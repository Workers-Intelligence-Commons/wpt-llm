"""wpt-train's steps as a gpu worker's job (docs/machines.md): the part of
the training that needs the GPUs, started by the wpt manager (wpt_train)
with the spec it built.

With --processes above 1 or --max-restarts above 0, or when the run
spans several gpu workers, the job's first process starts the training's
processes (wpt_training.launch), each again `python -m wpt_worker.job`
with its own channel, and starts them again after a crash (on one gpu
worker only: across machines, a crash ends the job on every one, and the
manager resumes the run from its checkpoint, wpt_train.CRASHED); each of
them, or the first process alone otherwise:

    1. loads the strategy and the recipe's steps (the files outside the
       repository from what the manager sent: wpt_training.SENT)
    2. tells the manager it is ready (ready: its device, its Python and
       libraries' versions for the provenance, the setup the scheduler's pass times are measured on) and gets the
       plan back: each step's fingerprint, how many steps are reused, the
       state the run starts from, the pass times to replay
    3. trains each step that is not reused, its environment's part on the
       manager called through data.call (env.call), its checkpoint written
       on this machine and, as WPT_LINK_CHECKPOINTS says, sent to the
       manager (LinkCheckpoint); then step.done, which keeps its result.
       The first start of a job removes the checkpoint folders earlier
       runs' jobs left on this machine (after a stop or a crash), which no
       later run reads
    4. trains the drafter, measures or replays the pass times, writes a
       sample's token ids, and sends them with the drafter (finished)

The manager writes the model folder from what it kept and received. The
job deals only in token ids: the stop tokens it needs come in the spec.

On several gpu workers (spec["machines"]: this worker's number, how many
there are, and how they exchange gradients; placement()), each worker's
processes are ranks number * --processes onward of every worker's
processes together. In one place (transport nccl), they all meet at gpu
worker 0's private address, on the port after its own, through NCCL (or
gloo on the CPU); a worker refuses a meeting address or an address of
its own (spec["machines"]["private"]) that is not private, and keeps
NCCL's and gloo's traffic to the network interface that has its own
(NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME). Across places (transport link), each machine's processes meet on
that machine only, and the strategy is wpt_worker.peers.Bridged, which
exchanges the gradients between the machines over their TLS links; only
--strategy ddp runs that way.

Exit codes are wpt-train's: 2 refused, 3 the record could not be written,
130 stopped.

It lives in wpt-worker rather than in wpt-train because wpt_train imports
the tokenizer when it loads, which a job refuses (wpt_worker.job), and
because wpt-worker's dependencies are the libraries a worker installs for
its job: what this module imports.
"""

import importlib.metadata
import ipaddress
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import psutil
import torch

from wpt_inference import COSTS_FILE, costs_file, costs_setup, measure_costs, sample_tokens
from wpt_inference.drafting import DrafterFailed, train_drafter
from wpt_inference.drafting import settings as drafter_settings
from wpt_model import PARTS, WPT, ModelConfig
from wpt_model.sparse import KernelUnavailable, kernels
from wpt_store import StoreError
from wpt_store.shards import documents
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry
from wpt_training import (SENT, WINDOWS, Checkpoint, Data, FolderCheckpoint, Interrupted, Options, PluginError,
                          SettingsMismatch, StepFailed, load_state, load_step, load_strategy, save_state, train, weights)
from wpt_training.launch import REFUSED, STOPPED, launch
from wpt_utils.device import describe

from .job import LinkStore, call, download, remote, upload
from .link import RemoteError, addresses, is_private
from .peers import HUB, MACHINE, MACHINES
from .writing import Writing

EVERY, STEP_ENDS = "every", "step-ends"  # WPT_LINK_CHECKPOINTS: what the manager gets of the checkpoints
CHECKPOINTS = (EVERY, STEP_ENDS)
NCCL, LINK = "nccl", "link"  # spec["machines"]["transport"]: how a run's gpu workers exchange gradients
ONE_MACHINE = {"index": 0, "count": 1, "transport": None}
BRIDGED = "WPT_BRIDGED"  # set for the processes of a run across places: their strategy is wpt_worker.peers.Bridged
WORKER_ONLY = ("; with step-ends, that checkpoint is on the gpu worker's disk only, where only this job's restarts "
               "read it: the manager's checkpoint does not hold it, so --resume starts the step over")
# the outside libraries the gpu worker trains with, whose versions the model's provenance names; a step's
# fingerprint holds the code instead (wpt_utils.programs.code, on the manager, whose checkout the worker runs)
LIBRARIES = ("torch", "numpy", "pyarrow")


class Refused(Exception):
    """The run cannot go on as asked; trying again would refuse again."""


def tools() -> dict:
    """The Python, the system and the versions of LIBRARIES (None when
    one is not installed) that train here, for the model's provenance."""
    versions = {}
    for package in LIBRARIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), **versions}


def run(spec, link, work: Path) -> int:
    options = spec["options"]
    machines = spec.get("machines") or ONE_MACHINE
    store = LinkStore(link, work)
    environ = {**spec["join"], **os.environ}  # a process the job's launcher started joins under its span
    try:
        several = machines["count"] > 1
        if (options["processes"] > 1 or options["max_restarts"] > 0 or several) and "LOCAL_RANK" not in os.environ:
            with Telemetry.join(store, "wpt-train", environ) as telemetry:
                log = lambda text: telemetry.log(text, error=True)  # noqa: E731
                try:
                    placed = placement(machines, options)
                except Refused as error:
                    log(str(error))
                    return REFUSED
                restarts = options["max_restarts"]
                if several and restarts:
                    log(f"{machines['count']} gpu workers: a crashed process ends the job on every one of them, "
                        "with no restart here (restarts stay on one machine); wpt-train then exits with code 5, "
                        "which wpt-manager resumes from the checkpoint")
                    restarts = 0
                return launch(options["processes"], ["-m", "wpt_worker.job"], max_restarts=restarts, log=log,
                              environ={**telemetry.child_environment(), **placed["environ"]},
                              first_rank=placed["first_rank"], world_size=placed["world_size"],
                              master=placed["master"])
        main = os.environ.get("RANK", "0") == "0"
        with (Telemetry.join(store, "wpt-train", environ) if main else Telemetry.off()) as telemetry:
            return _run(spec, link, store, work, telemetry)
    except RecordError as error:
        print(f"RecordError: {error}; stopped, nothing more was recorded", file=sys.stderr)
        return RECORD_FAILED


def placement(machines, options) -> dict:
    """Where this machine's processes stand in the run (spec["machines"],
    which the manager gives each gpu worker): their first rank, every
    machine's processes together, where they meet, and what they are told
    besides. Refused when the run cannot be placed as asked: NCCL's peers
    at an address that is not private, or a strategy other than ddp across
    places."""
    count, index, processes = machines["count"], machines["index"], options["processes"]
    if count == 1:
        return {"environ": {}, "first_rank": 0, "world_size": processes, "master": None}
    first, world = index * processes, count * processes
    if machines["transport"] == NCCL:
        host, own = machines["master"], machines["private"]
        for address, whose in ((host, "the gpu workers of one place meet at"), (own, "this gpu worker's own is")):
            if not is_private(address):
                raise Refused(f"{whose} {address}, which is not a private address: a gpu worker joins NCCL peers at "
                              "private addresses only, since NCCL's ports have no password")
        interface = interface_of(own)
        if interface is None:
            raise Refused(f"this gpu worker's private address is {own}, which none of this machine's network "
                          "interfaces has: give its key the private= address this machine has")
        # NCCL and gloo keep to that interface: none of their traffic goes out on another one, public or not
        pinned = {"NCCL_SOCKET_IFNAME": f"={interface}", "GLOO_SOCKET_IFNAME": interface}
        return {"environ": pinned, "first_rank": first, "world_size": world, "master": (host, machines["port"])}
    if options["strategy"] != "ddp":
        raise Refused(f"--strategy {options['strategy']}: gpu workers in different places exchange gradients over "
                      "their link, which only ddp does; use --strategy ddp")
    hub = {HUB: machines["hub"]} if index else {}
    return {"environ": {MACHINE: str(index), MACHINES: str(count), BRIDGED: "1", **hub}, "first_rank": first,
            "world_size": world, "master": None}


def interface_of(host) -> str | None:
    """The name of this machine's network interface that has an address
    host stands for; None when none has."""
    wanted = addresses(host)
    for name, found in psutil.net_if_addrs().items():
        for entry in found:
            try:
                if ipaddress.ip_address(entry.address.split("%")[0]) in wanted:
                    return name
            except ValueError:  # a hardware address
                continue
    return None


def sent(files: dict, folder: Path):
    """The files outside the repository the manager sent ({the path a recipe
    or an option gave: its text}), written into folder under their own
    names, and named in wpt_training.SENT."""
    for number, (given, text) in enumerate(sorted(files.items()), 1):
        path = folder / str(number) / Path(given).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        SENT[given] = str(path)


def _run(spec, link, store, work, telemetry) -> int:
    sent(spec["files"], work / "files")
    try:
        if os.environ.get(BRIDGED):  # gpu workers in different places: ddp, bridged over their links
            from .peers import Bridged
            strategy = Bridged()
        else:
            strategy_plugin, strategy = load_strategy(spec["options"]["strategy"])
    except PluginError as error:
        telemetry.log(f"PluginError: {error}", error=True)
        return REFUSED
    try:
        _train(spec, link, store, work, strategy, telemetry)
        return 0
    except Refused as error:
        if strategy.is_main:
            telemetry.log(str(error), error=True)
        return REFUSED
    except Interrupted as error:
        if strategy.is_main:
            telemetry.log(str(error) + (WORKER_ONLY if error.saved and spec["checkpoints"] != EVERY else ""),
                          error=True)
        return STOPPED
    finally:
        strategy.close()


class LinkCheckpoint(Checkpoint):
    """The run's checkpoint as the job sees it: always written on this
    machine (local, a folder of the job's own for the run's record), where
    a restart of the job's processes finds it; with every, each save is
    also sent to the manager, which keeps it in its own checkpoint (the
    model folder's or the store's). exists() and remove() are the
    manager's; load() reads the local one first."""

    def __init__(self, link, local: FolderCheckpoint, every: bool, shown: str):
        self.link, self.local, self.every, self.shown = link, local, every, shown

    def __str__(self):
        return self.shown

    def exists(self):
        return call(self.link, "checkpoint.exists")[0]["exists"]

    def load(self):
        if self.local.path.is_file():
            return self.local.load()
        header, _ = call(self.link, "checkpoint.load")
        if header.get("file") is None:
            return None
        download(self.link, header, self.local.path)
        return self.local.load()

    def save(self, state):
        self.local.save(state)
        if self.every:
            call(self.link, "checkpoint.save", upload=upload(self.link, self.local.path))

    def remove(self):
        self.local.remove()
        call(self.link, "checkpoint.remove")


def environment_call(link, number):
    """data.call of step number: a method of its environment's part on the
    manager; a StepFailed there is a StepFailed here."""
    def environment(name, *arguments):
        try:
            _, body = link.call("env.call", json.dumps({"number": number, "name": name,
                                                         "arguments": arguments}).encode())
        except RemoteError as error:
            if error.kind == "StepFailed":
                raise StepFailed(error.message) from None
            raise remote(error) from None
        return json.loads(body)
    return environment


def _asked(strategy, ask):
    """ask() in the main process, its answer in every process; a Refused
    there is one in every process."""
    found = None
    if strategy.is_main:
        try:
            found = ("answer", ask())
        except Refused as error:
            found = ("refused", str(error))
        except RemoteError as error:
            if error.kind != "Refused":
                raise
            found = ("refused", error.message)
    kind, value = strategy.broadcast(found)
    if kind == "refused":
        raise Refused(value)
    return value


def _train(spec, link, store, work, strategy, telemetry):
    options = spec["options"]
    log = telemetry.log if strategy.is_main else (lambda *_: None)
    try:
        steps = [load_step(entry, PARTS) for entry in spec["steps"]]
    except PluginError as error:
        raise Refused(f"PluginError: {error}") from None
    other = next((step for step in steps if step.reads != WINDOWS), None)
    if strategy.splits_windows and other is not None:
        raise Refused(f"InputError: --strategy split splits text windows only, and the step {other.name} reads "
                      f"{other.reads}")
    restarted = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")) > 0  # set by the launcher
    resume = options["resume"] or restarted
    model_config = ModelConfig(**spec["model"])
    drafter_config, drafting = drafter_settings(model_config, spec["drafter"])  # the manager checked them
    setup = costs_setup(model_config, 1, drafter_config.block, strategy.device)  # what the times are measured on
    seed = spec["seed"]
    torch.manual_seed(seed)  # the same initial weights in every process
    model = WPT(model_config)
    model.set_canonical_ids(torch.tensor(spec["canonical"], dtype=torch.long))  # Engram's, from the vocabulary
    model.recompute = options["recompute"]
    if strategy.device.type == "cuda":  # decided once, before training (docs/model.md, "Attention")
        try:
            chosen = kernels(strategy.device)
        except KernelUnavailable as error:
            raise Refused(f"KernelUnavailable: {error}") from None
        log("attention: " + (f"{chosen.NAME} kernels" if chosen
                             else "PyTorch (neither TileLang nor Triton is installed)"))
    model = model.to(strategy.device)
    parameters = sum(p.numel() for p in model.parameters())
    log(f"parameters: {parameters:,}")
    plan = _asked(strategy, lambda: call(link, "ready", device=describe(strategy.device),
                                         processes=strategy.world_size, setup=setup, tools=tools(),
                                         parameters=parameters)[0])

    record = spec["record"]
    marks, reused = plan["marks"], plan["reused"]
    if reused:
        state = None
        if strategy.is_main:
            state = load_state(download(link, plan["state"], work / "reused.safetensors"),
                               only=("model", "best_model"))
            (work / "reused.safetensors").unlink()
        model.load_state_dict(strategy.broadcast(weights(state) if strategy.is_main else None))
        del state
    local = FolderCheckpoint(work.parent / "checkpoints" / telemetry_key(spec))
    if strategy.is_main and not restarted:
        _remove_others(local.folder, log)
    checkpoint = LinkCheckpoint(link, local, spec["checkpoints"] == EVERY, spec["checkpoint"])
    writing = Writing(link)
    data_options = {"stream_shards": options["stream_shards"] if options["stream"] else None,
                    "micro_batch": options["micro_batch"] or 0}
    for number, (step, mark) in enumerate(zip(steps, marks), 1):
        if number <= reused:
            continue
        data = Data(step.name, store, record, model_config, strategy, telemetry, log, seed, number, **data_options,
                    call=environment_call(link, number), writing=writing)
        try:
            with telemetry.span("step", step=step.name, number=number, fingerprint=mark) as span:
                method = step.start_method()
                environment = step.start_environment(data)
                try:
                    trained = train(model, strategy, step, method, environment, seed, number,
                                    Options(options["checkpoint_every"], options["segment"]), checkpoint=checkpoint,
                                    resume=resume, identity=mark, done=marks[:number - 1], log=log,
                                    telemetry=telemetry, save_last=True)  # its result is kept from the checkpoint
                    sources = environment.sources if strategy.is_main else {}
                finally:
                    environment.close()
                span.set(**asdict(trained))
        except StepFailed as error:
            left = strategy.is_main and checkpoint.exists()
            raise Refused(f"StepFailed: step {number} {step.name}: {error.reason}; "
                          + (f"its checkpoint stays in {checkpoint}: --resume continues the step from it, and "
                             "deleting it starts the step over" if left else "nothing of it is kept")
                          + (f"; the results of the {number - 1} steps before it stay in the store"
                             if number > 1 else "")) from None
        except (ValueError, SettingsMismatch, PluginError, StoreError) as error:
            raise Refused(f"{type(error).__name__}: step {number} {step.name}: {error}") from None
        if strategy.is_main:
            about = {"number": number, "reached": _reached(trained), "sources": dict(sorted(sources.items()))}
            sent_state = None if checkpoint.every else upload(link, local.path)
            call(link, "step.done", json.dumps(about).encode(), state=sent_state)
            local.remove()  # the manager keeps the step's result
        strategy.barrier()
        resume = False  # the step that stopped, when one did, is done
    if strategy.is_main:
        _finish(spec, link, store, work, model, model_config, drafter_config, drafting, setup, plan, steps,
                strategy, telemetry)


def _remove_others(own: Path, log):
    """Remove the checkpoint folders of other runs' jobs beside own: each
    is keyed by its run's record folder, which no later command uses again,
    so nothing reads them any more."""
    if not own.parent.is_dir():
        return
    for folder in sorted(own.parent.iterdir()):
        if folder != own and folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
            log(f"removed the gpu worker's checkpoint of an earlier run ({folder.name}): nothing reads it")


def telemetry_key(spec) -> str:
    """The name of the job's own checkpoint folder: the run's record
    folder, so a restart of this run finds it and no other run does."""
    return spec["join"]["WPT_RECORD_FOLDER"]


def _reached(trained) -> dict:
    """What a step's provenance says it reached (wpt_training.Trained)."""
    return {"best_validation_loss": trained.best_val,
            "updates": trained.steps,  # the last update made: fewer than its stages' when the optimizer stopped
            "stages": list(trained.stages),  # each stage's window length, updates, batch, first and last update
            "weights_update": trained.weights_step,  # the update whose weights the step leaves
            "indexer_loss": trained.indexer_loss,  # the CSA indexers' loss at that update (docs/model.md, "Attention")
            "skipped": trained.skipped}  # updates with nothing to learn


def _finish(spec, link, store, work, model, model_config, drafter_config, drafting, setup, plan, steps, strategy,
            telemetry):
    """The drafter, the scheduler's pass times and a sample, sent to the manager."""
    end, im_end = spec["tokens"]["end"], spec["tokens"]["im_end"]
    # a model that trained on turns ends its answers with <|im_end|> as well as the end of a document
    turns = any(step.reads != WINDOWS for step in steps)
    stops = (end, im_end) if turns else (end,)
    try:
        with telemetry.span("drafter", **asdict(drafter_config), stops=list(stops)):
            drafter = train_drafter(model, (tokens for _, _, tokens in documents(store, spec["record"]["shards"])),
                                    drafter_config, drafting, end=end, stops=stops, device=strategy.device,
                                    telemetry=telemetry)
    except DrafterFailed as error:
        raise Refused(f"DrafterFailed: {error}; nothing was written; every step's result is kept in the store: "
                      "the same command trains the drafter again without training") from None
    # the scheduler's pass times, once per run, in the record; or those an earlier run recorded, replayed
    with telemetry.span("scheduler.costs", replayed=spec["options"]["scheduler_costs"]):
        costs = plan["costs"] if plan["costs"] is not None else measure_costs(model, 1, drafter.block)
        telemetry.write_file(COSTS_FILE, costs_file(costs, setup))
    model.eval()  # a sample, so the manager shows at once what it learned
    sample = list(sample_tokens(model, [end], max_new_tokens=120, drafter=drafter, costs=costs))
    with tempfile.TemporaryDirectory(dir=work) as folder:
        path = Path(folder) / "drafter.safetensors"
        save_state({"state": drafter.state_dict(), "provenance": drafter.provenance}, path)
        call(link, "finished", json.dumps({"sample": sample}).encode(), drafter=upload(link, path))
