"""Train a model as a model recipe says, on the dataset built from it.

Usage:
    wpt-train RECIPE DATASET_ID [options]
    wpt-train RECIPE DATASET_ID --strategy ddp --processes N [options]
    torchrun --nproc-per-node N -m wpt_train RECIPE DATASET_ID --strategy ddp [options]

RECIPE is a file name in data/recipes/models/, without .json, in the
repository holding the current folder. Everything that changes the
resulting model comes from it: the architecture's numbers, the method and
its settings, the stages (each its window length, steps and batch,
trained in order) and the seed. DATASET_ID is what
`wpt-dataset build RECIPE` prints; the dataset comes from the store named
by WPT_STORE (environment or .env), and a dataset built from another data
part than the recipe's is refused. Training never builds anything: it
reads the dataset and checks every input against its recorded hash first.

The options say only how the hardware runs it. --strategy names how the
hardware is used (`single`: one device; `ddp`: several processes, started
by --processes on this machine or by torchrun on one machine or several),
a built-in name or a .py file (see wpt_training); `split` divides each
window between the processes instead, for windows too long for one GPU.
On one GPU, --segment runs a longer window as consecutive segments of that
many tokens; with `split`, each process runs its slice of a longer window
in segments of that many tokens. Each gives the same training as whole
windows.

A stage's batch is the number of windows per update over every process
together, so the same recipe trains the same way on any hardware. Each
process takes --micro-batch windows per pass (default: all of its share of
each stage's batch) and adds passes up until the batch is complete. Lower --micro-batch when a
GPU runs out of memory; the training does not change.

Every option can also come from an environment variable, WPT_ plus the
option name in capitals with '_' for '-' (--micro-batch ->
WPT_MICRO_BATCH; WPT_RESUME=1); the command line wins. The recipe may come
from WPT_RECIPE and the dataset id from WPT_DATASET.

Every --checkpoint-every steps the main process writes MODEL/checkpoint/,
or with --run NAME the store's checkpoint of that name, which any machine
using the same store can continue. Run the same command with --resume to
continue (on other hardware too). The checkpoint is removed once the model
is saved. Ctrl+C or SIGTERM finishes the step and saves before stopping.
After a crash, every process starts again, up to --max-restarts times
(default 3), and continues from the checkpoint; 0 runs in this process.

The main process records the run in the store, in its own folder under
records/ (wpt_telemetry; docs/observability.md): what it read, every step,
evaluation and checkpoint, and every file it wrote; a run started by the
launcher keeps one folder over its restarts. When that record cannot be
written, the run finishes its step unrecorded, saves its checkpoint
(which the broken record does not name), and stops with code 3.

Exit codes: 0 done, 2 refused (bad input, other settings than the
checkpoint's; never restarted), 3 the record could not be written (never
restarted), 130 stopped by a signal, others a crash.

The model folder gets config.json, the weights (their headers name the
licenses), tokenizer.json (the dataset's vocabulary, for wpt-chat),
provenance.json (what produced it, with the sources it learned from), and
LICENSE, NOTICE and README.md (the model card). It is written beside the
target as a hidden folder, recorded, then renamed into place: a run that
fails before leaves the target as it was. A folder already there, with
everything in it (an earlier model, its checkpoint, the results wpt-eval
wrote for other weights), is replaced as a whole.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_inference import sample_tokens
from wpt_model import WPT, ModelConfig, canonical_ids, save_model, weight_files
from wpt_model.notice import WEIGHT_METADATA, write_credits, write_documents
from wpt_model.sparse import KernelUnavailable, kernels
from wpt_store import NAME as STORE_NAME
from wpt_store import StoreError, store_from_settings
from wpt_store.shards import ShardStream, sources_of, token_file
from wpt_telemetry import JOIN_FOLDER, RECORD_FAILED, RecordError, Telemetry, file_hashes, set_ref, short
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_training import (Checkpoint, FolderCheckpoint, Interrupted, PluginError, RunSettings, SettingsMismatch,
                          batching, load_method, load_strategy, train)
from wpt_training.launch import REFUSED, STOPPED, launch
from wpt_utils.device import describe
from wpt_utils.programs import repository

SIZE_UNITS = {"": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9}
CHECKPOINT_FOLDER = "checkpoint"


class Refused(Exception):
    """The run cannot start as asked; trying again would refuse again."""


class InputError(Exception):
    """The dataset is missing, its record describes another id, it has a
    record shape this version does not read, it was built from another data
    part than the recipe's, or the batch does not split over the processes. Nothing has been trained. (A shard or vocabulary
    that does not match its hash is the store's CorruptObject.)"""


def parse_size(text):
    """'500MB' -> 500_000_000. Units: KB, MB, GB, or none for bytes."""
    match = re.fullmatch(r"(\d+)\s*([KMG]B)?", text.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (examples: 500MB, 2GB, 1000000)")
    return int(match[1]) * SIZE_UNITS[match[2] or ""]


# option name, type, default, help
OPTIONS = [
    ("out", str, "runs/default", "model folder to write"),
    ("micro-batch", int, None, "windows per pass in one process (default: its whole share of each stage's batch)"),
    ("checkpoint-every", int, 500, "write a checkpoint every N steps (0: never)"),
    ("segment", int, 0, "run a window longer than this many tokens as consecutive segments (with --strategy split, "
                        "each process's slice): the same training with less memory, each segment but the last run "
                        "twice (a multiple of 128; 0: whole windows)"),
    ("strategy", str, "single", "how the hardware is used: a built-in name or a .py file"),
    ("processes", int, 1, "start this many processes on this machine, one per GPU (with --strategy ddp or split)"),
    ("max-shard-size", parse_size, None, "split the weights into files of at most this size, e.g. 500MB"),
    ("stream-shards", int, 8, "with --stream: dataset shards in use at once (windows are drawn from them)"),
    ("run", str, None, "keep the checkpoint in the store under this name, so any machine using the store can "
                       "continue the run"),
    ("max-restarts", int, 3, "after a crash, start every process again this many times at most, continuing from "
                             "the checkpoint (0: run in this process, never restart)"),
]


def parse_args(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", nargs="?", default=environ.get("WPT_RECIPE"),
                        help="model recipe: a file name in data/recipes/models/, without .json (WPT_RECIPE)")
    parser.add_argument("dataset", nargs="?", default=environ.get("WPT_DATASET"),
                        help="dataset id that `wpt-dataset build RECIPE` printed (WPT_DATASET)")
    for name, kind, default, text in OPTIONS:
        variable = "WPT_" + name.upper().replace("-", "_")
        if variable in environ:
            try:
                default = kind(environ[variable])
            except (ValueError, argparse.ArgumentTypeError) as error:
                parser.error(f"{variable}={environ[variable]!r}: {error}")
        parser.add_argument(f"--{name}", type=kind, default=default, help=f"{text} ({variable})")
    parser.add_argument("--stream", action="store_true", default=environ.get("WPT_STREAM", "") in ("1", "true", "yes"),
                        help="read the training shards a few at a time instead of copying them all to disk first "
                             "(WPT_STREAM)")
    parser.add_argument("--recompute", action="store_true",
                        default=environ.get("WPT_RECOMPUTE", "") in ("1", "true", "yes"),
                        help="keep only each layer's input and run the layer again for the backward pass: the same "
                             "training with much less memory, about a third slower (WPT_RECOMPUTE)")
    parser.add_argument("--resume", action="store_true", default=environ.get("WPT_RESUME", "") in ("1", "true", "yes"),
                        help="continue from the checkpoint: the model folder's, or with --run the store's (WPT_RESUME)")
    args = parser.parse_args(argv)
    if not args.recipe or not args.dataset:
        parser.error("give the model recipe and the dataset id (or set WPT_RECIPE and WPT_DATASET)")
    if args.run is not None and not STORE_NAME.fullmatch(args.run):
        parser.error(f"--run {args.run!r}: use letters, digits, '.', '_' and '-' only")
    return args


def load_recipe(name):
    """The model recipe of that name, checked against its schema, from the
    repository holding the current folder."""
    try:
        return Repo.find().recipe("model", name)
    except DatasetError as error:
        raise InputError(str(error)) from None


def load_dataset(store, dataset_id, recipe):
    """(record, record object, tokenizer), the record and vocabulary checked,
    and the dataset built from the recipe's data part."""
    record_object = store.get_ref("datasets", dataset_id)
    if record_object is None:
        raise InputError(f"no dataset {dataset_id} in {store}; build it with `wpt-dataset build <recipe>`")
    record = json.loads(store.get_bytes(record_object))
    if record.get("id") != dataset_id:
        raise InputError(f"dataset record {record_object} describes {record.get('id')}, not {dataset_id}")
    if record.get("schema") != 3:
        raise InputError(f"dataset {dataset_id} has record shape {record.get('schema')}, this wpt-train reads 3; "
                         "build the recipe again with the current wpt-dataset")
    if record.get("recipe") != recipe["data"]:
        raise InputError(f"dataset {dataset_id} was built from {record.get('recipe_name')!r} with another data part "
                         "than this recipe's; build this recipe with `wpt-dataset build` and use the id it prints")
    tokenizer = Tokenizer.from_bytes(store.get_bytes(record["tokenizer"]["vocabulary"]))  # sha256-checked by the store
    return record, record_object, tokenizer


def token_arrays(store, record, strategy, stream_shards=None):
    """(training tokens, held-out tokens) as arrays read from disk on demand.
    One process per machine builds the files, checking every shard. With
    stream_shards, the training tokens are a ShardStream that keeps only
    that many shards on disk at once."""
    parts = ("held_out_shards",) if stream_shards else ("shards", "held_out_shards")
    if strategy.is_local_main:
        for part in parts:
            token_file(store, record[part])
    strategy.barrier()
    held_out = (np.memmap(token_file(store, record["held_out_shards"]), dtype="<u4", mode="r")
                if record["held_out_shards"] else np.zeros(0, dtype="<u4"))
    if stream_shards:
        return ShardStream(store, record["shards"], stream_shards), held_out
    training = (np.memmap(token_file(store, record["shards"]), dtype="<u4", mode="r") if record["shards"] else
                np.zeros(0, dtype="<u4"))
    return training, held_out


def provenance(args, recipe, record, record_object, sources, model_config, n_params, strategy, plugins, trained,
               batch):
    versions = {}
    for package in ("torch", "numpy", "pyarrow", "wpt-model", "wpt-training", "wpt-tokenizer", "wpt-store", "wpt-train"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (strategy_plugin, method_plugin, method_settings) = plugins
    return {
        "schema": 4,
        "recipe_name": args.recipe,
        "recipe": recipe,  # everything that changes the model: data, architecture, training, post-training steps
        "dataset": record["id"],
        "dataset_record": record_object,
        "sources": {sid: record["sources"][sid] for sid in sorted(sources)},  # address, revision, licenses
        "skipped_sources": record.get("skipped", []),  # admitted, but left out of the build: it failed
        "vocabulary": record["tokenizer"]["vocabulary"],
        "shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["shards"]],
        "held_out_shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["held_out_shards"]],
        "options": {k: v for k, v in vars(args).items() if k not in ("recipe", "dataset")},  # how the hardware ran it
        "strategy": {"name": strategy_plugin.name, "file": strategy_plugin.file, "file_sha256": strategy_plugin.sha256,
                     "processes": strategy.world_size},
        "method": {"name": method_plugin.name, "file": method_plugin.file, "file_sha256": method_plugin.sha256,
                   "settings": method_settings},
        "batch": batch,  # each stage's windows, micro-batch and accumulation (wpt_training.batching)
        "model": asdict(model_config),
        "parameters": n_params,
        "best_validation_loss": trained.best_val,
        "steps_trained": trained.steps,  # the last step reached: fewer than the recipe's when the method stopped early
        "stages": list(trained.stages),  # each stage's window length, steps, first step and the last step it trained
        "weights_step": trained.weights_step,  # the step whose weights the folder holds
        "indexer_loss": trained.indexer_loss,  # the CSA indexers' loss at that step (docs/model.md, "Attention")
        **repository(),  # repository_commit, repository_dirty
        "device": describe(strategy.device) + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""),
        "tools": {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION,
                  **versions},
    }


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # the sample is any text; Windows pipes default to a narrower code page
    args = parse_args(argv)
    arguments = sys.argv[1:] if argv is None else list(argv)
    try:
        store = store_from_settings(Path.cwd() / ".env")
    except StoreError as error:  # no store, no record: nothing has started
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(REFUSED)
    try:
        if (args.processes > 1 or args.max_restarts > 0) and "LOCAL_RANK" not in os.environ:  # not yet started by one
            with _record(store, args, arguments, main=True) as telemetry:
                code = launch(args.processes, ["-m", "wpt_train", *arguments], max_restarts=args.max_restarts,
                              log=lambda text: telemetry.log(text, error=True), environ=telemetry.child_environment())
                if code:
                    sys.exit(code)
            return
        # the main process records the run; the others' events are the same, and only it writes the folder
        with _record(store, args, arguments, main=os.environ.get("RANK", "0") == "0") as telemetry:
            _run(args, store, telemetry)
    except RecordError as error:
        print(f"RecordError: {error}; stopped, nothing more was recorded", file=sys.stderr)
        sys.exit(RECORD_FAILED)


def _record(store, args, arguments, main):
    """The run's record: a new one, the one the launcher opened (JOIN_FOLDER),
    or none in a process other than the main one."""
    if not main:
        return Telemetry.off()
    if JOIN_FOLDER in os.environ:
        return Telemetry.join(store, "wpt-train", os.environ, env_file=Path.cwd() / ".env")
    return Telemetry.start(store, "wpt-train", arguments, short(args.dataset),
                           inputs={"recipe": args.recipe, "dataset": args.dataset}, env_file=Path.cwd() / ".env")


def _run(args, store, telemetry):
    try:
        with telemetry.span("inputs.recipe", recipe=args.recipe):
            recipe = load_recipe(args.recipe)
            training = recipe["training"]
            method_plugin, method_settings = load_method(training["method"], training["settings"])
            strategy_plugin, strategy = load_strategy(args.strategy)
    except InputError as error:
        _refuse(f"InputError: {error}", telemetry)
    except PluginError as error:
        _refuse(f"PluginError: {error}", telemetry)
    try:
        _train(args, recipe, strategy, (strategy_plugin, method_plugin, method_settings), store, telemetry)
    except Refused as error:
        _refuse(str(error), telemetry)
    except Interrupted as error:
        if strategy.is_main:
            telemetry.log(str(error), error=True)
        sys.exit(STOPPED)
    finally:
        strategy.close()


def _refuse(text, telemetry):
    telemetry.log(text, error=True)
    sys.exit(REFUSED)


class StoreCheckpoint(Checkpoint):
    """The checkpoint of a named run in the store (ref checkpoints/NAME):
    any machine using the same store continues it. Each save replaces the
    previous one, which is then deleted. The ref's change and every
    deletion are recorded in the run's record, naming the object, and that
    line is in the store before the step (RecordError otherwise). When the
    record breaks, a save still sets the ref, unrecorded, then raises the
    RecordError; nothing is deleted then: the previous object stays."""

    KIND = "checkpoints"

    def __init__(self, store, name, telemetry):
        self.store, self.name, self.telemetry = store, name, telemetry

    def __str__(self):
        return f"{self.store} as {self.KIND}/{self.name}"

    def exists(self):
        return self.store.get_ref(self.KIND, self.name) is not None

    def load(self):
        sha = self.store.get_ref(self.KIND, self.name)
        if sha is None:
            return None
        state = torch.load(self.store.path(sha), map_location="cpu", weights_only=True)  # path() checks the sha256
        self.store.release(sha)
        return state

    def save(self, state):
        self.store.scratch.mkdir(parents=True, exist_ok=True)
        partial = self.store.scratch / f"checkpoint-{self.name}.partial"
        try:
            torch.save(state, partial)
            sha = self.store.put_file(partial)
        finally:
            partial.unlink(missing_ok=True)
        previous = self.store.get_ref(self.KIND, self.name)
        try:
            set_ref(self.store, self.KIND, self.name, sha, self.telemetry)  # the previous one stays until this
        except RecordError:
            self.store.set_ref(self.KIND, self.name, sha)  # the checkpoint is saved all the same, unrecorded
            raise
        if previous and previous != sha:
            self._record_deletion(previous)
            self.store.delete(previous)

    def remove(self):
        sha = self.store.get_ref(self.KIND, self.name)
        if sha:
            self._record_deletion(sha)
        self.store.delete_ref(self.KIND, self.name)
        if sha:
            self.store.delete(sha)

    def _record_deletion(self, sha):
        self.telemetry.irreversible("training.checkpoint_deleted", object=sha, checkpoint=str(self))


class RestoreFailed(Exception):
    """A folder moved aside to make room for a new model could not be moved
    back after the new one failed to take its place: the target is missing,
    the earlier folder is at `aside`, the new one at `new`."""

    def __init__(self, target, aside, new, cause):
        super().__init__(f"{target} could not be replaced ({cause}), and the earlier folder, moved to {aside}, could "
                         f"not be moved back; the new model is in {new}: move one of them to {target} by hand")
        self.target, self.aside, self.new = target, aside, new


def _replace_folder(new, target: Path, telemetry):
    """Rename the folder new to target. A folder already at target is moved
    aside first and removed once new is in place. Moving aside and putting
    new in place are one step: on any failure or interruption between them,
    the earlier folder is moved back (and new with it, to where it was)
    before the error goes on, or RestoreFailed names both folders. Once new
    is in place, an earlier folder that cannot be removed, or whose removal
    is interrupted, stays beside the target, with a warning that names it
    (an interruption then goes on)."""
    if not target.exists():
        os.replace(new, target)
        return
    aside = target.parent / f".{target.name}-replaced-{secrets.token_hex(4)}"
    try:
        os.replace(target, aside)
        os.replace(new, target)
    except BaseException as error:
        if aside.exists():  # moved aside: put it back, whatever happened since
            try:
                if target.exists():
                    os.replace(target, new)
                os.replace(aside, target)
            except BaseException as failed:
                raise RestoreFailed(target, aside, new, f"{type(error).__name__}: {error}") from failed
        raise
    try:
        shutil.rmtree(aside)
    except (OSError, KeyboardInterrupt) as error:
        telemetry.event("model.replaced_folder_left", folder=str(aside), error=_describe(error))
        telemetry.log(f"warning: the earlier {target}, moved to {aside}, could not be removed "
                      f"({_describe(error)}): remove it by hand", error=True)
        if isinstance(error, KeyboardInterrupt):
            raise


def _describe(error):
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _train(args, recipe, strategy, plugins, store, telemetry):
    log = telemetry.log if strategy.is_main else (lambda *_: None)
    out = Path(args.out)
    restarted = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")) > 0  # set by the launcher and by torchrun
    resume = args.resume or restarted
    try:
        with telemetry.span("inputs.dataset", dataset=args.dataset, stream=args.stream) as span:
            checkpoint = (StoreCheckpoint(store, args.run, telemetry) if args.run
                          else FolderCheckpoint(out / CHECKPOINT_FOLDER))
            if checkpoint.exists() and not resume:
                raise Refused(f"{checkpoint} holds an unfinished run: add --resume to continue it, or delete it")
            record, record_object, tokenizer = load_dataset(store, args.dataset, recipe)
            train_data, val_data = token_arrays(store, record, strategy, args.stream_shards if args.stream else None)
            span.set(training_tokens=len(train_data), held_out_tokens=len(val_data),
                     shards=len(record["shards"]), held_out_shards=len(record["held_out_shards"]))
        telemetry.input(recipe=args.recipe, dataset=args.dataset, dataset_record=record_object,
                        vocabulary=record["tokenizer"]["vocabulary"])
        shape, training = recipe["model"], recipe["training"]
        settings = RunSettings(stages=training["stages"], micro_batch=args.micro_batch or 0,
                               eval_every=training["eval_every"], eval_batches=training["eval_batches"],
                               checkpoint_every=args.checkpoint_every, seed=training["seed"], segment=args.segment)
    except (StoreError, InputError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    if len(train_data) <= settings.longest + 1:
        raise Refused(f"the dataset has {len(train_data):,} training tokens, fewer than one window of "
                      f"{settings.longest}")
    log(f"device: {describe(strategy.device)}" + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""))
    log(f"dataset {args.dataset}: {len(train_data):,} training tokens, {len(val_data):,} held-out tokens, "
        f"vocabulary {tokenizer.vocab_size:,} tokens")

    torch.manual_seed(settings.seed)  # the same initial weights in every process
    # the model's context: the last stage's windows
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=settings.stages[-1].block_size, **shape)
    model = WPT(model_config)
    model.set_canonical_ids(canonical_ids(tokenizer.decode, tokenizer.vocab_size))  # Engram's, from the vocabulary
    model.recompute = args.recompute
    if strategy.device.type == "cuda":  # decided once, before training (docs/model.md, "Attention")
        try:
            chosen = kernels(strategy.device)
        except KernelUnavailable as error:
            raise Refused(f"KernelUnavailable: {error}") from None
        log("attention: " + (f"{chosen.NAME} kernels" if chosen
                             else "PyTorch (neither TileLang nor Triton is installed)"))
    model = model.to(strategy.device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_params:,}")
    strategy_plugin, method_plugin, method_settings = plugins
    telemetry.event("training.settings", recipe=recipe, settings=asdict(settings), model=asdict(model_config),
                    parameters=n_params, device=describe(strategy.device), processes=strategy.world_size,
                    strategy={"name": strategy_plugin.name, "file_sha256": strategy_plugin.sha256},
                    method={"name": method_plugin.name, "file_sha256": method_plugin.sha256, "settings": method_settings},
                    resume=resume, checkpoint=str(checkpoint))
    try:
        with telemetry.span("training", steps=settings.steps) as span:
            trained = train(model, strategy, method_plugin, method_settings, train_data, val_data, settings,
                            checkpoint=checkpoint, resume=resume,
                            identity={"dataset": args.dataset, "model": asdict(model_config), "training": training,
                                      # the shards in use decide the windows, so a resumed run must stream the same way
                                      **({"stream_shards": args.stream_shards} if args.stream else {})},
                            log=log, telemetry=telemetry)
            span.set(**asdict(trained))
    except (ValueError, SettingsMismatch, PluginError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    finally:
        if isinstance(train_data, ShardStream):
            train_data.close()
    if not strategy.is_main:
        return

    # each stage's windows, micro-batch and accumulation, as the run split them
    batch = batching(settings, 1 if strategy.splits_windows else strategy.world_size)
    named = record["tokenizer"]
    # written beside the target, then renamed: a failure midway leaves the target as it was
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    renaming = False
    try:
        with telemetry.span("model.write", folder=str(out)):
            with telemetry.span("model.weights"):
                save_model(model, staging, store.get_bytes(named["vocabulary"]),
                           f"{named['recipe']} version {named['recipe_version']}", max_shard_bytes=args.max_shard_size,
                           metadata=WEIGHT_METADATA)
            with telemetry.span("model.provenance") as span:
                sources = sources_of(store, record["shards"])
                written = provenance(args, recipe, record, record_object, sources, model_config, n_params, strategy,
                                     plugins, trained, batch)
                (staging / "provenance.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8",
                                                         newline="\n")
                span.set(sources=sorted(sources), skipped_sources=written["skipped_sources"])
            with telemetry.span("model.notice"):  # the credits each source asks for, NOTICE, LICENSE, the model card
                write_credits(staging, written, store.get_bytes)
                write_documents(staging)
            files = file_hashes(staging)
            telemetry.event("model.files", files=files)
            # recorded before the folder appears: a record that cannot be written leaves the target as it was
            telemetry.output(model_folder=str(out), weights={name: files[name] for name in weight_files(staging)},
                             provenance=files["provenance.json"])
            # the run is complete; recorded before the checkpoint goes (a FolderCheckpoint goes with the old folder)
            telemetry.event("training.checkpoint_removed", checkpoint=str(checkpoint))
            telemetry.irreversible("model.folder_replaced", folder=str(out))
            renaming = True
            _replace_folder(staging, out, telemetry)
    except BaseException as error:
        # decided by the folders, not by the error: a RestoreFailed may arrive inside a RecordError
        if renaming and not out.exists() and staging.exists():
            if not isinstance(error, RestoreFailed):  # its own message already names both folders
                earlier = sorted(out.parent.glob(f".{out.name}-replaced-*"))
                print(f"{out} is missing: the new model is kept in {staging}"
                      + "".join(f", the earlier one in {folder.absolute()}" for folder in earlier)
                      + f"; move one of them to {out} by hand", file=sys.stderr)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    checkpoint.remove()
    log(f"saved {out} (licenses and sources: {out / 'NOTICE'})")
    # show a sample so you immediately see what it learned
    model.eval()
    ids = sample_tokens(model, [Tokenizer.END], context_size=model_config.block_size, max_new_tokens=120)
    log(tokenizer.decode(list(ids)).decode("utf-8", errors="replace"))
