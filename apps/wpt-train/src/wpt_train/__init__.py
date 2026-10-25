"""Train a model as a model recipe says, on the dataset built from it.

Usage:
    wpt-train RECIPE DATASET_ID [options]
    wpt-train RECIPE DATASET_ID --strategy ddp --processes N [options]
    torchrun --nproc-per-node N -m wpt_train RECIPE DATASET_ID --strategy ddp [options]

RECIPE is a file name in data/recipes/models/, without .json, in the
repository holding the current folder. Everything that changes the
resulting model comes from it: the architecture's numbers, the seed, and
its steps, which run in order on one model (docs/training.md, "Steps"):
each a built-in step of wpt_training (read-text, answer-as-assistant,
finish-tasks, finish-tasks-with-less-writing) or a .py file, with its
method, environment, optimizer, settings and locked parts. The model's
context is the last window length of the last step that reads text
windows. DATASET_ID is what `wpt-dataset build RECIPE` prints; the dataset
comes from the store named by WPT_STORE (environment or .env), and a
dataset built from another data part than the recipe's is refused. It
holds everything the steps read: the text's shards, and the conversations
and tasks the build found. Training never builds anything: it reads the
dataset and checks every input against its recorded hash first. Every
step's settings, the drafter's, the steps' locked parts, and that the
dataset holds something for each step to learn from are checked before
any step runs.

The options say only how the hardware runs it. --strategy names how the
hardware is used (`single`: one device; `ddp`: several processes, started
by --processes on this machine or by torchrun on one machine or several),
a built-in name or a .py file (see wpt_training); `split` divides each
text window between the processes instead, for windows too long for one
GPU, and is refused for a recipe with a step that reads something else. On
one GPU, --segment runs a longer window as consecutive segments of that
many tokens; with `split`, each process runs its slice of a longer window
in segments of that many tokens. Each gives the same training as whole
windows.

A step's batch is the number of examples per update over every process
together, so the same recipe trains the same way on any hardware. Each
process takes --micro-batch examples per pass (default: all of its share of
each batch) and adds passes up until the batch is complete. Lower
--micro-batch when a GPU runs out of memory; the training does not change.

Every option can also come from an environment variable, WPT_ plus the
option name in capitals with '_' for '-' (--micro-batch ->
WPT_MICRO_BATCH; WPT_RESUME=1); the command line wins. The recipe may come
from WPT_RECIPE and the dataset id from WPT_DATASET.

After each step, the store keeps its result (ref steps/<fingerprint>): the
weights and the optimizer's state, under a fingerprint of the step before
it, the step itself (method, optimizer, environment, settings, locked
parts), the fingerprint of the part of the dataset it reads, the seed and
the code versions; no option is part of it. A run whose first steps
have a kept result starts from the result of the last of them and runs the
steps after it: a recipe that shares its first steps with an earlier run
reuses them, and a run that failed after some steps continues from the
first step not kept. Reuse means the same inputs, not the same bits.

Inside a step, every --checkpoint-every updates the main process writes
MODEL/checkpoint/, or with --run NAME the store's checkpoint of that name,
which any machine using the same store can continue. Run the same command
with --resume to continue the step that stopped (on other hardware too);
the steps before it are reused. Once the step ends, its last checkpoint
becomes its kept result, as it is. Ctrl+C or SIGTERM finishes the update and saves before
stopping. After a crash, every process starts again, up to --max-restarts
times (default 3), and continues the same way; 0 runs in this process.

The main process records the run in the store, in its own folder under
records/ (wpt_telemetry; docs/observability.md): what it read, every step,
update, evaluation and checkpoint, every step result kept or reused, and
every file it wrote; a run started by the launcher keeps one folder over
its restarts. When that record cannot be written, the run finishes its
update unrecorded, saves its checkpoint (which the broken record does not
name), and stops with code 3.

After the last step, the main process trains the model's drafter for
speculative decoding against the final weights, the model frozen, with the
recipe's `drafter` section (wpt_inference.drafting; docs/model.md, "The
drafter"): its prompts are the beginnings of the dataset's training
documents, and the model alone writes the answers, which stop at the end
of a document and, when a step trained on conversations or tasks, at
<|im_end|>, the end of a turn. A model and its drafter are one unit: no
option skips it, and a drafter that cannot be trained refuses the run
(code 2) and writes nothing; every step's result is kept, so the same
command trains the drafter again without training. Drafter settings no
drafter can have (feature layers the model does not have, an unknown
setting) are refused before training starts. Then it measures the
scheduler's pass times once (wpt_inference.measure_costs) and writes them
into the run's record as scheduler-costs.json, with the setup they were
measured on (the model's config, the drafter's block, the device);
--scheduler-costs FOLDER replays those an earlier run recorded instead,
refused before training when they were measured on another setup.

Exit codes: 0 done, 2 refused (bad input, other settings than the
checkpoint's, a step that failed, a drafter that could not be trained;
never restarted), 3 the record could not be written (never restarted), 130
stopped by a signal, others a crash. A step that fails keeps no result;
a checkpoint it wrote stays, which --resume continues and deleting it
starts the step over; the results of the steps before it stay in the
store.

The model folder gets config.json, the weights (their headers name the
licenses), tokenizer.json (the dataset's vocabulary, for wpt-chat), the
drafter (drafter/: config.json, model.safetensors, and provenance.json,
which names the sha256 of the weights it was trained against),
provenance.json (what produced it: each step, its fingerprint and result,
and the sources each one learned from), and LICENSE, NOTICE and README.md
(the model card). It is written beside the target as a hidden folder,
recorded, then renamed into place: a run that fails before leaves the
target as it was. A folder already there, with everything in it (an
earlier model, its checkpoint, the results wpt-eval wrote for other
weights), is replaced as a whole. When the new folder cannot take the
target's place and the target is then missing, or holds only a checkpoint
of this run (without --run), the new folder is kept and named on the
error output, to be moved into place by hand.
"""

import argparse
import hashlib
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

import torch

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_dataset.sources import fingerprint
from wpt_inference import COSTS_FILE, costs_file, costs_setup, measure_costs, replayed_costs, sample_tokens
from wpt_inference.drafting import DrafterFailed, train_drafter
from wpt_inference.drafting import settings as drafter_settings
from wpt_model import PARTS, WPT, ModelConfig, canonical_ids, save_drafter, save_model, weight_files, weight_hashes
from wpt_model.notice import WEIGHT_METADATA, write_credits, write_documents
from wpt_model.sparse import KernelUnavailable, kernels
from wpt_store import NAME as STORE_NAME
from wpt_store import StoreError, store_from_settings
from wpt_store.shards import documents
from wpt_telemetry import JOIN_FOLDER, RECORD_FAILED, RecordError, Telemetry, file_hashes, set_ref, short
from wpt_tokenizer import RUSTC_VERSION, Tokenizer, chat
from wpt_training import (WINDOWS, Checkpoint, Data, FolderCheckpoint, Interrupted, Options, PluginError,
                          SettingsMismatch, StepFailed, load_step, load_strategy, train, weights)
from wpt_training.launch import REFUSED, STOPPED, launch
from wpt_utils.device import describe
from wpt_utils.programs import repository

SIZE_UNITS = {"": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9}
CHECKPOINT_FOLDER = "checkpoint"
# the code whose version a step's result depends on, beside the files of the step's own parts
CODE = ("torch", "numpy", "wpt-model", "wpt-training", "wpt-tokenizer", "wpt-store", "wpt-dataset", "wpt-agent")


class Refused(Exception):
    """The run cannot start as asked; trying again would refuse again."""


class InputError(Exception):
    """The dataset is missing, its record describes another id, it has a
    record shape this version does not read, it was built from another data
    part than the recipe's, the recipe's steps cannot run as asked, or the
    batch does not split over the processes. Nothing has been trained. (A
    shard or vocabulary that does not match its hash is the store's
    CorruptObject.)"""


def parse_size(text):
    """'500MB' -> 500_000_000. Units: KB, MB, GB, or none for bytes."""
    match = re.fullmatch(r"(\d+)\s*([KMG]B)?", text.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (examples: 500MB, 2GB, 1000000)")
    return int(match[1]) * SIZE_UNITS[match[2] or ""]


# option name, type, default, help
OPTIONS = [
    ("out", str, "runs/default", "model folder to write"),
    ("micro-batch", int, None, "examples per pass in one process (default: its whole share of each batch)"),
    ("checkpoint-every", int, 500, "write a checkpoint every N updates of a step (0: none during a step); each "
                                   "step's result is kept in the store when it ends"),
    ("segment", int, 0, "run a window longer than this many tokens as consecutive segments (with --strategy split, "
                        "each process's slice): the same training with less memory, each segment but the last run "
                        "twice (a multiple of 128; 0: whole windows)"),
    ("strategy", str, "single", "how the hardware is used: a built-in name or a .py file"),
    ("processes", int, 1, "start this many processes on this machine, one per GPU (with --strategy ddp or split)"),
    ("max-shard-size", parse_size, None, "split the weights into files of at most this size, e.g. 500MB"),
    ("stream-shards", int, 8, "with --stream: dataset shards in use at once (windows are drawn from them)"),
    ("run", str, None, "keep the checkpoint in the store under this name, so any machine using the store can "
                       "continue the run"),
    ("scheduler-costs", str, None, "replay the drafter scheduler's pass times that an earlier run recorded on the "
                                   f"same setup: the name of its record folder, which holds {COSTS_FILE} (default: "
                                   "measure them)"),
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
                        help="continue the step that stopped from its checkpoint: the model folder's, or with --run "
                             "the store's (WPT_RESUME)")
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
    if record.get("schema") != 4:
        raise InputError(f"dataset {dataset_id} has record shape {record.get('schema')}, this wpt-train reads 4; "
                         "build the recipe again with the current wpt-dataset")
    if record.get("recipe") != recipe["data"]:
        raise InputError(f"dataset {dataset_id} was built from {record.get('recipe_name')!r} with another data part "
                         "than this recipe's; build this recipe with `wpt-dataset build` and use the id it prints")
    tokenizer = Tokenizer.from_bytes(store.get_bytes(record["tokenizer"]["vocabulary"]))  # sha256-checked by the store
    return record, record_object, tokenizer


def plan(recipe, strategy) -> tuple[list, int]:
    """(the recipe's steps, each a wpt_training.Step, every setting typed
    and checked, the model's context: the last window length of the last
    step that reads text windows). PluginError for a step that cannot be;
    InputError for steps that cannot run together as asked."""
    steps = [load_step(entry, PARTS) for entry in recipe["steps"]]
    contexts = [step.context() for step in steps if step.context() is not None]
    if not contexts:
        raise InputError("no step reads text windows, so nothing sets the model's context; start the recipe with "
                         "read-text")
    other = next((step for step in steps if step.reads != WINDOWS), None)
    if strategy.splits_windows and other is not None:
        raise InputError(f"--strategy split splits text windows only, and the step {other.name} reads "
                         f"{other.reads}")
    return steps, contexts[-1]


def check_dataset(steps, record, store):
    """InputError naming the first step whose environment finds nothing to
    learn from in the dataset (wpt_training.Step.check_dataset)."""
    for number, step in enumerate(steps, 1):
        try:
            step.check_dataset(record, store)
        except ValueError as error:
            raise InputError(f"step {number} {step.name}: {error}") from None


def code_versions() -> dict:
    """{package: its version} of the code a step's result depends on."""
    versions = {}
    for package in CODE:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


# the part of the dataset record's `fingerprints` that a step reads, by the kind of examples it learns from
PART_READ = {WINDOWS: "text", "conversations": "conversations", "attempts": "tasks"}


def fingerprints(steps, model_config, record, seed, code) -> list[str]:
    """Each step's fingerprint, in order: the previous step's (for the first,
    the initial weights': the model's shape and vocabulary size, without
    block_size, which no weight depends on and a later step may change),
    the step (wpt_training.Step.record: its files, method, environment,
    optimizer, settings and locked parts), the fingerprint of the part of
    the dataset it reads (the record's `fingerprints`), the recipe's seed
    and the code versions. The step's position in the list, which its draws
    mix into the seed, is the length of the chain before it. No option is
    part of it."""
    shape = {key: value for key, value in asdict(model_config).items() if key != "block_size"}
    previous = fingerprint({"initial": shape})
    found = []
    for step in steps:
        previous = fingerprint({"previous": previous, "step": step.record(),
                                "data": record["fingerprints"][PART_READ[step.reads]], "seed": seed, "code": code})
        found.append(previous)
    return found


class StepResults:
    """The results of the steps runs made, in the store: ref steps/<the
    step's fingerprint> names a JSON object, {"about": what the run recorded
    of the step, "state": the object of its checkpoint state} (the weights,
    those of its best evaluation, the optimizer's, method's and
    environment's state, and the step's fingerprint as its signature;
    wpt_training.train), the bytes of the step's last checkpoint as they
    are. Setting a ref is recorded in the run's record first (RecordError
    otherwise, and the ref is not set). The main process alone reads and
    writes them."""

    KIND = "steps"

    def __init__(self, store, telemetry):
        self.store, self.telemetry = store, telemetry

    def object(self, mark) -> str | None:
        """The result object of the step of fingerprint mark, None when none is kept."""
        return self.store.get_ref(self.KIND, mark)

    def read(self, sha) -> dict:
        return json.loads(self.store.get_bytes(sha))

    def state(self, result: dict) -> dict:
        sha = result["state"]
        state = torch.load(self.store.path(sha), map_location="cpu", weights_only=True)  # path() checks the sha256
        self.store.release(sha)
        return state

    def keep(self, mark, state_sha, about) -> str:
        """Keep a step's checkpoint state, the store object state_sha; its result object."""
        sha = self.store.put_bytes(json.dumps({"about": about, "state": state_sha}, sort_keys=True).encode())
        set_ref(self.store, self.KIND, mark, sha, self.telemetry)
        return sha


def provenance(args, recipe, record, record_object, model_config, n_params, strategy, strategy_plugin, done):
    sources = {sid: source for step in done for sid, source in step["sources"].items()}
    versions = {}
    for package in (*CODE, "pyarrow", "wpt-train"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema": 5,
        "recipe_name": args.recipe,
        "recipe": recipe,  # everything that changes the model: data, architecture, seed, steps, drafter
        "dataset": record["id"],
        "dataset_record": record_object,
        "sources": dict(sorted(sources.items())),  # every step's, as the dataset record describes them
        "skipped_sources": record.get("skipped", []),  # admitted, but left out of the build: it failed
        "vocabulary": record["tokenizer"]["vocabulary"],
        "shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["shards"]],
        "held_out_shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["held_out_shards"]],
        "options": {k: v for k, v in vars(args).items() if k not in ("recipe", "dataset")},  # how the hardware ran it
        "strategy": {"name": strategy_plugin.name, "file": strategy_plugin.file, "file_sha256": strategy_plugin.sha256,
                     "processes": strategy.world_size},
        # each step in order: what it is, its fingerprint and kept result, whether this run reused it, what it
        # reached, and the sources it learned from
        "steps": [{**step, "sources": sorted(step["sources"])} for step in done],
        "model": asdict(model_config),
        "parameters": n_params,
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
            strategy_plugin, strategy = load_strategy(args.strategy)
    except InputError as error:
        _refuse(f"InputError: {error}", telemetry)
    except PluginError as error:
        _refuse(f"PluginError: {error}", telemetry)
    try:
        try:
            with telemetry.span("inputs.steps", steps=len(recipe["steps"])):
                steps, context = plan(recipe, strategy)
        except (InputError, PluginError) as error:
            raise Refused(f"{type(error).__name__}: {error}") from None
        _train(args, recipe, strategy, strategy_plugin, steps, context, store, telemetry)
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
    RecordError; nothing is deleted then: the previous object stays. kept:
    the objects that are also a kept step's state (StepResults), which only
    lose the ref, never their object."""

    KIND = "checkpoints"

    def __init__(self, store, name, telemetry, kept: set):
        self.store, self.name, self.telemetry, self.kept = store, name, telemetry, kept

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
        if previous and previous != sha and previous not in self.kept:
            self._record_deletion(previous)
            self.store.delete(previous)

    def remove(self):
        sha = self.store.get_ref(self.KIND, self.name)
        deleted = sha if sha and sha not in self.kept else None
        if deleted:
            self._record_deletion(deleted)
        self.store.delete_ref(self.KIND, self.name)
        if deleted:
            self.store.delete(deleted)

    def state_object(self) -> str:
        """The object of the saved state."""
        return self.store.get_ref(self.KIND, self.name)

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


def _state_object(checkpoint, store) -> str:
    """The store object of a step's last checkpoint, which becomes its kept
    state: a store checkpoint's own object, or a folder checkpoint's file
    put into the store."""
    if isinstance(checkpoint, StoreCheckpoint):
        return checkpoint.state_object()
    return store.put_file(checkpoint.path)


def _describe(error):
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def _train(args, recipe, strategy, strategy_plugin, steps, context, store, telemetry):
    log = telemetry.log if strategy.is_main else (lambda *_: None)
    out = Path(args.out)
    out_was_there = out.exists()  # a failed step then leaves the target as it was
    restarted = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")) > 0  # set by the launcher and by torchrun
    resume = args.resume or restarted
    try:
        with telemetry.span("inputs.dataset", dataset=args.dataset, stream=args.stream) as span:
            kept_states = set()  # the objects of the kept steps' states, which a store checkpoint never deletes
            checkpoint = (StoreCheckpoint(store, args.run, telemetry, kept_states) if args.run
                          else FolderCheckpoint(out / CHECKPOINT_FOLDER))
            if checkpoint.exists() and not resume:
                raise Refused(f"{checkpoint} holds an unfinished step: add --resume to continue it, or delete it")
            record, record_object, tokenizer = load_dataset(store, args.dataset, recipe)
            # what every step reads is there before the first one trains: the main process reads it, once
            problem = None
            if strategy.is_main:
                try:
                    check_dataset(steps, record, store)
                except InputError as error:
                    problem = str(error)
            problem = strategy.broadcast(problem)
            if problem:
                raise InputError(problem)
            span.set(shards=len(record["shards"]), held_out_shards=len(record["held_out_shards"]),
                     conversations=record.get("conversations", {}).get("count", 0),
                     tasks=record.get("tasks", {}).get("count", 0))
        telemetry.input(recipe=args.recipe, dataset=args.dataset, dataset_record=record_object,
                        vocabulary=record["tokenizer"]["vocabulary"])
    except (StoreError, InputError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    log(f"device: {describe(strategy.device)}" + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""))
    log(f"dataset {args.dataset}: vocabulary {tokenizer.vocab_size:,} tokens")

    seed = recipe["seed"]
    torch.manual_seed(seed)  # the same initial weights in every process
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=context, **recipe["model"])
    try:  # the drafter trained at the end, and the scheduler's pass times replayed: refused now, not after training
        drafter_config, drafting = drafter_settings(model_config, recipe["drafter"])  # its feature layers included
        setup = costs_setup(model_config, 1, drafter_config.block, strategy.device)  # what the times are measured on
        replayed = (replayed_costs(store.read_record(args.scheduler_costs, COSTS_FILE), setup)
                    if args.scheduler_costs else None)
    except (ValueError, StoreError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
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
    stream_shards = args.stream_shards if args.stream else None
    marks = fingerprints(steps, model_config, record, seed, code_versions())
    results = StepResults(store, telemetry)
    telemetry.event("training.settings", recipe=recipe, model=asdict(model_config), parameters=n_params,
                    device=describe(strategy.device), processes=strategy.world_size,
                    strategy={"name": strategy_plugin.name, "file_sha256": strategy_plugin.sha256},
                    steps=[{**step.record(), "fingerprint": mark} for step, mark in zip(steps, marks)],
                    resume=resume, checkpoint=str(checkpoint))

    # the last step whose result is kept: the run starts from it
    kept = strategy.broadcast([results.object(mark) for mark in marks] if strategy.is_main else None)
    reused = max((number for number, sha in enumerate(kept, 1) if sha), default=0)
    done = []
    if reused:
        state = None
        if strategy.is_main:
            kept_states.update(results.read(sha)["state"] for sha in kept if sha)
            found = [results.read(sha) for sha in kept[:reused]]
            done = [{**result["about"], "result": sha, "reused": True} for result, sha in zip(found, kept)]
            telemetry.input(reused_steps=dict(zip(marks[:reused], kept[:reused])))
            for step, about in zip(steps, done):
                telemetry.event("training.step_reused", step=step.name, number=about["number"],
                                fingerprint=about["fingerprint"], result=about["result"])
                log(f"step {about['number']} {step.name}: reused, result {about['result'][:12]}")
            state = results.state(found[-1])
        model.load_state_dict(strategy.broadcast(weights(state) if strategy.is_main else None))
        del state
    outputs = {}
    data_options = {"stream_shards": stream_shards, "micro_batch": args.micro_batch or 0}
    for number, (step, mark) in enumerate(zip(steps, marks), 1):
        if number <= reused:
            continue
        data = Data(step.name, store, record, tokenizer, model_config, strategy, telemetry, log, seed, number,
                    **data_options)
        try:
            with telemetry.span("step", step=step.name, number=number, fingerprint=mark) as span:
                method = step.start_method()
                environment = step.start_environment(data)
                try:
                    trained = train(model, strategy, step, method, environment, seed, number,
                                    Options(args.checkpoint_every, args.segment), checkpoint=checkpoint,
                                    resume=resume, identity=mark, done=marks[:number - 1], log=log,
                                    telemetry=telemetry, save_last=True)  # its result is kept from the checkpoint
                    sources = environment.sources if strategy.is_main else {}
                finally:
                    environment.close()
                span.set(**asdict(trained))
        except StepFailed as error:
            if strategy.is_main and not out_was_there and out.is_dir() and not any(out.iterdir()):
                out.rmdir()  # the folder the checkpoint alone made goes with it
            # a checkpoint written during the step stays: --resume continues the step from it
            left = strategy.is_main and checkpoint.exists()
            raise Refused(f"StepFailed: step {number} {step.name}: {error.reason}; "
                          + (f"its checkpoint stays in {checkpoint}: --resume continues the step from it, and "
                             "deleting it starts the step over" if left else "nothing of it is kept")
                          + (f"; the results of the {number - 1} steps before it stay in the store"
                             if number > 1 else "")) from None
        except (ValueError, SettingsMismatch, PluginError, StoreError, DatasetError) as error:
            raise Refused(f"{type(error).__name__}: step {number} {step.name}: {error}") from None
        if strategy.is_main:
            about = {"number": number, **step.record(), "fingerprint": mark, **_reached(trained),
                     "sources": dict(sorted(sources.items()))}
            with telemetry.span("training.step_kept", step=step.name, number=number, fingerprint=mark) as span:
                state = _state_object(checkpoint, store)
                sha = results.keep(mark, state, about)
                kept_states.add(state)  # the checkpoint's ref goes, its object stays as the step's state
                span.set(result=sha)
            outputs[mark] = sha
            telemetry.output(step_results=outputs)
            done.append({**about, "result": sha, "reused": False})
            checkpoint.remove()  # the step's result is kept
        strategy.barrier()
        resume = False  # the step that stopped, when one did, is done
    if not strategy.is_main:
        return
    _finish(args, recipe, record, record_object, model, model_config, n_params, strategy, strategy_plugin, done,
            drafter_config, drafting, setup, replayed, tokenizer, checkpoint, out, store, telemetry, log)


def _reached(trained) -> dict:
    """What a step's provenance says it reached (wpt_training.Trained)."""
    return {"best_validation_loss": trained.best_val,
            "updates": trained.steps,  # the last update made: fewer than its stages' when the optimizer stopped
            "stages": list(trained.stages),  # each stage's window length, updates, batch, first and last update
            "weights_update": trained.weights_step,  # the update whose weights the step leaves
            "indexer_loss": trained.indexer_loss,  # the CSA indexers' loss at that update (docs/model.md, "Attention")
            "skipped": trained.skipped}  # updates with nothing to learn


def _finish(args, recipe, record, record_object, model, model_config, n_params, strategy, strategy_plugin, done,
            drafter_config, drafting, setup, replayed, tokenizer, checkpoint, out, store, telemetry, log):
    """The drafter, the scheduler's pass times, then the model folder."""
    # a model that trained on turns ends its answers with <|im_end|> as well as the end of a document
    turns = any(step["reads"] != WINDOWS for step in done)
    try:
        with telemetry.span("drafter", **asdict(drafter_config)):
            drafter = train_drafter(model, (tokens for _, _, tokens in documents(store, record["shards"])),
                                    drafter_config, drafting, end=Tokenizer.END,
                                    stops=(Tokenizer.END, chat.IM_END) if turns else (Tokenizer.END,),
                                    device=strategy.device, telemetry=telemetry)
    except DrafterFailed as error:
        raise Refused(f"DrafterFailed: {error}; nothing was written; every step's result is kept in the store: "
                      "the same command trains the drafter again without training") from None
    drafter.provenance = {**drafter.provenance, "trained_by": {"command": "wpt-train", "recipe_name": args.recipe},
                          "dataset": args.dataset}
    # the scheduler's pass times, once per run, in the record; or those an earlier run recorded, replayed
    with telemetry.span("scheduler.costs", replayed=args.scheduler_costs):
        costs = replayed if replayed is not None else measure_costs(model, 1, drafter.block)
        telemetry.write_file(COSTS_FILE, costs_file(costs, setup))

    named = record["tokenizer"]
    # written beside the target, then renamed: a failure midway leaves the target as it was
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    renaming = False
    try:
        with telemetry.span("model.write", folder=str(out)):
            with telemetry.span("model.weights"):
                vocabulary = store.get_bytes(named["vocabulary"])
                tokenizer_name = f"{named['recipe']} version {named['recipe_version']}"
                save_model(model, staging, vocabulary, tokenizer_name, max_shard_bytes=args.max_shard_size,
                           metadata=WEIGHT_METADATA)
                save_drafter(drafter, staging, tokenizer_name, hashlib.sha256(vocabulary).hexdigest(),
                             metadata=WEIGHT_METADATA,
                             provenance={**drafter.provenance, "target": {"weights": weight_hashes(staging)}})
            with telemetry.span("model.provenance") as span:
                written = provenance(args, recipe, record, record_object, model_config, n_params, strategy,
                                     strategy_plugin, done)
                (staging / "provenance.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8",
                                                         newline="\n")
                span.set(sources=sorted(written["sources"]), skipped_sources=written["skipped_sources"])
            with telemetry.span("model.notice"):  # the credits each source asks for, NOTICE, LICENSE, the model card
                write_credits(staging, written, store.get_bytes)
                write_documents(staging)
            files = file_hashes(staging)
            telemetry.event("model.files", files=files)
            # recorded before the folder appears: a record that cannot be written leaves the target as it was
            telemetry.output(model_folder=str(out), weights={name: files[name] for name in weight_files(staging)},
                             provenance=files["provenance.json"], drafter=files["drafter/model.safetensors"])
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
        elif (renaming and not args.run and staging.exists() and out.is_dir()
              and {path.name for path in out.iterdir()} <= {CHECKPOINT_FOLDER}):
            # the target holds only what this run's checkpoint left there: no other model to go back to
            print(f"the new model could not take the place of {out}, which holds no other model: it is kept in "
                  f"{staging}; move it to {out} by hand", file=sys.stderr)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    checkpoint.remove()  # a checkpoint left by a step this run reused
    log(f"saved {out} (licenses and sources: {out / 'NOTICE'})")
    # show a sample so you immediately see what it learned
    model.eval()
    ids = sample_tokens(model, [Tokenizer.END], max_new_tokens=120, drafter=drafter, costs=costs)
    log(tokenizer.decode(list(ids)).decode("utf-8", errors="replace"))
