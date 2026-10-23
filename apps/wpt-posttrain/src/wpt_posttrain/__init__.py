"""Run a model recipe's post-training steps on a trained model into a new
model folder.

Usage:
    wpt-posttrain RECIPE BASE_FOLDER NEW_FOLDER

RECIPE is a file name in data/recipes/models/, without .json, in the
repository holding the current folder; its `posttrain` list names the
steps. Each step is the name of a built-in step (the files in
wpt_posttrain/steps/) or the path of any Python file, alone or as
{"step": NAME, "settings": {...}}. A step file defines

    SETTINGS = {name: default, ...}  # optional: the settings a recipe may change
    def check(settings)  # optional: raises ValueError, saying why, on settings that cannot go together
    def run(context: wpt_posttrain.Context) -> wpt_posttrain.Outcome

and changes context.model in place. context.settings holds its SETTINGS
with the recipe entry's settings, each of its default's type, as a
training method's (wpt_utils.settings); an unknown setting, one of
another type, or settings its check refuses stop the run before any step
runs. The steps run in order. Each one says
whether it succeeded or why it failed (Outcome.success() or
Outcome.failure(reason)); the first failure stops the run.

A step reads its training data with context.conversations(): the
conversations of every admitted source whose `chat` files carry a tag named
after the step (the `sft` step reads the files tagged `sft`), found in the
repository holding the current folder. The runner records those sources.
A step that trains on tasks reads them the same way with context.tasks():
the task folders whose task.json carries the step's name (the `grpo` step
reads those tagged `grpo`), or context.tasks(tag) those of another tag
(the `token-efficiency` step reads the tasks tagged `grpo`); the
measure-only tasks (tagged `eval`, or held out by the base model's recipe)
never reach a step.

After the last step, the run trains a fresh drafter for the new weights
(wpt_inference.drafting; docs/model.md, "The drafter"), the model frozen,
with the recipe's `drafter` section: its prompts are the beginnings of the
base model's training documents (the dataset its provenance names, read
from the store), and the post-trained model alone writes the answers. The
base folder's drafter is never used: it was trained against other weights.
A drafter that cannot be trained fails the run, and nothing is written.
Its training answers stop at the end of a document or at <|im_end|>, the
end of a turn, which a post-trained model writes. Then the run measures
the scheduler's pass times for the new weights once and writes them into
its record as scheduler-costs.json, with the setup they were measured on
(the model's config, the drafter's block, the device); --scheduler-costs
RECORD_FOLDER replays those an earlier run recorded instead, refused
before any step runs when they were measured on another setup.

NEW_FOLDER is written only when every step succeeded and the drafter was
trained: the weights, the tokenizer, the drafter (drafter/, with its own
provenance.json: the command and recipe that trained it, the dataset, the
sha256 of the weights it was trained against), and the base model's provenance.json with a "post_training"
section (the recipe's name, the base weights' sha256, each step's file
sha256, settings, outcome and the sources it read, those sources' revisions and licenses, the repository
commit, the tools), then LICENSE, NOTICE and README.md, which list the
sources read by the steps with the pre-training sources. It must
not exist yet. BASE_FOLDER is never changed, and its evaluation results are
not copied: they describe other weights.

The store named by WPT_STORE (environment or .env) is open for steps that
read data from it. The run is recorded there, in its own folder under
records/ (wpt_telemetry; docs/observability.md): the base weights, each
step with what it read and its outcome, and every file written. A step
records its own events, and writes its own files into that folder, through
context.telemetry. When the record cannot be written, the run stops at
once, writes nothing more and exits with code 3.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from wpt_dataset import conversations as chat_files
from wpt_dataset import tasks as task_files
from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_inference import COSTS_FILE, costs_file, costs_setup, measure_costs, replayed_costs
from wpt_inference.drafting import DrafterFailed, train_drafter
from wpt_inference.drafting import settings as drafter_settings
from wpt_model import (WPT, TokenizerMismatch, load_model, load_tokenizer, save_drafter, save_model,
                       tokenizer_identity, weight_files, weight_hashes)
from wpt_model.notice import CREDITS_NAME, WEIGHT_METADATA, write_documents
from wpt_store import Store, StoreError, store_from_settings
from wpt_store.shards import documents
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry, file_hashes, short
from wpt_tokenizer import RUSTC_VERSION, Tokenizer, chat
from wpt_utils.device import describe, select_device
from wpt_utils.files import sha256_file
from wpt_utils.programs import repository
from wpt_utils.settings import typed

BUILT_IN = Path(__file__).parent / "steps"
VERSION = importlib.metadata.version("wpt-posttrain")


class StepError(Exception):
    """A step cannot run: the recipe is missing, names no step or has
    drafter settings no drafter can have, a step's setting is unknown or of
    another type or its check refuses its settings, the scheduler's times to replay
    are missing or were measured on another setup, the base folder is not a trained
    model or its drafter is missing or was trained against other weights,
    NEW_FOLDER exists, the step is unknown,
    has no run(context), or returned something other than an Outcome.
    Nothing has been written."""


class StepFailed(Exception):
    """A step ran and reported failure; .step and .reason say which and why.
    Nothing has been written."""

    def __init__(self, step, reason):
        super().__init__(f"step {step} failed: {reason}")
        self.step, self.reason = step, reason


@dataclass(frozen=True)
class Outcome:
    succeeded: bool
    reason: str  # why it failed; empty when it succeeded

    @classmethod
    def success(cls):
        return cls(True, "")

    @classmethod
    def failure(cls, reason):
        if not reason:
            raise ValueError("a failure needs its reason")
        return cls(False, reason)


@dataclass
class Context:
    base: Path      # the base model folder, read only
    model: WPT  # the model the steps change, on device
    tokenizer: Tokenizer
    device: torch.device
    provenance: dict  # the base folder's provenance.json
    store: Store
    step: str = ""  # the running step's name, set by the runner
    settings: dict = field(default_factory=dict)  # the running step's settings, set by the runner
    sources: dict = field(default_factory=dict)  # the sources the running step read, recorded by the runner
    # the run's record: a step's own events (telemetry.event, .span, .log) and files (telemetry.write_file)
    telemetry: Telemetry = field(default_factory=Telemetry.off)

    def conversations(self) -> list[tuple[str, dict]]:
        """[(conversation id, conversation)] of the admitted sources whose
        chat files are tagged with the running step's name, in source and
        path order (docs/chat-format.md). Raises a wpt_dataset DatasetError
        when a source or a conversation breaks the rules."""
        found, sources = chat_files.read(Repo.find(), self.store, self.step, self.telemetry)
        self.sources.update(sources)
        return found

    def tasks(self, tag: str | None = None) -> list[task_files.Task]:
        """The tasks of the admitted sources whose task.json files are
        tagged `tag`, by default the running step's name, in source and
        folder order (docs/post-training.md), their pointers filled in; a
        step that trains on another step's tasks names that step's tag.
        The measure-only tasks are left out: those tagged `eval`, and the
        share of every task source that the base model's recipe holds out
        (its data.held_out, the rule that held its documents out), and the
        tasks pointing into the text of a held-out document of the base
        model's dataset (its document list). Raises a wpt_dataset
        DatasetError when a source or a task folder breaks the rules, or
        when tasks point into documents and that dataset has no document
        list."""
        found, sources = task_files.read(Repo.find(), self.store, tag or self.step,
                                         self.provenance["recipe"]["data"]["held_out"],
                                         dataset=self.provenance["dataset"])
        self.sources.update(sources)
        return found


def built_in_names() -> list[str]:
    return sorted(path.stem for path in BUILT_IN.glob("*.py") if not path.name.startswith("_"))


def find(step) -> tuple[str, Path, str]:
    """(step name, file, how the record names the file): a built-in by its
    package path, another file by its name only, so the model folder never
    records where it lives on this machine."""
    if step in built_in_names():
        return step, BUILT_IN / f"{step}.py", f"wpt_posttrain/steps/{step}.py"
    path = Path(step)
    if path.suffix != ".py" or not path.is_file():
        known = ", ".join(built_in_names()) or "none yet"
        raise StepError(f"{step}: not a built-in step ({known}) nor a Python file")
    return path.stem, path, path.name


def load(entry) -> tuple[str, Path, str, object, dict]:
    """(step name, file, how the record names the file, the loaded module,
    its settings) of a recipe's `posttrain` entry: a step, or {"step":
    step, "settings": {...}}, whose settings change the module's SETTINGS
    defaults, each of its default's type (wpt_utils.settings.typed), then
    pass the module's check(settings), when it has one."""
    step, overrides = (entry, {}) if isinstance(entry, str) else (entry["step"], entry["settings"])
    name, path, file_name = find(step)
    spec = importlib.util.spec_from_file_location(f"wpt_posttrain_step_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise StepError(f"{path.name}: defines no run(context)")
    try:
        settings = typed(getattr(module, "SETTINGS", {}), overrides, file_name)
    except ValueError as error:
        raise StepError(str(error)) from None
    try:
        getattr(module, "check", lambda settings: None)(settings)
    except ValueError as error:
        raise StepError(f"{file_name}: {error}") from None
    return name, path, file_name, module, settings


def run_step(context, entry) -> dict:
    """Run one step of the recipe's `posttrain` list; return its record, or
    raise StepFailed."""
    name, path, file_name, module, settings = load(entry)
    context.step, context.sources, context.settings = name, {}, settings
    file_sha256 = sha256_file(path)
    with context.telemetry.span("posttrain.step", step=name, file=file_name, file_sha256=file_sha256,
                                settings=settings) as span:
        outcome = module.run(context)
        if not isinstance(outcome, Outcome):
            raise StepError(f"{path.name}: run(context) returned {type(outcome).__name__}, not wpt_posttrain.Outcome")
        span.set(outcome="succeeded" if outcome.succeeded else "failed", reason=outcome.reason,
                 sources={sid: source["revision"] for sid, source in sorted(context.sources.items())})
        if not outcome.succeeded:
            raise StepFailed(name, outcome.reason)
    return {"name": name, "file": file_name, "file_sha256": file_sha256, "settings": settings,
            "outcome": "succeeded", "sources": dict(sorted(context.sources.items()))}


def open_context(base, store, device, telemetry=None) -> Context:
    base = Path(base)
    try:
        provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"{base}: no provenance.json; give a folder that wpt-train wrote") from None
    if provenance.get("schema") != 4:
        raise StepError(f"{base}: provenance shape {provenance.get('schema')}, this wpt-posttrain reads 4")
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(base))
        model = load_model(base, device)  # refuses a folder whose drafter is missing or of other weights
    # ValueError: a vocabulary of another tokenizer version, or a DrafterMismatch
    except (TokenizerMismatch, ValueError) as error:
        raise StepError(str(error)) from None
    return Context(base, model, tokenizer, device, provenance, store, telemetry=telemetry or Telemetry.off())


def recipe_parts(name) -> tuple[list, dict]:
    """(the post-training steps, each a name or {"step", "settings"}, the
    drafter section) of the model recipe of that name."""
    try:
        recipe = Repo.find().recipe("model", name)
    except DatasetError as error:
        raise StepError(str(error)) from None
    if not recipe["posttrain"]:
        raise StepError(f"model recipe {name!r} names no post-training step")
    return recipe["posttrain"], recipe["drafter"]


def post_train(context, recipe_name, steps, drafter_section, new_folder, base_weights, scheduler_costs=None) -> Path:
    """Run every step, train a fresh drafter for the new weights, measure
    the scheduler's pass times (or replay those of the record folder
    scheduler_costs) into the run's record, then write new_folder. A
    failure (StepFailed, DrafterFailed) leaves nothing. drafter_section:
    the recipe's `drafter` section; base_weights:
    wpt_model.weight_hashes(context.base)."""
    new_folder = Path(new_folder)
    if new_folder.exists():
        raise StepError(f"{new_folder} exists; give a new folder")
    for step in steps:
        load(step)  # an unknown step, or an unknown, mistyped or refused setting, stops the run before any step has run
    # settings no drafter can have, and times recorded on another setup, stop the run before any step has run
    try:
        drafter_config, drafting = drafter_settings(context.model.config, drafter_section)
        setup = costs_setup(context.model.config, 1, drafter_config.block, context.device)
        replayed = (replayed_costs(context.store.read_record(scheduler_costs, COSTS_FILE), setup)
                    if scheduler_costs else None)
    except (ValueError, StoreError) as error:
        raise StepError(str(error)) from None
    context.telemetry.input(recipe=recipe_name, base_weights=base_weights, dataset=context.provenance.get("dataset"))
    records = []
    for step in steps:
        records.append(run_step(context, step))
    # a fresh drafter, against the post-trained weights, on the beginnings of the base model's training documents
    with context.telemetry.span("drafter", **asdict(drafter_config)):
        beginnings = (tokens for _, _, tokens in documents(context.store, context.provenance["shards"]))
        # a post-trained model ends its answers with <|im_end|> as well as the end of a document
        drafter = train_drafter(context.model, beginnings, drafter_config, drafting, end=Tokenizer.END,
                                stops=(Tokenizer.END, chat.IM_END), device=context.device,
                                telemetry=context.telemetry)
    drafter.provenance = {**drafter.provenance,
                          "trained_by": {"command": "wpt-posttrain", "recipe_name": recipe_name},
                          "dataset": context.provenance.get("dataset")}
    # the scheduler's pass times for the new weights, once per run, in the record; or an earlier run's, replayed
    with context.telemetry.span("scheduler.costs", replayed=scheduler_costs):
        costs = replayed if replayed is not None else measure_costs(context.model, 1, drafter.block)
        context.telemetry.write_file(COSTS_FILE, costs_file(costs, setup))
    sources = {sid: source for record in records for sid, source in record["sources"].items()}
    context.telemetry.input(sources={sid: source["revision"] for sid, source in sorted(sources.items())})
    provenance = {**context.provenance, "post_training": {
        "recipe_name": recipe_name,
        "base_weights": base_weights,
        "steps": [{**record, "sources": sorted(record["sources"])} for record in records],
        "sources": dict(sorted(sources.items())),  # address, revision, licenses
        **repository(),  # repository_commit, repository_dirty
        "device": describe(context.device),
        "tools": {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION,
                  "torch": torch.__version__, "wpt-posttrain": VERSION},
    }}
    # written beside the target, then renamed: a failure midway leaves no new folder
    new_folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{new_folder.name}-", dir=new_folder.parent))
    telemetry = context.telemetry
    renaming = False
    try:
        with telemetry.span("model.write", folder=str(new_folder)):
            with telemetry.span("model.weights"):
                save_model(context.model, staging, load_tokenizer(context.base), tokenizer_identity(context.base)[0],
                           metadata=WEIGHT_METADATA)  # the base model's vocabulary, under the same name
                save_drafter(drafter, staging, *tokenizer_identity(context.base), metadata=WEIGHT_METADATA,
                             provenance={**drafter.provenance, "target": {"weights": weight_hashes(staging)}})
            with telemetry.span("model.provenance"):
                (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                         encoding="utf-8", newline="\n")
            with telemetry.span("model.notice"):  # the credits each source asks for, NOTICE, LICENSE, the model card
                if (context.base / CREDITS_NAME).is_dir():  # the base model's per-document credits still apply
                    shutil.copytree(context.base / CREDITS_NAME, staging / CREDITS_NAME)
                write_documents(staging)
            files = file_hashes(staging)
            telemetry.event("model.files", files=files)
            # recorded before the folder appears: a record that cannot be written leaves no new folder
            weights = {name: files[name] for name in weight_files(staging)}
            telemetry.output(model_folder=str(new_folder), weights=weights, provenance=files["provenance.json"],
                             drafter=files["drafter/model.safetensors"])
            telemetry.irreversible("model.folder_written", folder=str(new_folder))
            renaming = True
            os.replace(staging, new_folder)
    except BaseException:
        # the rename was recorded but did not happen: the finished model stays, named
        if renaming and not new_folder.exists() and staging.exists():
            print(f"{new_folder} could not be written: the new model is kept in {staging}; move it there by hand",
                  file=sys.stderr)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return new_folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", help="model recipe whose `posttrain` steps run: a file name in data/recipes/models/, "
                                       "without .json")
    parser.add_argument("base", help="model folder written by wpt-train")
    parser.add_argument("new", help="model folder to write; must not exist")
    parser.add_argument("--scheduler-costs", metavar="RECORD_FOLDER",
                        help="replay the drafter scheduler's pass times that an earlier run recorded on the same "
                             f"setup: the name of its record folder, which holds {COSTS_FILE} (default: measure "
                             "them)")
    args = parser.parse_args(argv)
    try:
        store = store_from_settings(Path.cwd() / ".env")
    except StoreError as error:  # no store, no record: nothing has started
        sys.exit(f"{type(error).__name__}: {error}\nnothing was written")
    try:
        base_weights = weight_hashes(args.base)
    except OSError:  # no model there: open_context names the problem, in the record
        base_weights = {}
    try:
        with Telemetry.start(store, "wpt-posttrain", sys.argv[1:] if argv is None else list(argv),
                             short(next(iter(base_weights.values()), "none")),
                             inputs={"recipe": args.recipe, "base_folder": args.base},
                             env_file=Path.cwd() / ".env") as telemetry:
            try:
                with telemetry.span("inputs"):
                    steps, drafter_section = recipe_parts(args.recipe)
                    device = select_device()
                    telemetry.log(f"device: {describe(device)}")
                    context = open_context(args.base, store, device, telemetry)
                new_folder = post_train(context, args.recipe, steps, drafter_section, args.new, base_weights,
                                        args.scheduler_costs)
            except (StoreError, StepError, StepFailed, DrafterFailed, DatasetError) as error:
                sys.exit(f"{type(error).__name__}: {error}\nnothing was written")  # end.json records it
            telemetry.log(f"saved {new_folder} (licenses and sources: {new_folder / 'NOTICE'})")
    except RecordError as error:
        print(f"RecordError: {error}\nstopped; nothing more was written", file=sys.stderr)
        sys.exit(RECORD_FAILED)
