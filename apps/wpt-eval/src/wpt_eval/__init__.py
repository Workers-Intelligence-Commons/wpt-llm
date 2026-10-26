"""Run tests on a trained model and record their results in its model card.

Usage:
    wpt-eval MODEL_FOLDER [EVAL ...] [--attempts N]

EVAL is the name of a built-in test (the files in wpt_eval/evals/) or the
path of any Python file; with none, every built-in test runs. A test file
defines

    def run(context: wpt_eval.Context) -> wpt_eval.Result

The context gives the model, its tokenizer, the folder's provenance and the
documents of the dataset it was trained on, read from the store named by
WPT_STORE (environment or .env) and checked against their recorded hashes.
The result carries data for programs and the same findings as text for
people. The model is on a gpu worker (wpt_eval.remote, docs/machines.md,
"Evaluation"): the first of WPT_GPU_WORKERS, or a local one; a test reaches
it through the context only: context.model.config, .device (where it runs,
in words), .generator(seed), .writer(prompt, rows, ...) and
.greedy(prompts, steps).

The agent measures share one run of attempts (wpt_eval.attempts):
context.attempts(plain) plays N attempts (--attempts, WPT_ATTEMPTS, default
4) at every measure-only task of the repository holding the current folder
(read once per command), each in a container of the task's image (built
first with `wpt-agent build-image NAME`, as for finish-tasks), scored by the task's
check, once per command for each value of plain; every test of the run
reads the same records. Every attempt is also kept in the run's record
folder, one JSON line each in attempts.jsonl, a dropped one marked so.

Each result goes to MODEL_FOLDER/evals/<name>.json, and the model card
(README.md) is written again to show it. A test that fails stops the run;
results already written stay.

The run is recorded in the store, in its own folder under records/
(wpt_telemetry; docs/observability.md): the weights it measured, each test
with its file's sha256 and its result, and every file written. A test can
record its own events through context.telemetry. When the record cannot be
written, the run stops (at its next record call, or sooner when a send
in the background was refused; docs/observability.md) and exits with code
3.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wpt_agent import Unready
from wpt_model import DrafterMismatch, TokenizerMismatch, check_drafter, load_tokenizer, weight_hashes
from wpt_model.notice import CARD_NAME, EVALS_NAME, documents
from wpt_store import Store, StoreError, read_env_file, store_from_settings
from wpt_store.shards import documents as documents_of
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry, short
from wpt_tokenizer import Tokenizer
from wpt_utils.files import sha256_file
from wpt_utils.programs import code
from wpt_worker.link import LinkError

from . import attempts as shared
from .attempts import Record
from .remote import Remote, WorkerError

BUILT_IN = Path(__file__).parent / "evals"


class EvalError(Exception):
    """A test cannot run: the folder is not a trained model, the test is
    unknown, has no run(context), or returned something other than a Result.
    Nothing is written for that test."""


@dataclass(frozen=True)
class Result:
    data: dict  # numbers and names that json can hold, for programs
    text: str   # the same findings for people; shown in the model card


@dataclass(frozen=True)
class Document:
    id: str             # <source id>/<path>
    source: str
    tokens: np.ndarray  # uint32, ending with Tokenizer.END


@dataclass
class Context:
    folder: Path
    model: object  # on a gpu worker (wpt_eval.remote.Remote), or wpt_worker.evaluation.Served in this process
    tokenizer: Tokenizer
    device: str  # where the model runs, in words
    provenance: dict
    store: Store
    telemetry: Telemetry = field(default_factory=Telemetry.off)  # the run's record, for a test's own events
    attempts_per_task: int = shared.ATTEMPTS  # wpt-eval's --attempts
    _attempts: dict = field(default_factory=dict, repr=False)  # {plain: [Record]}, played once per command
    _tasks: list | None = field(default=None, repr=False)  # the measure-only tasks, read once per command

    def measure_tasks(self) -> list:
        """The measure-only tasks (wpt_eval.attempts.measure_tasks), read on
        the first call, then given again."""
        if self._tasks is None:
            self._tasks = shared.measure_tasks(self.store, self.provenance)
        return self._tasks

    def attempts(self, plain=False) -> list[Record]:
        """The shared run of attempts at the measure-only tasks
        (wpt_eval.attempts): attempts_per_task attempts at each, scored by
        its check, each offered its seeded variation of the tools, or with
        plain only the tools of the same variants of the task's needs,
        under their real names. Played on the first call for each value of
        plain, then given again to every later test of the same command;
        every attempt is also a line of attempts.jsonl in the run's record
        folder, and only the scored ones are records here. [] when there is
        no measure task. wpt_agent.Unready when the tasks cannot be read or
        an image is not built, which stops the command."""
        if plain not in self._attempts:
            self._attempts[plain] = shared.run(self, plain, self.attempts_per_task)
        return self._attempts[plain]

    def documents(self, part, passes=None) -> Iterator[Document]:
        """"training" or "held_out": each document of that part once, in
        shard order (a document repeated by the mixture appears once). They
        are read one shard at a time, each shard checked before its
        documents are given, so a dataset larger than memory can be read;
        call again to read them again. passes, a collections.Counter when
        given, counts how many times the part holds each document (its
        rows), whole once every document was read
        (wpt_store.shards.documents)."""
        if part not in ("training", "held_out"):
            raise ValueError(f"part {part!r}: choose 'training' or 'held_out'")
        for document, source, tokens in documents_of(
                self.store, self.provenance["shards" if part == "training" else "held_out_shards"], passes):
            yield Document(document, source, tokens)


def built_in_names() -> list[str]:
    return sorted(path.stem for path in BUILT_IN.glob("*.py") if not path.name.startswith("_"))


def find(eval_) -> tuple[str, Path, str]:
    """(result name, file, how the result names the file). A built-in is
    named by its package path; another file by its name only, so the model
    folder never records where it lives on this machine."""
    if eval_ in built_in_names():
        return eval_, BUILT_IN / f"{eval_}.py", f"wpt_eval/evals/{eval_}.py"
    path = Path(eval_)
    if path.suffix != ".py" or not path.is_file():
        raise EvalError(f"{eval_}: not a built-in test ({', '.join(built_in_names())}) nor a Python file")
    return path.stem, path, path.name


def run_eval(context, eval_) -> dict:
    """Run one test, write its result into the folder and return it."""
    name, path, file_name = find(eval_)
    spec = importlib.util.spec_from_file_location(f"wpt_eval_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise EvalError(f"{path.name}: defines no run(context)")
    file_sha256 = sha256_file(path)
    telemetry = context.telemetry
    with telemetry.span("eval", name=name, file=file_name, file_sha256=file_sha256):
        result = module.run(context)
        if not isinstance(result, Result):
            raise EvalError(f"{path.name}: run(context) returned {type(result).__name__}, not wpt_eval.Result")
        record = {"name": name, "file": file_name, "file_sha256": file_sha256,
                  "code": code(), "device": context.device, "data": result.data, "text": result.text}
        telemetry.event("eval.result", name=name, data=result.data)
        folder = context.folder / EVALS_NAME
        text = json.dumps(record, indent=2) + "\n"
        with telemetry.span("eval.write", file=f"{EVALS_NAME}/{name}.json"):
            # each file is in the record, sent to the store, before it is written
            telemetry.output(**{f"eval_{name}": hashlib.sha256(text.encode()).hexdigest()})
            telemetry.irreversible("eval.file_written", file=f"{EVALS_NAME}/{name}.json")
            folder.mkdir(exist_ok=True)
            (folder / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
            _write_documents(context.folder, telemetry)  # the model card shows it
    return record


def _write_documents(folder, telemetry):
    """Write the folder's LICENSE, NOTICE and model card again from its
    files, results included: under hidden names beside them, then, once
    the model card's sha256 is recorded and sent, renamed into place. A
    record that fails leaves the earlier copies."""
    staged = {}
    try:
        for name, text in documents(folder).items():
            descriptor, path = tempfile.mkstemp(dir=folder, prefix=f".{name}-")
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as out:
                out.write(text)
            staged[name] = Path(path)
        telemetry.output(model_card=sha256_file(staged[CARD_NAME]))
        telemetry.irreversible("model.documents_written", files=sorted(staged))
        for name, path in staged.items():
            os.replace(path, folder / name)
    except BaseException:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise


def open_context(folder, store, telemetry=None, attempts=shared.ATTEMPTS) -> Context:
    """The context of the model folder, with no model yet (main puts the
    gpu worker's there). EvalError when the folder is not one wpt-train
    wrote, its drafter was trained against other weights, or its tokenizer
    is not the weights'."""
    folder = Path(folder)
    try:
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvalError(f"{folder}: no provenance.json; give a folder that wpt-train wrote") from None
    if "steps" not in provenance:
        raise EvalError(f"{folder}: provenance.json lists no steps, so it is of another shape than this wpt-eval "
                        "reads; train it again")
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(folder))
        check_drafter(folder)  # refuses a folder whose drafter is missing or of other weights
    # ValueError: a vocabulary of another tokenizer version; DrafterMismatch is one too
    except (TokenizerMismatch, DrafterMismatch, ValueError) as error:
        raise EvalError(str(error)) from None
    return Context(folder, None, tokenizer, "", provenance, store, telemetry or Telemetry.off(), attempts)


def _count(text) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("at least 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="model folder written by wpt-train")
    parser.add_argument("evals", nargs="*", help=f"built-in test names or .py files (default: {', '.join(built_in_names())})")
    default = shared.ATTEMPTS
    if "WPT_ATTEMPTS" in os.environ:
        try:
            default = _count(os.environ["WPT_ATTEMPTS"])
        except argparse.ArgumentTypeError as error:
            parser.error(f"WPT_ATTEMPTS={os.environ['WPT_ATTEMPTS']!r}: {error}")
    parser.add_argument("--attempts", type=_count, default=default,
                        help=f"attempts at each measure task in the agent measures' shared run (WPT_ATTEMPTS; "
                             f"default {shared.ATTEMPTS})")
    args = parser.parse_args(argv)
    environ = {**read_env_file(Path.cwd() / ".env"), **{key: value for key, value in os.environ.items() if value}}
    try:
        store = store_from_settings(Path.cwd() / ".env")
    except StoreError as error:  # no store, no record: nothing has started
        sys.exit(f"{type(error).__name__}: {error}")
    evals = args.evals or built_in_names()
    try:
        weights = weight_hashes(args.model)
    except OSError:  # no model there: open_context names the problem, in the record
        weights = {}
    try:
        with Telemetry.start(store, "wpt-eval", sys.argv[1:] if argv is None else list(argv),
                             short(next(iter(weights.values()), "none")),
                             inputs={"model_folder": args.model, "evals": evals, "attempts": args.attempts},
                             env_file=Path.cwd() / ".env") as telemetry:
            try:
                with telemetry.span("inputs"):
                    for eval_ in evals:
                        find(eval_)  # an unknown test stops the run before any other has run
                    context = open_context(args.model, store, telemetry, args.attempts)
                    telemetry.input(weights=weights, dataset=context.provenance.get("dataset"))
                with telemetry.span("gpu_worker.model"), Remote(args.model, environ, telemetry.log) as model:
                    context.model, context.device = model, model.device
                    for eval_ in evals:
                        record = run_eval(context, eval_)
                        telemetry.log(f"\n{record['name']}\n{record['text'].rstrip()}")
            except (StoreError, EvalError, Unready, LinkError, WorkerError) as error:
                sys.exit(f"{type(error).__name__}: {error}")  # end.json records it
            telemetry.log(f"\nresults in {Path(args.model) / EVALS_NAME}, model card {Path(args.model) / 'README.md'}")
    except RecordError as error:
        print(f"RecordError: {error}\nstopped; nothing more was written", file=sys.stderr)
        sys.exit(RECORD_FAILED)
