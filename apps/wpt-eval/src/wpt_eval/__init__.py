"""Run tests on a trained model and record their results in its model card.

Usage:
    wpt-eval MODEL_FOLDER [EVAL ...]

EVAL is the name of a built-in test (the files in wpt_eval/evals/) or the
path of any Python file; with none, every built-in test runs. A test file
defines

    def run(context: wpt_eval.Context) -> wpt_eval.Result

The context gives the model, its tokenizer, the device, the folder's
provenance and the documents of the dataset it was trained on, read from the
store named by WPT_STORE (environment or .env) and checked against their
recorded hashes. The result carries data for programs and the same findings
as text for people.

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
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from wpt_model import WPT, TokenizerMismatch, load_model, load_tokenizer, weight_hashes
from wpt_model.notice import CARD_NAME, EVALS_NAME, documents
from wpt_store import Store, StoreError, store_from_settings
from wpt_store.shards import read_verified
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry, short
from wpt_tokenizer import Tokenizer
from wpt_utils.device import describe, select_device
from wpt_utils.files import sha256_file

BUILT_IN = Path(__file__).parent / "evals"
VERSION = importlib.metadata.version("wpt-eval")


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
    model: WPT  # in eval mode, on device
    tokenizer: Tokenizer
    device: torch.device
    provenance: dict
    store: Store
    telemetry: Telemetry = field(default_factory=Telemetry.off)  # the run's record, for a test's own events

    def documents(self, part) -> Iterator[Document]:
        """"training" or "held_out": each document of that part once, in
        shard order (a document repeated by the mixture appears once). They
        are read one shard at a time, each shard checked before its
        documents are given, so a dataset larger than memory can be read;
        call again to read them again."""
        if part not in ("training", "held_out"):
            raise ValueError(f"part {part!r}: choose 'training' or 'held_out'")
        seen = set()
        for shard in self.provenance["shards" if part == "training" else "held_out_shards"]:
            for tokens, source, document, _, _ in read_verified(self.store, shard):
                if document not in seen:
                    seen.add(document)
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
        record = {"schema": 1, "name": name, "file": file_name, "file_sha256": file_sha256,
                  "wpt_eval": VERSION, "device": describe(context.device), "data": result.data, "text": result.text}
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


def open_context(folder, store, device, telemetry=None) -> Context:
    folder = Path(folder)
    try:
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvalError(f"{folder}: no provenance.json; give a folder that wpt-train wrote") from None
    if provenance.get("schema") != 4:
        raise EvalError(f"{folder}: provenance shape {provenance.get('schema')}, this wpt-eval reads 4; train it again")
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(folder))
    except (TokenizerMismatch, ValueError) as error:  # ValueError: a vocabulary of another tokenizer version
        raise EvalError(str(error)) from None
    model = load_model(folder, device)
    model.eval()
    return Context(folder, model, tokenizer, device, provenance, store, telemetry or Telemetry.off())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="model folder written by wpt-train")
    parser.add_argument("evals", nargs="*", help=f"built-in test names or .py files (default: {', '.join(built_in_names())})")
    args = parser.parse_args(argv)
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
                             inputs={"model_folder": args.model, "evals": evals},
                             env_file=Path.cwd() / ".env") as telemetry:
            try:
                with telemetry.span("inputs"):
                    for eval_ in evals:
                        find(eval_)  # an unknown test stops the run before any other has run
                    device = select_device()
                    telemetry.log(f"device: {describe(device)}")
                    context = open_context(args.model, store, device, telemetry)
                    telemetry.input(weights=weights, dataset=context.provenance.get("dataset"))
                for eval_ in evals:
                    record = run_eval(context, eval_)
                    telemetry.log(f"\n{record['name']}\n{record['text'].rstrip()}")
            except (StoreError, EvalError) as error:
                sys.exit(f"{type(error).__name__}: {error}")  # end.json records it
            telemetry.log(f"\nresults in {Path(args.model) / EVALS_NAME}, model card {Path(args.model) / 'README.md'}")
    except RecordError as error:
        print(f"RecordError: {error}\nstopped; nothing more was written", file=sys.stderr)
        sys.exit(RECORD_FAILED)
