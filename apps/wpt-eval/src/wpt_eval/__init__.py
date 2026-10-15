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
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wpt_device import describe, select_device
from wpt_model import TinyGPT, TokenizerMismatch, load_model, load_tokenizer
from wpt_model.notice import EVALS_NAME, write_documents
from wpt_store import Store, StoreError, store_from_settings
from wpt_store.shards import read_verified
from wpt_tokenizer import Tokenizer

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
    model: TinyGPT  # in eval mode, on device
    tokenizer: Tokenizer
    device: torch.device
    provenance: dict
    store: Store

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
    result = module.run(context)
    if not isinstance(result, Result):
        raise EvalError(f"{path.name}: run(context) returned {type(result).__name__}, not wpt_eval.Result")
    record = {"schema": 1, "name": name, "file": file_name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "wpt_eval": VERSION, "device": describe(context.device), "data": result.data, "text": result.text}
    folder = context.folder / EVALS_NAME
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_documents(context.folder)
    return record


def open_context(folder, store, device) -> Context:
    folder = Path(folder)
    try:
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvalError(f"{folder}: no provenance.json; give a folder that wpt-train wrote") from None
    if provenance.get("schema") != 3:
        raise EvalError(f"{folder}: provenance shape {provenance.get('schema')}, this wpt-eval reads 3; train it again")
    try:
        tokenizer = Tokenizer.from_bytes(load_tokenizer(folder))
    except (TokenizerMismatch, ValueError) as error:  # ValueError: a vocabulary of another tokenizer version
        raise EvalError(str(error)) from None
    model = load_model(folder, device)
    model.eval()
    return Context(folder, model, tokenizer, device, provenance, store)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="model folder written by wpt-train")
    parser.add_argument("evals", nargs="*", help=f"built-in test names or .py files (default: {', '.join(built_in_names())})")
    args = parser.parse_args(argv)
    try:
        evals = args.evals or built_in_names()
        for eval_ in evals:
            find(eval_)  # an unknown test stops the run before any other has run
        device = select_device()
        print(f"device: {describe(device)}")
        context = open_context(args.model, store_from_settings(Path.cwd() / ".env"), device)
        for eval_ in evals:
            record = run_eval(context, eval_)
            print(f"\n{record['name']}\n{record['text'].rstrip()}")
    except (StoreError, EvalError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
    print(f"\nresults in {Path(args.model) / EVALS_NAME}, model card {Path(args.model) / 'README.md'}")
