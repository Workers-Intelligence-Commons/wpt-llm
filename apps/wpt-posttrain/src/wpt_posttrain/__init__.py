"""Run post-training steps on a trained model into a new model folder.

Usage:
    wpt-posttrain BASE_FOLDER NEW_FOLDER STEP [STEP ...]

STEP is the name of a built-in step (the files in wpt_posttrain/steps/) or
the path of any Python file. A step file defines

    def run(context: wpt_posttrain.Context) -> wpt_posttrain.Outcome

and changes context.model in place. The steps run in order. Each one says
whether it succeeded or why it failed (Outcome.success() or
Outcome.failure(reason)); the first failure stops the run.

NEW_FOLDER is written only when every step succeeded: the weights, the
tokenizer, and the base model's provenance.json with a "post_training"
section (the base weights' sha256, each step's file sha256 and outcome, the
repository commit, the tools), then LICENSE, NOTICE and README.md. It must
not exist yet. BASE_FOLDER is never changed, and its evaluation results are
not copied: they describe other weights.

The store named by WPT_STORE (environment or .env) is open for steps that
read data from it.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from wpt_device import describe, select_device
from wpt_model import TinyGPT, load_model, save_model
from wpt_model.notice import WEIGHT_METADATA, write_documents
from wpt_store import Store, StoreError, store_from_settings
from wpt_tokenizer import RUSTC_VERSION, Tokenizer

BUILT_IN = Path(__file__).parent / "steps"
VERSION = importlib.metadata.version("wpt-posttrain")


class StepError(Exception):
    """A step cannot run: the base folder is not a trained model, NEW_FOLDER
    exists, the step is unknown, has no run(context), or returned something
    other than an Outcome. Nothing has been written."""


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
    model: TinyGPT  # the model the steps change, on device
    tokenizer: Tokenizer
    device: torch.device
    provenance: dict  # the base folder's provenance.json
    store: Store


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


def run_step(context, step) -> dict:
    """Run one step; return its record, or raise StepFailed."""
    name, path, file_name = find(step)
    spec = importlib.util.spec_from_file_location(f"wpt_posttrain_step_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise StepError(f"{path.name}: defines no run(context)")
    outcome = module.run(context)
    if not isinstance(outcome, Outcome):
        raise StepError(f"{path.name}: run(context) returned {type(outcome).__name__}, not wpt_posttrain.Outcome")
    if not outcome.succeeded:
        raise StepFailed(name, outcome.reason)
    return {"name": name, "file": file_name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "outcome": "succeeded"}


def open_context(base, store, device) -> Context:
    base = Path(base)
    try:
        provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise StepError(f"{base}: no provenance.json; give a folder that wpt-train wrote") from None
    if provenance.get("schema") != 3:
        raise StepError(f"{base}: provenance shape {provenance.get('schema')}, this wpt-posttrain reads 3")
    return Context(base, load_model(base, device), Tokenizer.load(base / "tokenizer.json"), device, provenance, store)


def post_train(context, steps, new_folder) -> Path:
    """Run every step, then write new_folder. A failure leaves nothing."""
    new_folder = Path(new_folder)
    if new_folder.exists():
        raise StepError(f"{new_folder} exists; give a new folder")
    for step in steps:
        find(step)  # an unknown step stops the run before any step has run
    records = [run_step(context, step) for step in steps]
    provenance = {**context.provenance, "post_training": {
        "base_weights": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(context.base.glob("model*.safetensors"))},
        "steps": records,
        **_repository(),
        "device": describe(context.device),
        "tools": {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION,
                  "torch": torch.__version__, "wpt-posttrain": VERSION},
    }}
    # written beside the target, then renamed: a failure midway leaves no new folder
    new_folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{new_folder.name}-", dir=new_folder.parent))
    try:
        save_model(context.model, staging, metadata=WEIGHT_METADATA)
        context.tokenizer.save(staging / "tokenizer.json")
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
        write_documents(staging)
        os.replace(staging, new_folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return new_folder


def _repository():
    def run(*command):
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("git", "status", "--porcelain")
    return {"repository_commit": run("git", "rev-parse", "HEAD"),
            "repository_dirty": bool(status) if status is not None else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base", help="model folder written by wpt-train")
    parser.add_argument("new", help="model folder to write; must not exist")
    parser.add_argument("steps", nargs="+", help="built-in step names or .py files, run in order")
    args = parser.parse_args(argv)
    try:
        device = select_device()
        print(f"device: {describe(device)}")
        context = open_context(args.base, store_from_settings(Path.cwd() / ".env"), device)
        new_folder = post_train(context, args.steps, args.new)
    except (StoreError, StepError, StepFailed) as error:
        sys.exit(f"{type(error).__name__}: {error}\nnothing was written")
    print(f"saved {new_folder} (licenses and sources: {new_folder / 'NOTICE'})")
