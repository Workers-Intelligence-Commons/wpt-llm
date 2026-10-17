"""Train a model as a model recipe says, on the dataset built from it.

Usage:
    wpt-train RECIPE DATASET_ID [options]
    wpt-train RECIPE DATASET_ID --strategy ddp --processes N [options]
    torchrun --nproc-per-node N -m wpt_train RECIPE DATASET_ID --strategy ddp [options]

RECIPE is a file name in data/recipes/models/, without .json, in the
repository holding the current folder. Everything that changes the
resulting model comes from it: the architecture's numbers, the method and
its settings, the steps, the batch and the seed. DATASET_ID is what
`wpt-dataset build RECIPE` prints; the dataset comes from the store named
by WPT_STORE (environment or .env), and a dataset built from another data
part than the recipe's is refused. Training never builds anything: it
reads the dataset and checks every input against its recorded hash first.

The options say only how the hardware runs it. --strategy names how the
hardware is used (`single`: one device; `ddp`: several processes, started
by --processes on this machine or by torchrun on one machine or several),
a built-in name or a .py file (see wpt_training).

The recipe's batch is the number of windows per update over every process
together, so the same recipe trains the same way on any hardware. Each
process takes --micro-batch windows per pass (default: all of its share)
and adds passes up until the batch is complete. Lower --micro-batch when a
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

Exit codes: 0 done, 2 refused (bad input, other settings than the
checkpoint's; never restarted), 130 stopped by a signal, others a crash.

The model folder gets config.json, the weights (their headers name the
licenses), tokenizer.json (the dataset's vocabulary, for wpt-chat),
provenance.json (what produced it, with the sources it learned from), and
LICENSE, NOTICE and README.md (the model card). Results that wpt-eval wrote
for a model saved there before are removed: they describe other weights.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_device import describe
from wpt_inference import sample_tokens
from wpt_model import WPT, ModelConfig, canonical_ids, save_model
from wpt_model.notice import EVALS_NAME, WEIGHT_METADATA, write_credits, write_documents
from wpt_model.sparse import KernelUnavailable, kernels
from wpt_store import NAME as STORE_NAME
from wpt_store import StoreError, store_from_settings
from wpt_store.shards import ShardStream, sources_of, token_file
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_training import (Checkpoint, FolderCheckpoint, Interrupted, PluginError, RunSettings, SettingsMismatch,
                          load_method, load_strategy, train)
from wpt_training.launch import REFUSED, STOPPED, launch

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
    ("micro-batch", int, None, "windows per pass in one process (default: its whole share)"),
    ("checkpoint-every", int, 500, "write a checkpoint every N steps (0: never)"),
    ("strategy", str, "single", "how the hardware is used: a built-in name or a .py file"),
    ("processes", int, 1, "start this many processes on this machine, one per GPU (with --strategy ddp)"),
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
    def run(*command):
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("git", "status", "--porcelain")
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
        "batch": batch,
        "model": asdict(model_config),
        "parameters": n_params,
        "best_validation_loss": trained.best_val,
        "steps_trained": trained.steps,  # the last step reached: fewer than the recipe's when the method stopped early
        "weights_step": trained.weights_step,  # the step whose weights the folder holds
        "indexer_loss": trained.indexer_loss,  # the CSA indexers' loss at that step (docs/model.md, "Attention")
        "repository_commit": run("git", "rev-parse", "HEAD"),
        "repository_dirty": bool(status) if status is not None else None,
        "device": describe(strategy.device) + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""),
        "tools": {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION,
                  **versions},
    }


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # the sample is any text; Windows pipes default to a narrower code page
    args = parse_args(argv)
    if (args.processes > 1 or args.max_restarts > 0) and "LOCAL_RANK" not in os.environ:  # not yet started by one
        sys.exit(launch(args.processes, ["-m", "wpt_train", *(sys.argv[1:] if argv is None else argv)],
                        max_restarts=args.max_restarts))
    try:
        recipe = load_recipe(args.recipe)
        training = recipe["training"]
        method_plugin, method_settings = load_method(training["method"], training["settings"])
        strategy_plugin, strategy = load_strategy(args.strategy)
    except InputError as error:
        _refuse(f"InputError: {error}")
    except PluginError as error:
        _refuse(f"PluginError: {error}")
    try:
        _train(args, recipe, strategy, (strategy_plugin, method_plugin, method_settings))
    except Refused as error:
        _refuse(str(error))
    except Interrupted as error:
        if strategy.is_main:
            print(error, file=sys.stderr)
        sys.exit(STOPPED)
    finally:
        strategy.close()


def _refuse(text):
    print(text, file=sys.stderr)
    sys.exit(REFUSED)


class StoreCheckpoint(Checkpoint):
    """The checkpoint of a named run in the store (ref checkpoints/NAME):
    any machine using the same store continues it. Each save replaces the
    previous one, which is then deleted."""

    KIND = "checkpoints"

    def __init__(self, store, name):
        self.store, self.name = store, name

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
        self.store.set_ref(self.KIND, self.name, sha)  # the previous one stays until this points elsewhere
        if previous and previous != sha:
            self.store.delete(previous)

    def remove(self):
        sha = self.store.get_ref(self.KIND, self.name)
        self.store.delete_ref(self.KIND, self.name)
        if sha:
            self.store.delete(sha)


def _train(args, recipe, strategy, plugins):
    log = print if strategy.is_main else (lambda *_: None)
    out = Path(args.out)
    restarted = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")) > 0  # set by the launcher and by torchrun
    resume = args.resume or restarted
    try:
        store = store_from_settings(Path.cwd() / ".env")
        checkpoint = StoreCheckpoint(store, args.run) if args.run else FolderCheckpoint(out / CHECKPOINT_FOLDER)
        if checkpoint.exists() and not resume:
            raise Refused(f"{checkpoint} holds an unfinished run: add --resume to continue it, or delete it")
        record, record_object, tokenizer = load_dataset(store, args.dataset, recipe)
        train_data, val_data = token_arrays(store, record, strategy, args.stream_shards if args.stream else None)
        shape, training = recipe["model"], recipe["training"]
        settings = RunSettings(block_size=shape["block_size"], batch_size=training["batch_size"],
                               micro_batch=args.micro_batch or training["batch_size"] // strategy.world_size,
                               steps=training["steps"], eval_every=training["eval_every"],
                               eval_batches=training["eval_batches"], checkpoint_every=args.checkpoint_every,
                               seed=training["seed"])
    except (StoreError, InputError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    if len(train_data) <= settings.block_size + 1:
        raise Refused(f"the dataset has {len(train_data):,} training tokens, fewer than one window of "
                      f"{settings.block_size}")
    log(f"device: {describe(strategy.device)}" + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""))
    log(f"dataset {args.dataset}: {len(train_data):,} training tokens, {len(val_data):,} held-out tokens, "
        f"vocabulary {tokenizer.vocab_size:,} tokens")

    torch.manual_seed(settings.seed)  # the same initial weights in every process
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **shape)
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
    _, method_plugin, method_settings = plugins
    try:
        trained = train(model, strategy, method_plugin, method_settings, train_data, val_data, settings,
                         checkpoint=checkpoint, resume=resume,
                         identity={"dataset": args.dataset, "model": asdict(model_config), "training": training,
                                   # the shards in use decide the windows, so a resumed run must stream the same way
                                   **({"stream_shards": args.stream_shards} if args.stream else {})}, log=log)
    except (ValueError, SettingsMismatch, PluginError) as error:
        raise Refused(f"{type(error).__name__}: {error}") from None
    finally:
        if isinstance(train_data, ShardStream):
            train_data.close()
    if not strategy.is_main:
        return

    batch = {"windows": settings.batch_size, "micro_batch": settings.micro_batch, "processes": strategy.world_size,
             "accumulation": settings.batch_size // (settings.micro_batch * strategy.world_size)}
    named = record["tokenizer"]
    save_model(model, out, store.get_bytes(named["vocabulary"]), f"{named['recipe']} version {named['recipe_version']}",
               max_shard_bytes=args.max_shard_size, metadata=WEIGHT_METADATA)
    shutil.rmtree(out / EVALS_NAME, ignore_errors=True)
    sources = sources_of(store, record["shards"])
    written = provenance(args, recipe, record, record_object, sources, model_config, n_params, strategy, plugins,
                         trained, batch)
    (out / "provenance.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_credits(out, written, store.get_bytes)
    write_documents(out)
    checkpoint.remove()  # the run is complete
    log(f"saved {out} (licenses and sources: {out / 'NOTICE'})")
    # show a sample so you immediately see what it learned
    model.eval()
    ids = sample_tokens(model, [Tokenizer.END], context_size=model_config.block_size, max_new_tokens=120)
    log(tokenizer.decode(list(ids)).decode("utf-8", errors="replace"))
