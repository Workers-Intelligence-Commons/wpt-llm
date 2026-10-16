"""Train a TinyGPT on a built dataset.

Usage:
    wpt-train DATASET_ID [options]
    wpt-train DATASET_ID --strategy ddp --processes N [options]
    torchrun --nproc-per-node N -m wpt_train DATASET_ID --strategy ddp [options]

DATASET_ID is what `wpt-dataset build <recipe>` prints; the dataset comes
from the store named by WPT_STORE (environment or .env). Training never
builds anything: it reads the dataset and checks every input against its
recorded hash first.

--strategy names how the hardware is used (`single`: one device; `ddp`:
several processes, started by --processes on this machine or by torchrun
on one machine or several) and --method how the model learns
(`adamw`), each a built-in name or a .py file (see wpt_training). A method
has its own settings, changed with --set NAME=VALUE (repeatable).

--batch-size is the number of windows per update over every process
together, so the same options train the same way on any hardware. Each
process takes --micro-batch windows per pass (default: all of its share)
and adds passes up until the batch is complete. Lower --micro-batch when a
GPU runs out of memory; the training does not change.

Every option can also come from an environment variable, WPT_ plus the
option name in capitals with '_' for '-' (--n-layer -> WPT_N_LAYER;
WPT_SET holds NAME=VALUE pairs separated by spaces; WPT_RESUME=1); the
command line wins. The dataset id may come from WPT_DATASET.

Every --checkpoint-every steps the main process writes MODEL/checkpoint/;
after a crash, run the same command with --resume to continue (on other
hardware too). The checkpoint is removed once the model is saved.

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

from wpt_device import describe
from wpt_inference import sample_tokens
from wpt_model import ModelConfig, TinyGPT, save_model
from wpt_model.notice import EVALS_NAME, WEIGHT_METADATA, write_credits, write_documents
from wpt_store import StoreError, store_from_settings
from wpt_store.shards import ShardStream, sources_of, token_file
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_training import PluginError, RunSettings, SettingsMismatch, load_method, load_strategy, train
from wpt_training.launch import launch

SIZE_UNITS = {"": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9}
DEFAULT_MODEL = ModelConfig(vocab_size=1)
CHECKPOINT_FOLDER = "checkpoint"


class InputError(Exception):
    """The dataset is missing, its record describes another id, it has a
    record shape this version does not read, or the batch does not split
    over the processes. Nothing has been trained. (A shard or vocabulary
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
    ("block-size", int, DEFAULT_MODEL.block_size, "tokens the model sees at once"),
    ("n-layer", int, DEFAULT_MODEL.n_layer, "transformer blocks"),
    ("n-head", int, DEFAULT_MODEL.n_head, "attention heads per block"),
    ("n-embd", int, DEFAULT_MODEL.n_embd, "embedding width"),
    ("dropout", float, DEFAULT_MODEL.dropout, "dropout rate"),
    ("batch-size", int, 64, "windows per update, over every process together"),
    ("micro-batch", int, None, "windows per pass in one process (default: its whole share)"),
    ("steps", int, 3000, "maximum training steps"),
    ("eval-every", int, 200, "measure validation loss every N steps"),
    ("checkpoint-every", int, 500, "write a checkpoint every N steps (0: never)"),
    ("seed", int, 1337, "random seed"),
    ("strategy", str, "single", "how the hardware is used: a built-in name or a .py file"),
    ("processes", int, 1, "start this many processes on this machine, one per GPU (with --strategy ddp)"),
    ("method", str, "adamw", "how the model learns: a built-in name or a .py file"),
    ("max-shard-size", parse_size, None, "split the weights into files of at most this size, e.g. 500MB"),
    ("stream-shards", int, 8, "with --stream: dataset shards in use at once (windows are drawn from them)"),
]


def parse_args(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default=environ.get("WPT_DATASET"), help="built dataset id (WPT_DATASET)")
    for name, kind, default, text in OPTIONS:
        variable = "WPT_" + name.upper().replace("-", "_")
        if variable in environ:
            try:
                default = kind(environ[variable])
            except (ValueError, argparse.ArgumentTypeError) as error:
                parser.error(f"{variable}={environ[variable]!r}: {error}")
        parser.add_argument(f"--{name}", type=kind, default=default, help=f"{text} ({variable})")
    parser.add_argument("--set", action="append", default=environ.get("WPT_SET", "").split(), metavar="NAME=VALUE",
                        help="a setting of the method, repeatable (WPT_SET)")
    parser.add_argument("--stream", action="store_true", default=environ.get("WPT_STREAM", "") in ("1", "true", "yes"),
                        help="read the training shards a few at a time instead of copying them all to disk first "
                             "(WPT_STREAM)")
    parser.add_argument("--resume", action="store_true", default=environ.get("WPT_RESUME", "") in ("1", "true", "yes"),
                        help="continue from the model folder's checkpoint (WPT_RESUME)")
    args = parser.parse_args(argv)
    if not args.dataset:
        parser.error("give the dataset id (or set WPT_DATASET)")
    try:
        args.set = dict(pair.split("=", 1) for pair in args.set)
    except ValueError:
        parser.error("--set takes NAME=VALUE")
    return args


def load_dataset(store, dataset_id):
    """(record, record object, tokenizer), the record and vocabulary checked."""
    record_object = store.get_ref("datasets", dataset_id)
    if record_object is None:
        raise InputError(f"no dataset {dataset_id} in {store}; build it with `wpt-dataset build <recipe>`")
    record = json.loads(store.get_bytes(record_object))
    if record.get("id") != dataset_id:
        raise InputError(f"dataset record {record_object} describes {record.get('id')}, not {dataset_id}")
    if record.get("schema") != 3:
        raise InputError(f"dataset {dataset_id} has record shape {record.get('schema')}, this wpt-train reads 3; "
                         "build the recipe again with the current wpt-dataset")
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


def provenance(args, record, record_object, sources, model_config, n_params, strategy, plugins, best_val, batch):
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
        "schema": 3,
        "dataset": record["id"],
        "dataset_record": record_object,
        "sources": {sid: record["sources"][sid] for sid in sorted(sources)},  # address, revision, licenses
        "vocabulary": record["tokenizer"]["vocabulary"],
        "shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["shards"]],
        "held_out_shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["held_out_shards"]],
        "options": {k: v for k, v in vars(args).items() if k not in ("dataset", "set")},
        "strategy": {"name": strategy_plugin.name, "file": strategy_plugin.file, "file_sha256": strategy_plugin.sha256,
                     "processes": strategy.world_size},
        "method": {"name": method_plugin.name, "file": method_plugin.file, "file_sha256": method_plugin.sha256,
                   "settings": method_settings},
        "batch": batch,
        "model": asdict(model_config),
        "parameters": n_params,
        "best_validation_loss": best_val,
        "repository_commit": run("git", "rev-parse", "HEAD"),
        "repository_dirty": bool(status) if status is not None else None,
        "device": describe(strategy.device) + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""),
        "tools": {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION,
                  **versions},
    }


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # the sample is any text; Windows pipes default to a narrower code page
    args = parse_args(argv)
    if args.processes > 1 and "LOCAL_RANK" not in os.environ:  # not yet one of the started processes
        sys.exit(launch(args.processes, ["-m", "wpt_train", *(sys.argv[1:] if argv is None else argv)]))
    try:
        method_plugin, method_settings = load_method(args.method, args.set)
        strategy_plugin, strategy = load_strategy(args.strategy)
    except PluginError as error:
        sys.exit(f"PluginError: {error}")
    try:
        _train(args, strategy, (strategy_plugin, method_plugin, method_settings))
    finally:
        strategy.close()


def _train(args, strategy, plugins):
    log = print if strategy.is_main else (lambda *_: None)
    out = Path(args.out)
    checkpoint_folder = out / CHECKPOINT_FOLDER
    if checkpoint_folder.exists() and not args.resume:
        sys.exit(f"{checkpoint_folder} holds an unfinished run: add --resume to continue it, or delete it")
    try:
        store = store_from_settings(Path.cwd() / ".env")
        record, record_object, tokenizer = load_dataset(store, args.dataset)
        train_data, val_data = token_arrays(store, record, strategy, args.stream_shards if args.stream else None)
        settings = RunSettings(block_size=args.block_size, batch_size=args.batch_size,
                               micro_batch=args.micro_batch or args.batch_size // strategy.world_size,
                               steps=args.steps, eval_every=args.eval_every, checkpoint_every=args.checkpoint_every,
                               seed=args.seed)
    except (StoreError, InputError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
    if len(train_data) <= args.block_size + 1:
        sys.exit(f"the dataset has {len(train_data):,} training tokens, fewer than one window of {args.block_size}")
    log(f"device: {describe(strategy.device)}" + (f", {strategy.world_size} processes" if strategy.world_size > 1 else ""))
    log(f"dataset {args.dataset}: {len(train_data):,} training tokens, {len(val_data):,} held-out tokens, "
        f"vocabulary {tokenizer.vocab_size:,} tokens")

    torch.manual_seed(args.seed)  # the same initial weights in every process
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                               n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = TinyGPT(model_config).to(strategy.device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_params:,}")
    _, method_plugin, method_settings = plugins
    try:
        best_val = train(model, strategy, method_plugin, method_settings, train_data, val_data, settings,
                         checkpoint_folder=checkpoint_folder, resume=args.resume,
                         identity={"dataset": args.dataset, "model": asdict(model_config),
                                   # the shards in use decide the windows, so a resumed run must stream the same way
                                   **({"stream_shards": args.stream_shards} if args.stream else {})}, log=log)
    except (ValueError, SettingsMismatch, PluginError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
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
    written = provenance(args, record, record_object, sources, model_config, n_params, strategy, plugins, best_val, batch)
    (out / "provenance.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_credits(out, written, store.get_bytes)
    write_documents(out)
    shutil.rmtree(checkpoint_folder, ignore_errors=True)  # the run is complete
    log(f"saved {out} (licenses and sources: {out / 'NOTICE'})")
    # show a sample so you immediately see what it learned
    model.eval()
    ids = sample_tokens(model, [Tokenizer.END], context_size=model_config.block_size, max_new_tokens=120)
    log(tokenizer.decode(list(ids)).decode("utf-8", errors="replace"))
