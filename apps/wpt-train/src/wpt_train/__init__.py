"""Train a TinyGPT on a built dataset.

Usage:
    wpt-train DATASET_ID [options]

DATASET_ID is what `wpt-dataset build <recipe>` prints; the dataset comes
from the store named by WPT_STORE (environment or .env). Training never
builds anything: it reads the dataset and checks every input against its
recorded hash first.

Every option can also come from an environment variable, WPT_ plus the
option name in capitals with '_' for '-' (--n-layer -> WPT_N_LAYER); the
command line wins. The dataset id may come from WPT_DATASET.

The model folder gets config.json, the weights, tokenizer.json (the
dataset's vocabulary, for wpt-chat) and provenance.json (what produced it).
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from wpt_device import describe, select_device
from wpt_inference import sample_tokens
from wpt_model import ModelConfig, TinyGPT, save_model
from wpt_store import StoreError, store_from_settings
from wpt_store.shards import TokenHash, read_shard
from wpt_tokenizer import RUSTC_VERSION, Tokenizer
from wpt_training import TrainConfig, train

SIZE_UNITS = {"": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9}
DEFAULT_MODEL, DEFAULT_TRAINING = ModelConfig(vocab_size=1), TrainConfig()


class InputError(Exception):
    """The dataset, or one of its shards or its vocabulary, is missing or
    does not match its recorded hash. Nothing has been trained."""


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
    ("batch-size", int, None, "windows per step (default: 16 on CPU, 64 on a GPU)"),
    ("steps", int, DEFAULT_TRAINING.steps, "maximum training steps"),
    ("lr", float, DEFAULT_TRAINING.lr, "learning rate"),
    ("eval-every", int, DEFAULT_TRAINING.eval_every, "measure validation loss every N steps"),
    ("patience", int, DEFAULT_TRAINING.patience, "evaluations without improvement before stopping"),
    ("seed", int, 1337, "random seed"),
    ("max-shard-size", parse_size, None, "split the weights into files of at most this size, e.g. 500MB"),
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
    args = parser.parse_args(argv)
    if not args.dataset:
        parser.error("give the dataset id (or set WPT_DATASET)")
    return args


def load_dataset(store, dataset_id):
    """(record, record object, tokenizer, training tokens, held-out tokens), every input checked."""
    record_object = store.get_ref("datasets", dataset_id)
    if record_object is None:
        raise InputError(f"no dataset {dataset_id} in {store}; build it with `wpt-dataset build <recipe>`")
    record = json.loads(store.get_bytes(record_object))
    if record.get("id") != dataset_id:
        raise InputError(f"dataset record {record_object} describes {record.get('id')}, not {dataset_id}")
    tokenizer = Tokenizer.from_bytes(store.get_bytes(record["tokenizer"]["vocabulary"]))  # sha256-checked by the store
    return (record, record_object, tokenizer,
            _read_tokens(store, record["shards"]), _read_tokens(store, record["held_out_shards"]))


def _read_tokens(store, shards):
    parts = []
    for shard in shards:
        digest = TokenHash()
        for tokens, *rest in read_shard(store.path(shard["object"])):  # the file's sha256 is checked here
            digest.add(tokens, *rest)
            parts.append(tokens)
        if digest.hexdigest() != shard["token_sha256"]:
            raise InputError(f"shard {shard['object']}: token content does not match its recorded hash")
    joined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)
    return torch.from_numpy(joined.astype(np.int64))


def provenance(args, record, record_object, model_config, n_params, device, best_val):
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
    return {
        "schema": 1,
        "dataset": record["id"],
        "dataset_record": record_object,
        "vocabulary": record["tokenizer"]["vocabulary"],
        "shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["shards"]],
        "held_out_shards": [{"object": s["object"], "token_sha256": s["token_sha256"]} for s in record["held_out_shards"]],
        "options": {k: v for k, v in vars(args).items() if k != "dataset"},
        "model": asdict(model_config),
        "parameters": n_params,
        "best_validation_loss": best_val,
        "repository_commit": run("git", "rev-parse", "HEAD"),
        "repository_dirty": bool(status) if status is not None else None,
        "device": describe(device),
        "tools": {"python": platform.python_version(), "platform": platform.platform(), "rustc": RUSTC_VERSION,
                  **versions},
    }


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    try:
        store = store_from_settings(Path.cwd() / ".env")
        record, record_object, tokenizer, train_data, val_data = load_dataset(store, args.dataset)
    except (StoreError, InputError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
    if len(train_data) <= args.block_size + 1:
        sys.exit(f"the dataset has {len(train_data):,} training tokens, fewer than one window of {args.block_size}")
    device = select_device()
    print(f"device: {describe(device)}")
    print(f"dataset {args.dataset}: {len(train_data):,} training tokens, {len(val_data):,} held-out tokens, "
          f"vocabulary {tokenizer.vocab_size:,} tokens")

    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                               n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = TinyGPT(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    batch_size = args.batch_size or (16 if device.type == "cpu" else 64)  # GPUs want big batches
    train_config = TrainConfig(batch_size=batch_size, steps=args.steps, lr=args.lr,
                               eval_every=args.eval_every, patience=args.patience)
    best_val = train(model, train_data, val_data, block_size=model_config.block_size, config=train_config, device=device)

    out = Path(args.out)
    save_model(model, out, max_shard_bytes=args.max_shard_size)
    tokenizer.save(out / "tokenizer.json")
    record_args = argparse.Namespace(**{**vars(args), "batch_size": batch_size})
    (out / "provenance.json").write_text(
        json.dumps(provenance(record_args, record, record_object, model_config, n_params, device, best_val), indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"saved {out}")
    # show a sample so you immediately see what it learned
    model.eval()
    ids = sample_tokens(model, [Tokenizer.END], context_size=model_config.block_size, max_new_tokens=120)
    print(tokenizer.decode(list(ids)).decode("utf-8", errors="replace"))
