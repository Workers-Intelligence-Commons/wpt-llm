"""Save and load a WPT model as a model folder, in the Hugging Face layout:

    config.json                          the model's numbers (ModelConfig)
    model.safetensors                    all weights, when not split
    model-00001-of-00003.safetensors     or one file per shard,
    model.safetensors.index.json         plus an index: tensor name -> shard file
    tokenizer.json                       the vocabulary the model was trained with
    drafter/                             its drafter for speculative decoding (drafter.py)
    LICENSE, NOTICE, README.md           license documents and model card (notice.py)

safetensors stores only tensors, so loading never runs code (unlike torch.load).
Every weight file's header names the tokenizer (`tokenizer`, such as
"classics version 1") and the sha256 of tokenizer.json (`tokenizer_sha256`),
so a model never runs with another vocabulary than its own: load_tokenizer
refuses a tokenizer.json that does not match (TokenizerMismatch).

A model and its drafter are one unit, with separate weights: load_model
refuses a folder whose drafter is missing, or whose drafter's provenance
names other weights than the folder's (their sha256, weight_hashes), as
DrafterMismatch. No option skips it: wpt-train writes the drafter with
the weights it was trained against.
"""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from wpt_utils.files import sha256_file

from .config import ModelConfig
from .model import WPT
from .notice import DRAFTER_NAME

CONFIG_NAME = "config.json"
SINGLE_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"
TOKENIZER_NAME = "tokenizer.json"
METADATA = {"format": "pt"}


class DrafterMismatch(ValueError):
    """The model folder has no drafter, or its drafter was trained against
    other weights than the folder's."""


class TokenizerMismatch(ValueError):
    """tokenizer.json is missing or is not the vocabulary named in the
    weight files' headers (or the headers name none, or disagree)."""


def save_model(model: WPT, directory, tokenizer: bytes, tokenizer_name: str, max_shard_bytes=None, metadata=None):
    """Write the model folder. Replaces any model already saved there.

    tokenizer: the vocabulary file's bytes, written as tokenizer.json and
    named in every weight file's header with tokenizer_name.
    metadata: other text entries added to the header of every weight file
    (for example notice.WEIGHT_METADATA).

    config.json is written last, so a save that fails midway leaves a folder
    that load_model refuses instead of one that loads mismatched weights.

    max_shard_bytes=None writes one file; otherwise tensors are packed in order
    into files of at most that size (a single larger tensor gets its own file).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _remove_model(directory)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    metadata = {**(metadata or {}), "tokenizer": tokenizer_name,
                "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(), **METADATA}
    (directory / TOKENIZER_NAME).write_bytes(tokenizer)

    shards = _split(state, max_shard_bytes)
    if len(shards) == 1:
        save_file(shards[0], directory / SINGLE_NAME, metadata=metadata)
    else:
        _save_shards(shards, state, directory, metadata)
    model.config.save(directory / CONFIG_NAME)


def _save_shards(shards, state, directory, metadata):
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, directory / name, metadata=metadata)
        weight_map.update({key: name for key in shard})
    index = {
        "metadata": {"total_size": sum(_nbytes(t) for t in state.values())},
        "weight_map": weight_map,
    }
    (directory / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_model(directory, device="cpu") -> WPT:
    """Rebuild the model from config.json and load its weights, split or
    not; DrafterMismatch when its drafter is missing or was trained against
    other weights (check_drafter)."""
    directory = Path(directory)
    check_drafter(directory)
    config = ModelConfig.load(directory / CONFIG_NAME)
    state = {}
    for name in weight_files(directory):
        state.update(load_file(directory / name))
    model = WPT(config)
    model.load_state_dict(state)
    return model.to(device)


def check_drafter(directory):
    """DrafterMismatch unless the folder holds a drafter (drafter/) whose
    provenance names the folder's weights, by their sha256, as those it was
    trained against."""
    directory = Path(directory)
    drafter = directory / DRAFTER_NAME
    try:
        provenance = json.loads((drafter / "provenance.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DrafterMismatch(f"{directory}: no drafter ({DRAFTER_NAME}/); a model folder holds the drafter "
                              "wpt-train trained for its weights") from None
    if not (drafter / CONFIG_NAME).exists():
        raise DrafterMismatch(f"{directory}: its drafter ({DRAFTER_NAME}/) was not saved whole")
    trained = (provenance.get("target") or {}).get("weights")
    weights = weight_hashes(directory)
    if trained != weights:
        raise DrafterMismatch(f"{directory}: its drafter was trained against weights {trained}, the folder holds "
                              f"{weights}; train the model again to train its drafter")


def tokenizer_identity(directory) -> tuple[str, str]:
    """(tokenizer name, sha256 of tokenizer.json) as the weight files' headers name them."""
    directory = Path(directory)
    found = set()
    for name in weight_files(directory):
        with safe_open(directory / name, framework="pt") as weights:
            header = weights.metadata() or {}
        found.add((header.get("tokenizer"), header.get("tokenizer_sha256")))
    if len(found) != 1 or None in next(iter(found)):
        raise TokenizerMismatch(f"{directory}: the weight files name no single tokenizer ({sorted(found, key=str)}); "
                                "this folder was written before tokenizers were named, so train it again")
    return next(iter(found))


def load_tokenizer(directory) -> bytes:
    """The bytes of tokenizer.json, only when they are the vocabulary the weights name."""
    directory = Path(directory)
    name, expected = tokenizer_identity(directory)
    try:
        data = (directory / TOKENIZER_NAME).read_bytes()
    except FileNotFoundError:
        raise TokenizerMismatch(f"{directory}: no {TOKENIZER_NAME}; the weights need tokenizer {name}") from None
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise TokenizerMismatch(f"{directory}: {TOKENIZER_NAME} has sha256 {actual}, but the weights were trained "
                                f"with tokenizer {name}, sha256 {expected}")
    return data


def weight_files(directory) -> list[str]:
    """The names of a model folder's weight files, sorted: model.safetensors,
    or the shards its index names. What "the weights" of a folder are,
    wherever they are hashed or read."""
    directory = Path(directory)
    index_path = directory / INDEX_NAME
    if index_path.exists():
        return sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
    return [SINGLE_NAME]


def weight_hashes(directory) -> dict[str, str]:
    """{name: sha256} of each of weight_files(directory), hashed as the store
    hashes files (wpt_utils.files), so that every command names the same weights alike.
    OSError when one cannot be read."""
    directory = Path(directory)
    return {name: sha256_file(directory / name) for name in weight_files(directory)}


def _split(state, max_shard_bytes):
    if max_shard_bytes is None:
        return [state]
    shards, current, size = [], {}, 0
    for key, tensor in state.items():
        n = _nbytes(tensor)
        if current and size + n > max_shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[key] = tensor
        size += n
    shards.append(current)
    return shards


def _nbytes(tensor: torch.Tensor):
    return tensor.numel() * tensor.element_size()


def _remove_model(directory: Path):
    for path in directory.glob("model*.safetensors"):
        path.unlink()
    (directory / INDEX_NAME).unlink(missing_ok=True)
    (directory / CONFIG_NAME).unlink(missing_ok=True)
    (directory / TOKENIZER_NAME).unlink(missing_ok=True)
