"""Save and load a TinyGPT as a model folder, in the Hugging Face layout:

    config.json                          model sizes (ModelConfig)
    model.safetensors                    all weights, when not split
    model-00001-of-00003.safetensors     or one file per shard,
    model.safetensors.index.json         plus an index: tensor name -> shard file

safetensors stores only tensors, so loading never runs code (unlike torch.load).
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import ModelConfig
from .gpt import TinyGPT

CONFIG_NAME = "config.json"
SINGLE_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"
METADATA = {"format": "pt"}


def save_model(model: TinyGPT, directory, max_shard_bytes=None):
    """Write the model folder. Replaces any model already saved there.

    config.json is written last, so a save that fails midway leaves a folder
    that load_model refuses instead of one that loads mismatched weights.

    max_shard_bytes=None writes one file; otherwise tensors are packed in order
    into files of at most that size (a single larger tensor gets its own file).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _remove_model(directory)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}

    shards = _split(state, max_shard_bytes)
    if len(shards) == 1:
        save_file(shards[0], directory / SINGLE_NAME, metadata=METADATA)
    else:
        _save_shards(shards, state, directory)
    model.config.save(directory / CONFIG_NAME)


def _save_shards(shards, state, directory):
    weight_map = {}
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, directory / name, metadata=METADATA)
        weight_map.update({key: name for key in shard})
    index = {
        "metadata": {"total_size": sum(_nbytes(t) for t in state.values())},
        "weight_map": weight_map,
    }
    (directory / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")


def load_model(directory, device="cpu") -> TinyGPT:
    """Rebuild the model from config.json and load its weights, split or not."""
    directory = Path(directory)
    config = ModelConfig.load(directory / CONFIG_NAME)
    index_path = directory / INDEX_NAME
    if index_path.exists():
        files = sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        files = [SINGLE_NAME]
    state = {}
    for name in files:
        state.update(load_file(directory / name))
    model = TinyGPT(config)
    model.load_state_dict(state)
    return model.to(device)


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
