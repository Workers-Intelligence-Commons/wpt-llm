"""A step's checkpoint state as a safetensors file, the only form a state
takes on disk, in the store and on the link between the wpt manager and a
gpu worker: loading one never runs code.

A state is what a step's parts hand to a checkpoint: dicts, lists, tuples,
numbers, strings, None and tensors, nested as deep as they go (the weights,
the optimizer's state with its integer keys, the method's and the
environment's). safetensors holds tensors only, so the file has two
layers:

    tensors   every tensor of the state, each under a name of its own; a tensor that
              another entry of the state also names (tied weights) is stored once
    layout    the state's shape without its tensors, JSON in the file's
              metadata under "layout": a dict is {"dict": [[key, value], ...]} (a key is
              a string, a number, a boolean, None or a tuple of those), a list
              {"list": [...]}, a tuple {"tuple": [...]}, a tensor {"tensor": its
              name}, a float that JSON cannot hold {"float": "nan" | "inf" | "-inf"};
              a string, an integer, a float, a boolean and None stand for themselves

save_state writes the file beside its place and renames it; load_state
gives the state back with its tensors on the CPU, or only the entries of
the top level it is asked for (the weights of a step's result, without
its optimizer's state). A value of another type is a TypeError, never
written.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAYOUT = "layout"


def save_state(state, path):
    """Write state to the file path, in place of the one there: a crash
    meanwhile leaves the previous one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors, seen = {}, {}
    layout = _encode(state, tensors, seen, "state")
    partial = path.with_name(path.name + ".partial")
    save_file(tensors, partial, metadata={LAYOUT: json.dumps(layout, separators=(",", ":"))})
    os.replace(partial, path)


def load_state(path, only=None):
    """The state the file path holds, its tensors on the CPU. only: names
    of the state's top-level entries; the state then holds those alone,
    and no other tensor is read."""
    with safe_open(path, framework="pt", device="cpu") as file:
        layout = json.loads((file.metadata() or {}).get(LAYOUT, "null"))
        if not isinstance(layout, dict) or "dict" not in layout:
            raise ValueError(f"{path}: not a state file (its layout is missing or is not a dict)")
        loaded = {}
        if only is not None:
            entries = [[key, value] for key, value in layout["dict"] if key in only]
            layout = {"dict": entries}
        return _decode(layout, file, loaded)


def _encode(value, tensors, seen, path):
    if isinstance(value, torch.Tensor):
        return {"tensor": _store(value, tensors, seen, path)}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, dict):
        return {"dict": [[_key(key), _encode(item, tensors, seen, f"{path}.{key}")] for key, item in value.items()]}
    if isinstance(value, list):
        return {"list": [_encode(item, tensors, seen, f"{path}.{index}") for index, item in enumerate(value)]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(item, tensors, seen, f"{path}.{index}") for index, item in enumerate(value)]}
    raise TypeError(f"{path}: a {type(value).__name__} is not a value a state holds (tensors, numbers, strings, "
                    "None, lists, tuples and dicts)")


def _key(key):
    if key is None or isinstance(key, (bool, int, str)):
        return key
    if isinstance(key, float) and math.isfinite(key):
        return key
    if isinstance(key, tuple):
        return {"tuple": [_key(item) for item in key]}
    raise TypeError(f"{key!r}: a {type(key).__name__} is not a key a state holds")


def _store(tensor, tensors, seen, path) -> str:
    """The name the tensor is stored under: the one it already has when the
    state names the same memory twice, a copy when only a part of that
    memory is named again."""
    tensor = tensor.detach()
    storage = tensor.untyped_storage().data_ptr() if tensor.device.type != "meta" else None
    key = (storage, tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
           str(tensor.device))
    if key in seen:
        return seen[key]
    if storage is not None and any(found[0] == storage for found in seen):
        tensor = tensor.clone()  # another view of the same memory: safetensors keeps no two tensors that overlap
    name = f"{len(tensors)}.{path}"
    tensors[name] = tensor.cpu().contiguous()
    seen[key] = name
    return name


def _decode(node, file, loaded):
    if not isinstance(node, dict):
        return node
    (kind, value), = node.items()
    if kind == "tensor":
        if value not in loaded:
            loaded[value] = file.get_tensor(value)
        return loaded[value]
    if kind == "float":
        return float(value)
    if kind == "list":
        return [_decode(item, file, loaded) for item in value]
    if kind == "tuple":
        return tuple(_decode(item, file, loaded) for item in value)
    if kind == "dict":
        return {_decode_key(key): _decode(item, file, loaded) for key, item in value}
    raise ValueError(f"{kind!r}: not a part of a state's layout")


def _decode_key(key):
    if isinstance(key, dict):
        return tuple(_decode_key(item) for item in key["tuple"])
    return key
