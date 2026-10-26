"""Model folders in the store, under a run's name: ref models/NAME names a
JSON object, the folder's manifest, {path inside the folder: {"sha256",
"size"}}, each file an object of its own.

    put_model(store, folder)          every file of folder into the store; the manifest's object,
                                      for the caller to set models/NAME to (recorded first)
    copy_out(store, name, target)     the folder models/NAME names, into target (a new folder),
                                      each file checked against its sha256
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .errors import CorruptObject, MissingObject

KIND = "models"


def put_model(store, folder) -> str:
    folder = Path(folder)
    manifest = {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        manifest[path.relative_to(folder).as_posix()] = {"sha256": store.put_file(path), "size": path.stat().st_size}
    return store.put_bytes(json.dumps(manifest, indent=1, sort_keys=True).encode())


def copy_out(store, name, target) -> int:
    """The model folder kept under name, written into target, which must
    not exist yet: beside it first, then renamed, so a failure leaves no
    partial target. The number of its files. MissingObject when no model
    has that name; CorruptObject when a file does not match its sha256."""
    target = Path(target)
    sha = store.get_ref(KIND, name)
    if sha is None:
        raise MissingObject(f"no model {name} in {store} (ref {KIND}/{name})")
    if target.exists():
        raise FileExistsError(f"{target} exists: copy the model into a new folder")
    manifest = json.loads(store.get_bytes(sha))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for path, entry in sorted(manifest.items()):
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise CorruptObject(f"model {name}: {path!r} is not a path inside its folder")
            local = staging / path
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store.path(entry["sha256"]), local)  # path() checks the sha256
            store.release(entry["sha256"])
            if local.stat().st_size != entry["size"]:
                raise CorruptObject(f"model {name}: {path} has {local.stat().st_size} bytes, not {entry['size']}")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return len(manifest)
