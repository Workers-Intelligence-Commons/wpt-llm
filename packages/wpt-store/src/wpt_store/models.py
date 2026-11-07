"""Model folders in the store: <id>/models/<name>/ holds the model's real
files, each under its own path, and FILES, {"files": {path inside the
folder: {"sha256", "size"}}}, which every read checks against.

    put_model(store, name, folder)    every file of folder into the session; what names them, for
                                      the caller to keep (recorded first)
    keep_model(store, name, files)    the model named: FILES written, last, and the index told
    has_model(store, name)            whether the session (or the one it started from) holds it
    model_file(store, name, path)     one file's bytes, None when it holds no such file
    copy_out(store, name, target)     the folder into target (a new folder), each file checked

A named run's checkpoint lies beside them until the model is kept: CHECKPOINT,
a record that names the state's file, which lies in the same folder
(wpt.train.StoreCheckpoint).

    checkpoint_keys(store, name)      the key of that record in the session, then in the one it started from
    state_key(store, name, sha256)    the key a checkpoint state of that sha256 is stored under, in the session
"""

import os
import shutil
import tempfile
from pathlib import Path

from .errors import CorruptFile, MissingFile
from .index import MODELS, held_row
from .names import File, check_name
from .settings import now

FILES = "files.json"  # the file of <id>/models/<name>/ that names the others (a model folder holds no file of that name)
CHECKPOINT = "checkpoint.json"  # the file of <id>/models/<name>/ that names a run's checkpoint state


def _prefix(session, name) -> str:
    check_name(name)
    return f"{session}/{MODELS}/{name}/"


def checkpoint_keys(store, name) -> list[str]:
    return [_prefix(session, name) + CHECKPOINT for session in store.reading()]


def state_key(store, name, sha256) -> str:
    return f"{_prefix(store.needed(), name)}state-{sha256}"


def put_model(store, name, folder) -> dict:
    """Every file of folder under the session's model name: {path: {"sha256",
    "size"}}, for keep_model. The model is not there until then."""
    folder, prefix, files = Path(folder), _prefix(store.needed(), name), {}
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        relative = path.relative_to(folder).as_posix()
        files[relative] = {"sha256": store.put_named(prefix + relative, path).sha256, "size": path.stat().st_size}
    return files


def keep_model(store, name, files: dict):
    """The session's model name is files (put_model): FILES written, then
    the index told."""
    session = store.needed()
    record = store.put_json(_prefix(session, name) + FILES, {"name": name, "time": now(), "files": files})
    store.note([held_row(session, MODELS, name, record)])


def _record(store, name) -> tuple[str, dict] | None:
    """(the key prefix of the model name, its FILES record) from the
    session, else from the one it started from; None when neither holds it."""
    for session in store.reading():
        prefix = _prefix(session, name)
        if (record := store.get_json(prefix + FILES)) is not None:
            return prefix, record
    return None


def has_model(store, name) -> bool:
    return _record(store, name) is not None


def model_file(store, name, path) -> bytes | None:
    """One file of the model name, checked against its sha256; None when
    the session holds no such model or file."""
    found = _record(store, name)
    if found is None or path not in found[1]["files"]:
        return None
    return store.get_bytes(File(found[0] + path, found[1]["files"][path]["sha256"]))


def copy_out(store, name, target) -> int:
    """The model folder kept under name, written into target, which must
    not exist yet: beside it first, then renamed, so a failure leaves no
    partial target. The number of its files. MissingFile when no model
    has that name; CorruptFile when a file does not match its sha256."""
    target = Path(target)
    found = _record(store, name)
    if found is None:
        raise MissingFile(f"no model {name} in session {store.session} of {store}")
    if target.exists():
        raise FileExistsError(f"{target} exists: copy the model into a new folder")
    prefix, record = found
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for path, entry in sorted(record["files"].items()):
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise CorruptFile(f"model {name}: {path!r} is not a path inside its folder")
            local = staging / path
            local.parent.mkdir(parents=True, exist_ok=True)
            file = File(prefix + path, entry["sha256"])
            shutil.copyfile(store.path(file), local)  # path() checks the sha256
            store.release(file)
            if local.stat().st_size != entry["size"]:
                raise CorruptFile(f"model {name}: {path} has {local.stat().st_size} bytes, not {entry['size']}")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return len(record["files"])
