"""Edit a source's files as plain files, and record the change as a patch.

    unpack(repo, source, store, folder)
        writes the source's final files (after cuts and patches) into folder
    write_patch(repo, source, store, folder, name)
        writes the difference between those files and folder as the source's
        next patch, NNNN-name.patch, through Repo.add_patch, and returns its
        place in the repository

Patches made this way are always well formed. The service layer checks the
result: the source with the new patch must give exactly the folder's files
and pass the rules of `wpt-dataset check --fetch`. Otherwise the patch file
is removed again and the error raised (a patch that applies to the wrong
result is PatchFailed).

The folder must lie outside the repository, so unpacked upstream content is
never committed by accident. On Windows, which has no executable bit, a
file keeps the mode it had in the source.
"""

import os
import stat
import tempfile
from pathlib import Path

from . import manifest
from .errors import EditError
from .manifest import EXECUTABLE, FILE, SYMLINK, Entry
from .repo import PATCH, Repo, Source
from .sources import load_index, source_tree, temporary_git


def unpack(repo: Repo, source: Source, store, folder) -> int:
    """Write the source's final files into folder (new or empty). Returns how many."""
    folder = _outside(repo, folder)
    if folder.exists() and any(folder.iterdir()):
        raise EditError(f"{folder} is not empty; unpack into a new folder")
    tree, _ = source_tree(source, store)
    for path, entry in tree.items():
        target = folder.joinpath(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = store.get_bytes(entry.sha256)
        if entry.mode == SYMLINK:
            os.symlink(content.decode(), target)
        else:
            target.write_bytes(content)
            if entry.mode == EXECUTABLE:
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    folder.mkdir(parents=True, exist_ok=True)
    return len(tree)


def write_patch(repo: Repo, source: Source, store, folder, name: str) -> str:
    """Record the difference between the source's final files and folder as
    its next patch, through the service layer (Repo.add_patch, which keeps
    it only if the source then gives exactly the folder's files and passes
    the rules of `wpt-dataset check --fetch`). Returns the patch's place in
    the repository."""
    folder = _outside(repo, folder)
    if not PATCH.fullmatch(f"0001-{name}.patch"):
        raise EditError(f"patch name {name!r}: use letters, digits, '.', '_' and '-'")
    before, _ = source_tree(source, store)
    after = read_folder(folder, store, before)
    manifest.check_paths(after)
    diff = _diff(before, after, store)
    if not diff:
        raise EditError(f"{folder} holds exactly the source's files: nothing to record")
    return repo.add_patch(source.id, name, diff, after, store)


def read_folder(folder: Path, store, known: dict[str, Entry]) -> dict[str, Entry]:
    """The folder as a tree, every content put into the store."""
    tree = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if path.is_symlink():
            tree[relative] = Entry(SYMLINK, store.put_bytes(os.readlink(path).replace("\\", "/").encode()))
        elif path.is_file():
            if os.name == "nt":
                mode = known[relative].mode if relative in known and known[relative].mode != SYMLINK else FILE
            else:
                mode = EXECUTABLE if path.stat().st_mode & stat.S_IXUSR else FILE
            tree[relative] = Entry(mode, store.put_bytes(path.read_bytes()))
        elif not path.is_dir():
            raise EditError(f"{path}: only files and symlinks can be part of a source")
    return tree


def _diff(before, after, store) -> bytes:
    with tempfile.TemporaryDirectory(prefix="wpt-edit-") as tmp:
        git, _ = temporary_git(tmp)
        trees = []
        for tree in (before, after):
            load_index(git, tree, store)
            trees.append(git("write-tree").decode().strip())
        return git("diff-tree", "-p", "--no-color", "--no-renames", "--full-index", *trees)


def _outside(repo: Repo, folder) -> Path:
    folder = Path(folder).resolve()
    if folder.is_relative_to(repo.root.resolve()):
        raise EditError(f"{folder} is inside the repository; use a folder outside it, so no file gets committed")
    return folder
