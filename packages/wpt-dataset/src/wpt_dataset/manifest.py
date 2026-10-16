"""Manifests: the exact list of what a source contains, and its checksum.

One line per entry, sorted by the UTF-8 bytes of the path:

    <mode> <sha256> <path>

mode 100644 is a file, 100755 an executable file, 120000 a symlink (its
sha256 is that of the target text). Empty folders do not appear. The
source checksum is "sha256:" + the sha256 of the manifest's bytes.

Contents live in the store under their sha256, so a tree is just a mapping
path -> Entry. Readers build trees from the downloaded data itself (archive
headers, git or mercurial objects), never from files written to a disk, so
Linux and Windows compute the same manifest.
"""

import concurrent.futures
import contextlib
import hashlib
import os
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import BadPath, FetchFailed, UnsupportedEntry
from .programs import program

FILE, EXECUTABLE, SYMLINK = "100644", "100755", "120000"
MODES = (FILE, EXECUTABLE, SYMLINK)
SUBMODULE = "160000"  # a git tree's pointer to a commit of another repository


@dataclass(frozen=True)
class Entry:
    mode: str
    sha256: str


def manifest_bytes(tree: dict[str, Entry]) -> bytes:
    lines = (f"{e.mode} {e.sha256} {p}\n" for p, e in sorted(tree.items(), key=lambda item: item[0].encode()))
    return "".join(lines).encode()


def checksum(tree: dict[str, Entry]) -> str:
    return "sha256:" + hashlib.sha256(manifest_bytes(tree)).hexdigest()


def parse_manifest(data: bytes) -> dict[str, Entry]:
    tree = {}
    for line in data.decode().splitlines():
        mode, sha, path = line.split(" ", 2)
        tree[path] = Entry(mode, sha)
    return tree


def glob_match(path: str, pattern: str) -> bool:
    """`*` matches within one folder, `**` any number of folders."""
    return PurePosixPath(path).full_match(pattern)


def check_paths(tree: dict[str, Entry]):
    """Paths must work on Linux and Windows alike."""
    seen = {}
    for path in tree:
        parts = path.split("/")
        if path.startswith("/") or any(p in ("", ".", "..") for p in parts) or "\\" in path or "\0" in path:
            raise BadPath(f"{path!r}: paths are relative, use '/', and have no empty, '.' or '..' parts")
        folded = path.casefold()
        if folded in seen:
            raise BadPath(f"{seen[folded]!r} and {path!r} differ only by letter case")
        seen[folded] = path


class Extractor:
    """Applies a source's `extract` rules while entries are read, storing
    each kept entry's content in the store."""

    def __init__(self, store, extract: dict):
        self.store = store
        self.strip = extract.get("strip", 0)
        self.exclude = extract.get("exclude", [])
        self.tree: dict[str, Entry] = {}

    def kept(self, raw_path: bytes | str) -> str | None:
        """The path an entry keeps after `strip`, or None when it is stripped
        away or excluded."""
        if isinstance(raw_path, bytes):
            try:
                raw_path = raw_path.decode("utf-8")
            except UnicodeDecodeError as error:
                raise BadPath(f"{raw_path!r} is not valid UTF-8") from error
        parts = [p for p in raw_path.split("/") if p not in ("", ".")]
        if len(parts) <= self.strip:
            return None  # gone with the stripped folders, like tar --strip-components
        path = "/".join(parts[self.strip:])
        if any(glob_match(path, pattern) for pattern in self.exclude):
            return None
        return path

    def add(self, raw_path: bytes | str, mode: str, content: bytes):
        path = self.kept(raw_path)
        if path is not None:
            self._record(path, mode, self.store.put_bytes(content))

    def add_file(self, path: str, mode: str, file: Path):
        """A kept path (see kept) whose content is a file on disk, stored
        without reading it into memory."""
        self._record(path, mode, self.store.put_file(file))

    def add_known(self, path: str, mode: str, sha256: str):
        """A kept path known only by its content's sha256, which is not
        stored: for a manifest computed from an upstream's own listing."""
        self._record(path, mode, sha256)

    def _record(self, path: str, mode: str, sha256: str):
        if path in self.tree:
            raise BadPath(f"{path!r} appears twice")
        self.tree[path] = Entry(mode, sha256)

    def finish(self) -> dict[str, Entry]:
        check_paths(self.tree)
        return self.tree


def _file_mode(unix_mode: int) -> str:
    return EXECUTABLE if unix_mode & 0o111 else FILE


def read_tar(path: Path, extractor: Extractor):
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if member.isdir():
                continue
            if member.issym():
                extractor.add(member.name, SYMLINK, member.linkname.encode())
            elif member.isfile() or member.islnk():  # a hard link holds its target's content
                extractor.add(member.name, _file_mode(member.mode), archive.extractfile(member).read())
            else:
                raise UnsupportedEntry(f"{member.name}: tar entry type {member.type!r} is neither a file nor a symlink")


def read_zip(path: Path, extractor: Extractor):
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            unix_mode = info.external_attr >> 16 if info.create_system == 3 else 0
            content = archive.read(info)
            if stat.S_ISLNK(unix_mode):
                extractor.add(info.filename, SYMLINK, content)
            elif unix_mode == 0 or stat.S_ISREG(unix_mode):
                extractor.add(info.filename, _file_mode(unix_mode), content)
            else:
                raise UnsupportedEntry(f"{info.filename}: zip entry is neither a file nor a symlink")


def read_archive(path: Path, extractor: Extractor):
    if zipfile.is_zipfile(path):
        read_zip(path, extractor)
    else:
        try:
            read_tar(path, extractor)
        except tarfile.ReadError as error:
            raise UnsupportedEntry(f"{path.name}: neither a zip nor a tar archive") from error


def read_git(repo: Path, revision: str, extractor: Extractor, prefix: str = "") -> dict[str, str]:
    """Entries of a commit, from git's own objects, each path after `prefix`
    (a submodule's folder, ending in "/"). An excluded file is never read, so
    a partial fetch need not hold it. Returns the kept submodules, each path
    with the commit it pins; an excluded submodule is skipped."""
    entries, submodules = [], {}
    for path, mode, kind, object_id in _git_tree(repo, revision, prefix):
        if extractor.kept(path) is None:
            continue
        if mode == SUBMODULE:
            submodules[path.decode()] = object_id
        elif kind != "blob" or mode not in MODES:
            raise UnsupportedEntry(f"{path.decode(errors='replace')}: git {kind} with mode {mode}")
        else:
            entries.append((path, mode, object_id))
    with contextlib.closing(_git_blobs(repo, [e[2] for e in entries])) as contents:
        for (path, mode, _), content in zip(entries, contents):
            extractor.add(path, mode, content)
    return submodules


def git_blobs_needed(repo: Path, revision: str, extractor: Extractor, since: str | None = None,
                     prefix: str = "") -> list[str]:
    """The blobs read_git and read_git_history read: the kept files of the
    commit and, with `since`, both sides of every kept file a commit since
    then changes. A partial fetch downloads only these."""
    needed = [object_id for path, _, kind, object_id in _git_tree(repo, revision, prefix)
              if kind == "blob" and extractor.kept(path) is not None]
    if since:
        for changes in _changed_paths(repo, _history_commits(repo, since, revision)).values():
            needed += [object_id for path, blobs in changes if extractor.kept(path) is not None for object_id in blobs]
    return list(dict.fromkeys(needed))


def missing_objects(repo: Path, object_ids: list[str]) -> list[str]:
    """The objects among object_ids that the repository does not hold."""
    if not object_ids:
        return []
    out = _git(repo, "cat-file", "--batch-check", input="".join(o + "\n" for o in object_ids).encode())
    return [line.split()[0] for line in out.decode().splitlines() if line.endswith(" missing")]


def _git_tree(repo, revision, prefix):
    """(path with prefix, mode, kind, object id) of each entry of a commit."""
    for record in filter(None, _git(repo, "ls-tree", "-r", "-z", "--full-tree", revision).split(b"\0")):
        meta, path = record.split(b"\t", 1)
        mode, kind, object_id = meta.decode().split()
        yield prefix.encode() + path, mode, kind, object_id


HISTORY = ".git/history/"
# git's output must not depend on the settings of the machine that fetches
PLAIN_GIT = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
             # an object a partial fetch left out is an error, never a download of its own
             "GIT_NO_LAZY_FETCH": "1"}


def read_git_history(repo: Path, since: str, revision: str, extractor: Extractor):
    """One file per commit from `since` to `revision`, merges left out:
    `.git/history/<commit>`, holding what `git format-patch` writes for the
    commit (author, date, message, changed files, diff), limited to the
    paths the extractor keeps. A commit that changes no kept path gives no
    file. git never allows a `.git` folder in a tree, so these paths cannot
    meet a file of the commit itself."""
    commits = _history_commits(repo, since, revision)
    changed = {commit: [path for path, _ in changes] for commit, changes in _changed_paths(repo, commits).items()}

    def patch(commit):
        paths = [p for p in changed[commit] if extractor.kept(p) is not None]
        if not paths:
            return commit, None
        limit = [] if len(paths) == len(changed[commit]) else ["--", *(f":(literal){p}" for p in paths)]
        text = _git(repo, "format-patch", "-1", "--root", "--stdout", "--no-signature", "--no-renames", "--no-binary",
                    commit, *limit)
        # a diff may hold bytes in any encoding; the file must be UTF-8 text
        return commit, text.decode("utf-8", errors="replace").encode("utf-8")

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for commit, text in pool.map(patch, commits):
            if text is not None:
                extractor.add(HISTORY + commit, FILE, text)


def _history_commits(repo, since, revision) -> list[str]:
    """The commits from `since` to `revision`, merges left out, newest first."""
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", since, revision],
                              capture_output=True, env=PLAIN_GIT)
    if ancestor.returncode != 0:
        raise UnsupportedEntry(f"history.since {since} is not an ancestor of the revision {revision}")
    parents = _git(repo, "rev-parse", f"{since}^@").decode().split()
    return _git(repo, "rev-list", "--no-merges", revision, "--not", *parents).decode().split()


def _changed_paths(repo, commits) -> dict[str, list[tuple[str, list[str]]]]:
    """Each commit's changed paths, with the blobs on both sides of the
    change (none for a submodule, or for the side where the file is
    absent), from one `git diff-tree --stdin`."""
    if not commits:
        return {}
    out = _git(repo, "diff-tree", "--stdin", "--always", "--root", "-r", "-z", "--no-renames",
               input="".join(c + "\n" for c in commits).encode())
    changed, current, tokens = {}, None, iter(out.split(b"\0"))
    for token in tokens:
        if token.startswith(b":"):  # ":<old mode> <new mode> <old blob> <new blob> <status>", then the path
            old_mode, new_mode, old, new, _ = token[1:].decode().split()
            path = next(tokens).decode("utf-8", errors="surrogateescape")
            blobs = [b for m, b in ((old_mode, old), (new_mode, new)) if m != SUBMODULE and b.strip("0")]
            changed[current].append((path, blobs))
        elif token.strip():
            current = token.decode().strip()
            changed[current] = []
    return changed


def _git(repo, *args, input=None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=input, capture_output=True, check=True,
                          env=PLAIN_GIT).stdout


def _git_blobs(repo, object_ids):
    """Contents of blobs, in order, through one `git cat-file --batch`. Close
    the generator when done (contextlib.closing): the git process stops only
    then, and on Windows it keeps its folder locked until it stops."""
    if not object_ids:
        return
    process = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=PLAIN_GIT)
    try:
        for object_id in object_ids:
            process.stdin.write(object_id.encode() + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if header[1:] == [b"missing"]:
                raise FetchFailed(f"git object {object_id} was not fetched")
            size = int(header[2])
            content = process.stdout.read(size)
            process.stdout.read(1)  # the newline after each object
            yield content
    finally:
        process.stdin.close()
        process.wait()


def read_mercurial(repo: Path, revision: str, extractor: Extractor):
    """Entries of a changeset, from mercurial's manifest and `hg cat`."""
    hg = program("hg")
    listing = subprocess.run([hg, "-R", str(repo), "manifest", "-v", "-r", revision],
                             capture_output=True, check=True).stdout
    for line in listing.splitlines():
        # "644   path", "755 * path", "644 @ path" (symlink)
        perms, flag, raw_path = line[:3], line[4:5], line[6:]
        mode = SYMLINK if flag == b"@" else (EXECUTABLE if flag == b"*" else FILE)
        if perms not in (b"644", b"755"):
            raise UnsupportedEntry(f"{raw_path!r}: unexpected mercurial manifest line {line!r}")
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BadPath(f"{raw_path!r} is not valid UTF-8") from error
        content = subprocess.run([hg, "-R", str(repo), "cat", "-r", revision, "--", "path:" + path],
                                 capture_output=True, check=True).stdout
        extractor.add(path, mode, content)
