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
import copy
import functools
import glob
import hashlib
import itertools
import os
import re
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from wpt_utils.programs import program

from .errors import BadPath, FetchFailed, UnsupportedEntry

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


WILDCARD = re.compile(r"[*?[]")  # what fnmatch reads as more than its own letter


def _normal(path: str) -> str:
    """A path or pattern as full_match sees it: PurePosixPath's spelling,
    with "" for an empty path rather than "."."""
    pure = PurePosixPath(path)
    return str(pure) if pure.parts else ""


class Globs:
    """A list of patterns compiled once, where `match` gives what any of
    glob_match gives for one of them. A pattern with no `*`, `?` or `[`
    matches only its own spelling, so those are a set; the others become one
    regular expression, each the very one full_match compiles (glob.translate,
    recursive, hidden files included, "/" as separator, letter case kept)."""

    def __init__(self, patterns):
        self.exact, wild = set(), []
        for pattern in map(_normal, patterns):
            if WILDCARD.search(pattern):
                wild.append(pattern)
            else:
                self.exact.add(pattern)
        translated = (glob.translate(p, recursive=True, include_hidden=True, seps="/") for p in dict.fromkeys(wild))
        self.wild = re.compile("|".join(translated)).match if wild else None

    def match(self, path: str) -> bool:
        path = _normal(path)
        return path in self.exact or (self.wild is not None and self.wild(path) is not None)


@functools.lru_cache(maxsize=4096)
def _globs(patterns: tuple[str, ...]) -> Globs:
    return Globs(patterns)


def glob_match_any(path: str, patterns) -> bool:
    """Whether path matches any of patterns (see glob_match); each list of
    patterns is compiled once, on its first use."""
    return _globs(tuple(patterns)).match(path)


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
        self.folder: list[str] = []  # the folder of the tree the entries go into (within)
        # compiled here once, not through glob_match_any, which turns the whole list into its cache key on
        # every call (once per entry read) and keeps it compiled after this fetch
        self.excluded = Globs(extract.get("exclude", []))
        self.paths: dict[bytes | str, str | None] = {}  # kept's answers: history commits repeat paths
        self.tree: dict[str, Entry] = {}

    def within(self, folder: str, strip: int | None = None) -> "Extractor":
        """An extractor adding to the same tree, each entry's path (after
        `strip`, the source's unless given) placed under folder ("" for the
        root) before `exclude` is matched: for one of several addresses, or
        the members of an archive a dataset holds."""
        inner = copy.copy(self)  # the same store, exclusions and tree
        inner.folder = [part for part in folder.split("/") if part]
        inner.strip = self.strip if strip is None else strip
        inner.paths = {}
        return inner

    def kept(self, raw_path: bytes | str) -> str | None:
        """The path an entry keeps after `strip`, or None when it is stripped
        away or excluded."""
        if raw_path not in self.paths:
            self.paths[raw_path] = self._kept(raw_path)
        return self.paths[raw_path]

    def _kept(self, raw_path: bytes | str) -> str | None:
        if isinstance(raw_path, bytes):
            try:
                raw_path = raw_path.decode("utf-8")
            except UnicodeDecodeError as error:
                raise BadPath(f"{raw_path!r} is not valid UTF-8") from error
        parts = [p for p in raw_path.split("/") if p not in ("", ".")]
        if len(parts) <= self.strip:
            return None  # gone with the stripped folders, like tar --strip-components
        path = "/".join(self.folder + parts[self.strip:])
        if self.excluded.match(path):
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
# the history's commit order, merges included, part of the history
ORDER = HISTORY + "order"
MERGE = "merge"  # an ORDER line's mark of a merge
# git's output must not depend on the settings of the machine that fetches
PLAIN_GIT = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
             # an object a partial fetch left out is an error, never a download of its own
             "GIT_NO_LAZY_FETCH": "1"}


def license_entries(licenses: list[dict], path: str) -> list[dict]:
    """The license entries covering path, in their order. A history file is
    matched only by patterns that start with `.git/history/`, so an entry
    written `**` covers every file of the tree and no history file; such patterns match no other path.
    Every place that maps a path to its license entries goes through here."""
    history = path.startswith(HISTORY)
    return [e for e in licenses
            if glob_match_any(path, [p for p in e["paths"] if p.startswith(HISTORY)] if history else e["paths"])]


def format_entries(formats: list[dict], path: str) -> list[dict]:
    """The format entries covering path, in their order: a records entry
    covers the files its `join` reads too (records.py). Every place that
    maps a path to its format entry goes through here."""
    return [e for e in formats if glob_match_any(path, e["paths"] + e.get("join", {}).get("paths", []))]


def read_git_history(repo: Path, since: str, revision: str, extractor: Extractor):
    """One file per commit from `since` to `revision`, merges left out:
    `.git/history/<commit>`, holding what `git format-patch` writes for the
    commit (author, date, message, changed files, diff), limited to the
    paths the extractor keeps. A commit that changes no kept path gives no
    file. git never allows a `.git` folder in a tree, so these paths cannot
    meet a file of the commit itself.

    One more file, ORDER, lists every commit from `since` to `revision`,
    merges included, newest first in an order where a commit always comes
    before its parents (git's topological order), one per line: the commit,
    then " merge" for a merge. It is
    part of the tree, so the source checksum covers the order too."""
    order = _history_order(repo, since, revision)
    extractor.add(ORDER, FILE, "".join(f"{commit} {MERGE}\n" if merge else f"{commit}\n"
                                       for commit, merge in order).encode())
    commits = [commit for commit, merge in order if not merge]
    changed = {commit: [path for path, _ in changes] for commit, changes in _changed_paths(repo, commits).items()}
    limits = {}  # commit -> None when every path it changes is kept, else the pathspecs of the kept paths
    for commit in commits:
        paths = [p for p in changed[commit] if extractor.kept(p) is not None]
        if paths:
            limits[commit] = None if len(paths) == len(changed[commit]) else _only(paths, changed[commit])

    def patch(commit):
        specs = limits[commit]
        if any("\n" in s or "\r" in s for s in specs):  # no line of standard input can hold such a path
            return commit, _git(repo, *FORMAT_PATCH, "-1", commit, "--", *specs)
        # the paths go on standard input, after "--": a command line has a limit (32,767 characters on
        # Windows) that a commit changing thousands of files passes
        lines = "".join(line + "\n" for line in (commit, "--", *specs))
        return commit, _git(repo, *FORMAT_PATCH, "-1", "--stdin", input=lines.encode("utf-8", "surrogateescape"))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        limited = pool.map(patch, [c for c, paths in limits.items() if paths is not None])
        for commit, text in itertools.chain(_whole_patches(repo, [c for c, paths in limits.items() if paths is None]),
                                            limited):
            # a diff may hold bytes in any encoding; the file must be UTF-8 text
            extractor.add(HISTORY + commit, FILE, text.decode("utf-8", errors="replace").encode("utf-8"))


def _only(kept, changed) -> list[str]:
    """Pathspecs that limit a commit's diff to the kept paths among the
    changed ones. git reads a literal pathspec naming a folder as a prefix,
    so a kept path that is also the folder of changed paths not kept (a
    submodule replacing a folder of the same name) would bring them in:
    those are excluded by name, and only then."""
    folders = set(kept)
    excluded = [p for p in changed if p not in folders
                and any(p[:i] in folders for i, letter in enumerate(p) if letter == "/")]
    return [f":(literal){p}" for p in kept] + [f":(exclude,literal){p}" for p in excluded]


FORMAT_PATCH = ("format-patch", "--root", "--stdout", "--no-signature", "--no-renames", "--no-binary")
HEADER = re.compile(rb"(?<![^\n])From ([0-9a-f]+) Mon Sep 17 00:00:00 2001\n")  # a patch's first line


PATCH_GROUP = 500  # commits per git call, so one call's output is in memory at a time


def _whole_patches(repo, commits):
    """(commit, what `format-patch -1 <commit>` writes) for commits given
    newest first, one group of PATCH_GROUP commits at a time, so only one
    group's patches are in memory: from one `git format-patch --stdin` per
    group (_group_patches), or one call per commit for a group that cannot
    be cut."""
    for start in range(0, len(commits), PATCH_GROUP):
        group = commits[start:start + PATCH_GROUP]
        cut = _group_patches(repo, group)
        for commit in group:
            yield commit, cut[commit] if commit in cut else _git(repo, *FORMAT_PATCH, "-1", commit)


def _group_patches(repo, commits) -> dict[str, bytes]:
    """{commit: its patch} from one `git format-patch --stdin`, which writes
    the commits oldest first with one blank line between two patches. Each
    patch is cut where the next commit's first line starts. A message or a
    diff may hold such a line too: when the commits' first lines appear more
    than once, or out of order, nothing is cut and the result is empty."""
    if not commits:
        return {}
    out = _git(repo, *FORMAT_PATCH, "--stdin", "--no-walk=unsorted", "-N", input="".join(c + "\n" for c in commits).encode())
    asked, order = set(commits), commits[::-1]
    found = [(match[1].decode(), match.start()) for match in HEADER.finditer(out) if match[1].decode() in asked]
    if [commit for commit, _ in found] != order or found[0][1] != 0:
        return {}
    ends = [start - 1 for _, start in found[1:]] + [len(out)]  # the next patch's start, less the blank line
    if any(out[end - 1:end + 1] != b"\n\n" for end in ends[:-1]):
        return {}
    return {commit: out[start:end] for (commit, start), end in zip(found, ends)}


def _history_commits(repo, since, revision) -> list[str]:
    """The commits from `since` to `revision`, merges left out, in ORDER's order."""
    return [commit for commit, merge in _history_order(repo, since, revision) if not merge]


def _history_order(repo, since, revision) -> list[tuple[str, bool]]:
    """(commit, whether it is a merge) of every commit from `since` to
    `revision`, newest first, each before its parents (ORDER). The one
    walk every list of history commits comes from."""
    lines = _git(repo, "rev-list", "--topo-order", "--parents", revision, "--not",
                 *_since_parents(repo, since, revision)).decode().splitlines()
    return [(line.split()[0], len(line.split()) > 2) for line in lines]


def _since_parents(repo, since, revision) -> list[str]:
    """The parents of `since`, where the history stops; UnsupportedEntry
    when `since` is not an ancestor of the revision."""
    ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", since, revision],
                              capture_output=True, env=PLAIN_GIT)
    if ancestor.returncode != 0:
        raise UnsupportedEntry(f"history.since {since} is not an ancestor of the revision {revision}")
    return _git(repo, "rev-parse", f"{since}^@").decode().split()


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
