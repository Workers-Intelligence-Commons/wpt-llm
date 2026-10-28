"""Manifests: the sha256 of the exact list of what a source contains.

The list holds one line per entry, sorted by the UTF-8 bytes of the path:

    <mode> <sha256> <path>

mode 100644 is a file, 100755 an executable file, 120000 a symlink (its
sha256 is that of the target text). Empty folders do not appear. A tree
is named by the sha256 of that list's bytes (manifest_bytes), its
`manifest`, as a plain hex digest wherever it appears (digest).

A tree is a mapping path -> Entry. A downloaded file lies in one of its
session's bundles (wpt_store.sessions), and its entry says where
(Entry.placed); any other content (a cut's or a patch's result, a file
of a tree downloaded before bundles) is a store object under its sha256.
content() and local_file() read either. Readers build trees from the downloaded data itself (archive
headers, git or mercurial objects), never from files written to a disk, so
Linux and Windows compute the same manifest.
"""

import concurrent.futures
import contextlib
import copy
import dataclasses
import functools
import glob
import hashlib
import io
import itertools
import os
import re
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from wpt_store import CorruptObject
from wpt_utils.programs import program

from .errors import BadPath, Incomplete, UnsupportedEntry

FILE, EXECUTABLE, SYMLINK = "100644", "100755", "120000"
MODES = (FILE, EXECUTABLE, SYMLINK)
SUBMODULE = "160000"  # a git tree's pointer to a commit of another repository


@dataclass(frozen=True)
class Entry:
    mode: str
    sha256: str
    # where a downloaded file lies (wpt_store.Placed); None for a store object. Not part of what the entry is:
    # two entries with the same mode and content are equal wherever their bytes lie
    placed: object = field(default=None, compare=False, repr=False)
    # the git object id of the blob a git source's entry holds as it was fetched (None otherwise): what the store
    # database's map from git object id to sha256 learns. Not part of what the entry is either, and never in a
    # manifest; an entry a cut or a patch changes is a new entry, without it
    git: str | None = field(default=None, compare=False, repr=False)


def content(store, entry: Entry) -> bytes:
    """An entry's bytes, checked against its sha256."""
    return store.read_file(entry.placed) if entry.placed is not None else store.get_bytes(entry.sha256)


def local_file(store, entry: Entry) -> Path:
    """A local file holding an entry's bytes, checked against its sha256
    (a copy in the store's cache folder when they are not local already)."""
    return store.file_path(entry.placed) if entry.placed is not None else store.path(entry.sha256)


class Stored:
    """One file of a tree as a reader of many records needs it: open()
    reads its bytes from the store as a stream, part by part (a downloaded
    file through wpt_store's file_parts, a store object, such as a patched
    file, through its read_range), never whole in memory and never copied
    to the disk, checked against its sha256 at its end; local() is a local
    file holding them, for a reader that must seek (Parquet)."""

    def __init__(self, store, entry: Entry, name: str):
        self.store, self.entry, self.name = store, entry, name

    def __str__(self):
        return self.name

    def open(self, mode="rb"):
        if mode != "rb":
            raise ValueError("a stored file is read as bytes")
        parts = (self.store.file_parts(self.entry.placed) if self.entry.placed is not None
                 else _object_parts(self.store, self.entry.sha256))
        return io.BufferedReader(_Parts(parts), BLOCK)

    def read_bytes(self) -> bytes:
        return content(self.store, self.entry)

    def local(self) -> Path:
        return local_file(self.store, self.entry)


BLOCK = 1 << 20  # bytes a stored file's stream reads at once


def _object_parts(store, sha256: str):
    """A store object's bytes in parts of at most BLOCK bytes; CorruptObject
    after the last part when they do not match its sha256."""
    digest, length = hashlib.sha256(), store.size(sha256)
    for start in range(0, length, BLOCK):
        part = store.read_range(sha256, start, min(BLOCK, length - start))
        digest.update(part)
        yield part
    if digest.hexdigest() != sha256:
        raise CorruptObject(f"object {sha256} reads back as {digest.hexdigest()}")


class _Parts(io.RawIOBase):
    """A readable stream over an iterator of byte strings."""

    def __init__(self, parts):
        self._parts, self._held, self._at = parts, memoryview(b""), 0

    def readable(self):
        return True

    def readinto(self, buffer) -> int:
        while self._at == len(self._held):
            part = next(self._parts, None)
            if part is None:
                return 0
            self._held, self._at = memoryview(part), 0
        count = min(len(buffer), len(self._held) - self._at)
        buffer[:count] = self._held[self._at:self._at + count]
        self._at += count
        return count

    def close(self):
        close = getattr(self._parts, "close", None)
        if close is not None:
            close()
        super().close()


def size(store, entry: Entry) -> int:
    """An entry's length in bytes, without reading it."""
    return entry.placed.length if entry.placed is not None else store.size(entry.sha256)


def placed_tree(tree: dict[str, Entry], files: dict) -> dict[str, Entry]:
    """The tree with each entry whose content a source's file list
    (wpt_store.parse_file_list: {path: (mode, Placed)}) holds placed where
    that list says; the others stay store objects."""
    where = {placed.sha256: placed for _, placed in files.values()}
    return {path: dataclasses.replace(e, placed=where[e.sha256]) if e.sha256 in where else e
            for path, e in tree.items()}


def manifest_bytes(tree: dict[str, Entry]) -> bytes:
    lines = (f"{e.mode} {e.sha256} {p}\n" for p, e in sorted(tree.items(), key=lambda item: item[0].encode()))
    return "".join(lines).encode()


def digest(tree: dict[str, Entry]) -> str:
    """The tree's manifest: the sha256 of its file list's bytes, plain hex."""
    return hashlib.sha256(manifest_bytes(tree)).hexdigest()


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


# No bound: callers match many lists (a records entry's join, a task's area), each compiled on its first use.
@functools.cache
def _globs(patterns: tuple[str, ...]) -> Globs:
    return Globs(patterns)


def glob_match_any(path: str, patterns) -> bool:
    """Whether path matches any of patterns (see glob_match); each list of
    patterns is compiled once, on its first use."""
    return _globs(tuple(patterns)).match(path)


class PatternIndex:
    """Several lists of patterns compiled once into one matcher: matching(path)
    gives the numbers of the lists that match path (as glob_match_any would
    for each), in order, without trying each list in turn. An exact path is a
    lookup. A pattern with wildcards is filed under its literal folder (the
    segments before its first wildcard) and, when its last segment names a
    kind of file (`*.js`) or a file (`CMakeLists.txt`), under that ending or
    name too: a path meets only the patterns of its own folders whose ending
    or name its own file name has, and those that name neither, each then
    confirmed by its own compiled expression (Globs)."""

    def __init__(self, lists):
        self.exact, wild = {}, {}
        for number, patterns in enumerate(lists):
            for pattern in map(_normal, patterns):
                if not WILDCARD.search(pattern):
                    self.exact.setdefault(pattern, []).append(number)
                    continue
                wild.setdefault(_filed(pattern), {}).setdefault(number, []).append(pattern)
        self.wild = {key: [(number, Globs(found)) for number, found in by_list.items()]
                     for key, by_list in wild.items()}

    def matching(self, path: str) -> list[int]:
        path = _normal(path)
        found = set(self.exact.get(path, ()))
        parts = path.split("/")
        name = parts[-1]
        keys = [("name", name), ("any", "")] + [("ending", name[i + 1:]) for i, letter in enumerate(name)
                                                 if letter == "."]
        for depth in range(len(parts)):
            folder = "/".join(parts[:depth])
            for kind, key in keys:
                for number, globs in self.wild.get((folder, kind, key), ()):
                    if number not in found and globs.match(path):
                        found.add(number)
        return sorted(found)


def _filed(pattern: str) -> tuple[str, str, str]:
    """Where PatternIndex files a pattern with wildcards: (its literal
    folder, "ending" and the ending its last segment `*.<ending>` names,
    "name" and the file its literal last segment names, or "any" and "")."""
    parts = pattern.split("/")
    literal = next(i for i, part in enumerate(parts) if WILDCARD.search(part))
    folder, last = "/".join(parts[:literal]), parts[-1]
    if literal < len(parts) - 1 and not WILDCARD.search(last):
        return folder, "name", last
    if last.startswith("*.") and not WILDCARD.search(last[2:]):
        return folder, "ending", last[2:]
    return folder, "any", ""


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
    each kept entry's content in store: what a download stores into
    (download.Staging, download.Bundled), whose keep(content) gives the
    content's sha256 and where it lies (a wpt_store.Placed, or None for
    an object under its sha256), and put_file(path) a local file's sha256."""

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

    def add(self, raw_path: bytes | str, mode: str, content: bytes, git: str | None = None) -> Entry | None:
        """Store an entry's content when its path is kept: its Entry, or
        None. git: the id of the git blob it holds (Entry.git)."""
        path = self.kept(raw_path)
        if path is not None:
            return self._record(path, mode, *self.store.keep(content), git=git)
        return None

    def add_same(self, raw_path: bytes | str, mode: str, like: Entry) -> Entry | None:
        """An entry holding the content of an entry already stored (a tar's
        hard link), when its path is kept: its Entry, or None."""
        path = self.kept(raw_path)
        if path is not None:
            return self._record(path, mode, like.sha256, like.placed)
        return None

    def add_placed(self, raw_path: bytes | str, mode: str, placed, git: str | None = None) -> Entry | None:
        """An entry whose content the store holds already where placed (a
        wpt_store.Placed) says, when its path is kept: its Entry, or None.
        Its content is neither read nor stored again. git: as add's."""
        path = self.kept(raw_path)
        if path is not None:
            return self._record(path, mode, placed.sha256, placed, git)
        return None

    def keep_blobs(self, repo: Path, object_ids: list[str]):
        """Store git blobs no entry holds, the ones the history's patches
        need (a store that has no use for them, one that does not give
        keep_mapped, ignores them): what a download keeps, so the store
        database's map of git objects learns them (download.Staging)."""
        if object_ids and hasattr(self.store, "keep_mapped"):
            with contextlib.closing(_git_blobs(repo, object_ids)) as contents:
                for object_id, content in zip(object_ids, contents):
                    self.store.keep_mapped(object_id, content)

    def add_file(self, path: str, mode: str, file: Path, git: str | None = None):
        """A kept path (see kept) whose content is a file on disk, stored
        without reading it into memory. git: as add's."""
        self._record(path, mode, self.store.put_file(file), git=git)

    def _record(self, path: str, mode: str, sha256: str, placed=None, git=None) -> Entry:
        if path in self.tree:
            raise BadPath(f"{path!r} appears twice")
        self.tree[path] = Entry(mode, sha256, placed, git)
        return self.tree[path]

    @contextlib.contextmanager
    def trying(self):
        """A block whose entries leave the tree again when it raises: one
        try at reading what an address holds."""
        before = set(self.tree)
        try:
            yield
        except BaseException:
            for path in set(self.tree) - before:
                del self.tree[path]
            raise

    def finish(self) -> dict[str, Entry]:
        check_paths(self.tree)
        return self.tree


def _file_mode(unix_mode: int) -> str:
    return EXECUTABLE if unix_mode & 0o111 else FILE


class LinkBack(Exception):
    """A tar read as it arrives holds hard links to members that were not
    kept: their content went by unread, and only reading the archive again,
    holding them (read_tar_members), gives it. targets: the normalized
    names of the members whose content those links need."""

    def __init__(self, targets):
        super().__init__("hard links to members not kept: " + ", ".join(sorted(targets)))
        self.targets = frozenset(targets)


def read_tar(path: Path, extractor: Extractor):
    with tarfile.open(path, "r:*") as archive:
        read_tar_members(archive, extractor)


def read_tar_members(archive: tarfile.TarFile, extractor: Extractor, streamed=False, held=frozenset()):
    """Every entry of an open tar archive. streamed: it is read once from
    start to end (mode "r|"), so a hard link takes the content of the
    member it names, the last of that name before it, as tarfile finds it.
    A kept hard link to a member that was not kept is left out, and once
    the whole archive is read, LinkBack names every member whose content
    such links need; read again with those names as held, a held member's
    content stays in memory, unstored, for the links that need it. A
    member's content is read whole."""
    read = {}  # streamed: each member's normalized name -> its Entry when kept, its content when held, else None
    links = {}  # streamed: each hard link's normalized name -> the normalized name it links to
    missing = set()
    for member in archive:
        name, found = os.path.normpath(member.name), None
        if member.isdir():
            pass
        elif member.issym():
            extractor.add(member.name, SYMLINK, member.linkname.encode())
        elif member.islnk() and streamed:  # a hard link holds its target's content
            links[name] = target = os.path.normpath(member.linkname)
            found = read.get(target)  # what a link to this one holds, kept or not
            if extractor.kept(member.name) is not None:
                if found is None:
                    missing.add(target)
                elif isinstance(found, Entry):
                    found = extractor.add_same(member.name, _file_mode(member.mode), found)
                else:
                    found = extractor.add(member.name, _file_mode(member.mode), found)
        elif member.isfile() or member.islnk():
            if not streamed or extractor.kept(member.name) is not None:
                found = extractor.add(member.name, _file_mode(member.mode), archive.extractfile(member).read())
            elif name in held:
                found = archive.extractfile(member).read()
        else:
            raise UnsupportedEntry(f"{member.name}: tar entry type {member.type!r} is neither a file nor a symlink")
        if streamed:
            read[name] = found
    if missing:
        raise LinkBack(_linked(missing, links))


def _linked(names, links) -> set[str]:
    """names, with every member a hard link among them leads to, link after link."""
    found, pending = set(), list(names)
    while pending:
        if (name := pending.pop()) not in found:
            found.add(name)
            if name in links:
                pending.append(links[name])
    return found


def read_zip(path: Path, extractor: Extractor):
    with zipfile.ZipFile(path) as archive:
        read_zip_members(archive, extractor)


def read_zip_members(archive: zipfile.ZipFile, extractor: Extractor):
    """Every entry of an open zip archive, in the order of its table of
    contents. A member's content is read whole."""
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


def read_git(repo: Path, revision: str, extractor: Extractor, prefix: str = "", known=None,
             large=None) -> dict[str, str]:
    """Entries of a commit, from git's own objects, each path after `prefix`
    (a submodule's folder, ending in "/"). An excluded file is never read, so
    a partial fetch need not hold it; nor is a blob whose content the store
    holds already (known: {git object id: wpt_store.Placed}), whose entry
    is placed where the store holds it, its sha256 taken from there. Each
    kept entry holds its blob's id (Entry.git). large(blob), when given,
    is a local file holding the file a blob stands for when the blob is not
    the file itself (an LFS pointer), None otherwise: that file's bytes are
    the entry's. Returns the kept submodules, each path with the commit it
    pins; an excluded submodule is skipped."""
    known = known or {}
    entries, submodules = [], {}
    for path, mode, kind, object_id in _git_tree(repo, revision, prefix):
        if extractor.kept(path) is None:
            continue
        if mode == SUBMODULE:
            submodules[path.decode()] = object_id
        elif kind != "blob" or mode not in MODES:
            raise UnsupportedEntry(f"{path.decode(errors='replace')}: git {kind} with mode {mode}")
        elif object_id in known:
            extractor.add_placed(path, mode, known[object_id], object_id)
        else:
            entries.append((path, mode, object_id))
    with contextlib.closing(_git_blobs(repo, [e[2] for e in entries])) as contents:
        for (path, mode, object_id), content in zip(entries, contents):
            file = large(content) if large is not None and mode != SYMLINK else None
            if file is None:
                extractor.add(path, mode, content, object_id)
            elif (kept := extractor.kept(path)) is not None:
                extractor.add_file(kept, mode, file, object_id)
    return submodules


def git_blobs_needed(repo: Path, revision: str, extractor: Extractor, since: str | None = None,
                     prefix: str = "") -> list[str]:
    """The blobs read_git and read_git_history read: the kept files of the
    commit and, with `since`, both sides of every kept file a commit since
    then changes (git_history_blobs). A partial fetch downloads only these."""
    needed = [object_id for path, _, kind, object_id in _git_tree(repo, revision, prefix)
              if kind == "blob" and extractor.kept(path) is not None]
    return list(dict.fromkeys(needed + git_history_blobs(repo, revision, extractor, since)))


def git_history_blobs(repo: Path, revision: str, extractor: Extractor, since: str | None) -> list[str]:
    """The blobs read_git_history reads to write its patches (none without
    `since`): both sides of every kept file a commit since then changes."""
    if not since:
        return []
    needed = []
    for changes in _changed_paths(repo, _history_commits(repo, since, revision)).values():
        needed += [object_id for path, blobs in changes if extractor.kept(path) is not None for object_id in blobs]
    return list(dict.fromkeys(needed))


def put_blobs(repo: Path, files: list[Path]) -> list[str]:
    """Write local files into a repository as git blobs, in one git call:
    contents the store holds already, so git can diff them without
    downloading them. Their object ids, in order."""
    if not files:
        return []
    return _git(repo, "hash-object", "-w", "--no-filters", "--stdin-paths",
                input="".join(f"{file}\n" for file in files).encode()).decode().split()


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


def format_entries(source, path: str) -> list[dict]:
    """The format entries of a source covering path, in their order: a
    records entry covers the files its `join` reads too (records.py). Every
    place that maps a path to its format entry goes through here, and so
    through the source's one matcher (source_paths)."""
    return source_paths(source).format_entries(path)


class SourcePaths:
    """A source's format entries and tag entries, each list compiled once
    into one matcher over all its patterns (PatternIndex): a path meets only
    the patterns of its own folders, never every entry in turn, so a source
    of 200,000 files and 10,000 entries is matched in seconds."""

    def __init__(self, data: dict):
        self.formats, self.tag_entries = data["formats"], data.get("tags", [])
        self._formats = PatternIndex([e["paths"] + e.get("join", {}).get("paths", []) for e in self.formats])
        self._tags = PatternIndex([e["paths"] for e in self.tag_entries])

    def format_entries(self, path: str) -> list[dict]:
        return [self.formats[number] for number in self._formats.matching(path)]

    def tags(self, path: str) -> frozenset[str]:
        return frozenset(tag for number in self._tags.matching(path) for tag in self.tag_entries[number]["tags"])


COMPILED = 64  # the sources whose matchers stay compiled, the last asked about
_compiled: dict[int, tuple[dict, SourcePaths]] = {}


def source_paths(source) -> SourcePaths:
    """The source's matcher (SourcePaths), compiled on its first use and
    kept while the same description (source.data, the very object) is asked
    about again."""
    found = _compiled.get(id(source.data))
    if found is None or found[0] is not source.data:
        if len(_compiled) >= COMPILED:
            _compiled.pop(next(iter(_compiled)), None)
        found = _compiled[id(source.data)] = (source.data, SourcePaths(source.data))
    return found[1]


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
    part of the tree, so the tree's manifest covers the order too."""
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
                raise Incomplete(f"git object {object_id} was not fetched")
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
