"""Sessions: an ID a store's files are stored under, and what a session
holds of its sources and datasets.

A session is only an ID: nothing opens it, nothing closes it, it has no
status. Files are stored under it and a record says what happened. One ID
covers the whole store: every command that touches the store takes it
(--session ID, or WPT_SESSION in the environment or in .env: given, needed),
and Store.at(session, start) is the store as that command works in it. A
download given none makes a new short one (new_id). With start, the ID a
command began from (--from), it reads from start what the session does not
hold, and writes under the session; whatever it writes names start
(Store.put_json). No rule protects an ID another one started from.

    <id>/sources/<source>.tar.zst      a source's files: a normal tar (wpt_store.tars), compressed in
                                       frames of about 1 MiB at SOURCE_LEVEL, with its seek table
                                       (wpt_store.blocks), so one file is read alone by range
    <id>/sources/<source>.json         when, address, size, or the error, and what reading needs (Held)
    <id>/datasets/<name>/              dataset.json, its shards and every other file it needs: its vocabulary,
                                       document list, credits, conversations and tasks (<name> is
                                       <recipe or series>-<its dataset id's first 8 hex characters>)

Packing writes one source's .tar.zst: on this machine's disk first, in the
store's cache folder, as PARTIAL<random>, owned by its process through a
lock file beside it (wpt_store.local_files) while it fills, compressed as
the members come (keep, keep_file, keep_same). finish() sends the tar, then
its record; until the record is written the session reads what it held. A
store whose files are not on this machine keeps a copy of the tar in its
cache folder, unless the download streams; the copy's name holds the tar's
sha256, so a tar replaced elsewhere is never read through an old copy.
discard() removes the partial tar and leaves the session as it was. Every
new download run removes the partial tars no live process owns.

A source's record, Held, is one JSON file (Store.put_json), read by
Store.get_json, and held(store, source, key) finds it. Another process may replace
a record and its tar at any time, so every read asks the store for the
record's version (Store.version), and only a parsed record of that version
is reused. A record read anew makes the store forget what it keeps in memory
of the tar. A file in it is
[path, mode, sha256, start, length], sorted by the UTF-8 bytes of the path,
or for a symlink [path, mode, sha256, target]. Downloading a source again
in a session replaces its files. A download that fails keeps what the
session held for the source, and its record gains the `error`
(keep_failure); a source whose record has an `error` is not held: the next
download fetches it. A build reads the good file such a record still names
(Held.provides) and leaves out a source that has none. Should a crash fall
between the tar and the record, the record names another key or an error,
so the next download fetches the source again, and a read until then
refuses the bytes against their sha256 (CorruptFile).

Failures: SessionNeeded (no ID given), ValueError for an ID that cannot be
one; CorruptFile, MissingFile (a record or tar that is damaged or gone);
what the store raised while storing: the session then holds what it held
for that source, or the tar without its record (above).
"""

import hashlib
import os
import secrets
from pathlib import Path

from . import blocks
from .errors import SessionNeeded
from .index import DATASETS, SOURCES, held_row
from .local_files import LocalFile, unowned
from .names import check_name, check_session
from .settings import now, read_env_file
from .tars import Placed, Tar

SESSION = "WPT_SESSION"  # the setting naming a command's session, as --session does
START = "WPT_FROM"  # the setting naming the session a command starts from, as --from does
ID_BYTES = 4  # random bytes of a new session ID: eight hex digits
PARTIAL = ".partial-source-"  # the start of the name of a source's tar being filled on this machine's disk
SYMLINK = "120000"
DATASET = "dataset.json"
NO_SESSION = (f"this command needs a session: give --session ID or set {SESSION}; "
              "`wpt dataset download` makes a new one and prints it")


def new_id() -> str:
    return secrets.token_hex(ID_BYTES)


def given(value=None, environ=None, env_file=".env") -> str | None:
    """The session a command names: value (--session), else WPT_SESSION in
    the environment, else in env_file; None when none does. ValueError for
    one that cannot be an ID."""
    environ = os.environ if environ is None else environ
    session = value or environ.get(SESSION) or read_env_file(env_file).get(SESSION) or None
    if session is not None:
        check_session(session)
    return session


def needed(value=None, environ=None, env_file=".env") -> str:
    """given, for a command that cannot make its own: SessionNeeded when no
    session is named."""
    session = given(value, environ, env_file)
    if session is None:
        raise SessionNeeded(NO_SESSION)
    return session


def started(value=None, environ=None, env_file=".env") -> str | None:
    """The session a command starts from: value (--from), else WPT_FROM in
    the environment, else in env_file; None when none does. ValueError for
    one that cannot be an ID."""
    environ = os.environ if environ is None else environ
    start = value or environ.get(START) or read_env_file(env_file).get(START) or None
    if start is not None:
        check_session(start)
    return start


def source_key(session: str, source: str, ending: str) -> str:
    check_name(source)
    return f"{session}/{SOURCES}/{source}{ending}"


class Held:
    """What a source's record says: the session that holds it, and what
    any reader of it needs. A failed source's record keeps what the session
    held for it, if anything, beside its `error`."""

    def __init__(self, session: str, record: dict):
        self.session, self.record = session, record
        self.source = record["source"]
        self.key, self.address, self.error = record.get("key"), record.get("address"), record.get("error")
        self.time, self.size, self.stored = record.get("time"), record.get("size"), record.get("stored")
        self.sha256, self.manifest, self.got = record.get("sha256"), record.get("manifest"), record.get("got")
        self.archive = source_key(session, self.source, ".tar.zst")
        self._files = None

    def holds(self, key: str) -> bool:
        """Whether this is that download (its upstream key), not failed: the
        download of a session that holds it is not fetched again."""
        return self.error is None and self.key == key

    def provides(self, key: str) -> bool:
        """Whether this holds a good file of that download (its upstream
        key), even when a later attempt at it failed: what a build reads.
        A record of a download that failed with no file held before has
        none."""
        return self.sha256 is not None and self.key == key

    def _placed(self, listed) -> tuple[str, Placed]:
        if len(listed) == 4:
            _, mode, sha256, target = listed
            text = target.encode("utf-8", "surrogateescape")
            return mode, Placed(self.archive, self.sha256, 0, len(text), sha256, text)
        _, mode, sha256, start, length = listed
        return mode, Placed(self.archive, self.sha256, start, length, sha256)

    @property
    def files(self) -> dict[str, tuple[str, Placed]]:
        """{path: (mode, where it lies)}, sorted as the tree's manifest is."""
        if self._files is None:
            self._files = {listed[0]: self._placed(listed) for listed in self.record.get("files", [])}
        return self._files

    @property
    def pages(self) -> dict[str, Placed]:
        """{address: where it lies} of the license pages downloaded with it."""
        return {address: self._placed(listed)[1] for address, listed in self.record.get("pages", {}).items()}


def held_here(store, source: str) -> Held | None:
    """The record of a source in the session itself (the one a download
    writes under), or None."""
    return _read(store, store.needed(), source)


def held_all(store, source: str) -> list[Held]:
    """The records of a source in the sessions the store reads, in the order
    it reads them: its own session's, then the one it started from. A
    record with an `error` is given too: holds() and provides() tell them
    apart."""
    return [found for session in store.reading() if (found := _read(store, session, source)) is not None]


def held(store, source: str, key: str | None = None) -> Held | None:
    """The record of a source the session reads: with key (its upstream
    key), the first that provides a good file of that download; else, or
    when none does, the first there is, whose `error` says why a build
    leaves the source out; None without any."""
    found = held_all(store, source)
    for record in found:
        if key is not None and record.provides(key):
            return record
    return found[0] if found else None


def _read(store, session, source) -> Held | None:
    key = source_key(session, source, ".json")
    version = store.version(key)  # asked first: a record replaced meanwhile is read again next time
    if version is None:
        return None
    kept = store.held_records.get((session, source))
    if kept is not None and kept[0] == version:
        return kept[1]
    record = store.get_json(key)
    if record is None:
        return None
    found = Held(session, record)
    store.forget(found.archive)
    store.held_records.put((session, source), (version, found))
    return found


class _Hashing:
    """A binary file open for writing that also hashes what goes in."""

    def __init__(self, out):
        self.out, self.digest = out, hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.out.write(data)


class Packing:
    """One source's .tar.zst being made: what a fetch stores into. keep,
    keep_file and keep_same each add one path and give the sha256 of its
    content; a content written already under the same mode is a hard link,
    and a path written already with the same content adds nothing, so a new
    try of a read puts nothing more in the tar. replaced is True once a
    path came again with other bytes: the tar then holds them twice."""

    def __init__(self, store, source: str, stream=False):
        store.needed()
        self.store, self.source, self.stream = store, source, stream
        self.local = LocalFile(store.cache / (PARTIAL + secrets.token_hex(8)))  # a new name: taken at once
        self.replaced, self.members, self._first, self._content = False, {}, {}, {}
        try:
            self._file = open(self.local.path(), "xb")
        except BaseException:
            self.local.remove()
            raise
        self._out = _Hashing(self._file)
        self._writer = blocks.Writer(self._out, *store.compressing(), level=blocks.SOURCE_LEVEL)
        self._tar = Tar(self._writer)

    def _new(self, path, sha256) -> bool:
        """Whether a path is to be written: False for one that holds this
        content already."""
        known = self.members.get(path)
        if known is not None and known[1] == sha256:
            return False
        self.replaced = self.replaced or known is not None
        return True

    def _same(self, path, mode, sha256) -> bool:
        """Whether the path went in as a hard link to the same content,
        written earlier under that mode."""
        first = self._first.get((sha256, mode))
        if first is None:
            return False
        self._tar.link(path, mode, first)
        self.members[path] = (mode, *self.members[first][1:])
        return True

    def _written(self, path, mode, sha256, start, length):
        self.members[path] = (mode, sha256, start, length, None)
        self._first[sha256, mode] = self._content[sha256] = path

    def keep(self, path: str, mode: str, data: bytes) -> str:
        sha256 = hashlib.sha256(data).hexdigest()
        if self._new(path, sha256):
            if mode == SYMLINK:
                self._tar.symlink(path, data)
                self.members[path] = (mode, sha256, 0, len(data), data)
            elif not self._same(path, mode, sha256):
                self._written(path, mode, sha256, self._tar.add(path, mode, data), len(data))
        return sha256

    def keep_file(self, path: str, mode: str, file, sha256: str | None = None) -> str:
        """A local file, never whole in memory."""
        sha256 = sha256 or _sha256_file(file)
        if self._new(path, sha256) and not self._same(path, mode, sha256):
            length = os.path.getsize(file)
            self._written(path, mode, sha256, self._tar.add_file(path, mode, file, length), length)
        return sha256

    def matches(self, tree) -> bool:
        """Whether the tar holds the paths of tree (a mapping by path) and no
        others, each once."""
        return not self.replaced and self.members.keys() == tree.keys()

    def keep_same(self, path: str, mode: str, sha256: str) -> str:
        """A path holding a content kept already (a tar's hard link)."""
        if self._new(path, sha256):
            first = self._content[sha256]
            self._tar.link(path, mode, first)
            self.members[path] = (mode, *self.members[first][1:])
        return sha256

    def finish(self, tree: dict, pages: dict, about: dict, previous: Held | None = None) -> Held:
        """The tar sent to the store under the session, then its record: the
        paths of tree ({path: mode}), pages ({address: its path in the tar})
        and about (the record's own fields: key, address, manifest, got,
        code). previous: what the session held, whose local copy goes. The
        partial tar is gone afterwards, stored or not."""
        store, session = self.store, self.store.needed()
        try:
            self._tar.close()
            original, stored = self._writer.close()
            self._file.close()
            version = self._out.digest.hexdigest()
            key = source_key(session, self.source, ".tar.zst")
            store.keep_archive(key, self.local.path(), version, previous.sha256 if previous else None, not self.stream)
            files = [self._listed(path) for path in sorted(tree, key=str.encode)]
            size = sum({self.members[path][1]: self.members[path][3] for path in tree}.values())
            record = {"source": self.source, "session": session, "time": now(), "size": size, "stored": stored,
                      "original": original, "sha256": version, **about, "files": files,
                      "pages": {address: self._listed(path) for address, path in pages.items()}}
            record = store.put_json(source_key(session, self.source, ".json"), record)
        finally:
            self.discard()
        return Held(session, record)

    def _listed(self, path) -> list:
        mode, sha256, start, length, text = self.members[path]
        if text is not None:
            return [path, mode, sha256, text.decode("utf-8", "surrogateescape")]
        return [path, mode, sha256, start, length]

    def discard(self):
        """The partial tar gone, if still there, and its lock released."""
        self._file.close()
        self.local.remove()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def failed_addresses(failures) -> list[dict]:
    """Each of failures, an address's (address, failure), as a record holds it."""
    return [{"address": address, "kind": type(failed).__name__, "message": str(failed)} for address, failed in failures]


def keep_failure(store, source: str, previous: Held | None, key: str, address: str, error: Exception,
                 failures=()) -> Held:
    """The source's record, in the session, says that its download failed:
    error (a failure of a download of the upstream key), each of failures
    an address's (address, failure). What the session held stays."""
    session = store.needed()
    record = {name: value for name, value in previous.record.items() if name != "error"} if previous else \
        {"source": source, "session": session, "address": address}
    record["error"] = {"time": now(), "key": key, "address": address, "kind": type(error).__name__,
                       "message": str(error), "failures": failed_addresses(failures)}
    record = store.put_json(source_key(session, source, ".json"), record)
    return Held(session, record)


def source_row(found: Held) -> tuple:
    """What the index says of a source's record."""
    return held_row(found.session, SOURCES, found.source, found.record)


def remove_unowned_partials(cache: Path) -> list[str]:
    """Remove every partial tar in cache (PARTIAL...) that no live process
    owns (local_files.unowned). Their names."""
    removed = []
    for left in unowned(cache, PARTIAL):
        if left.path().exists():
            removed.append(left.path().name)
        left.remove()
    return removed


def dataset_key(session: str, name: str, file: str = DATASET) -> str:
    """The key of a file of the dataset name of a session."""
    check_name(name)
    return f"{session}/{DATASETS}/{name}/{file}"


def keep_dataset(store, name: str, record: dict) -> dict:
    """The session's dataset name is record, stamped with this moment as its
    `time`: its dataset.json written, last of its files, and the index told.
    The record as stored."""
    session = store.needed()
    record = store.put_json(dataset_key(session, name), {**record, "time": now()})
    store.note([held_row(session, DATASETS, name, record)])
    return record


def find_dataset(store, dataset_id: str) -> tuple[str, dict] | None:
    """(the key of its record, the record) of the dataset of that id, from
    the session, else from the one it started from, found through the
    index; None when neither holds it."""
    with store.index() as found:
        rows = found.execute("SELECT session, name FROM held WHERE kind = ? AND json_extract(about, '$.id') = ?",
                             (DATASETS, dataset_id)).fetchall()
    for session in store.reading():
        for name in sorted(name for holder, name in rows if holder == session):
            key = dataset_key(session, name)
            if (record := store.get_json(key)) is not None and record.get("id") == dataset_id:
                return key, record
    return None


def latest_dataset(store, title: str) -> dict | None:
    """The record of the dataset last built for a recipe or a series (its
    `names`, which dataset_title gives), found through the index: the
    session's own latest, else the one the session started from; None when
    neither holds one."""
    with store.index() as found:
        rows = found.execute("SELECT session, name FROM held WHERE kind = ? AND EXISTS "
                             "(SELECT 1 FROM json_each(held.about, '$.names') WHERE value = ?) "
                             "ORDER BY time DESC, name", (DATASETS, title)).fetchall()
    for session in store.reading():
        for name in (name for holder, name in rows if holder == session):
            record = store.get_json(dataset_key(session, name))
            if record is not None and title in record.get("names", ()):
                return record
    return None


def dataset_vocabulary(store, sha256: str) -> dict | None:
    """The reference ({"key", "sha256"}) of a vocabulary file of that sha256
    that a dataset of the session, or of the one it started from, holds in
    its folder, found through the index; None when none does. A locked
    vocabulary is found there again when the store's cache folder is
    empty."""
    with store.index() as found:
        rows = found.execute("SELECT session, name FROM held WHERE kind = ? AND json_extract(about, '$.vocabulary') = ?",
                             (DATASETS, sha256)).fetchall()
    for session in store.reading():
        for name in sorted(name for holder, name in rows if holder == session):
            record = store.get_json(dataset_key(session, name))
            reference = None if record is None else record.get("tokenizer", {}).get("vocabulary")
            if isinstance(reference, dict) and reference.get("sha256") == sha256:
                return reference
    return None
