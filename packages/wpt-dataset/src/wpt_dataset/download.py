"""The download step: sources fetched into the store, each checked against
its checksum, in one session (wpt_store.sessions) of the command that
runs it.

For each source, several at once (each):

    its upstream tree in the store      -> a `reused` line; nothing is downloaded (unless again: then it
                                           is downloaded and checked as below, and its files, held
                                           already, are not stored twice)
    otherwise, each address in turn     -> fetched into a staging folder on this machine (sources.fetch),
    (address, then each mirror)            or, for an http-archive source, each member straight into the
                                           session's bundles as it is read (Bundled); its tree's checksum
                                           compared with the source's
    the first that matches              -> its files go into the session's bundles, in path order, each
                                           content once (an http-archive source's are there already);
                                           its file list and manifest are stored, and once the bundles
                                           holding them are, the refs upstream-files/<checksum> and
                                           upstream/<checksum> name them (sources.py); a `downloaded` line
    none                                -> the source fails as before: ChecksumMismatch, CrawlForbidden
                                           or FetchFailed, listing each address; a `failed` line for each
                                           address that failed, with the checksum it gave; nothing of
                                           what it gave is named (an http-archive source's members stay
                                           in the bundles, in no file list)

An empty source has no upstream: it needs no download and gets no line.
An http-archive source downloaded again (again) whose tree the store
holds puts nothing in the bundles: only its members' sha256 are computed.
The staging folder is removed in every case. fill (fill-checksum) and
unchecked (manifest) fetch the main address the same way and keep what
they fetched under the checksum they found, so checking the source
afterwards downloads nothing.

How many run at once: the rule of wpt_utils.parallel, each download's
memory measured as it runs (its git processes' too), and at most the limit at once sites.txt gives the source's site (the
host of its address; sites.py), whose limits per time pace every request.
The results come in the order of the ids, whatever ran at once; only where
the files lie in the bundles depends on it.

Failures: each source's own (DatasetError, StoreError, MissingTool) is its
result, so one run names every failure. A failure of the session (the store refused a
bundle or the database) stops the run.
"""

import contextlib
import copy
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from wpt_store import StoreError
from wpt_telemetry import Telemetry, set_ref
from wpt_utils.files import sha256_file
from wpt_utils.parallel import Done, Sizing, each as each_item
from wpt_utils.programs import MissingTool, code_name, repository

from . import manifest, retry, sites
from .errors import ChecksumMismatch, CrawlForbidden, CutFailed, DatasetError, FetchFailed, NotText, PatchFailed
from .repo import Source
from .sources import (UPSTREAM, UPSTREAM_FILES, apply_cuts, apply_patches, exact_pattern, fetch, not_text,
                      stored_files, stored_upstream)

DOWNLOADED, REUSED, FAILED = "downloaded", "reused", "failed"  # the store database's outcomes (wpt_store.database)


class Staging:
    """A download's files on this machine, each under its sha256, until
    they are checked and kept: what sources.fetch stores into, and what
    cuts, patches and the text check read, as they read a store. With a
    session, a git source's files the store holds already
    (known_objects) are read from the store, through the session."""

    def __init__(self, folder, session=None):
        self.folder, self.session = Path(folder), session
        self.folder.mkdir(parents=True, exist_ok=True)
        self.mapped = {}  # git object id -> sha256 of the blobs kept for the map alone (keep_mapped)

    def known_objects(self, object_ids) -> dict:
        """{git object id: wpt_store.Placed} of those the store holds (none without a session)."""
        return self.session.known_objects(object_ids) if self.session is not None else {}

    def read_file(self, placed) -> bytes:
        return self.session.read(placed)

    def file_path(self, placed) -> Path:
        """A local file holding a stored file's bytes: a copy in the staging folder."""
        if not self.has(placed.sha256):
            self.put_bytes(self.read_file(placed))
        return self.path(placed.sha256)

    def path(self, sha256: str) -> Path:
        return self.folder / sha256

    def has(self, sha256: str) -> bool:
        return self.path(sha256).is_file()

    def put_bytes(self, data: bytes) -> str:
        sha256 = hashlib.sha256(data).hexdigest()
        if not self.has(sha256):
            self.path(sha256).write_bytes(data)
        return sha256

    def put_file(self, path) -> str:
        """Keep a downloaded file as a second name of the same bytes (a hard
        link: a file of several GB is not copied), or a copy where linking
        fails (another file system). Its writer may remove the file or
        write a new one in its place, never rewrite it."""
        sha256 = sha256_file(path)
        if not self.has(sha256):
            try:
                os.link(path, self.path(sha256))
            except OSError:
                shutil.copyfile(path, self.path(sha256))
        return sha256

    def keep_mapped(self, object_id: str, data: bytes):
        """A git blob no tree entry holds (one the history's patches read),
        kept so that _keep puts it in the bundles and the store database's
        map of git objects names it."""
        self.mapped[object_id] = self.put_bytes(data)

    def keep(self, data: bytes) -> tuple[str, None]:
        """data kept as a file under its sha256 (manifest.Extractor)."""
        return self.put_bytes(data), None

    def get_bytes(self, sha256: str) -> bytes:
        return self.path(sha256).read_bytes()

    def size(self, sha256: str) -> int:
        return self.path(sha256).stat().st_size


class Bundled:
    """What an http-archive source's fetch stores into: each entry the
    extractor keeps goes straight into the session's bundles (with no
    session: only its sha256 is computed, for a tree the store holds
    already), and is read back from the session, its bundle sealed or
    not; anything else (a cut's or a patch's result) goes into the
    staging folder, as with Staging. A content it placed once is never
    added again, so a new try of a read (retry.call, a tar read again
    for its hard links) puts nothing more in the bundles."""

    def __init__(self, staging: Staging, session=None):
        self.staging, self.session = staging, session
        self.placed = {}  # sha256 -> the wpt_store.Placed of each content this object put in the bundles

    def keep(self, data: bytes):
        """(sha256, where it lies: a wpt_store.Placed, or None with no session)."""
        sha256 = hashlib.sha256(data).hexdigest()
        if self.session is None:
            return sha256, None
        if sha256 not in self.placed:
            self.placed[sha256] = self.session.add(data)
        return sha256, self.placed[sha256]

    def read_file(self, placed) -> bytes:
        return self.session.read(placed)

    def file_path(self, placed) -> Path:
        """A local file holding a kept entry's bytes: a copy in the staging folder."""
        if not self.staging.has(placed.sha256):
            self.staging.put_bytes(self.read_file(placed))
        return self.staging.path(placed.sha256)

    def put_bytes(self, data: bytes) -> str:
        return self.staging.put_bytes(data)

    def put_file(self, path) -> str:
        return self.staging.put_file(path)

    def path(self, sha256: str) -> Path:
        return self.staging.path(sha256)

    def get_bytes(self, sha256: str) -> bytes:
        return self.staging.get_bytes(sha256)


def each(work, ids, address_of, log=print):
    """[wpt_utils.parallel.Done] of work(id) for every id, in the ids'
    order, as many at once as wpt_utils.parallel allows and each site's
    limit (address_of(id) gives its address, None when it has none). A
    DatasetError, StoreError or MissingTool work raises is the id's
    error."""
    limits, default, _ = sites.limits()
    sizing = Sizing("downloads")
    log(sizing.why)

    def group(sid):
        address = address_of(sid)
        site = sites.site_of(address, limits) if address else None
        return (site or f"default:{urlparse(address or '').hostname}"), limits.get(site, default)

    return list(each_item(work, ids, (DatasetError, StoreError, MissingTool), sizing, group=group))


@contextlib.contextmanager
def recorded(store, command, arguments, ids, env_file):
    """The telemetry of one downloading command's run: its record,
    records/<UTC>-wpt-dataset-<command>-<the id, or how many>-<random>/,
    with every new try of a download in it (retry.reporting). Its session
    (session()) opens inside it and ends before it."""
    main_id = ids[0] if len(ids) == 1 else f"{len(ids)}-sources"
    with (Telemetry.start(store, f"wpt-dataset-{command}", arguments, main_id, inputs={"sources": list(ids)},
                          env_file=env_file) as telemetry,
          retry.reporting(telemetry.log)):
        yield telemetry


def session(store, telemetry, stream=False):
    """A new session of store named by telemetry's run record, at this
    repository's commit (wpt_store.sessions): the session a downloading
    command's run downloads into."""
    where = repository()
    return store.session(telemetry.folder, telemetry.command, where["repository_commit"] or "",
                         bool(where["repository_dirty"]), stream=stream)


def address_in(repo):
    """address_of for each: a source's address, None when it cannot be
    read (its work names why)."""
    def address_of(sid):
        try:
            return repo.source(sid).data["address"]
        except DatasetError:
            return None
    return address_of


def download(repo, ids, store, session, telemetry=None, log=print) -> list[Done]:
    """The download step for the sources ids names, into session (see the
    module's docstring): each one's Done, whose value is its outcome
    (one)."""
    return each(lambda sid: one(repo.source(sid), store, session, telemetry), ids, address_in(repo), log)


def one(source: Source, store, session, telemetry=None, again=False) -> str | None:
    """Download one source into session, or find it in the store (unless
    again): its outcome (see the module's docstring)."""
    if source.data["type"] == "empty":
        return None
    telemetry = telemetry or Telemetry.off()
    expected = source.data["checksum"]
    stored = stored_upstream(expected, store) is not None
    if stored and not again:
        session.line(source.id, REUSED, code_name(), revision=source.data["revision"], checksum=expected,
                     size=_stored_size(stored_files(expected, store)))
        return REUSED
    addresses = [source.data["address"]] + [m["address"] for m in source.data.get("mirrors", [])]
    unreachable, forbidden, mismatched = [], [], []
    for address in addresses:
        with tempfile.TemporaryDirectory(prefix="wpt-download-") as tmp:
            staging, folder = _folders(tmp, session)
            sink = _sink(source.data, staging, None if stored else session)
            try:
                tree = _fetch(session, source, source.data, address, sink, folder)
            except (CrawlForbidden, FetchFailed) as error:  # this address only; another may work
                (forbidden if isinstance(error, CrawlForbidden) else unreachable).append(str(error))
                continue
            actual = manifest.checksum(tree)
            if actual == expected:
                if stored:  # its files lie where the store holds them already: the map learns where
                    files = stored_files(expected, store)  # the same paths: the checksum matched
                    session.remember({entry.git: files[path][1] for path, entry in tree.items() if entry.git})
                    size = _stored_size(files)
                else:
                    size = _keep(tree, sink, expected, store, session, telemetry)
                session.line(source.id, DOWNLOADED, code_name(), revision=source.data["revision"], checksum=expected,
                             address=address, size=size)
                return DOWNLOADED
            mismatched.append(f"{address}: {actual}")
            _failed(session, source, address, actual)
    others = "".join(f"; refused {f}" for f in forbidden) + "".join(f"; unreachable {u}" for u in unreachable)
    if mismatched:
        raise ChecksumMismatch(f"{source.id}: expected {expected}, got " + "; ".join(mismatched) + others)
    if forbidden:
        raise CrawlForbidden(f"{source.id}: every address failed{others}")
    raise FetchFailed(f"{source.id}: " + "; ".join(unreachable))


def used(source: Source, store, session):
    """A `reused` line for a source whose upstream tree a reading command
    (a build) used from the store; none for an empty source, which has no
    upstream."""
    if source.data["type"] != "empty":
        session.line(source.id, REUSED, code_name(), revision=source.data["revision"], checksum=source.data["checksum"],
                     size=_stored_size(stored_files(source.data["checksum"], store)))


def fill(source: Source, store, session, telemetry=None) -> tuple[dict, dict[str, str]]:
    """For a draft source (checksum all zeros): (its description with the
    checksum filled in, {path: why} of the files it now excludes). One
    fetch of the main address (of each of the source's `addresses`) gives
    the upstream tree; every file of the final tree that is not text, as
    sources.source_tree finds them (encoding.py), is excluded by its exact
    path (sources.exact_pattern). The tree is then read again from what
    that fetch downloaded for a git source, since an exclusion also
    changes its history files, and the excluded files are left out of it
    for the others. What it fetched is kept in session under the checksum
    found, so checking the source downloads nothing more."""
    telemetry = telemetry or Telemetry.off()
    data = copy.deepcopy(source.data)
    address = data["address"]
    with tempfile.TemporaryDirectory(prefix="wpt-download-") as tmp:
        staging, folder = _folders(tmp, session)
        sink = _sink(data, staging, session)
        try:
            tree = _fetch(session, source, data, address, sink, folder)
            try:
                final = apply_patches(source.id, apply_cuts(source.id, tree, data.get("cuts", []), sink),
                                      source.patches, sink)
            except (CutFailed, PatchFailed):
                final = {}  # the check names that failure, before any text is read
            found = not_text(source, final, sink)
            if made := [path for path in found if path not in tree]:
                raise NotText(f"{source.id}: {found[made[0]]} and comes from a patch, not upstream; "
                              f"extract cannot leave it out")
            if found:
                data.setdefault("extract", {}).setdefault("exclude", []).extend(map(exact_pattern, found))
                if data["type"] == "git":
                    tree = _fetch(session, source, data, address, sink, folder, again=True)
                else:
                    tree = {path: entry for path, entry in tree.items() if path not in found}
        except FetchFailed as error:
            raise FetchFailed(f"{source.id}: {error}") from error
        data["checksum"] = manifest.checksum(tree)
        _kept(source, data, tree, sink, address, store, session, telemetry)
    return data, found


def unchecked(source: Source, store, session, telemetry=None) -> dict:
    """The upstream tree at the source's main address, its checksum not
    checked (to fill it in by hand): kept in session under the checksum
    found, as fill keeps it."""
    telemetry = telemetry or Telemetry.off()
    address = source.data["address"]
    with tempfile.TemporaryDirectory(prefix="wpt-download-") as tmp:
        staging, folder = _folders(tmp, session)
        sink = _sink(source.data, staging, session)
        tree = _fetch(session, source, source.data, address, sink, folder)
        _kept(source, {**source.data, "checksum": manifest.checksum(tree)}, tree, sink, address, store, session,
              telemetry)
    return tree


def _folders(tmp, session):
    """(the staging folder, reading what the store holds through session,
    and the folder downloads go into) inside tmp."""
    folder = Path(tmp) / "download"
    folder.mkdir()
    return Staging(Path(tmp) / "files", session), folder


def _sink(data, staging, session):
    """What a fetch of data stores into: Bundled into session for an
    http-archive source (None: only the sha256), staging for any other."""
    return Bundled(staging, session) if data["type"] == "http-archive" else staging


def _fetch(session, source, data, address, sink, folder, again=False):
    """sources.fetch, with a `failed` line for the address when what it
    serves cannot be taken (a DatasetError: FetchFailed, CrawlForbidden, an
    entry the tree cannot hold). Any other error (the store's, a missing
    program, an interrupt) is not the address's failure: no line."""
    try:
        return fetch(data, address, sink, folder, again=again)
    except DatasetError:
        _failed(session, source, address)
        raise


def _kept(source, data, tree, sink, address, store, session, telemetry):
    """Keep a fetched tree under data's checksum, with its `downloaded` line."""
    size = _keep(tree, sink, data["checksum"], store, session, telemetry)
    session.line(source.id, DOWNLOADED, code_name(), revision=data["revision"], checksum=data["checksum"], address=address,
                 size=size)


def _keep(tree, sink, checksum, store, session, telemetry) -> int:
    """Put a checked tree's files into the session's bundles (an entry
    placed already, in this session's bundles or an earlier one's, stays
    where it lies), and once they are stored, name its file list and
    manifest by its checksum; the git object id each of its git files
    holds (Entry.git) goes into the store database's map with them, and so
    do the blobs its history needed (they stay in the bundles, though no
    tree holds them). Returns the bytes its files take, each content
    once."""
    placed = {}
    for path in sorted(tree, key=str.encode):
        entry = tree[path]
        if entry.sha256 not in placed:
            placed[entry.sha256] = entry.placed or session.add(sink.path(entry.sha256))
    size = sum(where.length for where in placed.values())
    # the blobs the history's patches read that no entry of the tree holds go into the bundles and the map too,
    # named by no file list: the session's end keeps them (Session.name with no action)
    mapped = {}
    for object_id, sha256 in getattr(sink, "mapped", {}).items():
        if sha256 not in placed:
            placed[sha256] = session.add(sink.path(sha256))
        mapped[object_id] = placed[sha256]
    described = store.put_bytes(manifest.manifest_bytes(tree))
    name = checksum.removeprefix("sha256:")

    def name_it(listed):  # the file list first: a tree named by upstream/ always has its files
        set_ref(store, UPSTREAM_FILES, name, listed, telemetry)
        set_ref(store, UPSTREAM, name, described, telemetry)

    session.name({path: (entry.mode, placed[entry.sha256]) for path, entry in tree.items()}, name_it,
                 {entry.git: path for path, entry in tree.items() if entry.git})
    if mapped:
        session.name({object_id: (manifest.FILE, where) for object_id, where in mapped.items()}, None,
                     {object_id: object_id for object_id in mapped})
    return size


def _failed(session, source, address, checksum=None):
    session.line(source.id, FAILED, code_name(), revision=source.data["revision"], checksum=checksum, address=address)


def _stored_size(files) -> int | None:
    """The bytes a file list's files take, each content once; None for a
    tree downloaded before bundles."""
    return sum({placed.sha256: placed.length for _, placed in files.values()}.values()) if files else None
