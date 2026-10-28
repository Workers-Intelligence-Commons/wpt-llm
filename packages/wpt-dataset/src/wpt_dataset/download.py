"""The download step: sources fetched into the store, each as its address
serves it when asked, in one session (wpt_store.sessions) of the command
that runs it. It decides which sources to fetch, asks their protocol
(protocols/) and writes what they give into the store; it never judges
licenses or content.

A source is downloaded when, and only when (its three triggers):

    the store does not hold it          the store keys what it holds by sources.upstream_key: its
                                        address and how it is fetched (`extract`, `history.since`,
                                        `submodules`, its protocol, its license pages), so a source
                                        whose address or fetch options changed is not held either
    it is marked                        the store database holds a mark for it, or for every source,
                                        made after its last download (Session.marked; `wpt-store
                                        mark`): a person's way to ask for fresh content

Nothing else: no question to upstream about changes, no server hint. A
stale store is accepted.

For each source, several at once (each):

    held and not marked                 -> a `reused` line; nothing is downloaded
    otherwise, its license pages        -> each outside page its entries give as their place
                                           (sources.license_pages), through the HTTP protocol, into a
                                           staging folder on this machine
    then each address in turn           -> its protocol's files, into the staging folder, or for an
    (address, then each mirror)            http source, each member straight into the session's bundles
                                           as it is read (Bundled)
    the first that gives its files      -> its files and pages go into the session's bundles, in path
                                           order, each content once; once the bundles holding them are
                                           stored, the refs upstream-pages/<key>, upstream-files/<key>
                                           and upstream/<key> name them (sources.py), the last one last;
                                           a `downloaded` line with what it got, for the record only:
                                           the protocol's answer (the commit, for example) and the
                                           tree's manifest (`manifest`; a `reused` line names it too)
    none, or a page that failed         -> the source fails, its failure named (below); a `failed` line
                                           for each address or page that failed

A failed download stores nothing for that source: nothing it fetched is
named, so the session's end leaves it out of the bundles (wpt_store.sessions),
and the store keeps what it held before.

Upstream git repositories fetch only the objects whose ids are new: the
git object ids the store database maps to the contents it holds are taken
from the store (protocols/git.py).

An empty source has no upstream: it needs no download and gets no line.
The staging folder is removed in every case.

How many run at once: the rule of wpt_utils.parallel, each download's
memory measured as it runs (its git processes' too), and at most the limit at once sites.txt gives the source's site (the
host of its address; sites.py), whose limits per time pace every request.
The results come in the order of the ids, whatever ran at once; only where
the files lie in the bundles depends on it.

Failures: each source's own is its result, so one run names every failure:
the protocol failures (SourceGone: upstream gone; TooManyRequests: refused
for too many requests; CrawlForbidden: refused by robots rules;
RobotsUnreadable: robots rules unreadable; HostUnreachable: host
unreachable; ServerError: server error through every try; AccessRefused:
access refused; Incomplete: the stream stopped before its end;
FetchFailed, their base: any other), of the kind its main address failed
with, every address and its failure named;
a DatasetError for files the tree cannot hold; StoreError, MissingTool. A
failure of the session (the store refused a bundle or the database) stops
the run.
"""

import contextlib
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from wpt_store import StoreError
from wpt_store.database import DOWNLOADED, FAILED, REUSED  # the store database's outcomes
from wpt_telemetry import Telemetry, set_ref
from wpt_utils.files import sha256_file
from wpt_utils.parallel import Done, Sizing, each as each_item
from wpt_utils.programs import MissingTool, code_name, repository

from . import manifest, retry, sites
from .errors import DatasetError, FetchFailed
from .protocols import EMPTY, fetch, http, named
from .protocols.base import UNREACHABLE
from .repo import Source
from .sources import (UPSTREAM, UPSTREAM_FILES, UPSTREAM_PAGES, license_pages, stored_files, stored_upstream,
                      upstream_key)


class Staging:
    """A download's files on this machine, each under its sha256, until
    they are checked and kept: what protocols.fetch stores into, and what
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
    """What an http source's fetch stores into: each entry the
    extractor keeps goes straight into the session's bundles, and is read
    back from the session, its bundle sealed or not; anything else (a cut's or a patch's result) goes into the
    staging folder, as with Staging. A content it placed once is never
    added again, so a new try of a read (retry.call, a tar read again
    for its hard links) puts nothing more in the bundles."""

    def __init__(self, staging: Staging, session):
        self.staging, self.session = staging, session
        self.placed = {}  # sha256 -> the wpt_store.Placed of each content this object put in the bundles

    def keep(self, data: bytes):
        """(sha256, where it lies: a wpt_store.Placed)."""
        sha256 = hashlib.sha256(data).hexdigest()
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


def wanted(repo, ids) -> list[str]:
    """The sources a download takes: the ids given, or every admitted
    source when none is given."""
    return list(ids) or [s.id for s in repo.sources() if s.admitted]


def run(repo, ids, store, command: str, arguments, stream=False) -> list[Done]:
    """One downloading command's run (download, check --fetch, manifest):
    its record (recorded) and its session (session), the download step for
    the sources ids names into that session (download), the session ended
    (its last bundle sealed) before the record ends. Each source's Done, as
    download gives it; a failure of the record (RecordError) or of the
    session (StoreError) stops the run."""
    with recorded(store, command, arguments, ids, repo.root / ".env") as telemetry:
        with session(store, telemetry, stream) as opened:
            return download(repo, ids, store, opened, telemetry, log=telemetry.log)


def download(repo, ids, store, session, telemetry=None, log=print) -> list[Done]:
    """The download step for the sources ids names, into session (see the
    module's docstring): each one's Done, whose value is its outcome
    (one)."""
    return each(lambda sid: one(repo.source(sid), store, session, telemetry), ids, address_in(repo), log)


def one(source: Source, store, session, telemetry=None) -> str | None:
    """Download one source into session, or find it in the store: its
    outcome (see the module's docstring)."""
    if source.data["type"] == EMPTY:
        return None
    telemetry = telemetry or Telemetry.off()
    key = upstream_key(source)
    if stored_upstream(key, store) is not None and not session.marked(source.id):
        _reused(source, key, store, session)
        return REUSED
    addresses = [source.data["address"]] + [m["address"] for m in source.data.get("mirrors", [])]
    failures = []
    with tempfile.TemporaryDirectory(prefix="wpt-download-") as tmp:
        pages = _pages(session, source, Path(tmp) / "pages")
        for number, address in enumerate(addresses):
            staging, folder = _folders(Path(tmp) / f"address-{number}", session)
            sink = _sink(source.data, staging, session)
            try:
                fetched = _fetch(session, source, address, sink, folder)
            except FetchFailed as error:  # this address only; another may work
                failures.append(error)
                continue
            size, described = _keep(fetched.tree, pages, sink, key, store, session, telemetry)
            session.line(source.id, DOWNLOADED, code_name(), got={**fetched.got, "manifest": described},
                         address=address, size=size)
            return DOWNLOADED
    raise _failure(source, failures)


def _pages(session, source, folder) -> dict[str, Path]:
    """{address: a local file holding it} of the source's license pages
    (sources.license_pages), each fetched through the HTTP protocol; the
    first that fails is the source's failure, with a `failed` line."""
    pages = {}
    for number, address in enumerate(license_pages(source)):
        folder.mkdir(exist_ok=True)
        try:
            try:
                pages[address] = http.page(address, folder / str(number))
            except UNREACHABLE as error:
                raise named(error, address) from error
        except FetchFailed as error:
            _failed(session, source, address)
            raise type(error)(f"{source.id}: license page {error}") from error
    return pages


def _failure(source, failures: list[FetchFailed]) -> FetchFailed:
    """The source's failure when no address gave its files: of the kind
    its main address (the first of failures) failed with, each address and
    its failure named."""
    return type(failures[0])(f"{source.id}: every address failed: " + "; ".join(f"{type(e).__name__}: {e}" for e in failures))


def used(source: Source, store, session):
    """A `reused` line for a source whose upstream tree a reading command
    (a build) used from the store; none for an empty source, which has no
    upstream."""
    if source.data["type"] != EMPTY:
        _reused(source, upstream_key(source), store, session)


def _reused(source, key, store, session):
    """A `reused` line naming the tree the store holds under key (its
    manifest, as got records it), None when it holds none."""
    described = store.get_ref(UPSTREAM, key)
    session.line(source.id, REUSED, code_name(), got={"manifest": described} if described else None,
                 size=_stored_size(stored_files(key, store)))


def _folders(tmp, session):
    """(the staging folder, reading what the store holds through session,
    and the folder downloads go into) inside tmp, made new."""
    folder = Path(tmp) / "download"
    folder.mkdir(parents=True)
    return Staging(Path(tmp) / "files", session), folder


def _sink(data, staging, session):
    """What a fetch of data stores into: Bundled into session for an http
    source (an archive's members go there as they are read), staging for
    any other."""
    return Bundled(staging, session) if data["type"] == "http" else staging


def _fetch(session, source, address, sink, folder):
    """protocols.fetch, with a `failed` line for the address when what it
    serves cannot be taken (a DatasetError: a protocol failure, an entry
    the tree cannot hold). Any other error (the store's, a missing
    program, an interrupt) is not the address's failure: no line."""
    try:
        return fetch(source.data, address, sink, folder)
    except DatasetError:
        _failed(session, source, address)
        raise


def _keep(tree, pages, sink, key, store, session, telemetry) -> int:
    """Put a fetched tree's files and its license pages ({address: local
    file}) into the session's bundles (an entry placed already, in this
    session's bundles or an earlier one's, stays where it lies), and once
    they are stored, name them by the source's key (sources.upstream_key):
    the pages' list first, then the tree's file list and manifest; the git
    object id each of its git files holds (Entry.git) goes into the store
    database's map with them, and so do the blobs its history needed (they
    stay in the bundles, though no tree holds them). Returns (the bytes its
    files take, each content once; the sha256 the store names its manifest
    by)."""
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
    if pages:
        session.name({address: (manifest.FILE, session.add(file)) for address, file in pages.items()},
                     lambda listed: set_ref(store, UPSTREAM_PAGES, key, listed, telemetry))
    described = store.put_bytes(manifest.manifest_bytes(tree))

    def name_it(listed):  # the file list first: a tree named by upstream/ always has its files
        set_ref(store, UPSTREAM_FILES, key, listed, telemetry)
        set_ref(store, UPSTREAM, key, described, telemetry)

    session.name({path: (entry.mode, placed[entry.sha256]) for path, entry in tree.items()}, name_it,
                 {entry.git: path for path, entry in tree.items() if entry.git})
    if mapped:
        session.name({object_id: (manifest.FILE, where) for object_id, where in mapped.items()}, None,
                     {object_id: object_id for object_id in mapped})
    return size, described


def _failed(session, source, address):
    session.line(source.id, FAILED, code_name(), address=address)


def _stored_size(files) -> int | None:
    """The bytes a file list's files take, each content once; None for a
    tree downloaded before bundles."""
    return sum({placed.sha256: placed.length for _, placed in files.values()}.values()) if files else None
