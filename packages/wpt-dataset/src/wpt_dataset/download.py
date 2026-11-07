"""The download step: sources fetched into the store, each as its address
serves it when asked (a git source at the commit its `revision` names), under
one session ID (wpt_store.sessions). It decides which sources to fetch, asks
their protocol (protocols/) and writes what they give into the session; it
never judges licenses or content.

A source is downloaded when, and only when neither the session nor the one
it started from (--from) holds it: each one's record is missing, failed, or
names another download (sources.upstream_key: its address and how it is
fetched: `revision`, `extract`, `history.since`, `submodules`, its protocol,
its license pages). So a rerun, or one new source, costs only what the
session lacks, a redo is a new ID, whose every source is fetched again, and
a new ID started from an old one fetches only what the old one lacks. No
question goes to upstream about changes. What the old session holds stays
there: a build under the new ID reads it from there.

Every source follows one path. For each, several at once:

    held, here or in the session it  -> nothing is downloaded (the index learns of it again,
    started from                        when it is the session's own)
    otherwise, its license pages        -> each outside page its entries give as their place
                                           (sources.license_pages), through the HTTP protocol, into a
                                           folder on this machine
    then each address in turn           -> its protocol's files go straight into the source's tar as
    (address, then each mirror)            they are read (wpt_store.Packing), compressed on this machine's
                                           disk
    the first that gives its files      -> its license pages join its files in the tar (PAGES), the tar
                                           and then the source's record go into the session, once the
                                           event store.source_kept is in the run's record; the record
                                           says what it got, for the record only: the protocol's answer
                                           (the commit, for example) and the tree's manifest
    none, or a page that failed         -> the source fails, its failure named (below); its record gains
                                           the `error`, each address's failure in it, and the session
                                           keeps any file it held for the source, which a build
                                           still reads (wpt_store.sessions)

An empty source has no upstream: it needs no download and gets no record.

The index learns of every source of a run when the run ends.

Each download works in a temporary folder of its own, WORK<random> in the
system's temporary folder: what its protocol fetches lies there. The process
holds a lock file beside it while the download runs (wpt_store.local_files),
and removes it, then the lock file, once the download ends, whatever its
outcome. A process that a signal stops or that crashes leaves both: every
new run on the machine (run()) removes each such folder no live process
owns, and never one a live process owns, and each partial tar likewise. A
folder it cannot open or remove, such as another user's in a temporary
folder several users share, stays.

How many run at once: the rule of wpt_utils.parallel, each download's
memory measured as it runs (its git processes' too), and at most the limit at once sites.txt gives the source's site (the
host of its address; sites.py), whose limits per time pace every request.
The results come in the order of the ids, whatever ran at once.

Failures: each source's own is its result, so one run names every failure:
the protocol failures (SourceGone: upstream gone; TooManyRequests: refused
for too many requests; CrawlForbidden: refused by robots rules;
RobotsUnreadable: robots rules unreadable; HostUnreachable: host
unreachable; ServerError: server error through every try; AccessRefused:
access refused; Incomplete: the stream stopped before its end, or gave other
bytes on a new try; FetchFailed, their base: any other), of the kind its
main address failed with, every address and its failure named;
a DatasetError for files the tree cannot hold; StoreError, MissingTool. A
failure of the store while a source is kept is that source's result, and
the session holds what wpt_store.sessions says. Each of these goes into the
source's record; only a store that refuses that write leaves it unrecorded.
"""

import contextlib
import hashlib
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from wpt_store import (Packing, StoreError, failed_addresses, held_all, held_here, keep_failure,
                       remove_unowned_partials, source_row)
from wpt_store.local_files import LocalFile, unowned
from wpt_telemetry import Telemetry
from wpt_utils.files import sha256_file
from wpt_utils.lines import say
from wpt_utils.parallel import Done, Sizing, each as each_item
from wpt_utils.programs import MissingTool, code_name

from . import manifest, retry, sites
from .errors import BadPath, DatasetError, FetchFailed, Incomplete
from .protocols import EMPTY, fetch, http, named
from .protocols.base import UNREACHABLE
from .source_data import Source
from .sources import license_pages, upstream_key

WORK = "wpt-download-"  # the start of the name of a download's temporary folder
PAGES = ".wpt-license-pages/"  # the folder of a source's tar that its license pages lie in, by their number
DOWNLOADED, REUSED = "downloaded", "reused"  # what a download did for a source
FAILURES = (DatasetError, StoreError, MissingTool)  # what fails one source and leaves the others going


class Staging:
    """A fetch's files on this machine, each under its sha256: what
    protocols.fetch stores into when nothing is to be kept in a session
    (unusable.py), and what the text check then reads, as it reads a store."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def path(self, sha256: str) -> Path:
        return self.folder / sha256

    def has(self, sha256: str) -> bool:
        return self.path(sha256).is_file()

    def keep(self, _path, _mode, data: bytes) -> str:
        sha256 = hashlib.sha256(data).hexdigest()
        if not self.has(sha256):
            self.path(sha256).write_bytes(data)
        return sha256

    def keep_file(self, _path, _mode, file) -> str:
        """Keep a downloaded file as a second name of the same bytes (a hard
        link: a file of several GB is not copied), or a copy where linking
        fails (another file system). Its writer may remove the file or
        write a new one in its place, never rewrite it."""
        sha256 = sha256_file(file)
        if not self.has(sha256):
            try:
                os.link(file, self.path(sha256))
            except OSError:
                shutil.copyfile(file, self.path(sha256))
        return sha256

    def keep_same(self, _path, _mode, sha256: str) -> str:
        return sha256

    def get_bytes(self, sha256: str) -> bytes:
        return self.path(sha256).read_bytes()


def each(work, ids, address_of, log=say):
    """[wpt_utils.parallel.Done] of work(id) for every id, in the ids'
    order, as many at once as wpt_utils.parallel allows and each site's
    limit (address_of(id) gives its address, None when it has none). One
    of FAILURES work raises is the id's error."""
    limits, default, _ = sites.limits()
    sizing = Sizing("downloads")
    log(sizing.why)

    def group(sid):
        address = address_of(sid)
        site = sites.site_of(address, limits) if address else None
        return (site or f"default:{urlparse(address or '').hostname}"), limits.get(site, default)

    return list(each_item(work, ids, FAILURES, sizing, group=group))


@contextlib.contextmanager
def recorded(store, command, arguments, ids, env_file):
    """The telemetry of one downloading command's run: its record,
    <id>/runs/<UTC>-wpt-dataset-<command>-<the id, or how many>-<random>/,
    with every new try of a download in it (retry.reporting)."""
    main_id = ids[0] if len(ids) == 1 else f"{len(ids)}-sources"
    with (Telemetry.start(store, f"wpt dataset {command}", arguments, main_id, inputs={"sources": list(ids)},
                          env_file=env_file) as telemetry,
          retry.reporting(telemetry.log)):
        yield telemetry


def remove_unowned_folders() -> list[str]:
    """Remove every download's temporary folder (WORK...) in the system's
    temporary folder that no live process owns (local_files.unowned): the
    leftovers of a process that stopped without removing them. Their
    names. One the system refuses to remove stays, for a later session:
    removing it is not the session's work, so it never stops one."""
    removed = []
    for left in unowned(tempfile.gettempdir(), WORK):
        name = left.path().name if left.path().exists() else None
        try:
            left.remove()
        except OSError:
            continue
        if name:
            removed.append(name)
    return removed


@contextlib.contextmanager
def work_folder():
    """A new temporary folder for one download, owned by this process
    through its lock file while the with block runs, then removed."""
    local = LocalFile(Path(tempfile.gettempdir()) / (WORK + secrets.token_hex(8)))
    try:
        local.path().mkdir()
        yield local.path()
    finally:
        local.remove()


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
    """One downloading command's run (download, check --fetch, manifest),
    in the session of store (a view of Store.at): its record (recorded),
    the machine cleaned of what stopped processes left, the download step
    for the sources ids names (download). Each source's Done, as download
    gives it; a failure of the record (RecordError) stops the run. stream:
    no copy of a tar stays on this machine."""
    with recorded(store, command, arguments, ids, repo.root / ".env") as telemetry:
        remove_unowned_folders()
        remove_unowned_partials(store.cache)
        return download(repo, ids, store, telemetry, stream, log=telemetry.log)


def download(repo, ids, store, telemetry=None, stream=False, log=say) -> list[Done]:
    """The download step for the sources ids names, into the session of
    store (see the module's docstring): each one's Done, whose value is its
    outcome (one). The index learns of them at the end."""
    rows = []
    try:
        return each(lambda sid: one(repo.source(sid), store, rows, telemetry, stream), ids, address_in(repo), log)
    finally:
        store.note(rows)


def one(source: Source, store, rows: list, telemetry=None, stream=False) -> str | None:
    """Download one source into the session, or find it there: DOWNLOADED,
    REUSED, or None for an empty source (see the module's docstring). What
    the index is to learn goes into rows."""
    if source.data["type"] == EMPTY:
        return None
    key, previous, failures = upstream_key(source), held_here(store, source.id), []
    if previous is not None and previous.holds(key):
        rows.append(source_row(previous))
        return REUSED
    if any(found.holds(key) for found in held_all(store, source.id)):  # the session it started from holds it
        return REUSED
    try:
        rows.append(source_row(_fetched(source, key, previous, store, failures, telemetry or Telemetry.off(), stream)))
    except FAILURES as error:
        rows.append(source_row(keep_failure(store, source.id, previous, key, source.data["address"], error, failures)))
        raise
    return DOWNLOADED


def _fetched(source, key, previous, store, failures, telemetry, stream):
    """The source fetched from the first address that gives it, and kept:
    its record (wpt_store.Held). Each address or page that failed goes into
    failures, (address, failure)."""
    with work_folder() as tmp:
        pages = _pages(source, Path(tmp) / "pages", failures)
        for number, address in enumerate(addresses(source)):
            _, folder = folders(Path(tmp) / f"address-{number}")
            packing = Packing(store, source.id, stream)
            try:
                try:
                    fetched = fetch(source.data, address, packing, folder)
                    if not packing.matches(fetched.tree):
                        raise Incomplete(f"{address}: gave other files on a new try")
                except FetchFailed as error:  # this address only; another may work
                    failures.append((address, error))
                    continue
                except DatasetError as error:  # what it serves cannot be taken
                    failures.append((address, error))
                    raise
                return _keep(source, fetched, pages, packing, key, address, previous, failures, telemetry)
            finally:
                packing.discard()
    raise failure(source, [error for _, error in failures])


def addresses(source: Source) -> list[str]:
    """The addresses to try in turn for a source's files: its own, then its
    mirrors'."""
    return [source.data["address"]] + [mirror["address"] for mirror in source.data.get("mirrors", [])]


def _pages(source, folder, failures) -> dict[str, Path]:
    """{address: a local file holding it} of the source's license pages
    (sources.license_pages), each fetched through the HTTP protocol; the
    first that fails is the source's failure, and goes into failures."""
    pages = {}
    for number, address in enumerate(license_pages(source)):
        folder.mkdir(exist_ok=True)
        try:
            try:
                pages[address] = http.page(address, folder / str(number))
            except UNREACHABLE as error:
                raise named(error, address) from error
        except FetchFailed as error:
            failures.append((address, error))
            raise type(error)(f"{source.id}: license page {error}") from error
    return pages


def failure(source, failures: list[FetchFailed]) -> FetchFailed:
    """The source's failure when no address gave its files: of the kind
    its main address (the first of failures) failed with, each address and
    its failure named."""
    return type(failures[0])(f"{source.id}: every address failed: " + "; ".join(f"{type(e).__name__}: {e}" for e in failures))


def folders(tmp):
    """(a Staging, and the folder downloads go into) inside tmp, made new."""
    folder = Path(tmp) / "download"
    folder.mkdir(parents=True)
    return Staging(Path(tmp) / "files"), folder


def _keep(source, fetched, pages, packing, key, address, previous, failures, telemetry):
    """The license pages ({address: local file}) added to the fetched tree's
    tar, then both kept in the session (Packing.finish), once the event
    store.source_kept is in the run's record: the source's record, which
    names the addresses that failed before this one gave it (failures)."""
    paths = {}
    for number, (page, file) in enumerate(pages.items()):
        paths[page] = f"{PAGES}{number}"
        if paths[page] in fetched.tree:
            raise BadPath(f"{paths[page]!r} is where the source's license pages go")
        packing.keep_file(paths[page], manifest.FILE, file)
    described = manifest.digest(fetched.tree)
    telemetry.irreversible("store.source_kept", source=source.id, manifest=described)
    return packing.finish({path: entry.mode for path, entry in fetched.tree.items()}, paths,
                          {"key": key, "address": address, "manifest": described, "got": fetched.got,
                           "code": code_name(), **({"failures": failed_addresses(failures)} if failures else {})},
                          previous)
