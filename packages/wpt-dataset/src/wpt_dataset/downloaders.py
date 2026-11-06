"""Downloaders: worker processes that fetch the files of the coming parts of
a read round while the workers read, so that a part finds its files on this
machine and no worker waits on the store. Without them each worker fetches
for itself, only one batch ahead and inside its own part (reading.py), and
the link stands idle at every part's opening and end.

ahead_of runs them for one round. They take the parts in the order the
workers will, one part each, as many at once as wpt_utils.parallel allows
(a Sizing of network work, at most DOWNLOADERS, whose disk rule keeps its
quarter of the free disk). A downloader opens a part as its read will
(download), and copies its chosen files, the small and the large ones, and
the license statements it reads on the way, into a folder of that part, in a
temporary folder of the store's cache folder. A read given that folder
(Downloaded) finds them there, and what it does not find it reads for itself,
so what it gives never depends on what was copied. What is
downloaded and not yet used stays within CAP bytes, and within a quarter of
the disk's free bytes when the round starts, so the workers' own disk rule
still has room: a part counts its chosen bytes from its download's start to
its release. At the limit the downloaders wait. A part larger than the limit
is downloaded only when nothing else is held.

A part that starts takes its folder (Downloads.take), and waits for its
download if that is under way. A part whose download has not started takes
none: the downloaders pass it by, and it fetches for itself as it does
without them. So no part ever waits on the limit, on which the parts that
hold it may be waiting in their turn. The read releases each file it used
(reading.py), and the folder goes when the part ends, before its result is
given. Leaving ahead_of ends the downloaders, at once when it is left by an
error or a stop, and removes the temporary folder. A live process owns that
folder through a lock (wpt_store.local_files): one killed outright leaves it
behind, and the next round on this machine removes it.

A failed download changes nothing but timing: it is recorded, every error
stays in the downloader, and the part reads what the folder lacks for itself,
meeting that error itself if it is the part's. Every file a downloader keeps
was checked against its sha256 first (wpt_store's fetch_files, file_parts).

Only a store whose reads wait (a bucket) gets downloaders: a folder store's
files are on this machine already. The event "sources.ahead" says, for each
round, what the downloaders did.

All of this changes timing only, so it changes no result.
"""

import collections
import contextlib
import secrets
import shutil
import threading
import time
from pathlib import Path

from wpt_store.local_files import LocalFile, unowned
from wpt_utils.parallel import each

from . import reading, workers

CAP = 100 << 30  # bytes downloaded and not yet used
# downloaders at once: each holds wpt_store.ahead.REQUESTS connections and R2 gives a connection a few MiB/s, so a
# link fills only with many of them; as many as the workers of a 16-core machine, whose memory rule holds them back
DOWNLOADERS = 12
UNDER_WAY, DONE, GONE = "under way", "done", "gone"  # a part's download; gone: its part took its turn, or ended
FOLDER = "ahead-"  # the start of the name of a round's temporary folder


def _name(part) -> tuple[str, int]:
    return part.source.id, part.number


class _Copies:
    """A view of copies_in that also copies into its folder a file it reads
    whole, so the statements a downloader reads wait for the part's read."""

    def __init__(self, view):
        self.view = view

    def __getattr__(self, name):
        return getattr(self.view, name)

    def __repr__(self):
        return repr(self.view)  # the store a failed download's error names

    def read_file(self, placed) -> bytes:
        self.view.fetch_files([placed])
        return self.view.read_file(placed)


class Downloaded:
    """A store for one part's read: its local copies are those of folder,
    which a downloader filled (download) and which outlives the read."""

    def __init__(self, store, folder):
        self.store, self.folder = store, folder

    def __getattr__(self, name):
        return getattr(self.store, name)

    def __repr__(self):
        return repr(self.store)

    def __reduce__(self):
        return Downloaded, (self.store, self.folder)

    def copies_in(self, _):
        return self.store.copies_in(self.folder)


def download(source, store, positions, vocabulary, cache, folder, telemetry) -> int:
    """Copy into folder what a read of a source's chosen files (read_tokens
    of that vocabulary, or read_markdown without one) will ask for, in a
    worker process: its statements, and each chosen file that no pack holds
    and that lies in a bundle, unless the read's tokens are known. The bytes
    of those files."""
    copies = _Copies(store.copies_in(folder))
    tree, _, paths, key = reading.opened(source, copies, positions, telemetry)
    if vocabulary is not None and store.get_ref("source-tokens", reading.tokens_ref(key, vocabulary, paths)):
        return 0
    packed = frozenset().union(*(pack.files for pack in reading.packs(Path(cache) / key)))
    places = [tree[path].placed for path in paths if path not in packed and tree[path].placed is not None]
    copies.fetch_files([placed for placed in places if placed.length < reading.FETCHED_AT_MOST])
    for placed in places:
        if placed.length >= reading.FETCHED_AT_MOST:
            copies.file_path(placed)
    return sum(placed.length for placed in places)


class Downloads:
    """What a round's downloaders have done, part by part. Without parts it
    is off: every part reads from store as it is."""

    def __init__(self, store, telemetry=None, parts=(), folder=None, limit=0):
        self.store, self.telemetry, self.limit = store, telemetry, limit
        self.counts = collections.Counter()  # what the event "sources.ahead" says
        self._folders = {_name(part): Path(folder or "") / str(number) for number, part in enumerate(parts)}
        self._states = {}  # each part's whose turn came, by _name
        self._held = 0  # the bytes counted against the limit
        self._stopped = False
        self._changed = threading.Condition()

    def upcoming(self, parts):
        """The parts, in their order, each once the limit has room for it
        (counted from then on), past those whose turn came first; none more
        after stop()."""
        for part in parts:
            began = time.monotonic()
            with self._changed:
                while not self._stopped and _name(part) not in self._states and self._full(part):
                    self._changed.wait()
                self.counts["full_seconds"] += time.monotonic() - began
                if self._stopped:
                    return
                if _name(part) in self._states:
                    continue
                self._states[_name(part)] = UNDER_WAY
                self._held += part.size
            yield part

    def _full(self, part) -> bool:
        return self._held > 0 and self._held + part.size > self.limit

    def run(self, part, work):
        """A part's download, work(its folder), which gives the bytes it
        copied. An error of work is recorded and stops here."""
        copied, failed = 0, 0
        try:
            copied = work(self._folders[_name(part)])
        except Exception as error:
            failed = 1
            self.telemetry.event("sources.ahead_failed", source=part.source.id, part=part.number,
                                 error=f"{type(error).__name__}: {error}")
        with self._changed:
            self.counts["bytes"] += copied
            self.counts["failed"] += failed
            ended = self._states[_name(part)] == GONE  # its part ended first
            self._states[_name(part)] = DONE
            self._changed.notify_all()
        if ended:
            self.release(part)

    def take(self, part):
        """The store a part reads from, when its turn comes: Downloaded once
        its download ended, the store itself when that never started."""
        if _name(part) not in self._folders:
            return self.store
        began = time.monotonic()
        with self._changed:
            state = self._states.setdefault(_name(part), GONE)
            if state == GONE:
                self.counts["alone"] += 1
                self._changed.notify_all()
                return self.store
            self.counts["waited" if state == UNDER_WAY else "ready"] += 1
            while self._states[_name(part)] == UNDER_WAY:
                self._changed.wait()
            if state == UNDER_WAY:
                self.counts["waited_seconds"] += time.monotonic() - began
        return Downloaded(self.store, self._folders[_name(part)])

    def release(self, part):
        """Remove what is left of a part that ended, and free its bytes;
        a download under way does so when it ends."""
        if _name(part) not in self._folders:
            return
        with self._changed:
            state = self._states.get(_name(part))
            self._states[_name(part)] = GONE
            self._changed.notify_all()
        if state == DONE:
            shutil.rmtree(self._folders[_name(part)], ignore_errors=True)
            with self._changed:
                self._held -= part.size
                self._changed.notify_all()

    def stop(self):
        with self._changed:
            self._stopped = True
            self._changed.notify_all()


@contextlib.contextmanager
def _temporary_folder(store):
    """A new temporary folder in the store's cache folder, owned by this
    process through its lock file while the with block runs, then removed,
    once those no live process owns are removed (local_files.unowned). One
    the system refuses to remove stays, for a later round: removing it is
    not the build's work, so it never stops one."""
    for left in unowned(store.cache, FOLDER):
        with contextlib.suppress(OSError):
            left.remove()
    local = LocalFile(store.cache / (FOLDER + secrets.token_hex(8)))
    try:
        local.path().mkdir()
        yield local.path()
    finally:
        with contextlib.suppress(OSError):
            local.remove()


@contextlib.contextmanager
def ahead_of(parts, arguments, repo, store, telemetry):
    """Downloads of these parts of a round, downloaders running for the
    with block's length, on a store whose reads wait; off when arguments is
    None. arguments(part): the positions, vocabulary and cache folder of
    download."""
    parts = [part for part in parts if part.size]
    if arguments is None or not store.kind.ahead or not parts:
        yield Downloads(store)
        return
    with _temporary_folder(store) as folder:
        pool = workers.Pool(repo, "downloads ahead", disk=lambda part: part.size, folder=folder, network=DOWNLOADERS)
        free = pool.sizing.free_at_start
        downloads = Downloads(store, telemetry, parts, folder, CAP if free is None else min(CAP, free // 4))
        parent = telemetry.current()

        def fetch(part):
            def copied(into):
                return pool.call(download, part.source, store, *arguments(part), str(into), telemetry=telemetry)

            with telemetry.within(parent):
                downloads.run(part, copied)

        def feed():
            for _ in each(fetch, downloads.upcoming(parts), Exception, pool.sizing):
                pass

        feeding = threading.Thread(target=feed, name="downloads-ahead")
        try:
            with pool:
                feeding.start()
                try:
                    yield downloads
                finally:
                    downloads.stop()
        finally:
            feeding.join()
            telemetry.event("sources.ahead", parts=len(parts), limit=downloads.limit, why=pool.sizing.why,
                            values={name: round(value, 3) for name, value in sorted(downloads.counts.items())})
