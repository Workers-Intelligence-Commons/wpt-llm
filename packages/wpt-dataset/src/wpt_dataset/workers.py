"""Conversion in worker processes.

A source's conversion, and every reading of a source's conversations, of
its tasks and of the documents tasks point into, runs in a worker process,
never in the process that asked for it: several run at once, each on its
own processor and in its own memory, as many as wpt_utils.parallel allows,
each worker's memory measured as it runs (Processes) and, with a disk to
count, the size of the files a source's conversion will copy, which it states
beforehand (sources.copy_size). The results
come back to the asking process and are put together there in the items'
order, so they are the ones a conversion of one item at a time gives.

A worker opens the asking process's repository (Repo), uses its code()
and its code outside the data folder (wpt_utils.programs.codes), so its
fingerprints are the asking process's, and the store each item is given
(wpt_store opens the same store there). What it records goes into the
run's record under the span the asking thread is in (wpt_telemetry's
join), each item a part of its own.

    with Pool(repo, "sources at once") as pool:
        pool.call(function, *arguments, telemetry=telemetry)   # from any thread

function is a module-level function the worker imports; it gets the
arguments and, as telemetry=, the record it writes to.
"""

import os

from wpt_telemetry import Telemetry
from wpt_utils.parallel import Sizing, process_pool
from wpt_utils.programs import codes, set_code

from .license_check import LicenseTexts
from .repo import DATA_FOLDER, Repo

_REPO = None  # in a worker process: the asking process's repository, set once by _start
_TEXTS = None  # in a worker process: its repository's license texts, read on their first use


def repo() -> Repo:
    """The repository of the process that asked for this work (in a worker
    process), as Repo gives it."""
    if _REPO is None:
        raise RuntimeError("wpt_dataset.workers.repo() is for a worker process")
    return _REPO


def license_texts() -> LicenseTexts:
    """The license texts of repo() (license_check.LicenseTexts), read once
    per worker process: a worker lives for one piece of work, in which the
    repository does not change."""
    global _TEXTS
    if _TEXTS is None:
        _TEXTS = LicenseTexts(repo().license_texts())
    return _TEXTS


def _start(root: str, found: dict):
    global _REPO
    set_code(found)
    _REPO = Repo(root)


def _run(function, arguments, record_store, command, environment):
    if not environment:
        return function(*arguments, telemetry=Telemetry.off())
    with Telemetry.join(record_store, command, {**os.environ, **environment}) as telemetry:
        return function(*arguments, telemetry=telemetry)


class Pool:
    """Worker processes for one piece of work, at most sizing.limit of
    them, each started when first needed; sizing is the work's
    wpt_utils.parallel.Sizing (what names its items; disk and folder as
    there), measuring the workers' memory. A with block owns them: leaving
    it ends them, at once when it is left by an error or a stop."""

    def __init__(self, repo: Repo, what: str, disk=None, folder=None):
        self.sizing = Sizing(what, disk=disk, folder=folder)
        self._root = str(repo.root)
        self._pool = None

    def __enter__(self):
        self._pool = process_pool(self.sizing, self.sizing.limit, _start, (self._root, codes(DATA_FOLDER)))
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self._pool.terminate_workers()
        self._pool.shutdown(wait=error is None, cancel_futures=True)
        return False

    def call(self, function, *arguments, telemetry=None):
        """function(*arguments, telemetry=...) in a worker, waited for; its
        error raised here. With a telemetry, the worker records under the
        span this thread is in."""
        telemetry = telemetry or Telemetry.off()
        environment = telemetry.child_environment()
        return self._pool.submit(_run, function, arguments, telemetry.store, telemetry.command,
                                 environment).result()
