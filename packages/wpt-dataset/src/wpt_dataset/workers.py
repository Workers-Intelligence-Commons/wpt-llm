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
and its code_of() of every module whose code a worker's fingerprints hold
(KEYED, wpt_utils.programs.codes), so its fingerprints are the asking
process's, and the store each item is given
(wpt_store opens the same store there). What it records goes into the
run's record under the span the asking thread is in (wpt_telemetry's
join): the worker holds its lines back (Held) and the asking process adds
them to the record when the item ends, so an item writes no part of the
record of its own, and a worker that dies loses the lines of its item. A
ref a worker sets names a result that can be computed again
(wpt_telemetry.set_result_ref), never a step that cannot be undone.

    with Pool(repo, "sources at once") as pool:
        pool.call(function, *arguments, telemetry=telemetry)   # from any thread

function is a module-level function the worker imports; it gets the
arguments and, as telemetry=, the record it writes to.
"""

import os

from wpt_telemetry import START, Telemetry
from wpt_utils.parallel import Sizing, process_pool
from wpt_utils.programs import codes, set_code

from .license_check import LicenseTexts

# the modules whose code (wpt_utils.programs.code_of) the fingerprints a worker computes hold: the final tree and
# its text check (sources), what a source keeps (kept), a source's Markdown and tokens (reading), the index of a
# records entry under `newest` (records)
KEYED = ("wpt_dataset.sources", "wpt_dataset.kept", "wpt_dataset.reading", "wpt_dataset.records")
# every file that is not Python in the src folders of the packages of the KEYED modules, and whether it enters the
# keys. None does: the schemas only refuse a document, and the site list only decides which sites a download
# may reach, so neither decides a saved result. (A compiled module is no entry: it enters by the files that build it.)
NOT_PYTHON = {
    "wpt_dataset/sites.txt": False,
    **{f"wpt_dataset/schemas/{kind}.schema.json": False for kind in (
        "conversation", "image-licenses", "image-lock", "image-tools", "license-text", "lock", "mixture", "model",
        "probe", "rating", "series", "source", "task", "tokenizer")},
}

_REPO = None  # in a worker process: the asking process's repository, set once by _start
_TEXTS = None  # in a worker process: its repository's license texts, read on their first use


def repo():
    """The repository of the process that asked for this work (in a worker
    process), as Repo gives it. (This module does not import repo.py: the
    class comes from the asking process, so the code of a worker's
    fingerprints stays that of the modules it computes them with.)"""
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


def _start(opener, root: str, found: dict):
    global _REPO
    set_code(found)
    _REPO = opener(root)


class Held:
    """The store a worker records through: the lines it adds wait in
    `lines`, [(file name, lines)], for the asking process. Every other call
    goes to the store itself."""

    def __init__(self, store):
        self.store, self.lines = store, []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def read_record(self, folder, name):
        return b"" if name == START else self.store.read_record(folder, name)  # the asking process wrote it

    def append_record(self, folder, name, data):
        self.lines.append((name, data))

    def flush_records(self):
        pass

    def __str__(self):
        return str(self.store)


def _run(function, arguments, record_store, command, environment):
    """(function's result, the error it raised, the lines its record holds
    back) in a worker."""
    if not environment:
        return function(*arguments, telemetry=Telemetry.off()), None, []
    held = Held(record_store)
    try:
        with Telemetry.join(held, command, {**os.environ, **environment}) as telemetry:
            return function(*arguments, telemetry=telemetry), None, held.lines
    except Exception as error:
        return None, error, held.lines


class Pool:
    """Worker processes for one piece of work, at most sizing.limit of
    them, each started when first needed; sizing is the work's
    wpt_utils.parallel.Sizing (what names its items; disk, folder and
    network as there), measuring the workers' memory. A with block owns them: leaving
    it ends them, at once when it is left by an error or a stop."""

    def __init__(self, repo, what: str, disk=None, folder=None, network=None):
        self.sizing = Sizing(what, disk=disk, folder=folder, network=network)
        self._opener, self._root = type(repo), str(repo.root)
        self._pool = None

    def __enter__(self):
        self._pool = process_pool(self.sizing, self.sizing.limit, _start, (self._opener, self._root, codes(*KEYED)))
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
        environment = telemetry.child_environment(flush=False)
        value, error, lines = self._pool.submit(_run, function, arguments, telemetry.store, telemetry.command,
                                                environment).result()
        telemetry.add_lines(lines)
        if error is not None:
            raise error
        return value
