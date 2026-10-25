"""The always-on source server: rounds that keep this checkout's sources
current, one after another, forever, with a pause between two (serve).

A round (one_round), in the checkout it runs in:

1. Each source drafter (by default every built-in) runs with its default
   settings and drafts every candidate again (`again`), each run with its
   own record, as `wpt-dataset source-drafter` runs it. Its saves follow
   the source drafter rules: only a license change sends a source back to
   under-review. A source drafter that cannot run is logged in the round's
   output and the round goes on.
2. Every source is downloaded again in full and checked, in one session of
   the round's command: a draft (checksum all zeros) is filled in as
   `wpt-dataset fill-checksum` fills it; any other is downloaded even when
   its tree is in the store (download.one, again), so a source whose
   upstream no longer gives its checksum fails, and only the failure is
   recorded. Each source that came is then checked as `check --fetch`
   checks it.

Nothing is committed: the drafts and the filled checksums stay in the
checkout. A round returns its failures, one line each, and serve logs
them; the next round comes all the same. A record the store refuses (RecordError) stops the
server, as it stops any recorded command.
"""

import time

from wpt_store import StoreError
from wpt_telemetry import RecordError
from wpt_utils.parallel import Done
from wpt_utils.programs import MissingTool

from . import download, source_drafter
from .errors import DatasetError
from .repo import DRAFT_CHECKSUM

PAUSE = 7 * 24 * 3600  # seconds between the end of a round and the start of the next: one week
FAILURES = (DatasetError, StoreError, MissingTool, OSError)  # what one source or one source drafter may fail with


def serve(repo, store, by, command, arguments, pause=PAUSE, drafters=None, log=print, sleep=time.sleep):
    """Rounds (one_round) without end, pause seconds between two; after
    each, its failures through log."""
    while True:
        failed = one_round(repo, store, by, command, arguments, drafters, log)
        log(f"round done, {len(failed)} failed" + "".join(f"\n  {line}" for line in failed))
        log(f"next round in {pause:,} seconds")
        sleep(pause)


def one_round(repo, store, by, command, arguments, drafters=None, log=print) -> list[str]:
    """One round (see the module's docstring): the source drafters
    (built-in names or files; None: every built-in) with by as the drafts'
    `by`, then every source downloaded again and checked, the session's
    record named after command (wpt-dataset-<command>) with arguments.
    Returns the failures."""
    failed = []
    for target in source_drafter.built_in_names() if drafters is None else drafters:
        failed += _drafted(repo, store, target, by, log)
    return failed + _downloaded(repo, store, command, arguments)


def _drafted(repo, store, target, by, log) -> list[str]:
    """Run one source drafter on every candidate; its failures."""
    name = source_drafter.find(target)[0]
    try:
        with source_drafter.recorded(store, target, source_drafter.arguments(target, by, {}, again=True), "",
                                     repo.root / ".env") as telemetry:
            counts = source_drafter.run(repo, store, target, {}, by, telemetry, again=True)
    except RecordError:
        raise
    except FAILURES as error:
        log(f"{name}: {type(error).__name__}: {error}")
        return [f"{name}: {type(error).__name__}: {error}"]
    return [f"{name}: {counts['failed']} candidates failed (its record lists them)"] if counts["failed"] else []


def _downloaded(repo, store, command, arguments) -> list[str]:
    """Download every source again in one session, filling in the drafts,
    then check each one that came; the failures."""
    sources = [source for source in repo.sources() if not source.unwritten]  # as `check` does: no text yet
    drafts = {source.id for source in sources if source.data["checksum"] == DRAFT_CHECKSUM}
    ids = [source.id for source in sources]
    failed = []
    with download.recorded(store, command, arguments, ids, repo.root / ".env") as telemetry:
        with download.session(store, telemetry) as session:
            def work(sid):
                if sid in drafts:
                    return repo.fill_checksum(sid, store, session, telemetry)
                return download.one(repo.source(sid), store, session, telemetry, again=True)
            results = download.each(work, ids, download.address_in(repo), log=telemetry.log)
        for result in results:
            if result.error is None:
                try:
                    repo.check_fetched(result.item, store)
                except FAILURES as error:
                    result = Done(result.item, error=error)
            if result.error is not None:
                failed.append(f"{result.item}: {type(result.error).__name__}: {result.error}")
                telemetry.log(failed[-1], error=True)
        telemetry.log(f"{len(ids)} sources downloaded again and checked, {len(drafts)} drafts filled in, "
                      f"{len(failed)} failed")
    return failed
