"""What the store database (database.py) says about the sources'
downloads and the sessions that made them, read from it alone: the
listing `wpt-store list` prints."""

from dataclasses import dataclass

from .database import DOWNLOADED

# 1 for each source's last line: the latest time, then session, then place in it
LATEST = "ROW_NUMBER() OVER (PARTITION BY source ORDER BY time DESC, session DESC, number DESC) AS latest"


@dataclass(frozen=True)
class SourceState:
    """One source as the store database knows it: its last line's
    outcome, revision and checksum, and its last download's time, address,
    repository commit and size (None when it was never downloaded)."""

    source: str
    outcome: str
    revision: str | None
    checksum: str | None
    downloaded: str | None
    address: str | None
    commit: str | None
    size: int | None


@dataclass(frozen=True)
class SessionRow:
    folder: str
    command: str
    started: str
    ended: str | None
    commit: str
    dirty: bool


def listing(store) -> tuple[list[SourceState], list[SessionRow]]:
    """(every source the database has a line for, by id; every session, in
    the order they started)."""
    with store.database() as found:
        # each source's last line, of all and of its downloads: one pass over the lines each, not one per line
        last = found.execute(
            f"SELECT source, outcome, revision, checksum FROM (SELECT *, {LATEST} FROM lines) WHERE latest = 1"
            " ORDER BY source").fetchall()
        downloads = {row[0]: row[1:] for row in found.execute(
            "SELECT source, time, address, wpt_commit, size FROM ("
            f" SELECT line.*, sessions.wpt_commit, {LATEST} FROM lines AS line"
            " JOIN sessions ON sessions.folder = line.session WHERE outcome = ?) WHERE latest = 1", (DOWNLOADED,))}
        sessions = [SessionRow(folder, command, started, ended, commit, bool(dirty))
                    for folder, command, started, ended, commit, dirty in found.execute(
                        "SELECT folder, command, started, ended, wpt_commit, dirty FROM sessions"
                        " ORDER BY started, folder")]
    return [SourceState(source, outcome, revision, checksum, *downloads.get(source, (None,) * 4))
            for source, outcome, revision, checksum in last], sessions
