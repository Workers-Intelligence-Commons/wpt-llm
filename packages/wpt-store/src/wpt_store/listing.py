"""What the store's index (index.py) says about its sessions, read from it
alone: the listing `wpt store list` prints."""

import json
from dataclasses import dataclass

from .index import SOURCES


@dataclass(frozen=True)
class SourceState:
    """One source as one session holds it: the time of its file or its
    failure, the address used, its bytes, what the download got (a dict,
    for the record), and the error that failed it, None when it did not."""

    session: str
    source: str
    time: str
    address: str | None
    size: int | None
    got: dict | None
    error: str | None


@dataclass(frozen=True)
class SessionRow:
    """One session: its ID, and how many things of each kind it holds."""

    session: str
    held: dict[str, int]


def listing(store) -> tuple[list[SourceState], list[SessionRow]]:
    """(every source of every session, by session then id; every session,
    by ID)."""
    with store.index() as found:
        rows = found.execute("SELECT session, name, time, about FROM held WHERE kind = ? ORDER BY session, name",
                             (SOURCES,)).fetchall()
        counts = found.execute("SELECT session, kind, COUNT(*) FROM held GROUP BY session, kind ORDER BY session, kind"
                               ).fetchall()
    states = []
    for session, name, time, about in rows:
        about = json.loads(about)
        error = about.get("error")
        states.append(SourceState(session, name, time, about.get("address"), about.get("size"), about.get("got"),
                                  f"{error['kind']}: {error['message']}" if error else None))
    sessions = {}
    for session, kind, count in counts:
        sessions.setdefault(session, {})[kind] = count
    return states, [SessionRow(session, held) for session, held in sessions.items()]
