"""What a store's records say about its runs, read back for people.

    runs(store)            every run, <id>/<run>, with its command, its start
                           and how it ended, from its start.json and end.json
    summary(store, folder) one run: its stage (the spans open at the last
                           line read), how many spans, events and lines it
                           holds, its last events and its errors
    follow(store, folder)  the lines of a run as they reach the store, until
                           its end.json is there

Each line read back for people is its text and the level of wpt_utils.lines
it prints at (described).

A run whose end.json is missing is running, or it crashed: its record cannot
tell which. The lines of a run reach the store in groups (wpt_telemetry), so
a run that is going shows up to FLUSH_SECONDS late.
"""

import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from wpt_store import MissingFile
from wpt_utils.lines import ERROR, INFO

from . import END, EVENTS, START

READERS = 16  # record folders read at once
LAST = 20  # the last events and errors a summary keeps
FOLLOW_SECONDS = 5  # how often follow looks for new lines


@dataclass(frozen=True)
class Run:
    folder: str
    command: str | None  # None when the record has no start.json
    started: str | None
    status: str  # "succeeded", "failed", or "no end" while end.json is not there
    seconds: float | None


@dataclass
class Summary:
    run: Run
    arguments: list
    error: str | None  # the error its end.json names
    stage: list[str]  # the names of the spans open, outermost first
    spans: dict[str, int]  # how many of each span ended
    events: dict[str, int]  # how many of each event
    lines: int  # how many printed lines
    last: list[tuple[str, str]] = field(default_factory=list)  # its last events and printed lines (described)
    errors: list[tuple[str, str]] = field(default_factory=list)  # its last spans that failed and error lines


def _json_file(store, folder, name) -> dict | None:
    try:
        return json.loads(store.read_record(folder, name))
    except MissingFile:
        return None


def run_of(store, folder) -> Run:
    start = _json_file(store, folder, START) or {}
    end = _json_file(store, folder, END) or {}
    return Run(folder, start.get("command"), start.get("started"), end.get("status", "no end"), end.get("seconds"))


def runs(store) -> list[Run]:
    """Every run of the store, oldest first."""
    with ThreadPoolExecutor(READERS) as pool:
        return list(pool.map(lambda folder: run_of(store, folder), store.record_folders()))


def find(store, given: str) -> str:
    """The record folder a name gives: itself, or the only folder holding
    it. ValueError when none does, or several."""
    folders = store.record_folders()
    if given in folders:
        return given
    found = [folder for folder in folders if given in folder]
    if len(found) != 1:
        raise ValueError(f"{len(found)} record folders hold {given!r}" + (f": {', '.join(found[:5])}" if found else ""))
    return found[0]


def described(line: dict) -> tuple[str, str] | None:
    """One recorded line as one line for people and the level it prints at,
    None for a line that only opens or closes a span without a failure."""
    time_ = line.get("time", "")
    match line.get("kind"):
        case "log":
            return f"{time_} {line.get('text')}", line.get("severity", "info").upper()
        case "event":
            return f"{time_} {line.get('name')} {json.dumps(line.get('attributes', {}), ensure_ascii=False)}", INFO
        case "span_end" if line.get("status") == "error":
            return f"{time_} {line.get('name')} failed: {line.get('error')}", ERROR
    return None


def summary(store, folder) -> Summary:
    start = _json_file(store, folder, START) or {}
    end = _json_file(store, folder, END) or {}
    try:
        data = store.read_record(folder, EVENTS)
    except MissingFile:
        data = b""
    opened, spans, events, lines = {}, collections.Counter(), collections.Counter(), 0
    last, errors = collections.deque(maxlen=LAST), collections.deque(maxlen=LAST)
    for raw in data.splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue  # a line a crash left cut
        kind = line.get("kind")
        if kind == "span_start":
            opened[line["span"]] = line
        elif kind == "span_end":
            opened.pop(line["span"], None)
            spans[line["name"]] += 1
        elif kind == "event":
            events[line["name"]] += 1
        elif kind == "log":
            lines += 1
        shown = described(line)
        if shown is not None:
            last.append(shown)
            if shown[1] == ERROR:
                errors.append(shown)
    return Summary(run_of(store, folder), start.get("arguments", []), end.get("error"),
                   [line["name"] for line in opened.values()], dict(spans), dict(events), lines, list(last),
                   list(errors))


def follow(store, folder, seconds=FOLLOW_SECONDS, wait=time.sleep):
    """Each line of the run as described gives it, as it reaches the store,
    from the run's first; it ends once the run's end.json is there and every
    line is read."""
    reader = store.record_reader(folder, EVENTS)
    while True:
        ended = END in store.record_files(folder)
        for raw in reader.read().splitlines():
            try:
                shown = described(json.loads(raw))
            except ValueError:
                continue  # a line a crash left cut
            if shown is not None:
                yield shown
        if ended:
            return
        wait(seconds)
