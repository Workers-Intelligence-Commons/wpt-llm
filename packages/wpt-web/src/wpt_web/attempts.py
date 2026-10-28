"""The attempts of the wpt-eval runs in the store, which people rate (the
rating page is their only reader): which runs wrote attempts, and one
run's attempts (each with its request, its result and its changes to
/work) with the weights it measured. It reads the store's run records and
changes nothing.

A wpt-eval run's record folder (wpt_telemetry; docs/observability.md)
holds attempts.jsonl, which wpt-eval writes (wpt_eval.attempts.RECORD_FILE,
whose name ATTEMPTS repeats: wpt-web, a package, cannot import an
application), and ids.json, whose inputs name the weights it measured."""

import json

from wpt_dataset.errors import NotFound
from wpt_telemetry import IDS

ATTEMPTS = "attempts.jsonl"  # in a wpt-eval run's record folder: one attempt per line
EVAL = "-wpt-eval-"  # in the name of every wpt-eval run's record folder (wpt_telemetry.Telemetry.start)


def runs(store) -> list[str]:
    """The record folders of the wpt-eval runs in the store that wrote
    attempts, oldest first (the folders' names start with their UTC time)."""
    return [folder for folder in store.record_folders() if EVAL in folder and ATTEMPTS in store.record_files(folder)]


def attempts(store, folder: str) -> tuple[dict, list[dict]]:
    """({weight file: sha256} the run measured, [each line of its
    attempts.jsonl]) of a wpt-eval run's record folder; a last line cut
    short (a run that stopped while writing it) is left out. NotFound when
    the folder is not a wpt-eval run's or holds no attempts."""
    try:
        files = store.record_files(folder)
    except ValueError:  # not a record folder's name
        files = []
    if EVAL not in folder or ATTEMPTS not in files:
        raise NotFound(f"no attempts of a wpt-eval run in the record folder {folder!r}")
    ids = json.loads(store.read_record(folder, IDS)) if IDS in files else {}
    data = store.read_record(folder, ATTEMPTS)
    lines = data.decode("utf-8", errors="replace").split("\n")
    found = [json.loads(line) for line in lines[:-1] if line.strip()]
    if lines[-1].strip():  # no line break after it: the run stopped while writing it
        try:
            found.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            pass
    return ids.get("inputs", {}).get("weights", {}), found
