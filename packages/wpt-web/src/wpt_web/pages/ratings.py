"""The rating page's routes: the wpt-eval runs of the store that wrote
attempts, one run's attempts (each with its request, its result and its
changes to /work) and the weights it measured; and the ratings of a
rating file of a source, one replaced, added or removed as the next patch
(the rating rules of docs/evaluation.md, "People's ratings"). The ratings
go through the service layer, as `wpt-dataset ratings` and
`save-rating`; the attempts come from the store's run records, read here,
the page being their only reader.

A wpt-eval run's record folder (wpt_telemetry; docs/observability.md)
holds attempts.jsonl, which wpt-eval writes (wpt_eval.attempts.RECORD_FILE,
whose name ATTEMPTS repeats: wpt-web, a package, cannot import an
application), and ids.json, whose inputs name the weights it measured."""

import json

from wpt_dataset.errors import NotFound
from wpt_telemetry import IDS

from ..server import Route
from .common import PATCH, SAVED, STRING, VERSION, refusals

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


def routes(context):
    repo = context.repo

    @refusals
    def listed(values, body):
        return 200, runs(context.store)

    @refusals
    def attempts_of(values, body):
        weights, found = attempts(context.store, values["folder"])
        return 200, {"weights": weights, "attempts": found}

    @refusals
    def ratings(values, body):
        version, found = repo.ratings(values["id"], values["path"], context.store)
        return 200, {"version": version, "ratings": found}

    @refusals
    def save(values, body):
        if not (values["line"] == "new" or values["line"].isdigit() and int(values["line"]) >= 1):
            return 404, {"problems": [f"no line {values['line']!r}: a number from 1, or new"]}
        line = None if values["line"] == "new" else int(values["line"])
        return 200, {"version": repo.save_rating(values["id"], values["path"], line, body["rating"], body["patch"],
                                                 body["version"], context.store)}

    return [
        Route("GET", "/api/runs", listed,
              "the record folders of the wpt-eval runs in the store that wrote attempts (attempts.jsonl), oldest "
              "first", answer={"type": "array", "items": STRING}),
        Route("GET", "/api/runs/{folder}/attempts", attempts_of,
              "the attempts a wpt-eval run wrote into its record folder (task, kind, index, plain, dropped, "
              "conversation, and for a scored attempt its score and its changes to /work), and the weights it "
              "measured ({weight file: sha256})",
              answer={"type": "object", "required": ["weights", "attempts"],
                      "properties": {"weights": {"type": "object", "additionalProperties": STRING},
                                     "attempts": {"type": "array", "items": {"type": "object"}}}}),
        Route("GET", "/api/sources/{id}/ratings/{path}", ratings,
              "the ratings of a rating file, one per line: the attempt each rates, the verdict, who and when (a "
              "person's text)",
              answer={"type": "object", "required": ["version", "ratings"],
                      "properties": {"version": STRING, "ratings": {"type": "array"}}}),
        Route("PUT", "/api/sources/{id}/ratings/{path}/{line}", save,
              "replace the rating on a line (from 1), add one (new), or remove it (null), in a rating file of an "
              "empty source, as the next patch; a new rating file gets its own rating format entry (a person's "
              "text)",
              body={"type": "object", "required": ["rating", "patch", "version"],
                    "properties": {"rating": {"type": ["object", "null"]}, "patch": PATCH, "version": VERSION}},
              answer=SAVED),
    ]
