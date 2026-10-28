"""The rating page's routes: the wpt-eval runs of the store that wrote
attempts, one run's attempts and the weights it measured (attempts.py);
and the ratings of a rating file of a source, one replaced, added or
removed as the next patch (the rating rules of docs/evaluation.md,
"People's ratings"), through the service layer, as `wpt-dataset ratings`
and `save-rating`."""

from wpt_dataset.repo import line_number

from .. import attempts
from ..server import Route
from .common import PATCH, SAVED, STRING, VERSION, refusals


def routes(context):
    repo = context.repo

    @refusals
    def listed(values, body):
        return 200, attempts.runs(context.store)

    @refusals
    def attempts_of(values, body):
        weights, found = attempts.attempts(context.store, values["folder"])
        return 200, {"weights": weights, "attempts": found}

    @refusals
    def ratings(values, body):
        version, found = repo.ratings(values["id"], values["path"], context.store)
        return 200, {"version": version, "ratings": found}

    @refusals
    def save(values, body):
        return 200, {"version": repo.save_rating(values["id"], values["path"], line_number(values["line"]),
                                                 body["rating"], body["patch"], body["version"], context.store)}

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
