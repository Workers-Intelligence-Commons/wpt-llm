"""The documents pages: every kind of data file other than a source's own
files (recipes, Dockerfiles, the signer lists, and the generated locks and
license lists, which are only read here), and the JSON schema of each JSON
kind, which the recipe editors are built from. As `wpt-dataset kinds`,
`list`, `show` and `save`, through the service layer."""

from wpt_dataset.errors import NotFound
from wpt_dataset.repo import KINDS, schema

from ..server import Route
from .common import SAVED, STRING, VERSION, refusals

INSIDE_SOURCES = ("conversation", "task")  # shapes of files inside a source: a chat file's lines, a task.json


def routes(context):
    repo = context.repo

    def kinds(values, body):
        return 200, [kind | {"names": repo.names(kind["name"])} for kind in repo.kinds()]

    @refusals
    def names(values, body):
        return 200, repo.names(values["kind"])

    @refusals
    def read(values, body):
        found = repo.read(values["kind"], values["name"])
        return 200, {"kind": found.kind, "name": found.name, "text": found.text, "document": found.document,
                     "version": found.version, "problems": found.problems}

    @refusals
    def save(values, body):
        kind = values["kind"]
        if kind in KINDS and KINDS[kind].form == "json":
            if "document" not in body:
                return 400, {"problems": [f"{kind} is JSON: send {{document, version}}"]}
            content = body["document"]
        else:
            if not isinstance(body.get("text"), str):
                return 400, {"problems": [f"{kind} is text: send {{text, version}}"]}
            content = body["text"]
        return 200, {"version": repo.save(kind, values["name"], content, body["version"])}

    def kind_schema(values, body):
        kind = KINDS.get(values["kind"])
        name = kind.schema if kind is not None else values["kind"] if values["kind"] in INSIDE_SOURCES else None
        if name is None:
            return 404, {"problems": [str(NotFound(f"no JSON schema for {values['kind']!r}"))]}
        return 200, schema(name)

    return [
        Route("GET", "/api/kinds", kinds, "every kind of data file, who writes it, and the names that exist",
              answer={"type": "array", "items": {"type": "object"}}),
        Route("GET", "/api/documents/{kind}", names, "the names of the files of a kind",
              answer={"type": "array", "items": STRING}),
        Route("GET", "/api/documents/{kind}/{name}", read,
              "a file: its text, its JSON, its version and what the check finds in it",
              answer={"type": "object", "required": ["text", "document", "version", "problems"]}),
        Route("PUT", "/api/documents/{kind}/{name}", save,
              "write a file people write (not a generated one) through the rules of the check; version \"\" "
              "makes a new one",
              body={"type": "object", "required": ["version"],
                    "properties": {"document": {}, "text": STRING, "version": VERSION}}, answer=SAVED),
        Route("GET", "/api/schemas/{kind}", kind_schema,
              "the JSON schema of a JSON kind, or of a conversation or a task", answer={"type": "object"}),
    ]
