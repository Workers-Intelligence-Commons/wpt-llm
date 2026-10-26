"""The sources pages: list every source with its problems; show and save its
source.json (the same rules as `wpt-dataset check`, without downloading:
`wpt-dataset check --fetch ID` still checks the license and format maps);
start a new empty source for a person's own text; list, read and change
its files, always as a new patch; give a path its own tags. Every route
goes through the service layer (wpt_dataset.repo), as the `wpt-dataset`
commands of the same names do."""

import base64

from wpt_dataset import encoding
from wpt_dataset.errors import NotFound, NotText, RuleViolation
from wpt_dataset.repo import empty_source

from ..server import Route
from .common import PATCH, SAVED, STRING, VERSION, refusals

FORMATS = ["text", "markdown", "chat"]  # a person's own text, or conversations (written in the conversation editor)
FILE = {"type": "object", "required": ["mode", "size", "format", "tags"],
        "properties": {"mode": STRING, "size": {"type": "integer"}, "format": {"type": ["string", "null"]},
                       "tags": {"type": "array", "items": STRING}}}


def routes(context):
    repo = context.repo

    def summary(sid):
        """One source's line in the list, its problems from the one read that
        checks it; a folder with no readable source.json is listed with its
        problem, never a failure of the list."""
        try:
            found = repo.read("source", sid)
            document, problems = found.document, found.problems
        except (NotFound, RuleViolation):
            document, problems = None, repo.check_source(sid)
        document = document if isinstance(document, dict) else {}
        decision = document.get("decision") if isinstance(document.get("decision"), dict) else {}
        licenses = document.get("licenses") if isinstance(document.get("licenses"), list) else []
        return {"id": sid, "address": document.get("address"),
                "type": document.get("type"), "decision": decision.get("decision"), "reason": decision.get("reason"),
                "licenses": [e.get("license") for e in licenses if isinstance(e, dict)],
                "problems": problems}

    def list_sources(values, body):
        return 200, [summary(sid) for sid in repo.source_ids()]

    @refusals
    def read_source(values, body):
        found = repo.read("source", values["id"])
        return 200, {"id": values["id"], "document": found.document, "version": found.version,
                     "problems": found.problems}

    @refusals
    def save_source(values, body):
        if values["id"] not in repo.source_ids():
            return 404, {"problems": [f"no source {values['id']}"]}
        return 200, {"version": repo.save_source(values["id"], body["document"], body["version"]), "problems": []}

    @refusals
    def new_source(values, body):
        formats = {path: file["format"] for path, file in body["files"].items()}
        document = empty_source(body["name"], license=body["license"], attribution=body["attribution"],
                                text=body["license_text"], by=body["by"], formats=formats, tags=body.get("tags", []))
        files = {path: file["text"].encode("utf-8") for path, file in body["files"].items() if file["text"]}
        sid = repo.create_source(body["origin"], body.get("item") or None, document, files, body["patch"],
                                 context.store if files else None)
        return 200, {"id": sid}

    @refusals
    def list_files(values, body):
        version, files = repo.source_files(values["id"], context.store)
        return 200, {"version": version, "files": files}

    @refusals
    def read_file(values, body):
        sid, path = values["id"], values["path"]
        version, data = repo.read_source_file(sid, path, context.store)
        try:
            return 200, {"version": version, "text": encoding.source_text(repo.source(sid), path, data)}
        except NotText:
            return 200, {"version": version, "base64": base64.b64encode(data).decode()}

    def encoded(sid, path, text):
        """text in the encoding the file it replaces states (encoding.encode),
        so a patch changes only what the person changed; UTF-8 for a new file."""
        try:
            _, data = repo.read_source_file(sid, path, context.store)
        except NotFound:
            return text.encode("utf-8")
        return encoding.encode(text, data, encoding.entry_of(repo.source(sid), path), f"{sid}/{path}")

    @refusals
    def write_file(values, body):
        content = None if body["text"] is None else encoded(values["id"], values["path"], body["text"])
        return 200, {"version": repo.change_source_files(values["id"], {values["path"]: content}, body["patch"],
                                                         body["version"], context.store)}

    @refusals
    def set_tags(values, body):
        return 200, {"version": repo.set_tags(values["id"], values["path"], body["tags"], body["version"])}

    new_file = {"type": "object", "required": ["format", "text"],
                "properties": {"format": {"enum": FORMATS}, "text": STRING}}
    return [
        Route("GET", "/api/sources", list_sources, "every source, with what the check finds in it",
              answer={"type": "array", "items": {"type": "object"}}),
        Route("GET", "/api/sources/{id}", read_source, "a source's source.json, its version and problems",
              answer={"type": "object", "required": ["document", "version", "problems"]}),
        Route("PUT", "/api/sources/{id}", save_source, "write source.json through the source rules",
              body={"type": "object", "required": ["document", "version"],
                    "properties": {"document": {}, "version": VERSION}}, answer=SAVED),
        Route("POST", "/api/sources", new_source,
              "start an empty source for a person's own text (decision \"under-review\"): each file's format entry, "
              "and the files with text as its first patch (a chat file with no text yet gets only its entry)",
              body={"type": "object", "required": ["origin", "name", "license", "attribution", "license_text", "by",
                                                   "files", "patch"],
                    "properties": {"origin": STRING, "item": {"type": ["string", "null"]}, "name": STRING,
                                   "license": STRING, "attribution": STRING, "license_text": STRING, "by": STRING,
                                   "tags": {"type": "array", "items": STRING},
                                   "files": {"type": "object", "additionalProperties": new_file}, "patch": PATCH}},
              answer={"type": "object", "required": ["id"], "properties": {"id": STRING}}),
        Route("GET", "/api/sources/{id}/files", list_files,
              "a source's final files (after cuts and patches; format \"joined\" for a file a records entry's `join` reads)",
              answer={"type": "object", "required": ["version", "files"],
                      "properties": {"version": STRING, "files": {"type": "object", "additionalProperties": FILE}}}),
        Route("GET", "/api/sources/{id}/files/{path}", read_file,
              "one final file: its text (UTF-8, or the encoding it states), or base64 when it is not text "
              "(a person's text)",
              answer={"type": "object", "required": ["version"],
                      "properties": {"version": STRING, "text": STRING, "base64": STRING}}),
        Route("PUT", "/api/sources/{id}/files/{path}", write_file,
              "add, replace (text) or remove (null) a final file, recorded as the next patch (a person's text)",
              body={"type": "object", "required": ["text", "patch", "version"],
                    "properties": {"text": {"type": ["string", "null"]}, "patch": PATCH, "version": VERSION}},
              answer=SAVED),
        Route("PUT", "/api/sources/{id}/tags/{path}", set_tags,
              "give exactly that path its own tags entry in source.json (no tags: remove it)",
              body={"type": "object", "required": ["tags", "version"],
                    "properties": {"tags": {"type": "array", "items": STRING}, "version": VERSION}}, answer=SAVED),
    ]
