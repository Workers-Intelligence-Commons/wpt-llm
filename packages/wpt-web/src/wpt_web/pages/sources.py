"""The sources page: list every source with its problems, show one as its
file says, and save an edited one through Repo.save_source, which applies
the same rules as `wpt-dataset check` (without downloading: `wpt-dataset
check --fetch ID` still checks the license and format maps)."""

import json

from wpt_dataset.errors import InvalidFile, OutdatedSchema, RuleViolation, StaleFile
from wpt_dataset.repo import file_version, schema

from ..server import Route


def routes(repo):
    def path(sid):
        return repo.data / "sources" / sid / "source.json"

    def raw(sid):
        try:
            return json.loads(path(sid).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def summary(sid):
        document = raw(sid)
        document = document if isinstance(document, dict) else {}
        decision = document.get("decision") if isinstance(document.get("decision"), dict) else {}
        licenses = document.get("licenses") if isinstance(document.get("licenses"), list) else []
        return {"id": sid, "address": document.get("address"), "schema": document.get("schema"),
                "decision": decision.get("decision"), "reason": decision.get("reason"),
                "licenses": [e.get("license") for e in licenses if isinstance(e, dict)],
                "problems": repo.check_source(sid)}

    def list_sources(values, body):
        return 200, [summary(sid) for sid in repo.source_ids()]

    def read_source(values, body):
        sid = values["id"]
        if sid not in repo.source_ids():
            return 404, {"problems": [f"no source {sid}"]}
        return 200, {"id": sid, "document": raw(sid), "version": file_version(path(sid)),
                     "problems": repo.check_source(sid)}

    def save_source(values, body):
        sid = values["id"]
        if sid not in repo.source_ids():
            return 404, {"problems": [f"no source {sid}"]}
        if not isinstance(body, dict) or "document" not in body or not isinstance(body.get("version"), str):
            return 400, {"problems": ["send {document, version}"]}
        try:
            version = repo.save_source(sid, body["document"], body["version"])
        except StaleFile as error:
            return 409, {"problems": [str(error)]}
        except (InvalidFile, OutdatedSchema, RuleViolation) as error:
            return 422, {"problems": [str(error)]}
        return 200, {"version": version, "problems": []}

    def source_schema(values, body):
        return 200, schema("source")

    return [Route("GET", "/api/sources", list_sources),
            Route("GET", "/api/sources/{id}", read_source),
            Route("PUT", "/api/sources/{id}", save_source),
            Route("GET", "/api/schemas/source", source_schema)]
