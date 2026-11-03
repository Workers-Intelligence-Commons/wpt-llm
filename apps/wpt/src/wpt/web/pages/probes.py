"""The probe page's routes: the probes of a probe file of a source, one
replaced, added or removed as the next patch (the probe rules of
docs/evaluation.md, "Probes"). They go through the service layer, as
`wpt dataset probes` and `save-probe`."""

from ..server import Route
from .common import PATCH, SAVED, STRING, VERSION, refusals


def routes(context):
    repo = context.repo

    @refusals
    def probes(values, body):
        version, found = repo.probes(values["id"], values["path"], context.store)
        return 200, {"version": version, "probes": found}

    @refusals
    def save(values, body):
        if not (values["line"] == "new" or values["line"].isdigit() and int(values["line"]) >= 1):
            return 404, {"problems": [f"no line {values['line']!r}: a number from 1, or new"]}
        line = None if values["line"] == "new" else int(values["line"])
        return 200, {"version": repo.save_probe(values["id"], values["path"], line, body["probe"], body["patch"],
                                                body["version"], context.store)}

    return [
        Route("GET", "/api/sources/{id}/probes/{path}", probes,
              "the probes of a probe file, one per line: the question the probes test asks and the patterns that "
              "flag an alarming reply (a person's text)",
              answer={"type": "object", "required": ["version", "probes"],
                      "properties": {"version": STRING, "probes": {"type": "array"}}}),
        Route("PUT", "/api/sources/{id}/probes/{path}/{line}", save,
              "replace the probe on a line (from 1), add one at the end (new), or remove it (null), in a probe file "
              "of an empty source, as the next patch; a new probe file gets its own probe format entry (a "
              "person's text)",
              body={"type": "object", "required": ["probe", "patch", "version"],
                    "properties": {"probe": {"type": ["object", "null"]}, "patch": PATCH, "version": VERSION}},
              answer=SAVED),
    ]
