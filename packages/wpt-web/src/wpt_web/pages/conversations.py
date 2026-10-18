"""The conversation editor's routes: the conversations of a chat file of a
source, one replaced, added or removed as the next patch (the chat rules of
docs/chat-format.md, model drafts included, apply to the whole file); and
a model's draft of the next assistant message (drafts.py), which the
person edits or confirms before it is saved. As `wpt-dataset
conversations`, `conversation` and `save-conversation`, through the
service layer."""

from ..server import Route
from .common import PATCH, SAVED, STRING, VERSION, refusals


def routes(context, drafter):
    repo = context.repo

    @refusals
    def conversations(values, body):
        version, found = repo.conversations(values["id"], values["path"], context.store)
        return 200, {"version": version, "conversations": found}

    @refusals
    def save(values, body):
        if not (values["line"] == "new" or values["line"].isdigit() and int(values["line"]) >= 1):
            return 404, {"problems": [f"no line {values['line']!r}: a number from 1, or new"]}
        line = None if values["line"] == "new" else int(values["line"])
        return 200, {"version": repo.save_conversation(values["id"], values["path"], line, body["conversation"],
                                                       body["patch"], body["version"], context.store)}

    def draft(values, body):
        return drafter.draft(body["model"], body["conversation"], body.get("max_new_tokens"))

    return [
        Route("GET", "/api/sources/{id}/conversations/{path}", conversations,
              "the conversations of a chat file, one per line, in the current shape (a person's text)",
              answer={"type": "object", "required": ["version", "conversations"],
                      "properties": {"version": STRING, "conversations": {"type": "array"}}}),
        Route("PUT", "/api/sources/{id}/conversations/{path}/{line}", save,
              "replace the conversation on a line (from 1), add one (new), or remove it (null), in a chat file of "
              "an empty source, as the next patch; each model draft's edited is set from its content; a new chat "
              "file gets its own chat format entry (a person's text)",
              body={"type": "object", "required": ["conversation", "patch", "version"],
                    "properties": {"conversation": {"type": ["object", "null"]}, "patch": PATCH,
                                   "version": VERSION}}, answer=SAVED),
        Route("POST", "/api/drafts", draft,
              "a WPT model folder's draft of the next assistant message: content, and the metadata.draft that "
              "records it (docs/chat-format.md, \"Model drafts\")",
              body={"type": "object", "required": ["model", "conversation"],
                    "properties": {"model": {"type": "string", "minLength": 1,
                                             "description": "the model folder, on this machine"},
                                   "conversation": {"type": "object"},
                                   "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 4096}}},
              answer={"type": "object", "required": ["message"], "properties": {"message": {"type": "object"}}}),
    ]
