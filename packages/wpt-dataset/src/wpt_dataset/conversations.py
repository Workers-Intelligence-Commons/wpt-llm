"""Conversations for post-training (docs/chat-format.md).

A file in the `chat` format is JSON Lines: one conversation per line, each
matching schemas/conversation.schema.json, and each tool_result answering a
tool_call made earlier in the same conversation. Such a file is never a
pre-training document (markdown.py gives it no Markdown); a post-training
step reads it through read(), which picks the files tagged with the step's
name.

A chat format entry with `tree` reads instead a file of messages, one per
record (JSON Lines or Parquet, as a records file), each naming its parent:
every path from a first message to a last one becomes a conversation (see
from_tree).
"""

import json
from pathlib import Path

from . import markdown, records
from .errors import InvalidFile
from .repo import Repo, validate
from .sources import source_tree, tags_of


def parse(source_id: str, path: str, data: bytes) -> list[dict]:
    """The conversations of one chat file; InvalidFile names the line at fault."""
    conversations = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        where = f"{source_id}/{path} line {number}"
        if not line.strip():
            raise InvalidFile(f"{where}: empty line; a chat file holds one conversation per line")
        try:
            conversation = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidFile(f"{where}: {error}") from error
        validate(conversation, "conversation", where)
        calls = set()
        for message in conversation["messages"]:
            for block in message["content"]:
                if block["type"] == "tool_call":
                    calls.add(block["id"])
                elif block["type"] == "tool_result" and block["call"] not in calls:
                    raise InvalidFile(f"{where}: tool_result answers {block['call']!r}, which no earlier tool_call made")
        conversations.append(conversation)
    return conversations


def from_tree(source_id: str, path: str, file: Path, spec: dict) -> list[tuple[str, dict]]:
    """[(id of its last message, conversation)] from a file of messages that
    name their parent. spec (a format entry's `tree`) names the fields `id`,
    `parent`, `role` and `text`, maps the file's roles to WPT's (`roles`),
    and lists `drop` rules: a message whose field `is` a value, or is a
    number `above` one, is dropped, and so is everything after it, as is a
    message with no text. Every path from a first message (no parent) to a
    last one becomes a conversation; trailing messages that are not the
    assistant's are left off, and a path with no assistant message, or one
    that repeats another exactly, gives none."""
    where = f"{source_id}/{path}"
    messages, children, roots = {}, {}, []
    for row in records.rows(file, path):
        message_id = records.field(row, spec["id"])
        if message_id is None:
            raise InvalidFile(f"{where}: a message has no {spec['id']}")
        message_id = str(message_id)
        if message_id in messages:
            raise InvalidFile(f"{where}: message {message_id} appears twice")
        role = records.field(row, spec["role"])
        if not isinstance(role, str) or role not in spec["roles"]:
            raise InvalidFile(f"{where}: message {message_id}: role {role!r} is not in tree.roles")
        text = records.field(row, spec["text"])
        dropped = (not isinstance(text, str) or not text.strip()
                   or any(_drops(records.field(row, rule["field"]), rule) for rule in spec.get("drop", [])))
        messages[message_id] = (spec["roles"][role], text, dropped)
        parent = records.field(row, spec["parent"])
        if parent is None:
            roots.append(message_id)
        else:
            children.setdefault(str(parent), []).append(message_id)
    orphans = sorted(set(children) - set(messages))
    if orphans:
        raise InvalidFile(f"{where}: {len(orphans)} messages name a parent the file does not hold, such as {orphans[0]}")

    found, seen = [], set()
    stack = [(root, ()) for root in reversed(roots)]
    while stack:  # depth first, in the file's order
        message_id, before = stack.pop()
        role, text, dropped = messages[message_id]
        if dropped:
            continue
        here = (*before, (message_id, role, text))
        live = [child for child in children.get(message_id, []) if not messages[child][2]]
        if live:
            stack += [(child, here) for child in reversed(live)]
            continue
        kept = list(here)
        while kept and kept[-1][1] != "assistant":
            kept.pop()
        key = tuple((role, text) for _, role, text in kept)
        if not kept or key in seen:
            continue
        seen.add(key)
        conversation = {"schema": 1, "messages": [{"role": role, "content": [{"type": "text", "text": text}]}
                                                  for role, text in key]}
        validate(conversation, "conversation", f"{where} message {kept[-1][0]}")
        found.append((kept[-1][0], conversation))
    return found


def _drops(value, rule) -> bool:
    if "is" in rule:
        return value == rule["is"]
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > rule["above"]


def of_file(source_id: str, path: str, entry: dict, store, sha256: str) -> list[tuple[str, dict]]:
    """[(key, conversation)] of one chat file: its line number, or with
    `tree` its last message's id. InvalidFile names what is wrong."""
    if "tree" in entry:
        return from_tree(source_id, path, store.path(sha256), entry["tree"])
    return list(enumerate(parse(source_id, path, store.get_bytes(sha256)), start=1))


def read(repo: Repo, store, tag: str, telemetry=None) -> tuple[list[tuple[str, dict]], dict]:
    """([(conversation id, conversation)], {source id: address, revision,
    licenses}) for every chat file tagged `tag` in an admitted source, in
    source and path order. A conversation id is "<source>/<path>:<line>", or
    with `tree` "<source>/<path>:<id of its last message>". telemetry: as
    for source_tree."""
    conversations, sources = [], {}
    for source in repo.sources():
        if not source.admitted:
            continue
        tree, _ = source_tree(source, store, telemetry)
        for path in sorted(tree, key=str.encode):
            entry = markdown.format_of(source, path)
            if entry["format"] != "chat" or tag not in tags_of(source, path):
                continue
            found = of_file(source.id, path, entry, store, tree[path].sha256)
            conversations += [(f"{source.id}/{path}:{key}", c) for key, c in found]
            sources[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                                  "licenses": source.data["licenses"]}
    return conversations, sources
