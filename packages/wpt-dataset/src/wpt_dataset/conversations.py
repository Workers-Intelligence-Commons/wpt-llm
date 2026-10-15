"""Conversations for post-training (docs/chat-format.md).

A file in the `chat` format is JSON Lines: one conversation per line, each
matching schemas/conversation.schema.json, and each tool_result answering a
tool_call made earlier in the same conversation. Such a file is never a
pre-training document (markdown.py gives it no Markdown); a post-training
step reads it through read(), which picks the files tagged with the step's
name.
"""

import json

from . import markdown
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


def read(repo: Repo, store, tag: str) -> tuple[list[tuple[str, dict]], dict]:
    """([(conversation id, conversation)], {source id: address, revision,
    licenses}) for every chat file tagged `tag` in an admitted source, in
    source and path order. A conversation id is "<source>/<path>:<line>"."""
    conversations, sources = [], {}
    for source in repo.sources():
        if not source.admitted:
            continue
        tree, _ = source_tree(source, store)
        for path in sorted(tree, key=str.encode):
            if markdown.format_of(source, path)["format"] != "chat" or tag not in tags_of(source, path):
                continue
            found = parse(source.id, path, store.get_bytes(tree[path].sha256))
            conversations += [(f"{source.id}/{path}:{n}", c) for n, c in enumerate(found, start=1)]
            sources[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                                  "licenses": source.data["licenses"]}
    return conversations, sources
