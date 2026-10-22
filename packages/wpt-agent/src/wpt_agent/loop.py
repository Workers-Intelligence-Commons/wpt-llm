"""The agent loop: the model writes a turn, its tool calls run in the
sandbox, their results come back as the next turn, until the model stops,
a limit is reached, or the model's own processes or files fill its
container so that no command can start there (SandboxFull: docker's text
never reaches the model).

Several attempts at one task run together, one per sandbox (attempts):
the model reads the start their openings share once for all of them (the
whole opening when every attempt is offered the same tools), then each
row the rest of its own, and writes their turns in one batch, each
attempt in its own row (wpt_inference.Writer); while one attempt's tool
calls run, the others wait.

Each attempt is offered its tools (tools.Offer: by default every tool of
the task under its own name), and its conversation's metadata keeps
`tools`, the name each offered tool had for the model mapped to the real
tool's, `looking`, the real names of the offered tools that look at
sources (wpt_dataset.tasks.looks; wpt_check's looking rule reads it),
`answered_by`, the task's looking tools that can answer what it leaves
open (wpt_check.asked_last reads it), and `answer`, the value of the attempt's answer block: the last
json block of its last assistant message when it is an object of exactly
`answer`, a string, and `confidence`, a number from 0 to 1
(docs/chat-format.md, "The answer"); otherwise null. A turn without a
tool call still ends the attempt: a question to the user goes through
the ask tool.

The tokens are kept exactly as the model sampled them, next to the
conversation (docs/chat-format.md) they read as: a learner needs the
tokens the model chose, which encoding the conversation again would not
always give back. Everything around the model's turns (the opening, each
tool turn, the start of each assistant turn) comes from the chat template.
"""

import os
from dataclasses import dataclass

from wpt_dataset.repo import CURRENT_SCHEMA
from wpt_dataset.tasks import looks
from wpt_tokenizer import END, chat

from .sandbox import SandboxError, SandboxFull
from .tools import call, plain

# why an attempt ended
ANSWERED = "answered"  # a turn closed without a tool call
ENDED = "ended"  # the model wrote END, the end of a document
TURNS = "turns"  # the task's max_turns
TOKENS = "tokens"  # the task's max_tokens
CONTEXT = "context"  # the context the caller allows is full
FULL = "full"  # the model's processes or files filled its container: no command can start there


@dataclass
class Attempt:
    conversation: dict  # every message, as docs/chat-format.md records them
    ids: list[int]  # every token, starting with END
    learn: list[bool]  # for each token, whether the model wrote it
    ended: str  # ANSWERED, ENDED, TURNS, TOKENS, CONTEXT or FULL


def opening(task, offer=None) -> dict:
    """The conversation an attempt starts from: the tools offered (by
    default plain(task)), the task's instructions as the user's message,
    and in its metadata the offered tools' real names, those of them that
    look, and the task's answered_by."""
    offer = offer or plain(task)
    return {"schema": CURRENT_SCHEMA["conversation"], "tools": offer.tools,
            "messages": [{"role": "user", "content": [{"type": "text", "text": task.instructions}]}],
            "metadata": {"tools": dict(offer.names),
                         "looking": [name for name, tool in offer.real.items() if looks(tool)],
                         "answered_by": list(task.answered_by)}}


def attempts(task, sandboxes, tokenizer, start, *, result_bytes: int, context: int | None = None,
             offers: list | None = None) -> list:
    """One attempt at the task in each sandbox, written together, each
    offered its tools (offers, one per sandbox; by default plain(task)).
    start(ids, rows, stops) gives the writer of `rows` rows that all start
    from the tokens ids, whose write(inputs, limits) reads each row's
    inputs and writes up to its limit, ending after a stop token
    (wpt_inference.Writer). context, when given, is the most tokens an
    attempt may hold; without it an attempt ends only at the task's limits.
    A tool result longer than result_bytes keeps its start and end
    (tools.py). Returns, for each sandbox, its Attempt, or the SandboxError
    that ended it (Docker failed, which is not the model's doing)."""
    offers = offers or [plain(task)] * len(sandboxes)
    prompts = [chat.prompt(tokenizer, opening(task, offer)) for offer in offers]
    shared = len(os.path.commonprefix(prompts))  # the tokens every opening starts with
    writer = start(prompts[0][:shared], len(sandboxes), (chat.IM_END, END))
    runs = {row: _attempt(task, sandbox, tokenizer, prompts[row], shared, offers[row], result_bytes, context)
            for row, sandbox in enumerate(sandboxes)}
    done, asks = {}, {}

    def advance(row, written):  # runs the row until it asks for tokens again, or ends
        try:
            asks[row] = runs[row].send(written)
        except StopIteration as stopped:
            done[row] = stopped.value
        except SandboxError as error:
            done[row] = error

    for row in runs:
        advance(row, None)
    while asks:
        inputs = {row: following for row, (following, _) in asks.items()}
        written = writer.write(inputs, {row: limit for row, (_, limit) in asks.items()})
        asks.clear()
        for row, tokens in written.items():
            advance(row, tokens)
    return [done[row] for row in range(len(sandboxes))]


def attempt(task, sandbox, tokenizer, start, *, result_bytes: int, context: int | None = None,
            offer=None) -> Attempt:
    """One attempt at the task in the sandbox (attempts, for one); raises
    SandboxError when Docker failed."""
    [done] = attempts(task, [sandbox], tokenizer, start, result_bytes=result_bytes, context=context,
                      offers=[offer] if offer else None)
    if isinstance(done, SandboxError):
        raise done
    return done


def _attempt(task, sandbox, tokenizer, prompt, shared, offer, result_bytes, context):
    """One attempt, as a generator: it yields (tokens the model reads next,
    the most tokens it may write now) and is sent what the model wrote;
    it returns the Attempt. The writer has read prompt[:shared]."""
    conversation = opening(task, offer)
    ids = list(prompt)
    learn = [False] * len(ids)
    written = turns = 0
    following = prompt[shared:]  # tokens the model has not read yet, besides those it wrote
    while True:
        room = task.max_tokens - written if context is None else min(task.max_tokens - written, context - len(ids))
        if room <= 0:
            return _ended(conversation, ids, learn, TOKENS if written >= task.max_tokens else CONTEXT)
        reply = []
        while not reply or (reply[-1] not in (chat.IM_END, END) and len(reply) < room):
            reply += yield following, room - len(reply)
            following = []
        ids, learn, written, turns = ids + reply, learn + [True] * len(reply), written + len(reply), turns + 1
        closed = reply[-1] == chat.IM_END
        blocks = chat.reply(tokenizer, reply[:-1] if closed or reply[-1] == END else reply)
        conversation["messages"].append({"role": "assistant", "content": blocks})
        if reply[-1] == END:
            return _ended(conversation, ids, learn, ENDED)
        if not closed:  # cut by a limit before the turn closed: its calls do not run
            return _ended(conversation, ids, learn, TOKENS if written >= task.max_tokens else CONTEXT)
        calls = [block for block in blocks if block["type"] == "tool_call"]
        if not calls:
            return _ended(conversation, ids, learn, ANSWERED)
        results = []
        for block in calls:
            try:
                text, failed = call(sandbox, block["name"], block["arguments"], task.command_seconds, result_bytes,
                                    offer, task.replies)
            except SandboxFull:  # the turn's results never reach the model
                return _ended(conversation, ids, learn, FULL)
            results.append({"type": "tool_result", "call": block["id"], "content": [{"type": "text", "text": text}],
                            **({"error": True} if failed else {})})
        message = {"role": "tool", "content": results}
        conversation["messages"].append(message)
        if turns >= task.max_turns:
            return _ended(conversation, ids, learn, TURNS)
        following = _after_turn(tokenizer, message)
        if context is not None and len(ids) + len(following) >= context:  # no room left for the model's next token
            return _ended(conversation, ids, learn, CONTEXT)
        ids, learn = ids + following, learn + [False] * len(following)


def _ended(conversation, ids, learn, ended) -> Attempt:
    """The Attempt, its answer block read and kept in the conversation's metadata."""
    last = next((m for m in reversed(conversation["messages"]) if m["role"] == "assistant"), {"content": []})
    value = next((block["value"] for block in reversed(last["content"]) if block["type"] == "json"), None)
    if not (isinstance(value, dict) and set(value) == {"answer", "confidence"} and isinstance(value["answer"], str)
            and type(value["confidence"]) in (int, float) and 0 <= value["confidence"] <= 1):
        value = None
    conversation["metadata"]["answer"] = value
    return Attempt(conversation, ids, learn, ended)


def _after_turn(tokenizer, message) -> list[int]:
    """The template's tokens after an assistant turn's <|im_end|>: the line
    break that ends the turn, the tool turn, and the start of the next
    assistant turn."""
    closed = chat.encode(tokenizer, {"messages": [{"role": "assistant", "content": []}]})[0]
    line_break = closed[closed.index(chat.IM_END) + 1:]
    return line_break + chat.prompt(tokenizer, {"messages": [message]})[1:]  # [1:]: without the opening END
