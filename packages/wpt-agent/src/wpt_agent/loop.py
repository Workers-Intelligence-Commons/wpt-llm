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

An attempt opens with the task's full instructions, or with its one-line
form (terse; the measure of terse against full requests) when the caller
asks for it.

Each attempt is offered its tools (tools.Offer: by default every tool of
the task under its own name), and its conversation's metadata keeps
`tools`, the name each offered tool had for the model mapped to the real
tool's, `looking`, the real names of the offered tools that look at
sources (wpt_dataset.tasks.looks; wpt_check's looking rule reads it),
`answered_by`, the task's looking tools that can answer what it leaves
open (wpt_check.asked_last reads it), and `answer` (docs/chat-format.md,
"The answer"). A turn without a tool call is the model's final reply,
its answer: a question to the user goes through the ask tool. The loop
then adds one user turn, CONFIDENCE, and the model's reply to it, a
number from 0 to 100 and nothing else (confidence()), divided by 100,
gives `answer` its value: {"answer": the final reply's text,
"confidence": that number}. The model writes text only, never the JSON
of that value. `answer` is null when the reply to CONFIDENCE is not such
a number, and when the attempt ended with no final reply (a limit, or
END in a turn with a tool call), so the check library scores it 0. The
confidence turn counts against max_tokens and the context, not
max_turns.

The tokens are kept exactly as the model sampled them, next to the
conversation (docs/chat-format.md) they read as: a learner needs the
tokens the model chose, which encoding the conversation again would not
always give back. Everything around the model's turns (the opening, each
tool turn, the start of each assistant turn) comes from the chat template.
"""

import os
import re
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

# the one question every attempt, and every measure that asks for a confidence, puts after the final reply
CONFIDENCE = ("How sure are you that you did what was asked, and did it right? Reply with a number from 0 to 100 "
              "and nothing else.")
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")  # a number as the reply to CONFIDENCE writes it


@dataclass
class Attempt:
    conversation: dict  # every message, as docs/chat-format.md records them
    ids: list[int]  # every token, starting with END
    learn: list[bool]  # for each token, whether the model wrote it
    ended: str  # ANSWERED, ENDED, TURNS, TOKENS, CONTEXT or FULL


def opening(task, offer=None, terse=False) -> dict:
    """The conversation an attempt starts from: the tools offered (by
    default plain(task)), the task's instructions as the user's message
    (with terse, its one-line form; ValueError for a task without one),
    and in its metadata the offered tools' real names, those of them that
    look, and the task's answered_by."""
    if terse and not task.terse:
        raise ValueError(f"{task.id}: the task has no one-line form (terse)")
    offer = offer or plain(task)
    request = task.terse if terse else task.instructions
    return {"schema": CURRENT_SCHEMA["conversation"], "tools": offer.tools,
            "messages": [{"role": "user", "content": [{"type": "text", "text": request}]}],
            "metadata": {"tools": dict(offer.names),
                         "looking": [name for name, tool in offer.real.items() if looks(tool)],
                         "answered_by": list(task.answered_by)}}


def attempts(task, sandboxes, tokenizer, start, *, result_bytes: int, context: int | None = None,
             offers: list | None = None, terse: bool = False) -> list:
    """One attempt at the task in each sandbox, written together, each
    offered its tools (offers, one per sandbox; by default plain(task)),
    each opening with the task's instructions, or with terse its one-line
    form (opening).
    start(ids, rows, stops) gives the writer of `rows` rows that all start
    from the tokens ids, whose write(inputs, limits) reads each row's
    inputs and writes up to its limit, ending after a stop token
    (wpt_inference.Writer). context, when given, is the most tokens an
    attempt may hold; without it an attempt ends only at the task's limits.
    A tool result longer than result_bytes keeps its start and end
    (tools.py). Returns, for each sandbox, its Attempt, or the SandboxError
    that ended it (Docker failed, which is not the model's doing)."""
    offers = offers or [plain(task)] * len(sandboxes)
    prompts = [chat.prompt(tokenizer, opening(task, offer, terse)) for offer in offers]
    shared = len(os.path.commonprefix(prompts))  # the tokens every opening starts with
    writer = start(prompts[0][:shared], len(sandboxes), (chat.IM_END, END))
    runs = {row: _attempt(task, sandbox, tokenizer, prompts[row], shared, offers[row], result_bytes, context, terse)
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
            offer=None, terse: bool = False) -> Attempt:
    """One attempt at the task in the sandbox (attempts, for one); raises
    SandboxError when Docker failed."""
    [done] = attempts(task, [sandbox], tokenizer, start, result_bytes=result_bytes, context=context,
                      offers=[offer] if offer else None, terse=terse)
    if isinstance(done, SandboxError):
        raise done
    return done


def _attempt(task, sandbox, tokenizer, prompt, shared, offer, result_bytes, context, terse):
    """One attempt, as a generator: it yields (tokens the model reads next,
    the most tokens it may write now) and is sent what the model wrote;
    it returns the Attempt. The writer has read prompt[:shared]."""
    conversation = opening(task, offer, terse)
    ids = list(prompt)
    learn = [False] * len(ids)
    written = turns = 0
    final = None  # (the final reply's text, why the attempt ended) once CONFIDENCE is asked
    following = prompt[shared:]  # tokens the model has not read yet, besides those it wrote
    while True:
        room = task.max_tokens - written if context is None else min(task.max_tokens - written, context - len(ids))
        if room <= 0:
            return _ended(conversation, ids, learn, TOKENS if written >= task.max_tokens else CONTEXT)
        reply = []
        while not reply or (reply[-1] not in (chat.IM_END, END) and len(reply) < room):
            reply += yield following, room - len(reply)
            following = []
        ids, learn, written = ids + reply, learn + [True] * len(reply), written + len(reply)
        turns += final is None  # the reply to CONFIDENCE is not one of the task's turns
        closed, finished = reply[-1] == chat.IM_END, reply[-1] == END
        blocks = chat.reply(tokenizer, reply[:-1] if closed or finished else reply)
        conversation["messages"].append({"role": "assistant", "content": blocks})
        if not (closed or finished):  # cut by a limit before the turn closed: its calls do not run
            return _ended(conversation, ids, learn, TOKENS if written >= task.max_tokens else CONTEXT)
        if final is not None:
            sure = confidence(blocks)
            return _ended(conversation, ids, learn, final[1],
                          None if sure is None else {"answer": final[0], "confidence": sure})
        calls = [block for block in blocks if block["type"] == "tool_call"]
        if not calls:  # the final reply: its text is the answer, and the confidence comes next
            final = (text(blocks), ENDED if finished else ANSWERED)
            message = {"role": "user", "content": [{"type": "text", "text": CONFIDENCE}]}
        elif finished:  # END in a turn with calls: they do not run, and there is no final reply
            return _ended(conversation, ids, learn, ENDED)
        else:
            results = []
            for block in calls:
                try:
                    found, failed = call(sandbox, block["name"], block["arguments"], task.command_seconds,
                                         result_bytes, offer, task.replies)
                except SandboxFull:  # the turn's results never reach the model
                    return _ended(conversation, ids, learn, FULL)
                results.append({"type": "tool_result", "call": block["id"],
                                "content": [{"type": "text", "text": found}], **({"error": True} if failed else {})})
            message = {"role": "tool", "content": results}
        conversation["messages"].append(message)
        if final is None and turns >= task.max_turns:
            return _ended(conversation, ids, learn, TURNS)
        following = _after_turn(tokenizer, message)
        if context is not None and len(ids) + len(following) >= context:  # no room left for the model's next token
            return _ended(conversation, ids, learn, CONTEXT)
        ids, learn = ids + following, learn + [False] * len(following)


def _ended(conversation, ids, learn, ended, answer=None) -> Attempt:
    """The Attempt, its answer (None without one) kept in the conversation's metadata."""
    conversation["metadata"]["answer"] = answer
    return Attempt(conversation, ids, learn, ended)


def text(blocks) -> str:
    """The text of a reply: its text blocks, one per line, its thinking left out."""
    return "\n".join(block["text"] for block in blocks if block["type"] == "text")


def confidence(blocks) -> float | None:
    """The reply to CONFIDENCE as a number from 0 to 1: its text, spaces
    around it aside, a number from 0 to 100 (NUMBER) and nothing else,
    divided by 100; None for any other reply, one with a tool call
    included. Thinking before it is not part of its text."""
    if any(block["type"] not in ("text", "thinking") for block in blocks):
        return None
    written = text(blocks).strip()
    if not NUMBER.fullmatch(written) or float(written) > 100:
        return None
    return float(written) / 100


def _after_turn(tokenizer, message) -> list[int]:
    """The template's tokens after an assistant turn's <|im_end|>: the line
    break that ends the turn, the tool turn, and the start of the next
    assistant turn."""
    closed = chat.encode(tokenizer, {"messages": [{"role": "assistant", "content": []}]})[0]
    line_break = closed[closed.index(chat.IM_END) + 1:]
    return line_break + chat.prompt(tokenizer, {"messages": [message]})[1:]  # [1:]: without the opening END
