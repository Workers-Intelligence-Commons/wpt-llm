"""The agent loop: the model writes a turn, its tool calls run in the
sandbox, their results come back as the next turn, until the model stops,
a limit is reached, or the model's own processes or files fill its
container so that no command can start there (SandboxFull: docker's text
never reaches the model).

The tokens are kept exactly as the model sampled them, next to the
conversation (docs/chat-format.md) they read as: a learner needs the
tokens the model chose, which encoding the conversation again would not
always give back. Everything around the model's turns (the opening, each
tool turn, the start of each assistant turn) comes from the chat template.
"""

from dataclasses import dataclass

from wpt_tokenizer import END, chat

from .sandbox import SandboxFull
from .tools import TOOLS, call

# why an attempt ended
ANSWERED = "answered"  # a turn closed without a tool call
ENDED = "ended"  # the model wrote END, the end of a document
TURNS = "turns"  # the task's max_turns
TOKENS = "tokens"  # the task's max_tokens
CONTEXT = "context"  # the model's context is full
FULL = "full"  # the model's processes or files filled its container: no command can start there


@dataclass
class Attempt:
    conversation: dict  # every message, as docs/chat-format.md records them
    ids: list[int]  # every token, starting with END
    learn: list[bool]  # for each token, whether the model wrote it
    ended: str  # ANSWERED, ENDED, TURNS, TOKENS, CONTEXT or FULL


def opening(task) -> dict:
    """The conversation an attempt starts from: the tools and the task's
    instructions as the user's message."""
    return {"schema": 1, "tools": TOOLS,
            "messages": [{"role": "user", "content": [{"type": "text", "text": task.instructions}]}]}


def attempt(task, sandbox, tokenizer, generate, *, context: int, result_bytes: int) -> Attempt:
    """One attempt at the task in the sandbox. generate(ids, n) yields up to
    n next tokens after ids; context is the most tokens the attempt may
    hold; a tool result longer than result_bytes keeps its start and end
    (tools.py)."""
    conversation = opening(task)
    ids = chat.prompt(tokenizer, conversation)
    learn = [False] * len(ids)
    written = turns = 0
    while True:
        room = min(task.max_tokens - written, context - len(ids))
        if room <= 0:
            return Attempt(conversation, ids, learn, TOKENS if written >= task.max_tokens else CONTEXT)
        reply = []
        for token in generate(ids, room):
            reply.append(token)
            if token in (chat.IM_END, END):
                break
        ids, learn, written, turns = ids + reply, learn + [True] * len(reply), written + len(reply), turns + 1
        closed = bool(reply) and reply[-1] == chat.IM_END
        blocks = chat.reply(tokenizer, reply[:-1] if closed or reply[-1:] == [END] else reply)
        conversation["messages"].append({"role": "assistant", "content": blocks})
        if reply[-1:] == [END]:
            return Attempt(conversation, ids, learn, ENDED)
        if not closed:  # cut by a limit before the turn closed: its calls do not run
            return Attempt(conversation, ids, learn, TOKENS if written >= task.max_tokens else CONTEXT)
        calls = [block for block in blocks if block["type"] == "tool_call"]
        if not calls:
            return Attempt(conversation, ids, learn, ANSWERED)
        results = []
        for block in calls:
            try:
                text, failed = call(sandbox, block["name"], block["arguments"], task.command_seconds, result_bytes)
            except SandboxFull:  # the turn's results never reach the model
                return Attempt(conversation, ids, learn, FULL)
            results.append({"type": "tool_result", "call": block["id"], "content": [{"type": "text", "text": text}],
                            **({"error": True} if failed else {})})
        message = {"role": "tool", "content": results}
        conversation["messages"].append(message)
        if turns >= task.max_turns:
            return Attempt(conversation, ids, learn, TURNS)
        following = _after_turn(tokenizer, message)
        if len(ids) + len(following) >= context:  # no room left for the model's next token
            return Attempt(conversation, ids, learn, CONTEXT)
        ids, learn = ids + following, learn + [False] * len(following)


def _after_turn(tokenizer, message) -> list[int]:
    """The template's tokens after an assistant turn's <|im_end|>: the line
    break that ends the turn, the tool turn, and the start of the next
    assistant turn."""
    closed = chat.encode(tokenizer, {"messages": [{"role": "assistant", "content": []}]})[0]
    line_break = closed[closed.index(chat.IM_END) + 1:]
    return line_break + chat.prompt(tokenizer, {"messages": [message]})[1:]  # [1:]: without the opening END
