"""The chat template: a conversation (docs/chat-format.md) to token ids and back.

The layout is ChatML with the tags of Qwen and Hermes, each tag a special
token (wpt_tokenizer.SPECIAL), so no text inside a message can imitate one:

    END
    <|im_start|>system\\n{text}\\n\\n<tools>\\n{one tool as JSON per line}\\n</tools><|im_end|>\\n
    <|im_start|>user\\n{text}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n{thinking}\\n</think>\\n\\n{text}\\n<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call><|im_end|>\\n
    <|im_start|>user\\n<tool_response>\\n{result}\\n</tool_response><|im_end|>\\n
    <|im_start|>assistant\\n{text}\\n<json>\\n{value}\\n</json><|im_end|>\\n

A message's blocks keep their order; blocks after the first are separated by
a line break, and a thinking block ends with a blank line. Tools offered in
a conversation without a system message get a system message of their own.
A `tool` message is written as a user turn of <tool_response> blocks, as
Qwen does. A `json` block is written as its JSON text between <json> and
</json>, so a structured output is told apart from text, also inside a
tool result. Left out, while recorded:
`file` blocks (no model reads media yet) and a tool_result's `error` flag.
"""

import json

from . import END, SPECIAL

IM_START, IM_END = SPECIAL["<|im_start|>"], SPECIAL["<|im_end|>"]
THINK, THINK_END = SPECIAL["<think>"], SPECIAL["</think>"]
TOOLS, TOOLS_END = SPECIAL["<tools>"], SPECIAL["</tools>"]
CALL, CALL_END = SPECIAL["<tool_call>"], SPECIAL["</tool_call>"]
RESPONSE, RESPONSE_END = SPECIAL["<tool_response>"], SPECIAL["</tool_response>"]
JSON, JSON_END = SPECIAL["<json>"], SPECIAL["</json>"]
NAMES = {i: name for name, i in SPECIAL.items()}


def encode(tokenizer, conversation) -> tuple[list[int], list[bool]]:
    """(ids, learn): the conversation's tokens, starting with END, and for
    each token whether a model learns to predict it, which holds for every
    assistant message's content and its closing <|im_end|>."""
    ids, learn = [END], [False]
    for role, parts in _turns(tokenizer, conversation):
        body = [i for part in parts for i in part]
        ids += [IM_START, *tokenizer.encode(f"{role}\n".encode())]
        learn += [False] * (len(ids) - len(learn))
        ids += [*body, IM_END]
        learn += [role == "assistant"] * (len(body) + 1)
        newline = tokenizer.encode(b"\n")
        ids += newline
        learn += [False] * len(newline)
    return ids, learn


def prompt(tokenizer, conversation) -> list[int]:
    """The conversation's tokens followed by the start of an assistant turn:
    what a model continues to answer."""
    return encode(tokenizer, conversation)[0] + [IM_START, *tokenizer.encode(b"assistant\n")]


def show(tokenizer, ids) -> str:
    """The text of the tokens with every special token written as its tag."""
    out, pending = [], []
    for i in ids:
        if i in NAMES or i == END:
            out.append(tokenizer.decode(pending).decode("utf-8", errors="replace"))
            out.append(NAMES.get(i, ""))
            pending = []
        else:
            pending.append(i)
    out.append(tokenizer.decode(pending).decode("utf-8", errors="replace"))
    return "".join(out)


def reply(tokenizer, ids) -> list[dict]:
    """The blocks of an assistant message from its generated tokens (up to,
    not including, <|im_end|>). A tool call whose text is not a JSON object
    with a name and arguments, and a json block whose text is not JSON,
    stay text, tags included."""
    blocks, text, inside, buffer, calls = [], [], None, [], 0

    def flush_text():
        words = _words(tokenizer, text)
        if words:
            blocks.append({"type": "text", "text": words})
        text.clear()

    for i in ids:
        if inside is None and i in (THINK, CALL, JSON):
            flush_text()
            inside, buffer = i, []
        elif inside == THINK and i == THINK_END:
            blocks.append({"type": "thinking", "text": _words(tokenizer, buffer)})
            inside = None
        elif inside == CALL and i == CALL_END:
            call = _call(_words(tokenizer, buffer))
            if call is None:
                text += [CALL, *buffer, CALL_END]
            else:
                calls += 1
                blocks.append({"type": "tool_call", "id": f"call-{calls}", **call})
            inside = None
        elif inside == JSON and i == JSON_END:
            try:
                blocks.append({"type": "json", "value": _json(_words(tokenizer, buffer))})
            except ValueError:  # not JSON, or NaN or Infinity, which no JSON file holds
                text += [JSON, *buffer, JSON_END]
            inside = None
        elif inside is not None:
            buffer.append(i)
        else:
            text.append(i)
    if inside is not None:  # an unclosed tag: keep what was written
        text += [inside, *buffer]
    flush_text()
    return blocks


def _words(tokenizer, ids) -> str:
    return show(tokenizer, ids).strip("\n")


def _json(text):
    def refuse(constant):
        raise ValueError(constant)
    return json.loads(text, parse_constant=refuse)


def _call(text):
    try:
        call = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not (isinstance(call, dict) and set(call) == {"name", "arguments"} and isinstance(call["name"], str)
            and isinstance(call["arguments"], dict)):
        return None
    return {"name": call["name"], "arguments": call["arguments"]}


def _turns(tokenizer, conversation):
    """(role written in the header, [token runs of the body]) per turn."""
    def text(words):
        return tokenizer.encode(words.encode())

    def wrapped(open_, close, words):
        return [open_, *text(f"\n{words}\n"), close]

    def body(blocks):
        parts, after_thinking = [], False
        for block in blocks:
            kind = block["type"]
            if kind == "file":
                continue
            if parts and not after_thinking:
                parts.append(text("\n"))
            after_thinking = kind == "thinking"
            if kind == "text":
                parts.append(text(block["text"]))
            elif kind == "json":
                parts.append(wrapped(JSON, JSON_END, json.dumps(block["value"], ensure_ascii=False)))
            elif kind == "thinking":
                parts.append([*wrapped(THINK, THINK_END, block["text"]), *text("\n\n")])
            elif kind == "tool_call":
                call = {"name": block["name"], "arguments": block["arguments"]}
                parts.append(wrapped(CALL, CALL_END, json.dumps(call, ensure_ascii=False)))
            elif kind == "tool_result":
                parts.append([RESPONSE, *result(block["content"]), RESPONSE_END])
        return parts

    def result(blocks):  # "\n{each block, one per line}\n", the text between json blocks encoded at once
        ids, words = [], ["\n"]
        for n, block in enumerate(b for b in blocks if b["type"] != "file"):
            words.append("\n" if n else "")
            if block["type"] == "json":
                ids += [*text("".join(words)), *wrapped(JSON, JSON_END, json.dumps(block["value"], ensure_ascii=False))]
                words = []
            else:
                words.append(block["text"])
        return ids + text("".join(words) + "\n")

    tools = conversation.get("tools") or []
    listed = [TOOLS, *text("\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + "\n"), TOOLS_END]
    messages = conversation["messages"]
    if tools and messages[0]["role"] != "system":
        yield "system", [listed]
    for n, message in enumerate(messages):
        parts = body(message["content"])
        if n == 0 and tools and message["role"] == "system":
            parts += [text("\n\n"), listed] if parts else [listed]
        yield ("user" if message["role"] == "tool" else message["role"]), parts
