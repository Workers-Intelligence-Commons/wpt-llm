"""The tools a model calls in its sandbox: the four every task offers
(read, write, edit and bash), ask when the task holds the user's replies,
and the tools a task or its image's pool declares, each answered by a
program in the image (docs/training.md, "Tools").

TOOLS lists the four as a conversation's `tools` (docs/chat-format.md).
An Offer is what one attempt is offered: offer() chooses it with a seeded
generator (one variant of each need, its tools under their own names and
the files it hides; each other tool of the task and of the pool left out
or kept, under its own name or another, with even odds; every tool in a
shuffled order), plain() offers every tool of the task under its own
name and hides nothing more, and needed() offers the tools of the same
variants offer() chooses with the same generator, under their own names,
and nothing else. arranged() gives an attempt's arrangement: the task's
files, those its offer hides moved among the tool_files (a folder outside
/work, such as a repository's remote, under its own path there), and the
image's folders the task shows empty (absent). call()
runs one call in a sandbox under the
name the model saw and returns its result text and whether it failed. A
relative path is read from /work, the sandbox's working folder.
ask gives the reply of the first of the task's replies whose pattern is
found in the question, or NOT_KNOWN. A program reads the call's
arguments as JSON on its standard input and runs in /work as
sandbox.TOOL, which reads the task's tool_files; its standard output is
the result, and an exit code other than 0 makes the call fail with its
standard error (or its output) as the result.
A call that cannot run (unknown tool, missing argument, an argument holding
a NUL character or text UTF-8 cannot encode, file not found, a file edit
cannot rewrite) fails with a message for the model, never with an
exception: only the sandbox's own failures (SandboxError, and SandboxFull
when the model filled its container) are raised.
The refusal covers every argument, write's content and edit's strings
included: a file holding a NUL byte, or other bytes that are not UTF-8
text, is written through bash, by a command that prints them from escapes
(printf '\\0').

A result longer than the call's limit keeps its first and last limit / 2
bytes, with a line between them saying how many bytes were cut. edit reads
the whole file, up to EDIT_BYTES, and rewrites only UTF-8 text.
"""

import json
import posixpath
import random
import re
from dataclasses import dataclass
from typing import NamedTuple

from wpt_dataset.tasks import ASK as ASK_NAME

from .sandbox import TOOL

TOOLS = [
    {"name": "read", "description": "Read a file and return its content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write", "description": "Write a file, replacing its content; creates its folders.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "edit", "description": "Replace old_string, which must occur exactly once in the file, with new_string.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"},
                                                     "new_string": {"type": "string"}},
                    "required": ["path", "old_string", "new_string"]}},
    {"name": "bash", "description": "Run a command with bash and return its output and exit code.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]
ASK = {"name": ASK_NAME, "description": "Ask the user a question and return their reply.",
       "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}
NOT_KNOWN = "I don't know that"  # ask's reply to a question no pattern matches
# the other names the built-in tools and ask may be offered under
OTHER_NAMES = {"read": ["read_file", "view"], "write": ["write_file", "create_file"],
               "edit": ["edit_file", "str_replace"], "bash": ["shell", "run_command"], "ask": ["ask_user", "question"]}
SHOWN = ("name", "description", "parameters")  # what the model sees of a tool's declaration
ARGUMENTS = {tool["name"]: tool["parameters"]["required"] for tool in [*TOOLS, ASK]}
WRITE = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
EDIT_BYTES = 1 << 20  # the largest file edit rewrites


@dataclass(frozen=True)
class Offer:
    tools: list[dict]  # what the model sees, in order: each tool's name, description and parameters
    names: dict[str, str]  # each name the model sees -> the real tool's name, in the same order
    real: dict[str, dict]  # each real tool's name -> its declaration
    hidden: tuple = ()  # the paths under files/, or the folders outside /work, the chosen variants hide


def available(task, pool=()) -> dict[str, dict]:
    """{name: declaration} of every tool the task can offer: the four, ask
    when it has replies, its own tools, then those of the image's pool
    whose name it does not take."""
    found = {tool["name"]: tool for tool in [*TOOLS, *([ASK] if task.replies else []), *task.tools]}
    return found | {tool["name"]: tool for tool in pool if tool["name"] not in found}


def plain(task) -> Offer:
    """Every tool of the task, under its own name, in order: no variation."""
    real = available(task)
    return _offer({name: name for name in real}, real)


def offer(task, pool, rng: random.Random) -> Offer:
    """One attempt's tools, drawn from rng: one variant of each of the
    task's needs, its tools under their own names and its tool_files
    hidden; each other tool of the task and of the pool, the tools of
    unchosen variants aside, left out or kept with even odds, and a kept
    one offered under one of its other names with even odds, when one is
    free; every tool in a shuffled order."""
    real = available(task, pool)
    variants = {name for need in task.needs for variant in need for name in variant.tools}
    chosen = _variants(task, rng)
    shown = {name: name for variant in chosen for name in variant.tools}
    for name in [name for name in real if name not in variants and rng.random() < 0.5]:
        free = [other for other in real[name].get("names", OTHER_NAMES.get(name, []))
                if other not in real and other not in shown]
        shown[rng.choice(free) if free and rng.random() < 0.5 else name] = name
    order = list(shown)
    rng.shuffle(order)
    return _offer({name: shown[name] for name in order}, real,
                  tuple(path for variant in chosen for path in variant.tool_files))


def needed(task, rng: random.Random) -> Offer:
    """The tools of the variants offer(task, pool, rng) chooses from a
    generator in the same state, one of each of the task's needs, under
    their own names, in the needs' order, with the same tool_files hidden:
    nothing renamed, shuffled or added, so that an attempt offered it
    differs from the varied one only in the tools it sees."""
    chosen = _variants(task, rng)
    names = {name: name for variant in chosen for name in variant.tools}
    # a need names only the task's own tools (wpt_dataset.tasks), never the pool's
    return _offer(names, available(task), tuple(path for variant in chosen for path in variant.tool_files))


def _variants(task, rng):
    """One variant of each of the task's needs, drawn from rng: the first
    draws offer() and needed() make."""
    return [rng.choice(need) for need in task.needs]


class Arranged(NamedTuple):
    """What one attempt's container holds, from its task and its offer."""
    files: dict  # {path under /work: (manifest mode, content)}
    tool_files: dict  # the same, for the files only the tools' programs read
    outside: dict  # {absolute folder outside /work: its files, as files}
    absent: tuple  # the image's folders the attempt sees empty (the task's absent)


def arranged(task, offer: Offer) -> Arranged:
    """The arrangement of an attempt offered `offer`: the task's files, the
    files under each path the offer hides moved from /work among the files
    only the tools' programs read, each folder outside /work it hides
    (task.outside) moved there too, under its own absolute path
    (/tmp/origin.git's files under tmp/origin.git/), and the image's
    folders the task shows empty."""
    if not offer.hidden:
        return Arranged(task.files, task.tool_files, task.outside, task.absent)
    moved = {path: entry for path, entry in task.files.items()
             if any(path == hidden or path.startswith(hidden + "/") for hidden in offer.hidden)}
    moved |= {posixpath.join(place.lstrip("/"), path): entry for place, files in task.outside.items()
              if place in offer.hidden for path, entry in files.items()}
    return Arranged({path: entry for path, entry in task.files.items() if path not in moved},
                    {**task.tool_files, **moved},
                    {place: files for place, files in task.outside.items() if place not in offer.hidden},
                    task.absent)


def _offer(names, real, hidden=()):
    """The Offer of names ({name seen: real name}), from real's declarations."""
    return Offer([{**{key: real[actual][key] for key in SHOWN}, "name": seen} for seen, actual in names.items()],
                 names, {actual: real[actual] for actual in names.values()}, hidden)


def call(sandbox, name: str, arguments: dict, seconds: float, limit: int, offer: Offer,
         replies=()) -> tuple[str, bool]:
    """(result text, failed) of one call of the tool the model saw as
    `name` in offer, run in the sandbox for at most
    `seconds`, its output kept within `limit` bytes; replies: the task's
    (pattern, reply) for ask."""
    if name not in offer.names:
        return f"unknown tool {name!r}; the tools are {', '.join(offer.names)}", True
    real = offer.real[offer.names[name]]
    if "program" in real:
        return _program(sandbox, name, real["program"], arguments, seconds, limit)
    needed = ARGUMENTS[real["name"]]
    wrong = [key for key in needed if not isinstance(arguments.get(key), str)]
    if wrong:
        return f"{name} needs the string arguments {', '.join(needed)}; missing or not a string: " \
               f"{', '.join(wrong)}", True
    for key in needed:
        if "\0" in arguments[key]:
            return f"{name}: {key} holds a NUL character, which no tool argument may hold", True
        try:
            arguments[key].encode("utf-8")
        except UnicodeEncodeError as error:
            return f"{name}: {key} is not valid text (character {error.start} cannot be encoded as UTF-8)", True
    if real["name"] == "ask":
        return next((reply for pattern, reply in replies if re.search(pattern, arguments["question"])), NOT_KNOWN), \
            False
    if real["name"] == "read":
        return _read(sandbox, arguments["path"], seconds, limit)
    if real["name"] == "write":
        return _write(sandbox, arguments["path"], arguments["content"], seconds, limit)
    if real["name"] == "edit":
        return _edit(sandbox, arguments["path"], arguments["old_string"], arguments["new_string"], seconds, limit)
    return _bash(sandbox, arguments["command"], seconds, limit)


def _program(sandbox, name, program, arguments, seconds, limit):
    # ASCII JSON: a lone surrogate the model wrote stays an escape the program may refuse
    ran = sandbox.run([program], seconds=seconds, limit=limit, stdin=json.dumps(arguments).encode(), user=TOOL)
    if ran.code:
        return _stopped(ran, seconds) or _text(ran.stderr).strip() or _text(ran.stdout).strip() \
            or f"{name} failed with exit code {ran.code}", True
    return _text(ran.stdout), False


def _cat(sandbox, path, seconds, limit):
    """(Ran, failure text or "") of reading a file."""
    ran = sandbox.run(["cat", "--", path], seconds=seconds, limit=limit)
    if ran.code:
        return ran, _stopped(ran, seconds) or _text(ran.stderr).strip() or f"cannot read {path}"
    return ran, ""


def _read(sandbox, path, seconds, limit):
    ran, failure = _cat(sandbox, path, seconds, limit)
    return (failure, True) if failure else (_text(ran.stdout), False)


def _write(sandbox, path, content, seconds, limit):
    data = content.encode()
    ran = sandbox.run(["sh", "-c", WRITE, "sh", path], seconds=seconds, limit=limit, stdin=data)
    if ran.code:
        return _stopped(ran, seconds) or _text(ran.stderr).strip() or f"cannot write {path}", True
    return f"wrote {len(data)} bytes to {path}", False


def _edit(sandbox, path, old, new, seconds, limit):
    ran, failure = _cat(sandbox, path, seconds, EDIT_BYTES)
    if failure:
        return failure, True
    if ran.cut:
        return f"{path} is larger than {EDIT_BYTES} bytes, more than edit rewrites", True
    try:
        text = ran.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        return f"{path} is not UTF-8 text (byte {error.start}), so edit cannot rewrite it", True
    found = text.count(old) if old else 0
    if found != 1:
        return f"old_string occurs {found} times in {path}; it must occur exactly once", True
    result, failed = _write(sandbox, path, text.replace(old, new), seconds, limit)
    return (result, True) if failed else (f"edited {path}", False)


def _bash(sandbox, command, seconds, limit):
    ran = sandbox.run(["bash", "-c", command], seconds=seconds, limit=limit, merged=True)
    output = _text(ran.stdout)
    ending = _stopped(ran, seconds) or (f"exit code {ran.code}" if ran.code else "")
    return (f"{output}\n[{ending}]" if output else f"[{ending}]") if ending else output, bool(ran.code)


def _stopped(ran, seconds):
    return f"stopped after {seconds} seconds" if ran.timed_out else ""


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")
