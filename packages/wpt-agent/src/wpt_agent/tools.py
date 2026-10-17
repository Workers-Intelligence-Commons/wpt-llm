"""The four tools a model calls in its sandbox: read, write, edit and bash.

TOOLS lists them as a conversation's `tools` (docs/chat-format.md); call()
runs one call in a sandbox and returns its result text and whether it
failed. A relative path is read from /work, the sandbox's working folder.
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
ARGUMENTS = {tool["name"]: tool["parameters"]["required"] for tool in TOOLS}
WRITE = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
EDIT_BYTES = 1 << 20  # the largest file edit rewrites


def call(sandbox, name: str, arguments: dict, seconds: float, limit: int) -> tuple[str, bool]:
    """(result text, failed) of one tool call, run in the sandbox for at
    most `seconds`, its output kept within `limit` bytes."""
    if name not in ARGUMENTS:
        return f"unknown tool {name!r}; the tools are {', '.join(ARGUMENTS)}", True
    wrong = [key for key in ARGUMENTS[name] if not isinstance(arguments.get(key), str)]
    if wrong:
        return f"{name} needs the string arguments {', '.join(ARGUMENTS[name])}; missing or not a string: " \
               f"{', '.join(wrong)}", True
    for key in ARGUMENTS[name]:
        if "\0" in arguments[key]:
            return f"{name}: {key} holds a NUL character, which no tool argument may hold", True
        try:
            arguments[key].encode("utf-8")
        except UnicodeEncodeError as error:
            return f"{name}: {key} is not valid text (character {error.start} cannot be encoded as UTF-8)", True
    if name == "read":
        return _read(sandbox, arguments["path"], seconds, limit)
    if name == "write":
        return _write(sandbox, arguments["path"], arguments["content"], seconds, limit)
    if name == "edit":
        return _edit(sandbox, arguments["path"], arguments["old_string"], arguments["new_string"], seconds, limit)
    return _bash(sandbox, arguments["command"], seconds, limit)


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
        return f"{path} is larger than {EDIT_BYTES} bytes, more than edit rewrites; change it with bash", True
    try:
        text = ran.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        return f"{path} is not UTF-8 text (byte {error.start}), so edit cannot rewrite it; change it with bash", True
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
