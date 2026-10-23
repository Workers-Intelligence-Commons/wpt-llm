"""The rules many tasks' checks share (docs/post-training.md, "The check
library"; maintainer's decision, 2026-10-18), tested once in the
repository (packages/wpt-agent/tests/test_agent_image.py). A check calls
them on the attempt's conversation, conversation.json next to it, as
Python started with -P or -I:

    #!/usr/bin/python3 -I
    import wpt_check
    conversation = wpt_check.load()
    right = ...  # the task's own judgement of the files and the answer
    print(wpt_check.report(conversation, right, sources_first=wpt_check.sources_first(conversation)))

    load()                       the attempt's conversation, next to the check
    calls(c)                     every tool call, in order, by its real tool's name
    answer(c)                    the answer: {"answer": ..., "confidence": ...}, or None without one
    answer_text(c)               the answer's text, "" without one
    scored(c, score)             score, or 0 without an answer
    report(c, correct, **rules)  what a check prints: the JSON object of its parts, "correct" (scored) and
                                 each rule's, as numbers
    brier(c, correct)            (confidence - correct) squared, 1 without an answer
    before(c, first, then)       whether every call to a tool of `then` comes after a call to one of `first`
    read_before_write(c, new)    whether every file written or edited was read first
    sources_first(c)             whether the sources were looked at before anything was changed or asked
    asked_last(c, found)         whether every question to the user came after every file where the answer
                                 lives (the task's answered_in) was opened by a looking tool that can answer
                                 (its answered_by), and before any look showed found; true at once when no
                                 such file exists
    opened(call, path)           whether a call showed the file at path: its path argument, or a bash command
                                 that names it and runs a program that shows content (SHOWING)
    read_before_ask(c, text)     whether every question to the user came after a look whose result showed text
    programs(command)            the programs a bash command runs
    path_of(path)                a tool's path argument as an absolute path (a relative one starts at /work)
    mentions(command, path)      whether a shell command names the file at path
    beside(name)                 the text of a file next to the check (one a task's pointer put there)
    file_text(path)              the text of a file the attempt left in /work, "" when there is none
    holds(text, expected)        whether text holds expected's words, in order, case and marks aside
    not_enough(c, missing, n)    whether the answer says the files do not hold enough ("not enough: ..."), and
                                 names what is missing by the first n words of missing
    pushed(expected)             whether the remote got exactly one new commit, the expected tree, any message

The rules read the tools by their real names (read, write, edit, bash,
ask, a declared tool's name): an attempt sees them under varied names, and
the conversation's metadata.tools maps each name it saw to the real one;
metadata.looking lists the offered tools that look at sources (read, bash,
and each declared tool marked `looks`; wpt_dataset.tasks.looks), and
metadata.answered_by is the task's answered_by, and metadata.answered_in the
files under /work the attempt started with that the task's answered_in
matches.

Looking is one rule, which sources_first, asked_last and read_before_ask
share (maintainer's decision, 2026-10-22): a call looks (Call.looks) when
it calls a looking tool other than bash, or bash with a command that runs
one of the image's reading programs (READERS), found by parsing the
command (programs): every simple command of its pipelines and lists (|,
;, &, &&, ||, line breaks, parentheses), its first word after any
VAR=value, by its name or its path, comments left out.
"""

import json
import os
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import NamedTuple

WORK = "/work"
CONVERSATION = "conversation.json"
WRITES = ("write", "edit")  # the tools that change a file by its path
# the image's programs that show what files or folders hold: a bash command running one of them looks
READERS = frozenset({"cat", "head", "tail", "more", "tac", "nl", "od", "wc", "cut", "sort", "uniq", "paste", "fold",
                     "fmt", "pr", "rev", "expand", "unexpand", "shuf", "comm", "join", "csplit", "base64", "base32",
                     "basenc", "iconv", "grep", "egrep", "fgrep", "rgrep", "zgrep", "zegrep", "zfgrep", "zcat",
                     "zless", "zmore", "zdiff", "sed", "awk", "mawk", "find", "ls", "dir", "vdir", "stat", "du",
                     "diff", "diff3", "sdiff", "cmp"})
SEPARATING = frozenset("|&;()\n")  # a word made of these alone ends a simple command: |, ||, &&, ;, ( ), a line break
# the READERS that show what a file holds, not only its name, size or counts: a bash command opens a file with them
SHOWING = READERS - {"wc", "csplit", "find", "ls", "dir", "vdir", "stat", "du", "cmp"}
PUNCTUATION = "();<>|&\n"  # what bash splits words at, outside quotes
# bash's reserved words that may come before a simple command's program
RESERVED = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}"})
# the programs that run a program named after their options: {name: (their options that take the next word
# as their value, the words they take before the program)}
RUNNERS = {"time": ((), 0), "exec": (("-a",), 0), "command": ((), 0), "builtin": ((), 0), "nohup": ((), 0),
           "env": (("-u", "--unset", "-C", "--chdir", "-S", "--split-string"), 0),
           "nice": (("-n", "--adjustment"), 0), "stdbuf": (("-i", "-o", "-e", "--input", "--output", "--error"), 0),
           "timeout": (("-k", "--kill-after", "-s", "--signal"), 1),
           "xargs": (("-a", "--arg-file", "-d", "--delimiter", "-E", "-I", "-L", "--max-lines", "-n", "--max-args",
                      "-P", "--max-procs", "-s", "--max-chars", "--process-slot-var"), 0)}
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*", re.S)  # VAR=value before a program
CHANGES = ("write", "edit", "ask")  # what sources_first wants the sources looked at before
REPOSITORY = "repository.json"  # next to the check: what a repository task's pushed commit must be
NOT_ENOUGH = "not enough"  # how an answer starts that says the files do not hold what was asked
WORD = re.compile(r"\w+")


class Call(NamedTuple):
    tool: str | None  # the real tool's name; None for a name the attempt was not offered
    name: str  # the name the model called it by
    arguments: dict
    result: str  # the result's text, "" when the call got none
    failed: bool
    looks: bool  # whether it looks at sources: a looking tool, bash only when its command runs a reader


def load(path: str | None = None) -> dict:
    """The attempt's conversation: path, by default conversation.json in
    the folder of the running check."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), CONVERSATION)
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def calls(conversation: dict) -> list[Call]:
    """Every tool call of the attempt, in the order the model made them."""
    metadata = conversation.get("metadata", {})
    names, looking = metadata.get("tools", {}), set(metadata.get("looking", []))
    results = {block["call"]: block for message in conversation["messages"] if message["role"] == "tool"
               for block in message["content"] if block["type"] == "tool_result"}
    found = []
    for message in conversation["messages"]:
        for block in message["content"] if message["role"] == "assistant" else ():
            if block["type"] == "tool_call":
                result = results.get(block["id"], {"content": []})
                text = "".join(part.get("text", "") for part in result["content"])
                tool = names.get(block["name"])
                found.append(Call(tool, block["name"], block["arguments"], text, bool(result.get("error")),
                                  _looks(tool, block["arguments"], looking)))
    return found


def answer(conversation: dict) -> dict | None:
    """The answer, as the agent loop built it in the conversation's
    metadata (docs/chat-format.md, "The answer"): the final reply's text
    and the confidence the model gave when asked after it, from 0 to 1;
    None when the attempt ended with no final reply or its reply to the
    confidence question was not a number from 0 to 100."""
    return conversation.get("metadata", {}).get("answer")


def answer_text(conversation: dict) -> str:
    """The answer's text, the model's final reply, "" without an answer."""
    found = answer(conversation)
    return "" if found is None else found["answer"]


def scored(conversation: dict, score: float) -> float:
    """The attempt's score: score, or 0 without an answer (no final reply,
    or no confidence from 0 to 100 after it), whatever the files hold
    (maintainer's decisions, 2026-10-18 and 2026-10-23)."""
    return score if answer(conversation) is not None else 0.0


def report(conversation: dict, correct, **rules) -> str:
    """What a check prints (docs/post-training.md, "Rewards"; maintainer's
    decision, 2026-10-23): the JSON object of its named parts, each a
    number from 0 to 1: "correct", the task's own judgement as scored()
    gives it, and each rule given, such as sources_first=True; a task's
    check may report any part of its own, such as its conventions."""
    return json.dumps({"correct": scored(conversation, float(correct)),
                       **{name: float(value) for name, value in rules.items()}})


def brier(conversation: dict, correct: bool) -> float:
    """The Brier score of the confidence: (confidence - 1 or 0) squared, 0
    for a confidence that was exactly right, 1 for one exactly wrong, and 1
    without an answer, since no confidence was given."""
    found = answer(conversation)
    return 1.0 if found is None else (found["confidence"] - (1.0 if correct else 0.0)) ** 2


def before(conversation: dict, first, then) -> bool:
    """Whether every call to a tool of `then` comes after at least one call
    to a tool of `first` (true when no tool of `then` was called)."""
    seen = False
    for call in calls(conversation):
        if call.tool in then and not seen:
            return False
        seen = seen or call.tool in first
    return True


def _uncommented(command: str) -> str:
    """The command without its comments (a # outside quotes at the start
    of a word, to the end of its line) and its line continuations (a
    backslash before a line break), as bash reads it."""
    kept, quote, index = [], None, 0
    while index < len(command):
        char = command[index]
        if quote != "'" and char == "\\":
            if command[index + 1:index + 2] != "\n":
                kept.append(command[index:index + 2])
            index += 2
            continue
        if quote:
            quote = None if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char == "#" and (not kept or kept[-1][-1] in " \t\r\n" + PUNCTUATION):
            index = command.find("\n", index)
            if index < 0:
                break
            continue
        kept.append(char)
        index += 1
    return "".join(kept)


def programs(command: str) -> list[str]:
    """The names of the programs a bash command runs: the first word of
    each simple command, after any VAR=value and RESERVED words, and past
    RUNNERS with their options and the words they take, without its
    folder; [] when the command does not parse (an unclosed quote)."""
    lexer = shlex.shlex(_uncommented(command), posix=True, punctuation_chars=PUNCTUATION)
    lexer.whitespace, lexer.whitespace_split, lexer.commenters = " \t\r", True, ""
    try:
        words = list(lexer)
    except ValueError:
        return []
    found, starting, runner, value, operands = [], True, None, False, 0
    for word in words:
        name = posixpath.basename(word)
        if word and set(word) <= SEPARATING:
            starting, runner, value, operands = True, None, False, 0
        elif not starting:
            continue
        elif value:  # the value of a runner's option
            value = False
        elif runner is not None and word.startswith("-") and word != "-":
            value = word in runner[0]
        elif operands:
            operands -= 1
        elif ASSIGNMENT.fullmatch(word) or word in RESERVED:
            continue
        elif name in RUNNERS:
            runner = RUNNERS[name]
            operands = runner[1]
        else:
            found.append(name)
            starting = False
    return found


def _looks(tool, arguments, looking) -> bool:
    """Whether a call of the real tool `tool` looks at sources: a tool of
    `looking`, the offered looking tools, and bash only when its command
    runs a reading program (READERS)."""
    if tool not in looking:
        return False
    if tool == "bash":
        command = arguments.get("command")
        return isinstance(command, str) and any(name in READERS for name in programs(command))
    return True


def path_of(path: str) -> str:
    """A tool's path argument as an absolute path: a relative path starts at /work."""
    return posixpath.normpath(posixpath.join(WORK, path))


def mentions(command: str, path: str) -> bool:
    """Whether a shell command names the file at path (absolute, or relative to /work)."""
    names = {path, posixpath.relpath(path, WORK)} if path.startswith(WORK + "/") else {path}
    return any(re.search(rf"(?<![\w./-])(\./)?{re.escape(name)}(?![\w/.-])", command) for name in names)


def read_before_write(conversation: dict, new=()) -> bool:
    """Whether every file the attempt wrote or edited was read before: by
    read, by an earlier bash command that names it, or written by the
    attempt itself; new lists the files the task expects the model to
    create, which need no reading."""
    known, commands = {path_of(path) for path in new}, []
    for call in calls(conversation):
        if call.tool == "bash" and isinstance(call.arguments.get("command"), str):
            commands.append(call.arguments["command"])
        path = call.arguments.get("path")
        if call.tool not in ("read", *WRITES) or not isinstance(path, str):
            continue
        path = path_of(path)
        if call.tool in WRITES and path not in known and not any(mentions(command, path) for command in commands):
            return False
        if not call.failed:
            known.add(path)
    return True


def sources_first(conversation: dict, changes=CHANGES) -> bool:
    """Whether the attempt looked at sources (Call.looks) before it changed a
    file or asked the user (by default write, edit, ask), and looked at
    all when it did neither."""
    looked = False
    for call in calls(conversation):
        if call.tool in changes and not looked:
            return False
        looked = looked or call.looks
    return looked


def opened(call: Call, path: str) -> bool:
    """Whether the call showed the file at path (absolute) and did not
    fail: it looks (Call.looks) and its path argument is that file, or, for
    bash, its command names it (mentions) and runs a program that shows
    what a file holds (SHOWING)."""
    if not call.looks or call.failed:
        return False
    if call.tool == "bash":
        command = call.arguments["command"]
        return mentions(command, path) and not SHOWING.isdisjoint(programs(command))
    argument = call.arguments.get("path")
    return isinstance(argument, str) and path_of(argument) == path


def asked_last(conversation: dict, found: str = "") -> bool:
    """Whether every question to the user (ask) came after the attempt had
    opened every file where the answer lives (maintainer's decisions,
    2026-10-22 and 2026-10-23): the task's answered_in, as the files of
    /work the attempt started with (metadata.answered_in), each opened by
    one of the looking tools that can answer, the task's answered_by
    (metadata.answered_by). A question counts at once when no such file
    exists (the answer is absent), and when the attempt was offered none of
    those tools, since no look could find it. Given found, the answer's
    text, a question no longer counts once a call that looks showed it,
    its words in order as holds reads them: the answer was found."""
    metadata = conversation.get("metadata", {})
    offered = set(metadata.get("tools", {}).values())
    wanted = {name for name in metadata.get("answered_by", []) if name in offered}
    unopened = {path_of(path) for path in metadata.get("answered_in", [])} if wanted else set()
    shown = False
    for call in calls(conversation):
        if call.tool == "ask" and (unopened or shown):
            return False
        shown = shown or bool(found and call.looks and holds(call.result, found))
        if call.tool in wanted:
            unopened = {path for path in unopened if not opened(call, path)}
    return True


def read_before_ask(conversation: dict, text: str) -> bool:
    """Whether every question to the user (ask) came after a call that
    looks (Call.looks) and whose result showed text: its words, in order, as
    holds reads them; true when the attempt asked nothing."""
    shown = False
    for call in calls(conversation):
        if call.tool == "ask" and not shown:
            return False
        shown = shown or (call.looks and holds(call.result, text))
    return True


def beside(name: str) -> str:
    """The text of the file name in the folder of the running check."""
    with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), name), encoding="utf-8") as file:
        return file.read()


def file_text(path: str) -> str:
    """The text of a regular file the attempt left (relative to /work), ""
    when there is none there: a folder, a pipe or a device is read as no
    file, and a symbolic link is not followed."""
    try:
        descriptor = os.open(path_of(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return ""
    with os.fdopen(descriptor, "rb") as file:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return ""
        return file.read().decode("utf-8", errors="replace")


def _words(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def holds(text: str, expected: str) -> bool:
    """Whether text holds the words of expected one after the other, letter
    case, punctuation, Markdown marks and spacing aside; false when expected
    has no word."""
    want, have = _words(expected), _words(text)
    return bool(want) and any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


def not_enough(conversation: dict, missing: str = "", words: int | None = None) -> bool:
    """Whether the answer starts with NOT_ENOUGH, letter case and spacing
    aside: the attempt says the files do not hold what was asked; given
    missing, the text the task's files lack, the answer must also name it:
    hold its first `words` words (all of them by default), in order, as
    holds reads them (maintainer's decision, 2026-10-22)."""
    text = answer_text(conversation)
    if not text.strip().casefold().startswith(NOT_ENOUGH):
        return False
    return not missing or holds(text, " ".join(_words(missing)[:words]))


def pushed(expected: dict | None = None) -> bool:
    """Whether the remote of a repository task (expected: REPOSITORY next to
    the check, by default) holds on its branch exactly one new commit, whose
    parent is the commit it started from, whose tree is the expected tree,
    and whose message is not empty, spaces aside: the model writes its own
    message, never compared (maintainer's decision, 2026-10-22).

    The remote is outside /work (wpt_dataset.tasks.REMOTE), where the
    model could also have changed its settings: only its objects and
    references are copied into a fresh repository of the check's own, so
    git reads none of its settings."""
    expected = expected or json.loads(beside(REPOSITORY))
    remote = path_of(expected["remote"])
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GIT_CONFIG_NOSYSTEM": "1",
                   "GIT_CONFIG_GLOBAL": os.devnull, "HOME": "/nonexistent"}
    with tempfile.TemporaryDirectory() as folder:
        def git(*args):
            return subprocess.run(["git", "--git-dir", folder, *args], env=environment, stdin=subprocess.DEVNULL,
                                  capture_output=True, check=True).stdout.decode("utf-8", errors="replace")

        try:
            git("init", "-q", "--bare", "--template=")
            for part in ("objects", "refs"):
                shutil.copytree(os.path.join(remote, part), os.path.join(folder, part), symlinks=True,
                                dirs_exist_ok=True)
            if os.path.isfile(os.path.join(remote, "packed-refs")):
                shutil.copyfile(os.path.join(remote, "packed-refs"), os.path.join(folder, "packed-refs"),
                                follow_symlinks=False)
            head = git("rev-parse", "--verify", f"refs/heads/{expected['branch']}^{{commit}}").strip()
            parents = git("rev-list", "--parents", "-n", "1", head).split()[1:]
            tree = git("rev-parse", f"{head}^{{tree}}").strip()
            message = git("cat-file", "commit", head).split("\n\n", 1)[1]
        except (OSError, shutil.Error, subprocess.CalledProcessError, IndexError):
            return False
    return parents == [expected["parent"]] and tree == expected["tree"] and message.strip() != ""
