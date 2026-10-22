"""Scoring an attempt: the task's check runs on what the attempt left.

When the attempt has ended, every process of its container is stopped
(Sandbox.freeze), also when they filled it, and the check runs in a fresh container of the same
image and limits, as root, holding only a copy of the attempt's /work and
of the task's folders outside /work (Sandbox's `outside`, such as a
repository's remote): nothing the model started or changed elsewhere can
reach the check. The
check enters that container in a folder of its own and runs from /work,
so the model's files are its current folder: a check must not start an
interpreter that loads code from there (docs/post-training.md, "Scores").
Next to the check, in its folder, is CONVERSATION: the attempt's
conversation in the chat format, whose metadata maps each tool's name as
the model saw it to the real tool's and holds the answer block's value
(loop.py), so the check scores the files, the trace of tool calls and the
confidence; the check finds it as "$(dirname "$0")/conversation.json".
The task's check_files (the texts its pointers put there,
wpt_dataset.tasks) are there too, by their paths. The files the attempt's
variant hid (tools.Offer.hidden, such as the push tool's remote) are in
the check's container at their place, in /work or at their folder outside
it, as the tools' programs left them, in place of whatever the model put
there.
Its standard output, without the white space around it, must be one number
from 0 to 1: the score. Its exit code and its standard error are not read.
"""

import json
import math
import secrets

from wpt_dataset.manifest import EXECUTABLE, FILE
from wpt_dataset.tasks import CONVERSATION

from .sandbox import ROOT, TOOL_FILES, WORK, Sandbox, SandboxFull

OUTPUT = 4096  # bytes of the check's output kept


def score(attempt: Sandbox, check: bytes, seconds: float, conversation: dict,
          beside: dict | None = None, hidden=(), outside=()) -> tuple[float | None, str]:
    """(score, "") or, when the check gave no score, (None, why); the check
    reads the attempt's conversation next to it, and the files `beside`
    ({path: (manifest mode, content)}, a task's check_files), finds the
    hidden paths (paths under files/, or folders outside /work, the
    attempt's variant hid) at their place, and the attempt's folders
    `outside` /work (absolute paths) at theirs. Raises SandboxError when
    docker fails."""
    # (folder in the attempt, folder in the check, paths under both)
    carried = [(TOOL_FILES, WORK, [path for path in hidden if not path.startswith("/")]),
               (TOOL_FILES, "/", [path.lstrip("/") for path in hidden if path.startswith("/")]),
               ("/", "/", [path.lstrip("/") for path in outside])]
    carried = [(root, place, paths) for root, place, paths in carried if paths]
    saved = []
    try:
        for root, place, paths in carried:
            saved.append((attempt.save(root, paths), place, paths))
        attempt.freeze()
        with Sandbox(attempt.image, {}, user=ROOT, limits=attempt.limits) as clean:
            attempt.copy_work(clean)
            for archive, place, paths in saved:
                if archive is not None:
                    clean.restore(archive, place, paths)
            folder = f"/tmp/wpt-check-{secrets.token_hex(8)}"
            path = f"{folder}/check"
            clean.put(path, check, EXECUTABLE)
            # ASCII JSON: a lone surrogate in a tool call's arguments stays an escape
            clean.put(f"{folder}/{CONVERSATION}", json.dumps(conversation).encode(), FILE)
            for name, (mode, content) in (beside or {}).items():
                clean.put(f"{folder}/{name}", content, mode)
            try:
                ran = clean.run([path], seconds=seconds, limit=OUTPUT)
            except SandboxFull as error:  # the copy of /work left no room to start the check
                return None, f"the check could not start: {error}"
    finally:
        for archive, _, _ in saved:
            if archive is not None:
                archive.close()
    if ran.timed_out:
        return None, f"the check was stopped after {seconds} seconds"
    printed = ran.stdout.decode("utf-8", errors="replace").strip()
    try:
        value = float(printed)
    except ValueError:
        value = math.nan
    if not 0 <= value <= 1:  # also refuses nan
        detail = ran.stderr.decode("utf-8", errors="replace").strip()[-500:]
        return None, (f"the check printed {printed[:100]!r}, not a number from 0 to 1 (exit code {ran.code})"
                      + (f": {detail}" if detail else ""))
    return value, ""
