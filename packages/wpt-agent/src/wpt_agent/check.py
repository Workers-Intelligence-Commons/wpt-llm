"""Scoring an attempt: the task's check runs on what the attempt left.

When the attempt has ended, every process of its container is stopped
(Sandbox.freeze), also when they filled it, and the check runs in a fresh container of the same
image and limits, as root, holding only a copy of the attempt's /work:
nothing the model started or changed elsewhere can reach the check. The
check enters that container in a folder of its own and runs from /work,
so the model's files are its current folder: a check must not start an
interpreter that loads code from there (docs/post-training.md, "Scores").
Its standard output, without the white space around it, must be one number
from 0 to 1: the score. Its exit code and its standard error are not read.
"""

import math
import secrets

from wpt_dataset.manifest import EXECUTABLE

from .sandbox import ROOT, Sandbox, SandboxFull

OUTPUT = 4096  # bytes of the check's output kept


def score(attempt: Sandbox, check: bytes, seconds: float) -> tuple[float | None, str]:
    """(score, "") or, when the check gave no score, (None, why). Raises
    SandboxError when docker fails."""
    attempt.freeze()
    with Sandbox(attempt.image, {}, user=ROOT, limits=attempt.limits) as clean:
        attempt.copy_work(clean)
        path = f"/tmp/wpt-check-{secrets.token_hex(8)}/check"
        clean.put(path, check, EXECUTABLE)
        try:
            ran = clean.run([path], seconds=seconds, limit=OUTPUT)
        except SandboxFull as error:  # the copy of /work left no room to start the check
            return None, f"the check could not start: {error}"
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
