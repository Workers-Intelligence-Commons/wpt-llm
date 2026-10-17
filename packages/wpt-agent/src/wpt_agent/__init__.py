"""A small agent harness: a model works on a task inside a container, with
four tools, and the task's check scores what it left.

    images.built(root, name)    the task's image, built by `wpt-agent build-image` from
                                data/images/<name>/ and its lock
    Sandbox(image, files, ...)  one fresh container per attempt: no network, limited sizes, the
                                files in /work, the model's commands run as a non-root user
    loop.attempt(task, ...)     the model writes, its tool calls run, results come back, until it
                                stops or a limit is reached
    check.score(sandbox, ...)   the attempt's processes stopped, the task's check runs on a copy
                                of its /work in a fresh container and prints 0 to 1

The tools are read, write, edit and bash (tools.py). Tasks come from the
`task` format of sources (wpt_dataset.tasks); this package holds no
learning: a post-training step, or a chat or evaluation command, decides
what to do with an attempt and its score.
"""

from .check import score
from .images import ImageError
from .loop import Attempt, attempt, opening
from .sandbox import AGENT, Limits, Sandbox, SandboxError, SandboxFull
from .tools import TOOLS

__all__ = ["AGENT", "Attempt", "ImageError", "Limits", "Sandbox", "SandboxError", "SandboxFull", "TOOLS", "attempt",
           "opening", "score"]
