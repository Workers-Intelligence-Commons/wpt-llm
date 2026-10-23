"""A small agent harness: a model works on a task inside a container, with
the tools offered to it, and the task's check scores what it left and the
attempt's conversation.

    images.built(root, name)    the task's image, built by `wpt-agent build-image` from
                                data/images/<name>/ and its lock
    Sandbox(image, files, ...)  one fresh container per attempt: no network, limited sizes, the
                                files in /work, the model's commands run as a non-root user
    loop.attempts(task, ...)    the model writes, its tool calls run, results come back, until it
                                stops or a limit is reached; several attempts written together
                                (loop.attempt: one)
    tools.offer(task, ...)      the tools one attempt is offered: one variant of each need, the
                                others varied (tools.plain: every tool under its own name;
                                tools.needed: the same variants, nothing else);
                                tools.arranged: the attempt's files, those its variants hide
                                among the hidden ones, the image's folders it sees empty
    check.score(sandbox, ...)   the attempt's processes stopped, the task's check runs on a copy
                                of its /work in a fresh container, next to the attempt's
                                conversation, and prints 0 to 1
    played.play(task, ...)      attempts written by a model and scored, what a post-training
                                step and the measures share (played.ready: the images built)

The tools are read, write, edit and bash, ask when the task holds the
user's replies, and those the task or its image's pool declares, which
programs in the image answer (tools.py). Tasks come from the
`task` format of sources (wpt_dataset.tasks); this package holds no
learning: a post-training step, or a chat or evaluation command, decides
what to do with an attempt and its score.
"""

from .check import score
from .images import ImageError
from .loop import Attempt, attempt, attempts, opening
from .played import GREEDY, Played, Sampling, Unready, attempt_box, changes, play, ready
from .sandbox import AGENT, Limits, Sandbox, SandboxError, SandboxFull
from .tools import TOOLS, Arranged, Offer, arranged, needed, offer, plain

__all__ = ["AGENT", "GREEDY", "Arranged", "Attempt", "ImageError", "Limits", "Offer", "Played", "Sampling", "Sandbox",
           "SandboxError", "SandboxFull", "TOOLS", "Unready", "arranged", "attempt", "attempt_box", "attempts",
           "changes", "needed", "offer", "opening", "plain", "play", "ready", "score"]
