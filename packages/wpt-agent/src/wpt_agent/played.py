"""Playing attempts at a task: what a training step and the measures
share, with no learning and no record of their own.

    ready(names)               the built images and their pools of tools, or Unready
    attempt_box(image, task, o) one attempt's container, its files arranged for the offer o
    play(task, offers, keep)   one attempt per offer, each in a fresh container, written together
                               by the model alone (no drafter) from one reading of what their
                               openings share; then, attempt by attempt in order, its parts and
                               score, and keep(index, outcome) at once: a Played, or the
                               SandboxError that dropped it (Docker failed, not the model's doing)
    gated(task, parts)         whether every rule the task's gates name holds (its part is 1)
    changes(before, after)     an attempt's changes to /work, as text
    changed(before, after)     the paths under /work an attempt changed

The caller passes the model, its tokenizer, the built images, how tokens
are drawn (Sampling: GREEDY always takes the likeliest token) and the
generator that draws them, so the same generator state gives the same
attempts again; or, instead of the model and the generator, writer: what
makes a writer like wpt_inference.Writer (the wpt manager's, whose model
writes on its gpu worker). A step that learns asks for each attempt's routes (R3);
a measure asks for its changes to /work, which people read when they
rate it.
"""

import difflib
import hashlib
from dataclasses import dataclass

from wpt_dataset.errors import DatasetError
from wpt_dataset.manifest import SYMLINK, glob_match_any
from wpt_dataset.repo import Repo

from . import images
from .check import CORRECT, score
from .images import ImageError
from .loop import Attempt, attempts
from .sandbox import AGENT, Limits, Sandbox, SandboxError
from .tools import Arranged, arranged

FILE_BYTES = 1 << 16  # a file of /work larger than this is named in the changes, never shown
CHANGES_BYTES = 1 << 16  # the changes of one attempt, as text, at most; the rest is cut
WITHIN_AREA = "within_area"  # the part of a task with an area: 1 when every path the attempt changed is inside it


class Unready(Exception):
    """Attempts cannot start: no repository, a task's image is not built
    on this machine, or its folder or pool is broken. Nothing was played."""


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_k: int | None  # None: every token


GREEDY = Sampling(1.0, 1)  # always the likeliest token (two equally likely ones aside)


@dataclass(frozen=True)
class Played:
    """An attempt that was scored."""
    attempt: Attempt
    score: float  # its CORRECT part, 0 when a gate is broken (gated) or the check gave no score
    routes: object = None  # its routed experts (wpt_model.Routes), when the caller asked for them
    changes: str | None = None  # its changes to /work (changes()), when the caller asked for them
    parts: dict | None = None  # the check's parts, and WITHIN_AREA for a task with an area; None without a score


def ready(names) -> tuple[dict, dict]:
    """({image name: the tag of its image built on this machine}, {image
    name: its pool of tools, data/images/<name>/tools.json}) of these
    images, from the repository holding the working folder; Unready naming
    the command that builds a missing image, or what is broken."""
    try:
        repo = Repo.find()
        return ({name: images.built(repo.root, name) for name in names},
                {name: repo.image_tools(name) for name in names})
    except (ImageError, DatasetError) as error:
        raise Unready(str(error)) from None


def attempt_box(image, task, offer) -> tuple[Sandbox, Arranged]:
    """A fresh container of image for one attempt at the task offered
    `offer`: the model's user, the task's container sizes, and the
    attempt's arrangement (tools.arranged), returned too."""
    found = arranged(task, offer)
    box = Sandbox(image, found.files, user=AGENT, limits=Limits(**task.container), tool_files=found.tool_files,
                  outside=found.outside, absent=found.absent)
    return box, found


def gated(task, parts: dict) -> bool:
    """Whether every rule the task's gates name holds: its part is 1 (a
    part the check did not print counts as broken)."""
    return all(parts.get(name, 0.0) >= 1 for name in task.gates)


def play(task, offers, keep, *, tokenizer, built, sampling: Sampling, check_seconds: float, result_bytes: int,
         model=None, generator=None, writer=None, routes=False, with_changes=False, terse=False,
         name="attempts") -> list:
    """One attempt at the task for each offer (wpt_agent.Offer), each in a
    fresh container of its image (built: {image name: tag}), written
    together by the model, its tokens drawn by sampling and generator, each
    opened with the task's instructions, or with terse its one-line form;
    then, attempt by attempt in order, its parts, its score and its
    container's removal, and at once keep(index, outcome), index from 1,
    outcome a Played (with routes, its routes; with with_changes, its
    changes to /work) or the SandboxError that dropped it. A task with an
    area gets the part WITHIN_AREA: 1 when every path under /work the
    attempt changed matches one of its patterns, else 0. A check that
    gives no score counts as 0, printed with the reason under name.
    writer(prompt, rows, *, temperature, top_k, stops, routes), when
    given, makes the writers in place of wpt_inference.Writer on model and
    generator. Returns what keep returned, in order."""
    if writer is None:
        from wpt_inference import Writer  # PyTorch is loaded only by what writes: wpt-agent's own commands need none

        def writer(prompt, rows, **settings):
            return Writer(model, prompt, rows, generator=generator, **settings)

    outcomes, boxes, placed, writers = {}, {}, {}, []
    for i, offer in enumerate(offers):
        try:
            boxes[i], placed[i] = attempt_box(built[task.image], task, offer)
        except SandboxError as error:
            outcomes[i] = error

    def start(ids, rows, stops):  # the opening read once for every attempt
        writers.append(writer(ids, rows, temperature=sampling.temperature, top_k=sampling.top_k, stops=stops,
                              routes=routes))
        return writers[-1]

    kept = []
    try:
        rows = {i: row for row, i in enumerate(boxes)}
        if boxes:
            done = attempts(task, list(boxes.values()), tokenizer, start, result_bytes=result_bytes,
                            offers=[offers[i] for i in boxes], terse=terse)
            outcomes.update(zip(boxes, done))
        for i in range(len(offers)):
            outcome, why = outcomes[i], None
            if not isinstance(outcome, SandboxError):
                try:
                    found, why = score(boxes[i], task.check, check_seconds, outcome.conversation, task.check_files,
                                       offers[i].hidden, tuple(placed[i].outside))
                    # what /work holds after the check stopped its processes
                    left = boxes[i].work_files(FILE_BYTES) if with_changes or task.area is not None else None
                    if found is not None and task.area is not None:
                        found[WITHIN_AREA] = float(all(glob_match_any(path, task.area)
                                                       for path in changed(placed[i].files, left)))
                    value = found[CORRECT] if found is not None and gated(task, found) else 0.0
                    outcome = Played(outcome, value, writers[-1].routes[rows[i]] if routes else None,
                                     changes(placed[i].files, left) if with_changes else None, found)
                except SandboxError as error:
                    outcome = error
            if i in boxes:
                try:
                    boxes.pop(i).close()
                except SandboxError as error:
                    outcome = error
            if why and not isinstance(outcome, SandboxError):  # score() gives "" with a score
                print(f"{name}: {task.id}: no score, counted as 0: {why}")
            kept.append(keep(i + 1, outcome))  # its line, before the next attempt is scored
    finally:
        for box in boxes.values():  # left open only when an error stops the attempts
            try:
                box.close()
            except SandboxError:
                pass  # the error that stopped them goes on
    return kept


def changes(before: dict, after: dict, limit: int = CHANGES_BYTES) -> str:
    """The changes from before ({path under /work: (manifest mode,
    content)}, what an attempt started with) to after ({path: (mode,
    content or None when larger than FILE_BYTES, size, sha256)}, what it
    left; Sandbox.work_files), as text: a unified diff of each changed text
    file, path by path; a file that is not UTF-8 text, or is larger than
    FILE_BYTES, is named with its size, never shown; a change of mode is
    named. Text past limit bytes is cut, with a line saying how much. ""
    when nothing changed."""
    parts = []
    for path in sorted(before.keys() | after.keys(), key=str.encode):
        old, new = before.get(path), after.get(path)
        if old is not None:
            old = (old[0], old[1] if len(old[1]) <= FILE_BYTES else None, len(old[1]),
                   hashlib.sha256(old[1]).hexdigest())
        if old is not None and new is not None and old[0] == new[0] and old[3] == new[3]:
            continue
        if old is not None and new is not None and old[0] != new[0]:
            parts.append(f"mode of {path}: {old[0]} -> {new[0]}\n")
            if old[3] == new[3]:
                continue
        old_text, new_text = _text(old), _text(new)
        if (old is None or old_text is not None) and (new is None or new_text is not None):
            lines = difflib.unified_diff((old_text or "").splitlines(keepends=True),
                                         (new_text or "").splitlines(keepends=True),
                                         "/dev/null" if old is None else f"a/{path}",
                                         "/dev/null" if new is None else f"b/{path}")
            parts += [line if line.endswith("\n") else f"{line}\n\\ No newline at end of file\n" for line in lines]
        else:
            what = "added" if old is None else "removed" if new is None else "changed"
            parts.append(f"{path}: {what}, not shown ({(new or old)[2]} bytes, not UTF-8 text or more than "
                         f"{FILE_BYTES})\n")
    data = "".join(parts).encode()
    if len(data) <= limit:
        return data.decode()
    return data[:limit].decode(errors="ignore") + f"\n[cut: {len(data) - limit} more bytes of changes]\n"


def changed(before: dict, after: dict) -> list[str]:
    """The paths under /work, in order, that an attempt added, removed, or
    whose mode or content it changed, from before and after as changes()
    takes them."""
    return sorted(path for path in before.keys() | after.keys()
                  if path not in before or path not in after or before[path][0] != after[path][0]
                  or hashlib.sha256(before[path][1]).hexdigest() != after[path][3])


def _text(found) -> str | None:
    """A file's content as text, None when it is not UTF-8 text (a NUL
    byte included) or was too large to read; a link as its target."""
    if found is None or found[1] is None:
        return None
    mode, content = found[:2]
    if mode == SYMLINK:
        return f"-> {content.decode(errors='replace')}\n"
    if b"\0" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None
