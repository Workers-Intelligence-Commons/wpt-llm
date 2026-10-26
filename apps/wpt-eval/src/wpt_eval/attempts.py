"""The shared run of attempts the agent measures read (docs/evaluation.md,
"The shared run of attempts").

    measure_tasks(store, provenance)  the measure-only tasks, read once per command (Context)
    run(context, plain)     ATTEMPTS attempts at every measure-only task, scored by its check:
                            one Record each, kept as a line of RECORD_FILE in the run's record
    library(image)          the check library (wpt_check.py) of an image, loaded on this machine,
                            so a measure reads an attempt by the same rules as the checks
    summary(records)        attempts, tasks, passed (a score of 1), pass rate, mean score;
                            described(summary) says it for people

The tasks are the measure-only ones of every admitted source
(wpt_dataset.tasks.read(..., measure=True): tagged eval, the hash share of
every task source, the pointer tasks on held-out text), read with the
model's own held-out share and dataset. The attempts are played by
wpt_agent.play, as the finish-tasks step plays its own: each in a fresh container
of the task's image, written together by the model alone, then scored by
the task's check. The model always writes its likeliest token
(wpt_agent.GREEDY), so the attempts at
one task differ only in the tools they are offered. plain=False offers
each attempt its seeded variation of the tools (wpt_agent.offer, from a
generator seeded with SEED, the task and the attempt); plain=True offers
the same attempt, from a generator in the same state, only the tools of
the same variants of the task's needs, under their real names, nothing
renamed, shuffled or added (wpt_agent.needed): the same hidden files and
workspace, so only the tools differ between the two runs.

Every model runs the same way, a pre-trained one included: an attempt
without an answer (no final reply, or no confidence from 0 to 100 after
it) scores what its check gives, 0 with the check library's rule.

Every attempt is a line of RECORD_FILE: a scored one with its
conversation, its score and its changes to /work (wpt_agent.changes: a
text diff from its start to its end, a file that is not text or is too
large named, never shown, the whole cut past a size), which the rating
page shows; one Docker dropped
(wpt_agent.SandboxError, not the model's doing), marked dropped, with
the error and the opening it would have started from, as the finish-tasks step
keeps its own. A dropped attempt has no Record: it is printed and
recorded as an event too. A line the store refuses raises
wpt_telemetry.RecordError.
"""

import json
import random
import types
from dataclasses import dataclass

import wpt_agent
from wpt_agent import Unready
from wpt_dataset import tasks as task_files
from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo

ATTEMPTS = 4  # attempts per task, unless wpt-eval's --attempts says otherwise
# how the attempts are checked: the finish-tasks step's values (wpt_training/environments/agent-tasks.py)
CHECK_SECONDS = 300
RESULT_BYTES = 4000
SEED = 0  # the tools' variation, and the generator (whose draws matter only between equally likely tokens)
RECORD_FILE = "attempts.jsonl"  # in the run's record folder, one line per attempt
LIBRARY = "wpt_check.py"  # the check library, in an image's folder


@dataclass(frozen=True)
class Record:
    """One scored attempt at a measure task."""
    task: task_files.Task
    kind: str  # the first part of the task's folder: "look-or-ask", "read-the-manual", ...
    index: int  # the attempt's number at its task, from 1
    plain: bool  # offered only the needed tools under their real names, or its seeded variation
    conversation: dict  # as conversation.json holds it for the check: messages, metadata tools, looking,
    #                     answered_by, answered_in, answer
    score: float  # the check's, from 0 to 1
    changes: str = ""  # its changes to /work (wpt_agent.changes)

    def line(self) -> dict:
        """The record as a line of RECORD_FILE: the task by its id."""
        return {"task": self.task.id, "kind": self.kind, "index": self.index, "plain": self.plain,
                "dropped": False, "conversation": self.conversation, "score": self.score, "changes": self.changes}


def kind(task) -> str:
    """The first part of the task's folder, in its id "<source id>/<folder>"."""
    return task.id.split("/", 1)[1].split("/", 1)[0]


def measure_tasks(store, provenance) -> list[task_files.Task]:
    """The measure-only tasks of every admitted source, their pointers
    filled in from the store, for the model whose provenance this is;
    wpt_agent.Unready when they cannot be read."""
    try:
        found, _ = task_files.read(Repo.find(), store, None, provenance["recipe"]["data"]["held_out"],
                                   measure=True, dataset=provenance.get("dataset"))
    except DatasetError as error:
        raise Unready(str(error)) from None
    return found


def run(context, plain: bool, count: int = ATTEMPTS) -> list[Record]:
    """count attempts at every measure-only task (context.measure_tasks()),
    task by task in reading order: one Record per attempt that was scored;
    every attempt, a dropped one included, kept as a line of RECORD_FILE
    as soon as it is scored."""
    tasks = context.measure_tasks()
    if not tasks:
        return []
    built, pools = wpt_agent.ready(sorted({task.image for task in tasks}))
    telemetry = context.telemetry
    generator = context.model.generator(SEED)  # a new one for every run: its draws start from the same place
    records = []
    for task in tasks:
        draws = [random.Random(f"{SEED}/{task.id}/{index}") for index in range(1, count + 1)]
        offers = [wpt_agent.needed(task, rng) if plain else wpt_agent.offer(task, pools.get(task.image, []), rng)
                  for rng in draws]

        def keep(index, outcome, task=task, offers=offers):
            if isinstance(outcome, wpt_agent.SandboxError):
                print(f"eval: {task.id}: attempt {index} dropped, Docker failed: {outcome}")
                telemetry.event("eval.attempt_dropped", task=task.id, attempt=index, plain=plain, error=str(outcome))
                line = {"task": task.id, "kind": kind(task), "index": index, "plain": plain, "dropped": True,
                        "error": str(outcome), "conversation": wpt_agent.opening(task, offers[index - 1])}
            else:
                record = Record(task, kind(task), index, plain, outcome.attempt.conversation, outcome.score,
                                outcome.changes or "")
                records.append(record)
                line = record.line()
            telemetry.append_file(RECORD_FILE, (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))

        with telemetry.span("eval.attempts", task=task.id, plain=plain, attempts=count):
            wpt_agent.play(task, offers, keep, tokenizer=context.tokenizer, built=built, sampling=wpt_agent.GREEDY,
                           writer=lambda prompt, rows, **settings: context.model.writer(
                               prompt, rows, generator=generator, **settings),
                           check_seconds=CHECK_SECONDS, result_bytes=RESULT_BYTES, with_changes=True, name="eval")
    return records


def summary(records) -> dict:
    """The attempts, the tasks they are at, how many passed, the pass rate
    and the mean score; both None without an attempt. An attempt passes
    when its score is 1, the check's full marks: a provisional rule, until
    the rewards stage defines a pass (docs/evaluation.md)."""
    records = list(records)
    passed = sum(record.score == 1 for record in records)
    return {"attempts": len(records), "tasks": len({record.task.id for record in records}), "passed": passed,
            "pass_rate": passed / len(records) if records else None,
            "mean_score": sum(record.score for record in records) / len(records) if records else None}


def described(found) -> str:
    """A summary for people: "3 of 8 attempts passed (37.5%) at 2 tasks, mean score 0.41"."""
    if not found["attempts"]:
        return "no attempt"
    return (f"{found['passed']} of {found['attempts']} attempts passed ({found['pass_rate']:.1%}) at "
            f"{found['tasks']} tasks, mean score {found['mean_score']:.2f}")


_libraries: dict[tuple[str, str], types.ModuleType] = {}  # by repository root and image


def library(image: str = "agent") -> types.ModuleType:
    """The check library of the image's folder, in the repository holding
    the current folder (wpt_check.py, read through the service layer),
    loaded as a module once per process; wpt_agent.Unready when there is
    no repository or the image holds none."""
    try:
        repo = Repo.find()
        key = (str(repo.root), image)
        if key not in _libraries:
            files = repo.image_files(image)
            if LIBRARY not in files:
                raise Unready(f"image {image!r} holds no check library ({LIBRARY})")
            module = types.ModuleType("wpt_check")
            exec(compile(files[LIBRARY][1], f"{image}/{LIBRARY}", "exec"), module.__dict__)
            _libraries[key] = module
    except DatasetError as error:
        raise Unready(str(error)) from None
    return _libraries[key]
