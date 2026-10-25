"""Ask versus look: does the model look where it can and ask the user only
for what only the user knows?

From the shared run of attempts (context.attempts()), the look-or-ask
tasks (docs/training.md, "The task builder"), in two halves: those
where NOTES.md names the file the result goes into (the task does not
need ask: asking scores down), and those where only the reply of the ask
tool has it (the task needs ask). For each half, the pass rate and the
share of attempts that asked: called ask, by its real name, at least once.
"""

from wpt_eval import Result
from wpt_eval.attempts import described, library, summary

KIND = "look-or-ask"
HALVES = {"in_file": "NOTES.md names the file", "ask_only": "only the user knows the file"}


def needs_ask(task) -> bool:
    return any("ask" in variant.tools for need in task.needs for variant in need)


def asked(record) -> bool:
    return any(call.tool == "ask" for call in library(record.task.image).calls(record.conversation))


def run(context):
    records = [record for record in context.attempts() if record.kind == KIND]
    halves = {"in_file": [r for r in records if not needs_ask(r.task)],
              "ask_only": [r for r in records if needs_ask(r.task)]}
    data = {}
    for half, found in halves.items():
        count = sum(asked(record) for record in found)
        data[half] = {**summary(found), "asked": count, "asked_share": count / len(found) if found else None}
    if not records:
        return Result(data, f"No {KIND} measure task, so no attempt.\n")
    lines = []
    for half, label in HALVES.items():
        found = data[half]
        lines.append(f"- {label}: {described(found)}"
                     + (f"; {found['asked']} asked the user ({found['asked_share']:.1%})" if found["attempts"] else "")
                     + "\n")
    return Result(data, "".join(lines))
