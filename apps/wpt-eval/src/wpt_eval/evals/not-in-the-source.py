"""Not in the source: does the model say so when its files do not hold the
answer?

From the shared run of attempts (context.attempts()), the evidence-removed
tasks (docs/post-training.md, "The task builder"), in two halves: those
whose paragraph was removed from its text (the removed paragraph is next
to the check, as missing.txt, and the right answer starts with "not
enough" and names it), and those that can be answered (rest.txt next to
the check, and "not enough" is wrong). The pass rate of each half, apart.
"""

from wpt_eval import Result
from wpt_eval.attempts import described, summary

KIND = "evidence-removed"
MISSING = "missing.txt"  # next to the check of a task whose paragraph was removed
HALVES = {"removed": "paragraph removed", "answerable": "answerable"}


def run(context):
    records = [record for record in context.attempts() if record.kind == KIND]
    halves = {"removed": [r for r in records if MISSING in r.task.check_files],
              "answerable": [r for r in records if MISSING not in r.task.check_files]}
    data = {half: summary(found) for half, found in halves.items()}
    if not records:
        return Result(data, f"No {KIND} measure task, so no attempt.\n")
    return Result(data, "".join(f"- {label}: {described(data[half])}\n" for half, label in HALVES.items()))
