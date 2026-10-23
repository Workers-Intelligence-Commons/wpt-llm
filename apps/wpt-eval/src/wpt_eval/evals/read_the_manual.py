"""Reading gain, agent form: does the model use a program better when its
manual is there to read (docs/evaluation.md, "Reading the manual")?

The read-the-manual tasks (docs/post-training.md, "The task builder") ask
for a result only ondrik gives, a compiled program of the agent image
written for WPT, so no training text describes it and the image holds no
readable documentation of it besides the manual. Half of them remove the
manual (task.json's `absent` names its folder); the instructions are the
same in both halves. The measure reads the attempts every agent measure
shares (context.attempts(): the measure-only tasks, the usual tool
variation), keeps those at read-the-manual tasks, and gives for each half
the attempts, the tasks, the pass rate and the mean score, and the gain,
the first pass rate minus the second. An attempt passes when its check
scores 1 (wpt_eval.attempts.summary). A model that gives no answer (no
final reply, or no confidence from 0 to 100 after it) scores what it
scores, as in every measure.
"""

from wpt_dataset.builder import MANUAL, READ_THE_MANUAL
from wpt_eval import Result
from wpt_eval.attempts import described, summary

HALVES = {"with_manual": "with the manual", "without_manual": "without the manual"}


def run(context):
    halves = {half: [] for half in HALVES}
    for record in context.attempts():
        if record.kind == READ_THE_MANUAL:
            halves["without_manual" if MANUAL in record.task.absent else "with_manual"].append(record)
    data = {half: summary(records) for half, records in halves.items()}
    rates = [data[half]["pass_rate"] for half in HALVES]
    data["gain"] = None if None in rates else rates[0] - rates[1]
    if not any(halves.values()):
        return Result(data, f"No attempt at a {READ_THE_MANUAL} task: the measure-only tasks hold none, or Docker "
                            f"dropped every attempt.\n")
    lines = [f"- {label}: {described(data[half])}\n" for half, label in HALVES.items()]
    gain = "unknown, a half has no attempt" if data["gain"] is None else f"{data['gain']:+.1%}"
    return Result(data, "".join(lines) + f"- gain from the manual, in pass rate: {gain}\n")
