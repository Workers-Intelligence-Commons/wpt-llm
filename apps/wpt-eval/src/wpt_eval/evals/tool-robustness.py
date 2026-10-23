"""Robustness to tool changes: does the model do as well when its tools are
renamed, reshuffled, padded or missing?

The pass rate of the shared run's attempts with each attempt's seeded
variation of the tools (context.attempts()) against the same attempts
with plain tools (context.attempts(plain=True): only the tools of the same
variants of the task's needs, under their real names, nothing renamed,
shuffled or added), and the gap, plain minus varied. The gap should shrink
as training goes on. Only the tasks that have scored attempts in both runs
count (an attempt Docker dropped has none).
"""

from wpt_eval import Result
from wpt_eval.attempts import described, summary


def run(context):
    varied, plain = context.attempts(), context.attempts(plain=True)
    both = {record.task.id for record in varied} & {record.task.id for record in plain}
    data = {"varied": summary(r for r in varied if r.task.id in both),
            "plain": summary(r for r in plain if r.task.id in both)}
    rates = data["plain"]["pass_rate"], data["varied"]["pass_rate"]
    data["gap"] = None if None in rates else rates[0] - rates[1]
    if not context.measure_tasks():
        return Result(data, "No measure task, so no attempt.\n")
    if not both:
        return Result(data, "No measure task has scored attempts in both runs: Docker dropped every attempt of each "
                            "task in one of them.\n")
    return Result(data, f"- plain tools: {described(data['plain'])}\n- varied tools: {described(data['varied'])}\n"
                        f"- gap, plain minus varied: {data['gap']:+.1%}\n")
