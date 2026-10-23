"""Code versus other tasks: is the model drifting toward a code specialist?

The pass rates of the shared run's attempts (context.attempts()) at code
tasks against those at every other task, with their counts. A task is a
code task when any document it points to is tagged code; for a repository
pointer, the tags of the files the commit changes count (maintainer's
decision, 2026-10-23): wpt_dataset.tasks.document_tags. A task that
points to no document is another task. The gap should stay small.
"""

from wpt_dataset.repo import Repo
from wpt_dataset.tasks import document_tags
from wpt_eval import Result
from wpt_eval.attempts import described, summary

CODE = "code"


def run(context):
    records = context.attempts()
    repo, code = Repo.find(), {}
    for record in records:
        task = record.task
        if task.id not in code:
            code[task.id] = any(CODE in document_tags(repo, context.store, sid, document)
                                for sid, document, _ in task.documents)
    data = {"code": summary(r for r in records if code[r.task.id]),
            "other": summary(r for r in records if not code[r.task.id])}
    rates = data["code"]["pass_rate"], data["other"]["pass_rate"]
    data["gap"] = None if None in rates else rates[0] - rates[1]
    if not records:
        return Result(data, "No attempt was scored: there is no measure task, or Docker dropped every attempt.\n")
    return Result(data, f"- code tasks: {described(data['code'])}\n- other tasks: {described(data['other'])}\n"
                        + (f"- gap, code minus other: {data['gap']:+.1%}\n" if data["gap"] is not None else ""))
