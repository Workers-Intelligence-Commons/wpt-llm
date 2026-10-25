"""Sources opened first: does the model open the sources it is given before
it answers, without opening everything blindly?

From the shared run of attempts (context.attempts()), the tasks that give
sources: files under docs/ in /work, or web pages under web/ that the
search and fetch tools answer from (docs/training.md, "The task
builder"). The share of attempts that opened sources before answering,
by the check library's sources_first (a look before any change or
question, and a look at all when there was neither), and the mean number
of tool calls of an attempt, so that opening everything shows.
"""

from wpt_eval import Result
from wpt_eval.attempts import library

DOCS, PAGES = "docs/", "web/"  # where a task's given files and its pages are, under files/


def gives_sources(task) -> bool:
    return any(path.startswith(DOCS) for path in task.files) or any(path.startswith(PAGES) for path in task.tool_files)


def run(context):
    records = [record for record in context.attempts() if gives_sources(record.task)]
    first = calls = 0
    for record in records:
        check = library(record.task.image)
        first += check.sources_first(record.conversation)
        calls += len(check.calls(record.conversation))
    data = {"attempts": len(records), "tasks": len({record.task.id for record in records}), "opened_first": first,
            "share": first / len(records) if records else None,
            "mean_tool_calls": calls / len(records) if records else None}
    if not records:
        return Result(data, "No measure task gives sources, so no attempt.\n")
    return Result(data, f"{first} of {len(records)} attempts at {data['tasks']} tasks ({data['share']:.1%}) opened "
                        f"sources before answering; {data['mean_tool_calls']:.1f} tool calls per attempt on average.\n")
