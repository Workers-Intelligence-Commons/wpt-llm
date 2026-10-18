"""The task editor's routes: the task folders of a source (task.json, the
check, the starting files), one written or removed as the next patch (the
task rules of docs/post-training.md, "Tasks"); and git's status of the
data files, which every page shows. As `wpt-dataset tasks`, `task`,
`save-task` and `status`, through the service layer."""

from ..server import Route
from .common import PATCH, SAVED, STRING, VERSION, refusals


def routes(context):
    repo = context.repo

    @refusals
    def tasks(values, body):
        version, found = repo.tasks(values["id"], context.store)
        return 200, {"version": version, "tasks": found}

    @refusals
    def save(values, body):
        return 200, {"version": repo.save_task(values["id"], values["folder"], body["task"], body.get("check", ""),
                                               body.get("files", {}), body["patch"], body["version"], context.store)}

    def status(values, body):
        found = repo.status()
        return 200, {"git": found is not None, "files": found or []}

    return [
        Route("GET", "/api/sources/{id}/tasks", tasks,
              "the task folders of a source: task.json, the check, the starting files (a person's text) and "
              "the links kept as they are",
              answer={"type": "object", "required": ["version", "tasks"],
                      "properties": {"version": STRING, "tasks": {"type": "object"}}}),
        Route("PUT", "/api/sources/{id}/tasks/{folder}", save,
              "write a task folder of an empty source as exactly these files, or remove it (task null), as the "
              "next patch holding only the files that change (each keeps its mode; links stay as they are; a task "
              "at the source's root is refused); a new folder gets a task format entry (a person's text)",
              body={"type": "object", "required": ["task", "patch", "version"],
                    "properties": {"task": {"type": ["object", "null"]}, "check": STRING,
                                   "files": {"type": "object", "additionalProperties": STRING},
                                   "patch": PATCH, "version": VERSION}}, answer=SAVED),
        Route("GET", "/api/status", status,
              "git's status of every changed or new file under data/ (commits stay in git)",
              answer={"type": "object", "required": ["git", "files"],
                      "properties": {"git": {"type": "boolean"}, "files": {"type": "array", "items": {
                          "type": "object", "properties": {"path": STRING, "state": STRING}}}}}),
    ]
