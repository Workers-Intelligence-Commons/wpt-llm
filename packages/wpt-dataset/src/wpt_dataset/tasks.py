"""Tasks for agentic post-training (docs/post-training.md).

Files in the `task` format form task folders, one folder per task:

    <folder>/task.json   instructions, image, limits and container sizes (schemas/task.schema.json)
    <folder>/files/...   the starting files, copied into the attempt's container
    <folder>/check       any program; run after the attempt, it prints a score from 0 to 1

A task folder is a folder holding task.json that is not inside another
task's files/. Every other file of the format belongs to exactly one task
folder, as its check or under its files/. A task's image must be a folder
of data/images/ holding a Dockerfile. The attempt's container, and the
check's, take SIZES, each raised by task.json's `container` when it asks
for more; /work and /tmp are kept in memory and may each hold `disk` MiB,
so 2 x disk + HEADROOM must stay below `memory`, or the task is refused.
Such files are never pre-training
documents (markdown.py gives them no Markdown); a post-training step reads
the tasks whose task.json carries a tag named after the step, through read().
"""

import json
import posixpath
from dataclasses import dataclass, field

from . import markdown
from .errors import InvalidFile, RuleViolation
from .manifest import SYMLINK
from .repo import Repo, validate
from .sources import source_tree, tags_of

TASK, CHECK, FILES = "task.json", "check", "files/"
# the containers' sizes when task.json asks for no more: MiB of memory (the files in /work and /tmp included),
# MiB that /work and /tmp may each hold, processes and threads at once, CPUs' worth of time
SIZES = {"memory": 4096, "disk": 1024, "processes": 256, "cpus": 2.0}
HEADROOM = 1024  # MiB of memory left to the processes when /work and /tmp are both full


@dataclass(frozen=True)
class Task:
    id: str  # "<source>/<folder>"
    instructions: str
    image: str  # a folder name in data/images/
    max_turns: int
    max_tokens: int
    command_seconds: float
    files: dict[str, tuple[str, bytes]]  # path under files/ -> (manifest mode, content; a link's is its target)
    check: bytes
    container: dict = field(default_factory=lambda: dict(SIZES))  # SIZES, raised by task.json's `container`


def folders(paths) -> dict[str, list[str]]:
    """{task folder ("" for the root): its paths}. InvalidFile names a path
    that belongs to no task folder."""
    found = []
    for path in sorted(paths, key=lambda p: (p.count("/"), p)):  # outer folders first
        folder = posixpath.dirname(path)
        if posixpath.basename(path) == TASK and not any(path.startswith(_prefix(f) + FILES) for f in found):
            found.append(folder)
    grouped = {folder: [] for folder in found}
    for path in paths:
        owner = max((f for f in found if path.startswith(_prefix(f))), key=len, default=None)
        inside = path.removeprefix(_prefix(owner)) if owner is not None else ""
        if inside not in (TASK, CHECK) and not inside.startswith(FILES):
            raise InvalidFile(f"{path}: a task file must be a task folder's {TASK}, {CHECK} or under its {FILES}")
        grouped[owner].append(path)
    return grouped


def _prefix(folder):
    return folder + "/" if folder else ""


def parse(source_id: str, entries: dict, store, repo: Repo) -> dict[str, Task]:
    """{task folder: Task} of a source's files in the task format ({path:
    Entry}), in folder order. InvalidFile names the file at fault."""
    tasks = {}
    for folder, paths in sorted(folders(entries).items()):
        prefix = _prefix(folder)
        where = f"{source_id}/{prefix}"
        check = entries.get(prefix + CHECK)
        if check is None:
            raise InvalidFile(f"{where}: a task folder needs its {CHECK} program")
        if SYMLINK in (check.mode, entries[prefix + TASK].mode):
            raise InvalidFile(f"{where}: {TASK} and {CHECK} must be files, not links")
        try:
            document = json.loads(store.get_bytes(entries[prefix + TASK].sha256))
        except json.JSONDecodeError as error:
            raise InvalidFile(f"{where}{TASK}: {error}") from error
        validate(document, "task", f"{where}{TASK}")
        try:
            repo.check_image(document["image"])
        except RuleViolation as error:
            raise InvalidFile(f"{where}{TASK}: {error}") from None
        container = {name: max(size, document.get("container", {}).get(name, 0)) for name, size in SIZES.items()}
        if 2 * container["disk"] + HEADROOM >= container["memory"]:
            raise InvalidFile(f"{where}{TASK}: /work and /tmp may hold {container['disk']} MiB each, which with "
                              f"{HEADROOM} MiB for the processes needs more than {container['memory']} MiB of "
                              f"memory; raise container.memory above {2 * container['disk'] + HEADROOM}")
        limits = document["limits"]
        files = {path.removeprefix(prefix + FILES): (entries[path].mode, store.get_bytes(entries[path].sha256))
                 for path in sorted(paths) if path.startswith(prefix + FILES)}
        tasks[folder] = Task(f"{source_id}/{folder}" if folder else source_id, document["instructions"],
                             document["image"], limits["max_turns"], limits["max_tokens"], limits["command_seconds"],
                             files, store.get_bytes(check.sha256), container)
    return tasks


def has_tasks(source) -> bool:
    return any(entry["format"] == "task" for entry in source.data["formats"])


def of_source(source, tree, store, repo: Repo) -> dict[str, Task]:
    """{task folder: Task} of a source's final tree: its paths in the task
    format; their images are folders of repo's data/images/."""
    if not has_tasks(source):
        return {}
    entries = {path: entry for path, entry in tree.items() if markdown.format_of(source, path)["format"] == "task"}
    return parse(source.id, entries, store, repo)


def read(repo: Repo, store, tag: str, log=print) -> tuple[list[Task], dict]:
    """([Task], {source id: address, revision, licenses}) for every task of
    an admitted source whose task.json is tagged `tag`, in source and folder
    order. A source whose license text is empty is left out, and log is
    called with a note naming it."""
    found, sources = [], {}
    for source in repo.sources(log):
        if not source.admitted:
            continue
        tree, _ = source_tree(source, store)
        tagged = [task for folder, task in of_source(source, tree, store, repo).items()
                  if tag in tags_of(source, _prefix(folder) + TASK)]
        if tagged:
            found += tagged
            sources[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                                  "licenses": source.data["licenses"]}
    return found, sources
