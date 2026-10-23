"""Tasks for agentic post-training (docs/post-training.md).

Files in the `task` format form task folders, one folder per task:

    <folder>/task.json   instructions (and, optionally, the same request in one line, terse), image,
                         limits, container sizes, the tools it needs and offers, the user's replies to
                         the ask tool (schemas/task.schema.json)
    <folder>/files/...   the starting files, copied into the attempt's container; those task.json's
                         tool_files names go where only the tools' programs read them
    <folder>/check       any program; run after the attempt, it prints a score from 0 to 1

A task folder is a folder holding task.json that is not inside another
task's files/. Every other file of the format belongs to exactly one task
folder, as its check or under its files/. A task's image must be a folder
of data/images/ holding a Dockerfile. The attempt's container, and the
check's, take SIZES, each raised by task.json's `container` when it asks
for more; /work and /tmp are kept in memory and may each hold `disk` MiB,
so 2 x disk + HEADROOM must stay below `memory`, or the task is refused.
A task's tools are the built-in ones (BUILT_IN), `ask` when it has replies,
and those it declares, which may not take those names; each name in
`needs` is one of them, in one need only (a need's variants may share
one), and `ask` needs replies. A variant is one tool, or tools offered
together with the paths under files/ that it hides (Variant): in an
attempt offered it, they go where only the tools' programs read them. A
task with replies names, in `answered_by`, the looking tools that can
answer what it leaves open: looking tools (looks) it or its image's pool
offers. Every
reply's pattern must compile, and every tool_files entry, a variant's
included, must name a file of files/ (or one a pointer makes) or a
folder holding some; a variant's may also name REMOTE, a repository
pointer's remote outside /work. Each `absent` entry is a folder of the
image, outside /work and /tmp (OWN), that the task's attempts see empty:
the image installs something there that the task removes, such as the
manual of the read-the-manual tasks' program. A task's terse form is one
line: no line break, as str.splitlines() finds them.

A task built by a program is a pointer task: task.json's `pointers` name,
for each text it needs, the source, the document (its path in the
source, as markdown.documents gives it) and the span (bytes of the
document's Markdown, the end excluded), the sha256 of that span, and the
parts of it the builder removed; the repository holds the pointer, never
the text. Each pointer's text becomes a file of the task, under files/ (in
/work, or hidden when tool_files names it) or next to the check. fill()
reads the texts from the store and checks each span against its sha256,
so a task whose source changed is refused until its builder makes it again.
A repository pointer names instead a commit of a git source's history
(its .git/history/<commit> file) and the source checksum the task was
built from: fill() makes /work itself a git repository at the commit's
parent, the real state just before the commit (every later commit undone,
newest first, in the history's order, then the commit), with the
commit's changes in its working tree, not committed, and a bare
repository outside /work, at REMOTE, as its `origin`; next to the check,
REPOSITORY says what the pushed commit must be (docs/post-training.md,
"Pointer tasks").

Such files are never pre-training
documents (markdown.py gives them no Markdown); a post-training step reads
the tasks whose task.json carries a tag named after the step, through
read(), which fills in the pointers and leaves out the measure-only tasks
(measure_only: among them every task pointing into a document whose
Markdown is the text of a held-out document of the base model's dataset,
held_out_texts); read(..., measure=True) gives the measure-only ones
instead.
"""

import hashlib
import json
import os
import posixpath
import re
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

from . import markdown
from .errors import DocumentListMissing, InvalidFile, RuleViolation
from .manifest import FILE, HISTORY, MERGE, ORDER, SYMLINK
from .repo import Repo, validate
from .sources import held_out_group, load_index, source_tree, tags_of, temporary_git

TASK, CHECK, FILES = "task.json", "check", "files/"
# the containers' sizes when task.json asks for no more: MiB of memory (the files in /work and /tmp included),
# MiB that /work and /tmp may each hold, processes and threads at once, CPUs' worth of time
SIZES = {"memory": 4096, "disk": 1024, "processes": 256, "cpus": 2.0}
HEADROOM = 1024  # MiB of memory left to the processes when /work and /tmp are both full
BUILT_IN = ("read", "write", "edit", "bash")  # the tools every task can offer (wpt_agent.tools)
ASK = "ask"  # the tool a task with replies offers
LOOKING = ("read", "bash")  # the built-in tools that look at sources; a declared tool looks when marked `looks`
EVAL = "eval"  # the tag of the measure tasks people write: no training step reads them
CONVERSATION = "conversation.json"  # the attempt's conversation, next to the check (wpt_agent.check puts it there)
REPOSITORY = "repository.json"  # next to the check: what a repository pointer's pushed commit must be
BESIDE_CHECK = (CHECK, CONVERSATION, REPOSITORY)  # names next to the check that a pointer's text may not take
BRANCH = "main"  # the branch of a repository pointer's repository and of its remote
# a repository pointer's bare remote, outside /work, so that `git add -A` in /work never takes it (maintainer's
# decision, 2026-10-22); a variant hides it by naming this path in its tool_files
REMOTE = "/tmp/origin.git"
OWN = ("/work", "/tmp")  # the attempt's own folders, kept in memory, which `absent` may not name
PATCHES = 200  # history files undone per `git apply`, so the command line stays short


@dataclass(frozen=True)
class Pointer:
    """A text a task takes from a source's document (task.json's `pointers`)."""
    to: str  # where the text goes: "files" (/work), "tool_files" (hidden), "check_files" (next to the check),
    #          or "repository" (/work is the repository)
    path: str  # its path there; "" for a repository
    source: str
    document: str  # the document's path in its source; a repository pointer's .git/history/<commit>
    span: tuple[int, int] | None  # bytes of the document's Markdown, the end excluded; None for a repository
    sha256: str  # of those bytes; "" for a repository
    removed: tuple = ()  # (start, end) of each part of the span left out, in the document's bytes
    checksum: str = ""  # a repository's: the source checksum its task was built from


class Variant(NamedTuple):
    """One way to meet a need: tools offered together, and the paths under
    files/ that go to the tools' programs in an attempt offered them."""
    tools: tuple[str, ...]
    tool_files: tuple[str, ...] = ()


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
    needs: tuple = ()  # each need's variants, as tuples of Variants
    replies: tuple = ()  # (pattern, reply) of each reply to the ask tool, in order
    tools: tuple = ()  # the task's own tool declarations (name, description, parameters, program, names)
    tool_files: dict = field(default_factory=dict)  # the files under files/ only the tools' programs read, as `files`
    outside: dict = field(default_factory=dict)  # {absolute folder outside /work: its files, as `files`}: a
    #                                              repository pointer's remote at REMOTE
    check_files: dict = field(default_factory=dict)  # files put next to the check, as `files`
    pointers: tuple = ()  # the Pointers fill() has not filled in yet
    answered_by: tuple = ()  # the looking tools that can answer what the task leaves open (wpt_check.asked_last)
    documents: tuple = ()  # (source, document, sha256 of its Markdown or None) of each document its pointers name
    held_out_keys: tuple = ()  # what the held-out hash reads for each of them: its held-out group, or its id
    absent: tuple = ()  # the image's folders the attempts see empty (task.json's `absent`)
    terse: str = ""  # the same request as instructions, in one line; "" when the task has none


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
    known = set(repo.source_ids())  # read once: the sources the pointers may name
    pools = {}  # {image: its pool's tool declarations}
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
        terse = document.get("terse", "")
        if terse and terse.splitlines() != [terse]:
            raise InvalidFile(f"{where}{TASK}: terse is the request in one line, and it holds a line break")
        limits = document["limits"]
        files = {path.removeprefix(prefix + FILES): (entries[path].mode, store.get_bytes(entries[path].sha256))
                 for path in sorted(paths) if path.startswith(prefix + FILES)}
        replies = tuple((reply["pattern"], reply["reply"]) for reply in document.get("replies", []))
        tools = tuple(document.get("tools", []))
        needs = tuple(tuple(Variant((variant,)) if isinstance(variant, str) else
                            Variant(tuple(variant["tools"]), tuple(variant.get("tool_files", [])))
                            for variant in variants) for variants in document["needs"])
        answered_by = tuple(document.get("answered_by", []))
        absent = tuple(document.get("absent", []))
        for place in absent:
            if posixpath.normpath(place) != place or any(place == own or place.startswith(own + "/") for own in OWN):
                raise InvalidFile(f"{where}{TASK}: absent names {place!r}; it names a folder of the image, "
                                  f"written without . or .., outside {' and '.join(OWN)}")
        pointed, made = _check_pointers(f"{where}{TASK}", document.get("pointers", []), files, known)
        if document["image"] not in pools:  # read once per image
            pools[document["image"]] = repo.image_tools(document["image"])
        pool = pools[document["image"]]
        hidden = _check_tools(f"{where}{TASK}", needs, replies, tools, document.get("tool_files", []),
                              [*files, *pointed], answered_by, pool, [*files, *pointed, *made])
        pointers = tuple(Pointer({"check": "check_files", "repository": "repository"}.get(pointer["for"]) or
                                 ("tool_files" if pointer["path"] in hidden else "files"),
                                 pointer.get("path", ""), pointer["source"], pointer["document"],
                                 tuple(pointer["span"]) if "span" in pointer else None, pointer.get("sha256", ""),
                                 tuple(tuple(part) for part in pointer.get("removed", [])),
                                 pointer.get("checksum", ""))
                         for pointer in document.get("pointers", []))
        tasks[folder] = Task(f"{source_id}/{folder}" if folder else source_id, document["instructions"],
                             document["image"], limits["max_turns"], limits["max_tokens"], limits["command_seconds"],
                             {path: entry for path, entry in files.items() if path not in hidden},
                             store.get_bytes(check.sha256), container, needs, replies, tools,
                             {path: files[path] for path in hidden if path in files}, pointers=pointers,
                             answered_by=answered_by, absent=absent, terse=terse)
    return tasks


def _check_pointers(where, pointers, files, known) -> tuple[list[str], list[str]]:
    """(the paths under files/ that text pointers fill, what a repository
    pointer makes outside /work: REMOTE); InvalidFile when one names a
    source that does not exist, takes the place of another file or folder,
    or has parts removed outside its span or overlapping, and when a task
    has more than one repository pointer, or one beside any other file in
    /work (/work is the repository itself: maintainer's decision,
    2026-10-22)."""
    taken = {"files": list(files), "check": list(BESIDE_CHECK)}
    repositories = [pointer for pointer in pointers if pointer["for"] == "repository"]
    if len(repositories) > 1:
        raise InvalidFile(f"{where}: {len(repositories)} repository pointers; a task has at most one")
    if repositories and (files or any(pointer["for"] == "files" for pointer in pointers)):
        raise InvalidFile(f"{where}: /work is the repository of its repository pointer, so the task has no other "
                          f"file there: no {FILES} and no other pointer to files")
    for pointer in pointers:
        if pointer["source"] not in known:
            raise InvalidFile(f"{where}: a pointer names the source {pointer['source']!r}, which does not exist")
        if pointer["for"] == "repository":
            continue
        path = pointer["path"]
        if any(path == other or path.startswith(other + "/") or other.startswith(path + "/")
               for other in taken[pointer["for"]]):
            raise InvalidFile(f"{where}: the pointer to {path!r} takes the place of another file or folder")
        taken[pointer["for"]].append(path)
        start, end = pointer["span"]
        if end <= start:
            raise InvalidFile(f"{where}: the pointer to {path!r} has the empty span {[start, end]}")
        last = start
        for part in pointer.get("removed", []):
            if not last <= part[0] < part[1] <= end:
                raise InvalidFile(f"{where}: the pointer to {path!r} removes {part}, which is not inside its span "
                                  f"{[start, end]} after the parts before it")
            last = part[1]
    return [pointer["path"] for pointer in pointers if pointer["for"] == "files"], [REMOTE] if repositories else []


def fill(tasks, repo: Repo, store) -> tuple[list[Task], dict]:
    """(the tasks, each pointer's text put in its place, its pointers gone
    and its documents (source, document, sha256 of the document's whole
    Markdown, for measure_only) kept; {source id: address, revision,
    licenses} of every source they point into}). Each source's final tree
    comes from the store (fetched
    when it is not there), and each span must hold the bytes its sha256
    names; a repository pointer's source must still have the checksum it
    names, and the commit must be made as repository() makes it.
    InvalidFile names the task and the pointer that cannot be filled: its
    source not admitted or changed, its document gone, its span changed or
    cutting a character, its commit not made (repository())."""
    wanted = {}
    for task in tasks:
        for pointer in task.pointers:  # a repository pointer's history file is a document too
            wanted.setdefault(pointer.source, set()).add(pointer.document)
    texts, trees, sources, checksums, read_sources = {}, {}, {}, {}, {}
    for sid, paths in sorted(wanted.items()):
        source = read_sources[sid] = repo.source(sid)
        if not source.admitted:
            raise InvalidFile(f"{sid}: tasks point into it, but it is not admitted")
        trees[sid], _ = source_tree(source, store)
        texts |= {(sid, path): text for path, text in markdown.chosen(source, trees[sid], store, repo, paths).items()}
        sources[sid] = {"address": source.data["address"], "revision": source.data["revision"],
                        "licenses": source.data["licenses"]}
        checksums[sid] = source.data["checksum"]
    filled = []
    for task in tasks:
        parts = {"files": dict(task.files), "tool_files": dict(task.tool_files), "check_files": dict(task.check_files),
                 "outside": dict(task.outside)}
        read = {(pointer.source, pointer.document) for pointer in task.pointers}
        documents = tuple((sid, path, hashlib.sha256(texts[sid, path]).hexdigest() if (sid, path) in texts else None)
                          for sid, path in sorted(read))
        keys = tuple(held_out_group(read_sources[sid], path) or f"{sid}/{path}" for sid, path in sorted(read))
        for pointer in task.pointers:
            where = f"{task.id}/{TASK}: the pointer to {pointer.path!r}"
            if pointer.to == "repository":
                where = f"{task.id}/{TASK}: the repository pointer"
                if pointer.checksum != checksums[pointer.source]:
                    raise InvalidFile(f"{where}: {pointer.source} now has the checksum {checksums[pointer.source]}, "
                                      f"not {pointer.checksum}, which the task was built from; build the task again")
                files, remote, expected = repository(where, pointer, trees[pointer.source], store)
                parts["files"] |= files
                parts["outside"][REMOTE] = remote
                parts["check_files"][REPOSITORY] = (FILE, json.dumps(expected, indent=1).encode() + b"\n")
                continue
            text = texts.get((pointer.source, pointer.document))
            if text is None:
                raise InvalidFile(f"{where}: {pointer.source} has no document {pointer.document!r}")
            start, end = pointer.span
            if end > len(text) or hashlib.sha256(text[start:end]).hexdigest() != pointer.sha256:
                raise InvalidFile(f"{where}: the span {[start, end]} of {pointer.source}/{pointer.document} no "
                                  f"longer holds the text it pointed to; build the task again")
            bounds = [start, *(edge for part in pointer.removed for edge in part), end]
            pieces = [text[bounds[i]:bounds[i + 1]] for i in range(0, len(bounds), 2)]
            cut = next((edge for edge in bounds if edge < len(text) and text[edge] & 0xC0 == 0x80), None)
            if cut is not None:  # a byte inside a UTF-8 character
                raise InvalidFile(f"{where}: byte {cut} cuts a character of {pointer.source}/{pointer.document}")
            parts[pointer.to][pointer.path] = (FILE, b"".join(pieces))
        filled.append(replace(task, **parts, pointers=(), documents=task.documents + documents,
                              held_out_keys=task.held_out_keys + keys))
    return filled, sources


def chain(where, pointer: Pointer, tree, store) -> list[str]:
    """The history files to undo from the source's final tree to reach
    the state just before pointer's commit, in order: every later commit's,
    newest first, in the history's order (ORDER), then the commit's own; a
    commit that changed no kept path has none (maintainer's decision,
    2026-10-22). InvalidFile when the tree has no ORDER or no history file
    of the commit, the commit is not in ORDER, or a merge sits between the
    commit and the final tree: undoing cannot reach its state."""
    for path in (ORDER, pointer.document):
        if path not in tree:
            raise InvalidFile(f"{where}: {pointer.source} has no file {path!r}")
    commit = pointer.document.removeprefix(HISTORY)
    order = _order(tree, store)
    position = next((n for n, line in enumerate(order) if line[0] == commit), None)
    if position is None:
        raise InvalidFile(f"{where}: {pointer.source}/{ORDER} does not list {commit}")
    reachable = _reachable(order)
    if position >= len(reachable):
        raise InvalidFile(f"{where}: the merge {order[len(reachable)][0]} sits between {commit} and the final tree "
                          f"of {pointer.source}")
    return [HISTORY + line[0] for line in order[:position + 1] if HISTORY + line[0] in tree]


def undoable(tree, store) -> list[str]:
    """The history files of a tree that has ORDER whose commits chain()
    can reach: those newer than the newest merge (every commit when there
    is none), in ORDER's order. A commit behind a merge cannot be reached
    by undoing, so the builder drops it before drawing (maintainer's
    decision, 2026-10-22)."""
    return [HISTORY + line[0] for line in _reachable(_order(tree, store)) if HISTORY + line[0] in tree]


def _reachable(order: list[list[str]]) -> list[list[str]]:
    """ORDER's lines newer than its newest merge (every line when there is
    none): the commits that undoing can reach. chain() and undoable() both
    go through here, so the builder draws only what chain() accepts."""
    newest = next((n for n, line in enumerate(order) if MERGE in line[1:]), len(order))
    return order[:newest]


def _order(tree, store) -> list[list[str]]:
    """ORDER's lines, each split into the commit and its marks."""
    return [line.split() for line in store.get_bytes(tree[ORDER].sha256).decode().splitlines()]


def repository(where, pointer: Pointer, tree, store) -> tuple[dict, dict, dict]:
    """({path in /work: (manifest mode, content)}, {path in REMOTE: the
    same}, what REPOSITORY holds) of a repository pointer. The source's
    final tree with the history files of chain() undone in order is the
    commit's parent (the real state just before it); the final tree with
    only the later commits undone is the commit's own tree, the expected
    tree (maintainer's decision, 2026-10-22). The parent is committed once
    on BRANCH, both in /work, the repository itself (its .git and, in its
    working tree, the commit's tree, so the changes are there and not
    committed), and in the bare repository at REMOTE, outside /work, which
    is its `origin`; the model writes its own commit message
    (maintainer's decision, 2026-10-22), so none is given or expected. The
    parent commit stands for the history the store does not keep: its
    author and date are the commit's, its message names the commit it
    comes before. InvalidFile when chain() refuses, a history file does
    not undo, or its author and date do not make a commit (an empty name or
    email); any other git failure is not the pointer's and goes through."""
    undone = chain(where, pointer, tree, store)
    patches = [store.get_bytes(tree[path].sha256) for path in undone]
    commit = pointer.document.removeprefix(HISTORY)
    work = {path: found for path, found in tree.items() if not path.startswith(".git/")}
    with tempfile.TemporaryDirectory(prefix="wpt-repository-") as tmp:
        git, env = temporary_git(tmp)
        load_index(git, work, store)
        undo = [os.path.join(tmp, f"undo-{number:08d}") for number in range(len(patches))]
        for file, patch in zip(undo, patches):
            Path(file).write_bytes(patch)
        trees = []
        for group in (undo[:-1], undo[-1:]):  # the later commits, then the commit itself
            for start in range(0, len(group), PATCHES):
                try:  # git apply undoes several files' diffs one after the other, all or none
                    git("apply", "--cached", "-R", "--whitespace=nowarn", *group[start:start + PATCHES])
                except subprocess.CalledProcessError as error:
                    raise InvalidFile(f"{where}: undoing {pointer.source}'s commits down to {commit} failed: "
                                      f"{error.stderr.decode(errors='replace').strip()}") from None
            trees.append(git("write-tree").decode().strip())
        after, before = trees
        try:  # the commit's author, date and email, read from its history file, make the parent commit
            info = dict(line.split(": ", 1) for line in git("mailinfo", os.path.join(tmp, "message"),
                                                             os.path.join(tmp, "diff"), input=patches[-1]).decode()
                        .splitlines() if ": " in line)
            who = {"GIT_AUTHOR_NAME": info.get("Author", ""), "GIT_AUTHOR_EMAIL": info.get("Email", ""),
                   "GIT_AUTHOR_DATE": info.get("Date", "")}
            who |= {key.replace("AUTHOR", "COMMITTER"): value for key, value in who.items()}
            parent = subprocess.run(["git", "commit-tree", before, "-m", f"The repository before {commit}"],
                                    env={**env, **who}, cwd=tmp, capture_output=True,
                                    check=True).stdout.decode().strip()
        except subprocess.CalledProcessError as error:
            raise InvalidFile(f"{where}: the commit before {commit} of {pointer.source} cannot be made from its "
                              f"history file: {error.stderr.decode(errors='replace').strip()}") from None
        git("update-ref", f"refs/heads/{BRANCH}", parent)
        git("repack", "-a", "-d", "-q")  # only what the parent reaches: the commit's own blobs stay loose
        working = _tree_files(git, after)
        git("read-tree", parent)
        packed = Path(env["GIT_DIR"]) / "objects" / "pack"
        objects = {f"objects/pack/{found.name}": found.read_bytes() for found in sorted(packed.iterdir())}
        index = Path(env["GIT_INDEX_FILE"]).read_bytes()
    head = f"{parent}\n".encode()
    config = (f"[core]\n\trepositoryformatversion = 0\n\tbare = false\n[remote \"origin\"]\n\turl = {REMOTE}\n"
              f"\tfetch = +refs/heads/*:refs/remotes/origin/*\n[branch \"{BRANCH}\"]\n\tremote = origin\n"
              f"\tmerge = refs/heads/{BRANCH}\n[user]\n\tname = {_quoted(who['GIT_AUTHOR_NAME'])}\n"
              f"\temail = {_quoted(who['GIT_AUTHOR_EMAIL'])}\n")
    dot = {"HEAD": f"ref: refs/heads/{BRANCH}\n".encode(), f"refs/heads/{BRANCH}": head, **objects}
    remote = {path: (FILE, data)
              for path, data in {**dot, "config": b"[core]\n\trepositoryformatversion = 0\n\tbare = true\n"}.items()}
    files = {posixpath.join(".git", path): (FILE, data)
             for path, data in {**dot, "config": config.encode(), "index": index,
                                f"refs/remotes/origin/{BRANCH}": head}.items()}
    return files | working, remote, {"remote": REMOTE, "branch": BRANCH, "parent": parent, "tree": after}


def _tree_files(git, tree) -> dict:
    """{path: (manifest mode, content)} of every file of a tree of the
    temporary repository."""
    listed = [record.split(b"\t", 1) for record in git("ls-tree", "-r", "-z", tree).split(b"\0") if record]
    entries = [(meta.decode().split(), path.decode()) for meta, path in listed]
    contents = git("cat-file", "--batch", input="".join(f"{meta[2]}\n" for meta, _ in entries).encode())
    files, at = {}, 0
    for (mode, _, object_id), path in entries:
        header_end = contents.index(b"\n", at)
        size = int(contents[at:header_end].split()[2])
        files[path] = (mode, contents[header_end + 1:header_end + 1 + size])
        at = header_end + 1 + size + 1  # the content, then a line break
    return files


def _quoted(value: str) -> str:
    """A value as git's configuration file reads it back exactly."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def measure_only(task: Task, tags, share: float, held=frozenset()) -> bool:
    """Whether only the measures read the task (maintainer's decisions,
    2026-10-18 and 2026-10-22): tagged EVAL; or in the share `share` of
    every task source that a hash of the task id holds out, the rule that
    holds documents out of training (the model recipe's held_out); or, once
    filled (fill), pointing into a document that rule holds out (by its
    held-out group when it has one: the source's held_out_by,
    sources.held_out_group; Task.held_out_keys), or one whose Markdown's
    sha256 is in held, the texts of the held-out documents
    (held_out_texts): the same text under another id."""
    from .build import is_held_out

    return EVAL in tags or is_held_out(task.id, share) \
        or any(is_held_out(key, share) for key in task.held_out_keys) \
        or any(sha256 in held for _, _, sha256 in task.documents)


def document_tags(repo: Repo, store, sid: str, document: str) -> frozenset[str]:
    """The tags of a document a task points to (Task.documents), through
    sources.tags_of: a file's, a record's file's (<file>#<record id>), and
    for a repository pointer's history file (HISTORY + commit), the tags of
    every file the commit changes (changed_paths)."""
    source = repo.source(sid)
    if not document.startswith(HISTORY):
        return tags_of(source, document.split("#", 1)[0])
    tree, _ = source_tree(source, store)
    return frozenset(tag for path in changed_paths(store.get_bytes(tree[document].sha256))
                     for tag in tags_of(source, path))


def changed_paths(history: bytes) -> list[str]:
    """The paths a history file's commit changes (what `git format-patch`
    wrote for it, manifest.read_git_history), as `git apply --numstat`
    reads them from its diff."""
    with tempfile.TemporaryDirectory(prefix="wpt-changed-") as tmp:
        git, _ = temporary_git(tmp)
        listed = git("apply", "--numstat", "-z", input=history)
    return [record.split(b"\t", 2)[2].decode("utf-8", errors="replace")
            for record in listed.split(b"\0") if record.count(b"\t") >= 2]


def held_out_texts(store, dataset: str | None) -> set[str]:
    """The sha256 of the Markdown of every held-out document of the dataset
    `dataset` (the base model's, which its provenance names), read from the
    dataset's document list (build.document_list; maintainer's decision,
    2026-10-22): no source is converted. DocumentListMissing when the
    dataset has no list, built before builds wrote one, when the store does
    not hold the dataset, or when none is named; nothing falls back and
    nothing is rebuilt (maintainer's decision, 2026-10-22)."""
    from .build import HELD_OUT, document_list

    if not dataset:
        raise DocumentListMissing("pointer tasks need the base model's dataset, to keep held-out texts out of "
                                  "training, and none is named")
    listed = document_list(store, dataset)
    if listed is None and store.get_ref("datasets", dataset) is None:
        raise DocumentListMissing(f"dataset {dataset}, the base model's, is not in this store, so its document "
                                  f"list cannot say which texts it held out, and pointer tasks need that")
    if listed is None:
        raise DocumentListMissing(f"dataset {dataset} predates document lists, so it cannot say which texts it held "
                                  f"out, and pointer tasks need that; this base model is used with the repository "
                                  f"at the commit that built it")
    return {entry["sha256"] for entry in listed if entry["tag"] == HELD_OUT}


def looks(tool: dict) -> bool:
    """Whether a tool, by its declaration, looks at sources (wpt_check's
    looking rule; maintainer's decision, 2026-10-22): read and bash
    (LOOKING), or a declared tool marked `looks`."""
    return tool["name"] in LOOKING or tool.get("looks") is True


def check_declared(where, names) -> None:
    """InvalidFile when a declared tool takes the name of a tool every task
    has, or is declared twice (a task's tools, an image's pool)."""
    for name in names:
        if name in BUILT_IN or name == ASK or names.count(name) > 1:
            raise InvalidFile(f"{where}: the tool {name!r} is declared twice, or takes the name of a tool every "
                              f"task has ({', '.join(BUILT_IN)}, {ASK})")


def _check_tools(where, needs, replies, tools, tool_files, files, answered_by, pool, made) -> set[str]:
    """The paths of files that tool_files hides; InvalidFile when the
    task's tools, needs, answered_by, replies or tool_files (a variant's
    among them, which may also name what a repository pointer makes) break
    the rules above."""
    declared = [tool["name"] for tool in tools]
    check_declared(where, declared)
    offered = {*BUILT_IN, *declared, *([ASK] if replies else [])}
    looking = {*LOOKING, *(tool["name"] for tool in [*tools, *(t for t in pool if t["name"] not in declared)]
                           if looks(tool))}
    named = [{name for variant in variants for name in variant.tools} for variants in needs]
    for number, names in enumerate(named):
        for name in sorted(names):
            if name not in offered:
                raise InvalidFile(f"{where}: needs {name!r}, which the task does not offer"
                                  + (" (ask is offered only with replies)" if name == ASK else ""))
            if any(name in other for other in named[number + 1:]):
                raise InvalidFile(f"{where}: {name!r} is in two needs; a tool is the variant of one need only")
    if replies and not answered_by:
        raise InvalidFile(f"{where}: a task with replies names in answered_by the looking tools that can answer "
                          f"before the user is asked")
    for name in answered_by:
        if name not in looking:
            raise InvalidFile(f"{where}: answered_by names {name!r}, which is no looking tool the task or its "
                              f"image's pool offers (read, bash, or a declared tool marked looks)")
    for pattern, _ in replies:
        try:
            re.compile(pattern)
        except re.error as error:
            raise InvalidFile(f"{where}: the reply pattern {pattern!r} is not a regular expression: {error}") from None
    for variants in needs:
        for variant in variants:
            _hidden(where, variant.tool_files, made)
    return _hidden(where, tool_files, files)


def _hidden(where, tool_files, files) -> set[str]:
    """The paths of files that the tool_files entries name; InvalidFile
    for an entry that names none."""
    hidden = set()
    for entry in tool_files:
        found = {path for path in files if path == entry or path.startswith(entry + "/")}
        if not found:
            raise InvalidFile(f"{where}: tool_files names {entry!r}, which is no file or folder under {FILES}")
        hidden |= found
    return hidden


def has_tasks(source) -> bool:
    return any(entry["format"] == "task" for entry in source.data["formats"])


def of_source(source, tree, store, repo: Repo) -> dict[str, Task]:
    """{task folder: Task} of a source's final tree: its paths in the task
    format; their images are folders of repo's data/images/."""
    if not has_tasks(source):
        return {}
    entries = {path: entry for path, entry in tree.items() if markdown.format_of(source, path)["format"] == "task"}
    return parse(source.id, entries, store, repo)


def pointed_sources(source, tree, store, repo: Repo) -> set[str]:
    """The ids of the sources a source's pointer tasks point into."""
    return {pointer.source for task in of_source(source, tree, store, repo).values() for pointer in task.pointers}


def read(repo: Repo, store, tag: str | None, held_out: float, log=print, measure=False,
         dataset: str | None = None) -> tuple[list[Task], dict]:
    """([Task], {source id: address, revision, licenses}) for every task of
    an admitted source whose task.json is tagged `tag` (any tag or none,
    when tag is None: the measures read every measure-only task) and that is not
    measure-only (measure_only, with held_out the model recipe's share and,
    when a task points into documents, the texts of the held-out documents
    of `dataset`, the base model's, read once for the pass: held_out_texts,
    which refuses a dataset without its document list), in source and
    folder order, its pointers filled in; the sources include those the
    kept tasks' pointers read. With measure, the measure-only ones instead.
    A source whose license text is empty is left out, and log is called
    with a note naming it."""
    found, sources = [], {}
    for source in repo.sources(log):
        if not source.admitted or not has_tasks(source):  # no tree read for a source without tasks
            continue
        tree, _ = source_tree(source, store)
        for folder, task in of_source(source, tree, store, repo).items():
            tags = tags_of(source, _prefix(folder) + TASK)
            if tag is None or tag in tags:
                found.append((task, tags, source))
    filled, pointed = fill([task for task, _, _ in found], repo, store)
    held = held_out_texts(store, dataset) if any(task.documents for task in filled) else set()
    kept = []
    for task, (_, tags, source) in zip(filled, found):
        if measure_only(task, tags, held_out, held) == measure:
            kept.append(task)
            sources[source.id] = {"address": source.data["address"], "revision": source.data["revision"],
                                  "licenses": source.data["licenses"]}
            sources |= {sid: pointed[sid] for sid, _, _ in task.documents}
    return kept, sources
