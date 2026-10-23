"""The task builder: program-built tasks, written as pointer tasks into a
task source (docs/post-training.md, "The task builder"). scripts/build_tasks.py runs it.

Each kind draws its tasks from the documents of admitted sources already
in the store, with a generator seeded by the run's seed, so the same seed,
sources and store give the same tasks; the repository holds only the
pointers (tasks.Pointer), filled in when a task is read:

- find-the-passage: quote.txt holds the start of a paragraph of one of the
  task's texts; the answer is the rest of it. Half the tasks give the texts
  as files of docs/, half as web pages that the declared search and fetch
  tools answer from (hidden files, tool_files).
- evidence-removed: the same, but in half the tasks the paragraph is removed
  from its text, and the right answer starts with "not enough" (wpt_check
  NOT_ENOUGH) and names what the question asked for: the paragraph, by the
  words of its quoted start, read from missing.txt; the other half can be
  answered.
- look-or-ask: the rest of the paragraph goes into a file whose name is
  the one missing detail: in half the tasks NOTES.md names it, and the
  attempt must have read it before any question; in the other half only
  the reply of the ask tool does. In both, the user is asked only after
  NOTES.md, where the answer lives (its answered_in), was opened through a
  looking tool that can answer (ANSWERED_BY), and, where NOTES.md names
  the file, never after a look showed that line: the answer was found.
- commit-and-push: a repository pointer to a commit of an admitted git
  source's history, drawn from every commit of every such source newer
  than its newest merge (the others are dropped before drawing) whose
  state just before it can be made, later commits undone in the
  history's order: /work is the
  repository, and the model commits the prepared changes with a message
  of its own, then pushes them, one
  way per attempt, the seeded choice of a need's variants: `git push` through bash to the
  bare `origin` outside /work (tasks.REMOTE), with no push tool offered,
  or the declared push tool, with the remote hidden so that only the
  tool's program reaches it.
- read-the-manual: the result of ondrik, a program of the agent image
  written for WPT, so that no training text describes it, and compiled,
  so that the half without the manual has no readable documentation of
  it. A task asks for the seal of a
  drawn text folded a drawn number of times, or the slack knot of two
  drawn numbers, each given by an option the instructions do not name. Half the tasks remove
  the program's manual (`absent`: its folder in the image is shown
  empty); the check runs the program in its own container and wants its
  output in the answer. Neither sources nor a pool expression are read.
  These tasks are measure-only:
  training on them would teach the program, against "no training text
  describes it". write() tags their folder `eval` (tasks.EVAL), so the
  task reader gives them to the measures only (MEASURE_ONLY).

Every kind's check prints its parts (wpt_check.report): `correct`, and
the trace rules the kind gives something to judge (RULES): sources_first
for the kinds built on text, and for look-or-ask read_before_write,
asked_last and, where NOTES.md names the file, read_before_ask, the last
two its `gates`. Each task's `area` is what its attempts may change:
nothing ([]) for the kinds that only answer, the result's file for
look-or-ask, git's own records under .git/ (REPOSITORY_AREA) for
commit-and-push, whose project files the attempt commits but never changes.

No kind's instructions state an answer block: the model's final reply is
its answer, as text, and the harness asks for its confidence after it
(docs/chat-format.md, "The answer").
Each kind acts, checks, or both (CLASSES): commit-and-push acts, find-the-passage,
evidence-removed and look-or-ask check, read-the-manual does both.
shares() gives a task source's shares of tasks that act only, check only
and do both, which add up to 1.

A text is a window of whole paragraphs of a document's Markdown, at most
WINDOW bytes, so a read tool's result (4000 bytes) holds it whole. A
paragraph a task asks for is plain text (no heading, list, quotation,
table, code or markup at its start), and its quoted start appears in no
other text of the task, letter case and marks aside. Every task's check
calls the shared check library (wpt_check, in the agent image), which
reads the attempt's answer.

A task's folder is <kind>/<the first 16 hex digits of the sha256 of its
task.json>, so a task keeps its id, and so its measure-only share, over
reruns; write() replaces everything under <kind>/ in the task source.
"""

import hashlib
import json
import math
import posixpath
import random
import re
from dataclasses import dataclass

from . import markdown, pools
from .errors import InvalidFile, RuleViolation
from .manifest import HISTORY, ORDER
from .repo import Repo
from .sources import in_store, source_tree, tags_of
from .tasks import CHECK, EVAL, FILES, LOOKING, REMOTE, TASK, Pointer, looks, repository, undoable

READ_THE_MANUAL = "read-the-manual"
KINDS = ("find-the-passage", "evidence-removed", "look-or-ask", "commit-and-push", READ_THE_MANUAL)
ACT, CHECK_KIND, BOTH = "act", "check", "both"
CLASSES = {"find-the-passage": CHECK_KIND, "evidence-removed": CHECK_KIND, "look-or-ask": CHECK_KIND,
           "commit-and-push": ACT, READ_THE_MANUAL: BOTH}  # what each kind's tasks do
MEASURE_ONLY = (READ_THE_MANUAL,)  # the kinds whose folder write() tags eval
IMAGE = "agent"
LIMITS = {"max_turns": 20, "max_tokens": 4000, "command_seconds": 30}
WINDOW = 3000  # bytes of a text at most
PARAGRAPH = (120, 800)  # bytes of a paragraph a task asks for, at least and at most
QUOTED, REST = 6, 8  # words of the quoted start and of the rest, at least
DOCUMENT = 4 << 20  # bytes of a document read at most
GIVEN, PAGES = 3, 5  # texts of a task: files in docs/, or web pages
REPOSITORY_BYTES, REPOSITORY_FILES = 4 << 20, 2000  # the largest working tree a commit-and-push task holds
TRIES = 20  # draws of a paragraph before a document is left aside
# what a commit-and-push attempt may change: git's own records; a project file it adds, removes or edits
# while committing is over-reach
REPOSITORY_AREA = ".git/**"
WORD = re.compile(rb"\w+")
MARKUP = re.compile(rb"[#>|*+`~<\[!-]|\d+[.)]\s|\\")  # what a plain paragraph does not start with
SEARCH = {"name": "search", "description": "Search the web; returns the best pages, each with its address.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
          "program": "/usr/local/lib/wpt-tools/search", "names": ["web_search"], "looks": True}
FETCH = {"name": "fetch", "description": "Fetch a web page and return its text.",
         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
         "program": "/usr/local/lib/wpt-tools/fetch", "names": ["web_fetch"], "looks": True}
PUSH = {"name": "push", "description": "Push the commits of the current branch of a git repository to its remote.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description":
                                                                 "the repository's folder, by default the working "
                                                                 "folder"}}},
        "program": "/usr/local/lib/wpt-tools/push", "names": ["git_push", "push_changes"]}
PROGRAM = "/usr/local/bin/ondrik"  # the program of the read-the-manual tasks, in the agent image
MANUAL = "/usr/share/doc/ondrik"  # the folder of its manual, which half those tasks remove
TURNS = (2, 9)  # the folds of a seal a task asks for, at least and at most
KNOTTED = (100, 99999)  # the numbers of a knot, at least and at most
# the tools that can read NOTES.md's text, kept when they look (tasks.looks) and the task or the image's pool offers them
ANSWERED_BY = ("read", "bash", "search_text")
INSTRUCTIONS = {
    "given": "quote.txt holds the start of a paragraph of one of the documents in docs/. Find that paragraph and "
             "give the rest of it, word for word, as your answer.",
    "pages": "quote.txt holds the start of a paragraph of a web page. Find the page and give the rest of the "
             "paragraph, word for word, as your answer.",
    "removed": " If the paragraph is not there, answer \"not enough: \" followed by what is missing.",
    "look-or-ask": "quote.txt holds the start of a paragraph of one of the documents in docs/. Write the rest of "
                   "that paragraph, word for word, into the file the user wants it in, and answer with a short "
                   "summary of what you did.",
    # these words only; the model writes its own commit message. No answer
    # block after them
    "commit-and-push": "Based on the current diffs, could you commit and push the changes in this repository?",
    "seal": "Use the program ondrik, installed in this container, to find the seal of the text \"{text}\" folded "
            "{turns} times, and give the seal as your answer.",
    "knot": "Use the program ondrik, installed in this container, to find the slack knot of {first} and {second}, "
            "and give the knot as your answer.",
}
# where a look-or-ask task's answer lives, its answered_in: in half the tasks it holds the file's name, and in
# the other half the attempt reads it without finding the name before it asks
NOTES_FILE = "NOTES.md"
NOTES = "# Notes\n\nThe documents are in docs/.\n"
WHERE = "Put results in {name}.\n"
ASKED = r"(?i)\b(file|name|where|save|put|write|store)"  # a question about where the result goes
REPLY = "Put it in {name}."
CHECKS = {
    "answer": "right = wpt_check.holds(wpt_check.answer_text(conversation), wpt_check.beside('rest.txt'))",
    "answerable": "right = wpt_check.holds(wpt_check.answer_text(conversation), wpt_check.beside('rest.txt')) "
                  "and not wpt_check.not_enough(conversation)",
    "removed": "right = wpt_check.not_enough(conversation, wpt_check.beside('missing.txt'), {words})",
    "file": "right = wpt_check.holds(wpt_check.file_text({name!r}), wpt_check.beside('rest.txt'))",
    "pushed": "right = wpt_check.pushed()",
    # the program's output, from the check's own container, where the model's files cannot change it
    "program": "import subprocess\n"
               "expected = subprocess.run({argv!r}, stdin=subprocess.DEVNULL, capture_output=True, "
               "text=True).stdout.strip()\n"
               "right = expected != '' and wpt_check.holds(wpt_check.answer_text(conversation), expected)",
}
# the trace rules of the check library a check reports as parts next to `correct` (#37's rules, each a named part)
RULES = {"sources_first": "wpt_check.sources_first(conversation)",
         "read_before_write": "wpt_check.read_before_write(conversation, [{name!r}])",
         "asked_last": "wpt_check.asked_last(conversation, {found!r})",
         "read_before_ask": "wpt_check.read_before_ask(conversation, {where!r})"}
CHECK_TEXT = """#!/usr/bin/python3 -I
import wpt_check
conversation = wpt_check.load()
{right}
print(wpt_check.report(conversation, right{rules}))
"""


def check_text(right: str, rules=(), **fields) -> str:
    """A task's check: right, the code that sets `right`, the task's own
    judgement, then the parts it prints (wpt_check.report): `correct`
    and each of the RULES named in rules, filled in with fields."""
    return CHECK_TEXT.format(right=right, rules="".join(f", {name}={RULES[name].format(**fields)}" for name in rules))


@dataclass(frozen=True)
class Text:
    """A window of a document's Markdown."""
    source: str
    document: str
    address: str  # the source's address
    markdown: bytes  # the whole document
    span: tuple[int, int]
    paragraphs: list  # (start, end) of each paragraph in the window


def build(repo: Repo, store, kind: str, count: int, seed: int, select: str | None = None, log=print) -> dict:
    """{task folder: (task.json's document, check text, {path under files/:
    text})} of `count` tasks of `kind` (fewer when the sources do not hold
    enough, with a note), drawn with `seed` from the admitted sources in the
    store whose documents `select` (a pool expression) selects; the
    commit-and-push kind reads the git sources with a history instead."""
    if kind not in KINDS:
        raise RuleViolation(f"no task kind {kind!r}; the kinds are {', '.join(KINDS)}")
    rng = random.Random(f"{kind}:{seed}")
    if kind == "commit-and-push":
        made = _commits(repo, store, count, rng, log)
    elif kind == READ_THE_MANUAL:
        made = [_manual(number, rng) for number in range(count)]
    else:
        if not select:
            raise RuleViolation(f"{kind}: say which sources' documents the tasks come from (a pool expression)")
        looking = {*LOOKING, *(tool["name"] for tool in repo.image_tools(IMAGE) if looks(tool))}
        answered_by = [name for name in ANSWERED_BY if name in looking]
        made = _passages(repo, store, kind, count, rng, pools.parse(select), answered_by, log)
    # two identical drawn tasks are one folder
    built = {f"{kind}/{hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]}":
             (task, check, files) for task, check, files in made}
    if len(built) < count:
        log(f"{kind}: {len(built)} tasks of the {count} asked; the sources in the store hold no more, "
            f"or drew the same task twice")
    return built


def write(repo: Repo, store, sid: str, kind: str, built: dict, version: str) -> str:
    """Write the built tasks into the task source sid, an `empty` source a
    person made: everything under <kind>/ becomes exactly these task
    folders, through one patch (Repo.change_source_files, which applies
    the rules of `check --fetch`, pointers filled in); a kind of
    MEASURE_ONLY gets its folder tagged eval in source.json first, when no
    entry tags it so. Returns the new version."""
    source = repo.source(sid)
    if source.data["type"] != "empty":
        raise RuleViolation(f"{sid} is a {source.data['type']} source: built tasks go into an empty source")
    _, existing = repo.source_files(sid, store)
    changes = {path: None for path in existing if path.startswith(kind + "/")}
    for folder, (task, check, files) in built.items():
        changes[f"{folder}/{TASK}"] = (json.dumps(task, indent=1, ensure_ascii=False) + "\n").encode()
        changes[f"{folder}/{CHECK}"] = check.encode()
        changes |= {f"{folder}/{FILES}{path}": text.encode() for path, text in files.items()}
    covered = any(entry["format"] == "task" and f"{kind}/**" in entry["paths"] for entry in source.data["formats"])
    formats = [] if covered else [{"format": "task", "paths": [f"{kind}/**"]}]
    tagged = any(EVAL in entry["tags"] and f"{kind}/**" in entry["paths"] for entry in source.data.get("tags", []))
    tags = [{"tags": [EVAL], "paths": [f"{kind}/**"]}] if kind in MEASURE_ONLY and not tagged else []
    return repo.change_source_files(sid, changes, f"build-{kind}", version, store, formats, tags)


def shares(repo: Repo, store, sid: str) -> dict:
    """What the task source sid holds of each built kind: {"tasks": {kind:
    its task folders}, "act", "check", "both": the shares of them whose
    kind acts only, checks only, or does both (CLASSES)}, each from 0 to 1
    and adding up to 1 (each 0 for a source without built tasks)."""
    _, existing = repo.source_files(sid, store)
    found = {kind: sum(1 for path in existing if path.startswith(kind + "/") and path.count("/") == 2
                       and path.endswith("/" + TASK)) for kind in KINDS}
    total = sum(found.values())
    return {"tasks": found, **{what: sum(found[kind] for kind in KINDS if CLASSES[kind] == what) / total
                               if total else 0.0 for what in (ACT, CHECK_KIND, BOTH)}}


# -- texts ----------------------------------------------------------------------

def paragraphs(text: bytes) -> list[tuple[int, int]]:
    """(start, end) of each paragraph of Markdown: runs of lines between blank lines."""
    found, start = [], None
    for line in re.finditer(rb"[^\n]*\n?", text):
        blank = not line[0].strip()
        if blank and start is not None:
            found.append((start, end))
            start = None
        elif not blank:
            start = line.start() if start is None else start
            end = line.start() + len(line[0].rstrip(b"\n"))
    if start is not None:
        found.append((start, end))
    return found


def plain(text: bytes, span) -> bool:
    """Whether a paragraph can be asked for: plain text of the right size."""
    part = text[span[0]:span[1]]
    return PARAGRAPH[0] <= len(part) <= PARAGRAPH[1] and not MARKUP.match(part) \
        and len(WORD.findall(part)) >= QUOTED + REST and b"```" not in part


def window(text: bytes, found, target: int, rng) -> tuple[int, int]:
    """(first, last) paragraph numbers of a window holding the paragraph
    `target`, grown on both sides at random while it stays within WINDOW."""
    first = last = target
    while True:
        sides = [side for side in ("before", "after") if (first if side == "before" else len(found) - 1 - last)]
        sides = [side for side in sides
                 if _size(text, found, first - (side == "before"), last + (side == "after")) <= WINDOW]
        if not sides:
            return first, last
        if rng.choice(sides) == "before":
            first -= 1
        else:
            last += 1


def _size(text, found, first, last):
    return _end(text, found[last][1]) - found[first][0]


def _end(text, end):
    """A window's end: the paragraph's, and its line break when one follows."""
    return end + 1 if text[end:end + 1] == b"\n" else end


def _text(document, first, last, found) -> Text:
    return Text(document["source"], document["path"], document["address"], document["markdown"],
                (found[first][0], _end(document["markdown"], found[last][1])), found[first:last + 1])


def quote_end(text: bytes, span) -> int | None:
    """Where the quoted start ends: a space near 40% of the paragraph, with
    at least QUOTED words before it and REST after it."""
    spaces = [m.start() for m in re.finditer(rb" ", text[span[0]:span[1]])]
    good = [span[0] + s for s in spaces if len(WORD.findall(text[span[0]:span[0] + s])) >= QUOTED
            and len(WORD.findall(text[span[0] + s:span[1]])) >= REST]
    return min(good, key=lambda at: abs(at - span[0] - 0.4 * (span[1] - span[0])), default=None)


def words(data: bytes) -> list[bytes]:
    return [word.lower() for word in WORD.findall(data)]


def occurs(needle: list, texts) -> int:
    """How many times the words of needle occur in the texts."""
    return sum(sum(found[i:i + len(needle)] == needle for i in range(len(found) - len(needle) + 1))
               for found in (words(text) for text in texts))


def _documents(repo, store, expression, wanted, rng, log):
    """{source id: [document]} of the admitted sources in the store, each
    with up to `wanted` documents drawn from those the expression selects
    (a reservoir: every document has the same odds), each holding a plain
    paragraph."""
    found, absent = {}, 0
    for source in repo.sources(log):
        if not source.admitted or source.data["type"] == "empty":
            continue
        if not in_store(source, store):
            absent += 1
            continue
        tree, _ = source_tree(source, store)
        selected = {path for path in tree if not path.startswith(HISTORY)
                    and pools.matches(expression, tags_of(source, path))}
        if not selected:
            continue
        kept, seen = [], 0
        for converted in markdown.documents(source, tree, store, repo, only=selected):
            if len(converted.markdown) > DOCUMENT or not any(plain(converted.markdown, p)
                                                             for p in paragraphs(converted.markdown)):
                continue
            seen += 1
            document = {"source": source.id, "path": converted.path, "address": source.data["address"],
                        "markdown": converted.markdown}
            if len(kept) < wanted:
                kept.append(document)
            elif (place := rng.randrange(seen)) < wanted:
                kept[place] = document
        if kept:
            found[source.id] = kept
    if absent:
        log(f"{absent} admitted sources are not in the store and gave no task (wpt-dataset check --fetch fetches "
            f"them)")
    return found


def _passages(repo, store, kind, count, rng, expression, answered_by, log) -> list:
    """The tasks of a kind built on a paragraph of a document."""
    # sources in a drawn order, the same share of tasks from each
    by_source = _documents(repo, store, expression, 4 * PAGES, rng, log)
    order = sorted(by_source)
    rng.shuffle(order)
    each = math.ceil(count / len(order)) if order else 0
    made = []
    for sid in order:
        for _ in range(each):
            if len(made) == count:
                return made
            number = len(made)
            # halves crossed: pages or files, removed or answerable, asked or written down
            pages = kind != "look-or-ask" and number // 2 % 2 == 0
            removed = kind == "evidence-removed" and number % 2 == 0
            task = _passage(kind, by_source[sid], rng, pages, removed, number % 2 == 0, answered_by)
            if task is not None:
                made.append(task)
    return made


def _passage(kind, documents, rng, pages, removed, ask, answered_by):
    """One task on a paragraph of one of the documents, its other texts
    drawn from the others (or elsewhere in the same document); None when no
    draw gives a paragraph whose start appears only once."""
    texts = PAGES if pages else GIVEN
    for _ in range(TRIES):
        document = rng.choice(documents)
        text, found = document["markdown"], paragraphs(document["markdown"])
        candidates = [n for n, p in enumerate(found) if plain(text, p) and quote_end(text, p) is not None]
        if not candidates:
            continue
        target = rng.choice(candidates)
        first, last = window(text, found, target, rng)
        if removed and first == last:
            continue
        main = _text(document, first, last, found)
        others = _others(documents, document, (first, last), texts - 1, rng)
        start, end = found[target]
        cut = quote_end(text, found[target])
        quoted = words(text[start:cut])
        if occurs(quoted, [main.markdown[main.span[0]:main.span[1]], *(t.markdown[t.span[0]:t.span[1]]
                                                                        for t in others)]) != 1:
            continue
        return _task(kind, main, others, target - first, cut, rng, pages, removed, ask, answered_by)
    return None


def _others(documents, document, taken, wanted, rng) -> list[Text]:
    """Up to `wanted` windows of other documents, or of the same document
    away from the paragraphs `taken`."""
    found = []
    pool = [d for d in documents if d is not document] or [document]
    for _ in range(4 * wanted):
        if len(found) == wanted:
            break
        other = rng.choice(pool)
        parts = paragraphs(other["markdown"])
        free = [n for n in range(len(parts)) if other is not document or not taken[0] - 1 <= n <= taken[1] + 1]
        if not free:
            continue
        first, last = window(other["markdown"], parts, rng.choice(free), rng)
        if other is document and not (last < taken[0] - 1 or first > taken[1] + 1):
            continue
        found.append(_text(other, first, last, parts))
    return found


def _pointer(to, path, text: Text, span, removed=()):
    data = text.markdown[span[0]:span[1]]
    pointer = {"for": to, "path": path, "source": text.source, "document": text.document, "span": list(span),
               "sha256": hashlib.sha256(data).hexdigest()}
    if removed:
        pointer["removed"] = [list(part) for part in removed]
    return pointer


def _names(texts: list[Text], pages) -> list[str]:
    """Each text's path under files/: docs/<document>.md, or, for a web
    page, web/<the source's address without its scheme>/<document>."""
    names = []
    for text in texts:
        if pages:
            base = re.sub(r"^[a-z]+://", "", text.address.rstrip("/")) + "/" + text.document.replace("#", "/")
        else:
            base = "docs/" + posixpath.basename(text.document).replace("#", "-").rsplit(".", 1)[0] + ".md"
        base = re.sub(r"[^A-Za-z0-9._/-]+", "-", base).strip("/")
        name, number = base, 2
        while name in names:
            name, number = f"{base}-{number}", number + 1
        names.append(name)
    return [f"web/{name}" if pages else name for name in names]


def _task(kind, main: Text, others, target, cut, rng, pages, removed, ask, answered_by):
    """(task.json, check, files) of a task on paragraph `target` of main's window."""
    texts = [main, *others]
    rng.shuffle(texts)
    names = _names(texts, pages)
    start, end = main.paragraphs[target]
    gap = ()
    if removed:  # the paragraph and the blank line after it, or before it when it ends the window
        following = main.paragraphs[target + 1][0] if target + 1 < len(main.paragraphs) else None
        gap = ((start, following),) if following is not None else ((main.paragraphs[target - 1][1], end),)
    pointers = [_pointer("files", name, text, text.span, gap if text is main else ())
                for name, text in zip(names, texts)]
    pointers += [_pointer("files", "quote.txt", main, (start, cut)),
                 _pointer("check", "missing.txt" if removed else "rest.txt", main,
                          (start, end) if removed else (cut + 1, end))]
    task = {"schema": 3, "image": IMAGE, "limits": LIMITS}
    files = {}
    if kind == "look-or-ask":
        name = f"result-{rng.getrandbits(24):06x}.txt"
        where = WHERE.format(name=name)
        task |= {"instructions": INSTRUCTIONS[kind],
                 "needs": [["read", "bash"], ["write"], *([["ask"]] if ask else [])],
                 "replies": [{"pattern": ASKED, "reply": REPLY.format(name=name)}], "answered_by": answered_by,
                 "answered_in": [NOTES_FILE]}
        files[NOTES_FILE] = NOTES + ("" if ask else where)
        # the user asked only after a look, and, where NOTES.md names the file, after reading it: gates
        gates = ["asked_last", *([] if ask else ["read_before_ask"])]
        task |= {"area": [name], "gates": gates}
        # where NOTES.md names the file, a question after the attempt saw that line no longer counts: asked_last
        check = check_text(CHECKS["file"].format(name=name), ["sources_first", "read_before_write", *gates],
                           name=name, where=where.strip(), found="" if ask else where.strip())
    else:
        task |= {"instructions": INSTRUCTIONS["pages" if pages else "given"]
                 + (INSTRUCTIONS["removed"] if kind == "evidence-removed" else ""),
                 "needs": [["read", "bash"], *([["search"], ["fetch"]] if pages else [])]}
        if pages:
            task |= {"tools": [SEARCH, FETCH], "tool_files": ["web"]}
        # the removed paragraph is named by the words of its start that quote.txt gives, counted as wpt_check counts
        words = len(re.findall(r"\w+", main.markdown[start:cut].decode("utf-8")))
        right = CHECKS["removed"].format(words=words) if removed else \
            CHECKS["answerable" if kind == "evidence-removed" else "answer"]
        task["area"] = []  # it answers only: no path of /work is its to change
        check = check_text(right, ["sources_first"])
    task["pointers"] = pointers
    return task, check, files


# -- commits --------------------------------------------------------------------

def _commits(repo, store, count, rng, log) -> list:
    """Commit-and-push tasks: every
    commit of each admitted git source's stored history newer than its
    newest merge (tasks.undoable: its history files; a merge has none) is
    a candidate, the source's working tree within REPOSITORY_BYTES and
    REPOSITORY_FILES; the commits behind that merge, which undoing cannot
    reach, are dropped before drawing, unlogged. Candidates are taken in a seeded order until `count` are
    kept; a commit whose state just before it cannot be made
    (tasks.repository: a failed undo, or git refusing the parent commit,
    turned into InvalidFile) is logged and skipped. Only a taken commit is
    made into a repository. A source whose stored history has no order
    file (ORDER), stored before histories kept one, gives none."""
    sources = [s for s in repo.sources(log) if s.admitted and s.data["type"] == "git" and s.data.get("history")]
    candidates, absent, unordered = [], 0, 0
    for source in sources:
        if not in_store(source, store):
            absent += 1
            continue
        tree, _ = source_tree(source, store)
        if ORDER not in tree:
            unordered += 1
            continue
        work = [entry for path, entry in tree.items() if not path.startswith(".git/")]
        if len(work) > REPOSITORY_FILES or sum(store.path(entry.sha256).stat().st_size for entry in work) \
                > REPOSITORY_BYTES:
            continue
        candidates += [(source, tree, path) for path in undoable(tree, store)]
    if absent:
        log(f"{absent} git sources with a history are not in the store and gave no task (wpt-dataset check --fetch "
            f"fetches them)")
    if unordered:
        log(f"{unordered} git sources with a history have no {ORDER} in the store and gave no task (their history "
            f"was stored before histories kept their order; they give tasks once fetched under a new checksum)")
    rng.shuffle(candidates)
    made = []
    for source, tree, document in candidates:
        if len(made) == count:
            break
        task = _commit(source, tree, document, store, log)
        if task is not None:
            made.append(task)
    return made


def _commit(source, tree, document, store, log):
    """(task.json, check, files) of a task on the commit of the history
    file `document` of a source's stored tree, its pointer naming the
    source's checksum; None, named in log, when tasks.repository refuses
    it (InvalidFile); any other failure stops the build."""
    pointer = {"for": "repository", "source": source.id, "document": document, "checksum": source.data["checksum"]}
    try:
        repository(source.id, Pointer("repository", "", source.id, document, None, "",
                                      checksum=pointer["checksum"]), tree, store)
    except InvalidFile as error:  # a commit that cannot be made is left out, named
        log(f"{source.id}/{document}: left out, {error}")
        return None
    return ({"schema": 3, "instructions": INSTRUCTIONS["commit-and-push"],
             # one way to push per attempt: git through bash, or the push tool with the remote hidden
             "image": IMAGE, "limits": LIMITS, "tools": [PUSH],
             "needs": [["bash", {"tools": ["bash", "push"], "tool_files": [REMOTE]}]],
             "area": [REPOSITORY_AREA], "pointers": [pointer]}, check_text(CHECKS["pushed"]), {})


# -- the program's manual -------------------------------------------------------

def _manual(number, rng):
    """(task.json, check, files) of the read-the-manual task `number`:
    the halves crossed, the manual removed in every other task, and the
    seal or the knot in every other pair. The instructions are the same
    with and without the manual."""
    if number // 2 % 2 == 0:
        text, turns = f"{rng.getrandbits(32):08x}", rng.randint(*TURNS)
        instructions = INSTRUCTIONS["seal"].format(text=text, turns=turns)
        argv = [PROGRAM, "press", "--turns", str(turns), text]
    else:
        first, second = rng.randint(*KNOTTED), rng.randint(*KNOTTED)
        instructions = INSTRUCTIONS["knot"].format(first=first, second=second)
        argv = [PROGRAM, "tie", "--give", str(first), str(second)]
    task = {"schema": 3, "instructions": instructions, "image": IMAGE, "limits": LIMITS,
            "needs": [["bash"]], "area": []}
    if number % 2 == 0:
        task["absent"] = [MANUAL]
    return task, check_text(CHECKS["program"].format(argv=argv)), {}
