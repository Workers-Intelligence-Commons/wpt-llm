"""The first agent image, data/images/agent/: its shared check library
(wpt_check.py), tested once here for every check that calls it, and the
programs that answer the tools tasks declare (search, fetch) and its pool
(tools.json). The library and the programs are read from the image's
folder, as the image holds them; the last test runs them in the image
itself, built from its lock (skipped without Docker)."""

import ast
import importlib.machinery
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from wpt_agent import AGENT, Limits, Sandbox, arranged, attempt, images, offer, score
from wpt_dataset import builder
from wpt_dataset.manifest import FILE, ORDER, Entry
from wpt_dataset.repo import Repo
from wpt_dataset.tasks import REMOTE, REPOSITORY, Pointer, Task, Variant, looks, repository
from wpt_store import open_store
from wpt_tokenizer import Tokenizer, chat

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "data" / "images" / "agent"
WEB = "https://example.org"
LAMP = b"# The lighthouse\n\nThe keeper lights the lamp at dusk and writes the hour in his log.\n"
TIDES = b"# Tides\n\nThe tide turns twice a day.\n\nA spring tide comes after a new moon.\n"
TOKENIZER = Tokenizer([])  # bytes only: any text encodes
# the declarations a task copies for the image's search and fetch (docs/post-training.md, "The agent image")
SEARCH = {"name": "search", "description": "Search the web; returns the best pages, each with its address.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
          "program": "/usr/local/lib/wpt-tools/search", "names": ["web_search"], "looks": True}
FETCH = {"name": "fetch", "description": "Fetch a web page and return its text.",
         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
         "program": "/usr/local/lib/wpt-tools/fetch", "names": ["web_fetch"], "looks": True}

# the tools that look, by the one rule (wpt_dataset.tasks.looks): the built-in ones, the emulations, the pool's
LOOKING = {tool["name"] for tool in [{"name": "read"}, {"name": "bash"}, SEARCH, FETCH,
                                     *json.loads((FOLDER / "tools.json").read_text())["tools"]] if looks(tool)}


def load(name):
    """A file of the image's folder as a module, its folder on the path as
    in the image (the programs import wpt_tools)."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    path = FOLDER / (f"{name}.py" if name.startswith("wpt_") else f"tools/{name}")
    spec = importlib.util.spec_from_loader(name, importlib.machinery.SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    writes = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # a __pycache__ in the image's folder would change the image
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = writes
    return module


def turn(*blocks) -> list[int]:
    """The tokens a model writes for an assistant message, <|im_end|> included."""
    ids = chat.encode(TOKENIZER, {"messages": [{"role": "assistant", "content": list(blocks)}]})[0]
    start = ids.index(chat.IM_START) + len(TOKENIZER.encode(b"assistant\n")) + 1
    return ids[start:ids.index(chat.IM_END) + 1]


class Scripted:
    """A model that writes the given turns, one per call, whatever it read."""

    def __init__(self, *turns):
        self.turns = list(turns)

    def __call__(self, ids, rows, stops):
        return self

    def write(self, inputs, limits):
        return {row: self.turns.pop(0)[:limit] for row, limit in limits.items()}


def call(n, name, arguments):
    return {"type": "tool_call", "id": f"c{n}", "name": name, "arguments": arguments}


def conversation(calls, tools=None, answer=None, answered_by=(), results=None):
    """An attempt's conversation: each call (name, arguments) in a turn of
    its own with its result (results, one text each, or "ok"), then in
    metadata the offered tools that look (LOOKING), as the agent loop
    writes them, the answer block's value and the task's answered_by."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Fix calc.py."}]}]
    for n, (name, arguments) in enumerate(calls):
        messages += [{"role": "assistant", "content": [{"type": "tool_call", "id": f"c{n}", "name": name,
                                                        "arguments": arguments}]},
                     {"role": "tool", "content": [{"type": "tool_result", "call": f"c{n}", "content": [
                         {"type": "text", "text": results[n] if results else "ok"}]}]}]
    names = tools or {name: name for name, _ in calls}
    return {"schema": 3, "messages": messages,
            "metadata": {"tools": names, "looking": [name for name in names.values() if name in LOOKING],
                         "answered_by": list(answered_by), "answer": answer}}


def test_a_missing_answer_block_scores_0_and_the_confidence_gets_its_brier_score():
    # every check scores a missing or malformed block 0, whatever the files
    # hold, and gives a confidence its Brier score (maintainer's decision, 2026-10-18)
    check = load("wpt_check")
    given = conversation([], answer={"answer": "Fixed the sign.", "confidence": 0.8})
    missing = conversation([])
    assert check.scored(given, 0.7) == 0.7 and check.scored(missing, 1.0) == 0.0
    assert check.brier(given, True) == pytest.approx(0.04) and check.brier(given, False) == pytest.approx(0.64)
    assert check.brier(missing, True) == 1.0


def test_the_trace_rules_read_each_tool_by_its_real_name():
    # an attempt sees its tools under varied names; the rules judge the real
    # tools, through the conversation's map of the names the model saw
    check = load("wpt_check")
    names = {"view": "read", "str_replace": "edit", "shell": "bash", "question": "ask", "web_search": "search"}
    edit = ("str_replace", {"path": "calc.py", "old_string": "+", "new_string": "-"})

    def trace(*calls):
        return conversation(calls, names)

    assert check.read_before_write(trace(("view", {"path": "/work/calc.py"}), edit))
    assert check.read_before_write(trace(("shell", {"command": "cat ./calc.py"}), edit))
    assert not check.read_before_write(trace(("shell", {"command": "cat old_calc.py"}), edit))
    assert not check.read_before_write(trace(edit))
    assert check.read_before_write(trace(("str_replace", {"path": "notes.md"})), new=["notes.md"])

    ask = ("question", {"question": "Which Python version?"})
    assert check.sources_first(trace(("web_search", {"query": "python version"}), ask, edit))
    assert not check.sources_first(trace(ask, ("web_search", {"query": "python version"})))


def test_looking_is_a_looking_tool_or_bash_running_a_reading_program():
    # maintainer's decision, 2026-10-22: every looking tool opens a source,
    # the pool's among them, and a bash call looks when its command, parsed,
    # runs one of the image's reading programs; a word that only names one
    # does not
    check = load("wpt_check")
    names = {"view": "read", "str_replace": "edit", "shell": "bash", "grep": "search_text", "question": "ask"}
    edit = ("str_replace", {"path": "calc.py", "old_string": "+", "new_string": "-"})

    def shell(command):
        return ("shell", {"command": command})

    def first(*calls):
        return check.sources_first(conversation(calls, names))

    for command in ("cat calc.py", "cd docs && /usr/bin/head -n 5 a.md", "ls | sort", "LC_ALL=C grep -rn x .",
                    "for f in *.md; do sed -n 1p \"$f\"; done", "(cd docs; tail a.md) 2>&1", "xargs -0 cat < list",
                    # a line break ends a command, as ; does; a # inside a word starts no comment
                    "cd docs\ncat a.md", "echo a#b; cat x", "echo '#'; cat x", "cd docs && \\\n  zless a.md.gz",
                    # a function's body, and the programs that run a program after their own words
                    "f() { cat x; }", "timeout 5 cat x", "timeout -s KILL 5s comm a b", "xargs -n 1 cat",
                    "env -u HOME nice -n 5 diff3 a b c"):
        assert first(shell(command), edit), command
    for command in ("echo cat calc.py", "python3 --version", "git status", "echo 'cat", "printf '%s' ls",
                    "echo done # cat calc.py", "python3 x.py\n# then cat it", "xargs -n 1 echo"):
        assert not first(shell(command), edit), command
    assert check.programs("xargs -n 1 cat") == ["cat"] and check.programs("cd a\nls # cat b") == ["cd", "ls"]
    assert first(("grep", {"pattern": "def "}), edit) and first(("view", {"path": "calc.py"}))
    assert not first(edit, ("view", {"path": "calc.py"})) and not first(shell("python3 calc.py"))
    # a tool looks only when the agent loop listed it as looking: here bash is not offered as such
    unlisted = conversation([shell("cat calc.py"), edit], names)
    unlisted["metadata"]["looking"] = ["read"]
    assert not check.sources_first(unlisted)


def test_asking_comes_after_a_look_through_a_tool_that_can_answer():
    # maintainer's decision, 2026-10-22: before a question, at least one call
    # that looks, to one of the looking tools the task names as able to
    # answer; not every one of them; a bash call counts when it looks
    check = load("wpt_check")
    names = {"view": "read", "shell": "bash", "find": "find_files", "question": "ask"}  # search_text is not offered
    ask = ("question", {"question": "Where do I put it?"})
    view, grep, run, find = ("view", {"path": "NOTES.md"}), ("shell", {"command": "grep -rn put ."}), \
        ("shell", {"command": "python3 x.py"}), ("find", {"pattern": "*.md"})

    def last(*calls, answered_by=("read", "bash", "search_text")):
        return check.asked_last(conversation(calls, names, answered_by=answered_by))

    assert last(view, ask) and last(grep, ask) and last(run, view, ask, ask) and last(view, grep)
    assert not last(ask, view) and not last(run, ask) and not last(view, ask, answered_by=("bash",))
    assert not last(find, ask)  # a look, but through a tool that cannot answer
    assert check.asked_last(conversation([ask], names))  # a task that names none


def test_a_question_comes_after_a_look_that_showed_the_notes():
    # maintainer's decision, 2026-10-22: where NOTES.md holds the detail,
    # the attempt reads it, through any looking tool, before any question
    check = load("wpt_check")
    names = {"view": "read", "shell": "bash", "grep": "search_text", "question": "ask"}
    where = "Put results in result-0a1b2c.txt."
    ask = ("question", {"question": "Where do I put it?"})
    notes = f"# Notes\n\nThe documents are in docs/.\n{where}\n"

    def seen(calls, results):
        return check.read_before_ask(conversation(calls, names, results=results), where)

    assert seen([("view", {"path": "NOTES.md"}), ask], [notes, "Put it there."])
    assert seen([("grep", {"pattern": "Put"}), ask], [f"NOTES.md:4:{where}", "Put it there."])
    assert seen([("shell", {"command": "cat NOTES.md"}), ask], [notes, "Put it there."])
    assert seen([("shell", {"command": "ls"})], ["NOTES.md\ndocs/\n"])  # nothing asked
    assert not seen([("shell", {"command": "ls"}), ask], ["NOTES.md\ndocs/\n", "Put it there."])
    assert not seen([ask, ("view", {"path": "NOTES.md"})], ["Put it there.", notes])
    assert not seen([("shell", {"command": f"echo '{where}'"}), ask], [where, "Put it there."])


REST = "and wrote the hour, the wind and the state of the sea in his log."
GIT = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}  # no setting of this machine's
COMMIT = "Count the lamps\n\nThe keeper counts them at dusk, every night.\n"


def checked(tmp_path, right, conversation, beside, **fields):
    """The score a check that the task builder writes prints (its fields
    filled in), run as in the image (the library on its path) from a folder
    holding conversation.json and the files a task's pointers put next to
    it."""
    folder = Path(tempfile.mkdtemp(dir=tmp_path))
    (folder / "check").write_text(builder.CHECK_TEXT.format(right=builder.CHECKS[right].format(**fields)))
    (folder / "conversation.json").write_text(json.dumps(conversation))
    for name, text in beside.items():
        (folder / name).write_text(text)
    ran = subprocess.run([sys.executable, "-P", "-B", str(folder / "check")], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(FOLDER)}, check=True)
    return float(ran.stdout)


def test_a_built_check_scores_the_rest_of_the_paragraph_or_not_enough(tmp_path):
    # the task builder's checks: the rest of the paragraph, word for word
    # but letter case and marks aside, scores 1; a missing passage wants
    # "not enough" naming the paragraph asked for by its quoted start
    # (maintainer's decision, 2026-10-22), and an answerable one refuses it;
    # no answer block, 0
    def said(text):
        return conversation([], answer={"answer": text, "confidence": 0.5})

    rest = {"rest.txt": "and wrote the hour, the *wind* and the state of the sea in his log.\n"}
    assert checked(tmp_path, "answer", said(f"The rest: {REST.upper()}"), rest) == 1.0
    assert checked(tmp_path, "answer", said("and wrote the hour in his log."), rest) == 0.0
    assert checked(tmp_path, "answer", conversation([]), rest) == 0.0
    assert checked(tmp_path, "answerable", said(f"Not enough: {REST}"), rest) == 0.0
    missing = {"missing.txt": f"The keeper climbed the stairs at dusk, {REST}\n"}
    quoted = "The keeper climbed the stairs at dusk,"
    assert checked(tmp_path, "removed", said(f"Not enough: no text holds the paragraph that starts "
                                             f"\"{quoted.lower()}\"."), missing, words=7) == 1.0
    assert checked(tmp_path, "removed", said("Not enough: the paragraph after the quote is not in docs/."), missing,
                   words=7) == 0.0
    assert checked(tmp_path, "removed", said(REST), missing, words=7) == 0.0
    check = load("wpt_check")
    (tmp_path / "out.txt").write_text(REST)
    os.mkfifo(tmp_path / "pipe")
    assert check.file_text(str(tmp_path / "out.txt")) == REST
    assert check.file_text(str(tmp_path / "pipe")) == "" and check.file_text(str(tmp_path / "gone.txt")) == ""
    assert not check.holds("anything", "...")


def prepared(tmp_path):
    """The files in /work of a commit-and-push task, those of its remote,
    and what REPOSITORY next to its check holds, made by the task reader
    from a two-commit history (the next commit's history file as
    format-patch writes it, and the history's order)."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args, cwd=upstream):
        return subprocess.run(["git", *args], cwd=cwd, env={**GIT, "GIT_AUTHOR_NAME": "A. Keeper",
                                                             "GIT_AUTHOR_EMAIL": "keeper@example.org",
                                                             "GIT_COMMITTER_NAME": "A. Keeper",
                                                             "GIT_COMMITTER_EMAIL": "keeper@example.org"},
                              capture_output=True, check=True).stdout

    git("init", "-q")
    (upstream / "lamps.py").write_text("LAMPS = 1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "start")
    start = git("rev-parse", "HEAD").decode().strip()
    (upstream / "lamps.py").write_text("LAMPS = 2\n")
    (upstream / "count.md").write_text("# Count\n")
    git("add", "-A")
    git("commit", "-q", "-m", COMMIT.strip())
    head = git("rev-parse", "HEAD").decode().strip()
    patch = git("format-patch", "-1", "--stdout", head)
    store = open_store((tmp_path / "store").as_uri())
    tree = {path: Entry(FILE, store.put_bytes((upstream / path).read_bytes())) for path in ("lamps.py", "count.md")}
    tree[f".git/history/{head}"] = Entry(FILE, store.put_bytes(patch))
    tree[ORDER] = Entry(FILE, store.put_bytes(f"{head}\n{start}\n".encode()))
    pointer = Pointer("repository", "", "example-lamps-0123abcd", f".git/history/{head}", None, "",
                      checksum="sha256:" + "0" * 64)
    return repository("test", pointer, tree, store)


def unpacked(files, work):
    for path, (_, content) in files.items():
        (work / path).parent.mkdir(parents=True, exist_ok=True)
        (work / path).write_bytes(content)


def test_pushed_wants_one_new_commit_with_the_next_commit_s_tree_and_any_message(tmp_path):
    # the commit-and-push check reads the remote: exactly one new commit on
    # its branch, on top of the commit it started from, its tree the next
    # commit's; the model writes its own message, which is not compared but
    # must not be empty (maintainer's decision, 2026-10-22)
    check = load("wpt_check")
    files, remote, expected = prepared(tmp_path)

    def attempt(name, *commands):
        work, origin = tmp_path / name, tmp_path / f"{name}-origin.git"
        unpacked(files, work)
        unpacked(remote, origin)
        # the remote at REMOTE in a container; in this folder here
        subprocess.run(["git", "remote", "set-url", "origin", str(origin)], cwd=work, env=GIT, check=True)
        for command in commands:
            subprocess.run(command, shell=True, cwd=work, env=GIT, capture_output=True, check=True)
        return check.pushed({**expected, "remote": str(origin)})

    assert attempt("right", "git add -A", "git commit -q -m 'Count lamps'", "git push -q")
    assert not attempt("unpushed", "git add -A", "git commit -q -m 'Count lamps'")
    assert not attempt("empty", "git add -A", "git commit -q --allow-empty-message -m ''", "git push -q")
    assert not attempt("part", "git add lamps.py", "git commit -q -m 'Count lamps'", "git push -q")
    assert not attempt("two", "git add lamps.py", "git commit -q -m one", "git add -A",
                       "git commit -q -m two", "git push -q")


def test_search_ranks_the_pages_by_the_query_and_fetch_gives_a_page_s_text(tmp_path):
    # the emulations give a real tool's kind of answer: ranked snippets with
    # their addresses for a search, the page's text for a fetch, 404 for a
    # page that is not there
    for path, text in (("example.org/lighthouse", LAMP), ("example.org/tides/index", TIDES)):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(text)
    search, fetch = load("search"), load("fetch")
    found = search.search("when is the lamp lit at dusk", search.pages(str(tmp_path)))
    assert found[0] == (f"{WEB}/lighthouse", "The lighthouse",
                        "The keeper lights the lamp at dusk and writes the hour in his log.")
    assert [address for address, _, _ in found] == [f"{WEB}/lighthouse", f"{WEB}/tides/index"]  # "the" alone
    assert [address for address, _, _ in search.search("the tide after a moon", search.pages(str(tmp_path)))] == \
           [f"{WEB}/tides/index", f"{WEB}/lighthouse"]
    assert fetch.page(f"{WEB}/lighthouse#log", str(tmp_path)) == LAMP.decode()
    assert fetch.page("http://example.org/tides/", str(tmp_path)) == TIDES.decode()
    assert fetch.page(f"{WEB}/../../etc/passwd", str(tmp_path)) is None
    assert fetch.page(f"{WEB}/gone", str(tmp_path)) is None


def test_the_pool_s_tools_answer_as_the_tools_they_stand_for(tmp_path):
    # padding tools really work (maintainer's decision, 2026-10-22): a
    # listing, a glob, a grep and a calculator, links never followed
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("import os\nx = 1\n")
    (tmp_path / "src" / "deep" / "b.py").write_text("y = 2\nx = 3\n")
    (tmp_path / "notes.md").write_text("x marks it\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\xff x = 9\n")
    (tmp_path / "loop").symlink_to(tmp_path)

    def at(path="."):
        return os.open(tmp_path / path, os.O_RDONLY)

    assert load("list_files").listing(at()) == ["image.png", "loop@", "notes.md", "src/"]
    assert load("find_files").found("**/*.py", at()) == ["src/a.py", "src/deep/b.py"]
    assert load("find_files").found("src/*", at()) == ["src/a.py", "src/deep"]
    grep = load("search_text")
    assert grep.lines(re.compile(r"^x = "), ".", at(), True) == ["src/a.py:2:x = 1", "src/deep/b.py:2:x = 3"]
    assert grep.lines(re.compile("y"), "src/deep/b.py", at("src/deep/b.py"), False) == ["src/deep/b.py:1:y = 2"]
    calculator = load("calculator")
    assert calculator.value(ast.parse("(3 + 4) * sqrt(4) - 2 ** 3", mode="eval")) == 6.0
    with pytest.raises(ValueError):
        calculator.value(ast.parse("__import__('os')", mode="eval"))


def test_a_pool_tool_never_reaches_outside_work(tmp_path):
    # the pool's programs read the hidden files' group: a path or a link
    # leading out of /work must fail, or the model would read through them
    tools = load("wpt_tools")
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "hidden").write_text("secret\n")
    (work / "link").symlink_to(tmp_path / "hidden")
    (work / "folder").symlink_to(tmp_path)
    with open(tools.in_work("src/a.py", root=str(work)), closefd=True) as file:
        assert file.read() == "x = 1\n"
    for path in ("../hidden", "link", "folder/hidden", str(tmp_path / "hidden")):
        with pytest.raises(SystemExit):
            tools.in_work(path, root=str(work))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a folder whose mode is 0")
def test_a_pool_tool_names_the_real_cause_of_a_refusal(tmp_path, capsys):
    tools = load("wpt_tools")
    work = tmp_path / "work"
    (work / "shut").mkdir(parents=True)
    (tmp_path / "hidden").write_text("secret\n")
    (work / "link").symlink_to(tmp_path / "hidden")
    (work / "shut").chmod(0)
    try:
        for path, said in (("shut/a", "permission denied"), ("link", "symbolic link")):
            with pytest.raises(SystemExit):
                tools.in_work(path, root=str(work))
            assert said in capsys.readouterr().err
    finally:
        (work / "shut").chmod(0o755)


@pytest.fixture(scope="module")
def agent(docker_runs, tmp_path_factory):
    """(a repository holding a copy of the agent image's folder, the image
    built from it and its lock, at once when this machine has it): a Repo
    of the real repository would make its audit hook refuse pytest's own
    clean-up of folders named data."""
    root = tmp_path_factory.mktemp("repository")
    shutil.copytree(FOLDER, root / "data" / "images" / "agent", ignore=shutil.ignore_patterns("__pycache__"))
    return root, images.build(root, "agent", log=lambda _: None)


@pytest.mark.image_build
@pytest.mark.xdist_group("agent-image")  # one worker builds the image: two would share its snapshot layer
def test_the_reading_programs_are_programs_the_agent_image_holds(agent):
    # wpt_check.READERS names the image's reading programs (maintainer's
    # decision, 2026-10-22): each of them is on the image's path
    _, image = agent
    readers = sorted(load("wpt_check").READERS)
    listed = subprocess.run(["docker", "run", "--rm", "--network", "none", image, "bash", "-c",
                             'for name in "$@"; do command -v "$name" >/dev/null && echo "$name"; done', "bash",
                             *readers], capture_output=True, text=True, check=True).stdout.split()
    assert listed == readers


@pytest.mark.image_build
@pytest.mark.xdist_group("agent-image")  # one worker builds the image: two would share its snapshot layer
def test_in_the_agent_image_the_tools_answer_and_the_check_library_scores(agent):
    # in the image itself: search and fetch answer from the task's hidden
    # pages, a pool tool works in /work without reaching them, and a check
    # started as python3 -I imports the library, reads the conversation and
    # the file a pointer put next to it
    root, image = agent
    pool = Repo(root).image_tools("agent")
    hidden = {"web/example.org/lighthouse": (FILE, LAMP)}
    made = Task("example/lamp", "When is the lamp lit?", "agent", 8, 10_000, 10, {"notes.txt": (FILE, b"start\n")},
                b"#!/usr/bin/python3 -I\nimport os, sys, wpt_check\nc = wpt_check.load()\n"
                b"expected = open(os.path.join(os.path.dirname(sys.argv[0]), 'expected.txt')).read()\n"
                b"right = expected in wpt_check.answer(c)['answer']\n"
                b"print(wpt_check.scored(c, float(right)))\n",
                needs=((Variant(("search",)),), (Variant(("fetch",)),), (Variant(("bash",)),)), tools=(SEARCH, FETCH),
                tool_files=hidden)
    chosen = next(chosen for chosen in (offer(made, pool, random.Random(seed)) for seed in range(100))
                  if "list_files" in chosen.names.values())
    listing = next(shown for shown, actual in chosen.names.items() if actual == "list_files")
    turns = [turn(call(1, "search", {"query": "lamp dusk"}), call(2, "fetch", {"url": f"{WEB}/lighthouse"}),
                  call(3, listing, {"path": "/tmp/wpt-tools"}), call(4, listing, {}),
                  call(5, "bash", {"command": "cat /tmp/wpt-tools/web/example.org/lighthouse"})),
             turn({"type": "json", "value": {"answer": "The keeper lights it at dusk.", "confidence": 0.9}})]
    limits = Limits(memory=256, disk=16, processes=64, cpus=1.0)
    with Sandbox(image, made.files, user=AGENT, limits=limits, tool_files=made.tool_files) as box:
        done = attempt(made, box, TOKENIZER, Scripted(*turns), result_bytes=4000, offer=chosen)
        assert score(box, made.check, 30, done.conversation, {"expected.txt": (FILE, b"at dusk")}) == \
               (1.0, "")
    searched, fetched, outside, inside, seen = [block["content"][0]["text"] for m in done.conversation["messages"]
                                                if m["role"] == "tool" for block in m["content"]]
    assert searched == f"1. The lighthouse\n   {WEB}/lighthouse\n   {LAMP.decode().splitlines()[2]}\n"
    assert fetched == LAMP.decode()
    assert outside == "/tmp/wpt-tools: outside /work" and inside == "notes.txt\n"
    assert "Permission denied" in seen and "keeper" not in seen


@pytest.mark.image_build
@pytest.mark.xdist_group("agent-image")  # one worker builds the image: two would share its snapshot layer
def test_in_the_agent_image_each_attempt_pushes_one_way_and_the_check_reads_the_remote(agent):
    # maintainer's decision, 2026-10-22: the seeded choice of a variant
    # gives each commit-and-push attempt one way to push. With git, no push
    # tool is offered and git push through bash reaches the remote; with the
    # push tool, the remote is hidden, git push through bash cannot reach it
    # and the tool's program does. /work is the repository and the remote is
    # outside it, so `git add -A` there never takes the remote. Either way
    # the check, as root, reads the remote the attempt pushed to at its place
    root, image = agent
    files, remote, expected = prepared(Path(tempfile.mkdtemp()))
    check = builder.CHECK_TEXT.format(right=builder.CHECKS["pushed"]).encode()
    beside = {REPOSITORY: (FILE, json.dumps(expected).encode())}
    made = Task("example/lamps", "Commit and push.", "agent", 8, 10_000, 10, files, check,
                needs=((Variant(("bash",)), Variant(("bash", "push"), (REMOTE,))),), tools=(builder.PUSH,),
                outside={REMOTE: remote})
    limits = Limits(memory=256, disk=16, processes=64, cpus=1.0)
    offers = [offer(made, [], random.Random(seed)) for seed in range(20)]
    with_git = next(chosen for chosen in offers if "push" not in chosen.names.values())
    with_tool = next(chosen for chosen in offers if "push" in chosen.names.values())
    assert with_git.hidden == () and with_tool.hidden == (REMOTE,)
    commit = call(1, "bash", {"command": "git status --porcelain && git add -A && git commit -q -m 'Count the lamps'"})
    by_git = call(2, "bash", {"command": "git push -q"})
    tool = next(shown for shown, actual in with_tool.names.items() if actual == "push")
    by_tool = call(2, tool, {})
    for chosen, push, scored in ((with_git, by_git, 1.0), (with_tool, by_git, 0.0), (with_tool, by_tool, 1.0)):
        turns = [turn(commit, push), turn({"type": "json", "value": {"answer": "Committed, pushed.", "confidence": 1}})]
        work, hidden, outside = arranged(made, chosen)
        with Sandbox(image, work, user=AGENT, limits=limits, tool_files=hidden, outside=outside) as box:
            done = attempt(made, box, TOKENIZER, Scripted(*turns), result_bytes=4000, offer=chosen)
            assert score(box, made.check, 30, done.conversation, beside, chosen.hidden, tuple(outside)) == \
                (scored, "")
        status, pushing = (block["content"][0]["text"] for block in done.conversation["messages"][2]["content"])
        assert status.splitlines() == [" M lamps.py", "?? count.md"]
        assert ("does not appear to be a git repository" in pushing) == (not scored)
