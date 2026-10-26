import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from wpt_dataset import tasks
from wpt_dataset.build import Document, is_held_out, write_document_list
from wpt_dataset.cli import main
from wpt_dataset.errors import DocumentListMissing, InvalidFile
from wpt_dataset.manifest import EXECUTABLE, FILE
from wpt_dataset.markdown import documents
from wpt_dataset.sources import source_tree
from wpt_telemetry import Telemetry

TASK = {"instructions": "Write hello into out.txt.", "image": "tiny",
        "limits": {"max_turns": 4, "max_tokens": 200, "command_seconds": 5}, "needs": [["write", "bash"]]}
SEARCH = {"name": "search", "description": "Search the notes.", "program": "/usr/local/bin/search",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
          "looks": True}
CHECK = b"#!/bin/sh\ntest -f out.txt && echo 1 || echo 0\n"
FORMATS = [{"format": "task", "paths": ["tasks/**"]}, {"format": "text", "paths": ["notes.txt"]}]


def add(kit, repo, store, folder, files, tags):
    (repo.root / "data" / "images" / "tiny").mkdir(parents=True, exist_ok=True)
    (repo.root / "data" / "images" / "tiny" / "Dockerfile").write_text("ARG BASE=debian:trixie-slim\nFROM ${BASE}\n")
    address, commit = kit.git_project(folder, files=files, executable={p for p in files if p.endswith("/check")},
                                      symlinks={})
    sid = kit.add_source(repo, store, "example", folder.name, {"type": "git", "address": address, "revision": commit,
                                                               "formats": FORMATS, "tags": tags})
    return repo.source(sid)


def test_tagged_task_folders_reach_their_step_and_never_pretraining(kit, repo, store, tmp_path):
    # a step reads exactly the tasks tagged with its name, each with its
    # instructions, limits, starting files, area, gates and check, and no
    # task file may become a pre-training document: the model would see the
    # checks; a task.json of shape 1, a source's bytes that migrate never
    # rewrites, is read in the current shape
    task = json.dumps({**TASK}).encode()
    hello = {**TASK, "container": {"memory": 8192, "cpus": 4}, "area": ["out.txt"], "gates": ["sources_first"]}
    files = {"tasks/hello/task.json": json.dumps(hello).encode(), "tasks/hello/check": CHECK, "tasks/hello/files/README": b"start\n",
             "tasks/hello/files/sub/task.json": b"a starting file, not a task\n",
             "tasks/other/task.json": task, "tasks/other/check": CHECK, "notes.txt": b"words\n"}
    source = add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["tasks/hello/**"]},
                                                              {"tags": ["en"], "paths": ["notes.txt"]}])
    tree, _ = source_tree(source, store)
    assert [c.path for c in documents(source, tree, store, repo)] == ["notes.txt"]

    found, sources = tasks.read(repo, store, "finish-tasks", 0)
    assert found == [tasks.Task(f"{source.id}/tasks/hello", "Write hello into out.txt.", "tiny", 4, 200, 5,
                                {"README": (FILE, b"start\n"),
                                 "sub/task.json": (FILE, b"a starting file, not a task\n")}, CHECK,
                                {"memory": 8192, "disk": 1024, "processes": 256, "cpus": 4},
                                ((tasks.Variant(("write",)), tasks.Variant(("bash",))),), area=("out.txt",),
                                gates=("sources_first",), tags=frozenset({"finish-tasks"}))]
    assert list(sources) == [source.id] and sources[source.id]["revision"] == source.data["revision"]
    assert tasks.read(repo, store, "answer-as-assistant", 0) == ([], {})
    other = tasks.of_source(source, tree, store, repo)["tasks/other"]
    assert other.files == {} and (other.area, other.gates) == (None, ())
    assert tree["tasks/hello/check"].mode == EXECUTABLE


def test_a_task_s_tools_replies_and_hidden_files_reach_its_step(kit, repo, store, tmp_path):
    # the tools a task declares, its needs (a variant may be tools offered
    # together that hide files), the user's replies and the looking tools
    # that can answer before the user is asked, with where its answers
    # live, come as written; the files
    # tool_files names leave /work for the tools' programs
    variant = {"tools": ["edit", "bash"], "tool_files": ["corpus.txt"]}
    document = {**TASK, "tools": [SEARCH], "needs": [["search"], ["write", variant]],
                "replies": [{"pattern": "(?i)colou?r", "reply": "Blue."}], "answered_by": ["read", "search"],
                "answered_in": ["README", "notes/*.txt"], "tool_files": ["corpus"]}
    files = {"tasks/a/task.json": json.dumps(document).encode(), "tasks/a/check": CHECK,
             "tasks/a/files/README": b"start\n", "tasks/a/files/corpus/one.txt": b"one\n",
             "tasks/a/files/corpus.txt": b"seen by the model\n"}
    add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    [found], _ = tasks.read(repo, store, "finish-tasks", 0)
    assert (found.tools, found.replies, found.answered_by, found.answered_in) == (
        (SEARCH,), (("(?i)colou?r", "Blue."),), ("read", "search"), ("README", "notes/*.txt"))
    assert found.needs == ((tasks.Variant(("search",)),),
                           (tasks.Variant(("write",)), tasks.Variant(("edit", "bash"), ("corpus.txt",))))
    assert found.files == {"README": (FILE, b"start\n"), "corpus.txt": (FILE, b"seen by the model\n")}
    assert found.tool_files == {"corpus/one.txt": (FILE, b"one\n")}


def listed(store, documents=(), share=0.0, name="d" * 64) -> str:
    """The id of a dataset whose build kept the document list of these
    (source, path, Markdown) documents, held out by share as a build holds
    them out."""
    listing = [Document(f"{sid}/{path}", sid, hashlib.sha256(text).hexdigest(), frozenset())
               for sid, path, text in documents]
    write_document_list(store, name, listing, {d.id for d in listing if is_held_out(d.id, share)}, Telemetry.off())
    return name


BOOK = b"The lamp was lit at dusk. The keeper wrote the hour in his log. Then he slept.\n"


def pointer(source, text, path, start, end, removed=(), to="files"):
    """A pointer to text[start:end] of the source's notes.txt, its sha256 included."""
    return {"for": to, "path": path, "source": source.id, "document": "notes.txt", "span": [start, end],
            "sha256": hashlib.sha256(text[start:end]).hexdigest(), "removed": [list(part) for part in removed]}


def test_a_task_holds_its_request_in_full_and_in_one_line(kit, repo, store, tmp_path):
    # the measure of terse against full requests opens the same task with
    # either form; a task without the
    # one-line form keeps "" for it
    files = {"tasks/a/task.json": json.dumps({**TASK, "terse": "Say hello."}).encode(), "tasks/a/check": CHECK,
             "tasks/b/task.json": json.dumps(TASK).encode(), "tasks/b/check": CHECK}
    add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    found, _ = tasks.read(repo, store, "finish-tasks", 0)
    assert [(task.instructions, task.terse) for task in found] == [("Write hello into out.txt.", "Say hello."),
                                                                   ("Write hello into out.txt.", "")]


def test_a_pointer_task_gets_its_texts_from_the_document_it_points_into(kit, repo, store, tmp_path):
    # the repository holds only the pointers; reading the task fills in each
    # text from the store, the parts the builder removed left out, and puts
    # it in /work, among the hidden files or next to the check, as each says;
    # the pointed source is recorded with the task's for the model's NOTICE
    book = add(kit, repo, store, tmp_path / "book", {"notes.txt": BOOK}, [])
    tree, _ = source_tree(book, store)
    [text] = [c.markdown for c in documents(book, tree, store, repo)]
    second = text.index(b"The keeper")
    third = text.index(b"Then")
    document = {**TASK, "tool_files": ["web"], "pointers": [
        pointer(book, text, "book.txt", 0, len(text), [(second, third)]),
        pointer(book, text, "web/example.org/lamp", 0, second),
        pointer(book, text, "missing.txt", second, third, to="check")]}
    task = add(kit, repo, store, tmp_path / "work", {"tasks/a/task.json": json.dumps(document).encode(),
                                                     "tasks/a/check": CHECK}, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    [found], sources = tasks.read(repo, store, "finish-tasks", 0, dataset=listed(store))
    assert found.files == {"book.txt": (FILE, text[:second] + text[third:])}
    assert found.tool_files == {"web/example.org/lamp": (FILE, b"The lamp was lit at dusk. ")}
    assert found.check_files == {"missing.txt": (FILE, b"The keeper wrote the hour in his log. ")}
    assert found.pointers == () and set(sources) == {task.id, book.id}


def test_a_pointer_whose_span_changed_or_whose_source_is_not_admitted_stops_the_fetched_check(kit, repo, store,
                                                                                              tmp_path):
    # a task whose source moved on would hand the model another text than
    # its check expects: `check --fetch` refuses it until it is built again
    book = add(kit, repo, store, tmp_path / "book", {"notes.txt": BOOK}, [])
    tree, _ = source_tree(book, store)
    [text] = [c.markdown for c in documents(book, tree, store, repo)]
    stale = {**pointer(book, text, "book.txt", 0, 10), "sha256": hashlib.sha256(b"other text").hexdigest()}
    task = add(kit, repo, store, tmp_path / "work",
               {"tasks/a/task.json": json.dumps({**TASK, "pointers": [stale]}).encode(), "tasks/a/check": CHECK},
               [{"tags": ["finish-tasks"], "paths": ["**"]}])
    with pytest.raises(InvalidFile, match=rf"{task.id}/tasks/a/task.json: the pointer to 'book.txt': the span "
                                          rf"\[0, 10\] of {book.id}/notes.txt no longer holds the text it pointed to"):
        repo.check_fetched(task.id, store)

    path = repo.root / "data" / "sources" / book.id / "source.json"
    path.write_text(path.read_text().replace('"admitted"', '"refused", "reason": "no license"'))
    with pytest.raises(InvalidFile, match=f"{book.id}: tasks point into it, but it is not admitted"):
        repo.check_fetched(task.id, store)


def test_a_changed_source_checks_the_task_sources_that_point_into_it(kit, repo, store, tmp_path, monkeypatch,
                                                                     capsys):
    # CI fetches only the changed sources; the pointer tasks built on one
    # must be checked with it, the others need not be
    book = add(kit, repo, store, tmp_path / "book", {"notes.txt": BOOK}, [])
    tree, _ = source_tree(book, store)
    [text] = [c.markdown for c in documents(book, tree, store, repo)]
    task = add(kit, repo, store, tmp_path / "work",
               {"tasks/a/task.json": json.dumps({**TASK, "pointers": [pointer(book, text, "a.txt", 0, 8)]}).encode(),
                "tasks/a/check": CHECK}, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    other = add(kit, repo, store, tmp_path / "other",
                {"tasks/b/task.json": json.dumps(TASK).encode(), "tasks/b/check": CHECK}, [])
    kit.run("git", "init", "-q", cwd=repo.root)
    kit.run("git", "add", "-A", cwd=repo.root)
    kit.run("git", "commit", "-q", "-m", "sources", cwd=repo.root)
    repo.save("source", book.id, {**book.data, "note": "changed"}, repo.version("source", book.id))
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    assert main(["--root", str(repo.root), "check", "--changed", "HEAD"]) == 0
    out = capsys.readouterr().out
    assert f"{book.id}: " in out and f"{task.id}: " in out and f"{other.id}: " not in out


def test_a_deleted_source_gets_the_task_sources_that_pointed_into_it_checked(
        kit, repo, store, tmp_path, monkeypatch, capsys):
    # a pointer task whose source was deleted is broken although nothing
    # else changed: `check --changed` must name it
    book = add(kit, repo, store, tmp_path / "book", {"notes.txt": BOOK}, [])
    tree, _ = source_tree(book, store)
    [text] = [c.markdown for c in documents(book, tree, store, repo)]
    task = add(kit, repo, store, tmp_path / "work",
               {"tasks/a/task.json": json.dumps({**TASK, "pointers": [pointer(book, text, "a.txt", 0, 8)]}).encode(),
                "tasks/a/check": CHECK}, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    kit.run("git", "init", "-q", cwd=repo.root)
    kit.run("git", "add", "-A", cwd=repo.root)
    kit.run("git", "commit", "-q", "-m", "sources", cwd=repo.root)
    shutil.rmtree(repo.root / "data" / "sources" / book.id)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    assert main(["--root", str(repo.root), "check", "--changed", "HEAD"]) == 1
    captured = capsys.readouterr()
    assert f"{task.id}: FAILED" in captured.err and "which does not exist" in captured.err


def test_a_pointer_with_an_empty_span_is_refused_as_such_even_with_removed_parts(kit, repo, store, tmp_path):
    book = add(kit, repo, store, tmp_path / "book", {"notes.txt": BOOK}, [])
    empty = {**pointer(book, BOOK, "a.txt", 5, 5), "removed": [[1, 2]]}
    task = add(kit, repo, store, tmp_path / "work",
               {"tasks/a/task.json": json.dumps({**TASK, "pointers": [empty]}).encode(), "tasks/a/check": CHECK},
               [{"tags": ["finish-tasks"], "paths": ["**"]}])
    with pytest.raises(InvalidFile, match=r"the pointer to 'a.txt' has the empty span \[5, 5\]"):
        repo.check_fetched(task.id, store)


def test_the_image_tool_pool_declares_its_tools_as_a_task_does():
    # image-tools.schema.json copies task.schema.json's tool declaration:
    # one gaining a field the other refuses would break pool tools
    folder = Path(tasks.__file__).parent / "schemas"
    pool, task = (json.loads((folder / f"{name}.schema.json").read_text()) for name in ("image-tools", "task"))
    assert {key: pool["$defs"][key] for key in ("name", "tool")} == \
           {key: task["$defs"][key] for key in ("name", "tool")}


def test_a_task_kept_in_a_dataset_reads_back_as_the_same_task():
    # a dataset keeps its tasks as JSON lines, which the training steps read
    # back: every field comes back as it was, the bytes of every kind of file
    # and every need's variants included
    files = {"notes.txt": ("100644", b"start\n"), "run": ("100755", b"#!/bin/sh\n\xff\n")}
    task = tasks.Task(
        "example-tasks/hello", "Write hello into out.txt.", "tiny", 4, 200, 5.0, files, CHECK,
        container={**tasks.SIZES, "memory": 2048},
        needs=((tasks.Variant(("write",)), tasks.Variant(("bash",), ("bin",))),),
        replies=(("which file", "out.txt"),), tools=(SEARCH,), tool_files={"index": ("100644", b"hello\n")},
        outside={"/tmp/origin.git": {"HEAD": ("100644", b"ref: refs/heads/main\n")}},
        check_files={"expected": ("100644", b"hello\n")}, answered_by=("search",), answered_in=("notes/**",),
        documents=(("example-text", "book.md", "a" * 64),), held_out_keys=("example-text/book.md",),
        absent=("/usr/share/doc",), terse="Write hello.", area=("out.txt",), gates=("correct",),
        tags=frozenset({"finish-tasks", "finish-tasks-with-less-writing"}))
    defaults = tasks.Task("", "", "", 0, 0, 0.0, {}, b"")
    assert [name for name in tasks.JSON_FIELDS if getattr(task, name) == getattr(defaults, name)] == []
    assert tasks.from_json(json.loads(json.dumps(tasks.to_json(task)))) == task


def test_measure_tasks_never_reach_training_and_the_measures_get_them(kit, repo, store, tmp_path):
    # tasks tagged eval, and the share of every task source the recipe's
    # held_out holds out by a hash of the task id (as documents are held
    # out), are measure-only: a training step never reads them, the
    # measures read exactly them
    names = [f"t{i}" for i in range(12)]
    files = {f"tasks/{name}/{part}": content for name in names
             for part, content in (("task.json", json.dumps(TASK).encode()), ("check", CHECK))}
    source = add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["**"]},
                                                              {"tags": ["eval"], "paths": ["tasks/t0/**"]}])
    held = {f"{source.id}/tasks/{name}" for name in names if is_held_out(f"{source.id}/tasks/{name}", 0.5)}
    assert 0 < len(held) < len(names)  # the share leaves some on both sides
    measures = held | {f"{source.id}/tasks/t0"}
    training, _ = tasks.read(repo, store, "finish-tasks", 0.5)
    measured, _ = tasks.read(repo, store, "finish-tasks", 0.5, measure=True)
    assert {task.id for task in measured} == measures
    assert {task.id for task in training} == {f"{source.id}/tasks/{name}" for name in names} - measures


def test_a_task_on_the_text_of_a_held_out_document_is_measure_only(kit, repo, store, tmp_path):
    # a held-out document never reaches
    # training through a task, also when the task points into another
    # source's copy of its text, whose own id is not held out; the held-out
    # texts are those the model's dataset lists as held out
    held, kept = [], []
    for number in range(16):  # sources with the same text, until one is held out by its id and one is not
        source = add(kit, repo, store, tmp_path / f"copy{number}", {"notes.txt": BOOK}, [])
        (held if is_held_out(f"{source.id}/notes.txt", 0.5) else kept).append(source)
        if held and kept:
            break
    other = next(source for number in range(16)  # a text of its own
                 for source in [add(kit, repo, store, tmp_path / f"other{number}", {"notes.txt": BOOK + b"%d\n" % number}, [])]
                 if not is_held_out(f"{source.id}/notes.txt", 0.5))
    tree, _ = source_tree(kept[0], store)
    [text] = [c.markdown for c in documents(kept[0], tree, store, repo)]
    names = [f"t{i}" for i in range(12)]
    files = {f"tasks/{name}/{part}": content for name in names for part, content in (
        ("task.json", json.dumps({**TASK, "pointers": [pointer(kept[0] if int(name[1:]) % 2 else other, text,
                                                               "a.txt", 0, 10)]}).encode()), ("check", CHECK))}
    work = add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    ids = {f"{work.id}/tasks/{name}" for name in names if not is_held_out(f"{work.id}/tasks/{name}", 0.5)}
    copied = {f"{work.id}/tasks/{name}" for name in names if int(name[1:]) % 2}
    assert ids & copied and ids - copied  # some tasks of each kind not held out by their own id
    dataset = listed(store, [(held[0].id, "notes.txt", text)], 0.5)
    training, _ = tasks.read(repo, store, "finish-tasks", 0.5, dataset=dataset)
    measured, _ = tasks.read(repo, store, "finish-tasks", 0.5, measure=True, dataset=dataset)
    assert {task.id for task in training} == ids - copied
    assert {task.id for task in measured} == {f"{work.id}/tasks/{name}" for name in names} - (ids - copied)


def test_a_task_pointing_into_a_document_held_out_by_its_group_is_measure_only(kit, repo, store, tmp_path):
    # a document of a source with held_out_by (as eBible's chapters are held
    # out by book and chapter) is held out
    # by its group, not its id; a task pointing into it is measure-only by
    # the same rule, whatever its own id and its id say
    path = "notes.txt"
    group = next(path[i:j] for i in range(len(path)) for j in range(i + 1, len(path) + 1)
                 if is_held_out(path[i:j], 0.5))  # a part of the path whose hash holds it out
    book = next(source for number in range(16)  # a source whose document's own id is not held out
                for source in [add(kit, repo, store, tmp_path / f"book{number}", {path: BOOK}, [])]
                if not is_held_out(f"{source.id}/{path}", 0.5))
    data = json.loads((repo.root / "data" / "sources" / book.id / "source.json").read_text())
    (repo.root / "data" / "sources" / book.id / "source.json").write_text(
        json.dumps({**data, "held_out_by": {"pattern": f"({re.escape(group)})"}}, indent=2) + "\n")
    tree, _ = source_tree(book, store)
    [text] = [c.markdown for c in documents(book, tree, store, repo)]
    names = [f"t{i}" for i in range(12)]
    files = {f"tasks/{name}/{part}": content for name in names for part, content in (
        ("task.json", json.dumps({**TASK, "pointers": [pointer(book, text, "a.txt", 0, 10)]}).encode()),
        ("check", CHECK))}
    add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    training, _ = tasks.read(repo, store, "finish-tasks", 0.5, dataset=listed(store))
    measured, _ = tasks.read(repo, store, "finish-tasks", 0.5, measure=True, dataset=listed(store))
    assert training == [] and len(measured) == len(names)


def test_pointer_tasks_are_refused_when_the_base_dataset_kept_no_document_list(kit, repo, store, tmp_path):
    # a dataset built before builds kept
    # their document list cannot say which texts are held out; reading
    # pointer tasks against it stops, saying such a base is used with the
    # repository at the commit that built it, never converting the sources
    # instead and promising no rebuild
    book = add(kit, repo, store, tmp_path / "book", {"notes.txt": BOOK}, [])
    tree, _ = source_tree(book, store)
    [text] = [c.markdown for c in documents(book, tree, store, repo)]
    add(kit, repo, store, tmp_path / "work", {"tasks/a/task.json": json.dumps(
        {**TASK, "pointers": [pointer(book, text, "a.txt", 0, 10)]}).encode(), "tasks/a/check": CHECK},
        [{"tags": ["finish-tasks"], "paths": ["**"]}])
    # a dataset this store does not hold is named as absent, not as old
    with pytest.raises(DocumentListMissing, match=f"dataset {'e' * 64}, the model's, is not in this store"):
        tasks.read(repo, store, "finish-tasks", 0, dataset="e" * 64)
    store.set_ref("datasets", "e" * 64, store.put_bytes(b"{}\n"))  # the dataset's record, and no document list
    with pytest.raises(DocumentListMissing, match=f"dataset {'e' * 64} predates document lists.*used with the "
                                                   f"repository at the commit that built it$"):
        tasks.read(repo, store, "finish-tasks", 0, dataset="e" * 64)


@pytest.mark.parametrize("files, error", [
    ({"tasks/a/task.json": json.dumps(TASK).encode(), "tasks/a/check": CHECK, "tasks/a/hint.txt": b"x\n"},
     "tasks/a/hint.txt: a task file must be a task folder's task.json, check or under its files/"),
    ({"tasks/a/task.json": json.dumps(TASK).encode()}, "tasks/a/: a task folder needs its check program"),
    ({"tasks/a/task.json": json.dumps({**TASK, "limits": {"max_turns": 0, "max_tokens": 1, "command_seconds": 1}})
      .encode(), "tasks/a/check": CHECK}, "tasks/a/task.json: at limits/max_turns"),
    ({"tasks/a/task.json": json.dumps({**TASK, "needs": [["ask"]]}).encode(), "tasks/a/check": CHECK},
     r"tasks/a/task.json: needs 'ask', which the task does not offer \(ask is offered only with replies\)"),
    ({"tasks/a/task.json": json.dumps({**TASK, "needs": [["write"], ["write", "edit"]]}).encode(),
      "tasks/a/check": CHECK},
     "tasks/a/task.json: 'write' is in two needs"),
    ({"tasks/a/task.json": json.dumps({**TASK, "tools": [{**SEARCH, "name": "bash"}]}).encode(),
      "tasks/a/check": CHECK},
     "tasks/a/task.json: the tool 'bash' is declared twice, or takes the name of a tool every task has"),
    ({"tasks/a/task.json": json.dumps({**TASK, "replies": [{"pattern": "(", "reply": "x"}],
                                       "answered_by": ["read"], "answered_in": ["README"]}).encode(),
      "tasks/a/check": CHECK}, r"tasks/a/task.json: the reply pattern '\(' is not a regular expression"),
    ({"tasks/a/task.json": json.dumps({**TASK, "replies": [{"pattern": "x", "reply": "y"}],
                                       "answered_in": ["README"]}).encode(),
      "tasks/a/check": CHECK}, "tasks/a/task.json: a task with replies names in answered_by the looking tools"),
    # each task states where its answers live; only a person knows where, so
    # a task with replies and no answered_in is read with answered_in empty, which the check names
    ({"tasks/a/task.json": json.dumps({**TASK, "replies": [{"pattern": "x", "reply": "y"}],
                                       "answered_by": ["read"]}).encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: at answered_in"),
    # answered_by names only looking tools: read, bash, or a declared tool marked looks, so
    # wpt_check.asked_last can see a look through one of them
    *(({"tasks/a/task.json": json.dumps({**TASK, "tools": tools, "replies": [{"pattern": "x", "reply": "y"}],
                                         "answered_by": ["read", name], "answered_in": ["README"]}).encode(),
        "tasks/a/check": CHECK},
       f"tasks/a/task.json: answered_by names '{name}', which is no looking tool the task or its image's pool offers")
      for tools, name in (([], "write"), ([{**SEARCH, "looks": False}], "search"), ([], "search"))),
    ({"tasks/a/task.json": json.dumps({**TASK, "needs": [["write", {"tools": ["bash"], "tool_files": ["gone"]}]]})
      .encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: tool_files names 'gone', which is no file or folder under files/"),
    ({"tasks/a/task.json": json.dumps({**TASK, "tool_files": ["corpus"]}).encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: tool_files names 'corpus', which is no file or folder under files/"),
    ({"tasks/a/task.json": json.dumps({**TASK, "pointers": [
        {"for": "files", "path": "a.txt", "source": "example-gone-0123abcd", "document": "notes.txt", "span": [0, 1],
         "sha256": "0" * 64}]}).encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: a pointer names the source 'example-gone-0123abcd', which does not exist"),
    # the one-line form of the request is one line
    ({"tasks/a/task.json": json.dumps({**TASK, "terse": "Write hello.\n"}).encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: terse is the request in one line, and it holds a line break"),
    ({"tasks/a/task.json": json.dumps({**TASK, "image": "missing"}).encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: no image 'missing': data/images/missing/Dockerfile does not exist"),
    # absent names a folder of the image: /work and /tmp are the attempt's own, and a path with .. could reach them
    *(({"tasks/a/task.json": json.dumps({**TASK, "absent": [place]}).encode(), "tasks/a/check": CHECK},
       rf"tasks/a/task.json: absent names '{place}'; it names a folder of the image")
      for place in ("/work", "/tmp/wpt-tools", "/usr/../tmp")),
    # /work and /tmp full, at 2 x 2048 MiB, would leave the processes no memory in the default 4096 MiB
    ({"tasks/a/task.json": json.dumps({**TASK, "container": {"disk": 2048}}).encode(), "tasks/a/check": CHECK},
     r"tasks/a/task.json: /work and /tmp may hold 2048 MiB each, .* more than 4096 MiB of memory; "
     r"raise container.memory above 5120"),
])
def test_a_broken_task_folder_stops_the_check(kit, repo, store, tmp_path, files, error):
    # a task that cannot run must be caught when the source is checked, not
    # in the middle of a training run
    source = add(kit, repo, store, tmp_path / "work", files, [{"tags": ["finish-tasks"], "paths": ["**"]}])
    with pytest.raises(InvalidFile, match=error):
        list(documents(source, source_tree(source, store)[0], store, repo))


def test_a_changed_image_folder_checks_every_source_with_tasks(kit, repo, store, tmp_path, monkeypatch, capsys):
    # CI fetches only the changed sources; a task naming an image that a
    # change removed must still be caught before it merges
    add(kit, repo, store, tmp_path / "work", {"tasks/a/task.json": json.dumps(TASK).encode(), "tasks/a/check": CHECK},
        [{"tags": ["finish-tasks"], "paths": ["**"]}])
    kit.run("git", "init", "-q", cwd=repo.root)
    kit.run("git", "add", "-A", cwd=repo.root)
    kit.run("git", "commit", "-q", "-m", "tasks", cwd=repo.root)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    assert main(["--root", str(repo.root), "check", "--changed", "HEAD"]) == 0  # nothing changed, nothing fetched
    assert "entries" not in capsys.readouterr().out

    shutil.rmtree(repo.root / "data" / "images" / "tiny")
    assert main(["--root", str(repo.root), "check", "--changed", "HEAD"]) == 1
    assert "tasks/a/task.json: no image 'tiny': data/images/tiny/Dockerfile does not exist" in capsys.readouterr().err
