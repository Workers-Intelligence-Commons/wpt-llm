import json
import shutil

import pytest

from wpt_dataset import tasks
from wpt_dataset.cli import main
from wpt_dataset.errors import InvalidFile
from wpt_dataset.manifest import EXECUTABLE, FILE
from wpt_dataset.markdown import documents
from wpt_dataset.sources import source_tree

TASK = {"schema": 1, "instructions": "Write hello into out.txt.", "image": "tiny",
        "limits": {"max_turns": 4, "max_tokens": 200, "command_seconds": 5}}
CHECK = b"#!/bin/sh\ntest -f out.txt && echo 1 || echo 0\n"
FORMATS = [{"format": "task", "paths": ["tasks/**"]}, {"format": "text", "paths": ["notes.txt"]}]


def add(kit, repo, store, folder, files, tags):
    (repo.data / "images" / "tiny").mkdir(parents=True, exist_ok=True)
    (repo.data / "images" / "tiny" / "Dockerfile").write_text("ARG BASE=debian:trixie-slim\nFROM ${BASE}\n")
    address, commit = kit.git_project(folder, files=files, executable={p for p in files if p.endswith("/check")},
                                      symlinks={})
    sid = kit.add_source(repo, store, "example", folder.name, {"type": "git", "address": address, "revision": commit,
                                                               "formats": FORMATS, "tags": tags})
    return repo.source(sid)


def test_tagged_task_folders_reach_their_step_and_never_pretraining(kit, repo, store, tmp_path):
    # a step reads exactly the tasks tagged with its name, each with its
    # instructions, limits, starting files and check, and no task file may
    # become a pre-training document: the model would see the checks
    task = json.dumps(TASK).encode()
    files = {"tasks/hello/task.json": json.dumps({**TASK, "container": {"memory": 8192, "cpus": 4}}).encode(), "tasks/hello/check": CHECK, "tasks/hello/files/README": b"start\n",
             "tasks/hello/files/sub/task.json": b"a starting file, not a task\n",
             "tasks/other/task.json": task, "tasks/other/check": CHECK, "notes.txt": b"words\n"}
    source = add(kit, repo, store, tmp_path / "work", files, [{"tags": ["grpo"], "paths": ["tasks/hello/**"]},
                                                              {"tags": ["en"], "paths": ["notes.txt"]}])
    tree, _ = source_tree(source, store)
    assert [c.path for c in documents(source, tree, store, repo)] == ["notes.txt"]

    found, sources = tasks.read(repo, store, "grpo")
    assert found == [tasks.Task(f"{source.id}/tasks/hello", "Write hello into out.txt.", "tiny", 4, 200, 5,
                                {"README": (FILE, b"start\n"),
                                 "sub/task.json": (FILE, b"a starting file, not a task\n")}, CHECK,
                                {"memory": 8192, "disk": 1024, "processes": 256, "cpus": 4})]
    assert list(sources) == [source.id] and sources[source.id]["revision"] == source.data["revision"]
    assert tasks.read(repo, store, "sft") == ([], {})
    assert tasks.of_source(source, tree, store, repo)["tasks/other"].files == {}
    assert tree["tasks/hello/check"].mode == EXECUTABLE


@pytest.mark.parametrize("files, error", [
    ({"tasks/a/task.json": json.dumps(TASK).encode(), "tasks/a/check": CHECK, "tasks/a/hint.txt": b"x\n"},
     "tasks/a/hint.txt: a task file must be a task folder's task.json, check or under its files/"),
    ({"tasks/a/task.json": json.dumps(TASK).encode()}, "tasks/a/: a task folder needs its check program"),
    ({"tasks/a/task.json": json.dumps({**TASK, "limits": {"max_turns": 0, "max_tokens": 1, "command_seconds": 1}})
      .encode(), "tasks/a/check": CHECK}, "tasks/a/task.json: at limits/max_turns"),
    ({"tasks/a/task.json": json.dumps({**TASK, "image": "missing"}).encode(), "tasks/a/check": CHECK},
     "tasks/a/task.json: no image 'missing': data/images/missing/Dockerfile does not exist"),
    # /work and /tmp full, at 2 x 2048 MiB, would leave the processes no memory in the default 4096 MiB
    ({"tasks/a/task.json": json.dumps({**TASK, "container": {"disk": 2048}}).encode(), "tasks/a/check": CHECK},
     r"tasks/a/task.json: /work and /tmp may hold 2048 MiB each, .* more than 4096 MiB of memory; "
     r"raise container.memory above 5120"),
])
def test_a_broken_task_folder_stops_the_check(kit, repo, store, tmp_path, files, error):
    # a task that cannot run must be caught when the source is checked, not
    # in the middle of a post-training run
    source = add(kit, repo, store, tmp_path / "work", files, [{"tags": ["grpo"], "paths": ["**"]}])
    with pytest.raises(InvalidFile, match=error):
        list(documents(source, source_tree(source, store)[0], store, repo))


def test_a_changed_image_folder_checks_every_source_with_tasks(kit, repo, store, tmp_path, monkeypatch, capsys):
    # CI fetches only the changed sources; a task naming an image that a
    # change removed must still be caught before it merges
    add(kit, repo, store, tmp_path / "work", {"tasks/a/task.json": json.dumps(TASK).encode(), "tasks/a/check": CHECK},
        [{"tags": ["grpo"], "paths": ["**"]}])
    kit.run("git", "init", "-q", cwd=repo.root)
    kit.run("git", "add", "-A", cwd=repo.root)
    kit.run("git", "commit", "-q", "-m", "tasks", cwd=repo.root)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    assert main(["--root", str(repo.root), "check", "--changed", "HEAD"]) == 0  # nothing changed, nothing fetched
    assert "entries" not in capsys.readouterr().out

    shutil.rmtree(repo.data / "images" / "tiny")
    assert main(["--root", str(repo.root), "check", "--changed", "HEAD"]) == 1
    assert "tasks/a/task.json: no image 'tiny': data/images/tiny/Dockerfile does not exist" in capsys.readouterr().err
