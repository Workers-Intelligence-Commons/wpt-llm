import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check.py"


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("wpt_check_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clone(tmp_path):
    """A git repository of one commit that ignores `ignored.txt`, and a
    function that runs git in it; returns (its folder, git, the commit)."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=tmp_path, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "conftest.py").write_text("one\n")
    (tmp_path / "other.py").write_text("one\n")
    git("add", ".")
    git("commit", "-qm", "one")
    return tmp_path, git, git("rev-parse", "HEAD")


def test_the_working_tree_counts_what_is_not_committed_and_a_push_only_what_is(check, clone, monkeypatch):
    # the image build test must run on work not committed yet, staged or not,
    # new or not; a run for a push reads only what the push sends
    root, git, base = clone
    monkeypatch.setattr(check, "ROOT", root)
    monkeypatch.setattr(check, "images_changed", lambda base, working_tree: False)
    (root / "conftest.py").write_text("two\n")
    (root / "new.py").write_text("new\n")
    (root / "ignored.txt").write_text("ignored\n")
    assert sorted(check.changed_paths(base, True)) == ["conftest.py", "new.py"]
    assert check.changed_paths(base, False) == []
    assert check.image_tests(base, False)[0] == ["-m", "not image_build"]
    deselect, why = check.image_tests(base, True)
    assert deselect == [] and "the working tree changes conftest.py" in why

    git("add", "conftest.py")  # staged counts the same
    assert sorted(check.changed_paths(base, True)) == ["conftest.py", "new.py"]
    git("commit", "-qm", "two")  # committed, a push sends it
    assert check.changed_paths(base, False) == ["conftest.py"]
    assert "this push changes conftest.py" in check.image_tests(base, False)[1]


def test_a_new_file_of_an_image_s_folder_counts_in_the_working_tree(check, clone):
    # what the service layer answers: git diff alone never lists a file git does not track
    root, git, base = clone
    image = root / "data" / "images" / "one"
    image.mkdir(parents=True)
    (image / "Dockerfile").write_text("FROM scratch\n")

    def answer(code):
        return subprocess.run([sys.executable, "-c", code, base], cwd=root, check=True, capture_output=True,
                              text=True).stdout.strip()

    assert answer(check.IMAGES_DIFFER) == "True" and answer(check.IMAGES_CHANGED) == "False"
    (image / "Dockerfile").unlink()
    assert answer(check.IMAGES_DIFFER) == "False"


def test_the_working_tree_mode_runs_every_check_but_the_commits_and_says_so(check, monkeypatch, capsys):
    ran, asked = [], []
    monkeypatch.setattr(check, "image_tests", lambda base, working_tree: asked.append(working_tree) or ([], "why"))
    monkeypatch.setattr(check, "test_processes", lambda deselect: (1, "sized"))
    monkeypatch.setattr(check, "with_python_library", dict)
    monkeypatch.setattr(check, "run", lambda commands, folder, env: ran.extend(commands) or True)
    monkeypatch.setattr(sys, "argv", ["check.py", "--working-tree", "--base", "main"])
    with pytest.raises(SystemExit) as ended:
        check.main()
    assert ended.value.code == 0 and asked == [True]
    out = capsys.readouterr().out
    assert [line.split()[1] for line in out.splitlines() if line.startswith("passed")] == [
        "lock", "tests", "rust", "data", "sources"]
    assert "left out  commits" in out
    assert not any("check-commits" in command for command in ran)


def test_a_range_of_commits_is_refused_with_the_working_tree(check, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check.py", "--working-tree", "--range", "a..b"])
    with pytest.raises(SystemExit) as refused:
        check.main()
    assert refused.value.code == 2 and "--working-tree checks none" in capsys.readouterr().err


def test_the_sample_is_every_nth_test_in_collection_order(check):
    ids = [f"t{i}.py::test_{i}" for i in range(10)]
    assert check.sample_of(ids, 4) == ["t0.py::test_0", "t4.py::test_4", "t8.py::test_8"]
    assert check.sample_of(ids, 4) == check.sample_of(list(ids), 4)  # the same tests, the same sample
    assert check.sample_of([], 4) == []


def test_the_test_processes_are_sized_by_the_memory_the_sample_held_alone(check, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(check, "collected", lambda deselect: [f"a.py::t{i}" for i in range(100)])
    monkeypatch.setattr(check, "peak", lambda command, **options: ran.append(command) or (0, 5e9))
    asked = []

    def limit(memory, what):
        asked.append((memory, what))
        return 3, "3 test workers: sized"

    monkeypatch.setattr(check, "limit", limit)
    count, line = check.test_processes(["-m", "not image_build"])
    assert count == 3 and asked == [(5e9, "test workers")]
    assert ran[0][-3:] == ["a.py::t0", "a.py::t40", "a.py::t80"] and "-n" in ran[0]  # alone: no other processes
    assert "3 sample tests ran alone (passed); it held 5.0 GB at most" in line and "3 test workers: sized" in line


def test_a_sample_whose_memory_cannot_be_measured_leaves_the_cores_to_decide(check, monkeypatch):
    monkeypatch.setattr(check, "collected", lambda deselect: ["a.py::t"])
    monkeypatch.setattr(check, "peak", lambda command, **options: (1, None))
    count, line = check.test_processes([])
    assert count >= 1 and "FAILED" in line and "its memory could not be measured" in line
    assert "unknown" in line


def test_no_fixed_memory_figure_sizes_the_test_processes(check):
    assert not hasattr(check, "WORKER_MEMORY")
