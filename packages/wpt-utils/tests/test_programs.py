import os
import subprocess

import pytest

from wpt_utils import programs
from wpt_utils.programs import checkout_outside

GIT_ENV = {"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def test_the_code_outside_data_stays_the_same_through_changes_to_data_only(tmp_path, monkeypatch):
    # a source's own results are keyed by the code outside data/: a commit,
    # or an uncommitted change, that touches only data/ keeps it, and any
    # change to a file outside data/, committed, uncommitted or new, makes
    # a new one
    for name, value in GIT_ENV.items():
        monkeypatch.setenv(name, value)

    def git(*command):
        subprocess.run(["git", "-C", str(tmp_path), *command], capture_output=True, check=True)

    def write(path, text):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8", newline="\n")

    def commit(message):
        git("add", "-A")
        git("commit", "-q", "-m", message)

    git("init", "-q")
    write("packages/convert.py", "print('one')\n")
    write("data/sources/a/source.json", "{}\n")
    commit("first")
    first = checkout_outside(tmp_path, "data")

    write("data/sources/b/source.json", "{}\n")  # another source, not yet committed
    write("data/sources/a/source.json", '{"changed": true}\n')
    assert checkout_outside(tmp_path, "data") == first
    commit("another source")
    assert checkout_outside(tmp_path, "data") == first

    write("packages/convert.py", "print('two')\n")  # the code changed, not committed
    uncommitted = checkout_outside(tmp_path, "data")
    assert uncommitted != first
    commit("the code")
    committed = checkout_outside(tmp_path, "data")
    assert committed != first

    write("packages/new.py", "print('three')\n")  # a new file git does not ignore
    assert checkout_outside(tmp_path, "data") != committed


def test_a_worker_uses_the_code_outside_its_starter_gives_and_refuses_a_folder_not_given(monkeypatch):
    # a worker's fingerprints are its starter's even when the checkout
    # changed meanwhile, so it never finds a folder's code outside itself
    for name in ("_CODE", "_OUTSIDE", "_GIVEN"):  # put back as they were when the test ends
        monkeypatch.setattr(programs, name, getattr(programs, name))
    programs.set_code({"code": {"commit": "0" * 40, "changes": None}, "outside": {"data": "1" * 64}})
    assert programs.code_outside("data") == "1" * 64
    with pytest.raises(KeyError):
        programs.code_outside("other")
