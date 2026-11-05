"""wpt doctor: one pass over what a machine needs, naming what is missing
and how to fix it."""

import os
import subprocess

import pytest
import torch

from wpt.doctor import main
from wpt_utils.lines import bare

GIT = ["git", "-c", "user.name=T", "-c", "user.email=t@example.org"]


@pytest.fixture
def clean(built, tmp_path, monkeypatch):
    """A checkout with nothing uncommitted, and gpu workers named, so this
    machine's own GPU does not matter."""
    (tmp_path / ".gitignore").write_text(".env\n")
    subprocess.run([*GIT, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*GIT, "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run([*GIT, "commit", "-qm", "c"], cwd=tmp_path, check=True)
    monkeypatch.setenv("WPT_GPU_WORKERS", "wpt-worker://t@example.org:8470#pin")


def test_a_ready_machine_has_nothing_missing(clean, built, capsys):
    main([])
    printed = bare(capsys.readouterr().out)
    assert "MISSING" not in printed and printed.splitlines()[-1] == "nothing is missing"
    assert "store: " in printed and "no GPU here, and WPT_GPU_WORKERS names the gpu workers" in printed
    assert "attention" not in printed  # decided on the gpu workers
    assert "GB free in" in printed and "GB of sources" in printed


def test_a_machine_that_trains_here_names_what_computes_the_attention(clean, monkeypatch, capsys):
    monkeypatch.delenv("WPT_GPU_WORKERS")
    monkeypatch.setenv("WPT_KERNELS", "pytorch")
    monkeypatch.setattr("wpt.doctor.check_torch", lambda environ: ([], torch.device("cpu")))
    main([])
    assert "ok  attention: PyTorch" in bare(capsys.readouterr().out)


def test_every_gap_is_named_with_its_fix_in_one_pass(clean, built, tmp_path, monkeypatch, capsys):
    # three gaps at once: the store does not open, the tools are not found, a file changes every key on the code
    (tmp_path / "new.txt").write_text("untracked")
    monkeypatch.delenv("WPT_STORE")
    (tmp_path / "bin").mkdir()
    os.symlink(subprocess.run(["which", "git"], capture_output=True, text=True).stdout.strip(),
               tmp_path / "bin" / "git")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    with pytest.raises(SystemExit) as gaps:
        main([])
    assert gaps.value.code == 1
    printed = bare(capsys.readouterr().out)
    assert "MISSING  store: " in printed and "fix: set WPT_STORE" in printed
    assert "MISSING  rustc: not found" in printed and "MISSING  cargo: not found" in printed
    assert "fix: install a Rust toolchain" in printed
    assert "MISSING  checkout: 0 uncommitted and 2 untracked files" in printed  # bin/ and new.txt
    assert printed.splitlines()[-1] == "4 thing(s) missing"


def test_a_file_store_whose_folder_does_not_exist_is_missing_and_stays_absent(clean, tmp_path, monkeypatch, capsys):
    # opening a store creates its folders: a mistyped path must not read as a healthy, empty store
    absent = tmp_path / "typed-wrong" / "store"
    monkeypatch.setenv("WPT_STORE", f"file://{absent}")
    with pytest.raises(SystemExit) as gaps:
        main([])
    assert gaps.value.code == 1
    printed = bare(capsys.readouterr().out)
    assert f"MISSING  store: {absent} does not exist" in printed and "fix: correct WPT_STORE" in printed
    assert not absent.parent.exists()
