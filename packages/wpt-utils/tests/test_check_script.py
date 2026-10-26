import importlib.util
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
