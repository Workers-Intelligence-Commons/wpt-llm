"""The built-in read_the_manual measure, on hand-made attempts as
context.attempts() gives them (the run every agent measure shares)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpt_dataset.builder import MANUAL
from wpt_dataset.tasks import Task

FILE = Path(__file__).resolve().parents[1] / "src" / "wpt_eval" / "evals" / "read_the_manual.py"


def measure():
    spec = importlib.util.spec_from_file_location("read_the_manual", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(folder, absent=()):
    return Task(f"built-0123abcd/{folder}", "Find the seal.", "agent", 20, 4000, 30, {}, b"", absent=absent)


def records(*attempts):
    """Each (task, kind, score) as a record of context.attempts()."""
    return [SimpleNamespace(task=made, kind=kind, index=index, plain=False, conversation={"messages": []},
                            score=score) for index, (made, kind, score) in enumerate(attempts, start=1)]


def context(found):
    return SimpleNamespace(attempts=lambda plain=False: found)


def test_the_pass_rates_with_and_without_the_manual_and_the_gain():
    # the manual is removed by naming its folder in absent; other kinds' attempts do not count,
    # and an attempt passes only when its check scores 1
    shown, shown_too = task("read-the-manual/aa"), task("read-the-manual/bb")
    removed = task("read-the-manual/cc", absent=(MANUAL,))
    other = task("look-or-ask/dd")
    result = measure().run(context(records(
        (shown, "read-the-manual", 1.0), (shown, "read-the-manual", 1.0), (shown_too, "read-the-manual", 1.0),
        (shown_too, "read-the-manual", 0.0), (removed, "read-the-manual", 1.0), (removed, "read-the-manual", 0.0),
        (removed, "read-the-manual", 0.0), (removed, "read-the-manual", 0.5), (other, "look-or-ask", 1.0))))
    assert result.data["with_manual"] == {"tasks": 2, "attempts": 4, "passed": 3, "pass_rate": 0.75,
                                          "mean_score": 0.75}
    assert result.data["without_manual"] == {"tasks": 1, "attempts": 4, "passed": 1, "pass_rate": 0.25,
                                             "mean_score": 0.375}
    assert result.data["gain"] == pytest.approx(0.5)
    assert "with the manual: 3 of 4 attempts passed (75.0%)" in result.text
    assert "without the manual: 1 of 4 attempts passed (25.0%)" in result.text
    assert "gain from the manual, in pass rate: +50.0%" in result.text


def test_without_attempts_in_a_half_the_gain_is_unknown():
    result = measure().run(context(records((task("read-the-manual/aa"), "read-the-manual", 1.0))))
    assert result.data["without_manual"]["pass_rate"] is None and result.data["gain"] is None
    assert "unknown" in result.text
    none = measure().run(context([]))
    assert none.data["gain"] is None and "hold none" in none.text
