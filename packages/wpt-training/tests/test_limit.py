"""A run's limit and what its speed projects (wpt train --stop-after-minutes
and --stop-after-updates; docs/training.md, "Checkpoints and resuming")."""

from types import SimpleNamespace

from wpt_training import Limit, window_tokens_left
from wpt_training import runner


def windows(name, *stages, continues=False):
    return SimpleNamespace(name=name, continues=continues, environment_settings={"stages": [
        {"steps": steps, "batch_size": batch, "block_size": block} for steps, batch, block in stages]})


def test_the_first_limit_reached_stops_the_run_and_only_updates_after_the_first_are_measured(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: now[0])
    both = Limit(updates=3, minutes=1)
    both.count(64)  # the first update also compiles the kernels: it starts the clock, and is not measured
    assert not both.reached() and both.measured == 0
    now[0] += 30
    both.count(64)
    assert not both.reached() and (both.measured, both.tokens, both.seconds()) == (1, 64, 30)
    both.count(64)
    assert both.reached()  # the third update

    minutes = Limit(minutes=1)
    minutes.count(0)
    now[0] += 59
    assert not minutes.reached()
    now[0] += 1
    assert minutes.reached()


def test_the_tokens_left_count_later_stages_and_steps_and_name_the_steps_that_list_no_windows():
    steps = [windows("read", (10, 2, 4), (10, 2, 8)),
             SimpleNamespace(name="chat", continues=False, environment_settings={}),
             windows("again", (5, 1, 2))]
    assert window_tokens_left(steps) == (10 * 8 + 10 * 16 + 5 * 2, ["chat"])
    assert window_tokens_left(steps, 1, 15) == (5 * 16 + 5 * 2, ["chat"])
    assert window_tokens_left(steps, 3, 4) == (2, [])


def test_a_step_that_continues_the_one_before_counts_only_the_updates_past_it():
    steps = [windows("trunk", (10, 1, 4)), windows("branch", (15, 1, 4), continues=True)]
    assert window_tokens_left(steps) == (10 * 4 + 5 * 4, [])
