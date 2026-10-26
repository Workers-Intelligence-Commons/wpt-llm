"""A checkpoint state as a safetensors file (wpt_training.states): what
a step's parts hand to a checkpoint comes back exactly, and loading a file
never runs code."""

import math

import pytest
import torch

from wpt_training import load_state, save_state


def state_like_a_step_s():
    """What a checkpoint holds: weights, an optimizer's state with integer keys, tuples and
    several types, and what the method and the environment add."""
    weights = {"embed.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
               "norm.weight": torch.ones(3, dtype=torch.bfloat16), "steps": torch.tensor(7)}
    return {"signature": {"identity": "abc"}, "step": 3, "next_step": 4, "model": weights,
            "optimizer": {"state": {0: {"step": torch.tensor(3.0), "exp_avg": torch.full((2, 3), 0.5)},
                                    1: {"step": torch.tensor(3.0), "exp_avg": torch.zeros(3)}},
                          "param_groups": [{"lr": 0.01, "betas": (0.9, 0.95), "params": [0, 1], "nesterov": True,
                                            "weight_decay": None, "name": "all"}],
                          "best": float("inf"), "history": [1.5, 1.25]},
            "method": {"next_step": 4}, "environment": {"sampling": torch.tensor([1, 2, 3], dtype=torch.uint8),
                                                        "manager": {"seen": [1, 2], "ratio": float("nan")}},
            "best_val": None, "best_model": None, "reached": {"best": 2.5}}


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        [same(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        [same(a, b) for a, b in zip(left, right)]
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert type(left) is type(right) and left == right


def test_a_state_comes_back_with_its_tensors_types_keys_and_nesting_as_it_was_saved(tmp_path):
    state = state_like_a_step_s()
    save_state(state, tmp_path / "checkpoint.safetensors")
    same(state, load_state(tmp_path / "checkpoint.safetensors"))
    assert [path.name for path in tmp_path.iterdir()] == ["checkpoint.safetensors"]  # no partial file stays


def test_a_second_save_replaces_the_first(tmp_path):
    save_state({"step": 1, "model": {"w": torch.zeros(2)}}, tmp_path / "state")
    save_state({"step": 2, "model": {"w": torch.ones(2)}}, tmp_path / "state")
    assert load_state(tmp_path / "state")["step"] == 2


def test_only_the_entries_asked_for_are_given(tmp_path):
    state = state_like_a_step_s()
    save_state(state, tmp_path / "state")
    found = load_state(tmp_path / "state", only=("model", "best_model"))
    assert list(found) == ["model", "best_model"] and found["best_model"] is None
    same(state["model"], found["model"])


def test_tensors_that_share_memory_are_saved_whole_and_come_back_as_values(tmp_path):
    # tied weights, and two views of one memory: safetensors keeps no two tensors that overlap
    base = torch.arange(8, dtype=torch.float32)
    state = {"tied_a": base, "tied_b": base, "first_half": base[:4], "view": base.reshape(2, 4)}
    save_state(state, tmp_path / "state")
    found = load_state(tmp_path / "state")
    same(state, found)
    found["tied_a"][0] = 99
    assert found["tied_b"][0] == 99  # the memory the state shared is shared again, once
    assert found["first_half"][0] == 0 and base[0] == 0


def test_a_value_a_state_does_not_hold_is_refused_and_nothing_is_written(tmp_path):
    with pytest.raises(TypeError, match=r"state\.mine: a set is not a value a state holds"):
        save_state({"mine": {1, 2}}, tmp_path / "state")
    assert not list(tmp_path.iterdir())


def test_a_file_that_is_not_a_state_is_refused_and_a_pickle_is_never_run(tmp_path):
    torch.save({"step": 1}, tmp_path / "pickled.pt")
    with pytest.raises(Exception, match="HeaderTooLarge|header|Metadata|InvalidHeader"):
        load_state(tmp_path / "pickled.pt")
    from safetensors.torch import save_file

    save_file({"w": torch.zeros(1)}, tmp_path / "plain.safetensors")
    with pytest.raises(ValueError, match="not a state file"):
        load_state(tmp_path / "plain.safetensors")
