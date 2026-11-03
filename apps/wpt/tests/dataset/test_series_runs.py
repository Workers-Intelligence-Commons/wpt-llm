"""A size's other runs (docs/model.md, "Model series"): decay branches
that go on from one trunk, and the learning-rate check at the smallest
size."""

import copy
import math
from pathlib import Path

import pytest

from wpt_dataset import series
from wpt_dataset.errors import RuleViolation
from wpt_dataset.repo import Repo, validate

ROOT = Path(__file__).resolve().parents[4]
WINDOW = 32768


@pytest.fixture(scope="module")
def ladder():
    """eldest-titmouse, which has branches at a quarter, half and all of each size's updates, and the
    learning-rate check at a quarter, half, once, twice and four times the law's rate."""
    found = copy.deepcopy(Repo(ROOT).recipe("series", "eldest-titmouse"))
    assert found["branches"] == [0.25, 0.5, 1] and found["learning_rate_check"] == [0.25, 0.5, 1, 2, 4]
    return found


def without_continues(step):
    return {key: value for key, value in step.items() if key != "continues"}


def test_a_branch_is_the_trunk_up_to_its_start_then_its_decay_to_its_own_updates(ladder):
    # 256x5 makes 9,271 updates of 8 windows after a warm-up of 371; its branches make 2,318 (a quarter, 2,317.75
    # rounded), 4,636 (half, 4,635.5) and 9,271 updates, and each starts its decay after the last update at the
    # peak: 80% of its updates, 1,854, 3,708 and 7,416
    plan = series.plan(ladder, "256x5", 32768)
    assert (plan.updates, plan.warmup, plan.batch_size) == (9271, 371, 8)
    assert [(b.name, b.updates, b.start) for b in series.branches(ladder, plan)] == [
        ("branch-0.25", 2318, 1854), ("branch-0.5", 4636, 3708), ("branch-1", 9271, 7416)]
    [own] = series.run_recipe(ladder, plan)["steps"]
    half = series.run_recipe(ladder, plan, "branch-0.5")
    full = series.run_recipe(ladder, plan, "branch-1")
    for recipe in (half, full):
        validate(recipe, "model", "a branch")
        assert {key: value for key, value in recipe.items() if key != "steps"} == \
            {key: value for key, value in series.model_recipe(ladder, plan).items() if key != "steps"}

    def trunk(updates, continues):
        # the size's step at the peak rate and the early weight of multi-token prediction to its end, reading the
        # size's own prefix (no branch is longer than the size)
        step = {"step": "read-text",
                "optimizer": {"name": "by-part", "settings": {"lr": plan.learning_rate, "warmup": 371,
                                                              "first_drop": 1.0, "second_drop": 1.0}},
                "settings": {"stages": [{"block_size": WINDOW, "steps": updates, "batch_size": 8}],
                             "prefix_tokens": plan.prefix_tokens, "mtp_switch": 1.0},
                "locked": []}
        return {**step, "continues": True} if continues else step

    # the trunk cut at every shorter branch's start: the half branch holds the trunk to 3,708, then decays
    assert half["steps"][:2] == [trunk(1854, False), trunk(3708, True)]
    assert half["steps"][2]["continues"] is True
    assert without_continues(half["steps"][2]) == {**own, "settings": {**own["settings"], "stages": [
        {"block_size": WINDOW, "steps": 4636, "batch_size": 8}]}}
    # the full branch shares those trunk steps (wpt train reuses their results), goes on to 7,416, and decays as
    # the size's own step does
    assert full["steps"][:3] == [trunk(1854, False), trunk(3708, True), trunk(7416, True)]
    assert without_continues(full["steps"][3]) == own and full["steps"][3]["continues"] is True


def test_at_every_size_the_full_length_branch_decays_as_the_size_s_own_run_does(ladder):
    # the same schedule: the decay step is the size's own step (its updates, rates, warm-up, window and batch, and
    # the switch of the multi-token prediction's weight after 80% of the updates) and it begins where the own
    # run's first drop begins, after the last update at the peak
    for size in series.sizes(ladder):
        plan = series.plan(ladder, size, 32768)
        [own] = series.run_recipe(ladder, plan)["steps"]
        assert own["settings"]["mtp_switch"] == 0.8
        full = series.run_recipe(ladder, plan, "branch-1")
        assert without_continues(full["steps"][-1]) == own
        [branch] = [b for b in series.branches(ladder, plan) if b.name == "branch-1"]
        assert branch.updates == plan.updates and branch.start == math.floor(0.8 * plan.updates)
        assert full["steps"][-2]["settings"]["stages"][0]["steps"] == branch.start


def test_the_learning_rate_check_runs_the_smallest_size_at_multiples_of_the_law_s_rate(ladder):
    plan = series.plan(ladder, "256x5", 32768)
    assert series.smallest(ladder) == "256x5" and plan.learning_rate == 0.001694
    assert series.runs(ladder, "256x5") == ["branch-0.25", "branch-0.5", "branch-1",
                                            "lr-0.25", "lr-0.5", "lr-1", "lr-2", "lr-4"]
    assert series.runs(ladder, "512x10") == ["branch-0.25", "branch-0.5", "branch-1"]
    own = series.run_recipe(ladder, plan)
    for run, rate in (("lr-0.25", 0.0004235), ("lr-0.5", 0.000847), ("lr-1", 0.001694), ("lr-2", 0.003388),
                      ("lr-4", 0.006776)):
        recipe = series.run_recipe(ladder, plan, run)
        [step] = recipe["steps"]
        assert step["optimizer"]["settings"]["lr"] == rate
        # everything else is the size's own run
        changed = copy.deepcopy(own)
        changed["steps"][0]["optimizer"]["settings"]["lr"] = rate
        assert recipe == changed
    with pytest.raises(ValueError, match="no run 'lr-2' of the size 512x10"):
        series.run_recipe(ladder, series.plan(ladder, "512x10", 32768), "lr-2")


def test_a_branch_longer_than_its_size_makes_the_series_dataset_hold_its_trunk(ladder):
    # twice 768x15's 18,585 updates of 32 windows: the one dataset holds the tokens of 37,170 updates
    longer = {**ladder, "branches": [0.5, 2]}
    assert series.data_part(longer)["total_tokens"] == 37170 * 32 * (WINDOW + 1)
    assert series.data_part(ladder)["total_tokens"] == series.plan(ladder, "768x15", 0).prefix_tokens


def test_a_branch_whose_decay_would_start_inside_the_warm_up_is_refused(ladder):
    # 4% of 256x5's updates is 371 updates; 80% of them, 296, is inside the warm-up of 371
    with pytest.raises(ValueError, match="size 256x5: branch-0.04 has 371 updates, so its decay would start after "
                                         "update 296, inside the warm-up of 371"):
        series.check({**ladder, "branches": [0.04, 0.5]})


def test_a_run_a_size_does_not_have_is_refused():
    with pytest.raises(RuleViolation, match="no run 'lr-2' of the size 384x8; its runs are branch-0.25, "
                                            "branch-0.5, branch-1"):
        Repo(ROOT).model_recipe("eldest-titmouse:384x8:lr-2")


def test_the_series_command_prints_the_tokens_of_the_dataset_the_recipe_holds(capsys):
    from wpt.dataset import main

    name = "eldest-titmouse"
    repo = Repo(ROOT)
    total = series.data_part(repo.recipe("series", name))["total_tokens"]
    assert repo.series_total_tokens(name) == total
    assert main(["series", name]) == 0
    assert f"one dataset of {total:,} training tokens" in capsys.readouterr().out
