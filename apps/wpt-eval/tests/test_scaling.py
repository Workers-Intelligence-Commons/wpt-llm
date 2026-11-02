"""The scaling curve (wpt_eval.scaling, docs/evaluation.md, "Scaling
curve"): the fit recovers a known power law from points on it, and the
command reads each model's point from its provenance and predicts a size
of a series before it trains."""

import json
from pathlib import Path

import pytest

from wpt_dataset.repo import Repo
from wpt_eval.scaling import fit, main, planned, point
from wpt_store import LocalStore
from wpt_store.models import KIND, put_model

ROOT = Path(__file__).resolve().parents[3]
COMPUTE = [1e18, 4e18, 2e19, 7e19, 3e20]


def test_the_fit_recovers_a_and_b_of_a_power_law_from_three_points_on_it():
    a, b = 31.0, 0.052
    curve = fit(COMPUTE[:3], [a * c ** -b for c in COMPUTE[:3]])
    assert not curve.with_irreducible and curve.irreducible == 0
    assert curve.a == pytest.approx(a, rel=1e-6) and curve.b == pytest.approx(b, rel=1e-6)
    assert curve.loss(1e22) == pytest.approx(a * 1e22 ** -b, rel=1e-6)


def test_from_four_points_the_fit_also_recovers_the_irreducible_loss():
    a, b, irreducible = 420.0, 0.11, 1.7
    curve = fit(COMPUTE, [a * c ** -b + irreducible for c in COMPUTE])
    assert curve.with_irreducible
    assert curve.a == pytest.approx(a, rel=1e-4) and curve.b == pytest.approx(b, rel=1e-4)
    assert curve.irreducible == pytest.approx(irreducible, rel=1e-4)


def model_folder(repo, folder, given, loss, held_out="h"):
    """The model folder of a run of recipe given that did every update of its stages: the parts of
    provenance.json the command reads (wpt-train writes it), and, unless loss is None, evals/held-out-loss.json
    (wpt-eval writes it) with that overall loss. The best validation loss it also holds is never the point's."""
    recipe = repo.model_recipe(given)
    [step] = recipe["steps"]
    [stage] = step["settings"]["stages"]
    folder.mkdir()
    (folder / "provenance.json").write_text(json.dumps({
        "recipe_name": given, "recipe": recipe, "held_out_shards": [{"object": "o", "token_sha256": held_out}],
        "steps": [{"step": {"name": "read-text"}, "best_validation_loss": 99.0,
                   "stages": [{**stage, "first_step": 1, "last_step": stage["steps"]}]}]}))
    if loss is not None:
        (folder / "evals").mkdir()
        (folder / "evals" / "held-out-loss.json").write_text(json.dumps({"name": "held-out-loss",
                                                                         "data": {"loss": loss, "tokens": 5}}))
    return str(folder)


def test_the_command_fits_the_trained_sizes_and_predicts_the_loss_of_a_larger_size(tmp_path, monkeypatch, capsys):
    # given three sizes of the series trained to losses on a known curve, two as model folders and one kept in the
    # store under its run's name, the command must find that curve and give the next size the loss it predicts
    repo = Repo(ROOT)
    a, b = 28.0, 0.048
    # C = M x D: the series' M of each size (6 x its active parameters + the attention term) x the tokens its
    # updates train on
    compute = {plan.size: plan.flops_per_token * plan.updates * plan.batch_size * plan.block_size
               for plan in repo.series_plans("eldest-titmouse")}
    store = LocalStore(tmp_path / "store")
    models = []
    for size in ("256x5", "384x8", "512x10"):
        models.append(model_folder(repo, tmp_path / size, f"eldest-titmouse:{size}", a * compute[size] ** -b))
    store.set_ref(KIND, "run-512", put_model(store, tmp_path / "512x10"))
    models[-1] = "run-512"
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.chdir(ROOT)

    assert main([*models, "--predict", "eldest-titmouse:640x13"]) == 0
    out = capsys.readouterr().out
    printed = out.split("fit: L = ", 1)[1].split("\n", 1)[0]
    found_a, found_b = float(printed.split(" x C^-")[0]), float(printed.split(" x C^-")[1].split()[0])
    assert found_a == pytest.approx(a, rel=1e-4) and found_b == pytest.approx(b, rel=1e-4)
    larger = compute["640x13"]
    assert f"predicted: eldest-titmouse:640x13: C {larger:.4g}" in out
    assert f"loss {a * larger ** -b:.4f}" in out.split("predicted:", 1)[1]
    assert "run-512 (eldest-titmouse:512x10)" in out


def test_a_model_without_a_held_out_loss_evaluation_is_named_and_left_out_of_the_fit(tmp_path, capsys):
    # given three sizes on a known curve and a fourth with no evaluation (its best validation loss is in its
    # provenance, and is not used), the fit is of the three, the output names the fourth and says what makes
    # the evaluation
    repo = Repo(ROOT)
    a, b = 28.0, 0.048
    compute = {plan.size: plan.flops_per_token * plan.updates * plan.batch_size * plan.block_size
               for plan in repo.series_plans("eldest-titmouse")}
    folders = [model_folder(repo, tmp_path / size, f"eldest-titmouse:{size}",
                            a * compute[size] ** -b if size != "640x13" else None)
               for size in ("256x5", "384x8", "512x10", "640x13")]
    assert main(folders) == 0
    out = capsys.readouterr().out
    assert out.startswith("3 points")
    assert f"left out of the fit, with no evals/held-out-loss.json: {folders[3]}" in out
    assert "WPT_EVALUATE=1" in out and "wpt-eval" in out
    assert str(tmp_path / "256x5") in out and "loss 99" not in out


def test_when_the_models_left_out_leave_fewer_than_two_points_the_error_names_them(tmp_path, capsys):
    repo = Repo(ROOT)
    folders = [model_folder(repo, tmp_path / "256x5", "eldest-titmouse:256x5", 3.0),
               model_folder(repo, tmp_path / "384x8", "eldest-titmouse:384x8", None)]
    assert main(folders) == 2
    err = capsys.readouterr().err
    assert "1 points: a fit needs 2 at least" in err and f"left out of the fit, with no evals/held-out-loss.json: {folders[1]}" in err


def test_models_that_hold_out_other_text_are_refused(tmp_path, capsys):
    repo = Repo(ROOT)
    folders = [model_folder(repo, tmp_path / size, f"eldest-titmouse:{size}", 3.0, held_out)
               for size, held_out in (("256x5", "h"), ("384x8", "other"))]
    assert main(folders) == 2
    assert "hold out other text" in capsys.readouterr().err


def test_a_step_that_continues_another_counts_only_the_updates_it_adds():
    # given a trunk of 50 updates and a decay branch that continues it to 75 (which reports updates 1 to 75), the
    # model trained 75 updates, not 125, and the recipe plans the same
    stage = {"block_size": 8, "batch_size": 2}
    repo = Repo(ROOT)
    model = repo.model_recipe("eldest-titmouse:256x5")["model"]

    def step(updates, continues):
        entry = {"settings": {"stages": [{**stage, "steps": updates}]}}
        return {**entry, "continues": True} if continues else entry

    recipe = {"model": model, "steps": [step(50, False), step(75, True)]}
    reported = [{"step": {"name": "read-text"}, "best_validation_loss": 3.0,
                 "stages": [{**stage, "steps": updates, "first_step": 1, "last_step": updates}]}
                for updates in (50, 75)]
    found = point("run", {"recipe_name": "r", "recipe": recipe, "steps": reported}, 3.0)
    assert found.tokens == 75 * 2 * 8
    assert planned(recipe) == found.compute
