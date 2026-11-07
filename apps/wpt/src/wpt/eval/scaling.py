"""wpt scaling: the scaling curve of trained models, loss against compute,
and the loss it predicts for a model not trained yet (docs/evaluation.md,
"Scaling curve").

Usage:
    wpt scaling MODEL MODEL [MODEL ...] [--predict RECIPE ...]

MODEL is a model folder that wpt train wrote, or the name of a run whose
model the store named by WPT_STORE (environment or .env) keeps
(models/NAME, as wpt manager keeps it). Of each model, its provenance.json
and its evals/held-out-loss.json give a point:

    compute   C = M x D: M, the training compute per token, is
              wpt_dataset.series.flops_per_token of the recipe's model at
              the window length of its last text-window stage (the count a
              series' plan reads: 6 x the active parameters + the attention
              term), and D the tokens its text-window steps trained on:
              each stage's updates done x its windows per update x its
              window length, where a step that continues the one before it
              counts only the updates it adds
    loss      the model's held-out-loss evaluation's overall loss: its loss
              per token over every held-out token

A model without that evaluation is named in the output and left out of the
fit; `WPT_EVALUATE=1` in wpt manager's run, or `wpt eval FOLDER
held-out-loss`, makes it. Every model in the fit must have trained on the
same held-out shards, so the losses are of the same text. The fit is least
squares on the losses:

    L = a x C^-b          2 or 3 points
    L = a x C^-b + E      4 points or more; E, the irreducible loss, at least 0

--predict RECIPE (a model recipe's name or SERIES:SIZE) prints the loss
the fit gives for that recipe's planned compute, the same count from its
stages before it trains. Nothing is written.
"""

import argparse
import functools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_dataset.series import flops_per_token, parameters
from wpt_model.notice import EVALS_NAME
from wpt import sessions
from wpt_store import StoreError, store_from_settings
from wpt_store.models import has_model, model_file
from wpt_utils.lines import ERROR, say

HELD_OUT_LOSS = f"{EVALS_NAME}/held-out-loss.json"  # the evaluation whose overall loss is a point's loss
IRREDUCIBLE_FROM = 4  # points from which the fit has an irreducible loss E
EXPONENTS = np.geomspace(1e-4, 4.0, 4001)  # the exponents b searched first, then refined between neighbours
REFINE = 100  # golden-section steps between the best searched exponent's neighbours


class ScalingError(Exception):
    """A model gives no point, the points cannot be fitted, or a recipe cannot be predicted."""


@dataclass(frozen=True)
class Point:
    name: str             # the model folder or the run's name, as given
    recipe: str           # the recipe it was trained from (provenance's recipe_name)
    parameters: int       # without the two vocabulary tables
    tokens: int           # D
    compute: float        # C = M x D
    loss: float


@dataclass(frozen=True)
class Fit:
    a: float
    b: float
    irreducible: float    # E; 0 for a fit without it
    with_irreducible: bool
    points: int

    def loss(self, compute: float) -> float:
        return self.a * compute ** -self.b + self.irreducible

    def formula(self) -> str:
        return (f"L = {self.a:.6g} x C^-{self.b:.6g}"
                + (f" + {self.irreducible:.6g}" if self.with_irreducible else ""))


def per_token(recipe: dict) -> int:
    """M of a recipe's model, at the window length of its last text-window stage."""
    stages = [stage for step in recipe["steps"] for stage in text_stages(step.get("settings", {}).get("stages", []))]
    return flops_per_token(recipe["model"], stages[-1]["block_size"])


def text_stages(stages) -> list[dict]:
    """The stages that read text windows (a window length)."""
    return [stage for stage in stages if stage.get("block_size") is not None]


def new_tokens(steps) -> int:
    """The tokens trained on by steps, each (tokens, continues): the tokens
    it reports from update 1, of which a step that continues the one before
    it counts again that step's, already counted."""
    total, before = 0, 0
    for tokens, continues in steps:
        total += tokens - (before if continues else 0)
        before = tokens
    return total


def point(name: str, provenance: dict, loss: float) -> Point:
    """The point of a model's provenance.json and the overall loss of its held-out-loss evaluation (the module's docstring)."""
    if not [step for step in provenance["steps"] if text_stages(step["stages"])]:
        raise ScalingError(f"{name}: no step read text windows, so it has no held-out loss of text")
    recipe_steps = provenance["recipe"]["steps"]
    tokens = new_tokens([
        (sum((stage["last_step"] - stage["first_step"] + 1) * stage["batch_size"] * stage["block_size"]
             for stage in text_stages(step["stages"]) if stage["last_step"] is not None),
         recipe_steps[number].get("continues", False))
        for number, step in enumerate(provenance["steps"])])
    M = per_token(provenance["recipe"])
    return Point(name, provenance["recipe_name"], parameters(provenance["recipe"]["model"], 0)[0], tokens,
                 float(M) * tokens, float(loss))


def planned(recipe: dict) -> float:
    """The compute a model recipe plans: M x the tokens of its text-window stages, every update done."""
    steps = [(sum(s["steps"] * s["batch_size"] * s["block_size"]
                  for s in text_stages(step.get("settings", {}).get("stages", []))), step.get("continues", False))
             for step in recipe["steps"]]
    if not any(tokens for tokens, _ in steps):
        raise ScalingError("the recipe has no step that reads text windows")
    return float(per_token(recipe)) * new_tokens(steps)


def _solve(compute, loss, b, irreducible):
    """(a, E, the sum of squared residuals) of the best fit at exponent b,
    a linear least-squares problem once b is fixed; E is 0 without
    irreducible, or when the best E would be below 0. The compute is
    divided by its smallest value so the columns stay of similar size; a
    is given back for C itself."""
    scale = compute.min()
    x = (compute / scale) ** -b
    if irreducible:
        (slope, constant), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), loss, rcond=None)
    if not irreducible or constant < 0:
        slope, constant = float(x @ loss / (x @ x)), 0.0
    if slope <= 0:  # a loss that does not fall as compute grows
        return None, None, math.inf
    residual = slope * x + constant - loss
    return float(slope * scale ** b), float(constant), float(residual @ residual)


def fit(compute, loss) -> Fit:
    """The least-squares fit of L = a x C^-b (+ E from IRREDUCIBLE_FROM
    points) to the points: b is searched over EXPONENTS, then refined by
    golden-section search between the best one's neighbours; a and E are
    solved at each b. ScalingError with fewer than 2 points, or losses
    that do not fall as compute grows."""
    compute, loss = np.asarray(compute, dtype=np.float64), np.asarray(loss, dtype=np.float64)
    if len(compute) < 2:
        raise ScalingError(f"{len(compute)} points: a fit needs 2 at least")
    if len(set(compute.tolist())) < 2:
        raise ScalingError("every point has the same compute: a fit needs 2 different ones")
    irreducible = len(compute) >= IRREDUCIBLE_FROM
    errors = [_solve(compute, loss, b, irreducible)[2] for b in EXPONENTS]
    best = int(np.argmin(errors))
    if not math.isfinite(errors[best]):
        raise ScalingError("the losses do not fall as compute grows: no power law with b > 0 fits them")
    low, high = EXPONENTS[max(best - 1, 0)], EXPONENTS[min(best + 1, len(EXPONENTS) - 1)]
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(REFINE):
        left, right = high - ratio * (high - low), low + ratio * (high - low)
        if _solve(compute, loss, left, irreducible)[2] <= _solve(compute, loss, right, irreducible)[2]:
            high = right
        else:
            low = left
    b = (low + high) / 2
    a, constant, _ = _solve(compute, loss, b, irreducible)
    return Fit(a, b, constant, irreducible, len(compute))


def file_of(given: str, path: str, store) -> bytes | None:
    """The bytes of path inside the model folder given, or else inside the
    model the store keeps under the run's name given; None when the model
    holds no such file. store: a function that opens the store, called only
    for a run's name. ScalingError when given is neither a folder nor a
    model of the store."""
    folder = Path(given)
    if (folder / "provenance.json").is_file():
        found = folder / path
        return found.read_bytes() if found.is_file() else None
    store = store()
    if not has_model(store, given):
        raise ScalingError(f"{given}: neither a model folder nor a model session {store.session} keeps")
    return model_file(store, given, path)


def provenance_of(given: str, store) -> dict:
    """The provenance.json of a model folder, or else of the model the store keeps under the run's name given."""
    found = file_of(given, "provenance.json", store)
    if found is None:
        raise ScalingError(f"{given}: the store's model {given} holds no provenance.json")
    return json.loads(found)


def held_out_loss(given: str, store) -> float | None:
    """The overall loss of the model's held-out-loss evaluation; None when it has none (not run, or run
    with no held-out token)."""
    found = file_of(given, HELD_OUT_LOSS, store)
    return None if found is None else json.loads(found)["data"]["loss"]


def points(models, store) -> tuple[list[Point], list[str]]:
    """(the point of each model with a held-out-loss evaluation, the models
    without one, left out); ScalingError when the models of the points were not all
    evaluated on the same held-out shards."""
    found, left_out, held_out = [], [], None
    for given in models:
        loss = held_out_loss(given, store)
        if loss is None:
            left_out.append(given)
            continue
        provenance = provenance_of(given, store)
        shards = [shard["token_sha256"] for shard in provenance["held_out_shards"]]
        if held_out is not None and shards != held_out[1]:
            raise ScalingError(f"{given} and {held_out[0]} hold out other text (their held-out shards differ): "
                               "their losses cannot be compared")
        held_out = held_out or (given, shards)
        found.append(point(given, provenance, float(loss)))
    return found, left_out


def left_out_note(left_out) -> str:
    return (f"left out of the fit, with no {HELD_OUT_LOSS}: {', '.join(left_out)} "
            "(WPT_EVALUATE=1 in wpt manager's run, or `wpt eval FOLDER held-out-loss`, makes it)")


def report(found: list[Point], curve: Fit, predictions: dict[str, float], left_out=()) -> str:
    largest = max(p.compute for p in found)
    lines = [f"{len(found)} points (C = M x tokens trained, M the training compute per token: 6 x the parameters "
             "one token uses + 12 x layers x width x window):"]
    lines += [f"  {p.name} ({p.recipe}): {p.parameters:,} parameters, {p.tokens:,} tokens, C {p.compute:.4g}, "
              f"loss {p.loss:.4f}, fit {curve.loss(p.compute):.4f}" for p in sorted(found, key=lambda p: p.compute)]
    lines.append(f"fit: {curve.formula()}"
                 + ("" if curve.with_irreducible else
                    f" (an irreducible loss is fitted from {IRREDUCIBLE_FROM} points)")
                 + (" (2 points: the curve passes through both, so nothing checks it)" if curve.points == 2 else ""))
    if left_out:
        lines.append(left_out_note(left_out))
    for name, compute in predictions.items():
        lines.append(f"predicted: {name}: C {compute:.4g} ({compute / largest:.3g} x the largest trained), "
                     f"loss {curve.loss(compute):.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wpt scaling", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", metavar="MODEL",
                        help="model folders wpt train wrote, or names of runs whose model the store keeps")
    parser.add_argument("--predict", action="append", default=[], metavar="RECIPE",
                        help="a model recipe's name or SERIES:SIZE whose loss the fit predicts; may repeat")
    sessions.options(parser)
    args = parser.parse_args(argv)
    env_file = Path.cwd() / ".env"
    try:
        # opened only for a run's name
        store = functools.cache(lambda: sessions.opened(store_from_settings(env_file), args, env_file))
        found, left_out = points(args.models, store)
        try:
            curve = fit([p.compute for p in found], [p.loss for p in found])
        except ScalingError as error:
            if left_out:
                raise ScalingError(f"{error}; {left_out_note(left_out)}") from None
            raise
        predictions = {}
        if args.predict:
            repo = Repo.find()
            for name in args.predict:
                predictions[name] = planned(repo.model_recipe(name))
    except (ScalingError, DatasetError, StoreError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return 2
    say(report(found, curve, predictions, left_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
