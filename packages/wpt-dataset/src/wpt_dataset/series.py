"""Model series: one file, several sizes of one model (docs/model.md,
"Model series").

A series file (data/recipes/series/<name>.json, read through the service
layer) holds the rules every size shares and the list of sizes, each a
name and a width. Everything else of a size is derived here, by pure
functions of the series document:

    shape         the reference model scaled to the size's width: every
                  width inside the model (WIDTHS) in proportion, every count
                  kept, the layers in the reference's width per layer
                  (rounded half up), the Engram layers at the reference's
                  fractions of depth, the global layers' kinds taken in turn
                  from the series' `global_attention`, Engram's table size
                  in proportion to the cube of the width (the nearest prime)
    parameters    counted as wpt_model builds them, without and with the
                  two vocabulary tables (the embedding and the output head)
    active_parameters
                  the parameters one token uses: without the two vocabulary
                  tables, with only the top_k routed experts of each layer,
                  and without Engram's lookup tables
    plan          the size's training from the series' `laws` (DeepSeek
                  LLM's scaling laws): the non-embedding compute per token
                  M = 6 x the active parameters + 12 x n_layer x d_model x
                  block_size (the paper's attention term), the
                  compute C the model law pairs with M, the tokens D the
                  data law pairs with C, the learning rate and the batch at
                  C; the batch in whole windows (at least 1), the updates
                  that read D tokens (at least 1), the warm-up the series'
                  share of them (at least 1), and the prefix: the tokens its
                  updates read, the first ones of the series' dataset
    model_recipe  the size as a model recipe, the shape of
                  data/recipes/models/<name>.json: the series' data part,
                  its shape, the seed, one read-text step of one stage
                  reading its prefix, the drafter
    data_part     the series' data part: its mixture, tokenizer and held-out
                  share, and the total tokens of the largest prefix, so one
                  dataset serves every size

`model_recipe` builds the recipe from a Plan: a caller that changes what a
run does with a size (dataclasses.replace on the plan's numbers) gets the
recipe of that change, with the same shape and data.

A size has other runs besides its own, each named SERIES:SIZE:RUN
(`runs`, `run_recipe`):

    branch-SHARE  a decay branch, one per share of the series' `branches`:
                  SHARE times the size's updates. Every branch of a size
                  goes on from one trunk, the size's read-text step at its
                  peak rate after the warm-up, with no drop; the trunk is
                  cut into consecutive steps that end at each branch's
                  start, the last update before its first drop, so a
                  branch's recipe is the trunk's steps up to its start,
                  each continuing the one before it, then its decay, which
                  continues the last of them to the branch's updates. Two
                  branches share the trunk's steps they both hold, and
                  wpt-train reuses their kept results: the trunk trains
                  once. The trunk reads the prefix of the size's longest
                  branch, or the size's own when it is longer.
    lr-FACTOR     at the smallest size (the fewest parameters) only, one per
                  factor of the series' `learning_rate_check`: the size's
                  own run at FACTOR times the law's learning rate.
"""

import math
from dataclasses import dataclass, replace
from fractions import Fraction

# what wpt_model fixes for every size, as docs/model.md says (wpt-training's tests check this count against the model)
GLOBAL_EVERY = 4                 # a global layer every 4th layer and the last (wpt_model.config.global_layers)
CSA_RATIO, HCA_RATIO = 4, 128    # tokens merged into one entry (wpt_model.attention)
STREAMS = 4                      # mHC's residual streams
SHARED_EXPERTS = 2
HASH_LAYERS = 3                  # the first 3 layers' experts are hash-routed, without a router
CONV_KERNEL = 4                  # the short convolutions of KDA and Engram

# the reference model's numbers that are widths: they scale with d_model; every other number is a count, kept
WIDTHS = ("kda_heads", "attention_heads", "query_rank", "output_rank", "index_heads", "index_top_k",
          "expert_width", "latent_width", "shared_width")
FLOPS_PER_PARAMETER = 6          # a training token's forward and backward pass through each weight
ATTENTION_FLOPS = 12             # the paper's attention term: 12 x layers x width x window
MTP_SWITCH = 0.8                 # a size's own run switches the multi-token prediction's weight after this share of its updates
FIRST_DROP = 0.8                 # by-part's first_drop (wpt-training's tests check): a branch's decay starts after it
BRANCH, LEARNING_RATE = "branch-", "lr-"  # the names of a size's other runs, SERIES:SIZE:RUN


@dataclass(frozen=True)
class Plan:
    """One size's numbers. parameters: without the two vocabulary tables;
    total_parameters: with them. flops_per_token: M; compute: C; tokens:
    D; batch_tokens: the law's batch, before rounding; learning_rate:
    AdamW's, rounded to 4 significant digits; batch_size: windows per
    update; updates; warmup: updates; prefix_tokens: the first training
    tokens of the series' dataset this size reads."""
    size: str
    model: dict
    parameters: int
    total_parameters: int
    flops_per_token: int
    compute: float
    tokens: float
    batch_tokens: float
    learning_rate: float
    block_size: int
    batch_size: int
    updates: int
    warmup: int
    prefix_tokens: int


def _half_up(value) -> int:
    """value rounded to the nearest whole number, halves up, computed exactly."""
    return math.floor(Fraction(value) + Fraction(1, 2))


def _exact(number) -> Fraction:
    """A JSON number as the decimal it is written as (0.04 is 1/25)."""
    return Fraction(str(number))


def global_layers(n_layer: int) -> list[int]:
    """The numbers (from 1) of the global layers of an n_layer model."""
    return [n for n in range(1, n_layer + 1) if n % GLOBAL_EVERY == 0 or n == n_layer]


def sizes(series: dict) -> list[str]:
    """The series' size names, in its order."""
    return [size["name"] for size in series["sizes"]]


def _size(series: dict, size: str) -> dict:
    for found in series["sizes"]:
        if found["name"] == size:
            return found
    raise ValueError(f"no size {size!r} in the series; its sizes are {', '.join(sizes(series))}")


def shape(series: dict, size: str) -> dict:
    """The size's model numbers (a model recipe's `model`); ValueError when
    a width does not scale to a whole number."""
    reference, width = series["reference"], _size(series, size)["d_model"]
    scale = Fraction(width, reference["d_model"])
    model = dict(reference)
    model["d_model"] = width
    for key in WIDTHS:
        value = reference[key] * scale
        if value.denominator != 1:
            raise ValueError(f"size {size}: {key} {reference[key]} x {width}/{reference['d_model']} is not a whole "
                             "number; choose a width that scales every width of the reference to whole numbers")
        model[key] = int(value)
    model["engram_table_size"] = _nearest_prime(reference["engram_table_size"] * scale ** 3)
    n_layer = max(2, _half_up(reference["n_layer"] * scale))
    model["n_layer"] = n_layer
    depth = Fraction(n_layer, reference["n_layer"])
    model["engram_layers"] = sorted({max(1, _half_up(layer * depth)) for layer in reference["engram_layers"]})
    kinds = series["global_attention"]
    model["global_attention"] = [kinds[i % len(kinds)] for i in range(len(global_layers(n_layer)))]
    if model["attention_heads"] % model["output_groups"]:
        raise ValueError(f"size {size}: {model['attention_heads']} attention heads do not split into "
                         f"{model['output_groups']} output groups")
    return model


def _primes_from(start: int, count: int) -> list[int]:
    """The count smallest primes at or above start (wpt_model.engram's table sizes)."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
        n += 1
    return found


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % p for p in range(2, math.isqrt(n) + 1))


def _nearest_prime(target: Fraction) -> int:
    """The prime nearest to target (the smaller when two are as near; 2 at least)."""
    below, above = math.floor(target), math.ceil(target)
    while below >= 2 and not _is_prime(below):
        below -= 1
    while not _is_prime(above):
        above += 1
    return below if below >= 2 and target - below <= above - target else above


def parameters(model: dict, vocab_size: int) -> tuple[int, int]:
    """(parameters without the two vocabulary tables, every parameter) of
    the model wpt_model builds from these numbers (a model recipe's
    `model`)."""
    count = _count(model, active=False)
    return count, count + 2 * vocab_size * model["d_model"]


def active_parameters(model: dict) -> int:
    """The parameters one token uses, of the model wpt_model builds from
    these numbers: `parameters` without the vocabulary tables, with the
    top_k routed experts of each layer instead of all n_experts, and
    without Engram's lookup tables (the rows read by hash; its projections
    count)."""
    return _count(model, active=True)


def _count(model: dict, active: bool) -> int:
    D, n_layer = model["d_model"], model["n_layer"]
    norm, connection = D, STREAMS * D * (2 * STREAMS + STREAMS ** 2) + 2 * STREAMS + STREAMS ** 2 + 3

    def feed_forward(width, hidden):
        return 3 * width * hidden

    def compressor(width, ratio, parts):
        return 2 * D * parts * width + ratio * parts * width + width

    def attention(ratio):
        H, c, q = model["attention_heads"], model["attention_head_dim"], model["query_rank"]
        g, r = model["output_groups"], model["output_rank"]
        count = D * q + q + q * H * c + D * c + c + H + (H * c // g) * g * r + g * r * D
        if ratio:
            count += compressor(c, ratio, 2 if ratio == CSA_RATIO else 1)
        if ratio == CSA_RATIO:
            n, ci = model["index_heads"], model["index_head_dim"]
            count += q * n * ci + D * n + compressor(ci, CSA_RATIO, 2)
        return count

    def kda():
        H, d = model["kda_heads"], model["kda_head_dim"]
        return 3 * D * H * d + 3 * H * d * CONV_KERNEL + D * H + D * d + d * H * d + H + D * H * d + d + H * d * D

    def experts(hashed):
        L, E, n = model["latent_width"], model["expert_width"], model["n_experts"]
        routed = model["top_k"] if active else n
        count = SHARED_EXPERTS * feed_forward(D, model["shared_width"]) + 2 * D * L + L + 3 * routed * E * L
        return count if hashed else count + D * n

    def engram():
        tables = (model["engram_order"] - 1) * model["engram_heads"]
        width = tables * model["engram_dim"]
        rows = 0 if active else sum(_primes_from(model["engram_table_size"], tables))
        return rows * model["engram_dim"] + (1 + STREAMS) * width * D + 3 * D + D * CONV_KERNEL

    kinds = dict(zip(global_layers(n_layer), model["global_attention"]))
    count = 0
    for number in range(1, n_layer + 1):
        kind = kinds.get(number)
        count += kda() if kind is None else attention(CSA_RATIO if kind == "CSA" else HCA_RATIO)
        count += experts(number <= HASH_LAYERS) + 2 * norm + 2 * connection
        count += engram() if number in model["engram_layers"] else 0
    multi_token = (2 * norm + 2 * D * D + norm + attention(0) + norm
                   + feed_forward(D, SHARED_EXPERTS * model["shared_width"]) + norm)
    return count + norm + multi_token  # the final norm, the multi-token prediction block


def _law(series: dict, name: str, value: float) -> float:
    law = series["laws"][name]
    return law["coefficient"] * value ** law["exponent"]


def flops_per_token(model: dict, block_size: int) -> int:
    """M, the non-embedding compute per token of the model trained on
    windows of block_size tokens: the paper's form with the active
    parameters, 6 x them + 12 x n_layer x d_model x block_size (the
    attention term, whatever the layers attend to)."""
    return (FLOPS_PER_PARAMETER * active_parameters(model)
            + ATTENTION_FLOPS * model["n_layer"] * model["d_model"] * block_size)


def plan(series: dict, size: str, vocab_size: int) -> Plan:
    """The size's numbers (see the module's docstring)."""
    model = shape(series, size)
    counted, total = parameters(model, vocab_size)
    M = flops_per_token(model, series["block_size"])
    model_law = series["laws"]["model_flops"]
    C = (M / model_law["coefficient"]) ** (1 / model_law["exponent"])  # the compute whose optimal M is this M
    D = _law(series, "tokens", C)
    block = series["block_size"]
    batch_tokens = _law(series, "batch_tokens", C)
    batch = max(1, _half_up(batch_tokens / block))
    updates = max(1, _half_up(D / (batch * block)))
    warmup = max(1, _half_up(_exact(series["warmup_share"]) * updates))
    return Plan(size=size, model=model, parameters=counted, total_parameters=total, flops_per_token=M, compute=C,
                tokens=D, batch_tokens=batch_tokens,
                learning_rate=float(f"{_law(series, 'learning_rate', C):.4g}"), block_size=block, batch_size=batch,
                updates=updates, warmup=warmup, prefix_tokens=updates * batch * (block + 1))


def plans(series: dict, vocab_size: int) -> list[Plan]:
    """Every size's plan, in the series' order."""
    return [plan(series, size, vocab_size) for size in sizes(series)]


def data_part(series: dict) -> dict:
    """The data part every size's recipe holds: the series' own, with the
    total tokens of the largest prefix, its trunks' included (the
    vocabulary's size does not change a prefix)."""
    largest = max(_trunk_prefix(series, plan(series, size, 0)) for size in sizes(series))
    data = series["data"]
    return {"mixture": data["mixture"], "tokenizer": data["tokenizer"], "total_tokens": largest,
            "held_out": data["held_out"]}


def _read_text(found: Plan, updates: int, prefix: int, drops=True, continues=False) -> dict:
    """The size's read-text step of `updates` updates reading `prefix`
    tokens; without drops, a trunk: at the peak rate after the warm-up and
    at the early weight of multi-token prediction to its end, every late
    change left to the decay that continues it; continues: it goes on from
    the step before it (docs/training.md, "Continued steps")."""
    optimizer = {"lr": found.learning_rate, "warmup": found.warmup}
    settings = {"stages": [{"block_size": found.block_size, "steps": updates, "batch_size": found.batch_size}],
                "prefix_tokens": prefix}
    if drops:
        settings["mtp_switch"] = MTP_SWITCH
    else:
        optimizer.update(first_drop=1.0, second_drop=1.0)  # after every update: never within the step
        settings["mtp_switch"] = 1.0
    entry = {
        "step": "read-text",
        "optimizer": {"name": "by-part", "settings": optimizer},
        "settings": settings,
        "locked": [],
    }
    if continues:
        entry["continues"] = True
    return entry


def model_recipe(series: dict, found: Plan) -> dict:
    """The model recipe of a plan of this series (a plan from `plan`, or one
    a caller changed)."""
    return {
        "data": data_part(series),
        "model": found.model,
        "seed": series["seed"],
        "steps": [_read_text(found, found.updates, found.prefix_tokens)],
        "drafter": series["drafter"],
    }


@dataclass(frozen=True)
class Branch:
    """A decay branch of a size: its run's name, its share of the size's
    updates, its updates in all (the trunk's it goes on from included), and
    start, the trunk's update it goes on from: the last before its first
    drop."""
    name: str
    share: float
    updates: int
    start: int


def branches(series: dict, found: Plan) -> list[Branch]:
    """The size's decay branches, the shortest first."""
    made = []
    for share in series.get("branches", []):
        updates = max(1, _half_up(_exact(share) * found.updates))
        # the last update by-part runs at the peak: it drops after first_drop x updates, compared as floats
        made.append(Branch(f"{BRANCH}{share}", share, updates, math.floor(FIRST_DROP * updates)))
    return sorted(made, key=lambda branch: branch.updates)


def _trunk_prefix(series: dict, found: Plan) -> int:
    """The tokens the size's trunk reads: its longest branch's, or the size's own when they are more."""
    window = found.batch_size * (found.block_size + 1)
    return max([found.prefix_tokens, *(branch.updates * window for branch in branches(series, found))])


def smallest(series: dict) -> str:
    """The size with the fewest parameters, where the learning-rate check runs."""
    return min(sizes(series), key=lambda size: parameters(shape(series, size), 0)[0])


def runs(series: dict, size: str) -> list[str]:
    """The names of the size's other runs (SERIES:SIZE:RUN), in order: its
    branches, then, at the smallest size, the learning-rate check's."""
    found = [f"{BRANCH}{share}" for share in series.get("branches", [])]
    if size == smallest(series):
        found += [f"{LEARNING_RATE}{factor}" for factor in series.get("learning_rate_check", [])]
    return found


def run_numbers(series: dict, found: Plan) -> list[dict]:
    """What each of the size's other runs does, in the order of `runs`:
    its name, its updates in all, the trunk's update it goes on from (a
    branch's; None for a learning rate of the check), and its peak
    learning rate."""
    made = {branch.name: branch for branch in branches(series, found)}
    found_runs = []
    for run in runs(series, found.size):
        if run in made:
            found_runs.append({"run": run, "updates": made[run].updates, "start": made[run].start,
                               "learning_rate": found.learning_rate})
        else:
            [step] = run_recipe(series, found, run)["steps"]
            found_runs.append({"run": run, "updates": found.updates, "start": None,
                               "learning_rate": step["optimizer"]["settings"]["lr"]})
    return found_runs


def run_recipe(series: dict, found: Plan, run: str = "") -> dict:
    """The model recipe of a run of the plan's size: its own ('') or one
    of `runs`; ValueError for a run the size does not have."""
    if not run:
        return model_recipe(series, found)
    if run not in runs(series, found.size):
        listed = runs(series, found.size)
        raise ValueError(f"no run {run!r} of the size {found.size}; "
                         + (f"its runs are {', '.join(listed)}" if listed else "it has no other run"))
    if run.startswith(LEARNING_RATE):
        factor = next(f for f in series["learning_rate_check"] if f"{LEARNING_RATE}{f}" == run)
        return model_recipe(series, replace(found, learning_rate=float(f"{found.learning_rate * factor:.4g}")))
    made = branches(series, found)
    mine = next(branch for branch in made if branch.name == run)
    prefix = _trunk_prefix(series, found)
    ends = sorted({branch.start for branch in made if branch.start <= mine.start})
    steps = [_read_text(found, end, prefix, drops=False, continues=number > 0) for number, end in enumerate(ends)]
    steps.append(_read_text(found, mine.updates, prefix, continues=True))
    return {**model_recipe(series, found), "steps": steps}


def check(series: dict):
    """ValueError when the series cannot be: size names, branch shares or
    learning-rate factors that repeat, a size whose shape cannot be
    derived, or a branch whose decay would start inside the warm-up or
    has as many updates as another branch of its size."""
    for what, found in (("size names", sizes(series)), ("branch shares", series.get("branches", [])),
                        ("learning-rate factors", series.get("learning_rate_check", []))):
        if len({_exact(value) if not isinstance(value, str) else value for value in found}) != len(found):
            raise ValueError(f"{what} repeat")
    for size in sizes(series):
        shape(series, size)
        found = plan(series, size, 0)  # the vocabulary changes no update
        made = branches(series, found)
        for branch in made:
            if branch.start < found.warmup:
                raise ValueError(f"size {size}: {branch.name} has {branch.updates:,} updates, so its decay would "
                                 f"start after update {branch.start:,}, inside the warm-up of {found.warmup:,}")
        if len({branch.updates for branch in made}) != len(made):
            raise ValueError(f"size {size}: two branches have the same number of updates")
