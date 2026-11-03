"""Model series (docs/model.md, "Model series"): the repository's series
file derives each size's shape and training settings, and every size
shares one dataset."""

import copy
from pathlib import Path

import pytest

from wpt_dataset import series
from wpt_dataset.errors import RuleViolation
from wpt_dataset.repo import Repo, validate

ROOT = Path(__file__).resolve().parents[3]
SERIES = "eldest-titmouse"
WINDOW = 32768


@pytest.fixture(scope="module")
def repo():
    return Repo(ROOT)


@pytest.fixture(scope="module")
def ladder(repo):
    return repo.recipe("series", SERIES)


def test_the_sizes_keep_the_reference_s_width_per_layer_and_scale_every_width_with_it(repo, ladder):
    # widths in multiples of 128, 51.2 of width per layer rounded half up, every width inside
    # the model in docs/model.md's proportion to d (the indexer as V4-Flash's: 64 d / 4096
    # heads and a top 512 d / 4096), every count kept; the global layers every 4th and the
    # last, CSA and HCA in turn; Engram at 20% of the depth, its table size in proportion to the cube of the width
    # (50021 x (d / 512)^3, to the nearest prime)
    layers = {256: 5, 384: 8, 512: 10, 640: 13, 768: 15}
    global_kinds = {5: ["CSA", "HCA"], 8: ["CSA", "HCA"], 10: ["CSA", "HCA", "CSA"],
                    13: ["CSA", "HCA", "CSA", "HCA"], 15: ["CSA", "HCA", "CSA", "HCA"]}
    engram = {5: [1], 8: [2], 10: [2], 13: [3], 15: [3]}
    table_size = {256: 6257, 384: 21101, 512: 50021, 640: 97687, 768: 168803}
    assert series.sizes(ladder) == ["256x5", "384x8", "512x10", "640x13", "768x15"]
    for size in series.sizes(ladder):
        model = series.shape(ladder, size)
        d = model["d_model"]
        assert size == f"{d}x{layers[d]}"
        n = layers[d]
        assert model == {
            "n_layer": n, "global_attention": global_kinds[n], "d_model": d,
            "kda_heads": d // 128, "kda_head_dim": 128,
            "attention_heads": d // 64, "attention_head_dim": 128, "query_rank": d // 4,
            "output_groups": 2, "output_rank": d // 2,
            "index_heads": 64 * d // 4096, "index_head_dim": 128, "index_top_k": 512 * d // 4096,
            "n_experts": 32, "top_k": 4, "expert_width": d // 2, "latent_width": d // 2, "shared_width": d,
            "engram_layers": engram[n], "engram_order": 3, "engram_heads": 8, "engram_dim": 16,
            "engram_table_size": table_size[d],
        }
    # the reference size is interstitial-pecker's model, and every size uses the 32,768-token vocabulary
    assert series.shape(ladder, "512x10") == repo.recipe("model", "interstitial-pecker")["model"]
    assert repo.vocab_size(ladder) == 32768


def test_a_size_s_training_follows_deepseek_llm_s_laws_at_the_compute_its_size_pairs_with(repo, ladder):
    # DeepSeek LLM (arXiv 2401.02954), eq. 1 and 4, written here from the paper: M_opt = 0.1715 C^0.5243
    # gives the C of a size's M = 6 x the parameters a token uses + 12 x layers x width x window, then
    # D_opt = 5.8316 C^0.4757, eta_opt = 0.3118 C^-0.1250 and B_opt = 0.2920 C^0.3271 (tokens)
    plan = series.plan(ladder, "512x10", repo.vocab_size(ladder))
    assert (plan.parameters, plan.total_parameters) == (112_723_864, 146_278_296)  # the model's own count
    # a token uses 44,853,336 parameters (4 of the 32 routed experts of each layer, no Engram table; checked against
    # the model in wpt-training's tests), through 10 layers of width 512, over a window of 32,768 tokens
    M = 6 * 44_853_336 + 12 * 10 * 512 * WINDOW
    C = (M / 0.1715) ** (1 / 0.5243)
    assert plan.flops_per_token == M == 2_282_385_936
    assert plan.compute == pytest.approx(C, rel=1e-12) and C == pytest.approx(2.041e19, rel=1e-3)
    assert plan.tokens == pytest.approx(5.8316 * C ** 0.4757, rel=1e-12)
    assert plan.learning_rate == pytest.approx(0.3118 * C ** -0.125, rel=5e-4)
    assert plan.learning_rate == 0.001203
    assert plan.batch_tokens == pytest.approx(0.2920 * C ** 0.3271, rel=1e-12)
    assert plan.batch_size == 18  # 604,806 tokens: 18.46 windows of 32K
    assert plan.updates == 15_161  # 8.942e9 tokens / (18 x 32,768) = 15,161.2
    assert plan.warmup == 606  # 4% of 15,161 = 606.4
    assert plan.prefix_tokens == 15_161 * 18 * (WINDOW + 1)


def test_the_batch_is_whole_windows_at_least_one_and_the_warm_up_a_share_of_the_updates(ladder):
    def planned(batch_windows, tokens=1000 * WINDOW, warmup_share=0.04):
        # laws of exponent 0 give the batch and the tokens straight away
        changed = copy.deepcopy(ladder)
        changed["laws"]["batch_tokens"] = {"coefficient": batch_windows * WINDOW, "exponent": 0}
        changed["laws"]["tokens"] = {"coefficient": tokens, "exponent": 0}
        changed["warmup_share"] = warmup_share
        return series.plan(changed, "256x5", 32768)

    assert planned(2.5).batch_size == 3  # halves up
    assert planned(2.49).batch_size == 2
    assert planned(0.3).batch_size == 1  # at least one window
    assert planned(2, 2000 * WINDOW).updates == 1000
    assert planned(2, 2000 * WINDOW).warmup == 40
    assert planned(2, 2000 * WINDOW, 0.0125).warmup == 13  # 12.5: halves up
    assert planned(2, 20 * WINDOW).warmup == 1  # 0.4 of 10 updates: at least one


def test_each_size_is_a_model_recipe_and_every_size_shares_the_data_of_the_largest(repo, ladder):
    plans = repo.series_plans(SERIES)
    largest = max(plans, key=lambda plan: plan.parameters)
    assert largest.size == "768x15"
    data = {"mixture": "admitted", "tokenizer": "admitted", "total_tokens": largest.prefix_tokens, "held_out": 0.001}
    for plan in plans:
        recipe = repo.model_recipe(f"{SERIES}:{plan.size}")
        validate(recipe, "model", f"{SERIES}:{plan.size}")
        assert recipe["data"] == data  # one dataset: `wpt dataset build` of any size finds the same one
        assert recipe["model"] == plan.model and recipe["seed"] == ladder["seed"]
        assert recipe["drafter"] == ladder["drafter"]
        [step] = recipe["steps"]
        assert step["step"] == "read-text" and step["locked"] == []
        assert step["optimizer"] == {"name": "by-part", "settings": {"lr": plan.learning_rate, "warmup": plan.warmup}}
        assert step["settings"] == {"stages": [{"block_size": WINDOW, "steps": plan.updates,
                                                "batch_size": plan.batch_size}],
                                    "prefix_tokens": plan.prefix_tokens, "mtp_switch": 0.8}
        assert plan.prefix_tokens <= data["total_tokens"]
    # a plain name is still a model recipe's
    assert repo.model_recipe("interstitial-pecker") == repo.recipe("model", "interstitial-pecker")


def test_a_size_or_a_series_that_does_not_exist_is_refused(repo, ladder):
    with pytest.raises(RuleViolation, match="no size '1024x20' in the series eldest-titmouse"):
        repo.model_recipe(f"{SERIES}:1024x20")
    with pytest.raises(RuleViolation, match="no series recipe named 'nothing'"):
        repo.model_recipe("nothing:256x5")
    odd = copy.deepcopy(ladder)
    odd["sizes"].append({"name": "300x6", "d_model": 300})  # 300 / 512 of 4 KDA heads is not whole
    with pytest.raises(ValueError, match="size 300x6: kda_heads"):
        series.check(odd)
