"""A size of a model series built and trained through the commands
(docs/model.md, "Model series"): SERIES:SIZE goes wherever a model
recipe's name goes."""

import json
from pathlib import Path

from wpt_dataset import build as dataset_build
from wpt_dataset.build import build_dataset, last_dataset
from wpt_dataset.repo import Repo
from wpt.train import main as train

# a series of two sizes of the application tests' tiny model; laws chosen so that C = M and D = M / 2000:
# the small size (66,266 parameters, M 397,596) reads 199 tokens as 3 updates of 4 windows of 16 tokens, the
# large one (314,548, M 1,887,288) 944 tokens as 15 updates
TINY = {
    "data": {"mixture": "m", "tokenizer": "t", "held_out": 0.5},
    "reference": {"n_layer": 4, "d_model": 16, "kda_heads": 2, "kda_head_dim": 8, "attention_heads": 2,
                  "attention_head_dim": 128, "query_rank": 8, "output_groups": 1, "output_rank": 8, "index_heads": 2,
                  "index_head_dim": 128, "index_top_k": 2, "n_experts": 4, "top_k": 2, "expert_width": 8,
                  "latent_width": 8, "shared_width": 16, "engram_layers": [2], "engram_order": 3, "engram_heads": 2,
                  "engram_dim": 4, "engram_table_size": 31},
    "global_attention": ["CSA", "HCA"],
    "block_size": 16,
    "laws": {"learning_rate": {"coefficient": 0.002, "exponent": 0}, "batch_tokens": {"coefficient": 64, "exponent": 0},
             "model_flops": {"coefficient": 1, "exponent": 1}, "tokens": {"coefficient": 0.0005, "exponent": 1}},
    "warmup_share": 0.04,
    "seed": 1337,
    "sizes": [{"name": "small", "d_model": 16}, {"name": "large", "d_model": 32}],
    "drafter": {"epochs": 1, "prompts": 4, "prompt_tokens": 8, "answer_tokens": 8, "batch": 2, "anchors": 4,
                "held_out": 0.25},
}


def test_every_size_of_a_series_builds_one_dataset_and_a_size_trains_on_its_start(built, tmp_path):
    folder = tmp_path / "data" / "recipes" / "series"
    folder.mkdir(parents=True)
    (folder / "tiny.json").write_text(json.dumps(TINY))
    repo = Repo(tmp_path)
    small = build_dataset(repo, "tiny:small", built.store, log=lambda _: None)
    assert build_dataset(repo, "tiny:large", built.store, log=lambda _: None) == small  # one build for the series
    record = json.loads(built.store.get_bytes(built.store.get_ref("datasets", small)))
    assert record["recipe"]["total_tokens"] == 15 * 4 * 17  # the large size's prefix
    train(["tiny:small", small, "--out", "model"])
    provenance = json.loads((Path("model") / "provenance.json").read_text())
    assert provenance["recipe_name"] == "tiny:small"
    assert provenance["recipe"] == repo.model_recipe("tiny:small")
    [step] = provenance["recipe"]["steps"]
    assert step["settings"]["stages"] == [{"block_size": 16, "steps": 3, "batch_size": 4}]
    assert step["settings"]["prefix_tokens"] == 3 * 4 * 17
    assert step["optimizer"]["settings"] == {"lr": 0.002, "warmup": 1}
    assert provenance["model"]["n_layer"] == 4 and provenance["model"]["d_model"] == 16


def test_a_build_names_its_dataset_for_every_size_and_run_of_the_series(built, tmp_path):
    folder = tmp_path / "data" / "recipes" / "series"
    folder.mkdir(parents=True)
    (folder / "tiny.json").write_text(json.dumps(TINY))
    assert last_dataset(built.store, "tiny:small") is None
    built_id = dataset_build.run(Repo(tmp_path), "tiny:small", built.store, ["build", "tiny:small"])
    assert last_dataset(built.store, "tiny:large") == built_id
    assert last_dataset(built.store, "tiny:large:lr-0.5") == built_id
    assert last_dataset(built.store, "tiny") is None  # a model recipe of that name is another thing
