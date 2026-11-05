"""A decay branch of a series' size trained through wpt train (docs/model.md,
"Model series"): the trunk's steps train once, and a branch goes on from
the trunk's state at its start."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from wpt_dataset.build import build_dataset
from wpt_dataset.repo import Repo
from wpt.train import main as train

# the application tests' tiny model as a series, laws chosen so that C = M and D = M / 2000: the small size reads
# 199 tokens as 3 updates of 4 windows of 16 tokens, after a warm-up of 1; its branches make 2 updates (half, 1.5
# rounded up), decaying after update 1, and 3, decaying after update 2; the large size makes the one dataset hold
# enough documents for the drafter
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
    "branches": [0.5, 1],
    "drafter": {"epochs": 1, "prompts": 4, "prompt_tokens": 8, "answer_tokens": 8, "batch": 2, "anchors": 4,
                "held_out": 0.25},
}


def steps_of(folder):
    """(fingerprint, reused, updates reached) of each step the model folder's provenance lists."""
    provenance = json.loads((Path(folder) / "provenance.json").read_text())
    return [(step["fingerprint"], step["reused"], step["updates"]) for step in provenance["steps"]]


def test_branches_share_the_trunk_and_the_full_branch_trains_as_the_size_s_own_run(built, tmp_path):
    folder = tmp_path / "data" / "recipes" / "series"
    folder.mkdir(parents=True)
    (folder / "tiny.json").write_text(json.dumps(TINY))
    dataset = build_dataset(Repo(tmp_path), "tiny:small:branch-1", built.store, log=lambda *_: None)
    # on the CPU, where the same updates give the same bits
    (tmp_path / "cpu.py").write_text("import torch\nfrom wpt_training import Strategy\n\n"
                                     "def start():\n    return Strategy(torch.device('cpu'))\n")
    cpu = ["--strategy", str(tmp_path / "cpu.py")]
    train(["tiny:small:branch-0.5", dataset, "--out", "half", *cpu])
    half = steps_of("half")
    assert [found[1:] for found in half] == [(False, 1), (False, 2)]
    trunk = half[0][0]
    # the full branch reuses the trunk's first step, goes on to update 2 and decays to update 3: the reused
    # state, then its own, carried on to the next step
    train(["tiny:small:branch-1", dataset, "--out", "full", *cpu])
    assert steps_of("full")[0] == (trunk, True, 1)
    assert [found[1:] for found in steps_of("full")[1:]] == [(False, 2), (False, 3)]
    # the same weights as the size's own run of 3 updates: the same updates at the same rates on the same
    # windows (its switch of the multi-token prediction's weight, after 80% of the updates, is the branch's
    # too: it falls at update 3 in both)
    train(["tiny:small", dataset, "--out", "own", *cpu])
    full, own = load_file("full/model.safetensors"), load_file("own/model.safetensors")
    assert full.keys() == own.keys() and all(torch.equal(full[name], own[name]) for name in own)
