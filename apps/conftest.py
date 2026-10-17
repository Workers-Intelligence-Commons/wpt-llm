"""Shared setup for the application tests: a dataset built through
wpt-dataset from a real git project, and a tiny model trained on it through
wpt-train. Test files reach them as fixtures (they cannot import this file
under --import-mode=importlib)."""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from wpt_dataset import manifest
from wpt_dataset.build import build_dataset
from wpt_dataset.repo import CURRENT_SCHEMA, Repo, address_hash, dump_json
from wpt_dataset.sources import compute_upstream
from wpt_store import open_store
from wpt_train import main as train

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}
ATTRIBUTION = "Fox Stories, copyright (c) 2026 Ada Writer, https://example.org/fox"
LICENSE_TEXT = "Copyright (c) 2026 Ada Writer\n\nPermission is hereby granted, free of charge, to any person.\n"


TINY_MODEL = {"block_size": 16, "n_layer": 4, "global_attention": ["CSA"], "d_model": 16, "kda_heads": 2,
              "kda_head_dim": 8, "attention_heads": 2, "attention_head_dim": 128, "query_rank": 8, "output_groups": 1,
              "output_rank": 8, "index_heads": 2, "index_head_dim": 128, "index_top_k": 2, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
              "shared_width": 16, "engram_layers": [2], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
              "engram_table_size": 31}
ONE_STEP = {"method": "muon", "settings": {}, "steps": 1, "batch_size": 4, "eval_every": 2, "eval_batches": 20,
            "seed": 1337}


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A dataset of one git project of 16 documents, about half of them held
    out; its 3000 training tokens draw only some of the training documents.
    The working folder tmp_path is the repository, and the store is
    WPT_STORE. Gives store, dataset_id (built from the model recipe "d", a
    tiny model trained one step), source_id, attribution, and
    recipe(name, model=..., training=..., posttrain=...), which writes a
    model recipe of the same data part, with those changes to "d", and
    returns its name."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for i in range(16):
        (upstream / f"t{i}.txt").write_text(f"story {i}: the quick brown fox jumps over the lazy dog. " * 30)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=GIT_ENV, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()

    root = tmp_path
    repo = Repo(root)
    store = open_store((tmp_path / "store").as_uri())
    data = {"schema": 5, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "MIT", "paths": ["**"], "attribution": ATTRIBUTION, "text": LICENSE_TEXT}],
            "formats": [{"format": "text", "paths": ["**"]}],
            "tags": [{"tags": ["en"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-14", "reason": "test"}}
    data["checksum"] = manifest.checksum(compute_upstream(data, data["address"], store))
    source_id = f"example-fox-{address_hash(str(upstream))}"
    (root / "data" / "sources" / source_id).mkdir(parents=True)
    (root / "data" / "sources" / source_id / "source.json").write_text(dump_json(data))
    def write(kind, name, document):
        repo.recipe_path(kind, name).parent.mkdir(parents=True, exist_ok=True)
        repo.recipe_path(kind, name).write_text(json.dumps({"schema": CURRENT_SCHEMA[kind], **document}))
        return name

    def recipe(name="d", model=None, training=None, posttrain=()):
        return write("model", name, {"data": {"mixture": "m", "tokenizer": "t", "total_tokens": 3000, "held_out": 0.5},
                                     "model": {**TINY_MODEL, **(model or {})},
                                     "training": {**ONE_STEP, **(training or {})}, "posttrain": list(posttrain)})

    write("mixture", "m", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    write("tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    dataset_id = build_dataset(repo, recipe(), store, log=lambda _: None)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.setenv("WPT_MAX_RESTARTS", "0")  # wpt-train runs in the test's own process, unless a test says
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(store=store, dataset_id=dataset_id, source_id=source_id, attribution=ATTRIBUTION,
                           license_text=LICENSE_TEXT, recipe=recipe)


@pytest.fixture
def trained(built, tmp_path):
    """The built dataset plus tmp_path/model, trained by the recipe "d"."""
    train(["d", built.dataset_id, "--out", "model"])
    return SimpleNamespace(**vars(built), folder=tmp_path / "model")
