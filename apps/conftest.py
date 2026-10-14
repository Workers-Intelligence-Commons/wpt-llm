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


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A dataset of one git project of 16 documents, about half of them held
    out; its 3000 training tokens draw only some of the training documents.
    The store is WPT_STORE and the working folder is tmp_path. Gives store,
    dataset_id, source_id and attribution."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for i in range(16):
        (upstream / f"t{i}.txt").write_text(f"story {i}: the quick brown fox jumps over the lazy dog. " * 30)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=GIT_ENV, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()

    root = tmp_path / "wpt"
    repo = Repo(root)
    store = open_store((tmp_path / "store").as_uri())
    data = {"schema": 3, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "MIT", "paths": ["**"], "attribution": ATTRIBUTION}],
            "formats": [{"format": "text", "paths": ["**"]}],
            "tags": [{"tags": ["en"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-14", "reason": "test"}}
    data["checksum"] = manifest.checksum(compute_upstream(data, data["address"], store))
    source_id = f"example-fox-{address_hash(str(upstream))}"
    (root / "data" / "sources" / source_id).mkdir(parents=True)
    (root / "data" / "sources" / source_id / "source.json").write_text(dump_json(data))
    for kind, name, document in [("mixture", "m", {"pools": [{"name": "en", "select": "en", "weight": 1}]}),
                                 ("tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 300}),
                                 ("dataset", "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 3000, "held_out": 0.5})]:
        repo.recipe_path(kind, name).parent.mkdir(parents=True, exist_ok=True)
        repo.recipe_path(kind, name).write_text(json.dumps({"schema": CURRENT_SCHEMA[kind], **document}))
    dataset_id = build_dataset(repo, "d", store, log=lambda _: None)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(store=store, dataset_id=dataset_id, source_id=source_id, attribution=ATTRIBUTION)


@pytest.fixture
def trained(built, tmp_path):
    """The built dataset plus tmp_path/model, a tiny model trained 1 step on it."""
    train([built.dataset_id, "--out", "model", "--steps", "1", "--block-size", "16", "--n-layer", "1",
           "--n-head", "2", "--n-embd", "16", "--batch-size", "4"])
    return SimpleNamespace(**vars(built), folder=tmp_path / "model")
