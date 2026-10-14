import json
import os
import subprocess
from pathlib import Path

import pytest
from safetensors import safe_open

from wpt_dataset import manifest
from wpt_dataset.build import build_dataset
from wpt_dataset.repo import Repo, address_hash, dump_json
from wpt_dataset.sources import compute_upstream
from wpt_model import load_model
from wpt_store import open_store
from wpt_tokenizer import Tokenizer
from wpt_train import main

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}
ROOT = Path(__file__).resolve().parents[3]
ATTRIBUTION = "Fox Stories, copyright (c) 2026 Ada Writer, https://example.org/fox"


@pytest.fixture
def built(tmp_path, monkeypatch):
    """(store, dataset id) built from a small git project through wpt-dataset."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for i in range(8):
        (upstream / f"t{i}.txt").write_text(f"story {i}: the quick brown fox jumps over the lazy dog. " * 30)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=GIT_ENV, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()

    root = tmp_path / "wpt"
    repo = Repo(root)
    store = open_store((tmp_path / "store").as_uri())
    data = {"schema": 2, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "MIT", "paths": ["**"], "attribution": ATTRIBUTION}],
            "tags": [{"tags": ["en"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-14", "reason": "test"}}
    data["checksum"] = manifest.checksum(compute_upstream(data, data["address"], store))
    folder = root / "data" / "sources" / f"example-fox-{address_hash(str(upstream))}"
    folder.mkdir(parents=True)
    (folder / "source.json").write_text(dump_json(data))
    for kind, name, document in [("mixture", "m", {"pools": [{"name": "en", "select": "en", "weight": 1}]}),
                                 ("tokenizer", "t", {"mixture": "m", "vocab_size": 300}),
                                 ("dataset", "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 3000, "held_out": 0.25})]:
        repo.recipe_path(kind, name).parent.mkdir(parents=True, exist_ok=True)
        repo.recipe_path(kind, name).write_text(json.dumps({"schema": 1, **document}))
    dataset_id = build_dataset(repo, "d", store, log=lambda _: None)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.chdir(tmp_path)
    return store, dataset_id


def test_training_writes_a_model_with_its_provenance(built, tmp_path, monkeypatch):
    store, dataset_id = built
    monkeypatch.setenv("WPT_N_LAYER", "1")  # settings come from the environment too
    main([dataset_id, "--out", "model", "--steps", "3", "--eval-every", "2", "--block-size", "16",
          "--n-head", "2", "--n-embd", "16", "--batch-size", "4"])

    model = load_model(tmp_path / "model")
    assert model.config.n_layer == 1 and model.config.block_size == 16
    tokenizer = Tokenizer.load(tmp_path / "model" / "tokenizer.json")
    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))
    assert tokenizer.to_bytes() == store.get_bytes(record["tokenizer"]["vocabulary"])
    assert model.config.vocab_size == tokenizer.vocab_size
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["dataset"] == dataset_id
    assert provenance["vocabulary"] == record["tokenizer"]["vocabulary"]
    assert provenance["options"]["n_layer"] == 1 and provenance["options"]["steps"] == 3
    assert [s["token_sha256"] for s in provenance["shards"]] == [s["token_sha256"] for s in record["shards"]]


def test_a_trained_model_folder_names_its_licenses_and_sources(built, tmp_path):
    # whoever receives the folder, or only a weight file, must learn the licenses
    # and the credit each source asks for
    store, dataset_id = built
    main([dataset_id, "--out", "model", "--steps", "1", "--block-size", "16", "--n-layer", "1",
          "--n-head", "2", "--n-embd", "16", "--batch-size", "4"])
    folder = tmp_path / "model"
    sid = f"example-fox-{address_hash(str(tmp_path / 'upstream'))}"

    notice = (folder / "NOTICE").read_text()
    assert "Weights: AGPL-3.0-or-later" in notice and "Generated text: CC0-1.0" in notice
    assert f"{sid}\n" in notice and f"  MIT for **:\n    {ATTRIBUTION}\n" in notice
    assert notice.endswith((ROOT / "OUTPUT-NOTICE").read_text())
    assert (folder / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
    card = (folder / "README.md").read_text()
    assert card.startswith("---\nlicense: agpl-3.0\n---\n") and "Weights: AGPL-3.0-or-later" in card
    assert f"| {sid} |" in card and "None yet: run `wpt-eval" in card
    with safe_open(folder / "model.safetensors", "pt") as weights:
        assert weights.metadata()["license"] == "AGPL-3.0-or-later"
        assert weights.metadata()["output_license"] == "CC0-1.0"


def test_a_damaged_shard_stops_training_before_it_starts(built, tmp_path):
    store, dataset_id = built
    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))
    shard = record["shards"][0]["object"]
    path = store.root / "objects" / shard[:2] / shard
    path.write_bytes(path.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(SystemExit, match="CorruptObject"):
        main([dataset_id, "--out", "model", "--steps", "1"])
    assert not (tmp_path / "model").exists()
