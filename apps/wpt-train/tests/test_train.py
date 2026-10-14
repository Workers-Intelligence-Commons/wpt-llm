import json
from pathlib import Path

import pytest
from safetensors import safe_open

from wpt_model import load_model
from wpt_tokenizer import Tokenizer
from wpt_train import main

ROOT = Path(__file__).resolve().parents[3]


def test_training_writes_a_model_with_its_provenance(built, tmp_path, monkeypatch):
    store, dataset_id = built.store, built.dataset_id
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
    dataset_id, sid = built.dataset_id, built.source_id
    main([dataset_id, "--out", "model", "--steps", "1", "--block-size", "16", "--n-layer", "1",
          "--n-head", "2", "--n-embd", "16", "--batch-size", "4"])
    folder = tmp_path / "model"

    notice = (folder / "NOTICE").read_text()
    assert "Weights: AGPL-3.0-or-later" in notice and "Generated text: CC0-1.0" in notice
    assert f"{sid}\n" in notice and f"  MIT for **:\n    {built.attribution}\n" in notice
    assert notice.endswith((ROOT / "OUTPUT-NOTICE").read_text())
    assert (folder / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
    card = (folder / "README.md").read_text()
    assert card.startswith("---\nlicense: agpl-3.0\n---\n") and "Weights: AGPL-3.0-or-later" in card
    assert f"| {sid} |" in card and "None yet: run `wpt-eval" in card
    with safe_open(folder / "model.safetensors", "pt") as weights:
        assert weights.metadata()["license"] == "AGPL-3.0-or-later"
        assert weights.metadata()["output_license"] == "CC0-1.0"


def test_a_damaged_shard_stops_training_before_it_starts(built, tmp_path):
    store, dataset_id = built.store, built.dataset_id
    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))
    shard = record["shards"][0]["object"]
    path = store.root / "objects" / shard[:2] / shard
    path.write_bytes(path.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(SystemExit, match="CorruptObject"):
        main([dataset_id, "--out", "model", "--steps", "1"])
    assert not (tmp_path / "model").exists()
