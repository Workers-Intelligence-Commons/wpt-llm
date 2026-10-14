import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

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


SMALL = ["--block-size", "16", "--n-layer", "1", "--n-head", "2", "--n-embd", "16", "--batch-size", "4",
         "--eval-every", "2"]
ON_CPU = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}  # gloo on every system, no GPU needed
STARTERS = {
    "processes": ["-m", "wpt_train", "--processes", "2"],
    "torchrun": pytest.param(["-m", "torch.distributed.run", "--nproc-per-node", "2", "-m", "wpt_train"],
                             marks=pytest.mark.skipif(os.name == "nt", reason="torchrun does not start on Windows "
                                                                              "with PyTorch 2.14 (docs/training.md)")),
}


def weights(folder):
    return load_file(folder / "model.safetensors")


@pytest.mark.parametrize("starter", STARTERS.values(), ids=STARTERS.keys())
def test_two_processes_train_the_same_model_as_one(built, tmp_path, starter):
    # the batch is global: 2 processes of 2 windows each, or 1 process adding up
    # 2 passes of 2 windows, must take the same steps, dropout included
    env = {**os.environ, **ON_CPU}
    for out, command in [("one", ["-m", "wpt_train", "--micro-batch", "2"]), ("two", [*starter, "--strategy", "ddp"])]:
        subprocess.run([sys.executable, *command, built.dataset_id, "--out", out, "--steps", "4", *SMALL],
                       cwd=tmp_path, env=env, check=True)

    one, two = weights(tmp_path / "one"), weights(tmp_path / "two")
    assert one.keys() == two.keys()
    # gloo adds gradients in varying order: up to 4e-6 apart; other dropout gives 2e-3
    assert all(torch.allclose(one[k], two[k], atol=1e-4) for k in one)
    provenance = json.loads((tmp_path / "two" / "provenance.json").read_text())
    assert provenance["strategy"]["processes"] == 2 and provenance["batch"]["accumulation"] == 1


def test_a_crashed_run_resumes_where_its_checkpoint_left_it(built, tmp_path, monkeypatch):
    # after a crash, --resume must give the same model as a run never interrupted
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "crashing.py"
    method.write_text(
        "import os\nimport torch\nfrom wpt_training import Method\n\nSETTINGS = {'lr': 0.01}\n\n"
        "class Crashing(Method):\n"
        "    def __init__(self, model, settings):\n"
        "        self.optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'])\n\n"
        "    def before_update(self, model, step):\n"
        "        if step == 3 and os.environ.get('CRASH'):\n"
        "            raise RuntimeError('power cut')\n\n"
        "def start(model, settings, run):\n    return Crashing(model, settings)\n")
    options = ["--steps", "5", "--checkpoint-every", "2", "--strategy", str(strategy), "--method", str(method), *SMALL]
    main([built.dataset_id, "--out", "whole", *options])

    monkeypatch.setenv("CRASH", "1")
    with pytest.raises(RuntimeError, match="power cut"):
        main([built.dataset_id, "--out", "resumed", *options])
    assert (tmp_path / "resumed" / "checkpoint" / "checkpoint.pt").is_file()
    with pytest.raises(SystemExit, match="--resume"):
        main([built.dataset_id, "--out", "resumed", *options])
    monkeypatch.delenv("CRASH")
    main([built.dataset_id, "--out", "resumed", "--resume", *options])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    assert not (tmp_path / "resumed" / "checkpoint").exists()
