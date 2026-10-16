import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from wpt_model import load_model, load_tokenizer, tokenizer_identity
from wpt_tokenizer import Tokenizer
from wpt_train import main

ROOT = Path(__file__).resolve().parents[3]


def test_training_writes_a_model_with_its_provenance(built, tmp_path, monkeypatch):
    store, dataset_id = built.store, built.dataset_id
    recipe = built.recipe("three", training={"steps": 3})
    monkeypatch.setenv("WPT_RECIPE", recipe)  # the recipe and the dataset come from the environment too
    monkeypatch.setenv("WPT_DATASET", dataset_id)
    main(["--out", "model"])

    model = load_model(tmp_path / "model")
    assert model.config.n_layer == 4 and model.config.block_size == 16  # the recipe's numbers
    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))
    # the weights name the dataset's vocabulary, and only that file loads with them
    assert tokenizer_identity(tmp_path / "model") == ("t version 1", record["tokenizer"]["vocabulary"])
    tokenizer = Tokenizer.from_bytes(load_tokenizer(tmp_path / "model"))
    assert model.config.vocab_size == tokenizer.vocab_size
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert provenance["dataset"] == dataset_id
    assert provenance["vocabulary"] == record["tokenizer"]["vocabulary"]
    assert provenance["recipe_name"] == "three" and provenance["recipe"]["training"]["steps"] == 3
    assert provenance["steps_trained"] == 3
    assert [s["token_sha256"] for s in provenance["shards"]] == [s["token_sha256"] for s in record["shards"]]


def test_a_trained_model_folder_names_its_licenses_and_sources(built, tmp_path):
    # whoever receives the folder, or only a weight file, must learn the licenses
    # and the credit each source asks for
    dataset_id, sid = built.dataset_id, built.source_id
    main(["d", dataset_id, "--out", "model"])
    folder = tmp_path / "model"

    notice = (folder / "NOTICE").read_text()
    assert "Weights: AGPL-3.0-or-later" in notice and "Generated text: CC0-1.0" in notice
    assert f"{sid}\n" in notice and f"  MIT for **:\n    {built.attribution}\n" in notice
    assert f"License texts\n\n--- {sid} (MIT)\n\n{built.license_text}" in notice  # what MIT asks to travel
    assert notice.endswith((ROOT / "OUTPUT-NOTICE").read_text())
    assert (folder / "LICENSE").read_text() == (ROOT / "LICENSE").read_text()
    card = (folder / "README.md").read_text()
    assert card.startswith("---\nlicense: agpl-3.0\n---\n") and "Weights: AGPL-3.0-or-later" in card
    assert f"| {sid} |" in card and "None yet: run `wpt-eval" in card
    with safe_open(folder / "model.safetensors", "pt") as weights:
        assert weights.metadata()["license"] == "AGPL-3.0-or-later"
        assert weights.metadata()["output_license"] == "CC0-1.0"


def test_a_damaged_shard_stops_training_before_it_starts(built, tmp_path, capsys):
    store, dataset_id = built.store, built.dataset_id
    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))
    shard = record["shards"][0]["object"]
    path = store.root / "objects" / shard[:2] / shard
    path.write_bytes(path.read_bytes()[:-10] + b"0123456789")
    with pytest.raises(SystemExit) as refused:
        main(["d", dataset_id, "--out", "model"])
    assert refused.value.code == 2 and "CorruptObject" in capsys.readouterr().err  # 2: never restarted
    assert not (tmp_path / "model").exists()


def test_a_dataset_built_from_another_recipe_data_is_refused(built, tmp_path, capsys):
    # the model must train on what its recipe names, so a dataset id built
    # from another data part stops the run before it starts
    other = built.recipe("other")
    recipe = json.loads((tmp_path / "data" / "recipes" / "models" / "other.json").read_text())
    recipe["data"]["total_tokens"] = 2000
    (tmp_path / "data" / "recipes" / "models" / "other.json").write_text(json.dumps(recipe))
    with pytest.raises(SystemExit) as refused:
        main([other, built.dataset_id, "--out", "model"])
    assert refused.value.code == 2 and "another data part" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()


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
    recipe = built.recipe("four", training={"steps": 4})
    for out, command in [("one", ["-m", "wpt_train", "--micro-batch", "2"]), ("two", [*starter, "--strategy", "ddp"])]:
        subprocess.run([sys.executable, *command, recipe, built.dataset_id, "--out", out],
                       cwd=tmp_path, env=env, check=True)

    one, two = weights(tmp_path / "one"), weights(tmp_path / "two")
    assert one.keys() == two.keys()
    # gloo adds gradients in varying order: up to 4e-6 apart; other dropout gives 2e-3
    assert all(torch.allclose(one[k], two[k], atol=1e-4) for k in one)
    provenance = json.loads((tmp_path / "two" / "provenance.json").read_text())
    assert provenance["strategy"]["processes"] == 2 and provenance["batch"]["accumulation"] == 1


def interrupting(built, tmp_path, at_step_3):
    """The recipe and options of a 5-step run on the CPU whose method does
    at_step_3 (Python code) at step 3, just before the update."""
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "interrupting.py"
    method.write_text(
        "import os\nimport signal\nimport torch\nfrom wpt_training import Method\n\nSETTINGS = {'lr': 0.01}\n\n"
        "class Interrupting(Method):\n"
        "    def __init__(self, model, settings):\n"
        "        self.optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'])\n\n"
        "    def before_update(self, model, step):\n"
        "        if step == 3:\n"
        f"            {at_step_3}\n\n"
        "def start(model, settings, run):\n    return Interrupting(model, settings)\n")
    return [built.recipe("interrupting", training={"steps": 5, "method": str(method)}), built.dataset_id,
            "--strategy", str(strategy)]


def test_a_crashed_run_resumes_where_its_checkpoint_left_it(built, tmp_path, monkeypatch, capsys):
    # after a crash, --resume must give the same model as a run never interrupted
    options = [*interrupting(built, tmp_path, "if os.environ.get('CRASH'): raise RuntimeError('power cut')"),
               "--checkpoint-every", "2"]
    main([*options, "--out", "whole"])

    monkeypatch.setenv("CRASH", "1")
    with pytest.raises(RuntimeError, match="power cut"):
        main([*options, "--out", "resumed"])
    assert (tmp_path / "resumed" / "checkpoint" / "checkpoint.pt").is_file()
    with pytest.raises(SystemExit) as refused:
        main([*options, "--out", "resumed"])
    assert refused.value.code == 2 and "--resume" in capsys.readouterr().err
    monkeypatch.delenv("CRASH")
    main([*options, "--out", "resumed", "--resume"])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    assert not (tmp_path / "resumed" / "checkpoint").exists()


def test_the_model_card_names_the_steps_trained_and_the_step_of_its_weights(built, tmp_path):
    # a published model says how far it trained and which step its weights
    # come from, even when its method stopped early and kept an earlier step
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "early.py"
    method.write_text(
        "import torch\nfrom wpt_training import Method, Verdict\n\nSETTINGS = {}\n\n"
        "class Early(Method):\n"
        "    def __init__(self, model):\n"
        "        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)\n\n"
        "    def evaluated(self, step, val_loss):\n"
        "        return Verdict(best=step == 2, stop=step == 4)\n\n"
        "def start(model, settings, run):\n    return Early(model)\n")
    recipe = built.recipe("early", training={"steps": 10, "method": str(method)})
    main([recipe, built.dataset_id, "--out", "model", "--strategy", str(strategy)])

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert (provenance["steps_trained"], provenance["weights_step"]) == (4, 2)
    card = (tmp_path / "model" / "README.md").read_text()
    assert "Steps: 4 trained (limit 10); the weights are those of step 2" in card


def test_streamed_training_never_copies_the_whole_dataset_to_disk(built, tmp_path):
    # --stream trains from a few shards at a time, so a machine needs disk
    # only for those, and the model folder names its sources as usual
    store, dataset_id = built.store, built.dataset_id
    main([built.recipe("two", training={"steps": 2}), dataset_id, "--out", "model", "--stream", "--stream-shards", "1"])

    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))
    held_out_only = [p.name for p in (store.scratch / "tokens").glob("*.u32")]
    assert len(held_out_only) == 1  # the held-out tokens; no file of every training shard
    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert set(provenance["sources"]) == {built.source_id}
    assert load_model(tmp_path / "model").config.block_size == 16
    assert record["shards"][0]["sources"] == [built.source_id]


def test_a_stopped_run_saves_its_step_and_continues_on_another_machine(built, tmp_path, monkeypatch):
    # a machine taken back (SIGTERM) or Ctrl+C loses no step; with --run the
    # checkpoint is in the store, so another model folder, as on another machine
    # using the same store, continues it and gives the model of an unbroken run.
    # SIGTERM, not SIGINT: a shell starts background commands with SIGINT
    # ignored, and an ignored signal stays ignored
    options = [*interrupting(built, tmp_path, "if os.environ.get('STOP'): signal.raise_signal(signal.SIGTERM)"),
               "--checkpoint-every", "100"]
    main([*options, "--out", "whole"])

    monkeypatch.setenv("STOP", "1")
    with pytest.raises(SystemExit) as stopped:
        main([*options, "--out", "here", "--run", "fox-1"])
    monkeypatch.delenv("STOP")
    assert stopped.value.code == 130 and not (tmp_path / "here" / "checkpoint").exists()
    saved = built.store.get_ref("checkpoints", "fox-1")
    assert torch.load(built.store.path(saved), weights_only=True)["step"] == 3  # not a multiple of 100: the signal
    main([*options, "--out", "there", "--run", "fox-1", "--resume"])

    whole, there = weights(tmp_path / "whole"), weights(tmp_path / "there")
    assert all(torch.equal(whole[k], there[k]) for k in whole)
    assert built.store.get_ref("checkpoints", "fox-1") is None and not built.store.has(saved)


def test_a_crashed_run_starts_again_on_its_own(built, tmp_path):
    # a long run must survive a crash nobody watches: the processes start again
    # and continue from the checkpoint, without anyone typing --resume
    options = [*interrupting(
        built, tmp_path, "if os.environ.get('CRASH') and os.environ['TORCHELASTIC_RESTART_COUNT'] == '0': "
                         "raise RuntimeError('power cut')"), "--checkpoint-every", "2"]
    env = {**os.environ, **ON_CPU}
    subprocess.run([sys.executable, "-m", "wpt_train", *options, "--out", "whole"],
                   cwd=tmp_path, env=env, check=True)
    run = subprocess.run([sys.executable, "-m", "wpt_train", *options, "--out", "restarted"],
                         cwd=tmp_path, env={**env, "CRASH": "1", "WPT_MAX_RESTARTS": "1"}, capture_output=True,
                         text=True)

    assert run.returncode == 0, run.stderr
    assert "power cut" in run.stderr and "starting every process again (1 of 1 restarts)" in run.stderr
    assert "resumed from step 2" in run.stdout
    whole, restarted = weights(tmp_path / "whole"), weights(tmp_path / "restarted")
    assert all(torch.equal(whole[k], restarted[k]) for k in whole)
