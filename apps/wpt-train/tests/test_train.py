import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from wpt_model import load_model, load_tokenizer, tokenizer_identity
import wpt_telemetry
from wpt_store import LocalStore
from wpt_telemetry import FLUSH_THREAD
from wpt_tokenizer import Tokenizer
from wpt_train import main

ROOT = Path(__file__).resolve().parents[3]


def test_training_writes_a_model_with_its_provenance(built, tmp_path, monkeypatch, capsys):
    store, dataset_id = built.store, built.dataset_id
    recipe = built.recipe("three", training={"steps": 3})
    monkeypatch.setenv("WPT_RECIPE", recipe)  # the recipe and the dataset come from the environment too
    monkeypatch.setenv("WPT_DATASET", dataset_id)
    main(["--out", "model"])

    assert "| indexer loss " in capsys.readouterr().out  # beside the training loss, at each evaluation
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
    assert isinstance(provenance["indexer_loss"], float)  # the CSA indexers' loss at the kept step
    assert [s["token_sha256"] for s in provenance["shards"]] == [s["token_sha256"] for s in record["shards"]]


def test_a_training_run_records_what_it_read_every_step_and_what_it_wrote(built, tmp_path, capsys):
    # afterwards one must know exactly what happened, from the store alone:
    # the dataset read, each step's loss, the evaluations, the files written
    main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    printed = capsys.readouterr().out

    [run] = built.records("wpt-train")
    assert run.end["status"] == "succeeded" and run.end["exit_code"] == 0
    assert run.ids["inputs"]["recipe"] == "three" and run.ids["inputs"]["dataset"] == built.dataset_id
    assert f"-wpt-train-{built.dataset_id[:8]}-" in run.folder  # its folder is found by the dataset it trained on
    steps = [e for e in run.events if e.get("name") == "training.step"]
    assert [e["attributes"]["step"] for e in steps] == [1, 2, 3]
    assert all(math.isfinite(e["values"]["train_loss"]) for e in steps)
    evaluations = [e["attributes"]["step"] for e in run.events if e.get("name") == "training.evaluation"]
    assert evaluations == [1, 2, 3]  # eval_every 2, and the first and last steps
    folder = tmp_path / "model"
    weights = hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()
    assert run.ids["outputs"]["weights"] == {"model.safetensors": weights}
    [files] = [e["attributes"]["files"] for e in run.events if e.get("name") == "model.files"]
    assert files["NOTICE"] == hashlib.sha256((folder / "NOTICE").read_bytes()).hexdigest()
    assert {"LICENSE", "README.md", "provenance.json", "tokenizer.json", "config.json"} <= set(files)
    logged = "".join(e["text"] + "\n" for e in run.events if e["kind"] == "log")
    assert logged == printed  # every line it printed, nothing else


def test_a_record_the_store_refuses_stops_the_run_once_its_checkpoint_is_saved(built, tmp_path, monkeypatch, capsys):
    # without its record a run must do nothing more than keep its work: it
    # finishes the step and saves its checkpoint, takes no further step,
    # writes no model, and says why
    real = LocalStore._raw_append_record

    def disk_full_at_step_3(self, folder, name, data):
        if b'"name": "training.step"' in data and b'"step": 3,' in data:
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full_at_step_3)
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("five", training={"steps": 5}), built.dataset_id, "--out", "model"])

    assert stopped.value.code == 3 and "No space left on device" in capsys.readouterr().err
    [run] = built.records("wpt-train")
    assert [e["attributes"]["step"] for e in run.events if e.get("name") == "training.step"] == [1, 2]
    assert run.end is None and [path.name for path in (tmp_path / "model").iterdir()] == ["checkpoint"]
    assert torch.load(tmp_path / "model" / "checkpoint" / "checkpoint.pt", weights_only=True)["step"] == 3


def test_a_send_refused_in_the_background_keeps_the_step_in_the_store_checkpoint(built, tmp_path, monkeypatch, capsys):
    # a record that breaks while the process trains (here a send refused in
    # the background, which arrives as a stop signal) must not cost the steps
    # done: the step ends, the store's checkpoint holds it, unrecorded, and
    # --resume gives the model of an unbroken run. The checkpoint it replaced
    # is not deleted, since its deletion could not be recorded first
    real = LocalStore._raw_flush_records

    def refused_in_the_background(self):
        if threading.current_thread().name == FLUSH_THREAD and os.environ.get("REFUSE") == "now":
            raise OSError(28, "No space left on device")
        real(self)

    monkeypatch.setattr(wpt_telemetry, "FLUSH_SECONDS", 0.02)
    monkeypatch.setattr(LocalStore, "_raw_flush_records", refused_in_the_background)
    options = [*interrupting(built, tmp_path, "if os.environ.get('REFUSE'): os.environ['REFUSE'] = 'now'; "
                                              "__import__('time').sleep(2)"), "--checkpoint-every", "1"]
    main([*options, "--out", "whole"])

    monkeypatch.setenv("REFUSE", "1")
    with pytest.raises(SystemExit) as stopped:
        main([*options, "--out", "here", "--run", "fox-1"])
    monkeypatch.delenv("REFUSE")
    assert stopped.value.code == 3 and "No space left on device" in capsys.readouterr().err
    saved = built.store.get_ref("checkpoints", "fox-1")
    assert torch.load(built.store.path(saved), weights_only=True)["step"] == 3
    run = built.records("wpt-train")[-1]
    named = [e["attributes"]["object"] for e in run.events
             if e.get("name") == "store.ref_set" and e["attributes"]["ref"] == "checkpoints/fox-1"]
    assert len(named) == 2 and saved not in named and built.store.has(named[-1])  # steps 1 and 2; 2's object stays

    main([*options, "--out", "there", "--run", "fox-1", "--resume"])
    whole, there = weights(tmp_path / "whole"), weights(tmp_path / "there")
    assert all(torch.equal(whole[k], there[k]) for k in whole)
    assert built.store.get_ref("checkpoints", "fox-1") is None and built.store.has(named[-1])


def test_training_into_a_model_folder_replaces_it_whole_and_only_once_recorded(built, tmp_path, monkeypatch):
    # the folder appears only after the record names its files, so a record
    # that fails leaves the earlier folder as it was; a run that succeeds
    # replaces it as a whole, results of other weights and stray files included
    recipe = built.recipe("two", training={"steps": 2})
    main([recipe, built.dataset_id, "--out", "model"])
    folder = tmp_path / "model"
    (folder / "evals").mkdir()
    (folder / "evals" / "memorization.json").write_text("{}\n")
    (folder / "notes.txt").write_text("mine\n")
    before = {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}

    real = LocalStore._raw_append_record

    def disk_full_at_the_files(self, record, name, data):
        if b'"name": "model.files"' in data:
            raise OSError(28, "No space left on device")
        real(self, record, name, data)

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "_raw_append_record", disk_full_at_the_files)
        with pytest.raises(SystemExit) as stopped:
            main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    assert stopped.value.code == 3
    assert {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()} == before
    assert (folder / "evals" / "memorization.json").exists()

    main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    assert not (folder / "evals").exists() and not (folder / "notes.txt").exists()
    assert json.loads((folder / "provenance.json").read_text())["recipe_name"] == "three"
    run = built.records("wpt-train")[-1]
    assert run.ids["outputs"]["weights"] == {
        "model.safetensors": hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()}
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []  # nothing left beside it


def test_on_s3_the_model_folder_is_replaced_only_once_its_record_is_in_the_bucket(built, s3, tmp_path):
    # S3 holds record lines back: the lines naming the new files must be in
    # the bucket before the earlier folder goes, so a refused send leaves it
    # as it was
    s3.copy(built.store)
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "notes.txt").write_text("mine\n")
    s3.refuse(b'"training.checkpoint_removed"')
    with pytest.raises(SystemExit) as stopped:
        main(["d", built.dataset_id, "--out", "model"])
    assert stopped.value.code == 3
    assert [path.name for path in folder.iterdir()] == ["notes.txt"]
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []


def test_every_checkpoint_deleted_from_the_store_is_named_in_the_record_first(built, tmp_path, monkeypatch):
    # an object leaves the store only once the record says which one went:
    # each replaced checkpoint, and the last one once the model is saved
    real, deleted = LocalStore.delete, []

    def delete(self, sha):
        [run] = built.records("wpt-train")
        named = [e["attributes"]["object"] for e in run.events if e.get("name") == "training.checkpoint_deleted"]
        deleted.append((sha, sha in named))
        real(self, sha)

    monkeypatch.setattr(LocalStore, "delete", delete)
    main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model", "--checkpoint-every", "1",
          "--run", "fox-1"])
    assert len({sha for sha, _ in deleted}) == 3 and all(named for _, named in deleted)
    assert built.store.get_ref("checkpoints", "fox-1") is None


def test_on_s3_a_checkpoint_is_deleted_only_once_its_deletion_is_in_the_bucket(built, s3, tmp_path, monkeypatch):
    # S3 holds record lines back: a deletion the bucket did not record must
    # not happen, so the previous checkpoint stays
    from wpt_store import S3Store

    s3.copy(built.store)
    deleted = []
    monkeypatch.setattr(S3Store, "delete", lambda self, sha: deleted.append(sha))
    s3.refuse(b'"training.checkpoint_deleted"')
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model",
              "--checkpoint-every", "1", "--run", "fox-1"])
    assert stopped.value.code == 3 and deleted == []
    assert s3.store.get_ref("checkpoints", "fox-1") is not None


@pytest.mark.parametrize("error", [PermissionError(13, "Permission denied"), KeyboardInterrupt()],
                         ids=["refused", "interrupted"])
def test_an_earlier_folder_that_cannot_be_removed_is_left_beside_with_a_warning(built, tmp_path, monkeypatch, capsys,
                                                                                error):
    # the new model is in place and complete: an earlier folder that the
    # system will not remove must not fail the run, but must be named; a
    # Ctrl+C during its removal names it the same way, and still stops
    main(["d", built.dataset_id, "--out", "model"])
    real = shutil.rmtree

    def refuse_the_replaced_one(path, *args, **options):
        if "-replaced-" in Path(path).name:
            raise error
        return real(path, *args, **options)

    monkeypatch.setattr(shutil, "rmtree", refuse_the_replaced_one)
    with pytest.raises(KeyboardInterrupt) if isinstance(error, KeyboardInterrupt) else contextlib.nullcontext():
        main(["d", built.dataset_id, "--out", "model"])

    [left] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-replaced-")]
    assert (left / "model.safetensors").exists() and (tmp_path / "model" / "model.safetensors").exists()
    run = built.records("wpt-train")[-1]
    assert run.end["exit_code"] == (130 if isinstance(error, KeyboardInterrupt) else 0)
    assert [e["attributes"]["folder"] for e in run.events if e.get("name") == "model.replaced_folder_left"] == [
        left.name]  # as --out named the folder: relative to the working folder
    assert left.name in capsys.readouterr().err


def test_an_earlier_folder_is_put_back_when_the_new_one_cannot_take_its_place(built, tmp_path, monkeypatch):
    # moving the earlier folder aside and the new one in are one step: when
    # the second fails, the target must be the earlier folder again
    main(["d", built.dataset_id, "--out", "model"])
    before = {path.name: path.read_bytes() for path in (tmp_path / "model").iterdir() if path.is_file()}
    real = os.replace

    def refuse_the_new_one(source, target):
        if Path(target).name == "model" and not Path(source).name.startswith(".model-replaced-"):
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_new_one)
    with pytest.raises(PermissionError):
        main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    assert {path.name: path.read_bytes() for path in (tmp_path / "model").iterdir() if path.is_file()} == before
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []


def test_an_earlier_folder_that_cannot_be_put_back_is_named_with_the_new_one(built, tmp_path, monkeypatch):
    # when the earlier folder cannot be moved back either, the target is
    # missing: neither model may be deleted, and the error names both
    from wpt_train import RestoreFailed

    main(["d", built.dataset_id, "--out", "model"])
    earlier = (tmp_path / "model" / "model.safetensors").read_bytes()
    real = os.replace

    def refuse_every_move_into_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_every_move_into_the_target)
    with pytest.raises(RestoreFailed) as failed:
        main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    assert not (tmp_path / "model").exists() and str(failed.value.aside) in str(failed.value)
    assert (failed.value.aside / "model.safetensors").read_bytes() == earlier
    assert json.loads((failed.value.new / "provenance.json").read_text())["recipe_name"] == "three"


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
    # one command, one record: both attempts and the restart between them
    record = built.records("wpt-train")[-1]
    assert record.start["arguments"][-2:] == ["--out", "restarted"] and record.end["status"] == "succeeded"
    assert [e["attributes"]["step"] for e in record.events if e.get("name") == "training.step"] == [1, 2, 3, 4, 5]
    assert any("starting every process again" in e.get("text", "") for e in record.events)
    assert record.ids["outputs"]["weights"] == {
        "model.safetensors": hashlib.sha256((tmp_path / "restarted" / "model.safetensors").read_bytes()).hexdigest()}


def test_a_new_model_that_cannot_take_an_empty_place_is_kept_and_named(built, tmp_path, monkeypatch, capsys):
    # with no earlier folder, a refused rename must not delete the only copy of the new model
    real = os.replace

    def refuse_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_target)
    with pytest.raises(PermissionError):
        main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    [kept] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-")]
    assert not (tmp_path / "model").exists()
    assert json.loads((kept / "provenance.json").read_text())["recipe_name"] == "three"
    assert str(kept) in capsys.readouterr().err


def test_a_restore_failure_inside_a_broken_record_still_keeps_both_models(built, tmp_path, monkeypatch, capsys):
    # a broken record turns RestoreFailed into RecordError on its way out of
    # the span: the folders, not the error's type, decide what is kept
    import wpt_train
    from wpt_telemetry import RecordError

    main(["d", built.dataset_id, "--out", "model"])
    earlier = (tmp_path / "model" / "model.safetensors").read_bytes()
    real_replace, real_folder = os.replace, wpt_train._replace_folder

    def refuse_every_move_into_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real_replace(source, target)

    def replace_while_the_record_breaks(new, target, telemetry):
        try:
            real_folder(new, target, telemetry)
        except wpt_train.RestoreFailed as failed:
            raise RecordError("the record broke too") from failed

    monkeypatch.setattr(os, "replace", refuse_every_move_into_the_target)
    monkeypatch.setattr(wpt_train, "_replace_folder", replace_while_the_record_breaks)
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("three", training={"steps": 3}), built.dataset_id, "--out", "model"])
    assert stopped.value.code == 3 and not (tmp_path / "model").exists()
    [aside] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-replaced-")]
    [new] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-") and path != aside]
    assert (aside / "model.safetensors").read_bytes() == earlier
    assert json.loads((new / "provenance.json").read_text())["recipe_name"] == "three"
    err = capsys.readouterr().err
    assert str(new) in err and str(aside) in err
