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

from wpt_inference import COSTS_FILE, DRAFT, costs_setup, replayed_costs
from wpt_model import load_drafter, load_model, load_tokenizer, tokenizer_identity, weight_hashes
import wpt_telemetry
from wpt_store import LocalStore
from wpt_telemetry import FLUSH_THREAD
from wpt_tokenizer import Tokenizer
from wpt_utils.device import select_device
from wpt_train import main

ROOT = Path(__file__).resolve().parents[3]


def one_stage(steps, block_size=16, batch_size=4):
    """A read-text step of one stage (docs/training.md, "Stages")."""
    return {"stages": [{"block_size": block_size, "steps": steps, "batch_size": batch_size}]}


def forget_steps(store):
    """Forget every step result the store keeps, as a store that never saw
    these recipes: the next run trains every step again."""
    shutil.rmtree(store.root / "refs" / "steps", ignore_errors=True)


def test_training_writes_a_model_with_its_provenance(built, tmp_path, monkeypatch, capsys):
    store, dataset_id = built.store, built.dataset_id
    recipe = built.recipe("three", training=one_stage(3))
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
    assert provenance["recipe_name"] == "three"
    assert provenance["recipe"]["steps"][0]["settings"]["stages"] == [{"block_size": 16, "steps": 3, "batch_size": 4}]
    [step] = provenance["steps"]
    assert step["step"]["name"] == "read-text" and step["updates"] == 3 and not step["reused"]
    assert isinstance(step["indexer_loss"], float)  # the CSA indexers' loss at the kept update
    # the step's result is kept in the store, under its fingerprint
    assert store.get_ref("steps", step["fingerprint"]) == step["result"]
    assert [s["token_sha256"] for s in provenance["shards"]] == [s["token_sha256"] for s in record["shards"]]


def test_a_training_run_records_what_it_read_every_step_and_what_it_wrote(built, tmp_path, capsys):
    # afterwards one must know exactly what happened, from the store alone:
    # the dataset read, each step's loss, the evaluations, the files written
    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
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


def test_training_ends_with_the_drafter_against_the_final_weights_and_the_scheduler_s_times(built, tmp_path,
                                                                                             capsys):
    # a model and its drafter are one
    # unit; the run trains the drafter at its end, the model frozen, and
    # measures the scheduler's pass times once, into its record; a later run
    # replays them
    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
    folder = tmp_path / "model"
    drafter = load_drafter(folder).provenance
    assert drafter["target"] == {"weights": weight_hashes(folder)}
    assert drafter["trained_by"] == {"command": "wpt-train", "recipe_name": "three"}
    assert drafter["dataset"] == built.dataset_id and drafter["prompts"]["count"] == 4
    [run] = built.records("wpt-train")
    names = [e.get("name") for e in run.events]
    assert names.index("drafter.settings") > max(i for i, name in enumerate(names) if name == "training.step")
    assert run.ids["outputs"]["drafter"] == hashlib.sha256((folder / "drafter" / "model.safetensors").read_bytes()
                                                           ).hexdigest()
    recorded = built.store.read_record(run.folder, COSTS_FILE)
    # the times carry the setup they were measured on
    setup = costs_setup(load_model(folder).config, 1, DRAFT, select_device())
    assert json.loads(recorded)["setup"] == setup
    costs = replayed_costs(recorded, setup)
    assert len(costs) == DRAFT + 1 and all(seconds > 0 for seconds in costs)

    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "again", "--scheduler-costs",
          run.folder])
    replayed = built.records("wpt-train")[-1]
    assert built.store.read_record(replayed.folder, COSTS_FILE) == recorded
    # refused before training, not after: no such record, or times measured on another setup
    for recipe, record, why in ((built.recipe("three", training=one_stage(3)), "no-such-run", "MissingObject"),
                                (built.recipe("block", training=one_stage(3), drafter={"block": 4}), run.folder,
                                 "another setup: draft 5, here 4"),
                                (built.recipe("longer", training=one_stage(3, block_size=32)), run.folder,
                                 "another setup: model block_size 16, here 32")):
        with pytest.raises(SystemExit) as refused:
            main([recipe, built.dataset_id, "--out", "never", "--scheduler-costs", record])
        assert refused.value.code == 2 and not (tmp_path / "never").exists()
        assert why in capsys.readouterr().err
        assert not [e for e in built.records("wpt-train")[-1].events if e.get("name") == "training.step"]


def test_drafter_feature_layers_the_model_does_not_have_are_refused_before_training(built, tmp_path, capsys):
    # checked against the model's layers at startup
    with pytest.raises(SystemExit) as refused:
        main([built.recipe("three", training=one_stage(3), drafter={"features": [9]}), built.dataset_id, "--out",
              "model"])
    assert refused.value.code == 2 and "9 is not a layer of a" in capsys.readouterr().err
    assert not [e for e in built.records("wpt-train")[-1].events if e.get("name") == "training.step"]


def test_a_drafter_that_cannot_be_trained_refuses_the_run_and_its_steps_stay_kept(built, tmp_path, capsys):
    # held_out keeps both prompts apart: nothing is left to learn from. The
    # step's result was kept when it ended, even with no checkpoint during
    # training, so the same command, its drafter fixed, takes no update and
    # trains the drafter again
    failing = built.recipe("three", training=one_stage(3), drafter={"prompts": 2, "held_out": 0.9})
    with pytest.raises(SystemExit) as refused:
        main([failing, built.dataset_id, "--out", "model", "--checkpoint-every", "0"])
    assert refused.value.code == 2
    assert capsys.readouterr().err.strip().endswith(
        "DrafterFailed: held_out 0.9 keeps every one of the 2 prompts apart; nothing was written; every step's "
        "result is kept in the store: the same command trains the drafter again without training")
    assert not (tmp_path / "model" / "config.json").exists() and not (tmp_path / "model" / "checkpoint").exists()
    [run] = built.records("wpt-train")
    names = [e.get("name") for e in run.events if e["kind"] != "span_start"]
    assert names.count("training.checkpoint") == 1 and names.count("store.ref_set") == 1  # the step's result
    assert names.index("training.checkpoint") > max(i for i, name in enumerate(names) if name == "training.step")
    # the drafter's section takes no part in a step's fingerprint: a fixed one reuses the kept step
    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
    again = built.records("wpt-train")[-1]
    assert not [e for e in again.events if e.get("name") == "training.step"]
    assert [e["attributes"]["step"] for e in again.events if e.get("name") == "training.step_reused"] == ["read-text"]
    assert load_drafter(tmp_path / "model").provenance["target"] == {"weights": weight_hashes(tmp_path / "model")}


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
        main([built.recipe("five", training=one_stage(5)), built.dataset_id, "--out", "model"])

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
    forget_steps(built.store)

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
    recipe = built.recipe("two", training=one_stage(2))
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
            main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
    assert stopped.value.code == 3
    assert {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()} == before
    assert (folder / "evals" / "memorization.json").exists()
    # the step's result is kept: the same command finishes the run without training
    assert not (folder / "checkpoint").exists()

    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
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
    # as it was: the step's checkpoint went once its result was kept
    assert sorted(path.name for path in folder.iterdir()) == ["notes.txt"]
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".model")] == []


def test_every_checkpoint_deleted_from_the_store_is_named_in_the_record_first(built, tmp_path, monkeypatch):
    # an object leaves the store only once the record says which one went:
    # each replaced checkpoint; the last one loses its name only, its object
    # kept as it is as the step's state
    real, deleted = LocalStore.delete, []

    def delete(self, sha):
        [run] = built.records("wpt-train")
        named = [e["attributes"]["object"] for e in run.events if e.get("name") == "training.checkpoint_deleted"]
        deleted.append((sha, sha in named))
        real(self, sha)

    monkeypatch.setattr(LocalStore, "delete", delete)
    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model", "--checkpoint-every", "1",
          "--run", "fox-1"])
    assert len({sha for sha, _ in deleted}) == 2 and all(named for _, named in deleted)
    assert built.store.get_ref("checkpoints", "fox-1") is None
    [step] = json.loads((tmp_path / "model" / "provenance.json").read_text())["steps"]
    state = json.loads(built.store.get_bytes(step["result"]))["state"]
    assert built.store.has(state) and state not in {sha for sha, _ in deleted}


def test_on_s3_a_checkpoint_is_deleted_only_once_its_deletion_is_in_the_bucket(built, s3, tmp_path, monkeypatch):
    # S3 holds record lines back: a deletion the bucket did not record must
    # not happen, so the previous checkpoint stays
    from wpt_store import S3Store

    s3.copy(built.store)
    deleted = []
    monkeypatch.setattr(S3Store, "delete", lambda self, sha: deleted.append(sha))
    s3.refuse(b'"training.checkpoint_deleted"')
    with pytest.raises(SystemExit) as stopped:
        main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model",
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
        main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
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
        main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
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
    recipe = built.recipe("four", training=one_stage(4))
    for out, command in [("one", ["-m", "wpt_train", "--micro-batch", "2"]), ("two", [*starter, "--strategy", "ddp"])]:
        forget_steps(built.store)  # the hardware is no part of a step's fingerprint: each run trains
        subprocess.run([sys.executable, *command, recipe, built.dataset_id, "--out", out],
                       cwd=tmp_path, env=env, check=True)

    one, two = weights(tmp_path / "one"), weights(tmp_path / "two")
    assert one.keys() == two.keys()
    # gloo adds gradients in varying order: up to 4e-6 apart; other dropout gives 2e-3
    assert all(torch.allclose(one[k], two[k], atol=1e-4) for k in one)
    provenance = json.loads((tmp_path / "two" / "provenance.json").read_text())
    assert provenance["strategy"]["processes"] == 2


def test_split_windows_in_segments_train_the_same_model_as_one_process(built, tmp_path):
    # docs/training.md, "Long windows": 2 processes split each 512-token
    # window, each running its slice in segments of 128 tokens, one window
    # per pass; the batch of 2 windows takes 2 passes in each process, and
    # the model is the one a single process trains on whole windows
    env = {**os.environ, **ON_CPU}
    recipe = built.recipe("split", training=one_stage(2, 512, 2))
    for out, options in [("one", []), ("two", ["--processes", "2", "--strategy", "split", "--micro-batch", "1",
                                              "--segment", "128"])]:
        forget_steps(built.store)
        subprocess.run([sys.executable, "-m", "wpt_train", recipe, built.dataset_id, "--out", out, *options],
                       cwd=tmp_path, env=env, check=True)

    one, two = weights(tmp_path / "one"), weights(tmp_path / "two")
    assert one.keys() == two.keys()
    assert all(torch.allclose(one[k], two[k], atol=1e-4) for k in one)
    provenance = json.loads((tmp_path / "two" / "provenance.json").read_text())
    assert provenance["strategy"]["processes"] == 2


def interrupting(built, tmp_path, at_step_3, at_step=3, best=None):
    """The recipe and options of a 5-update run on the CPU whose optimizer
    does at_step_3 (Python code) at update 3 (or at_step), just before the
    update; with best (an update), it judges only that update's evaluation
    the best."""
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "interrupting.py"
    method.write_text(
        "import os\nimport signal\nimport torch\nfrom wpt_training import Optimizer, Verdict\n\n"
        "SETTINGS = {'lr': 0.01}\n\n"
        "class Interrupting(Optimizer):\n"
        "    def __init__(self, model, settings):\n"
        "        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], "
        "lr=settings['lr'])\n\n"
        "    def before_update(self, step):\n"
        f"        if step == {at_step}:\n"
        f"            {at_step_3}\n\n"
        + ("" if best is None else
           "    def evaluated(self, step, val_loss):\n"
           f"        return Verdict(best=step == {best}, stop=False)\n\n")
        + "def start(model, settings, run):\n    return Interrupting(model, settings)\n")
    return [built.recipe("interrupting", training={**one_stage(5), "optimizer": str(method)}), built.dataset_id,
            "--strategy", str(strategy)]


def test_a_crashed_run_resumes_where_its_checkpoint_left_it(built, tmp_path, monkeypatch, capsys):
    # after a crash, --resume must give the same model as a run never interrupted
    options = [*interrupting(built, tmp_path, "if os.environ.get('CRASH'): raise RuntimeError('power cut')"),
               "--checkpoint-every", "2"]
    main([*options, "--out", "whole"])
    forget_steps(built.store)

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


def staged(built, tmp_path, crash_at, best=None):
    """The options of a run of two stages on the CPU (docs/training.md,
    "Stages"): 2 updates of 4 windows of 16 tokens, then 4 updates of 2
    windows of 256 tokens, whose optimizer raises at update crash_at when
    CRASH is set."""
    options = interrupting(built, tmp_path, "if os.environ.get('CRASH'): raise RuntimeError('power cut')", crash_at,
                           best)
    options[0] = built.recipe("staged", training={
        "stages": [{"block_size": 16, "steps": 2, "batch_size": 4}, {"block_size": 256, "steps": 4, "batch_size": 2}],
        "optimizer": str(tmp_path / "interrupting.py")})
    return [*options, "--checkpoint-every", "2"]


def stage_spans(run):
    return [event["attributes"] for event in run.events
            if event["kind"] == "span_start" and event["name"] == "training.stage"]


def step_windows(run):
    return [(event["attributes"]["step"], event["attributes"]["block_size"]) for event in run.events
            if event.get("name") == "training.step"]


def test_stages_train_in_order_and_a_run_resumes_across_their_boundary(built, tmp_path, monkeypatch):
    # a crash at the first step of the second stage resumes from the
    # checkpoint the first stage's end wrote, and gives the model of a run
    # never interrupted (the long windows in segments of 128 tokens, as a
    # 16 GB GPU would run them); each stage draws its own batch
    options = staged(built, tmp_path, 3)
    main([*options, "--out", "whole", "--segment", "128"])
    forget_steps(built.store)
    monkeypatch.setenv("CRASH", "1")
    with pytest.raises(RuntimeError, match="power cut"):
        main([*options, "--out", "resumed"])
    monkeypatch.delenv("CRASH")
    main([*options, "--out", "resumed", "--segment", "128", "--resume"])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    assert load_model(tmp_path / "resumed").config.block_size == 256  # the model's context: the last stage's
    provenance = json.loads((tmp_path / "resumed" / "provenance.json").read_text())
    assert provenance["steps"][0]["stages"] == [
        {"block_size": 16, "steps": 2, "batch_size": 4, "first_step": 1, "last_step": 2},
        {"block_size": 256, "steps": 4, "batch_size": 2, "first_step": 3, "last_step": 6}]

    uninterrupted, crashed, again = built.records("wpt-train")
    assert [(s["stage"], s["block_size"], s["batch_size"], s["first_step"], s["last_step"])
            for s in stage_spans(uninterrupted)] == [(1, 16, 4, 1, 2), (2, 256, 2, 3, 6)]
    assert step_windows(uninterrupted) == [(1, 16), (2, 16), (3, 256), (4, 256), (5, 256), (6, 256)]
    assert [(s["stage"], s["resumed_at"]) for s in stage_spans(again)] == [(2, 3)]
    assert step_windows(again) == [(3, 256), (4, 256), (5, 256), (6, 256)]


def test_a_run_resumed_in_the_middle_of_a_later_stage_keeps_its_best_weights(built, tmp_path, monkeypatch):
    # the method judges only step 4's evaluation the best: a crash at step 5
    # resumes from step 4's checkpoint, inside the second stage, with that
    # best kept, so the folder holds the same weights from the same step as
    # a run never interrupted, and the resumed run trains steps 5 and 6 only
    options = staged(built, tmp_path, 5, best=4)
    main([*options, "--out", "whole"])
    forget_steps(built.store)
    monkeypatch.setenv("CRASH", "1")
    with pytest.raises(RuntimeError, match="power cut"):
        main([*options, "--out", "resumed"])
    monkeypatch.delenv("CRASH")
    main([*options, "--out", "resumed", "--resume"])

    whole, resumed = weights(tmp_path / "whole"), weights(tmp_path / "resumed")
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    for out in ("whole", "resumed"):
        provenance = json.loads((tmp_path / out / "provenance.json").read_text())
        assert (provenance["steps"][0]["weights_update"], provenance["steps"][0]["updates"]) == (4, 6)
    uninterrupted, crashed, again = built.records("wpt-train")
    assert [(s["stage"], s["first_step"], s["resumed_at"]) for s in stage_spans(again)] == [(2, 3, 5)]
    assert step_windows(again) == [(5, 256), (6, 256)]


def test_the_model_card_names_the_updates_trained_and_the_update_of_its_weights(built, tmp_path):
    # a published model says how far each step trained and which update its
    # weights come from, even when its optimizer stopped early and kept an
    # earlier update
    strategy = tmp_path / "cpu.py"
    strategy.write_text("import torch\nfrom wpt_training import Strategy\n\n"
                        "def start():\n    return Strategy(torch.device('cpu'))\n")
    method = tmp_path / "early.py"
    method.write_text(
        "import torch\nfrom wpt_training import Optimizer, Verdict\n\nSETTINGS = {}\n\n"
        "class Early(Optimizer):\n"
        "    def __init__(self, model):\n"
        "        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)\n\n"
        "    def evaluated(self, step, val_loss):\n"
        "        return Verdict(best=step == 2, stop=step == 4)\n\n"
        "def start(model, settings, run):\n    return Early(model)\n")
    recipe = built.recipe("early", training={**one_stage(10), "optimizer": str(method)})
    main([recipe, built.dataset_id, "--out", "model", "--strategy", str(strategy)])

    provenance = json.loads((tmp_path / "model" / "provenance.json").read_text())
    assert (provenance["steps"][0]["updates"], provenance["steps"][0]["weights_update"]) == (4, 2)
    card = (tmp_path / "model" / "README.md").read_text()
    assert "1. `read-text`: method next-token, optimizer early, environment text-windows; 4 updates (limit 10), " \
           "the weights of update 2" in card


def test_streamed_training_never_copies_the_whole_dataset_to_disk(built, tmp_path):
    # --stream trains from a few shards at a time, so a machine needs disk
    # only for those, and the model folder names its sources as usual
    store, dataset_id = built.store, built.dataset_id
    main([built.recipe("two", training=one_stage(2)), dataset_id, "--out", "model", "--stream", "--stream-shards", "1"])

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
    forget_steps(built.store)

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
    forget_steps(built.store)
    run = subprocess.run([sys.executable, "-m", "wpt_train", *options, "--out", "restarted"],
                         cwd=tmp_path, env={**env, "CRASH": "1", "WPT_MAX_RESTARTS": "1"}, capture_output=True,
                         text=True)

    assert run.returncode == 0, run.stderr
    assert "power cut" in run.stderr and "starting every process again (1 of 1 restarts)" in run.stderr
    assert "read-text: resumed from update 2" in run.stdout
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
    # with no earlier folder, a refused rename must not delete the only copy of the new model; the checkpoint
    # lives in the store (--run), so no checkpoint folder takes the target's place first
    real = os.replace

    def refuse_the_target(source, target):
        if Path(target).name == "model":
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_target)
    with pytest.raises(PermissionError):
        main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model", "--run", "fox-2"])
    [kept] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-")]
    assert not (tmp_path / "model").exists()
    assert json.loads((kept / "provenance.json").read_text())["recipe_name"] == "three"
    assert str(kept) in capsys.readouterr().err


def test_a_new_model_that_cannot_take_the_place_of_its_own_checkpoint_folder_is_kept_and_named(built, tmp_path,
                                                                                             monkeypatch, capsys):
    # without --run, the target holds only what this run's checkpoint left
    # there: a refused rename keeps the new model, which has no other copy,
    # and names it
    real = os.replace

    def refuse_the_new_one(source, target):
        if Path(target).name == "model" and not Path(source).name.startswith(".model-replaced-"):
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_new_one)
    with pytest.raises(PermissionError):
        main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
    assert list((tmp_path / "model").iterdir()) == []
    [kept] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-")]
    assert json.loads((kept / "provenance.json").read_text())["recipe_name"] == "three"
    assert f"it is kept in {kept}" in capsys.readouterr().err
    monkeypatch.setattr(os, "replace", real)
    main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])  # reuses the step
    assert (tmp_path / "model" / "config.json").exists()
    assert kept.is_dir()  # the run wrote its own folder: the kept one is left to remove by hand


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
        main([built.recipe("three", training=one_stage(3)), built.dataset_id, "--out", "model"])
    assert stopped.value.code == 3 and not (tmp_path / "model").exists()
    [aside] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-replaced-")]
    [new] = [path for path in tmp_path.iterdir() if path.name.startswith(".model-") and path != aside]
    assert (aside / "model.safetensors").read_bytes() == earlier
    assert json.loads((new / "provenance.json").read_text())["recipe_name"] == "three"
    err = capsys.readouterr().err
    assert str(new) in err and str(aside) in err
