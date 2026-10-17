import hashlib
import json
import os
import subprocess

import pytest
import torch

from wpt_dataset import manifest
from wpt_dataset.repo import address_hash, dump_json
from wpt_dataset.sources import compute_upstream
from wpt_model import load_model
from wpt_eval import main as evaluate
from wpt_posttrain import main
from wpt_train import main as train

ZERO_WEIGHTS = ("import torch\nfrom wpt_posttrain import Outcome\n\n"
             "def run(context):\n"
             "    with torch.no_grad():\n"
             "        for p in context.model.parameters():\n"
             "            p.zero_()\n"
             "    return Outcome.success()\n")


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


def test_steps_that_succeed_give_a_new_folder_and_leave_the_base_untouched(trained, tmp_path):
    # the recipe names the steps, and the new folder records which recipe ran
    base_before = digest(trained.folder)
    step = tmp_path / "zero_weights.py"
    step.write_text(ZERO_WEIGHTS)
    check = tmp_path / "check_records.py"
    check.write_text("from wpt_posttrain import Outcome\n\ndef run(context):\n    return Outcome.success()\n")
    main([trained.recipe("tune", posttrain=[str(step), str(check)]), str(trained.folder), str(tmp_path / "tuned")])

    tuned = tmp_path / "tuned"
    assert all(torch.count_nonzero(p) == 0 for p in load_model(tuned).parameters())  # the steps' change was saved
    assert digest(trained.folder) == base_before
    provenance = json.loads((tuned / "provenance.json").read_text())
    post = provenance["post_training"]
    assert post["recipe_name"] == "tune"
    assert [s["file"] for s in post["steps"]] == ["zero_weights.py", "check_records.py"]
    assert post["steps"][0]["file_sha256"] == hashlib.sha256(step.read_bytes()).hexdigest()
    assert post["base_weights"] == {"model.safetensors": base_before["model.safetensors"]}
    assert provenance["sources"] == json.loads((trained.folder / "provenance.json").read_text())["sources"]
    assert trained.attribution in (tuned / "NOTICE").read_text()
    card = (tuned / "README.md").read_text()
    assert "## Post-training" in card and "1. `zero_weights.py`" in card and "2. `check_records.py`" in card
    assert not (tuned / "evals").exists()


def test_a_failed_step_stops_the_run_with_its_reason_and_writes_nothing(trained, tmp_path, capsys):
    step = tmp_path / "zero_weights.py"
    step.write_text(ZERO_WEIGHTS)
    failing = tmp_path / "needs_records.py"
    failing.write_text("from wpt_posttrain import Outcome\n\n"
                       "def run(context):\n    return Outcome.failure('no demonstration records in the store')\n")
    never = tmp_path / "never.py"
    never.write_text("raise SystemExit('this step must not run')\n")
    with pytest.raises(SystemExit, match="step needs_records failed: no demonstration records in the store"):
        main([trained.recipe("tune", posttrain=[str(step), str(failing), str(never)]), str(trained.folder),
              str(tmp_path / "tuned")])
    assert list(tmp_path.glob("*tuned*")) == []


def test_a_step_that_returns_no_outcome_writes_nothing(trained, tmp_path):
    step = tmp_path / "broken.py"
    step.write_text("def run(context):\n    return True\n")
    with pytest.raises(SystemExit, match="returned bool, not wpt_posttrain.Outcome"):
        main([trained.recipe("tune", posttrain=[str(step)]), str(trained.folder), str(tmp_path / "tuned")])
    assert list(tmp_path.glob("*tuned*")) == []


def test_sft_learns_from_the_sources_tagged_sft_and_credits_them(built, tmp_path, monkeypatch):
    # the built-in step fine-tunes on the conversations a person adds as a
    # source tagged sft; until one exists it says so and writes nothing, and
    # once it has trained, the model's NOTICE credits that source
    recipe = built.recipe("chat", model={"block_size": 64}, posttrain=["sft"])
    train([recipe, built.dataset_id, "--out", "model"])
    base, root = tmp_path / "model", tmp_path  # the step reads the sources of the repository holding the current folder
    with pytest.raises(SystemExit, match="step sft failed: no conversations: no admitted source has chat files tagged sft"):
        main([recipe, str(base), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()

    talk = {"schema": 1, "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                                      {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}
    upstream = tmp_path / "talks"
    upstream.mkdir()
    (upstream / "talks.jsonl").write_text(json.dumps(talk) + "\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=env, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()
    data = {"schema": 5, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Talks, by Bo Speaker, 2026",
                          "text": "Bo Speaker dedicates these talks to the public domain.\n"}],
            "formats": [{"format": "chat", "paths": ["**"]}], "tags": [{"tags": ["sft"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-15"}}
    data["checksum"] = manifest.checksum(compute_upstream(data, data["address"], built.store))
    sid = f"example-talks-{address_hash(str(upstream))}"
    (root / "data" / "sources" / sid).mkdir()
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))

    main([recipe, str(base), str(tmp_path / "tuned")])
    tuned = tmp_path / "tuned"
    assert digest(tuned)["model.safetensors"] != digest(base)["model.safetensors"]
    post = json.loads((tuned / "provenance.json").read_text())["post_training"]
    assert post["steps"][0]["file"] == "wpt_posttrain/steps/sft.py" and post["steps"][0]["sources"] == [sid]
    assert post["sources"][sid]["revision"] == commit
    notice = (tuned / "NOTICE").read_text()
    assert sid in notice and "Talks, by Bo Speaker, 2026" in notice and "dedicates these talks" in notice
    assert built.attribution in notice  # the pre-training sources stay credited
    assert f"1. `wpt_posttrain/steps/sft.py`" in (tuned / "README.md").read_text()


def test_the_records_of_training_post_training_and_evaluation_follow_one_model(trained, tmp_path):
    # each run's record names the ids it read and wrote, so the chain from the
    # dataset to the evaluated model can be followed across their folders;
    # a step records its own events and files in its run's folder
    step = tmp_path / "keeps_attempts.py"
    step.write_text("from wpt_posttrain import Outcome\n\n"
                    "def run(context):\n"
                    "    context.telemetry.event('attempts', values={'reward': 1.0}, count=1)\n"
                    "    context.telemetry.append_file('attempts.jsonl', b'{\"messages\": []}\\n')\n"
                    "    return Outcome.success()\n")
    main([trained.recipe("tune", posttrain=[str(step)]), str(trained.folder), str(tmp_path / "tuned")])
    evaluate([str(tmp_path / "tuned")])

    [training], [post], [evaluation] = (trained.records(c) for c in ("wpt-train", "wpt-posttrain", "wpt-eval"))
    assert training.ids["inputs"]["dataset"] == post.ids["inputs"]["dataset"] == trained.dataset_id
    assert post.ids["inputs"]["base_weights"] == training.ids["outputs"]["weights"]
    assert evaluation.ids["inputs"]["weights"] == post.ids["outputs"]["weights"] == {
        "model.safetensors": digest(tmp_path / "tuned")["model.safetensors"]}
    assert all(run.end["status"] == "succeeded" for run in (training, post, evaluation))
    # each folder's name carries what its run is about: the weights it started from, or measured
    assert f"-wpt-posttrain-{training.ids['outputs']['weights']['model.safetensors'][:8]}-" in post.folder
    assert f"-wpt-eval-{post.ids['outputs']['weights']['model.safetensors'][:8]}-" in evaluation.folder
    assert any(e.get("name") == "attempts" for e in post.events)
    assert trained.store.read_record(post.folder, "attempts.jsonl") == b'{"messages": []}\n'


def test_on_s3_the_new_folder_appears_only_once_its_record_is_in_the_bucket(trained, s3, tmp_path):
    # S3 holds record lines back: the lines naming the new folder's files must
    # be in the bucket before the folder appears, so a refused send leaves none
    step = tmp_path / "nothing.py"
    step.write_text("from wpt_posttrain import Outcome\n\ndef run(context):\n    return Outcome.success()\n")
    s3.refuse(b'"model.files"')
    with pytest.raises(SystemExit) as stopped:
        main([trained.recipe("tune", posttrain=[str(step)]), str(trained.folder), str(tmp_path / "tuned")])
    assert stopped.value.code == 3
    assert [path.name for path in tmp_path.iterdir() if "tuned" in path.name] == []


def test_a_new_folder_that_cannot_be_renamed_into_place_is_kept_and_named(trained, tmp_path, monkeypatch, capsys):
    # the rename is recorded before it happens: when it fails, the finished model stays, named
    step = tmp_path / "nothing.py"
    step.write_text("from wpt_posttrain import Outcome\n\ndef run(context):\n    return Outcome.success()\n")
    real = os.replace

    def refuse_the_target(source, target):
        if os.path.basename(target) == "tuned":
            raise PermissionError(13, "Permission denied", str(source))
        real(source, target)

    monkeypatch.setattr(os, "replace", refuse_the_target)
    with pytest.raises(PermissionError):
        main([trained.recipe("tune", posttrain=[str(step)]), str(trained.folder), str(tmp_path / "tuned")])
    [kept] = [path for path in tmp_path.iterdir() if path.name.startswith(".tuned-")]
    assert not (tmp_path / "tuned").exists() and (kept / "provenance.json").is_file()
    assert str(kept) in capsys.readouterr().err
