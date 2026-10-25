import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import wpt_agent
import wpt_inference
import wpt_posttrain
from wpt_agent import images, played
from wpt_dataset.build import is_held_out
from wpt_dataset.repo import address_hash, dump_json
from wpt_inference import COSTS_FILE, DRAFT, costs_setup, replayed_costs
from wpt_model import load_drafter, load_model, weight_hashes
from wpt_eval import main as evaluate
from wpt_dataset import conversations
from wpt_posttrain import agentic, main
from wpt_store import LocalStore
from wpt_telemetry import RECORD_FAILED, Telemetry
from wpt_tokenizer import Tokenizer, chat
from wpt_train import main as train
from wpt_utils.device import select_device


def one_stage(steps, block_size=16, batch_size=4):
    """A recipe's training part of one stage (docs/training.md, "Stages")."""
    return {"stages": [{"block_size": block_size, "steps": steps, "batch_size": batch_size}]}


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


def test_every_run_trains_a_fresh_drafter_against_the_new_weights(trained, tmp_path, monkeypatch):
    # a model and its drafter are one
    # unit; after the steps, the run trains a new drafter, the model frozen,
    # on the beginnings of the base model's training documents, and the new
    # folder holds it, named against the new weights. Its training answers stop at <|im_end|> too
    stops, real = [], wpt_posttrain.train_drafter
    monkeypatch.setattr(wpt_posttrain, "train_drafter", lambda *args, **kwargs: stops.append(kwargs["stops"])
                        or real(*args, **kwargs))
    step = tmp_path / "zero_weights.py"
    step.write_text(ZERO_WEIGHTS)
    main([trained.recipe("tune", posttrain=[str(step)]), str(trained.folder), str(tmp_path / "tuned")])
    assert stops == [(Tokenizer.END, chat.IM_END)]

    tuned = tmp_path / "tuned"
    drafter, base = load_drafter(tuned).provenance, load_drafter(trained.folder).provenance
    assert drafter["target"] == {"weights": weight_hashes(tuned)} != base["target"]
    assert drafter["trained_by"] == {"command": "wpt-posttrain", "recipe_name": "tune"}
    assert drafter["dataset"] == trained.dataset_id and drafter["prompts"]["count"] == 4
    assert all(torch.count_nonzero(p) == 0 for p in load_model(tuned).parameters())  # the model learned nothing more
    [run] = trained.records("wpt-posttrain")
    names = {event.get("name") for event in run.events}
    assert {"drafter", "drafter.settings", "drafter.update", "drafter.calibration", "drafter.acceptance"} <= names
    assert "## Drafter" in (tuned / "README.md").read_text()
    assert "Trained by `wpt-posttrain`, recipe `tune`" in (tuned / "README.md").read_text()


def test_post_training_records_the_scheduler_s_times_and_replays_them_only_on_their_setup(trained, tmp_path):
    # wpt-posttrain measures the
    # scheduler's pass times for the new weights, records them with their
    # setup, and replays an earlier run's; another setup is refused before
    # any step runs
    check = tmp_path / "check.py"
    check.write_text("from wpt_posttrain import Outcome\n\ndef run(context):\n    return Outcome.success()\n")
    main([trained.recipe("tune", posttrain=[str(check)]), str(trained.folder), str(tmp_path / "tuned")])
    [run] = trained.records("wpt-posttrain")
    recorded = trained.store.read_record(run.folder, COSTS_FILE)
    setup = costs_setup(load_model(tmp_path / "tuned").config, 1, DRAFT, select_device())
    assert len(replayed_costs(recorded, setup)) == DRAFT + 1
    assert "scheduler.costs" in {event.get("name") for event in run.events}

    main([trained.recipe("tune", posttrain=[str(check)]), str(trained.folder), str(tmp_path / "again"),
          "--scheduler-costs", run.folder])
    assert trained.store.read_record(trained.records("wpt-posttrain")[-1].folder, COSTS_FILE) == recorded
    never = tmp_path / "never.py"  # its file is read before any step runs; its run must never start
    never.write_text("def run(context):\n    raise SystemExit('this step must not run')\n")
    with pytest.raises(SystemExit, match=r"StepError: the recorded costs were measured on another setup: draft 5, "
                                         r"here 4\nnothing was written"):
        main([trained.recipe("tune", drafter={"block": 4}, posttrain=[str(never)]), str(trained.folder),
              str(tmp_path / "never"), "--scheduler-costs", run.folder])
    assert not (tmp_path / "never").exists()


def test_a_drafter_that_cannot_be_trained_fails_the_run_and_writes_nothing(trained, tmp_path):
    # held_out keeps both prompts apart: nothing is left to learn from
    check = tmp_path / "check.py"
    check.write_text("from wpt_posttrain import Outcome\n\ndef run(context):\n    return Outcome.success()\n")
    recipe = trained.recipe("tune", drafter={"prompts": 2, "held_out": 0.9}, posttrain=[str(check)])
    with pytest.raises(SystemExit, match=r"DrafterFailed: held_out 0.9 keeps every one of the 2 prompts apart\n"
                                         r"nothing was written"):
        main([recipe, str(trained.folder), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()
    assert not list(tmp_path.glob(".tuned-*"))


def test_a_base_folder_whose_drafter_was_trained_against_other_weights_is_refused(trained, tmp_path):
    # no option skips the check
    provenance = trained.folder / "drafter" / "provenance.json"
    document = json.loads(provenance.read_text())
    provenance.write_text(json.dumps({**document, "target": {"weights": {"model.safetensors": "0" * 64}}}))
    check = tmp_path / "check.py"
    check.write_text("raise SystemExit('this step must not run')\n")
    with pytest.raises(SystemExit, match=r"DrafterMismatch|StepError: .*its drafter was trained against weights"):
        main([trained.recipe("tune", posttrain=[str(check)]), str(trained.folder), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()


def test_a_failed_step_stops_the_run_with_its_reason_and_writes_nothing(trained, tmp_path, capsys):
    step = tmp_path / "zero_weights.py"
    step.write_text(ZERO_WEIGHTS)
    failing = tmp_path / "needs_records.py"
    failing.write_text("from wpt_posttrain import Outcome\n\n"
                       "def run(context):\n    return Outcome.failure('no demonstration records in the store')\n")
    never = tmp_path / "never.py"  # its file is read before any step runs; its run must never start
    never.write_text("def run(context):\n    raise SystemExit('this step must not run')\n")
    with pytest.raises(SystemExit, match="step needs_records failed: no demonstration records in the store"):
        main([trained.recipe("tune", posttrain=[str(step), str(failing), str(never)]), str(trained.folder),
              str(tmp_path / "tuned")])
    assert list(tmp_path.glob("*tuned*")) == []


def test_a_recipe_entry_s_settings_reach_its_step_typed_from_the_step_s_defaults(trained, tmp_path):
    # a posttrain entry may carry the
    # settings it changes, typed from the step file's SETTINGS as a training
    # method's are; the new folder records them, and an unknown or mistyped
    # setting stops the run before any step has run
    step = tmp_path / "counted.py"
    step.write_text("from wpt_posttrain import Outcome\n\nSETTINGS = {'rounds': 10, 'rate': 0.5}\n\n"
                    "def run(context):\n    assert context.settings == {'rounds': 3, 'rate': 1.0}\n"
                    "    return Outcome.success()\n")
    zero = tmp_path / "zero_weights.py"
    zero.write_text(ZERO_WEIGHTS)
    entry = {"step": str(step), "settings": {"rounds": 3, "rate": 1}}
    main([trained.recipe("tune", posttrain=[entry]), str(trained.folder), str(tmp_path / "tuned")])
    post = json.loads((tmp_path / "tuned" / "provenance.json").read_text())["post_training"]
    assert post["steps"][0]["settings"] == {"rounds": 3, "rate": 1.0}
    for settings, why in (({"round": 3}, r"no setting 'round' \(settings: rounds, rate\)"),
                          ({"rounds": "many"}, r"setting rounds='many' is not a int")):
        recipe = trained.recipe("tune", posttrain=[str(zero), {"step": str(step), "settings": settings}])
        with pytest.raises(SystemExit, match=rf"counted.py: {why}\nnothing was written"):
            main([recipe, str(trained.folder), str(tmp_path / "again")])
    # stating a low confidence must never pay more than being right: refused before sft or any step runs
    recipe = trained.recipe("tune", posttrain=[str(zero), {"step": "grpo", "settings": {"weight_calibration": 2}}])
    with pytest.raises(SystemExit, match=r"grpo.py: weight_correct \(1.0\) must be above weight_calibration \(2.0\)"):
        main([recipe, str(trained.folder), str(tmp_path / "again")])
    assert not (tmp_path / "again").exists()
    assert all(torch.count_nonzero(p) != 0 for p in load_model(trained.folder).parameters())


def test_a_step_that_returns_no_outcome_writes_nothing(trained, tmp_path):
    step = tmp_path / "broken.py"
    step.write_text("def run(context):\n    return True\n")
    with pytest.raises(SystemExit, match="returned bool, not wpt_posttrain.Outcome"):
        main([trained.recipe("tune", posttrain=[str(step)]), str(trained.folder), str(tmp_path / "tuned")])
    assert list(tmp_path.glob("*tuned*")) == []


def test_sft_learns_from_the_sources_tagged_sft_and_credits_them(built, tmp_path, monkeypatch, capsys):
    # the built-in step fine-tunes on the conversations a person adds as a
    # source tagged sft; until one exists it says so and writes nothing, and
    # once it has trained, the model's NOTICE credits that source. The CSA
    # indexers keep learning by their own loss, which the step shows
    recipe = built.recipe("chat", training=one_stage(1, 64), posttrain=["sft"])
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
    data = {"schema": 7, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Talks, by Bo Speaker, 2026",
                          "text": "Bo Speaker dedicates these talks to the public domain.\n"}],
            "formats": [{"format": "chat", "paths": ["**"]}], "tags": [{"tags": ["sft"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-15"}}
    data["checksum"] = built.checksum_of(data)
    sid = f"example-talks-{address_hash(str(upstream))}"
    (root / "data" / "sources" / sid).mkdir()
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
    built.download(sid)

    main([recipe, str(base), str(tmp_path / "tuned")])
    tuned = tmp_path / "tuned"
    assert digest(tuned)["model.safetensors"] != digest(base)["model.safetensors"]
    assert "indexer loss" in capsys.readouterr().out
    indexers = [load_model(folder).layers[3].attention.indexer.q.weight for folder in (base, tuned)]
    assert not torch.equal(*indexers)
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
    s3.copy(trained.store)  # the dataset, whose training documents' beginnings the drafter's prompts are
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


def test_grpo_says_so_and_writes_nothing_until_a_source_holds_tasks(trained, tmp_path):
    with pytest.raises(SystemExit, match="step grpo failed: no tasks: no admitted source has task files tagged grpo"):
        main([trained.recipe("agent", posttrain=["grpo"]), str(trained.folder), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()


TASK = {"schema": 1, "instructions": "Write hello into out.txt.", "image": "tiny",
        "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}, "needs": []}


def task_source(built, root, upstream, check='#!/bin/bash\necho "0.$RANDOM"\n') -> str:
    """Writes data/images/tiny/Dockerfile and an admitted source of one task
    tagged grpo, whose check by default prints scores that differ; returns
    its id. The source's upstream is the folder upstream, or the first
    upstream-N whose task the recipes' held_out of 0.5 leaves to training
    (wpt_dataset.tasks.measure_only)."""
    for number in range(100):
        candidate = upstream.with_name(f"{upstream.name}-{number}") if number else upstream
        if not is_held_out(f"example-tasks-{address_hash(str(candidate))}/hello", 0.5):
            upstream = candidate
            break
    (root / "data" / "images" / "tiny").mkdir(parents=True)
    (root / "data" / "images" / "tiny" / "Dockerfile").write_text("ARG BASE=ubuntu:24.04\nFROM ${BASE}\nARG PACKAGES\n")
    (upstream / "hello" / "files").mkdir(parents=True)
    (upstream / "hello" / "task.json").write_text(json.dumps(TASK))
    (upstream / "hello" / "files" / "notes.txt").write_text("start\n")
    (upstream / "hello" / "check").write_text(check)
    (upstream / "hello" / "check").chmod(0o755)
    env = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=env, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()
    data = {"schema": 7, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Tasks, by Cy Author, 2026",
                          "text": "Cy Author dedicates these tasks to the public domain.\n"}],
            "formats": [{"format": "task", "paths": ["**"]}], "tags": [{"tags": ["grpo"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-17"}}
    data["checksum"] = built.checksum_of(data)
    sid = f"example-tasks-{address_hash(str(upstream))}"
    (root / "data" / "sources" / sid).mkdir()
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
    built.download(sid)
    return sid


# a lock of data/images/tiny/ whose image no machine builds: its tag is absent
NEVER_BUILT = {"schema": 1, "base": f"ubuntu@sha256:{'0' * 64}", "snapshot": "20000101T000000Z",
               "packages": {"hello": "0"}}
TINY_IMAGE = "wpt-posttrain/tiny-image"  # in pytest's cache (.pytest_cache/): the pinned and built image of tiny


@pytest.fixture(scope="session")
def tiny_image(request, docker_runs):
    """Gives pinned(root), which pins and builds data/images/tiny/ of root,
    as task_source writes it, and returns the image's tag. The lock and the
    image are kept, in pytest's cache and in Docker, for as long as the
    folder's files (but its lock) and wpt_agent/images.py stay the same: a
    later pinned() writes the kept lock.json and licenses.json into the
    folder, and builds again only when Docker lost the image. When they
    change, the image kept for the old ones is removed."""
    cache = request.config.cache

    def pinned(root):
        digest = hashlib.sha256(Path(images.__file__).read_bytes())
        for path, (executable, content) in sorted(images.files(root, "tiny").items()):
            if path not in ("lock.json", "licenses.json"):
                digest.update(path.encode() + b"\0" + str(executable).encode() + b"\0" + content + b"\0")
        key, folder = digest.hexdigest(), root / "data" / "images" / "tiny"
        kept = cache.get(TINY_IMAGE, None)
        if kept and kept["key"] == key:
            (folder / "lock.json").write_text(kept["lock"])
            (folder / "licenses.json").write_text(kept["licenses"])
            return images.build(root, "tiny", log=lambda _: None)  # at once when Docker has it
        if kept:
            subprocess.run(["docker", "rmi", "-f", kept["tag"]], capture_output=True)
        images.lock(root, "tiny", force=True, log=lambda _: None)
        tag = images.build(root, "tiny", log=lambda _: None)
        cache.set(TINY_IMAGE, {"key": key, "tag": tag, "lock": (folder / "lock.json").read_text(),
                               "licenses": (folder / "licenses.json").read_text()})
        return tag

    return pinned


GRPO = Path(wpt_posttrain.__file__).parent / "steps" / "grpo.py"
TOKEN_EFFICIENCY = Path(wpt_posttrain.__file__).parent / "steps" / "token-efficiency.py"


def answered(monkeypatch):
    """Every attempt ends with an answer, its confidence 0.5, as a model
    that has learned the chat format gives one: a tiny model never does,
    and an attempt without an answer is worth 0 whatever its check says."""
    real = wpt_agent.loop._ended
    monkeypatch.setattr(wpt_agent.loop, "_ended", lambda conversation, ids, learn, ended, answer=None: real(
        conversation, ids, learn, ended, answer or {"answer": "", "confidence": 0.5}))


def step_module(path):
    spec = importlib.util.spec_from_file_location(f"{path.stem.replace('-', '_')}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grpo_fails_and_writes_nothing_when_no_group_ever_teaches_anything(docker_runs, built, tmp_path,
                                                                           monkeypatch):
    # a check that always prints 0 gives every group equal scores: the run
    # makes no update at all, and the step fails, saying why
    recipe = built.recipe("agent", training=one_stage(1, 1024))
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks", check="#!/bin/sh\necho 0\n")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")  # a plain image is enough here
    step = {"step": "grpo", "settings": {"rounds": 2, "group": 2, "tasks_per_update": 1}}
    recipe = built.recipe("agent", training=one_stage(1, 1024), posttrain=[step])
    with pytest.raises(SystemExit, match=r"step grpo failed: no update in 2 rounds: every group was skipped, 2 "
                                         r"because its rewards were all equal, 0 because it kept fewer than 2 "
                                         r"attempts\nnothing was written"):
        main([recipe, str(tmp_path / "model"), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()


class _Logits(torch.nn.Module):
    """A model of 8 tokens whose logits at each position come from the token there."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.table = torch.nn.Embedding(8, 8)

    def forward(self, idx, replay=None, indexer_loss=False):
        logits = self.table(idx)
        return (logits, logits.new_zeros(())) if indexer_loss else logits


def fake_task(name, **fields):
    """A task as agentic reads it, without files or a check."""
    return type("Task", (), {"id": name, "image": "tiny", "needs": (), "replies": (), "tools": (), "answered_by": (),
                             "answered_in": (), "files": {}, "tool_files": {}, "outside": {}, "absent": (),
                             "terse": "", "area": None, "gates": (), **fields})


def test_a_group_whose_rewards_are_all_equal_never_dilutes_the_update(monkeypatch):
    # the loss is averaged over the tokens written in the groups that teach
    # something: a skipped group, whose rewards are all equal, adds none
    scores = {"differ": [1.0, 0.0], "equal": [0.5, 0.5]}
    given = {}

    def play(task, count, keep, offers, terse=False, with_changes=False):
        ids = [0, 1, 2, 3] if task.id == "differ" else [0, 4, 5, 6, 7, 1, 2]
        attempt = wpt_agent.Attempt({"metadata": {}, "messages": []}, ids, [False] + [True] * (len(ids) - 1),
                                    "answered")
        return [keep(index, wpt_agent.Played(attempt, next(given[task.id]))) for index in range(1, count + 1)]

    losses, update = [], agentic.update
    monkeypatch.setattr(agentic, "update", lambda *args: losses.append(update(*args)) or losses[-1])
    grpo = step_module(GRPO)
    settings = {**grpo.SETTINGS, "rounds": 1, "group": 2}
    for names in (["differ", "equal"], ["differ"]):
        given = {name: iter(values) for name, values in scores.items()}
        tasks = [fake_task(name) for name in names]
        learner = agentic.Learner("grpo", _Logits(), None, "cpu", {"tiny": None}, settings)
        learner.play = play
        assert learner.train(tasks, lambda task, index, played: played.score).succeeded
    assert len(losses) == 2 and losses[0] == pytest.approx(losses[1])


def scripted(calls=0, confidence=0.5):
    """An attempt that made `calls` tool calls and answered with that
    confidence (None: no answer)."""
    messages = [{"role": "assistant", "content": [{"type": "tool_call", "id": f"c{n}", "name": "bash",
                                                    "arguments": {}}]} for n in range(calls)]
    answer = None if confidence is None else {"answer": "Done.", "confidence": confidence}
    return wpt_agent.Attempt({"messages": messages, "metadata": {"answer": answer}}, [0, 1], [False, True], "answered")


def test_an_attempt_s_parts_add_up_by_their_weights_and_a_broken_gate_or_no_answer_scores_0():
    # correct 1, calibration 0.5 (Brier:
    # 1 - (confidence - correct)²), each rule 0.25, each tool call -0.01; a
    # gated rule below 1, or no answer, makes the whole attempt worth 0
    settings = step_module(GRPO).SETTINGS
    task, gated = fake_task("t"), fake_task("t", gates=("asked_last",))
    parts = {"correct": 1.0, "sources_first": 1.0, "asked_last": 0.0}
    assert agentic.combined(task, scripted(3, 0.8), parts, settings) == pytest.approx(
        1 + 0.5 * (1 - 0.2 ** 2) + 0.25 * 1 - 0.01 * 3)
    assert agentic.combined(gated, scripted(3, 0.8), parts, settings) == 0.0
    assert agentic.combined(task, scripted(3, None), parts, settings) == 0.0
    assert agentic.combined(task, scripted(), None, settings) == 0.0  # the check gave no score
    # stating a low confidence never pays more than being right
    right = agentic.combined(task, scripted(0, 0.0), {"correct": 1.0}, settings)
    assert right > agentic.combined(task, scripted(0, 0.0), {"correct": 0.0}, settings) == 0.5


def test_each_terse_attempt_is_rewarded_for_its_closeness_to_the_full_attempt_of_the_same_index(monkeypatch):
    # a task with a one-line form is
    # played in full, then terse, both forms one group, so each advantage is
    # taken from the mean of all their attempts; terse attempt i adds,
    # weighed, how alike its changes to /work are to full attempt i's; an
    # answer-only task (an empty area) gets no closeness; where closeness
    # counts, a terse attempt whose full attempt Docker dropped is not
    # trained on
    full = {1: "+a\n+b\n", 2: "+c\n", 3: None}  # None: dropped
    terse = {1: "+a\n+b\n", 2: "+x\n", 3: "+c\n"}
    played_forms = []

    def play(task, count, keep, offers, terse_form=False, with_changes=False):
        changes = terse if terse_form else full
        played_forms.append((terse_form, with_changes))
        return [keep(index, wpt_agent.SandboxError("lost") if changes[index] is None else
                     wpt_agent.Played(scripted(), 1.0, None, changes[index] if with_changes else None,
                                      {"correct": 1.0})) for index in range(1, count + 1)]

    settings = {**step_module(GRPO).SETTINGS, "rounds": 1, "group": 3}
    updated, update = [], agentic.update
    monkeypatch.setattr(agentic, "update", lambda model, optimizer, groups, device:
                        updated.append([[value for *_, value in group] for group in groups])
                        or update(model, optimizer, groups, device))
    for area, near in ((None, [1.0, 0.0]), ([], None)):
        kept = []
        learner = agentic.Learner("grpo", _Logits(), None, "cpu", {"tiny": None}, settings, terse=True)
        learner.play = play
        monkeypatch.setattr(learner, "_keep", lambda task, round_, group, index, one, offer, terse_form=False:
                            kept.append((terse_form, index, one.reward)) or len(kept))
        learner.train([fake_task("t", terse="Say hi.", area=area)])
        base = agentic.combined(fake_task("t"), scripted(), {"correct": 1.0}, settings)
        expected = [base + settings["weight_closeness"] * value for value in near] + [None] if near else [base] * 3
        assert [reward for terse_form, _, reward in kept if terse_form] == [
            pytest.approx(value) if value is not None else None for value in expected]
        assert [reward for terse_form, _, reward in kept if not terse_form] == [base, base, None]
    # the first task's rewards differ, so it is learned from, as one group of the four kept; the second's are equal
    assert updated == [[[base, base, *map(pytest.approx, [base + settings["weight_closeness"], base])]], []]
    assert played_forms == [(False, True), (True, True), (False, False), (True, False)]


def test_grpo_fails_and_writes_nothing_when_every_attempt_of_a_round_is_dropped(built, tmp_path, monkeypatch):
    # a sandbox that never starts is not the model's doing, but a round that
    # could make no update at all stops the step, naming the round and why
    recipe = built.recipe("agent", training=one_stage(1, 1024))
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks")

    def refused(*args, **kwargs):
        raise wpt_agent.SandboxError("docker create: no space left on device")

    monkeypatch.setattr(images, "built", lambda root, name: f"wpt-agent/{name}:test")  # no image to build
    monkeypatch.setattr(played, "Sandbox", refused)
    recipe = built.recipe("agent", training=one_stage(1, 1024), posttrain=["grpo"])
    with pytest.raises(SystemExit, match=r"step grpo failed: round 1: every attempt was dropped, Docker failed "
                                         r"8 times; the last time: docker create: no space left on device"):
        main([recipe, str(tmp_path / "model"), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()
    # the dropped attempts are kept too, each as the conversation it would have
    # started from, and the round's event names their lines
    [post] = built.records("wpt-posttrain")
    lines = attempt_lines(built.store, post.folder, "grpo-attempts.jsonl")
    assert [(line["metadata"]["round"], line["metadata"]["group"], line["metadata"]["attempt"]) for line in lines] == [
        (1, 1, i) for i in range(1, 9)]
    for line in lines:
        metadata = line["metadata"]
        assert metadata["dropped"] and "no space left" in metadata["error"]
        assert metadata["ended"] is metadata["score"] is metadata["reward"] is metadata["written"] is None
        assert [m["role"] for m in line["messages"]] == ["user"]
        assert line["messages"][0]["content"][0]["text"] == TASK["instructions"]
    [round_] = [e for e in post.events if e.get("name") == "grpo.round"]
    assert round_["attributes"]["file"] == "grpo-attempts.jsonl" and round_["attributes"]["lines"] == [1, 8]


def attempt_lines(store, folder, name) -> list[dict]:
    """The attempts file's lines, each checked as a chat-format conversation."""
    return conversations.parse("record", name, store.read_record(folder, name))


def test_a_refused_attempt_line_stops_the_run_with_the_record_error_code(built, tmp_path, monkeypatch):
    # an attempt the record cannot keep stops the command at once, as any
    # part of the record does: exit code 3, nothing written
    recipe = built.recipe("agent", training=one_stage(1, 1024))
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks")

    def refused(*args, **kwargs):
        raise wpt_agent.SandboxError("docker create: no space left on device")

    real = LocalStore._raw_append_record

    def disk_full(self, folder, name, data):
        if name == "grpo-attempts.jsonl":
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(images, "built", lambda root, name: f"wpt-agent/{name}:test")
    monkeypatch.setattr(played, "Sandbox", refused)  # a dropped attempt is kept too: no Docker needed
    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full)
    recipe = built.recipe("agent", training=one_stage(1, 1024), posttrain=["grpo"])
    with pytest.raises(SystemExit) as stopped:
        main([recipe, str(tmp_path / "model"), str(tmp_path / "tuned")])
    assert stopped.value.code == RECORD_FAILED
    assert not (tmp_path / "tuned").exists()


@pytest.mark.xdist_group("tiny-image")  # one worker: the first of these tests pins and builds the image
def test_grpo_learns_from_scored_attempts_at_the_tasks_tagged_grpo(docker_runs, built, tmp_path, tiny_image,
                                                                    monkeypatch):
    # each attempt runs in a container of the task's image and its check
    # scores it; attempts that score apart change the weights, but not the
    # routers', and the new model's NOTICE credits the source of the tasks;
    # the image must be built first, and the step says how. The opening
    # alone (the four tools' descriptions) is longer than block_size: the
    # model reads the attempts through its cache, up to the task's max_tokens
    recipe = built.recipe("agent", training=one_stage(1, 64))
    train([recipe, built.dataset_id, "--out", "model"])
    base, root = tmp_path / "model", tmp_path
    sid = task_source(built, root, tmp_path / "tasks")
    (root / "data" / "images" / "tiny" / "lock.json").write_text(dump_json(NEVER_BUILT))  # no image built yet

    answered(monkeypatch)
    step = {"step": "grpo", "settings": {"rounds": 1, "group": 2, "tasks_per_update": 1, "learning_rate": 1e-2}}
    recipe = built.recipe("agent", training=one_stage(1, 64), posttrain=[step])
    with pytest.raises(SystemExit, match="image 'tiny' is not built on this machine.*wpt-agent build-image tiny"):
        main([recipe, str(base), str(tmp_path / "tuned")])
    tiny_image(root)
    main([recipe, str(base), str(tmp_path / "tuned")])

    tuned = tmp_path / "tuned"
    before, after = load_file(base / "model.safetensors"), load_file(tuned / "model.safetensors")
    routers = [name for name in before if name.endswith("experts.router.weight")]
    assert routers and all(torch.equal(before[name], after[name]) for name in routers)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    post = json.loads((tuned / "provenance.json").read_text())["post_training"]
    assert post["steps"][0]["file"] == "wpt_posttrain/steps/grpo.py" and post["steps"][0]["sources"] == [sid]
    assert post["steps"][0]["settings"] == {**step_module(GRPO).SETTINGS, **step["settings"]}
    notice = (tuned / "NOTICE").read_text()
    assert "Tasks, by Cy Author, 2026" in notice and "dedicates these tasks" in notice
    # every attempt is kept in the run's record, as a conversation in the chat format, and its span names its line
    record = built.records("wpt-posttrain")[-1]
    lines = attempt_lines(built.store, record.folder, "grpo-attempts.jsonl")
    assert [tuple(line["metadata"][key] for key in ("step", "task", "round", "group", "attempt")) for line in lines] == [
        ("grpo", f"{sid}/hello", 1, 1, 1), ("grpo", f"{sid}/hello", 1, 1, 2)]
    for line in lines:
        metadata = line["metadata"]
        assert not metadata["dropped"] and metadata["error"] is None
        # the check prints one number, its correct part; the reward weighs it with the calibration and tool calls
        assert metadata["parts"] == {"correct": metadata["score"]} and 0 <= metadata["score"] <= 1
        assert metadata["reward"] == pytest.approx(metadata["score"] + 0.5 * (1 - (0.5 - metadata["score"]) ** 2)
                                                   - 0.01 * metadata["tool_calls"])
        assert metadata["written"] > 0 and metadata["terse"] is False
        assert metadata["ended"] in ("answered", "ended", "turns", "tokens", "context", "full")
        assert line["messages"][0]["content"][0]["text"] == TASK["instructions"]
        assert line["messages"][1]["role"] == "assistant"
    spans = [e for e in record.events if e["kind"] == "span_end" and e["name"] == "grpo.attempt"]
    assert [(span["attributes"]["file"], span["attributes"]["line"]) for span in spans] == [
        ("grpo-attempts.jsonl", 1), ("grpo-attempts.jsonl", 2)]
    [group] = [e for e in record.events if e.get("name") == "grpo.group"]
    assert group["attributes"]["lines"] == [1, 2]


def attempt_of(written, read=3):
    """An attempt whose model wrote `written` tokens between the `read`
    tokens of its instructions and a tool result of `read` tokens."""
    learn = [False] * read + [True] * written + [False] * read + [True]
    return wpt_agent.Attempt({"messages": [], "metadata": {"answer": {"answer": "Hello.", "confidence": 0.5}}},
                             [0] * len(learn), learn, "answered")


def test_token_efficiency_rewards_a_passing_attempt_by_how_few_tokens_it_wrote():
    # below the best start score an attempt earns nothing, whatever its
    # weighed parts; at or above it, its weighed parts times the reference
    # length over the tokens the model wrote, never counting the
    # instructions, the tool results or the template's tokens
    step = step_module(TOKEN_EFFICIENCY)
    reference = (0.5, 12.0)  # best start score, reference length
    assert step.reward(reference, attempt_of(2), 0.4, 1.5) == 0.0  # short, but below the best start score
    assert step.reward(reference, attempt_of(11, read=50), 0.5, 1.0) == pytest.approx(1.0)  # 12 tokens, as long
    assert step.reward(reference, attempt_of(5), 1.0, 1.0) == pytest.approx(2.0)  # half as long, scoring higher
    assert step.reward(reference, attempt_of(23), 0.5, 1.0) == pytest.approx(0.5)  # twice as long
    assert step.reward(reference, attempt_of(11), 0.5, 1.5) == pytest.approx(1.5)  # as long, its parts weigh more


def test_each_attempt_s_line_is_kept_as_soon_as_it_is_scored(monkeypatch):
    # a task's attempts are written
    # together, but each attempt's line is kept right after it is scored,
    # before the next one is, as when attempts ran one by one; a dropped
    # attempt takes its place in the order
    class Box:
        made = 0

        def __init__(self, *args, **kwargs):
            Box.made += 1
            if Box.made == 2:
                raise wpt_agent.SandboxError("docker create: no space left on device")

        def close(self):
            pass

    def attempts(task, boxes, tokenizer, start, result_bytes, offers, terse):
        start([0], len(boxes), ())
        return [wpt_agent.Attempt({}, [0, 1], [False, True], "answered") for _ in boxes]

    kept, scored = [], []
    monkeypatch.setattr(played, "Sandbox", Box)
    monkeypatch.setattr(played, "attempts", attempts)
    monkeypatch.setattr(played, "score",
                        lambda box, check, seconds, conversation, beside, hidden, outside: scored.append(list(kept)) or
                        ({"correct": 1.0}, None))
    monkeypatch.setattr(wpt_inference, "Writer",
                        lambda model, ids, rows, **kwargs: type("W", (), {"routes": [None] * rows}))
    grpo = step_module(GRPO)
    settings = {**grpo.SETTINGS, "rounds": 1, "tasks_per_update": 1, "group": 3}
    learner = agentic.Learner("grpo", _Logits(), None, "cpu", {"tiny": None}, settings)
    monkeypatch.setattr(learner, "_keep",
                        lambda task, round_, group, index, one, offer, terse: kept.append(index) or len(kept))
    task = type("Task", (), {"id": "t", "image": "tiny", "files": None, "check": None, "needs": (), "replies": (),
                             "tools": (), "tool_files": {}, "outside": {}, "check_files": {}, "answered_by": (),
                             "answered_in": (), "absent": (), "area": None, "gates": (),
                             "container": {"memory": 1, "disk": 1, "processes": 1, "cpus": 1}})
    outcomes = learner.attempts(task, 1, 1, 3, lambda task, index, played: played.score)
    assert scored == [[], [1, 2]]  # attempt 1 scored before any line, attempt 3 after the lines of 1 and 2
    assert kept == [1, 2, 3] and [one.attempt is None for one in outcomes] == [False, True, False]


def test_token_efficiency_takes_the_mean_length_of_the_attempts_that_reach_the_best_start_score():
    # per task: the best result the incoming model reached (its correct
    # part, never the weighed parts, which calibration and tool calls move),
    # and the mean length of its attempts that reach it; a task the model
    # was always wrong on, or whose every attempt Docker dropped, is left out
    step = step_module(TOKEN_EFFICIENCY)
    plays = {"mixed": [(0.5, 4), (0.75, 10), (0.25, 2), (0.75, 20)], "zero": [(0.0, 3), (0.0, 7)], "lost": []}

    class Learner(agentic.Learner):
        def __init__(self):  # no model: play() stands for the attempts, attempts() keeps them as the step does
            self.name, self.file, self.telemetry = "token-efficiency", "token-efficiency-attempts.jsonl", Telemetry.off()
            self.settings = {**step.SETTINGS, "reference": 4}
            self.pools = {}

        def play(self, task, count, keep, offers, terse=False, with_changes=False):
            outcomes = [(score, length) for score, length in plays[task.id][:count]]
            outcomes += [wpt_agent.SandboxError("docker create: no space left on device")] * (count - len(outcomes))
            return [keep(index, outcome if isinstance(outcome, Exception)
                         else wpt_agent.Played(attempt_of(outcome[1] - 1), outcome[0], parts={"correct": outcome[0]}))
                    for index, outcome in enumerate(outcomes, 1)]

    tasks = [fake_task(name, instructions="Say hello.") for name in plays]
    references, zero, lost = step._references(Learner(), tasks)
    assert references == {"mixed": (0.75, pytest.approx(15.0))}
    assert (zero, lost) == (1, 1)


def test_token_efficiency_fails_and_writes_nothing_when_no_task_is_left(docker_runs, built, tmp_path, monkeypatch):
    # the incoming model scores 0 on every reference attempt: the task has
    # no result to keep, and with no task left the step fails, saying why
    recipe = built.recipe("agent", training=one_stage(1, 1024))
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks", check="#!/bin/sh\necho 0\n")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")  # a plain image is enough here
    recipe = built.recipe("agent", training=one_stage(1, 1024),
                          posttrain=[{"step": "token-efficiency", "settings": {"reference": 2}}])
    with pytest.raises(SystemExit, match=r"step token-efficiency failed: no task left of the 1 tasks tagged grpo: 1 "
                                         r"scored 0 on every reference attempt, Docker failed every reference "
                                         r"attempt of 0\nnothing was written"):
        main([recipe, str(tmp_path / "model"), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()


@pytest.mark.xdist_group("tiny-image")
def test_token_efficiency_learns_from_the_tasks_tagged_grpo(docker_runs, built, tmp_path, tiny_image, monkeypatch):
    # the incoming model's attempts at grpo's own tasks set the reference,
    # then attempts that pass and fail it change the weights, but not the
    # routers', and the new model credits the source of the tasks
    recipe = built.recipe("agent", training=one_stage(1, 1024))
    train([recipe, built.dataset_id, "--out", "model"])
    base, root = tmp_path / "model", tmp_path
    sid = task_source(built, root, tmp_path / "tasks", check="#!/bin/bash\necho $((RANDOM % 2))\n")
    tiny_image(root)
    # 16 reference attempts all fail 1 time in 65536; 8 attempts all equal, in each of 2 rounds, 1 time in 16384
    answered(monkeypatch)
    step = {"step": "token-efficiency", "settings": {"rounds": 2, "group": 8, "reference": 16, "tasks_per_update": 1,
                                                     "learning_rate": 1e-2}}
    recipe = built.recipe("agent", training=one_stage(1, 1024), posttrain=[step])
    main([recipe, str(base), str(tmp_path / "tuned")])

    tuned = tmp_path / "tuned"
    before, after = load_file(base / "model.safetensors"), load_file(tuned / "model.safetensors")
    routers = [name for name in before if name.endswith("experts.router.weight")]
    assert routers and all(torch.equal(before[name], after[name]) for name in routers)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    post = json.loads((tuned / "provenance.json").read_text())["post_training"]
    assert post["steps"][0]["file"] == "wpt_posttrain/steps/token-efficiency.py" and post["steps"][0]["sources"] == [sid]
    assert "Tasks, by Cy Author, 2026" in (tuned / "NOTICE").read_text()
    # the reference attempts are kept first, with no round and no reward, then the rounds' attempts
    [record] = built.records("wpt-posttrain")
    lines = attempt_lines(built.store, record.folder, "token-efficiency-attempts.jsonl")
    metadata = [line["metadata"] for line in lines]
    assert [(m["round"], m["attempt"]) for m in metadata] == (
        [(None, i) for i in range(1, 17)] + [(r, i) for r in (1, 2) for i in range(1, 9)])
    assert all(m["step"] == "token-efficiency" for m in metadata)
    assert all(m["reward"] is None for m in metadata[:16])
    assert all(m["reward"] is not None for m in metadata[16:] if not m["dropped"])
