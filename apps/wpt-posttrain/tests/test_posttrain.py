import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import wpt_agent
import wpt_posttrain
from wpt_agent import images
from wpt_dataset import manifest
from wpt_dataset.repo import address_hash, dump_json
from wpt_dataset.sources import compute_upstream
from wpt_model import load_model
from wpt_eval import main as evaluate
from wpt_dataset import conversations
from wpt_posttrain import agentic, main
from wpt_store import LocalStore
from wpt_telemetry import RECORD_FAILED, Telemetry
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


def test_sft_learns_from_the_sources_tagged_sft_and_credits_them(built, tmp_path, monkeypatch, capsys):
    # the built-in step fine-tunes on the conversations a person adds as a
    # source tagged sft; until one exists it says so and writes nothing, and
    # once it has trained, the model's NOTICE credits that source. The CSA
    # indexers keep learning by their own loss, which the step shows
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
        "limits": {"max_turns": 2, "max_tokens": 6, "command_seconds": 5}}


def task_source(built, root, upstream, check='#!/bin/bash\necho "0.$RANDOM"\n') -> str:
    """Writes data/images/tiny/Dockerfile and an admitted source of one task
    tagged grpo, whose check by default prints scores that differ; returns
    its id."""
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
    data = {"schema": 5, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Tasks, by Cy Author, 2026",
                          "text": "Cy Author dedicates these tasks to the public domain.\n"}],
            "formats": [{"format": "task", "paths": ["**"]}], "tags": [{"tags": ["grpo"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-17"}}
    data["checksum"] = manifest.checksum(compute_upstream(data, data["address"], built.store))
    sid = f"example-tasks-{address_hash(str(upstream))}"
    (root / "data" / "sources" / sid).mkdir()
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
    return sid


GRPO = Path(wpt_posttrain.__file__).parent / "steps" / "grpo.py"
TOKEN_EFFICIENCY = Path(wpt_posttrain.__file__).parent / "steps" / "token-efficiency.py"


def step_copy(folder, original=GRPO, **settings) -> Path:
    """The documented way to change a step's settings: a copy of its file,
    under the same name, with the given constants changed."""
    text = original.read_text()
    for name, value in settings.items():
        text = re.sub(rf"^{name} = .*$", f"{name} = {value}", text, count=1, flags=re.MULTILINE)
    step = folder / original.name
    step.write_text(text)
    return step


def step_module(path):
    spec = importlib.util.spec_from_file_location(f"{path.stem.replace('-', '_')}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grpo_fails_and_writes_nothing_when_no_group_ever_teaches_anything(docker_runs, built, tmp_path,
                                                                           monkeypatch):
    # a check that always prints 0 gives every group equal scores: the run
    # makes no update at all, and the step fails, saying why
    recipe = built.recipe("agent", model={"block_size": 1024})
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks", check="#!/bin/sh\necho 0\n")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")  # a plain image is enough here
    step = step_copy(tmp_path, ROUNDS="2", GROUP="2", TASKS_PER_UPDATE="1")
    recipe = built.recipe("agent", model={"block_size": 1024}, posttrain=[str(step)])
    with pytest.raises(SystemExit, match=r"step grpo failed: no update in 2 rounds: every group was skipped, 2 "
                                         r"because its scores were all equal, 0 because it kept fewer than 2 "
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


def test_a_group_whose_scores_are_all_equal_never_dilutes_the_update(monkeypatch):
    # the loss is averaged over the tokens written in the groups that teach
    # something: a skipped group, whose scores are all equal, adds none
    scores = {"differ": [1.0, 0.0], "equal": [0.5, 0.5]}
    played = {}

    def play(task):
        ids = [0, 1, 2, 3] if task.id == "differ" else [0, 4, 5, 6, 7, 1, 2]
        attempt = wpt_agent.Attempt({}, ids, [False] + [True] * (len(ids) - 1), "answered")
        return attempt, None, next(played[task.id])

    losses, update = [], agentic.update
    monkeypatch.setattr(agentic, "update", lambda *args: losses.append(update(*args)) or losses[-1])
    grpo = step_module(GRPO)
    settings = agentic.Settings(rounds=1, tasks_per_update=grpo.TASKS_PER_UPDATE, group=2,
                                learning_rate=grpo.LEARNING_RATE, weight_decay=grpo.WEIGHT_DECAY,
                                temperature=grpo.TEMPERATURE, top_k=grpo.TOP_K, check_seconds=grpo.CHECK_SECONDS,
                                result_bytes=grpo.RESULT_BYTES, seed=grpo.SEED)
    task = lambda name: type("Task", (), {"id": name, "image": "tiny"})
    for names in (["differ", "equal"], ["differ"]):
        played = {name: iter(values) for name, values in scores.items()}
        tasks = [task(name) for name in names]
        learner = agentic.Learner("grpo", _Logits(), None, "cpu", {"tiny": None}, 100, settings)
        learner.play = play
        assert learner.train(tasks).succeeded
    assert len(losses) == 2 and losses[0] == pytest.approx(losses[1])


def test_grpo_fails_and_writes_nothing_when_every_attempt_of_a_round_is_dropped(built, tmp_path, monkeypatch):
    # a sandbox that never starts is not the model's doing, but a round that
    # could make no update at all stops the step, naming the round and why
    recipe = built.recipe("agent", model={"block_size": 1024})
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks")

    def refused(*args, **kwargs):
        raise wpt_agent.SandboxError("docker create: no space left on device")

    monkeypatch.setattr(images, "built", lambda root, name: f"wpt-agent/{name}:test")  # no image to build
    monkeypatch.setattr(wpt_agent, "Sandbox", refused)
    recipe = built.recipe("agent", model={"block_size": 1024}, posttrain=["grpo"])
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
    recipe = built.recipe("agent", model={"block_size": 1024})
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
    monkeypatch.setattr(wpt_agent, "Sandbox", refused)  # a dropped attempt is kept too: no Docker needed
    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full)
    recipe = built.recipe("agent", model={"block_size": 1024}, posttrain=["grpo"])
    with pytest.raises(SystemExit) as stopped:
        main([recipe, str(tmp_path / "model"), str(tmp_path / "tuned")])
    assert stopped.value.code == RECORD_FAILED
    assert not (tmp_path / "tuned").exists()


def test_grpo_learns_from_scored_attempts_at_the_tasks_tagged_grpo(docker_runs, built, tmp_path):
    # each attempt runs in a container of the task's image and its check
    # scores it; attempts that score apart change the weights, but not the
    # routers', and the new model's NOTICE credits the source of the tasks;
    # the image must be built first, and the step says how
    recipe = built.recipe("agent", model={"block_size": 1024})  # the four tools' descriptions take most of it
    train([recipe, built.dataset_id, "--out", "model"])
    base, root = tmp_path / "model", tmp_path
    sid = task_source(built, root, tmp_path / "tasks")
    images.lock(root, "tiny", log=lambda _: None)  # no image built yet

    step = step_copy(tmp_path, ROUNDS="1", GROUP="2", TASKS_PER_UPDATE="1", LEARNING_RATE="1e-2")
    recipe = built.recipe("agent", model={"block_size": 1024}, posttrain=[str(step)])
    with pytest.raises(SystemExit, match="image 'tiny' is not built on this machine.*wpt-agent build-image tiny"):
        main([recipe, str(base), str(tmp_path / "tuned")])
    try:
        images.build(root, "tiny", log=lambda _: None)
        main([recipe, str(base), str(tmp_path / "tuned")])
    finally:
        subprocess.run(["docker", "rmi", "-f", images.tag(root, "tiny")], capture_output=True)

    tuned = tmp_path / "tuned"
    before, after = load_file(base / "model.safetensors"), load_file(tuned / "model.safetensors")
    routers = [name for name in before if name.endswith("experts.router.weight")]
    assert routers and all(torch.equal(before[name], after[name]) for name in routers)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    post = json.loads((tuned / "provenance.json").read_text())["post_training"]
    assert post["steps"][0]["file"] == "grpo.py" and post["steps"][0]["sources"] == [sid]
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
        assert metadata["reward"] == metadata["score"] and 0 <= metadata["score"] <= 1 and metadata["written"] > 0
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
    return wpt_agent.Attempt({}, [0] * len(learn), learn, "answered")


def test_token_efficiency_rewards_a_passing_attempt_by_how_few_tokens_it_wrote():
    # below the best start score an attempt earns nothing; at or above it,
    # the reference length over the tokens the model wrote, never counting
    # the instructions, the tool results or the template's tokens
    step = step_module(TOKEN_EFFICIENCY)
    reference = (0.5, 12.0)  # best start score, reference length
    assert step.reward(reference, attempt_of(2), 0.4) == 0.0  # short, but below the best start score
    assert step.reward(reference, attempt_of(11, read=50), 0.5) == pytest.approx(1.0)  # 12 tokens written, as long
    assert step.reward(reference, attempt_of(5), 1.0) == pytest.approx(2.0)  # half as long, scoring higher
    assert step.reward(reference, attempt_of(23), 0.5) == pytest.approx(0.5)  # twice as long


def test_token_efficiency_takes_the_mean_length_of_the_attempts_that_reach_the_best_start_score():
    # per task: the best score the incoming model reached, and the mean
    # length of its attempts that reach it; a task the model always scored
    # 0 on, or whose every attempt Docker dropped, is left out
    step = step_module(TOKEN_EFFICIENCY)
    plays = {"mixed": [(0.5, 4), (0.75, 10), (0.25, 2), (0.75, 20)], "zero": [(0.0, 3), (0.0, 7)], "lost": []}

    class Learner(agentic.Learner):
        def __init__(self):  # no model: play() stands for the attempts, attempt() keeps them as the step does
            self.name, self.file, self.telemetry = "token-efficiency", "token-efficiency-attempts.jsonl", Telemetry.off()

        def play(self, task):
            if not plays[task.id]:
                raise wpt_agent.SandboxError("docker create: no space left on device")
            score, length = plays[task.id].pop(0)
            return attempt_of(length - 1), None, score

    step.REFERENCE = 4
    tasks = [type("Task", (), {"id": name, "instructions": "Say hello."}) for name in plays]
    references, zero, lost = step._references(Learner(), tasks)
    assert references == {"mixed": (0.75, pytest.approx(15.0))}
    assert (zero, lost) == (1, 1)


def test_token_efficiency_fails_and_writes_nothing_when_no_task_is_left(docker_runs, built, tmp_path, monkeypatch):
    # the incoming model scores 0 on every reference attempt: the task has
    # no result to keep, and with no task left the step fails, saying why
    recipe = built.recipe("agent", model={"block_size": 1024})
    train([recipe, built.dataset_id, "--out", "model"])
    task_source(built, tmp_path, tmp_path / "tasks", check="#!/bin/sh\necho 0\n")
    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    monkeypatch.setattr(images, "built", lambda root, name: "ubuntu:24.04")  # a plain image is enough here
    step = step_copy(tmp_path, TOKEN_EFFICIENCY, REFERENCE="2")
    recipe = built.recipe("agent", model={"block_size": 1024}, posttrain=[str(step)])
    with pytest.raises(SystemExit, match=r"step token-efficiency failed: no task left of the 1 tasks tagged grpo: 1 "
                                         r"scored 0 on every reference attempt, Docker failed every reference "
                                         r"attempt of 0\nnothing was written"):
        main([recipe, str(tmp_path / "model"), str(tmp_path / "tuned")])
    assert not (tmp_path / "tuned").exists()


def test_token_efficiency_learns_from_the_tasks_tagged_grpo(docker_runs, built, tmp_path):
    # the incoming model's attempts at grpo's own tasks set the reference,
    # then attempts that pass and fail it change the weights, but not the
    # routers', and the new model credits the source of the tasks
    recipe = built.recipe("agent", model={"block_size": 1024})
    train([recipe, built.dataset_id, "--out", "model"])
    base, root = tmp_path / "model", tmp_path
    sid = task_source(built, root, tmp_path / "tasks", check="#!/bin/bash\necho $((RANDOM % 2))\n")
    images.lock(root, "tiny", log=lambda _: None)
    # 16 reference attempts all fail 1 time in 65536; 8 attempts all equal, in each of 2 rounds, 1 time in 16384
    step = step_copy(tmp_path, TOKEN_EFFICIENCY, ROUNDS="2", GROUP="8", REFERENCE="16", TASKS_PER_UPDATE="1",
                     LEARNING_RATE="1e-2")
    recipe = built.recipe("agent", model={"block_size": 1024}, posttrain=[str(step)])
    try:
        images.build(root, "tiny", log=lambda _: None)
        main([recipe, str(base), str(tmp_path / "tuned")])
    finally:
        subprocess.run(["docker", "rmi", "-f", images.tag(root, "tiny")], capture_output=True)

    tuned = tmp_path / "tuned"
    before, after = load_file(base / "model.safetensors"), load_file(tuned / "model.safetensors")
    routers = [name for name in before if name.endswith("experts.router.weight")]
    assert routers and all(torch.equal(before[name], after[name]) for name in routers)
    assert any(not torch.equal(before[name], after[name]) for name in before)
    post = json.loads((tuned / "provenance.json").read_text())["post_training"]
    assert post["steps"][0]["file"] == "token-efficiency.py" and post["steps"][0]["sources"] == [sid]
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
