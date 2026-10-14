import json
import os
import subprocess
from types import SimpleNamespace

import pytest
import torch

from wpt_dataset import manifest
from wpt_dataset.build import build_dataset
from wpt_dataset.repo import Repo, address_hash, dump_json
from wpt_dataset.sources import compute_upstream
from wpt_eval import main, open_context, run_eval
from wpt_store import open_store
from wpt_tokenizer import Tokenizer
from wpt_train import main as train

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """(model folder, source id) trained through wpt-dataset and wpt-train on
    one git project of 16 documents, about half of them held out; the
    3000 training tokens draw only some of the training documents."""
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
    data = {"schema": 2, "type": "git", "address": str(upstream), "revision": commit,
            "licenses": [{"license": "MIT", "paths": ["**"], "attribution": "Ada Writer"}],
            "tags": [{"tags": ["en"], "paths": ["**"]}],
            "decision": {"decision": "admitted", "by": "T", "date": "2026-10-14", "reason": "test"}}
    data["checksum"] = manifest.checksum(compute_upstream(data, data["address"], store))
    sid = f"example-fox-{address_hash(str(upstream))}"
    (root / "data" / "sources" / sid).mkdir(parents=True)
    (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
    for kind, name, document in [("mixture", "m", {"pools": [{"name": "en", "select": "en", "weight": 1}]}),
                                 ("tokenizer", "t", {"mixture": "m", "vocab_size": 300}),
                                 ("dataset", "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 3000, "held_out": 0.5})]:
        repo.recipe_path(kind, name).parent.mkdir(parents=True, exist_ok=True)
        repo.recipe_path(kind, name).write_text(json.dumps({"schema": 1, **document}))
    dataset_id = build_dataset(repo, "d", store, log=lambda _: None)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.chdir(tmp_path)
    train([dataset_id, "--out", "model", "--steps", "1", "--block-size", "16", "--n-layer", "1",
           "--n-head", "2", "--n-embd", "16", "--batch-size", "4"])
    return tmp_path / "model", sid


class Recall(torch.nn.Module):
    """A stand-in model that knows its training documents by heart: after
    any start of one of them it gives the next token, otherwise token 0."""

    def __init__(self, documents, vocab_size):
        super().__init__()
        self.config = SimpleNamespace(block_size=1000)
        self.vocab_size = vocab_size
        self.next = {tuple([Tokenizer.END, *d.tokens[:i].tolist()]): int(d.tokens[i])
                     for d in documents for i in range(len(d.tokens))}

    def forward(self, ids):
        logits = torch.zeros(ids.size(0), ids.size(1), self.vocab_size)
        for row, seen in enumerate(ids.tolist()):
            logits[row, -1, self.next.get(tuple(seen), 0)] = 1
        return logits


def test_memorization_counts_the_training_documents_a_model_reproduces(trained):
    # given a model that recalls every training document and no held-out one,
    # the test must report all training documents and no held-out document
    folder, sid = trained
    context = open_context(folder, open_store(os.environ["WPT_STORE"]), torch.device("cpu"))
    training, held_out = context.documents("training"), context.documents("held_out")
    assert training and held_out
    context.model = Recall(training, context.tokenizer.vocab_size)

    data = run_eval(context, "memorization")["data"]
    assert data["training"]["exact"] == data["training"]["documents"] == len(training)
    assert data["training"]["exact_by_source"] == {sid: len(training)}
    assert data["held_out"]["exact"] == 0 and data["held_out"]["documents"] == len(held_out)


def test_the_command_runs_every_built_in_test_into_the_model_card(trained, capsys):
    folder, _ = trained
    main([str(folder)])

    result = json.loads((folder / "evals" / "memorization.json").read_text())
    assert result["name"] == "memorization" and result["file"] == "wpt_eval/evals/memorization.py"
    store = open_store(os.environ["WPT_STORE"])
    counts = json.loads(store.get_bytes(json.loads((folder / "provenance.json").read_text())["dataset_record"]))["documents"]
    assert result["data"]["held_out"]["documents"] == counts["held_out"]  # every one is long enough
    assert 0 < result["data"]["training"]["documents"] <= counts["training"]
    card = (folder / "README.md").read_text()
    assert "### memorization" in card and result["text"].rstrip() in card
    assert result["text"].rstrip() in capsys.readouterr().out


def test_any_python_file_can_be_a_test(trained, tmp_path):
    folder, _ = trained
    test = tmp_path / "count_tokens.py"
    test.write_text(
        "from wpt_eval import Result\n\n"
        "def run(context):\n"
        "    n = sum(len(d.tokens) for d in context.documents('held_out'))\n"
        "    return Result({'held_out_tokens': n}, f'{n} held-out tokens')\n")
    main([str(folder), str(test)])

    result = json.loads((folder / "evals" / "count_tokens.json").read_text())
    assert result["file"] == "count_tokens.py" and result["data"]["held_out_tokens"] > 0
    assert f"{result['data']['held_out_tokens']} held-out tokens" in (folder / "README.md").read_text()


def test_a_test_that_returns_no_result_writes_nothing(trained, tmp_path):
    folder, _ = trained
    test = tmp_path / "broken.py"
    test.write_text("def run(context):\n    return 'all good'\n")
    with pytest.raises(SystemExit, match="returned str, not wpt_eval.Result"):
        main([str(folder), str(test)])
    assert not (folder / "evals" / "broken.json").exists()
