"""The held-out-loss test (docs/evaluation.md, "Held-out loss"): every
held-out token's loss once, grouped by its document's source and by the
kind of text --kind-by names; and the model's losses as the gpu worker
computes them."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wpt.eval import Context
from wpt_model import load_model
from wpt_store import open_store
from wpt_store.shards import write_shard
from wpt_tokenizer import END
from wpt_worker import evaluation
from wpt_worker.evaluation import Served

FILE = Path(__file__).resolve().parents[2] / "src" / "wpt" / "eval" / "evals" / "held-out-loss.py"
ROOT = Path(__file__).resolve().parents[4]


def held_out_loss():
    spec = importlib.util.spec_from_file_location("held_out_loss", FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TokenLoss:
    """A stand-in model whose loss on a token is the token's own id, so the
    loss of a set of tokens is known from the tokens alone."""

    def __init__(self, block_size):
        self.config = SimpleNamespace(block_size=block_size)
        self.windows = []

    def losses(self, windows):
        self.windows += windows
        return [np.asarray(window[1:], dtype=np.float32) for window in windows]


def test_each_held_out_token_counts_once_toward_its_source_and_its_kind_of_text(tmp_path, monkeypatch):
    # given held-out documents of two gutenberg sources and one github source, read in windows shorter than the
    # documents, every token must be scored once and grouped under its source and its source's origin
    held_out = {"gutenberg-1-aaaaaaaa/a.txt": [7, 9, 11, 13, 15, 17, 19, END],
                "github-x-bbbbbbbb/main.py": [100, 200, END],
                "gutenberg-2-cccccccc/b.txt": [3, 4, 5, 6, END]}
    store = open_store((tmp_path / "store").as_uri())
    path = tmp_path / "held.parquet"
    digest = write_shard(path, [(np.asarray(tokens, dtype="<u4"), document.split("/")[0], document, "all", 0)
                                for document, tokens in held_out.items()])
    provenance = {"shards": [], "held_out_shards": [{"object": store.put_file(path), "token_sha256": digest}]}
    model = TokenLoss(block_size=4)
    context = Context(tmp_path, model, None, "cpu", provenance, store, kind_by="origin")
    monkeypatch.chdir(ROOT)

    data = held_out_loss().run(context).data

    def expected(documents):
        tokens = [token for document in documents for token in held_out[document]]
        return {"documents": len(documents), "tokens": len(tokens), "loss": sum(tokens) / len(tokens)}
    gutenberg = ["gutenberg-1-aaaaaaaa/a.txt", "gutenberg-2-cccccccc/b.txt"]
    assert data["by_kind"] == {"gutenberg": expected(gutenberg), "github": expected(["github-x-bbbbbbbb/main.py"])}
    assert data["by_source"]["gutenberg-1-aaaaaaaa"] == expected(["gutenberg-1-aaaaaaaa/a.txt"])
    assert data["tokens"] == 16 and data["loss"] == pytest.approx(expected(list(held_out))["loss"])
    # the stream starts after one end-of-document token, and each window starts at the last token of the one before
    stream = [END] + [token for tokens in held_out.values() for token in tokens]
    assert [token for window in model.windows for token in window[1:]] == stream[1:]
    assert all(len(window) <= 5 for window in model.windows)


def test_without_a_field_only_the_whole_held_out_loss_and_each_source_s_are_given(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    path = tmp_path / "held.parquet"
    digest = write_shard(path, [(np.asarray([5, 6, END], dtype="<u4"), "gutenberg-1-aaaaaaaa",
                                 "gutenberg-1-aaaaaaaa/a.txt", "all", 0)])
    provenance = {"shards": [], "held_out_shards": [{"object": store.put_file(path), "token_sha256": digest}]}
    result = held_out_loss().run(Context(tmp_path, TokenLoss(8), None, "cpu", provenance, store))
    assert result.data["by_kind"] is None and result.data["kind_by"] is None
    assert result.data["by_source"] == {"gutenberg-1-aaaaaaaa": {"documents": 1, "tokens": 3,
                                                                 "loss": (5 + 6 + END) / 3}}
    assert "--kind-by" in result.text


def test_the_worker_s_losses_are_the_model_s_next_token_losses_read_in_chunks(trained, monkeypatch):
    # the losses come through the generation cache a chunk at a time; they must be what one pass over the whole
    # window gives
    model = load_model(trained.folder)
    monkeypatch.setattr(evaluation, "LOSS_CHUNK", 5)
    window = [END, *range(20, 37)]
    [found] = Served(model).losses([window])
    with torch.no_grad():
        scores = model(torch.tensor([window[:-1]]))[0].float()
    whole = torch.nn.functional.cross_entropy(scores, torch.tensor(window[1:]), reduction="none").numpy()
    assert found.shape == (len(window) - 1,)
    np.testing.assert_allclose(found, whole, rtol=1e-4, atol=1e-4)


def test_the_test_groups_a_trained_model_s_held_out_loss_by_its_documents_tags(trained):
    # the source of the trained fixture tags every file "en": the one kind holds every held-out token
    from wpt.eval import open_context

    context = open_context(trained.folder, trained.store, kind_by="tag")
    context.model = Served(load_model(trained.folder))
    data = held_out_loss().run(context).data
    tokens = sum(len(document.tokens) for document in context.documents("held_out"))
    assert data["tokens"] == tokens > 0
    assert data["by_kind"] == {"en": {"documents": data["documents"], "tokens": tokens,
                                      "loss": pytest.approx(data["loss"])}}
