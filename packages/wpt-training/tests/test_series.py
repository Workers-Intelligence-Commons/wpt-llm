"""A size of a model series against the model it trains (docs/model.md,
"Model series"): the series counts the parameters the model has, and a
size reads exactly the start of the series' one dataset."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from wpt_dataset import series
from wpt_dataset.repo import Repo
from wpt_model import PARTS, ModelConfig, WPT
from wpt_model.config import global_layers
from wpt_model.engram import Engram
from wpt_model.experts import Experts
from wpt_store import LocalStore
from wpt_store.shards import write_shard
from wpt_telemetry import Telemetry
from wpt_training import Data, Strategy, load_step

ROOT = Path(__file__).resolve().parents[3]


def test_the_series_counts_the_parameters_of_the_model_it_derives():
    # the count without the vocabulary tables decides a size's compute and tokens: it must be
    # the model's own, so it is checked against the model wpt_model builds at every size
    repo = Repo(ROOT)
    ladder = repo.recipe("series", "eldest-titmouse")
    for size in series.sizes(ladder):
        model = series.shape(ladder, size)
        config = ModelConfig(vocab_size=32768, block_size=32768, **model)
        with torch.device("meta"):
            built = WPT(config)
        parts = built.parts()
        tables = sum(p.numel() for part in ("embedding", "output-head") for p in parts[part])
        everything = sum(p.numel() for p in built.parameters())
        assert series.parameters(model, 32768) == (everything - tables, everything), size
        assert series.global_layers(model["n_layer"]) == global_layers(model["n_layer"])


def test_the_series_counts_the_parameters_one_token_uses_as_the_model_it_derives_has_them():
    # the model's own count without the vocabulary tables, less the routed experts a token does not reach
    # (n_experts - top_k of each layer's stacked experts) and less Engram's lookup tables
    repo = Repo(ROOT)
    ladder = repo.recipe("series", "eldest-titmouse")
    for size in series.sizes(ladder):
        model = series.shape(ladder, size)
        with torch.device("meta"):
            built = WPT(ModelConfig(vocab_size=32768, block_size=32768, **model))
        unused = tables = 0
        for module in built.modules():
            if isinstance(module, Experts):
                routed = sum(p.numel() for p in (module.w_gate, module.w_up, module.w_down))
                unused += routed - routed // module.n * module.k
            if isinstance(module, Engram):
                tables += sum(p.numel() for p in module.tables.parameters())
        assert tables > 0 and unused > 0
        assert series.active_parameters(model) == series.parameters(model, 0)[0] - unused - tables, size


@pytest.fixture
def dataset(tmp_path):
    """A store holding 4 shards of 100 training tokens each, token i at
    place i of the training tokens, each shard its own source, and a
    held-out shard; and its record."""
    store = LocalStore(tmp_path / "store").at("test-session")
    shards = []
    for i in range(4):
        path = tmp_path / f"{i}.parquet"
        digest = write_shard(path, [(list(range(100 * i, 100 * i + 100)), f"s{i}", f"s{i}/d", "p", 0)])
        shards.append({**asdict(store.put_named(f"test-session/datasets/d/{digest}.parquet", path)), "token_sha256": digest, "rows": 1, "tokens": 100})
    path = tmp_path / "held.parquet"
    digest = write_shard(path, [(list(range(1000, 1100)), "s0", "s0/h", "held-out", 0)])
    held = [{**asdict(store.put_named(f"test-session/datasets/d/{digest}.parquet", path)), "token_sha256": digest, "rows": 1, "tokens": 100}]
    record = {"shards": shards, "held_out_shards": held, "total_tokens": 400,
              "sources": {f"s{i}": {"name": f"s{i}"} for i in range(4)}}
    return store, record


def windows(store, record, prefix, stream_shards=None, updates=40):
    """Every window the read-text step draws in `updates` updates of 8 windows of 16 tokens."""
    step = load_step({"step": "read-text", "settings": {
        "stages": [{"block_size": 16, "steps": updates, "batch_size": 8}], "prefix_tokens": prefix}}, PARTS)
    step.check_dataset(record, store)
    data = Data(step.name, store, record, None, Strategy(torch.device("cpu")), Telemetry.off(), lambda _: None, 1337,
                1, stream_shards=stream_shards)
    environment = step.start_environment(data)
    try:
        found = [x.numpy() for update in range(1, updates + 1) for x, _ in environment.batch(update, None, None)]
        return np.concatenate(found), environment.sources
    finally:
        environment.close()


@pytest.mark.parametrize("stream_shards", [None, 2])
def test_a_smaller_size_reads_exactly_the_start_of_the_dataset_built_for_a_larger_one(dataset, stream_shards):
    # a prefix of 250 tokens ends inside the third shard: every window is a run of the first
    # 250 training tokens (a stream walks them over and over: after token 249 comes token 0),
    # the windows reach its end, and the step learned from the sources of those three shards
    # alone; the same windows on every run
    store, record = dataset
    found, sources = windows(store, record, 250, stream_shards)
    following = np.roll(found, -1, axis=1)[:, :-1]
    assert ((following == found[:, :-1] + 1) | (stream_shards is not None) & (found[:, :-1] == 249)
            & (following == 0)).all()
    assert found.min() >= 0 and found.max() <= 249 and found.max() >= 240
    assert sorted(sources) == ["s0", "s1", "s2"]
    again, _ = windows(store, record, 250, stream_shards)
    assert (found == again).all()
    # without a prefix, the whole dataset
    everything, sources = windows(store, record, 0, stream_shards)
    assert everything.max() >= 250 and sorted(sources) == ["s0", "s1", "s2", "s3"]


def test_a_prefix_longer_than_the_dataset_is_refused_before_any_step_runs(dataset):
    store, record = dataset
    with pytest.raises(ValueError, match="reads the first 401 training tokens; the dataset has 400"):
        windows(store, record, 401)
