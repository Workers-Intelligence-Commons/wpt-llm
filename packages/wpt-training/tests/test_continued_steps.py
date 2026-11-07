"""A step that continues the one before it (docs/training.md, "Continued
steps"): a decay branch of a model series is a trunk at the peak rate
and a decay that goes on from its last state, and together they must
train as the one step whose schedule drops at the same updates."""

from dataclasses import asdict

import pytest
import torch

from wpt_dataset import series
from wpt_model import PARTS, WPT, ModelConfig
from wpt_store import LocalStore
from wpt_store.shards import write_shard
from wpt_telemetry import Telemetry
from wpt_training import (Data, FolderCheckpoint, Strategy, check_continuations, draw_numbers, load_state, load_step,
                          train)

TINY = dict(vocab_size=50, block_size=16, n_layer=4, global_attention=["CSA"], d_model=16, kda_heads=2, kda_head_dim=8,
            attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1, output_rank=8, index_heads=2,
            index_head_dim=128, index_top_k=2, n_experts=4, top_k=2, expert_width=8, latent_width=8, shared_width=16,
            engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31)
CPU = Strategy(torch.device("cpu"))


@pytest.fixture
def dataset(tmp_path):
    """A store with 2 training shards of 300 random tokens and a held-out shard; and its record."""
    store = LocalStore(tmp_path / "store").at("test-session")
    generator = torch.Generator().manual_seed(0)

    def shard(name, part):
        path = tmp_path / f"{name}.parquet"
        digest = write_shard(path, [(torch.randint(50, (300,), generator=generator).tolist(), "s", f"s/{name}",
                                     part, 0)])
        return {**asdict(store.put_named(f"test-session/datasets/d/{digest}.parquet", path)), "token_sha256": digest, "rows": 1, "tokens": 300}

    record = {"shards": [shard("a", "p"), shard("b", "p")], "held_out_shards": [shard("h", "held-out")],
              "total_tokens": 600, "sources": {"s": {"name": "s"}}}
    return store, record


def read_text(updates, drops=True, continues=False, **settings):
    """A read-text step as a series' recipe holds it: 2 windows of 16 tokens per update, a warm-up of 2;
    without drops, a trunk: no drop, no switch of the multi-token prediction's weight."""
    optimizer = {"lr": 0.01, "warmup": 2}
    if not drops:
        optimizer.update(first_drop=1.0, second_drop=1.0)
        settings["mtp_switch"] = 1.0
    entry = {"step": "read-text", "optimizer": {"name": "by-part", "settings": optimizer},
             "settings": {"stages": [{"block_size": 16, "steps": updates, "batch_size": 2}], "prefix_tokens": 500,
                          **settings}}
    if continues:
        entry["continues"] = True
    return load_step(entry, PARTS)


def run(steps, dataset, folder):
    """The steps trained in order on one model as the gpu worker trains them; each step's last state."""
    store, record = dataset
    torch.manual_seed(1337)
    model = WPT(ModelConfig(**TINY))
    states, previous = [], None
    for number, (step, draws) in enumerate(zip(steps, draw_numbers(steps)), 1):
        data = Data(step.name, store, record, None, CPU, Telemetry.off(), lambda _: None, 1337, draws)
        environment = step.start_environment(data)
        checkpoint = FolderCheckpoint(folder / str(number))
        try:
            train(model, CPU, step, step.start_method(), environment, 1337, draws, checkpoint=checkpoint,
                  identity=number, log=lambda *_: None, save_last=True, start=previous if step.continues else None)
        finally:
            environment.close()
        states.append(load_state(checkpoint.path))
        previous = checkpoint
    return states


def test_a_decay_that_continues_a_trunk_trains_as_the_one_step_of_its_schedule(dataset, tmp_path):
    # 10 updates whose rate drops after update 8 (80%) and 9 (90%), where the multi-token prediction's weight
    # switches too; the trunk runs updates 1 to 8 at the peak rate and the early weight, and the decay goes on
    # from its last state at update 9 with the 10 updates' schedule, every late change in it
    [whole] = run([read_text(10, mtp_switch=0.8)], dataset, tmp_path / "whole")
    trunk, decay = run([read_text(8, drops=False), read_text(10, continues=True)], dataset, tmp_path / "branch")
    assert trunk["step"] == 8 and decay["step"] == 10
    assert whole["model"].keys() == decay["model"].keys()
    for name, value in whole["model"].items():
        assert torch.equal(decay["model"][name], value), name


def test_a_step_continues_only_one_like_it_with_updates_left():
    check_continuations([read_text(8, drops=False), read_text(10, continues=True)])
    with pytest.raises(ValueError, match="there is none"):
        check_continuations([read_text(10, continues=True)])
    with pytest.raises(ValueError, match="which makes 10 updates: it must have more than that, and it has 8"):
        check_continuations([read_text(10, drops=False), read_text(8, continues=True)])
    other = load_step({"step": "answer-as-assistant", "continues": True}, PARTS)
    with pytest.raises(ValueError, match="same step, method, environment, optimizer and locked parts"):
        check_continuations([read_text(8), other])


def test_a_series_branches_where_by_part_s_first_drop_is():
    from wpt_training.plugins import _load
    assert series.FIRST_DROP == _load("optimizers", "by-part").module.SETTINGS["first_drop"]
