"""A window trained in parts (docs/training.md, "Long windows") gives the
loss and gradients of the whole window at once: in segments on one
process, and split across processes."""

import contextlib
from pathlib import Path

import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_training import Method, Strategy, parts
from wpt_training.launch import launch

# every layer kind: KDA (layers 1-3, 5-7), CSA (4) with an indexer choosing 8 of its entries, HCA (8), the window of
# each attention, Engram (layers 2 and 5), the experts (hash-routed, then routed and balanced), mHC and the MTP block
MODEL = dict(vocab_size=50, block_size=448, n_layer=8, global_attention=["CSA", "HCA"], d_model=16, kda_heads=2,
             kda_head_dim=8, attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1, output_rank=8,
             index_heads=4, index_head_dim=128, index_top_k=8, n_experts=4, top_k=2, expert_width=8, latent_width=8,
             shared_width=16, engram_layers=[2, 5], engram_order=3, engram_heads=2, engram_dim=4,
             engram_table_size=31)


def model_and_window(T):
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**MODEL, "block_size": T})).train()
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randint(50, (2, T + 1), generator=generator)
    return model, tokens[:, :-1], tokens[:, 1:]


def results(model, loss, indexer):
    """The pass's losses, every gradient, and the balancing counts it recorded."""
    counts = [layer.experts.histogram.clone() for layer in model.layers if not layer.experts.hashed]
    return float(loss), float(indexer), [p.grad.clone() for p in model.parameters()], counts


def whole_window(T):
    model, x, y = model_and_window(T)
    loss, indexer = Method().loss(model, x, y)
    loss.backward()
    return results(model, loss.detach(), indexer)


def assert_same(got, expected, tolerance=1e-6):
    (loss, indexer, grads, counts), (loss_w, indexer_w, grads_w, counts_w) = got, expected
    assert abs(loss - loss_w) < 1e-5 * abs(loss_w) and abs(indexer - indexer_w) < 1e-4 * abs(indexer_w)
    assert indexer_w > 0
    scale = max(g.abs().max().item() for g in grads_w)  # FP32 rounding, measured against the largest gradient
    for g, g_w in zip(grads, grads_w, strict=True):
        assert torch.allclose(g, g_w, rtol=1e-3, atol=tolerance * scale)
    assert len(counts) == len(counts_w) and all(torch.equal(c, c_w) for c, c_w in zip(counts, counts_w))


RECOMPUTE = pytest.mark.parametrize("recompute", [False, True], ids=["kept", "recomputed"])  # --recompute


@RECOMPUTE
def test_a_window_in_segments_trains_as_the_whole_window(recompute):
    # 448 tokens in segments of 128, 128, 128 and 64: KDA memories,
    # convolution tails, windows, Engram's earlier ids and the merged entries
    # cross three boundaries, and every gradient flows back across them,
    # with each layer's activations kept or recomputed in the backward pass
    expected = whole_window(448)
    model, x, y = model_and_window(448)
    model.recompute = recompute
    loss, indexer = parts.learn(Method(), model, model, x, y, 1.0, Strategy(torch.device("cpu")), 128,
                                lambda last: contextlib.nullcontext())
    assert_same(results(model, loss, indexer), expected)

    model.eval()  # and the validation loss in segments is the whole window's
    with torch.no_grad():
        assert torch.allclose(parts.validation_loss(Method(), model, model, x, y, Strategy(torch.device("cpu")), 128),
                              Method().validation_loss(model, x, y), rtol=1e-5)


WORKER = '''
import json, os, sys, torch
sys.path.insert(0, sys.argv[2])
from test_parts import model_and_window, results
from wpt_training import Method, load_strategy, parts
_, strategy = load_strategy("split")
model, x, y = model_and_window(int(sys.argv[3]))
segment, model.recompute = int(sys.argv[4]), sys.argv[5] == "1"
loss, indexer = parts.learn(Method(), model, strategy.wrap(model), x, y, 1.0, strategy, segment,
                            lambda last: strategy.no_sync(model))
strategy.synchronize(model)
loss, indexer = strategy.mean(float(loss)) * strategy.world_size, strategy.mean(float(indexer)) * strategy.world_size
found = results(model, loss, indexer)
model.after_update(strategy.reduce)  # the balancing counts, added over the processes
model.eval()
with torch.no_grad():
    validation = strategy.mean(float(parts.validation_loss(Method(), model, model, x, y, strategy, segment)))
if strategy.rank == 0:
    torch.save((found[:3], [layer.experts.bias.clone() for layer in model.layers if not layer.experts.hashed],
                validation * strategy.world_size), sys.argv[1])
strategy.close()
'''


def split_runs(tmp_path, monkeypatch, recompute, runs):
    """Each run (processes, window length, segment) on CPU processes (gloo)
    gives the whole window's losses and gradients, the gradients summed over
    the processes, its experts' balancing counts and its validation loss."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8", newline="\n")
    for processes, T, segment in runs:
        model, x, y = model_and_window(T)
        loss, indexer = Method().loss(model, x, y)
        loss.backward()
        loss, indexer, grads, _ = results(model, loss.detach(), indexer)
        model.after_update(lambda tensor, op: tensor)
        biases = [layer.experts.bias for layer in model.layers if not layer.experts.hashed]
        model.eval()
        with torch.no_grad():
            validation = float(Method().validation_loss(model, x, y))
        out = tmp_path / f"{processes}-{T}-{segment}.pt"
        assert launch(processes, [str(script), str(out), str(Path(__file__).parent), str(T), str(segment),
                                  "1" if recompute else "0"]) == 0
        (got_loss, got_indexer, got_grads), got_biases, got_validation = torch.load(out)
        # KDA's slices compose their memories' transforms in FP32 (the same algebra in float64 agrees to 1e-15):
        # its gradients differ from the whole window's by up to 6e-5 of the largest gradient
        assert_same((got_loss, got_indexer, got_grads, []), (loss, indexer, grads, []), 2e-4)
        assert all(torch.allclose(b, b_w, atol=1e-6) for b, b_w in zip(got_biases, biases, strict=True))
        assert abs(got_validation - validation) < 1e-5 * abs(validation)


@RECOMPUTE
def test_a_window_split_across_processes_trains_as_the_whole_window(tmp_path, monkeypatch, recompute):
    # DeepSeek-V4's contextual parallelism on 2 and 4 CPU processes: 512
    # tokens in slices of 256 or 128, the KDA memory combined from every
    # slice's transform, the merged entries gathered
    split_runs(tmp_path, monkeypatch, recompute, [(2, 512, 0), (4, 512, 0)])


@RECOMPUTE
def test_a_window_split_across_processes_in_segments_trains_as_the_whole_window(tmp_path, monkeypatch, recompute):
    # 768 tokens on 2 processes in segments of 128: three parts of 256
    # tokens, each split in two, so every process trains its slice as three
    # segments and the parts hand on what they pass on, gradients included
    split_runs(tmp_path, monkeypatch, recompute, [(2, 768, 128)])
