"""A window trained in parts (docs/training.md, "Long windows") gives the
loss and gradients of the whole window at once: in segments on one
process, and split across processes."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_model.experts import BINS, WINDOW, Experts
from wpt_training import Strategy, TextMethod, parts
from wpt_training.launch import launch

# every layer kind: KDA (layers 1-3, 5-7), CSA (4) with an indexer choosing 8 of its entries, HCA (8), the window of
# each attention, Engram (layers 2 and 5), the experts (hash-routed, then routed and balanced), mHC and the MTP block
MODEL = dict(vocab_size=50, block_size=448, n_layer=8, global_attention=["CSA", "HCA"], d_model=16, kda_heads=2,
             kda_head_dim=8, attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1, output_rank=8,
             index_heads=4, index_head_dim=128, index_top_k=8, n_experts=4, top_k=2, expert_width=8, latent_width=8,
             shared_width=16, engram_layers=[2, 5], engram_order=3, engram_heads=2, engram_dim=4,
             engram_table_size=31)


# ROUNDING (2^-14, about 6.1e-5) and ARITHMETIC are the `routing` fixture's (conftest.py; docs/training.md, "Comparing
# routing in tests"). Whole and in segments, on x86 CPUs, the scores here differ by up to 6e-8 at the first routed layer
# and 7.6e-6 at the last.


def model_and_window(T):
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**MODEL, "block_size": T})).train()
    generator = torch.Generator().manual_seed(1)
    tokens = torch.randint(50, (2, T + 1), generator=generator)
    return model, tokens[:, :-1], tokens[:, 1:]


@contextlib.contextmanager
def counted_scores():
    """While open, keeps what each routed layer counts for balancing: per
    Experts module, one (scores, cutoffs, bias) per counted batch, in the
    order counted (a window's positions in order, whole or in parts)."""
    kept, count = {}, Experts._count

    def keep(experts, s, cutoff):
        kept.setdefault(experts, []).append((s.clone(), cutoff.clone(), experts.bias.clone()))
        count(experts, s, cutoff)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Experts, "_count", keep)
        yield kept


def results(model, loss, indexer, kept=None, rows=2):
    """The pass's losses, every gradient, and, with kept (counted_scores),
    each routed layer's scores, cutoffs, bias, k and balancing counts, the
    scores in the order of the window's tokens (rows of the batch, then
    positions)."""
    routed = []
    for experts in [layer.experts for layer in model.layers if not layer.experts.hashed] if kept is not None else []:
        batches = kept[experts]
        s = torch.cat([s.view(rows, -1, s.shape[-1]) for s, _, _ in batches], 1).reshape(-1, experts.n)
        cutoff = torch.cat([c.view(rows, -1) for _, c, _ in batches], 1).reshape(-1)
        routed.append((s, cutoff, batches[0][2], experts.k, experts.histogram.clone()))
    return float(loss), float(indexer), [p.grad.clone() for p in model.parameters()], routed


def whole_window(T):
    model, x, y = model_and_window(T)
    with counted_scores() as kept:
        loss, indexer = TextMethod().loss(model, x, y)
        loss.backward()
    return results(model, loss.detach(), indexer, kept)


def balancing_counts(bins, n):
    """The counts of each expert's bins, as Experts._count adds them."""
    return torch.bincount((bins + torch.arange(n) * BINS).reshape(-1), minlength=n * BINS).view(n, BINS)


def balancing_bins(s, cutoff, bias):
    """Each (token, expert)'s balancing bin, as Experts._count computes it,
    and how far its offset s_j - alpha lies from the nearest edge between
    two bins, in units of score."""
    place = (s - cutoff[:, None] + bias + WINDOW) * (BINS / (2 * WINDOW))
    edge = place.round().clamp(1, BINS - 1)
    return place.floor().long().clamp(0, BINS - 1), (place - edge).abs() * (2 * WINDOW / BINS)


def assert_routed_alike(got, expected, routing):
    """Two passes over the same tokens give each routed layer the same
    scores within ROUNDING (routing.ROUNDING), and the same routing and
    balancing counts for every token whose choice is clear. That bound on
    the scores is what limits the differences to near ties: a token whose choice is decided by two values
    no farther apart than the passes moved them (the token's own score
    differences, plus ARITHMETIC) may go either way. The rerouting and bin
    checks below document this, by testing it for the scores at hand; they
    do not enforce it. Its experts may differ only when its k-th and
    (k+1)-th biased scores are near a tie, and its count for an expert may
    move only when that offset s_j - alpha is near the edge between two
    bins, and then to the next bin. Each pass's counts are exactly the bins
    of the scores it counted (docs/training.md, "Comparing routing in
    tests")."""
    ROUNDING, ARITHMETIC = routing.ROUNDING, routing.ARITHMETIC
    assert len(got) == len(expected)
    for (s, cutoff, bias, k, counts), (s_w, cutoff_w, bias_w, k_w, counts_w) in zip(got, expected):
        assert s.shape == s_w.shape and k == k_w and torch.equal(bias, bias_w)
        moved_s, moved_cutoff = (s - s_w).abs(), (cutoff - cutoff_w).abs()
        assert moved_s.max() <= ROUNDING and moved_cutoff.max() <= ROUNDING  # the same scores, within rounding
        n = s.shape[1]
        top, top_w = torch.topk(s + bias, k + 1), torch.topk(s_w + bias_w, k + 1)
        near_tie = top_w.values[:, k - 1] - top_w.values[:, k] <= 2 * moved_s.max(-1).values + ARITHMETIC
        rerouted = (top.indices[:, :k].sort(-1).values != top_w.indices[:, :k].sort(-1).values).any(-1)
        assert not (rerouted & ~near_tie).any(), "a token whose routing is clear went to other experts"
        (bins, _), (bins_w, from_edge) = balancing_bins(s, cutoff, bias), balancing_bins(s_w, cutoff_w, bias_w)
        for b, c in ((bins, counts), (bins_w, counts_w)):  # the counts are those tokens' bins, nothing else
            assert torch.equal(balancing_counts(b, n), c)
        near_edge = from_edge <= moved_s + moved_cutoff[:, None] + ARITHMETIC
        assert not ((bins != bins_w) & ~near_edge).any(), "a token far from a bin's edge changed bins"
        assert (bins - bins_w).abs().max() <= 1


def assert_same(got, expected, routing, tolerance=2e-5):
    # tolerance: of the largest gradient; whole and in segments differ by up to 8.3e-7
    # on GitHub's AMD runners and 8.2e-6 on its Intel ones (docs/training.md)
    (loss, indexer, grads, routed), (loss_w, indexer_w, grads_w, routed_w) = got, expected
    assert abs(loss - loss_w) < 1e-5 * abs(loss_w) and abs(indexer - indexer_w) < 1e-4 * abs(indexer_w)
    assert indexer_w > 0
    scale = max(g.abs().max().item() for g in grads_w)  # FP32 rounding, measured against the largest gradient
    for g, g_w in zip(grads, grads_w, strict=True):
        assert torch.allclose(g, g_w, rtol=1e-3, atol=tolerance * scale)
    assert_routed_alike(routed, routed_w, routing)


def test_routing_and_balancing_counts_may_differ_only_at_near_ties(routing):
    # two passes whose router scores differ by one unit of FP32 rounding, as
    # another processor's arithmetic gives: token 0's 2nd and 3rd scores are
    # one unit apart, the other way round in each pass, so its experts
    # differ; token 1's offset for expert 0 lies exactly on the edge between
    # two bins in one pass and one unit below it in the other, so its count
    # moves to the bin below; every other score moves by one unit only (each
    # token's (k+1)-th expert, whose offset is 0, sits on an edge in both).
    # The comparison accepts both, and refuses a count moved for a clear token.
    experts = Experts(ModelConfig(**MODEL), hashed=False).train()
    k, bias = experts.k, experts.bias
    generator = torch.Generator().manual_seed(0)
    s_w = 0.05 + 0.9 * torch.rand(64, experts.n, generator=generator)
    half, below = torch.tensor(0.5), lambda t: torch.nextafter(torch.as_tensor(t), torch.tensor(0.0))
    s_w[0] = torch.stack([torch.tensor(0.75), half, below(0.5), torch.tensor(0.25)])
    s_w[1] = torch.tensor([0.875, 0.625, 0.5, 0.25])  # expert 0: s - alpha = 0.375, the edge of bin 3584
    up = torch.rand(s_w.shape, generator=generator) < 0.5
    s = torch.nextafter(s_w, torch.where(up, 1.0, 0.0))
    s[0] = torch.stack([torch.tensor(0.75), below(0.5), half, torch.tensor(0.25)])
    s[1] = s_w[1]
    s[1, 0] = below(0.875)

    def counted(scores):
        cutoff = torch.topk(scores + bias, k + 1).values[:, k]
        experts.histogram = None
        experts._count(scores, cutoff)
        return scores, cutoff, bias.clone(), k, experts.histogram

    got, expected = counted(s), counted(s_w)
    assert not torch.equal(got[4], expected[4])  # the counts differ
    assert set(torch.topk(s[0], k).indices.tolist()) != set(torch.topk(s_w[0], k).indices.tolist())  # and the routing
    assert_routed_alike([got], [expected], routing)

    counts = got[4].clone()  # token 2's offset farthest from a bin's edge: its count moved to the next bin
    bins, from_edge = balancing_bins(*got[:3])
    j = int(from_edge[2].argmax())
    assert from_edge[2, j] > routing.ROUNDING and bins[2, j] < BINS - 1
    counts[j, bins[2, j]] -= 1
    counts[j, bins[2, j] + 1] += 1
    with pytest.raises(AssertionError):
        assert_routed_alike([(*got[:4], counts)], [expected], routing)


RECOMPUTE = pytest.mark.parametrize("recompute", [False, True], ids=["kept", "recomputed"])  # --recompute


@RECOMPUTE
def test_a_window_in_segments_trains_as_the_whole_window(recompute, routing):
    # 448 tokens in segments of 128, 128, 128 and 64: KDA memories,
    # convolution tails, windows, Engram's earlier ids and the merged entries
    # cross three boundaries, and every gradient flows back across them,
    # with each layer's activations kept or recomputed in the backward pass
    expected = whole_window(448)
    model, x, y = model_and_window(448)
    model.recompute = recompute
    with counted_scores() as kept:
        loss, indexer = parts.learn(TextMethod(), model, model, x, y, 1.0, Strategy(torch.device("cpu")), 128,
                                    lambda last: contextlib.nullcontext())
    assert_same(results(model, loss, indexer, kept), expected, routing)

    model.eval()  # and the validation loss in segments is the whole window's
    with torch.no_grad():
        assert torch.allclose(parts.validation_loss(TextMethod(), model, model, x, y, Strategy(torch.device("cpu")), 128),
                              TextMethod().validation_loss(model, x, y), rtol=1e-5)


WORKER = '''
import json, os, sys, torch
sys.path.insert(0, sys.argv[2])
from test_parts import counted_scores, model_and_window, results
from wpt_training import TextMethod, load_strategy, parts
_, strategy = load_strategy("split")
model, x, y = model_and_window(int(sys.argv[3]))
segment, model.recompute = int(sys.argv[4]), sys.argv[5] == "1"
with counted_scores() as kept:
    loss, indexer = parts.learn(TextMethod(), model, strategy.wrap(model), x, y, 1.0, strategy, segment,
                                lambda last: strategy.no_sync(model))
strategy.synchronize(model)
loss, indexer = strategy.mean(float(loss)) * strategy.world_size, strategy.mean(float(indexer)) * strategy.world_size
found = results(model, loss, indexer, kept)
torch.save(found[3], f"{sys.argv[1]}.{strategy.rank}")  # this process's slice: its scores and balancing counts
model.after_update(strategy.reduce)  # the balancing counts, added over the processes
model.eval()
with torch.no_grad():
    validation = strategy.mean(float(parts.validation_loss(TextMethod(), model, model, x, y, strategy, segment)))
if strategy.rank == 0:
    torch.save((found[:3], validation * strategy.world_size), sys.argv[1])
strategy.close()
'''


def positions_of(model, rank, processes, T, segment):
    """The window positions process `rank` trains, in the order it counts them."""
    mine = SimpleNamespace(splits_windows=True, world_size=processes, rank=rank)
    return torch.cat([torch.arange(a, end) for a, end, _ in parts._pieces(model, T, mine, segment)])


def slice_of(model, routed, rank, processes, T, segment):
    """The whole window's routed layers (results) restricted to the tokens
    process `rank` trains, with the balancing counts of those tokens."""
    positions = positions_of(model, rank, processes, T, segment)
    sliced = []
    for s, cutoff, bias, k, _ in routed:
        s, cutoff = s.view(2, T, -1)[:, positions].reshape(-1, s.shape[-1]), cutoff.view(2, T)[:, positions].reshape(-1)
        sliced.append((s, cutoff, bias, k, balancing_counts(balancing_bins(s, cutoff, bias)[0], s.shape[-1])))
    return sliced


def assert_counts_add_up(found, model, routed, processes, T, segment, routing):
    """The processes' balancing counts, added, are the whole window's
    counts, up to near ties: the processes' scores put back in the order of
    the window, with the sum of their counts, are compared with the whole
    window's as two passes over the same tokens (assert_routed_alike)."""
    added = []
    for layer, (_, _, bias, k, _) in enumerate(routed):
        s = torch.full_like(routed[layer][0], torch.nan).view(2, T, -1)
        cutoff = torch.full_like(routed[layer][1], torch.nan).view(2, T)
        for rank in range(processes):
            positions = positions_of(model, rank, processes, T, segment)
            s[:, positions] = found[rank][layer][0].view(2, len(positions), -1)
            cutoff[:, positions] = found[rank][layer][1].view(2, len(positions))
        assert not s.isnan().any() and not cutoff.isnan().any()  # every token trained by one process
        added.append((s.view(-1, s.shape[-1]), cutoff.view(-1), bias, k, sum(part[layer][4] for part in found)))
    assert_routed_alike(added, routed, routing)


def split_runs(tmp_path, monkeypatch, recompute, runs, routing):
    """Each run (processes, window length, segment) on CPU processes (gloo)
    gives the whole window's losses and gradients, the gradients summed over
    the processes, its validation loss, and, for each process, the scores
    and balancing counts of its slice of the window."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    script = tmp_path / "worker.py"
    script.write_text(WORKER, encoding="utf-8", newline="\n")
    for processes, T, segment in runs:
        model, x, y = model_and_window(T)
        with counted_scores() as kept:
            loss, indexer = TextMethod().loss(model, x, y)
            loss.backward()
        loss, indexer, grads, routed = results(model, loss.detach(), indexer, kept)
        model.after_update(lambda tensor, op: tensor)
        model.eval()
        with torch.no_grad():
            validation = float(TextMethod().validation_loss(model, x, y))
        out = tmp_path / f"{processes}-{T}-{segment}.pt"
        assert launch(processes, [str(script), str(out), str(Path(__file__).parent), str(T), str(segment),
                                  "1" if recompute else "0"]) == 0
        (got_loss, got_indexer, got_grads), got_validation = torch.load(out)
        # KDA's slices compose their memories' transforms in FP32 (the same algebra in float64 agrees to 1e-15):
        # its gradients differ from the whole window's by up to 6e-5 of the largest gradient
        assert_same((got_loss, got_indexer, got_grads, []), (loss, indexer, grads, []), routing, 2e-4)
        found = [torch.load(f"{out}.{rank}") for rank in range(processes)]
        for rank in range(processes):  # each process's counts are those of its tokens, up to near ties
            assert_routed_alike(found[rank], slice_of(model, routed, rank, processes, T, segment), routing)
        assert_counts_add_up(found, model, routed, processes, T, segment, routing)  # and together the whole window's
        assert abs(got_validation - validation) < 1e-5 * abs(validation)


@RECOMPUTE
def test_a_window_split_across_processes_trains_as_the_whole_window(tmp_path, monkeypatch, recompute, routing):
    # DeepSeek-V4's contextual parallelism on 2 and 4 CPU processes: 512
    # tokens in slices of 256 or 128, the KDA memory combined from every
    # slice's transform, the merged entries gathered
    split_runs(tmp_path, monkeypatch, recompute, [(2, 512, 0), (4, 512, 0)], routing)


@RECOMPUTE
def test_a_window_split_across_processes_in_segments_trains_as_the_whole_window(tmp_path, monkeypatch, recompute,
                                                                                routing):
    # 768 tokens on 2 processes in segments of 128: three parts of 256
    # tokens, each split in two, so every process trains its slice as three
    # segments and the parts hand on what they pass on, gradients included
    split_runs(tmp_path, monkeypatch, recompute, [(2, 768, 128)], routing)
