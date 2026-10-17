import importlib
import math
import sys
import types

import pytest
import torch

import wpt_model
from wpt_device import select_device
from wpt_model import WPT, ModelConfig
from wpt_model import sparse
from wpt_model.attention import CSA_RATIO, HCA_RATIO, WINDOW, Attention, _window, indexed
from wpt_model.formats import fp4, fp8, fp8_parts, hadamard
from wpt_model.sparse import KernelUnavailable, _SparseAttention, index_alignment, kernels, top_entries

CONFIG = ModelConfig(vocab_size=50, block_size=300, n_layer=8, global_attention=["CSA", "HCA"], d_model=32, kda_heads=2, kda_head_dim=8,
                     attention_heads=4, attention_head_dim=128, query_rank=16, output_groups=2, output_rank=16,
                     index_heads=8, index_head_dim=128, index_top_k=8, n_experts=4, top_k=2, expert_width=8,
                     latent_width=8, shared_width=16, engram_layers=[2], engram_order=3, engram_heads=2,
                     engram_dim=4, engram_table_size=31)


# -- the formulas of DeepSeek-V4's paper (section 2.3), one token at a time --

def v4_frequencies(merged):
    """V4-Flash's precompute_freqs_cis with its config.json: YaRN (factor 16
    over 65536 tokens, beta_fast 32, beta_slow 1) on base 160000 in layers
    with merged entries; base 10000 without YaRN in the window alone."""
    dim, base = 64, 160000 if merged else 10000
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if merged:
        def correction(rotations):
            return dim * math.log(65536 / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(math.floor(correction(32)), 0), min(math.ceil(correction(1)), dim - 1)
        smooth = 1 - torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        freqs = freqs / 16 * (1 - smooth) + freqs * smooth
    return freqs


def rope(x, position, merged=True):
    """RoPE on the last 64 dimensions, as complex numbers (V4's inference code)."""
    frequencies = v4_frequencies(merged)
    turned = torch.view_as_complex(x[..., -64:].reshape(*x.shape[:-1], 32, 2).contiguous())
    turned = torch.view_as_real(turned * torch.polar(torch.ones(32), position * frequencies)).flatten(-2)
    return torch.cat([x[..., :-64], turned], -1)


def rms(x, weight=None):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return x if weight is None else x * weight


def as_stored(entry):
    """V4's key-value entries: FP8 (blocks of 64) except the 64 rotary dimensions."""
    return torch.cat([fp8(entry[..., :-64]), entry[..., -64:]], -1)


def as_indexed(x):
    """V4's indexer queries and keys: a Hadamard rotation, then FP4 (blocks of
    32), then FP8 with one scale per vector (V4's lossless FP4-to-FP8 step)."""
    return fp8(fp4(hadamard(x)), block=x.shape[-1])


def merged(compressor, x, i):
    """C^Comp_i of one text x (T, D), before its format: eq. 11-12 with
    overlap, 22-23 without."""
    m, c = compressor.ratio, compressor.width
    W, Z, B = compressor.value.weight, compressor.score.weight, compressor.position
    values, scores = [], []
    if compressor.parts == 2:
        for j in range(m * i, m * (i + 1)):  # C^a over block i
            values.append(W[c:] @ x[j])
            scores.append(Z[c:] @ x[j] + B[j - m * i, c:])
        for j in range(m * (i - 1), m * i) if i > 0 else []:  # C^b over block i - 1
            values.append(W[:c] @ x[j])
            scores.append(Z[:c] @ x[j] + B[j - m * (i - 1), :c])
    else:
        for j in range(m * i, m * (i + 1)):
            values.append(W @ x[j])
            scores.append(Z @ x[j] + B[j - m * i])
    entry = (torch.stack(scores).softmax(0) * torch.stack(values)).sum(0)
    return rope(rms(entry, compressor.norm.weight), m * i)


def attention_by_formula(layer, x):
    """The layer's output for one text x (T, D), query by query."""
    T, H, c = x.shape[0], layer.heads, layer.head_dim
    entries = [as_stored(merged(layer.compressor, x, i)) for i in range(T // layer.ratio)]
    if layer.indexer is not None:
        keys = [as_indexed(merged(layer.indexer.keys, x, i)) for i in range(T // layer.ratio)]
    out = []
    for t in range(T):
        latent = rms(layer.q_down.weight @ x[t], layer.q_norm.weight)
        q = rope(rms((layer.q_up.weight @ latent).view(H, c)), t)  # each head RMS-normalized
        window = [as_stored(rope(rms(layer.kv.weight @ x[j], layer.kv_norm.weight), j))
                  for j in range(max(0, t - WINDOW + 1), t + 1)]
        visible = list(range((t + 1) // layer.ratio))  # blocks that end at or before t
        if layer.indexer is not None and visible:  # eq. 13-17
            n, d = layer.indexer.heads, layer.indexer.head_dim
            q_index = as_indexed(rope((layer.indexer.q.weight @ latent).view(n, d), t))
            w = (layer.indexer.weights.weight @ x[t]) * (n * d) ** -0.5
            scores = torch.stack([(w * (q_index @ keys[s]).relu()).sum() for s in visible])
            ranked = scores.sort(descending=True)
            k = min(layer.indexer.top_k, len(visible))
            assert k == len(visible) or ranked.values[k - 1] - ranked.values[k] > 1e-6  # no near tie decides the choice
            visible = [visible[s] for s in ranked.indices[:k].tolist()]
        kv = torch.stack(window + [entries[s] for s in visible])
        z = q @ kv.T * c ** -0.5
        s = z.exp() / (z.exp().sum(-1, keepdim=True) + layer.sink.exp()[:, None])  # eq. 27: the sink
        o = rope(s @ kv, -t)  # position -t on the output
        groups = o.reshape(layer.groups, -1)
        projected = [layer.out_down.weight.view(layer.groups, -1, groups.shape[-1])[g] @ groups[g]
                     for g in range(layer.groups)]
        out.append(layer.out_up.weight @ torch.cat(projected))
    return torch.stack(out)


def randomized(layer):
    """The layer with its sinks and positional biases away from their starting zeros, so they count."""
    torch.manual_seed(1)
    with torch.no_grad():
        layer.sink.normal_()
        for module in layer.modules():
            if hasattr(module, "position") and isinstance(module.position, torch.nn.Parameter):
                module.position.normal_()
    return layer


@pytest.mark.parametrize("ratio", [CSA_RATIO, HCA_RATIO])
def test_csa_and_hca_give_their_formulas_result(ratio):
    # docs/model.md: DeepSeek-V4's merging, indexer, window, sink, partial
    # rotary encoding with YaRN, entries in FP8 and the indexer's in FP4
    # after a Hadamard rotation, and grouped output, over enough tokens for
    # the window to slide and for entries to merge (75 CSA entries, 2 HCA)
    torch.manual_seed(0)
    layer = randomized(Attention(CONFIG, ratio)).eval()
    x = torch.randn(2, 300, CONFIG.d_model)
    with torch.no_grad():
        out, _ = layer(x)
        expected = torch.stack([attention_by_formula(layer, text) for text in x])
    assert torch.allclose(out, expected, atol=1e-4)


def sparse_inputs(B=2, T=40, H=3, c=16, N=50, K=12):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(B, T, H, c, generator=generator)
    kv = torch.randn(B, N, c, generator=generator)
    sink = torch.randn(H, generator=generator)
    idx = torch.randint(N, (B, T, K), generator=generator)
    idx[torch.rand(B, T, K, generator=generator) < 0.3] = -1
    idx[0, 5] = -1  # a query with nothing to attend to: only the sink
    return q, kv, sink, idx


def sparse_by_formula(q, kv, sink, idx, scale):
    entries = torch.stack([kv[b][idx[b].clamp(min=0)] for b in range(kv.shape[0])])  # (B, T, K, c)
    z = torch.einsum("bthc,btkc->bthk", q, entries) * scale
    z = z.masked_fill(idx[:, :, None] < 0, float("-inf"))
    z = torch.cat([z, sink[:, None].expand(*z.shape[:-1], 1)], -1)
    return torch.einsum("bthk,btkc->bthc", z.softmax(-1)[..., :-1], entries)


def test_sparse_attention_and_its_gradients_follow_the_formula(monkeypatch):
    # the PyTorch form computes both passes a block of queries at a time,
    # with a backward pass written by hand: it must give autograd's
    # gradients of the formula, including the sink's, across block edges
    monkeypatch.setattr(sparse, "BLOCK_VALUES", 2000)  # blocks of a few queries
    inputs = [t.requires_grad_() if t.is_floating_point() else t for t in sparse_inputs()]
    out, _ = _SparseAttention.apply(*inputs, 0.25)
    expected = sparse_by_formula(*inputs, 0.25)
    assert torch.allclose(out, expected, atol=1e-5)
    weights = torch.randn_like(out)
    got = torch.autograd.grad((out * weights).sum(), inputs[:3])
    want = torch.autograd.grad((expected * weights).sum(), inputs[:3])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(got, want))


def test_the_indexer_loss_and_its_gradients_follow_the_formula(monkeypatch):
    # DeepSeek-V3.2: KL(target || softmax(I)) over the chosen entries, mean
    # over queries; a query with no entry adds nothing
    monkeypatch.setattr(sparse, "BLOCK_VALUES", 2000)
    generator = torch.Generator().manual_seed(0)
    B, T, n, c, N, K = 2, 30, 4, 8, 20, 6
    q, w, keys = (torch.randn(*shape, generator=generator).requires_grad_()
                  for shape in ((B, T, n, c), (B, T, n), (B, N, c)))
    chosen = torch.randint(N, (B, T, K), generator=generator)
    chosen[torch.rand(B, T, K, generator=generator) < 0.3] = -1
    chosen[1, 3] = -1
    target = torch.rand(B, T, K, generator=generator).masked_fill(chosen < 0, 0)
    target = target / target.sum(-1, keepdim=True).clamp(min=1e-30)

    entries = torch.stack([keys[b][chosen[b].clamp(min=0)] for b in range(B)])
    scores = torch.einsum("bth,bthk->btk", w, torch.einsum("bthc,btkc->bthk", q, entries).relu())
    log_q = scores.masked_fill(chosen < 0, float("-inf")).log_softmax(-1)
    expected = torch.where(target > 0, target * (target.log() - log_q), 0).sum() / (B * T)

    loss = index_alignment(q, w, keys, chosen, target)
    assert torch.allclose(loss, expected, atol=1e-6)
    got, want = torch.autograd.grad(loss, (q, w, keys)), torch.autograd.grad(expected, (q, w, keys))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(got, want))


def runnable_kernels():
    """The kernel modules that can run on this machine's GPU: TileLang on an
    NVIDIA GPU under Linux with TileLang and nvcc, Triton on any GPU it
    supports. Each is tested, whichever kernels() would choose."""
    device = select_device()  # select_device: AMD GPUs need it
    found = []
    if device.type != "cuda":
        return found
    for name, needs in (("tilelang_kernels", torch.version.cuda and sparse._nvcc()), ("triton_kernels", True)):
        try:
            module = importlib.import_module(f"wpt_model.{name}") if needs else None
        except ImportError:
            module = None
        if module is not None:
            found.append(module)
    return found


KERNELS = runnable_kernels()
needs_kernels = pytest.mark.skipif(not KERNELS, reason="needs a GPU with TileLang (NVIDIA, Linux, nvcc) or Triton")
each_kernel = pytest.mark.parametrize("chosen", KERNELS, ids=lambda module: module.NAME)
RECIPES = [(4, 16), (8, 64), (4, 40), (2, 16), (5, 40)]  # attention heads and top k; indexer heads and top k


def recipe_rows(B, T, k, generator):
    """Index rows as a CSA layer builds them: the window (its -1s at the end
    of its part, so in mid-row), then the top k of the visible merged
    entries, -1 where fewer are visible."""
    q = torch.randn(B, T, 2, 128, generator=generator)
    keys, w = torch.randn(B, T // CSA_RATIO, 128, generator=generator), torch.rand(B, T, 2, generator=generator)
    chosen = top_entries(q, w, keys, (torch.arange(T) + 1) // CSA_RATIO, k)
    return torch.cat([_window(T, "cpu").expand(B, -1, -1), torch.where(chosen >= 0, chosen + T, -1)], -1)


def relative_error(got, expected):
    return ((got.cpu() - expected).abs().max() / expected.abs().max()).item()


@needs_kernels
@each_kernel
@pytest.mark.parametrize("heads, k", RECIPES[:3])
def test_the_attention_kernels_give_the_pytorch_result(chosen, heads, k):
    # docs/model.md: kernels compute from BF16 inputs; against the PyTorch
    # form on the same BF16 values, on rows as the recipes' CSA layers make
    # them: the output, the weights and the log-sum-exp within 1e-2, 1e-4 and
    # 1e-4 of their largest value, and every gradient within 1e-2
    device, generator = select_device(), torch.Generator().manual_seed(0)
    B, T, c = 2, 600, 128
    idx = recipe_rows(B, T, k, generator)
    q, kv = (torch.randn(*shape, generator=generator).bfloat16().float()
             for shape in ((B, T, heads, c), (B, T + T // CSA_RATIO, c)))
    sink = torch.randn(heads, generator=generator)
    inputs = [t.requires_grad_() for t in (q, kv, sink)]
    out, weights = _SparseAttention.apply(*inputs, idx, c ** -0.5)
    d_out = torch.randn_like(out)
    expected = torch.autograd.grad((out * d_out).sum(), inputs)
    lse = torch.logaddexp(torch.logsumexp(torch.einsum("bthc,btkc->bthk", q, torch.stack(
        [kv[b][idx[b].clamp(min=0)] for b in range(B)])).masked_fill(idx[:, :, None] < 0, float("-inf")) * c ** -0.5,
        -1), sink).detach()

    on = [t.detach().to(device) for t in (q.bfloat16(), kv.bfloat16(), sink)] + [idx.to(device, torch.int32)]
    got_out = chosen.attention(*on, c ** -0.5)
    got_lse, got_weights = chosen.statistics(*on, c ** -0.5)
    delta = (d_out * out.detach()).sum(-1)
    dq, dkv = chosen.attention_backward(on[0], on[1], on[3], c ** -0.5, got_lse, d_out.to(device), delta.to(device))
    d_sink = -((sink.detach() - got_lse.cpu()).exp() * delta).sum((0, 1))
    assert relative_error(got_out, out.detach()) < 1e-2
    assert relative_error(got_weights, weights) < 1e-4 and relative_error(got_lse, lse) < 1e-4
    assert all(relative_error(a, b) < 1e-2 for a, b in zip((dq, dkv, d_sink), expected))


@needs_kernels
@each_kernel
@pytest.mark.parametrize("heads, k", RECIPES[3:] + [(8, 64)])
def test_the_indexer_kernel_gives_the_pytorch_scores_and_choice(chosen, heads, k):
    # DeepSeek's FP8 indexer kernel (and its translation) reads the
    # indexer's FP8 values: its scores within 1e-3 of the largest score of
    # the PyTorch form (FP8 matrix units may accumulate in less than FP32),
    # and its top k as good, every -1 in the same place
    device, generator = select_device(), torch.Generator().manual_seed(0)
    B, T, N, c = 2, 300, 75, 128
    q = indexed(torch.randn(B, T, heads, c, generator=generator))
    keys = indexed(torch.randn(B, N, c, generator=generator))
    w = torch.randn(B, T, heads, generator=generator)
    expected = torch.einsum("bth,bthn->btn", w, torch.einsum("bthc,bnc->bthn", q, keys).relu())
    numbers, scales = fp8_parts(q, c)
    key_numbers, key_scales = fp8_parts(keys, c)
    on = (numbers.to(torch.float8_e4m3fn), w * scales[..., 0], key_numbers.to(torch.float8_e4m3fn), key_scales[..., 0])
    assert relative_error(chosen.index_scores(*(t.to(device).contiguous() for t in on)), expected) < 1e-3

    visible = (torch.arange(T) + 1) // CSA_RATIO
    picked = top_entries(q, w, keys, visible, k)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sparse, "_kernels", chosen)
        got = top_entries(*(t.to(device) for t in (q, w, keys, visible)), k).cpu()
    assert torch.equal(got < 0, picked < 0)
    score = lambda chosen: expected.gather(-1, chosen.clamp(min=0)).masked_fill(chosen < 0, 0).sort(-1).values
    assert relative_error(score(got), score(picked)) < 1e-3


@needs_kernels
def test_a_csa_layer_trains_with_the_kernels_as_with_pytorch():
    # docs/model.md: on a GPU with kernels, training passes use them too
    # (the attention, its statistics for the indexer's target, its backward
    # pass, and the indexer's scores): a CSA layer's output, indexer loss and
    # gradients agree with the PyTorch form's on the CPU within 1e-2 of
    # their largest value
    torch.manual_seed(0)
    device = select_device()
    layer = randomized(Attention(CONFIG, CSA_RATIO))
    x = torch.randn(2, 300, CONFIG.d_model)
    results = []
    for where in ("cpu", device):
        layer.zero_grad()
        here = layer.to(where)
        out, loss = here(x.to(where), indexer_loss=True)
        (out.square().mean() + loss).backward()
        results.append([out.detach().cpu(), loss.detach().cpu()] + [p.grad.cpu() for p in here.parameters()])
    assert kernels(device) is not None
    assert all(relative_error(got, want) < 1e-2 for got, want in zip(results[1], results[0]))


def fake_kernels(fails):
    """A stand-in for a kernel module, whose kernels fail with fails."""
    def kernel(*_):
        raise fails
    return types.SimpleNamespace(NAME="Triton", attention=kernel, statistics=kernel, attention_backward=kernel,
                                 index_scores=kernel)


def test_installed_kernels_that_cannot_work_stop_instead_of_falling_back(monkeypatch):
    # docs/model.md: which form runs is decided once; TileLang installed
    # without nvcc, or kernels that do not run on this GPU, are an error
    # with its reason, and neither TileLang nor Triton installed means
    # PyTorch. Stand-ins play the installed packages: neither the real
    # import nor a GPU is exercised
    gpu = torch.device("cuda")
    monkeypatch.setattr(sparse, "_kernels", sparse._UNDECIDED)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(sparse, "_installed", lambda name: True)
    monkeypatch.setattr(sparse, "_nvcc", lambda: None)
    with pytest.raises(KernelUnavailable, match="nvcc"):
        kernels(gpu)

    monkeypatch.setattr(sparse, "_installed", lambda name: name == "triton")
    monkeypatch.setitem(sys.modules, "wpt_model.triton_kernels", fake_kernels(RuntimeError("no image")))
    monkeypatch.delattr(wpt_model, "triton_kernels", raising=False)
    monkeypatch.setattr(sparse, "_try", lambda chosen, device: chosen.attention())
    with pytest.raises(KernelUnavailable, match="Triton kernels do not run on this GPU: RuntimeError: no image"):
        kernels(gpu)

    monkeypatch.setattr(sparse, "_installed", lambda name: False)
    assert kernels(gpu) is None and kernels(torch.device("cpu")) is None


def test_the_mtp_block_sees_only_the_last_128_tokens():
    # its attention is a sliding window (MiMo-V2-Flash, DeepSeek-V4's MTP
    # block): a change at position 10 reaches positions 10 .. 137 and no other
    torch.manual_seed(0)
    mtp = WPT(CONFIG).mtp.eval()
    hidden, following = torch.randn(1, 300, CONFIG.d_model), torch.randn(1, 300, CONFIG.d_model)
    changed = hidden.clone()
    changed[:, 10] += 1
    with torch.no_grad():
        before, after = mtp(hidden, following), mtp(changed, following)
    differs = (before - after).abs().amax(-1)[0] > 1e-5
    assert differs[10:10 + WINDOW].all() and not differs[:10].any() and not differs[10 + WINDOW:].any()

