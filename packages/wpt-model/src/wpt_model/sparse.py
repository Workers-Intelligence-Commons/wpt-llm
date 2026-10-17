"""The two computations of DeepSeek-V4's attention whose size grows with the
text (docs/model.md, "Attention"), each over a table of entries that rows
of indices name (-1 names none):

- sparse_attention: every query head attends to its row's entries, each
  entry both key and value (one head shared by all query heads), with a
  learned sink logit per head added to the softmax's denominator:

      z_{t,h,j} = scale q_{t,h} . kv_{idx[t,j]}
      o_{t,h}   = sum_j exp(z_{t,h,j}) kv_{idx[t,j]} / (sum_j exp(z_{t,h,j}) + exp(sink_h))

- the lightning indexer (DeepSeek-V3.2, V4): each query's score of an
  entry, I_{t,s} = sum_h w_{t,h} ReLU(q_{t,h} . k_s); top_entries keeps the
  k best visible ones, and index_alignment is its training loss, the
  KL divergence from the main attention's weights on those entries.

Each has two forms. The PyTorch form runs a block of queries at a time, in
the forward and the backward pass, so memory holds one block's gathered
entries instead of every query's; it runs on every device and computes in
FP32. On a GPU, kernels compute them instead, for every pass, training
included (kernels()):

    TileLang  an NVIDIA GPU under Linux with TileLang installed
              (tilelang_kernels.py: DeepSeek's attention and FP8 indexer
              kernels, and WPT's backward kernels)
    Triton    any other NVIDIA or AMD GPU where Triton is installed
              (triton_kernels.py: translations of the same)

Kernels read the queries and entries in BF16 (the entries' FP8 values are
exact in BF16) and the indexer's queries and keys in FP8. The indexer's
loss (index_alignment) stays in PyTorch: it reads only the chosen entries.
Tests check each form against the formulas, and the kernels against the
PyTorch form.

Which form runs is decided once per process, at the first GPU pass (or
before, by wpt-train): neither TileLang nor Triton installed means PyTorch;
an installed one that cannot work (no CUDA compiler for TileLang, an import
that fails, a kernel that does not compile or run on this GPU) is
KernelUnavailable, which stops the run. A failure inside a later call is an
error like any other, never a switch to PyTorch.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import torch

from .formats import fp8_parts

BLOCK_VALUES = 1 << 24  # gathered values per block of queries: 64 MiB in FP32

_UNDECIDED = object()
_kernels = _UNDECIDED


class KernelUnavailable(RuntimeError):
    """TileLang or Triton is installed for this GPU but its kernels cannot
    work here: the reason says why. Nothing has been computed."""


def kernels(device):
    """The kernels that compute the attention on device (a module with
    NAME, attention, statistics, attention_backward and index_scores), or
    None for the PyTorch form; decided at the first call on a GPU. Raises
    KernelUnavailable when TileLang or Triton is installed but cannot work."""
    global _kernels
    if device.type != "cuda":
        return None
    if _kernels is _UNDECIDED:
        _kernels = _decide(device)
    return _kernels


def _decide(device):
    if torch.version.cuda and sys.platform == "linux" and _installed("tilelang"):
        if _nvcc() is None:
            raise KernelUnavailable("TileLang compiles its attention kernels with the CUDA compiler (nvcc), which "
                                    "is not here: install the CUDA Toolkit and put nvcc on PATH or set CUDA_HOME, "
                                    "or remove tilelang to use the Triton kernels")
        try:
            from . import tilelang_kernels as chosen
        except Exception as error:  # an ImportError, or an error of TileLang's own while it starts
            raise KernelUnavailable(f"TileLang is installed but does not import: {error}") from error
    elif _installed("triton"):
        try:
            from . import triton_kernels as chosen
        except Exception as error:
            raise KernelUnavailable(f"Triton is installed but does not import: {error}") from error
    else:
        return None
    try:  # every kernel compiled and run once, so that a GPU they cannot run on stops the run now
        _try(chosen, device)
    except Exception as error:
        raise KernelUnavailable(f"{chosen.NAME} kernels do not run on this GPU: {type(error).__name__}: {error}"
                                ) from error
    return chosen


def _installed(name):
    """Whether a module is installed, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # a module already imported without a spec
        return True


def _try(chosen, device):
    """Each kernel once, on a small input of the recipes' shapes."""
    generator = torch.Generator().manual_seed(0)
    B, T, H, c, N, K = 1, 4, 4, 128, 8, 64
    q = torch.randn(B, T, H, c, generator=generator).to(device, torch.bfloat16)
    kv = torch.randn(B, N, c, generator=generator).to(device, torch.bfloat16)
    sink = torch.zeros(H, device=device)
    idx = torch.full((B, T, K), -1, dtype=torch.int32)
    idx[..., 0] = torch.arange(T)
    idx = idx.to(device)
    chosen.attention(q, kv, sink, idx, c ** -0.5)
    lse, _ = chosen.statistics(q, kv, sink, idx, c ** -0.5)
    chosen.attention_backward(q, kv, idx, c ** -0.5, lse, q.float(), lse)
    numbers, scales = fp8_parts(q.float(), c)
    chosen.index_scores(numbers.to(torch.float8_e4m3fn), scales, numbers[:, :, 0].to(torch.float8_e4m3fn),
                        scales[:, :, 0])
    torch.cuda.synchronize(device)


def _nvcc():
    """The CUDA compiler's path, looked for where TileLang looks (CUDA_HOME,
    CUDA_PATH, PATH, /usr/local/cuda), or None."""
    homes = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    found = [Path(home) / "bin" / "nvcc" for home in homes if home] + [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"]
    return next((str(path) for path in found if path and Path(path).is_file()), None)


def _rows(batch, *sizes):
    """Queries per block, so one block's largest tensor holds about BLOCK_VALUES values."""
    values = batch
    for size in sizes:
        values *= max(1, size)
    return max(1, BLOCK_VALUES // values)


def _gather(table, idx):
    """table (B, N, c), idx (B, R, K) -> (B, R, K, c). An index of -1 reads
    entry 0, which the caller masks."""
    B, N, c = table.shape
    flat = idx.clamp(min=0) + torch.arange(B, device=idx.device)[:, None, None] * N
    return table.reshape(B * N, c)[flat]


def _scatter_add(table_grad, idx, values):
    """The reverse of _gather: adds values (B, R, K, c) into table_grad (B, N, c)."""
    B, N, c = table_grad.shape
    flat = idx.clamp(min=0) + torch.arange(B, device=idx.device)[:, None, None] * N
    table_grad.view(B * N, c).index_add_(0, flat.reshape(-1), values.reshape(-1, c))


class _SparseAttention(torch.autograd.Function):
    """sparse_attention in FP32. Saves only its inputs and outputs; the
    backward pass gathers each block's entries again."""

    @staticmethod
    def forward(ctx, q, kv, sink, idx, scale):
        B, T, H, c = q.shape
        out, lse = q.new_empty(B, T, H, c), q.new_empty(B, T, H)
        weights = q.new_empty(idx.shape)
        rows = _rows(B, idx.shape[-1], max(c, H))
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            entries = _gather(kv, idx[:, part])
            z = torch.einsum("brhc,brkc->brhk", q[:, part], entries) * scale
            z = z.masked_fill(idx[:, part, None, :] < 0, float("-inf"))
            lse[:, part] = torch.logaddexp(torch.logsumexp(z, -1), sink)  # the sink joins the denominator
            p = (z - lse[:, part, :, None]).exp()
            out[:, part] = torch.einsum("brhk,brkc->brhc", p, entries)
            weights[:, part] = p.sum(2)
        ctx.save_for_backward(q, kv, sink, idx, out, lse)
        ctx.scale = scale
        ctx.mark_non_differentiable(weights)
        return out, weights

    @staticmethod
    def backward(ctx, d_out, _d_weights):
        q, kv, sink, idx, out, lse = ctx.saved_tensors
        B, T, H, c = q.shape
        delta = (d_out * out).sum(-1)  # (B, T, H): sum_j p_j dp_j
        dq, dkv = torch.empty_like(q), torch.zeros_like(kv)
        rows = _rows(B, idx.shape[-1], max(c, H))
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            entries = _gather(kv, idx[:, part])
            z = torch.einsum("brhc,brkc->brhk", q[:, part], entries) * ctx.scale
            z = z.masked_fill(idx[:, part, None, :] < 0, float("-inf"))
            p = (z - lse[:, part, :, None]).exp()
            dp = torch.einsum("brhc,brkc->brhk", d_out[:, part], entries)
            dz = p * (dp - delta[:, part, :, None])
            dq[:, part] = torch.einsum("brhk,brkc->brhc", dz, entries) * ctx.scale
            d_entries = (torch.einsum("brhk,brhc->brkc", dz, q[:, part]) * ctx.scale
                         + torch.einsum("brhk,brhc->brkc", p, d_out[:, part]))
            _scatter_add(dkv, idx[:, part], d_entries)
        d_sink = -((sink - lse).exp() * delta).sum((0, 1))  # the sink carries no value
        return dq, dkv, d_sink, None, None


class _KernelAttention(torch.autograd.Function):
    """sparse_attention by a GPU's kernels, from BF16 queries and entries.
    The forward pass also computes the softmax's statistics when a gradient
    or the weights are needed; the backward pass reads them."""

    @staticmethod
    def forward(ctx, q, kv, sink, idx, scale, chosen, weights):
        qb, kvb = q.to(torch.bfloat16).contiguous(), kv.to(torch.bfloat16).contiguous()
        sinkf, idx32 = sink.float().contiguous(), idx.to(torch.int32).contiguous()
        out = chosen.attention(qb, kvb, sinkf, idx32, scale)
        lse = summed = None
        if weights or any(ctx.needs_input_grad[:3]):
            lse, summed = chosen.statistics(qb, kvb, sinkf, idx32, scale)
            ctx.mark_non_differentiable(summed)
        ctx.save_for_backward(qb, kvb, sinkf, idx32, out, lse)
        ctx.scale, ctx.chosen, ctx.dtypes = scale, chosen, (q.dtype, kv.dtype, sink.dtype)
        return out, summed

    @staticmethod
    def backward(ctx, d_out, _d_weights):
        qb, kvb, sink, idx, out, lse = ctx.saved_tensors
        d_out = d_out.float()
        delta = (d_out * out).sum(-1)  # (B, T, H): sum_j p_j dp_j
        dq, dkv = ctx.chosen.attention_backward(qb, kvb, idx, ctx.scale, lse, d_out, delta)
        d_sink = -((sink - lse).exp() * delta).sum((0, 1))  # the sink carries no value
        q_type, kv_type, sink_type = ctx.dtypes
        return dq.to(q_type), dkv.to(kv_type), d_sink.to(sink_type), None, None, None, None


def sparse_attention(q, kv, sink, idx, scale, weights=False):
    """q: (B, T, H, c); kv: (B, N, c); sink: (H,); idx: (B, T, K), -1 for
    none. Returns the output (B, T, H, c) in FP32, and with weights, each
    entry's weight summed over the heads (B, T, K) (None without). On a GPU
    with kernels (kernels()), they compute it, from BF16 inputs."""
    chosen = kernels(q.device)
    with torch.autocast(q.device.type, enabled=False):
        if chosen is not None:
            out, summed = _KernelAttention.apply(q, kv, sink, idx, scale, chosen, weights)
        else:
            out, summed = _SparseAttention.apply(q.float(), kv.float(), sink.float(), idx, scale)
    return out, summed if weights else None


@torch.no_grad()
def top_entries(q, w, keys, visible, k):
    """The indexer's choice: for each query, the indices of its k best
    entries among the first visible[t], -1 where fewer are visible.
    q: (B, T, n, c); w: (B, T, n); keys: (B, N, c); visible: (T,).
    Returns (B, T, min(k, N)). On a GPU with kernels, the FP8 indexer kernel
    computes the scores."""
    B, T, n, c = q.shape
    N = keys.shape[1]
    chosen = torch.full((B, T, min(k, N)), -1, dtype=torch.long, device=q.device)
    if N == 0:
        return chosen
    kernel = kernels(q.device)
    with torch.autocast(q.device.type, enabled=False):
        q, w, keys = q.float(), w.float(), keys.float()
        if kernel is not None:  # in FP8, one scale per head and key: exact for the indexer's values (attention.py)
            q, q_scale = fp8_parts(q, c)
            q, w = q.to(torch.float8_e4m3fn), w * q_scale[..., 0]
            keys, key_scale = fp8_parts(keys, c)
            keys, key_scale = keys.to(torch.float8_e4m3fn).contiguous(), key_scale[..., 0].contiguous()
        positions = torch.arange(N, device=q.device)
        rows = _rows(B, n, N)
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            if kernel is not None:
                scores = kernel.index_scores(q[:, part].contiguous(), w[:, part].contiguous(), keys, key_scale)
            else:
                scores = torch.einsum("brh,brhn->brn", w[:, part],
                                      torch.einsum("brhc,bnc->brhn", q[:, part], keys).relu_())
            hidden = positions >= visible[part, None]  # (R, N): entries not complete before the query
            best = scores.masked_fill_(hidden, float("-inf")).topk(chosen.shape[-1], -1).indices
            chosen[:, part] = best.masked_fill(best >= visible[part][None, :, None], -1)
    return chosen


class _IndexAlignment(torch.autograd.Function):
    """The mean over queries of KL(target_t || softmax(I_{t,S_t})), in FP32.
    Saves only its inputs; the backward pass gathers each block again."""

    @staticmethod
    def forward(ctx, q, w, keys, chosen, target):
        B, T, n, c = q.shape
        total = q.new_zeros(())
        rows = _rows(B, chosen.shape[-1], max(n, c))
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            scores, _ = _index_scores(q[:, part], w[:, part], keys, chosen[:, part])
            p = target[:, part]
            log_q = scores.log_softmax(-1)
            total = total + torch.where(p > 0, p * (p.log() - log_q), 0).sum()
        ctx.save_for_backward(q, w, keys, chosen, target)
        return total / (B * T)

    @staticmethod
    def backward(ctx, d_total):
        q, w, keys, chosen, target = ctx.saved_tensors
        B, T, n, c = q.shape
        dq, dw, dkeys = torch.empty_like(q), torch.empty_like(w), torch.zeros_like(keys)
        rows = _rows(B, chosen.shape[-1], max(n, c))
        for start in range(0, T, rows):
            part = slice(start, start + rows)
            scores, (entries, a) = _index_scores(q[:, part], w[:, part], keys, chosen[:, part])
            p = target[:, part]
            has = p.sum(-1, keepdim=True) > 0  # queries without entries add nothing
            d_scores = torch.where(has, scores.softmax(-1).nan_to_num() - p, 0) * (d_total / (B * T))
            dw[:, part] = torch.einsum("brk,brhk->brh", d_scores, a.relu())
            da = w[:, part, :, None] * d_scores[:, :, None, :] * (a > 0)
            dq[:, part] = torch.einsum("brhk,brkc->brhc", da, entries)
            _scatter_add(dkeys, chosen[:, part], torch.einsum("brhk,brhc->brkc", da, q[:, part]))
        return dq, dw, dkeys, None, None


def _index_scores(q, w, keys, chosen):
    """I_{t,s} for the chosen entries (-inf where none), with the gathered
    entries and the scores before ReLU."""
    entries = _gather(keys, chosen)
    a = torch.einsum("brhc,brkc->brhk", q, entries)
    scores = torch.einsum("brh,brhk->brk", w, a.relu()).masked_fill(chosen < 0, float("-inf"))
    return scores, (entries, a)


def index_alignment(q, w, keys, chosen, target):
    """The indexer's loss (DeepSeek-V3.2's sparse stage): over the chosen
    entries (B, T, K), the KL divergence from target, the main attention's
    weights summed over heads and normalized over those entries, to the
    softmax of the index scores, averaged over queries. Its gradient
    reaches only q, w and keys."""
    with torch.autocast(q.device.type, enabled=False):
        return _IndexAlignment.apply(q.float(), w.float(), keys.float(), chosen, target.float())
