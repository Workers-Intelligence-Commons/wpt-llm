"""The attention kernels for Triton, on an NVIDIA or AMD GPU (Linux, and
Windows through triton-windows): what sparse.py computes, with the same
contract as tilelang_kernels.py.

- attention: a translation of DeepSeek's sparse attention kernel
  (deepseek_kernels.py): one query per program, its entries gathered in
  blocks of 64, an online softmax in FP32 with the sink added to the
  denominator at the end, the weights cast to BF16 before they multiply
  the entries, as there.
- statistics and attention_backward: the same as tilelang_kernels.py's,
  for training.
- index_scores: a translation of DeepSeek's FP8 indexer kernel
  (deepseek_index_kernel.py), one program per query and block of 128
  keys. It reads the FP8 queries and keys as bytes and turns them into
  BF16, which holds every FP8 value exactly, before multiplying them, so it
  runs on GPUs without FP8 arithmetic too and gives the same products.

Inputs: q (B, T, H, c) and kv (B, N, c) in BF16, sink (H,) in FP32, idx
(B, T, K) in int32 with -1 for none; outputs in FP32. Heads are padded to
16 inside the kernels (Triton's matrix products need 16 rows), and the head
width to a power of 2.

_attention and _index_scores translate DeepSeek's MIT-licensed kernels,
whose notice follows:

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import triton
import triton.language as tl

NAME = "Triton"
BLOCK = 64  # entries per block, as DeepSeek's kernel
KEYS = 128  # keys per program of the indexer, DeepSeek's inner block


def attention(q, kv, sink, idx, scale):
    """(B, T, H, c): the attention's output."""
    B, T, H, c = q.shape
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    _attention[(B * T,)](q, kv, sink, idx, out, scale, T, kv.shape[1], idx.shape[-1], **_shape(H, c))
    return out


def statistics(q, kv, sink, idx, scale):
    """(lse (B, T, H), weights (B, T, K)): each head's log of its softmax's
    denominator (the sink included), and each entry's weight summed over
    heads."""
    B, T, H, c = q.shape
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    weights = torch.empty(idx.shape, dtype=torch.float32, device=q.device)
    _statistics[(B * T,)](q, kv, sink, idx, lse, weights, scale, T, kv.shape[1], idx.shape[-1], **_shape(H, c))
    return lse, weights


def attention_backward(q, kv, idx, scale, lse, d_out, delta):
    """(dq (B, T, H, c), dkv (B, N, c)): the gradients of the queries and
    entries, from lse (statistics), d_out (B, T, H, c) and delta (B, T, H),
    the sum over c of d_out times the output. The entries' gradients are
    added up across queries by atomic additions."""
    B, T, H, c = q.shape
    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    dkv = torch.zeros(kv.shape, dtype=torch.float32, device=kv.device)
    _backward[(B * T,)](q, kv, idx, lse, d_out.contiguous(), delta, dq, dkv, scale, T, kv.shape[1], idx.shape[-1],
                        **_shape(H, c))
    return dq, dkv


def index_scores(q, q_scale, k, k_scale):
    """(B, T, N): sum_h q_scale * ReLU(q . k) * k_scale, from FP8 queries q
    (B, T, n, c) and keys k (B, N, c), q_scale (B, T, n) and k_scale (B, N)
    in FP32."""
    B, T, n, c = q.shape
    N = k.shape[1]
    out = torch.empty(B, T, N, dtype=torch.float32, device=q.device)
    grid = (B * T, triton.cdiv(N, KEYS))
    _index_scores[grid](q.view(torch.uint8), q_scale, k.view(torch.uint8), k_scale, out, T, N,
                        H=n, HP=max(16, triton.next_power_of_2(n)), C=c, KEYS=KEYS)
    return out


def _shape(H, c):
    return dict(H=H, HP=max(16, triton.next_power_of_2(H)), C=c, CP=triton.next_power_of_2(c), BLOCK=BLOCK)


@triton.jit
def _block(q, kv, idx, row, b, start, scale, N, K, C: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    """One block of a query's entries: (entries (BLOCK, CP) in BF16, 0 where
    none; scores (HP, BLOCK) in FP32, -inf where none; the indices; which
    are there)."""
    cols = start + tl.arange(0, BLOCK)
    dims = tl.arange(0, CP)
    entry = tl.load(idx + row * K + cols, mask=cols < K, other=-1)
    there = entry >= 0
    rows = b * N + tl.where(there, entry, 0).to(tl.int64)
    entries = tl.load(kv + rows[:, None] * C + dims[None, :], mask=there[:, None] & (dims[None, :] < C), other=0.0)
    z = tl.dot(q, tl.trans(entries)) * scale
    return entries, tl.where(there[None, :], z, float("-inf")), rows, there


@triton.jit
def _load_heads(pointer, row, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr):
    """A query's (HP, CP) values at row, 0 for padded heads and dimensions."""
    heads, dims = tl.arange(0, HP), tl.arange(0, CP)
    return tl.load(pointer + row * H * C + heads[:, None] * C + dims[None, :],
                   mask=(heads[:, None] < H) & (dims[None, :] < C), other=0.0)


@triton.jit
def _softmax_denominator(q, kv, idx, row, b, scale, N, K, C: tl.constexpr, CP: tl.constexpr, HP: tl.constexpr,
                         BLOCK: tl.constexpr):
    """(the largest score per head, or 0 where there is none; the sum of exp
    of the scores minus it), over every entry of the query."""
    largest = tl.full([HP], float("-inf"), tl.float32)
    total = tl.zeros([HP], tl.float32)
    for start in range(0, K, BLOCK):
        _, z, _, _ = _block(q, kv, idx, row, b, start, scale, N, K, C, CP, BLOCK)
        new = tl.maximum(largest, tl.max(z, 1))
        safe = tl.where(new == float("-inf"), 0.0, new)
        total = total * tl.exp(largest - safe) + tl.sum(tl.exp(z - safe[:, None]), 1)
        largest = new
    return tl.where(largest == float("-inf"), 0.0, largest), total


@triton.jit
def _attention(q_ptr, kv, sink_ptr, idx, out, scale, T, N, K, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr,
               CP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = row // T
    heads, dims = tl.arange(0, HP), tl.arange(0, CP)
    q = _load_heads(q_ptr, row, H, HP, C, CP)
    largest = tl.full([HP], float("-inf"), tl.float32)
    total = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, CP], tl.float32)
    for start in range(0, K, BLOCK):
        entries, z, _, _ = _block(q, kv, idx, row, b, start, scale, N, K, C, CP, BLOCK)
        new = tl.maximum(largest, tl.max(z, 1))
        safe = tl.where(new == float("-inf"), 0.0, new)
        rescale = tl.exp(largest - safe)
        p = tl.exp(z - safe[:, None])
        total = total * rescale + tl.sum(p, 1)
        acc = acc * rescale[:, None] + tl.dot(p.to(tl.bfloat16), entries)
        largest = new
    safe = tl.where(largest == float("-inf"), 0.0, largest)
    sink = tl.load(sink_ptr + heads, mask=heads < H, other=0.0)
    total += tl.exp(sink - safe)
    tl.store(out + row * H * C + heads[:, None] * C + dims[None, :], acc / total[:, None],
             mask=(heads[:, None] < H) & (dims[None, :] < C))


@triton.jit
def _statistics(q_ptr, kv, sink_ptr, idx, lse_ptr, weights, scale, T, N, K, H: tl.constexpr, HP: tl.constexpr,
                C: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = row // T
    heads = tl.arange(0, HP)
    q = _load_heads(q_ptr, row, H, HP, C, CP)
    largest, total = _softmax_denominator(q, kv, idx, row, b, scale, N, K, C, CP, HP, BLOCK)
    sink = tl.load(sink_ptr + heads, mask=heads < H, other=0.0)
    lse = largest + tl.log(total + tl.exp(sink - largest))
    tl.store(lse_ptr + row * H + heads, lse, mask=heads < H)
    lse = tl.where(heads < H, lse, float("inf"))  # padded heads take no weight
    for start in range(0, K, BLOCK):
        _, z, _, _ = _block(q, kv, idx, row, b, start, scale, N, K, C, CP, BLOCK)
        cols = start + tl.arange(0, BLOCK)
        tl.store(weights + row * K + cols, tl.sum(tl.exp(z - lse[:, None]), 0), mask=cols < K)


@triton.jit
def _backward(q_ptr, kv, idx, lse_ptr, d_out, delta_ptr, dq_ptr, dkv, scale, T, N, K, H: tl.constexpr,
              HP: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = row // T
    heads, dims = tl.arange(0, HP), tl.arange(0, CP)
    q = _load_heads(q_ptr, row, H, HP, C, CP)
    do = _load_heads(d_out, row, H, HP, C, CP).to(tl.bfloat16)
    lse = tl.load(lse_ptr + row * H + heads, mask=heads < H, other=float("inf"))
    delta = tl.load(delta_ptr + row * H + heads, mask=heads < H, other=0.0)
    dq = tl.zeros([HP, CP], tl.float32)
    for start in range(0, K, BLOCK):
        entries, z, rows, there = _block(q, kv, idx, row, b, start, scale, N, K, C, CP, BLOCK)
        p = tl.exp(z - lse[:, None])
        dp = tl.dot(do, tl.trans(entries))
        dz = (p * (dp - delta[:, None]) * scale).to(tl.bfloat16)
        dq += tl.dot(dz, entries)
        d_entries = tl.dot(tl.trans(dz), q) + tl.dot(tl.trans(p.to(tl.bfloat16)), do)
        tl.atomic_add(dkv + rows[:, None] * C + dims[None, :], d_entries, mask=there[:, None] & (dims[None, :] < C))
    tl.store(dq_ptr + row * H * C + heads[:, None] * C + dims[None, :], dq,
             mask=(heads[:, None] < H) & (dims[None, :] < C))


@triton.jit
def _fp8(byte):
    """FP8 E4M3 bytes as their values in BF16 (exact): sign, 4 exponent bits
    with bias 7, 3 mantissa bits, subnormals below exponent 1."""
    bits = byte.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = (bits & 7).to(tl.float32)
    magnitude = tl.where(exponent == 0, mantissa * 0.001953125,  # 2^-9
                         (8.0 + mantissa) * tl.exp2((exponent - 10).to(tl.float32)))
    return tl.where(bits >= 128, -magnitude, magnitude).to(tl.bfloat16)


@triton.jit
def _index_scores(q_ptr, q_scale, k_ptr, k_scale, out, T, N, H: tl.constexpr, HP: tl.constexpr, C: tl.constexpr,
                  KEYS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = row // T
    keys = tl.program_id(1) * KEYS + tl.arange(0, KEYS)
    heads, dims = tl.arange(0, HP), tl.arange(0, C)
    q = _fp8(tl.load(q_ptr + row * H * C + heads[:, None] * C + dims[None, :], mask=heads[:, None] < H, other=0))
    k = _fp8(tl.load(k_ptr + (b * N + keys[:, None]) * C + dims[None, :], mask=keys[:, None] < N, other=0))
    logits = tl.dot(k, tl.trans(q))  # (KEYS, HP)
    weights = tl.load(q_scale + row * H + heads, mask=heads < H, other=0.0)
    summed = tl.sum(tl.maximum(logits, 0) * weights[None, :], 1)
    scale = tl.load(k_scale + b * N + keys, mask=keys < N, other=0.0)
    tl.store(out + row * N + keys, summed * scale, mask=keys < N)
