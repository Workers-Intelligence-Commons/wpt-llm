"""The attention kernels for TileLang, on an NVIDIA GPU under Linux: what
sparse.py computes, with the same contract as triton_kernels.py.

- attention: DeepSeek's sparse attention kernel (deepseek_kernels.py).
- statistics and attention_backward: WPT's own, written in the shape of
  DeepSeek's kernel (one query per program, its entries in blocks of 64,
  heads padded to 16), for training, since DeepSeek published no backward
  pass. statistics gives each head's log of its softmax's denominator (the
  sink included) and each entry's weight summed over heads (the indexer's
  target); attention_backward gives the gradients of the queries and
  entries, the entries' added up across queries by atomic additions.
- index_scores: DeepSeek's FP8 indexer kernel (deepseek_index_kernel.py).

Inputs: q (B, T, H, c) and kv (B, N, c) in BF16, sink (H,) in FP32, idx
(B, T, K) in int32 with -1 for none; outputs in FP32. Importing this
module imports TileLang and compiles nothing; sparse.py imports it once
it has decided that TileLang is used.
"""

import torch
import tilelang
import tilelang.language as T

from .deepseek_index_kernel import fp8_index
from .deepseek_kernels import BF16, FP32, INT32, pass_configs, sparse_attn

NAME = "TileLang (DeepSeek's kernels)"
HEADS, BLOCK = 16, 64  # heads padded to 16 for the matrix units, as DeepSeek's kernel; entries per block
KEYS = 512  # the FP8 indexer kernel's block of keys


def attention(q, kv, sink, idx, scale):
    """(B, T, H, c): the attention's output, by DeepSeek's kernel."""
    return sparse_attn(q, kv, sink, idx, scale).float()


def statistics(q, kv, sink, idx, scale):
    """(lse (B, T, H), weights (B, T, K)): each head's log of its softmax's
    denominator, and each entry's weight summed over heads."""
    H, K = q.shape[2], idx.shape[-1]
    q, sink, idx = _padded(q, sink, idx)
    lse = q.new_empty(q.shape[:3], dtype=torch.float32)
    weights = q.new_empty(idx.shape, dtype=torch.float32)
    statistics_kernel(q.shape[2], q.shape[3], scale)(q, kv, sink, idx, lse, weights)
    return lse[:, :, :H].contiguous(), weights[..., :K].contiguous()


def attention_backward(q, kv, idx, scale, lse, d_out, delta):
    """(dq (B, T, H, c), dkv (B, N, c)): the gradients of the queries and
    entries, from lse (statistics), d_out (B, T, H, c) and delta (B, T, H),
    the sum over c of d_out times the output."""
    H = q.shape[2]
    q, _, idx = _padded(q, None, idx)
    extra = q.shape[2] - H
    lse = torch.nn.functional.pad(lse, (0, extra), value=float("inf"))  # padded heads: no weight
    d_out = torch.nn.functional.pad(d_out.to(torch.bfloat16), (0, 0, 0, extra))
    delta = torch.nn.functional.pad(delta, (0, extra))
    dq = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    dkv = torch.zeros(kv.shape, dtype=torch.float32, device=kv.device)
    backward_kernel(q.shape[2], q.shape[3], scale)(q, kv, idx, lse.contiguous(), d_out.contiguous(),
                                                    delta.contiguous(), dq, dkv)
    return dq[:, :, :H], dkv


def index_scores(q, q_scale, k, k_scale):
    """(B, T, N): sum_h q_scale * ReLU(q . k) * k_scale, from FP8 queries q
    (B, T, n, c) and keys k (B, N, c), q_scale (B, T, n) and k_scale (B, N)
    in FP32, by DeepSeek's FP8 indexer kernel. Heads are padded to 16 and
    keys to a multiple of 512 with zeros, which add nothing."""
    n, N = q.shape[2], k.shape[1]
    heads, keys = -n % HEADS, -N % KEYS
    q = torch.cat([q, q.new_zeros(*q.shape[:2], heads, q.shape[3])], 2) if heads else q
    q_scale = torch.nn.functional.pad(q_scale, (0, heads))
    k = torch.cat([k, k.new_zeros(k.shape[0], keys, k.shape[2])], 1) if keys else k
    k_scale = torch.nn.functional.pad(k_scale, (0, keys))
    return fp8_index(q.contiguous(), q_scale.contiguous(), k.contiguous(), k_scale.contiguous())[..., :N]


def _padded(q, sink, idx):
    """Heads padded to a multiple of HEADS (zero queries; a sink of +inf, so
    padded heads take no weight) and rows to a multiple of BLOCK entries
    (-1), the shapes WPT's kernels take."""
    B, T, H, c = q.shape
    heads, entries = -H % HEADS, -idx.shape[-1] % BLOCK
    if heads:
        q = torch.cat([q, q.new_zeros(B, T, heads, c)], 2)
        sink = None if sink is None else torch.cat([sink, sink.new_full((heads,), float("inf"))])
    if entries:
        idx = torch.nn.functional.pad(idx, (0, entries), value=-1)
    return q.contiguous(), sink, idx.contiguous()


@tilelang.jit(pass_configs=pass_configs)
def statistics_kernel(h: int, d: int, scale: float):
    """Two passes over a query's entries: the first finds each head's
    log-sum-exp as DeepSeek's kernel runs its softmax, the second adds each
    entry's weights over the heads."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_blocks = topk // BLOCK

    @T.prim_func
    def statistics_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
        lse: T.Tensor[(b, m, h), FP32],
        weights: T.Tensor[(b, m, topk), FP32],
    ):
        with T.Kernel(m, b, threads=256) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((BLOCK, d), BF16)
            idxs = T.alloc_fragment(BLOCK, INT32)
            acc_s = T.alloc_fragment((h, BLOCK), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)
            lse_frag = T.alloc_fragment(h, FP32)
            weight_sum = T.alloc_fragment(BLOCK, FP32)

            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=2):
                for i in T.Parallel(BLOCK):
                    idxs[i] = topk_idxs[by, bx, t * BLOCK + i]
                for i, j in T.Parallel(BLOCK, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] *= scale
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]

            for i in T.Parallel(h):
                lse_frag[i] = scores_max[i] + T.log(sum_exp[i] + T.exp(attn_sink[i] - scores_max[i]))
            T.copy(lse_frag, lse[by, bx, :])

            for t in T.Pipelined(num_blocks, num_stages=2):
                for i in T.Parallel(BLOCK):
                    idxs[i] = topk_idxs[by, bx, t * BLOCK + i]
                for i, j in T.Parallel(BLOCK, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, BLOCK):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - lse_frag[i])
                T.reduce_sum(acc_s, weight_sum, dim=0)
                T.copy(weight_sum, weights[by, bx, t * BLOCK])

    return statistics_kernel_


@tilelang.jit(pass_configs=pass_configs)
def backward_kernel(h: int, d: int, scale: float):
    """One query's gradients: p = exp(z - lse) again from its entries, then
    dz = p (dp - delta) scale, dq = sum_j dz_j kv_j, and each entry's
    dz^T q + p^T d_out added into dkv."""
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    num_blocks = topk // BLOCK

    @T.prim_func
    def backward_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
        lse: T.Tensor[(b, m, h), FP32],
        d_out: T.Tensor[(b, m, h, d), BF16],
        delta: T.Tensor[(b, m, h), FP32],
        dq: T.Tensor[(b, m, h, d), FP32],
        dkv: T.Tensor[(b, n, d), FP32],
    ):
        with T.Kernel(m, b, threads=256) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            do_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((BLOCK, d), BF16)
            p_cast = T.alloc_shared((h, BLOCK), BF16)
            dz_cast = T.alloc_shared((h, BLOCK), BF16)
            dkv_shared = T.alloc_shared((BLOCK, d), FP32)
            idxs = T.alloc_fragment(BLOCK, INT32)
            acc_p = T.alloc_fragment((h, BLOCK), FP32)
            acc_dp = T.alloc_fragment((h, BLOCK), FP32)
            acc_dq = T.alloc_fragment((h, d), FP32)
            acc_dkv = T.alloc_fragment((BLOCK, d), FP32)
            lse_frag = T.alloc_fragment(h, FP32)
            delta_frag = T.alloc_fragment(h, FP32)

            T.copy(q[by, bx, :, :], q_shared)
            T.copy(d_out[by, bx, :, :], do_shared)
            T.copy(lse[by, bx, :], lse_frag)
            T.copy(delta[by, bx, :], delta_frag)
            T.clear(acc_dq)

            for t in T.serial(num_blocks):
                for i in T.Parallel(BLOCK):
                    idxs[i] = topk_idxs[by, bx, t * BLOCK + i]
                for i, j in T.Parallel(BLOCK, d):
                    kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                for i, j in T.Parallel(h, BLOCK):
                    acc_p[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                T.gemm(q_shared, kv_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(h, BLOCK):
                    acc_p[i, j] = T.exp(acc_p[i, j] * scale - lse_frag[i])
                T.copy(acc_p, p_cast)
                T.gemm(do_shared, kv_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow,
                       clear_accum=True)
                for i, j in T.Parallel(h, BLOCK):
                    acc_dp[i, j] = acc_p[i, j] * (acc_dp[i, j] - delta_frag[i]) * scale
                T.copy(acc_dp, dz_cast)
                T.gemm(dz_cast, kv_shared, acc_dq, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(dz_cast, q_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow,
                       clear_accum=True)
                T.gemm(p_cast, do_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_dkv, dkv_shared)
                for i, j in T.Parallel(BLOCK, d):  # a missing entry's row is 0, added to entry 0
                    T.atomic_add(dkv[by, T.max(idxs[i], 0), j], dkv_shared[i, j])

            T.copy(acc_dq, dq[by, bx, :, :])

    return backward_kernel_
