"""DeepSeek-V4's attention (docs/model.md, "Attention"): queries from a
low-rank latent, one key-value head shared by every query head, the last
128 tokens uncompressed (the sliding window), a learned sink per head,
rotary position encoding on the last 64 dimensions, and a grouped output
projection. The ratio sets the kind:

    CSA_RATIO (4)     every 4 tokens merged into one entry, each from 8
                      overlapping tokens; an indexer picks the top k
                      entries for each query
    HCA_RATIO (128)   every 128 tokens merged into one entry, all attended
    0                 the window alone (the multi-token prediction block)

A query sees a merged entry only once every token in it is at or before
the query, so no query sees a later token.

Entries are kept as V4 keeps them (formats.py): the window's and the merged
entries in FP8 except their 64 rotary dimensions; the indexer's queries and
keys rotated by a Hadamard matrix, then in FP4, held in FP8.

When its caller asks, a CSA layer also returns its indexer's loss
(sparse.py, index_alignment), computed from inputs cut off from the rest of
the model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .formats import fp4, fp8, hadamard
from .layers import EPS
from .sparse import index_alignment, sparse_attention, top_entries

WINDOW = 128              # uncompressed tokens each query sees, itself included
ROPE_DIMS = 64            # rotary dimensions at the end of every query, key-value entry and output
CSA_RATIO, HCA_RATIO = 4, 128
ROPE_BASE = 160000        # layers with merged entries (V4's compress_rope_theta), with YaRN
WINDOW_ROPE_BASE = 10000  # the window alone (V4's rope_theta), without YaRN
# YaRN in the layers with merged entries, V4's values: original_seq_len, rope_factor, beta_fast, beta_slow
YARN_LENGTH, YARN_FACTOR, YARN_BETA_FAST, YARN_BETA_SLOW = 65536, 16, 32, 1


def frequencies(merged, device=None):
    """The ROPE_DIMS / 2 rotary frequencies of a layer. With merged entries
    (CSA, HCA): base ROPE_BASE stretched by YaRN as DeepSeek-V4's
    precompute_freqs_cis does: the pairs that turn more than YARN_BETA_FAST
    times over YARN_LENGTH tokens keep their frequency, those that turn
    fewer than YARN_BETA_SLOW times are slowed YARN_FACTOR times, and a
    linear ramp joins the two. The window alone: base WINDOW_ROPE_BASE,
    unchanged."""
    half = ROPE_DIMS // 2
    base = ROPE_BASE if merged else WINDOW_ROPE_BASE
    pairs = torch.arange(half, dtype=torch.float32, device=device)
    plain = 1.0 / base ** (2 * pairs / ROPE_DIMS)
    if not merged:
        return plain

    def pair(turns):  # the (fractional) pair that turns `turns` times over YARN_LENGTH tokens
        return ROPE_DIMS * math.log(YARN_LENGTH / (turns * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(pair(YARN_BETA_FAST)), 0)
    high = min(math.ceil(pair(YARN_BETA_SLOW)), ROPE_DIMS - 1)
    kept = 1 - ((pairs - low) / max(high - low, 0.001)).clamp(0, 1)
    return plain / YARN_FACTOR * (1 - kept) + plain * kept


def rotate(x, positions, frequencies, inverse=False):
    """Rotary position encoding of the last ROPE_DIMS dimensions of x (B, T,
    d) or (B, T, H, d), in adjacent pairs, at positions (T,), turning at
    frequencies (ROPE_DIMS / 2,); inverse turns them back by the same
    angles."""
    angles = positions.float()[:, None] * frequencies
    cos, sin = angles.cos(), angles.sin() * (-1 if inverse else 1)
    if x.ndim == 4:
        cos, sin = cos[:, None], sin[:, None]
    rest, pairs = x[..., :-ROPE_DIMS], x[..., -ROPE_DIMS:].float()
    even, odd = pairs[..., 0::2], pairs[..., 1::2]
    turned = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
    return torch.cat([rest, turned.to(x.dtype)], -1)


def stored(entries):
    """Key-value entries as V4 keeps them: FP8 except the rotary dimensions."""
    return torch.cat([fp8(entries[..., :-ROPE_DIMS]), entries[..., -ROPE_DIMS:]], -1)


def indexed(x):
    """The indexer's queries and keys as V4 keeps them: rotated by a Hadamard
    matrix, which leaves their dot products as they were but spreads each
    vector over every dimension, then FP4; then held in FP8 with one scale
    per vector, the form DeepSeek's FP8 indexer kernel reads. That last step
    loses nothing unless the vector's blocks of 32 differ in scale by more
    than 2^13 (V4's FP4-to-FP8 threshold), so kernel and PyTorch read the
    same values."""
    return fp8(fp4(hadamard(x)), block=x.shape[-1])


class Compressor(nn.Module):
    """Merges every `ratio` tokens into one entry (V4 eq. 9-12 and 20-23):
    per channel, the tokens' values C weighted by a softmax of their scores
    Z plus a learned bias per place in the block. With overlap (CSA), an
    entry also takes the block before it through a second pair of
    projections, 2 * ratio tokens in one softmax. (B, T, D) -> (B, T //
    ratio, width), RMS-normalized, rotated at each block's first token.
    Tokens after the last full block make no entry yet. In FP32, as V4."""

    def __init__(self, width_in, width, ratio, overlap):
        super().__init__()
        self.ratio, self.width, self.parts = ratio, width, 2 if overlap else 1
        self.value = nn.Linear(width_in, self.parts * width, bias=False)  # [C^b | C^a] with overlap, else C
        self.score = nn.Linear(width_in, self.parts * width, bias=False)  # [Z^b | Z^a], else Z
        self.position = nn.Parameter(torch.zeros(ratio, self.parts * width))  # [B^b | B^a], else B; starting at 0
        self.norm = nn.RMSNorm(width, eps=EPS)

    def forward(self, x, frequencies):
        B, T, _ = x.shape
        m, N = self.ratio, T // self.ratio
        with torch.autocast(x.device.type, enabled=False):
            x = x[:, :N * m].float()
            value = self.value(x).view(B, N, m, self.parts, self.width)
            score = (self.score(x).view(B, N, m, self.parts * self.width) + self.position).view(value.shape)
            if self.parts == 2:  # entry i: block i through C^a, block i - 1 through C^b (none before the first)
                value = torch.cat([_previous(value[..., 0, :], 0.0), value[..., 1, :]], 2)
                score = torch.cat([_previous(score[..., 0, :], float("-inf")), score[..., 1, :]], 2)
            else:
                value, score = value[..., 0, :], score[..., 0, :]
            merged = (score.softmax(2) * value).sum(2)
            return rotate(self.norm(merged), torch.arange(N, device=x.device) * m, frequencies)


def _previous(blocks, fill):
    """(B, N, m, c) -> each block's previous block, fill before the first."""
    return torch.cat([torch.full_like(blocks[:, :1], fill), blocks[:, :-1]], 1)


class Indexer(nn.Module):
    """CSA's lightning indexer (V4 eq. 13-17): its own merged keys, and
    queries from the attention's query latent, with per-head weights from
    the token. Gives the top k entries for each query."""

    def __init__(self, config):
        super().__init__()
        n, c = config.index_heads, config.index_head_dim
        self.heads, self.head_dim, self.top_k = n, c, config.index_top_k
        self.q = nn.Linear(config.query_rank, n * c, bias=False)  # W^IUQ
        self.weights = nn.Linear(config.d_model, n, bias=False)    # W^w
        self.keys = Compressor(config.d_model, c, CSA_RATIO, overlap=True)

    def forward(self, x, latent, positions, frequencies):
        """(queries, head weights, merged keys) for the scores."""
        B, T, _ = x.shape
        q = indexed(rotate(self.q(latent).view(B, T, self.heads, self.head_dim), positions, frequencies))
        w = self.weights(x) * (self.head_dim * self.heads) ** -0.5  # V4's scale of the weights
        return q, w, indexed(self.keys(x, frequencies))


class Attention(nn.Module):
    """One attention sublayer: CSA (ratio 4), HCA (ratio 128) or the window
    alone (ratio 0)."""

    def __init__(self, config, ratio):
        super().__init__()
        D, H, c = config.d_model, config.attention_heads, config.attention_head_dim
        g, r = config.output_groups, config.output_rank
        self.ratio, self.heads, self.head_dim, self.groups = ratio, H, c, g
        self.q_down = nn.Linear(D, config.query_rank, bias=False)  # W^DQ
        self.q_norm = nn.RMSNorm(config.query_rank, eps=EPS)
        self.q_up = nn.Linear(config.query_rank, H * c, bias=False)  # W^UQ
        self.kv = nn.Linear(D, c, bias=False)  # the window's entries
        self.kv_norm = nn.RMSNorm(c, eps=EPS)
        self.sink = nn.Parameter(torch.zeros(H))  # one sink logit per head, starting at 0
        self.out_down = nn.Linear(H * c // g, g * r, bias=False)  # one (r x H c / g) matrix per group
        self.out_up = nn.Linear(g * r, D, bias=False)
        self.compressor = Compressor(D, c, ratio, overlap=ratio == CSA_RATIO) if ratio else None
        self.indexer = Indexer(config) if ratio == CSA_RATIO else None

    def forward(self, x, indexer_loss=False):
        """(B, T, D) -> (output, the indexer's loss), the loss only from a
        CSA layer asked for it (indexer_loss), None otherwise."""
        B, T, _ = x.shape
        H, c = self.heads, self.head_dim
        positions = torch.arange(T, device=x.device)
        turning = frequencies(self.ratio > 0, x.device)
        latent = self.q_norm(self.q_down(x))
        q = F.rms_norm(self.q_up(latent).view(B, T, H, c), (c,), eps=EPS)  # V4 normalizes each head, unweighted
        q = rotate(q, positions, turning)
        table = stored(rotate(self.kv_norm(self.kv(x)), positions, turning))
        idx = _window(T, x.device).expand(B, -1, -1)
        if self.ratio:
            merged = stored(self.compressor(x, turning))
            visible = (positions + 1) // self.ratio  # entries complete at or before each query
            if self.indexer is not None:
                index = self.indexer(x.detach(), latent.detach(), positions, turning)  # its own loss alone trains it
                chosen = top_entries(*index, visible, self.indexer.top_k)
            else:
                every = torch.arange(merged.shape[1], device=x.device)
                chosen = every.masked_fill(every >= visible[:, None], -1).expand(B, -1, -1)
            idx = torch.cat([idx, torch.where(chosen >= 0, chosen + T, -1)], -1)
            table = torch.cat([table, merged.to(table.dtype)], 1)
        asked = indexer_loss and self.indexer is not None
        o, weights = sparse_attention(q, table, self.sink, idx, c ** -0.5, weights=asked)
        alignment = None
        if asked:
            on_chosen = weights[..., idx.shape[-1] - chosen.shape[-1]:]
            target = on_chosen / on_chosen.sum(-1, keepdim=True).clamp(min=1e-30)
            alignment = index_alignment(*index, chosen, target)
        o = rotate(o, positions, turning, inverse=True).to(x.dtype)  # each entry's share then depends on distance
        groups = o.reshape(B, T, self.groups, -1)
        o = torch.einsum("btgi,gri->btgr", groups, self.out_down.weight.view(self.groups, -1, groups.shape[-1]))
        return self.out_up(o.flatten(2)), alignment


def _window(T, device):
    """(1, T, min(T, WINDOW)): the last WINDOW tokens up to each query, in
    order, -1 for positions before the text (at the end of the row, as
    DeepSeek's kernel needs a valid entry first)."""
    width = min(T, WINDOW)
    t = torch.arange(T, device=device)[:, None]
    rows = (t - width + 1).clamp(min=0) + torch.arange(width, device=device)
    return rows.masked_fill(rows > t, -1)[None]
