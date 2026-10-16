"""Engram memory (docs/model.md, "Engram memory"): embeddings of the last
tokens' N-grams, read from hashed tables, mixed into each residual stream
through a learned gate.

The hash is multiplicative-XOR, as DeepSeek describes it; DeepSeek does not
publish its constants, so HASH_MULTIPLIERS are WPT's. Each (order, head)
pair has its own table, of a distinct prime size.
"""

import math
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CONV_KERNEL, EPS, ShortConv

# odd constants below 2^32, one per hash head (WPT's own): with table sizes
# below 2^31, every product stays inside 64-bit integers
HASH_MULTIPLIERS = (2654435761, 2246822519, 3266489917, 668265263, 374761393, 2870177453, 3624381961, 1597334677,
                    2034339539, 3337309187, 1181783497, 4022958977, 2479686631, 1013904223, 1664525143, 2891336453)


def canonical_ids(decode, vocab_size) -> torch.Tensor:
    """Each token id's canonical id: tokens whose decoded text is the same
    after NFKC normalization and lowercasing share the smallest of their
    ids; a token that is not whole UTF-8 text, or is empty, keeps its own.
    decode: list of ids -> bytes."""
    first, ids = {}, []
    for token in range(vocab_size):
        try:
            text = decode([token]).decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        ids.append(first.setdefault(unicodedata.normalize("NFKC", text).lower(), token) if text else token)
    return torch.tensor(ids, dtype=torch.long)


def primes_from(start, count) -> list[int]:
    """The count smallest primes at or above start."""
    found, n = [], max(start, 2)
    while len(found) < count:
        if all(n % p for p in range(2, math.isqrt(n) + 1)):
            found.append(n)
        n += 1
    return found


class Engram(nn.Module):
    def __init__(self, config, streams):
        super().__init__()
        D, self.order, heads = config.d_model, config.engram_order, config.engram_heads
        if heads > len(HASH_MULTIPLIERS):
            raise ValueError(f"model config: engram_heads is at most {len(HASH_MULTIPLIERS)}")
        self.pad = config.vocab_size  # the canonical id of the tokens before the window
        self.sizes = primes_from(config.engram_table_size, (self.order - 1) * heads)
        self.tables = nn.ModuleList(nn.Embedding(size, config.engram_dim) for size in self.sizes)
        width = len(self.sizes) * config.engram_dim
        self.value = nn.Linear(width, D, bias=False)                                           # W_V, shared
        self.keys = nn.ModuleList(nn.Linear(width, D, bias=False) for _ in range(streams))     # W_K per stream
        self.hidden_norm, self.key_norm, self.value_norm = (nn.RMSNorm(D, eps=EPS) for _ in range(3))
        self.conv = ShortConv(D, kernel=CONV_KERNEL, dilation=self.order)
        nn.init.zeros_(self.conv.conv.weight)  # the module starts as nothing

    def lookup(self, canonical):
        """(B, T) canonical ids -> e_t (B, T, tables * row width)."""
        T, rows, table = canonical.shape[1], [], 0
        for n in range(2, self.order + 1):
            gram = [F.pad(canonical, (back, 0), value=self.pad)[:, :T] for back in range(n - 1, -1, -1)]
            for head in range(len(self.sizes) // (self.order - 1)):
                size, index = self.sizes[table], torch.zeros_like(canonical)
                for ids in gram:
                    index = ((index * HASH_MULTIPLIERS[head]) ^ ids) % size
                rows.append(self.tables[table](index))
                table += 1
        return torch.cat(rows, -1)

    def forward(self, streams, canonical):
        """streams (B, T, n, D) -> streams with each one's Y added."""
        e = self.lookup(canonical)
        value = self.value(e)
        scale = streams.shape[-1] ** -0.5
        added = []
        for s, key in enumerate(self.keys):
            a = torch.sigmoid((self.hidden_norm(streams[:, :, s]) * self.key_norm(key(e))).sum(-1, keepdim=True) * scale)
            V = a * value
            added.append(F.silu(self.conv(self.value_norm(V))) + V)
        return streams + torch.stack(added, 2)
