"""What a layer passes on to later tokens (docs/model.md, "What a layer
passes on"), said once per module. Every module that reads earlier tokens
has passes_on(): {name: what}, each one of

    Tail(steps, fill)  the last `steps` time steps of a value each token
                       has: KDA's and Engram's convolution inputs, the
                       window's entries, a merging's rows not merged into
                       an entry yet, Engram's canonical ids; before the
                       text, fill's value
    Entries(ratio)     entries merged from every `ratio` tokens, every one
                       kept: CSA's and HCA's merged entries, the indexer's
                       keys
    Memory()           a state updated token by token: KDA's memory

Training's places (sequence.py: a window in segments, or split across
processes) carry these from one part of a window to the next, and the
generation cache (cache.py) keeps them from one call to the next; both read
how many steps a tail holds, what comes before the text and how to take
tokens back from here.
"""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Tail:
    """The last `steps` time steps of a value each token has (B, T, ...).
    fill(zeros) gives the steps before the text from zeros of their shape;
    zeros when None."""
    steps: int
    fill: Callable | None = None

    def start(self, x, rows):
        """What comes before the text, for rows rows of values like x."""
        empty = x.new_zeros(rows, self.steps, *x.shape[2:])
        return empty if self.fill is None else self.fill(empty)

    def undo(self, before, new, kept):
        """The tail once the first kept[b] of row b's new steps follow the
        tail before them."""
        return last(torch.cat([before, new], 1), kept, self.steps)


@dataclass(frozen=True)
class Entries:
    """Entries merged from every `ratio` tokens: entry i once tokens
    i * ratio to (i + 1) * ratio - 1 are read; every one is kept. Taking
    tokens back needs nothing: an entry past a row's tokens is never read,
    and is merged again when the row reaches it."""
    ratio: int


@dataclass(frozen=True)
class Memory:
    """KDA's memory (B, H, dk, dv), zeros before the text."""

    @staticmethod
    def undo(before, inputs, kept):
        """The memory once the first kept[b] of row b's new tokens follow the
        memory before them, replayed from it (Kimi K3's way); inputs are
        their (k, v, g, beta)."""
        from .kda import delta_step  # kda.py describes its memory with this class

        k, v, g, beta = inputs
        beta, g = padded(beta, g, kept)
        return delta_step(k, k, v, g, beta, before)[1]  # the output is not needed: k stands in for q


def described(module, name):
    """What module's passes_on() says it passes on under name."""
    found = module.passes_on().get(name)
    if found is None:
        raise KeyError(f"{type(module).__name__} passes on no {name!r}")
    return found


def padded(beta, g, counts):
    """KDA's beta and g with the tokens past counts[b] of row b made padding:
    no write (beta 0) and no decay (g 0), so they leave the memory as it was."""
    valid = torch.arange(beta.shape[1], device=beta.device)[None] < counts[:, None]
    return beta * valid[..., None], g * valid[..., None, None]


def last(rows, counts, size):
    """rows (B, size + n, ...) -> (B, size, ...): for each row b, the `size`
    rows that end where its first counts[b] rows after the first `size` end."""
    index = counts[:, None] + torch.arange(size, device=rows.device)
    return rows[torch.arange(len(rows), device=rows.device)[:, None], index]
