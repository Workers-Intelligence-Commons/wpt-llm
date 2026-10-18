"""The generation cache (docs/model.md, "Generation"): what a model keeps of
the tokens it has read, so that the next ones cost only themselves, for B
rows (attempts) at once. WPT.new_cache makes one, WPT.extend reads new
tokens through it, WPT.undo takes back the last tokens of an undoable
extend (rejected draft tokens), and Cache.select copies rows (a prompt read
once, shared by several attempts).

The cache keeps what each layer says it passes on to later tokens
(carry.py), the same description training's segments and splits carry
(sequence.py):

    KDA          its memory S and the last 3 inputs of each short convolution
    CSA, HCA     DeepSeek-V4's inference layout: the window's last 127
                 entries (V4 keeps 128, the new token's included), the
                 merged entries (FP8 but their rotary dimensions), CSA's
                 indexer keys (FP4, held in FP8), and each merging's rows
                 not merged into an entry yet
    Engram       the last order - 1 canonical ids and each stream's last
                 inputs of its convolution; the tables stay where they are
    MTP block    its window, like CSA's and HCA's, for the pairs it read

Nothing grows with the text but the merged entries, a quarter of the tokens
in CSA (and as many indexer keys) and 1/128 in HCA, so generation goes on
past block_size. The experts keep nothing, but a cache made with routes
records each row's routed experts (R3) as its tokens are read.

New tokens are read by the model's own forward pass, through a Reading:
the place (sequence.py) of B rows each at its own length, which gives each
module what the cache keeps and keeps what the pass leaves. An extend may
give each row its own number of new tokens (counts), the rest of its row
being padding: a tail keeps the last steps of a row's own tokens, KDA's
padding writes nothing, and only the blocks a row's own tokens complete
are merged. Undoing goes back to what an undoable reading started from and
keeps the first kept[b] tokens of row b, as each kind of carried value
says (carry.py): tails cut, the memory replayed, entries left (one past a
row's tokens is never read, and is merged again when the row reaches it).
"""

import torch

from .carry import described, padded
from .experts import Routes
from .kda import delta_step
from .sequence import Place


class Rows:
    """One stream of B rows' carried values: the text the model reads, or
    the (hidden state, next token) pairs its multi-token prediction block
    reads. lengths: the tokens each row has read; values: what each module
    passes on, by (module, name), made at the first reading; pending: what
    undo needs after an undoable reading, None otherwise."""

    def __init__(self, rows, device):
        self.lengths = torch.zeros(rows, dtype=torch.long, device=device)
        self.values = {}
        self.pending = None

    def reading(self, counts, undoable):
        return Reading(self, counts, undoable)

    def undo(self, kept):
        """Back to the state before the last undoable reading, then its first
        kept[b] tokens of row b again."""
        if self.pending is None:
            raise ValueError("no undoable extend to undo")
        lengths, saved = self.pending
        for key, (item, before, new) in saved.items():
            self.values[key] = item.undo(before, new, kept)
        self.lengths, self.pending = lengths + kept, None


class Reading(Place):
    """The place of new tokens read through rows (Rows): row b's first
    counts[b] tokens follow its lengths[b] tokens, the rest are padding.
    The model's forward pass reads through it; finish then records the
    tokens read. With undoable, it keeps what Rows.undo needs."""

    collect = True  # a cache made with routes records them (balancing counts only in training mode)

    def __init__(self, rows, counts, undoable):
        self.rows, self.counts = rows, counts
        self.start = rows.lengths
        self.latest = int(rows.lengths.max()) if len(rows.lengths) else 0
        self.saved = {} if undoable else None

    def blocks(self, ratio, T):
        first = self.start // ratio
        return first, int(((self.start + self.counts) // ratio - first).max()) if len(first) else 0

    def before(self, module, name, x):
        key, tail = (id(module), name), described(module, name)
        earlier = self.rows.values.get(key)
        earlier = tail.start(x, x.shape[0]) if earlier is None else earlier.to(x.dtype)
        self.rows.values[key] = tail.undo(earlier, x, self.counts)  # the last steps of each row's own tokens
        if self.saved is not None:
            self.saved[key] = (tail, earlier, x)
        return earlier

    def everything(self, module, name, new):
        """Writes the entries each row's own tokens complete after its
        earlier ones; returns every row's entries so far (B, N, ...), N the
        most any row has, a row's entries past its own number stale."""
        key, ratio = (id(module), name), described(module, name).ratio
        first = self.start // ratio
        made = (self.start + self.counts) // ratio - first
        need = int((first + made).max()) if len(first) else 0
        held = self.rows.values.get(key)
        if held is None:
            held = new.new_zeros(len(first), 0, *new.shape[2:])
        if need > held.shape[1]:
            grown = max(need, 2 * held.shape[1])
            held = torch.cat([held, held.new_zeros(len(first), grown - held.shape[1], *held.shape[2:])], 1)
        row, j = (torch.arange(new.shape[1], device=new.device)[None] < made[:, None]).nonzero(as_tuple=True)
        held = held.to(new.dtype)
        held[row, first[row] + j] = new[row, j]
        self.rows.values[key] = held
        return held[:, :need]

    def delta(self, module, q, k, v, g, beta):
        key, memory = (id(module), "memory"), described(module, "memory")
        beta, g = padded(beta, g, self.counts)
        before = self.rows.values.get(key)
        if before is None:
            before = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32)
        out, self.rows.values[key] = delta_step(q, k, v, g, beta, before)
        if self.saved is not None:
            self.saved[key] = (memory, before, (k, v, g, beta))
        return out

    def finish(self):
        """The rows have read their tokens."""
        rows = self.rows
        rows.pending = (rows.lengths, self.saved) if self.saved is not None else None
        rows.lengths = rows.lengths + self.counts


class Cache:
    """B rows' cache of one model (WPT.new_cache): text, what its layers
    keep of the tokens each row has read; mtp, what its multi-token
    prediction block keeps of the pairs it has read; routes: each row's
    Routes, when the cache records them."""

    def __init__(self, model, rows, routes=False):
        device = model.embedding.weight.device
        self.text, self.mtp = Rows(rows, device), Rows(rows, device)
        self.routes = [Routes() for _ in range(rows)] if routes else None

    @property
    def lengths(self):
        """The tokens each row has read."""
        return self.text.lengths

    @property
    def rows(self):
        return len(self.text.lengths)

    def select(self, index):
        """A new cache of the rows index names (a row may come several times),
        copied: the copies change apart from each other and from this one."""
        if self.text.pending is not None or self.mtp.pending is not None:
            raise ValueError("an undoable extend is pending: undo it before copying rows")
        index = torch.as_tensor(index, dtype=torch.long, device=self.lengths.device)
        copy = object.__new__(Cache)
        copy.text, copy.mtp = (_selected(rows, index) for rows in (self.text, self.mtp))
        copy.routes = None if self.routes is None else [self.routes[i].copy() for i in index.tolist()]
        return copy


def _selected(rows, index):
    copy = Rows(len(index), rows.lengths.device)
    copy.lengths = rows.lengths[index]
    copy.values = {key: value[index] for key, value in rows.values.items()}
    return copy


class Recorder:
    """Records the routed experts of each row's real tokens (not its
    padding) into its own Routes; Experts.route calls record."""

    def __init__(self, routes, counts):
        self.routes, self.counts = routes, counts

    def record(self, experts, chosen):
        chosen = chosen.view(len(self.routes), -1, chosen.shape[-1])
        for routes, row, count in zip(self.routes, chosen, self.counts):
            routes.append(experts, row[:count])

