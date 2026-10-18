"""Where a pass's tokens sit (docs/model.md, "What a layer passes on"). A
module that reads earlier tokens says once what it passes on to later
tokens (carry.py: Tail, Entries, Memory), and asks the place of its pass
for what came before its first token:

    before(module, name, x)          a Tail's steps before x's first token
    everything(module, name, new)    an Entries': every entry so far, the
                                     new ones (those this pass completes)
                                     after the earlier ones
    delta(module, q, k, v, g, beta)  KDA's output, its Memory starting from
                                     what the earlier tokens left

A place also gives the pass's positions (positions), the entries it can
complete (blocks), and whether training statistics are recorded (collect).

Four places answer, from that one description:

    Whole    the pass is the whole window: nothing came before
    Carried  one segment of a window trained on one GPU (wpt_training's
             parts.py): what the segment before passed on, as tensors whose
             gradients the runner carries back across the boundary
    Split    one process's slice of a window, or of one part of it,
             split across processes (DeepSeek-V4's contextual
             parallelism, section 3.4.3): rank i gets the tails of rank
             i - 1, every merged entry by an all-gather, and the KDA memory
             by combining every slice's transform of it; rank 0 gets what
             the part before passed on, as Carried does, and the part
             passes on what its last slice ends with
    Reading  generation (cache.py): B rows, each at its own place, reading
             what the cache keeps and keeping it up to date, with undo

Training's places are exact: a window trained in segments or split across
processes gives the loss and gradients of the whole window at once. A
segment or slice holds a multiple of the longest merging's tokens
(WPT.PART); a window's last segment may be shorter.
"""

import torch
import torch.distributed as dist

from .carry import described
from .kda import delta_attention


class Place:
    """Where a pass's tokens sit. start: the first token's position, an int
    for every row alike, or (B,) for rows at different places; latest: the
    largest start, an int."""

    start = 0
    latest = 0
    collect = True  # training statistics (expert balancing) are recorded

    def positions(self, T, device):
        """The pass's T positions: (T,), or (B, T) for rows at different places."""
        steps = torch.arange(T, device=device)
        return self.start + steps if isinstance(self.start, int) else self.start[:, None] + steps

    def blocks(self, ratio, T):
        """(the index of the first entry a merging of ratio tokens can
        complete in this pass, the most entries any row completes): an int,
        or (B,) for rows at different places."""
        first = self.start // ratio
        return first, (self.start + T) // ratio - first


class Whole(Place):
    """The pass is the whole window."""

    def before(self, module, name, x):
        return described(module, name).start(x, x.shape[0])

    def everything(self, module, name, new):
        return new

    def delta(self, module, q, k, v, g, beta):
        return delta_attention(q, k, v, g, beta)[0]


WHOLE = Whole()


class Carried(Place):
    """One segment of a window: carried holds what the segment before passed
    on (empty for the first), by key (module, name). After the pass, passed
    holds what this one passes on (each module's tails and KDA memories), and
    added the entries it added to everything (the merged entries and keys
    of all segments so far are what came before plus each one's added).
    collect: record training statistics in this pass."""

    def __init__(self, offset, carried, collect=True):
        self.start = self.latest = offset
        self.carried, self.collect = carried, collect
        self.passed, self.added = {}, {}

    def before(self, module, name, x):
        key, tail = (id(module), name), described(module, name)
        earlier = self.carried.get(key)
        if earlier is None:
            earlier = tail.start(x, x.shape[0])
        self.passed[key] = torch.cat([earlier, x], 1)[:, -tail.steps:]
        return earlier

    def everything(self, module, name, new):
        key = (id(module), name)
        described(module, name)
        self.added[key] = new
        earlier = self.carried.get(key)
        return new if earlier is None else torch.cat([earlier.to(new.dtype), new], 1)

    def delta(self, module, q, k, v, g, beta):
        key = (id(module), "memory")
        described(module, "memory")
        out, self.passed[key] = delta_attention(q, k, v, g, beta, self.carried.get(key))
        return out


class Split(Place):
    """Process rank's slice of a window split across world processes, each
    slice of `length` tokens, in order from token start of the window: the
    whole window, or one part of it when the window runs as consecutive
    parts each split across the processes (wpt_training's parts.py). carried
    holds what the part before passed on (empty for the first part or a
    whole window), by key (module, name), as for Carried; after the pass,
    passed and added hold what this part passes on, the same on every
    process. Every exchange is an all-gather whose backward pass sums each
    slice's gradient over the processes, so the processes' gradients added
    up are the whole window's. Each process builds the same graph
    (rank-dependent values, never rank-dependent operations), so their
    backward passes meet in the same order. collect: record training
    statistics in this pass."""

    def __init__(self, rank, world, length, start=0, carried=None, collect=True, group=None):
        self.rank, self.world, self.group = rank, world, group
        self.start = self.latest = start + rank * length
        self.collect = collect
        self.carried, self.passed, self.added = carried or {}, {}, {}
        self.later = torch.tensor(rank > 0)  # False on the first slice: what came before it is the part before's

    def before(self, module, name, x):
        key, tail = (id(module), name), described(module, name)
        n = tail.steps
        tails = gather(x[:, x.shape[1] - n:], self.group)  # stage 1: each rank's last n steps
        self.passed[key] = tails[self.world - 1]  # the part's last n steps: its last slice's
        earlier = self.carried.get(key)
        if earlier is None:  # the window's start: nothing came before
            earlier = tail.start(x, x.shape[0])
        return torch.where(self.later.to(x.device), tails[(self.rank - 1) % self.world], earlier.to(tails.dtype))

    def everything(self, module, name, new):
        key = (id(module), name)
        described(module, name)
        entries = gather(new, self.group).transpose(0, 1).flatten(1, 2)  # stage 2: every rank's entries, in order
        self.added[key] = entries
        earlier = self.carried.get(key)
        return entries if earlier is None else torch.cat([earlier.to(entries.dtype), entries], 1)

    def delta(self, module, q, k, v, g, beta):
        """KDA across slices: the memory's update is linear in the memory,
        S_end = A S_start + B, so each slice computes its own A and B in one
        pass (memory and values widened: starting from [I | 0] with values
        [0 | v], the memory ends as [A | B] and each output as [A_t^T q_t |
        B_t^T q_t]); the processes exchange them, each composes the slices
        before its own, after the memory the part before passed on, into its
        starting memory, and its output follows exactly. The last slice's
        composed end is what the part passes on."""
        described(module, "memory")
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        start = torch.cat([torch.eye(dk, device=q.device).expand(B, H, dk, dk),
                           q.new_zeros(B, H, dk, dv, dtype=torch.float32)], -1)
        out, transform = delta_attention(q, k, torch.cat([v.new_zeros(B, T, H, dk), v], -1), g, beta, start)
        transforms = gather(transform.float(), self.group)  # (world, B, H, dk, dk + dv)
        key = (id(module), "memory")
        memory = self.carried.get(key)
        memory = transforms.new_zeros(B, H, dk, dv) if memory is None else memory.float()
        starts = [memory]
        for j in range(self.world):  # every slice's starting memory, then the part's end; each rank takes its own
            memory = transforms[j, ..., :dk] @ memory + transforms[j, ..., dk:]
            starts.append(memory)
        self.passed[key] = starts.pop()
        memory = torch.stack(starts)[self.rank]
        out = out.float()
        return torch.einsum("bhkv,bthk->bthv", memory, out[..., :dk]) + out[..., dk:]


class _Gather(torch.autograd.Function):
    """all_gather with a backward pass that sums each slice's gradient over
    the processes (an all-reduce, which every backend has)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.stack(parts)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return grad[dist.get_rank(ctx.group)], None


def gather(x, group=None):
    """(world, *x.shape): every process's x, in order of rank."""
    if not x.is_floating_point():
        parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, x.contiguous(), group=group)
        return torch.stack(parts)
    return _Gather.apply(x, group)
