"""Where a pass's tokens sit in a longer window (docs/model.md, "Long
windows"). A layer that reads earlier tokens asks its pass's place for what
came before its first token, in three ways:

    before(module, name, x, n)   the n time steps of x before this pass
                                 (zeros before the window's first token)
    everything(module, name, x)  x's entries after every earlier pass's
                                 (the merged entries of CSA and HCA, the
                                 indexer's keys), indexed from the window's
                                 first token
    delta(module, q, k, v, g, beta)
                                 KDA's output, its memory starting from what
                                 the earlier tokens left

Three places answer them:

    Whole    the pass is the whole window: nothing came before
    Carried  one segment of a window trained on one GPU (wpt_training's
             runner): what the segment before passed on, as tensors whose
             gradients the runner carries back across the boundary
    Split    one process's slice of a window, or of one part of it,
             split across processes (DeepSeek-V4's contextual
             parallelism, section 3.4.3): rank i gets the tails of rank
             i - 1, every merged entry by an all-gather, and the KDA memory
             by combining every slice's transform of it; rank 0 gets what
             the part before passed on, as Carried does, and the part
             passes on what its last slice ends with

Every answer is exact: a window trained in segments or split across
processes gives the loss and gradients of the whole window at once. A
segment or slice holds a multiple of the longest merging's tokens
(WPT.PART), so no merged entry straddles two of them; a window's last
segment may be shorter.
"""

import torch
import torch.distributed as dist

from .kda import delta_attention


class Whole:
    """The pass is the whole window."""

    offset = 0       # the pass's first token's place in the window
    collect = True   # training statistics (expert balancing) are recorded

    def before(self, module, name, x, n):
        return x.new_zeros(x.shape[0], n, *x.shape[2:])

    def everything(self, module, name, x):
        return x

    def delta(self, module, q, k, v, g, beta):
        return delta_attention(q, k, v, g, beta)[0]


WHOLE = Whole()


class Carried:
    """One segment of a window: carried holds what the segment before passed
    on (empty for the first), by key (module, name). After the pass, passed
    holds what this one passes on (each module's tails and KDA memories), and
    added the entries it added to everything (the merged entries and keys
    of all segments so far are what came before plus each one's added).
    collect: record training statistics in this pass."""

    def __init__(self, offset, carried, collect=True):
        self.offset, self.carried, self.collect = offset, carried, collect
        self.passed, self.added = {}, {}

    def before(self, module, name, x, n):
        key = (id(module), name)
        earlier = self.carried.get(key)
        if earlier is None:
            earlier = x.new_zeros(x.shape[0], n, *x.shape[2:])
        self.passed[key] = torch.cat([earlier, x], 1)[:, -n:]
        return earlier

    def everything(self, module, name, x):
        key = (id(module), name)
        self.added[key] = x
        earlier = self.carried.get(key)
        return x if earlier is None else torch.cat([earlier.to(x.dtype), x], 1)

    def delta(self, module, q, k, v, g, beta):
        key = (id(module), "memory")
        out, self.passed[key] = delta_attention(q, k, v, g, beta, self.carried.get(key))
        return out


class Split:
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
        self.offset, self.collect = start + rank * length, collect
        self.carried, self.passed, self.added = carried or {}, {}, {}
        self.first = torch.tensor(float(rank > 0))  # 0 on the first slice: what came before it is the part before's

    def before(self, module, name, x, n):
        key = (id(module), name)
        tails = gather(x[:, x.shape[1] - n:], self.group)  # stage 1: each rank's last n steps
        self.passed[key] = tails[self.world - 1]  # the part's last n steps: its last slice's
        earlier = self.carried.get(key)
        if earlier is None:  # the window's start: nothing came before
            earlier = x.new_zeros(x.shape[0], n, *x.shape[2:])
        first = self.first.to(x.device, tails.dtype)
        return tails[(self.rank - 1) % self.world] * first + earlier * (1 - first)

    def everything(self, module, name, x):
        key = (id(module), name)
        entries = gather(x, self.group).transpose(0, 1).flatten(1, 2)  # stage 2: every rank's entries, in order
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
