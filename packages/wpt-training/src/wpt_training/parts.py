"""A window trained in parts (docs/training.md, "Long windows"), with the
loss and gradients of the whole window at once.

Segments, on one process: a window longer than the segment length runs as
consecutive segments. A first pass without gradients runs every segment
but the last, each handing the next what it passes on (the model's
sequence.Carried: KDA memories, convolution and window tails, merged
entries); then the segments run again from the last to the first, each
with gradients, starting from what the one before passed on, and its
backward pass starts from its loss plus the gradient that the later
segments sent back to what it passed on. So memory holds one segment's
activations at a time, and each segment except the last runs twice.

Slices, on several processes (a strategy whose splits_windows is true):
every process takes the same windows and process r trains its slice r of
each, the model exchanging what crosses the slices (sequence.Split); the
gradients the processes compute add up to the whole window's.

Both at once, when a slice is longer than the segment length: the window
runs as consecutive parts of world x segment tokens (the last may be
shorter), each split across the processes, so process r trains the r-th
segment of every part, and the parts hand on what they pass on as
segments do.

The model gives each part its place (model.carried, model.split), and the
method its loss (Method.part_loss): the part's share of the whole window's
loss. Validation losses are the parts' validation losses weighted by their
share of the window's tokens.
"""

from functools import partial

import torch

from .plugins import Method, PluginError


def check(model, method, stages, segment, strategy):
    """ValueError or PluginError when the stages cannot run in parts as asked;
    nothing when no stage needs parts."""
    splits = strategy.splits_windows
    if not splits and not any(segment and stage.block_size > segment for stage in stages):
        return
    if not hasattr(model, "carried") or not hasattr(model, "split"):
        raise PluginError("the model cannot train a window in parts (it has no carried and split)")
    if type(method).loss is not Method.loss and type(method).part_loss is Method.part_loss:
        raise PluginError("the method defines its own loss but not part_loss, so it cannot train a window in parts")
    part = getattr(model, "PART", 1)
    if segment % part:
        raise ValueError(f"--segment {segment} is not a multiple of {part} tokens, the longest merging")
    for stage in stages:
        if splits and stage.block_size % (strategy.world_size * part):
            raise ValueError(f"a window of {stage.block_size} tokens does not split into {strategy.world_size} slices "
                             f"of a multiple of {part} tokens")


def _pieces(model, T, strategy, segment):
    """This process's parts of a window of T tokens, in order, as (first
    token, end, place(carried, collect)); None when it runs whole."""
    if strategy.splits_windows:
        world, rank = strategy.world_size, strategy.rank
        chunk = world * segment if segment and T > world * segment else T  # tokens split across the processes at once
        pieces = []
        for start in range(0, T, chunk):
            length = min(chunk, T - start) // world
            a = start + rank * length
            pieces.append((a, a + length, partial(model.split, rank, world, length, start)))
        return pieces
    if not segment or T <= segment:
        return None
    return [(a, min(a + segment, T), partial(model.carried, a)) for a in range(0, T, segment)]


def learn(method, model, wrapped, x, y, scale, strategy, segment, quiet):
    """Forward and backward passes of windows x, y (B, T) with the loss
    times scale; returns (this process's loss, its indexer loss), detached:
    the whole window's, or with split windows, this process's share. quiet(last)
    is the context of a backward pass, last for the last of the
    micro-batch."""
    pieces = _pieces(model, x.shape[1], strategy, segment)
    if pieces is None:
        with quiet(True):
            loss, indexer = method.loss(wrapped, x, y)
            (loss * scale).backward()
        return loss.detach(), indexer
    return _in_pieces(method, wrapped, x, y, scale, pieces, quiet)


def _following(y, end):
    """The token after y[:, :end] in the window, or None at its end."""
    return y[:, end:end + 1] if end < y.shape[1] else None


def _handed_on(carried, place):
    """What the parts so far hand the next one: the last part's passed, and
    everything's entries of every part so far."""
    return {**place.passed, **{key: torch.cat([carried[key], new], 1) if key in carried else new
                               for key, new in place.added.items()}}


def _in_pieces(method, wrapped, x, y, scale, pieces, quiet):
    """A first pass without gradients runs every part but the last, keeping
    what each passes on; then the parts run again from the last to the
    first, with gradients (see the module's docstring)."""
    T = x.shape[1]

    def run(i, carried, collect):
        a, b, place_of = pieces[i]
        place = place_of(carried, collect)
        loss, indexer = method.part_loss(partial(wrapped, sequence=place), x[:, a:b], y[:, a:b], _following(y, b), T)
        return place, loss, indexer

    passed, added, carried = [], [], {}
    with torch.no_grad(), quiet(False):  # what each part passes on; balancing counts them here
        for i in range(len(pieces) - 1):
            place, _, _ = run(i, carried, True)
            passed.append(place.passed)
            added.append(place.added)
            carried = _handed_on(carried, place)
    del carried
    grown = set(added[0]) if added else set()
    grads, total, indexer_total = {}, 0.0, 0.0
    for i in reversed(range(len(pieces))):
        leaves = {}
        if i:
            leaves = {key: _leaf(t) for key, t in passed[i - 1].items()}
            leaves.update({key: _leaf(torch.cat([added[j][key] for j in range(i)], 1)) for key in grown})
        with quiet(i == 0):
            place, loss, indexer = run(i, leaves, i == len(pieces) - 1)
            roots, given = [loss * scale], [torch.ones_like(loss)]
            for key, t in [*place.passed.items(), *place.added.items()]:
                if key in grads and t.requires_grad:  # what later parts sent back
                    roots.append(t)
                    given.append(grads[key][:, grads[key].shape[1] - t.shape[1]:])
            torch.autograd.backward(roots, given)
        sent = {}
        for key, leaf in leaves.items():
            if not leaf.requires_grad:
                continue
            g = leaf.grad if leaf.grad is not None else torch.zeros_like(leaf)
            if key in grown and key in grads:  # the earlier entries' share of the later parts' gradient
                g = g + grads[key][:, :leaf.shape[1]]
            sent[key] = g
        grads = sent
        total = total + loss.detach()
        indexer_total = indexer_total + indexer
    return total, indexer_total


def _leaf(t):
    return t.detach().requires_grad_(t.is_floating_point())


@torch.no_grad()
def validation_loss(method, model, wrapped, x, y, strategy, segment):
    """The validation loss of windows x, y: this process's share with split
    windows (the processes' shares add up to it)."""
    T = x.shape[1]
    pieces = _pieces(model, T, strategy, segment)
    if pieces is None:
        return method.validation_loss(wrapped, x, y)
    total, carried = 0.0, {}
    for a, b, place_of in pieces:
        place = place_of(carried, False)
        total = total + method.validation_loss(partial(wrapped, sequence=place), x[:, a:b], y[:, a:b]) * ((b - a) / T)
        carried = _handed_on(carried, place)
    return total
