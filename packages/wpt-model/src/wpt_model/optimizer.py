"""The update rule of WPT's weights (docs/model.md, "Training"): Muon for
matrix weights, AdamW for the rest, Adam for Engram's tables. The model
and its drafter each name their parameters' roles (optimizer_roles, by
the one rule of roles()); training wraps MuonAdamW with a schedule and
its stopping (wpt_training's `by-part` optimizer, wpt_inference.drafting).

Muon as DeepSeek-V4 runs it: the momentum of each matrix is orthogonalized
by hybrid Newton-Schulz steps (8 that bring the singular values near 1, then
2 that settle them at 1), then scaled by 0.18 sqrt(max(A, B)) for an A x B
matrix, so the update's RMS is 0.18, with decoupled weight decay:
W -= lr (0.18 sqrt(max(A, B)) O + wd W). Query, key and value projections
are orthogonalized head by head, routed experts one by one and the other
logically separate matrices (an output group, a merging's projection pair)
apart. No QK-Clip: CSA and HCA normalize their queries and key-value
entries, which keeps their scores bounded (DeepSeek-V4).
"""

import math

import torch
import torch.nn as nn

from .attention import Attention, Compressor
from .engram import Engram
from .kda import KDA
from .mhc import Connection

NEWTON_SCHULZ = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2  # DeepSeek-V4's hybrid steps
UPDATE_RMS = 0.18  # DeepSeek-V4's gamma


def orthogonalize(G):
    """(matrices, A, B) -> each matrix's nearest semi-orthogonal matrix, approximately."""
    X = G.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in NEWTON_SCHULZ:
        A = X @ X.transpose(-1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    return X.transpose(-1, -2) if tall else X


def roles(module: nn.Module, vectors=()):
    """(name, parameter, role, view) for every parameter of module. Role
    "matrix" (Muon): view is the (matrices, rows, columns) shape it is
    orthogonalized as, one matrix per head for the query, key and value
    projections of KDA and of attention and its indexer, one per output
    group, one per merging projection, one per routed expert, one
    elsewhere. Role "vector" (AdamW): the parameters given in vectors,
    those of fewer than 2 dimensions (norms, biases, sinks, gating
    factors), mHC's biases and gating factors, the convolutions and the
    merging's positional biases. Role "table": Engram's tables. view is
    None for the last two."""
    heads, chosen, tables = {}, {id(p) for p in vectors}, set()
    for part in module.modules():
        if isinstance(part, KDA):
            heads.update({id(part.q.weight): part.heads, id(part.k.weight): part.heads, id(part.v.weight): part.heads})
        elif isinstance(part, Attention):  # its one key-value head is one matrix
            heads.update({id(part.q_up.weight): part.heads, id(part.out_down.weight): part.groups})
            if part.indexer is not None:
                heads[id(part.indexer.q.weight)] = part.indexer.heads
        elif isinstance(part, Compressor):
            heads.update({id(part.value.weight): part.parts, id(part.score.weight): part.parts})
            chosen.add(id(part.position))
        elif isinstance(part, nn.Conv1d):
            chosen.add(id(part.weight))
        elif isinstance(part, Connection):
            chosen.update(id(p) for name, p in part.named_parameters() if name != "phi.weight")
        elif isinstance(part, Engram):
            tables.update(id(table.weight) for table in part.tables)
    for name, p in module.named_parameters():
        if id(p) in tables:
            yield name, p, "table", None
        elif id(p) in chosen or p.ndim < 2:
            yield name, p, "vector", None
        elif p.ndim == 3:
            yield name, p, "matrix", tuple(p.shape)
        else:
            n = heads.get(id(p), 1)
            yield name, p, "matrix", (n, p.shape[0] // n, p.shape[1])


class MuonAdamW(torch.optim.Optimizer):
    """Muon over the "matrix" parameters of roles, each in its view, AdamW
    over the "vector" ones, and Adam at table_lr_factor times lr, with no
    weight decay, over the "table" ones; a parameter that does not require
    a gradient (a locked part, one the loss never reaches) is left out, and
    so is a role with no parameter. Each group's "lr" is lr times its
    "lr_factor": a schedule sets it before each step."""

    def __init__(self, roles, lr, weight_decay, momentum=0.95, betas=(0.9, 0.95), eps=1e-8, table_lr_factor=1.0):
        found = {"matrix": ([], []), "vector": ([], []), "table": ([], [])}
        for _, parameter, role, view in roles:
            if parameter.requires_grad:
                found[role][0].append(parameter)
                found[role][1].append(view)
        groups = [
            {"params": found["matrix"][0], "views": found["matrix"][1], "muon": True, "lr_factor": 1.0,
             "weight_decay": weight_decay},
            {"params": found["vector"][0], "muon": False, "lr_factor": 1.0, "weight_decay": weight_decay},
            {"params": found["table"][0], "muon": False, "lr_factor": table_lr_factor, "weight_decay": 0.0},
        ]
        self.counts = {role: len(params) for role, (params, _) in found.items()}
        super().__init__([{**g, "lr": lr * g["lr_factor"]} for g in groups if g["params"]],
                         {"lr": lr, "momentum": momentum, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["muon"]:
                self._muon(group)
            else:
                self._adamw(group)

    def _muon(self, group):
        lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
        for p, view in zip(group["params"], group["views"]):
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p, dtype=torch.float32)
            g = p.grad.float()
            state["momentum"].mul_(mu).add_(g)
            update = orthogonalize(g.add(state["momentum"], alpha=mu).view(view)).view(p.shape)  # Nesterov
            p.mul_(1 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr * UPDATE_RMS * math.sqrt(max(view[-2:])))

    def _adamw(self, group):
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["mean"] = torch.zeros_like(p, dtype=torch.float32)
                state["square"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
            g = p.grad.float()
            state["mean"].lerp_(g, 1 - beta1)
            state["square"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            mean = state["mean"] / (1 - beta1 ** state["step"])
            root = (state["square"] / (1 - beta2 ** state["step"])).sqrt_().add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(mean.to(p.dtype), root.to(p.dtype), value=-lr)
