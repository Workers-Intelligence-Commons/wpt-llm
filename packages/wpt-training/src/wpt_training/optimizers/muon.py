"""Muon for matrix weights, AdamW for the rest, on a multi-step schedule
(docs/model.md, "Training").

Muon as DeepSeek-V4 runs it: the momentum of each matrix is orthogonalized
by hybrid Newton-Schulz steps (8 that bring the singular values near 1, then
2 that settle them at 1), then scaled by 0.18 sqrt(max(A, B)) for an A x B
matrix, so the update's RMS is 0.18, with decoupled weight decay:
W -= lr (0.18 sqrt(max(A, B)) O + wd W). The
model names each weight's role (WPT.optimizer_roles): query, key and value
projections are orthogonalized head by head, routed experts one by one and
the other logically separate matrices (an output group, a merging's
projection pair) apart; the embedding, output head, norms, mHC's biases and
gating factors, attention sinks and merging biases take AdamW; Engram's
tables take Adam at table_lr_factor times the rate, with no weight decay.
No QK-Clip: CSA and HCA normalize their queries and key-value entries, which
keeps their scores bounded (DeepSeek-V4).

Schedule (DeepSeek LLM): linear warm-up to lr over `warmup` updates, then
first_factor of it after first_drop of the step's updates and
second_factor after second_drop; the updates are those of every stage of
the step together. The step keeps the weights of the lowest validation
loss of its last stage, or its last weights when it has no evaluation.
Only the parameters the step trains are in it: those of the parts it
locks, and of the parts its method never reaches, are not.

lr and weight_decay default to Kimi Linear's smallest model (2.006e-3) and
Moonlight (0.1); a recipe sets its own.
"""

import math

import torch

from wpt_training import Optimizer, PluginError, Verdict

SETTINGS = {
    "lr": 0.002, "weight_decay": 0.1, "momentum": 0.95,
    "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
    "table_lr_factor": 5.0,
    "warmup": 2000, "first_drop": 0.8, "first_factor": 0.316, "second_drop": 0.9, "second_factor": 0.1,
}
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


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer for both kinds of groups: "muon" groups hold matrices,
    each with its view in "views"; the others are AdamW groups."""

    def __init__(self, groups, momentum, betas, eps):
        super().__init__(groups, {"lr": 0.0, "momentum": momentum, "betas": betas, "eps": eps})  # lr: set each step

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


class Muon(Optimizer):
    def __init__(self, model, settings, run):
        if not hasattr(model, "optimizer_roles"):
            raise PluginError("muon: the model does not name its parameters' roles (optimizer_roles)")
        self.settings, self.steps = settings, run.steps
        roles = {"matrix": ([], []), "vector": ([], []), "table": ([], [])}
        for _, parameter, role, view in model.optimizer_roles():
            if not parameter.requires_grad:  # a locked part, or one the method never reaches
                continue
            roles[role][0].append(parameter)
            roles[role][1].append(view)
        groups = [
            {"params": roles["matrix"][0], "views": roles["matrix"][1], "muon": True, "lr_factor": 1.0,
             "weight_decay": settings["weight_decay"]},
            {"params": roles["vector"][0], "muon": False, "lr_factor": 1.0, "weight_decay": settings["weight_decay"]},
            {"params": roles["table"][0], "muon": False, "lr_factor": settings["table_lr_factor"], "weight_decay": 0.0},
        ]
        self.optimizer = MuonAdamW([g for g in groups if g["params"]], settings["momentum"],
                                   (settings["beta1"], settings["beta2"]), settings["eps"])
        self.best = float("inf")
        run.log(f"muon: {len(roles['matrix'][0])} matrices, {len(roles['vector'][0])} AdamW tensors, "
                f"{len(roles['table'][0])} Engram tables")

    def factor(self, step):
        """The share of lr an update uses."""
        s = self.settings
        if step <= s["warmup"]:
            return step / s["warmup"]
        if step > s["second_drop"] * self.steps:
            return s["second_factor"]
        if step > s["first_drop"] * self.steps:
            return s["first_factor"]
        return 1.0

    def before_update(self, step):
        for group in self.optimizer.param_groups:
            group["lr"] = self.settings["lr"] * group["lr_factor"] * self.factor(step)

    def stage(self, number, block_size):
        self.best = float("inf")  # the step keeps the best weights of its last stage, at its window length

    def evaluated(self, step, val_loss):
        note = f" | lr x{self.factor(step):.3f}"
        if val_loss < self.best:
            self.best = val_loss
            return Verdict(best=True, stop=False, note=note + " <- best")
        return Verdict(best=False, stop=False, note=note)

    def state_dict(self):
        return {**super().state_dict(), "best": self.best}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.best = state["best"]


def start(model, settings, run):
    return Muon(model, settings, run)
