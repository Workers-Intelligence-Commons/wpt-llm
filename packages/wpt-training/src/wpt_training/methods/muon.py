"""Muon for matrix weights, AdamW for the rest, on a multi-step schedule
(docs/model.md, "Training").

Muon (Moonlight): the momentum of each matrix is orthogonalized by
Newton-Schulz steps, then scaled by 0.2 sqrt(max(A, B)) for an A x B matrix,
with decoupled weight decay: W -= lr (0.2 sqrt(max(A, B)) O + wd W). The
model names each weight's role (WPT.optimizer_roles): query, key and value
projections are orthogonalized head by head and routed experts one by one;
the embedding, output head, norms and mHC's biases and gating factors take
AdamW; Engram's tables take Adam at table_lr_factor times the rate, with no
weight decay. After each step, QK-Clip scales the MLA heads whose largest
query-key score passed qk_clip (0 turns it off).

Schedule (DeepSeek LLM): linear warm-up to lr over `warmup` steps, then
first_factor of it after first_drop of the steps and second_factor after
second_drop. The multi-token prediction loss weighs mtp_weight, then
mtp_late_weight after mtp_switch of the steps (DeepSeek-V3: after 10T of
14.8T tokens). The folder keeps the weights of the lowest validation loss.

lr and weight_decay default to Kimi Linear's smallest model (2.006e-3) and
Moonlight (0.1); a recipe sets its own.
"""

import math

import torch

from wpt_training import Method, PluginError, Verdict

SETTINGS = {
    "lr": 0.002, "weight_decay": 0.1, "momentum": 0.95,
    "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
    "table_lr_factor": 5.0,
    "warmup": 2000, "first_drop": 0.8, "first_factor": 0.316, "second_drop": 0.9, "second_factor": 0.1,
    "mtp_weight": 0.3, "mtp_late_weight": 0.1, "mtp_switch": 0.676,
    "qk_clip": 100.0,
}
NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)  # Moonlight's coefficients, 5 steps
NEWTON_SCHULZ_STEPS = 5


def orthogonalize(G):
    """(matrices, A, B) -> each matrix's nearest semi-orthogonal matrix, approximately."""
    X = G.float()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = NEWTON_SCHULZ
    for _ in range(NEWTON_SCHULZ_STEPS):
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
            p.add_(update.to(p.dtype), alpha=-lr * 0.2 * math.sqrt(max(view[-2:])))

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


class Muon(Method):
    def __init__(self, model, settings, run):
        if not hasattr(model, "optimizer_roles"):
            raise PluginError("muon: the model does not name its parameters' roles (optimizer_roles)")
        self.settings, self.steps = settings, run.steps
        roles = {"matrix": ([], []), "vector": ([], []), "table": ([], [])}
        for _, parameter, role, view in model.optimizer_roles():
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
        model.track_query_keys(settings["qk_clip"] > 0)
        self.best, self.next_step = float("inf"), 1
        self._weigh_mtp()
        run.log(f"muon: {len(roles['matrix'][0])} matrices, {len(roles['vector'][0])} AdamW tensors, "
                f"{len(roles['table'][0])} Engram tables")

    def factor(self, step):
        """The share of lr a step uses."""
        s = self.settings
        if step <= s["warmup"]:
            return step / s["warmup"]
        if step > s["second_drop"] * self.steps:
            return s["second_factor"]
        if step > s["first_drop"] * self.steps:
            return s["first_factor"]
        return 1.0

    def before_update(self, model, step):
        for group in self.optimizer.param_groups:
            group["lr"] = self.settings["lr"] * group["lr_factor"] * self.factor(step)

    def after_update(self, model, step, reduce):
        if self.settings["qk_clip"] > 0:
            model.clip_query_keys(self.settings["qk_clip"], reduce)
        self.next_step = step + 1
        self._weigh_mtp()

    def _weigh_mtp(self):
        s = self.settings
        self.mtp_weight = s["mtp_weight"] if self.next_step <= s["mtp_switch"] * self.steps else s["mtp_late_weight"]

    def evaluated(self, step, val_loss):
        note = f" | lr x{self.factor(step):.3f}"
        if val_loss < self.best:
            self.best = val_loss
            return Verdict(best=True, stop=False, note=note + " <- best")
        return Verdict(best=False, stop=False, note=note)

    def state_dict(self):
        return {**super().state_dict(), "best": self.best, "next_step": self.next_step}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.best, self.next_step = state["best"], state["next_step"]
        self._weigh_mtp()


def start(model, settings, run):
    return Muon(model, settings, run)
