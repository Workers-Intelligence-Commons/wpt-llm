"""Each kind of weight under its own update rule (docs/model.md,
"Training"), on a schedule, with an optional early stop.

The update rule is wpt_model.MuonAdamW over the model's optimizer_roles:
Muon for matrix weights (DeepSeek-V4's hybrid Newton-Schulz steps, updates
of RMS 0.18, query, key and value projections head by head, routed experts
one by one, the other logically separate matrices apart), AdamW for the
embedding, the output head, norms, mHC's biases and gating factors,
attention sinks and merging biases, and Adam at table_lr_factor times the
rate, with no weight decay, for Engram's tables. Only the parameters the
step trains are in it: those of the parts it locks, and of the parts its
method never reaches, are not.

Schedule: "steps" (DeepSeek LLM), a linear warm-up to lr over `warmup`
updates, then first_factor of it after first_drop of the step's updates
and second_factor after second_drop, the updates being those of every
stage of the step together; "constant", lr at every update.

Without early stop (`patience` 0), the step keeps the weights of the
lowest validation loss of its last stage. With it, a stage stops when the
smoothed loss (the mean of the last `smoothing` evaluations) has not
improved by `min_improve` for `patience` evaluations, and the step keeps
the weights of the last improvement in its last stage. The smaller the
tokens-per-parameter ratio, the more overfitting is tolerated: at
Chinchilla balance (`chinchilla` tokens per parameter) an improvement must
be `min_improve`; a data-starved run relaxes toward "never stop early,
memorizing is the only available mode". An early stop ends the stage: the
next one starts, with its own count. A step without evaluations never
stops early and keeps its last weights.

lr and weight_decay default to Kimi Linear's smallest model (2.006e-3) and
Moonlight (0.1); a recipe sets its own.
"""

from wpt_model import MuonAdamW

from wpt_training import Optimizer, PluginError, Verdict

SETTINGS = {
    "lr": 0.002, "weight_decay": 0.1, "momentum": 0.95,
    "beta1": 0.9, "beta2": 0.95, "eps": 1e-8,
    "table_lr_factor": 5.0,
    "schedule": "steps",
    "warmup": 2000, "first_drop": 0.8, "first_factor": 0.316, "second_drop": 0.9, "second_factor": 0.1,
    "patience": 0, "min_improve": 0.01, "smoothing": 3, "chinchilla": 20,
}
SCHEDULES = ("steps", "constant")


def check(settings):
    if settings["schedule"] not in SCHEDULES:
        raise ValueError(f"schedule {settings['schedule']!r}: one of {', '.join(SCHEDULES)}")
    if settings["patience"] < 0:
        raise ValueError(f"patience {settings['patience']}: 0 (no early stop) or more evaluations")


class ByPart(Optimizer):
    def __init__(self, model, settings, run):
        if not hasattr(model, "optimizer_roles"):
            raise PluginError("by-part: the model does not name its parameters' roles (optimizer_roles)")
        self.settings, self.steps = settings, run.steps
        self.optimizer = MuonAdamW(model.optimizer_roles(), settings["lr"], settings["weight_decay"],
                                   settings["momentum"], (settings["beta1"], settings["beta2"]), settings["eps"],
                                   settings["table_lr_factor"])
        counts = self.optimizer.counts
        run.log(f"by-part: {counts['matrix']} Muon matrices, {counts['vector']} AdamW tensors, "
                f"{counts['table']} Engram tables; schedule {settings['schedule']}")
        self.min_improve = None
        if settings["patience"]:
            ratio = run.tokens / run.parameters
            self.min_improve = max(0.0005, settings["min_improve"] * min(1.0, ratio / settings["chinchilla"]))
            run.log(f"tokens: {run.tokens:,} | params: {run.parameters:,} | ratio: {ratio:.3f} tok/param "
                    f"(Chinchilla {settings['chinchilla']}) -> improvement threshold {self.min_improve * 100:.2f}%")
        self.best, self.without_improvement, self.history = float("inf"), 0, []

    def factor(self, step):
        """The share of lr an update uses."""
        s = self.settings
        if s["schedule"] == "constant":
            return 1.0
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
        # the step keeps the best weights of its last stage, at its window length; longer windows: another scale
        self.best, self.without_improvement, self.history = float("inf"), 0, []

    def evaluated(self, step, val_loss):
        note = f" | lr x{self.factor(step):.3f}"
        if self.min_improve is None:
            if val_loss < self.best:
                self.best = val_loss
                return Verdict(best=True, stop=False, note=note + " <- best")
            return Verdict(best=False, stop=False, note=note)
        self.history.append(val_loss)
        recent = self.history[-self.settings["smoothing"]:]
        smoothed = sum(recent) / len(recent)
        if smoothed < self.best * (1 - self.min_improve):
            self.best, self.without_improvement = smoothed, 0
            return Verdict(best=True, stop=False, note=note + f" <- best (smoothed {smoothed:.4f})")
        self.without_improvement += 1
        if self.without_improvement >= self.settings["patience"]:
            return Verdict(best=False, stop=True,
                           note=note + f"\nearly stop: smoothed val loss has not improved by "
                                       f"{self.min_improve * 100:.2f}% for {self.settings['patience']} evaluations "
                                       f"(best {self.best:.4f})")
        return Verdict(best=False, stop=False, note=note)

    def state_dict(self):
        return {**super().state_dict(), "best": self.best, "without_improvement": self.without_improvement,
                "history": list(self.history)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.best, self.without_improvement, self.history = state["best"], state["without_improvement"], state["history"]


def start(model, settings, run):
    return ByPart(model, settings, run)
