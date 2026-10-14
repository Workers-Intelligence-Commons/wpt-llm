"""AdamW at a constant learning rate, with early stopping on a smoothed
validation loss.

Training stops when the smoothed loss (the mean of the last `smoothing`
evaluations) has not improved by `min_improve` for `patience` evaluations.
The smaller the tokens-per-parameter ratio, the more overfitting is
tolerated: at Chinchilla balance (`chinchilla` tokens per parameter) an
improvement must be `min_improve`; a data-starved run relaxes toward "never
stop early, memorizing is the only available mode".
"""

import torch

from wpt_training import Method, Verdict

SETTINGS = {"lr": 3e-4, "patience": 10, "min_improve": 0.01, "smoothing": 3, "chinchilla": 20}


class AdamW(Method):
    def __init__(self, model, settings, run):
        self.settings = settings
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
        ratio = run.tokens / run.parameters
        self.min_improve = max(0.0005, settings["min_improve"] * min(1.0, ratio / settings["chinchilla"]))
        run.log(f"tokens: {run.tokens:,} | params: {run.parameters:,} | ratio: {ratio:.3f} tok/param "
                f"(Chinchilla {settings['chinchilla']}) -> improvement threshold {self.min_improve * 100:.2f}%")
        self.best, self.without_improvement, self.history = float("inf"), 0, []

    def evaluated(self, step, val_loss):
        self.history.append(val_loss)
        recent = self.history[-self.settings["smoothing"]:]
        smoothed = sum(recent) / len(recent)
        if smoothed < self.best * (1 - self.min_improve):
            self.best, self.without_improvement = smoothed, 0
            return Verdict(best=True, stop=False, note=f" <- best (smoothed {smoothed:.4f})")
        self.without_improvement += 1
        if self.without_improvement >= self.settings["patience"]:
            return Verdict(best=False, stop=True,
                           note=f"\nearly stop: smoothed val loss has not improved by {self.min_improve * 100:.2f}% "
                                f"for {self.settings['patience']} evaluations (best {self.best:.4f})")
        return Verdict(best=False, stop=False)

    def state_dict(self):
        return {**super().state_dict(), "best": self.best, "without_improvement": self.without_improvement,
                "history": list(self.history)}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.best, self.without_improvement, self.history = state["best"], state["without_improvement"], state["history"]


def start(model, settings, run):
    return AdamW(model, settings, run)
