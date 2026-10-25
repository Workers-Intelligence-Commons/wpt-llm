"""Next-token prediction on text windows (docs/training.md, "Steps").

The loss of a window is the next-token loss of every position, plus the
multi-token prediction loss (the token after next) at mtp_weight, then at
mtp_late_weight after mtp_switch of the step's updates (DeepSeek-V3: after
10T of 14.8T tokens), plus the gradient of the CSA indexers' own loss,
which reaches only the indexers (wpt_training.TextMethod). Evaluations
compare the next-token loss alone. Windows longer than --segment, or split
across processes, train in parts with the same loss (parts.py).
"""

from wpt_training import WINDOWS, TextMethod

READS = WINDOWS
SETTINGS = {"mtp_weight": 0.3, "mtp_late_weight": 0.1, "mtp_switch": 0.676}


class NextToken(TextMethod):
    def __init__(self, settings):
        self.settings, self.steps, self.next_step = settings, 1, 1
        self._weigh_mtp()

    def begin(self, model, environment, run, segment, strategy):
        super().begin(model, environment, run, segment, strategy)
        self.steps = run.steps
        self._weigh_mtp()

    def after_update(self, step):
        self.next_step = step + 1
        self._weigh_mtp()

    def _weigh_mtp(self):
        s = self.settings
        self.mtp_weight = s["mtp_weight"] if self.next_step <= s["mtp_switch"] * self.steps else s["mtp_late_weight"]

    def state_dict(self):
        return {"next_step": self.next_step}

    def load_state_dict(self, state):
        self.next_step = state["next_step"]
        self._weigh_mtp()


def start(settings):
    return NextToken(settings)
