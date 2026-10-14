"""Next-token training: one runner, with two plug points.

A strategy knows the hardware: how many processes train together and which
one this is, its device, how the model is wrapped, how a number is averaged
over the processes. Built-ins (strategies/): `single`, one device; `ddp`,
several processes started by `torchrun`, on one machine or several.

A method knows how the model learns: its optimizer, its learning-rate
schedule, the loss of one batch, what happens just before each update, and
when training stops. Built-ins (methods/): `adamw`.

Each is a Python file, named by a built-in name or a path:

    strategy file:  def start() -> wpt_training.Strategy
    method file:    SETTINGS = {name: default, ...}
                    def start(model, settings, run) -> wpt_training.Method

The runner (train) owns everything else: the data, the global batch and
how it is split over processes and accumulation steps, evaluation,
checkpoints and resuming. The batches and the dropout of a step are seeded
by the run's seed and the step only, so the same settings train the same
way on any number of processes, and a resumed run continues exactly where
it stopped.
"""

from .plugins import Method, PluginError, Run, Strategy, Verdict, load_method, load_strategy
from .runner import RunSettings, SettingsMismatch, train

__all__ = ["Method", "PluginError", "Run", "RunSettings", "SettingsMismatch", "Strategy", "Verdict",
           "load_method", "load_strategy", "train"]
