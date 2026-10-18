"""Next-token training: one runner, with two plug points.

A strategy knows the hardware: how many processes train together and which
one this is, its device, how the model is wrapped, how a number is averaged
over the processes. Built-ins (strategies/): `single`, one device; `ddp`,
several processes started by `torchrun`, on one machine or several, each
on its own windows; `split`, several processes that split each window
between them (DeepSeek-V4's contextual parallelism).

A method knows how the model learns: its optimizer, its learning-rate
schedule, the loss of one batch, what happens just before and after each
update, and when training stops. Built-ins (methods/): `adamw`.

Each is a Python file, named by a built-in name or a path:

    strategy file:  def start() -> wpt_training.Strategy
    method file:    SETTINGS = {name: default, ...}
                    def start(model, settings, run) -> wpt_training.Method

The runner (train) owns everything else: the data, the recipe's stages
(each its window length, steps and global batch), how each batch is split
over processes and accumulation steps (batching), windows trained in parts
(parts.py: segments on one process, slices across processes, slices in
segments), evaluation,
checkpoints (in a folder, or anywhere a Checkpoint subclass saves them),
resuming, and stopping cleanly on a signal. The batches and the random draws of a step are seeded
by the run's seed and the step only, so the same settings train the same
way on any number of processes, and a resumed run continues exactly where
it stopped.
"""

from .plugins import Method, PluginError, Run, Strategy, Verdict, load_method, load_strategy
from .runner import (Checkpoint, FolderCheckpoint, Interrupted, RunSettings, SettingsMismatch, Stage, Trained, batching,
                     train)

__all__ = ["Checkpoint", "FolderCheckpoint", "Interrupted", "Method", "PluginError", "Run", "RunSettings",
           "SettingsMismatch", "Stage", "Strategy", "Trained", "Verdict", "batching", "load_method", "load_strategy",
           "train"]
