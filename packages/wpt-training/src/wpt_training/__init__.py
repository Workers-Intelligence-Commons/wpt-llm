"""Training: one runner, for every step of a model recipe.

A step is what is done to the model (steps/: `read-text`,
`answer-as-assistant`, `finish-tasks`, `finish-tasks-with-less-writing`). It
chooses one of each of these roles, its settings, and the parts of the
model it locks (wpt_model.PARTS):

    a method       how a loss is made from a batch of examples (methods/:
                   `next-token`, `assistant-tokens`, `grpo`)
    an optimizer   how the weights move, with its schedule and its verdict on
                   each evaluation (optimizers/: `muon`, `adamw`); a recipe may
                   give a step another one, so any loss runs with any optimizer
    an environment where the examples come from (environments/: `text-windows`,
                   `conversations`, `agent-tasks`)

A strategy knows the hardware: how many processes train together and which
one this is, its device, how the model is wrapped, how a number is averaged
over the processes. Built-ins (strategies/): `single`, one device; `ddp`,
several processes on one machine, each on its own examples; `split`,
several processes that split each text window between them (DeepSeek-V4's
contextual parallelism). Every one runs inside a gpu worker (wpt_worker;
docs/machines.md): the training's processes are the worker's job, and an
environment's part that needs what only the wpt manager holds runs there
(plugins.py).

Each is a Python file, named by a built-in name or a path (plugins.py says
what each defines). The runner (train) owns everything else: the updates,
each batch's passes, the gradients and their exchange between processes,
the locked parts, windows trained in parts (parts.py: segments on one
process, slices across processes, slices in segments), evaluation,
checkpoints (in a folder, or anywhere a Checkpoint subclass saves them),
resuming, stopping cleanly on a signal, and the step's telemetry. The
batches and the random draws of an update are seeded by the seeds and the
update only, so the same settings train the same way on any number of
processes, and a resumed step continues exactly where it stopped.
"""

from .plugins import (SENT, Context, Data, Environment, Method, Optimizer, PluginError, Run, Stage, Step, StepFailed,
                      Strategy, TextMethod, Verdict, WINDOWS, built_in, in_process, load_step, load_strategy, step_seed,
                      strategy_plugin)
from .runner import Checkpoint, FolderCheckpoint, Interrupted, Options, SettingsMismatch, Trained, train, weights
from .states import load_state, save_state

__all__ = ["SENT", "Checkpoint", "Context", "Data", "Environment", "FolderCheckpoint", "Interrupted", "Method", "Optimizer", "Options",
           "PluginError", "Run", "SettingsMismatch", "Stage", "Step", "StepFailed", "Strategy", "TextMethod",
           "Trained", "Verdict", "WINDOWS", "built_in", "in_process", "load_state", "load_step", "load_strategy",
           "save_state", "step_seed", "strategy_plugin", "train", "weights"]
