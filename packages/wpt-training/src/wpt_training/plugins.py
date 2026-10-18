"""What a strategy and a method are, and how their files are found."""

import contextlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from wpt_utils.files import sha256_file

HERE = Path(__file__).parent


class PluginError(Exception):
    """A strategy or method cannot be used: unknown name, not a Python file,
    no start(), start() returned the wrong type, an unknown or mistyped
    setting, or a strategy that cannot start here (ddp outside torchrun).
    Nothing has been trained."""


class Strategy:
    """One process training alone on one device. A strategy for several
    processes overrides what differs. With splits_windows, every process
    takes the same windows and trains its own slice of each (wpt_training's
    parts.py); its synchronize then adds the processes' gradients up."""

    splits_windows = False

    def __init__(self, device, rank=0, world_size=1, local_rank=0):
        self.device, self.rank, self.world_size, self.local_rank = device, rank, world_size, local_rank

    @property
    def is_main(self) -> bool:
        """The one process that logs and writes the model folder."""
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        """The one process per machine that prepares local files."""
        return self.local_rank == 0

    def wrap(self, model):
        """The model as the training steps call it."""
        return model

    def no_sync(self, wrapped):
        """A context for a micro-batch whose gradients need not be shared yet."""
        return contextlib.nullcontext()

    def synchronize(self, model):
        """Called once a step's gradients are complete in this process, before
        the update: share them, when wrap does not."""

    def mean(self, value: float) -> float:
        """The average of value over every process."""
        return value

    def reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """The elementwise sum (op "sum") or maximum ("max") of tensor over
        every process, in every process."""
        return tensor

    def broadcast(self, value):
        """The main process's value, in every process (value is ignored
        elsewhere). It must survive pickle."""
        return value

    def barrier(self):
        """Wait until every process reaches this point."""

    def close(self):
        """Release what start() took."""


@dataclass(frozen=True)
class Run:
    """What a method may know about the run."""
    tokens: int      # training plus validation tokens
    parameters: int
    steps: int       # the maximum number of steps
    log: callable    # a no-op except in the main process


@dataclass(frozen=True)
class Verdict:
    best: bool       # keep these weights as the best so far
    stop: bool       # stop training now
    note: str = ""   # added to the evaluation's log line


class Method:
    """How a model learns. A method sets self.optimizer and may override the
    rest; the defaults are the next-token loss plus the multi-token
    prediction loss at mtp_weight, plus the gradient of the model's indexer
    loss, and no stopping."""

    optimizer: torch.optim.Optimizer
    mtp_weight = 0.3  # DeepSeek-V3's first weight; a method may change it as training goes

    def loss(self, model, x, y):
        """The losses of a batch, windows x and y, the tokens after them:
        (training loss, indexer loss). The training loss is what the
        backward pass starts from: its value is the language-model loss,
        which the log shows next to the validation loss, and it carries the
        gradient of the CSA indexers' loss too, which reaches only the
        indexers (DeepSeek-V3.2). The indexer loss is that loss's value,
        which the log shows beside it."""
        return self.part_loss(model, x, y, None, x.shape[1])

    def part_loss(self, model, x, y, following, window):
        """The share of a part of the windows, x and y (B, t), in loss()'s
        two values over the whole windows, so that the parts' shares add up
        to them (parts.py): model computes the part where it sits in the
        windows; following is the token after y's last (B, 1), None at the
        windows' end; window is the windows' length. A method that defines
        its own loss defines this too, or trains whole windows only."""
        logits, ahead, indexers = model(x, next_tokens=y, indexer_loss=True)
        B, t = y.shape
        targets = y[:, 1:] if following is None else torch.cat([y[:, 1:], following], 1)  # the token after next
        loss = (_cross_entropy(logits, y, "sum") / (B * window)
                + self.mtp_weight * _cross_entropy(ahead[:, :targets.shape[1]], targets, "sum") / (B * (window - 1)))
        share = indexers * (t / window)  # the model's is the mean over the part's queries
        return loss + (share - share.detach()), share.detach()

    def validation_loss(self, model, x, y):
        """The next-token loss alone, which evaluations compare."""
        return _cross_entropy(model(x), y)

    def stage(self, number: int, block_size: int):
        """Called when a stage after the first starts (number from 1): the
        windows are now block_size tokens long, and the evaluations that
        follow are not comparable with earlier ones."""

    def before_update(self, model, step: int):
        """Called after the gradients of a step are complete, before the optimizer steps."""

    def after_update(self, model, step: int, reduce):
        """Called after the optimizer steps. reduce(tensor, op) sums ("sum")
        or takes the maximum ("max") of a tensor over every process."""

    def evaluated(self, step: int, val_loss: float) -> Verdict:
        return Verdict(best=True, stop=False)

    def state_dict(self) -> dict:
        """Everything a resumed run needs: tensors, numbers, strings, lists, dicts."""
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict):
        self.optimizer.load_state_dict(state["optimizer"])


def _cross_entropy(logits, targets, reduction="mean"):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)


@dataclass(frozen=True)
class Plugin:
    name: str
    file: str     # a built-in by its package path, another file by its name only
    sha256: str
    module: object


def _load(kind, name_or_path) -> Plugin:
    folder = HERE / kind
    built_in = sorted(p.stem for p in folder.glob("*.py") if not p.name.startswith("_"))
    if name_or_path in built_in:
        path, file = folder / f"{name_or_path}.py", f"wpt_training/{kind}/{name_or_path}.py"
    else:
        path = Path(name_or_path)
        if path.suffix != ".py" or not path.is_file():
            raise PluginError(f"{name_or_path}: not a built-in {kind[:-1]} ({', '.join(built_in)}) nor a Python file")
        file = path.name
    spec = importlib.util.spec_from_file_location(f"wpt_training_{kind}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "start", None)):
        raise PluginError(f"{path.name}: defines no start()")
    return Plugin(path.stem, file, sha256_file(path), module)


def load_strategy(name_or_path) -> tuple[Plugin, Strategy]:
    plugin = _load("strategies", name_or_path)
    strategy = plugin.module.start()
    if not isinstance(strategy, Strategy):
        raise PluginError(f"{plugin.file}: start() returned {type(strategy).__name__}, not wpt_training.Strategy")
    return plugin, strategy


def load_method(name_or_path, overrides: dict) -> tuple[Plugin, dict]:
    """The method's file and its settings: its SETTINGS defaults with the
    overrides, each of its default's type (an int where a float is expected
    is taken; text is converted)."""
    plugin = _load("methods", name_or_path)
    defaults = getattr(plugin.module, "SETTINGS", {})
    settings = dict(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise PluginError(f"{plugin.file}: no setting {key!r} (settings: {', '.join(defaults) or 'none'})")
        kind = type(defaults[key])
        if isinstance(value, str) and kind is not str:
            try:
                value = (value.lower() in ("1", "true", "yes")) if kind is bool else kind(value)
            except ValueError:
                raise PluginError(f"{plugin.file}: setting {key}={value!r} is not a {kind.__name__}") from None
        elif kind is float and type(value) is int:
            value = float(value)
        if type(value) is not kind:
            raise PluginError(f"{plugin.file}: setting {key}={value!r} is not a {kind.__name__}")
        settings[key] = value
    return plugin, settings


def start_method(plugin: Plugin, model, settings: dict, run: Run) -> Method:
    method = plugin.module.start(model, settings, run)
    if not isinstance(method, Method):
        raise PluginError(f"{plugin.file}: start() returned {type(method).__name__}, not wpt_training.Method")
    return method
