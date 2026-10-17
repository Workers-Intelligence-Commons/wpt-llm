"""The training loop shared by every strategy and method."""

import contextlib
import hashlib
import os
import shutil
import signal
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .plugins import Run, start_method

CHECKPOINT = "checkpoint.pt"


class SettingsMismatch(Exception):
    """--resume found a checkpoint written with other settings (data, model,
    method, batch, seed); continuing would mix two runs. Nothing changed."""


class Interrupted(Exception):
    """The run was asked to stop (Ctrl+C, SIGTERM, a machine being shut
    down): it finished its step, wrote a checkpoint when it keeps them, and
    stopped. `step` is the last step done; `saved` says whether the
    checkpoint holds it."""

    def __init__(self, step: int, saved: bool):
        super().__init__(f"stopped after step {step}" + ("; the checkpoint holds it" if saved else ""))
        self.step, self.saved = step, saved


class Checkpoint:
    """Where a run's checkpoint lives: the state of an unfinished run, which
    only the main process writes and reads."""

    def exists(self) -> bool:
        raise NotImplementedError

    def load(self) -> dict | None:
        """The saved state, or None when there is none."""
        raise NotImplementedError

    def save(self, state: dict):
        """Replace the saved state; a crash meanwhile leaves the previous one."""
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError


class FolderCheckpoint(Checkpoint):
    """checkpoint.pt in a folder of this machine."""

    def __init__(self, folder: Path):
        self.folder, self.path = Path(folder), Path(folder) / CHECKPOINT

    def __str__(self):
        return str(self.folder)

    def exists(self):
        return self.folder.exists()

    def load(self):
        return torch.load(self.path, map_location="cpu", weights_only=True) if self.path.is_file() else None

    def save(self, state):
        _save(self.path, state)

    def remove(self):
        shutil.rmtree(self.folder, ignore_errors=True)


@dataclass(frozen=True)
class RunSettings:
    block_size: int
    batch_size: int        # windows per update, over every process together
    micro_batch: int       # windows per forward pass in one process
    steps: int             # maximum number of updates
    eval_every: int        # evaluate every N steps (and at the first and last steps)
    eval_batches: int = 20  # global batches averaged per evaluation
    checkpoint_every: int = 0  # write a checkpoint every N steps (0: never)
    seed: int = 1337


def accumulation(settings: RunSettings, world_size: int) -> int:
    """Micro-batches each process adds up before an update."""
    per_update = settings.micro_batch * world_size
    if settings.batch_size % per_update:
        raise ValueError(f"the batch of {settings.batch_size} windows does not split into micro-batches of "
                         f"{settings.micro_batch} on {world_size} processes; choose sizes that divide it")
    return settings.batch_size // per_update


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") & (2**63 - 1)


def _windows(tokens, block_size, count, *seed_parts):
    """count random windows of block_size + 1 tokens, the same on every machine."""
    generator = torch.Generator().manual_seed(_seed(*seed_parts))
    starts = torch.randint(len(tokens) - block_size - 1, (count,), generator=generator).tolist()
    return torch.from_numpy(np.stack([np.asarray(tokens[s:s + block_size + 1], dtype=np.int64) for s in starts]))


def _tokens_at(tokens, step, settings):
    """The tokens a step draws its windows from: the array itself, or for a
    stream (an object with `at`, such as wpt_store.shards.ShardStream) the
    part in use once the steps before it have used their tokens."""
    if hasattr(tokens, "at"):
        return tokens.at((step - 1) * settings.batch_size * (settings.block_size + 1))
    return tokens


def _micro_batches(tokens, settings, strategy, accumulate, *seed_parts):
    """This process's micro-batches of one global batch: micro-batch a of
    process r is slice (a * world_size + r) of the batch, so the batch holds
    the same windows whatever the number of processes."""
    batch = _windows(tokens, settings.block_size, settings.batch_size, *seed_parts)
    for a in range(accumulate):
        start = (a * strategy.world_size + strategy.rank) * settings.micro_batch
        part = batch[start:start + settings.micro_batch].to(strategy.device)
        yield a, part[:, :-1], part[:, 1:]


@dataclass(frozen=True)
class Trained:
    """What a run reached: the validation loss of the evaluation the method
    judged best, the last step trained, the step whose weights the model
    holds (that best evaluation's, or the last step's when none was best),
    and the indexer loss of that step's last micro-batch."""
    best_val: float
    steps: int
    weights_step: int
    indexer_loss: float


def train(model, strategy, method_plugin, method_settings, train_tokens, val_tokens, settings: RunSettings,
          checkpoint: Checkpoint | None = None, resume=False, identity=None, log=print) -> Trained:
    """Train model in place, then load the weights of the evaluation the
    method judged best back into it; returns a Trained. train_tokens and val_tokens are 1-D arrays of token ids (np.memmap
    works); train_tokens may instead be a stream (see _tokens_at). With a
    checkpoint and settings.checkpoint_every, the main process saves there
    every N steps; resume=True continues from it. identity: what a resumed
    run must share with the checkpoint besides the settings (for example the
    dataset id and the model shape).

    A stop signal (SIGINT, SIGTERM, and SIGBREAK on Windows) to any process
    lets every process finish the step; they then save the checkpoint, when
    the run keeps them, and raise Interrupted. A second signal acts as
    usual."""
    log = log if strategy.is_main else (lambda *_: None)
    if len(val_tokens) <= settings.block_size + 1:
        log("text is small: validation loss is measured on training data")
        val_tokens = train_tokens
    accumulate = accumulation(settings, strategy.world_size)
    parameters = sum(p.numel() for p in model.parameters())
    run = Run(tokens=len(train_tokens) + len(val_tokens), parameters=parameters, steps=settings.steps, log=log)
    method = start_method(method_plugin, model, method_settings, run)
    wrapped = strategy.wrap(model)
    signature = {"settings": {k: v for k, v in asdict(settings).items() if k not in ("micro_batch", "checkpoint_every")},
                 "method": [method_plugin.sha256, method_settings], "identity": identity}

    first_step, best_val, best_state, best_step, best_indexer = 1, float("inf"), None, None, None
    if resume and checkpoint:
        # only the main process wrote it, maybe on another machine: it reads and sends it
        state = strategy.broadcast(checkpoint.load() if strategy.is_main else None)
        if state is None:
            log(f"no checkpoint in {checkpoint}: starting at step 1")
        else:
            if state["signature"] != signature:
                raise SettingsMismatch(f"the checkpoint in {checkpoint} was written by a run with other settings")
            model.load_state_dict(state["model"])
            method.load_state_dict(state["method"])
            first_step, best_val, best_state = state["step"] + 1, state["best_val"], state["best_model"]
            best_step = state.get("best_step")  # absent from checkpoints written before it was kept
            best_indexer = state.get("best_indexer")
            log(f"resumed from step {state['step']}")

    def evaluate(step):
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(settings.eval_batches):
                for _, x, y in _micro_batches(_tokens_at(val_tokens, step, settings), settings, strategy, accumulate,
                                              settings.seed, "eval", step, i):
                    losses.append(method.validation_loss(wrapped, x, y).item())
        model.train()
        return strategy.mean(sum(losses) / len(losses))

    def _step(step):
        """One update, and its evaluation when one is due; True when the method says stop."""
        nonlocal best_val, best_state, best_step, best_indexer, indexer
        for a, x, y in _micro_batches(_tokens_at(train_tokens, step, settings), settings, strategy, accumulate,
                                      settings.seed, "train", step):
            # by the slice's place in the batch, so dropout does not depend on the number of processes
            torch.manual_seed(_seed(settings.seed, "dropout", step, a * strategy.world_size + strategy.rank))
            with strategy.no_sync(wrapped) if a < accumulate - 1 else contextlib.nullcontext():
                loss, indexer = method.loss(wrapped, x, y)
                (loss / accumulate).backward()
        method.before_update(model, step)
        method.optimizer.step()
        method.optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "after_update"):  # the model's own updates, such as expert balancing
            model.after_update(strategy.reduce)
        method.after_update(model, step, strategy.reduce)

        if step % settings.eval_every == 0 or step == 1 or step == settings.steps:
            val_loss = evaluate(step)
            verdict = method.evaluated(step, val_loss)
            if verdict.best:
                best_val, best_step, best_indexer = val_loss, step, indexer.item()
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log(f"step {step:5d} | train loss {loss.item():.4f} | indexer loss {indexer.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"{time.time() - t0:.0f}s{verdict.note}")
            return verdict.stop
        return False

    def save(step):
        if strategy.is_main:
            checkpoint.save({"signature": signature, "step": step, "model": model.state_dict(),
                             "method": method.state_dict(), "best_val": best_val, "best_model": best_state,
                             "best_step": best_step, "best_indexer": best_indexer})
        strategy.barrier()

    keeps = checkpoint is not None and settings.checkpoint_every > 0
    model.train()
    t0 = time.time()
    last, indexer = first_step - 1, None
    with _StopSignals(log) as signals:
        for step in range(first_step, settings.steps + 1):
            stop = _step(step)
            last = step
            if keeps and step % settings.checkpoint_every == 0:
                save(step)
            # every process must stop at the same step, wherever the signal came
            if (strategy.mean(1.0 if signals.asked else 0.0) if strategy.world_size > 1 else signals.asked) > 0:
                if keeps and step % settings.checkpoint_every:
                    save(step)
                raise Interrupted(step, keeps)
            if stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    kept = best_state is not None and best_step is not None
    return Trained(best_val, last, best_step if kept else last,
                   best_indexer if kept else (indexer.item() if indexer is not None else None))


def _save(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    os.replace(partial, path)


class _StopSignals:
    """While active, a stop signal only sets `asked` (and puts the usual
    action back, so a second one stops at once). Signals reach Python's main
    thread only: elsewhere this does nothing."""

    NAMES = ("SIGINT", "SIGTERM", "SIGBREAK")  # SIGBREAK: Windows' Ctrl+Break, which a launcher forwards

    def __init__(self, log):
        self.log, self.asked, self.previous = log, False, {}

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in self.NAMES:
                # an ignored signal stays ignored, as nohup and background jobs ask
                if hasattr(signal, name) and signal.getsignal(getattr(signal, name)) is not signal.SIG_IGN:
                    number = getattr(signal, name)
                    self.previous[number] = signal.signal(number, self._handle)
        return self

    def _handle(self, number, frame):
        self.asked = True
        signal.signal(number, self.previous.get(number, signal.SIG_DFL))
        self.log(f"{signal.Signals(number).name}: stopping after this step (again to stop at once)")

    def __exit__(self, *exc):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        return False

