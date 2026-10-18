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

from wpt_telemetry import RecordError, Telemetry

from . import parts
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
        """Replace the saved state; a crash meanwhile leaves the previous one.
        A checkpoint that records what it does raises the run's RecordError
        when its record breaks, once the state is saved all the same."""
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
class Stage:
    """Part of a run: steps updates on windows of block_size tokens,
    batch_size windows per update over every process together."""
    block_size: int
    steps: int
    batch_size: int


@dataclass(frozen=True)
class RunSettings:
    stages: tuple          # Stage (or a dict of its fields) for each stage, in order: the windows grow
    eval_every: int        # evaluate every N steps (and at the first and last steps of each stage)
    micro_batch: int = 0   # windows per forward pass in one process (0: its whole share of each stage's batch)
    eval_batches: int = 20  # global batches of the stage averaged per evaluation
    checkpoint_every: int = 0  # write a checkpoint every N steps (0: never)
    seed: int = 1337
    segment: int = 0       # tokens per pass of a longer window, in segments (parts.py); 0: whole windows

    def __post_init__(self):
        stages = tuple(stage if isinstance(stage, Stage) else Stage(**stage) for stage in self.stages)
        if not stages:
            raise ValueError("a run has at least one stage")
        object.__setattr__(self, "stages", stages)

    @property
    def steps(self) -> int:
        """Updates over every stage."""
        return sum(stage.steps for stage in self.stages)

    @property
    def longest(self) -> int:
        return max(stage.block_size for stage in self.stages)

    def stage(self, step) -> tuple[int, Stage, int, int]:
        """(index, stage, its first step, its last step) of a step."""
        first = 1
        for index, stage in enumerate(self.stages):
            if step < first + stage.steps:
                return index, stage, first, first + stage.steps - 1
            first += stage.steps
        raise ValueError(f"step {step} is after the last stage")

    def tokens_before(self, step) -> int:
        """Training tokens the steps before this one drew, windows of each stage's length."""
        used, first = 0, 1
        for stage in self.stages:
            done = min(max(step - first, 0), stage.steps)
            used += done * stage.batch_size * (stage.block_size + 1)
            first += stage.steps
        return used


def batching(settings: RunSettings, sharing: int) -> list[dict]:
    """How each stage's batch splits, in order: its windows (the stage's
    batch_size), micro_batch (windows per pass in one process: the
    settings', or the process's whole share) and accumulation (passes each
    process adds up before an update). sharing: the processes that share out
    the windows (1 when each splits every window). ValueError when a batch
    does not split evenly."""
    found = []
    for number, stage in enumerate(settings.stages, 1):
        micro = settings.micro_batch or stage.batch_size // sharing
        if not micro or stage.batch_size % (micro * sharing):
            raise ValueError(f"stage {number}'s batch of {stage.batch_size} windows does not split into micro-batches "
                             f"of {micro} on {sharing} processes; choose sizes that divide it")
        found.append({"windows": stage.batch_size, "micro_batch": micro,
                      "accumulation": stage.batch_size // (micro * sharing)})
    return found


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
        return tokens.at(settings.tokens_before(step))
    return tokens


def _micro_batches(tokens, stage, split, strategy, *seed_parts):
    """This process's micro-batches of one global batch of the stage (split:
    its entry in batching): micro-batch a of process r is slice
    (a * world_size + r) of the batch, so the batch holds the same windows
    whatever the number of processes; with split windows, every process
    takes every micro-batch."""
    batch = _windows(tokens, stage.block_size, stage.batch_size, *seed_parts)
    shared, rank = (1, 0) if strategy.splits_windows else (strategy.world_size, strategy.rank)
    micro = split["micro_batch"]
    for a in range(split["accumulation"]):
        start = (a * shared + rank) * micro
        part = batch[start:start + micro].to(strategy.device)
        yield a, part[:, :-1], part[:, 1:]


@dataclass(frozen=True)
class Trained:
    """What a run reached: the validation loss of the evaluation the method
    judged best in the last stage, the last step trained, the step whose
    weights the model holds (that best evaluation's, or the last step's when
    none was best), the indexer loss of that step's last micro-batch, and
    each stage: its window length, its steps, its batch, its first step and
    the last step it trained (an early stop ends a stage before its
    steps)."""
    best_val: float
    steps: int
    weights_step: int
    indexer_loss: float
    stages: tuple = ()


def train(model, strategy, method_plugin, method_settings, train_tokens, val_tokens, settings: RunSettings,
          checkpoint: Checkpoint | None = None, resume=False, identity=None, log=print, telemetry=None) -> Trained:
    """Train model in place, then load the weights of the evaluation the
    method judged best back into it; returns a Trained. train_tokens and val_tokens are 1-D arrays of token ids (np.memmap
    works); train_tokens may instead be a stream (see _tokens_at). With a
    checkpoint and settings.checkpoint_every, the main process saves there
    every N steps; resume=True continues from it. identity: what a resumed
    run must share with the checkpoint besides the settings (for example the
    dataset id and the model shape).

    The stages run in order, each drawing batches of its own number of
    windows of its own length (a step's windows depend only on the seed and
    the step); a method's stop
    ends the current stage, and the next one starts. When a stage after the
    first starts, the method hears of it (Method.stage) and the best
    evaluation so far is forgotten: the folder keeps the best weights of the
    last stage. Windows longer than settings.segment run in segments, and a
    strategy that splits windows trains each process's slice, in segments
    too when a slice is longer than settings.segment (parts.py): either
    way, the same training as whole windows.

    A stop signal (SIGINT, SIGTERM, and SIGBREAK on Windows) to any process
    lets every process finish the step; they then save the checkpoint, when
    the run keeps them, and raise Interrupted. A second signal acts as
    usual.

    telemetry (a wpt_telemetry.Telemetry) records, from the main process,
    every step (its loss over the whole batch, its duration, the learning
    rates), every evaluation, checkpoint, resume and stop. When the record
    breaks (RecordError from a call, log included, or a send refused in
    the background, which arrives as a stop signal), the step goes on
    unrecorded; then every process saves the checkpoint, when the run keeps
    them, and raises RecordError: the main process the record's own, the
    others one that says the main process's record broke. A record that
    breaks at the last event, once the steps are done, raises RecordError
    with no further save: the last checkpoint written stays."""
    record = _Unbroken(telemetry if telemetry is not None and strategy.is_main else Telemetry.off())
    log = record.guard(log) if strategy.is_main else (lambda *_: None)
    splits = strategy.splits_windows
    batches = batching(settings, 1 if splits else strategy.world_size)  # each stage's
    if len(train_tokens) <= settings.longest + 1:
        raise ValueError(f"the training data has {len(train_tokens):,} tokens, fewer than one window of "
                         f"{settings.longest}")
    parameters = sum(p.numel() for p in model.parameters())
    run = Run(tokens=len(train_tokens) + len(val_tokens), parameters=parameters, steps=settings.steps, log=log)
    method = start_method(method_plugin, model, method_settings, run)
    parts.check(model, method, settings.stages, settings.segment, strategy)
    wrapped = strategy.wrap(model)
    signature = {"settings": {k: v for k, v in asdict(settings).items()
                              if k not in ("micro_batch", "checkpoint_every", "segment")},
                 "method": [method_plugin.sha256, method_settings], "identity": identity}

    first_step, best_val, best_state, best_step, best_indexer, reached = 1, float("inf"), None, None, None, {}
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
            first_step, best_val, best_state = state["next_step"], state["best_val"], state["best_model"]
            best_step, best_indexer, reached = state["best_step"], state["best_indexer"], dict(state["reached"])
            record.event("training.resumed", step=state["step"], checkpoint=str(checkpoint))
            log(f"resumed from step {state['step']}")

    def whole(value):
        """A number over every process: their mean, or with split windows the
        sum of their shares."""
        return strategy.mean(value) * (strategy.world_size if splits else 1)

    def evaluate(step, index):
        model.eval()
        losses, stage = [], settings.stages[index]
        # a held-out part shorter than a window: the training data, as the stage's start said
        tokens = val_tokens if len(val_tokens) > stage.block_size + 1 else train_tokens
        with torch.no_grad():
            for i in range(settings.eval_batches):
                for _, x, y in _micro_batches(_tokens_at(tokens, step, settings), stage, batches[index], strategy,
                                              settings.seed, "eval", step, i):
                    losses.append(float(parts.validation_loss(method, model, wrapped, x, y, strategy,
                                                              settings.segment)))
        model.train()
        return whole(sum(losses) / len(losses))

    def _step(step, index, first, last):
        """One update, and its evaluation when one is due; True when the method says stop."""
        nonlocal best_val, best_state, best_step, best_indexer, indexer
        started, losses, stage = time.monotonic(), [], settings.stages[index]
        accumulate = batches[index]["accumulation"]
        for a, x, y in _micro_batches(_tokens_at(train_tokens, step, settings), stage, batches[index], strategy,
                                      settings.seed, "train", step):
            # by the slice's place in the batch, so dropout does not depend on the number of processes
            torch.manual_seed(_seed(settings.seed, "dropout", step, a * strategy.world_size + strategy.rank))

            def quiet(final, a=a):  # the step's gradients are shared at its very last backward pass
                return (contextlib.nullcontext() if final and a == accumulate - 1
                        else strategy.no_sync(wrapped))
            loss, indexer = parts.learn(method, model, wrapped, x, y, 1 / accumulate, strategy, settings.segment,
                                        quiet)
            losses.append(loss)
        strategy.synchronize(model)
        method.before_update(model, step)
        method.optimizer.step()
        method.optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "after_update"):  # the model's own updates, such as expert balancing
            model.after_update(strategy.reduce)
        method.after_update(model, step, strategy.reduce)
        # the loss over the whole batch: every micro-batch of every process
        train_loss = whole(torch.stack(losses).float().mean().item())
        indexer = torch.tensor(whole(float(indexer)) if splits else float(indexer))
        record.event("training.step", values={"train_loss": train_loss, "indexer_loss": indexer.item(),
                                             "seconds": time.monotonic() - started},
                     step=step, block_size=stage.block_size,
                     learning_rates=[group["lr"] for group in method.optimizer.param_groups])

        if step % settings.eval_every == 0 or step == first or step == last:
            val_loss = evaluate(step, index)
            verdict = method.evaluated(step, val_loss)
            if verdict.best:
                best_val, best_step, best_indexer = val_loss, step, indexer.item()
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            record.event("training.evaluation", values={"val_loss": val_loss}, step=step, best=verdict.best,
                         stop=verdict.stop, note=verdict.note)
            log(f"step {step:5d} | train loss {train_loss:.4f} | indexer loss {indexer.item():.4f} | "
                f"val loss {val_loss:.4f} | "
                f"{time.time() - t0:.0f}s{verdict.note}")
            return verdict.stop
        return False

    def save(step, next_step):
        if strategy.is_main:
            with record.span("training.checkpoint", step=step, where=str(checkpoint)):
                try:
                    checkpoint.save({"signature": signature, "step": step, "next_step": next_step,
                                     "model": model.state_dict(), "method": method.state_dict(),
                                     "best_val": best_val, "best_model": best_state, "best_step": best_step,
                                     "best_indexer": best_indexer, "reached": list(reached.items())})
                except RecordError as error:  # saved all the same; see Checkpoint.save
                    record.broken = record.broken or error
        strategy.barrier()

    keeps = checkpoint is not None and settings.checkpoint_every > 0
    model.train()
    t0 = time.time()
    last, indexer, step = first_step - 1, None, first_step
    with _StopSignals(log) as signals:
        while step <= settings.steps:
            index, stage, begin, end = settings.stage(step)
            if step == begin and index:  # longer windows: the best so far is no longer comparable
                best_val, best_state, best_step, best_indexer = float("inf"), None, None, None
                method.stage(index + 1, stage.block_size)
            log(f"stage {index + 1} of {len(settings.stages)}: windows of {stage.block_size:,} tokens, "
                f"{stage.batch_size} per step, steps {step} to {end}" + (f" (from {begin})" if step != begin else ""))
            if len(val_tokens) <= stage.block_size + 1:
                log("text is small: validation loss is measured on training data")
            with record.span("training.stage", stage=index + 1, block_size=stage.block_size,
                             batch_size=stage.batch_size, first_step=begin,
                             last_step=end, resumed_at=step):
                while step <= end:
                    stop = _step(step, index, begin, end)
                    last, reached[index] = step, step
                    next_step = end + 1 if stop else step + 1  # a method's stop ends the stage
                    if keeps and step % settings.checkpoint_every == 0:
                        save(step, next_step)
                    # every process must stop at the same step, wherever the signal came or the record broke
                    asked, broken = (float(signals.asked), float(record.broken is not None))
                    if strategy.world_size > 1:
                        flags = torch.tensor([asked, broken], device=strategy.device)
                        asked, broken = strategy.reduce(flags, "max").tolist()
                    if asked or broken:
                        if keeps and step % settings.checkpoint_every:
                            save(step, next_step)
                        record.event("training.stopped", step=step, saved=keeps)
                        if record.broken is not None:
                            raise record.broken
                        if broken:
                            raise RecordError(f"the main process's record broke; stopped after step {step}")
                        raise Interrupted(step, keeps)
                    step = next_step
                    if stop:
                        log(f"stage {index + 1} ended at step {last}")
                        break

    if best_state is not None:
        model.load_state_dict(best_state)
    kept = best_state is not None and best_step is not None
    stages, begin = [], 1
    for index, stage in enumerate(settings.stages):
        stages.append({"block_size": stage.block_size, "steps": stage.steps, "batch_size": stage.batch_size,
                       "first_step": begin, "last_step": reached.get(index)})
        begin += stage.steps
    trained = Trained(best_val, last, best_step if kept else last,
                      best_indexer if kept else (indexer.item() if indexer is not None else None), tuple(stages))
    record.event("training.finished", values={"best_validation_loss": best_val}, steps=trained.steps,
                 weights_step=trained.weights_step, stages=stages)
    if record.broken is not None:  # at the last event, once training ended: the last checkpoint written stays
        raise record.broken
    return trained


class _Unbroken:
    """The run's record, through which a RecordError does not stop the
    step: it is kept in `broken`, and every later call does nothing."""

    def __init__(self, telemetry):
        self.telemetry, self.broken = telemetry, None

    def guard(self, call):
        def guarded(*args, **options):
            if self.broken is None:
                try:
                    call(*args, **options)
                except RecordError as error:
                    self.broken = error
        return guarded

    def event(self, name, /, values=None, **attributes):
        self.guard(self.telemetry.event)(name, values, **attributes)

    @contextlib.contextmanager
    def span(self, name, /, **attributes):
        """The telemetry's span; a RecordError at its start or its end is
        kept, and one from inside the block goes on as itself."""
        if self.broken is not None:
            yield None
            return
        opened = self.telemetry.span(name, **attributes)
        try:
            opened.__enter__()
        except RecordError as error:
            self.broken = error
            yield None
            return
        try:
            yield None
        except BaseException as error:
            if not opened.__exit__(type(error), error, error.__traceback__):
                raise
        else:
            try:
                opened.__exit__(None, None, None)
            except RecordError as error:
                self.broken = error


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

