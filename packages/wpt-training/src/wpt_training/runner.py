"""The training loop every step shares, whatever its method, optimizer and
environment."""

import contextlib
import hashlib
import os
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from wpt_telemetry import RecordError, Telemetry

from .plugins import PluginError, Run
from .states import load_state, save_state

CHECKPOINT = "checkpoint.safetensors"


class SettingsMismatch(Exception):
    """--resume found a checkpoint written by another step, or with other
    settings (data, model, method, optimizer, batch, seed); continuing would
    mix two runs. Nothing changed."""


class Interrupted(Exception):
    """The run was asked to stop (Ctrl+C, SIGTERM, a machine being shut
    down): it finished its update, wrote a checkpoint when it keeps them,
    and stopped. `step` is the last update done; `saved` says whether the
    checkpoint holds it."""

    def __init__(self, step: int, saved: bool):
        super().__init__(f"stopped after update {step}" + ("; the checkpoint holds it" if saved else ""))
        self.step, self.saved = step, saved


class Checkpoint:
    """Where a run's checkpoint lives: the state of an unfinished step,
    which only the main process writes and reads."""

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

    def save_file(self, path: Path):
        """save() of a state already written to the file path (a state
        file, wpt_training.states), such as one a gpu worker sent; the file
        stays the caller's."""
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError


class FolderCheckpoint(Checkpoint):
    """checkpoint.safetensors, a state file (states.py), in a folder of this machine."""

    def __init__(self, folder: Path):
        self.folder, self.path = Path(folder), Path(folder) / CHECKPOINT

    def __str__(self):
        return str(self.folder)

    def exists(self):
        return self.folder.exists()

    def load(self):
        return load_state(self.path) if self.path.is_file() else None

    def save(self, state):
        save_state(state, self.path)

    def save_file(self, path):
        self.folder.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        shutil.copyfile(path, partial)
        os.replace(partial, self.path)

    def remove(self):
        shutil.rmtree(self.folder, ignore_errors=True)


@dataclass(frozen=True)
class Options:
    """How the hardware runs a step, which never changes what it learns:
    checkpoint_every (write a checkpoint every N updates, 0 never),
    segment (tokens per pass of a longer window, in segments: parts.py; 0
    whole windows)."""
    checkpoint_every: int = 0
    segment: int = 0


@dataclass(frozen=True)
class Trained:
    """What a step reached: the validation loss of the evaluation the
    optimizer judged best in the last stage (None without evaluations), the
    last update made, the update whose weights the model holds (that best
    evaluation's, or the last one's when none was best), the indexer loss
    of that update, each stage (its steps, its batch, its window length,
    its first update and the last it made: an early stop ends a stage
    before its steps), and the updates it skipped (nothing to learn)."""
    best_val: float | None
    steps: int
    weights_step: int
    indexer_loss: float | None
    stages: tuple = ()
    skipped: int = 0


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little") & (2**63 - 1)


def stage_of(stages, step) -> tuple[int, object, int, int]:
    """(index, stage, its first update, its last update) of an update."""
    first = 1
    for index, stage in enumerate(stages):
        if step < first + stage.steps:
            return index, stage, first, first + stage.steps - 1
        first += stage.steps
    raise ValueError(f"update {step} is after the last stage")


def train(model, strategy, step, method, environment, seed: int, number: int, options: Options = Options(),
          checkpoint: Checkpoint | None = None, resume=False, identity=None, done=(), log=print, telemetry=None,
          save_last=False) -> Trained:
    """Run one step (a wpt_training.Step, its method and environment
    started; number, its place in the recipe from 1, which dropout reads
    with the seed) on model, in place, then load the weights of the evaluation
    its optimizer judged best back into it; returns a Trained. The parts
    the step locks, and those its method never reaches, keep their
    weights: they are left out of the optimizer and of the processes'
    gradient exchange, and with BALANCING locked the routers' biases stay
    as they are. With a checkpoint and options.checkpoint_every, the main
    process saves there every N updates; with a checkpoint and save_last,
    also after the last update (unless that update's save already did),
    so whatever the caller does next can fail and still be resumed: the
    resumed step takes no update, loads the best weights and returns.
    resume=True continues from it. identity: what a resumed step must
    share with the checkpoint (the step's fingerprint, how the shards
    stream); done: the identities of steps already finished, whose
    checkpoint counts as none. A checkpoint holds the weights and the
    optimizer's, method's and environment's state; weights(state) gives
    the weights the step leaves the model with.

    The environment's stages run in order, each drawing batches of its own
    number of examples (a batch depends only on the seeds and the update);
    an optimizer's stop ends the current stage, and the next one starts.
    When a stage after the first starts, the optimizer hears of it
    (Optimizer.stage) and the best evaluation so far is forgotten.

    A stop signal (SIGINT, SIGTERM, and SIGBREAK on Windows) to any process
    lets every process finish the update; they then save the checkpoint,
    when the run keeps them, and raise Interrupted. A second signal acts as
    usual.

    telemetry (a wpt_telemetry.Telemetry) records, from the main process,
    every update (its loss, its duration, the learning rates), every
    evaluation, checkpoint, resume and stop. When the record breaks
    (RecordError from a call, log included, or a send refused in the
    background, which arrives as a stop signal), the update goes on
    unrecorded; then every process saves the checkpoint, when the run keeps
    them, and raises RecordError: the main process the record's own, the
    others one that says the main process's record broke. A record that
    breaks at the last event, once the updates are done, raises RecordError
    with no further save: the last checkpoint written stays. StepFailed
    from the method or the environment stops the step as it is."""
    from wpt_model import BALANCING  # the model's own names for its parts

    record = _Unbroken(telemetry if telemetry is not None and strategy.is_main else Telemetry.off())
    log = record.guard(log) if strategy.is_main else (lambda *_: None)
    signature = {"identity": identity}
    state = None
    if resume and checkpoint:
        # only the main process wrote it, maybe on another machine: it reads and sends it
        state = strategy.broadcast(checkpoint.load() if strategy.is_main else None)
        if state is not None and state["signature"] in [{"identity": found} for found in done]:
            state = None  # a step already finished left it
        elif state is not None and state["signature"] != signature:
            raise SettingsMismatch(f"the checkpoint in {checkpoint} was written by another step or with other "
                                   "settings")
        if state is None:
            log(f"no checkpoint of {step.name} in {checkpoint}: starting at update 1")
        else:
            environment.load_state_dict(state["environment"])
    environment.begin(model, strategy)  # with the weights the step starts from
    stages = environment.stage_list
    updates = sum(stage.steps for stage in stages)
    frozen = _freeze(model, (*step.locked, *method.unreached))
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not trainable:
            raise PluginError(f"{step.name}: its locked parts leave nothing of the model to train")
        run = Run(tokens=environment.tokens, parameters=trainable, steps=updates, log=log)
        optimizer = step.start_optimizer(model, run)
        method.begin(model, environment, run, options.segment, strategy)
        wrapped = strategy.wrap(model)
        return _loop(model, strategy, step, method, environment, optimizer, wrapped, (seed, number), options,
                     checkpoint, state, signature, log, record, save_last, stages, updates,
                     BALANCING not in step.locked)
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)


def weights(state: dict) -> dict:
    """The weights a step's checkpoint state leaves the model with: those of
    the evaluation judged best, else the last."""
    return state["best_model"] if state["best_model"] is not None else state["model"]


def _freeze(model, names) -> list:
    """The parameters of the parts named, which no longer ask for a gradient."""
    found = [] if not names else [p for name in sorted(set(names)) for p in model.parts()[name] if p.requires_grad]
    for parameter in found:
        parameter.requires_grad_(False)
    return found


def _loop(model, strategy, step, method, environment, optimizer, wrapped, draws, options, checkpoint, state,
          signature, log, record, save_last, stages, updates, balance):
    first_step, best_val, best_state, best_step, best_indexer, reached, skipped = 1, None, None, None, None, {}, 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        method.load_state_dict(state["method"])
        first_step, best_val, best_state = state["next_step"], state["best_val"], state["best_model"]
        best_step, best_indexer, reached = state["best_step"], state["best_indexer"], dict(state["reached"])
        skipped = state["skipped"]
        record.event("training.resumed", step=state["step"], checkpoint=str(checkpoint))
        log(f"{step.name}: resumed from update {state['step']}")

    def evaluate(update):
        model.eval()
        value = environment.evaluate(update, method, model, wrapped, strategy, options.segment)
        model.train()
        return value

    def _update(update, index, first, last):
        """One update, and its evaluation when one is due; True when the optimizer says stop."""
        nonlocal best_val, best_state, best_step, best_indexer, indexer, skipped
        started, stage = time.monotonic(), stages[index]
        passes = environment.batch(update, model, strategy)
        model.train()
        if passes is None:  # nothing to learn: no update
            skipped += 1
            environment.after_update(update, None, None)
            record.event("training.skipped", step=update)
            return False
        losses, indexers = [], []
        for a, examples in enumerate(passes):
            # by the pass's place in the batch, so dropout does not depend on the number of processes
            torch.manual_seed(_seed(*draws, "dropout", update, a * strategy.world_size + strategy.rank))

            def quiet(final, a=a):  # the update's gradients are shared at its very last backward pass
                return (contextlib.nullcontext() if final and a == len(passes) - 1
                        else strategy.no_sync(wrapped))
            loss, found = method.learn(model, wrapped, examples, len(passes), strategy, options.segment, quiet)
            losses.append(loss)
            indexers.append(found)
        strategy.synchronize(model)
        optimizer.before_update(update)
        optimizer.optimizer.step()
        optimizer.optimizer.zero_grad(set_to_none=True)
        if hasattr(model, "after_update"):  # the model's own updates, such as expert balancing
            model.after_update(strategy.reduce, balance)
        optimizer.after_update(update, strategy.reduce)
        method.after_update(update)
        train_loss, indexer = method.update_loss(losses, indexers, strategy)
        environment.after_update(update, train_loss, indexer)
        record.event("training.step", values=_known({"train_loss": train_loss, "indexer_loss": indexer,
                                                    "seconds": time.monotonic() - started}),
                     step=update, block_size=stage.block_size,
                     learning_rates=[group["lr"] for group in optimizer.optimizer.param_groups])

        if environment.evaluates(update, first, last):
            val_loss = evaluate(update)
            verdict = optimizer.evaluated(update, val_loss)
            if verdict.best:
                best_val, best_step, best_indexer = val_loss, update, indexer
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            record.event("training.evaluation", values={"val_loss": val_loss}, step=update, best=verdict.best,
                         stop=verdict.stop, note=verdict.note)
            log(f"update {update:5d} | train loss {train_loss:.4f}"
                + ("" if indexer is None else f" | indexer loss {indexer:.4f}")
                + f" | val loss {val_loss:.4f} | {time.time() - t0:.0f}s{verdict.note}")
            return verdict.stop
        return False

    def save(update, next_step):
        if strategy.is_main:
            with record.span("training.checkpoint", step=update, where=str(checkpoint)):
                try:
                    checkpoint.save({"signature": signature, "step": update, "next_step": next_step,
                                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                     "method": method.state_dict(), "environment": environment.state_dict(),
                                     "best_val": best_val, "best_model": best_state, "best_step": best_step,
                                     "best_indexer": best_indexer, "reached": list(reached.items()),
                                     "skipped": skipped})
                except RecordError as error:  # saved all the same; see Checkpoint.save
                    record.broken = record.broken or error
        strategy.barrier()

    keeps = checkpoint is not None and options.checkpoint_every > 0
    model.train()
    t0 = time.time()
    last, indexer, update = first_step - 1, None, first_step
    with _StopSignals(log) as signals:
        while update <= updates:
            index, stage, begin, end = stage_of(stages, update)
            if update == begin and index:  # longer windows: the best so far is no longer comparable
                best_val, best_state, best_step, best_indexer = None, None, None, None
                optimizer.stage(index + 1, stage.block_size)
            log(f"{step.name}: "
                + (f"stage {index + 1} of {len(stages)}: windows of {stage.block_size:,} tokens, "
                   if stage.block_size is not None else "")
                + f"{stage.batch_size} per update, updates {update} to {end}"
                + (f" (from {begin})" if update != begin else ""))
            environment.stage(index)
            with record.span("training.stage", stage=index + 1, block_size=stage.block_size,
                             batch_size=stage.batch_size, first_step=begin,
                             last_step=end, resumed_at=update):
                while update <= end:
                    stop = _update(update, index, begin, end)
                    last, reached[index] = update, update
                    next_step = end + 1 if stop else update + 1  # an optimizer's stop ends the stage
                    if keeps and update % options.checkpoint_every == 0:
                        save(update, next_step)
                    # every process must stop at the same update, wherever the signal came or the record broke
                    asked, broken = (float(signals.asked), float(record.broken is not None))
                    if strategy.world_size > 1:
                        flags = torch.tensor([asked, broken], device=strategy.device)
                        asked, broken = strategy.reduce(flags, "max").tolist()
                    if asked or broken:
                        if keeps and update % options.checkpoint_every:
                            save(update, next_step)
                        record.event("training.stopped", step=update, saved=keeps)
                        if record.broken is not None:
                            raise record.broken
                        if broken:
                            raise RecordError(f"the main process's record broke; stopped after update {update}")
                        raise Interrupted(update, keeps)
                    update = next_step
                    if stop:
                        log(f"stage {index + 1} ended at update {last}")
                        break
    if last >= first_step:  # a resumed step that had every update done already heard its end then
        environment.finished(last)
    saved = keeps and last % options.checkpoint_every == 0  # the last update's own save
    if save_last and checkpoint is not None and last >= first_step and not saved:
        save(last, updates + 1)  # every update is done: a resumed step takes none
    if best_state is not None:
        model.load_state_dict(best_state)
    kept = best_state is not None and best_step is not None
    found, begin = [], 1
    for index, stage in enumerate(stages):
        found.append({"block_size": stage.block_size, "steps": stage.steps, "batch_size": stage.batch_size,
                      "first_step": begin, "last_step": reached.get(index)})
        begin += stage.steps
    trained = Trained(best_val, last, best_step if kept else last,
                      best_indexer if kept else (indexer if indexer is not None else None), tuple(found), skipped)
    record.event("training.finished", values=_known({"best_validation_loss": best_val}), steps=trained.steps,
                 weights_step=trained.weights_step, stages=found, skipped=skipped)
    if record.broken is not None:  # at the last event, once training ended: the last checkpoint written stays
        raise record.broken
    return trained


def _known(values: dict) -> dict:
    """The values that are numbers: a step without evaluations has no
    validation loss, and a method without indexers no indexer loss."""
    return {name: value for name, value in values.items() if value is not None}


class _Unbroken:
    """The run's record, through which a RecordError does not stop the
    update: it is kept in `broken`, and every later call does nothing."""

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
        self.log(f"{signal.Signals(number).name}: stopping after this update (again to stop at once)")

    def __exit__(self, *exc):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        return False
