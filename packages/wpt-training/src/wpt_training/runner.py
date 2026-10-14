"""The training loop shared by every strategy and method."""

import contextlib
import hashlib
import os
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


@dataclass(frozen=True)
class RunSettings:
    block_size: int
    batch_size: int        # windows per update, over every process together
    micro_batch: int       # windows per forward pass in one process
    steps: int             # maximum number of updates
    eval_every: int        # evaluate every N steps (and at step 1)
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


def _micro_batches(tokens, settings, strategy, accumulate, *seed_parts):
    """This process's micro-batches of one global batch: micro-batch a of
    process r is slice (a * world_size + r) of the batch, so the batch holds
    the same windows whatever the number of processes."""
    batch = _windows(tokens, settings.block_size, settings.batch_size, *seed_parts)
    for a in range(accumulate):
        start = (a * strategy.world_size + strategy.rank) * settings.micro_batch
        part = batch[start:start + settings.micro_batch].to(strategy.device)
        yield a, part[:, :-1], part[:, 1:]


def train(model, strategy, method_plugin, method_settings, train_tokens, val_tokens, settings: RunSettings,
          checkpoint_folder: Path | None = None, resume=False, identity=None, log=print):
    """Train model in place, then load the weights of the evaluation the
    method judged best back into it; returns that evaluation's validation
    loss. train_tokens and val_tokens are 1-D arrays of token ids (np.memmap
    works). With checkpoint_folder and settings.checkpoint_every, the main
    process writes checkpoint.pt there every N steps; resume=True continues
    from it. identity: what a resumed run must share with the checkpoint
    besides the settings (for example the dataset id and the model shape)."""
    log = log if strategy.is_main else (lambda *_: None)
    if len(val_tokens) <= settings.block_size + 1:
        log("text is small: validation loss is measured on training data")
        val_tokens = train_tokens
    accumulate = accumulation(settings, strategy.world_size)
    parameters = sum(p.numel() for p in model.parameters())
    run = Run(tokens=len(train_tokens) + len(val_tokens), parameters=parameters, steps=settings.steps, log=log)
    method = start_method(method_plugin, model, method_settings, run)
    wrapped = strategy.wrap(model)
    checkpoint_path = checkpoint_folder / CHECKPOINT if checkpoint_folder else None
    signature = {"settings": {k: v for k, v in asdict(settings).items() if k not in ("micro_batch", "checkpoint_every")},
                 "method": [method_plugin.sha256, method_settings], "identity": identity}

    first_step, best_val, best_state = 1, float("inf"), None
    if resume and checkpoint_path:
        # only the main process wrote it, maybe on another machine: it reads and sends it
        found = checkpoint_path.is_file() if strategy.is_main else None
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if found else None
        state = strategy.broadcast(state)
        if state is None:
            log(f"no checkpoint in {checkpoint_folder}: starting at step 1")
        else:
            if state["signature"] != signature:
                raise SettingsMismatch(f"{checkpoint_path} was written by a run with other settings")
            model.load_state_dict(state["model"])
            method.load_state_dict(state["method"])
            first_step, best_val, best_state = state["step"] + 1, state["best_val"], state["best_model"]
            log(f"resumed from step {state['step']}")

    def evaluate(step):
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(settings.eval_batches):
                for _, x, y in _micro_batches(val_tokens, settings, strategy, accumulate, settings.seed, "eval", step, i):
                    losses.append(method.loss(wrapped, x, y).item())
        model.train()
        return strategy.mean(sum(losses) / len(losses))

    model.train()
    t0 = time.time()
    for step in range(first_step, settings.steps + 1):
        for a, x, y in _micro_batches(train_tokens, settings, strategy, accumulate, settings.seed, "train", step):
            # by the slice's place in the batch, so dropout does not depend on the number of processes
            torch.manual_seed(_seed(settings.seed, "dropout", step, a * strategy.world_size + strategy.rank))
            with strategy.no_sync(wrapped) if a < accumulate - 1 else contextlib.nullcontext():
                loss = method.loss(wrapped, x, y)
                (loss / accumulate).backward()
        method.before_update(model, step)
        method.optimizer.step()
        method.optimizer.zero_grad(set_to_none=True)

        stop = False
        if step % settings.eval_every == 0 or step == 1:
            val_loss = evaluate(step)
            verdict = method.evaluated(step, val_loss)
            if verdict.best:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            log(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | "
                f"{time.time() - t0:.0f}s{verdict.note}")
            stop = verdict.stop
        if checkpoint_path and settings.checkpoint_every and step % settings.checkpoint_every == 0:
            if strategy.is_main:
                _save(checkpoint_path, {"signature": signature, "step": step, "model": model.state_dict(),
                                        "method": method.state_dict(), "best_val": best_val, "best_model": best_state})
            strategy.barrier()
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def _save(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    os.replace(partial, path)
