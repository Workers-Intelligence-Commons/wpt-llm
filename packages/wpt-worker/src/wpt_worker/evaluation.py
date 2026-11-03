"""wpt-eval's model as a gpu worker's job (docs/machines.md, "Evaluation"):
the part of an evaluation that needs the GPUs. The wpt manager runs the
tests, which hold the tokenizer, the store, the sandboxes and the model
folder; the job holds the model and answers what the tests ask of it,
through the link. It deals only in token ids, as every job does.

The manager's spec holds the model's config, the weight files it offers
and the form of the attention it names (spec["kernels"], one of
wpt_model.sparse.FORMS, or None); the job decides the attention's form
first, as wpt-train's does (docs/model.md, "Attention"), and a form that
cannot work here ends it with exit code 2 and the reason on its standard
error. It then downloads the files (checked against their sha256), loads the
model on the machine's best device in eval mode, tells the manager it is
ready, and then answers until the manager ends it:

    generator.new, writer.start, writer.write
                                 a seeded generator and the writers on the model
                                 (wpt_worker.writing)
    model.greedy                 the tokens the model writes greedily after each of several
                                 prompts of one length (greedy())
    model.losses                 the next-token loss of every token of several windows after
                                 the first (losses())
    eval.end                     the manager is done: the job ends with code 0

Served answers the same calls for a model in the caller's own process, with
the names the manager's side has (config, device, generator, writer,
greedy and losses): a test cannot tell where the model is.
"""

import sys
import threading

import numpy as np
from pathlib import Path

import torch
from safetensors.torch import load_file

from wpt_model import WPT, ModelConfig
from wpt_model.sparse import KernelUnavailable, kernels
from wpt_training.launch import REFUSED
from wpt_utils.device import describe, select_device

from .job import call, download
from .writing import LocalWriter, Writers, Writing


class Served:
    """A model in this process: config (the model's config), device (a
    description of where it runs), generator() (a new seeded generator: its
    number), writer() (a writer on the model, as wpt_worker.writing.RemoteWriter
    is one on a gpu worker's), greedy() and losses()."""

    def __init__(self, model):
        self.model, self.config = model, model.config
        self.device = describe(next(model.parameters()).device)
        self.writers = Writers(model)

    def generator(self, seed: int) -> int:
        return self.writers.new_generator(seed)

    def writer(self, prompt, rows=1, *, temperature, top_k, stops, routes=False, generator=None) -> LocalWriter:
        return LocalWriter(self.writers, prompt, rows, temperature=temperature, top_k=top_k, stops=stops,
                           routes=routes, generator=generator)

    def greedy(self, prompts: list[list[int]], steps: int) -> list[list[int]]:
        return greedy(self.model, prompts, steps)

    def losses(self, windows: list[list[int]]) -> list[np.ndarray]:
        return losses(self.model, windows)


@torch.no_grad()
def greedy(model, prompts: list[list[int]], steps: int) -> list[list[int]]:
    """The tokens model writes after each of prompts (token lists of one
    length), steps of them, always taking the likeliest."""
    ids = torch.tensor(prompts, device=next(model.parameters()).device)
    for _ in range(steps):
        logits = model(ids[:, -model.config.block_size:])[:, -1, :]
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[:, len(prompts[0]):].tolist()


LOSS_CHUNK = 4096  # tokens of a window read in one pass: the scores of one pass are LOSS_CHUNK x vocab_size


@torch.no_grad()
def losses(model, windows: list[list[int]]) -> list[np.ndarray]:
    """For each window (token ids), the next-token loss, in nats, of each
    of its tokens after the first, read with only the window's earlier
    tokens before it: float32, one fewer than the window. The window is
    read through the model's generation cache (docs/model.md,
    "Generation"), LOSS_CHUNK tokens at a time, so the scores held at once
    stay LOSS_CHUNK x vocab_size whatever the window's length."""
    device, found = next(model.parameters()).device, []
    for window in windows:
        ids = torch.tensor(window, dtype=torch.long, device=device)
        cache, parts = model.new_cache(1), []
        for start in range(0, len(window) - 1, LOSS_CHUNK):
            end = min(start + LOSS_CHUNK, len(window) - 1)
            scores, _ = model.extend(ids[start:end].unsqueeze(0), cache)
            parts.append(torch.nn.functional.cross_entropy(scores[0].float(), ids[start + 1:end + 1],
                                                           reduction="none"))
        found.append(torch.cat(parts).cpu().numpy().astype(np.float32) if parts else np.zeros(0, np.float32))
    return found


def run(spec, link, work: Path) -> int:
    config = ModelConfig(**spec["model"])
    device, named = select_device(), spec["kernels"]
    if device.type == "cuda" or named:  # decided once, before the model loads (docs/model.md, "Attention")
        try:
            chosen = kernels(device, named)
        except KernelUnavailable as error:
            print(f"KernelUnavailable: {error}", file=sys.stderr)
            return REFUSED
        print("attention: " + (f"{chosen.NAME} kernels" if chosen else "PyTorch (--kernels pytorch)" if named
                               else "PyTorch (neither TileLang nor Triton is installed)"))
    state = {}
    for name, offer in spec["weights"].items():
        path = download(link, offer, work / "weights" / name)
        state.update(load_file(path))
        path.unlink()
    model = WPT(config)
    model.load_state_dict(state)
    del state
    model = model.to(device)
    model.eval()
    where, done = describe(device), threading.Event()
    writing = Writing(link)

    def answer(header, body):
        return {"tokens": greedy(model, header["prompts"], header["steps"])}, b""

    def scored(header, body):
        tokens, windows, start = np.frombuffer(body, dtype="<u4"), [], 0
        for length in header["lengths"]:
            windows.append(tokens[start:start + length].tolist())
            start += length
        return {}, b"".join(found.astype("<f4").tobytes() for found in losses(model, windows))

    def end(header, body):
        done.set()

    with writing(model):
        link.handlers.update({"model.greedy": answer, "model.losses": scored, "eval.end": end})
        call(link, "ready", device=where)
        link.serve(until=done.is_set)
    return 0
