"""wpt-eval's model as a gpu worker's job (docs/machines.md, "Evaluation"):
the part of an evaluation that needs the GPUs. The wpt manager runs the
tests, which hold the tokenizer, the store, the sandboxes and the model
folder; the job holds the model and answers what the tests ask of it,
through the link. It deals only in token ids, as every job does.

The manager's spec holds the model's config and the weight files it
offers; the job downloads them (checked against their sha256), loads the
model on the machine's best device in eval mode, tells the manager it is
ready, and then answers until the manager ends it:

    generator.new, writer.start, writer.write
                                 a seeded generator and the writers on the model
                                 (wpt_worker.writing)
    model.greedy                 the tokens the model writes greedily after each of several
                                 prompts of one length (greedy())
    eval.end                     the manager is done: the job ends with code 0

Served answers the same calls for a model in the caller's own process, with
the names the manager's side has (config, device, generator, writer and
greedy): a test cannot tell where the model is.
"""

import threading
from pathlib import Path

import torch
from safetensors.torch import load_file

from wpt_model import WPT, ModelConfig
from wpt_utils.device import describe, select_device

from .job import call, download
from .writing import LocalWriter, Writers, Writing


class Served:
    """A model in this process: config (the model's config), device (a
    description of where it runs), generator() (a new seeded generator: its
    number), writer() (a writer on the model, as wpt_worker.writing.RemoteWriter
    is one on a gpu worker's) and greedy()."""

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


@torch.no_grad()
def greedy(model, prompts: list[list[int]], steps: int) -> list[list[int]]:
    """The tokens model writes after each of prompts (token lists of one
    length), steps of them, always taking the likeliest."""
    ids = torch.tensor(prompts, device=next(model.parameters()).device)
    for _ in range(steps):
        logits = model(ids[:, -model.config.block_size:])[:, -1, :]
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[:, len(prompts[0]):].tolist()


def run(spec, link, work: Path) -> int:
    config = ModelConfig(**spec["model"])
    state = {}
    for name, offer in spec["weights"].items():
        path = download(link, offer, work / "weights" / name)
        state.update(load_file(path))
        path.unlink()
    model = WPT(config)
    model.load_state_dict(state)
    del state
    model = model.to(select_device())
    model.eval()
    device, done = describe(next(model.parameters()).device), threading.Event()
    writing = Writing(link)

    def answer(header, body):
        return {"tokens": greedy(model, header["prompts"], header["steps"])}, b""

    def end(header, body):
        done.set()

    with writing(model):
        link.handlers.update({"model.greedy": answer, "eval.end": end})
        call(link, "ready", device=device)
        link.serve(until=done.is_set)
    return 0
