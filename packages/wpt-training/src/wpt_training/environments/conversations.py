"""Conversations the dataset holds, tagged `tag` (docs/chat-format.md;
docs/training.md, "The answer-as-assistant step").

Each conversation of the dataset whose chat file carries the tag becomes
tokens with the chat template (wpt_tokenizer.chat), with the tokens the
assistant wrote marked: those are the only ones a method learns. Each
epoch goes over every conversation once, in an order drawn from the recipe's seed
and the step's number in the recipe,
`batch` conversations per update (the last update of an epoch may hold
fewer). Conversation j of an update goes to process j % world_size, and
each process takes --micro-batch of its conversations per pass (default:
all of them); a process left with fewer passes than the others makes
passes that learn nothing, so that every process makes as many.

A dataset with no conversation tagged `tag` is refused before any step
runs (check_dataset). Each conversation must fit the model's context
(block_size + 1 tokens, as text windows); a longer one fails the step,
since cutting it would drop the answers it holds, and so does one without
an assistant message.
"""

import json
import math

import torch

from wpt_tokenizer import chat
from wpt_training import Environment, Stage, StepFailed

GIVES = "conversations"
SETTINGS = {"tag": "answer-as-assistant", "epochs": 3, "batch": 8}
IGNORED = -100  # the target of a token the model does not learn: PyTorch's cross_entropy leaves it out


def check(settings):
    for key in ("epochs", "batch"):
        if settings[key] < 1:
            raise ValueError(f"{key}: at least 1")


def tagged(record, store, tag) -> list[dict]:
    """The dataset's conversations ({id, tags, conversation}) whose chat file carries tag."""
    part = record.get("conversations")
    lines = [] if part is None else store.get_bytes(part["object"]).decode("utf-8").splitlines()
    return [one for one in map(json.loads, lines) if tag in one["tags"]]


def check_dataset(settings, record, store):
    if not tagged(record, store, settings["tag"]):
        raise ValueError(f"no conversations: the dataset holds no chat file tagged {settings['tag']}")


class Conversations(Environment):
    def __init__(self, settings, data):
        self.settings, self.data = settings, data
        found = tagged(data.record, data.store, settings["tag"])
        limit = data.config.block_size + 1
        self.examples = []
        for one in found:
            ids, learn = chat.encode(data.tokenizer, one["conversation"])
            if len(ids) > limit:
                raise StepFailed(f"{one['id']}: {len(ids)} tokens, more than the model's context ({limit} with the "
                                 "first token)")
            if not any(learn[1:]):
                raise StepFailed(f"{one['id']}: no assistant message to learn from")
            self.examples.append((ids, learn))
        sources = {one["id"].split("/", 1)[0] for one in found}
        self.sources = {sid: data.record["sources"][sid] for sid in sorted(sources)}
        self.tokens = sum(len(ids) for ids, _ in self.examples)
        self.per_epoch = math.ceil(len(self.examples) / settings["batch"])
        self.stage_list = (Stage(settings["epochs"] * self.per_epoch, settings["batch"]),)
        order = torch.Generator().manual_seed(data.step_seed)
        self.orders = [torch.randperm(len(self.examples), generator=order).tolist() for _ in range(settings["epochs"])]
        self.losses, self.indexers = [], []
        data.telemetry.event(f"{data.step}.conversations", conversations=len(self.examples), tokens=self.tokens)

    def batch(self, step, model, strategy):
        epoch, index = divmod(step - 1, self.per_epoch)
        start = index * self.settings["batch"]
        chosen = [self.examples[i] for i in self.orders[epoch][start:start + self.settings["batch"]]]
        learned = sum(sum(learn[1:]) for _, learn in chosen)
        mine = chosen[strategy.rank::strategy.world_size]
        share = math.ceil(len(chosen) / strategy.world_size)
        micro = self.data.micro_batch or share
        passes = []
        for a in range(math.ceil(share / micro)):
            examples = mine[a * micro:(a + 1) * micro]
            # a pass that learns nothing: every process makes as many passes, each process's gradients shared at once
            passes.append((*_batch(examples or chosen[:1], strategy.device, learn=bool(examples)), learned,
                           1.0 if examples else 0.0))
        return passes

    def after_update(self, step, loss, indexer):
        self.losses.append(loss)
        self.indexers.append(indexer)
        epoch, index = divmod(step - 1, self.per_epoch)
        if index == self.per_epoch - 1:
            self.data.log(f"{self.data.step} epoch {epoch + 1}/{self.settings['epochs']}: mean loss "
                          f"{sum(self.losses) / len(self.losses):.4f}, indexer loss "
                          f"{sum(self.indexers) / len(self.indexers):.4f} over {len(self.examples)} conversations")
            self.losses, self.indexers = [], []

    def state_dict(self):
        return {"losses": list(self.losses), "indexers": list(self.indexers)}

    def load_state_dict(self, state):
        self.losses, self.indexers = list(state["losses"]), list(state["indexers"])


def _batch(examples, device, learn=True):
    """(inputs, targets) padded to the longest example: each target is the
    next token where the model learns it, IGNORED elsewhere (everywhere
    without learn). Padding sits after every real token, so causal
    attention never lets it change them."""
    width = max(len(ids) for ids, _ in examples) - 1
    inputs = torch.zeros(len(examples), width, dtype=torch.long)
    targets = torch.full((len(examples), width), IGNORED, dtype=torch.long)
    for row, (ids, marks) in enumerate(examples):
        inputs[row, :len(ids) - 1] = torch.tensor(ids[:-1])
        if learn:
            targets[row, :len(ids) - 1] = torch.tensor([t if keep else IGNORED
                                                        for t, keep in zip(ids[1:], marks[1:])])
    return inputs.to(device), targets.to(device)


def start(settings, data):
    return Conversations(settings, data)
