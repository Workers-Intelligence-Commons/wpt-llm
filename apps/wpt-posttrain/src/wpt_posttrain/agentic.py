"""Learning from scored attempts at agentic tasks, the machinery the `grpo`
and `token-efficiency` steps share (docs/post-training.md, "The grpo
step").

    prepare(context, tag)      the tasks tagged `tag`, their built images and the context size,
                               or Unready when the step cannot start
    frozen_routers(model)      the routed expert layers' routers, unchanged while it lasts
    written(attempt)           how many tokens the model wrote in an attempt
    Learner(...).play(task)    one attempt in a fresh container, its routes (R3) and its score
    Learner(...).train(...)    rounds of groups of attempts: advantage = reward minus the group
                               mean, the loss over the tokens the model wrote, one update per
                               TASKS_PER_UPDATE tasks; drops, skips and failures as grpo says

Each step says what an attempt's reward is: grpo uses the check's score
as it is, token-efficiency its own rule on the score and the attempt's
length. Everything else is the same for both.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import wpt_agent
from wpt_agent import images
from wpt_dataset.repo import Repo
from wpt_inference import sample_tokens
from wpt_model import Routes
from wpt_model.experts import Experts
from wpt_posttrain import Outcome
from wpt_tokenizer import chat


@dataclass(frozen=True)
class Settings:
    """A step's constants (steps/grpo.py names each one)."""
    rounds: int
    tasks_per_update: int
    group: int
    learning_rate: float
    weight_decay: float
    temperature: float
    top_k: int | None
    check_seconds: float
    result_bytes: int
    seed: int


class Unready(Exception):
    """The step cannot start: no task is tagged, a task's instructions fill
    the model's context, or an image is not built. Nothing has changed."""


def prepare(context, tag=None):
    """(tasks tagged `tag`, by default the running step's name, {image name:
    built image}, the most tokens an attempt may hold), or Unready."""
    tasks = context.tasks(tag)
    if not tasks:
        raise Unready(f"no tasks: no admitted source has task files tagged {tag or context.step}")
    limit = context.model.config.block_size + 1
    for task in tasks:
        opening = len(chat.prompt(context.tokenizer, wpt_agent.opening(task)))
        if opening >= limit:
            raise Unready(f"{task.id}: its instructions and the tools take {opening} tokens, as much as "
                          f"the model's context ({limit} with the first token)")
    try:
        root = Repo.find().root
        built = {name: images.built(root, name) for name in sorted({task.image for task in tasks})}
    except wpt_agent.ImageError as error:
        raise Unready(str(error)) from None
    return tasks, built, limit


@contextmanager
def frozen_routers(model):
    """The routers of the routed expert layers learn nothing while it lasts."""
    routers = [module.router.weight for module in model.modules() if isinstance(module, Experts) and not module.hashed]
    for weight in routers:
        weight.requires_grad_(False)
    try:
        yield
    finally:
        for weight in routers:
            weight.requires_grad_(True)


def written(attempt) -> int:
    """How many tokens the model wrote in the attempt: the tokens the loss
    learns, never the instructions, the tool results or the template's."""
    return sum(attempt.learn)


class Learner:
    """One step's attempts and updates on one model. Sampling draws from a
    generator seeded with settings.seed when the learner is made, so every
    attempt it plays, before and during train, comes from one sequence."""

    def __init__(self, name, model, tokenizer, device, built, limit, settings):
        self.name, self.model, self.tokenizer, self.device = name, model, tokenizer, device
        self.built, self.limit, self.settings = built, limit, settings
        self.sampling = torch.Generator(device=device).manual_seed(settings.seed)

    def play(self, task):
        """(attempt, its routes, score) of one attempt in a fresh container;
        raises wpt_agent.SandboxError when Docker failed."""
        routes = Routes()
        recording = _Recording(self.model, routes)
        settings = self.settings

        def generate(ids, count):  # every call starts at the attempt's first token, as Routes needs
            return sample_tokens(recording, ids, context_size=self.model.config.block_size, max_new_tokens=count,
                                 temperature=settings.temperature, top_k=settings.top_k, generator=self.sampling)

        limits = wpt_agent.Limits(**task.container)  # the task format's sizes, raised by the task (wpt_dataset.tasks)
        with wpt_agent.Sandbox(self.built[task.image], task.files, user=wpt_agent.AGENT, limits=limits) as sandbox:
            attempt = wpt_agent.attempt(task, sandbox, self.tokenizer, generate, context=self.limit,
                                        result_bytes=settings.result_bytes)
            score, why = wpt_agent.score(sandbox, task.check, settings.check_seconds)
        if score is None:
            print(f"{self.name}: {task.id}: no score, counted as 0: {why}")
        return attempt, routes, score or 0.0

    def train(self, tasks, reward=lambda task, attempt, score: score, measure="score"):
        """settings.rounds passes over the tasks; reward(task, attempt,
        score) is what an attempt is worth, and measure names it in what the
        step prints and in its failures. The model is left in eval mode."""
        model, settings = self.model, self.settings
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings.learning_rate,
                                      weight_decay=settings.weight_decay)
        order = torch.Generator().manual_seed(settings.seed)
        total = equal = small = 0  # updates, and groups skipped because their rewards were equal or they were too small
        for round_ in range(1, settings.rounds + 1):
            rewards, updates, dropped, indexers = [], 0, [], []
            permutation = torch.randperm(len(tasks), generator=order).tolist()
            for start in range(0, len(tasks), settings.tasks_per_update):
                model.eval()
                groups = []
                for task in [tasks[i] for i in permutation[start:start + settings.tasks_per_update]]:
                    group = []
                    for _ in range(settings.group):
                        try:
                            attempt, routes, score = self.play(task)
                        except wpt_agent.SandboxError as error:
                            print(f"{self.name}: {task.id}: attempt dropped, Docker failed: {error}")
                            dropped.append(error)
                            continue
                        group.append((attempt, routes, reward(task, attempt, score)))
                    rewards += [value for *_, value in group]
                    if len(group) < 2:
                        print(f"{self.name}: {task.id}: group skipped, {len(group)} attempt left")
                        small += 1
                    elif len({value for *_, value in group}) == 1:  # every advantage is 0: nothing to learn
                        equal += 1
                    else:
                        groups.append(group)
                model.train()
                loss, indexer = update(model, optimizer, groups, self.device)
                if loss is not None and not math.isfinite(loss):
                    model.eval()
                    return Outcome.failure(f"round {round_}: the loss is {loss}; lower LEARNING_RATE")
                updates += loss is not None
                if indexer is not None:
                    indexers.append(indexer)
            if not rewards:  # every attempt was dropped, so no update could be made
                model.eval()
                return Outcome.failure(f"round {round_}: every attempt was dropped, Docker failed {len(dropped)} "
                                       f"times; the last time: {dropped[-1]}")
            print(f"{self.name} round {round_}/{settings.rounds}: mean {measure} {sum(rewards) / len(rewards):.4f} "
                  f"over {len(rewards)} attempts at {len(tasks)} tasks, {updates} updates"
                  + (f", indexer loss {sum(indexers) / len(indexers):.4f}" if indexers else ""))
            total += updates
        model.eval()
        if not total:
            return Outcome.failure(f"no update in {settings.rounds} rounds: every group was skipped, {equal} because "
                                   f"its {measure}s were all equal, {small} because it kept fewer than 2 attempts")
        return Outcome.success()


class _Recording(torch.nn.Module):
    """The model, keeping the experts its routers choose in routes."""

    def __init__(self, model, routes):
        super().__init__()
        self.model, self.routes = model, routes

    def forward(self, idx):
        return self.model(idx, record=self.routes)


def update(model, optimizer, groups, device):
    """One optimizer step on the groups' attempts, given as (attempt, routes,
    reward), whose rewards differ within each group: (the loss, the mean of
    the CSA indexers' loss over its passes), or (None, None) when there is
    nothing to learn (no group, or no token written). The indexers learn by
    their own loss, as in pre-training (docs/model.md, "Attention")."""
    weighted = []
    for group in groups:
        mean = sum(value for *_, value in group) / len(group)
        weighted += [(attempt, routes, value - mean) for attempt, routes, value in group]
    count = sum(sum(attempt.learn[1:]) for attempt, _, _ in weighted)
    if count == 0 or all(advantage == 0 for *_, advantage in weighted):
        return None, None
    optimizer.zero_grad(set_to_none=True)
    total, indexers = 0.0, []
    for attempt, routes, advantage in weighted:  # one attempt at a time: the gradients add up
        if advantage == 0 or not any(attempt.learn[1:]):
            continue
        # up to the last token the model wrote: the template's tokens after it teach nothing,
        # and the model never read them, so no route was recorded for them
        end = max(i for i, learned in enumerate(attempt.learn) if learned) + 1
        inputs = torch.tensor([attempt.ids[:end - 1]], device=device)
        targets = torch.tensor([attempt.ids[1:end]], device=device)
        mask = torch.tensor([attempt.learn[1:end]], dtype=torch.float32, device=device)
        logits, indexer = model(inputs, replay=routes, indexer_loss=True)
        logits = logits.float()
        chosen = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        loss = -advantage * (chosen * mask).sum() / count
        if not math.isfinite(loss.item()):
            return loss.item(), None
        (loss + (indexer - indexer.detach())).backward()  # the indexers' gradient, not their value
        total += loss.item()
        indexers.append(indexer.item())
    optimizer.step()
    return total, sum(indexers) / len(indexers)
