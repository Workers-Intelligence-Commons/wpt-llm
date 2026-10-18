"""Learning from scored attempts at agentic tasks, the machinery the `grpo`
and `token-efficiency` steps share (docs/post-training.md, "The grpo
step").

    prepare(context, tag)          the tasks tagged `tag` and their built images, or Unready when
                                   the step cannot start
    frozen_routers(model)          the routed expert layers' routers, unchanged while it lasts
    written(attempt)               how many tokens the model wrote in an attempt
    Learner(...).play(task, n, k)  n attempts, each in a fresh container, written together in one
                                   batch from one reading of the task's opening (the model's
                                   generation cache), by the model alone (no drafter); then each
                                   one's routes (R3) and score, handed to k as soon as it is scored
    Learner(...).attempts(...)     play, and keep each attempt as a line of the step's attempts
                                   file as soon as it is scored
    Learner(...).train(...)        rounds of groups of attempts: advantage = reward minus the group
                                   mean, the loss over the tokens the model wrote, one update per
                                   TASKS_PER_UPDATE tasks; drops, skips and failures as grpo says

Each step says what an attempt's reward is: grpo uses the check's score
as it is, token-efficiency its own rule on the score and the attempt's
length. Everything else is the same for both.

Every attempt, a dropped one included, becomes one line of
`<step>-attempts.jsonl` in the run's record folder (context.telemetry;
docs/post-training.md, "Runs"): a conversation in the chat format
(docs/chat-format.md) whose `metadata` identifies the attempt; a dropped
attempt's conversation is only its opening (wpt_agent.opening). The
attempt's span, its
group's event and its round's event name the file and the line numbers.
A line the store refuses raises wpt_telemetry.RecordError, which stops
the command (exit code 3).
"""

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import wpt_agent
from wpt_agent import images
from wpt_dataset.repo import Repo
from wpt_inference import Writer
from wpt_model import Routes
from wpt_model.experts import Experts
from wpt_posttrain import Outcome
from wpt_telemetry import Telemetry

# the lines already in each attempts file, by (record folder, file name): a
# recipe that runs a step twice adds to the same file, and its lines go on counting
_lines: dict[tuple[str | None, str], int] = {}


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
    """The step cannot start: no task is tagged, or an image is not built.
    Nothing has changed."""


def prepare(context, tag=None):
    """(tasks tagged `tag`, by default the running step's name, {image name:
    built image}), or Unready. An attempt has no context limit: the model
    reads it through its generation cache, past block_size, until the
    task's max_tokens."""
    tasks = context.tasks(tag)
    if not tasks:
        raise Unready(f"no tasks: no admitted source has task files tagged {tag or context.step}")
    try:
        root = Repo.find().root
        built = {name: images.built(root, name) for name in sorted({task.image for task in tasks})}
    except wpt_agent.ImageError as error:
        raise Unready(str(error)) from None
    return tasks, built


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
    attempt it plays, before and during train, comes from one sequence.
    The model alone writes the attempts, never a drafter: the tokens drawn
    then depend on the seed alone, never on how fast this machine's passes
    are. attempts() keeps each attempt in telemetry's record, as a line of
    `<name>-attempts.jsonl`."""

    def __init__(self, name, model, tokenizer, device, built, settings, telemetry=None):
        self.name, self.model, self.tokenizer, self.device = name, model, tokenizer, device
        self.built, self.settings = built, settings
        self.sampling = torch.Generator(device=device).manual_seed(settings.seed)
        self.telemetry = telemetry or Telemetry.off()
        self.file = f"{name}-attempts.jsonl"

    def play(self, task, count, keep):
        """count attempts at the task, each in a fresh container, written
        together; then, attempt by attempt in order, its score and its
        container's removal, and at once keep(index, outcome), index from
        1, outcome (attempt, its routes, score) or the wpt_agent.SandboxError
        that dropped it (Docker failed). Returns what keep returned, in
        order."""
        settings = self.settings
        limits = wpt_agent.Limits(**task.container)  # the task format's sizes, raised by the task (wpt_dataset.tasks)
        outcomes, boxes, writers = {}, {}, []
        for i in range(count):
            try:
                boxes[i] = wpt_agent.Sandbox(self.built[task.image], task.files, user=wpt_agent.AGENT, limits=limits)
            except wpt_agent.SandboxError as error:
                outcomes[i] = error

        def start(ids, rows, stops):  # the opening read once, every attempt's routes recorded (R3)
            writers.append(Writer(self.model, ids, rows, temperature=settings.temperature, top_k=settings.top_k,
                                  generator=self.sampling, stops=stops, routes=True))
            return writers[-1]

        kept = []
        try:
            rows = {i: row for row, i in enumerate(boxes)}
            if boxes:
                done = wpt_agent.attempts(task, list(boxes.values()), self.tokenizer, start,
                                          result_bytes=settings.result_bytes)
                outcomes.update(zip(boxes, done))
            for i in range(count):
                outcome, why = outcomes[i], None
                if not isinstance(outcome, wpt_agent.SandboxError):
                    try:
                        score, why = wpt_agent.score(boxes[i], task.check, settings.check_seconds)
                        outcome = outcome, writers[-1].routes[rows[i]], score or 0.0
                    except wpt_agent.SandboxError as error:
                        outcome = error
                if i in boxes:
                    try:
                        boxes.pop(i).close()
                    except wpt_agent.SandboxError as error:
                        outcome = error
                if why is not None and not isinstance(outcome, wpt_agent.SandboxError):
                    print(f"{self.name}: {task.id}: no score, counted as 0: {why}")
                kept.append(keep(i + 1, outcome))  # its line, before the next attempt is scored
        finally:
            for box in boxes.values():  # left open only when an error stops the attempts
                try:
                    box.close()
                except wpt_agent.SandboxError:
                    pass  # the error that stopped them goes on
        return kept

    def attempts(self, task, round_, group, count, reward=None) -> list["Kept"]:
        """Play count attempts at the task together and keep each as a line
        of self.file, in order, as soon as it is scored. round_ is None for
        attempts made before any round (token-efficiency's reference), group
        says which group of the round (or which task), and each attempt is
        numbered from 1; reward(task, attempt, score) is what an attempt is
        worth, None for one never trained on. A dropped attempt (Docker
        failed) is kept too. RecordError when the store refuses a line."""
        def keep(index, outcome):
            with self.telemetry.span(f"{self.name}.attempt", task=task.id, round=round_, group=group,
                                     attempt=index) as span:
                if isinstance(outcome, wpt_agent.SandboxError):
                    print(f"{self.name}: {task.id}: {'reference ' if round_ is None else ''}attempt dropped, "
                          f"Docker failed: {outcome}")
                    one = Kept(0, None, None, None, None, outcome)
                else:
                    attempt, routes, score = outcome
                    one = Kept(0, attempt, routes, score, None if reward is None else reward(task, attempt, score),
                               None)
                one.line = self._keep(task, round_, group, index, one)
                span.set(file=self.file, line=one.line, dropped=one.attempt is None,
                         **({} if one.attempt is None else {"ended": one.attempt.ended, "score": one.score,
                                                            "reward": one.reward, "written": written(one.attempt)}))
            return one

        with self.telemetry.span(f"{self.name}.play", task=task.id, round=round_, group=group, attempts=count):
            return self.play(task, count, keep)

    def _keep(self, task, round_, group, index, kept) -> int:
        """Add the attempt's line to self.file, its conversation with the
        attempt's metadata; its line number, from 1."""
        attempt = kept.attempt
        metadata = {"step": self.name, "task": task.id, "round": round_, "group": group, "attempt": index,
                    "dropped": attempt is None, "error": None if kept.error is None else str(kept.error),
                    "ended": None if attempt is None else attempt.ended, "score": kept.score,
                    "reward": kept.reward, "written": None if attempt is None else written(attempt)}
        conversation = wpt_agent.opening(task) if attempt is None else attempt.conversation
        line = {**conversation, "metadata": metadata}
        self.telemetry.append_file(self.file, (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        key = (self.telemetry.folder, self.file)
        _lines[key] = _lines.get(key, 0) + 1
        return _lines[key]

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
            rewards, updates, dropped, indexers, lines = [], 0, [], [], []
            permutation = torch.randperm(len(tasks), generator=order).tolist()
            number = 0  # the round's groups, from 1
            for start in range(0, len(tasks), settings.tasks_per_update):
                model.eval()
                groups = []
                for task in [tasks[i] for i in permutation[start:start + settings.tasks_per_update]]:
                    number += 1
                    played = self.attempts(task, round_, number, settings.group, reward)
                    lines += [kept.line for kept in played]
                    dropped += [kept.error for kept in played if kept.attempt is None]
                    group = [(kept.attempt, kept.routes, kept.reward) for kept in played if kept.attempt is not None]
                    rewards += [value for *_, value in group]
                    if len(group) < 2:
                        print(f"{self.name}: {task.id}: group skipped, {len(group)} attempt left")
                        small += 1
                        outcome = "skipped: fewer than 2 attempts"
                    elif len({value for *_, value in group}) == 1:  # every advantage is 0: nothing to learn
                        equal += 1
                        outcome = f"skipped: every {measure} equal"
                    else:
                        groups.append(group)
                        outcome = "learned"
                    self.telemetry.event(f"{self.name}.group", task=task.id, round=round_, group=number,
                                         file=self.file, lines=[kept.line for kept in played], outcome=outcome)
                model.train()
                loss, indexer = update(model, optimizer, groups, self.device)
                if loss is not None and not math.isfinite(loss):
                    model.eval()
                    return Outcome.failure(f"round {round_}: the loss is {loss}; lower LEARNING_RATE")
                updates += loss is not None
                if indexer is not None:
                    indexers.append(indexer)
            mean = sum(rewards) / len(rewards) if rewards else None
            self.telemetry.event(f"{self.name}.round", values=None if mean is None else {f"mean_{measure}": mean},
                                 round=round_, file=self.file, lines=[lines[0], lines[-1]], attempts=len(lines),
                                 dropped=len(dropped), updates=updates)
            if not rewards:  # every attempt was dropped, so no update could be made
                model.eval()
                return Outcome.failure(f"round {round_}: every attempt was dropped, Docker failed {len(dropped)} "
                                       f"times; the last time: {dropped[-1]}")
            print(f"{self.name} round {round_}/{settings.rounds}: mean {measure} {mean:.4f} "
                  f"over {len(rewards)} attempts at {len(tasks)} tasks, {updates} updates"
                  + (f", indexer loss {sum(indexers) / len(indexers):.4f}" if indexers else ""))
            total += updates
        model.eval()
        if not total:
            return Outcome.failure(f"no update in {settings.rounds} rounds: every group was skipped, {equal} because "
                                   f"its {measure}s were all equal, {small} because it kept fewer than 2 attempts")
        return Outcome.success()


@dataclass
class Kept:
    """An attempt Learner.attempts played and kept: its line number in the
    learner's file, then the attempt, its routes, score and reward, or the
    error that dropped it (the others None)."""
    line: int
    attempt: wpt_agent.Attempt | None
    routes: Routes | None
    score: float | None
    reward: float | None
    error: Exception | None


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
