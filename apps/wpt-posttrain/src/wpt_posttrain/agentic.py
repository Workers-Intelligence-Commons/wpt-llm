"""Learning from scored attempts at agentic tasks, the machinery the `grpo`
and `token-efficiency` steps share (docs/post-training.md, "The grpo
step").

    WEIGHTS                        the defaults of the weights of an attempt's parts, which both
                                   steps' SETTINGS hold
    check(settings)                ValueError when weight_correct is not above weight_calibration
    prepare(context, tag)          the tasks tagged `tag`, their built images and the images' pools
                                   of tools, or wpt_agent.Unready when the step cannot start
    frozen_routers(model)          the routed expert layers' routers, unchanged while it lasts
    written(attempt)               how many tokens the model wrote in an attempt
    tool_calls(attempt)            how many tool calls it made
    combined(task, a, parts, s)    an attempt's parts weighed into one number by the step's
                                   settings s, 0 without an answer or with a broken gate
    result(attempt, score)         its score (wpt_agent.Played.score), 0 without an answer
    closeness(full, terse)         how alike two attempts' changes to /work are, from 0 to 1
    Learner(...).play(t, n, k, o)  n attempts at t through wpt_agent.play (each in a fresh
                                   container, offered its tools o, written together by the model
                                   alone), each one's routes (R3) recorded, handed to k as soon
                                   as it is scored
    Learner(...).attempts(...)     each attempt's tools chosen from a generator seeded with the
                                   step's seed, the task, the round, the group and the attempt
                                   (wpt_agent.offer), play, and keep each attempt as a line of the
                                   step's attempts file as soon as it is scored
    Learner(...).train(...)        rounds of groups of attempts: advantage = reward minus the group
                                   mean, the loss over the tokens the model wrote, one update per
                                   tasks_per_update tasks; drops, skips and failures as grpo says;
                                   with terse, a task's one-line form after its full form, in the
                                   same group

Each step says what an attempt's reward is: grpo uses combined() as it
is, token-efficiency its own rule on result(), combined() and the
attempt's length. Everything else is the same for both. A step's settings
(context.settings) are a dict holding SETTINGS's names: those of
steps/grpo.py, the weights of the parts among them (docs/post-training.md,
"Rewards").

Every attempt, a dropped one included, becomes one line of
`<step>-attempts.jsonl` in the run's record folder (context.telemetry;
docs/post-training.md, "Runs"): a conversation in the chat format
(docs/chat-format.md) whose `metadata` identifies the attempt, next to
the tools' real names and the answer the attempt's own metadata holds,
the final reply's text and the confidence the loop built from the
model's reply to its confidence question (wpt_agent.loop); a dropped
attempt's conversation is only its opening (wpt_agent.opening). The
reply to that question is written by the model, so it is learned and
counted in `written`, as every token the model writes. The
attempt's span, its
group's event and its round's event name the file and the line numbers.
A line the store refuses raises wpt_telemetry.RecordError, which stops
the command (exit code 3).
"""

import difflib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import wpt_agent
from wpt_model import Routes
from wpt_model.experts import Experts
from wpt_posttrain import Outcome
from wpt_telemetry import Telemetry

# the lines already in each attempts file, by (record folder, file name): a
# recipe that runs a step twice adds to the same file, and its lines go on counting
_lines: dict[tuple[str | None, str], int] = {}


CORRECT = "correct"  # the check's part for the task's own judgement (wpt_agent.check)

# the weights of an attempt's parts (maintainer's decision, 2026-10-23), starting values until experiments on
# the 1B model set them (#53); weight_correct stays above weight_calibration (check)
WEIGHTS = {
    "weight_correct": 1.0,
    "weight_calibration": 0.5,
    "weight_rule": 0.25,  # each rule's part: the check's other parts, and within_area
    "weight_tool_call": -0.01,  # each tool call
}


def check(settings):
    """ValueError when weight_correct is not above weight_calibration:
    stating a low confidence must never pay more than being right
    (maintainer's decision, 2026-10-23). The runner calls a step file's
    check(settings) before any step has run."""
    if settings["weight_correct"] <= settings["weight_calibration"]:
        raise ValueError(f"weight_correct ({settings['weight_correct']}) must be above weight_calibration "
                         f"({settings['weight_calibration']}), so stating a low confidence never pays more than "
                         f"being right")


def prepare(context, tag=None):
    """(tasks tagged `tag`, by default the running step's name, {image name:
    built image}, {image name: its pool of tools, data/images/<name>/tools.json}),
    or wpt_agent.Unready (wpt_agent.ready). An attempt has no context
    limit: the model reads it through its generation cache, past
    block_size, until the task's max_tokens."""
    tasks = context.tasks(tag)
    if not tasks:
        raise wpt_agent.Unready(f"no tasks: no admitted source has task files tagged {tag or context.step}")
    built, pools = wpt_agent.ready(sorted({task.image for task in tasks}))
    return tasks, built, pools


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


def tool_calls(attempt) -> int:
    """How many tool calls the model made in the attempt."""
    return sum(block["type"] == "tool_call" for message in attempt.conversation["messages"]
               if message["role"] == "assistant" for block in message["content"])


def combined(task, attempt, parts, settings, closeness=None) -> float:
    """What the attempt's parts are worth together (maintainer's decision,
    2026-10-23): 0 when the check gave no score, the attempt has no answer
    (no final reply, or no confidence from 0 to 100) or a gate of the task
    is broken (wpt_agent.gated); otherwise the weighted sum of `correct`,
    the calibration 1 - (confidence - correct)², each other part (a rule,
    such as sources_first or within_area), the number of tool calls and,
    for a terse attempt, its closeness to the full attempt, weighed by
    settings' weight_correct, weight_calibration, weight_rule,
    weight_tool_call and weight_closeness."""
    answer = attempt.conversation.get("metadata", {}).get("answer")
    if parts is None or answer is None or not wpt_agent.gated(task, parts):
        return 0.0
    correct = parts[CORRECT]
    value = (settings["weight_correct"] * correct
             + settings["weight_calibration"] * (1 - (answer["confidence"] - correct) ** 2)
             + settings["weight_rule"] * sum(part for name, part in parts.items() if name != CORRECT)
             + settings["weight_tool_call"] * tool_calls(attempt))
    return value if closeness is None else value + settings["weight_closeness"] * closeness


def result(attempt, score) -> float:
    """The attempt's score (wpt_agent.Played.score: its correct part, 0 with
    a broken gate or no score from the check), 0 when it has no answer, as
    combined() counts it."""
    return score if attempt.conversation.get("metadata", {}).get("answer") is not None else 0.0


def closeness(full: str, terse: str) -> float:
    """How alike two attempts' changes to /work are (wpt_agent.changes), from
    0 to 1: the share of their lines in common, in order
    (difflib.SequenceMatcher's ratio); 1 when neither changed anything."""
    return difflib.SequenceMatcher(None, full.splitlines(), terse.splitlines(), autojunk=False).ratio()


class Learner:
    """One step's attempts and updates on one model. Sampling draws from a
    generator seeded with settings["seed"] when the learner is made, so
    every attempt it plays, before and during train, comes from one
    sequence. The model alone writes the attempts, never a drafter: the
    tokens drawn then depend on the seed alone, never on how fast this
    machine's passes are. attempts() keeps each attempt in telemetry's
    record, as a line of `<name>-attempts.jsonl`. pools: {image name: its
    pool of tools}. With terse, train plays a task that has a one-line form
    in both forms (maintainer's decision, 2026-10-23)."""

    def __init__(self, name, model, tokenizer, device, built, settings, telemetry=None, pools=None, terse=False):
        self.name, self.model, self.tokenizer, self.device = name, model, tokenizer, device
        self.built, self.settings, self.pools, self.terse = built, settings, pools or {}, terse
        self.sampling = torch.Generator(device=device).manual_seed(settings["seed"])
        self.telemetry = telemetry or Telemetry.off()
        self.file = f"{name}-attempts.jsonl"

    def play(self, task, count, keep, offers, terse=False, with_changes=False):
        """count attempts at the task, each offered its tools (offers, one
        wpt_agent.Offer each), opened with its instructions or, with terse,
        its one-line form, through wpt_agent.play with the step's sampling,
        check seconds and result bytes, each one's routes recorded (with
        with_changes, its changes to /work too); keep(index, outcome), index
        from 1, outcome a wpt_agent.Played or the wpt_agent.SandboxError
        that dropped it. Returns what keep returned, in order."""
        settings = self.settings
        sampling = wpt_agent.Sampling(settings["temperature"], settings["top_k"] or None)  # top_k 0: every token
        return wpt_agent.play(task, offers[:count], keep, model=self.model, tokenizer=self.tokenizer,
                              built=self.built, sampling=sampling, generator=self.sampling,
                              check_seconds=settings["check_seconds"], result_bytes=settings["result_bytes"],
                              routes=True, with_changes=with_changes, terse=terse, name=self.name)

    def attempts(self, task, round_, group, count, reward=None, terse=False, with_changes=False) -> list["Kept"]:
        """Play count attempts at the task together, in its full or (terse)
        its one-line form, and keep each as a line of self.file, in order,
        as soon as it is scored. round_ is None for attempts made before any
        round (token-efficiency's reference), group says which group of the
        round (or which task), and each attempt is numbered from 1, offered
        the same tools in both forms; reward(task, index, played) is what an
        attempt is worth, None for one never trained on. A dropped attempt
        (Docker failed) is kept too. RecordError when the store refuses a
        line."""
        offers = [wpt_agent.offer(task, self.pools.get(task.image, []),
                                  random.Random(f"{self.settings['seed']}/{task.id}/{round_}/{group}/{index}"))
                  for index in range(1, count + 1)]

        def keep(index, outcome):
            with self.telemetry.span(f"{self.name}.attempt", task=task.id, round=round_, group=group,
                                     attempt=index, terse=terse) as span:
                if isinstance(outcome, wpt_agent.SandboxError):
                    print(f"{self.name}: {task.id}: {'reference ' if round_ is None else ''}attempt dropped, "
                          f"Docker failed: {outcome}")
                    one = Kept(0, None, None, None, None, outcome)
                else:
                    one = Kept(0, outcome.attempt, outcome.routes, outcome.score,
                               None if reward is None else reward(task, index, outcome), None, outcome.parts,
                               outcome.changes)
                one.line = self._keep(task, round_, group, index, one, offers[index - 1], terse)
                span.set(file=self.file, line=one.line, dropped=one.attempt is None,
                         **({} if one.attempt is None else {"ended": one.attempt.ended, "score": one.score,
                                                            "reward": one.reward, "written": written(one.attempt)}))
            return one

        with self.telemetry.span(f"{self.name}.play", task=task.id, round=round_, group=group, attempts=count,
                                 terse=terse):
            return self.play(task, count, keep, offers, terse, with_changes)

    def _keep(self, task, round_, group, index, kept, offer, terse=False) -> int:
        """Add the attempt's line to self.file, its conversation (a dropped
        attempt's opening, with the tools it was offered) with the attempt's
        metadata added to its own; its line number, from 1."""
        attempt = kept.attempt
        metadata = {"step": self.name, "task": task.id, "round": round_, "group": group, "attempt": index,
                    "terse": terse, "dropped": attempt is None,
                    "error": None if kept.error is None else str(kept.error),
                    "ended": None if attempt is None else attempt.ended, "parts": kept.parts, "score": kept.score,
                    "reward": kept.reward, "written": None if attempt is None else written(attempt),
                    "tool_calls": None if attempt is None else tool_calls(attempt)}
        conversation = wpt_agent.opening(task, offer, terse) if attempt is None else attempt.conversation
        line = {**conversation, "metadata": {**conversation["metadata"], **metadata}}
        self.telemetry.append_file(self.file, (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        key = (self.telemetry.folder, self.file)
        _lines[key] = _lines.get(key, 0) + 1
        return _lines[key]

    def train(self, tasks, reward=None, measure="reward"):
        """settings["rounds"] passes over the tasks; reward(task, index,
        played) is what an attempt is worth, by default combined(), and
        measure names it in what the step prints and in its failures. With
        self.terse, a task with a one-line form is played in its full form,
        then in its one-line form, and both forms' attempts make one group
        (maintainer's decision, 2026-10-23): each terse attempt's reward
        adds, weighed, its closeness to the full attempt of the same index,
        except on a task whose area is empty (it answers only; maintainer's
        decision, 2026-10-23), and, where closeness is part of the reward, a
        terse attempt whose full attempt Docker dropped is kept but not
        trained on: its reward would lack the part every other terse attempt
        has. The model is left in eval mode."""
        model, settings = self.model, self.settings
        if reward is None:
            def reward(task, index, played):
                return combined(task, played.attempt, played.parts, settings)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        order = torch.Generator().manual_seed(settings["seed"])
        total = equal = small = 0  # updates, and groups skipped because their rewards were equal or they were too small
        for round_ in range(1, settings["rounds"] + 1):
            rewards, updates, dropped, indexers, lines = [], 0, [], [], []
            permutation = torch.randperm(len(tasks), generator=order).tolist()
            number = 0  # the round's groups, from 1
            for start in range(0, len(tasks), settings["tasks_per_update"]):
                model.eval()
                groups = []
                for task in [tasks[i] for i in permutation[start:start + settings["tasks_per_update"]]]:
                    number += 1
                    both = self.terse and bool(task.terse)
                    with_changes = both and task.area != []  # an answer-only task gets no closeness
                    played = self.attempts(task, round_, number, settings["group"], reward, with_changes=with_changes)
                    if both:
                        def terse_reward(task, index, terse, full=played, with_changes=with_changes):
                            reference = full[index - 1]
                            if not with_changes:
                                return combined(task, terse.attempt, terse.parts, settings)
                            if reference.attempt is None:  # its full attempt was dropped: nothing to compare with
                                return None
                            near = closeness(reference.changes, terse.changes)
                            return combined(task, terse.attempt, terse.parts, settings, near)

                        played = played + self.attempts(task, round_, number, settings["group"], terse_reward,
                                                        terse=True, with_changes=with_changes)
                    lines += [kept.line for kept in played]
                    dropped += [kept.error for kept in played if kept.attempt is None]
                    group = [(kept.attempt, kept.routes, kept.reward) for kept in played if kept.reward is not None]
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
                                         terse=both, file=self.file, lines=[kept.line for kept in played],
                                         outcome=outcome)
                model.train()
                loss, indexer = update(model, optimizer, groups, self.device)
                if loss is not None and not math.isfinite(loss):
                    model.eval()
                    return Outcome.failure(f"round {round_}: the loss is {loss}; lower learning_rate")
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
            print(f"{self.name} round {round_}/{settings['rounds']}: mean {measure} {mean:.4f} "
                  f"over {len(rewards)} attempts at {len(tasks)} tasks, {updates} updates"
                  + (f", indexer loss {sum(indexers) / len(indexers):.4f}" if indexers else ""))
            total += updates
        model.eval()
        if not total:
            return Outcome.failure(f"no update in {settings['rounds']} rounds: every group was skipped, {equal} "
                                   f"because its {measure}s were all equal, {small} because it kept fewer than 2 "
                                   f"attempts")
        return Outcome.success()


@dataclass
class Kept:
    """An attempt Learner.attempts played and kept: its line number in the
    learner's file, then the attempt, its routes, score and reward, or the
    error that dropped it (the others None), and its parts and changes to
    /work (wpt_agent.Played)."""
    line: int
    attempt: wpt_agent.Attempt | None
    routes: Routes | None
    score: float | None
    reward: float | None
    error: Exception | None
    parts: dict | None = None
    changes: str | None = None


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
