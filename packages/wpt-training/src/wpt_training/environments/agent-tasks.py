"""Attempts at the agentic tasks the dataset holds, tagged `tag`, each
scored by its task's check (docs/training.md, "The finish-tasks step").
A dataset with no task tagged `tag` is refused before any step runs
(check_dataset).

Each task's image must be built where the wpt manager's sandboxes run
first (`wpt-agent build-image NAME`, on the Docker host DOCKER_HOST names,
the manager's machine by default): the manager plays the attempts, its
gpu worker's model writes their tokens (data.writing; the manager turns
text into tokens and back, the worker deals only in token ids), and every
process of the worker then trains on its share of them. For each task the
model makes `group` attempts, each in a fresh container of the task's
image with no network and limited sizes (wpt_agent), written together in
one batch from one reading of what the attempts' openings share (the
model's generation cache), each attempt offered its own choice of the
task's tools (wpt_agent.offer, seeded with the step's seed): it writes a turn,
its tool calls run in the container as a non-root user, their results
come back as the next turn, until it stops or a limit is reached. Then the
attempt's processes are stopped and the task's check scores a copy of the
attempt's /work in a fresh container, next to the attempt's conversation,
as named parts from 0 to 1 (`correct` and the rules it reports; a task
with an area also gets within_area); a check that prints no such score
counts as 0.

An attempt's reward, with reward "weighed", is its parts weighed together
(agentic.combined; docs/training.md, "Rewards"): 0 without an answer or
with a broken gate of the task, else the weighted sum of correct, the
calibration of its confidence (1 - (confidence - correct)²), each rule,
and its number of tool calls, each weighed by its weight_ setting. With
terse, a task with a one-line form is played twice in each round: `group`
attempts in its full form, then `group` in its one-line form, all of them
one group; a terse attempt's reward adds, weighed by weight_closeness, its
closeness to the full attempt of the same index (how alike their changes
to /work are; none on a task whose area is empty, which only answers), so
the model's own full result is its reference, and, where closeness
counts, a terse attempt whose full attempt was dropped is not trained on.

With reward "shorter", the model first makes `reference` attempts at each
task, before any update, scored the same way and never trained on. An
attempt's score here is its result (agentic.result): its correct part, 0
with a broken gate or without an answer. A task's best start score is the
best score among them; an attempt "passes" when its score is at least
that best score, and the task's reference length is the mean length of
its passing attempts (a length counts only the tokens the model wrote).
A task whose every reference attempt scored 0, or was dropped, is left
out, with a line for each; no task left fails the step. An attempt's
reward is then 0 when its score is below the task's best start score, and
otherwise its weighed parts times the task's reference length divided by
its own length: every passing attempt whose weighed parts are above 0
beats every failing one, and of two passing attempts with the same weighed
parts the shorter one wins. The reference is kept with the step's state,
so a resumed step does not play it again.

The rounds: `rounds` passes over the tasks, in an order drawn from the step's seed
for each round, `tasks_per_update` tasks per update. Each attempt's
advantage is its reward minus the mean reward of its group, not divided by
the group's standard deviation (which would make a group whose scores
barely differ weigh as much as one whose scores differ widely). A group
whose rewards are all equal teaches nothing and is skipped, and so is a
group left with fewer than 2 attempts. When Docker fails, which is not the
model's doing, the attempt is dropped from its group; a round whose every
attempt was dropped fails the step, and so does a step that made no
update. An update with nothing to learn is skipped.

An attempt ends at the latest at the task's max_tokens, past the model's
block_size if need be: the model reads it through its generation cache.
The model alone writes the attempts, without a drafter. Sampling draws
from one generator on the gpu worker seeded with the step's seed, for the
reference and then the rounds, so a step gives the same attempts again on
the same machine and setup. Every attempt, a dropped one included, is kept
as one line of `<step>-attempts.jsonl` in the run's record folder
(wpt_training.agentic).

Two parts: AgentTasks, on the wpt manager, plays and scores the attempts
and keeps the rounds' account; Attempts, on the gpu worker, holds the
sampling generator and turns what the manager sends (token ids, the marks
of the tokens the model wrote, each attempt's routes as the worker's own
handle, its advantage) into the passes the method learns from.
"""

import json
import math

import torch

from wpt_model import Routes
from wpt_training import Environment, Stage, StepFailed, agentic

GIVES = "attempts"
REWARDS = ("weighed", "shorter")
SETTINGS = {
    "tag": "finish-tasks",  # the tasks it plays
    "rounds": 10,  # passes over every task
    "tasks_per_update": 4,
    "group": 8,  # attempts per task, in each of its forms
    "terse": False,  # a task with a one-line form is played in both forms, in one group
    "reward": "weighed",  # or "shorter": the reference attempts, then the weighed parts times the reference length
    "reference": 0,  # with reward "shorter": attempts per task before any update
    "temperature": 1.0,  # sampling from the model's own probabilities, which the loss assumes
    "top_k": 0,  # 0: every token
    "check_seconds": 300.0,  # how long the check may run
    "result_bytes": 4000,  # a longer tool result keeps its first and last 2000 bytes
    **agentic.WEIGHTS,  # the weights of an attempt's parts
    "weight_closeness": 0.25,  # a terse attempt's closeness to the full attempt of the same index
}


def check(settings):
    """ValueError on settings that cannot go together."""
    agentic.check(settings)
    if settings["reward"] not in REWARDS:
        raise ValueError(f"reward {settings['reward']!r}: one of {', '.join(REWARDS)}")
    if settings["reward"] == "shorter" and settings["reference"] < 1:
        raise ValueError("reward \"shorter\" needs reference attempts: set reference to 1 or more")
    if settings["reward"] == "shorter" and settings["terse"]:
        raise ValueError("reward \"shorter\" plays each task in its full form only: set terse to false")
    for key in ("rounds", "tasks_per_update", "group"):
        if settings[key] < 1:
            raise ValueError(f"{key}: at least 1")


def tagged(record, store, tag) -> list[dict]:
    """The dataset's tasks (tasks.to_json) whose task.json carries tag."""
    part = record.get("tasks")
    lines = [] if part is None else store.get_bytes(part["object"]).decode("utf-8").splitlines()
    return [found for found in map(json.loads, lines) if tag in found["tags"]]


def check_dataset(settings, record, store):
    if not tagged(record, store, settings["tag"]):
        raise ValueError(f"no tasks: the dataset holds no task tagged {settings['tag']}")


class AgentTasks:
    """The environment's part on the wpt manager: the tasks, the built
    images, the reference and the rounds' account. Its methods are what
    the part on the gpu worker calls (data.call): opening, begin, play,
    after_update, finished, state and load_state."""

    def __init__(self, settings, context):
        import wpt_agent  # the manager's own: the sandboxes run where it reaches Docker
        from wpt_dataset import tasks as task_files

        self.settings, self.data = settings, context
        self.tasks = [task_files.from_json(found) for found in tagged(context.record, context.store, settings["tag"])]
        found = {task.id.split("/", 1)[0] for task in self.tasks} | {sid for task in self.tasks
                                                                      for sid, _, _ in task.documents}
        self.sources = {sid: context.record["sources"][sid] for sid in sorted(found)}
        try:
            self.built, self.pools = wpt_agent.ready(sorted({task.image for task in self.tasks}))
        except wpt_agent.Unready as unready:
            raise StepFailed(str(unready)) from None
        self.references = None  # {task id: (best start score, reference length)}, with reward "shorter"
        self.lines = {"folder": context.telemetry.folder and str(context.telemetry.folder), "count": 0}
        self.round = _Round()
        self.updates = self.equal = self.small = 0
        self.ending = None

    def opening(self) -> dict:
        """What the worker's part needs before its step starts: the sources of the tasks."""
        return {"sources": self.sources}

    def begin(self) -> dict:
        """The reference, with reward "shorter", played once (a resumed step
        has it in its state), the tasks kept and the rounds' orders: {"steps":
        the step's updates, "batch": tasks per update}."""
        if self.settings["reward"] == "shorter" and self.references is None:
            self.references = self._references()
        kept = self.tasks if self.references is None else [task for task in self.tasks if task.id in self.references]
        if not kept:
            raise StepFailed(f"no task left of the {len(self.tasks)} tasks tagged {self.settings['tag']}: each "
                             "scored 0 on every reference attempt, or Docker failed every one")
        self.kept = kept
        self.per_round = math.ceil(len(kept) / self.settings["tasks_per_update"])
        order = torch.Generator().manual_seed(self.data.step_seed)
        self.orders = [torch.randperm(len(kept), generator=order).tolist() for _ in range(self.settings["rounds"])]
        return {"steps": self.settings["rounds"] * self.per_round, "batch": self.settings["tasks_per_update"]}

    def _player(self):
        same = self.lines["folder"] == (self.data.telemetry.folder and str(self.data.telemetry.folder))
        return agentic.Player(self.data.step, self.data.writer, self.data.tokenizer, self.built, self.pools,
                              self.settings, self.data.step_seed, self.data.telemetry,
                              self.lines["count"] if same else 0)

    def _references(self) -> dict:
        """{task id: (best start score, reference length)} of the tasks kept."""
        player, references = self._player(), {}
        for number, task in enumerate(self.tasks, start=1):
            scored = []  # (score, length) of the reference attempts Docker did not drop
            for kept in player.attempts(task, None, number, self.settings["reference"]):  # never trained on
                if kept.attempt is not None:
                    scored.append((agentic.result(kept.attempt, kept.score), agentic.written(kept.attempt)))
            best = max((score for score, _ in scored), default=0.0)
            if best <= 0:
                self.data.log(f"{self.data.step}: {task.id}: left out, "
                              + (f"its {len(scored)} reference attempts all scored 0" if scored
                                 else "Docker failed every reference attempt"))
                continue
            lengths = [length for score, length in scored if score >= best]
            references[task.id] = (best, sum(lengths) / len(lengths))
            self.data.log(f"{self.data.step}: {task.id}: best start score {best:.4f}, reference length "
                          f"{references[task.id][1]:.1f} tokens")
        self._count(player)
        return references

    def _count(self, player):
        self.lines = {"folder": self.data.telemetry.folder and str(self.data.telemetry.folder), "count": player.lines}

    def _reward(self, task, index, played):
        weighed = agentic.combined(task, played.attempt, played.parts, self.settings)
        if self.references is None:
            return weighed
        best, length = self.references[task.id]
        score = agentic.result(played.attempt, played.score)
        return 0.0 if score < best else weighed * length / agentic.written(played.attempt)

    def play(self, step) -> dict:
        """The attempts of update `step`, and what the worker trains on:
        {"items": [(ids, learn, routes, advantage)], the attempts to learn
        from, each one's routes the worker's handle, "count": the tokens the
        model wrote in the update's attempts, "ending": the round's summary
        when this update ends it, else None}."""
        settings, round_ = self.settings, self.round
        number, index = divmod(step - 1, self.per_round)
        chunk = [self.kept[i] for i in self.orders[number][index * settings["tasks_per_update"]:
                                                           (index + 1) * settings["tasks_per_update"]]]
        player, groups = self._player(), []
        for position, task in enumerate(chunk):
            group_number = index * settings["tasks_per_update"] + position + 1
            both = settings["terse"] and bool(task.terse)
            with_changes = both and task.area != []  # an answer-only task gets no closeness
            played = player.attempts(task, number + 1, group_number, settings["group"], self._reward,
                                     with_changes=with_changes)
            if both:
                def terse_reward(task, index, terse, full=played, with_changes=with_changes):
                    reference = full[index - 1]
                    if not with_changes:
                        return agentic.combined(task, terse.attempt, terse.parts, settings)
                    if reference.attempt is None:  # its full attempt was dropped: nothing to compare with
                        return None
                    near = agentic.closeness(reference.changes, terse.changes)
                    return agentic.combined(task, terse.attempt, terse.parts, settings, near)

                played = played + player.attempts(task, number + 1, group_number, settings["group"], terse_reward,
                                                  terse=True, with_changes=with_changes)
            round_.lines += [kept.line for kept in played]
            round_.dropped += [str(kept.error) for kept in played if kept.attempt is None]
            group = [kept for kept in played if kept.reward is not None]
            round_.rewards += [kept.reward for kept in group]
            if len(group) < 2:
                self.data.log(f"{self.data.step}: {task.id}: group skipped, {len(group)} attempt left")
                self.small += 1
                outcome = "skipped: fewer than 2 attempts"
            elif len({kept.reward for kept in group}) == 1:  # every advantage is 0: nothing to learn
                self.equal += 1
                outcome = "skipped: every reward equal"
            else:
                groups.append(group)
                outcome = "learned"
            self.data.telemetry.event(f"{self.data.step}.group", task=task.id, round=number + 1,
                                      group=group_number, terse=both, file=player.file,
                                      lines=[kept.line for kept in played], outcome=outcome)
        self._count(player)
        weighted = []
        for group in groups:
            mean = sum(kept.reward for kept in group) / len(group)
            weighted += [(kept, kept.reward - mean) for kept in group]
        count = sum(sum(kept.attempt.learn[1:]) for kept, _ in weighted)
        items = [] if count == 0 else [
            (kept.attempt.ids, kept.attempt.learn, kept.routes, advantage)
            for kept, advantage in weighted if advantage != 0 and any(kept.attempt.learn[1:])]
        ending = None
        if index == self.per_round - 1:
            ending = {"round": number + 1, "file": player.file, "rewards": len(round_.rewards),
                      "mean": sum(round_.rewards) / len(round_.rewards) if round_.rewards else None,
                      "lines": [round_.lines[0], round_.lines[-1]], "attempts": len(round_.lines),
                      "dropped": len(round_.dropped), "last_dropped": round_.dropped[-1] if round_.dropped else None}
        self.ending = ending
        return {"items": items, "count": count, "ending": ending}

    def after_update(self, step, loss, indexer):
        round_ = self.round
        if loss is not None:
            self.updates += 1
            round_.updates += 1
            if indexer is not None:
                round_.indexers.append(indexer)
        ending, self.ending = self.ending, None
        if ending is None:
            return
        self.data.telemetry.event(f"{self.data.step}.round",
                                  values=None if ending["mean"] is None else {"mean_reward": ending["mean"]},
                                  round=ending["round"], file=ending["file"], lines=ending["lines"],
                                  attempts=ending["attempts"], dropped=ending["dropped"], updates=round_.updates)
        if not ending["rewards"]:  # every attempt was dropped, so no update could be made
            raise StepFailed(f"round {ending['round']}: every attempt was dropped, Docker failed {ending['dropped']} "
                             f"times; the last time: {ending['last_dropped']}")
        self.data.log(f"{self.data.step} round {ending['round']}/{self.settings['rounds']}: mean reward "
                      f"{ending['mean']:.4f} over {ending['rewards']} attempts at {len(self.kept)} tasks, "
                      f"{round_.updates} updates"
                      + (f", indexer loss {sum(round_.indexers) / len(round_.indexers):.4f}" if round_.indexers
                         else ""))
        self.round = _Round()

    def finished(self, last):
        if not self.updates:
            raise StepFailed(f"no update in {self.settings['rounds']} rounds: every group was skipped, {self.equal} "
                             f"because its rewards were all equal, {self.small} because it kept fewer than 2 "
                             "attempts")

    def state(self) -> dict:
        return {"references": self.references, "lines": dict(self.lines), "round": vars(self.round),
                "updates": self.updates, "equal": self.equal, "small": self.small}

    def load_state(self, state):
        self.references = None if state["references"] is None else {
            task: tuple(found) for task, found in state["references"].items()}
        self.lines = dict(state["lines"])
        self.round = _Round(**state["round"])
        self.updates, self.equal, self.small = state["updates"], state["equal"], state["small"]


def manager(settings, context):
    return AgentTasks(settings, context)


class Attempts(Environment):
    """The environment's part on the gpu worker: the sampling generator,
    seeded with the step's seed, and the passes. Its main process asks the
    manager's part and shares the answer with the other processes; a
    StepFailed there fails the step in every process."""

    def __init__(self, settings, data):
        self.settings, self.data, strategy = settings, data, data.strategy
        self.sources = self._ask("opening")["sources"]
        self.sampling = torch.Generator(device=strategy.device).manual_seed(data.step_seed)

    def _ask(self, name, *arguments, model=None):
        """The manager's answer to name(*arguments), in every process; with
        model, the manager's writers write with it meanwhile (and the answer's
        items get their routes back)."""
        strategy, found = self.data.strategy, None
        if strategy.is_main:
            try:
                if model is None:
                    found = ("answer", self.data.call(name, *arguments))
                else:
                    model.eval()
                    with self.data.writing(model, self.sampling) as writing:
                        answer = self.data.call(name, *arguments)
                        for item in answer.get("items", []):
                            item[2] = routes_out(model, writing.routes(item[2]))
                    found = ("answer", answer)
            except StepFailed as error:
                found = ("failed", error.reason)
        kind, value = strategy.broadcast(found)
        if kind == "failed":
            raise StepFailed(value)
        return value

    def begin(self, model, strategy):
        found = self._ask("begin", model=model)
        self.stage_list = (Stage(found["steps"], found["batch"]),)

    def batch(self, step, model, strategy):
        payload = self._ask("play", step, model=model)
        items = [tuple(item) for item in payload["items"]]
        if not items:
            return None
        mine = items[strategy.rank::strategy.world_size]
        passes = []
        for a in range(math.ceil(len(items) / strategy.world_size)):
            # a pass that learns nothing: every process makes as many passes
            ids, learn, routes, advantage, weight = mine[a] + (1.0,) if a < len(mine) else items[0] + (0.0,)
            end = max(i for i, learned in enumerate(learn) if learned) + 1
            device = strategy.device
            passes.append((torch.tensor([ids[:end - 1]], device=device), torch.tensor([ids[1:end]], device=device),
                           torch.tensor([learn[1:end]], dtype=torch.float32, device=device),
                           routes_in(model, routes, device), advantage, payload["count"], weight))
        return passes

    def after_update(self, step, loss, indexer):
        self._ask("after_update", step, loss, indexer)

    def finished(self, last):
        self._ask("finished", last)

    def state_dict(self):
        return {"sampling": self.sampling.get_state(), "manager": self.data.call("state")}  # the main process's

    def load_state_dict(self, state):
        self.sampling.set_state(state["sampling"])
        if self.data.strategy.is_main:
            self.data.call("load_state", state["manager"])


def routes_out(model, routes: Routes) -> dict:
    """{index of the layer: its recorded choices, on the CPU}: the routes of
    an attempt, without the model's modules, to send to another process."""
    return {index: routes.chosen[layer.experts].cpu() for index, layer in enumerate(model.layers)
            if layer.experts in routes.chosen}


def routes_in(model, found: dict, device) -> Routes:
    """The Routes routes_out gave, on this process's model."""
    routes = Routes()
    routes.chosen = {model.layers[index].experts: chosen.to(device) for index, chosen in found.items()}
    return routes


class _Round:
    """What the main process gathers over a round: its attempts' rewards,
    their lines, the errors that dropped some, its updates and their
    indexer losses."""

    def __init__(self, rewards=(), lines=(), dropped=(), updates=0, indexers=()):
        self.rewards, self.lines, self.dropped = list(rewards), list(lines), list(dropped)
        self.updates, self.indexers = updates, list(indexers)


def start(settings, data):
    return Attempts(settings, data)
