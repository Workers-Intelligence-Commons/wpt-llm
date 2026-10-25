"""Scored attempts at agentic tasks: what the agent-tasks environment and
the steps that train on tasks share (docs/training.md, "The finish-tasks
step").

    WEIGHTS                        the defaults of the weights of an attempt's parts, which the
                                   agent-tasks environment's SETTINGS hold
    check(settings)                ValueError when weight_correct is not above weight_calibration
    written(attempt)               how many tokens the model wrote in an attempt
    tool_calls(attempt)            how many tool calls it made
    combined(task, a, parts, s)    an attempt's parts weighed into one number by the step's
                                   settings s, 0 without an answer or with a broken gate
    result(attempt, score)         its score (wpt_agent.Played.score), 0 without an answer
    closeness(full, terse)         how alike two attempts' changes to /work are, from 0 to 1
    Player(...).attempts(...)      count attempts at a task through wpt_agent.play (each in a
                                   fresh container, offered its tools, written together by the
                                   model alone, each one's routes (R3) recorded), each kept as a
                                   line of the step's attempts file as soon as it is scored
    routes_out(model, routes)      an attempt's routes as {layer index: tensor}, which pickle
    routes_in(model, found)        the wpt_model.Routes routes_out gave, on another process's model

Every attempt, a dropped one included, becomes one line of
`<step>-attempts.jsonl` in the run's record folder (docs/observability.md):
a conversation in the chat format (docs/chat-format.md) whose `metadata`
identifies the attempt, next to the tools' real names and the answer the
attempt's own metadata holds, the final reply's text and the confidence
the loop built from the model's reply to its confidence question
(wpt_agent.loop); a dropped attempt's conversation is only its opening
(wpt_agent.opening). The reply to that question is written by the model,
so it is learned and counted in `written`, as every token the model
writes. The attempt's span, its group's event and its round's event name
the file and the line numbers. A line the store refuses raises
wpt_telemetry.RecordError, which stops the command (exit code 3).
"""

import difflib
import json
import random
from dataclasses import dataclass

import wpt_agent
from wpt_model import Routes
from wpt_telemetry import Telemetry

CORRECT = "correct"  # the check's part for the task's own judgement (wpt_agent.check)

# the weights of an attempt's parts, starting values until experiments on
# the 1B model set them (#53); weight_correct stays above weight_calibration (check)
WEIGHTS = {
    "weight_correct": 1.0,
    "weight_calibration": 0.5,
    "weight_rule": 0.25,  # each rule's part: the check's other parts, and within_area
    "weight_tool_call": -0.01,  # each tool call
}


def check(settings):
    """ValueError when weight_correct is not above weight_calibration:
    stating a low confidence must never pay more than being right."""
    if settings["weight_correct"] <= settings["weight_calibration"]:
        raise ValueError(f"weight_correct ({settings['weight_correct']}) must be above weight_calibration "
                         f"({settings['weight_calibration']}), so stating a low confidence never pays more than "
                         f"being right")


def written(attempt) -> int:
    """How many tokens the model wrote in the attempt: the tokens the loss
    learns, never the instructions, the tool results or the template's."""
    return sum(attempt.learn)


def tool_calls(attempt) -> int:
    """How many tool calls the model made in the attempt."""
    return sum(block["type"] == "tool_call" for message in attempt.conversation["messages"]
               if message["role"] == "assistant" for block in message["content"])


def combined(task, attempt, parts, settings, closeness=None) -> float:
    """What the attempt's parts are worth together: 0 when the check gave no score, the attempt has no answer
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


@dataclass
class Kept:
    """An attempt Player.attempts played and kept: its line number in the
    player's file, then the attempt, its routes, score and reward, or the
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


class Player:
    """One step's attempts, played by one model in the main process.
    Sampling draws from `sampling`, a generator the caller seeded, so every
    attempt comes from one sequence. The model alone writes the attempts,
    never a drafter: the tokens drawn then depend on the seed alone, never
    on how fast this machine's passes are. Each attempt is kept in
    telemetry's record, as a line of `<name>-attempts.jsonl`, numbered from
    lines + 1. seed: the step's (Data.step_seed), which the tools offered are
    drawn from. built: {image name: built image}; pools: {image name: its
    pool of tools}."""

    def __init__(self, name, model, tokenizer, built, pools, settings, seed, sampling, telemetry=None, lines=0):
        self.name, self.model, self.tokenizer, self.built, self.pools = name, model, tokenizer, built, pools
        self.settings, self.seed, self.sampling, self.lines = settings, seed, sampling, lines
        self.telemetry = telemetry or Telemetry.off()
        self.file = f"{name}-attempts.jsonl"

    def attempts(self, task, round_, group, count, reward=None, terse=False, with_changes=False) -> list[Kept]:
        """Play count attempts at the task together, in its full or (terse)
        its one-line form, and keep each as a line of self.file, in order,
        as soon as it is scored. round_ is None for attempts made before any
        round (a reference), group says which group of the round (or which
        task), and each attempt is numbered from 1, offered the same tools in
        both forms; reward(task, index, played) is what an attempt is worth,
        None for one never trained on. A dropped attempt (Docker failed) is
        kept too. RecordError when the store refuses a line."""
        settings = self.settings
        offers = [wpt_agent.offer(task, self.pools.get(task.image, []),
                                  random.Random(f"{self.seed}/{task.id}/{round_}/{group}/{index}"))
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

        sampling = wpt_agent.Sampling(settings["temperature"], settings["top_k"] or None)  # top_k 0: every token
        with self.telemetry.span(f"{self.name}.play", task=task.id, round=round_, group=group, attempts=count,
                                 terse=terse):
            return wpt_agent.play(task, offers, keep, model=self.model, tokenizer=self.tokenizer,
                                  built=self.built, sampling=sampling, generator=self.sampling,
                                  check_seconds=settings["check_seconds"], result_bytes=settings["result_bytes"],
                                  routes=True, with_changes=with_changes, terse=terse, name=self.name)

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
        self.lines += 1
        return self.lines
