"""Token efficiency: teach the model to reach the same result as the model
it starts from, writing fewer tokens. It comes after grpo in a recipe's
posttrain list.

Reads grpo's own tasks: those of every admitted source whose task files
are tagged `grpo` (docs/post-training.md), with the same images, which
must be built on this machine first (`wpt-agent build-image NAME`).

First the reference. The incoming model makes `reference` attempts at each
task, written together, exactly as grpo's attempts run and are scored. An
attempt's score here is its result (agentic.result): its correct part,
0 with a broken gate or without an answer. A task's best start score is
the best score among them; an attempt "passes" when its score is
at least that best score, and the task's reference length is the mean
length of its passing attempts. A length counts only the tokens the model
wrote, the same tokens the loss learns, its reply to the confidence
question after its final reply included (never the instructions, the
tool results, the confidence question or the template's own tokens). The reference stays fixed for the
whole step, and its attempts are not trained on. A task whose every
reference attempt scored 0 has nothing to keep and is left out, and so is
a task whose every reference attempt was dropped because Docker failed;
the step prints a line for each. When no task is left, the step fails and
writes nothing.

Then grpo's learning, unchanged except for the reward: rounds of `group`
attempts per task, in its full form only, in fresh containers, advantage =
reward minus the mean reward of the group (not divided by the standard deviation), the loss over
the tokens the model wrote, R3 routes and frozen routers, dropped attempts
and the same failures (steps/grpo.py says each). An attempt's reward
(maintainer's decision, 2026-10-23) is 0 when it fails, its score below
the task's best start score, and when it passes, its parts weighed
together as grpo weighs them (agentic.combined, with this step's weight_
settings) times the
task's reference length divided by the attempt's length: the weighed
parts for an attempt as long as the reference, more for a shorter one,
less for a longer one. Every passing attempt whose weighed parts are above
0 beats every failing one, and of two passing attempts with the same
weighed parts the shorter one wins.

Sampling draws from one generator seeded with the seed, for the reference
and then the rounds, so a run gives the same attempts again on the same
machine and setup; the model alone writes them, without a drafter, as in
steps/grpo.py. Every attempt, the reference's (round null) and the
dropped ones included, is kept as one line of
token-efficiency-attempts.jsonl in the run's record folder, as grpo keeps
its own (steps/grpo.py). SETTINGS below is the step's whole
configuration, its defaults, which a recipe's `posttrain` entry
{"step": "token-efficiency", "settings": {...}} changes. The machinery
is wpt_posttrain/agentic.py, shared with steps/grpo.py.
"""

import wpt_agent
from wpt_posttrain import Outcome, agentic

SETTINGS = {
    "tag": "grpo",  # the tasks it trains on: grpo's own
    "rounds": 10,  # passes over every task
    "tasks_per_update": 4,
    "group": 8,  # attempts per task
    "reference": 8,  # attempts per task by the incoming model, before any update
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "temperature": 1.0,  # sampling from the model's own probabilities, which the loss assumes
    "top_k": 0,  # 0: every token
    "check_seconds": 300.0,  # how long the check may run
    "result_bytes": 4000,  # a longer tool result keeps its first and last 2000 bytes
    "seed": 0,  # the reference attempts, the order of the tasks in each round, and the sampled tokens
    **agentic.WEIGHTS,  # a passing attempt's parts weighed together as steps/grpo.py weighs them
}
check = agentic.check  # weight_correct above weight_calibration, before any step runs


def run(context):
    settings = context.settings
    try:
        tasks, built, pools = agentic.prepare(context, settings["tag"])
    except wpt_agent.Unready as error:
        return Outcome.failure(str(error))
    with agentic.frozen_routers(context.model):
        learner = agentic.Learner("token-efficiency", context.model, context.tokenizer, context.device, built,
                                  settings, context.telemetry, pools)
        context.model.eval()
        references, zero, lost = _references(learner, tasks)
        kept = [task for task in tasks if task.id in references]
        if not kept:
            return Outcome.failure(f"no task left of the {len(tasks)} tasks tagged {settings['tag']}: {zero} scored 0 "
                                   f"on every reference attempt, Docker failed every reference attempt of {lost}")
        return learner.train(kept, lambda task, index, played: reward(
            references[task.id], played.attempt, agentic.result(played.attempt, played.score),
            agentic.combined(task, played.attempt, played.parts, settings)))


def _references(learner, tasks):
    """({task id: (best start score, reference length)} of the tasks kept,
    how many were left out because every reference attempt scored 0, and
    how many because Docker failed every one)."""
    references, zero, lost = {}, 0, 0
    for number, task in enumerate(tasks, start=1):
        scored = []  # (score, length) of the reference attempts Docker did not drop
        for kept in learner.attempts(task, None, number, learner.settings["reference"]):  # round None, not trained on
            if kept.attempt is not None:
                scored.append((agentic.result(kept.attempt, kept.score), agentic.written(kept.attempt)))
        reference = _reference(scored)
        if reference is None and scored:
            print(f"token-efficiency: {task.id}: left out, its {len(scored)} reference attempts all scored 0")
            zero += 1
        elif reference is None:
            print(f"token-efficiency: {task.id}: left out, Docker failed every reference attempt")
            lost += 1
        else:
            references[task.id] = reference
            print(f"token-efficiency: {task.id}: best start score {reference[0]:.4f}, reference length "
                  f"{reference[1]:.1f} tokens")
    return references, zero, lost


def _reference(scored):
    """(best start score, reference length) of a task's reference attempts,
    given as (score, length); None when none scored above 0."""
    best = max((score for score, _ in scored), default=0.0)
    if best <= 0:
        return None
    lengths = [length for score, length in scored if score >= best]
    return best, sum(lengths) / len(lengths)


def reward(reference, attempt, score, weighed):
    """0 when the attempt's score is below the task's best start score, else
    its weighed parts times the task's reference length divided by the
    number of tokens the model wrote in the attempt."""
    best, length = reference
    return 0.0 if score < best else weighed * length / agentic.written(attempt)
