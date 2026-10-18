"""Token efficiency: teach the model to reach the same result as the model
it starts from, writing fewer tokens. It comes after grpo in a recipe's
posttrain list.

Reads grpo's own tasks: those of every admitted source whose task files
are tagged `grpo` (docs/post-training.md), with the same images, which
must be built on this machine first (`wpt-agent build-image NAME`).

First the reference. The incoming model makes REFERENCE attempts at each
task, written together, exactly as grpo's attempts run and are scored. A task's best start
score is the best score among them; an attempt "passes" when its score is
at least that best score, and the task's reference length is the mean
length of its passing attempts. A length counts only the tokens the model
wrote, the same tokens the loss learns (never the instructions, the tool
results or the template's own tokens). The reference stays fixed for the
whole step, and its attempts are not trained on. A task whose every
reference attempt scored 0 has nothing to keep and is left out, and so is
a task whose every reference attempt was dropped because Docker failed;
the step prints a line for each. When no task is left, the step fails and
writes nothing.

Then grpo's learning, unchanged except for the reward: rounds of GROUP
attempts per task in fresh containers, advantage = reward minus the mean
reward of the group (not divided by the standard deviation), the loss over
the tokens the model wrote, R3 routes and frozen routers, dropped attempts
and the same failures (steps/grpo.py says each). An attempt's reward is 0
when its score is below the task's best start score, and otherwise the
task's reference length divided by the attempt's length: 1 for an attempt
as long as the reference, more for a shorter one, less for a longer one.
Every passing attempt beats every failing one, and a shorter passing
attempt beats a longer one.

Sampling draws from one generator seeded with SEED, for the reference and
then the rounds, so a run gives the same attempts again on the same
machine and setup; the model alone writes them, without a drafter, as in
steps/grpo.py. Every attempt, the reference's (round null) and the
dropped ones included, is kept as one line of
token-efficiency-attempts.jsonl in the run's record folder, as grpo keeps
its own (steps/grpo.py). The settings below are the step's whole
configuration; to change them, copy this file and give its path to
wpt-posttrain. The machinery is wpt_posttrain/agentic.py, shared with
steps/grpo.py.
"""

from wpt_posttrain import Outcome, agentic

TAG = "grpo"  # the tasks it trains on: grpo's own
ROUNDS = 10  # passes over every task
TASKS_PER_UPDATE = 4
GROUP = 8  # attempts per task
REFERENCE = GROUP  # attempts per task by the incoming model, before any update
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.0
TEMPERATURE = 1.0  # sampling from the model's own probabilities, which the loss assumes
TOP_K = None
CHECK_SECONDS = 300  # how long the check may run
RESULT_BYTES = 4000  # a longer tool result keeps its first and last 2000 bytes
SEED = 0  # the reference attempts, the order of the tasks in each round, and the sampled tokens


def run(context):
    try:
        tasks, built = agentic.prepare(context, TAG)
    except agentic.Unready as error:
        return Outcome.failure(str(error))
    settings = agentic.Settings(ROUNDS, TASKS_PER_UPDATE, GROUP, LEARNING_RATE, WEIGHT_DECAY, TEMPERATURE, TOP_K,
                                CHECK_SECONDS, RESULT_BYTES, SEED)
    with agentic.frozen_routers(context.model):
        learner = agentic.Learner("token-efficiency", context.model, context.tokenizer, context.device, built,
                                  settings, context.telemetry)
        context.model.eval()
        references, zero, lost = _references(learner, tasks)
        kept = [task for task in tasks if task.id in references]
        if not kept:
            return Outcome.failure(f"no task left of the {len(tasks)} tasks tagged {TAG}: {zero} scored 0 on every "
                                   f"reference attempt, Docker failed every reference attempt of {lost}")
        return learner.train(kept, lambda task, attempt, score: reward(references[task.id], attempt, score),
                             measure="reward")


def _references(learner, tasks):
    """({task id: (best start score, reference length)} of the tasks kept,
    how many were left out because every reference attempt scored 0, and
    how many because Docker failed every one)."""
    references, zero, lost = {}, 0, 0
    for number, task in enumerate(tasks, start=1):
        scored = []  # (score, length) of the reference attempts Docker did not drop
        for kept in learner.attempts(task, None, number, REFERENCE):  # kept as round None, never trained on
            if kept.attempt is not None:
                scored.append((kept.score, agentic.written(kept.attempt)))
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


def reward(reference, attempt, score):
    """0 below the task's best start score, else its reference length
    divided by the number of tokens the model wrote in the attempt."""
    best, length = reference
    return 0.0 if score < best else length / agentic.written(attempt)
