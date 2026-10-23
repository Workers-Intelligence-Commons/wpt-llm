"""Group relative policy optimization: teach the model to finish tasks as
an agent.

Reads the tasks of every admitted source whose task files are tagged
`grpo` (docs/post-training.md). Each task's image must be built on this
machine first (`wpt-agent build-image NAME`). For each task the model makes
`group` attempts, each in a fresh container of the task's image with no
network and limited sizes (wpt_agent), written together in one batch from
one reading of what the attempts' openings share (the model's generation
cache), each attempt offered its own choice of the task's tools
(wpt_agent.offer, seeded with the seed): it writes a turn, its tool calls run
in the container as a non-root user, their results come back as the next
turn, until it stops or a limit is reached. Then the attempt's processes
are stopped and the task's check scores a copy of the attempt's /work in
a fresh container, next to the attempt's conversation, as named parts
from 0 to 1 (`correct` and the rules it reports; a task with an area also
gets within_area); a check that prints no such score counts as 0. An
attempt's reward is its parts weighed together (agentic.combined;
docs/post-training.md, "Rewards"): 0 without an answer or with a broken
gate of the task, else the weighted sum of correct, the calibration of
its confidence (1 - (confidence - correct)²), each rule, and its number
of tool calls, each weighed by its weight_ setting below. A task with a
one-line form (terse) is played twice in each round: `group` attempts in
its full form, then `group` in its one-line form, all of them one group;
a terse attempt's reward adds, weighed, its closeness to the full attempt
of the same index (how alike their changes to /work are; none on a task
whose area is empty, which only answers), so the model's own full result
is its reference, and, where closeness counts, a terse attempt whose
full attempt was dropped is not trained on (maintainer's decisions, 2026-10-18 and 2026-10-23). An
attempt whose own processes or files fill
its container so that no command can start there ends, and is scored
the same way. When Docker fails, which is not the model's doing, the
attempt is dropped from its group, and a group left with fewer than 2
attempts is skipped; a round whose every attempt was dropped makes no
update and stops the step, and so does a whole run that made no update.

Each attempt's advantage is its reward minus the mean reward of its group.
It is not divided by the group's standard deviation: that would make a
group whose scores barely differ weigh as much as one whose scores differ
widely, and give the tasks that are almost always passed or failed the
largest steps. The loss is minus the advantage times the log-probability of
each token the model wrote (never the instructions, the tool results or the
template's own tokens, as in steps/sft.py), averaged over every token the
model wrote in the update's attempts. There is no frozen reference model
and no KL penalty: each update learns from attempts the current model just
made. A group whose rewards are all equal teaches nothing and is skipped.
The group's size matters: when few attempts of a group differ, the group
mean can take away a penalty a weight was meant to give (paper 13 of the
plan, issue #40), so `group` is checked by experiment with the weights.

The routed expert layers keep, during the loss pass, the experts their
routers chose while the model wrote the attempt (R3: wpt_model.Routes), so
the loss scores each token through the experts that wrote it; the routers'
weights do not change during the step. Hash-routed layers need neither.

An attempt ends at the latest at the task's max_tokens, past the model's
block_size if need be: the model reads it through its generation cache,
which has no length limit. The model alone writes the attempts, without a
drafter (the weights change at every update, and the drafter is trained
only at the end of the run). Sampling draws from a generator seeded with
the seed, so a run gives the same attempts again on the same machine and
setup. Every attempt, a dropped one included, is kept as one line of
grpo-attempts.jsonl in the run's record folder: a conversation in the chat
format whose metadata identifies the attempt; the attempt's span names the
line. SETTINGS below is the step's whole configuration, its defaults: a
recipe's `posttrain` entry {"step": "grpo", "settings": {...}} changes
any of them (docs/post-training.md, "Running post-training"). The weights
are starting values, until experiments on the 1B model set them (#53).
The machinery is wpt_posttrain/agentic.py, which
steps/token-efficiency.py shares.
"""

import wpt_agent
from wpt_posttrain import Outcome, agentic

SETTINGS = {
    "rounds": 10,  # passes over every task
    "tasks_per_update": 4,
    "group": 8,  # attempts per task, in each of its forms
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "temperature": 1.0,  # sampling from the model's own probabilities, which the loss assumes
    "top_k": 0,  # 0: every token
    "check_seconds": 300.0,  # how long the check may run
    "result_bytes": 4000,  # a longer tool result keeps its first and last 2000 bytes
    "seed": 0,  # the order of the tasks in each round, and the sampled tokens
    **agentic.WEIGHTS,  # the weights of an attempt's parts
    "weight_closeness": 0.25,  # a terse attempt's closeness to the full attempt of the same index
}
check = agentic.check  # weight_correct above weight_calibration, before any step runs


def run(context):
    try:
        tasks, built, pools = agentic.prepare(context)
    except wpt_agent.Unready as error:
        return Outcome.failure(str(error))
    with agentic.frozen_routers(context.model):
        learner = agentic.Learner("grpo", context.model, context.tokenizer, context.device, built, context.settings,
                                  context.telemetry, pools, terse=True)
        return learner.train(tasks)
