"""Group relative policy optimization: teach the model to finish tasks as
an agent.

Reads the tasks of every admitted source whose task files are tagged
`grpo` (docs/post-training.md). Each task's image must be built on this
machine first (`wpt-agent build-image NAME`). For each task the model makes
GROUP attempts, each in a fresh container of the task's image with no
network and limited sizes (wpt_agent): it writes a turn, its read, write,
edit and bash calls run in the container as a non-root user, their results
come back as the next turn, until it stops or a limit is reached. Then the
attempt's processes are stopped and the task's check scores, from 0 to 1,
a copy of the attempt's /work in a fresh container; a check that prints no
such number counts as 0. An attempt whose own processes or files fill
its container so that no command can start there ends, and is scored
the same way. When Docker fails, which is not the model's doing, the
attempt is dropped from its group, and a group left with fewer than 2
attempts is skipped; a round whose every attempt was dropped makes no
update and stops the step, and so does a whole run that made no update.

Each attempt's advantage is its score minus the mean score of its group.
It is not divided by the group's standard deviation: that would make a
group whose scores barely differ weigh as much as one whose scores differ
widely, and give the tasks that are almost always passed or failed the
largest steps. The loss is minus the advantage times the log-probability of
each token the model wrote (never the instructions, the tool results or the
template's own tokens, as in steps/sft.py), averaged over every token the
model wrote in the update's attempts. There is no frozen reference model
and no KL penalty: each update learns from attempts the current model just
made. A group whose scores are all equal teaches nothing and is skipped.

The routed expert layers keep, during the loss pass, the experts their
routers chose while the model wrote the attempt (R3: wpt_model.Routes), so
the loss scores each token through the experts that wrote it; the routers'
weights do not change during the step. Hash-routed layers need neither.

Each attempt must fit the model's context (block_size + 1 tokens): an
attempt that fills it ends there and is scored as it is; a task whose
instructions alone fill it stops the step. Sampling draws from a generator
seeded with SEED, so a run gives the same attempts again on the same
machine and setup. Attempts live in memory only and are dropped after
their update. The settings below are the step's whole configuration; to
change them, copy this file and give its path to wpt-posttrain. The
machinery is wpt_posttrain/agentic.py.
"""

from wpt_posttrain import Outcome, agentic

ROUNDS = 10  # passes over every task
TASKS_PER_UPDATE = 4
GROUP = 8  # attempts per task
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.0
TEMPERATURE = 1.0  # sampling from the model's own probabilities, which the loss assumes
TOP_K = None
CHECK_SECONDS = 300  # how long the check may run
RESULT_BYTES = 4000  # a longer tool result keeps its first and last 2000 bytes
SEED = 0  # the order of the tasks in each round, and the sampled tokens


def run(context):
    try:
        tasks, built, limit = agentic.prepare(context)
    except agentic.Unready as error:
        return Outcome.failure(str(error))
    settings = agentic.Settings(ROUNDS, TASKS_PER_UPDATE, GROUP, LEARNING_RATE, WEIGHT_DECAY, TEMPERATURE, TOP_K,
                                CHECK_SECONDS, RESULT_BYTES, SEED)
    with agentic.frozen_routers(context.model):
        learner = agentic.Learner("grpo", context.model, context.tokenizer, context.device, built, limit, settings)
        return learner.train(tasks)
