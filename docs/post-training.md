# Post-training

Post-training uses both:

- **recorded human demonstrations**, and
- **interactive environments**.

## Rules

Post-training data follows the corpus rules. Each example counts, so review
is systematic.

- Text written by people: existing sources under allowed licenses, original
  texts written for the project, and contributions under the DCO.
- No output of another company's language model.
- Environments use no fake web pages or apps: they replay recorded content,
  only when the project has the authorization to use it.
- Scores come from programs or from humans, never from another company's
  model.

## Where records live

Records are files of a source, like any other text: they get the same
license entries, attribution, signed decision, NOTICE listing and checks as
every source. Records written for WPT go in an `empty` source (see
CONTRIBUTING.md, "Adding your own texts"). Conversations are files in the
`chat` format, tagged with the name of the step that reads them (`sft` for
the `sft` step); a `chat` file is never a pre-training document
([chat-format.md](chat-format.md)). Tasks for agentic post-training are
files in the `task` format, tagged the same way (`grpo` for the `grpo`
step, below, whose tasks the `token-efficiency` step reads too); they are
never pre-training documents either.

- A record holds only what a person made: commands typed, edits, messages
  written. From a session with an AI assistant, only the person's own
  messages may be kept, never the assistant's replies.
- No other person's personal data, and no secrets: terminal output and
  paths are checked before a record is committed.

## Record format

JSON Lines: one record per line. Media such as screenshots are separate
files named by their sha256 and referenced from the record. Each record
carries its provenance: the contributor, the commit that added it (signed
off under the DCO), the reference of the authorization it relies on, and its
license. A conversation in a record follows [chat-format.md](chat-format.md):
one JSON object holding every element, turned into tokens by a template
before training.

## Running post-training

```sh
uv run wpt-posttrain <model recipe> <base model folder> <new model folder>
```

The steps are the model recipe's `posttrain` list, run in order; the new
folder's provenance records the recipe's name.

A step is one Python file (or the name of a built-in one, in
`apps/wpt-posttrain/src/wpt_posttrain/steps/`) that defines `run`:

```python
from wpt_posttrain import Outcome

def run(context):
    conversations = context.conversations()  # [(id, conversation)] tagged with this step's name
    if not conversations:
        return Outcome.failure("no conversations")
    ...  # train context.model on them
    return Outcome.success()
```

`context` gives the `model` to change (in place, on `device`), its
`tokenizer`, the `base` folder, its `provenance`, the `store`, and
`telemetry`, the run's record: a step records its own events and spans
there, and writes its own files (such as one JSON line per attempt) into
the run's store folder ([observability.md](observability.md)).
`context.conversations()` reads the conversations of every admitted source
whose `chat` files carry a tag named after the step, from the repository
holding the current folder, and the runner records those sources.
`context.tasks()` reads the tasks the same way: the task folders whose
`task.json` carries the step's name. `context.tasks(tag)` reads those
tagged `tag` instead, for a step that trains on another step's tasks (the
`token-efficiency` step reads `context.tasks("grpo")`); the runner records
those sources under the step that read them. Every step
returns the same type, `Outcome`: success, or failure with its reason. A
step does not write a report; it says whether it managed to do its part,
and if not, why.

The steps run in order, and the first failure stops the run: the command
prints that step's reason and writes no model folder (its record in the
store keeps what happened, with that reason). When every step succeeded, it
writes the new folder: the weights, the tokenizer, and the base model's
`provenance.json` with a `post_training` section (the sha256 of the base
weights, each step's file and its sha256, the repository commit, the device
and the tools), then `LICENSE`, `NOTICE` and the model card, which lists the
steps. The new folder must not exist yet; the base folder is never changed,
and its `evals/` results are not copied, since they describe other weights.
The folder is written beside its place, then renamed into it; when that
rename fails, the finished folder is kept, hidden as
`.<folder name>-<random>`, and named on the error output: move it into
place by hand. Run `wpt-eval` on the new folder to measure it.

Each step's record in `provenance.json` names the sources it read, and the
`post_training` section keeps their address, revision and licenses. The
new folder's `NOTICE` lists them with the pre-training sources, with the
credit and the license text of each, and the model card names them under
each step.

## The `sft` step

The `sft` step, in a recipe's `posttrain` list, is supervised fine-tuning:
it reads the conversations tagged `sft`, turns each into tokens with the
chat template, and trains the model to predict the assistant's turns only. Until a source
holds such conversations, it fails with "no conversations" and writes
nothing. Each conversation must fit the model's context; a longer one
stops the step, since cutting it would drop the answers it holds. Its
settings (epochs, learning rate, batch, seed) are the constants at the top
of `steps/sft.py`; to change them, copy the file and put its path in the
recipe's `posttrain` list instead.

The base model must use a vocabulary of tokenizer 0.3.0 or later, which
reserves the chat template's ids. Talk to the result with
`uv run wpt-chat NEW --chat`.

## The `grpo` step

The `grpo` step teaches the model to finish tasks as an agent, by
reinforcement learning (group relative policy optimization). The model
works in a small harness with four tools, `read` (a file), `write` (a
file), `edit` (replace a string that occurs exactly once in a file) and
`bash` (run a command), on tasks that people write. Its code is the
`wpt-agent` package (tools, containers, images, the agent loop and the
scoring) and `steps/grpo.py` (the learning). It needs Docker on the
machine. Until a source holds tasks tagged `grpo`, it fails with "no
tasks" and writes nothing.

### Tasks

Tasks are written by people, in an `empty` source (CONTRIBUTING.md,
"Adding your own texts"), as files in the `task` format tagged `grpo`. One
folder per task:

```text
<folder>/task.json   what the model is asked, the image, the limits and, if needed, larger containers
<folder>/files/...   the starting files, copied into the container's /work
<folder>/check       any program (its first line names its interpreter, such as #!/bin/sh)
```

```json
{"schema": 1, "instructions": "Fix the failing test in calc.py.", "image": "python",
 "limits": {"max_turns": 20, "max_tokens": 4000, "command_seconds": 30},
 "container": {"memory": 8192}}
```

- `instructions`: the user message that starts every attempt.
- `image`: the folder in `data/images/` the attempt's container comes from.
- `limits`: `max_turns`, the assistant turns in one attempt; `max_tokens`,
  the tokens the model writes in one attempt, over every turn;
  `command_seconds`, how long one tool call may run.
- `container` (optional): sizes above the defaults (Attempts, below):
  `memory` and `disk` in MiB, `processes`, `cpus`. Each can only raise its
  default; a smaller value leaves the default. Since `/work` and `/tmp`
  are kept in memory, the sizes that result must leave room for the
  processes: 2 x `disk` + 1024 MiB (`HEADROOM`) must stay below `memory`.
  A task that raises `disk` must raise `memory` with it; the defaults leave
  1024 MiB more than that.

A task folder is a folder holding `task.json` that is not inside another
task's `files/`; every other file of the format is a task folder's `check`
or under its `files/`. `wpt-dataset check --fetch` checks each folder
against `packages/wpt-dataset/src/wpt_dataset/schemas/task.schema.json`,
and checks that its `image` is a folder of `data/images/` holding a
`Dockerfile`. CI checks the sources a change touches
(`wpt-dataset check --changed`, run by `scripts/check.py`); when anything
under `data/images/` changed, it also checks every source with files in the
`task` format, so a task naming an image that is gone is caught before the
change merges. The same check refuses a task whose sizes break the rule
on `disk` and `memory` above, naming its `task.json`. The step reads the tasks whose `task.json` carries the tag.
Like every file of a source, task files are UTF-8 text.

### Images

The containers come from Dockerfiles in the repository,
`data/images/<name>/Dockerfile`. Each machine builds them itself; a
committed lock pins what goes in, not the bytes that come out, since
builds differ between machines. A Dockerfile names its base as
`ARG BASE=<image>` before `FROM ${BASE}` and installs `${PACKAGES}`
(`ARG PACKAGES="<names>"`) with apt, so an image is Debian or Ubuntu. The
supported bases are Debian 13 (trixie) or later and Ubuntu 24.04 (noble) or
later, whose apt reads the dated archive (below); an older apt ignores it
and would read today's archive, which no lock pins:

```dockerfile
ARG BASE=debian:trixie-slim
FROM ${BASE}
ARG PACKAGES="bash coreutils python3"
RUN apt-get update && apt-get install -y --no-install-recommends ${PACKAGES} && rm -rf /var/lib/apt/lists/*
```

Only the base's apt is checked against the snapshot (below): the
Dockerfile's own steps are trusted not to add package sources or to
download anything, and nothing checks them. A person reviews them like
any other change.

`uv run wpt-agent lock-image NAME` pulls the base, pins it by digest,
takes the current moment as the snapshot, checks that apt on the base,
after `apt-get update`, reads its package lists only from the snapshot
hosts (it refuses the base otherwise), builds the image, and writes
next to the Dockerfile `lock.json` (the base by digest, the snapshot and
the version of every installed package) and `licenses.json` (every
installed package, its version and its licenses; [licenses.md](licenses.md),
"Container images of agentic tasks"). Commit both. It refuses to replace
an existing lock unless given `--force`: locking again is the person's
choice. `wpt-dataset check` checks both files against their schemas
(`image-lock.schema.json`, `image-licenses.schema.json`).

Every build reads its packages from the dated archive of the base's
distribution (snapshot.debian.org or snapshot.ubuntu.com) as it was at the
lock's snapshot, so a lock keeps building for years. The base gets one
layer that sets apt's `APT::Snapshot` before the Dockerfile runs; apt
finds the archive itself. apt fetches it over HTTPS without checking the
server's certificate, since the base images hold no certificates; it
checks the archive's signatures and every package's checksum itself, and
accepts the dated archive's expired Release files.

`uv run wpt-agent build-image NAME` builds the image from its lock on this
machine (the base by digest, the snapshot, every package as
`name=version`), and stops when the build installed other packages than
the lock pins. Like `lock-image`, and through the same code, it first
checks that apt, after `apt-get update`, reads its package lists only
from snapshot.debian.org or snapshot.ubuntu.com, and refuses to build
otherwise. The `grpo` step never builds an image: it stops before any
attempt when a task's image is not built for the folder's current files,
and names the command that builds it. A changed Dockerfile or lock needs
a new build.

### Attempts

For each task, the model makes several attempts (`GROUP`). Each attempt
runs in a fresh container of the task's image, removed after the attempt.
The container has no network and a read-only root: it can write only
`/work`, its working folder holding the task's `files/`, and `/tmp`, two
folders kept in memory. Its sizes are the defaults of the task format
(`SIZES` in `packages/wpt-dataset/src/wpt_dataset/tasks.py`), each raised
by the task's `container` field when it asks for more: `memory` (4096 MiB,
the files in `/work` and `/tmp` included), `disk` (1024 MiB, what `/work`
and `/tmp` may each hold), `processes` (256, processes and threads at
once) and `cpus` (2). The check's container gets the same sizes, and must
hold a full copy of `/work`: that is why `HEADROOM` (1024 MiB) of memory
stays free for the processes when `/work` and `/tmp` are both full. The model's commands run as user 1000, not root, and
nothing in the container can gain privileges (no `setuid` program works).
The container keeps only three capabilities, `CHOWN`, `DAC_OVERRIDE` and
`FOWNER`, which root uses to unpack the task's `files/` into `/work` with
user 1000 as their owner; the model's commands, as user 1000, hold none of
them. The container's first process is Docker's init (`--init`), which
collects every process that ended, so background commands that finished
never count against the process limit; the check's container starts the
same way.

The conversation follows [chat-format.md](chat-format.md): the four tools,
then the instructions as the user's message. The model writes an assistant
turn; each tool call in it runs in the container (a relative path starts at
`/work`, and a call that cannot run returns its error to the model); the
results come back as a tool turn; and the model writes the next turn. The
attempt ends when a turn closes without a tool call, when the model writes
the end of a document, after `max_turns` turns (the last turn's calls still
run), or when `max_tokens` or the model's context (`block_size + 1`
tokens) is reached in the middle of a turn, whose calls then do not run;
an attempt that fills the context is scored as it is. The attempt also
ends when the model's own processes or files fill its container, so that
Docker can start no command there: processes left running reach the
process limit, or files in `/work` and `/tmp` exhaust the memory. That
turn's calls get no result, and Docker's own error never
reaches the model. The attempt is scored like any other, on what `/work`
holds; when running out of memory also stopped the container, `/work` is
lost and the check scores an empty folder. A command that runs
longer than `command_seconds` is stopped, and its result says so (one
that ignores the signal to stop is killed 5 seconds later); a command that
only exits with the code `timeout` uses is reported with that exit code. A tool
result longer than `RESULT_BYTES` (4000) keeps its first and last 2000
bytes, with a line between them saying how many bytes were cut; the output
is cut while it is read, so a command that prints without end never fills
the machine's memory. `edit` rewrites only a UTF-8 file of at most 1 MiB,
and tells the model to use `bash` otherwise. No tool argument may hold a
NUL character or text that is not UTF-8, `write`'s content and `edit`'s
strings included: such a call fails with a message to the model. A file
holding a NUL byte, or other bytes that are not UTF-8 text, is written
through `bash`, with a command that prints them from escapes, such as
`printf '\0'`. A task whose instructions and tools alone fill the model's
context stops the step.

### Scores

When the attempt has ended, every process in its container is stopped
(the container is paused). The task's `check` then runs in a fresh
container of the same image and sizes, holding only a copy of the
attempt's `/work`: what the model started, or left anywhere else, cannot
reach the check. The check is copied in, in a folder of its own, and runs
as root from `/work` for at most `CHECK_SECONDS`.

Since the check runs from `/work`, the files the model left are in its
current folder. A check must not start an interpreter that loads code from
the current folder: run Python as `python3 -P` (or `-I`), and call
programs and scripts by absolute paths. Otherwise a `conftest.py`,
`unittest.py` or similar file the model planted would run as root inside
the check and could print any score it likes. Its standard output,
without the white space around it, must be one number from 0 to 1: the
attempt's score. Its exit code and standard error are not read. A check
that prints anything else, crashes or runs out of time gives no score;
the attempt counts as 0, and the step prints a line naming the task and
why.

When Docker or the container fails during an attempt or its check, which
is not the model's doing, the attempt is dropped from its group and the
step prints a line naming the task and the reason. Each command starts
by printing one byte before it runs, so a command that never started
(no such container, a container that is not running, a refusal of
Docker's daemon or runtime) is told apart from a command that exited
with an error. When Docker could not start it, the sandbox asks Docker
whether the container is full (memory ran out in it, or fewer than 16
processes are free): if so, the model filled it (Attempts, above);
otherwise Docker failed. A group left with fewer than 2 attempts teaches
nothing and is skipped. The step goes on, unless every attempt of a round was dropped: that round makes no update,
and the step fails, naming the round, how many attempts Docker failed and
its last error, and writes nothing. A round with at least one update goes
on as usual. A whole run that made no update at all fails too, and
writes nothing: every group was skipped, and the step says how many
because their scores were all equal and how many because they kept fewer
than 2 attempts.

### Learning

Each attempt's advantage is its score minus the mean score of its group
(the attempts at the same task). It is not divided by the group's
standard deviation: that would make a group whose scores barely differ
weigh as much as one whose scores differ widely, and give tasks that are
almost always passed or failed the largest steps. The loss is minus the
advantage times the log-probability of each token the model wrote, the
same tokens `sft` learns (never the instructions, the tool results or the
template's own tokens), averaged over every token the model wrote in the
update's attempts; a skipped group (below) adds none. The tokens are the ones the model sampled, kept as they
were, since writing the conversation as tokens again would not always give
the same ones. There is no frozen reference model and no KL penalty: each
update learns from attempts the current model just made. A group whose
scores are all equal teaches nothing and is skipped. The model samples
from its own probabilities (`TEMPERATURE` 1, no `TOP_K`), which the loss
assumes.

The routed expert layers ([model.md](model.md), "Experts") choose experts
for each token while the model writes an attempt. The step records those
choices, and the loss pass routes each token through the same experts
instead of choosing again (routing replay, "R3"), so the loss scores the
tokens through the experts that wrote them. The routers' weights do not
change during the step, and their balancing biases do not either. The
hash-routed layers choose from the token alone and need neither.

### Runs

The order of the tasks in each round and every sampled token come from
generators seeded with `SEED`, so a run gives the same attempts again on
the same machine and setup (the same model, tasks, images and devices);
the checks themselves may differ, as any program may. WPT's own attempts
live in memory only: each one is dropped after its update, and nothing is
kept on disk yet. The step prints one line per round (`ROUNDS`, passes
over every task): the mean score, the number of attempts and of updates.
Its settings are the constants at the top of `steps/grpo.py`; to change
them, copy the file and put its path in the recipe's `posttrain` list. The
containers' sizes are not among them: they belong to each task.

## The `token-efficiency` step

The `token-efficiency` step comes after `grpo` in a recipe's `posttrain`
list. Its goal is the same result as the model it starts from, with fewer
tokens. It trains on `grpo`'s own tasks, those tagged `grpo` (it has no tag
of its own), with the same images, containers, limits and checks, and
`grpo`'s learning unchanged (Attempts, Scores, Learning and Runs above):
groups of `GROUP` attempts, the advantage as the reward minus the group's
mean, the loss over the tokens the model wrote, R3 routes and routers that
do not change, dropped attempts and the same failures. Only the reward
differs. Its code is `steps/token-efficiency.py`; the machinery both steps
share is `apps/wpt-posttrain/src/wpt_posttrain/agentic.py`. Until a source
holds tasks tagged `grpo`, it fails with "no tasks" and writes nothing.

First, the reference. Before any update, the incoming model makes
`REFERENCE` attempts at each task (by default `GROUP`), run and scored as
`grpo`'s are. A task's best start score is the best score among them. An
attempt passes when its score is at least that best score, and the task's
reference length is the mean length of its passing attempts. A length
counts only the tokens the model wrote, the same tokens the loss learns,
never the instructions, the tool results or the template's own tokens. The
reference stays fixed for the whole step, and the step does not train on
its attempts. A task whose every reference attempt scored 0 has no result
to keep and is left out, and so is a task whose every reference attempt
Docker failed; the step prints a line for each. When no task is left, the
step fails, saying how many tasks were left out for each reason, and
writes nothing.

Then the rounds, as `grpo`'s, with this reward for each attempt:

- 0 when its score is below the task's best start score;
- otherwise the task's reference length divided by the attempt's length:
  1 for an attempt as long as the reference, 2 for one half as long, 0.5
  for one twice as long.

Every passing attempt is worth more than every failing one, and among
passing attempts the shorter one is worth more. The step prints one line
per task for the reference (its best start score and reference length) and
one line per round (the mean reward, the number of attempts and of
updates). Sampling draws from one generator seeded with `SEED`, for the
reference and then the rounds. Its settings are the constants at the top
of `steps/token-efficiency.py`, `TAG` (the tag of the tasks it reads) and
`REFERENCE` among them; to change them, copy the file and put its path in
the recipe's `posttrain` list.
