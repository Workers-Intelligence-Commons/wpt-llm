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
- A demonstration's `assistant` message may start as a draft written by a
  WPT model, in the conversation editor of `wpt-web`: the person edits it or
  confirms it as it is, and the message's `metadata.draft` records the model
  folder, the sha256 of its weights and of the draft, and whether the
  person changed it ([chat-format.md](chat-format.md), "Model drafts"). An
  unedited confirmed draft is allowed.
- An AI assistant (a coding agent helping a contributor) works with the
  data only through the `wpt-dataset` commands for structure and metadata,
  never through the ones that read or write people's texts; it may start a
  source with `new-source` without `--file` and `--license-text`
  (CONTRIBUTING.md, "Models and generated text"; maintainer's decision,
  2026-10-18).
- Environments use no fake web pages or apps: they replay recorded content,
  only when the project has the authorization to use it. A task's replies
  to the `ask` tool are content a person wrote and the task records, which
  the environment replays: no person waits for the model's question
  (maintainer's decision, 2026-10-18; "Tools", below).
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
`task.json` carries the step's name, except the measure-only ones
("Measure tasks", below), with the texts of pointer tasks filled in
("Pointer tasks", below). `context.tasks(tag)` reads those
tagged `tag` instead, for a step that trains on another step's tasks (the
`token-efficiency` step reads `context.tasks("grpo")`); the runner records
those sources under the step that read them, and the sources pointer tasks
point into with them, so the new model's `NOTICE` credits them too. Every step
returns the same type, `Outcome`: success, or failure with its reason. A
step does not write a report; it says whether it managed to do its part,
and if not, why.

The steps run in order, and the first failure stops the run: the command
prints that step's reason and writes no model folder (its record in the
store keeps what happened, with that reason). When every step succeeded, it
trains the new weights' drafter ("The drafter", below), and once that
succeeded too, it writes the new folder: the weights, the tokenizer, the
drafter, and the base model's
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
chat template, and trains the model to predict the assistant's turns only.
The CSA indexers go on learning by their own loss, as in pre-training
(model.md, "Attention"): the step asks the model for it, adds its gradient
and shows its value beside the loss of each epoch. A post-training step
that changes the weights by gradients does the same:
`model(inputs, indexer_loss=True)` returns the scores and that loss, and
the step adds `indexers - indexers.detach()` to its own loss. Until a source
holds such conversations, it fails with "no conversations" and writes
nothing. Each conversation must fit the model's context; a longer one
stops the step, since cutting it would drop the answers it holds. Its
settings (epochs, learning rate, batch, seed) are the constants at the top
of `steps/sft.py`; to change them, copy the file and put its path in the
recipe's `posttrain` list instead.

The base model must use a vocabulary of tokenizer 0.3.0 or later, which
reserves the chat template's ids. Talk to the result with
`uv run wpt-chat NEW --chat`.

## The drafter

Maintainer's decisions, 2026-10-17: a model and its drafter for
speculative decoding ([model.md](model.md), "The drafter") are one unit
and a hard dependency, with separate weights, and the drafter is no longer
a post-training step. After the last step succeeded, every run trains a
fresh drafter against the post-trained weights; the base folder's drafter,
trained against other weights, is never used or copied. A base folder
whose drafter is missing or was trained against other weights is refused
before any step runs (`load_model`), and no option skips either.

The drafter's shape and training come from the model recipe's `drafter`
section, as after pre-training. Its prompts are the beginnings of the base
model's training documents (the dataset its `provenance.json` names, read
from the store), each the document start and its first tokens, and the
post-trained model alone writes the answers, each up to the end of a
document or `<|im_end|>`, the end of a turn (maintainer's decision,
2026-10-18): no text people wrote as an answer, and no conversation of a
step, reaches it. The model is frozen
while the drafter learns DSpark's three losses from random anchors in the
answers; a tenth of the prompts is held out, and on them the drafter
drafts for the model, every draft verified, to calibrate its confidence on
the drafts the model kept and rejected and to measure how many it keeps.
The run records its settings, each update's losses, the calibration (each
position's temperature, the calibration error before and after) and the
measure through the run's record. A drafter that cannot be trained (too
few prompts, a loss that is not finite, no draft verified) fails the run
with `DrafterFailed`, and nothing is written.

Then the run measures the drafter scheduler's pass times for the new
weights once and writes them into its record as `scheduler-costs.json`,
with the setup they were measured on; `--scheduler-costs FOLDER` replays
those an earlier run recorded in its record folder `FOLDER` instead,
and times recorded on another setup are refused before any step runs
(maintainer's decision, 2026-10-18; [model.md](model.md), "The drafter",
says exactly what a replay guarantees). Drafter settings that cannot be,
such as feature layers the model does not have, are refused before any
step runs too.

The new folder holds it in `drafter/`, with its own `provenance.json`: the
command and recipe that trained it, the dataset, the sha256 of each weight
file it was trained against, its settings, its losses, the calibration and
the measure. Its prompts come from the dataset's sources, which `NOTICE`
already credits, so it adds none; the model card describes it.

## The `grpo` step

The `grpo` step teaches the model to finish tasks as an agent, by
reinforcement learning (group relative policy optimization). The model
works in a small harness, on tasks that people write, with four built-in
tools, `read` (a file), `write` (a file), `edit` (replace a string that
occurs exactly once in a file) and `bash` (run a command), and the tools a
task adds ("Tools", below). Its code is the `wpt-agent` package (tools,
containers, images, the agent loop and the scoring) and `steps/grpo.py`
(the learning). It needs Docker on the machine. Until a source holds tasks
tagged `grpo`, it fails with "no tasks" and writes nothing.

### Tasks

Tasks are written by people, in an `empty` source (CONTRIBUTING.md,
"Adding your own texts"), as files in the `task` format tagged `grpo`. One
folder per task:

```text
<folder>/task.json   what the model is asked, the image, the limits, the tools it needs and, if needed,
                     larger containers, its own tools and the user's replies
<folder>/files/...   the starting files, copied into the container's /work (those tool_files names
                     go where only the tools' programs read them)
<folder>/check       any program (its first line names its interpreter, such as #!/bin/sh)
```

```json
{"schema": 1, "instructions": "Fix the failing test in calc.py.", "image": "python",
 "limits": {"max_turns": 20, "max_tokens": 4000, "command_seconds": 30},
 "container": {"memory": 8192}, "needs": [["read", "bash"], ["edit", "write"]],
 "replies": [{"pattern": "(?i)python.*version", "reply": "Python 3.13."}]}
```

- `instructions`: the user message that starts every attempt. No task's
  instructions state an answer block: the model's final reply is its
  answer, as text, and the harness asks for its confidence after it
  ([chat-format.md](chat-format.md), "The answer"; maintainer's decision,
  2026-10-23). The harness adds no sentence to the opening (maintainer's
  decision, 2026-10-22).
- `terse` (optional): the same request in one line, for the measure of
  terse against full requests ([evaluation.md](evaluation.md), "Terse
  and full requests"): a task can hold its request in two forms, its full
  `instructions` and this one-line form (maintainer's decision,
  2026-10-23). The agent loop opens an attempt with `instructions`, or
  with `terse` when its caller asks (`wpt_agent.attempts(...,
  terse=True)`; a task without the field is refused there). It holds no
  line break: `wpt-dataset check --fetch` refuses one, naming the
  `task.json`.
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
- `needs`: the tools the task cannot be done without, each need the list
  of its variants (two different tools that write a file, say); every
  attempt is offered exactly one variant of each need ("Tools", below).
  `[]` when the task needs none.
- `replies` (optional): the user's replies, written in advance, to the
  `ask` tool, each a `pattern` and its `reply`; only a task with replies
  offers `ask`.
- `answered_by` (required with `replies`): the looking tools that can
  answer what the task leaves open, `read`, `bash` or a declared tool
  marked `looks` (the task's or its image's pool's). An attempt asks the
  user only after at least one look through one of them (`asked_last`,
  "The check library", below; maintainer's decision, 2026-10-22).
- `tools` (optional): the task's own tools beyond the built-in ones and
  `ask`, each with its `name`, `description`, `parameters` (the JSON
  Schema of its arguments), `program` (the absolute path, in the image, of
  the program that answers it) and, optionally, `names` (other names an
  attempt may offer it under) and `looks` (`true` for a tool that looks
  at sources as `read` does, such as a search or a fetch; "The check
  library", below).
- `tool_files` (optional): paths under `files/`, each a file or a folder
  with everything in it, that only the tools' programs read: they go to
  `/tmp/wpt-tools/` instead of `/work`.
- `absent` (optional): absolute folders of the image, outside `/work`
  and `/tmp`, their names made of letters, digits, `.`, `_`, `+` and `-`,
  that the attempts do not see: each is covered by an empty,
  read-only folder kept in memory (Docker's `--tmpfs`), so a task removes
  what its image installs there, such as the manual of `ondrik` ("The
  agent image", below). The check's container shows the image as it is.
- `pointers` (optional): the texts the task takes from documents of other
  sources ("Pointer tasks", below).

A task folder is a folder holding `task.json` that is not inside another
task's `files/`; every other file of the format is a task folder's `check`
or under its `files/`. `wpt-dataset check --fetch` checks each folder
against `packages/wpt-dataset/src/wpt_dataset/schemas/task.schema.json`,
and checks that its `image` is a folder of `data/images/` holding a
`Dockerfile`. CI does not fetch the sources a change touches: the
maintainer runs `wpt-dataset check --fetch` on a pull request's sources
before merging, which catches a task naming an image that is gone
(maintainer's decision, 2026-10-21). The same check refuses a task whose
sizes break the rule on `disk` and `memory` above, naming its `task.json`,
and one whose tools break the rules of "Tools" (below): a declared tool
taking a built-in name or `ask`, or declared twice; a need naming a tool
the task does not offer (`ask` without replies included); a tool in two
needs; a task with replies and no `answered_by`, or an `answered_by`
naming a tool that is not a looking tool; a reply pattern that is not a
regular expression; a `terse` that holds a line break; a `tool_files`
entry that names nothing under `files/`; an `absent` entry that is `/work`
or `/tmp`, lies under them, or holds `.` or `..`; a pointer that names a source that does not exist or takes the
place of another file, or whose text cannot be filled in ("Pointer tasks",
below). The step reads the tasks whose `task.json` carries the tag. Like
every file of a source, task files are UTF-8 text.

The task editor of `wpt-web` and `wpt-dataset save-task` write a task
folder through the service layer:

- only in an `empty` source;
- never at the source's root: a `task.json` there (folder "") is shown,
  and changed only with `unpack` and `patch`;
- as a patch that holds only the files whose bytes change, so every other
  file keeps its bytes and its mode, and a changed file keeps its
  execute bit; `task.json` keeps its layout where its meaning did not
  change;
- the text of each file is UTF-8, and a file that is not refuses the read;
  a symbolic link under `files/` is not shown as text, stays as it is, and
  a save that gives it text is refused, so no write goes through a link.

### Pointer tasks

Tasks a program builds (the task builder) are committed like the others,
but the repository holds pointers to data, never the data
([architecture.md](architecture.md)): such a task file holds, for each text
it needs, the source id, the document and the span, plus what the builder
changed, such as the sentence it removed. The step and the measures fill in
the text from the store, checked against its checksum (maintainer's
decision, 2026-10-18). Each entry of `pointers`:

```json
{"for": "files", "path": "web/example.org/lamp", "source": "<source id>", "document": "chapter-3.xhtml",
 "span": [1200, 4810], "sha256": "<sha256 of those bytes>", "removed": [[2210, 2290]]}
```

- `source`, `document`: the document's source and its path there, as the
  build names it (a file's path, or `<file>#<record id>` for a record).
- `span`: `[start, end)`, bytes of the document's Markdown, the text the
  model's training would read ([architecture.md](architecture.md),
  "Documents are Markdown"); `sha256`: the sha256 of those bytes.
- `removed` (optional): the parts of the span the builder left out, in
  order, in the same bytes.
- `for` and `path`: where the text goes. `files`: a file at `path` under
  `files/`, in `/work`, or among the hidden files when `tool_files` names it
  (a page a search answers from, say); `check`: a file at `path` next to
  the check, which only the check reads (the passage the builder removed,
  say; "Scores", below).

When a task is read, every source its pointers name must be admitted; its
final tree comes from the store (fetched when it is missing there), the
document is converted as the build converts it, and the span must hold
the bytes its `sha256` names and must not cut a character. The text is
then the span without its removed parts. A task breaks when its source's
revision changes what the span holds, and is refused, naming the task and
the pointer, until the builder makes it again (maintainer's decision,
2026-10-18). `wpt-dataset check --fetch` fills in every pointer of the
sources it checks, and `check --changed` also checks every task source
whose pointers point into a source the change touches.

A repository pointer names a commit of a git source's history instead, for
the tasks that commit prepared changes (maintainer's decision, 2026-10-18:
the task file names the source and the commit):

```json
{"for": "repository", "source": "<git source id>", "document": ".git/history/<commit>",
 "checksum": "sha256:<the source's checksum when the task was built>"}
```

The task starts from the real state just before the commit (maintainer's
decision, 2026-10-22). When it is read, the source must still have the
checksum the pointer names, which covers every history file and the
history's order file ([architecture.md](architecture.md), `history`), so
the same pointer always gives the same repository; a source whose
checksum changed refuses the task until the builder makes it again. From
the source's final tree, every commit after the pointer's commit is
undone, newest first, in the order the order file stores, then the
commit itself. Undoing the later commits gives the commit's own tree, the
tree the pushed commit must hold; undoing the commit too gives its
parent. A merge between the commit and the final tree, or a diff that
does not undo, makes the task fail to be filled, as does a history file
whose author and date git cannot make the parent commit from (an empty
name or email); the builder never draws a commit behind a merge, and
skips the others. Any other git failure is not the pointer's and stops
the build.

`/work` is the repository itself (maintainer's decision, 2026-10-22: the
request "Based on the current diffs, could you commit and push the
changes in this repository?" means `/work`). Its one commit, on the
branch `main`, holds the parent tree (the store keeps no older commit, so
this commit stands for the history: its author and date are the
commit's, its message names the commit it comes before). Its working tree
holds the commit's own tree, so the commit's changes are there and not
committed, and its git settings name the commit's author. A task with a
repository pointer has no other file in `/work`: no `files/` and no
pointer to files. Its `origin` is a bare repository holding the same
commit, outside `/work`, at `/tmp/origin.git` in the container, so `git
add -A` in `/work` never takes it (maintainer's decision, 2026-10-22). No
commit message is given: the model writes its own (maintainer's decision,
2026-10-22). Next to the check, `repository.json` holds what the pushed
commit must be: the remote (`/tmp/origin.git`), the branch, the commit it
starts from (`parent`) and the tree the pushed commit must hold (`tree`,
the commit's own tree). A task has at most one repository pointer. The
task counts as built from the commit's document, `.git/history/<commit>`,
for "Measure tasks", below.

### Measure tasks

The measures' tasks are kept out of training by two rules (maintainer's
decision, 2026-10-18), which the task reader (`wpt_dataset.tasks.read`)
applies:

- people tag the measure tasks they write `eval`, and no training step
  (`grpo`, `token-efficiency`) reads a task so tagged, whatever its other
  tags;
- a fixed share of every task source is measure-only, chosen by a hash of
  the task id the same way held-out documents are chosen
  ([data-selection.md](data-selection.md), "Held-out text and
  contamination"): the share is the model recipe's `held_out`, the same
  rule as documents (maintainer's decision, 2026-10-22). A step reads it
  from the base model's `provenance.json`, the recipe whose `held_out`
  kept its documents out.

A pointer task built from a held-out document is measure-only too (a
document of a source with `held_out_by` is held out by its group, as the
build holds it out: [data-selection.md](data-selection.md), "Held-out text
and contamination"; maintainer's decision, 2026-10-23), and so
is a pointer task whose document's text equals a held-out document's
(maintainer's decision, 2026-10-22): when the pointers are filled in
(`tasks.fill`), the sha256 of each named document's whole Markdown is kept,
and `measure_only` compares it with the sha256 of the Markdown of every
held-out document of the base model's dataset, the one its
`provenance.json` names (`held_out_texts`, read once for each pass that
reads tasks), so the same text under another source or path is caught
too. Those sha256 come from the dataset's document list, which its build
keeps ([data-selection.md](data-selection.md), "Held-out text and
contamination"); no source is converted to find them (maintainer's
decision, 2026-10-22). A dataset built before builds kept that list has
none: reading pointer tasks against it is refused
(`DocumentListMissing`), saying that the base model's dataset predates
document lists and that such a base is used with the repository at the
commit that built it; nothing falls back to converting the sources, and
nothing is rebuilt (maintainer's decision, 2026-10-22). A dataset the
store does not hold at all is refused the same way, the message saying
it is not in the store. A held-out document never reaches training
through a task. A repository pointer names only its commit's document:
the other files of the repository's tree are not looked at by this rule.
The measures read exactly the measure-only tasks
(`read(..., measure=True)`), whatever their tags (`tag=None`): those
tagged `eval`, the hash share of every task source and the pointer tasks
on held-out text, all in one run of attempts that every agent measure
reads (maintainer's decision, 2026-10-23; [evaluation.md](evaluation.md),
"The shared run of attempts").

### Tools

Every attempt is offered the tools of its task, chosen for that attempt
("Variation", below). A task's tools are:

- the four built-in ones, `read`, `write`, `edit` and `bash`;
- `ask`, when the task has `replies`;
- the tools the task declares;
- and the working tools of the image's pool (`tools.json`, "Images",
  below), whose names the task does not take.

**`ask`** (maintainer's decision, 2026-10-18). The model calls it with its
question, and the harness answers from the task's replies, written in
advance; the attempt goes on. Each reply's `pattern` is a Python regular
expression searched for anywhere in the question, and the first reply
whose pattern is found is given. A question that no pattern matches gets
"I don't know that", so a precise question pays and a vague one does not.
No person waits for the model's question: the reply is written ahead and
the harness gives it, as content a person wrote that the task records
(Rules, above). `ask` is offered only in a task that declares replies
(maintainer's decision, 2026-10-22). It serves asking the user last: a
turn without a tool call still ends the attempt, so a question goes
through `ask`.

**Declared tools** (maintainer's decision, 2026-10-18). A task declares
each tool it offers beyond the built-in ones and `ask`: its name,
description, parameters, and the program in the agent image that answers
it. The program runs inside the attempt's container, under its limits and
`command_seconds`: it reads the call's arguments as JSON on its standard
input (as ASCII, any other character escaped) and runs from `/work`; its
standard output is the result, and an exit code other than 0 makes the
call fail, with its standard error (or, when it printed none there, its
output) as the result. Like a check, a program must not load code from
its current folder: the model's files are there. Most of these tools are
emulations, such as a web search answered from local files; an emulation
gives the real tool's kind and shape of answer: a search returns ranked
snippets with their sources, a fetch returns a page's text. So the model
learns what kind of answer to expect from a kind of tool and adapts to
whatever harness it is put in. Real MCP servers are not used in training.

**Hidden files** (maintainer's decision, 2026-10-22). An emulated tool's
backing files are hidden from the model's own tools (`read`, `bash`): only
the tools' programs read them. The task names them in `tool_files`; they
are unpacked into `/tmp/wpt-tools/`, owned by root and readable only by
group 1001 (folders 750, files 640). A program runs as user 1000, the
model's, in group 1001, so it reads them and changes `/work` as the
model's own tools do; the model's commands run in group 1000, so they can
neither list nor read them, and since the groups differ the kernel lets
them neither look into the program's process (`ptrace`,
`/proc/<pid>/fd`, `/proc/<pid>/mem`) nor gain group 1001 (no setgid).
The files count against what `/tmp` may hold. The check's container does
not get the task's `tool_files`, but it does get the files an attempt's
variant hid (a variant's `tool_files`, "Variation", below): they are back
at their place, in its `/work` or at their folder outside it, as the
tools' programs left them, in place of whatever the model put there, so
the check of a commit-and-push task reads the remote the push tool pushed
to (maintainer's decision, 2026-10-22). A folder a task puts outside
`/work` (a repository pointer's remote, `/tmp/origin.git`) belongs to user
1000 in the attempt, and the check's container gets it too, at the same
place, as the attempt left it.

**Variation** (maintainer's decision, 2026-10-18). Every task lists the
tools it absolutely needs (`needs`), and a needed tool may have variants.
For each attempt the step (and the measures) draws, from a generator
seeded with the step's `SEED`, the task, the round, the group and the
attempt, so runs are reproducible (`wpt_agent.offer`):

- one variant of each need, under its own name; a variant may be tools
  offered together with paths under `files/` that it hides (its
  `tool_files`, `{"tools": [...], "tool_files": [...]}`), which go to
  `/tmp/wpt-tools/` in that attempt only; a repository pointer's remote is
  hidden by naming its path, `/tmp/origin.git`, and goes to
  `/tmp/wpt-tools/tmp/origin.git`;
- each other tool of the task and of the pool (the variants not drawn
  aside) left out or kept, with even odds; a kept one is offered under
  one of its other names, with even odds, when one is free (the built-in
  tools' and `ask`'s other names are `OTHER_NAMES` in
  `packages/wpt-agent/src/wpt_agent/tools.py`, a declared tool's its
  `names`);
- every offered tool in a shuffled order.

The tools of an image's pool pad the offer with tools the task never
needs; each really works, answered by its own program in the image
(maintainer's decision, 2026-10-22). The map from each name the model saw
to the real tool goes into the conversation the check reads ("Scores",
below). The aim is asking the user last without habits tied to particular
tools (maintainer's decision, 2026-10-18).

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

An image's pool of tools is `data/images/<name>/tools.json` (optional),
written by people like the Dockerfile: `{"schema": 1, "tools": [...]}`,
each tool declared as a task declares its own, its program a file the
Dockerfile puts in the image. `wpt-dataset check` checks it against
`image-tools.schema.json` and refuses a tool declared twice or taking the
name of a built-in tool or `ask`. A changed pool is a changed image
folder, which needs a new build.

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
otherwise. A failed `docker build` (in `lock-image` or `build-image`) and
a failed `apt-get update` in that check are tried again by the same rule,
6 times in all, with a growing pause before each new try: 10 s, 30 s,
1 min, 2 min and 5 min, about 9 minutes of waiting, because
snapshot.ubuntu.com was once down for over 2 hours (maintainer's decision,
2026-10-21). The error of a failed build ends with the output of the step
that failed. The `grpo` step never builds an image: it stops before any
attempt when a task's image is not built for the folder's current files,
and names the command that builds it. A changed Dockerfile or lock needs
a new build.

### The agent image

`data/images/agent/` is the first image of agentic tasks (maintainer's
decision, 2026-10-18): Debian 13 with `bash`, `git` and Python 3, and
four parts of its own, which its Dockerfile copies in:

- `wpt_check.py`, the shared check library ("The check library", below),
  in `/usr/lib/python3/dist-packages/`, where every `python3` finds it, `-I`
  and `-P` included;
- the programs that answer tools, in `/usr/local/lib/wpt-tools/`, written
  in Python and started as `python3 -I`, with the helpers they share,
  `wpt_tools.py`, next to the library;
- `tools.json`, its pool of working tools;
- `ondrik`, a small program written for WPT, so no training text
  describes it, in `/usr/local/bin/`, and its manual,
  `/usr/share/doc/ondrik/manual.txt` (maintainer's decision, 2026-10-23;
  `ondrik/` in the image's folder), for the task builder's
  `read-the-manual` ("The task builder", below). It is compiled
  (maintainer's decision, 2026-10-23): its source, `ondrik/ondrik.c`, is
  compiled in a build stage of the Dockerfile, and the image keeps only
  the binary, owned by root, which the model's user may run but not
  read, and no compiler.

A task declares the two emulations it uses as its own tools:

```json
{"name": "search", "description": "Search the web; returns the best pages, each with its address.",
 "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
 "program": "/usr/local/lib/wpt-tools/search", "names": ["web_search"], "looks": true}
{"name": "fetch", "description": "Fetch a web page and return its text.",
 "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
 "program": "/usr/local/lib/wpt-tools/fetch", "names": ["web_fetch"], "looks": true}
```

Both answer from the task's pages, hidden from the model's own tools: the
files under `files/web/` that `tool_files` names (`"tool_files": ["web"]`),
each the text of the page whose address is `https://` and its path there,
`web/example.org/lamp` being `https://example.org/lamp`. `search` ranks the
pages' paragraphs by BM25 and gives, as a search engine does, the 5 best
pages, each with its title (its first heading, or else its first line),
its address and a snippet of its best paragraph, or "No results found".
`fetch` gives a page's text; an address ending with `/` is its folder's
`index`, the query and the fragment do not count, and a page that is not
there fails with "404 Not Found".

The pool (`tools.json`) pads the offer with tools that really work in
`/work` (maintainer's decision, 2026-10-22): `list_files` (a folder's
entries, as `ls -p`), `find_files` (paths matching a glob pattern),
`search_text` (lines matching a regular expression, as `grep -rn`) and
`calculator` (an arithmetic expression); the first three are marked
`looks`. Since a program runs in the group
that reads the hidden files, these tools open a path only inside `/work`,
one folder at a time and never through a symbolic link, so neither a path
leading out of `/work` nor a link the model made reaches the hidden files.

A task that commits and pushes (the task builder's `commit-and-push`,
below) declares the emulated `push` (maintainer's decision, 2026-10-18):

```json
{"name": "push", "description": "Push the commits of the current branch of a git repository to its remote.",
 "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
 "program": "/usr/local/lib/wpt-tools/push", "names": ["git_push", "push_changes"]}
```

It runs `git push` in the repository at `path` (by default `/work`) and
gives what git printed. In the attempt whose variant hides the remote
(the task builder's `commit-and-push`, below), the remote is among the
hidden files under its own path: when the remote git would push to is a
local path outside `/work` (`/tmp/origin.git`) whose hidden copy exists
under `/tmp/wpt-tools/` (`/tmp/wpt-tools/tmp/origin.git`), the program
pushes there, so only it reaches the remote and `git push` through `bash`
does not (maintainer's decision, 2026-10-22). git reads the repository's own
settings, which the model can change, and the program runs in the group
that reads the hidden files: it turns hooks and the file system monitor
off, and a task whose hidden files are more than the remote it pushes to
must not offer it.

`ondrik` prints results that only it gives, by rules of its own: `ondrik
press [--turns N] TEXT` the seal of a text (10 signs, folded `N` times),
and `ondrik tie [--give] A B` the knot of two whole numbers (6 digits;
with `--give`, the slack knot). It prints no help of its own: a call it
cannot read fails with code 2 and names its manual, so a model that has
not read the manual can only guess the commands and options. The manual
is a plain text file under `/usr/share/doc/`, since the image installs
no program that shows manual pages. A task removes it by naming its
folder, `/usr/share/doc/ondrik`, in `absent` ("Tasks", above). Without
the manual, nothing in the image documents the program: the binary
cannot be read by the model's user, so its source, its strings and its
rules stay out of reach. The text of a seal starts with no `-`, so it
never reads as an option, and `--turns` takes 1 to 1,000,000.

A change to the library or to a program changes the image's folder, so
every machine builds the image again (`wpt-agent build-image agent`); a
change to its packages also locks it again (`lock-image agent --force`).
The library and the programs are tested in the repository
(`packages/wpt-agent/tests/test_agent_image.py`), once for every check
that calls them, and in the built image itself.

### The check library

Rules many checks share live in one library inside the agent images,
tested once in the repository, and each check calls it (maintainer's
decision, 2026-10-18). A check started as `python3 -I` imports it:

```python
#!/usr/bin/python3 -I
import wpt_check
conversation = wpt_check.load()  # conversation.json, next to the check
right = ...  # the task's own judgement of the files and the answer
print(wpt_check.scored(conversation, float(right)))
```

How `brier` weighs in a task's score is the rewards stage's rule (#10, #40:
correctness plus the Brier score, correctness weighted above 1), added to
this library; the example weighs nothing yet.

The rules read the attempt's trace by real tool names (`metadata.tools`
maps each name the model saw to the real one):

- `scored(conversation, score)`: the missing-answer rule. The attempt's
  score is 0 when it has no answer, whatever its files hold: no final
  reply, or a reply to the confidence question that is not a number from
  0 to 100 ([chat-format.md](chat-format.md), "The answer"; maintainer's
  decisions, 2026-10-18 and 2026-10-23).
- `brier(conversation, correct)`: the Brier score of the confidence,
  `(confidence - 1)²` for a right result and `confidence²` for a wrong one;
  1, the worst, without an answer.
- `read_before_write(conversation, new=())`: every file written or edited
  was read first, by `read`, by an earlier `bash` command that names it,
  or written by the attempt itself; `new` lists the files the task expects
  the model to create.
- `sources_first(conversation, changes=("write", "edit", "ask"))`: the
  attempt looked at sources (below) before its first change or question,
  and looked at all when it did neither.
- `asked_last(conversation)`: every `ask` came after at least one look
  that could find the answer: a call that looks (below) to one of the
  task's `answered_by` (`metadata.answered_by`), not to every one of them
  it was offered; true when it was offered none of them (maintainer's
  decision, 2026-10-22).
- `read_before_ask(conversation, text)`: every `ask` came after a call
  that looks and whose result showed `text`, its words in order as `holds`
  reads them; true when the attempt asked nothing (maintainer's decision,
  2026-10-22).
- `before(conversation, first, then)`: every call to a tool of `then`
  after a call to one of `first`; `calls` gives the trace, each call by
  its real tool's name and whether it looks, `answer` the answer the loop
  built, `{"answer": ..., "confidence": ...}` or `None`, and `answer_text`
  its text, the final reply's, `""` without one.
- `holds(text, expected)`: `text` holds the words of `expected` one after
  the other, letter case, punctuation, Markdown marks and spacing aside.
- `not_enough(conversation, missing="", words=None)`: the answer starts
  with "not enough", the answer of a task whose files do not hold what it
  asks; given `missing`, the text the files lack, the answer must also
  name it: hold its first `words` words (all of them by default) in
  order, as `holds` reads them (maintainer's decision, 2026-10-22).
- `pushed(expected=None)`: the remote of a repository pointer
  (`repository.json` next to the check, by default) holds on its branch
  exactly one new commit, on the commit it started from, with the expected
  tree and a message that is not empty, spaces aside. The model writes its
  own message, and the check never compares it (maintainer's decision,
  2026-10-22). The remote is outside `/work`, at `/tmp/origin.git`. Since
  the model could have changed the remote's settings, only its objects and
  references are copied into a fresh repository of the check's own, so git
  reads none of them.
- `beside(name)` and `file_text(path)`: the text of a file next to the
  check, and of a regular file the attempt left (`""` for none, a folder,
  a pipe or a link).
- `programs(command)`: the programs a `bash` command runs, as the looking
  rule finds them (below).

Looking is one rule, which `sources_first`, `asked_last` and
`read_before_ask` share (maintainer's decision, 2026-10-22): every looking
tool counts as opening a source. A call looks when it calls `read`, a
declared tool marked `"looks": true` (the agent image's `search` and
`fetch`, and its pool's `list_files`, `find_files` and `search_text`), or
`bash` with a command that runs one of the image's reading programs. Those
programs are listed once, as `READERS` in `wpt_check.py`: the programs of
the image that show what files or folders hold, such as `cat`, `head`,
`grep`, `sed`, `ls` and `find`, and a test checks that the built image
holds each of them. The command is parsed as bash reads it, never searched
for a name: each simple command of its pipelines and lists (`|`, `;`,
`&`, `&&`, `||`, a line break, parentheses, a function's body) runs the
program named by its first word, after any `VAR=value` and bash's reserved
words, and after programs that run another program (`env`, `nice`,
`nohup`, `stdbuf`, `time`, `timeout`, `xargs` and the like) with their
options and the words they take; a comment does not count. So `cd docs &&
cat a.md` looks and `echo cat a.md` does not. Which offered tools look is
decided in one place, `wpt_dataset.tasks.looks` (`read`, `bash`, and a
declaration marked `looks`): the agent loop writes their real names into
the conversation's `metadata.looking`, which the library reads, and
`wpt-dataset check` refuses an `answered_by` naming any other tool.

The rewards stage adds its rules here.

### The task builder

Program-built tasks are generated by a script in the repository,
`scripts/build_tasks.py`, as committed task sources made of pointer tasks,
so adding, removing or varying tasks is a rerun of the script (maintainer's
decision, 2026-10-18). Its code is `wpt_dataset/builder.py`:

```sh
uv run scripts/build_tasks.py KIND SOURCE --count N [--seed S] [--select EXPRESSION]
```

`SOURCE` is an `empty` source a person makes for the tasks, like any text
written for WPT (`wpt-dataset new-source`, then its license text, tagged
`grpo`); a maintainer admits it. A run replaces everything under `KIND/` in
it by `N` tasks, as one new patch checked as `check --fetch` checks it, every
pointer filled in. The tasks come from the admitted sources already in the
store, drawn with a generator seeded by `S` (default 1): the same seed,
sources and store give the same tasks. A source that is not in the store
gives no task, and the run says how many were left out. Each task's folder
is `KIND/<16 hex digits of the sha256 of its task.json>`, so an unchanged
task keeps its id over reruns, and with it its place in or out of the
measure-only share ("Measure tasks", above). Every task runs in the agent
image, and its check calls the check library, which reads the attempt's
answer: the final reply's text and the confidence the model gave when the
harness asked for it after that reply ([chat-format.md](chat-format.md),
"The answer"). No kind's instructions say anything about a block or how
to answer (maintainer's decision, 2026-10-23), and the harness adds
nothing to the opening (maintainer's decision, 2026-10-22).

The three kinds built on text take the documents of the sources that
`--select`, a pool expression ([data-selection.md](data-selection.md)),
selects by their tags: from each source, in a drawn order, the same share
of the tasks, from documents drawn with equal odds. A task's texts are
windows of whole paragraphs of a document's Markdown, at most 3,000 bytes,
so one read (whose result keeps 4,000 bytes) gives a whole text. The
paragraph a task asks for is plain text of 120 to 800 bytes, and
`quote.txt` holds its start, about its first 40%, which occurs in no other
text of the task. Every kind but `look-or-ask` gives its texts in half its
tasks as files of `docs/` (3 texts), and in the other half as web pages
only `search` and `fetch` reach (5 texts, hidden files named after the
source's address and the document).

- `find-the-passage`: the answer is the rest of the paragraph, word for
  word (`holds`).
- `evidence-removed`: the same, but in half the tasks the paragraph is
  removed from its text (the pointer's `removed`), and the right answer
  starts with "not enough" and names what the question asked for: the
  paragraph, by the words of its quoted start. The removed paragraph is
  next to the check, as `missing.txt`, and the check reads it: the answer
  must hold the first words of `missing.txt`, as many as `quote.txt`
  holds, in order, matched by words as `holds` matches them
  (`not_enough`). The other half can be answered, and "not enough" is
  wrong there (maintainer's decisions, 2026-10-22).
- `look-or-ask`: the rest of the paragraph goes into a file whose name is
  the one missing detail: in half the tasks `NOTES.md` names it, in the
  other half only the reply of the `ask` tool does (and `ask` is needed);
  both halves declare the same reply. The task's `answered_by` names the
  looking tools that can read `NOTES.md`: `read`, `bash` and the pool's
  `search_text`. The check wants the file's text and `asked_last`: no
  question before at least one look through one of those. In the
  half where `NOTES.md` names the file, it also wants `read_before_ask`:
  the attempt read `NOTES.md`, through any looking tool, before any
  question (maintainer's decisions, 2026-10-22).
- `commit-and-push`: every commit of each admitted git source's stored
  history that undoing can reach is a candidate, not only the newest
  (maintainer's decision, 2026-10-22), when the source's working tree
  stays within 4 MiB and 2,000 files. A source gives one candidate per
  history file (a merge has none) of a commit newer than its newest
  merge, read from its stored order file (every commit's, when the
  history holds no merge); the commits behind that merge, which undoing
  cannot reach, are dropped before drawing and never logged
  (maintainer's decision, 2026-10-22). A source whose stored history has
  no order file (stored before histories kept one) gives none, and the
  builder says how many such sources it left aside. The builder draws
  about `--count` of them (2,500 per kind) from all the candidates, in an
  order seeded by `S`, so the same seed, sources and store give the same
  tasks. A drawn commit's task starts from the real state just before it:
  every later commit undone, newest first, in the stored order, then the
  commit ("Pointer tasks", above). Only a real failure is logged per
  draw: a drawn commit whose undoing fails, or whose parent commit git
  refuses to make, is left out, named, and the next drawn takes its place
  (maintainer's decision, 2026-10-22). Only a drawn commit is made into a
  repository: the candidates are history files, not trees. Each kept
  commit is a repository pointer naming the source's checksum ("Pointer
  tasks", above), and `/work` is the repository, its remote outside it. The
  model is asked only "Based on the current diffs, could you commit and
  push the changes in this repository?", with nothing after it
  (maintainer's decisions, 2026-10-22 and 2026-10-23); it writes its own
  commit message, then pushes. The check is `pushed`: exactly one new commit
  pushed to the remote branch, whose tree is the commit's own tree, its
  message any that is not empty, never compared (maintainer's decision,
  2026-10-22). Each attempt pushes one way, the seeded
  choice between the two variants of the task's one need ("Variation",
  above; maintainer's decision, 2026-10-22): `bash` alone, so no push tool
  is offered and `git push` through `bash` reaches the bare `origin` at
  `/tmp/origin.git`; or `bash` with the declared `push`, a variant that
  hides the remote (its `tool_files` name `/tmp/origin.git`), so `git
  push` through `bash` cannot reach it and only the push tool's program
  does ("The agent image", above). Over a task's
  attempts both ways come up, so no habit is tied to one way of pushing
  (maintainer's decision, 2026-10-18). Its commits come from the admitted
  GitHub repositories (admitted on 2026-10-21).
- `read-the-manual` (maintainer's decision, 2026-10-23): each task asks
  for a result only the image's `ondrik` gives ("The agent image",
  above): the seal of a drawn text (8 hex digits) folded a drawn number
  of times (2 to 9), or the slack knot of two drawn numbers (100 to
  99,999). The instructions name the program and the result, never the
  option that gives it, and are the same whether the manual is there or
  not. The halves are crossed: every other task names the manual's
  folder in `absent`, so its attempts run without the manual, and every
  other pair asks for a seal instead of a knot. The task needs `bash`
  only (`[["bash"]]`) and has no files. Its check runs the program, by
  its absolute path, in the check's own container, which the model's
  files cannot change, and wants the program's output in the answer
  (`holds`). It reads no source and takes no `--select`. It is the
  agent form of the reading-gain measure ([evaluation.md](evaluation.md),
  "Reading the manual"). Its tasks are for the measures only
  (maintainer's decision, 2026-10-23): `builder.write` tags the kind's
  folder `eval` in the task source's `source.json`, through the service
  layer, when no entry tags it so, and the task reader gives tasks tagged
  `eval` to the measures only ("Measure tasks", below). A model trained
  on them would learn `ondrik` from the attempts themselves, and the
  measure would no longer tell whether it reads the manual.

Each kind is marked as acting, checking or both (maintainer's decision,
2026-10-23; `CLASSES` in `builder.py`): `commit-and-push` acts;
`find-the-passage`, `evidence-removed` and `look-or-ask` check;
`read-the-manual` does both, since it runs a program and checks its
output. After each run, `scripts/build_tasks.py` reports what the task
source holds: its built tasks of each kind, and the shares of them that
act only, that check only and that do both, which sum to 100%
(`builder.shares`), such as:

```text
2500 look-or-ask tasks written into <source id>
<source id> holds 10,000 built tasks (find-the-passage 2,500, evidence-removed 2,500, look-or-ask 2,500, commit-and-push 2,500, read-the-manual 0): acting only 25.0%, checking only 75.0%, both 0.0%
```

The first committed task set holds all 4 kinds, about 2,500 tasks of
each, about 10,000 in all, and reports its shares this way (maintainer's
decision, 2026-10-22). The 70/30 mix between the two shares is settled
later, once more kinds of tasks are defined: only the program-built
kinds are defined so far. Building the set needs every admitted source
it reads fetched into the store first (`uv run wpt-dataset check
--fetch`), on a machine that holds them, the task sources made and
admitted, and one run per kind with `--count 2500`.

### Attempts

For each task, the model makes several attempts (`GROUP`), written
together: the model reads the start the attempts' openings share once,
through its generation cache ([model.md](model.md), "Generation"), copies
it into one row per attempt, reads in each row the rest of that attempt's
opening (its offered tools differ), and writes the attempts' turns in one
batch; while one attempt's tool calls run, the others wait. Each attempt
runs in a fresh container of the task's image, removed after the attempt;
a group's containers run at the same time.
The container has no network and a read-only root: it can write only
`/work`, its working folder holding the task's `files/`, and `/tmp`, two
folders kept in memory. Its sizes are the defaults of the task format
(`SIZES` in `packages/wpt-dataset/src/wpt_dataset/tasks.py`), each raised
by the task's `container` field when it asks for more: `memory` (4096 MiB,
the files in `/work` and `/tmp` included), `disk` (1024 MiB, what `/work`
and `/tmp` may each hold), `processes` (256, processes and threads at
once) and `cpus` (2). The check's container gets the same sizes, and must
hold a full copy of `/work`: that is why `HEADROOM` (1024 MiB) of memory
stays free for the processes when `/work` and `/tmp` are both full. The
model's commands run as user 1000, not root, and nothing in the container
can gain privileges (no `setuid` program works).
The container keeps only three capabilities, `CHOWN`, `DAC_OVERRIDE` and
`FOWNER`, which root uses to unpack the task's `files/` into `/work` with
user 1000 as their owner; the model's commands, as user 1000, hold none of
them. The container's first process is Docker's init (`--init`), which
collects every process that ended, so background commands that finished
never count against the process limit; the check's container starts the
same way.

The conversation follows [chat-format.md](chat-format.md): the tools the
attempt is offered, then the instructions as the user's message. The model
writes an assistant turn; each tool call in it runs in the container (a
relative path starts at `/work`, and a call that cannot run returns its
error to the model); the
results come back as a tool turn; and the model writes the next turn. The
attempt ends when a turn closes without a tool call, when the model writes
the end of a document, after `max_turns` turns (the last turn's calls still
run), or when `max_tokens` is reached in the middle of a turn, whose calls
then do not run (maintainer's decision, 2026-10-17). The turn that closes
without a tool call is the final reply, the answer; the loop then adds one
user turn asking for the confidence, a number from 0 to 100 and nothing
else, and the model replies; that turn counts against `max_tokens` and
the context, not `max_turns` ([chat-format.md](chat-format.md), "The
answer"; maintainer's decision, 2026-10-23). The loop builds
`{"answer": <the final reply's text>, "confidence": <number / 100>}`, or
`null` when there is no final reply or the reply is not such a number,
and keeps it in the conversation's `metadata.answer`, next to `metadata.tools`, the map from each name the
model saw to the real tool, `metadata.looking`, the real names of the
offered tools that look at sources, and `metadata.answered_by`, the
task's ("The check library", above). The model's
`block_size` does not limit an attempt: the cache reads it past
`block_size`, and the loss pass reads it whole. The attempt also
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
`printf '\0'`.

### Scores

When the attempt has ended, every process in its container is stopped
(the container is paused). The task's `check` then runs in a fresh
container of the same image and sizes, holding only a copy of the
attempt's `/work`: what the model started, or left anywhere else, cannot
reach the check. The check is copied in, in a folder of its own, and runs
as root from `/work` for at most `CHECK_SECONDS`.

Next to the check, in its folder, is `conversation.json`: the attempt's
conversation in the chat format, as the attempts file keeps it (Runs,
below) but with only the attempt's own `metadata`: `tools`, the map from
each name the model saw to the real tool; `looking`, the real names of
the offered tools that look at sources ("The check library", below);
`answered_by`, the task's; and `answer`, the answer the loop built
(the final reply's text and the confidence) or `null`. It is written as ASCII JSON, any other character
escaped. A check finds it as `"$(dirname "$0")/conversation.json"`, and
scores everything: the files, the trace of tool calls and the confidence
(maintainer's decision, 2026-10-18). The texts the task's pointers put
`for` the check are there too, by their `path`.

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
choices, each attempt's apart, as the generation cache reads its tokens,
and the loss pass routes each token through the same experts
instead of choosing again (routing replay, "R3"), so the loss scores the
tokens through the experts that wrote them. The routers' weights do not
change during the step, and their balancing biases do not either. The
hash-routed layers choose from the token alone and need neither. The CSA
indexers go on learning by their own loss in each loss pass, as in `sft`.

### Runs

The order of the tasks in each round and every sampled token come from
generators seeded with `SEED`, so a run gives the same attempts again on
the same machine and setup (the same model, tasks, images and devices);
the checks themselves may differ, as any program may. The model alone
writes the attempts, without a drafter (maintainer's decision,
2026-10-17): the weights change at every update, the drafter is trained
only at the end of the run ("The drafter", above), and the attempts then
depend on the seed alone, never on this machine's times. A task's
attempts are written together in one batch, from one reading of the
task's opening (model.md, "Generation"). The step prints one line per
round (`ROUNDS`, passes over every task): the mean score, the number of
attempts and of updates, and the indexers' mean loss over the round's
loss passes.

Every attempt is kept in the run's record folder
([observability.md](observability.md)), as one line of
`grpo-attempts.jsonl` (maintainer's decisions, 2026-10-17: "GRPO attempts
later: chat JSON Lines in the post-training run's folder, linked from
traces", and "Add metadata to chat"), written through `context.telemetry`
as soon as the attempt is scored, so a line on a local store is on the
disk before the next attempt is scored. A group's attempts are written
together, then scored one by one, in order, each attempt's container
removed and its line kept before the next attempt is scored
(maintainer's decision, 2026-10-18: each attempt's line is kept as soon
as that attempt is scored, even though attempts are written together).
Each line is a conversation in
the chat format ([chat-format.md](chat-format.md)) whose `metadata` says
which attempt it is, so the file passes the same check as any `chat` file:

```json
{"schema": 3, "tools": [...], "messages": [...],
 "metadata": {"tools": {"shell": "bash", "...": "..."}, "answer": {"answer": "...", "confidence": 0.5},
              "step": "grpo", "task": "<task id>", "round": 1, "group": 3, "attempt": 2,
              "dropped": false, "error": null, "ended": "answered", "score": 0.5, "reward": 0.5,
              "written": 412}}
```

- `tools`, `messages`: the attempt: the tools it was offered, as the
  model saw them, the instructions, every assistant turn as the model's
  tokens read, every tool result as the model received it, and after the
  final reply the confidence question and the model's reply to it. A dropped
  attempt has only the tools and the instructions, the conversation it
  would have started from.
- `metadata.tools`: each name the model saw, mapped to the real tool's;
  `metadata.looking`: the real names of the offered tools that look at
  sources; `metadata.answered_by`: the task's; `metadata.answer`: the
  answer the loop built, the final reply's text and the confidence, `null`
  without a final reply or a confidence from 0 to 100, and absent for a
  dropped attempt.
- `step`, `task`: the step and the task's id.
- `round`: the round, from 1; `null` for an attempt made before the rounds
  (the `token-efficiency` step's reference, below).
- `group`: the group within the round, from 1, in the order played (for a
  reference attempt, the task's place in the list of tasks); `attempt`:
  the attempt within its group, from 1.
- `dropped`: `true` when Docker failed; then `error` says how, and
  `ended`, `score`, `reward` and `written` are `null`.
- `ended`: why the attempt ended (`answered`, `ended`, `turns`, `tokens`,
  `context` or `full`; Attempts, above).
- `score`: the check's score; `reward`: what the attempt is worth to the
  step (for `grpo`, the score), `null` for an attempt never trained on.
- `written`: the tokens the model wrote, the tokens the loss learns.

The trace names each line: the attempts of a group are written, scored
and kept in a span `grpo.play` (`task`, `round`, `group`, `attempts`);
inside it, each attempt's line is kept in a span `grpo.attempt` (`task`, `round`, `group`, `attempt`, then `file` and
`line`, its line number from 1, with `dropped`, `ended`, `score`,
`reward` and `written`); each group is an event `grpo.group` (`task`,
`round`, `group`, `file`, `lines`, and `outcome`: `learned`, or why it was
skipped); each round is an event `grpo.round` (`file`, `lines`: its first
and last, `attempts`, `dropped`, `updates`, and the value `mean_score`). A
recipe that runs the step twice adds to the same file, and the line
numbers go on. A line the store refuses stops the command at once with
exit code 3, as any part of the record does.
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
share is `apps/wpt-posttrain/src/wpt_posttrain/agentic.py`, which plays
attempts through `wpt_agent.play`, as the agent measures do. Until a source
holds tasks tagged `grpo`, it fails with "no tasks" and writes nothing.

First, the reference. Before any update, the incoming model makes
`REFERENCE` attempts at each task (by default `GROUP`), written together,
run and scored as `grpo`'s are. A task's best start score is the best score among them. An
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
reference and then the rounds. Every attempt is kept as `grpo` keeps its
own (Runs, above), in `token-efficiency-attempts.jsonl`, under the spans
and events `token-efficiency.attempt`, `.group` and `.round`: the
reference attempts first, with `round` and `reward` `null` in their
`metadata`, then the
rounds' attempts with their reward (the round's value is `mean_reward`).
Its settings are the constants at the top
of `steps/token-efficiency.py`, `TAG` (the tag of the tasks it reads) and
`REFERENCE` among them; to change them, copy the file and put its path in
the recipe's `posttrain` list.
