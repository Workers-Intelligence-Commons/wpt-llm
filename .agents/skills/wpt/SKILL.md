---
name: wpt
description: How to work in the WPT repository (Workers' Pre-trained Transformer) with its own tools. Use it before adding or changing a source, a recipe, the model, training, an evaluation or a training step, before running any wpt-* command, and before every push. It names the command for each job, what each one reads and writes, how to extend evaluation and training, and the rules that keep `scripts/check.py` green.
---

# WPT tooling

WPT trains a language model only on content whose license is clear. The
repository holds pointers to the data, not the data: every source is a
`data/sources/<id>/source.json` with an exact address, revision and
checksum, and everything fetched or built lives in a store named by the
`WPT_STORE` setting (environment, or `.env` at the root). The documents in
`docs/` hold every decision; read the one that covers your change before
changing it. Every text in the repository, this skill included, describes
only the current state: no decision dates, no history, and a change may
break compatibility (CONTRIBUTING.md, "Writing about the project").

## Before every push

```sh
uv run scripts/check.py
```

It runs what CI runs, the same way (CI calls this script): `uv.lock`
matches the project files, the Python tests, the Rust tests, the data
rules on every source (read, not downloaded), one small source fetched
and checked (`standardebooks-joseph-conrad-ford-madox-ford-the-nature-of-a-crime-5366b0f3`,
a short book fixed at one git commit: it proves the fetch works; the sources
a pull request adds or changes are fetched by the maintainer with
`check --fetch`) and the commits (sign-off
on every commit, an allowed signature on every commit that changes a
`decision`), compared with what a push would send. A green run means a
green CI. It needs a git clone with an upstream branch; elsewhere pass
`--base` and `--range`.

Commits are signed off (`git commit -s`): CI refuses any commit without a
`Signed-off-by` line.

## Setting up a machine

```sh
uv run scripts/setup.py          # installs the PyTorch build for this machine's GPU
```

A plain `uv sync` removes PyTorch, because no build is part of the default
install; run `scripts/setup.py` again after it. `uv run` does not remove it.
A Rust toolchain is needed: the tokenizer is built from source.

On Windows, work in Windows itself (PowerShell), never inside WSL: clone,
set up, build, train and run every wpt-* command there. Linux containers,
where a step needs them, run through Docker Desktop.

## The commands

| Job | Command | Reads | Writes |
| --- | --- | --- | --- |
| id of a new source | `uv run wpt-dataset source-id ORIGIN [ITEM] ADDRESS` | | prints the id |
| download sources into the store | `uv run wpt-dataset download [ID ...] [--stream]` (no ID: every admitted source; several at once, at most as many from one site, and at most as many requests to a site in a span of time, as `packages/wpt-dataset/src/wpt_dataset/sites.txt` says; a source already in the store is not downloaded again; `--stream`: no copy kept on this machine) | the upstream | the store: its bundles, its database, its run record `records/<folder>/` |
| what the store holds and when it was downloaded | `uv run wpt-dataset store` | the store database | prints every session and each source's last outcome and last download |
| manifest and checksum of a source | `uv run wpt-dataset manifest ID` (a download, kept under the checksum found) | the upstream | the store; prints the checksum |
| fill in new sources' checksums | `uv run wpt-dataset fill-checksum ID [ID ...]` (checksum all zeros; one download each: files that are not text (UTF-8, or the encoding they state) excluded by exact path, then checked as `check --fetch` does; what it downloads is kept in the store; failures listed at the end) | the upstream | `data/sources/ID/source.json`, the store |
| edit a source's files | `uv run wpt-dataset unpack ID FOLDER`, edit, `uv run wpt-dataset patch ID FOLDER NAME` | the source, the folder | the folder; then `data/sources/ID/NNNN-NAME.patch` |
| check the data rules | `uv run wpt-dataset check [--fetch [ID ...]] [--changed REV]` (`--fetch`: the sources are downloaded as `download` does, then checked; failures listed at the end) | `data/` | the store, with `--fetch` |
| build a vocabulary | `uv run wpt-dataset build-tokenizer NAME [--stream]` | a tokenizer recipe | the store; its run record `records/<folder>/`; `data/recipes/tokenizers/NAME.lock.json` (commit it) |
| check a vocabulary on the data | `uv run wpt-dataset tokenizer-check NAME` | the lock, admitted sources | prints characters per token per source |
| build a dataset | `uv run wpt-dataset build NAME [--stream] [--skip-failed]` (reads only downloaded sources: one not in the store is left out and listed in the record; sources converted in worker processes, several at once, the same dataset as one at a time; a task pointing into a source left out stops it, naming both; `--stream`: only tokens kept, no Markdown, only the files a conversion needs as files copied into a temporary folder deleted once the source is tokens; `--skip-failed`: any other failing source is left out and listed in the record instead of stopping) | the data part of the model recipe NAME | the store; its run record `records/<folder>/`; a session in the store database with a `reused` line per source it read; prints the dataset id |
| draft sources from what a site publishes | `uv run wpt-dataset source-drafters` (the built-ins and their settings), `uv run wpt-dataset source-drafter NAME\|FILE --by NAME [--set SETTING=VALUE ...] [--only CANDIDATE ...] [--again]` (a new source starts `under-review`; one that exists is updated in place, its revision, checksum, exclusions, file kinds, submodules and history start, and goes back to `under-review` when its licenses change; one with patches stops with a note; only candidates not done yet, so the same command resumes; `--again`: the done ones too; several drafts at once, as `wpt_utils.parallel` allows; a page fetched with an ETag is kept and asked for again only if it changed; docs/architecture.md, "Source drafters") | the site, the store's facts and pages, `data/` | `data/sources/ID/source.json` through the service layer; the store (facts, pages, what is done); its run record `records/<folder>/` |
| keep a checkout's sources current, without end (a machine that stays on) | `uv run wpt-dataset always-on-monitoring [--by NAME] [--pause SECONDS]` (each round: every built-in source drafter on every candidate, then every admitted source downloaded again, the admitted drafts' checksums filled in, each source checked as `check --fetch` does; the failures printed; then a pause, `--pause` or `WPT_MONITORING_PAUSE`, one week by default; without `--by`, the drafts' `by` is `machine:<host>`; nothing committed; docs/architecture.md, "Always-on monitoring") | the sites, the upstream, `data/` | `data/sources/ID/source.json` in this checkout; the store; a run record per source drafter run and per round's download |
| migrate data files | `uv run wpt-dataset migrate` | `data/` | rewrites files whose shape is behind |
| check the Markdown of sources | `uv run wpt-dataset markdown-check [ID ...]` | the sources | prints notes and problems per source |
| train | `uv run wpt-train RECIPE DATASET_ID [--out FOLDER] [--stream] [--recompute] [--segment N] [--run NAME] [--scheduler-costs RECORD_FOLDER] [options]`, started detached (skill `wpt-training-run`; runs the recipe's `steps` in order, starting from the result of the last of its first steps the store already holds, so a run reuses the steps it shares with an earlier one; a dataset that holds nothing for a step refuses the run before any step trains; a step that cannot do its part (`StepFailed`) stops the run with exit code 2, no result of it kept but the checkpoint it wrote, which `--resume` continues, the results of the steps before it kept; ends by training the model's drafter; `--scheduler-costs`: replay the drafter scheduler's pass times an earlier run recorded on the same setup; a checkpoint is always written just before the drafter, so a drafter failure resumes with `--resume`; `--stream`: a few shards on disk at a time; `--recompute`: much less memory, about a third slower; `--segment N`: windows longer than N tokens run as segments of N, the same training in the memory of N tokens plus what the segments hand on, up to twice the time; with `--strategy split`, each process's slice runs in segments of N; `--run`: the checkpoint in the store, so another machine can continue) | the model recipe, the store | a model folder; each step's result, the store's `steps/<fingerprint>`; its run record `records/<folder>/` in the store (one per command, over restarts); `scratch/` token files in the store's local folder; with `--run`, the store's `checkpoints/NAME` until the model is saved |
| train on several GPUs | `uv run wpt-train RECIPE DATASET_ID --strategy ddp --processes N [options]`, or `--strategy split` to split each window between the GPUs (windows too long for one; DeepSeek-V4's contextual parallelism) (several machines: `torchrun ... -m wpt_train`, Linux only) | the store | the same |
| evaluate | `uv run wpt-eval FOLDER [EVAL ...] [--attempts N]` (from inside the repository; the agent measures play N attempts, default 4, at every measure task, so each task's image must be built) | the folder, the store, the measure tasks | `FOLDER/evals/<name>.json`, `README.md`; its run record `records/<folder>/` in the store, with `attempts.jsonl` |
| build program-built tasks | `uv run scripts/build_tasks.py KIND SOURCE --count N [--seed S] [--select EXPRESSION]` (kinds `find-the-passage`, `evidence-removed`, `look-or-ask`, `commit-and-push`, `read-the-manual`; SOURCE an empty source a person made; docs/training.md, "The task builder") | admitted sources in the store | everything under `KIND/` in SOURCE, as its next patch |
| lock an image of agentic tasks | `uv run wpt-agent lock-image NAME [--force]` (needs Docker; `--force` replaces an existing lock) | `data/images/NAME/Dockerfile`, the base image's registry, the dated package archive | `data/images/NAME/lock.json` and `licenses.json` (commit them) |
| build an image of agentic tasks | `uv run wpt-agent build-image NAME` (needs Docker; before `wpt-train` with `finish-tasks` or `finish-tasks-with-less-writing`, and before `wpt-eval`'s agent measures) | `data/images/NAME/`, the base image's registry, the dated package archive | a local Docker image |
| continue text | `uv run wpt-chat FOLDER` (`--about` prints its NOTICE; `--chat` talks to a model trained on conversations; `--scheduler-costs RECORD_FOLDER` reads the scheduler's pass times a run recorded on the same setup instead of measuring them) | the folder; with `--scheduler-costs`, the store | |
| the kinds of data files, their names, one file | `uv run wpt-dataset kinds`, `list KIND`, `show KIND NAME` | `data/`, through the service layer | prints them (version and problems on stderr) |
| write a data file people write | `uv run wpt-dataset save KIND NAME FILE --version VERSION` (`''`: a new file) | the file | that file, after the rules of `check`; refused when it changed since VERSION |
| a source's files, tags and git status | `uv run wpt-dataset files ID`, `tags ID PATH [TAG ...] --version V`, `remove ID PATH --patch NAME --version V`, `status` | the source | `source.json`, or the next patch |
| a new empty source, without its texts | `uv run wpt-dataset new-source ORIGIN [ITEM] NAME --license L --attribution TEXT --by NAME [--tag TAG ...] [--chat PATH ...] [--format FORMAT=PATTERN ...]` (never `--file` or `--license-text`: the person writes the license text, then the files) | | `data/sources/ID/source.json`, `under-review`, its license text empty |
| a person's texts (people only, never an AI assistant) | `read`, `write`, `conversation`, `save-conversation`, `task`, `save-task`, `save-rating`, and `new-source --file` and `--license-text` (marked `*` in `--help`) | | |
| conversation, task and rating structure | `uv run wpt-dataset conversations ID PATH` (roles, drafts), `tasks ID` (image, limits, files), `ratings ID PATH` (the attempt each rates, who, when; never the verdict) | the source | prints them |
| edit every kind of data in a browser | `uv run wpt-web` (needs Bun), then http://127.0.0.1:8410/ (HTTP API described at `/api/openapi.json`; page "Ratings": people's pass or fail on the attempts of a `wpt-eval` run, a measure only, docs/evaluation.md; page "Source drafters": start a run and follow it, stopped with `wpt-web` and started again with it) | `data/`, a WPT model folder for drafts, the store's `wpt-eval` run records for ratings, `packages/wpt-web/ui` | data files through the service layer; `ui/node_modules`, `ui/dist`; the name it holds, in the git folder |

Everything that changes a model comes from its model recipe
(`data/recipes/models/NAME.json`: data, architecture, seed, the training
`steps` in order, the drafter; docs/model.md). Each step is
`{"step": NAME}` with, when they differ from the step file's, its
`optimizer` and that optimizer's settings, its `settings` and the parts it
keeps `locked`, named as the model names them (docs/model.md, "Parts";
docs/training.md, "Steps"). The built-in steps are `read-text`,
`answer-as-assistant`, `finish-tasks` and `finish-tasks-with-less-writing`.
The `read-text` step's `stages` list the window lengths in order, each
with its updates (`steps`) and its `batch_size`, which is global (over
every process) (docs/training.md, "Stages"); the model's context is the
last window length of the last step that reads text windows. Segments and split windows give
the whole window's loss and gradients, so they are options, never part of
the recipe (docs/training.md, "Long windows"). The `wpt-train` options only
say how the hardware runs it, and each also comes from `WPT_<OPTION>` in the
environment (`--micro-batch` -> `WPT_MICRO_BATCH`); `--help` on any command
lists them. A crash restarts the processes on their own, up to
`--max-restarts` times (default 3), from the last checkpoint; Ctrl+C or
`kill` finishes the current update and saves first; after a stop, the same command
with `--resume` continues from `FOLDER/checkpoint/` (or, with `--run NAME`,
from the store, on any machine). Exit codes: 2 refused, 3 the run's record
could not be written, 130 stopped.

Every run of `wpt-dataset download`, `check --fetch`, `fill-checksum`,
`manifest`, `build`, `build-tokenizer`, `wpt-dataset source-drafter`, each round of `wpt-dataset always-on-monitoring`, `wpt-train` and `wpt-eval` writes its record into its own store folder, `records/<UTC
time>-<command>-<main id>-<random>/` (main id: the source id or `<count>-sources`, the recipe name, a short
dataset id or a short weights hash): `start.json`, `events.jsonl` (spans, events,
every printed line), `ids.json` (the ids read and written, which chain the
runs together), `end.json`. With `OTEL_EXPORTER_OTLP_ENDPOINT` set
(environment or `.env`), the same goes live to an OpenTelemetry backend
(`deploy/otel-lgtm/compose.yaml`). A backend failure is one warning; a
record the store refuses stops the command with exit code 3 (`wpt-train`
first finishes its update and saves its checkpoint). A training step's
environment records its own events and files through `data.telemetry`, and
an evaluation through `context.telemetry`.
Details: `docs/observability.md`.

## Adding a source

Every rule for sources lives in the documentation: `docs/licenses.md`
(the allowed licenses, what to check for each, every rule and maintainer's
decision), `CONTRIBUTING.md` ("Adding a source"), `docs/data-selection.md`
and `docs/chat-format.md`. Read them before writing or reviewing a source,
and follow the `wpt-sources` skill for the order of work. Never apply a
rule they do not hold: a case they do not settle is a question for the
maintainer, and the answer goes into the documentation as a general rule
about the current state, with no date and no source named
(CONTRIBUTING.md, "Writing about the project"). A
new source starts `under-review`; only a maintainer admits a source, in a
signed commit (CONTRIBUTING.md, "Decisions are signed").

## The service layer

`wpt_dataset/repo.py` is the only code that reads or writes `data/`
(docs/architecture.md, "The service layer"): sources and patches, chat and
task files, recipes, Dockerfiles, the signer lists, and the generated
locks and license lists, which only the command that makes them writes.
The reviewed license texts (`reviewed-license-texts.json`, which the source
drafters read) are one of its kinds. Code elsewhere asks it for documents,
bytes and versions, never for a path under `data/`; `packages/wpt-dataset/tests/test_data_access.py` fails
otherwise, and an audit hook the layer sets refuses such an access at run
time (`DataAccessRefused`), except from tests. Every write applies the
rules of `wpt-dataset check`, refuses a stale save (one lock, across
processes, holds the version check and the write), writes a symbolic
link's target only inside the repository, writes U+2028, U+2029 and
U+0085 as escapes, and keeps a JSON file's layout where its meaning did
not change. A new kind of data gets both interfaces, a
`wpt-dataset` command and a `wpt-web` route with its JSON Schema, both thin
layers over the service layer.

## Patches and text written for WPT

Never write a patch by hand. `wpt-dataset unpack` writes a source's final
files into a folder outside the repository; after the edit,
`wpt-dataset patch` records the difference as the next patch and keeps it
only if the source then gives exactly the folder's files and passes its
license map.

Text a person writes for WPT is an `empty` source: no download, its
patches hold every file (CONTRIBUTING.md, "Adding your own texts").
Conversations for training are `chat` files of such a source, tagged with
the name of the step that reads them (`answer-as-assistant`), one
conversation per line as docs/chat-format.md says. Tasks for agentic
training are `task` files of such a source, one folder per task, tagged
`finish-tasks`; the `finish-tasks-with-less-writing` step, after
`finish-tasks`, reads the same tasks (docs/training.md). A dataset's
build keeps the tagged conversations and tasks of its admitted sources,
so a step reads them only from a dataset built after they were added. People's pass or fail on a model's measure
attempts are `rating` files of such a source, one rating per line, a
measure only that no training step reads (docs/evaluation.md, "People's
ratings"). Committed text
stays in the repository's history for good.

As an AI assistant, use the `wpt-dataset` commands only for structure and
metadata, never to read or write people's texts (CONTRIBUTING.md, "Models
and generated text"): never run `read`, `write`, `conversation`,
`save-conversation`, `task`, `save-task` or `save-rating`, and never call the matching
`wpt-web` routes. You may run `new-source` without `--file` and
`--license-text`, never with them. People write
their texts in `wpt-web` or with those commands themselves. An `assistant`
message may start as a WPT model's draft that a person edited or
confirmed; its `metadata.draft` records it (docs/chat-format.md, "Model
drafts").

## Formatting a person's text

A person may hand you raw text to turn into source files, asking you to
change only its form. The text must stay theirs, and no output of another
company's model may enter the data, so:

- Never open, read or print the content of those files.
- Learn their shape only through scripts that report structure, never
  words: line counts and lengths, blank lines, character classes,
  encodings, line endings.
- Change them only through scripts that transform the whole file
  mechanically (line endings, spacing, splitting into files), never by
  writing their text yourself.
- Pass the result to the tools with shell commands: `unpack`, copy the
  formatted files into the folder, `patch`, then `scripts/check.py`;
  never through the commands marked `*` in `wpt-dataset --help`.
- From a session with an AI assistant, keep only the person's own
  messages.

## Vocabularies

A tokenizer recipe's lock freezes its vocabulary: adding sources never
retrains it (docs/tokenizer.md). Raising the recipe's `version` trains a new
one and replaces the lock; it is a person's decision (it makes later models
incompatible with earlier ones), so never raise it unasked. A build that
stops with `TokenizerVersionNeeded` means the recipe changed: ask whether to
raise the version or undo the change. A change of the tokenizer's code asks
for no version. Commit the
lock file a build writes.

## Changing a file's shape

No data file carries a shape number: a step recognizes an old shape by its
fields. To change a shape: update the schema file, add a `Step` (what it
finds, how it rewrites) to the list of its kind in `wpt_dataset/migrate.py`,
with an old shape that leaves a field difference to find, and run
`wpt-dataset migrate`. A file `migrate` cannot rewrite, such as a source's `chat`
file (upstream's bytes), goes through `migrate.upgrade` where it is read.
Never edit data files by hand for a shape change. A field only a
person can fill is left empty by the migration, and the check then names
each file to fill in.

## The model folder

`wpt-train` writes `config.json`, the weights (each safetensors header
carries `license`, `output_license`, `notice`, `tokenizer` and
`tokenizer_sha256`), `tokenizer.json` (loaded only through
`wpt_model.load_tokenizer`, which checks it against those headers),
`provenance.json` (what produced it, including every source it learned
from), and `LICENSE`, `NOTICE` and `README.md` (the model card).
`NOTICE` and `README.md` come from `wpt_model.notice.write_documents`, which
reads only the folder's own files: change what they say there, never by
editing a generated folder. `wpt-train` writes the folder beside its
target and renames it into place; training into an existing folder replaces
it as a whole (its `evals/`, checkpoint and any other file go with it).
Every model folder holds `drafter/` (`config.json`, weights with the same
header, `provenance.json`): the drafter for speculative decoding, which
`wpt-train` trains at the end of every run against the final weights. A model and its drafter are one unit: `load_model` and
`load_drafter` refuse a folder whose drafter is missing or names other
weights' sha256 (`DrafterMismatch`), and nothing skips the check; a
folder written by hand gets one through `save_drafter`, whose provenance
names `{"target": {"weights": weight_hashes(folder)}}`. `NOTICE` and the
card describe it (docs/model.md, "The drafter").

Weights are AGPL-3.0-or-later; generated text is CC0-1.0 under
`OUTPUT-NOTICE`. `packages/wpt-model/src/wpt_model/LICENSE` and
`OUTPUT-NOTICE` are copies of the root files, and a test fails when they
differ: change both together.

## Writing a source drafter

One Python file, a built-in in
`packages/wpt-dataset/src/wpt_dataset/source_drafters/` or any path given
to `wpt-dataset source-drafter`:

```python
from wpt_dataset.source_drafter import Candidate, Drafted, NeedsPerson, Skipped

SETTINGS = {"run": "20250320"}  # changed by --set NAME=VALUE, typed from these defaults

def candidates(context):  # Candidate(name, source id when known, facts as JSON)
    ...

def draft(candidate, context):  # context: repo (read only), store, settings, by, today, work (an empty
                                # folder for this draft), fact(key, compute) (kept in the store),
                                # fetch(address) (robots.txt and retries, as a source's fetch)
    return Drafted(origin, item, document)  # source.json without its decision; or Skipped(reason),
                                            # or NeedsPerson(reason, notes)
```

Never write under `data/` from it, and never give a decision: the
framework saves every draft through the service layer, under review, and
records what is done in the store. A step that spans candidates is the
program's own. Check a new one by running it on a few real candidates
(`--only`), then restore `data/`. Details: docs/architecture.md, "Source
drafters".

## Writing an evaluation

One Python file, a built-in in `apps/wpt-eval/src/wpt_eval/evals/` or any
path given to `wpt-eval`:

```python
from wpt_eval import Result

def run(context):  # context: model, tokenizer, device, folder, provenance, store,
                   # telemetry (the run's record), documents("training") and
                   # documents("held_out"), read one shard at a time:
                   # iterate, never assume a list; attempts(plain=False),
                   # the agent measures' shared run of attempts at the
                   # measure tasks, played once per command
    ...
    return Result(data={...}, text="the same findings, for people")
```

Every evaluation returns `Result`: `data` that JSON can hold, and `text`
that the model card shows. Details: `docs/evaluation.md`.

## Writing a training strategy or step

A strategy (how the hardware is used) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/strategies/` or any path given to
`wpt-train --strategy`:

```python
from wpt_training import Strategy

def start():  # a subclass overrides wrap, no_sync, synchronize, mean, reduce, broadcast, barrier, close;
                # splits_windows = True: every process trains its slice of the same windows
    return Strategy(device, rank=0, world_size=1, local_rank=0)
```

A step is made of separate parts, each one Python file, a built-in in its
folder of `packages/wpt-training/src/wpt_training/` or any path:

```python
# methods/NAME.py: how a loss is made from a batch of examples
READS = "windows"  # the kind of examples it learns from
SETTINGS = {...}
def start(settings): return Mine()  # a wpt_training.Method (TextMethod on text windows): learn, update_loss,
                                    # optionally begin, unreached (parts its loss never reaches), after_update, state_dict

# environments/NAME.py: where the examples come from
GIVES = "windows"  # the kind of examples it gives
SETTINGS = {...}
def start(settings, data): ...  # a wpt_training.Environment: stage_list, tokens, sources, batch (None: the update
                                # is skipped), evaluates, evaluate, after_update, finished, state_dict, close;
                                # data: the step's name, store, dataset record, tokenizer, config, strategy,
                                # telemetry, log, seed, the step's number, step_seed (both mixed:
                                # what the step's draws are seeded by); optionally check(settings),
                                # context(settings) and check_dataset(settings, record, store): ValueError
                                # when the dataset holds nothing for it, before any step runs

# optimizers/NAME.py: how the weights move, with its schedule
SETTINGS = {...}
def start(model, settings, run): ...  # a wpt_training.Optimizer over run's parameters

# steps/NAME.py: puts them together
METHOD = "next-token"
ENVIRONMENT = "text-windows"
OPTIMIZER = {"name": "muon", "settings": {...}}
LOCKED = ()        # model parts left as they are (wpt_model.PARTS)
SETTINGS = {...}   # defaults for its method's and environment's settings
```

A method that reads what its environment does not give, an unknown part or
setting, an optimizer that refuses the model, and a dataset that holds
nothing for a step are refused before any step runs. The runner in `wpt_training/runner.py` runs one step: its
stages, each update's global batch, windows in parts (`parts.py`),
evaluation and checkpoints; a part never seeds randomness itself, so runs
stay the same on any hardware. Anything a resumed step needs goes in the
`state_dict` of its method, environment or optimizer. A part that finds it
cannot do its job (a loss that is not finite, every attempt dropped)
raises `wpt_training.StepFailed(reason)`. The step keeps its result in the store
under a fingerprint of everything that decides it: the dataset part the
step reads (the record's `fingerprints`: text, conversations or tasks), not
the whole dataset, and never the hardware options (`--stream` included). After the last step, every run trains a drafter for the final
weights; the drafter is not a step. Details: `docs/training.md`, "Steps".

The `finish-tasks` steps play their attempts through `wpt_agent.play` with
the model alone (no drafter), as the agent measures of `wpt-eval` do, and
keep every attempt as a line of `<step>-attempts.jsonl` in the run's
record folder: a chat-format conversation whose `metadata` identifies the
attempt. A task's check prints named parts (`correct` and the rules it
reports), and the step weighs them into the reward with its weight
settings, gated by the task's `gates` (docs/training.md, "Rewards").

## Generating text

Anything that makes a model write (a chat, an evaluation, a step's
attempts) goes through `wpt_inference`, never through a loop of its own
over `model(ids)`:

```python
from wpt_inference import Writer, next_message, sample_tokens

for token in sample_tokens(model, ids, max_new_tokens=200):  # one text
    ...
reply = next_message(model, tokenizer, conversation, max_new_tokens=512)  # the next chat message, as wpt-chat --chat
writer = Writer(model, prompt, rows=8, stops=[...], routes=True)  # several attempts from one prompt
written = writer.write({row: tokens_it_reads_first}, {row: most_tokens_it_may_write})
```

They read through the model's generation cache (`WPT.new_cache`,
`extend`, `undo`; docs/model.md, "Generation"): each token once, the
prompt once for every row, the rows in one batch, with no length limit
(past `block_size`). A `drafter` (`wpt_model.Drafter`, from
`load_drafter`) turns on speculative decoding: the same distribution,
fewer passes. Its scheduler needs `costs=`, the pass times a run measures
once (`measure_costs`) and records with the setup they were measured on
(`costs_setup`, `costs_file`), or replays on the same setup only
(`replayed_costs`); without them every draft is verified. For agentic
tasks, `wpt_agent.attempts` runs several attempts in one batch, and
`wpt_agent.play(...)` (what the task steps and the agent measures
share) scores them one by one and hands each to the caller as soon as it
is scored. What a layer passes
on to later tokens is said once, in its `passes_on()` (`carry.py`), which
training's segments and slices (`sequence.py`) and the cache (`cache.py`)
both read: a change to a layer's computation changes its description
there, and `test_cache.py` and `test_parts.py` check that both still give
what the whole text gives (docs/model.md, "What a layer passes on").

## Rules that keep the checks green

- Write text files with `encoding="utf-8", newline="\n"`: Windows must
  produce the same bytes as Linux.
- Start external programs through `wpt_utils.programs.program(name)`,
  which reports a missing tool clearly.
- A fingerprint holds `wpt_utils.programs.code()` (the commit and a hash of
  the uncommitted changes), never a version number of WPT's own code.
- Parallel work sizes itself with `wpt_utils.parallel.Sizing`, which
  measures each item's memory as it runs: never give a memory figure per
  item. A source's conversion or reading goes through
  `wpt_dataset.workers` (or `sources.each_read`), with a module-level
  function, since it runs in a worker process.
- Test isolation: a test never writes into the real store, cache or user
  settings; set `WPT_STORE`, `XDG_CACHE_HOME` and `LOCALAPPDATA` to the
  test's own folder when it touches them.
- The application tests share their setup (a dataset built from a git
  project, a tiny trained model) through the `built` and `trained`
  fixtures in `apps/conftest.py`.
- A new third-party package in `uv.lock` or `Cargo.lock` gets a row in
  `docs/licenses.md` (version and license, read from its metadata), and a
  changed version gets its row checked again.
- The repository is public: never commit personal paths, machine names,
  credentials, or data that is not a pointer to a licensed source. The one
  machine name committed is the `by` of a draft that always-on monitoring
  saved without `--by`, `machine:<host>`; a person's name holds no colon,
  and `check` refuses an `admitted` or `refused` decision by a machine.
