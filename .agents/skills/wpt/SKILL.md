---
name: wpt
description: How to work in the WPT repository (Workers' Pre-trained Transformer) with its own tools. Use it before adding or changing a source, a recipe, the model, training, an evaluation or a post-training step, before running any wpt-* command, and before every push. It names the command for each job, what each one reads and writes, how to extend evaluation and post-training, and the rules that keep `scripts/check.py` green.
---

# WPT tooling

WPT trains a language model only on content whose license is clear. The
repository holds pointers to the data, not the data: every source is a
`data/sources/<id>/source.json` with an exact address, revision and
checksum, and everything fetched or built lives in a store named by the
`WPT_STORE` setting (environment, or `.env` at the root). The documents in
`docs/` hold every decision; read the one that covers your change before
changing it.

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
| manifest and checksum of a source | `uv run wpt-dataset manifest ID` | the upstream | the store; prints the checksum |
| fill in new sources' checksums | `uv run wpt-dataset fill-checksum ID [ID ...]` (checksum all zeros; one download each: files that are not UTF-8 text excluded by exact path, then checked as `check --fetch` does; failures listed at the end) | the upstream | `data/sources/ID/source.json`, the store |
| edit a source's files | `uv run wpt-dataset unpack ID FOLDER`, edit, `uv run wpt-dataset patch ID FOLDER NAME` | the source, the folder | the folder; then `data/sources/ID/NNNN-NAME.patch` |
| check the data rules | `uv run wpt-dataset check [--fetch [ID ...]] [--changed REV]` (every source is fetched, failures listed at the end) | `data/` | the store, with `--fetch` |
| build a vocabulary | `uv run wpt-dataset build-tokenizer NAME [--stream]` | a tokenizer recipe | the store; `data/recipes/tokenizers/NAME.lock.json` (commit it) |
| check a vocabulary on the data | `uv run wpt-dataset tokenizer-check NAME` | the lock, admitted sources | prints characters per token per source |
| build a dataset | `uv run wpt-dataset build NAME [--stream] [--skip-failed]` (`--stream`: one source at a time, only tokens kept; `--skip-failed`: a failing source is left out and listed in the record instead of stopping) | the data part of the model recipe NAME | the store; its run record `records/<folder>/`; prints the dataset id |
| migrate data files | `uv run wpt-dataset migrate` | `data/` | rewrites files whose shape is behind |
| check the Markdown of sources | `uv run wpt-dataset markdown-check [ID ...]` | the sources | prints notes and problems per source |
| train | `uv run wpt-train RECIPE DATASET_ID [--out FOLDER] [--stream] [--recompute] [--segment N] [--run NAME] [--scheduler-costs RECORD_FOLDER] [options]`, started detached (skill `wpt-training-run`; ends by training the model's drafter; `--scheduler-costs`: replay the drafter scheduler's pass times an earlier run recorded on the same setup; a checkpoint is always written just before the drafter, so a drafter failure resumes with `--resume`; `--stream`: a few shards on disk at a time; `--recompute`: much less memory, about a third slower; `--segment N`: windows longer than N tokens run as segments of N, the same training in the memory of N tokens plus what the segments hand on, up to twice the time; with `--strategy split`, each process's slice runs in segments of N; `--run`: the checkpoint in the store, so another machine can continue) | the model recipe, the store | a model folder; its run record `records/<folder>/` in the store (one per command, over restarts); `scratch/` token files in the store's local folder; with `--run`, the store's `checkpoints/NAME` until the model is saved |
| train on several GPUs | `uv run wpt-train RECIPE DATASET_ID --strategy ddp --processes N [options]`, or `--strategy split` to split each window between the GPUs (windows too long for one; DeepSeek-V4's contextual parallelism) (several machines: `torchrun ... -m wpt_train`, Linux only) | the store | the same |
| evaluate | `uv run wpt-eval FOLDER [EVAL ...] [--attempts N]` (from inside the repository; the agent measures play N attempts, default 4, at every measure task, so each task's image must be built) | the folder, the store, the measure tasks | `FOLDER/evals/<name>.json`, `README.md`; its run record `records/<folder>/` in the store, with `attempts.jsonl` |
| post-train | `uv run wpt-posttrain RECIPE BASE NEW [--scheduler-costs RECORD_FOLDER]` (the steps are the recipe's `posttrain` list; ends by training a fresh drafter and measuring, or replaying, the scheduler's pass times) | the model recipe, the base folder, the store | the new folder, only if every step succeeded; its run record `records/<folder>/` in the store, always |
| build program-built tasks | `uv run scripts/build_tasks.py KIND SOURCE --count N [--seed S] [--select EXPRESSION]` (kinds `find-the-passage`, `evidence-removed`, `look-or-ask`, `commit-and-push`, `read-the-manual`; SOURCE an empty source a person made; docs/post-training.md, "The task builder") | admitted sources in the store | everything under `KIND/` in SOURCE, as its next patch |
| lock an image of agentic tasks | `uv run wpt-agent lock-image NAME [--force]` (needs Docker; `--force` replaces an existing lock) | `data/images/NAME/Dockerfile`, the base image's registry, the dated package archive | `data/images/NAME/lock.json` and `licenses.json` (commit them) |
| build an image of agentic tasks | `uv run wpt-agent build-image NAME` (needs Docker; before `wpt-posttrain` with `grpo` or `token-efficiency`, and before `wpt-eval`'s agent measures) | `data/images/NAME/`, the base image's registry, the dated package archive | a local Docker image |
| continue text | `uv run wpt-chat FOLDER` (`--about` prints its NOTICE; `--chat` talks to a post-trained model; `--scheduler-costs RECORD_FOLDER` reads the scheduler's pass times a run recorded on the same setup instead of measuring them) | the folder; with `--scheduler-costs`, the store | |
| the kinds of data files, their names, one file | `uv run wpt-dataset kinds`, `list KIND`, `show KIND NAME` | `data/`, through the service layer | prints them (version and problems on stderr) |
| write a data file people write | `uv run wpt-dataset save KIND NAME FILE --version VERSION` (`''`: a new file) | the file | that file, after the rules of `check`; refused when it changed since VERSION |
| a source's files, tags and git status | `uv run wpt-dataset files ID`, `tags ID PATH [TAG ...] --version V`, `remove ID PATH --patch NAME --version V`, `status` | the source | `source.json`, or the next patch |
| a new empty source, without its texts | `uv run wpt-dataset new-source ORIGIN [ITEM] NAME --license L --attribution TEXT --by NAME [--tag TAG ...] [--chat PATH ...] [--format FORMAT=PATTERN ...]` (never `--file` or `--license-text`: the person writes the license text, then the files) | | `data/sources/ID/source.json`, `under-review`, its license text empty |
| a person's texts (people only, never an AI assistant) | `read`, `write`, `conversation`, `save-conversation`, `task`, `save-task`, `save-rating`, and `new-source --file` and `--license-text` (marked `*` in `--help`) | | |
| conversation, task and rating structure | `uv run wpt-dataset conversations ID PATH` (roles, drafts), `tasks ID` (image, limits, files), `ratings ID PATH` (the attempt each rates, who, when; never the verdict) | the source | prints them |
| edit every kind of data in a browser | `uv run wpt-web` (needs Bun), then http://127.0.0.1:8410/ (HTTP API described at `/api/openapi.json`; page "Ratings": people's pass or fail on the attempts of a `wpt-eval` run, a measure only, docs/evaluation.md) | `data/`, a WPT model folder for drafts, the store's `wpt-eval` run records for ratings, `packages/wpt-web/ui` | data files through the service layer; `ui/node_modules`, `ui/dist` |

Everything that changes a model comes from its model recipe
(`data/recipes/models/NAME.json`: data, architecture, method and settings,
stages, seed, the drafter, post-training steps; docs/model.md). `training.stages` lists
the window lengths in order, each with its steps and its `batch_size`,
which is global (over every process) (docs/training.md, "Stages"); the
model's context is the last one's. Segments and split windows give
the whole window's loss and gradients, so they are options, never part of
the recipe (docs/training.md, "Long windows"). The `wpt-train` options only
say how the hardware runs it, and each also comes from `WPT_<OPTION>` in the
environment (`--micro-batch` -> `WPT_MICRO_BATCH`); `--help` on any command
lists them. A crash restarts the processes on their own, up to
`--max-restarts` times (default 3), from the last checkpoint; Ctrl+C or
`kill` finishes the step and saves first; after a stop, the same command
with `--resume` continues from `FOLDER/checkpoint/` (or, with `--run NAME`,
from the store, on any machine). Exit codes: 2 refused, 3 the run's record
could not be written, 130 stopped.

Every run of `wpt-dataset build`, `wpt-train`, `wpt-posttrain` and
`wpt-eval` writes its record into its own store folder, `records/<UTC
time>-<command>-<main id>-<random>/` (main id: the recipe name, a short
dataset id or a short weights hash): `start.json`, `events.jsonl` (spans, events,
every printed line), `ids.json` (the ids read and written, which chain the
runs together), `end.json`. With `OTEL_EXPORTER_OTLP_ENDPOINT` set
(environment or `.env`), the same goes live to an OpenTelemetry backend
(`deploy/otel-lgtm/compose.yaml`). A backend failure is one warning; a
record the store refuses stops the command with exit code 3 (`wpt-train`
first finishes its step and saves its checkpoint). A post-training step or
an evaluation records its own events and files through `context.telemetry`.
Details: `docs/observability.md`.

## Adding a source

Every rule for sources lives in the documentation: `docs/licenses.md`
(the allowed licenses, what to check for each, every rule and maintainer's
decision), `CONTRIBUTING.md` ("Adding a source"), `docs/data-selection.md`
and `docs/chat-format.md`. Read them before writing or reviewing a source,
and follow the `wpt-sources` skill for the order of work. Never apply a
rule they do not hold: a case they do not settle is a question for the
maintainer, and the answer goes into the documentation with its date. A
new source starts `under-review`; only a maintainer admits a source, in a
signed commit (CONTRIBUTING.md, "Decisions are signed").

## The service layer

`wpt_dataset/repo.py` is the only code that reads or writes `data/`
(docs/architecture.md, "The service layer"): sources and patches, chat and
task files, recipes, Dockerfiles, the signer lists, and the generated
locks and license lists, which only the command that makes them writes.
Code elsewhere asks it for documents, bytes and versions, never for a path
under `data/`; `packages/wpt-dataset/tests/test_data_access.py` fails
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
Post-training conversations are `chat` files of such a source, tagged with
the name of the step that reads them (`sft`), one conversation per line as
docs/chat-format.md says. Tasks for agentic post-training are `task`
files of such a source, one folder per task, tagged `grpo`; the
`token-efficiency` step, after `grpo`, reads the same tasks
(docs/post-training.md). People's pass or fail on a model's measure
attempts are `rating` files of such a source, one rating per line, a
measure only that no training step reads (docs/evaluation.md, "People's
ratings"). Committed text
stays in the repository's history for good.

As an AI assistant, use the `wpt-dataset` commands only for structure and
metadata, never to read or write people's texts (CONTRIBUTING.md, "Models
and generated text"): never run `read`, `write`, `conversation`,
`save-conversation`, `task`, `save-task` or `save-rating`, and never call the matching
`wpt-web` routes. You may run `new-source` without `--file` and
`--license-text`, never with them (maintainer's decision, 2026-10-18). People write
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
stops with `TokenizerVersionNeeded` means the recipe or the tokenizer code
changed: ask whether to raise the version or undo the change. Commit the
lock file a build writes.

## Changing a file's shape

Every data file carries its shape version (`schema`). To change a shape:
bump `CURRENT_SCHEMA` in `wpt_dataset/repo.py`, update the schema file,
register the step in `wpt_dataset/migrate.py`, and run `wpt-dataset
migrate`. A file `migrate` cannot rewrite, such as a source's `chat`
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
`wpt-train` and `wpt-posttrain` train at the end of every run against the
final weights. A model and its drafter are one unit: `load_model` and
`load_drafter` refuse a folder whose drafter is missing or names other
weights' sha256 (`DrafterMismatch`), and nothing skips the check; a
folder written by hand gets one through `save_drafter`, whose provenance
names `{"target": {"weights": weight_hashes(folder)}}`. `NOTICE` and the
card describe it (docs/model.md, "The drafter").

Weights are AGPL-3.0-or-later; generated text is CC0-1.0 under
`OUTPUT-NOTICE`. `packages/wpt-model/src/wpt_model/LICENSE` and
`OUTPUT-NOTICE` are copies of the root files, and a test fails when they
differ: change both together.

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

## Writing a training strategy or method

A strategy (how the hardware is used) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/strategies/` or any path given to
`wpt-train --strategy`:

```python
from wpt_training import Strategy

def start():  # a subclass overrides wrap, no_sync, synchronize, mean, reduce, broadcast, barrier, close;
                # splits_windows = True: every process trains its slice of the same windows
    return Strategy(device, rank=0, world_size=1, local_rank=0)
```

A method (how the model learns) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/methods/` (`muon`, `adamw`) or any
path, named by a model recipe's `training.method`:

```python
from wpt_training import Method, Verdict

SETTINGS = {"lr": 3e-4}  # changed by the recipe's training.settings, typed from these defaults

class Mine(Method):  # may override loss (then part_loss too, for windows in parts), validation_loss,
                     # stage, before_update, after_update, evaluated, state_dict, load_state_dict
    def __init__(self, model, settings, run):  # run: tokens, parameters, steps, log
        self.optimizer = ...

def start(model, settings, run):
    return Mine(model, settings, run)
```

The runner in `wpt_training/runner.py` owns the data, the stages, each
stage's global batch, windows in parts (`parts.py`), evaluation and checkpoints: a strategy or a method never draws windows or
seeds randomness itself, so runs stay the same on any hardware. Anything a
resumed run needs goes in the method's `state_dict`. Details:
`docs/training.md`.

## Writing a post-training step

One Python file, a built-in in `apps/wpt-posttrain/src/wpt_posttrain/steps/`
or any path given to `wpt-posttrain`:

```python
from wpt_posttrain import Outcome

def run(context):  # context: model (change it in place), tokenizer, device,
                   # base, provenance, store, telemetry (the run's record:
                   # events, spans, files in its store folder);
                   # context.conversations() reads the chat files tagged
                   # with this step's name, and context.tasks() its task
                   # folders (run them with wpt_agent; context.tasks("grpo")
                   # reads another step's tag)
    ...
    return Outcome.success()  # or Outcome.failure("why it could not do its part")
```

The built-in steps are `sft`, `grpo` and `token-efficiency` (after `grpo`,
on the same tasks: the same result with fewer written tokens); `grpo` and
`token-efficiency` share `wpt_posttrain/agentic.py`, which plays their
attempts through `wpt_agent.play` with the model alone (no drafter), as
the agent measures of `wpt-eval` do, and keeps every attempt as a
line of `<step>-attempts.jsonl` in the run's record folder: a chat-format
conversation whose `metadata` identifies the attempt. After the last step,
every run trains a fresh drafter for the new weights; the drafter is not a
step (docs/post-training.md, "The drafter").

A step reports only whether it managed to do its part, and why not. The
first failure stops the run and nothing is written. Details:
`docs/post-training.md`.

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
`wpt_agent.play(...)` (what the post-training steps and the agent measures
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
  credentials, or data that is not a pointer to a licensed source.
