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
rules, the changed sources (fetched and checked) and the commits (sign-off
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

## The commands

| Job | Command | Reads | Writes |
| --- | --- | --- | --- |
| id of a new source | `uv run wpt-dataset source-id ORIGIN [ITEM] ADDRESS` | | prints the id |
| manifest and checksum of a source | `uv run wpt-dataset manifest ID` | the upstream | the store; prints the checksum |
| edit a source's files | `uv run wpt-dataset unpack ID FOLDER`, edit, `uv run wpt-dataset patch ID FOLDER NAME` | the source, the folder | the folder; then `data/sources/ID/NNNN-NAME.patch` |
| check the data rules | `uv run wpt-dataset check [--fetch [ID ...]] [--changed REV]` (every source is fetched, failures listed at the end) | `data/` | the store, with `--fetch` |
| build a vocabulary | `uv run wpt-dataset build-tokenizer NAME [--stream]` | a tokenizer recipe | the store; `data/recipes/tokenizers/NAME.lock.json` (commit it) |
| check a vocabulary on the data | `uv run wpt-dataset tokenizer-check NAME` | the lock, admitted sources | prints characters per token per source |
| build a dataset | `uv run wpt-dataset build NAME [--stream] [--skip-failed]` (`--stream`: one source at a time, only tokens kept; `--skip-failed`: a failing source is left out and listed in the record instead of stopping) | the data part of the model recipe NAME | the store; its run record `records/<folder>/`; prints the dataset id |
| migrate data files | `uv run wpt-dataset migrate` | `data/` | rewrites files whose shape is behind |
| check the Markdown of sources | `uv run wpt-dataset markdown-check [ID ...]` | the sources | prints notes and problems per source |
| train | `uv run wpt-train RECIPE DATASET_ID [--out FOLDER] [--stream] [--recompute] [--run NAME] [options]`, started detached (skill `wpt-training-run`; `--stream`: a few shards on disk at a time; `--recompute`: much less memory, about a third slower; `--run`: the checkpoint in the store, so another machine can continue) | the model recipe, the store | a model folder; its run record `records/<folder>/` in the store (one per command, over restarts); `scratch/` token files in the store's local folder; with `--run`, the store's `checkpoints/NAME` until the model is saved |
| train on several GPUs | `uv run wpt-train RECIPE DATASET_ID --strategy ddp --processes N [options]` (several machines: `torchrun ... -m wpt_train`, Linux only) | the store | the same |
| evaluate | `uv run wpt-eval FOLDER [EVAL ...]` | the folder, the store | `FOLDER/evals/<name>.json`, `README.md`; its run record `records/<folder>/` in the store |
| post-train | `uv run wpt-posttrain RECIPE BASE NEW` (the steps are the recipe's `posttrain` list) | the model recipe, the base folder, the store | the new folder, only if every step succeeded; its run record `records/<folder>/` in the store, always |
| continue text | `uv run wpt-chat FOLDER` (`--about` prints its NOTICE; `--chat` talks to a post-trained model) | the folder | |
| review and edit sources in a browser | `uv run wpt-web` (needs Bun), then http://127.0.0.1:8410/ | `data/`, `packages/wpt-web/ui` | `source.json` through the source rules; `ui/node_modules`, `ui/dist` |

Everything that changes a model comes from its model recipe
(`data/recipes/models/NAME.json`: data, architecture, method and settings,
steps, batch, seed, post-training steps; docs/model.md); the recipe's
`batch_size` is global (over every process). The `wpt-train` options only
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
maintainer, and the answer goes into the documentation with its date. Only
a maintainer admits a source, in a signed commit.

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
docs/chat-format.md says. Committed text
stays in the repository's history for good.

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
  formatted files into the folder, `patch`, then `scripts/check.py`.
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
migrate`. Never edit data files by hand for a shape change. A field only a
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
                   # iterate, never assume a list
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

def start():  # a subclass overrides wrap, no_sync, mean, reduce, broadcast, barrier, close
    return Strategy(device, rank=0, world_size=1, local_rank=0)
```

A method (how the model learns) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/methods/` (`muon`, `adamw`) or any
path, named by a model recipe's `training.method`:

```python
from wpt_training import Method, Verdict

SETTINGS = {"lr": 3e-4}  # changed by the recipe's training.settings, typed from these defaults

class Mine(Method):  # may override loss, validation_loss, before_update, after_update, evaluated,
                     # state_dict, load_state_dict
    def __init__(self, model, settings, run):  # run: tokens, parameters, steps, log
        self.optimizer = ...

def start(model, settings, run):
    return Mine(model, settings, run)
```

The runner in `wpt_training/runner.py` owns the data, the global batch,
evaluation and checkpoints: a strategy or a method never draws windows or
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
                   # with this step's name
    ...
    return Outcome.success()  # or Outcome.failure("why it could not do its part")
```

A step reports only whether it managed to do its part, and why not. The
first failure stops the run and nothing is written. Details:
`docs/post-training.md`.

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
