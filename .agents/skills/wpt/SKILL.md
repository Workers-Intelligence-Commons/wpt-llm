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
| check the data rules | `uv run wpt-dataset check [--fetch [ID ...]] [--changed REV]` | `data/` | the store, with `--fetch` |
| build a vocabulary | `uv run wpt-dataset build-tokenizer NAME` | a tokenizer recipe | the store; `data/recipes/tokenizers/NAME.lock.json` (commit it) |
| check a vocabulary on the data | `uv run wpt-dataset tokenizer-check NAME` | the lock, admitted sources | prints characters per token per source |
| build a dataset | `uv run wpt-dataset build NAME` | a dataset recipe | the store; prints the dataset id |
| migrate data files | `uv run wpt-dataset migrate` | `data/` | rewrites files whose shape is behind |
| train | `uv run wpt-train DATASET_ID [--out FOLDER] [options]` | the store | a model folder; `scratch/` token files in the store's local folder |
| train on several GPUs | `uv run wpt-train DATASET_ID --strategy ddp --processes N [options]` (several machines: `torchrun ... -m wpt_train`, Linux only) | the store | the same |
| evaluate | `uv run wpt-eval FOLDER [EVAL ...]` | the folder, the store | `FOLDER/evals/<name>.json`, `README.md` |
| post-train | `uv run wpt-posttrain BASE NEW STEP [STEP ...]` | the base folder, the store | the new folder, only if every step succeeded |
| continue text | `uv run wpt-chat FOLDER` (`--about` prints its NOTICE) | the folder | |

Every `wpt-train` option also comes from `WPT_<OPTION>` in the environment
(`--n-layer` -> `WPT_N_LAYER`); `--help` on any command lists the rest.
`--batch-size` is global (over every process), `--set NAME=VALUE` changes a
setting of the method, and after a crash the same command with `--resume`
continues from `FOLDER/checkpoint/`.

## Adding a source

Follow CONTRIBUTING.md, "Adding a source". The parts an agent gets wrong:

- The license must be on the allowed list in
  `packages/wpt-dataset/src/wpt_dataset/licenses.py` and
  `docs/licenses.md`. When in doubt, refuse; grey-zone licenses (CC BY,
  CC BY-SA, ODbL) are refused until a lawyer confirms them.
- Every path matches exactly one `licenses` entry, and each entry needs an
  `attribution`: the credit that license asks for, copied from the source
  (authors, title, copyright line, link). Never invent it. Every model
  trained on the source repeats it in its `NOTICE`.
- Every path also matches exactly one `formats` entry: how it becomes
  Markdown (`markdown`, `text`, `code` with its `language`, `xhtml`).
  Training reads only Markdown; a new kind of file needs its format in
  `wpt_dataset/markdown.py` first.
- `wpt-dataset` downloads nothing from an address that its site's robots.txt
  forbids to every crawler, to wpt-dataset or to an AI training crawler, or
  whose robots.txt answers anything but the file, 404 or 410
  (`wpt_dataset/robots.py`); it then tries the next mirror, read with its
  own robots.txt, since a site's robots.txt speaks only for that site. A
  mirror is a real place the content is published (such as the author's
  own site); never download by hand around a refusal.
- Tags, licenses and formats are written by people; never generate them.
- A new source starts with `"decision": "refused"` and the reason "under
  review". Only a maintainer admits it, in a signed commit; an admission
  has no reason unless it is an exception (such as an explicit permission
  for content without a license), while a refusal or retirement always
  says why.
- The schema of each data file is in
  `packages/wpt-dataset/src/wpt_dataset/schemas/`.

## Patches and text written for WPT

Never write a patch by hand. `wpt-dataset unpack` writes a source's final
files into a folder outside the repository; after the edit,
`wpt-dataset patch` records the difference as the next patch and keeps it
only if the source then gives exactly the folder's files and passes its
license map.

Text a person writes for WPT is an `empty` source: no download, its
patches hold every file (CONTRIBUTING.md, "Adding your own texts").
Post-training records are files of such a source, tagged `demonstration`
(docs/post-training.md); their format is not written yet. Committed text
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
editing a generated folder. Training into an existing folder removes its
`evals/`, which describe other weights.

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
                   # documents("training") and documents("held_out"), read
                   # one shard at a time: iterate, never assume a list
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

def start():  # a subclass overrides wrap, no_sync, mean, broadcast, barrier, close
    return Strategy(device, rank=0, world_size=1, local_rank=0)
```

A method (how the model learns) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/methods/` or any path given to
`wpt-train --method`:

```python
from wpt_training import Method, Verdict

SETTINGS = {"lr": 3e-4}  # changed with --set lr=0.001, typed from these defaults

class Mine(Method):  # may override loss, before_update, evaluated, state_dict, load_state_dict
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
                   # base, provenance, store
    ...
    return Outcome.success()  # or Outcome.failure("why it could not do its part")
```

A step reports only whether it managed to do its part, and why not. The
first failure stops the run and nothing is written. Details:
`docs/post-training.md`.

## Rules that keep the checks green

- Write text files with `encoding="utf-8", newline="\n"`: Windows must
  produce the same bytes as Linux.
- Start external programs through `wpt_dataset.programs.program(name)`,
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
