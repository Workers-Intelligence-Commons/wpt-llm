---
name: wpt
description: How to work in the WPT repository (Workers' Pre-trained Transformer) with its own tools. Use it before adding or changing a source, a recipe, the model, training, an evaluation or a training step, before running any `wpt` subcommand, and before every push. It names the subcommand for each job, what each one reads and writes, how to extend evaluation and training, and the rules that keep `scripts/check.py` green.
---

# WPT tooling

WPT trains a language model only on content whose license is clear. The
repository holds pointers to the data, not the data: every source is a
`data/sources/<id>/source.json` with its address (and, for a git source,
the full id of the commit it downloads, `revision`), its license entries
(each with the place of its statement, the license texts expected there
and its own decision) and its decision; the license texts are items of
`data/license-texts/<id>/`; everything fetched or built lives in a store
named by the `WPT_STORE` setting (environment, or `.env` at the root), under
a session ID (below).
A git source is pinned to its `revision`, which whoever writes the source
writes; any other source is not pinned, and a download takes what upstream
serves when asked. The documents in
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
a short book: it proves the fetch works; the sources
a pull request adds or changes are fetched by the maintainer with
`check --fetch`) and the commits (sign-off
on every commit, an allowed signature on every commit that changes a
`decision`, of a source, a license entry or a license text), compared with what a push would send. A green run means a
green CI. It needs a git clone with an upstream branch; elsewhere pass
`--base` and `--range`.

To check work before committing it, run `uv run scripts/check.py
--working-tree`: every check but the commits, on the files as they are. The
push still needs the run above.

Commits are signed off (`git commit -s`): CI refuses any commit without a
`Signed-off-by` line.

## Setting up a machine

```sh
uv run scripts/setup.py          # installs the PyTorch build for this machine's GPU
```

A plain `uv sync` removes PyTorch, because no build is part of the default
install; run `scripts/setup.py` again after it. `uv run` does not remove it.
It detects the GPU, and says which settings a chip needs in the environment.
A Rust toolchain is needed: the tokenizer is built from source.

On Windows, work in Windows itself (PowerShell), never inside WSL: clone,
set up, build, train and run every `wpt` subcommand there. Linux containers,
where a step needs them, run through Docker Desktop.

## The commands

Every job is a subcommand of one command, `wpt` (`apps/wpt`; `uv run wpt
--help` lists them); `wpt web` serves the same services in a browser, on
127.0.0.1 only, and starts and follows every subcommand as a job, except the
`wpt dataset` commands marked `*` in `wpt dataset --help` (people's texts)
and the typing of `wpt chat` (docs/architecture.md, "The app" and "Two
interfaces").

Every command that touches the store works in a session, an ID its files
are stored under, with no open, close or status: `--session ID` or
`WPT_SESSION` (environment, or `.env`). `download`, `check --fetch` and
`manifest` make a new short ID and print it when none is given; any other
command with no ID is refused (`SessionNeeded`), except `wpt doctor`, `wpt store list`,
`runs` and `run`, which work with no ID and show every session, which is how a person
finds an ID (an ID narrows them to it), and `wpt store rebuild-index`, which needs none. `--from OLD_ID` (`WPT_FROM`)
reads from that session what this one does not hold; what the command writes
stays under its own ID and records the old one. In the table, "the session" is
that part of the store: `<id>/sources/`, `<id>/datasets/`, `<id>/models/`,
`<id>/runs/` (docs/architecture.md, "The store").

| Job | Command | Reads | Writes |
| --- | --- | --- | --- |
| id of a new source | `uv run wpt dataset source-id ORIGIN [ITEM] ADDRESS` (given once, when the source is created; an address that changes later keeps the id) | | prints the id |
| download sources into the store | `uv run wpt dataset download [ID ...] [--stream]` (no ID: every admitted source; several at once, at most as many from one site, and at most as many requests to a site in a span of time, as `packages/wpt-dataset/src/wpt_dataset/sites.txt` says; each as its address serves it when asked, a git source at exactly its `revision`; a source is downloaded only when neither the session nor the one it started from (`--from OLD_ID`) holds it with the same address, revision and fetch options (a new source, options that changed, a download that failed), so a rerun costs only what the session lacks, and a new session started from an old one fetches only what the old one lacks; a redo is a new session; a failed download keeps any file the session held for the source, and its `.json` holds the facts of that file and the error of the last attempt: a build reads that file when it is of the same download, and the dataset record notes the failed attempt; `--stream`: no copy kept on this machine) | the upstream | the session: each source's `<source>.tar.zst` and `<source>.json`; the index; its run record `<id>/runs/<run>/` |
| ask for fresh content | `uv run wpt dataset download [ID ...]` with no session: a new session fetches every source again, and the old one stays | the upstream | a new session |
| what a session holds and when it was downloaded | `uv run wpt store list [--session ID]` (no ID: every session of the store's index; an ID: that session and the one it started from) | the store's index | prints each with what it holds, and each source's time, error or address, size and what it got |
| make the store's index again | `uv run wpt store rebuild-index` (from the files of every session, a damaged index included) | every session's files | the store's index; prints how many things it holds |
| copy a kept model out of the store | `uv run wpt store copy-out NAME FOLDER --session ID` (NAME: the run's name; FOLDER must not exist; every file checked against its sha256) | the session's `<id>/models/NAME/` | FOLDER |
| download, build and train in one command | `docker compose up` with `COMPOSE_FILE=deploy/manager/compose.yaml` in `.env` (or `uv run wpt manager`), `WPT_RECIPE` required, `WPT_SESSION` the session (unset: the one of the run it goes on with, else a new ID, which it prints), `WPT_RUN` the run's name (unset: a new one, written into the git-ignored `wpt-run.txt` with its session and copy-out line), `WPT_EVALUATE=1` to evaluate at the end; `WPT_GPU_WORKERS`: the gpu workers' keys, or how many to rent (rented workers run the training and the evaluation with `WPT_KERNELS=triton` unless the settings set `WPT_KERNELS`; `WPT_RENTAL_GPUS`, `WPT_RENTAL_KIND`, `WPT_RENTAL_PROVIDER`; a lost rented worker, or a crash on several workers, resumes the run up to `WPT_REPLACEMENTS` times; `WPT_BUDGET` in dollars, `WPT_MAX_HOURS`, or both: the first reached stops the run, its checkpoint saved, and the cost is printed before each rental; `wpt manager stop [RUN]`, or a SIGTERM (`docker compose stop`), stops a run on purpose the same way, and `WPT_STOP_DEADLINE` seconds later (default 600) the manager ends what has not ended and gives the machines back, docs/machines.md; the container exits when the runs are over, and `docker compose up` again continues a failed run); goes on first with its own runs a crash cut short; builds the images of the dataset's tasks where `DOCKER_HOST` reaches Docker; settings from the environment, else from `.env` (docs/machines.md) | the recipe, the upstreams, the store | the session: its sources, dataset, checkpoint and the model under `<id>/models/NAME/` (the manager's own copy of the folder removed); the tasks' images on the Docker host; its run record `<id>/runs/<run>/`; its runs' gpu workers, tokens included, in `WPT_MANAGER_STATE` (never the store); rentals at the provider, stopped once the evaluation or training ends, and tried again until stopped |
| a gpu worker on a machine with GPUs | `docker compose -f compose.yaml -f compose.nvidia.yaml up` (or `compose.amd.yaml`; the files of `deploy/gpu-worker/` plus `wpt_worker/` and `wpt_utils/` copied beside them; prints its key `wpt-worker://TOKEN@HOST:PORT#PIN`; give it, one per machine, to `wpt train` in `WPT_GPU_WORKERS`; docs/machines.md) | what the wpt manager sends it | its own folder (certificate, checkout, libraries, checkpoints); never the store |
| check a machine before work | `uv run wpt doctor [--session ID]` (the store answers, and its credentials, the GPU and the PyTorch build, the attention kernels and nvcc, the Rust toolchain, free disk, and with an ID the sources the session reads, uncommitted or untracked files; exit code 1 when something is missing) | the settings, the store, the checkout | prints what is missing and how to fix it |
| manifest of a source's stored tree | `uv run wpt dataset manifest ID` (downloads it as `download` does, or finds it in the store) | the upstream, the store | the store; prints the list of the tree's files and the tree's manifest (the sha256 of that list) |
| erase a passage | `uv run wpt dataset cut ID PATH START END --version V` (bytes of the file as the cuts before left it; the cut names the passage by its length and two hashes keyed by the store's cut key, never its text; docs/erasure.md) | the store | `data/sources/ID/source.json`; the store's cut key on the first cut (keep a backup) |
| edit a source's files | `uv run wpt dataset unpack ID FOLDER`, edit, `uv run wpt dataset patch ID FOLDER NAME` | the source, the folder | the folder; then `data/sources/ID/NNNN-NAME.patch` |
| check the data rules | `uv run wpt dataset check [--fetch [ID ...]] [--changed REV]` (`--fetch`: the sources are downloaded as `download` does, then checked strictly: cuts, patches, license and format maps, every file text, pointer tasks; it names each file that is not text, to list in `extract`, and each license entry whose statement would keep its files out (not there, a text no item holds, with that text's id, one it does not expect, or one not admitted), without failing; failures listed at the end) | `data/` | the store, with `--fetch` |
| build a vocabulary | `uv run wpt dataset build-tokenizer NAME [--stream] [--strict] [--plan]` (NAME as for `build`: stops before any token; a failing source is left out and listed in the lock; `--strict` (`WPT_STRICT=1`): any failing source stops it, every source whose format map fails named before any text is read; `--plan`: as for `build`) | the data part of the model recipe NAME | the store; its run record `<id>/runs/<run>/`; `data/recipes/tokenizers/<its tokenizer>.lock.json` (commit it) |
| exclude a source's unusable files | `uv run wpt dataset exclude-unusable ID` (fetches it from upstream, nothing kept, past each file upstream does not give; then the text check) | the source | its `source.json`: exact-path excludes with their reasons in `extract.note`, for review before the commit |
| follow a run | `uv run wpt store runs [--session ID]`, `uv run wpt store run FOLDER [--session ID] [--follow]` | the store's run records | prints the runs, or one run's stage, counts, last events and errors |
| check a vocabulary on the data | `uv run wpt dataset tokenizer-check NAME` | the lock, admitted sources | prints characters per token per source |
| build a dataset | `uv run wpt dataset build NAME [--stream] [--strict] [--plan]` (NAME: a model recipe, or `SERIES:SIZE`, a size of a model series, every size of which builds the same dataset; never contacts upstream; reads only the sources of its session: a source that fails (not in the session, its download failed, a cut or patch that does not apply, a format map that fails, a broken file) is left out and listed in the record, and in the lock of a vocabulary the build trains; each item that fails alone is left out and reported in the record: a file no license entry covers, a file that is not text, the files and records of an entry that does not train (not admitted, or its statement in the stored bytes matches no admitted text it expects), a task whose pointer no longer holds its passage or points into a source left out; sources read in worker processes, several at once, the same dataset as one at a time; reads only the documents its mixture chooses from the index of the sources' records, drawn in a seeded order of whole 1 MiB frames (the files that start in one frame of one source's tar go together, so a pool's share may be passed by one frame's files, and the vocabulary's sample ends on whole frames); the first build naming a vocabulary with no lock trains and locks it; `--stream`: the Markdown of the chosen documents goes to a temporary folder deleted when the build ends; `--strict` (`WPT_STRICT=1`): any failing source stops the build instead, every source whose format map fails named before any text is read; a missing program stops it either way; `--plan`: stops after the choice and prints what it would read and compute) | the data part of the model recipe NAME | the session's `<id>/datasets/<name>/` (name: the recipe's, or `series-<series>`, a hyphen and the first 8 hex characters of the dataset id; the folder holds the record, shards, vocabulary, document list, credits, conversations and tasks, so a build with other content writes another folder and the same content finds its folder), the dataset last built for the recipe or series, which `wpt train` finds in the store's index; the store's `cache/` (only what can be made again); its run record `<id>/runs/<run>/`; prints the dataset id |
| check a dataset for contamination | `uv run wpt dataset contamination DATASET_ID` (by hand, never part of a build or of `wpt manager`; held-out runs of 13 words found in the dataset's training shards, reported only: an overlap never sets the exit code; a dataset with a saved report, or one inside its record, prints that report; docs/data-selection.md, "Held-out text and contamination") | the dataset's record, vocabulary and shards, one at a time | the store: the report `cache/dataset-contamination/<id>`; its run record `<id>/runs/<run>/`; prints the report |
| migrate data files | `uv run wpt dataset migrate` | `data/` | rewrites files whose shape is behind |
| check the Markdown of sources | `uv run wpt dataset markdown-check [ID ...]` | the sources | prints notes and problems per source |
| train | `uv run wpt train RECIPE [DATASET_ID] [--out FOLDER] [--stream] [--recompute] [--compile] [--segment N] [--run NAME] [--scheduler-costs RECORD_FOLDER] [--kernels FORM] [--stop-after-minutes M] [--stop-after-updates N] [--profile-update N] [options]`, started detached (skill `wpt-training-run`; RECIPE: a model recipe, or `SERIES:SIZE`, a size of a model series, which reads the start of the series' dataset, or `SERIES:SIZE:RUN`, another run of that size (`branch-SHARE`, a decay branch that goes on from the size's trunk; `lr-FACTOR`, at the smallest size, the learning-rate check); without DATASET_ID, the dataset last built for the recipe or series; `--stop-after-minutes`, `--stop-after-updates`: stop as a signal does, the first reached, and print the tokens per second and the projected time of the run (docs/training.md, "A limit and a measure"); `--profile-update N`: update N of each step runs under the profiler and its run record receives `profile.txt` (same section); trains through gpu workers, the keys of `WPT_GPU_WORKERS` (one per machine), or else a local one it starts; exit code 4 when a gpu worker is lost; `WPT_LINK_CHECKPOINTS`: `step-ends`, the default, or `every`, the checkpoints that come back from the worker; docs/machines.md; runs the recipe's `steps` in order, starting from the result of the last of its first steps the store already holds, so a run reuses the steps it shares with an earlier one; a dataset that holds nothing for a step refuses the run before any step trains; a step that cannot do its part (`StepFailed`) stops the run with exit code 2, no result of it kept but the checkpoint it wrote, which `--resume` continues (with `WPT_LINK_CHECKPOINTS=every`; with `step-ends` it stays on the worker and `--resume` starts the step over), the results of the steps before it kept; ends by training the model's drafter; `--scheduler-costs`: replay the drafter scheduler's pass times an earlier run recorded on the same setup; a checkpoint is always written just before the drafter, so a drafter failure resumes with `--resume`; `--stream`: a few shards on disk at a time; `--recompute`: much less memory, about a third slower; `--compile`: the passes over whole text windows through `torch.compile`, on a CUDA device only (docs/training.md, "Compiling the passes"); `--segment N`: windows longer than N tokens run as segments of N, the same training in the memory of N tokens plus what the segments hand on, up to twice the time; with `--strategy split`, each process's slice runs in segments of N; `--run`: the checkpoint in the store, so another machine can continue; `--kernels`: `triton`, `tilelang` or `pytorch` computes the attention, one that cannot work refusing the run (unset: the first kernels installed for the GPU; `wpt manager` names `triton` on rented workers); docs/model.md, "Attention") | the model recipe, the store | a model folder; each step's result, the store's `cache/steps/<fingerprint>`; its run record `<id>/runs/<run>/` (one per command, over restarts); `scratch/` token files in the store's local folder; with `--run`, the record `<id>/models/NAME/checkpoint.json` and the state it names, beside it in `<id>/models/NAME/`, until the model is saved |
| train on several GPUs | `uv run wpt train RECIPE DATASET_ID --strategy ddp --processes N [options]`, or `--strategy split` to split each window between the GPUs (windows too long for one; DeepSeek-V4's contextual parallelism) (on the gpu worker's machine) | the store | the same |
| evaluate | `uv run wpt eval FOLDER [EVAL ...] [--attempts N] [--probe-rate RATE] [--kind-by FIELD] [--kernels FORM]` (from inside the repository; the model runs on the first gpu worker of `WPT_GPU_WORKERS`, else a local one; the agent measures play N attempts, default 4, at every measure task, so each task's image must be built; the `probes` test asks each question of the probe files ceil(3 / RATE) times, default 1/100; the `held-out-loss` test gives the loss on every held-out token, each source's and each kind of text's, by `--kind-by origin`, `tag` (the default) or `format`, or `WPT_KIND_BY`; `--kernels tilelang`, `triton` or `pytorch`, or `WPT_KERNELS`, names what computes the attention on the gpu worker, and one that cannot work there refuses the evaluation) | the folder, the store, the measure tasks, the probe files | `FOLDER/evals/<name>.json`, `README.md`; its run record `<id>/runs/<run>/`, with `attempts.jsonl` |
| fit the scaling curve and predict a loss | `uv run wpt scaling MODEL MODEL [MODEL ...] [--predict RECIPE ...]` (MODEL: a model folder or a run's name the store keeps; loss against compute C = M x tokens trained, M as a series' plan counts it, L = a x C^-b, plus an irreducible loss from 4 points; the loss is the overall loss of the model's `evals/held-out-loss.json`, and a model without it is named and left out (`WPT_EVALUATE=1` or `wpt eval` makes it); RECIPE: a model recipe or `SERIES:SIZE`, predicted at its planned compute; docs/evaluation.md, "Scaling curve") | each model's `provenance.json` and `evals/held-out-loss.json`, from its folder or the session's `<id>/models/NAME/`; the recipes | prints the points, the fit and the predictions |
| build program-built tasks | `uv run scripts/build_tasks.py KIND SOURCE --count N [--seed S] [--select EXPRESSION]` (kinds `find-the-passage`, `evidence-removed`, `look-or-ask`, `commit-and-push`, `read-the-manual`; SOURCE an empty source a person made; docs/training.md, "The task builder") | admitted sources in the store | everything under `KIND/` in SOURCE, as its next patch |
| lock an image of agentic tasks | `uv run wpt agent lock-image NAME [--force]` (needs Docker; `--force` replaces an existing lock) | `data/images/NAME/Dockerfile`, the base image's registry, the dated package archive | `data/images/NAME/lock.json` and `licenses.json` (commit them) |
| build an image of agentic tasks | `uv run wpt agent build-image NAME` (needs Docker; before `wpt train` with `finish-tasks` or `finish-tasks-with-less-writing`, and before `wpt eval`'s agent measures) | `data/images/NAME/`, the base image's registry, the dated package archive | a local Docker image |
| continue text | `uv run wpt chat FOLDER` (`--kernels tilelang`, `triton` or `pytorch`, or `WPT_KERNELS`, names what computes the attention, and one that cannot work stops the command; `--about` prints its NOTICE; `--chat` talks to a model trained on conversations; `--scheduler-costs RECORD_FOLDER`, its `<id>/<run>`, reads the scheduler's pass times a run recorded on the same setup instead of measuring them) | the folder; with `--scheduler-costs`, the store | |
| the sizes of a model series | `uv run wpt dataset series NAME` (each size's shape, parameters without and with the vocabulary tables, compute, tokens, learning rate, batch, updates, warm-up and prefix, and its other runs `SERIES:SIZE:RUN`) | `data/recipes/series/NAME.json`, through the service layer | prints them |
| the kinds of data files, their names, one file | `uv run wpt dataset kinds`, `list KIND`, `show KIND NAME` | `data/`, through the service layer | prints them (version and problems on stderr) |
| write a data file people write | `uv run wpt dataset save KIND NAME FILE --version VERSION` (`''`: a new file; a save of a source that changes an entry's license, place or expected texts without deciding it anew sends that entry back to `under-review`) | the file | that file, after the rules of `check`; refused when it changed since VERSION |
| a source's files, tags and git status | `uv run wpt dataset files ID`, `tags ID PATH [TAG ...] --version V`, `remove ID PATH --patch NAME --version V`, `status` | the source | `source.json`, or the next patch |
| a new empty source, without its texts | `uv run wpt dataset new-source ORIGIN [ITEM] NAME --license L --attribution TEXT --by NAME [--tag TAG ...] [--chat PATH ...] [--format FORMAT=PATTERN ...]` (never `--file` or `--license-text`: the person writes the license text, then the files) | | `data/sources/ID/source.json`, `under-review`, its license text empty |
| add a license text a source holds | `uv run wpt dataset add-license-text ID --file PATH` or `--page ADDRESS` (the bytes at that file of its final tree, or that license page downloaded with it; nothing written when a license text holds them already) | the source in the store | `data/license-texts/<text id>/` (`license-text.json` under review, `license-text.txt`); prints the text id |
| a person's texts (people only, never an AI assistant) | `read`, `write`, `conversation`, `save-conversation`, `task`, `save-task`, `save-rating`, `save-probe`, and `new-source --file` and `--license-text` (marked `*` in `--help`) | | |
| conversation, task, rating and probe structure | `uv run wpt dataset conversations ID PATH` (roles, drafts), `tasks ID` (image, limits, files), `ratings ID PATH` (the attempt each rates, who, when; never the verdict), `probes ID PATH` (how many patterns each holds; never the question nor the patterns) | the source | prints them |
| edit every kind of data in a browser | `uv run wpt web` (needs Bun), then http://127.0.0.1:8410/ (HTTP API described at `/api/openapi.json`; page "Ratings": people's pass or fail on the attempts of a `wpt eval` run, a measure only, docs/evaluation.md; page "Probes": the questions of the `probes` test and the patterns that flag an alarming reply, a measure only; page "Sources": a source's description, its address and license entries included; page "Jobs": start a subcommand and follow what it prints and its run record) | `data/`, a WPT model folder for drafts, the store's `wpt eval` run records for ratings, `apps/wpt/ui` | data files through the service layer; `ui/node_modules`, `ui/dist` |

Everything that changes a model comes from its model recipe
(`data/recipes/models/NAME.json`: data, architecture, seed, the training
`steps` in order, the drafter; docs/model.md), or from a model series
(`data/recipes/series/NAME.json`: the rules its sizes share and the list
of sizes), from which `wpt_dataset.series` derives each size's model
recipe (docs/model.md, "Model series"). Each step is
`{"step": NAME}` with, when they differ from the step file's, its
`optimizer` and that optimizer's settings, its `settings` and the parts it
keeps `locked`, named as the model names them (docs/model.md, "Parts";
docs/training.md, "Steps"), and `continues: true` when it goes on from the
whole state the step before it left (docs/training.md, "Continued
steps"). The built-in steps are `read-text`,
`answer-as-assistant`, `finish-tasks` and `finish-tasks-with-less-writing`.
The `read-text` step's `stages` list the window lengths in order, each
with its updates (`steps`) and its `batch_size`, which is global (over
every process) (docs/training.md, "Stages"); the model's context is the
last window length of the last step that reads text windows. Segments and split windows give
the whole window's loss and gradients, so they are options, never part of
the recipe (docs/training.md, "Long windows"). The `wpt train` options only
say how the hardware runs it, and each also comes from `WPT_<OPTION>` in the
environment (`--micro-batch` -> `WPT_MICRO_BATCH`); `--help` on any command
lists them. A crash restarts the processes on the gpu worker on their own, up to
`--max-restarts` times (default 3), from the last checkpoint; Ctrl+C or
`kill` finishes the current update and saves first, and that checkpoint reaches
the manager whatever `WPT_LINK_CHECKPOINTS` says; after a stop, the same command
with `--resume` continues from `FOLDER/checkpoint/` (or, with `--run NAME`,
from the store, on any machine). Exit codes: 2 refused, 3 the run's record
could not be written, 130 stopped.

Every run of `wpt dataset download`, `check --fetch`, `manifest`, `build`,
`build-tokenizer`, `contamination`, `wpt train`, `wpt eval` and `wpt manager` writes its record into its own folder of the session, `<id>/runs/<UTC
time>-<command>-<main id>-<random>/` (main id: the source id or `<count>-sources`, the recipe name, a short
dataset id or a short weights hash): `start.json`, `events.jsonl` (spans, events,
every printed line, with its level and without the time and level it prints with), `ids.json` (the ids read and written, which chain the
runs together), `end.json`. With `OTEL_EXPORTER_OTLP_ENDPOINT` set
(environment or `.env`), the same goes live to an OpenTelemetry backend
(`deploy/otel-lgtm/compose.yaml`). A backend failure is one warning; a
record the store refuses stops the command with exit code 3 (`wpt train`
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
signed commit (CONTRIBUTING.md, "Decisions are signed"). The same holds
for each license entry and each license text: a file trains only when its
source, its entry and the text found at its entry's place are all
admitted, and no program admits or refuses at any level.

## The service layer

`wpt_dataset/repo.py` is the only code that reads or writes `data/`
(docs/architecture.md, "The service layer"): sources and patches, chat and
task files, recipes, Dockerfiles, the signer lists, and the generated
locks and license lists, which only the command that makes them writes.
The license texts (`data/license-texts/<id>/`) are one of its kinds. Code elsewhere asks it for documents,
bytes and versions, never for a path under `data/`; `packages/wpt-dataset/tests/test_data_access.py` fails
otherwise, and an audit hook the layer sets refuses such an access at run
time (`DataAccessRefused`), except from tests. Every write applies the
rules of `wpt dataset check`, refuses a stale save (one lock, across
processes, holds the version check and the write), writes a symbolic
link's target only inside the repository, writes U+2028, U+2029 and
U+0085 as escapes, and keeps a JSON file's layout where its meaning did
not change. A new kind of data gets both interfaces, a
`wpt dataset` command and a `wpt web` route with its JSON Schema, both thin
layers over the service layer. A new subcommand of `wpt` gets its job route
in `apps/wpt/src/wpt/web/pages/jobs.py`, unless it reads or writes a
person's text or reads what a person types.

## Patches and text written for WPT

Never write a patch by hand. `wpt dataset unpack` writes a source's final
files into a folder outside the repository; after the edit,
`wpt dataset patch` records the difference as the next patch and keeps it
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
ratings"). The questions the `probes` test of `wpt eval` asks and the
patterns that flag an alarming reply are `probe` files of such a source,
one probe per line, a measure only too (docs/evaluation.md, "Probes").
Committed text
stays in the repository's history for good.

As an AI assistant, use the `wpt dataset` commands only for structure and
metadata, never to read or write people's texts (CONTRIBUTING.md, "Models
and generated text"): never run `read`, `write`, `conversation`,
`save-conversation`, `task`, `save-task`, `save-rating` or `save-probe`, and never call the matching
`wpt web` routes. You may run `new-source` without `--file` and
`--license-text`, never with them. People write
their texts in `wpt web` or with those commands themselves. An `assistant`
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
  never through the commands marked `*` in `wpt dataset --help`.
- From a session with an AI assistant, keep only the person's own
  messages.

## Vocabularies

A tokenizer recipe's lock freezes its vocabulary: adding sources never
retrains it (docs/tokenizer.md). A dataset recipe names its vocabulary, and
the first build that names one with no lock trains it on a sample of that
dataset's chosen documents. Raising the recipe's `version` trains a new
one and replaces the lock; it is a person's decision (it makes later models
incompatible with earlier ones), so never raise it unasked. A build that
stops with `TokenizerVersionNeeded` means the recipe changed: ask whether to
raise the version or undo the change. A change of the tokenizer's code asks
for no `version` of the recipe; one that changes the tokens of the same text
raises `TOKENS_VERSION`. Commit the
lock file a build writes.

## Changing a file's shape

No data file carries a shape number: a step recognizes an old shape by its
fields. To change a shape: update the schema file, add a `Step` (what it
finds, how it rewrites) to the list of its kind in `wpt_dataset/migrate.py`,
with an old shape that leaves a field difference to find, and run
`wpt dataset migrate`. A file `migrate` cannot rewrite, such as a source's `chat`
file (upstream's bytes), goes through `migrate.upgrade` where it is read.
Never edit data files by hand for a shape change. A field only a
person can fill is left empty by the migration, and the check then names
each file to fill in.

## The model folder

`wpt train` writes `config.json`, the weights (each safetensors header
carries `license`, `output_license`, `notice`, `tokenizer` and
`tokenizer_sha256`), `tokenizer.json` (loaded only through
`wpt_model.load_tokenizer`, which checks it against those headers),
`provenance.json` (what produced it, including every source it learned
from), and `LICENSE`, `NOTICE` and `README.md` (the model card).
`NOTICE` and `README.md` come from `wpt_model.notice.write_documents`, which
reads only the folder's own files: change what they say there, never by
editing a generated folder. `wpt train` writes the folder beside its
target and renames it into place; training into an existing folder replaces
it as a whole (its `evals/`, checkpoint and any other file go with it).
Every model folder holds `drafter/` (`config.json`, weights with the same
header, `provenance.json`): the drafter for speculative decoding, which
`wpt train` trains at the end of every run against the final weights. A model and its drafter are one unit: `load_model` and
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

One Python file, a built-in in `apps/wpt/src/wpt/eval/evals/` or any
path given to `wpt eval`:

```python
from wpt.eval import Result

def run(context):  # context: model (on a gpu worker: config, generator,
                   # writer, greedy, losses), tokenizer, device, folder, provenance, store,
                   # telemetry (the run's record), documents("training") and
                   # documents("held_out"), read one shard at a time:
                   # iterate, never assume a list; attempts(plain=False),
                   # the agent measures' shared run of attempts at the
                   # measure tasks, played once per command;
                   # probe_rate (wpt eval's --probe-rate); kind_by (--kind-by, default tag)
    ...
    return Result(data={...}, text="the same findings, for people")
```

Every evaluation returns `Result`: `data` that JSON can hold, and `text`
that the model card shows. Details: `docs/evaluation.md`.

## Writing a training strategy or step

A strategy (how the hardware is used) is one Python file, a built-in in
`packages/wpt-training/src/wpt_training/strategies/` or any path given to
`wpt train --strategy`:

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
def start(model, settings, run): ...  # a wpt_training.Optimizer over run's parameters; optionally check(settings)

# steps/NAME.py: puts them together
METHOD = "next-token"
ENVIRONMENT = "text-windows"
OPTIMIZER = {"name": "by-part", "settings": {...}}
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
the model alone (no drafter), as the agent measures of `wpt eval` do, and
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
reply = next_message(model, tokenizer, conversation, max_new_tokens=512)  # the next chat message, as wpt chat --chat
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
- Print a line through `telemetry.log(text, level)` in a recorded run, else
  through `wpt_utils.lines.say(text, level)`, never `print`: every line starts
  with its time and level (docs/observability.md, "Printed lines"). Code that
  reads a line a command printed takes its text with `wpt_utils.lines.bare`.
- A result of the data pipeline holds in its fingerprint every input of
  `data/` its computation reads and its stage's version, never WPT's code:
  a code change leaves stored results found. A training step's result
  holds `wpt_utils.programs.code()` (the commit and a hash of the uncommitted
  changes). A file of `data/` that every source shares enters only as what
  the source found in it, found again on every build, as the license
  check's answer enters what a source keeps.
- When a change to the code changes what a stage of the data pipeline gives
  for the same inputs, raise that stage's version in the same commit
  (`TREE_VERSION`, `NOT_TEXT_VERSION`, `KEPT_VERSION`, `INDEX_VERSION`,
  `NEWEST_VERSION`, `MARKDOWN_VERSION`, `TOKENS_VERSION`,
  `VOCABULARY_VERSION`, `DATASET_VERSION`; docs/architecture.md,
  "Fingerprints and the store"). Every later build on any machine then does
  that stage again, so raise none for a change that gives the same results.
  A package or Python upgrade that changes what a stage gives counts as such
  a change. A locked vocabulary is the exception: it is never trained again,
  so `VOCABULARY_VERSION` acts only on a recipe with no lock.
  A new stage gets a version, and sets and finds its result through
  `wpt_telemetry.set_result` and `get_result`, so a build warns when
  it reuses another commit's result.
- Parallel work sizes itself with `wpt_utils.parallel.Sizing`, which
  measures each item's memory as it runs: never give a memory figure per
  item. A source's conversion or reading goes through
  `wpt_dataset.workers` (or `sources.each_read`), with a module-level
  function, since it runs in a worker process.
- Test isolation: a test never writes into the real store, cache or user
  settings; set `WPT_STORE`, `XDG_CACHE_HOME` and `LOCALAPPDATA` to the
  test's own folder when it touches them.
- A test passes both alone and under pytest-xdist. One that compares two
  trainings bit for bit, or runs several processes of a training on this
  machine, takes the `on_the_cpu` fixture of the root `conftest.py`
  (docs/architecture.md, "Checks").
- A test that must see something happen waits for that thing, not for a
  length of time, and a limit in seconds stands far above what it takes.
- The app's tests share their setup (a dataset built from a git
  project, a tiny trained model) through the `built` and `trained`
  fixtures in `apps/wpt/tests/conftest.py`; the builders of throwaway
  repositories, upstream projects and HTTP servers (the `kit` fixture)
  that the tests of wpt-dataset and of `wpt dataset` use are in the root
  `conftest.py`.
- A new third-party package in `uv.lock` or `Cargo.lock` gets a row in
  `docs/licenses.md` (version and license, read from its metadata), and a
  changed version gets its row checked again.
- The repository is public: never commit personal paths, machine names,
  credentials, or data that is not a pointer to a licensed source. A person's
  name holds no colon; the one machine name committed is
  `machine:wpt dataset`, the `by` of a license entry the service layer
  sent back to `under-review`, and `check` refuses an `admitted` or
  `refused` decision by a machine.
