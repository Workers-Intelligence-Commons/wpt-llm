# Observability: the record of every run

Every run of `wpt-dataset download`, `check --fetch`, `fill-checksum`,
`manifest`, `build` and `source-drafter`, each round of `source-server`,
`wpt-train`, `wpt-eval` and `wpt-manager` keeps a record of what it did, from its first input to its last
file, so that afterwards one can know exactly what happened. The record is
written into the store, as OpenTelemetry's three kinds of data (traces,
logs and metrics) and in a form anyone can read without a backend. When an
OpenTelemetry backend is set, the same data is also sent to it live, to
watch a run while it goes (`packages/wpt-telemetry`).

## What is recorded

Each command records spans (a part of the run with its start, end,
duration and outcome), events (something that happened, with numbers when
it has some), every line it prints, the ids it read and wrote, and how it
ended.

| Command | Records |
| --- | --- |
| `wpt-dataset download`, `check --fetch`, `fill-checksum`, `manifest`, a round of `source-server` | the session of the download step: how many sources download at once and why; each download tried again; every ref set for a kept source (`upstream-files/<checksum>`, `upstream/<checksum>`). Inputs: the source ids. Each source's outcome (downloaded, reused or failed) is a line of the session in the store database alone, never in the record ([architecture.md](architecture.md), "The download step") |
| `wpt-dataset build` | the recipe and mixture; the vocabulary (built, or found from its lock), and each source the tokenizer training read; the lock file written (its path and the vocabulary's sha256); how many sources are converted at once and why; each source read and converted (its tree, documents, duplicates, credits, the seconds its conversion took) or why it was skipped (a source not in the store, `NotDownloaded`, included); the conversations and tasks kept for the training steps (spans `conversations` and `tasks`: how many, their object, their sources); the tokens; each pool; each shard written (training and held-out: object, rows, tokens, token sha256, sources); the contamination check; the dataset record. Inputs: the recipe, the tokenizer, the vocabulary, every source's checksum, the skipped sources. Outputs: the dataset id, its record object and its document list |
| `wpt-dataset source-drafter` | the source drafter (its file and sha256, its settings, the name of its record of what is done) and who started it; the candidates listed (a span), and how many were chosen, done before and left to draft (the file `candidates.json`); one line per outcome in `outcomes.jsonl` (candidate, outcome: drafted, skipped, needs-a-person, stopped or failed, the source, its change: new, updated, licenses-changed or unchanged, the reason and the notes for a person), and every line printed. Inputs: the source drafter, its file's sha256, its settings, the record of what is done |
| `wpt-train` | the recipe and its steps (a span `inputs.steps`); the dataset read, its token counts and the conversations and tasks it holds (a span `inputs.dataset`); the settings; each step reused from an earlier run (the event `training.step_reused`: its number, fingerprint and result); each step trained (a span `step`: its name, number and fingerprint, and what it reached), and inside it each of its stages (a span `training.stage`: its number, window length, batch, first and last updates, and the update a resumed run started at), every update (the event `training.step`: its loss over the whole batch, its window length, its duration, the learning rate of each parameter group), every update skipped, every evaluation (validation loss, best or not, stop or not), every checkpoint written, a resume, a stop, and what the step's environment records itself (`answer-as-assistant`: the conversations read and their tokens; `finish-tasks` and `finish-tasks-with-less-writing`: every attempt as a line of `<step>-attempts.jsonl` as soon as it is scored, a conversation in the chat format whose `metadata` identifies the attempt, with a span per group's attempts, a span per attempt inside it and an event per group and per round naming the lines, [training.md](training.md), "Runs"); each step's result kept in the store (a span `training.step_kept`: its fingerprint and the result's sha256); the drafter's training (a span `drafter`: the model's answers, its settings, every update's losses, the calibration and the share of drafts kept); the scheduler's pass times, measured or replayed (a span `scheduler.costs`), written as the file `scheduler-costs.json`, which `--scheduler-costs` replays in a later run; the weights, the drafter, `provenance.json`, the credits, `NOTICE`, `LICENSE` and the model card written, with the sha256 of every file of the model folder; the checkpoint's removal, and with `--run` every checkpoint object deleted from the store (the one each save replaces; a step's last one stays as its kept state), named by its sha256; the sample; a gpu worker lost (the event `gpu_worker.lost`, exit code 4). Inputs: the recipe, the dataset id, its record object, the vocabulary, the results of the steps reused. Outputs: the result of each step trained, the model folder, the sha256 of each weight file, of the drafter's weights and of `provenance.json` |
| `wpt-eval` | each test (its file and sha256, its result's data), each result file and the model card written. Inputs: the weights' sha256 and the dataset id. Outputs: the sha256 of each result file and of the model card |
| `wpt-manager` | each of its commands as a span (`manager.download`, `manager.build` with the dataset id, `manager.train`, `manager.evaluate`), every line they print, the run's gpu workers (a span `manager.workers`, and inside it a span `manager.rent` per rental: its name, its kind and the worker's key with its token hidden), each lost gpu worker (the event `manager.worker_lost`) and its replacement (`manager.worker_replaced`), each rental stopped (`manager.rental_stopped`, recorded before the stop) or not (`manager.rental_stop_failed`), a record named `rentals` for each try to stop the rentals of runs that ended, and the model kept in the store (a span `manager.model` with the manifest's sha256, and the ref `models/<run>` as `store.ref_set`). Each command it runs keeps its own record too. Inputs: the recipe, the run's name, the dataset id. Outputs: the model's manifest |

The drafter scheduler's pass times are measured once per run of `wpt-train`,
and written into the run's record as
`scheduler-costs.json`: its shape (`schema`: 2), the `setup` they were
measured on (`model`: the model's config; `rows`: 1; `draft`: the drafter's
block; `device`: its kind and name, the processor's on the CPU), and
`seconds`, the seconds of a pass for each width. A later run replays them
with `--scheduler-costs FOLDER` (`wpt-train`; `wpt-chat`
reads them the same way), naming the earlier run's record folder; times
recorded on another setup are refused. [model.md](model.md), "The drafter",
says exactly what a replay guarantees.

Each command also records every store ref it sets, before setting it, as
the event `store.ref_set` (`ref`: such as `datasets/<id>` or
`checkpoints/<name>`; `object`: the sha256 it will point at), and each
step that cannot be undone just before it (such as
`model.folder_replaced`, `eval.file_written`, `tokenizer.lock`).

In `wpt-train`'s jobs on the gpu workers, with several processes, only the
main process records (the first process of gpu worker 0):
every process takes the same steps, and the main one averages the loss over
all of them. The others record nothing.

## The record in the store

Each run writes its own folder, `records/<folder>/` in the store named by
`WPT_STORE`, through the store's own interface (`wpt_store`: `write_record`,
`append_record`, `flush_records`, `read_record`, `record_files`,
`record_folders`). The folder is named
`<UTC start time>-<command>-<main id>-<8 random hex digits>`, such as
`20261017T120501Z-wpt-train-5d41402a-3fa2b1c9`, where the main id is what
the run is about:

| Command | Main id |
| --- | --- |
| `wpt-dataset download`, `check`, `fill-checksum`, `manifest`, a round of `source-server` | the one source id, or `<count>-sources` |
| `wpt-dataset build` | the model recipe's name |
| `wpt-dataset source-drafter` | the source drafter's name |
| `wpt-train` | the first 8 characters of the dataset id |
| `wpt-eval` | the first 8 characters of the measured weights' sha256 (its first weight file's) |
| `wpt-manager` | the run's name |

A character the store does not allow in a name (it allows letters, digits,
`.`, `_` and `-`) becomes `-`; `none` stands for weights that could not be
read. The whole ids are in `ids.json`. Model folders, weights and model
cards stay where they were; the record only names them and their sha256.

| File | Holds |
| --- | --- |
| `start.json` | the command, its arguments, when it started (UTC), the ids known at the start, the tool versions (Python, platform, WPT's packages, torch), the repository commit and whether the working tree had changes, and `trace`: the trace id the live backend shows the run under |
| `events.jsonl` | one JSON object per line, in the order things happened (below) |
| `ids.json` | `{"inputs": {...}, "outputs": {...}}`: every id the run read and wrote, rewritten as each is learned (each process adds to the copy in the store, so no process writes back an older one) |
| `end.json` | `status` (`succeeded` or `failed`), `exit_code`, `error`, `traceback` (where the failure came from, as Python prints it, with the error it was raised while handling; `null` when it succeeded), when it ended, and how many seconds it took |
| other files | written by a part of the run, such as a training step |

A folder without `end.json` is a run that did not reach its end: a crash,
a power cut, or a record that could not be written.

Each line of `events.jsonl` has `time` (UTC), `kind`, and `span`, the
span it happened in:

| `kind` | Other fields |
| --- | --- |
| `span_start` | `name`, `parent` (the enclosing span), `attributes` |
| `span_end` | `name`, `status` (`ok` or `error`), `seconds`, `attributes` set during the span, `error` and `traceback` |
| `event` | `name`, `attributes`, `values` (its numbers) |
| `log` | `severity` (`info`, or `error` for a line printed to stderr), `text`: a line the command printed |
| `input`, `output` | `ids`: what `ids.json` gained |
| `file` | `name`, `bytes`, `sha256` of a file written whole into the folder (a file added to line by line has no such line: it is its own record) |

A span's id is the OpenTelemetry span id, so a line can be found in the
live backend too.

A reader skips a line that is not valid JSON. Only a crash makes one: the
last line of a local file, cut while it was written. The next process that
adds to the file (a restarted training process) first ends that line, so
its own lines start whole.

### Following a model through its runs

The ids connect the folders: a build's `outputs.dataset` is a training
run's `inputs.dataset`; a training run's `outputs.step_results` holds the
results a later run reuses, named in its `inputs.reused_steps`; a training
run's `outputs.weights` is an evaluation's `inputs.weights`. Each command is its own run; nothing groups them beyond
these ids.

```python
import json
from wpt_store import store_from_settings

store = store_from_settings()
for folder in store.record_folders():
    ids = json.loads(store.read_record(folder, "ids.json"))
    print(folder, ids["inputs"].get("dataset"), ids["outputs"].get("weights"))
```

### wpt-train's processes and restarts

`wpt-train` writes `start.json` and `end.json` on the wpt manager's side
and starts its job on the gpu worker ([machines.md](machines.md)). With
`--processes` or `--max-restarts` (by default, 3), the job runs a small
launcher that starts the training processes and starts them again after a
crash; the launcher records the restarts. The launcher and the main
training process of each attempt join the same folder (the job hands them
`WPT_RECORD_FOLDER` and `WPT_RECORD_PARENT`) and write it through the link
to the manager, which writes the store: one command is one folder however
many times it restarts, and the worker never holds the store's
credentials. A line the job records goes to the manager's store before its
call returns, and the job's flushes are the manager store's own: on S3,
the lines it holds back are sent when the job's flush returns.

### Durable on a local folder and on S3

A record file is written whole (`start.json`, `ids.json`, `end.json`) or
added to line by line (`events.jsonl`), never both: the store refuses the
other kind of write to a file, whichever process made it.

- A local store (`file://`) appends each line to `events.jsonl` and asks
  the system to put it on the disk (`fsync`) before the call returns; a
  new file's name is put on the disk the same way. Whole files are
  written beside their name and renamed over it. A power cut can lose at
  most the line being written. A file added to line by line has a hidden
  marker beside it, `.appended-<name>`, which tells a later process its
  kind.
- An S3 object cannot grow, and each request costs time and money, so an
  S3 store holds the lines back and sends them in groups: every 5 seconds,
  at the end of every span, before the job starts,
  and at the end of the run. Each group is one object,
  `records/<folder>/events.jsonl/<time>-<writer>-<number>`, holding its
  lines with the time each was added. A crash or a power cut loses the
  lines added since the last group was sent: at most the last 5 seconds,
  and less when a span ended meanwhile. Whole files are sent at once, one
  object each; S3 has stored an object when it answers the upload. A
  group that S3 refuses stops the command as any record failure does
  (below), also when it was sent in the background. A group whose send
  was cut short (Ctrl+C) is sent again later under the same name, which
  replaces the object: no line is stored twice. Before a step that
  cannot be undone (a model folder renamed into place, an evaluation's
  result file or model card written, a checkpoint deleted from the
  store, a tokenizer's lock file written, and every store ref set: a
  vocabulary, a source's tokens, tree or upstream manifest, a
  checkpoint, a dataset's id pointed at its record), the command
  records it as an event (a ref as `store.ref_set`, naming the ref and
  the object it will point at), sends the lines it holds and waits for
  S3's answer: a refused group stops it before that step. The one
  exception is a checkpoint saved once the record broke (below).
- Reading a file added to line by line on S3 lists its objects (one
  request per 1,000 objects) and reads each one (one request each). A
  training run of 100,000 steps at one second each sends about 20,000
  groups instead of 100,000 lines, so reading its `events.jsonl` takes
  about 20,000 requests: read a long run's record once and keep the copy.

The lines read back in the order they were added, on either store, even
when several processes added to one file (`wpt-train`, then its job's
launcher and training process, then the launcher again after a crash): on S3 each line
carries the time it was added, and reading sorts every line of every group
by that time, then by writer.

## When something fails

- **The live backend is unreachable or fails**: one warning on stderr,
  and the run goes on. Each later batch is still sent, in the background
  and without a word, so the live view resumes when the backend is back;
  a batch that fails is dropped. The store record is complete either way.
  At the end of the run, the command waits for what was left to be sent:
  with a backend that does not answer, up to the exporter's timeout
  (`OTEL_EXPORTER_OTLP_TIMEOUT`, 10 seconds by default) for each of the
  three kinds of data, so up to 30 seconds. The exporters' own messages
  about WPT's sends are left out, since the warning says it once; their
  messages about any other use in the same process, and every logging
  setting, are left as they are. A kind whose send fails then
  sends nothing more, however many batches are waiting: measured against
  a backend that accepts the connection and never answers, the end of a
  run waited 20 seconds with the default timeout, and 3.6 seconds with a
  timeout of 2 seconds after 5,000 events.
- **The store record cannot be written** (the store is unreachable, the
  disk is full, permissions): the command stops at its next record call
  (or sooner, when a send in the background was refused: below), prints
  `RecordError: the record of this run could not be written ...`, and does
  nothing more: no further step, no model folder, no result file.
  Everything recorded before stays. All four commands exit with code 3,
  which `wpt-train`'s job never restarts. A record that broke always
  ends the command with code 3, even when what the command was doing then
  ended on its own, or when a Ctrl+C reached it meanwhile. In `wpt-train`,
  the checkpoint is saved either way:
  - when the training process's record breaks (a record call refused,
    or a group of its lines refused in the background: the usual case,
    since it records far more often than `wpt-train` itself), it finishes its
    step without recording it, and every process stops after that step,
    as for Ctrl+C: the checkpoint is saved (unless `--checkpoint-every
    0`), and the processes exit with code 3. The broken record does not
    name that checkpoint, nor, with `--run`, the store ref pointing at it;
    `--resume` continues from it and loses no step;
  - when `wpt-train`'s own record breaks while the job runs (on S3, a
    group of its lines refused in the background), it asks the job to
    stop as Ctrl+C does, so its processes finish their step, save the
    checkpoint, and stop; `wpt-train` then exits with code 3.

  A checkpoint replaced in the store while the record broke stays there
  beside the new one, since its deletion is recorded first. The last
  checkpoint's deletion, once the model is saved, is also recorded and
  sent first: a record that breaks there turns a run whose model is
  saved into exit code 3, and leaves that checkpoint in the store.
- **A group of lines sent in the background is refused** (S3): the
  sending thread interrupts the command's main thread the way Ctrl+C
  does, and the command stops with the same `RecordError` and code 3,
  never with 130, which means a stop asked for by the user. A real
  Ctrl+C still gives 130. How soon it stops:
  - on Linux and macOS, the sending thread sends the main thread a real
    SIGINT. A wait ends at once (a pause, `wpt-train` waiting for its
    job, a download); a computation inside a library
    that does not run Python meanwhile (a GPU step, a large array
    operation) finishes first, and the command stops right after it;
  - on Windows, which cannot send a signal to one thread, Python's
    `_thread.interrupt_main` stops the command only once its main thread
    runs Python again: after the wait or the computation it was in.

  Where the command handles Ctrl+C itself, its handler runs instead:
  `wpt-train`'s training loop then finishes its step, saves its checkpoint
  and exits with code 3, as above; `wpt-train` asks its job to stop, as
  for Ctrl+C, then exits with code 3. A process that ignores
  Ctrl+C (one a shell that is not interactive started in the background,
  with `&`) is not interrupted: it stops at its next record call, at most
  one step or one line later.
- **No store at all** (`WPT_STORE` unset or wrong): the command stops
  before it starts.

## The live view

The backend is Grafana's `grafana/otel-lgtm` image: an OpenTelemetry
collector with Grafana, Tempo (traces), Loki (logs) and Prometheus
(metrics) in one container, one per machine. Its setup is
`deploy/otel-lgtm/compose.yaml`: the image pinned by its digest, its two
ports, and a volume, `otel-lgtm-data`, holding every component's data, so
it survives restarts. Its licenses are in [licenses.md](licenses.md). Each
component keeps data for its default time unless told otherwise, such as
`PROMETHEUS_EXTRA_ARGS=--storage.tsdb.retention.time=90d` (the image's
README lists one such setting per component). The store record is what
lasts.

Two settings, in the `.env` file at the repository's root (git-ignored:
never commit a password or a machine's address; `.env.example` lists them
empty) or in the environment:

- `WPT_GRAFANA_PASSWORD`: the password of Grafana's user `admin`. It is
  required: unset or empty, `docker compose` refuses to start the
  backend. It cannot be `admin`, Grafana's default: the container checks
  it (and that it is not empty) before anything starts, then runs the
  image's own start command. A refused password stops the container with a message that
  names the problem (`docker compose -f deploy/otel-lgtm/compose.yaml
  logs otel-lgtm`), and Docker starts it again, waiting longer each time
  (up to a minute), until the password is changed: `docker compose up -d`
  itself does not fail, so check `docker compose ... ps` after starting
  it. Grafana reads it only when it first creates its data in the
  volume; to change it later, change it in Grafana (the user's profile),
  or remove the volume. The same holds for a volume that already exists
  when this compose file is adopted: an `otel-lgtm-data` volume made
  before keeps its stored admin password (such as `admin`), and
  `WPT_GRAFANA_PASSWORD` does not replace it. Remove the volume
  (`docker compose -f deploy/otel-lgtm/compose.yaml down -v`, which
  deletes every component's data) or set the password in Grafana.
- `WPT_OTEL_ADDRESS`: the address Grafana (port 3000) and the OTLP port
  (4318) listen on. Unset, it is `127.0.0.1`: this machine only. To reach
  the backend from the other machines of a private network, such as a
  Tailscale network, set it to this machine's address on that network
  (`tailscale ip -4` prints it); the backend is then open to every
  machine of that network, and to no other.

Start it with Docker (or Podman's `podman compose`) from the repository's
root:

```sh
docker compose --env-file .env -f deploy/otel-lgtm/compose.yaml up -d
```

Point the commands at it, in the environment or in `.env` beside
`WPT_STORE`:

```
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
```

(on another machine of the network, with `WPT_OTEL_ADDRESS` in place of
`127.0.0.1`). The commands send OTLP over HTTP (`/v1/traces`, `/v1/logs`,
`/v1/metrics` under that address); OTLP over gRPC is not used. The other
standard settings come from the environment only:
`OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TIMEOUT`, and
`OTEL_METRIC_EXPORT_INTERVAL` (WPT sends values every 10 seconds unless it
is set).

Then open Grafana at http://127.0.0.1:3000 (or at `WPT_OTEL_ADDRESS`) and
sign in as `admin` with `WPT_GRAFANA_PASSWORD`. Anonymous access is off:
the image turns it on by default, as Admin, and the compose file sets
`GF_AUTH_ANONYMOUS_ENABLED=false`, so every visitor must sign in.

- **traces** (Tempo): one trace per run, its service named after the
  command (`wpt-train`), with the run's spans; `start.json` gives its
  trace id;
- **logs** (Loki): the printed lines and the events, with their
  attributes, under the same service;
- **metrics** (Prometheus): every value of an event as a gauge named
  `wpt.<event>.<value>`, which Prometheus writes with `_` for `.`:
  `wpt_training_step_train_loss`, `wpt_training_evaluation_val_loss`,
  `wpt_sft_update_loss`. Each run is its own series: its
  `service.instance.id`, which Prometheus shows as the `instance` label,
  is the record's folder name, and its `job` is the command. A gauge is
  sent every
  10 seconds with its last value, so the live curve can skip steps; the
  record has every one.

To watch a run on another machine, start the backend there, set
`WPT_OTEL_ADDRESS` as above, or forward its ports
(`ssh -L 3000:127.0.0.1:3000 -L 4318:127.0.0.1:4318 HOST`).

## Recording from a step or a test

A training step's environment gets the run's record as `data.telemetry`
(training.md, "Environments"), and an evaluation gets it as
`context.telemetry`:

```python
def run(context):
    telemetry = context.telemetry
    with telemetry.span("generation", prompts=8) as span:
        ...
        span.set(tokens=1234)
    telemetry.event("attempt", values={"reward": 1.0}, task="t-17")  # values also become live gauges
    telemetry.append_file("attempts.jsonl", line.encode())  # a file of this run's folder, one line at a time
    telemetry.write_file("summary.json", data)               # or written whole
    telemetry.log("a line to print and record")
```

Attributes are anything JSON holds. A file's name uses letters, digits,
`.`, `_` and `-` only, does not start with `.`, and is not one of the
record's own (`start.json`, `events.jsonl`, `ids.json`, `end.json`); a
file is added to with `append_file` or written with `write_file`, never
both. Breaking either rule is a mistake in the step or the test: the call
raises `ValueError` (a name) or `MixedRecord` (from `wpt_store`, a file
used both ways), writes nothing, and the record goes on.
Every call is in the store when it returns (on S3, an added line within
the next few seconds, as above), and raises `RecordError` when the store
refused it: let it propagate, the command then stops.
`telemetry.irreversible(name, **attributes)` records an event as
`telemetry.event` does, then sends the lines an S3 store holds and waits
for its answer: call it just before a step that cannot be undone, which
then happens only once its record is in the store.
Each thread records under the span it opened last, or under the run's own
span when it opened none. Work handed to other threads records under a
span of its own: `telemetry.begin(name, **attributes)` opens a span under
the current one without entering it, `telemetry.within(span)` makes a
thread record under a span for a with block (`telemetry.current()` gives
a thread's current span to hand over), and `telemetry.end(span)`, or
`telemetry.finish(span)` around a with block, ends it from any thread.
Once the run's record is closed,
every call raises `RuntimeError`: a step or a test must not keep the
telemetry for later.

`wpt-dataset build` also records each download it tries again after a
failure that may pass (a timeout, a server error), as a printed line.
