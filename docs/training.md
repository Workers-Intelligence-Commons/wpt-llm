# Training and repeatability

## Training

```sh
uv run wpt-dataset build classics-test     # prints the dataset id
uv run wpt-train classics-test <dataset id>
```

A run takes minutes to days, so start it detached from the terminal or SSH
session, which would otherwise end it and lose its output:
`.\scripts\detach.ps1 uv run wpt-train ...` on Windows,
`scripts/detach.sh uv run wpt-train ...` on Linux and macOS. Both write its
output to `runs/logs/<date-time>.log` and print how to follow and stop it.

`wpt-train` takes a model recipe (`data/recipes/models/<name>.json`, see
[model.md](model.md)) and the id of the dataset built from it, and never
builds: build and training fail separately. A dataset built from another
data part than the recipe's is refused. Everything that changes the
resulting model comes from the recipe: the architecture's numbers, the
method and its settings, the steps, the batch, the evaluations and the
seed. The command-line options say only how the hardware runs it; each
also comes from an environment variable (`WPT_` + the option name in
capitals, `--micro-batch` -> `WPT_MICRO_BATCH`; the recipe and the dataset
id from `WPT_RECIPE` and `WPT_DATASET`). `uv run wpt-train --help` lists
them.

Validation uses the dataset's held-out shards. The tokens are read from
disk as training needs them, so a dataset larger than memory trains: before
the first step, the shards are checked against their token hashes and
their tokens written once into one file per part, in the store's `scratch/`
folder (`<store folder>/scratch/` for a local store, the cache for S3). A
later run reuses that file when it still matches the sha256 recorded when
it was written. `scratch/` can be deleted at any time.

### Streaming: a dataset larger than the disk

`--stream` (or `WPT_STREAM=1`) never copies every shard to disk. The
shards are walked in order at the speed training uses tokens: before each
step, the shard holding the next unused token and the shards after it,
`--stream-shards` in all (8 by default, about 2 GB), are the ones in use,
and the step draws its windows from them only. The shard after them is
prepared while training goes on, and a shard that leaves the set loses
its token file and, from an S3 store, its local copy. After the last shard
comes the first again.

Which shards are in use depends only on the step, so a resumed run and any
number of processes take the same windows, but they are not the windows a
run without `--stream` takes: a streamed run mixes documents within the
shards in use (the build already put the rows of every shard in a
pseudo-random order), not over the whole dataset. A run must resume the
way it started: with the same `--stream-shards`, or without `--stream`.
The held-out shards are copied whole, as without `--stream`.

### The batch is the same on any hardware

The recipe's `batch_size` is the number of windows in one update, over
every process together. Each process takes `--micro-batch` windows per
pass (by default its whole share) and adds the passes up until the batch
is complete. Which windows a step uses depends only on the recipe's `seed`
and the step, never on the number of processes: one GPU with two passes
and two GPUs with one pass each take the same steps. When a GPU runs out of
memory, lower `--micro-batch`, then add `--recompute`; the training stays
the same. The batch must split evenly: `batch_size` a multiple of
`--micro-batch` times the number of processes.

### Recomputing layers: long windows on a small GPU

`--recompute` (or `WPT_RECOMPUTE=1`) keeps only each layer's input during
a pass and runs the layer again during the backward pass, instead of
keeping everything the layer computed. The result is the same; memory
drops and each step takes about a third longer. Measured with a 101M model
(8 layers, width 512), one window's peak memory was 5.5 GiB at 4K tokens
without it, and with it 1.5 GiB at 4K, 2.7 GiB at 8K and 4.6 GiB at 16K.
The `tardigrade` and `axolotl` recipes, 32K-token windows on one 16 GB or
one 8 GB GPU, need it
with `--micro-batch 1`.

### Strategies: how the hardware is used

`--strategy` names a built-in or a Python file:

- `single` (default): one process on the best device of the machine;
- `ddp`: several processes, one per GPU. Each process holds a full copy of
  the model, and their gradients are averaged after each pass (PyTorch's
  `DistributedDataParallel`).

On one machine, `--processes` starts them:

```sh
uv run wpt-train <recipe> <dataset id> --strategy ddp --processes 2
```

On several machines, `torchrun` (which comes with PyTorch) starts them:
each machine runs the same command with its `--nnodes`, `--node-rank` and
`--master-addr` options. torchrun works on one machine too:

```sh
uv run torchrun --nproc-per-node 2 -m wpt_train <recipe> <dataset id> --strategy ddp
```

On Windows, torchrun does not start with PyTorch 2.14: the store its
processes meet through needs a network library (libuv) that PyTorch's
Windows build lacks. Use `--processes` there. PyTorch's processes on
Windows talk only through the CPU library (gloo), so several GPUs gain
less than on Linux.

`single` refuses to start among several processes, since each would train
alone and write the same folder. One process per
machine prepares the token files; only the main process logs and writes
the model folder.

A strategy file defines `start()`, which returns a `wpt_training.Strategy`:
the device, which process this is among how many, how the model is
wrapped, and how processes average a number, share a value and wait for
each other. A model too large for one GPU needs a strategy that splits the
model itself (PyTorch's FSDP, or a library such as DeepSpeed): it is
written as such a file, without changes to the training loop. None ships
yet.

### Methods: how the model learns

The recipe's `training.method` names a built-in or a Python file, and
`training.settings` changes its settings; those left out keep the method's
defaults, and an unknown setting or a value of the wrong type stops before
training.

- `muon`: how WPT models train (model.md, "Training"). Muon for matrix
  weights, per head for the query, key and value projections and per
  expert for routed experts; AdamW for the embedding, the output head,
  norms and mHC's biases and gating factors; Adam at `table_lr_factor`
  times the rate, without weight decay, for Engram's tables; QK-Clip at
  `qk_clip` on MLA layers (0: off). The learning rate warms up over
  `warmup` steps to `lr`, then falls to `first_factor` of it after
  `first_drop` of the steps and to `second_factor` after `second_drop`;
  the multi-token prediction loss weighs `mtp_weight`, then
  `mtp_late_weight` after `mtp_switch` of the steps. The folder keeps the
  weights of the lowest validation loss. Settings and defaults: `lr`
  0.002, `weight_decay` 0.1, `momentum` 0.95, `beta1` 0.9, `beta2` 0.95,
  `eps` 1e-8, `table_lr_factor` 5, `warmup` 2000, `first_drop` 0.8,
  `first_factor` 0.316, `second_drop` 0.9, `second_factor` 0.1,
  `mtp_weight` 0.3, `mtp_late_weight` 0.1, `mtp_switch` 0.676, `qk_clip` 100.
- `adamw`: AdamW at a constant learning rate, for every weight. Training stops
  early when the validation loss, smoothed over the last `smoothing`
  evaluations, has not improved by `min_improve` for `patience`
  evaluations; the smaller the tokens-per-parameter ratio, the smaller the
  improvement it asks for. Settings: `lr` 0.0003, `patience` 10,
  `min_improve` 0.01, `smoothing` 3, `chinchilla` 20 (the tokens per
  parameter at which `min_improve` applies in full).

A method file defines `SETTINGS` (names and defaults, whose types the
values take) and `start(model, settings, run)`, which returns a
`wpt_training.Method`: its optimizer, and optionally the loss of a batch
(by default the next-token loss plus the multi-token prediction loss at
`mtp_weight`), what happens just before each update (a learning-rate
schedule, gradient clipping) and just after it (QK-Clip), which evaluation
is the best and when to stop, and the state a resumed run needs.
Evaluations measure the next-token loss alone, at step 1, every
`eval_every` steps and at the last step. The model folder keeps the weights
of the evaluation the method judged best.

### Checkpoints and resuming

Every `--checkpoint-every` steps (default 500, 0 for never), the main
process writes a checkpoint: the step, the weights, the method's state and
the best weights so far. The same command with `--resume` continues from
it, on other hardware too: the number of processes, `--micro-batch` and
`--recompute` may change; the recipe and the other options may not (the
run stops with `SettingsMismatch`).
Without `--resume`, a leftover checkpoint stops the run, so a crashed run
is never overwritten by mistake. The checkpoint is removed once the model
is saved.

Where it lives:

- without `--run`, in `<model folder>/checkpoint/checkpoint.pt`, on the
  machine of the main process;
- with `--run NAME` (or `WPT_RUN`), in the store, under the ref
  `checkpoints/NAME`. Any machine using the same `WPT_STORE` continues the
  run with the same command, `--run NAME --resume` and any `--out`: a run
  can move from one machine to another between sessions. Each save
  replaces the previous checkpoint of that name, which is then deleted;
  a crash while saving leaves the previous one. With an S3 store, every
  save is an upload of about four times the model's size.

#### Stopping a run

Ctrl+C, `SIGTERM` (a shutdown, a rented machine being taken back) or, on
Windows, Ctrl+Break, sent to any process of the run, lets every process
finish its step; they then write a checkpoint (unless `--checkpoint-every
0`) and exit with code 130. `--resume` continues from that step. A second
signal stops the run at once, losing the steps since the last checkpoint,
as `kill -9` or a power cut does.

#### Restarting after a crash

When a process crashes, the processes `wpt-train` started are stopped and
all start again, up to `--max-restarts` times (default 3, `WPT_MAX_RESTARTS`);
a restarted process continues from the checkpoint without `--resume`. A
one-process run is started this way too, by a small parent process, unless
`--max-restarts 0`, which runs the training in the command's own process
and never restarts. Two exits are never restarted: a refusal (exit code 2:
a missing or damaged dataset, a checkpoint written with other settings,
options that do not fit), which would refuse again, and a stop asked for
(exit code 130). A bug that crashes every time stops after the last
restart.

With torchrun, its own `--max-restarts` restarts the processes, and they
continue from the checkpoint the same way. On several machines, use
torchrun's meeting point (`--rdzv-backend c10d --rdzv-endpoint HOST:PORT
--rdzv-id NAME`) instead of `--master-addr`, so a machine that comes back
can join again; and `--run`, so the checkpoint is where every machine can
read it. A checkpoint folder without `--run` is only on the main
process's machine. torchrun restarts after any failed process, refusals
included.

The model folder holds:

- `config.json`, the model's numbers;
- the weights (safetensors, optionally split with `--max-shard-size`); each
  file's header carries `license` (AGPL-3.0-or-later), `output_license`
  (CC0-1.0) and `notice`, which points to the folder's `NOTICE` and the
  project, and `tokenizer` (the tokenizer recipe and version, such as
  `classics version 1`) and `tokenizer_sha256`;
- `tokenizer.json`, the dataset's vocabulary, exactly the bytes those
  headers name: `wpt-chat`, `wpt-eval` and `wpt-posttrain` refuse another
  (see [tokenizer.md](tokenizer.md));
- `provenance.json`, what produced the model, including every source it
  learned from (address, revision, licenses and attribution, copied from the
  dataset record);
- `LICENSE`, the AGPL text; `NOTICE`, the licenses of the weights and of
  generated text, every source with the credit it asks for, and the output
  notice; `README.md`, the model card, with the licenses at the top;
- `evals/`, results written by `wpt-eval` (see [evaluation.md](evaluation.md)),
  shown in the model card. Training into the folder again removes them.

`NOTICE` and `README.md` are generated from the folder's own records, so they
cannot disagree with it.

`wpt-chat <model folder>` continues text typed by the user, until the model
ends the document. Its start line names the licenses and where the notice
is; `wpt-chat <model folder> --about` prints the folder's `NOTICE`. A pre-trained model only continues text. `wpt-chat <model folder> --chat`
talks to a model post-trained on conversations: it keeps the history,
writes it with the chat template ([chat-format.md](chat-format.md)), and
shows the model's thinking and tool calls with their tags, without running
the calls.

## What is exactly repeatable

**The data is exactly repeatable**, for audits: sources, cuts, patches,
vocabulary and tokens. Shards are compared by their token hash, which covers
only their rows, not the bytes of the Parquet file (a newer pyarrow may write
different bytes for the same tokens).

**Tool versions**: the repository pins a reference version of each tool
(`uv.lock` for Python packages, `.python-version`, and
`packages/wpt-tokenizer/rust-toolchain.toml` for Rust). Each build
records the versions it actually used, and reports where they differ from
the reference; it never refuses to run for that. The data hashes catch any
real change. Most upgrades change nothing; for example a newer Unicode
version could only affect newly added characters in steps that use Unicode
tables, and the tokenizer uses none.

**Training is not bit-exact**: GPU arithmetic differs between GPUs, drivers
and PyTorch versions, so the same inputs give equivalent but not identical
weights. The steps themselves are the same: each step's windows, and any
random draw a model makes, come from the seed and the step, and a resumed run continues the same
steps. Around it, everything possible is checked:

- before training, every input is checked against its hash: the dataset
  record, each shard (file sha256 and token hash), and the vocabulary;
- `provenance.json` records what produced the model: the dataset id and
  record, the sources it learned from, the vocabulary, every shard, the recipe's name and content, the command-line options, the strategy and
  method (file, its sha256, the method's settings), how the batch was split, the model shape and
  parameter count, the steps trained and the step whose weights the folder holds (which the
  model card shows), the repository commit (and whether the working tree had
  uncommitted changes), the device and the tool versions.
