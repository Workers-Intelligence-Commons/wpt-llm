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

The main process records the run in the store: what it read, every step's
loss, duration and learning rates, every evaluation and checkpoint, and
every file it wrote ([observability.md](observability.md)). A record the
store refuses stops the run with exit code 3, which is never restarted.
When the training process's record breaks (a record call refused, or on
S3 a group of its lines refused in the background), it finishes its step
unrecorded and every process stops after it, as for Ctrl+C: the
checkpoint is saved, and `--resume` continues from it without losing a
step. The broken record does not name that checkpoint (nor, with `--run`,
its store ref). When the launcher's record breaks while the processes
train (on S3, a group of its lines refused in the background), it stops
them as Ctrl+C does: they finish their step and save the checkpoint, and
the launcher exits with code 3. The loss printed at each evaluation is the step's loss over the
whole batch, every micro-batch of every process.

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
drops and each step takes about a third longer. Measured on 2026-10-17
with `watery-tardigrade` (102M parameters, 8 layers, width 512, CSA and
HCA, an indexer of 8 heads and top 64, Muon as DeepSeek-V4 runs it) on an
integrated Radeon 860M, where the Triton kernels compute the attention
(model.md, "Attention"), one window of random tokens with it:

| Tokens | First step's peak | Later steps' peak | Time of a later step |
| --- | --- | --- | --- |
| 4K | 1.48 GiB | 1.95 GiB | 9.7 s |
| 8K | 2.26 GiB | 2.75 GiB | 14.7 s |
| 16K | 4.09 GiB | 4.58 GiB | 27.1 s |
| 32K | 7.74 GiB | 8.24 GiB | 53.2 s |

Later steps add Muon's momentum, about 0.5 GiB. `unimpaired-axolotl`
(50M parameters) at 32K tokens peaked at 6.28 GiB on its first step and
6.57 GiB on the next, which took 29.0 s. With the PyTorch form of the
attention, the same day, the same runs peaked at 1.94, 2.77, 4.62 and
8.33 GiB (9.2, 16.9, 25.2 and 47.1 s), and `unimpaired-axolotl` at
6.62 GiB (25.2 s): on this GPU the kernels save under 0.1 GiB and are
not faster in training, since their backward pass adds each entry's
gradients by atomic additions, which this GPU runs slowly (about 0.7 s of
the 1.6 s the backward kernel takes on 32K queries of 8 heads and 192
entries each). Without gradients, at 8K tokens, a CSA layer's attention
took 15% less time with them and an HCA layer's half. These times were measured with
`HIP_LAUNCH_BLOCKING=1`, which makes the CPU wait for each GPU kernel:
without it, on this GPU (gfx1152, which ROCm supports only through
`HSA_OVERRIDE_GFX_VERSION`), Muon's 10 Newton-Schulz steps on the
32-expert recipes crash the process (a segmentation fault in the GPU
stack) in the first update, while 8 steps do not; the memory is the same
either way, and the times are an upper bound. With the 4.6M-parameter
`classics-test` the crash does not occur. Before the indexer was scaled
and Muon changed, the same model peaked at 1.51, 2.28 and 4.13 GiB on its
first step at 4K, 8K and 16K tokens (7.4, 14.3 and 34.5 s a step, without
`HIP_LAUNCH_BLOCKING`); with the gated MLA it had before 2026-10-17, at
1.73, 2.74 and 5.06 GiB (7.8, 18.9 and 58.2 s); on another GPU it had
peaked at 5.5 GiB at 4K tokens without `--recompute`. On the Radeon 860M,
a 4K window without `--recompute` crashes the process (a segmentation
fault in the GPU stack) with either attention. The `watery-tardigrade` and
`unimpaired-axolotl` recipes, 32K-token windows on one 16 GB or one 8 GB
GPU, need it with `--micro-batch 1`.

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
  weights as DeepSeek-V4 runs it (8 then 2 Newton-Schulz steps, updates of
  RMS 0.18), per head for the query, key and value projections, per expert
  for routed experts and per output group or merging projection in CSA
  and HCA; AdamW for the embedding, the output head, norms, mHC's biases
  and gating factors, attention sinks and merging biases; Adam at
  `table_lr_factor` times the rate, without weight decay, for Engram's
  tables; no QK-Clip (DeepSeek-V4). The learning rate warms up over
  `warmup` steps to `lr`, then falls to `first_factor` of it after
  `first_drop` of the steps and to `second_factor` after `second_drop`;
  the multi-token prediction loss weighs `mtp_weight`, then
  `mtp_late_weight` after `mtp_switch` of the steps. The folder keeps the
  weights of the lowest validation loss. Settings and defaults: `lr`
  0.002, `weight_decay` 0.1, `momentum` 0.95, `beta1` 0.9, `beta2` 0.95,
  `eps` 1e-8, `table_lr_factor` 5, `warmup` 2000, `first_drop` 0.8,
  `first_factor` 0.316, `second_drop` 0.9, `second_factor` 0.1,
  `mtp_weight` 0.3, `mtp_late_weight` 0.1, `mtp_switch` 0.676.
- `adamw`: AdamW at a constant learning rate, for every weight. Training stops
  early when the validation loss, smoothed over the last `smoothing`
  evaluations, has not improved by `min_improve` for `patience`
  evaluations; the smaller the tokens-per-parameter ratio, the smaller the
  improvement it asks for. Settings: `lr` 0.0003, `patience` 10,
  `min_improve` 0.01, `smoothing` 3, `chinchilla` 20 (the tokens per
  parameter at which `min_improve` applies in full).

A method file defines `SETTINGS` (names and defaults, whose types the
values take) and `start(model, settings, run)`, which returns a
`wpt_training.Method`: its optimizer, and optionally the losses of a batch,
`loss(model, x, y)` returning (training loss, indexer loss): by default the
next-token loss plus the multi-token prediction loss at `mtp_weight`, plus
the gradient, not the value, of the CSA indexers' loss, which the model
returns when asked, `model(x, next_tokens=y, indexer_loss=True)` giving
(scores, multi-token prediction scores, indexer loss); a method that does
not ask leaves the indexers untrained. The log shows both losses at each
evaluation (`train loss`, then `indexer loss`, each of the step's last
micro-batch), and `provenance.json` records the indexer loss of the step
whose weights the folder holds. Then, optionally, what happens
just before each update (a learning-rate schedule, gradient clipping) and
just after it, which evaluation
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
  a crash while saving leaves the previous one. The ref's change, every
  deletion, and the last checkpoint's once the model is saved, are
  recorded with the object's sha256, and that line is in the store before
  the step: a deletion whose record breaks leaves the object in the
  store. A save once the record broke still sets the ref, unrecorded, and
  leaves the previous object beside the new one. The last checkpoint's
  deletion is recorded first too, so a record that breaks there ends a
  run whose model is saved with exit code 3. With an S3 store, every save
  is an upload of about four times the model's size.

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
and never restarts. Three exits are never restarted: a refusal (exit code
2: a missing or damaged dataset, a checkpoint written with other settings,
options that do not fit), which would refuse again, a record that could
not be written (exit code 3), and a stop asked for (exit code 130). A bug
that crashes every time stops after the last restart.

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
  shown in the model card.

The main process writes the folder beside its target, as a hidden folder
named `.<folder name>-<random>`, records the sha256 of every file in it,
then renames it into place. A run that stops before (a crash, a refused
record) leaves the target as it was. Training into a folder that already
exists still works, but replaces it as a whole: everything in it goes, not
only the earlier weights and the `evals/` results, which describe other
weights, but also its `checkpoint/` and any file put there by hand. Keep
such files elsewhere. The checkpoint's removal is recorded before it is
removed, and the record is in the store before the folder is renamed. The
earlier folder is first moved aside, then removed once the new one is in
place. When the new one cannot take its place, or the run is interrupted
meanwhile, the earlier folder is moved back; when that fails too, the run
stops with `RestoreFailed`, which names the earlier folder
(`.<folder name>-replaced-<random>`) and the new one, both kept: move one
of them into place by hand. Whenever the rename was tried and the target
is missing afterward, whatever error stopped the run (a first rename
refused, or a record that broke meanwhile), the new model is kept and
named on the error output, with the earlier folder when there is one.
When the earlier folder cannot be removed, or
its removal is interrupted, it stays beside the new one, as
`.<folder name>-replaced-<random>`, named in a warning and in the record:
remove it by hand. The run still succeeds, unless it was interrupted.

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
weights. On a GPU, the attention kernels also add up the entries' gradients
in the order the GPU's atomic additions take, so two runs on one GPU can
differ slightly too (docs/model.md, "Attention"); `wpt-train` logs whether
TileLang, Triton or PyTorch computes the attention. The steps themselves are the same: each step's windows, and any
random draw a model makes, come from the seed and the step, and a resumed run continues the same
steps. Around it, everything possible is checked:

- before training, every input is checked against its hash: the dataset
  record, each shard (file sha256 and token hash), and the vocabulary;
- `provenance.json` records what produced the model: the dataset id and
  record, the sources it learned from, the vocabulary, every shard, the recipe's name and content, the command-line options, the strategy and
  method (file, its sha256, the method's settings), how the batch was split, the model shape and
  parameter count, the steps trained and the step whose weights the folder holds (which the
  model card shows) and the CSA indexers' loss at that step, the repository commit (and whether the working tree had
  uncommitted changes), the device and the tool versions.
