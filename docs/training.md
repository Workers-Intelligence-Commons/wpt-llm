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
seed, the steps ("Steps", below) with their settings, and the drafter. The
command-line options say only how the hardware runs it; each also comes
from an environment variable (`WPT_` + the option name in capitals,
`--micro-batch` -> `WPT_MICRO_BATCH`; the recipe and the dataset id from
`WPT_RECIPE` and `WPT_DATASET`). `uv run wpt-train --help` lists them.

The dataset holds everything the steps read: the text's shards, and the
conversations and tasks its build found ([data-selection.md](data-selection.md)).
Its record gives a fingerprint (a sha256) for each of these parts, `text`,
`conversations` and `tasks`, which the steps' fingerprints read ("Step
results and reuse", below). A source added after the build reaches training
only once it is downloaded into the store (`wpt-dataset download`) and the
dataset is built again.

`wpt-train` trains through gpu workers, one per machine
([machines.md](machines.md)): each worker's job runs the steps on its GPUs, and `wpt-train` answers it from
the store, the tokenizer and the run's record. The job's main process
records the run in the store, through that link: what it read, every step
and every update's loss, duration and learning rates, every evaluation and
checkpoint, every step result kept or reused, and every file it wrote
([observability.md](observability.md)). A record the store refuses stops
the run with exit code 3, which is never restarted. When the training
process's record breaks (a record call refused, or on S3 a group of its
lines refused in the background), it finishes its update unrecorded and
every process stops after it, as for Ctrl+C: the checkpoint is saved, and
`--resume` continues from it without losing an update. The broken record
does not name that checkpoint (nor, with `--run`, its store ref). When
`wpt-train`'s own record breaks while the job trains (on S3, a group of
its lines refused in the background), it stops the job as Ctrl+C does: its
processes finish their update and save the checkpoint, and `wpt-train`
exits with code 3.

## Steps

A recipe's `steps` list says what is done to the model, in order, on one
model: each step starts from the weights the one before it left. The recipe
has one `seed`; each step mixes its number in the list into it, so two
identical steps in one recipe draw different windows, conversations and
tasks.

```json
"seed": 1337,
"steps": [
  {"step": "read-text", "optimizer": {"name": "muon", "settings": {"lr": 0.002}},
   "settings": {"stages": [{"block_size": 4096, "steps": 2000, "batch_size": 256}], "eval_every": 200,
                "eval_batches": 20},
   "locked": []},
  {"step": "answer-as-assistant"},
  {"step": "finish-tasks", "settings": {"rounds": 4}}
]
```

A step is made of four roles, each a Python file of
`packages/wpt-training/src/wpt_training/`:

- the **method** (`methods/`) makes a loss from a batch of examples;
- the **environment** (`environments/`) gives the examples, batch by
  batch, and says when to evaluate and when the step has failed;
- the **optimizer** (`optimizers/`) moves the weights the loss reaches,
  with its schedule, and judges the evaluations;
- the **step** file (`steps/`) names its method, environment and
  optimizer with that optimizer's settings, the parts of the model it
  locks, and its own defaults for its method's and environment's settings.

The **runner** (`wpt_training.train`) is the training loop every step
shares: it freezes the locked parts, starts the optimizer, asks the
environment for each update's batch, lets the method learn from it,
updates, evaluates, writes checkpoints, and stops on a signal.

An entry is `{"step": NAME}` with, optionally, `optimizer` (`{"name",
"settings"}`), `settings` and `locked`; each part left out takes the step
file's own. `NAME` is a built-in step or the path of a Python file. The
settings are the defaults of the step's method and environment, then the
step file's, then the entry's, each of its default's type (an integer
where a number is expected is taken, text is converted;
`wpt_utils.settings.typed`). A setting belongs to the method or to the
environment by its name, never to both. The optimizer the entry names
takes the step file's settings for it when it is the step's own, else its
own defaults, then the entry's. Every step file is read and every setting
checked before any step runs: an unknown step or setting, a value of
another type, settings that cannot go together, a method that reads
another kind of examples than its environment gives, and an unknown part
to lock all refuse the run (exit code 2), naming the file. So do drafter
settings that cannot be. Once the dataset is read, and still before any
step runs, a step whose environment finds nothing to learn from in it
(no conversation or no task of its tag) refuses the run the same way,
naming the step. `provenance.json` records each step: its files and their
sha256, its method, environment, optimizer and their settings, the kind
of examples it reads (`reads`), and its locked parts.

The built-in steps:

| step | method | environment | optimizer | locked |
| --- | --- | --- | --- | --- |
| `read-text` | `next-token` | `text-windows` | `muon` | none |
| `answer-as-assistant` | `assistant-tokens` | `conversations` | `adamw`, lr 1e-4, no weight decay | `balancing` |
| `finish-tasks` | `grpo` | `agent-tasks` | `adamw`, lr 1e-6, no weight decay | `routers`, `balancing` |
| `finish-tasks-with-less-writing` | `grpo` | `agent-tasks` | `adamw`, lr 1e-6, no weight decay | `routers`, `balancing` |

A recipe whose steps are `read-text` alone makes a base model. The model's
context (`config.json`'s `block_size`) is the last window length of the
last step that reads text windows; a recipe with no such step is refused.

### Locked parts

`locked` names parts of the model by the names the model gives them
([model.md](model.md), "Parts"); a name the model does not give is refused,
naming the parts there are. A locked part's weights do not change during
the step: they are left out of its optimizer and of the processes'
gradient exchange. `balancing` is the routers' balancing biases, which no
gradient trains: locked, they stay as they are after each update. A
method also names the parts its loss never reaches (`Method.unreached`:
the multi-token prediction block for `assistant-tokens` and `grpo`), which
are left out the same way. A step whose locked and unreached parts leave
nothing to train is refused.

### Step results and reuse

After each step, the store keeps its result under the ref
`steps/<fingerprint>`: a JSON object holding what the run recorded of the
step and the object of its state, the weights (and those of its best
evaluation) with the optimizer's, method's and environment's state: the
step's last checkpoint, as it is. The fingerprint covers the fingerprint
of the step before it (for the first step, the model's shape and
vocabulary size, without the context `block_size`, which no weight
depends on: a recipe that adds a later step reading longer windows still
reuses the first), the step as `provenance.json` records it, the
fingerprint of the dataset part it reads (the record's `fingerprints`: `text`
for `read-text`, `conversations` for `answer-as-assistant`, `tasks` for
`finish-tasks` and `finish-tasks-with-less-writing`), the seed and the
code (the commit it runs from and a hash of its uncommitted changes;
[architecture.md](architecture.md), "Fingerprints and the store"), so any
commit makes every kept result unknown. A change to the
conversations, or to a source that holds only conversations, leaves the
fingerprint of `read-text` as it was, and so its kept result stands. The
options, which only say how the hardware runs it, stay out, `--stream` and
`--stream-shards` included.

A run whose first steps have a kept result starts from the result of the
last of them and runs the steps after it. So a recipe that shares its
first steps with an earlier run reuses them, and a run that failed after
some steps continues from the first step not kept. The main process reads
the result and sends its weights to every process. Reuse means the same
inputs, not the same bits ("What is exactly repeatable", below). Each step
starts its own optimizer. `provenance.json` says, for each step, its
fingerprint, its result and whether this run reused it.

A step that cannot do its part, such as one whose loss is not finite or
whose attempts Docker drops every one, fails (`wpt_training.StepFailed`):
the run stops with exit code 2 and the reason, the step keeps no result,
and the results of the steps before it stay in the store. A checkpoint
the step wrote before it failed stays, and the message names it: the
same command with `--resume` continues the step from it, and deleting it
starts the step over (a run with other settings for that step needs it
deleted, "Checkpoints and resuming", below).

### Methods

A method file defines `READS`, the kind of examples it learns from,
`SETTINGS` and `start(settings)`, which returns a `wpt_training.Method`.
Its `learn` runs one pass's forward and backward passes; the runner adds
the passes of an update up, and `update_loss` gives the update's loss and
indexer loss, the same in every process. `begin` may refuse options the
method cannot train with, and `unreached` names the parts its loss never
reaches.

- `next-token` (reads `windows`): the next-token loss of every position,
  plus the multi-token prediction loss at `mtp_weight`, then at
  `mtp_late_weight` after `mtp_switch` of the step's updates (DeepSeek-V3:
  after 10T of 14.8T tokens), plus the gradient, not the value, of the CSA
  indexers' loss, which reaches only the indexers. Evaluations compare the
  next-token loss alone. Settings: `mtp_weight` 0.3, `mtp_late_weight`
  0.1, `mtp_switch` 0.676.
- `assistant-tokens` (reads `conversations`): the next-token loss of the
  tokens the assistant wrote only ("The answer-as-assistant step", below).
- `grpo` (reads `attempts`): group relative policy optimization on scored
  attempts ("The finish-tasks step", below).

A method on text windows is a `wpt_training.TextMethod`: its
`loss(model, x, y)` returns (training loss, indexer loss), the model
giving the indexer loss when asked (`model(x, next_tokens=y,
indexer_loss=True)` returns the scores, the multi-token prediction scores
and that loss); a method that does not ask leaves the indexers untrained.
A method that defines its own `loss` defines `part_loss(model, x, y,
following, window)` too, the share of a part of the windows (`following`:
the token after `y`'s last, or None at the windows' end; `window`: the
windows' length) in what `loss` gives over whole windows, or it trains
whole windows only (the run refuses `--segment` and `split` with it).
`--strategy split` is refused for a recipe with a step that reads
anything else than text windows.

### Environments

An environment file defines `GIVES`, the kind of examples it gives,
`SETTINGS`, `start(settings, data)`, which returns a
`wpt_training.Environment`, and optionally `check(settings)`, which raises
`ValueError` on settings that cannot go together, `context(settings)`,
the window length the step leaves the model reading at, and
`check_dataset(settings, record, store)`, which raises `ValueError` when
the dataset holds nothing for the step, before any step runs. `data` gives
the step's name, the store, the dataset's record, the tokenizer, the
model's configuration, the strategy, the run's record, the log, the seed
and the step's number in the recipe. The
environment gives its stages (one for most environments; text windows
have one per window length), the tokens of its examples, each update's
passes (or none, when the update has nothing to learn: it is then
skipped), when to evaluate, what happens after each update and at the
step's end, the state a resumed step needs, and the sources it read, each
as the dataset record's `sources` describes it, which the model's
`NOTICE` credits.

- `text-windows` gives `windows` of the dataset's text ("The read-text
  step", below).
- `conversations` gives the dataset's conversations of one tag ("The
  answer-as-assistant step", below).
- `agent-tasks` gives the model's own attempts at the dataset's tasks of
  one tag, scored ("The finish-tasks step", below).

### Optimizers

An optimizer file defines `SETTINGS` and `start(model, settings, run)`,
which returns a `wpt_training.Optimizer`: its torch optimizer over the
parameters the step trains, and optionally what happens just before each
update (a learning-rate schedule, gradient clipping) and just after it, at
the start of each stage after the first (`stage`), which evaluation is the
best and when to stop, and the state a resumed step needs. `run` gives the
tokens of the step's examples, the parameters it trains, its number of
updates and the log. The step keeps the weights of the evaluation its
optimizer judged best in its last stage, or its last weights when it has
no evaluation.

- `muon`: how WPT models read text (model.md, "Training"). Muon for matrix
  weights as DeepSeek-V4 runs it (8 then 2 Newton-Schulz steps, updates of
  RMS 0.18), per head for the query, key and value projections, per expert
  for routed experts and per output group or merging projection in CSA
  and HCA; AdamW for the embedding, the output head, norms, mHC's biases
  and gating factors, attention sinks and merging biases; Adam at
  `table_lr_factor` times the rate, without weight decay, for Engram's
  tables; no QK-Clip (DeepSeek-V4). The learning rate warms up over
  `warmup` updates to `lr`, then falls to `first_factor` of it after
  `first_drop` of the step's updates and to `second_factor` after
  `second_drop`. Settings and defaults: `lr` 0.002, `weight_decay` 0.1,
  `momentum` 0.95, `beta1` 0.9, `beta2` 0.95, `eps` 1e-8,
  `table_lr_factor` 5, `warmup` 2000, `first_drop` 0.8, `first_factor`
  0.316, `second_drop` 0.9, `second_factor` 0.1. It needs a model that
  names its parameters' roles.
- `adamw`: AdamW at a constant learning rate. It stops a stage early when
  the validation loss, smoothed over the last `smoothing` evaluations, has
  not improved by `min_improve` for `patience` evaluations; the smaller the
  tokens-per-parameter ratio, the smaller the improvement it asks for.
  Settings: `lr` 0.0003, `weight_decay` 0.01 (PyTorch's own), `patience`
  10, `min_improve` 0.01, `smoothing` 3, `chinchilla` 20 (the tokens per
  parameter at which `min_improve` applies in full).

### Writing a step

A step file defines `METHOD`, `ENVIRONMENT` and `OPTIMIZER` (`{"name":
..., "settings": {...}}`), and optionally `LOCKED` and `SETTINGS`, its
defaults for its method's and environment's settings, and
`check(settings)`. It names built-in roles or paths of Python files, so a
new kind of training is a new method, environment or optimizer file and
a step file that puts them together, without changes to the runner.

## The read-text step

The `read-text` step reads the dataset's text: next-token prediction
(`next-token`) on windows of the training shards (`text-windows`), the
windows growing stage by stage, under Muon. Its settings are `stages`,
`eval_every` (200) and `eval_batches` (20), with the method's. Each update
draws its windows at random places of the training tokens, seeded by the
seed, the step's number in the recipe and the update only: a second
`read-text` step draws other windows than the first. Dropout reads the
same three.

Validation uses the dataset's held-out shards. The tokens are read from
disk as training needs them, so a dataset larger than memory trains:
before the step's first update, the shards are checked against their
token hashes and their tokens written once into one file per part, in the
store's `scratch/` folder (`<store folder>/scratch/` for a local store,
the cache for S3). A later run reuses that file when it still matches the
sha256 recorded when it was written. `scratch/` can be deleted at any time.

### Streaming: a dataset larger than the disk

`--stream` (or `WPT_STREAM=1`) never copies every shard to disk. The
shards are walked in order at the speed training uses tokens: before each
update, the shard holding the next unused token and the shards after it,
`--stream-shards` in all (8 by default, about 2 GB), are the ones in use,
and the update draws its windows from them only. The shard after them is
prepared while training goes on, and a shard that leaves the set loses
its token file and, from an S3 store, its local copy. After the last shard
comes the first again.

Which shards are in use depends only on the update, so a resumed run and
any number of processes take the same windows, but they are not the
windows a run without `--stream` takes: a streamed run mixes documents
within the shards in use (the build already put the rows of every shard in
a pseudo-random order), not over the whole dataset. Both options only say
how the hardware reads the shards, so they are no part of the step's
fingerprint: a run reuses a step another run kept with other values of
them. The held-out shards are copied whole, as without `--stream`.

### The batch is the same on any hardware

A step's batch is the number of examples in one update, over every process
together: for text windows, each stage's `batch_size` ("Stages", below).
Each process takes `--micro-batch` examples per pass (by default its whole
share of the batch) and adds the passes up until the batch is complete.
Which windows an update uses depends only on the recipe's `seed`, the
step's number in the recipe and the update, never on the number of
processes: one GPU with two passes and two
GPUs with one pass each make the same updates. When a GPU runs out of
memory, lower `--micro-batch`, then add `--recompute`; the training stays
the same. Every stage's batch must split evenly: its `batch_size` a
multiple of `--micro-batch` times the number of processes (with
`--strategy split`, every process takes every window, so a multiple of
`--micro-batch`).

### Stages

Training targets 1M tokens and reaches it by stages of the `read-text`
step; the window lengths follow DeepSeek-V4 (section 4.2.2: 4K, then 16K,
64K and 1M tokens); V4 states no tokens or updates per stage, so each
recipe sets them. The step's `stages` setting lists them in order, each
with its window length, its length in updates and its batch:

```json
"stages": [
  {"block_size": 4096, "steps": 2000, "batch_size": 256},
  {"block_size": 16384, "steps": 500, "batch_size": 64},
  {"block_size": 65536, "steps": 200, "batch_size": 16},
  {"block_size": 1048576, "steps": 50, "batch_size": 1}
]
```

(an example of the shape, not a recipe's numbers). The stages are one
step: their updates are counted together, update 1 being the first
stage's first, and the learning-rate schedule and the multi-token
prediction weight's switch (model.md, "Training") run over them all. Each
stage draws batches of its own `batch_size` windows of its own length, so
a stage puts `batch_size` times `block_size` tokens in each update. A
stage's `batch_size` is global, over every process together ("The batch is
the same on any hardware", above), and its evaluations average
`eval_batches` batches of its own size.

Each stage is evaluated at its first and last updates and every
`eval_every` updates, on windows of its length: a longer window predicts
better, so losses are compared within a stage only. When a stage after
the first starts, the optimizer hears of it (`Optimizer.stage`) and the
best evaluation so far is forgotten: `muon` and `adamw` start counting
again, and the step keeps the best weights of its last stage. An
optimizer's early stop (`adamw`'s patience) ends its stage, and the next
stage starts. A held-out part shorter than a stage's window measures that
stage's validation loss on the training text, which the log says.

Checkpoints and `--resume` cross stage boundaries: the checkpoint holds
the update to continue from, and a resumed step starts within its stage.
Each stage is a span of the run's record (`training.stage`, with its
number, window length, first and last updates, and the update a resumed
run started at), each update's event names its window length, and
`provenance.json` lists, for the step, the stages with the last update
each made (`stages`). The recipes keep one stage (model.md, "Model
recipes").

DeepSeek-V4 trains dense attention first and turns sparse attention on at
64K tokens; WPT is sparse from the first update, at every stage (model.md,
"Attention").

The log shows, at each evaluation, the update's loss over the whole batch,
every micro-batch of every process, then the indexer loss, then the
validation loss; `provenance.json` records the indexer loss of the update
whose weights the step leaves.

## Recomputing layers: long windows on a small GPU

`--recompute` (or `WPT_RECOMPUTE=1`) keeps only each layer's input during a
pass and runs the layer again during the backward pass, instead of keeping
everything the layer computed. The result is the same; memory drops and
each update takes about a third longer. Measured with `watery-tardigrade`
(102M parameters, 8 layers, width 512, CSA and HCA, an indexer of 8 heads
and top 64, Muon as DeepSeek-V4 runs it) on an integrated Radeon 860M,
where the Triton kernels compute the attention (model.md, "Attention"), one
window of random tokens with it:

| Tokens | First update's peak | Later updates' peak | Time of a later update |
| --- | --- | --- | --- |
| 4K | 1.48 GiB | 1.95 GiB | 9.7 s |
| 8K | 2.26 GiB | 2.75 GiB | 14.7 s |
| 16K | 4.09 GiB | 4.58 GiB | 27.1 s |
| 32K | 7.74 GiB | 8.24 GiB | 53.2 s |

Later updates add Muon's momentum, about 0.5 GiB. `unimpaired-axolotl` (50M
parameters) at 32K tokens peaked at 6.28 GiB on its first update and 6.57 GiB
on the next, which took 29.0 s. With the PyTorch form of the attention, the
same runs peaked at 1.94, 2.77, 4.62 and 8.33 GiB (9.2, 16.9, 25.2 and 47.1
s), and `unimpaired-axolotl` at 6.62 GiB (25.2 s): on this GPU the kernels
save under 0.1 GiB and are not faster in training, since their backward
pass adds each entry's gradients by atomic additions, which this GPU runs
slowly (about 0.7 s of the 1.6 s the backward kernel takes on 32K queries
of 8 heads and 192 entries each). Without gradients, at 8K tokens, a CSA
layer's attention took 15% less time with them and an HCA layer's half.
These times were measured with `HIP_LAUNCH_BLOCKING=1`, which makes the CPU
wait for each GPU kernel: without it, on this GPU (gfx1152, which ROCm
supports only through `HSA_OVERRIDE_GFX_VERSION`), Muon's 10 Newton-Schulz
steps on the 32-expert recipes crash the process (a segmentation fault in
the GPU stack) in the first update, while 8 steps do not; the memory is the
same either way, and the times are an upper bound. With the small
`classics-test` the crash does not occur. On the Radeon 860M, a 4K window
without `--recompute` crashes the process (a segmentation fault in the GPU
stack). The `watery-tardigrade` and `unimpaired-axolotl` recipes, 32K-token
windows on one 16 GB or one 8 GB GPU, need it with `--micro-batch 1`.

## Long windows: segments and slices

A window too long for one GPU's memory, even with `--recompute`, runs in
parts, which give exactly the loss and gradients of the whole window at
once (model.md, "Long windows"):

- `--segment N` (or `WPT_SEGMENT`), on one GPU: a window longer than `N`
  tokens runs as consecutive segments of `N` (the last may be shorter). A
  first pass without gradients runs every segment but the last, each
  handing the next what its layers need of the earlier tokens; then the
  segments run again from the last to the first with gradients, each
  starting from what the one before handed on, and its backward pass adds
  the gradient the later segments sent back to what it handed on. Memory
  holds one segment's activations plus what the segments hand on, which
  grows with the window (below); every segment but the last runs twice,
  up to twice the time (15% more than whole with `--recompute`, below). `N` is a multiple of 128. Windows of `N` tokens or fewer run
  whole.
- `--strategy split`, on several processes (GPUs of one or several
  machines, started as for `ddp`): every process takes the same windows,
  and process `r` trains the `r`-th of equal, consecutive slices of each
  (DeepSeek-V4's contextual parallelism); the processes exchange what
  crosses the slices, and their gradients are added up before each update.
  Each stage's window length must split into slices of a multiple of 128
  tokens.
- Both, `--strategy split` with `--segment N`, when a slice is longer
  than `N`: every process trains its slice in segments of `N` tokens. The
  window runs as consecutive parts of `N` tokens per process (`N` times
  the number of processes; the last part may be shorter), each part split
  across the processes as above, so process `r` trains the `r`-th segment
  of every part, and each part hands the next what segments hand on, with
  the same two passes. Memory holds one segment's activations per
  process; the result stays exact. Windows whose slices are `N` tokens or
  fewer are split without segments.

These are options of the hardware, not of the recipe: a run may resume with
another `--segment`, or none. The training loss and the indexer loss are
the parts' shares of the whole window's (`TextMethod.part_loss`, "Methods", above); the
validation loss in parts is each part's validation loss weighted by its
share of the window's tokens. The tests check, on the CPU, that 448 tokens
in segments of 128 (three boundaries, the last segment shorter), 512 tokens
split over 2 and 4 processes (gloo), and 768 tokens split over 2 processes
in segments of 128 (three parts) give the whole window's losses, gradients,
validation loss and expert balancing, each with the layers' activations
kept and with `--recompute`, for a model with every layer kind (KDA, CSA
with its indexer choosing, HCA, the windows, Engram, the experts, mHC, the
MTP block). Split windows are checked in FP32 to `2e-4` of the largest
gradient, against `1e-6` for segments, since KDA's composed transforms
round differently (model.md, "Long windows"; the split tests keep that
tolerance in FP32). A run of `wpt-train` with `--strategy split --processes
2 --segment 128` trains the same model as one process on whole windows.

Measured on a Radeon 860M (an integrated AMD GPU sharing the machine's
memory; `HIP_LAUNCH_BLOCKING=1`, which its 32-expert recipes need), with
watery-tardigrade's model, Muon, `--recompute`, one window of random
tokens, the second update of each:

| window | parts | peak memory | time of an update |
| --- | --- | --- | --- |
| 32,768 | whole | 8.24 GiB | 53 s |
| 32,768 | segments of 8,192 | 7.46 GiB | 61 s |
| 65,536 | segments of 8,192 | 8.43 GiB | 121 s |
| 131,072 | segments of 8,192 | 10.39 GiB | 289 s |

Most of that memory is the weights and the optimizer's state, the same at
every length; what grows with the window, about 31 KiB per token, is what
the segments hand on (the merged entries and each segment's starting
state, kept for the backward pass). The first update's loss is the same
whole or in segments (11.922). Not measured: 1M-token windows (about 37
GiB by that growth, an estimate), the RTX 5080, `split` on GPUs (NCCL,
rented NVIDIA machines, DeepSeek's TileLang kernels), and real runs on
real data; `split` was only run on the CPU, by the tests.

## Comparing routing in tests

Two passes over the same tokens that compute in another order (whole or
in segments, other CPU instructions, other thread counts) give router
scores that differ by rounding: on x86 CPUs, whole and in segments, up to
6e-8 at the first routed layer and 7.6e-6 at the last, the FP32 rounding
of every earlier layer added up. A choice whose two deciding values lie
within that rounding of each other can then go either way: a token whose
k-th and (k+1)-th biased scores nearly tie can go to other experts, and a
token whose `s_j - alpha` lies on the edge between two of balancing's
4096 bins (model.md, "Experts") can be counted in the next bin. Every
token's (k+1)-th expert has an offset of 0, on an edge, by construction.

So a test that compares routing or balancing counts between two such passes
keeps them exact for every token whose choice is clear, and lets only near
ties differ. A token is near a tie when the two values that decide a choice
are no farther apart than the two passes moved them. The bound on those
moves is the one the test asserts: the scores each pass counted agree
within ROUNDING, 2^-14 (about 6.1e-5, the `routing` fixture in
`conftest.py`). That score bound is what limits the differences to near
ties. The other checks document it and do not enforce it: they test, for
the scores at hand, that the experts differ only where the k-th and
(k+1)-th biased scores lie within two such moves of each other (plus 4
units of FP32 rounding, ARITHMETIC, for the comparison's own arithmetic),
that each pass's counts are exactly the bins of the scores it counted, and
that a count moves only where its `s_j - alpha` lies that close to a bin's
edge, by one bin at most (`assert_routed_alike` in
`packages/wpt-training/tests/test_parts.py`; split windows compare each
process's counts with those of its tokens in the whole window, and the sum
of the processes' counts with the whole window's: the processes' scores,
put back in the window's order, are compared with the whole window's under
the same bounds, so a sum differs from the whole window's counts only where
a token's bin may differ). A test of that comparison builds scores one unit
of rounding apart, with an exact edge and a near tie, so the near-tie path
runs the same on any processor.

A test that reads text through the generation cache and compares the result
with the whole text (`test_cache.py`, `test_sampling.py`) records the
cache's routes, reads the text whole recording its own
(`routing.first_difference`), and accepts a position whose routes differ
only when the whole pass's k-th and (k+1)-th biased scores lie within two
ROUNDING plus ARITHMETIC of each other. It then compares scores or tokens
exactly before the first position that differs, and in full when none does.
Two passes that run the same operations on the same shapes, such as
`--recompute`'s (`test_model.py`), give the same scores bit for bit, and
their counts are compared exactly. An exact comparison of balancing counts
in segments fails on GitHub's runners; the same failure shows on an x86
machine with `ATEN_CPU_CAPABILITY=avx512 OMP_NUM_THREADS=2` (or `avx2`),
where one to six tokens' counts in a layer move to the next bin and no
token goes to other experts.

A test that trains in segments and compares the gradients with the whole
window's (`assert_same` in `test_parts.py`) accepts a difference of 2e-5 of
the largest gradient, beyond a relative 1e-3. The difference is the
rounding of each math library path: on GitHub's runners, up to 8.3e-7 on
AMD EPYC processors and 8.2e-6 on Intel Xeon ones, which their math
libraries compute by other routes; on AMD, 4.4e-6 with oneDNN switched off
and 4.3e-5 with `MKL_CBWR=COMPATIBLE`. A limit of 1e-6 fails on every Intel
runner.

The drafter test (`test_sampling.py`) checks only that training works: the
losses of a few steps are finite, every weight they reach changes, and the
writer verifies the trained drafter's drafts. It does not ask that they be
kept more often than uniform drafts: the test's model has random weights,
so its next-token distribution is nearly uniform (entropy 3.75 of at most
3.91), uniform drafts are already kept at about 0.78 per position, and over
6 seeds a drafter trained for 40 or 200 steps was kept no more often than
uniform drafts on average (+0.0003), and less often after 400. With one
seed the margin went from -0.031 on Intel runners to +0.075 on AMD ones.

## Strategies: how the hardware is used

`--strategy` names a built-in or a Python file:

- `single` (default): one process on the best device of the machine;
- `ddp`: several processes, one per GPU. Each process holds a full copy of
  the model, and their gradients are averaged after each pass (PyTorch's
  `DistributedDataParallel`).
- `split`: several processes, one per GPU, that split each window between
  them ("Long windows", above). Each holds a full copy of the model, and
  their gradients are added up before each update.

Every strategy runs inside the gpu worker ([machines.md](machines.md)),
whose job is the training's processes; `--processes` starts them on the
worker's machine:

```sh
uv run wpt-train <recipe> <dataset id> --strategy ddp --processes 2
```

(`split` starts the same way.) A run on several gpu workers starts
`--processes` processes on each machine, and they train as one run
([machines.md](machines.md), "Several machines"); across places, only
`ddp` runs. PyTorch's processes on Windows talk only through the CPU
library (gloo), so several GPUs gain less than on Linux.

`single` refuses to start among several processes, since each would train
alone and write the same folder. One process per
machine prepares the token files; only the main process logs and writes
the model folder.

A strategy file defines `start()`, which returns a `wpt_training.Strategy`:
the device, which process this is among how many, how the model is
wrapped, and how processes average a number, share a value and wait for
each other; a strategy that splits windows sets `splits_windows` and adds
the gradients up in `synchronize`. A model too large for one GPU needs a strategy that splits the
model itself (PyTorch's FSDP, or a library such as DeepSpeed): it is
written as such a file, without changes to the training loop. None ships.

## Checkpoints and resuming

Inside a step, every `--checkpoint-every` updates (default 500, 0 for none
during a step), the gpu worker's main process writes a checkpoint: the step's
fingerprint, the update, the weights, the optimizer's, method's and
environment's state and the best weights so far. After the step's last
update it always writes one, unless that update's own was just written,
and that checkpoint, as it is, becomes the step's kept state ("Step
results and reuse", above); the checkpoint's place is then emptied: the
folder is removed, and a store checkpoint loses its name while its object
stays as the step's state. The same command with `--resume`
continues the step that stopped from its checkpoint, on other hardware
too: the number of processes, `--micro-batch`, `--recompute` and
`--segment` may change; a checkpoint written by another step, or for
other inputs (its fingerprint differs), stops the run with
`SettingsMismatch`. The steps before it are reused. Without `--resume`, a
leftover checkpoint stops the run, so a crashed run is never overwritten
by mistake.

The gpu worker writes every checkpoint on its own disk, where a restart of
its processes finds it. What comes back to the wpt manager is a setting,
`WPT_LINK_CHECKPOINTS`: `step-ends` (the default) sends only each step's
last checkpoint, with the step's end, which becomes its kept state;
`every` sends each one as it is written, into the place below. With
`step-ends`, a step that stops or fails leaves no checkpoint there, so
`--resume` starts it over, and the stop's message says so. The first
start of a job removes the checkpoint folders that earlier runs' jobs
left on the worker, which no later run reads.

Where it lives, on the manager's side:

- without `--run`, in `<model folder>/checkpoint/checkpoint.safetensors`, on the
  wpt manager's machine; a run that fails leaves `checkpoint/` in an
  earlier folder at that place, and a model folder that did not exist
  before the run and holds nothing else goes with the checkpoint;
- with `--run NAME` (or `WPT_RUN`), in the store, under the ref
  `checkpoints/NAME`. Any machine using the same `WPT_STORE` continues the
  run with the same command, `--run NAME --resume` and any `--out`: a run
  can move from one machine to another between sessions. Each save
  replaces the previous checkpoint of that name, which is then deleted;
  a crash while saving leaves the previous one. The ref's change and every
  deletion are recorded with the object's sha256, and that line is in the
  store before the deletion: a deletion whose record breaks leaves the
  object in the store. A save once the record broke still sets the ref,
  unrecorded, and leaves the previous object beside the new one. With an
  S3 store, every save is an upload of about four times the model's size.

### Stopping a run

Ctrl+C, `SIGTERM` (a shutdown, a rented machine being taken back) or, on
Windows, Ctrl+Break, sent to any process of the run, lets every process
finish its update; they then write a checkpoint (unless `--checkpoint-every
0`) and exit with code 130. `--resume` continues from that update. A second
signal stops the run at once, losing the updates since the last checkpoint,
as `kill -9` or a power cut does.

### Restarting after a crash

When a process crashes, the processes the gpu worker's job started are
stopped and all start again, up to `--max-restarts` times (default 3,
`WPT_MAX_RESTARTS`); a restarted process continues from the worker's own
checkpoint without `--resume`. A one-process run is started this way too,
by a small parent process in the job, unless `--max-restarts 0`, which runs
the training in the job's first process and never restarts. Three exits are never restarted: a refusal (exit code
2: a missing or damaged dataset, a checkpoint written by another step,
settings or options that do not fit, a step that failed, a drafter that
could not be trained), which would refuse again, a record that could not be
written (exit code 3), and a stop asked for (exit code 130). A bug that
crashes every time stops after the last restart, and `wpt-train` exits
with the crash's exit code. Restarts stay on one machine: on several gpu
workers, a crash ends the job on every one, and `wpt-train` exits with
code 5; `wpt-manager` resumes the run from the checkpoint it holds, as for
a lost worker ([machines.md](machines.md), "A lost gpu worker").

A gpu worker lost (it cannot be reached, or its link breaks) stops the run
with exit code 4; `--resume` continues it from the checkpoint the manager
holds, and `wpt-manager` does so itself on a replacement for a worker it
rented ([machines.md](machines.md), "A lost gpu worker").

## The drafter

After the last step, the gpu worker's main process trains the model's drafter
([model.md](model.md), "The drafter"): a model and its drafter are one
unit, so every run trains one, against the final weights, the model frozen,
with the recipe's `drafter` section. Its prompts are the beginnings of the
dataset's training documents, and the model alone writes the answers, each
up to the end of a document and, when a step trained on conversations or
tasks, `<|im_end|>`, the end of a turn: no text people wrote as an answer,
and no conversation of a step, reaches it. No option skips it. A drafter
that cannot be trained (too few prompts, a loss that is not finite, no
draft verified) refuses the run (exit code 2) and writes no model folder;
every step's result is kept, so the same command trains the drafter again
without training. The drafter reads the training shards once at the end of
the run, one shard at a time, to draw its prompts; from a remote store,
each shard's local copy is released once read. The disk then holds at most
one shard only when the shards were not already local: with `--stream`,
which released each one once its tokens were in the token file, or when
every step that reads text windows was reused. Otherwise the shards
training fetched to build its token file are still on disk when the
drafter starts, and each one goes as the drafter reads it.

Then the run measures the drafter scheduler's pass times once and writes
them into its record as `scheduler-costs.json`, with the setup they were
measured on: the model's config, the drafter's block (the drafts per
round), the rows (1) and the device (its kind and name; on the CPU, the
processor's). `--scheduler-costs FOLDER` replays those an earlier run
recorded in its record folder `FOLDER` instead; times measured on another
setup are refused before training starts. [model.md](model.md), "The
drafter", says exactly what a replay guarantees, and
[observability.md](observability.md) what the file holds.

## The model folder

The model folder holds:

- `config.json`, the model's numbers;
- the weights (safetensors, optionally split with `--max-shard-size`); each
  file's header carries `license` (AGPL-3.0-or-later), `output_license`
  (CC0-1.0) and `notice`, which points to the folder's `NOTICE` and the
  project, and `tokenizer` (the tokenizer recipe and version, such as
  `classics version 1`) and `tokenizer_sha256`;
- `tokenizer.json`, the dataset's vocabulary, exactly the bytes those
  headers name: `wpt-chat` and `wpt-eval` refuse another (see
  [tokenizer.md](tokenizer.md));
- `provenance.json`, what produced the model: each step, its fingerprint,
  result and what it reached, and every source the steps learned from
  (address, the manifest of the upstream tree it was built from, licenses
  and attribution, copied from the dataset record);
- `LICENSE`, the AGPL text; `NOTICE`, the licenses of the weights and of
  generated text, every source with the credit it asks for, and the output
  notice; `README.md`, the model card, with the licenses at the top and
  the steps in order;
- `drafter/`, the drafter for speculative decoding trained against these
  weights, with its `config.json`, weights (the same header) and
  `provenance.json`, which names the command and recipe that trained it,
  the dataset, the sha256 of every weight file it was trained against, its
  settings, its losses, the calibration and the measure: `load_model`
  refuses a folder without it or whose drafter names other weights
  ([model.md](model.md), "The drafter");
- `evals/`, results written by `wpt-eval` (see [evaluation.md](evaluation.md)),
  shown in the model card.

The main process writes the folder beside its target, as a hidden folder
named `.<folder name>-<random>`, records the sha256 of every file in it,
then renames it into place. A run that stops before (a crash, a refused
record, a failed step) leaves the target as it was. Training into a folder
that already exists still works, but replaces it as a whole: everything in
it goes, not only the earlier weights and the `evals/` results, which
describe other weights, but also its `checkpoint/` and any file put there
by hand. Keep such files elsewhere. The checkpoint's removal is recorded
before it is removed, and the record is in the store before the folder is
renamed. The earlier folder is first moved aside, then removed once the
new one is in place. When the new one cannot take its place, or the run is
interrupted meanwhile, the earlier folder is moved back; when that fails
too, the run stops with `RestoreFailed`, which names the earlier folder
(`.<folder name>-replaced-<random>`) and the new one, both kept: move one
of them into place by hand. Whenever the rename was tried and the target is
missing afterward, whatever error stopped the run (a first rename refused,
or a record that broke meanwhile), the new model is kept and named on the
error output, with the earlier folder when there is one. Without `--run`,
when the target holds only this run's checkpoint folder and the new folder
cannot take its place, the new model is kept beside it and named on the
error output, as when there was no target, since no other model is there
to go back to: move it into place by hand. When the earlier folder cannot
be removed, or its removal is interrupted, it stays beside the new one, as
`.<folder name>-replaced-<random>`, named in a warning and in the record:
remove it by hand. The run still succeeds, unless it was interrupted.

`NOTICE` and `README.md` are generated from the folder's own records, so they
cannot disagree with it. Each step's record in `provenance.json` names the
sources it read; `NOTICE` lists every one of them, with the credit and the
license text of each, and the model card names them under each step. Run
`wpt-eval` on the folder to measure it.

`wpt-chat <model folder>` continues text typed by the user, until the model
ends the document. Its start line names the licenses and where the notice
is; `wpt-chat <model folder> --about` prints the folder's `NOTICE`. A model
trained on text only continues text. `wpt-chat <model folder> --chat`
talks to a model trained on conversations: it keeps the history, writes it
with the chat template ([chat-format.md](chat-format.md)), and shows the
model's thinking and tool calls with their tags, without running the
calls. It measures the drafter scheduler's pass times when it loads the
model; `--scheduler-costs FOLDER` reads instead those a run recorded in its
record folder `FOLDER` of the store, refused when they were measured on
another setup ([model.md](model.md), "The drafter").

## Records written for training

The `answer-as-assistant` and `finish-tasks` steps learn from recorded
human demonstrations and from interactive environments.

### Rules

These records follow the corpus rules. Each example counts, so review
is systematic.

- Text written by people: existing sources under allowed licenses, original
  texts written for the project, and contributions under the DCO.
- No output of another company's language model, and no synthetic data that
  a WPT model did not make. WPT's own models may be used in training, their
  output included ([data-selection.md](data-selection.md), "Models in
  training"), such as a model's own attempt as the reference of a reward
  ("Rewards", below).
- A demonstration's `assistant` message may start as a draft written by a
  WPT model, in the conversation editor of `wpt-web`: the person edits it
  or confirms it as it is, and the message's `metadata.draft` records the
  model folder, the sha256 of its weights and of the draft, and whether the
  person changed it ([chat-format.md](chat-format.md), "Model drafts"). An
  unedited confirmed draft is allowed.
- An AI assistant (a coding agent helping a contributor) works with the
  data only through the `wpt-dataset` commands for structure and metadata,
  never through the ones that read or write people's texts; it may start a
  source with `new-source` without `--file` and `--license-text`
  (CONTRIBUTING.md, "Models and generated text").
- Environments use no fake web pages or apps: they replay recorded content,
  only when the project has the authorization to use it. A task's replies
  to the `ask` tool are content a person wrote and the task records, which
  the environment replays: no person waits for the model's question
  ("Tools", below).
- Scores come from programs or from humans, never from another company's
  model.

### Where records live

Records are files of a source, like any other text: they get the same
license entries, attribution, signed decision, NOTICE listing and checks as
every source. Records written for WPT go in an `empty` source (see
CONTRIBUTING.md, "Adding your own texts"). Conversations are files in the
`chat` format, tagged with the name of the step that reads them
(`answer-as-assistant`); a `chat` file is never a pre-training document
([chat-format.md](chat-format.md)). Tasks for agentic training are files in
the `task` format, tagged the same way (`finish-tasks`, whose tasks the
`finish-tasks-with-less-writing` step reads too); they are never
pre-training documents either. A dataset's build keeps every conversation
with the tags of its file, and every task but the measure-only ones
("Measure tasks", below) with the tags of its `task.json`; a step reads
those of its `tag` setting.

- A record holds only what a person made: commands typed, edits, messages
  written. From a session with an AI assistant, only the person's own
  messages may be kept, never the assistant's replies.
- No other person's personal data, and no secrets: terminal output and
  paths are checked before a record is committed.

### Record format

The shape of recorded demonstrations and environment recordings is not
settled yet. The code reads three kinds of records, each a format of
a source's files: conversations (`chat`, one JSON object per line, as
[chat-format.md](chat-format.md) says, turned into tokens by a template
before training, with media such as screenshots as separate files named by
their sha256), tasks (`task`, "Tasks", below) and people's ratings
(`rating`, [evaluation.md](evaluation.md), "People's ratings"). Whatever
the shape, a record's license, credit and decision are its source's, and
its contributor is the signed-off commit that adds it.

## The answer-as-assistant step

The `answer-as-assistant` step learns to write the assistant's turns of
the dataset's conversations whose `chat` file is tagged
`answer-as-assistant` (`tag`). Each conversation becomes tokens with the
chat template ([chat-format.md](chat-format.md)), and the loss
(`assistant-tokens`) is the next-token loss of the tokens the assistant
wrote only (thinking, text, tool calls and the closing `<|im_end|>`),
averaged over every such token of the update's conversations: the other
turns are context the model reads and does not learn to write. The CSA
indexers go on learning by their own loss, as in `read-text` (model.md,
"Attention"): its gradient joins the update's, and its value is shown
beside the loss of each epoch. The multi-token prediction block is not
used, so it does not change, and the routers' balancing biases are locked.

Each epoch (`epochs`, 3) goes over every conversation once, in an order
drawn from the recipe's `seed` and the step's number in the recipe, `batch` conversations per update (8; the last
update of an epoch may hold fewer). Conversation `j` of an update goes to
process `j` modulo the number of processes, and a process left with fewer
passes than the others makes passes that learn nothing, so that every
process makes as many. A dataset that holds no conversation of its tag
refuses the run before any step trains. The step fails when a
conversation does not fit the model's context (cutting it would drop the
answers it holds), when one has no assistant message, and when the loss
is not finite. It needs a vocabulary
of tokenizer 0.3.0 or later, which reserves the chat template's ids. Talk
to the result with `uv run wpt-chat FOLDER --chat`.

## The finish-tasks step

The `finish-tasks` step teaches the model to finish tasks as an agent, by
reinforcement learning (group relative policy optimization). The model
works in a small harness, on tasks that people write, with four built-in
tools, `read` (a file), `write` (a file), `edit` (replace a string that
occurs exactly once in a file) and `bash` (run a command), and the tools a
task adds ("Tools", below). Its code is the `wpt-agent` package (tools,
containers, images, the agent loop and the scoring), the `agent-tasks`
environment and `wpt_training/agentic.py` (the attempts and their
rewards), and the `grpo` method (the learning). The wpt manager plays
every attempt, the gpu worker's model writing its tokens, so it needs
Docker where the manager's `docker` command reaches it: its own machine or
the Docker host `DOCKER_HOST` names ([machines.md](machines.md)); every
process on the worker then trains on its share of them. It reads the dataset's tasks tagged
`finish-tasks` (`tag`); a dataset that holds none refuses the run, with
"no tasks", before any step trains.

### Tasks

Tasks are written by people, in an `empty` source (CONTRIBUTING.md,
"Adding your own texts"), as files in the `task` format tagged `finish-tasks`. One
folder per task:

```text
<folder>/task.json   what the model is asked, the image, the limits, the tools it needs and, if needed,
                     larger containers, its own tools and the user's replies
<folder>/files/...   the starting files, copied into the container's /work (those tool_files names
                     go where only the tools' programs read them)
<folder>/check       any program (its first line names its interpreter, such as #!/bin/sh)
```

```json
{"instructions": "Fix the failing test in calc.py.", "image": "python",
 "limits": {"max_turns": 20, "max_tokens": 4000, "command_seconds": 30},
 "container": {"memory": 8192}, "needs": [["read", "bash"], ["edit", "write"]],
 "replies": [{"pattern": "(?i)python.*version", "reply": "Python 3.13."}],
 "answered_by": ["read", "bash"], "answered_in": ["NOTES.md"], "area": ["calc.py", "tests/**"],
 "gates": ["read_before_write"]}
```

- `instructions`: the user message that starts every attempt. No task's
  instructions state an answer block: the model's final reply is its
  answer, as text, and the harness asks for its confidence after it
  ([chat-format.md](chat-format.md), "The answer"). The harness adds no
  sentence to the opening.
- `terse` (optional): the same request in one line, for the measure of
  terse against full requests ([evaluation.md](evaluation.md), "Terse and
  full requests"): a task can hold its request in two forms, its full
  `instructions` and this one-line form. The agent loop opens an attempt
  with `instructions`, or with `terse` when its caller asks
  (`wpt_agent.attempts(..., terse=True)`; a task without the field is
  refused there). It holds no line break: `wpt-dataset check --fetch`
  refuses one, naming the `task.json`.
- `image`: the folder in `data/images/` the attempt's container comes from.
- `limits`: `max_turns`, the assistant turns in one attempt; `max_tokens`,
  the tokens the model writes in one attempt, over every turn;
  `command_seconds`, how long one tool call may run.
- `container` (optional): sizes above the defaults (Attempts, below):
  `memory` and `disk` in MiB, `processes`, `cpus`. Each can only raise its
  default; a smaller value leaves the default. Since `/work` and `/tmp` are
  kept in memory, the sizes that result must leave room for the processes:
  2 x `disk` + 1024 MiB (`HEADROOM`) must stay below `memory`. A task that
  raises `disk` must raise `memory` with it; the defaults leave 1024 MiB
  more than that.
- `needs`: the tools the task cannot be done without, each need the list of
  its variants (two different tools that write a file, say); every attempt
  is offered exactly one variant of each need ("Tools", below). `[]` when
  the task needs none.
- `replies` (optional): the user's replies, written in advance, to the
  `ask` tool, each a `pattern` and its `reply`; only a task with replies
  offers `ask`.
- `answered_by` (required with `replies`): the looking tools that can
  answer what the task leaves open, `read`, `bash` or a declared tool
  marked `looks` (the task's or its image's pool's), through which an
  attempt opens the files of `answered_in`.
- `answered_in` (required with `replies`): where the task's answers live,
  as glob patterns of paths under `/work` (as `area` reads them). An
  attempt asks the user only after it opened, through a tool of
  `answered_by`, every file it starts with that they match, and did not
  find the answer there (the check can name its text to `asked_last`); a
  question counts at once when they match none of its files, the answer
  being absent (`asked_last`, "The check library", below).
- `tools` (optional): the task's own tools beyond the built-in ones and
  `ask`, each with its `name`, `description`, `parameters` (the JSON Schema
  of its arguments), `program` (the absolute path, in the image, of the
  program that answers it) and, optionally, `names` (other names an attempt
  may offer it under) and `looks` (`true` for a tool that looks at sources
  as `read` does, such as a search or a fetch; "The check library", below).
- `tool_files` (optional): paths under `files/`, each a file or a folder
  with everything in it, that only the tools' programs read: they go to
  `/tmp/wpt-tools/` instead of `/work`.
- `absent` (optional): absolute folders of the image, outside `/work` and
  `/tmp`, their names made of letters, digits, `.`, `_`, `+` and `-`, that
  the attempts do not see: each is covered by an empty, read-only folder
  kept in memory (Docker's `--tmpfs`), so a task removes what its image
  installs there, such as the manual of `ondrik` ("The agent image",
  below). The check's container shows the image as it is.
- `pointers` (optional): the texts the task takes from documents of other
  sources ("Pointer tasks", below).
- `area` (optional): the paths under `/work` an attempt may change, as glob
  patterns (`*`, `**`, `?`, `[...]`, as a source's license paths read
  them), `[]` when the task only answers and may change nothing. An attempt
  that changed, added or removed a path no pattern matches breaks the
  `within_area` part ("Rewards", below). A task without `area` gets no such
  part.
- `gates` (optional): the parts that must be 1 for an attempt to score at
  all, by name: the check's own (such as `asked_last`) or `within_area`. A
  gated part below 1, or missing, makes the attempt worth 0 whatever the
  rest ("Rewards", below). There is no gate for every task beyond the
  missing answer.

Task files are a source's bytes, which `wpt-dataset migrate` never
rewrites: a `task.json` of an earlier shape (a task with `replies` and no
`answered_in`) is read in the current shape each time it is read
(`migrate.upgrade`), by the build, the measures and the task editor of
`wpt-web`, whose save then writes the current shape. Only a person
knows where a task's answers live, so a task with replies gets an empty
`answered_in`, which `wpt-dataset check --fetch` then refuses, naming its
`task.json` to fill in.

A task folder is a folder holding `task.json` that is not inside another
task's `files/`; every other file of the format is a task folder's `check`
or under its `files/`. `wpt-dataset check --fetch` checks each folder
against `packages/wpt-dataset/src/wpt_dataset/schemas/task.schema.json`,
and checks that its `image` is a folder of `data/images/` holding a
`Dockerfile`. CI does not fetch the sources a change touches: the
maintainer runs `wpt-dataset check --fetch` on a pull request's sources
before merging, which catches a task naming an image that is gone. The same
check refuses a task whose sizes break the rule on `disk` and `memory`
above, naming its `task.json`, and one whose tools break the rules of
"Tools" (below): a declared tool taking a built-in name or `ask`, or
declared twice; a need naming a tool the task does not offer (`ask` without
replies included); a tool in two needs; a task with replies and no
`answered_by` or no `answered_in`, or an `answered_by` naming a tool that
is not a looking tool; a reply pattern that is not a regular expression; a
`terse` that holds a line break; a `tool_files` entry that names nothing
under `files/`; an `absent` entry that is `/work` or `/tmp`, lies under
them, or holds `.` or `..`; a pointer that names a source that does not
exist or takes the place of another file, or whose text cannot be filled in
("Pointer tasks", below). A step reads the tasks whose `task.json`
carries its tag. Every file of a task folder is UTF-8 text, as the task
format defines: an encoding stated inside such a file is not read
([architecture.md](architecture.md), "Encodings").

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
changed, such as the sentence it removed. The build and the measures fill in
the text from the store, each span checked against its sha256. Each entry of
`pointers`:

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
final tree comes from the store, where `wpt-dataset download` put it
(nothing downloads it here), read as the build keeps it, the
document is converted as the build converts it, and the span must hold the
bytes its `sha256` names and must not cut a character (the pointer-task
check, [architecture.md](architecture.md), "Shared checks"). The text is
then the span without its removed parts. A task breaks when a new download
of its source changes what the span holds, or when the build leaves out
the document it points into, until the builder makes it again. The build
leaves a task out alone, and names it with why in the dataset record
(`left_out_tasks`), when it breaks, when a source it points into is not
admitted, and when the build left that source out (not in the store, or a
cut or patch that does not apply). `wpt-dataset check --fetch` fills in
every pointer of the sources it checks and refuses a task that breaks,
naming the task and the pointer, and `check --changed` also checks every
task source whose pointers point into a source the change touches.

A repository pointer names a commit of a git source's history instead, for
the tasks that commit prepared changes (the task file names the source and
the commit):

```json
{"for": "repository", "source": "<git source id>", "document": ".git/history/<commit>",
 "manifest": "<the manifest of the source's upstream tree the task was built from>"}
```

The task starts from the real state just before the commit. When it is
read, the source's upstream tree in the store must still have the manifest
the pointer names (the sha256 of its file list,
[architecture.md](architecture.md), "The manifest"), which covers every
history file and the history's order file
([architecture.md](architecture.md), `history`), so the same pointer always
gives the same repository; once a new download gives the tree another
manifest, the task breaks as above until the builder makes it again. From
the source's final tree, every commit after the pointer's commit is
undone, newest first, in the order the order file stores, then the commit
itself. Undoing the later commits gives
the commit's own tree, the tree the pushed commit must hold; undoing the
commit too gives its parent. A merge between the commit and the final tree,
or a diff that does not undo, makes the task fail to be filled, as does a
history file whose author and date git cannot make the parent commit from
(an empty name or email); the builder never draws a commit behind a merge,
and skips the others. Any other git failure is not the pointer's and stops
the build.

`/work` is the repository itself (the request "Based on the current diffs,
could you commit and push the changes in this repository?" means `/work`).
Its one commit, on the branch `main`, holds the parent tree (the store
keeps no older commit, so this commit stands for the history: its author
and date are the commit's, its message names the commit it comes before).
Its working tree holds the commit's own tree, so the commit's changes are
there and not committed, and its git settings name the commit's author. A
task with a repository pointer has no other file in `/work`: no `files/`
and no pointer to files. Its `origin` is a bare repository holding the same
commit, outside `/work`, at `/tmp/origin.git` in the container, so `git add
-A` in `/work` never takes it. No commit message is given: the model writes
its own. Next to the check, `repository.json` holds what the pushed commit
must be: the remote (`/tmp/origin.git`), the branch, the commit it starts
from (`parent`) and the tree the pushed commit must hold (`tree`, the
commit's own tree). A task has at most one repository pointer. The task
counts as built from the commit's document, `.git/history/<commit>`, for
"Measure tasks", below.

### Measure tasks

The measures' tasks are kept out of training by two rules, which the task
reader (`wpt_dataset.tasks.read`) applies:

- people tag the measure tasks they write `eval`, and no dataset holds a
  task so tagged, whatever its other tags;
- a fixed share of every task source is measure-only, chosen by a hash of
  the task id the same way held-out documents are chosen
  ([data-selection.md](data-selection.md), "Held-out text and
  contamination"): the share is the model recipe's `held_out`, the same
  rule as documents. The build applies the recipe's `held_out`, the share
  that keeps its documents out.

A pointer task built from a held-out document is measure-only too (a
document of a source with `held_out_by` is held out by its group, as the
build holds it out: [data-selection.md](data-selection.md), "Held-out text
and contamination"), and so is a pointer task whose document's text equals
a held-out document's: when the pointers are filled in (`tasks.fill`), the
sha256 of each named document's whole Markdown is kept, and `measure_only`
compares it with the sha256 of the Markdown of every held-out document of
the dataset, so the same text under another source or path is caught too.
The build compares with the documents it holds out itself. The measures
compare with the dataset the model's `provenance.json` names
(`held_out_texts`, read once for each pass that reads tasks), whose sha256
come from the dataset's document list, which its build keeps
([data-selection.md](data-selection.md), "Held-out text and
contamination"); no source is converted to find them. A dataset built
before builds kept that list has none: reading pointer tasks against it is
refused (`DocumentListMissing`), saying that the dataset predates document
lists and that such a model is used with the repository at the commit that
built it; nothing falls back to converting the sources, and nothing is
rebuilt. A dataset the store does not hold at all is refused the same way,
the message saying it is not in the store. A held-out document never
reaches training through a task. A repository pointer names only its
commit's document: the other files of the repository's tree are not looked
at by this rule. The measures read exactly the measure-only tasks
(`read(..., measure=True)`), whatever their tags (`tag=None`): those tagged
`eval`, the hash share of every task source and the pointer tasks on
held-out text, all in one run of attempts that every agent measure reads
([evaluation.md](evaluation.md), "The shared run of attempts").

### Tools

Every attempt is offered the tools of its task, chosen for that attempt
("Variation", below). A task's tools are:

- the four built-in ones, `read`, `write`, `edit` and `bash`;
- `ask`, when the task has `replies`;
- the tools the task declares;
- and the working tools of the image's pool (`tools.json`, "Images",
  below), whose names the task does not take.

**`ask`**. The model calls it with its question, and the harness answers
from the task's replies, written in advance; the attempt goes on. Each
reply's `pattern` is a Python regular expression searched for anywhere in
the question, and the first reply whose pattern is found is given. A
question that no pattern matches gets "I don't know that", so a precise
question pays and a vague one does not. No person waits for the model's
question: the reply is written ahead and the harness gives it, as content a
person wrote that the task records (Rules, above). `ask` is offered only in
a task that declares replies. It serves asking the user last: a turn
without a tool call still ends the attempt, so a question goes through
`ask`.

**Declared tools**. A task declares each tool it offers beyond the built-in
ones and `ask`: its name, description, parameters, and the program in the
agent image that answers it. The program runs inside the attempt's
container, under its limits and `command_seconds`: it reads the call's
arguments as JSON on its standard input (as ASCII, any other character
escaped) and runs from `/work`; its standard output is the result, and an
exit code other than 0 makes the call fail, with its standard error (or,
when it printed none there, its output) as the result. Like a check, a
program must not load code from its current folder: the model's files are
there. Most of these tools are emulations, such as a web search answered
from local files; an emulation gives the real tool's kind and shape of
answer: a search returns ranked snippets with their sources, a fetch
returns a page's text. So the model learns what kind of answer to expect
from a kind of tool and adapts to whatever harness it is put in. Real MCP
servers are not used in training.

**Hidden files**. An emulated tool's backing files are hidden from the
model's own tools (`read`, `bash`): only the tools' programs read them. The
task names them in `tool_files`; they are unpacked into `/tmp/wpt-tools/`,
owned by root and readable only by group 1001 (folders 750, files 640). A
program runs as user 1000, the model's, in group 1001, so it reads them and
changes `/work` as the model's own tools do; the model's commands run in
group 1000, so they can neither list nor read them, and since the groups
differ the kernel lets them neither look into the program's process
(`ptrace`, `/proc/<pid>/fd`, `/proc/<pid>/mem`) nor gain group 1001 (no
setgid). The files count against what `/tmp` may hold. The check's
container does not get the task's `tool_files`, but it does get the files
an attempt's variant hid (a variant's `tool_files`, "Variation", below):
they are back at their place, in its `/work` or at their folder outside it,
as the tools' programs left them, in place of whatever the model put there,
so the check of a commit-and-push task reads the remote the push tool
pushed to. A folder a task puts outside `/work` (a repository pointer's
remote, `/tmp/origin.git`) belongs to user 1000 in the attempt, and the
check's container gets it too, at the same place, as the attempt left it.

**Variation**. Every task lists the tools it absolutely needs (`needs`),
and a needed tool may have variants. For each attempt the step (and the
measures) draws, from a generator seeded with the recipe's `seed` and the
step's number in the recipe, the task,
the round, the group and the attempt, so runs are reproducible
(`wpt_agent.offer`):

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

The tools of an image's pool pad the offer with tools the task never needs;
each really works, answered by its own program in the image. The map from
each name the model saw to the real tool goes into the conversation the
check reads ("Scores", below). The aim is asking the user last without
habits tied to particular tools.

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
written by people like the Dockerfile: `{"tools": [...]}`,
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
`name=version`), and stops when the build installed other packages than the
lock pins. Like `lock-image`, and through the same code, it first checks
that apt, after `apt-get update`, reads its package lists only from
snapshot.debian.org or snapshot.ubuntu.com, and refuses to build otherwise.
A failed `docker build` (in `lock-image` or `build-image`) and a failed
`apt-get update` in that check are tried again by the same rule, 6 times in
all, with a growing pause before each new try: 10 s, 30 s, 1 min, 2 min and
5 min, about 9 minutes of waiting, because snapshot.ubuntu.com was once
down for over 2 hours. The error of a failed build ends with the output of
the step that failed. No step builds an image: the `agent-tasks`
environment stops before any attempt when a task's image is not built for
the folder's current files, and names the command that builds it. A changed Dockerfile
or lock needs a new build.

### The agent image

`data/images/agent/` is the first image of agentic tasks: Debian 13 with
`bash`, `git` and Python 3, and four parts of its own, which its Dockerfile
copies in:

- `wpt_check.py`, the shared check library ("The check library", below), in
  `/usr/lib/python3/dist-packages/`, where every `python3` finds it, `-I`
  and `-P` included;
- the programs that answer tools, in `/usr/local/lib/wpt-tools/`, written
  in Python and started as `python3 -I`, with the helpers they share,
  `wpt_tools.py`, next to the library;
- `tools.json`, its pool of working tools;
- `ondrik`, a small program written for WPT, so no training text describes
  it, in `/usr/local/bin/`, and its manual,
  `/usr/share/doc/ondrik/manual.txt` (`ondrik/` in the image's folder), for
  the task builder's `read-the-manual` ("The task builder", below). It is
  compiled: its source, `ondrik/ondrik.c`, is compiled in a build stage of
  the Dockerfile, and the image keeps only the binary, owned by root, which
  the model's user may run but not read, and no compiler.

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
`/work`: `list_files` (a folder's entries, as `ls -p`), `find_files` (paths
matching a glob pattern), `search_text` (lines matching a regular
expression, as `grep -rn`) and `calculator` (an arithmetic expression); the
first three are marked `looks`. Since a program runs in the group that
reads the hidden files, these tools open a path only inside `/work`, one
folder at a time and never through a symbolic link, so neither a path
leading out of `/work` nor a link the model made reaches the hidden files.

A task that commits and pushes (the task builder's `commit-and-push`,
below) declares the emulated `push`:

```json
{"name": "push", "description": "Push the commits of the current branch of a git repository to its remote.",
 "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
 "program": "/usr/local/lib/wpt-tools/push", "names": ["git_push", "push_changes"]}
```

It runs `git push` in the repository at `path` (by default `/work`) and
gives what git printed. In the attempt whose variant hides the remote (the
task builder's `commit-and-push`, below), the remote is among the hidden
files under its own path: when the remote git would push to is a local path
outside `/work` (`/tmp/origin.git`) whose hidden copy exists under
`/tmp/wpt-tools/` (`/tmp/wpt-tools/tmp/origin.git`), the program pushes
there, so only it reaches the remote and `git push` through `bash` does
not. git reads the repository's own settings, which the model can change,
and the program runs in the group that reads the hidden files: it turns
hooks and the file system monitor off, and a task whose hidden files are
more than the remote it pushes to must not offer it.

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

Rules many checks share live in one library inside the agent images, tested
once in the repository, and each check calls it. A check started as
`python3 -I` imports it:

```python
#!/usr/bin/python3 -I
import wpt_check
conversation = wpt_check.load()  # conversation.json, next to the check
right = ...  # the task's own judgement of the files and the answer
print(wpt_check.report(conversation, right, sources_first=wpt_check.sources_first(conversation)))
```

A check prints the named parts of its score ("Scores", below), and the
step weighs them ("Rewards", below); the library weighs nothing.

The rules read the attempt's trace by real tool names (`metadata.tools`
maps each name the model saw to the real one):

- `report(conversation, correct, **rules)`: what a check prints, the JSON
  object of its parts: `correct`, the task's own judgement through
  `scored`, and each rule given by name, as a number (`True` is 1). A
  task's check may report any part of its own the same way, such as a
  `conventions` part for the conventions its workspace shows (placement,
  naming, framework, existing commands): those are checked only by each
  task's own check, never by a rule of this library.
- `scored(conversation, score)`: the missing-answer rule. The attempt's
  score is 0 when it has no answer, whatever its files hold: no final
  reply, or a reply to the confidence question that is not a number from 0
  to 100 ([chat-format.md](chat-format.md), "The answer").
- `brier(conversation, correct)`: the Brier score of the confidence,
  `(confidence - 1)²` for a right result and `confidence²` for a wrong one;
  1, the worst, without an answer.
- `read_before_write(conversation, new=())`: every file written or edited
  was read first, by `read`, by an earlier `bash` command that names it, or
  written by the attempt itself; `new` lists the files the task expects the
  model to create.
- `sources_first(conversation, changes=("write", "edit", "ask"))`: the
  attempt looked at sources (below) before its first change or question,
  and looked at all when it did neither.
- `asked_last(conversation, found="")`: every `ask` came after the attempt
  opened every file where the answer lives: the files of `/work` it started
  with that the task's `answered_in` matches (`metadata.answered_in`), each
  opened by a call that looks (below) to one of the task's `answered_by`
  (`metadata.answered_by`) and did not fail, whose `path` argument is the
  file or, for `bash`, whose command names it and runs a program that shows
  what a file holds (`opened`, `SHOWING`: the reading programs but `wc`,
  `csplit`, `find`, `ls`, `dir`, `vdir`, `stat`, `du` and `cmp`, which show
  only its name, size or counts). Given `found`, the answer's text, a
  question no longer counts once a call that looks showed it, its words in
  order as `holds` reads them: the answer was found. A question counts at
  once when no such file exists (the answer is absent), and when the
  attempt was offered none of those tools. A look that does not name the
  file, such as `grep -r` over a folder, does not open it.
- `read_before_ask(conversation, text)`: every `ask` came after a call that
  looks and whose result showed `text`, its words in order as `holds` reads
  them; true when the attempt asked nothing.
- `before(conversation, first, then)`: every call to a tool of `then` after
  a call to one of `first`; `calls` gives the trace, each call by its real
  tool's name and whether it looks, `answer` the answer the loop built,
  `{"answer": ..., "confidence": ...}` or `None`, and `answer_text` its
  text, the final reply's, `""` without one.
- `holds(text, expected)`: `text` holds the words of `expected` one after
  the other, letter case, punctuation, Markdown marks and spacing aside.
- `not_enough(conversation, missing="", words=None)`: the answer starts
  with "not enough", the answer of a task whose files do not hold what it
  asks; given `missing`, the text the files lack, the answer must also name
  it: hold its first `words` words (all of them by default) in order, as
  `holds` reads them.
- `pushed(expected=None)`: the remote of a repository pointer
  (`repository.json` next to the check, by default) holds on its branch
  exactly one new commit, on the commit it started from, with the expected
  tree and a message that is not empty, spaces aside. The model writes its
  own message, and the check never compares it. The remote is outside
  `/work`, at `/tmp/origin.git`. Since the model could have changed the
  remote's settings, only its objects and references are copied into a
  fresh repository of the check's own, so git reads none of them.
- `beside(name)` and `file_text(path)`: the text of a file next to the
  check, and of a regular file the attempt left (`""` for none, a folder, a
  pipe or a link).
- `programs(command)`: the programs a `bash` command runs, as the looking
  rule finds them (below).

Looking is one rule, which `sources_first`, `asked_last` and
`read_before_ask` share: every looking tool counts as opening a source. A
call looks when it calls `read`, a declared tool marked `"looks": true`
(the agent image's `search` and `fetch`, and its pool's `list_files`,
`find_files` and `search_text`), or `bash` with a command that runs one of
the image's reading programs. Those programs are listed once, as `READERS`
in `wpt_check.py`: the programs of the image that show what files or
folders hold, such as `cat`, `head`, `grep`, `sed`, `ls` and `find`, and a
test checks that the built image holds each of them. The command is parsed
as bash reads it, never searched for a name: each simple command of its
pipelines and lists (`|`, `;`, `&`, `&&`, `||`, a line break, parentheses,
a function's body) runs the program named by its first word, after any
`VAR=value` and bash's reserved words, and after programs that run another
program (`env`, `nice`, `nohup`, `stdbuf`, `time`, `timeout`, `xargs` and
the like) with their options and the words they take; a comment does not
count. So `cd docs && cat a.md` looks and `echo cat a.md` does not. Which
offered tools look is decided in one place, `wpt_dataset.tasks.looks`
(`read`, `bash`, and a declaration marked `looks`): the agent loop writes
their real names into the conversation's `metadata.looking`, which the
library reads, and `wpt-dataset check` refuses an `answered_by` naming any
other tool.

The trace rules of #37 are `read_before_write`, `sources_first` and
`asked_last`, each reported as a part of that name by the checks that apply
it ("The task builder", below). Each task states where its answers live,
its `answered_in`, and the looking tools that can open them, its
`answered_by`; `asked_last` waits for both.

### The task builder

Program-built tasks are generated by a script in the repository,
`scripts/build_tasks.py`, as committed task sources made of pointer tasks,
so adding, removing or varying tasks is a rerun of the script. Its code is
`wpt_dataset/builder.py`:

```sh
uv run scripts/build_tasks.py KIND SOURCE --count N [--seed S] [--select EXPRESSION]
```

`SOURCE` is an `empty` source a person makes for the tasks, like any text
written for WPT (`wpt-dataset new-source`, then its license text, tagged
`finish-tasks`); a maintainer admits it. A run replaces everything under `KIND/` in
it by `N` tasks, as one new patch checked as `check --fetch` checks it,
every pointer filled in. The tasks come from the admitted sources already
in the store, drawn with a generator seeded by `S` (default 1): the same
seed, sources and store give the same tasks. A source that is not in the
store gives no task, and the run says how many were left out. Each task's
folder is `KIND/<16 hex digits of the sha256 of its task.json>`, so an
unchanged task keeps its id over reruns, and with it its place in or out of
the measure-only share ("Measure tasks", above). Every task runs in the
agent image, and its check calls the check library, which reads the
attempt's answer: the final reply's text and the confidence the model gave
when the harness asked for it after that reply
([chat-format.md](chat-format.md), "The answer"). No kind's instructions
say anything about a block or how to answer, and the harness adds nothing
to the opening.

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
  must hold the first words of `missing.txt`, as many as `quote.txt` holds,
  in order, matched by words as `holds` matches them (`not_enough`). The
  other half can be answered, and "not enough" is wrong there.
- `look-or-ask`: the rest of the paragraph goes into a file whose name is
  the one missing detail: in half the tasks `NOTES.md` names it, in the
  other half only the reply of the `ask` tool does (and `ask` is needed);
  both halves declare the same reply. The task's `answered_in` is
  `["NOTES.md"]` in both halves, the file where the answer would live, so
  the attempt looks before asking even where the notes hold nothing, and
  its `answered_by` names the looking tools that can read it: `read`,
  `bash` and the pool's `search_text`. The check wants the file's text and
  `asked_last`: no question before `NOTES.md` was opened through one of
  those; in the half where the notes do not name the file, the attempt
  reads them, does not find the name, and then asks. In the half where
  `NOTES.md` names the file, the answer is found there, so `asked_last`
  gets the line naming it as `found`: a question after a look showed that
  line does not count, and the check also wants `read_before_ask`: the
  attempt read `NOTES.md`, through any looking tool, before any question.
  Those two are the task's `gates`: `correct` is the file's text, and a
  broken gate makes the attempt worth 0.
- `commit-and-push`: every commit of each admitted git source's stored
  history that undoing can reach is a candidate, not only the newest, when
  the source's working tree stays within 4 MiB and 2,000 files. A source
  gives one candidate per history file (a merge has none) of a commit newer
  than its newest merge, read from its stored order file (every commit's,
  when the history holds no merge); the commits behind that merge, which
  undoing cannot reach, are dropped before drawing and never logged. A
  source whose stored history has no order file (stored before histories
  kept one) gives none, and the builder says how many such sources it left
  aside. The builder draws about `--count` of them (2,500 per kind) from
  all the candidates, in an order seeded by `S`, so the same seed, sources
  and store give the same tasks. A drawn commit's task starts from the real
  state just before it: every later commit undone, newest first, in the
  stored order, then the commit ("Pointer tasks", above). Only a real
  failure is logged per draw: a drawn commit whose undoing fails, or whose
  parent commit git refuses to make, is left out, named, and the next drawn
  takes its place. Only a drawn commit is made into a repository: the
  candidates are history files, not trees. Each kept commit is a repository
  pointer naming the manifest of the source's upstream tree ("Pointer tasks", above), and
  `/work` is the repository, its remote outside it. The model is asked only
  "Based on the current diffs, could you commit and push the changes in
  this repository?", with nothing after it; it writes its own commit
  message, then pushes. The check is `pushed`: exactly one new commit
  pushed to the remote branch, whose tree is the commit's own tree, its
  message any that is not empty, never compared. Each attempt pushes one
  way, the seeded choice between the two variants of the task's one need
  ("Variation", above): `bash` alone, so no push tool is offered and `git
  push` through `bash` reaches the bare `origin` at `/tmp/origin.git`; or
  `bash` with the declared `push`, a variant that hides the remote (its
  `tool_files` name `/tmp/origin.git`), so `git push` through `bash` cannot
  reach it and only the push tool's program does ("The agent image",
  above). Over a task's attempts both ways come up, so no habit is tied to
  one way of pushing. Its commits come from the admitted GitHub
  repositories.
- `read-the-manual`: each task asks for a result only the image's `ondrik`
  gives ("The agent image", above): the seal of a drawn text (8 hex digits)
  folded a drawn number of times (2 to 9), or the slack knot of two drawn
  numbers (100 to 99,999). The instructions name the program and the
  result, never the option that gives it, and are the same whether the
  manual is there or not. The halves are crossed: every other task names
  the manual's folder in `absent`, so its attempts run without the manual,
  and every other pair asks for a seal instead of a knot. The task needs
  `bash` only (`[["bash"]]`) and has no files. Its check runs the program,
  by its absolute path, in the check's own container, which the model's
  files cannot change, and wants the program's output in the answer
  (`holds`). It reads no source and takes no `--select`. It is the agent
  form of the reading-gain measure ([evaluation.md](evaluation.md),
  "Reading the manual"). Its tasks are for the measures only:
  `builder.write` tags the kind's folder `eval` in the task source's
  `source.json`, through the service layer, when no entry tags it so, and
  the task reader gives tasks tagged `eval` to the measures only ("Measure
  tasks", below). A model trained on them would learn `ondrik` from the
  attempts themselves, and the measure would no longer tell whether it
  reads the manual.

Every kind's check prints its parts through `wpt_check.report`, and the
builder sets each kind's `area` and `gates`:

| kind | parts besides `correct` | `area` | `gates` |
| --- | --- | --- | --- |
| `find-the-passage`, `evidence-removed` | `sources_first` | `[]`: it only answers | none |
| `look-or-ask` | `sources_first`, `read_before_write` (the result's file is new), `asked_last`, and `read_before_ask` where `NOTES.md` names the file | the result's file | `asked_last`, and `read_before_ask` where it is reported |
| `commit-and-push` | none | `[".git/**"]`: git's own records | none |
| `read-the-manual` | none | `[]`: it only answers | none |

The trace rules are reported where the kind gives what they judge: given
sources for `sources_first`, a file to write for `read_before_write`, an
`ask` tool for `asked_last`. A `look-or-ask` attempt may change the
result's file only: anything else it adds, removes or edits breaks
`within_area`. A `commit-and-push` attempt may change git's own records,
everything under `.git/`, and nothing else: committing and pushing the
prepared changes leaves the project's files as they are, so any project
file the attempt adds, removes or edits while committing is over-reach and
breaks `within_area`. Only `look-or-ask` has replies, so only its tasks get
an `answered_in` (`["NOTES.md"]`, above); the other kinds have no
`asked_last` rule and leave it out.

Each kind is marked as acting, checking or both (`CLASSES` in
`builder.py`): `commit-and-push` acts; `find-the-passage`,
`evidence-removed` and `look-or-ask` check; `read-the-manual` does both,
since it runs a program and checks its output. After each run,
`scripts/build_tasks.py` reports what the task source holds: its built
tasks of each kind, and the shares of them that act only, that check only
and that do both, which sum to 100% (`builder.shares`), such as:

```text
2500 look-or-ask tasks written into <source id>
<source id> holds 10,000 built tasks (find-the-passage 2,500, evidence-removed 2,500, look-or-ask 2,500, commit-and-push 2,500, read-the-manual 0): acting only 25.0%, checking only 75.0%, both 0.0%
```

The first committed task set holds all 4 kinds, about 2,500 tasks of each,
about 10,000 in all, and reports its shares this way. The 70/30 mix between
the two shares is settled later, once more kinds of tasks are defined: only
the program-built kinds are defined so far. Building the set needs every
admitted source it reads downloaded into the store first (`uv run wpt-dataset
download`), on a machine that holds them, the task sources made and
admitted, and one run per kind with `--count 2500`.

### Attempts

For each task, the model makes several attempts (`group`), written
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
error to the model); the results come back as a tool turn; and the model
writes the next turn. The attempt ends when a turn closes without a tool
call, when the model writes the end of a document, after `max_turns` turns
(the last turn's calls still run), or when `max_tokens` is reached in the
middle of a turn, whose calls then do not run. The turn that closes without
a tool call is the final reply, the answer; the loop then adds one user
turn asking for the confidence, a number from 0 to 100 and nothing else,
and the model replies; that turn counts against `max_tokens` and the
context, not `max_turns` ([chat-format.md](chat-format.md), "The answer").
The loop builds `{"answer": <the final reply's text>, "confidence": <number
/ 100>}`, or `null` when there is no final reply or the reply is not such a
number, and keeps it in the conversation's `metadata.answer`, next to
`metadata.tools`, the map from each name the model saw to the real tool,
`metadata.looking`, the real names of the offered tools that look at
sources, `metadata.answered_by`, the task's, and `metadata.answered_in`,
the files of `/work` the attempt starts with that the task's `answered_in`
matches ("The check library", above). The model's `block_size` does not
limit an attempt: the cache reads it past `block_size`, and the loss pass
reads it whole. The attempt also ends when the model's own processes or
files fill its container, so that Docker can start no command there:
processes left running reach the process limit, or files in `/work` and
`/tmp` exhaust the memory. That turn's calls get no result, and Docker's
own error never reaches the model. The attempt is scored like any other, on
what `/work` holds; when running out of memory also stopped the container,
`/work` is lost and the check scores an empty folder. A command that runs
longer than `command_seconds` is stopped, and its result says so (one that
ignores the signal to stop is killed 5 seconds later); a command that only
exits with the code `timeout` uses is reported with that exit code. A tool
result longer than `result_bytes` (4000) keeps its first and last 2000
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
as root from `/work` for at most `check_seconds` (300).

Next to the check, in its folder, is `conversation.json`: the attempt's
conversation in the chat format, as the attempts file keeps it (Runs,
below) but with only the attempt's own `metadata`: `tools`, the map from
each name the model saw to the real tool; `looking`, the real names of the
offered tools that look at sources ("The check library", below);
`answered_by`, the task's; `answered_in`, the files of `/work` the attempt
started with that the task's `answered_in` matches; and `answer`, the
answer the loop built (the final reply's text and the confidence) or
`null`. It is written as ASCII JSON, any other character escaped. A check
finds it as `"$(dirname "$0")/conversation.json"`, and scores everything:
the files, the trace of tool calls and the confidence. The texts the task's
pointers put `for` the check are there too, by their `path`.

Since the check runs from `/work`, the files the model left are in its
current folder. A check must not start an interpreter that loads code from
the current folder: run Python as `python3 -P` (or `-I`), and call programs
and scripts by absolute paths. Otherwise a `conftest.py`, `unittest.py` or
similar file the model planted would run as root inside the check and could
print any score it likes. Its standard output, without the white space
around it, must be one JSON object of named parts, each a number from 0 to
1, `correct` among them: the task's own judgement of the files and the
answer, next to the rules it reports, such as `{"correct": 0.8,
"read_before_write": 1, "sources_first": 0.5}` (`wpt_check.report` prints
it). A bare number from 0 to 1 reads as `{"correct": n}`, so a check that
prints one number keeps working. Its exit code and standard error are not
read. A check that prints anything else (a part outside 0 to 1, a `true`,
no `correct`), crashes or runs out of time gives no score; the attempt
counts as 0, and the step prints a line naming the task and why. A task
with an `area` gets one more part, `within_area`, from the harness
(`wpt_agent.play`): 1 when every path under `/work` the attempt changed,
added or removed matches one of the area's patterns, 0 otherwise. The
attempt's `score`, what its line and the measures keep, is its `correct`
part, or 0 when a gate of the task is broken (`wpt_agent.gated`).

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
nothing and is skipped. The step goes on, unless every attempt of a
round was dropped: that round makes no update, and the step fails, naming
the round, how many attempts Docker failed and its last error. A round
with at least one update goes on as usual. A step that made no update at
all fails too: every group was skipped, and the step says how many because
their rewards were all equal and how many because they kept fewer than 2
attempts. An update whose groups were all skipped has nothing to learn and
is skipped (`training.skipped`).

### Rewards

An attempt's reward is its parts weighed together, with gates. The check
reports its parts; the step combines them, in `wpt_training/agentic.py`
(`combined`), which both task steps share, never the check:

- The reward is 0, whatever the rest, when the attempt has no answer (no
  final reply, or a reply to the confidence question that is not a number
  from 0 to 100), when its check gave no score, or when a gate of the
  task is broken: a part its `gates` names is below 1, or missing.
- Otherwise it is the weighted sum of `correct`; the calibration of the
  confidence, 1 - (confidence - correct)², one minus its Brier score
  (#40; paper 12 of the plan); each other part, a rule (the check's, and
  `within_area`); the number of tool calls, a cost on every call (#37,
  so the model does not open everything blindly); and, for a terse
  attempt, its closeness to the full attempt (below).
- The calibration and the tool calls are computed by the step from the
  attempt, not by the check: the confidence the harness collected
  (`metadata.answer`) and the tool calls of its conversation, so every
  attempt gets them, whatever its check reports.
- The weights are the step's settings (`weight_correct` 1,
  `weight_calibration` 0.5, `weight_rule` 0.25 for each rule,
  `weight_tool_call` -0.01 for each call, `weight_closeness` 0.25),
  starting values until experiments on the 1B model set them (#53).
  `weight_correct` must stay above `weight_calibration`, so stating a low
  confidence never pays more than being right: otherwise the run stops
  before any step runs (`agentic.check`). The `agent-tasks` environment
  takes these defaults from one place, `agentic.WEIGHTS`.

Terse against full (#39): a task that holds a one-line form (`terse`) is
played twice in each round of `finish-tasks`, first `group` attempts with its full
instructions, then `group` attempts with its one-line form, offered the
same tools attempt by attempt. Both forms' attempts are one group for the
advantage: with `group` 8, each of the 16 attempts' advantage is its reward
minus the mean reward of all 16. Only the terse attempts carry the
closeness part. Terse attempt *i* is compared with full attempt *i*, the
one of the same index: its reward adds, weighed by `weight_closeness`, how
alike their changes to `/work` are (twice the lines their diffs share, in
order, over the lines of both, `wpt_agent.changes`; 1 when neither changed
anything), next to passing the task's check as any attempt does. The
model's own full result is the reference, which the rule on WPT models in
training allows ([data-selection.md](data-selection.md), "Models in
training"). A task whose `area` is empty only answers, and gets no
closeness part. On any other task, a terse attempt whose full attempt
Docker dropped has nothing to be compared with: it is kept in the attempts
file with a `null` reward but left out of the group, since its reward would
lack the part every other terse attempt of the group has. As for any group,
one left with fewer than 2 attempts is skipped, and one whose rewards are
all equal teaches nothing.

The group's size matters (#40; paper 13 of the plan): normalizing by the
group can erase a penalty the weights were designed to give when few
distinct answers appear in a group. The advantage here is the reward
minus the group's mean, never divided by its standard deviation, and a
group left with fewer than 2 attempts is skipped (below); the group size
is set by experiment with the weights.

### Learning

Each attempt's advantage is its reward minus the mean reward of its group
(the attempts at the same task in the same round, both its forms for
`finish-tasks`'s terse tasks). It is not divided by the group's
standard deviation: that would make a group whose scores barely differ
weigh as much as one whose scores differ widely, and give tasks that are
almost always passed or failed the largest steps. The loss is minus the
advantage times the log-probability of each token the model wrote, the
same tokens `answer-as-assistant` learns (never the instructions, the tool results or the
template's own tokens), averaged over every token the model wrote in the
update's attempts; a skipped group (below) adds none. The tokens are the ones the model sampled, kept as they
were, since writing the conversation as tokens again would not always give
the same ones. There is no frozen reference model and no KL penalty: each
update learns from attempts the current model just made. A group whose
rewards are all equal teaches nothing and is skipped. The model samples
from its own probabilities (`temperature` 1, `top_k` 0: every token),
which the loss assumes.

The routed expert layers ([model.md](model.md), "Experts") choose experts
for each token while the model writes an attempt. The step records those
choices, each attempt's apart, as the generation cache reads its tokens,
and the loss pass routes each token through the same experts
instead of choosing again (routing replay, "R3"), so the loss scores the
tokens through the experts that wrote them. The routers' weights do not
change during the step, and their balancing biases do not either: the
step locks both. The
hash-routed layers choose from the token alone and need neither. The CSA
indexers go on learning by their own loss in each loss pass, as in
`answer-as-assistant`. The multi-token prediction block is not used, so it
does not change.

### Runs

The order of the tasks in each round and every sampled token come from
generators seeded with the recipe's `seed` and the step's number in the
recipe, so a run gives the same attempts again on
the same machine and setup (the same model, tasks, images and devices); the
checks themselves may differ, as any program may. The model alone writes
the attempts, without a drafter: the weights change at every update, the
drafter is trained only at the end of the run ("The drafter", above), and
the attempts then depend on the seed alone, never on this machine's times.
A task's attempts are written together in one batch, from one reading of
the task's opening (model.md, "Generation"). The step prints one line per
round (`rounds`, passes over every task): the mean reward, the number of
attempts and of updates, and the indexers' mean loss over the round's loss
passes.

Every attempt is kept in the run's record folder
([observability.md](observability.md)), as one line of
`<step>-attempts.jsonl` (`finish-tasks-attempts.jsonl`), written through
the run's record as soon as the attempt is scored, so a line on a local store is on the disk before the next attempt
is scored. A group's attempts are written together, then scored one by one,
in order, each attempt's container removed and its line kept before the
next attempt is scored (each attempt's line is kept as soon as that attempt
is scored, even though attempts are written together). Each line is a
conversation in the chat format ([chat-format.md](chat-format.md)) whose
`metadata` says which attempt it is, so the file passes the same check as
any `chat` file:

```json
{"tools": [...], "messages": [...],
 "metadata": {"tools": {"shell": "bash", "...": "..."}, "answer": {"answer": "...", "confidence": 0.5},
              "step": "finish-tasks", "task": "<task id>", "round": 1, "group": 3, "attempt": 2, "terse": false,
              "dropped": false, "error": null, "ended": "answered",
              "parts": {"correct": 1.0, "sources_first": 1.0}, "score": 1.0, "reward": 1.66,
              "written": 412, "tool_calls": 4}}
```

- `tools`, `messages`: the attempt: the tools it was offered, as the
  model saw them, the instructions, every assistant turn as the model's
  tokens read, every tool result as the model received it, and after the
  final reply the confidence question and the model's reply to it. A dropped
  attempt has only the tools and the instructions, the conversation it
  would have started from.
- `metadata.tools`: each name the model saw, mapped to the real tool's;
  `metadata.looking`: the real names of the offered tools that look at
  sources; `metadata.answered_by`: the task's; `metadata.answered_in`:
  the starting files the task's `answered_in` matches; `metadata.answer`: the
  answer the loop built, the final reply's text and the confidence, `null`
  without a final reply or a confidence from 0 to 100, and absent for a
  dropped attempt.
- `step`, `task`: the step and the task's id.
- `round`: the round, from 1; `null` for an attempt made before the rounds
  (the `finish-tasks-with-less-writing` step's reference, below).
- `group`: the group within the round, from 1, in the order played (for a
  reference attempt, the task's place in the list of tasks); `attempt`:
  the attempt within its group, from 1; `terse`: whether it opened with
  the task's one-line form ("Rewards", above).
- `dropped`: `true` when Docker failed; then `error` says how, and
  `ended`, `parts`, `score`, `reward`, `written` and `tool_calls` are
  `null`.
- `ended`: why the attempt ended (`answered`, `ended`, `turns`, `tokens`,
  `context` or `full`; Attempts, above).
- `parts`: the check's parts, with `within_area` for a task with an area,
  `null` when the check gave no score; `score`: its `correct` part, 0
  with a broken gate ("Scores", above); `reward`: what the attempt is
  worth to the step (for `finish-tasks`, its parts weighed together, "Rewards",
  above), `null` for an attempt never trained on.
- `written`: the tokens the model wrote, the tokens the loss learns;
  `tool_calls`: the tool calls it made.

The trace names each line: the attempts of a group are written, scored
and kept in a span `<step>.play` (`finish-tasks.play`: `task`, `round`, `group`, `attempts`,
`terse`);
inside it, each attempt's line is kept in a span `<step>.attempt` (`task`, `round`, `group`, `attempt`, `terse`, then `file` and
`line`, its line number from 1, with `dropped`, `ended`, `score`,
`reward` and `written`); each group is an event `<step>.group` (`task`,
`round`, `group`, `terse`: whether it holds the one-line form's attempts too, `file`, `lines`, and `outcome`: `learned`, or why it was
skipped); each round is an event `<step>.round` (`file`, `lines`: its first
and last, `attempts`, `dropped`, `updates`, and the value `mean_reward`). A
recipe that runs the step twice adds to the same file, and the line
numbers go on. A line the store refuses stops the command at once with
exit code 3, as any part of the record does.

Its settings are the `agent-tasks` environment's, their defaults: `tag`
(`finish-tasks`), `rounds` (10), `tasks_per_update` (4), `group` (8),
`terse` (true for this step: a task with a one-line form is played in
both forms), `reward` (`weighed`), `reference` (0), `temperature` (1),
`top_k` (0, every token), `check_seconds` (300), `result_bytes` (4000)
and the five weights ("Rewards", above). Its optimizer is
`adamw` at a learning rate of 1e-6, without weight decay. A recipe's
entry changes them ("Steps", above). The containers' sizes are not among
them: they belong to each task.

## The finish-tasks-with-less-writing step

The `finish-tasks-with-less-writing` step comes after `finish-tasks` in a
recipe's steps. Its goal is the same result as the model it starts from,
with fewer tokens. It trains on the tasks `finish-tasks` reads, those
tagged `finish-tasks`, with the same images, containers, limits and
checks, and the same learning (Attempts, Scores, Learning and Runs
above): groups of `group` attempts, in each task's full form only (it
plays no one-line form), the advantage as the reward minus the group's
mean, the loss over the tokens the model wrote, R3 routes and routers that
do not change, dropped attempts and the same failures. Only the reward
differs: the step is the `agent-tasks` environment with `reward`
`shorter` and `reference` 8.

First, the reference. Before any update, the incoming model makes
`reference` attempts at each task, written together, run and scored as
`finish-tasks`'s are. An attempt's score here is its result: its
`correct` part, 0 when a gate is broken, its check gave no score or it has
no answer (`agentic.result`). A task's best start score is the best score
among them. An attempt passes when its score is at least that best score,
and the task's reference length is the mean length of its passing
attempts. A length counts only the tokens the model wrote, the same tokens
the loss learns, never the instructions, the tool results or the
template's own tokens. The reference stays fixed for the whole step and is
kept with the step's state, so a resumed step does not play it again; the
step does not train on its attempts. A task whose every reference attempt
scored 0 has no result to keep and is left out, and so is a task whose
every reference attempt Docker failed; the step prints a line for each.
When no task is left, the step fails.

Then the rounds, as `finish-tasks`'s, with this reward for each attempt:

- 0 when it fails: its result, the gated `correct` part
  (`agentic.result`), is below the task's best start score;
- when it passes, its parts weighed together as `finish-tasks` weighs them
  ("Rewards", above, with this step's own weight settings; the parts are
  combined in the step that learns, never in the check), times the
  task's reference length divided by the attempt's length: the weighed
  parts once for an attempt as long as the reference, twice for one half
  as long, half for one twice as long.

Every passing attempt whose weighed parts are above 0 is worth more than
every failing one, and of two passing attempts whose parts weigh the same
the shorter one is worth more. The step prints one line per task for the
reference (its best start score and reference length) and one line per
round (the mean reward, the number of attempts and of updates). Sampling
draws from one generator seeded with the recipe's `seed` and the step's number, for the reference and then
the rounds. Every attempt is kept as `finish-tasks` keeps its own (Runs,
above), in `finish-tasks-with-less-writing-attempts.jsonl`, under the spans
and events `finish-tasks-with-less-writing.attempt`, `.group` and
`.round`: the reference attempts first, with `round` and `reward` `null`
in their `metadata`, then the rounds' attempts with their reward (the
round's value is `mean_reward`). Its settings are those of `finish-tasks`
("Runs", above), with `terse` false, `reward` `shorter` and `reference` 8.

## What is exactly repeatable

**The data is exactly repeatable from the same store**, for audits: the
same stored bytes, cuts, patches, vocabulary and code give the same tokens.
A rebuild from upstream is not: a new download takes what upstream serves
when asked (a git source the commit it names, while upstream still serves
it), and the dataset's document list (each document's id and the
sha256 of its Markdown) records what a model trained on. Shards are compared by their token hash, which covers
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
TileLang, Triton or PyTorch computes the attention. The updates themselves are the same: each update's
examples, and any random draw a model makes, come from the seed and the update, and a resumed step
continues the same updates; a reused step's result stands for its inputs, whatever machine trained
it. The attempts of a task step depend on the model's own sampling, and the checks may differ, as
any program may. Around it, everything possible is checked:

- before training, every input is checked against its hash: the dataset
  record, each shard (file sha256 and token hash), and the vocabulary;
- `provenance.json` records what produced the model: the dataset id and
  record, the sources it learned from, the vocabulary, every shard, the
  recipe's name and content, the command-line options, the strategy (file,
  its sha256), the model shape and parameter count, each step (its files
  and their sha256, its settings, its optimizer's, its locked parts, its
  fingerprint and result, whether it was reused, its stages with the last
  update each made, the updates made and skipped, the update whose
  weights it left, which the model card shows, its best validation loss
  and the CSA indexers' loss at that update, and the sources it learned
  from), the repository commit (and whether the working tree had
  uncommitted changes), the device and the tool versions.
