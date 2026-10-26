# Machines for training

A training run has three roles, which may share one machine or each have
their own:

- the **wpt manager**: the machine that runs `wpt-train` (or `wpt-manager`,
  below). It holds the store and its credentials, the tokenizer, the run's
  record, the checkpoint and the model folder, and it plays every agentic
  attempt;
- a **gpu worker**: a process that does only the GPU work of the training,
  for one manager at a time (`packages/wpt-worker`);
- the **sandboxes**: the Docker containers agentic attempts run in.

## One code path

`wpt-train` always trains through gpu workers. `WPT_GPU_WORKERS` lists
their keys, one gpu worker per machine. With none, `wpt-train` starts a
local worker on its own machine (`python -m wpt_worker --local`), reached
over the same link: the training runs the same way on one machine and on
several. A local worker ends with its manager, and its folder is
`WPT_WORKER_FOLDER` when set, else a temporary folder it removes.

## The gpu worker

The worker deals only in token ids: its job cannot import the tokenizer,
so nothing on it turns text into tokens or back. The manager encodes and
decodes, and sends the stop tokens the job needs.

Started with `python -m wpt_worker` (`deploy/gpu-worker/compose.yaml`; see
"Starting a gpu worker by hand"), it makes its own certificate, kept in its folder
so the certificate stays the same from one start to the next, and prints
its key:

```
wpt-worker://TOKEN@HOST:PORT#PIN
```

With `WPT_WORKER_PLACE` and `WPT_WORKER_PRIVATE` set (a rented cluster's
machines get them), the key also carries `?place=NAME&private=ADDRESS`
before the pin ("Several machines").

`TOKEN` is its secret token (`WPT_WORKER_TOKEN`, else a new one at each
start), and `PIN` is its certificate's sha256. A token set in
`WPT_WORKER_TOKEN` is printed as its first 4 characters and stars, never
whole. Give the key to the manager in `WPT_GPU_WORKERS`, with `HOST` the
address the manager reaches the worker at. The manager talks to it over
TLS and accepts only the pinned certificate. Its first frame carries the
token, and the worker closes any connection whose first frame does not.
One manager at a time: a second one is refused while the first is
connected.

A worker with no manager for `WPT_WORKER_WAIT` seconds (default 3600, 1
hour) ends. Its container then stays down, and a rented machine is given
back.

Every value that crosses the link is a JSON header and a body of bytes, so
neither side ever runs what the other sends.

### Starting a gpu worker by hand

`deploy/gpu-worker/compose.yaml` starts a worker on any machine with
Docker. The worker's server files are two things the person copies from the
repository to the machine, beside the Compose file: `start.sh`
(`deploy/gpu-worker/`) and the folder `wpt_worker`
(`packages/wpt-worker/src/`). Compose mounts both read-only from there into
the container. Without `wpt_worker`, `start.sh` says so and exits with
code 2. There is no mount of a checkout: the libraries the jobs need come
from the manager's checkout, which the manager sends (below).

The main file asks for no GPU and installs the CPU build of PyTorch. A GPU
kind adds a small file of its own, by the command or by a setting:

```sh
docker compose -f compose.yaml -f compose.nvidia.yaml up   # NVIDIA, CUDA build
docker compose -f compose.yaml -f compose.amd.yaml up      # AMD, ROCm build
COMPOSE_FILE=compose.yaml:compose.nvidia.yaml              # the same, in the environment or in a .env beside them
```

`compose.nvidia.yaml` reserves every NVIDIA GPU. `compose.amd.yaml` gives
the container `/dev/kfd` and `/dev/dri` and the `video` group. Each sets
`WPT_WORKER_TORCH`. The worker's own settings are `WPT_WORKER_TOKEN`,
`WPT_WORKER_PORT` (default 8470), `WPT_WORKER_ADDRESS` (default
`0.0.0.0`) and `WPT_WORKER_WAIT`.

## The checkout

At the start, the manager sends the worker its own checkout: every file
git tracks and every file it does not ignore, as the working tree holds it
(uncommitted changes included), without `data/`. The job runs from that
checkout, so the worker always runs the manager's code. With
`WPT_WORKER_INSTALL=1` (its container sets it), the worker installs the
libraries the checkout's `uv.lock` names, with the PyTorch build
`WPT_WORKER_TORCH` names (`cuda`, `rocm`, `xpu` or `cpu`), into its own
folder, for Python 3.14. The container's own Python and PyTorch are not
used.

A step's fingerprint holds the versions of the code it trains with: the
worker gives its libraries' versions (`torch`, `numpy`), and the manager
gives the repository's packages' versions, since it sends that code. So a
local and a remote worker give the same fingerprints for the same
recipe.

## The data path

The worker never holds the store's settings or credentials: a local worker
gets the manager's environment without `WPT_STORE` and `AWS_*`. Everything
the job reads, the manager sends:

- the objects of the run's dataset and of the states it starts from,
  checked against their sha256 when they arrive, and no other object;
- the files of the recipe's steps and strategy that live outside the
  repository.

Everything the job writes goes back to the manager:

- each line of the run's record, which goes to the manager's store before
  the job's call returns; the job's flushes are that store's, so on S3 the
  lines it holds back are sent when the job's flush returns
  ([observability.md](observability.md));
- the checkpoints, as `WPT_LINK_CHECKPOINTS` says (below);
- each step's end, which the manager keeps as the step's result;
- the drafter and the sample's token ids, from which the manager writes the
  model folder.

An environment that needs the manager, such as `agent-tasks`, which plays
attempts in sandboxes, has a part on the manager that the job calls; that
part asks the model on the worker to write tokens.

## Checkpoints

The job writes every checkpoint on the worker's disk, where a restart of
its processes reads it. A checkpoint, like every state the link carries, is
a safetensors file: the tensors are stored by name, and the rest of the
state (the optimizer's lists and numbers, a step's counters) is a layout in
the file's own metadata as JSON, with a layout of its own for the optimizer
state. No pickle is read from the link or from a file the link carried.
`WPT_LINK_CHECKPOINTS` says which ones come back to the manager:

- `step-ends` (the default): only each step's last checkpoint, sent with
  the step's end;
- `every`: each one as it is written.

A checkpoint that comes back is the run's checkpoint on the manager, in the
model folder or, with `--run`, in the store ([training.md](training.md),
"Checkpoints and resuming"). With `step-ends`, a checkpoint written inside
a step stays on the worker: a stop says so, and `--resume` on the manager
starts that step over. Each job keeps its checkpoints in a folder of the
worker named after the run's record folder, and the first start of a job
removes the folders earlier runs left there.

## Several machines

A run on several machines has one gpu worker per machine, each running
`--processes` processes, and every worker's processes train as one synced
run: each update waits for the gradients of every process. How the
workers exchange them depends on where they are:

- in one place (a home network, one provider's cluster), they meet through
  NCCL at full speed (gloo on the CPU), at gpu worker 0's private address,
  on the port after its own. A worker joins only peers at a private
  address, since NCCL's ports have no password, and nothing is added on
  top of the network. Each worker refuses a meeting address or a private
  address of its own that is not private, and keeps NCCL's and gloo's
  traffic to the network interface that has its own private address. The
  worker's container uses its machine's own network for that;
- across places, NCCL's ports never cross the public internet. Each
  machine's processes meet on that machine only, and the machines'
  gradients go directly between the gpu workers over their TLS links,
  through each worker's own port: gpu worker 0's job adds every machine's
  gradients up and sends the sum back. The other workers' jobs reach it
  with its pinned certificate and a secret the manager makes for the run;
  worker 0's own token never leaves the manager. What crosses is JSON and
  tensors' bytes, never pickle. This is the
  same math as one machine with as many processes, and GPUs wait for the
  slowest link. Only `--strategy ddp` runs that way.

A key may say where its worker is: `?place=NAME` and `&private=ADDRESS`,
before `#PIN`. Workers whose keys all give the same place are in one
place, and meet at worker 0's private address (else its address). Keys
that give no place are in one place when every address is private.
Otherwise the workers are in different places. Worker 0's meeting
address may be this machine's own (`127.0.0.1`) only when every worker's
is: the others would reach themselves there, so the run is refused.

Every link and every connection between workers notices a machine that
is gone without closing its connections (power cut, network cut) within
about 90 seconds, through TCP's own checks: its link breaks, and the
worker is lost.

A crashed process restarts the job's processes on one machine only. On
several machines, a crash ends the job on every worker, and `wpt-train`
exits with code 5. The manager resumes the run from the last checkpoint
like a lost worker ("A lost gpu worker").

## Rented gpu workers

`wpt-manager` rents its gpu workers itself when `WPT_GPU_WORKERS` is a
number: how many to rent. A provider is one small interface of ours, with
three operations: rent machines running gpu workers, give their addresses,
stop them (`apps/wpt-manager/src/wpt_manager/rental.py`). SkyPilot
(Apache-2.0, its `sky` command, in the manager's image) is one
implementation, from `deploy/gpu-worker/skypilot.yaml`. It starts the
worker with SkyPilot's own file mounts, which copy `start.sh` and
`wpt_worker` onto the machine. Code of our own for a provider SkyPilot
lacks or handles badly is another implementation of the same interface.

Several workers at one provider are rented as one cluster in one request,
when the provider supports it (SkyPilot does: `--num-nodes`). A cluster is
one place, named after the rental. Each node's `start.sh` computes the
node's private address from the cluster's address list and its rank, gives
it to the worker (`WPT_WORKER_PRIVATE`, with `WPT_WORKER_PLACE`) and prints
it, so the keys the manager holds carry `place` and `private`, and the
workers meet through NCCL at worker 0's private address. The cluster shares
one token, and is one rental: a lost worker replaces the whole cluster. A
provider that cannot rent a cluster is asked once per worker, and those
workers are in different places.

SkyPilot gets only its own provider keys: the shell's basics, the names of
the clouds it supports (`RUNPOD_`, `LAMBDA_`, `GOOGLE_` and the like), and
`WPT_WORKER_WAIT`, which it passes on to the workers. It never gets the
store's settings or keys (`WPT_STORE`, `AWS_*`, which are the S3 store's),
so a cloud that reads `AWS_*` reads them from SkyPilot's own files under
`HOME`.

The manager sets each rented worker's token itself and never prints it
whole: at most its first 4 characters, then stars. Rental kind is a
setting: on-demand by default, interruptible (spot) when chosen. A
provider's keys come from the environment or `.env`, as other settings
do.

The run's workers stay on the manager's disk (`WPT_MANAGER_STATE`, one file
per run, readable by its owner only), never in the store: each worker's
key, its token included, and for a rented one its provider and rental
name. A rental's name is kept before the provider rents it, so a rental a
crash cuts short is one the manager can still stop. The rentals are
stopped once the evaluation (or, without one, the training) ends, whatever
its outcome: each one is tried, and each failed try is recorded
(`manager.rental_stop_failed`).

A rental that could not be stopped stays named in its run's file. The manager
tries to stop every rental still named in any run file at each start and,
until none is left, every 300 seconds, each try a record of its own
(command `wpt-manager`, name `rentals`). It stays up trying, since a
machine costs money while it runs.

One manager at a time runs a run: it holds the run's lock in
`WPT_MANAGER_STATE` while it runs it, which the system releases when the
manager ends, crash included. Another manager sharing that folder leaves
a run whose lock is held, and refuses to start it (exit code 2).

## A lost gpu worker

`wpt-train` exits with code 4 when a gpu worker is lost: it cannot be
reached, refuses the link, or its link breaks. Its record names it
(`gpu_worker.lost`). The manager then looks at each worker: one that
answers on its port with its pinned certificate is there. Each worker it
rented that is not there is reported (`manager.worker_lost`), stopped,
and replaced by a new rental (`manager.worker_replaced`), and the training
runs again with `--resume`, from the last checkpoint in the store. A
rental of several workers is replaced whole.

A crash of a training process on several workers (exit code 5) resumes the
same way, on the workers that answer, and no rental is replaced when every
worker answers. A crash and a lost worker count against the same limit:
the run resumes at most `WPT_REPLACEMENTS` times (default 3) from one
start of the run to its end, and a manager that comes back goes on with
the same count. A lost worker that a person started is not replaced: the
run stops, and the message says to start it again.

## The manager comes back

When it starts, the manager first goes on with each of its own runs that
did not end: no other manager holds its lock, there is no model under the
run's name in the store, and the run's last `wpt-manager` record there has
no end. A run's file is marked ended once the run ends, whatever its
outcome, and the manager reads only the others. It reconnects to that
run's workers with the keys its disk holds; a rented worker that is gone
by then, such as one that waited its hour and ended, is lost, and
replaced. Then it starts the run the settings ask for.

In its Compose file, the manager's container exits when its runs are over,
finished or failed (`WPT_MANAGER_CONTAINER=1` makes a failed or refused run
exit with code 0; the failure is in its output and its record). Compose
restarts it only after a crash or a Docker restart (`restart: on-failure`),
and it then goes on with the run the crash cut short. `docker compose up`
again continues a failed run: with no `WPT_RUN`, the manager takes the name
in `wpt-run.txt` when it is a run of the same recipe whose model is not in
the store yet, and otherwise makes a new name.

## Evaluation

With `WPT_EVALUATE=1`, or by hand with `wpt-eval FOLDER`, the evaluation
runs its model on a gpu worker through the link. The tests stay on the
manager, which holds the tokenizer, the store and the sandboxes. A job
(`wpt_worker.evaluation`) loads the model folder's weights on the worker,
which the manager offers as files, and answers what the tests ask of the
model: a seeded generator, writers that write until a stop token or a limit
(the same ones a training attempt uses), and greedy writing for the
memorization measure. The tests cannot tell where the model is.

`wpt-eval` uses the first key in `WPT_GPU_WORKERS` (the manager passes it
the rentals it still holds, before it gives them back), else a local
worker. A number in `WPT_GPU_WORKERS` is refused: `wpt-eval` rents nothing.

## Sandboxes

The sandboxes run wherever the `docker` command reaches its daemon: the
manager's machine, or the Docker host `DOCKER_HOST` names. On a remote
Docker host, the host's cores, memory and idle cores are measured there,
through a short-lived container with no network
(`wpt_agent.host`), for the sizing rule of `wpt_utils.parallel`.

## The manager's Compose file

`wpt-manager` runs the whole pipeline as one command, each step its own
command:

1. `wpt-dataset download`: every admitted source into the store;
2. `wpt-dataset build RECIPE`: the dataset, then `wpt-agent build-image
   NAME` for each image the dataset's tasks name, where `DOCKER_HOST`
   reaches Docker (nothing without tasks);
3. `wpt-train RECIPE DATASET_ID --run NAME --resume`: the training, through
   the run's gpu workers (rented first when `WPT_GPU_WORKERS` is a
   number), its checkpoint in the store under the run's name;
4. `wpt-eval`, only with `WPT_EVALUATE=1`;
5. the model folder into the store, under the run's name; the manager's
   own copy of the folder is then removed.

A failed step stops the run with its exit code, and the same command with
the same run name continues from what is kept.

`deploy/manager/compose.yaml` runs it in a container, from the repository's
root:

```sh
docker compose up
```

The settings file is one file, the `.env` at the repository's root.
Compose reads it, with `COMPOSE_FILE=deploy/manager/compose.yaml` in it
(or `-f deploy/manager/compose.yaml --env-file .env` on the command line).
The manager reads the same file again inside the container for what its
environment leaves unset or empty, so `wpt-train`'s options (such as
`WPT_STRATEGY` and `WPT_PROCESSES`) come from there. Where a setting names
a path of the host and the container has its own, the container's value
wins (`WPT_STORE`, the manager's folders, `HOME`, `DOCKER_HOST`).

| Setting | Meaning |
| --- | --- |
| `WPT_RECIPE` | the model recipe; required, Compose refuses to start without it |
| `WPT_RUN` | the run's name; unset, the manager makes one and writes it into `wpt-run.txt` at the repository's root (git-ignored), with the line that copies its model out of the store |
| `WPT_EVALUATE` | `1`: evaluate the model at the end; off by default |
| `WPT_STORE` | the store; unset, a folder on the Compose volume `wpt-store`. A `file://` store must name a folder on a volume or a mounted folder: the manager refuses one in the container's own files, which go with the container |
| `WPT_STORE_FOLDER` | a folder of the machine, mounted at the same path in the container in place of the volume `wpt-store`; set `WPT_STORE` to `file://` and that path |
| `WPT_GPU_WORKERS` | the gpu workers' keys, or how many to rent; unset, a local worker in the container |
| `WPT_RENTAL_PROVIDER` | who rents the gpu workers; default `skypilot` |
| `WPT_RENTAL_KIND` | `on-demand` (the default) or `interruptible` |
| `WPT_RENTAL_GPUS` | what each rented machine has, in the provider's own words (SkyPilot: `H100:8`); required to rent |
| `WPT_RENTAL_WAIT` | seconds a rented worker has to give its address; default 1800 |
| `WPT_REPLACEMENTS` | how many times a run resumes after a lost gpu worker; default 3 |
| `WPT_WORKER_WAIT` | seconds a rented worker waits with no manager, then ends; default 3600 |
| `WPT_LINK_CHECKPOINTS` | `every` or `step-ends` |
| `DOCKER_HOST` | where the sandboxes run; unset, the machine's own Docker, whose socket the file mounts |

The file includes `deploy/otel-lgtm/compose.yaml`, so the live view starts
with it ([observability.md](observability.md), "The live view") and
`WPT_GRAFANA_PASSWORD` is required too. It mounts the checkout at `/repo`:
the manager runs that code and sends it to its gpu workers. The volume
`wpt-manager` holds the manager's own files: its runs' workers
(`WPT_MANAGER_STATE`), the model folders before they go into the store,
the libraries, and its `HOME`, where SkyPilot keeps its clusters and keys.

## The final model

The model folder goes into the store under the run's name: each file is an
object, and the ref `models/NAME` names the folder's manifest, which lists
each file's sha256 and size. At the end, the manager prints the line that
copies it out:

```sh
uv run wpt-store copy-out NAME runs/NAME
```

`wpt-store copy-out` writes the folder into a new folder, beside it first,
then renames it into place, and checks every file against its sha256 and
size: a file that differs stops it and leaves no folder. `wpt-store list`
lists what the store database says: every session and each source's last
outcome and download.
