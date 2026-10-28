# WPT: Workers' Pre-trained Transformer

WPT is an open language model, trained from scratch only on content whose
license is clear, and published under copyleft (AGPL-3.0-or-later). It is
the first project of Workers' Intelligence Commons: intelligence is a means
of production, and it should stay a common good that anyone can use, host,
modify and improve. Anyone may serve WPT, for money too; whoever serves a
modified version must share it. The principles are in the
[charter](docs/charter.md).

This repository is everything needed to build a WPT model from nothing:
the description of every source, the tools that fetch and check them, and
the code that builds the data, trains, evaluates and runs the
model. No checkpoint has been published yet.

## How it works

**The repository describes a corpus; it does not hold one.** Each source
is a `data/sources/<id>/source.json`: the address its content is fetched
from, the license of every path with the place where its license
statement must be found and the credit it asks for, how each file becomes
Markdown, and the signed human decisions that admit or refuse it. The data
itself is downloaded into a store you control (a folder or an S3 bucket),
as upstream serves it when asked, and every build checks each license
statement against the bytes the store holds. A rebuild gives today's
content, license-checked, not the same bytes as an earlier one. The
document list a model's dataset keeps (the sha256 of each document's
Markdown) records what that model trained on. WPT hosts no data.

- **Clear licenses only.** A source enters only when its license lets the
  weights be distributed under the AGPL; when in doubt, it is refused
  ([docs/licenses.md](docs/licenses.md)). Owners' "no AI training" wishes
  and robots.txt are honored.
- **No distillation.** Trained from scratch, with no output of another
  company's model in the data and no existing base model.
- **Provenance travels with the model.** Every model folder carries a
  `NOTICE` naming each source it learned from, with its credit, and a
  `provenance.json` recording exactly what produced it.
- **Erasure on request**, through recorded cuts that apply to every later
  build ([docs/erasure.md](docs/erasure.md)).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), git and a Rust toolchain
(the tokenizer is built from source). Mercurial only for Mercurial
sources; GnuPG only to check GPG-signed decisions (on Windows, the
gnupg.org build). On Windows, run WPT in Windows itself (PowerShell), not
inside WSL: a second system adds failures of its own and keeps files on a
disk Windows does not see. Linux containers, where a step needs them, run
through Docker Desktop.

```sh
uv run scripts/setup.py                            # the PyTorch build for this machine's GPU
echo "WPT_STORE=file://$HOME/wpt-store" >> .env    # where fetched and built data lives

uv run wpt-dataset download                        # every admitted source, into the store
uv run wpt-dataset build classics-test --stream    # convert, tokenize; prints a dataset id
scripts/detach.sh uv run wpt-train classics-test <dataset id> --stream --out runs/first
uv run wpt-eval runs/first                         # memorization and other tests, into the model card
uv run wpt-chat runs/first                         # continue text with the model
```

`classics-test` is a small model recipe over the admitted public-domain
books, enough to see every step work on one machine. A model recipe
(`data/recipes/models/`) holds everything that changes the model: its
data, architecture and training steps
([model.md](docs/model.md)); the command-line options only say how the
hardware runs it.

**Streaming** keeps disk use small at each step, for data larger than the
disk:

- `wpt-dataset download --stream` keeps no copy on the downloading
  machine of what an S3 store holds.
- `wpt-dataset build --stream` takes each source from the store into a
  temporary folder, turns it into tokens and deletes the folder; the store
  keeps their tokens, not their Markdown. The dataset and its id are the
  same as without it.
- `wpt-train --stream` reads the training shards a few at a time (8 by
  default, about 2 GB, `--stream-shards`) instead of copying them all to
  disk first.

**Long runs** start detached (`scripts/detach.sh` on Linux and macOS,
`.\scripts\detach.ps1` on Windows), so closing the terminal or the SSH
session does not stop them; Ctrl+C or a shutdown writes a checkpoint, and
the same command with `--resume` continues.

**Bigger setups:**

| To | Use |
| --- | --- |
| share data and runs between machines | `WPT_STORE=s3://bucket/prefix` (any S3-compatible service) |
| move a run to another machine | `--run NAME`: the checkpoint lives in the store; `--run NAME --resume` there |
| use several GPUs | `--strategy ddp --processes N`, on the gpu worker's machine |
| train on another machine's GPUs | a gpu worker on each machine (`deploy/gpu-worker/compose.yaml`, with `compose.nvidia.yaml` or `compose.amd.yaml` for GPUs), their keys in `WPT_GPU_WORKERS` ([machines](docs/machines.md)) |
| rent the GPUs | `WPT_GPU_WORKERS=N` and `WPT_RENTAL_GPUS` for `wpt-manager`, which rents N gpu workers through SkyPilot (as one cluster in one request), replaces a lost one and gives them back at the end ([machines](docs/machines.md)) |
| download, build and train in one command | `docker compose up`, with `COMPOSE_FILE=deploy/manager/compose.yaml` and `WPT_RECIPE` set in `.env`; then `wpt-store copy-out NAME runs/NAME` ([machines](docs/machines.md)) |
| build despite a broken source | `wpt-dataset build --skip-failed`: it is left out and listed (a source not downloaded always is) |
| teach the model to chat | conversations tagged `answer-as-assistant` in a source, downloaded (`wpt-dataset download`) and the dataset built again, and a recipe whose `steps` add `answer-as-assistant` after `read-text`: `wpt-train RECIPE DATASET_ID --out runs/chat`, then `wpt-chat runs/chat --chat` ([training](docs/training.md)) |
| teach the model to finish tasks as an agent | tasks tagged `finish-tasks` in a source, downloaded and the dataset built again, their images in `data/images/` locked with `wpt-agent lock-image NAME` and built with `wpt-agent build-image NAME`, and a recipe whose `steps` hold `finish-tasks` (needs Docker; [training](docs/training.md)) |
| teach the agent to reach the same result with fewer tokens | `finish-tasks-with-less-writing` after `finish-tasks` in the recipe's `steps`: it trains on the same tasks ([training](docs/training.md)) |
| train long windows on a small GPU | `--recompute`: layers run again in the backward pass instead of keeping their values |
| write your own texts, conversations and tasks, and edit recipes, in a browser | `wpt-web` (needs Bun), then http://127.0.0.1:8410/; the same through `wpt-dataset` commands ([architecture](docs/architecture.md), "Two interfaces") |

Every option also comes from the environment (`--micro-batch` is
`WPT_MICRO_BATCH`), and `--help` lists them all. Details:
[training](docs/training.md), [architecture](docs/architecture.md).

## Contributing

Every commit is signed off (`git commit -s`), and `uv run scripts/check.py`
runs every check CI runs before you push. A new source starts under review
and enters only when a maintainer admits it in a signed commit; see
[CONTRIBUTING.md](CONTRIBUTING.md). License questions the project cannot
settle alone are listed for legal review in
[LEGAL-QUESTIONS.html](LEGAL-QUESTIONS.html).

## Documentation

- [Charter](docs/charter.md)
- [Architecture: sources, manifests, patches, store, builds](docs/architecture.md)
- [Licenses and what the corpus accepts](docs/licenses.md)
- [Questions for legal review, in English and French](LEGAL-QUESTIONS.html)
- [Erasure requests](docs/erasure.md)
- [Data selection: tags, pools, mixtures, held-out text](docs/data-selection.md)
- [Tokenizer](docs/tokenizer.md)
- [Model architecture: the research it comes from and the recipes that size it](docs/model.md)
- [Training and repeatability](docs/training.md)
- [Machines for training: the wpt manager, gpu workers, sandboxes](docs/machines.md)
- [Evaluation](docs/evaluation.md)
- [Observability: the record of every run, and its live view](docs/observability.md)
- [Chat format: conversations as JSON, turned into tokens by a template](docs/chat-format.md)
- [Output notice and attribution](docs/output-notice.md)
- [Publishing models: releases, names, where](docs/publishing.md)

## License

Code, documents and the weights of WPT models: the
[GNU Affero General Public License, version 3 or later](LICENSE). Text a
WPT model generates: the authors of WPT waive their rights in it under
CC0-1.0 ([OUTPUT-NOTICE](OUTPUT-NOTICE), pending legal review), but
anything it repeats from its training data or its context keeps its own
license. Data built by WPT's tools (token shards, vocabularies, dataset
records) is under the AGPL as well, while each source's content inside it
keeps its own license. [NOTICE](NOTICE) says where to find a given model's
notice.
