# WPT: Workers' Pre-trained Transformer

WPT is an open language model, trained from scratch only on content whose
license is clear, and published under copyleft (AGPL-3.0-or-later). It is
the first project of Workers' Intelligence Commons: intelligence is a means
of production, and it should stay a common good that anyone can use, host,
modify and improve. Anyone may serve WPT, for money too; whoever serves a
modified version must share it. The principles are in the
[charter](docs/charter.md).

This repository is everything needed to rebuild a WPT model from nothing:
the description of every source, the tools that fetch and check them, and
the code that builds the data, trains, evaluates, post-trains and runs the
model. No checkpoint has been published yet.

## How it works

**The repository is a lockfile, not a corpus.** Each source is a
`data/sources/<id>/source.json`: its exact address, revision and checksum,
the license of every path with the credit it asks for, how each file
becomes Markdown, and the signed human decision that admitted or refused
it. The data itself is fetched on demand into a store you control (a
folder or an S3 bucket) and checked against its checksum, so anyone can
rebuild exactly the same dataset. WPT hosts no data.

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
gnupg.org build).

```sh
uv run scripts/setup.py                            # the PyTorch build for this machine's GPU
echo "WPT_STORE=file://$HOME/wpt-store" >> .env    # where fetched and built data lives

uv run wpt-dataset build classics-test --stream    # fetch, convert, tokenize; prints a dataset id
scripts/detach.sh uv run wpt-train classics-test <dataset id> --stream --out runs/first
uv run wpt-eval runs/first                         # memorization and other tests, into the model card
uv run wpt-chat runs/first                         # continue text with the model
```

`classics-test` is a small model recipe over the admitted public-domain
books, enough to see every step work on one machine. A model recipe
(`data/recipes/models/`) holds everything that changes the model: its
data, architecture, training and post-training steps
([model.md](docs/model.md)); the command-line options only say how the
hardware runs it.

**Streaming** keeps disk use small at both ends, for data larger than the
disk:

- `wpt-dataset build --stream` fetches one source at a time, turns it into
  tokens and deletes its files before the next; the store keeps only
  tokens. The dataset and its id are the same as without it.
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
| use several GPUs | `--strategy ddp --processes N` (several machines: `torchrun`) |
| build despite a broken source | `wpt-dataset build --skip-failed`: it is left out and listed |
| teach the model to chat | a recipe whose `posttrain` list holds `sft`: `wpt-posttrain RECIPE runs/first runs/chat`, then `wpt-chat runs/chat --chat` |
| train long windows on a small GPU | `--recompute`: layers run again in the backward pass instead of keeping their values |
| review sources in a browser | `wpt-web` (needs Bun), then http://127.0.0.1:8410/ |

Every option also comes from the environment (`--micro-batch` is
`WPT_MICRO_BATCH`), and `--help` lists them all. Details:
[training](docs/training.md), [architecture](docs/architecture.md).

## Contributing

Every commit is signed off (`git commit -s`), and `uv run scripts/check.py`
runs every check CI runs before you push. A new source starts refused and
enters only when a maintainer admits it in a signed commit; see
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
- [Evaluation](docs/evaluation.md)
- [Post-training](docs/post-training.md)
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
