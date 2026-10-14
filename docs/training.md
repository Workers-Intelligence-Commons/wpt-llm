# Training and repeatability

## Training

```sh
uv run wpt-train <dataset id> --n-layer 4 --n-head 4 --n-embd 128 --block-size 256 --steps 3000
```

`wpt-train` takes a built dataset id and never builds: build and training
fail separately. Every setting is a command-line option or an environment
variable (`WPT_` + the option name in capitals, `--n-layer` ->
`WPT_N_LAYER`; the dataset id may come from `WPT_DATASET`); there is no
training recipe file. `uv run wpt-train --help` lists them.

Training stops early when the smoothed validation loss stops improving, and
keeps the best checkpoint. Validation uses the dataset's held-out shards.

The model folder holds `config.json`, the weights (safetensors, optionally
split with `--max-shard-size`), `tokenizer.json` and `provenance.json`.

`wpt-chat <model folder>` continues text typed by the user, until the model
ends the document. A pre-trained model only continues text; a chat format
comes with post-training.

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
weights. Around it, everything possible is checked:

- before training, every input is checked against its hash: the dataset
  record, each shard (file sha256 and token hash), and the vocabulary;
- `provenance.json` records what produced the model: the dataset id and
  record, the vocabulary, every shard, all options, the model shape and
  parameter count, the repository commit (and whether the working tree had
  uncommitted changes), the device and the tool versions.
