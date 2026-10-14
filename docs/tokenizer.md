# Tokenizer

WPT has its own byte-level BPE tokenizer (`packages/wpt-tokenizer`), with its
hot loops in Rust (PyO3, built by maturin when the workspace is installed).

- **Every byte has a token** (ids 0-255), so encoding never drops anything.
- Id 256 is the end-of-document token.
- Each learned merge adds the next id from 257 on.
- Text is first cut into chunks by ASCII whitespace only (a chunk is a word
  with at most one leading space, or a run of whitespace), so results never
  depend on a Unicode version.

## Vocabularies

A vocabulary (the list of merges) is a build result, trained once per
tokenizer recipe and shared by every dataset and model size that uses it.

`data/recipes/tokenizers/<name>.json`:

```json
{ "schema": 1, "mixture": "<mixture name>", "vocab_size": 32000, "note": "" }
```

The documents selected by the mixture's pools (each once) train the
vocabulary. Its fingerprint covers the tokenizer's version
(`packages/wpt-tokenizer/rust/Cargo.toml`), the recipe, the mixture and the
sha256 of every selected document.

Training is deterministic: it runs on one thread, and when two pairs are
equally frequent the pair with the smaller ids wins. The first build records
the vocabulary's sha256 in `data/recipes/tokenizers/<name>.lock.json`; commit
that file. A later build with the same fingerprint that gives another
vocabulary fails (`LockMismatch`).

Bump the tokenizer's version whenever a change could alter training or
encoding results.

A vocabulary file is canonical JSON:

```json
{"format":"wpt-bpe","merges":[[a,b],...],"tokenizer_version":"0.2.0"}
```
