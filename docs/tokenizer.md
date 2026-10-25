# Tokenizer

WPT has its own byte-level BPE tokenizer (`packages/wpt-tokenizer`), with its
hot loops in Rust (PyO3, built by maturin when the workspace is installed).

- **Every byte has a token** (ids 0-255), so encoding never drops anything.
- Id 256 is the end-of-document token.
- Ids 257 to 512 are reserved for special tokens, such as the chat
  template's tags ([chat-format.md](chat-format.md)). Encoding text never
  produces them, and decoding gives them no bytes.
- Each learned merge adds the next id from 513 on.
- Text is first cut into chunks by ASCII whitespace only (a chunk is a word
  with at most one leading space, or a run of whitespace), so results never
  depend on a Unicode version.
- Each thread that encodes keeps its own cache of encoded chunks (up to
  about a million, then it starts again empty), so threads encoding at
  once never wait for each other, and the tokens never depend on the cache.
  A thread's cache goes when the thread ends.

## Vocabularies

A vocabulary (the list of merges) is a build result: one per version of a
tokenizer recipe, shared by every dataset and model that uses it.

`data/recipes/tokenizers/<name>.json`:

```json
{ "schema": 2, "version": 1, "mixture": "<mixture name>", "vocab_size": 32000, "note": "" }
```

### The lock freezes it

The first build of a recipe version trains the vocabulary on the documents
its mixture selects (each once), and writes
`data/recipes/tokenizers/<name>.lock.json`: the recipe, the tokenizer code's
version, every document it was trained on (`[id, sha256 of its Markdown]`)
and the vocabulary's sha256. Commit it. From then on, every build of that
version uses that vocabulary:

- **Adding or changing sources never retrains it.** Many datasets and
  models, whatever their training settings, share one vocabulary, so a
  model can keep training on new data. A source the vocabulary was not
  trained on still encodes exactly, since every byte has a token, only less
  compactly; `wpt-dataset tokenizer-check <name>` shows by how much.
- **A new vocabulary takes a new version.** For a big change (much more
  data, another size) or a change of the tokenizing algorithm, raise
  `version` in the recipe: the next build trains a new vocabulary and
  replaces the lock (git history keeps the old one). Editing the recipe
  (other than its `note`), or a new tokenizer code version, without
  raising `version` stops the build (`TokenizerVersionNeeded`), so one name
  and version always mean one vocabulary.
- **Every machine gets the same vocabulary.** A store that lacks it trains
  it again from the documents the lock lists, and the result must have the
  locked sha256 (`LockMismatch` otherwise). Training is deterministic: it
  runs on one thread, and when two pairs are equally frequent the pair with
  the smaller ids wins. When a listed document is no longer available (its
  source changed or was cut), that store cannot rebuild it
  (`LockedDocumentsMissing`): copy the vocabulary from a store that holds
  it, or raise `version`.
- **Erasure does not reach a locked vocabulary** until `version` is raised:
  merges learned from text that was erased later stay in it (see
  [erasure.md](erasure.md)).

The tokenizer code's version is in `packages/wpt-tokenizer/rust/Cargo.toml`.
Bump it whenever a code change could alter training or encoding results.

### Where the vocabulary is checked

- The dataset build uses only the locked vocabulary; the dataset record
  names the recipe, its version and the vocabulary's sha256, and the
  dataset id covers it.
- Training reads the vocabulary from the store, checked against that
  sha256, and writes it into the model folder as `tokenizer.json`. Every
  weight file's header names it: `tokenizer` (such as
  `classics version 1`) and `tokenizer_sha256`.
- `wpt-chat`, `wpt-eval` and `wpt-posttrain` load `tokenizer.json` only when
  its sha256 is the one in the weights' headers (`TokenizerMismatch`
  otherwise), so a model never runs with another vocabulary.
  `wpt-posttrain` keeps the base model's vocabulary and name.

### Checking a vocabulary on the data

`wpt-dataset tokenizer-check <name>` encodes then decodes every document of
every admitted source with the locked vocabulary: each must give back its
exact bytes (`RoundTripFailed` otherwise). It prints, per source, the
documents, characters, tokens and characters per token, and marks the
sources the vocabulary was not trained on.

A vocabulary file is canonical JSON:

```json
{"format":"wpt-bpe","merges":[[a,b],...],"tokenizer_version":"0.3.0"}
```

A vocabulary file written by another tokenizer version is refused, since
its ids would mean other tokens: tokenizer 0.3.0 reserved ids 257 to 512,
so vocabularies of 0.2.0, and the models trained on them, need a new
vocabulary version and a new model.
