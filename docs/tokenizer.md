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
- Training reads every document, but counts the chunks in a table of a
  fixed number of slots (Space-Saving: Metwally, Agrawal and El Abbadi,
  2005), so the memory does not grow with the corpus. A chunk already in
  the table has its count raised; a new chunk takes a free slot with count 1;
  when the table is full, the slot with the smallest count is evicted and
  the new chunk takes it with the evicted count plus 1, recording the evicted
  count as its error. A chunk's true count is then between its count less
  its error and its count, and every chunk more frequent than the smallest
  count is in the table. Equal counts are told apart by the chunk's bytes,
  so the same documents in the same order always give the same table and
  the same vocabulary. The merges are learned from the table's chunks and
  counts. While the documents have no more distinct chunks than the slots,
  no chunk is evicted, every count is exact and the order of the documents
  does not matter. The number of slots (50,000,000) is fixed in the
  tokenizer's code, never measured from the machine, so a vocabulary never
  depends on the machine that trained it; it keeps reading and learning
  together within about 25 GB.
- The learning step uses every core of the machine: the merges come one
  after the other, and the work inside each one is shared between threads.
  It gives the same vocabulary as on one core, whatever the number of cores.

## Vocabularies

A vocabulary (the list of merges) is a build result: one per version of a
tokenizer recipe, shared by every dataset and model that names it.

`data/recipes/tokenizers/<name>.json`:

```json
{ "version": 1, "sample_bytes": 10000000000, "vocab_size": 32000, "note": "" }
```

`sample_bytes` is the size of the sample that trains it, in file bytes.

### The lock freezes it

A dataset recipe names its vocabulary. The first build of a dataset that
names a recipe version with no lock draws a seeded sample of its own chosen
documents, each pool taking its weight of `sample_bytes` at most, as the
start of its chosen documents up to a whole frame's group (the unit the
choice draws, [data-selection.md](data-selection.md)), trains the
vocabulary on it (each document once), and writes
`data/recipes/tokenizers/<name>.lock.json`: the recipe, the vocabulary's
sha256, and the list of the documents it was trained on, named by its sha256
(`documents_sha256`) and its number of lines (`document_count`). `wpt dataset
build-tokenizer <dataset recipe>` does the same and stops before any token.
The lock lists each source the build left out with the kind of its error
(`left_out`); with `--strict`, a failing source stops the build instead. The list is
one object of the store: a line per document, the JSON array
`["<id>", "<sha256 of its Markdown>"]`, sorted by the line's bytes, written while the documents are read
and sorted on disk, so it is never held in memory whole. Commit the lock.
From then on, every build of that version uses that vocabulary:

- **Adding or changing sources never retrains it.** Many datasets and
  models, whatever their training settings, share one vocabulary, so a
  model can keep training on new data. A source the vocabulary was not
  trained on still encodes exactly, since every byte has a token, only less
  compactly; `wpt dataset tokenizer-check <name>` shows by how much.
- **A new vocabulary takes a new version.** For a big change (much more
  data, another size) or a change of the tokenizing algorithm, raise
  `version` in the recipe: the next build trains a new vocabulary and
  replaces the lock (git history keeps the old one). Editing the recipe
  (other than its `note`) without raising `version` stops the build
  (`TokenizerVersionNeeded`), so one name and version always mean one
  vocabulary. A change of the tokenizer's code asks for no version: the
  lock keeps the vocabulary, and the code's hash, its Rust sources
  included, is in the fingerprint of what the code makes, so a build never
  reuses a result an older code made.
- **The vocabulary is shared through the store.** A locked vocabulary is
  never trained again from the documents its lock lists, on this machine
  or another: every build reads it from the store, checked against the
  sha256 the lock keeps. A store that does not hold it stops the build
  (`VocabularyMissing`), so a machine without access to the store that
  holds it cannot use it. The list of the trained documents is shared the
  same way: a store without it stops the build (`TrainedListMissing`).
- **Erasure does not reach a locked vocabulary** until `version` is raised:
  merges learned from text that was erased later stay in it (see
  [erasure.md](erasure.md)).

The tokenizer's version is in `packages/wpt-tokenizer/rust/Cargo.toml`; a
vocabulary file names it (see below). Bump it when a change makes the ids
mean other tokens.

### Where the vocabulary is checked

- The dataset build uses only the locked vocabulary; the dataset record
  names the recipe, its version and the vocabulary's sha256, and the
  dataset id covers it.
- Training reads the vocabulary from the store, checked against that
  sha256, and writes it into the model folder as `tokenizer.json`. Every
  weight file's header names it: `tokenizer` (such as
  `classics version 1`) and `tokenizer_sha256`.
- `wpt chat` and `wpt eval` load `tokenizer.json` only when its sha256
  is the one in the weights' headers (`TokenizerMismatch` otherwise), so a
  model never runs with another vocabulary.

### Checking a vocabulary on the data

`wpt dataset tokenizer-check <name>` encodes then decodes every document of
every admitted source with the locked vocabulary: each must give back its
exact bytes (`RoundTripFailed` otherwise). It prints, per source, the
documents, characters, tokens and characters per token, and marks the
sources the vocabulary was not trained on. It writes nothing to the store.

A vocabulary file is canonical JSON:

```json
{"format":"wpt-bpe","merges":[[a,b],...],"tokenizer_version":"0.3.0"}
```

A vocabulary file written by another tokenizer version is refused, since
its ids would mean other tokens: tokenizer 0.3.0 reserved ids 257 to 512,
so vocabularies of 0.2.0, and the models trained on them, need a new
vocabulary version and a new model.
