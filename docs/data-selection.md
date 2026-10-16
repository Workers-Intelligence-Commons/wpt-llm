# Data selection

## Documents

Every file of an admitted source's final tree (after cuts and patches) is a
document, identified as `<source id>/<path>`. Symlinks are not documents.

A file in the `records` format (a dataset holding many documents, such as a
JSON Lines or Parquet file) is not one document: each of its records whose
license enters, and that the format entry's `keep` rules keep, is one,
identified as `<source id>/<path>#<record id>`, and tagged with its file's
tags.

A git source with `history` adds one document per commit,
`<source id>/.git/history/<commit>`: the commit as `git format-patch`
writes it, limited to the files the source keeps.

**Exact duplicates are removed**: when several documents have the same
content, the first by source id, then by path, keeps it. Nothing judges
content. Near-duplicates stay.

## Tags and pools

Tags describe paths in `source.json` (language, domain...). They come from
the source's own statements or from a person, never from a guess: GitHub
repositories carry only `code`, since their human language is not stated
(maintainer's decision, 2026-10-16). A pool selects documents by a tag
expression:

| Expression | Selects |
| --- | --- |
| `fr` | documents tagged `fr` |
| `fr && literature` | both |
| `fr \|\| es` | either |
| `!(poetry)` | not; `!` must directly precede `(` |
| `fr && (literature \|\| poetry)` | grouping |

`!` binds tightest, then `&&`, then `||`.

A document matching several pools of one mixture goes to the **first**
matching pool, in the order the mixture lists them. Reordering pools
therefore changes the dataset (and its id).

## Mixtures

`data/recipes/mixtures/<name>.json`:

```json
{
  "schema": 1,
  "pools": [
    { "name": "fr", "select": "fr && !(literature)", "weight": 0.5, "max_epochs": 4, "note": "" },
    { "name": "literature", "select": "literature", "weight": 0.5, "note": "" }
  ],
  "note": ""
}
```

A weight is a share of the dataset's total tokens; weights add up to 1. A
pool smaller than its share repeats: whole passes over its documents, then a
partial pass. More passes than `max_epochs` (default 4, after Muennighoff et
al. 2023, where up to about 4 passes are about as good as fresh data) is
refused; a higher `max_epochs` is the explicit permission.

## The data part of a model recipe

A model recipe (`data/recipes/models/<name>.json`, see
[model.md](model.md)) says everything that changes a model; its `data`
part is what a build enforces:

```json
"data": { "mixture": "<mixture name>", "tokenizer": "<tokenizer recipe name>",
          "total_tokens": 2000000000, "held_out": 0.001 }
```

`wpt-dataset build <name>` builds the data part of the model recipe of
that name and prints the dataset id: the fingerprint of the data part, the
mixture, the vocabulary and every document. Recipes with the same data
part share one dataset, and `wpt-train` refuses a dataset built from
another data part than its recipe's. With `--stream`, it
fetches one source at a time and keeps only tokens in the store, for a
corpus larger than the disk; the dataset is the same
([architecture.md](architecture.md), "Fingerprints and the store"). There is no
tokens-per-parameter rule: a model may be data-starved or oversized for its
data. Model size is the recipe's `model` part.

The build record (in the store) lists the sources and their trees, the pools
with their documents, tokens, shares and passes, the shards, the
contamination report and the tool versions used.

## Held-out text and contamination

A fixed share of every source's documents (`held_out`), chosen by a hash of
the document id, never enters training; it goes to separate held-out shards
used to measure the model.

Contamination is evaluation text that also appears in training text. The
build checks held-out documents for runs of 13 words also found in training
documents (as in the GPT-3 paper) and **reports** them; it removes nothing,
so training data never depends on evaluation sets. External benchmarks are
not part of this repository. Training on a benchmark follows the normal
license rules.

## Languages and translation

As many languages as possible, plus translation as a capability. Clean
parallel text is scarce. It comes from:

- public-domain pairs: old works and their public-domain translations
  (Project Gutenberg, Wikisource), aligned by sentence;
- Europarl and EU texts (CC BY 4.0) and Tatoeba (CC BY 2.0 FR), whose
  licenses are on the allowed list (see [licenses.md](licenses.md));
- translations contributed under the DCO.

## Models in training

For now, no model takes part in training in any way. If WPT's own bigger
model ever produces data or plays a role in post-training for smaller
models, that is stated explicitly and must be open to review: the
contribution says exactly what the bigger model did, and the generated data
is a build result fingerprinted by model, prompts and seed. Whether such text
inherits the licenses of the bigger model's training data is a question for
a lawyer.

## Model size

The repository is recipes, scripts and environment setups usable for any
model size. Flags and features let a smaller model train on a subset (for
example a computer-use pre- and post-training subset).
