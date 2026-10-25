# Data selection

## Documents

Every file of an admitted source's final tree (after cuts and patches) is a
document, identified as `<source id>/<path>`. Symlinks are not documents.
A document is the file's Markdown, read in the encoding the file states,
else UTF-8, and written as UTF-8 ([architecture.md](architecture.md),
"Encodings"); an image in a web page
(`xhtml` or `html`) leaves the marker `[image]` or `[image: ALT]` where it
was ([architecture.md](architecture.md), "Documents are Markdown").

A file in the `records` format (a dataset holding many documents, such as a
JSON Lines or Parquet file) is not one document: each of its records whose
license enters, and that the format entry's `keep` rules keep, is one,
identified as `<source id>/<path>#<record id>`, and tagged with its file's
tags.

A git source with `history` adds one document per commit,
`<source id>/.git/history/<commit>`: the commit as `git format-patch`
writes it, limited to the files the source keeps. Its order file,
`.git/history/order` ([architecture.md](architecture.md), `history`),
stays a file of the source's stored tree, covered by its checksum, but it
is not a training document: the build, the vocabulary and every other
list of documents leave it out.

**Exact duplicates are removed**: when several documents have the same
content, the first by source id, then by path, keeps it. Nothing judges
content. Near-duplicates stay.

## Tags and pools

Tags describe paths in `source.json` (language, domain...). They come from
the source's own statements or from a person, never from a guess: GitHub
repositories carry only `code`, since their human language is not stated.
A pool selects documents by a tag
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
another data part than its recipe's. It reads the sources `wpt-dataset
download` put in the store and downloads none; a source that is not
there is left out and listed in the record. With `--stream`, it keeps
only the tokens of each source's documents in the store, not their
Markdown, and no source's files on the machine; the dataset is the same
([architecture.md](architecture.md), "Fingerprints and the store"). There is no
tokens-per-parameter rule: a model may be data-starved or oversized for its
data. Model size is the recipe's `model` part.

The build record (in the store) lists the sources and their trees, the
download each source's files came from (its session and time), the pools
with their documents, tokens, shares and passes, the shards, the
contamination report and the tool versions used. It also gives a
fingerprint (a sha256) for each part the training steps read: `text` (the
shards, training and held-out), and `conversations` and `tasks` when the
dataset holds some. Each holds the vocabulary too, since every part becomes
tokens with it. The documents a pool takes and the order of the rows come
from the dataset's documents and sources that hold documents, never from the
conversations and tasks or the sources left out, so adding a conversation or
a task, or leaving out a source that holds no documents, leaves the `text`
fingerprint as it was ([training.md](training.md), "Step results and
reuse").

## Held-out text and contamination

A fixed share of every source's documents (`held_out`), chosen by a hash of
the document id, never enters training; it goes to separate held-out shards
used to measure the model.

A source may hold out documents together, whatever source they belong to:
its optional `held_out_by` field gives a pattern (a Python regular
expression) searched in each document's path, and where it is found, the
hash reads the pattern's groups joined by one space (its whole match when
it has no group) instead of the document id. Documents of any source with
the same groups then share their fate. A document whose path the pattern
is not found in keeps its id's fate.

Translations of one text are held out together: sources whose documents
are translations of each other give them the same groups through
`held_out_by` (such as a book and chapter), so the same part is held out
in every language at once, since another translation of a held-out
passage would otherwise leak it. A measure that needs to know which parts
are held out reads the same field, so the rule has one owner
([evaluation.md](evaluation.md), "Text measures"). A document's group is
part of the dataset id's fingerprint, so a dataset holding such sources
gets a new id when it is built again, and the document list records the
split the groups give; a dataset without such sources keeps its id.

Every dataset build stores, beside the dataset's record, the list of all
its documents: each one's id, the sha256 of its Markdown (the same sha256
the dataset id's fingerprint reads) and its tag, `training` or `held-out`.
It is found from the dataset id,
which a model's `provenance.json` names and every training run is
given, so a step and a measure learn which texts the model held out from
that list alone: a pointer task whose
document's text is one of them is measure-only, and no source is
converted to Markdown to find out ([training.md](training.md),
"Measure tasks"). Reading pointer tasks against a dataset that has no
document list is refused (`DocumentListMissing`): such a dataset is
used with the repository at the commit that built it. Nothing
falls back and nothing is rebuilt. A dataset the store does not hold at
all is refused the same way, the message saying it is not in the store.

Contamination is evaluation text that also appears in training text. The
build checks held-out documents for runs of 13 words also found in training
documents (as in the GPT-3 paper) and **reports** them; it removes nothing,
so training data never depends on evaluation sets. External benchmarks are
not part of this repository. Training on a benchmark follows the normal
license rules.

## Languages and translation

As many languages as possible, plus translation as a capability. Clean
parallel text is scarce. It comes from:

- public-domain works and their public-domain translations, aligned by
  sentence;
- texts published in several languages under a license on the allowed list
  ([licenses.md](licenses.md));
- translations contributed under the DCO.

## Models in training

WPT's own models may be used in training, their output included: as a
reference a reward compares
with, such as a model's own full attempt in the terse-versus-full reward
of the `finish-tasks` step ([training.md](training.md), "Rewards"), as
a draft a person edits or confirms ([chat-format.md](chat-format.md),
"Model drafts"), or as `answer-as-assistant` conversations made of its
attempts that passed their
task's check (issue #50, not built yet). Every such use says exactly what the WPT model did, so it is
open to review.

Other companies' models stay completely forbidden: no output of theirs
enters the data or the training, and they take no role in it, not as
data, a simulator or a judge. Synthetic data not made by a WPT model is
forbidden too; puzzles and tasks a program generates are not that kind of
synthetic data ([licenses.md](licenses.md), "No distillation from another
company's model").

## Model size

The repository is recipes, scripts and environment setups usable for any
model size. Flags and features let a smaller model train on a subset (for
example a computer-use subset for every training step).
