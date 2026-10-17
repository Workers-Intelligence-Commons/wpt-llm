# Evaluation

`wpt-eval` runs tests on a trained model and records their results in its
model card.

```sh
uv run wpt-eval runs/default                      # every built-in test
uv run wpt-eval runs/default memorization         # one built-in test
uv run wpt-eval runs/default path/to/my_test.py   # any Python file
```

It reads the model folder that `wpt-train` wrote, and the dataset the model
was trained on from the store named by `WPT_STORE`, checking every shard
against its recorded hash. Each result goes to `<model folder>/evals/<name>.json`
and the model card (`README.md`) is written again to show it. Training a new
model into the same folder removes the old results: they describe other
weights.

## Writing a test

A test is one Python file that defines `run`:

```python
from wpt_eval import Result

def run(context):
    documents = context.documents("held_out")
    count = sum(len(d.tokens) for d in documents)
    return Result({"held_out_tokens": count}, f"{count} held-out tokens")
```

`context` gives:

- `model`: the model, in evaluation mode, on `device`;
- `tokenizer`, `device`, `folder`, and `provenance` (the folder's
  `provenance.json`);
- `store`: the store holding the dataset;
- `telemetry`: the run's record, for the test's own events
  ([observability.md](observability.md));
- `documents("training")` and `documents("held_out")`: each document of that
  part once, with its `id`, `source` and `tokens` (ending with the
  end-of-document token). They are read one shard at a time as the test
  goes through them, so a dataset larger than memory works; a test that
  needs them twice calls `documents` twice, or keeps only what it needs.

Every test returns the same type, `Result`: `data`, numbers and names for
programs (anything JSON can hold), and `text`, the same findings written for
people, which the model card shows. A test is plain code, not a recipe file,
so it can measure whatever the question needs. The result file also records
the test file's sha256, so a result names the exact code that produced it.

Built-in tests live in `apps/wpt-eval/src/wpt_eval/evals/`; a new file there
is a new built-in test.

## Memorization

A model can reproduce pieces of its training data, and those pieces keep
their source's license (see [output-notice.md](output-notice.md)). The
`memorization` test measures how often it happens:

- up to 500 documents of each part, the same on every run;
- the model reads the first 32 tokens of a document and writes 32 tokens,
  always taking its most likely token;
- a reproduction is exact when those 32 tokens equal the document's next 32.

Held-out documents give the baseline: text that is easy to predict, such as
repeated lines or license headers, is continued exactly without any
memorization. Exact reproductions of training documents well above the
baseline mean the model outputs pieces of its sources. The result lists
exact reproductions by source, so the sources whose attribution matters most
are visible.
