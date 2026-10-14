# Post-training

Post-training uses both:

- **recorded human demonstrations**, and
- **interactive environments**.

## Rules

Post-training data follows the corpus rules. Each example counts, so review
is systematic.

- Text written by people: existing sources under allowed licenses, original
  texts written for the project, and contributions under the DCO.
- No output of another company's language model.
- Environments use no fake web pages or apps: they replay recorded content,
  only when the project has the authorization to use it.
- Scores come from programs or from humans, never from another company's
  model.

## Record format

JSON Lines: one record per line. Media such as screenshots are separate
files named by their sha256 and referenced from the record. Each record
carries its provenance: the contributor, the commit that added it (signed
off under the DCO), the reference of the authorization it relies on, and its
license. Records are converted to tokens before training.

The exact fields are written when the first post-training data is added.

## Running post-training

```sh
uv run wpt-posttrain <base model folder> <new model folder> <step> [<step> ...]
```

A step is one Python file (or the name of a built-in one, in
`apps/wpt-posttrain/src/wpt_posttrain/steps/`; there is none yet) that
defines `run`:

```python
from wpt_posttrain import Outcome

def run(context):
    records = ...  # read the demonstrations, for example from context.store
    if not records:
        return Outcome.failure("no demonstration records in the store")
    ...  # train context.model on them
    return Outcome.success()
```

`context` gives the `model` to change (in place, on `device`), its
`tokenizer`, the `base` folder, its `provenance` and the `store`. Every step
returns the same type, `Outcome`: success, or failure with its reason. A
step does not write a report; it says whether it managed to do its part,
and if not, why.

The steps run in order, and the first failure stops the run: the command
prints that step's reason and writes nothing. When every step succeeded, it
writes the new folder: the weights, the tokenizer, and the base model's
`provenance.json` with a `post_training` section (the sha256 of the base
weights, each step's file and its sha256, the repository commit, the device
and the tools), then `LICENSE`, `NOTICE` and the model card, which lists the
steps. The new folder must not exist yet; the base folder is never changed,
and its `evals/` results are not copied, since they describe other weights.
Run `wpt-eval` on the new folder to measure it.

The records a step trains on will need their own provenance in `NOTICE`;
that part is written with the first post-training data.
