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

## Where records live

Records are files of a source, like any other text: they get the same
license entries, attribution, signed decision, NOTICE listing and checks as
every source. Records written for WPT go in an `empty` source (see
CONTRIBUTING.md, "Adding your own texts"). Conversations are files in the
`chat` format, tagged with the name of the step that reads them (`sft` for
the `sft` step); a `chat` file is never a pre-training document
([chat-format.md](chat-format.md)).

- A record holds only what a person made: commands typed, edits, messages
  written. From a session with an AI assistant, only the person's own
  messages may be kept, never the assistant's replies.
- No other person's personal data, and no secrets: terminal output and
  paths are checked before a record is committed.

## Record format

JSON Lines: one record per line. Media such as screenshots are separate
files named by their sha256 and referenced from the record. Each record
carries its provenance: the contributor, the commit that added it (signed
off under the DCO), the reference of the authorization it relies on, and its
license. A conversation in a record follows [chat-format.md](chat-format.md):
one JSON object holding every element, turned into tokens by a template
before training.

## Running post-training

```sh
uv run wpt-posttrain <model recipe> <base model folder> <new model folder>
```

The steps are the model recipe's `posttrain` list, run in order; the new
folder's provenance records the recipe's name.

A step is one Python file (or the name of a built-in one, in
`apps/wpt-posttrain/src/wpt_posttrain/steps/`) that defines `run`:

```python
from wpt_posttrain import Outcome

def run(context):
    conversations = context.conversations()  # [(id, conversation)] tagged with this step's name
    if not conversations:
        return Outcome.failure("no conversations")
    ...  # train context.model on them
    return Outcome.success()
```

`context` gives the `model` to change (in place, on `device`), its
`tokenizer`, the `base` folder, its `provenance` and the `store`.
`context.conversations()` reads the conversations of every admitted source
whose `chat` files carry a tag named after the step, from the repository
holding the current folder, and the runner records those sources. Every step
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

Each step's record in `provenance.json` names the sources it read, and the
`post_training` section keeps their address, revision and licenses. The
new folder's `NOTICE` lists them with the pre-training sources, with the
credit and the license text of each, and the model card names them under
each step.

## The `sft` step

The `sft` step, in a recipe's `posttrain` list, is supervised fine-tuning:
it reads the conversations tagged `sft`, turns each into tokens with the
chat template, and trains the model to predict the assistant's turns only. Until a source
holds such conversations, it fails with "no conversations" and writes
nothing. Each conversation must fit the model's context; a longer one
stops the step, since cutting it would drop the answers it holds. Its
settings (epochs, learning rate, batch, seed) are the constants at the top
of `steps/sft.py`; to change them, copy the file and put its path in the
recipe's `posttrain` list instead.

The base model must use a vocabulary of tokenizer 0.3.0 or later, which
reserves the chat template's ids. Talk to the result with
`uv run wpt-chat NEW --chat`.
