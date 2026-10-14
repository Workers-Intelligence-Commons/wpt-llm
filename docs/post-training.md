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
