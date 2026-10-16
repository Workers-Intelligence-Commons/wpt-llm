# Publishing models

No model has been published yet. This page says how it will be done
(maintainer's decisions, 2026-10-16).

## Releases

A release is a commit of this repository made in a given month and tagged
with that month, `YY.MM` (`26.10` for October 2026). Every model of the
release is trained on that commit: its sources, recipes and code. There is
at most one release a month, and only when the month brought enough
changes and progress to be worth one; a month spent testing gives none.
Anything not released still exists in the repository, and anyone can train
it from there.

## Names

`WPT-<YY.MM>-<size>[-<variation>]`, for example `WPT-26.10-M` or
`WPT-26.10-M-chat`.

- `<YY.MM>`: the release the model was trained on.
- `<size>`: one of `XXS`, `XS`, `S`, `M`, `L`, `XL`, `XXL`, relative to the
  other models of the same release, spread out from `M`: one model is `M`;
  two are `S` and `L`; three are `S`, `M`, `L`; four are `XS`, `S`, `L`,
  `XL`; five are `XS`, `S`, `M`, `L`, `XL`; six leave out `M`; seven use
  every letter. A letter says only the order inside its release: the model
  card gives the real size.
- `<variation>`: none for the base model; otherwise one short word, such as
  `chat` for a model post-trained on conversations, which the model card
  explains.

## Where

Each model is one repository on Hugging Face, under the
`Workers-Intelligence-Commons` organization, named as above. It holds the
final weights only, with the model folder's `LICENSE`, `NOTICE`,
`provenance.json`, `tokenizer.json` and model card. Training steps are not
published as branches or separate repositories: the model card gives the
steps and the repository commit the model was trained on, so anyone can
see exactly where it comes from.

## Before a release

The checklist in [output-notice.md](output-notice.md), "Before publishing
weights", applies: OUTPUT-NOTICE reviewed by a lawyer, memorization
measured on each model, and a stable address chosen for attribution.
