## What the change does

<!-- What it does and why, in a few lines. Link the issue it answers. -->

## Part it touches

<!-- One or more: sources and data rules, tokenizer, dataset build, training,
evaluation, generation and chat, web interface, model architecture,
documentation. -->

## Checks

- [ ] `uv run scripts/check.py` passed
- [ ] Every commit is signed off (`git commit -s`)
- [ ] The documentation is updated in the same change as the code
- [ ] A new package has its row in `docs/licenses.md`
- [ ] Nothing personal or secret is in the change, its commits or its logs
- [ ] A change of a data file's shape comes with its migration (`wpt-dataset migrate`)

## A source

<!-- Only for a pull request that adds or changes a source. -->

License, with the address of its text:

Why it is allowed (`docs/licenses.md`, "What to check"):

A maintainer decides every source: the commits of a source change are signed by
a maintainer afterwards (CONTRIBUTING.md, "Decisions are signed").

## The model

<!-- Only for a pull request that changes the model or its training. -->

Measurements (the run record's folder, the eval results, before and after):

Do earlier weights still load, and give the same output? If not, what must be
trained again:
