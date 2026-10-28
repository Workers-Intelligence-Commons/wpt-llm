# Data descriptions

This folder describes data; it never holds data itself.

- `sources/<id>/`: one folder per upstream source (`source.json` and
  patches). See [docs/architecture.md](../docs/architecture.md).
- `recipes/mixtures/`: pools and their weights.
- `recipes/tokenizers/`: tokenizer recipes and the lock files builds write.
- `recipes/models/`: model recipes, each everything that changes one model:
  its data (built by `wpt-dataset build <name>`), architecture and training
  steps. See [docs/model.md](../docs/model.md).
- `images/<name>/`: the container images of agentic training: a
  `Dockerfile`, and the `lock.json` and `licenses.json` that
  `wpt-agent lock-image <name>` writes; `wpt-agent build-image <name>`
  builds the image from them on each machine. See
  [docs/training.md](../docs/training.md).
- `allowed-signers`: SSH keys allowed to sign decisions (OpenSSH
  allowed_signers format). GPG keys, if any, go in `allowed-gpg-keys.asc`.
- `license-texts/<id>/`: one folder per license text (`license-text.json`,
  its decision, kind and license, and optionally `license-text.txt`, the
  text itself), `<id>` the sha256 of its normalized text. See
  [docs/licenses.md](../docs/licenses.md), "License texts".
- `reviewed-license-texts.json`: reviews of license texts in an earlier
  shape, which `wpt-dataset migrate` turns into items of `license-texts/`;
  a review held for one exact file stays here for a person to decide.
