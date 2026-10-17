# Data descriptions

This folder describes data; it never holds data itself.

- `sources/<id>/`: one folder per upstream source (`source.json` and
  patches). See [docs/architecture.md](../docs/architecture.md).
- `recipes/mixtures/`: pools and their weights.
- `recipes/tokenizers/`: tokenizer recipes and the lock files builds write.
- `recipes/models/`: model recipes, each everything that changes one model:
  its data (built by `wpt-dataset build <name>`), architecture, training and
  post-training steps. See [docs/model.md](../docs/model.md).
- `images/<name>/`: the container images of agentic post-training: a
  `Dockerfile`, and the `lock.json` and `licenses.json` that
  `wpt-agent lock-image <name>` writes; `wpt-agent build-image <name>`
  builds the image from them on each machine. See
  [docs/post-training.md](../docs/post-training.md).
- `allowed-signers`: SSH keys allowed to sign decisions (OpenSSH
  allowed_signers format). GPG keys, if any, go in `allowed-gpg-keys.asc`.
