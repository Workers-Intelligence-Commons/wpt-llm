# Data descriptions

This folder describes data; it never holds data itself.

- `sources/<id>/`: one folder per upstream source (`source.json` and
  patches). See [docs/architecture.md](../docs/architecture.md).
- `recipes/mixtures/`: pools and their weights.
- `recipes/tokenizers/`: tokenizer recipes and the lock files builds write.
- `recipes/datasets/`: dataset recipes, built by `wpt-dataset build <name>`.
- `allowed-signers`: SSH keys allowed to sign decisions (OpenSSH
  allowed_signers format). GPG keys, if any, go in `allowed-gpg-keys.asc`.
