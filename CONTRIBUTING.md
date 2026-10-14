# Contributing

Thank you for helping. WPT trains only on content whose license is clear, so
most of this page is about sources.

## Every commit: sign off (DCO)

Every commit carries a `Signed-off-by` line with your name and email, added
by `git commit -s`. By it you certify the
[Developer Certificate of Origin](https://developercertificate.org/): you
have the right to submit the change under the project's license. CI refuses
commits without it.

For a commit that adds or changes a source, signing off also certifies that:

- you read the source's license, and every path it covers;
- you looked for a "no AI training" wish (README, robots.txt, a
  machine-readable text-and-data-mining reservation) and found none;
- the source holds no personal data.

You take responsibility for those three checks.

## Adding a source

1. Check the license against [docs/licenses.md](docs/licenses.md). It must be
   on the allowed list, clearly. Grey-zone licenses (CC BY, CC BY-SA, ODbL)
   are refused until a lawyer confirms them. Public domain must hold in the
   United States and in the work's country of origin.
2. Choose the id: `uv run wpt-dataset source-id <origin> [<item>] <address>`.
3. Create `data/sources/<id>/source.json` (see
   [docs/architecture.md](docs/architecture.md)), with `checksum` set to
   `sha256:` followed by 64 zeros for now, and a `decision` stating
   `"refused"` with the reason "under review".
4. `uv run wpt-dataset manifest <id>` downloads it and prints its manifest and
   checksum. Put the checksum in `source.json`. Exclude binary files in
   `extract`, and map every path to exactly one license in `licenses`. In
   each license entry, write in `attribution` the credit that license asks
   for, as the source gives it: authors, title, copyright line, link. Every
   model trained on the source repeats it in its notice.
5. `uv run wpt-dataset check --fetch <id>` must pass, then `uv run scripts/check.py`.
6. Open a pull request. Say what the source is, its license, and anything a
   reviewer should know; use `note` fields for reasons that must stay with
   the data.

## Adding your own texts

Text written for WPT, by you, lives in the repository itself, as the
patches of an `empty` source (see [docs/architecture.md](docs/architecture.md)):

1. Choose its id: `uv run wpt-dataset source-id <origin> [<item>] empty:<name>`.
2. Create `data/sources/<id>/source.json` with `"type": "empty"`, the
   address `empty:<name>`, today's date as `revision`, the empty checksum,
   a license you choose for your text (for example `CC0-1.0`) with your
   `attribution`, its tags, and the decision.
3. `uv run wpt-dataset unpack <id> <folder>` (a folder outside the
   repository), write or edit your files there, then
   `uv run wpt-dataset patch <id> <folder> <name>`. Each later change is the
   same two commands and one more patch.

Committed text stays public in the repository's history, clones and forks:
an erasure can remove it from future builds, never from that history (see
[docs/erasure.md](docs/erasure.md)). Commit only text you want public for
good. Your signed-off commit certifies you wrote it, and it holds no other
person's personal data and no output of another company's model.

There is no automatic filtering or tagging: tags and licenses are written by
people, and a person reviews every source. Tools such as ScanCode may help
you find license texts; they never decide.

## Decisions are signed

A maintainer admits a source by setting `decision` to `admitted`, with their
name, the date and the reason. A commit that changes any `decision` must be
signed (`git commit -S`) by an SSH key listed in `data/allowed-signers` or a
GPG key in `data/allowed-gpg-keys.asc`; CI checks the signature against the
lists as they stood before the commit, so a new key can only be added by a
commit signed with a key already listed.

This applies to every commit that changes a decision, including the commit
that first adds a source. A pull request proposing a source therefore fails
the signature check until a maintainer decides: the maintainer sets the
decision, then signs each commit of the pull request (for example
`git rebase --exec 'git commit --amend --no-edit -S' <base>`; amending keeps
the original author) and merges with a merge commit. Squash and rebase merges
on GitHub replace the commits and their signatures, so they are not used for
decisions.

## Changing a source

- **New upstream revision**: update `revision` and `checksum` in place, and
  redo its cuts and patches in the same commit.
- **New address**: that is a new source with a new id; retire the old one
  (`"decision": "retired"`), update the mixtures naming it, and move its
  erasure cuts.
- **Patches**: `0001-<name>.patch`, `0002-<name>.patch`... in git's format,
  applied in order after the erasure cuts. Removal patches are welcome.

## Erasure requests

Every erasure request is honored. Open a public issue, or use the
repository's **Security** tab, **Report a vulnerability**, to ask privately.
See [docs/erasure.md](docs/erasure.md).

## Models and generated text

No output of another company's model is accepted, in data or in
post-training. If a contribution ever uses WPT's own bigger model (to
produce data, or in any post-training role), it must say exactly what that
model did; such use is not planned today.

## Post-training records

Demonstrations and environment recordings follow
[docs/post-training.md](docs/post-training.md): JSON Lines with their
provenance, recorded only with the authorizations they need.

## Code

```sh
uv run scripts/setup.py      # once per machine: installs the right PyTorch build
uv run scripts/check.py      # every check CI runs, before you push
```

`scripts/check.py` is what CI runs too: the tests of every package, the Rust
tests, the data rules, and the checks on changed sources and on your
commits, compared with the upstream branch. A green run means a green CI.
There is no local hook.
