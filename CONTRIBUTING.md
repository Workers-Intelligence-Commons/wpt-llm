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
- you looked for a "no AI training" wish from the owner (README, a
  machine-readable text-and-data-mining reservation) and found none
  (`wpt-dataset` also skips any address whose site's robots.txt forbids it
  and tries the next mirror; see [docs/licenses.md](docs/licenses.md));
- the source holds no personal data.

You take responsibility for those three checks.

## Issues and pull requests

Open an issue from one of the forms in
[.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE): **Bug**, **Enhancement**
(both cover the tooling and the model), **Model behavior**, **Source
proposal** and **Erasure request**. Each form asks for the part concerned.
The Bug and Enhancement forms also ask for the commit, the exact command,
the recipe, the dataset id and the weights' sha256, the run record's
folder, and your system, GPU and PyTorch build. The other forms ask for
what fits their kind. A blank issue stays possible. For an
erasure request that should not be public, or for a security problem, use
the repository's **Security** tab, **Report a vulnerability**, and never
post personal details in a public issue.

A pull request starts from
[.github/pull_request_template.md](.github/pull_request_template.md): what
the change does, the part it touches, and the checks the template lists.

### Labels

Every form adds the label of its kind: `bug`, `enhancement`,
`model behavior`, `source proposal` or `erasure request`. Every form also
asks for the part concerned, and the workflow
[.github/workflows/area-label.yml](.github/workflows/area-label.yml) adds
the matching area label when the issue is opened:

- `area: sources and data rules`
- `area: tokenizer`
- `area: dataset build`
- `area: training`
- `area: post-training`
- `area: evaluation`
- `area: generation and chat`
- `area: web interface`
- `area: model architecture`
- `area: documentation`

The three labels of kind (`model behavior`, `source proposal` and
`erasure request`) must exist too, like the 10 area labels. A maintainer
creates every label before its first use. The words of the dropdown and of
the workflow are the area label names without `area: `: change them
together.

## Adding a source

1. Check the license against [docs/licenses.md](docs/licenses.md). It must be
   on the allowed list, clearly; its "What to check" column says where each
   kind of license hides a trap. CC BY-SA 4.0, the one grey-zone license,
   is refused until a lawyer confirms it. Public domain must hold in the
   United States and in the work's country of origin.
2. Choose the id: `uv run wpt-dataset source-id <origin> [<item>] <address>`.
   For a git repository on GitHub, write the address without `.git`:
   GitHub's robots.txt forbids `/*.git` addresses to AI training
   crawlers, and the plain address serves the same repository.
3. Create `data/sources/<id>/source.json` (see
   [docs/architecture.md](docs/architecture.md)), with `checksum` set to
   `sha256:` followed by 64 zeros for now, and a `decision` stating
   `"refused"` with the reason "under review".
4. `uv run wpt-dataset fill-checksum <id>` downloads it once, excludes in
   `extract` every file that is not UTF-8 text (binary files), each by its
   exact path, writes the checksum in `source.json`, and checks the source
   as `check --fetch` does (step 5) without downloading it again. It takes
   several ids, fills each, and lists the failures at the end. Map record
   and chat files in `formats` before running it: a file no such format
   reads is checked as text, and excluded when it is not UTF-8.
   `uv run wpt-dataset manifest <id>` only prints a source's manifest and
   checksum. A publisher that rebuilds its archives gives another checksum
   on a later fetch: update `revision`, set `checksum` back to the zeros,
   and fill it again. Map every path to exactly one license in `licenses`,
   and to exactly one format in `formats` (how it becomes Markdown, below). In
   each license entry, write in `attribution` the credit that license asks
   for, as the source gives it: authors, title, copyright line, link, and
   in `text` the license as the source gives it (its license file with its
   copyright line, then an Apache `NOTICE` file's content; for public
   domain, why). Every model trained on the source repeats both in its
   notice.
   A dataset on huggingface.co is `"type": "huggingface"`, pinned to the
   dataset's commit; `fill-checksum` and `manifest` then read the Hub's
   listing and download only its small files. `fill-checksum` excludes
   nothing there, and its check downloads the dataset: exclude each file
   the check names as not UTF-8 text, then put in the checksum `manifest`
   prints. A file holding many documents, each with its own
   license (JSON Lines, optionally `.gz` or `.zst`, or Parquet), is in the
   `records` format: it names the record's `text`, `license`, `id` and
   `credit` fields, and each license entry covering it lists in `values`
   the license values it covers ([docs/licenses.md](docs/licenses.md),
   "Rules").
   A git source may add its history: `"history": {"since": "<commit>"}`
   gives one file `.git/history/<commit>` per commit from `since` to the
   revision (merges left out), in `git format-patch` layout and limited
   to the files `extract` keeps. `since` is the commit from which the root
   license file holds the license the source lists; give these files a
   `code` format with the language `patch` and a license entry of their
   own, `.git/history/**`: an entry written `**` covers only the tree's
   files ([docs/licenses.md](docs/licenses.md), "Rules").
   A git source downloads only the contents of the files it keeps. A
   submodule (a folder pointing to a commit of another repository) is
   either excluded in `extract` or read at that commit from the address
   `submodules` gives for its path, `{"vendor/lib": "https://..."}`; its
   own submodules follow the same rule, by their full path. Its files
   need a license entry like any other path.
5. `uv run wpt-dataset check --fetch <id>` must pass (it also converts every
   file to Markdown), and `uv run wpt-dataset markdown-check <id>` (every
   note complete, no leftover markup), then `uv run scripts/check.py`.
6. Open a pull request. Say what the source is, its license, and anything a
   reviewer should know; use `note` fields for reasons that must stay with
   the data.

## Adding your own texts

Text written for WPT, by you, lives in the repository itself, as the
patches of an `empty` source (see [docs/architecture.md](docs/architecture.md)).
The simplest way is the "New source" page of `uv run wpt-web`: write or
paste your text, and the page makes the source, under review, with your
text as its first patch; its "Conversations" and "Tasks" pages write
post-training conversations and tasks the same way, and every page shows
which data files git sees as changed. `uv run wpt-dataset new-source` does
the same on the command line. By hand:

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
name and the date. An admission needs no reason: the license map and its
texts already say why the source is allowed. It carries one only for an
exception, such as an explicit permission for content without a license.
Refusing or retiring a source always gives the reason, so nobody reviews it
again from scratch. A commit that changes any `decision` must be
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
post-training. A contribution that uses a WPT model (to produce data, or in
any post-training role) must say exactly what that model did; the one such
use today is a model's draft of a conversation's reply, below.

A conversation's `assistant` message may start as a draft of a WPT model:
in the conversation editor of
`wpt-web`, a WPT model folder writes the next reply, and you edit it or
confirm it as it is. The message's `metadata.draft` says exactly what the
model did: the model folder's name, the sha256 of each of its weights
files, the sha256 of the draft as the model wrote it, and `edited`,
whether you changed it, which the service layer computes from the content
when the conversation is saved; the check refuses an `edited` that does
not match the content in any other chat file. An unedited confirmed draft
is allowed. The rest of `metadata.draft` is a claim the client makes,
which the server does not check (maintainer's decision, 2026-10-18): that
a WPT model wrote the draft, and its model, weights and sha256. See
[docs/chat-format.md](docs/chat-format.md), "Model drafts".

An AI assistant (a coding agent helping you) uses the `wpt-dataset`
commands only for structure and metadata: the kinds and names of files,
`source.json`, recipes, file lists, tags, git's status; never to read or
write people's texts, so it never runs the commands and options that do
(`read`, `write`, `conversation`, `save-conversation`, `task`,
`save-task`, and `new-source`'s `--file` and `--license-text`, marked `*`
in `uv run wpt-dataset --help`). It may start a new source with
`new-source` without `--file` and `--license-text` (maintainer's
decision, 2026-10-18): the source is made with its license, attribution,
format entries and tags, and the license text left empty (`--format
FORMAT=PATTERN` declares a format entry, such as `markdown=**/*.md` or
`text=notes/*.txt`, without any file; an empty `--license-text` file is an
error, not a text to write later); you then write the license text
(`wpt-dataset save source`, or the "Sources" page of `wpt-web`), and the
check names the source until you do (maintainer's decision, 2026-10-18:
the text may be empty only while the decision is refused, and builds and
checks leave such a source out with a note), then add your files, in any
format. People write their texts themselves. When a
person asks an assistant only to change the form of a text they wrote
(line endings, spacing, splitting into files), the assistant may move it
with scripts that transform whole files mechanically and report only its
shape, never its words, then pass it to `unpack` and `patch`; it never
reads or writes the words itself.

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
tests, the data rules, and the checks on changed sources and on your commits,
compared with the upstream branch. A green run means a green CI. There is no
local hook. The tests run in parallel, each process with one PyTorch thread and
no GPU, as in CI (a plain `uv run pytest` uses the GPU), on as many processes
as both 75% of the cores and 75% of the available memory (2.5 GB per process)
allow, at least 1. The image build test runs only when your push changes image
code (`packages/wpt-agent/src/wpt_agent/images.py`, the wpt-agent tests
`test_images.py` and `conftest.py`, the image lock and image licenses schemas
of wpt-dataset, or `data/images/`); otherwise `uv run pytest` runs it. The GRPO
tests keep their pinned and built image in `.pytest_cache/` and in Docker, so
only the first check on a machine, and every CI run, which starts empty, pays
for locking and building it (maintainer's decision, 2026-10-20;
`docs/architecture.md`, "Checks").

A test checks that the code works, not that it works well (maintainer's
decision, 2026-10-21): it runs without crashing, its computation is valid
(finite numbers, values in their range, tokens come out, training changes
the weights it reaches), and two routes the code says are the same give the
same result within rounding. Whether a method is efficient or a good idea
is a question for an evaluation (`wpt-eval`, docs/evaluation.md), never for
a test: a test that asks for a quality, such as a trained part beating a
simple one, fails on one processor and passes on another, by rounding and
chance, not by a bug.
