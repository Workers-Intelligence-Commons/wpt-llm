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
   is allowed until legal review says otherwise. Public domain must hold
   under the law of the country where the work was first published.
2. Choose the id: `uv run wpt-dataset source-id <origin> [<item>] <address>`.
   For a git repository on GitHub, write the address without `.git`:
   GitHub's robots.txt forbids `/*.git` addresses to AI training
   crawlers, and the plain address serves the same repository.
3. Create `data/sources/<id>/source.json` (see
   [docs/architecture.md](docs/architecture.md)), with `checksum` set to
   `sha256:` followed by 64 zeros for now, and a `decision` stating
   `"under-review"`, with your name (`by`) and today's `date` (see
   "Decisions are signed").
4. `uv run wpt-dataset fill-checksum <id>` downloads it once into the
   store, excludes in `extract` every file that is not text (binary
   files), each by its exact path, writes the checksum in `source.json`,
   and checks the source as `check --fetch` does (step 5) without
   downloading it again. It takes
   several ids, fills each, and lists the failures at the end. A file is
   text when it is UTF-8, or when it states its own encoding (a
   byte-order mark, an HTML page's `<meta charset>`, an XML declaration,
   Project Gutenberg's `Character set encoding:` line, a Python coding
   line) and decodes under it; nothing is guessed ([docs/architecture.md](docs/architecture.md),
   "Encodings"). Map every file in `formats` before running it: the format
   says which statement the file can make, and a file no records or chat
   format reads is checked as text.
   `uv run wpt-dataset manifest <id>` downloads a source without checking
   its checksum, keeps it in the store, and prints its manifest and
   checksum. A publisher that rebuilds its archives gives another checksum
   on a later fetch: update `revision`, set `checksum` back to the zeros,
   and fill it again. Map every path to exactly one license in `licenses`,
   and to exactly one format in `formats` (how it becomes Markdown,
   [docs/architecture.md](docs/architecture.md), "Documents are Markdown":
   an EPUB page is `xhtml`, a web page that is not strict XML, such as
   Project Gutenberg's, is `html`). In
   each license entry, write in `attribution` the credit that license asks
   for, as the source gives it: authors, title, copyright line, link, and
   in `text` the license as the source gives it (its license file with its
   copyright line, then an Apache `NOTICE` file's content; for public
   domain, why). Every model trained on the source repeats both in its
   notice.
   A dataset on huggingface.co is `"type": "huggingface"`, pinned to the
   dataset's commit; `fill-checksum` downloads every file it keeps, as for
   any other source. An archive the dataset holds (tar, tar.gz, tar.bz2,
   tar.xz, zip) is unpacked into its folder and its members are the
   source's files: exclude in `extract` the archives the source must not
   hold by their own paths, which keeps them from being downloaded. A source published at several
   addresses (its documents in one archive and their metadata in another)
   lists them in `addresses`, each with the
   folder of the tree it fills, and repeats the first one in `address`
   ([docs/architecture.md](docs/architecture.md), "source.json"). A file
   holding many documents, each with its own
   license (JSON Lines, optionally `.gz` or `.zst`, or Parquet), is in the
   `records` format: it names the record's `text`, `license`, `id` and
   `credit` fields, and each license entry covering it lists in `values`
   the license values it covers ([docs/licenses.md](docs/licenses.md),
   "Rules"). A text field that holds Markdown says `"as": "markdown"`, and
   one that holds a web page says `"as": "html"`; a `title` field gives
   the document its heading, and a dataset holding several revisions of
   one record names their date field in `newest`. A dataset that states
   one license for all its records names
   no `license` field, and its files take their one license entry. A
   JSON document whose field holds the list of records names that field in
   `records`; an XML file is one record, its fields XPath expressions with
   the entry's `namespaces`, and a `join` reads the license and credit from
   another file of the same record ([docs/architecture.md](docs/architecture.md),
   "Documents are Markdown").
   A git source may add its history: `"history": {"since": "<commit>"}`
   gives one file `.git/history/<commit>` per commit from `since` to the
   revision (merges left out), in `git format-patch` layout and limited
   to the files `extract` keeps, and `.git/history/order`, the commits'
   order with the merges marked. `since` is the commit from which the root
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
paste your text, and the page makes the source, under review
(`under-review`), with your text as its first patch; its "Conversations"
and "Tasks" pages write post-training conversations and tasks the same
way, its "Ratings" page your pass or fail on a model's attempts (a
measure only, never a training signal:
[docs/evaluation.md](docs/evaluation.md), "People's ratings"), and every
page shows which data files git sees as changed.
`uv run wpt-dataset new-source` does the same on the command line. By
hand:

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

No program admits or refuses a source. A program such as a source
drafter may draft a source's licenses and tags as help: its output is not
a source of truth, a person checks it, and only a person admits a source,
after reviewing it. Tools such as ScanCode may help you find license
texts; they never decide.

## Decisions are signed

A source is in exactly one of three states, its `decision`:
`under-review`, a source not yet decided, where every new source starts
and where a doubt stays; `admitted`; or `refused`. Each gives who set it (`by`)
and when (`date`). The builds, the post-training readers and
`check --fetch` without ids read only admitted sources: an `under-review`
source is left out of them exactly as a refused one is.

A maintainer admits a source by setting `decision` to `admitted`, with their
name and the date. An admission needs no reason: the license map and its
texts already say why the source is allowed. It carries one only for an
exception, such as an explicit permission for content without a license.
Refusing a source always gives the reason, so nobody reviews it again from
scratch; a source under review may give one (a doubt gives the missing
fact). A refused source stays in the repository as a reference: it is
never deleted. A commit that changes any `decision`
must be signed (`git commit -S`) by an SSH key listed in
`data/allowed-signers` or a GPG key in `data/allowed-gpg-keys.asc`; CI
checks the signature against the lists as they stood before the commit, so
a new key can only be added by a commit signed with a key already listed.

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
- **New address**: that is a new source with a new id; refuse the old one
  (`"decision": "refused"`, its `reason` naming the new source's id),
  update the mixtures naming it, and
  move its erasure cuts.
- **Patches**: `0001-<name>.patch`, `0002-<name>.patch`... in git's format,
  applied in order after the erasure cuts. Removal patches are welcome.

## Erasure requests

Every erasure request is honored. Open a public issue, or use the
repository's **Security** tab, **Report a vulnerability**, to ask privately.
See [docs/erasure.md](docs/erasure.md).

## Models and generated text

No output of another company's model is accepted, in data or in
post-training, and no synthetic data that a WPT model did not make. WPT's
own models may be used in training, their output included
([docs/data-selection.md](docs/data-selection.md), "Models in training"). A contribution that uses a WPT model (to produce
data, or in any post-training role) must say exactly what that model did.
The uses today are a model's draft of a conversation's reply, below, and
the `grpo` step's reward on a task's one-line form, which compares an
attempt with the model's own attempt at the full form
([docs/post-training.md](docs/post-training.md), "Rewards").

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
which the server does not check: that
a WPT model wrote the draft, and its model, weights and sha256. See
[docs/chat-format.md](docs/chat-format.md), "Model drafts".

An AI assistant (a coding agent helping you) uses the `wpt-dataset`
commands only for structure and metadata: the kinds and names of files,
`source.json`, recipes, file lists, tags, git's status; never to read or
write people's texts, so it never runs the commands and options that do
(`read`, `write`, `conversation`, `save-conversation`, `task`,
`save-task`, `save-rating`, and `new-source`'s `--file` and
`--license-text`, marked `*`
in `uv run wpt-dataset --help`). It may start a new source with
`new-source` without `--file` and `--license-text`: the source is made with its license, attribution,
format entries and tags, and the license text left empty (`--format
FORMAT=PATTERN` declares a format entry, such as `markdown=**/*.md` or
`text=notes/*.txt`, without any file; an empty `--license-text` file is an
error, not a text to write later); you then write the license text
(`wpt-dataset save source`, or the "Sources" page of `wpt-web`), and the
check names the source until you do (the text may be empty only while
the source is not admitted, that is `under-review` or `refused`, and
builds and checks leave such a source out with a note), then add your
files, in any format. People write their texts themselves. When a
person asks an assistant only to change the form of a text they wrote
(line endings, spacing, splitting into files), the assistant may move it
with scripts that transform whole files mechanically and report only its
shape, never its words, then pass it to `unpack` and `patch`; it never
reads or writes the words itself.

## Post-training records

Demonstrations and environment recordings are files of a source, like any
other text ([docs/post-training.md](docs/post-training.md), "Where records
live"): their license, credit and decision are the source's, in its
`source.json`, and the signed-off commit that adds them names their
contributor. Their shape is not settled yet; today the code reads
conversations in the `chat` format, tasks in the `task` format and
ratings in the `rating` format.

## Writing about the project

The repository describes only its current state: the documentation, the
skills (`.agents/skills/`), code comments and docstrings, and recipe
notes say what is true now and why, never when it was decided or what it
was before. A change may break compatibility with what came before; the
text then describes the new state alone, and the old one stays only in
git's history. No text holds a decision's date, a "(maintainer's
decision)" tag (only "maintainer's decision alone" stays, for a license
choice with no outside ruling) or an account of earlier versions. Rules
are general: they name no source and no example, so they hold for any
number of sources. Dates that belong to the content stay, such as a
source's revision or a lawyer's answer in `LEGAL-QUESTIONS.html`. A
migration step (`wpt-dataset migrate`) says what it changes in a file of
an older shape, not when that shape changed.

## Code

```sh
uv run scripts/setup.py      # once per machine: installs the right PyTorch build
uv run scripts/check.py      # every check CI runs, before you push
```

`scripts/check.py` is what CI runs too: the tests of every package, the Rust
tests, the data rules on every source (read, not downloaded), the download and
check of one small source (`standardebooks-joseph-conrad-ford-madox-ford-the-nature-of-a-crime-5366b0f3`,
a short book fixed at one git commit, so its download never changes), which
proves the fetch and its checks work, and the checks on your commits, compared with the
upstream branch. A green run means a green CI. Neither check.py nor CI
downloads the sources a pull request adds or changes: the maintainer runs
`uv run wpt-dataset check --fetch` on them before merging. There is no
local hook. The tests run in parallel, each process with one PyTorch thread and
no GPU, as in CI (a plain `uv run pytest` uses the GPU), on as many processes
as both 75% of the cores and 75% of the available memory (2.5 GB per process)
allow, at least 1. The image build test runs only when your push changes image
code (`packages/wpt-agent/src/wpt_agent/images.py`, the wpt-agent tests
`test_images.py` and `conftest.py`, the image lock and image licenses schemas
of wpt-dataset, or `data/images/`); otherwise `uv run pytest` runs it. The GRPO
tests keep their pinned and built image in `.pytest_cache/` and in Docker, so
only the first check on a machine, and every CI run, which starts empty, pays
for locking and building it (`docs/architecture.md`, "Checks").

A test checks that the code works, not that it works well: it runs without crashing, its computation is valid
(finite numbers, values in their range, tokens come out, training changes
the weights it reaches), and two routes the code says are the same give the
same result within rounding. Whether a method is efficient or a good idea
is a question for an evaluation (`wpt-eval`, docs/evaluation.md), never for
a test: a test that asks for a quality, such as a trained part beating a
simple one, fails on one processor and passes on another, by rounding and
chance, not by a bug.
