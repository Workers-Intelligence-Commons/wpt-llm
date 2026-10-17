# Architecture

**The repository is a lockfile, not a corpus.** It holds verifiable pointers
(address, revision, checksum), patches and build recipes. The data itself is
fetched on demand, then checked against its fingerprint, the same way
`go.sum` in Go or `fetchurl` in Nix pin exact bytes.

- We always know exactly which bytes were used.
- The repository stays small, publishable and reviewable in git.
- Every change to a source is a patch, separate from the source itself.
- Provenance is not a promise written next to the data: CI checks it.

## Three layers

1. **Upstream**: the content as its source publishes it, never modified and
   never committed. Only its description is in the repository.
2. **Erasure cuts and patches**: changes made by hand, kept as data next to
   the description.
3. **Build**: everything mechanical and repeatable (deduplication, held-out
   split, tokenization, shards). It is not content, so it is not a patch.

## Sources

One folder per source: `data/sources/<id>/` holds `source.json` and its
patches `0001-<name>.patch`, `0002-<name>.patch`... applied in that order.

### The id

`<origin>-<item>-<hash>`, or `<origin>-<hash>` when the source has no
natural item (lowercase `a-z 0-9 -`), for example `gutenberg-1342-05bda734`.
`<hash>` is the first 8 hex characters of the sha256 of the source's
`address`, so the id stays the same when the source moves to a new revision.
`wpt-dataset source-id <origin> [<item>] <address>` prints it. An id is never
reused.

When a source moves to a new address, it becomes a new source with a new id;
the old one is retired (`"decision": "retired"`), mixtures naming it are
updated, and its erasure cuts move to the new folder. When it publishes a new
revision, the folder is updated in place: `revision` and `checksum` change,
and its cuts and patches are redone in the same commit. Rebuilding an old
recipe then needs the old commit of this repository.

### source.json

```json
{
  "schema": 5,
  "type": "http-archive",
  "address": "https://www.gutenberg.org/cache/epub/1342/pg1342-h.zip",
  "revision": "2026-03-01",
  "checksum": "sha256:...",
  "extract": { "strip": 1, "exclude": ["**/*.jpg"], "note": "" },
  "licenses": [ { "license": "LicenseRef-public-domain", "paths": ["**"],
                  "attribution": "Jane Austen, Pride and Prejudice, Project Gutenberg, https://www.gutenberg.org/ebooks/1342",
                  "text": "Public domain. Jane Austen died in 1817; ...", "note": "" } ],
  "formats": [ { "format": "xhtml", "paths": ["**/*.xhtml"], "note": "" } ],
  "tags": [ { "tags": ["en", "literature"], "paths": ["**"], "note": "" } ],
  "mirrors": [ { "address": "https://...", "note": "" } ],
  "cuts": [ { "path": "...", "start": 0, "end": 0, "sha256_after": "...", "note": "" } ],
  "decision": { "decision": "admitted", "by": "...", "date": "2026-10-14", "note": "" },
  "note": ""
}
```

- `schema`: the shape version of the file, for migrations.
- `type`: `http-file`, `http-archive` (tar, tar.gz, tar.bz2, tar.xz or zip),
  `git`, `mercurial`, `huggingface` (a dataset on huggingface.co, pinned to
  its commit), or `empty` for text written for WPT: it downloads
  nothing and starts with no files, and its patches hold every file. Its
  `address` is a name, `empty:<name>`, its `revision` the date it was
  started, and its `checksum` that of an empty manifest,
  `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `revision`: the full commit hash for git and mercurial; the fetch date
  (`YYYY-MM-DD`) for http, where the checksum is what pins the content; the
  start date for `empty`.
- `checksum`: the sha256 of the canonical manifest (below).
- `extract`: `strip` removes leading path components (entries left with no
  path are dropped), `exclude` leaves paths out, and `name` gives an
  `http-file` its path.
- `licenses`: SPDX identifiers (or `LicenseRef-<name>`, or alternatives
  joined by `OR`, each optionally `WITH` an exception) mapped to paths by
  glob patterns. Every path must match exactly one entry, or the check
  fails; a new upstream file that matches no entry blocks the update until
  someone assigns it. Each entry's `attribution` (required) is the credit
  that license asks for, written by a person: authors, title, copyright
  line, link. Its `text` (required) is the license as the source gives it,
  with its copyright line and, for Apache-2.0, the source's `NOTICE`
  content; for `LicenseRef-public-domain`, why the work is in the public
  domain. Every model trained on the source repeats both in its `NOTICE`.
- `formats`: how each file becomes Markdown, mapped to paths like
  `licenses`: every path matches exactly one entry. Training reads only
  Markdown (see "Documents are Markdown" below).
- `tags`: describe paths (language, domain); pools select documents by tags.
- `mirrors`: other addresses expected to give the same checksum.
- `history` (git only): adds one file per commit from `since` to the
  revision, `.git/history/<commit>`, in `git format-patch` layout.
- `submodules` (git only): the address of each submodule the source reads,
  by its path.
- `cuts`: erasure cuts; see [erasure.md](erasure.md).
- `decision`: the human decision on the source (required): `admitted`,
  `refused` or `retired`, by whom and when. `reason` (why) is required to
  refuse or retire; an admission gives one only for an exception, such as
  an explicit permission for content without a license.
- Every object may hold a free-text `note`. It is data, so it survives
  migrations (comments would not).

`CONTRIBUTING.md`, "Adding a source", says how to fill `history`,
`submodules` and the `records` format's fields, including its `keep`
rules.

Globs: `*` matches within one folder, `**` across any number of folders.

The schema is `packages/wpt-dataset/src/wpt_dataset/schemas/source.schema.json`.
Mixture, model recipe, tokenizer recipe and lock files have schemas there
too.

### The manifest and the checksum

One line per entry, sorted by the UTF-8 bytes of the path:

```
<mode> <sha256> <path>
```

`100644` is a file, `100755` an executable file, `120000` a symlink (its
sha256 is that of the target text). Empty folders do not appear, so a git
checkout and an archive of the same release give the same checksum and can
be mirrors of each other. The checksum is `sha256:` + the sha256 of the
manifest. `wpt-dataset manifest <id>` prints a source's manifest and
checksum, to fill in a new `source.json`.

The manifest is computed from the downloaded data itself (git's objects,
mercurial's manifest, archive headers), never from files written to a disk,
so Linux and Windows agree. A path that is not valid UTF-8, uses `\`, `.` or
`..`, or differs from another only by letter case makes the source fail; an
`extract` rule or a removal patch must leave it out first.

Files that are not UTF-8 text (images, binaries) make the source fail too:
exclude them in `extract`.

### Patches

Patches use git's format and are applied with `git apply` inside a temporary
git index, so no file is written to a disk and modes and symlinks behave the
same everywhere. Git is therefore a dependency of data processing. A patch
can add, change, delete files and change modes.

Patches are not written by hand: `wpt-dataset unpack <id> <folder>` writes a
source's final files (after cuts and patches) into a folder outside the
repository; after editing them, `wpt-dataset patch <id> <folder> <name>`
records the difference as the next patch, `NNNN-<name>.patch`. It then
applies every patch again and keeps the new one only if the source gives
exactly the folder's files and passes its license map.

Removal patches are allowed even though their `-` lines show the removed
text: this repository is the software that downloads and treats the data,
and the project is what comes out of it. Removed material is not used, not
endorsed, and explicitly removed; the removal is itself the record that a
license was respected.

### When a source fails

A download that fails for a reason that may pass (a timeout, a cut
connection, an HTTP answer of 408, 425, 429 or 5xx, a git or Mercurial
network error) is tried again, up to 5 attempts in all. The wait doubles
from 4 seconds up to 60, each one drawn at random below that limit, and
follows a server's `Retry-After` up to 60 seconds. Anything else (a 404, a
robots.txt refusal, a checksum that differs) fails at once, since asking
again gives the same answer (`wpt_dataset/retry.py`).

A source that still fails (its checksum differs because the site rebuilt
the file, every address fails, a patch no longer applies, a file is
broken) stops `wpt-dataset build`. With `--skip-failed` (or
`WPT_SKIP_FAILED=1`), the build leaves it out instead and lists it, with
its error, under `skipped` in the dataset record; the model's
`provenance.json` repeats the list as `skipped_sources`. A missing program
(such as `hg`) stops the build anyway, since it fails every source that
needs it. Training a new vocabulary never skips a source: its lock freezes
the vocabulary for every later model of that version, so the first build
of a new tokenizer version needs every source.

`wpt-dataset check --fetch` checks every source it was given, even after
one failed, and ends with the list of failures and their reasons.

## Documents are Markdown

Every training document is the Markdown of one file of a source's final
tree, as its `formats` entry says
(`packages/wpt-dataset/src/wpt_dataset/markdown.py`):

| Format | Becomes |
| --- | --- |
| `markdown` | the file as it is |
| `text` | paragraphs as written; each line break inside a paragraph is a Markdown line break; Markdown's own characters are escaped, so the text reads exactly as written |
| `code` | one fenced code block whose opening line names the language and the path: ```` ```python path="./src/main.py" ```` (the entry's `language`); in a git or Mercurial source the path starts with `./`, the repository's root, and in any other source it is written as it is (```` ```python path="src/main.py" ````), since no repository root exists |
| `xhtml` | web pages written as strict XML, such as EPUB books, read by lxml (a malformed file stops the build): headings, paragraphs, lists, quotations, emphasis, tables; a dialogue row becomes `**Speaker:** words`; a note reference becomes a Markdown note (`[^12]`) with the note's text at the end of the document that refers to it, or is dropped when the note is not in the source |
| `records` | a dataset file holding many documents (JSON Lines, optionally `.gz` or `.zst`, or Parquet): each record whose license value an entry lists, and that the `keep` rules keep, is one document, `<file>#<record id>`; its text field is read as `text` above, or kept as it is with `"as": "markdown"` |
| `chat` | conversations for post-training, one JSON object per line ([chat-format.md](chat-format.md)); each line is checked, and the file gives no pre-training document: only the post-training step named by its tags reads it |

A file whose Markdown is empty is no document: an EPUB file that holds only
notes gives none, since its notes moved into the texts that refer to them.
A new kind of data (another markup, another kind of code) needs its format
here first; a path without a format stops the check.

## Fingerprints and the store

Every result is stored under the fingerprint of what produced it: the sha256
of its inputs, including WPT's own code version.

| Result | Fingerprint covers |
| --- | --- |
| upstream tree | the source checksum |
| final source tree | checksum, cuts, each patch's sha256, wpt-dataset version |
| vocabulary | tokenizer code version, tokenizer recipe (with its `version`), the documents it is trained on (id and sha256 of their Markdown); frozen by the recipe's lock until `version` is raised ([tokenizer.md](tokenizer.md)) |
| dataset | wpt-dataset version, the model recipe's data part, mixture, vocabulary, every document's id, sha256 (of its Markdown) and tags, each source's address, revision, checksum, tree and licenses (with attribution), and the sources skipped with `--skip-failed` |

So "already done" is a lookup: nothing is downloaded or applied twice, and
changing one patch rebuilds only that source.

`wpt-dataset build --stream` (or `WPT_STREAM=1`) keeps less: each source is
fetched into a throwaway store, turned into Markdown and tokens, and the
throwaway store removed before the next source. The store keeps one
token object per source (every document's tokens, one after the other),
its final manifest and a summary, under a ref named after the final tree,
the formats, the licenses, the wpt-dataset version and the vocabulary; a
source that did not change is never fetched again. The documents, the
dataset id and the shards are the same as without `--stream`. What it no
longer has is the source's files: a new vocabulary version, a changed
format or patch, or `wpt-dataset unpack` fetches the source again.
Without `--stream`, the store keeps everything, as above.

The store is set by one setting, `WPT_STORE`, in the environment or in
`.env` at the repository root:

```
WPT_STORE=file:///home/me/wpt-store      a local folder
WPT_STORE=s3://bucket/prefix             an S3-compatible bucket
```

For S3, credentials and a non-AWS endpoint come from the usual
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_ENDPOINT_URL`
settings. Objects are named by their sha256 and checked on every read.

The store also holds the record of every run of `wpt-dataset build`,
`wpt-train`, `wpt-posttrain` and `wpt-eval`: one folder per run under
`records/`, written by name rather than by content
([observability.md](observability.md)).

## Packages

| Package | Role |
| --- | --- |
| `wpt-dataset` | all data rules: sources, manifests, cuts, patches, pools, builds, the checks CI runs, migrations; command `wpt-dataset` |
| `wpt-store` | the content-addressed store (local or S3), its record folders, and the token shard format with its checked reader, so training and evaluation read data without depending on the data rules |
| `wpt-telemetry` | the record of one command run in the store, sent live to an OpenTelemetry backend when one is set ([observability.md](observability.md)) |
| `wpt-tokenizer` | the byte-level BPE tokenizer, Rust inside |
| `wpt-web` | a local web interface: one small server on 127.0.0.1, one file per page (Python routes, React page); its interface is built with Bun when it starts; command `wpt-web` |
| `wpt-model` | the WPT network ([model.md](model.md)), its safetensors model folders, and their license documents (`LICENSE`, `NOTICE`, model card) |
| `wpt-training` | the training loop (global batch, evaluation, checkpoints, resuming), with two plug points: strategies (how the hardware is used) and methods (how the model learns) |
| `wpt-inference` | token-by-token sampling |
| `wpt-utils` | helpers several packages share, one submodule each: `wpt_utils.device` picks the best GPU, else the CPU; `wpt_utils.programs` starts the outside programs WPT runs (`program(name)`, with a clear error when one is missing) and reads the repository commit (`repository()`), without importing PyTorch; `wpt_utils.files.sha256_file` is the one way a file is hashed, by the store, the model folders and the run records alike |
| `wpt-train`, `wpt-eval`, `wpt-posttrain`, `wpt-chat` | the applications |

## Checks

`uv run scripts/check.py` runs every check, the same way on any machine
and in CI (`.github/workflows/ci.yml`, on every pull request and push), so
a green run before pushing means a green CI. It compares with the upstream
branch, as a push would:

- `uv.lock` matches every `pyproject.toml`.
- The tests of every package, and the Rust tests. None needs a server: the
  S3 test runs against moto, which answers S3 requests inside the test
  process.
- `wpt-dataset check`: every file matches its schema and current shape,
  source ids match their address, licenses are on the allowed list (with
  their text), patches are numbered without gaps, mixture
  weights add up to 1 and pool expressions parse, recipes name existing
  recipes.
- `wpt-dataset check --changed <base>`: changed sources are fetched; their
  checksum, cuts, patches, license map and text are checked.
- `wpt-dataset check-commits <range>`: DCO sign-off on every commit, and a
  signature from an allowed key on every commit that changes a decision.

It runs every check even after one failed, then lists which passed. There
is no local hook: run it before pushing.

## Migrations

Each data file carries its shape version (`schema`). When a shape changes,
the change comes with a migration step in
`packages/wpt-dataset/src/wpt_dataset/migrate.py`, and `wpt-dataset migrate`
rewrites every file that is behind. Nobody edits the files by hand for it,
except where a new field holds something only a person can write: shape 2 of
`source.json` adds `attribution`, which the migration leaves empty, and the
check then names each file to fill in. Shape 4 adds each license's `text`:
the migration moves a `LicenseRef-<name>.txt` file into it and leaves any
other text empty for a person to fill. Shape 5 adds the `huggingface`
type, the `records` format and a license entry's `values`; only the
number changes.
