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

When a source moves to a new address, it becomes a new source with a new
id; the old one is refused (`"decision": "refused"`), its reason naming the
new source's id, mixtures naming it are updated, and its erasure cuts move
to the new folder. When it publishes a new revision, the folder is updated
in place: `revision` and `checksum` change, and its cuts and patches are
redone in the same commit. Rebuilding an old recipe then needs the old
commit of this repository.

### source.json

```json
{
  "schema": 7,
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
  its commit; an archive it holds is unpacked, below), or `empty` for text
  written for WPT: it downloads
  nothing and starts with no files, and its patches hold every file. Its
  `address` is a name, `empty:<name>`, its `revision` the date it was
  started, and its `checksum` that of an empty manifest,
  `sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `addresses` (`http-file` and `http-archive` only): one source fetched
  from several addresses, each into its own folder of the source's tree,
  for a publisher that puts documents in one archive and their metadata
  in another: `[{"folder": "fulltext", "address": ".../fulltext.zip"},
  {"folder": "metadata", "address": ".../metadata.zip"}]`. Each
  address is read with the source's `type`, `revision` and `extract`
  (`strip` on the paths of what it holds, `exclude` on the paths with
  their folder), and the manifest and checksum cover the one tree. `address`
  stays, and repeats the first one's address, which gives the id; a source
  with `addresses` has no `mirrors`. Every other source keeps its one
  `address`, unpacked at the tree's root.
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
- `tags`: describe paths (language, domain); pools select documents by
  tags.
- `mirrors`: other addresses expected to give the same checksum.
- `history` (git only): adds one file per commit from `since` to the
  revision, `.git/history/<commit>`, in `git format-patch` layout. A commit
  that also changes files `extract` leaves out gets a diff limited to the
  kept paths; when a kept path is also the folder of paths left out (a
  submodule replacing a folder of the same name), those paths are excluded
  from the diff by name. One more file, `.git/history/order`, holds the
  commit order: every commit from `since` to the revision, merges included,
  newest first and each before its parents (git's topological order), one
  per line, a merge's line ending in ` merge`. Being part of the stored
  tree, the order is covered by the source checksum, so a git source with a
  history gets a new checksum once it is fetched with its order file; until
  then its stored tree has no order file and gives no commit-and-push task.
  The order file is not a training document: `markdown.documents`, which
  every list of documents reads, leaves it out. Only patterns starting with
  `.git/history/` give these files their license entry: `**` covers the
  tree's files only (docs/licenses.md, "Rules").
- `submodules` (git only): the address of each submodule the source reads,
  by its path.
- `cuts`: erasure cuts; see [erasure.md](erasure.md).
- `held_out_by`: a `pattern` (a Python regular expression) searched in each
  document's path; where it is found, the held-out hash reads its groups
  instead of the document id, so documents of any source with the same
  groups are held out together; see [data-selection.md](data-selection.md),
  "Held-out text and contamination".
- `decision`: the human decision on the source (required), one of three
  states: `under-review` (every new source, until a maintainer decides),
  `admitted` or `refused`, by whom and when. `reason` (why) is required to
  refuse, optional under review; an admission gives one only for an
  exception, such as an explicit permission for content without a license.
  The builds, the post-training readers and `check --fetch` without ids
  read only admitted sources; an `under-review` source is left out of them
  as a refused one is. A refused source stays in the repository as a
  reference and is never deleted.
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
checksum. `wpt-dataset fill-checksum <id> ...` fills in the checksum of a
new `source.json` (all zeros until then) from one download: it excludes in
`extract` each file that is not text ("Encodings", below) by its exact
path (`*`, `?` and `[` written `[*]`, `[?]` and `[[]`), reads the download
again with those exclusions (they change the history files too), keeps
the upstream tree in the store as the check keeps it, and checks that
source as `check --fetch` does, without downloading it again. A huggingface source's checksum comes
from the Hub's listing, with nothing excluded, and no upstream ref is set;
its check downloads it. For a dataset that holds archives (below),
`fill-checksum` downloads and unpacks every archive the source keeps, and
the check downloads them again. A source with `addresses` is fetched from
each of them, and a failed download names the address it was for.

A huggingface source unpacks the archives the dataset holds (`.tar`,
`.tar.gz`, `.tar.bz2`, `.tar.xz` and `.zip`, by the end of the file's name:
the kinds `http-archive` reads), so their members are the source's files,
in the archive's folder, where unpacking it in place puts them, and the
manifest and checksum cover the unpacked files, not the archives: a
dataset holding `part_1.tar.gz` that holds `part_1/a.json` gives the
source the file `part_1/a.json`. The listing gives no archive's members, so
`manifest` and `fill-checksum` download every archive the source keeps; an
`extract` exclusion matching an archive's own path leaves it out without
downloading it, and one matching a member's path leaves that member out.

The manifest is computed from the downloaded data itself (git's objects,
mercurial's manifest, archive headers), never from files written to a disk,
so Linux and Windows agree. A path that is not valid UTF-8, uses `\`, `.` or
`..`, or differs from another only by letter case makes the source fail; an
`extract` rule or a removal patch must leave it out first.

Files that are not text (images, binaries) make the source fail too:
exclude them in `extract`.

### Encodings

A file whose encoding is stated in the file itself is text in that
encoding: it is decoded to Unicode, and everything the pipeline stores or
emits from it (Markdown, tokens, wpt-web's views) is UTF-8; every encoding
stated this way converts without loss.
One function decides a file's encoding for every reader
(`packages/wpt-dataset/src/wpt_dataset/encoding.py`): the check,
`fill-checksum`, the Markdown of every format, `markdown-check`, the build,
wpt-web's file views and the evaluations that read a source's files. The
statements it reads, in this order:

| Where | Statement |
| --- | --- |
| any file | a byte-order mark: UTF-8, UTF-16 LE or BE, UTF-32 LE or BE |
| `html`, `code` in html | the HTML standard's prescan of the first 1024 bytes: `<meta charset="...">` or `<meta http-equiv="Content-Type" content="...; charset=...">`, read as the Encoding Standard reads its label (`iso-8859-1`, `latin1` and `ascii` are windows-1252, a UTF-16 label is UTF-8) |
| `html` and `code` in html when the prescan finds none (Project Gutenberg's older XHTML pages named `.html`) | the XML declaration's `encoding="..."`, its label read as the prescan reads one, as the HTML standard's prescan reads an XML declaration |
| `xhtml`, `code` in xml or svg | the XML declaration's `encoding="..."` |
| `text` | Project Gutenberg's header line `Character set encoding: NAME`, before its `*** START OF` line; `ISO-Latin-1` is Latin-1, and `ASCII` or `US-ASCII` with a byte above 127 does not decode |
| `code` in python | the PEP 263 coding line, in the first two lines |
| `code` in css | the `@charset "...";` that starts the file |

Nothing is guessed: a file that states no encoding is UTF-8 text, or no
text at all, as before. `chat` and `rating` files are JSON, UTF-8 by its
standard, and every file of a `task` folder is UTF-8 by the task format
([post-training.md](post-training.md)): no statement is read in them. A stated encoding Python does not know, or bytes
that do not decode under it, make the file no text: the check names the
file and why, and `fill-checksum` excludes it as it excludes a binary file.
A source drafted before this rule may hold such files in its `extract`
exclusions; to take them in, remove those exclusions, set `checksum` back
to the zeros and run `fill-checksum` again.

A file keeps its own bytes everywhere else: its checksum, `wpt-dataset
unpack` and the patches hold them as upstream wrote them, and git's diffs
carry them as they are, so a removal patch on a Latin-1 page is made and
applied like any other (edit the unpacked file in its own encoding).
wpt-web saves an edited file in the encoding the file it replaces states.
A git history's files (`.git/history/<commit>`) are WPT's own UTF-8 text,
as before: a diff's bytes that are not UTF-8 are replaced there, whatever
the changed file states.

### Patches

Patches use git's format and are applied with `git apply` inside a temporary
git index, each patch handed to git on its standard input, so no file is
written to a disk and modes and symlinks behave the same everywhere. Git is therefore a dependency of data processing. A patch
can add, change, delete files and change modes.

A git index silently drops every path under `.git/`, so every such path
(the files of a git history, `.git/history/<commit>` and
`.git/history/order`, or any other) is set aside while the patches apply
and comes back exactly as it was; a patch that names one fails, as git
refuses it. For the same reason `wpt-dataset patch` refuses a folder whose
`.git/` files differ from the source's: a patch cannot change them.

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
| `xhtml` | web pages written as strict XML, such as EPUB books, read by lxml (a malformed file stops the build): headings, paragraphs, lists, quotations, emphasis, tables; a dialogue row becomes `**Speaker:** words`; a note reference becomes a Markdown note (`[^12]`) with the note's text at the end of the document that refers to it, or is dropped when the note is not in the source; a label already taken by another note gets a number, `[^12-2]`, then `[^12-3]`; an image becomes a marker, below |
| `html` | web pages that are not strict XML, such as Project Gutenberg's, read by lxml's HTML parser (which refuses no page) and converted as `xhtml` pages are; EPUB's note marks exist only in XML, so an html page's notes are Project Gutenberg's: a link of class `fnanchor` (or `noteref`, `noteRef`) to an element of class `footnote` or `fndiv`, or to anything inside one, in the same page, is a note reference, and that element (or, when it holds the targets of links in several of its children, the child holding the target) becomes a Markdown note as an EPUB note does, with one label however many links refer to it; inside such a note, a link back to a reference (or to the empty anchor just before one), and the element of class `label` holding it, are left out; a note no such link refers to, and every other link, stay ordinary text where the page prints them, so `markdown-check` finds no unpaired note; printed page numbers, such as Gutenberg's `{92}` (class `pagenum`), stay as the page shows them |
| `records` | a dataset file holding many documents (JSON Lines, optionally `.gz` or `.zst`, or Parquet): each record whose license value an entry lists, and that the `keep` rules keep, is one document, `<file>#<record id>`; its text field is read as `text` above, kept as it is with `"as": "markdown"`, or converted as an `html` page with `"as": "html"`, for a dataset of web pages (no note comes from another file). A `title` field's value becomes the document's first line, `# <title>`, then a blank line. Ids repeat at no point in a file, unless `newest` names a date field (ISO 8601), which needs an `id` field that every record holds: of the records that share an id in all the files the entry covers, only the one with the latest date is read, and when the dates are equal the later one (in path order, then in its file), as a dataset holding several revisions of one document needs. Three more shapes: with `records`, a `.json` file (optionally `.gz` or `.zst`) is one JSON document, and that dotted field of it holds the list of records (`"records": "full articles"`); without a `license` field, every record takes the one license entry of its path, which lists no `values`, as a dataset stating one license for all its records needs; an `.xml` file is one record whose fields are XPath 1.0 expressions, their prefixes in the entry's `namespaces` (nodes give the string value of the first one, none when there is none; any other value as it is), and a `join` lets it read one more file of the tree: the one of the join's `paths` whose path holds the same key (`key`, a regular expression searched in the record file's path, `joined_key` in the joined file's, each giving its groups joined by one space, or its whole match), as the XPath variable the join's `name` names (an empty node set when no file holds the key); the joined files give no document. A newspaper issue published as a full-text file and a metadata file is read this way: the text from the first, the license value and the credit from the second, found by the issue's id in the other address's folder. XML is read with no external entity, no external DTD and no network; only the entities its own DTD declares are expanded |
| `chat` | conversations for post-training, one JSON object per line ([chat-format.md](chat-format.md)); each line is checked, and the file gives no pre-training document: only the post-training step named by its tags reads it |
| `task` | the files of task folders for agentic post-training (`task.json`, `files/`, `check`; [post-training.md](post-training.md)); each folder is checked, and its files give no pre-training document: only a post-training step that asks for a tag of its `task.json` reads it: by default each step asks for its own name, and `token-efficiency` asks for `grpo`. A task built by a program holds pointers into the documents of other sources, never their text, which the readers fill in from the store ([post-training.md](post-training.md), "Pointer tasks") |
| `rating` | people's pass or fail on the attempts of a `wpt-eval` run, one rating per line ([evaluation.md](evaluation.md), "People's ratings"); each line is checked, and the file gives no pre-training document: a measure only, which no training step reads |

In `xhtml` and `html` pages, an image the page shows (`<img>`, or an
inline `<svg>`) is never dropped silently: in its place the Markdown holds
the marker `[image]`, or `[image: ALT]` when it has alt text (an svg's
`<title>`), ALT being that text with its white space collapsed and
Markdown's characters escaped, so models learn where images were. A printed caption (`figcaption`,
`caption`) stays ordinary text; math, audio and video are left out.

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
| final source tree (and the check that its files are text) | checksum, cuts, each patch's sha256, the formats (they decide each file's encoding), wpt-dataset version |
| vocabulary | tokenizer code version, tokenizer recipe (with its `version`), the documents it is trained on (id and sha256 of their Markdown); frozen by the recipe's lock until `version` is raised ([tokenizer.md](tokenizer.md)) |
| dataset | wpt-dataset version, the model recipe's data part, mixture, vocabulary, every document's id, sha256 (of its Markdown) and tags, each source's address, revision, checksum, tree and licenses (with attribution), and the sources skipped with `--skip-failed` |

So "already done" is a lookup: nothing is downloaded or applied twice, and
changing one patch rebuilds only that source.

Beside the dataset's record (the ref `datasets/<id>`), the build keeps the
dataset's document list under the ref `dataset-documents/<id>`: one JSON
line per document, in the build's order, with its `id`, the `sha256` of its
Markdown (the one the dataset fingerprint reads) and its `tag`, `training`
or `held-out`, from the same split its shards are cut by. A new build
writes the list before the record's ref, so a dataset is never there
without it; a dataset built before builds kept the list never gets one. The
task reader takes the held-out texts from it
([data-selection.md](data-selection.md), "Held-out text and
contamination").

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
Because the cache is keyed by the wpt-dataset version, every change to
what the Markdown converter (`markdown.py`) makes of a file raises that
version in `packages/wpt-dataset/pyproject.toml` (then `uv lock`), in the
same change, so no cached source keeps its old Markdown; every dataset id
changes with it, which is accepted.
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

## The service layer

`packages/wpt-dataset/src/wpt_dataset/repo.py` is the only code that reads
or writes the repository's data: the
sources and their patches, the chat, task and rating files inside them, the
model, tokenizer and mixture recipes, the image Dockerfiles and their
pools of tools (`tools.json`), and the two
signer lists (`allowed-signers`, `allowed-gpg-keys.asc`). Every command
that uses this data goes through it: `wpt-dataset` (`check`, `migrate`,
`build`, `check-commits` and the rest), `wpt-train`, `wpt-posttrain`, the
image commands of `wpt-agent`, and `wpt-web`. It hands out documents, bytes
and versions, never a path under `data/`.

- Every file has a version: the sha256 of its bytes, or `""` for a file
  that does not exist yet. A source's files share one version, over its
  `source.json` and every patch. A save names the version it started from
  and is refused when the file changed since (a stale save).
- Every write applies the rules of `wpt-dataset check` to the new content
  first, and replaces the file in one step (a temporary file, then a
  rename), so a refused or interrupted save leaves the old file. Text files
  are written with `\n` line endings.
- One lock per repository holds each version check together with the write
  it allows, across processes as well as threads: two saves of one version,
  from two tabs of `wpt-web`, a double click, or `wpt-web` and
  `wpt-dataset` at the same moment, cannot both pass; the second is a stale
  save. Between processes it is a lock the operating system holds on the
  file `wpt-data.lock` in the repository's git folder (`.git`, or the
  folder a worktree's `.git` file names; `.wpt-data.lock` at the root of a
  folder that is not a git working tree), with `fcntl.flock` on Linux and
  macOS and `msvcrt.locking` on Windows. The system releases it when the
  process ends, a crash included, so a lock is never left behind to block
  the next save. Inside a process, the threads share one lock per
  repository (`wpt-web` serves each request on its own thread).
- A write through a symbolic link writes the link's target when that target
  is inside the repository, and is refused otherwise: a data file that is a
  link, or lies in a folder that is one, is written at the place the links
  lead to, after every link is resolved, and a place outside the repository
  refuses the write. So a write never lands outside the repository,
  whatever a link points to. The same rule holds for a link inside a
  source's own files: a change to a source's file whose path is a link, or
  lies under one, lands on the link's target when that target is inside the
  source, and is refused otherwise; removing the link itself stays allowed,
  whatever it points to. A link's target may be anywhere inside the
  repository, and the lock file stays in the git folder, or at the root of
  a folder that is not a git working tree.
- A license entry's `text` may be empty only while the source is not
  admitted, that is `under-review` or `refused`: the schema says so, and a
  person writes the text later (`wpt-dataset save source`, or the "Sources"
  page). Every reader of the sources (`wpt-dataset build`, the
  post-training readers of chat and task files, `check --fetch`, `check
  --changed`) leaves such a source out with a note naming it, instead of
  failing, and `wpt-dataset check` names it too, as a note that is not a
  failure. A source that is admitted needs its text, and a source with an
  empty text takes no files until it is written.
- JSON is written with two spaces of indent, characters outside ASCII kept
  as they are, except U+2028, U+2029 and U+0085, always written as escapes,
  even in values a save keeps unchanged, so that no line splitter cuts a
  line. A save over an existing JSON file keeps its bytes when its meaning
  is the same (key order and those escapes aside), and otherwise rewrites
  only the values that changed, the rest keeping its layout and key order;
  a new key goes after the key the saved document puts before it, and a new
  key or item is written with the indent and spacing of the last member of
  its object or array.
- Conversations, tasks and ratings are written only in an `empty` source,
  text written for WPT ([chat-format.md](chat-format.md), "Writing a chat
  file"; [post-training.md](post-training.md), "Tasks";
  [evaluation.md](evaluation.md), "People's ratings").
- A source's text changes only through a patch: the layer unpacks the
  source into a temporary folder, puts the change in, and records the
  difference as the next patch, kept only when the source then gives
  exactly those files and passes the checks of `wpt-dataset check --fetch`
  ([Patches](#patches)). A new chat file or task folder gets its own format
  entry in `source.json` when none covers it.
- Generated files (a tokenizer's `.lock.json`, an image's `lock.json` and
  `licenses.json`) are readable everywhere and written only by the command
  that makes them (`wpt-dataset build-tokenizer`, `wpt-agent lock-image`),
  through the layer, which checks them against their schema.
- The signer lists are checked too: each line of `allowed-signers` that is
  neither empty nor a comment holds a key (a key type such as
  `ssh-ed25519`, then the key), and `allowed-gpg-keys.asc` holds an
  ASCII-armored public key block.

Two checks hold the rule. A test
(`packages/wpt-dataset/tests/test_data_access.py`) reads the source code of
every package, application and script and fails when code outside `repo.py`
names the data folder in a way that could open a file under it: the name
`data` as a part of a path (either side of `/`, or an argument of a call
that builds or reaches a path, such as `Path()`, `joinpath()`,
`os.path.join()`, `os.walk()`, `os.listdir()`, `glob()` or `open()`), a
name bound to `"data"` or a comparison with it, a string or an f-string
that starts with `data/` or holds `/data/` or `:data/`, or a private member
of the service layer used from outside. A docstring, a command's help text
and a message (inside a `raise`, or given to `print()`, a log call or
`sys.exit()`) may name the folder for people. A syntax scan cannot follow
every way to build a path, so the service layer also sets an audit hook
(Python's `sys.addaudithook`) in every process, when its first `Repo` is
made: any opening, listing, removal or renaming of a path under a `Repo`'s
data folder, as written or with its symbolic links resolved, whose nearest
caller in the project's own code (not Python's library, not an installed
package) is neither `repo.py` nor a test fails with `DataAccessRefused`
before it happens, in every command and in the test suite. A test is a
`conftest.py` or a file under a `tests` folder, both inside the project's
checkout: a folder named `tests` above the checkout, or a `conftest.py`
outside it, does not make its code a test. The tests are left out of both:
they build the repositories they check and read what a command wrote.

### Two interfaces

Every kind of data has two interfaces, both thin layers over the service
layer: the HTTP API of `wpt-web`,
described in OpenAPI 3.1 at `/api/openapi.json` (the server checks every
request body against the same description), and `wpt-dataset` commands.

| Kind | `wpt-dataset` | `wpt-web` |
| --- | --- | --- |
| any kind of file: list, read, write | `kinds`, `list KIND`, `show KIND NAME`, `save KIND NAME FILE --version V` | `/api/kinds`, `/api/documents/{kind}[/{name}]`, `/api/schemas/{kind}`; pages "Recipes", "Images", "Signers" |
| a new empty source | `new-source ORIGIN [ITEM] NAME ...` (without `--license-text`, a person writes the license text later; an empty `--license-text` file is an error; `--format FORMAT=PATTERN` declares a format entry with no file) | `POST /api/sources`; page "New source" |
| a source's description | `show source ID`, `save source ID FILE --version V` | `/api/sources/{id}`; page "Sources" |
| a source's files | `files ID`, `read ID PATH`, `write ID PATH FILE`, `remove ID PATH`, `tags ID PATH [TAG ...]` | `/api/sources/{id}/files[/{path}]`, `/api/sources/{id}/tags/{path}` |
| conversations | `conversations ID PATH`, `conversation ID PATH LINE`, `save-conversation ID PATH LINE` (or `new`) | `/api/sources/{id}/conversations/{path}[/{line}]`, `POST /api/drafts`; page "Conversations" |
| tasks | `tasks ID`, `task ID FOLDER`, `save-task ID FOLDER` | `/api/sources/{id}/tasks[/{folder}]`; page "Tasks" |
| people's ratings of attempts ([evaluation.md](evaluation.md), "People's ratings") | `ratings ID PATH`, `save-rating ID PATH LINE` (or `new`) | `/api/sources/{id}/ratings/{path}[/{line}]`, and the attempts of `wpt-eval` runs from the store, `/api/runs[/{folder}/attempts]`; page "Ratings" |
| git's status of `data/` | `status` | `/api/status`, on every page |

The HTTP API answers 404 when the thing does not exist, 409 for a stale
save, 422 for content the rules refuse (with their messages) and 503 when
a tool or the store is missing on this machine. Both interfaces write files
and show git's status; commits stay in git.

`wpt-web` listens only on 127.0.0.1 and checks neither the `Host` header
nor a request's `Content-Type`: like
the commands, it asks for no login and no header. What that leaves open
while it runs: a web page open in the same browser can send it a
cross-origin `POST` with `Content-Type: text/plain`, which the browser
sends without asking first, so such a page can start sources
(`POST /api/sources`) or ask for drafts (`POST /api/drafts`); and a site
that makes its own name resolve to 127.0.0.1 (DNS rebinding) can reach
every route, reading people's texts and changing data files in the working
tree. Nothing is committed by `wpt-web`, so every change shows in
`git status` and in the page's list of changed files before a commit.

## Packages

| Package | Role |
| --- | --- |
| `wpt-dataset` | all data rules: sources, manifests, cuts, patches, pools, builds, the checks CI runs, migrations; the service layer (`repo.py`), the only code that reads or writes `data/`; the task builder (`builder.py`, run by `scripts/build_tasks.py`); command `wpt-dataset` |
| `wpt-store` | the content-addressed store (local or S3), its record folders, and the token shard format with its checked reader, so training and evaluation read data without depending on the data rules |
| `wpt-telemetry` | the record of one command run in the store, sent live to an OpenTelemetry backend when one is set ([observability.md](observability.md)) |
| `wpt-tokenizer` | the byte-level BPE tokenizer, Rust inside |
| `wpt-web` | a local web interface over the service layer: one small server on 127.0.0.1, one file per page (Python routes, React page), an editor for every kind of data, a conversation editor that asks a WPT model folder for drafts, and its OpenAPI description; its interface is built with Bun when it starts; command `wpt-web` |
| `wpt-model` | the WPT network ([model.md](model.md)) and its drafter, its safetensors model folders, and their license documents (`LICENSE`, `NOTICE`, model card) |
| `wpt-training` | the training loop (global batch, evaluation, checkpoints, resuming), with two plug points: strategies (how the hardware is used) and methods (how the model learns) |
| `wpt-inference` | sampling through the model's generation cache: several rows from one prompt, the model's next chat message (`next_message`, which `wpt-chat --chat` and the text measures of `wpt-eval` both write with), speculative decoding with a drafter and its scheduler, the scheduler's pass times (measured, recorded, replayed); training a model's drafter (`drafting.py`), which `wpt-train` and `wpt-posttrain` run at the end of every run, so it uses `wpt-model` |
| `wpt-utils` | helpers several packages share, one submodule each: `wpt_utils.device` picks the best GPU, else the CPU; `wpt_utils.programs` starts the outside programs WPT runs (`program(name)`, with a clear error when one is missing) and reads the repository commit (`repository()`), without importing PyTorch; `wpt_utils.files.sha256_file` is the one way a file is hashed, by the store, the model folders and the run records alike; `wpt_utils.settings.typed` types a training method's and a post-training step's settings from the defaults its file declares |
| `wpt-agent` | the agent harness of agentic post-training: the tools (the four built-in ones, `ask`, and those a task or its image's pool declares, answered by programs in the image) and each attempt's choice of them, one Docker container per attempt, the images of `data/images/` and their locks, the agent loop and the task's check, and the playing of a task's attempts (`wpt_agent.play`), which the `grpo` and `token-efficiency` steps and the agent measures of `wpt-eval` share, so `wpt-eval` does not depend on `wpt-posttrain`; commands `wpt-agent lock-image` and `wpt-agent build-image` |
| `wpt-train`, `wpt-eval`, `wpt-posttrain`, `wpt-chat` | the applications |

## Checks

`uv run scripts/check.py` runs every check, the same way on any machine
and in CI (`.github/workflows/ci.yml`, on every pull request and push), so
a green run before pushing means a green CI. It compares with the upstream
branch, as a push would:

- `uv.lock` matches every `pyproject.toml`.
- The tests of every package, and the Rust tests. None needs a server: the
  S3 test runs against moto, which answers S3 requests inside the test
  process. The tests run in parallel (pytest-xdist), each process limited
  to one PyTorch thread and kept off the GPU, as in CI, since a GPU that
  shares the machine's memory gives every process its own share of it; a
  plain `uv run pytest` still uses the GPU. The number of processes is the
  smaller of 75% of the cores and 75% of the memory available when the
  check starts divided by 2.5 GB, the measured memory of one test process
  rounded up, and at least 1; where the memory cannot be read, the cores
  alone decide. The check prints the number and both inputs.
- The image build tests (marked `image_build`, in
  `packages/wpt-agent/tests/test_images.py`, and the test of the agent
  image's tools and check library in `test_agent_image.py`) lock and build
  real images, which takes minutes. They run only when the push changes
  image code: `packages/wpt-agent/src/wpt_agent/images.py`,
  `packages/wpt-agent/tests/test_images.py`, `test_agent_image.py`,
  `packages/wpt-agent/tests/conftest.py`,
  `packages/wpt-dataset/src/wpt_dataset/repo.py` (its `Repo.image_files`
  decides which files an image's tag covers), the schemas
  `image-lock.schema.json` and `image-licenses.schema.json` in
  `packages/wpt-dataset/src/wpt_dataset/schemas/`, or anything under
  `data/images/`. Otherwise the check leaves it out and says so; a plain
  `uv run pytest` runs it with every other test.
- The two GRPO tests that run attempts in a real image
  (`apps/wpt-posttrain/tests/test_posttrain.py`) keep its lock in
  `.pytest_cache/` and the built image in Docker, for as long as the
  image's files and `wpt_agent/images.py` stay the same. Only the first
  check on a machine, and every CI run (which starts empty), pays for
  locking and building it; when those files change, the next check locks
  and builds again and removes the old image.
- `wpt-dataset check`: every file matches its schema and current shape,
  source ids match their address, licenses are on the allowed list (with
  their text), patches are numbered without gaps, mixture weights add up to
  1 and pool expressions parse, recipes name existing recipes.
- `wpt-dataset check --fetch
  standardebooks-joseph-conrad-ford-madox-ford-the-nature-of-a-crime-5366b0f3`:
  one short book, fixed at one git commit so its download never changes, is
  fetched, and its checksum, cuts, patches, license map and text are
  checked, which proves the fetch and its checks work. The sources a pull
  request adds or changes are not fetched here: the maintainer runs
  `wpt-dataset check --fetch` on them before merging.
- `wpt-dataset check-commits <range>`: DCO sign-off on every commit, and a
  signature from an allowed key on every commit that changes a decision.

It runs every check even after one failed, then lists which passed. There
is no local hook: run it before pushing.

## Migrations

Each data file carries its shape version (`schema`). When a shape changes,
the change comes with a migration step in
`packages/wpt-dataset/src/wpt_dataset/migrate.py`, and `wpt-dataset
migrate` rewrites every file that is behind. Nobody edits the files by hand
for it, except where a new field holds something only a person can write:
shape 2 of `source.json` adds `attribution`, which the migration leaves
empty, and the check then names each file to fill in. Shape 4 adds each
license's `text`: the migration moves a `LicenseRef-<name>.txt` file into
it and leaves any other text empty for a person to fill. Shape 5 adds the
`huggingface` type, the `records` format and a license entry's `values`;
only the number changes. Shape 6 gives a source three states: the migration
turns a refusal whose reason is "under review" into `under-review` without
that reason, and turns `retired` into `refused`, keeping its reason. Shape 7 adds `addresses` and the records format's
`records`, `namespaces`, `join`, `newest`, `title` and `"as": "html"`, and makes its
`license` field optional; only the number changes.
