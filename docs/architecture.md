# Architecture

**The repository describes a corpus; it does not hold one.** It holds the
description of every source (where it is fetched from, how, its license
entries and the decisions on them), the license texts, patches and build
recipes. The data itself is downloaded into a store, as upstream serves it
when asked (a git repository at the commit its source names), and every
build checks it, on the bytes the store holds.

- The repository stays small, publishable and reviewable in git.
- Every change to a source is a patch, separate from the source itself.
- Licenses are checked against the bytes actually held, at the places each
  source declares, by every build.
- A git source is pinned to one commit, its `revision`: upstream's later
  commits enter only when a person moves it. Every other source is not
  pinned, so a rebuild from upstream gives its current content,
  license-checked. A dataset's document list (each document's id and the
  sha256 of its Markdown) records what a model trained on.

## Three layers

1. **Upstream**: the content as its source publishes it, never modified and
   never committed. Only its description is in the repository.
2. **Erasure cuts and patches**: changes made by hand, kept as data next to
   the description.
3. **Build**: everything mechanical and repeatable from the stored bytes
   (license check, deduplication, held-out split, tokenization, shards). It
   is not content, so it is not a patch.

## Two pieces

WPT turns sources into datasets in two pieces. They make one whole, but
each is designed and changed on its own, and neither does the other's job.

1. **Download** puts a source's bytes into the store
   ([The download step](#the-download-step)). It decides nothing about
   licenses or content.
2. **Build** turns the bytes in the store into a dataset
   ([The build](#the-build)). It never contacts upstream.

What they share is defined once and owned by neither: `source.json`, the
license entries and the license texts, the three levels of decision
([licenses.md](licenses.md), "Who decides"), what a protocol provides
([Protocols](#protocols)) and the shared checks
([Shared checks](#shared-checks)).

Upkeep of the repository is commands a person runs; nothing runs on a
schedule and no machine stays on for it:

- A person adds a source, or moves an address to a newer version, by
  editing `source.json` (the "Sources" page of `wpt web`, `wpt dataset save
  source`), or with a script kept outside the repository.
- A person downloads into a new session to get fresh content
  ([The store](#the-store)).
- A person reads a build's report and adds the license entry or the
  non-text path it names, or runs the task builder again
  ([training.md](training.md), "The task builder").
- The maintainer commits.

## Components and services

Each component has one role. Every operation is a service: a function with
its inputs, its result and its named failures. The `wpt dataset` commands
and the `wpt web` routes are controllers: each reads its input (arguments,
a request), calls the same services, and presents the result. A controller
holds no rule of its own, so the two interfaces cannot disagree. A job
started from `wpt web` runs a `wpt` subcommand in its own process, which
calls the same services.

| Component | Role | Code |
| --- | --- | --- |
| service layer | the only code that reads or writes `data/` ([The service layer](#the-service-layer)) | `repo.py` |
| protocols | turn an address into files; each owns its network effects and its named failures | `protocols/`, with the helpers `robots.py`, `retry.py`, `sites.py` and `lfs.py` |
| download | decides which sources to fetch, asks their protocol, writes each source's file and record into a session of the store; never judges licenses | `download.py` |
| license check | changes nothing: given a source's entries, its files or records and the license texts, it answers which entry covers each file or record, and for each entry whether the statement at its place matches a text it expects, which one, and that text's decision | `license_check.py` |
| shared checks | change nothing; defined once and used by every command that applies them | `checks.py`, `erasure.py` |
| build | asks the store for bytes, the shared checks and the license check; decides what is left out; writes its results to the store; never contacts upstream | `build.py`, `kept.py` |
| contamination check | reads a built dataset's shards and saves a report in the store; removes nothing, and nothing later reads the report | `contamination.py` |
| task builder | builds pointer tasks from stored bytes, through the service layer | `builder.py` |

A controller asks a service; download asks protocols and the store; the
build asks the store, the shared checks and the license check. No
component asks a controller. The code lives in
`packages/wpt-dataset/src/wpt_dataset/`.

## Protocols

A protocol turns an address into files. Download and build know only
protocols and files: nothing below a protocol knows which site a source
comes from. A protocol provides only:

- **Fetching**: given an address and the protocol's own options, it
  delivers files, each a path and its bytes, and returns what it got, for
  the record only (the commit, for example).
- **Its failures, each named**: source gone (`SourceGone`), refused for too
  many requests (`TooManyRequests`), refused by robots rules
  (`CrawlForbidden`), robots rules unreadable (`RobotsUnreadable`), host
  unreachable (`HostUnreachable`), server error through every retry
  (`ServerError`), access refused (`AccessRefused`: HTTP 401, 403 or 451,
  or a password prompt), incomplete (`Incomplete`). Anything else, such as
  a message never seen before or a local error, is the general fetch
  failure, `FetchFailed`. A source whose main address and mirrors fail in
  different ways fails with its main address's kind.

A source's `type` names its protocol:

- `http`: one file per address. A file whose name (`extract.name`, or the
  address's last path part) ends as an archive's (`.zip`, `.tar`,
  `.tar.gz`, `.tgz`, `.tar.bz2`, `.tbz2`, `.tar.xz`, `.txz`) is unpacked
  into its members; any other is one file under that name.
- `git`: a repository on any host, at exactly the commit the source's
  `revision` names. Its options are `history.since`, `submodules`, and
  LFS: a file the
  repository stores outside itself is fetched by its sha256 through the
  host's LFS batch request. A Hugging Face dataset is such a repository
  (`https://huggingface.co/datasets/<owner>/<name>`): its file list comes
  from git, and a first download brings the whole history with every
  content, which what the source does not keep never leaves.
- `mercurial`: a repository at its tip.
- `empty`: text written for WPT. It has no protocol, downloads nothing and
  starts with no files; its patches hold every file.

Robots rules and request limits apply per site, through the helpers the
protocols use. Markdown conversion is chosen by file format. Another
protocol is one more module in `protocols/` and one more name in
`type`; nothing else changes.

## Sources

One folder per source: `data/sources/<id>/` holds `source.json` and its
patches `0001-<name>.patch`, `0002-<name>.patch`... applied in that order.

### The id

`<origin>-<item>-<hash>`, or `<origin>-<hash>` when the source has no
natural item (lowercase `a-z 0-9 -`), for example `gutenberg-1342-05bda734`.
`wpt dataset source-id <origin> [<item>] <address>` prints the id a new
source gets: `<hash>` is the first 8 hex characters of the sha256 of its
first address. The id is given once, when the source is created, and is
never derived from the address again: an address that changes, such as a
version moved in it, keeps the same source, its id, its decisions and its
erasure cuts. An id is never reused.

### source.json

```json
{
  "type": "http",
  "address": "https://www.gutenberg.org/cache/epub/1342/pg1342-h.zip",
  "extract": { "strip": 1, "exclude": ["**/*.jpg"], "note": "" },
  "licenses": [ { "license": "LicenseRef-public-domain", "paths": ["*.html"],
                  "attribution": "Jane Austen, Pride and Prejudice, Project Gutenberg, https://www.gutenberg.org/ebooks/1342",
                  "text": "Public domain. Jane Austen died in 1817; ...",
                  "place": { "entry": true },
                  "decision": { "decision": "admitted", "by": "...", "date": "2026-10-14" }, "note": "" } ],
  "formats": [ { "format": "html", "paths": ["*.html"], "note": "" } ],
  "tags": [ { "tags": ["en", "literature"], "paths": ["*.html"], "note": "" } ],
  "mirrors": [ { "address": "https://...", "note": "" } ],
  "cuts": [ { "path": "...", "length": 0, "hmac": "...", "rolling": 0, "note": "" } ],
  "decision": { "decision": "admitted", "by": "...", "date": "2026-10-14", "note": "" },
  "note": ""
}
```

- `type`: the protocol ([Protocols](#protocols)). An `empty` source's
  `address` is a name, `empty:<name>`.
- `address`: where to fetch it. What upstream serves there when asked is
  what is taken; only a git source names a version, its `revision`, and no
  source names a checksum.
- `revision` (git only, required): the full id of the commit the download
  fetches, 40 hexadecimal characters. Whoever writes a git source writes
  it; a later upstream commit enters only once a person writes its id
  here, and the check refuses a git source without one.
- `addresses` (`http` only): one source fetched from several addresses,
  each into its own folder of the source's tree, for a publisher that puts
  documents in one archive and their metadata in another:
  `[{"folder": "fulltext", "address": ".../fulltext.zip"},
  {"folder": "metadata", "address": ".../metadata.zip"}]`. Each address is
  read with the source's `extract` rules (`strip` on the paths of what it
  holds, `exclude` on the paths with their folder) into one tree.
  `address` repeats the first one's address; a source with `addresses` has
  no `mirrors`. Every other source keeps its one `address`, unpacked at the
  tree's root.
- `extract`: `strip` removes leading path components (entries left with no
  path are dropped), `exclude` leaves paths out, and `name` gives an `http`
  file its name when the address does not.
- `licenses`: the license entries. Each maps an SPDX identifier (or
  `LicenseRef-<name>`, or an expression) to the paths it covers, names the
  place where its license statement must be found and what is expected
  there, and carries its own decision ([licenses.md](licenses.md),
  "License entries"). Every path matches exactly one entry. Each entry's
  `attribution` (required) is the credit that license asks for, written by
  a person: authors, title, copyright line, link. Its `text` (required) is
  the license as the source gives it, with its copyright line and, for
  Apache-2.0, the source's `NOTICE` content; for
  `LicenseRef-public-domain`, why the work is in the public domain. Every
  model trained on the source repeats both in its `NOTICE`.
- `formats`: how each file becomes Markdown, mapped to paths like
  `licenses`: every path matches exactly one entry. Training reads only
  Markdown (see "Documents are Markdown" below).
- `tags`: describe paths (language, domain); pools select documents by
  tags.
- `mirrors`: other addresses for the same source, with no promise of
  identical bytes.
- `history` (git only): adds one file per commit from `since` to
  `revision`, `.git/history/<commit>`, in `git format-patch` layout;
  `since` must be an ancestor of `revision`. A commit that also changes
  files `extract` leaves out gets a
  diff limited to the kept paths; when a kept path is also the folder of
  paths left out (a submodule replacing a folder of the same name), those
  paths are excluded from the diff by name. One more file,
  `.git/history/order`, holds the commit order: every commit from `since`
  to `revision`, merges included, newest first and each before its
  parents (git's topological order), one per line, a merge's line ending
  in ` merge`. The order file is part of the stored tree, so the tree's
  manifest covers it. It is not a training document: `markdown.documents`,
  which every list of documents reads, leaves it out. Only patterns
  starting with `.git/history/` give these files their license entry
  ([licenses.md](licenses.md), "Rules"), and only format patterns
  starting with `.git/history/` give them their format entry; every other
  format pattern never matches them.
- `submodules` (git only): the address of each submodule the source reads,
  by its path, read at the commit the tree of `revision` names for it.
- `cuts`: erasure cuts; see [erasure.md](erasure.md).
- `held_out_by`: a `pattern` (a Python regular expression) searched in each
  document's path; where it is found, the held-out hash reads its groups
  instead of the document id, so documents of any source with the same
  groups are held out together; see [data-selection.md](data-selection.md),
  "Held-out text and contamination".
- `decision`: the source's decision (required): whether WPT wants the
  source at all ([data-selection.md](data-selection.md), "The source's
  decision"). One of three states: `under-review` (every new source, until
  a maintainer decides), `admitted` or `refused`, by whom and when.
  `reason` (why) is required to refuse, optional under review; an
  admission gives one only for an exception, such as an explicit
  permission for content without a license. The builds and `check --fetch`
  without ids read only admitted sources; an `under-review` source is left
  out of them as a refused one is. A refused source stays in the
  repository as a reference and is never deleted.
- Every object may hold a free-text `note`. It is data, so it survives
  migrations (comments would not).

`CONTRIBUTING.md`, "Adding a source", says how to fill `history`,
`submodules` and the `records` format's fields, including its `keep`
rules.

Globs: `*` matches within one folder, `**` across any number of folders.

The schema is `packages/wpt-dataset/src/wpt_dataset/schemas/source.schema.json`.
Mixture, model recipe, model series, tokenizer recipe, lock and license
text files have schemas there too.

### The manifest

The store names the content of a tree by its manifest: the sha256 of the
list of its files, one line per entry, sorted by the UTF-8 bytes of the
path:

```
<mode> <sha256> <path>
```

`100644` is a file, `100755` an executable file, `120000` a symlink (its
sha256 is that of the target text). Empty folders do not appear. The
manifest is a plain hex digest, and it has that name and that form
everywhere it appears: a source's record in the store, a dataset's
record, the conversations' and tasks' records, a task's repository pointer, and a
model's `NOTICE` and card, which name each source by the manifest of the
upstream tree it was built from.
`wpt dataset manifest <id>` downloads a source as `download` does (or
finds it in the store) and prints the list of its upstream tree's files
and that tree's manifest.

The list of a tree's files is computed from the downloaded data itself
(git's objects, mercurial's manifest, archive headers), never from files
written to a disk, so Linux and Windows agree. A path that is not valid
UTF-8, uses `\`, `.` or `..`, or differs from another only by letter
case makes the source fail; an `extract` rule or a removal patch must
leave it out first.

A file that is not text (an image, a binary) is listed in `source.json` by
its exact path, as an `extract` exclusion (`*`, `?` and `[` written `[*]`,
`[?]` and `[[]`), so the store never holds it. The build leaves out alone
a file that is not text and is not listed, and reports it, so a person
adds the path.

### Encodings

A file whose encoding is stated in the file itself is text in that
encoding: it is decoded to Unicode, and everything the pipeline stores or
emits from it (Markdown, tokens, wpt web's views) is UTF-8; every encoding
stated this way converts without loss.
One function decides a file's encoding for every reader
(`packages/wpt-dataset/src/wpt_dataset/encoding.py`), through the text
check ([Shared checks](#shared-checks)): the check, the build's text check,
the Markdown of every format, `markdown-check`, the build,
wpt web's file views and the evaluations that read a source's files. The
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
text at all. `chat`, `rating` and `probe` files are JSON, UTF-8 by its
standard, and every file of a `task` folder is UTF-8 by the task format
([training.md](training.md)): no statement is read in them. A stated encoding Python does not know, or bytes
that do not decode under it, make the file no text: `check --fetch` names
the file and why, and the build leaves it out as it leaves out a binary
file.

A file keeps its own bytes everywhere else: the store, `wpt dataset
unpack` and the patches hold them as upstream wrote them, and git's diffs
carry them as they are, so a removal patch on a Latin-1 page is made and
applied like any other (edit the unpacked file in its own encoding).
wpt web saves an edited file in the encoding the file it replaces states.
A git history's files (`.git/history/<commit>`) are WPT's own UTF-8 text: a diff's bytes that are not UTF-8 are replaced there, whatever
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
refuses it. For the same reason `wpt dataset patch` refuses a folder whose
`.git/` files differ from the source's: a patch cannot change them.

Patches are not written by hand: `wpt dataset unpack <id> <folder>` writes a
source's final files (after cuts and patches) into a folder outside the
repository; after editing them, `wpt dataset patch <id> <folder> <name>`
records the difference as the next patch, `NNNN-<name>.patch`. It then
applies every patch again and keeps the new one only if the source gives
exactly the folder's files and passes its license map.

Removal patches are allowed even though their `-` lines show the removed
text: this repository is the software that downloads and treats the data,
and the project is what comes out of it. Removed material is not used, not
endorsed, and explicitly removed; the removal is itself the record that a
license was respected.

### When a source fails

A failure is judged per item: a source, a license entry, a file, a task.
Training from a commit where some fail, each left out and named, is
allowed.

A download that fails for a reason that may pass (a timeout, a cut
connection, an HTTP answer of 408, 425, 429 or 5xx, a git or Mercurial
network error) is tried again, up to 5 attempts in all. The wait doubles
from 4 seconds up to 60, each one drawn at random below that limit, and
follows a server's `Retry-After` up to 60 seconds. Anything else (a 404, a
robots.txt refusal) fails at once, since asking again gives the same
answer (`wpt_dataset/retry.py`).

A download that still fails fails that source in the download step, with
its named failure ([Protocols](#protocols)). It stores nothing for that
source: the store keeps what it held before. `wpt dataset build` reads
what the store holds; a source that is not in the store is left out, and
listed, with its error (`NotDownloaded`), under `skipped` in the dataset
record and in the build's run record; the model's `provenance.json`
repeats the list as `skipped_sources`. A cut or a patch that does not
apply leaves the whole source out the same way, until a person makes the
cut or the patch again.

A source in the store that fails in the build for another reason (its
format map fails, a file is broken) is left out and listed the same way.
A build that trains a new vocabulary lists each source it left out in the
lock (`left_out`). With `--strict` (or `WPT_STRICT=1`), any failing
source stops the build instead, including one that is not in the store
or whose cut or patch does not apply; every source whose format map fails
is named before any text is read. A missing program (such as `hg`) stops
the build either way, since it fails every source that needs it.

`wpt dataset download` and `wpt dataset check --fetch` try every source
they were given, even after one failed, and end with the list of failures
and their reasons.

## Documents are Markdown

Every training document is the Markdown of one file of a source's final
tree, as its `formats` entry says
(`packages/wpt-dataset/src/wpt_dataset/markdown.py`):

| Format | Becomes |
| --- | --- |
| `markdown` | the file as it is |
| `text` | paragraphs as written; each line break inside a paragraph is a Markdown line break; Markdown's own characters are escaped, so the text reads exactly as written |
| `code` | one fenced code block whose opening line names the language and the path: ```` ```python path="./src/main.py" ```` (the entry's `language`); in a git or Mercurial source the path starts with `./`, the repository's root, and in any other source it is written as it is (```` ```python path="src/main.py" ````), since no repository root exists |
| `xhtml` | web pages written as strict XML, such as EPUB books, read by lxml (a malformed file stops the build): headings, paragraphs, lists, quotations, emphasis, tables; a table inside a table's cell follows that table as a table of its own; a dialogue row becomes `**Speaker:** words`; a note reference becomes a Markdown note (`[^12]`) with the note's text at the end of the document that refers to it, or is dropped when the note is not in the source; a label already taken by another note gets a number, `[^12-2]`, then `[^12-3]`; an image becomes a marker, below |
| `html` | web pages that are not strict XML, such as Project Gutenberg's, read by lxml's HTML parser (which refuses no page) and converted as `xhtml` pages are; EPUB's note marks exist only in XML, so an html page's notes are Project Gutenberg's: a link of class `fnanchor` (or `noteref`, `noteRef`) to an element of class `footnote` or `fndiv`, or to anything inside one, in the same page, is a note reference, and that element (or, when it holds the targets of links in several of its children, the child holding the target) becomes a Markdown note as an EPUB note does, with one label however many links refer to it; inside such a note, a link back to a reference (or to the empty anchor just before one), and the element of class `label` holding it, are left out; a note no such link refers to, and every other link, stay ordinary text where the page prints them, so `markdown-check` finds no unpaired note; printed page numbers, such as Gutenberg's `{92}` (class `pagenum`), stay as the page shows them |
| `records` | a dataset file holding many documents (JSON Lines, optionally `.gz` or `.zst`, or Parquet): each record that a license entry covers (the entry whose field place expects the value of that field in the record) and that the `keep` rules keep is one document, `<file>#<record id>`; its text field is read as `text` above, kept as it is with `"as": "markdown"`, or converted as an `html` page with `"as": "html"`, for a dataset of web pages (no note comes from another file). A `title` field's value becomes the document's first line, `# <title>`, then a blank line. Ids repeat at no point in a file, unless `newest` names a date field (ISO 8601), which needs an `id` field that every record holds: of the records that share an id in all the files the entry covers, only the one with the latest date is read, and when the dates are equal the later one (in path order, then in its file), as a dataset holding several revisions of one document needs. Three more shapes: with `records`, a `.json` file (optionally `.gz` or `.zst`) is one JSON document, and that dotted field of it holds the list of records (`"records": "full articles"`); when the one license entry covering the file has no field place, every record takes it, as a dataset stating one license for all its records needs; an `.xml` file is one record whose fields are XPath 1.0 expressions, their prefixes in the entry's `namespaces` (nodes give the string value of the first one, none when there is none; any other value as it is), and a `join` lets it read one more file of the tree: the one of the join's `paths` whose path holds the same key (`key`, a regular expression searched in the record file's path, `joined_key` in the joined file's, each giving its groups joined by one space, or its whole match), as the XPath variable the join's `name` names (an empty node set when no file holds the key); the joined files give no document. A newspaper issue published as a full-text file and a metadata file is read this way: the text from the first, the license value and the credit from the second, found by the issue's id in the other address's folder. XML is read with no external entity, no external DTD and no network; only the entities its own DTD declares are expanded |
| `chat` | conversations for training, one JSON object per line ([chat-format.md](chat-format.md)); each line is checked, and the file gives no pre-training document: a dataset's build keeps each conversation with its file's tags, and only a training step whose `tag` setting is one of them reads it |
| `task` | the files of task folders for agentic training (`task.json`, `files/`, `check`; [training.md](training.md)); each folder is checked, and its files give no pre-training document: a dataset's build keeps each task with the tags of its `task.json`, and only a training step whose `tag` setting is one of them reads it: `finish-tasks` and `finish-tasks-with-less-writing` both ask for `finish-tasks`. A task built by a program holds pointers into the documents of other sources, never their text, which the readers fill in from the store ([training.md](training.md), "Pointer tasks") |
| `rating` | people's pass or fail on the attempts of a `wpt eval` run, one rating per line ([evaluation.md](evaluation.md), "People's ratings"); each line is checked, and the file gives no pre-training document: a measure only, which no training step reads |
| `probe` | questions the `probes` test of `wpt eval` asks a model and the patterns that flag an alarming reply, one probe per line ([evaluation.md](evaluation.md), "Probes"); each line is checked, its patterns included, and the file gives no pre-training document: a measure only, which no dataset build keeps and no training step reads |

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
of its inputs. A result of the data pipeline (a source's final tree and its
text check, what a source gives the index, what a source keeps for training,
the index of a records entry under `newest`, a source's Markdown and tokens, a
vocabulary, a dataset) holds no code of WPT: a change to the code makes no such
result unknown, so work goes on from what the store holds whatever commit made
it. Its fingerprint holds every input of `data/` its computation reads, so a
commit that changes only other files of `data/` leaves it the same and it is
found again.

Each of those stages has a version beside it in the code, a named integer that
its fingerprint holds: `TREE_VERSION` and `NOT_TEXT_VERSION` (`sources.py`),
`KEPT_VERSION` (`kept.py`), `INDEX_VERSION` (`choice.py`), `NEWEST_VERSION`
(`records.py`), `MARKDOWN_VERSION` and `TOKENS_VERSION` (`reading.py`),
`VOCABULARY_VERSION` and `DATASET_VERSION` (`build.py`). A person raises a
stage's version when a change to the code changes what the stage gives for the
same inputs, and every later build on any machine then does that stage again.
A package or Python upgrade that changes what a stage gives counts as a change
to the code. A locked vocabulary is never trained again, so `VOCABULARY_VERSION`
acts only on a recipe with no lock, and a new vocabulary takes raising the
recipe's `version` ([tokenizer.md](tokenizer.md)).

What a stage gives enters the next stage's fingerprint, so a stage that gives
something else makes the stages after it again too. Three stages reach what
reads them by their version instead. A dataset's id names its documents by
their Markdown, so it holds `TOKENS_VERSION` as well. What a source keeps holds
`NOT_TEXT_VERSION`, because its stored result carries the text check's answer.
The Markdown of a source with a records entry under `newest` holds
`NEWEST_VERSION`, because a read loads that index only when it makes the
Markdown.

Each stored result says which code made it (`wpt_utils.programs.code_name`: the
commit, and a hash of the uncommitted changes when there are some). The first
line of a result holds it after the result's sha256 (`wpt_telemetry.set_result`),
so finding the result says it at no further read; a pack of Markdown holds it in the
file that describes the pack, and a dataset's record in its `tools`. A build
that reuses results another commit made prints one warning per stage, with how
many came from which commit, and carries on: it is a person's call whether that
code still gives those results.

Any other result (a training step's) holds the code as the commit of the
checkout it runs from plus a sha256 of its uncommitted changes, every
tracked file's difference and every untracked file git does not ignore
(`wpt_utils.programs.code`). The commit is the WPT repository's whole
commit, so it covers `data/` too. Any commit, and any uncommitted change,
makes every result keyed by it unknown.

The license check reads the license texts, which every source shares.
Each build runs it again for every source, outside the store's results,
before the work it keeps: a few small reads per source of the stored
bytes at each entry's place. Its answer, for each entry the license text
found at its place and that text's decision, is part of the fingerprint
of what a source keeps for training. A new license text that a source
does not match, or a change to another text, leaves that answer and that
fingerprint the same.

A task source's tasks are checked against every source id and the files
of `data/images/`. That check also runs on every build, so it is no part
of any fingerprint. A commit that changes only `data/` therefore converts
and tokenizes again only the sources whose own inputs changed.

A session keeps its downloads across commits. Code installed as packages,
outside a git checkout, is identified by a sha256 of the installed
`wpt-*` packages' files instead in `code`.

| Result | Fingerprint covers |
| --- | --- |
| upstream tree (what the download stored; the source's record holds this fingerprint, which tells one download from another) | the source's id and what its download reads: its `type`, `address`, `addresses`, `revision`, `extract`, `history`, `submodules`, and the outside license pages its entries declare |
| final source tree (and the check that its files are text) | the upstream tree's manifest, the license pages stored with it, cuts (and the cut key), each patch's sha256, the formats (they decide each file's encoding), `TREE_VERSION`; its text check the tree, the formats and `NOT_TEXT_VERSION` |
| what a source gives the index | the source's id and its whole `source.json`, the fingerprint of its final tree, the sha256 of its tar (which alone says where each file lies, and which a new download of the same files may change), `INDEX_VERSION` |
| what a source keeps for training, and its check result | the final tree, the license entries (with their decisions), the formats, the license check's answer (for each entry, the license text found at its place and that text's decision), `KEPT_VERSION`, `NOT_TEXT_VERSION` |
| the Markdown of a source's chosen files (on this machine) | the fingerprint of what the source keeps for training, `MARKDOWN_VERSION`, the source's `type`, and `NEWEST_VERSION` for a source with a records entry under `newest` |
| a part of a source's tokens | that Markdown's fingerprint, `TOKENS_VERSION`, the vocabulary, the paths of the part's chosen files |
| vocabulary | `VOCABULARY_VERSION`, the tokenizer recipe (with its `version`), the sha256 of the list of the documents it is trained on (id and sha256 of their Markdown); frozen by the recipe's lock until `version` is raised ([tokenizer.md](tokenizer.md)) |
| dataset | `DATASET_VERSION`, `TOKENS_VERSION`, the model recipe's data part, mixture, vocabulary, every document's id, sha256 (of its Markdown) and tags, each source's address, the manifest of its upstream tree, its final tree, licenses (with attribution) and check result, the sources skipped (each failing source: by its id and the kind of its error), and the files of its conversations and tasks |
| training step's result | the fingerprint of the step before it, the step (its files, method, environment, optimizer, settings and locked parts), the fingerprint of the dataset part it reads (`text`, `conversations` or `tasks`, in the record's `fingerprints`), the recipe's seed and the code; no hardware option ([training.md](training.md), "Step results and reuse") |

So "already done" is a lookup for the same data and versions in the store's
`cache/`, which every session shares: nothing is applied twice. A dataset whose record lacks a field the build writes is
built again.

A build reads the session it is given, and the one that session started
from, and writes the dataset under the session: `<id>/datasets/<name>/`
holds the dataset whole. The name is the recipe's, or `series-<series>` for
a size of a series, a hyphen, and the first 8 hex characters of the dataset
id. A build with other content writes another complete folder; the same
content finds the folder it was written in, by its id, and builds nothing;
the record's `names` then gains the recipe or series that asked, so
`wpt train` without a dataset reads the latest build of each of them.
The folder holds `dataset.json`, the dataset's record, written last, so a
build that fails leaves no dataset; its shards, each named by its token hash;
its vocabulary; its document list; the credits of its sources
(`credits/<source>.jsonl`); and its conversations and tasks
(`conversations.jsonl`, `tasks.jsonl`, each only when the sources hold
some). The record names each of these files, shards included, by one shape,
`{"key", "sha256"}`: the file's key in the store and the sha256 every read
checks. Each build stamps the record with its time, the same content found
again included, and the dataset `wpt train` reads when it is given no
dataset id is the latest build for the recipe or series (the record's
`name`), the session's own before the one it started from, found through the
store's index. A dataset found in the session it started from stays where it
lies; one found in the session itself has its record written again, with the
time and the downloads its sources have now.

Beside the dataset's record, the build keeps the dataset's document list in
the dataset's folder: one JSON
line per document, in the build's order, with its `id`, the `sha256` of its
Markdown (the one the dataset fingerprint reads) and its `tag`, `training`
or `held-out`, from the same split its shards are cut by. A new build
writes the list before the record, so a dataset is never there
without it. The task reader takes the held-out texts from it
([data-selection.md](data-selection.md), "Held-out text and
contamination").

A build works in three steps, and only the last reads text:

1. Index. The sources' records already say each downloaded file's tar,
   position, size and sha256, and each source's formats, licenses and tags.
   The index holds every file that can give a document, from those records
   alone. Files with the same sha256 are one document: the index keeps the
   first in source id order. What each source gives is kept as the result
   `cache/source-index/<fingerprint>`, so a build computes only the sources whose
   inputs changed. A source that fails keeps nothing: every build computes it
   again and lists it.
2. Choose. The recipe's mixture draws its documents from the index, by
   bytes, in a seeded order of its own for each pool. The unit of the draw is
   a frame: the files of a pool that start in the same 1 MiB frame of the
   same source's tar are one group, in tar order, and a file stored on its own
   is a group of its own. The seed of a group is fixed by the pool's name and
   the group's own name (the source's id and the frame number, or the
   file's sha256), never a position in the index or a session ID, so a source
   added later, or the same files in another session,
   moves no group before another. A build thus fetches about the bytes it
   chooses, where a draw of single files would touch nearly every frame of a
   pool of small files.
3. Read. Only the chosen files are read, source after source in tar
   order. A source's small files are fetched together, in requests of the
   frames next to each other, and each request's frames leave memory once
   its files are copied. Their Markdown is kept in packs, compressed files
   on this machine, so the vocabulary and the tokens read each file once.
   A worker fetches the next batch of files while it converts the current
   batch and turns it into tokens, and never goes further ahead by itself.
   A source's documents come out in file path order, whichever packs hold
   them.

The choice comes in rounds. The first guesses 4 bytes per token. A pool
that then holds fewer training tokens than its share draws again, at the
bytes per token it measured, until it reaches its share or the index has no
more of it, so every pool holds its share. The draw gives whole groups, so
a pool may pass its share by one group at most.

The Markdown never goes to the store. The packs stay in the store's cache
folder for the next build, under a folder named after what decides the
conversion ("Fingerprints and the store"); with `--stream` (or
`WPT_STREAM=1`) they go to a temporary folder deleted when the build ends.
The documents, the dataset id and the shards are the same either way. A
file its conversion needs as a file (a Parquet records file, a message
tree, a file its patches change) is copied from the store into a temporary
folder on this machine, deleted once the source is read; every other file
is read from the store by range, so a conversion never opens a file of the
store itself. The store keeps one token file per part of a source and
round (every document's tokens, one after the other), and the result
`cache/source-tokens/<fingerprint>` holds its final manifest and a summary,
the fingerprint covering the Markdown's fingerprint, the vocabulary
and the part's chosen paths; a part whose inputs did not change is never read
again, so a build stopped halfway through a source reads only the parts not
yet saved. The shard stage copies every token file its rows lie in to this
machine first, several at once (the copy counts its disk as `wpt_utils.parallel`
does), writes the training and held-out shards from the copies, and removes
the copies afterwards, also when the stage fails; the shards are the same as
if each row were read by range from the store.

`wpt dataset contamination` (`contamination.py`) reads a built dataset's
shards, one at a time, each copied to a temporary folder and removed once
read, also when the command fails. The training documents are searched in
worker processes, as many at once as `wpt_utils.parallel` allows. Its report
is the result `cache/dataset-contamination/<id>`
([data-selection.md](data-selection.md), "Held-out text and contamination").

`--plan` on `build` and `build-tokenizer` stops after the first choice and
prints what the build would read and compute: each pool's documents and
bytes, whether a vocabulary would be trained, and the files, bytes, sources
and tars to read.

The build reads sources and their conversations and tasks in worker
processes, several at once, as many as `wpt_utils.parallel` allows. Each
source is read on its own, and the results are put together in the order
of the source ids and of the documents, so the dataset id and the shards
are the same on any machine and with any number of workers. How many run at
once counts each part's chosen bytes on the disk its copies go to. The
conversations and tasks do not depend on the choice: every indexed source
that holds some is read.

A source of more than four batches of chosen files (1 GiB) is read in parts,
each four batches in tar order, by several workers at once. The size of a
part is a constant of the code, so the parts are the same on every machine.
The build takes a source once all its parts are read: its documents, its
check result and its failure are those of one read, and a part that fails
takes the whole source out. The parts after a failed part are not read; those
before it are, so the source's error is the first in part order, whichever
part failed first in time. A file is never split.

The index of a records entry under `newest` reads every file the entry covers,
so it is saved as a result of its own, `cache/newest-index/<fingerprint>`, the
fingerprint covering the entry, the
files it covers and its version, and any later read, build or round loads it. A
source with such an entry read in parts starts with one more item in the same
workers, which saves the index while other sources are read. Its other parts
wait for that item, and its failure is the source's failure.

On a bucket store, downloaders run ahead of the workers, so that the link
stays busy across the ends of parts and no worker waits for the store. These
are worker processes of their own, at most twelve, that take the coming parts
in the order the workers will. Each copies a part's chosen files, small and
large, and the license statements its opening reads, into a folder of that
part under a temporary folder of the store's cache folder. A file a pack
already holds is not copied, nor any file of a part whose tokens the store
holds. Every copied file is checked against its sha256 first.

What is downloaded and not yet used stays within 100 GiB, and within a quarter
of the disk's free bytes when the round starts. At that limit the downloaders
wait. A part that starts takes its folder, waiting for a download under way;
a part whose download never started fetches for itself, so a part never waits
on the limit. A read releases each file once used and the folder goes when the
part ends. Ending a round, a failure and a stop remove the temporary folder.
A process killed outright leaves it behind, and the next round on that machine
removes it.
A failed download changes only timing: the part reads what the folder lacks
for itself, and meets the error itself when the error is its own. On a folder
store the files are local already, and no downloader runs.

### The download step

Only `wpt dataset download`, `check --fetch` (before its other checks) and
`manifest` download sources, through one piece of code
(`wpt_dataset/download.py`). Every other command, the build included, reads
the upstream files from its session, and a source that is not there fails
with `NotDownloaded`, naming the command that fetches it. An `empty`
source has no upstream and needs no download.

A download works in the session it is given ([The store](#the-store)). Given
none, it makes a new short ID and prints it.

A source is downloaded when, and only when, neither the session nor the one
it started from (`--from`) holds it: the source has no record in either, its
record holds an error, or its record names another download. A record names a download by the source's id and
everything its download reads (its `type`, addresses, `revision`, `extract`,
`history.since`, `submodules` and the outside license pages its entries
declare), so a new source, an address that changed, a revision moved to
another commit and changed fetch options all find nothing held. A changed
license page downloads the whole source again, so the tree and its pages
come from the same moment. A rerun, or one new source, therefore costs only
what the session lacks.

Downloading a source again in the same session replaces its file. A redo is
a new session, which fetches every source again and leaves the old session
as it is. A git source gets the same commit again: only a new `revision`
brings a later one. With `--from OLD_ID`, a source the old session holds
with the same address, revision and fetch options is not fetched, as one
the session itself holds is not, and a build, `check --fetch` and `manifest`
read it from the old session's files; the download fetches the rest, and
what it writes stays in its own session and names that ID.

Nothing else triggers a download: no question to upstream about changes, no
server hint (length, date, ETag). A stale session is accepted.

Every source follows one path, git sources included. For each source to
download, every license page its entries declare is fetched first, through
the HTTP protocol, into a folder on the machine. Then each address is
fetched in turn (the main one, then each mirror) through the source's
protocol, its files going into the source's tar as they are read. The first
address that gives its files is kept: the license pages join its files in
the tar, under `.wpt-license-pages/`, and the tar and then the source's
record go into the session.
An upstream checksum, where a protocol has one (LFS gives each file's
sha256), only checks that the bytes arrived intact. A failed download keeps
any file the session held for that source, and the source's record gains the
`error`, with each address's failure, beside the facts of that file. A source
whose record holds an error is not held: the next download fetches it. A
build reads the good file such a record still names when it is the file of
the download the source's description asks for now, and the dataset record
notes that the source's last attempt failed (`downloads`, `failed`); a
source with no good file is left out and listed.

Each download works in a temporary folder of its own,
`wpt-download-<random>` in the system's temporary folder, beside a lock
file the process holds while the download runs; the download removes
both when it ends, whatever its outcome. A process stopped by a signal or
a crash leaves them, and every new download run on the machine removes each
such folder whose lock it can take, never one a live process holds. A
folder whose lock file or files it cannot open or remove, such as another
user's, stays, and the run goes on.

A git source fetches the commit its `revision` names, and only that
commit. A server that serves only branch and tag tips is asked for those,
and the commit must be among what they hold; an address that does not
serve the commit fails with `FetchFailed`, naming it.

A first fetch of the commit, without file contents, says which files the
source keeps. A second fetch then brings the commit, and its history when the
source has one, with every content, because git spends some twenty times
more per file on a request by id. That fetch also brings files the source
does not keep and versions older than `history.since`; they are deleted with
the download's temporary folder and never stored. A server that cannot send
every content fails that fetch, and the files the source keeps are then
asked for by id.

An archive of an `http` source is read while it downloads
(`wpt_dataset/archives.py`): each member goes straight into the source's
tar as it is read, with no copy of the archive on the disk, and its
sha256 comes from the bytes read, so its manifest is the one a copy on the
disk gives. The store keeps the members, never the archive. A zip, whose
table of contents is at its end, is read through range requests when its
server gives its size: its last bytes are asked for whether or not the
server says it serves byte ranges. A server that answers with the whole
file instead, or gives no size, has the zip saved to a file on the disk
first. A tar is read as it arrives. A hard link to a member the source
leaves out makes it read the archive a second time as it arrives, holding
in memory the members such links need, since their content went by
unread. One fetch puts each content in the source's tar once, however many
times it reads the archive. An answer that ends before its announced size
is a cut connection. A tar whose server gives its size, a validator (`ETag`,
else `Last-Modified`) and `Accept-Ranges: bytes` continues from the byte
reached through a range request tied to that version; a cut that came after
new bytes uses up no try, and any answer but those bytes (the file changed)
starts the try over. Any other archive is tried again like any other
download. When the download fails, the unfinished tar is removed.

Each run of these commands leaves its run record in the session
([observability.md](observability.md)), and the source's record says what
happened to the source: when, the address used, the size, what the download
got, for the record only (the commit, for example), and the addresses that
failed first, or the error. Nothing decides on what the download got. A
failure that is not the source's (the store's, a missing program, an
interrupted run) changes no record.

With `download --stream` (or `WPT_STREAM=1`), the downloading
machine keeps no copy of a tar once the store holds it; without
it, an S3 store keeps a copy in its local cache folder and reads it first.

`wpt store list` lists what the store's index says of every session, or,
given an ID, of that session and the one it started from, from the index
alone: how many things of each kind each holds, then each of its sources with
its time, its error or the address used, its size and what it got. With no
ID it is how a person finds one.

`wpt store rebuild-index` makes the index again from the files of every
session, without reading the stored index, so it also replaces a damaged one.
It needs no session.

Several sources download at once: as many as `wpt_utils.parallel` allows,
and at most, from one site (the host of the source's address, and the
hosts under it), the number
`packages/wpt-dataset/src/wpt_dataset/sites.txt` gives. That file holds
one line per site, from what each site publishes, and a small default for
every other site; a line is added or changed by a commit, like code. A
site may also have a limit per time, a number of requests in a number of
seconds: every HTTP request this process sends to the site waits while
it would pass that limit, LFS requests included. A site may have both
kinds. Requests sent by git or Mercurial themselves are not counted. The
results come in the order of the sources, whatever ran at once.

A dataset's record names, for each source, the session and time of the
download that kept the files it used (`downloads`), from the source's
record. It is not part of the dataset id, so the same files downloaded
again give the same dataset.

### The build

`wpt dataset build` turns the bytes of a session into a dataset
(`wpt_dataset/build.py`) and never contacts upstream. A source the session
does not hold, or whose download failed, is left out and listed. For each
other admitted source, several at once:

1. Its final tree: the upstream tree, then its erasure cuts, then its
   patches. A cut or a patch that does not apply takes the whole source
   out until a person acts.
2. What it keeps (`wpt_dataset/kept.py`). The build asks the shared checks
   and the license check, and leaves out alone, each reported with why:
   - a file no license entry covers;
   - a file that is not text and is not listed;
   - every file of an entry that does not let its files train: the entry
     is not admitted, or the statement at its place, in the stored bytes
     (a file of the final tree, or a license page stored with the
     source), matches no text the entry expects, or matches one that is
     not admitted;
   - each record whose field value no entry expects, or whose entry does
     not train.
3. The Markdown of every kept file and record, then tokens.

A pointer task whose passage is not in the stored bytes is left out alone
and reported the same way ([training.md](training.md), "Pointer tasks").
The dataset record keeps each document's id and sha256, and each source's
check result: which statement was found at which place, which license
text matched it and that text's decision, and what was left out and why.

Every build runs the license check of every source again; what the
build keeps of a source is kept in the store under the check's answer and
its other inputs ("Fingerprints and the store"). What every build reads
again is fast by design: the download stores an
archive's members as files of the source's tar, so the build never unpacks an archive;
sources are processed
in parallel, sized by `wpt_utils.parallel`; and a file's format, tags and
license entry are each found with one matcher compiled over all of a
source's patterns, never by trying each entry in turn
(`wpt_dataset.manifest.PatternIndex`).

### Shared checks

Each check is defined once and used by every command that applies it (the
build, the task builder, `check --fetch`, the writes of the service
layer), so they cannot disagree. None changes anything.

- **Text check** (`checks.py`): whether a file is text ("Encodings",
  above).
- **License match** (`license_check.py`): a statement is looked up among
  the license texts, by the sha256 of its bytes, then by its normalized
  text, and matches when it is a text its entry expects; it trains only
  when that text is admitted ([licenses.md](licenses.md), "License
  entries").
- **Pointer-task check** (`checks.py`): whether a task's passage, in the
  Markdown of its document, still holds its sha256.
- **Cut match** (`erasure.py`): every run of a cut's length in its file
  whose keyed hashes are the cut's ([erasure.md](erasure.md)).

### The store

The store is set by one setting, `WPT_STORE`, in the environment or in
`.env` at the repository root:

```
WPT_STORE=file:///home/me/wpt-store      a local folder
WPT_STORE=s3://bucket/prefix             an S3-compatible bucket
```

For S3, the credentials and a non-AWS endpoint come from the store's own
settings, in the environment or in `.env`: `WPT_STORE_ACCESS_KEY_ID` and
`WPT_STORE_SECRET_ACCESS_KEY` (both needed), `WPT_STORE_SESSION_TOKEN`,
`WPT_STORE_ENDPOINT_URL` and `WPT_STORE_REGION` (default `us-east-1`). The
store reads no `AWS_*` setting (not even `AWS_ENDPOINT_URL` or `AWS_PROFILE`)
and no AWS configuration or credentials file, so the keys a cloud provider's
other tools use never reach it.

The setting's prefix picks the store's kind, a class of `wpt-store`
(`wpt_store/kinds.py`). A kind supplies five operations on files: write a
whole file, read a byte range, replace a file only if it is unchanged
since it was read, list, delete. Everything else (the layout, compression,
the index) is built on them, alike for every kind.

A store holds real named things, the same on a folder and on a bucket:

```
index                              each session ID and what it holds
<id>/sources/<source>.tar.zst      a source's files
<id>/sources/<source>.json         when, address, size, or the error, and what reading needs
<id>/datasets/<name>/              dataset.json, the shards and every other file the dataset needs
<id>/models/<name>/                the model's real files
<id>/runs/<run>/                   each command's record
cache/<stage>/<fingerprint>        only what can be rebuilt, shared by every ID
cut-key                            the secret erasure cuts are matched under
```

Hashes do two jobs. Every read checks the bytes against their sha256: a
record names each file it points at by the one shape `{"key", "sha256"}` (the
file's key in the store and its sha256), a record holds the
sha256 of its own fields, and a result holds its own in its first line.
Finished work is found from its inputs: a result lies under its fingerprint
("Fingerprints and the store"). A file known only by its content, such as a
token file or the output of a patch, lies in `cache/files/<sha256>`.

`cache/` holds only what can be rebuilt, so emptying it leaves every dataset
and every model usable: a dataset's files and a model's files, a run's
checkpoint included, lie in their own folders, and a record never points
into `cache/` for a file it needs to be read.

#### Sessions

A session is only an ID, one for the whole store: nothing opens it, nothing
closes it, and it has no status. Files are stored under it, and a record says
what happened. Every command that touches the store takes it, through
`--session ID` or `WPT_SESSION`. A download with no ID makes a new short ID
and prints it. `wpt store list`, `wpt store runs`, `wpt store run` and
`wpt doctor` work with no ID: with none they show every session of the
store's index, which is how a person finds an ID, and with one they narrow to
it (`doctor` runs every check either way, and given an ID counts that
session's sources in its disk line). `wpt store rebuild-index` needs no ID,
since it covers the whole store. Every other command that needs a session
with no ID is refused with `SessionNeeded`. `wpt web` takes the same options
and hands them to its routes and jobs.

`--from OLD_ID` (or `WPT_FROM`) makes a command read from that session what
its own does not hold, and write under its own: what it writes records the
old ID in `from`. No rule protects an ID another one started from.

A build reads the session it is given, and the one that session started
from. A source that is missing, or failed with no good file of its download
held, is left out and listed.

#### A source's files

`<source>.tar.zst` is a normal tar, which the `tar` program opens by hand: a
regular file's bytes follow its header, a symlink is a header naming its
target, and a content the source holds under several paths is written once,
the other paths being hard links. Its headers hold no time and no owner, so
they are the same on any machine. The tar is compressed in zstd frames of
about 1 MiB, so one file is read alone by range.

`<source>.json` is the source's record. It holds the fingerprint of the
download, the time, the address, the sizes, the tar's sha256, the tree's
manifest and what the download got, or the `error`; after a failed attempt
at a download whose good file the session holds, both: the facts of the good
file and the `error` of the last attempt. It also lists each file
with its path, mode, sha256, and position and length in the tar's
uncompressed bytes, so a reader needs nothing else to take one file.

A file several sources hold is stored again for each.

A source's tar is filled in a file of the store's cache folder named
`.partial-source-<random>`, beside a lock file its process holds while the
tar fills; the system releases the lock when the process ends, crash
included. A download run that starts removes every such file whose lock it
can take. The tar goes into the store first, then its record: until the
record is written, the session reads what it held. A local store writes the
tar to the disk for good (fsync) before the record names it. Should a crash
fall between the two, the record still names another download or an error:
the next download fetches the source again, and a read until then refuses the
bytes against their sha256 (`CorruptFile`).

#### Compression

Every file is compressed with zstd while it is written, in frames of
about 1 MB compressed apart (zstd's seekable format, which plain zstd
reads, its seek table a skippable frame that standard tools pass over), so
a byte range is read without the rest. It is kept compressed, under its key
plus `.zst`, only when that is smaller, and as it is otherwise. A source's tar
is always compressed, at level 19: it is written once and read many times,
and in 1 MiB frames of text level 19 is about 17% smaller than level 3, reads
as fast, and the levels above it are no smaller. Every other file takes
level 3. An S3 answer cut while
it is read is asked again, up to 5 tries in all. Large files are
compressed several frames at once, as `wpt_utils.parallel` allows. They
go to S3 in parts of 64 MB, and every upload of a process sends its parts
through one shared limit of 32 at once: S3-compatible stores such as R2
give each connection a few MiB/s, so a link fills only with many
connections at once, and uploads that run together do not multiply them.
That limit is set for network work, not by the cores; a part starts only
while the memory rule of `wpt_utils.parallel` allows.

Reads from S3 keep the same many connections busy. A compressed file's
frames are fetched in blocks of 8 frames, one request each, and a read
requests the blocks it needs together, at most 8 requests in flight per
process. When a reader goes through a file in order (a large file, or a
tar's files one after another), the blocks after the one it works on
are requested ahead, so its next data arrives while it processes the
current data. The compressed bytes requested ahead and not yet taken stay
within 64 MB per process. A request ahead that fails raises at the read
that takes its block. What a read returns, and so what a build writes,
does not depend on the timing. A local folder reads in place, one request
at a time.

#### The index

`index`, at the store's root, is one compressed SQLite file for fast lookup:
each session ID and what it holds, a row per source, dataset, model and run,
with its time and what its record says of it. It says nothing the files do
not: every row comes from a file of a session, and `wpt_store.index.rebuild`
makes the index again from the files alone.

A source's row reaches the index when its run ends, a dataset's and a
model's when they are kept. A run that crashed leaves its rows out; the next
download into its ID puts them in for each source it finds held, and a
rebuild puts in every one.

Every read or change of the index goes through a copy on the
disk of the machine that runs it, in the store's cache folder, never
through one in memory, so only the disk bounds its size. A change is made
in that copy, and the compressed copy replaces the stored file whole, only
if it is unchanged since it was read: a conditional upload on S3, a lock
in a local folder. A copy larger than one part goes up in parts, and the
request that completes them carries the condition. When another command
replaced it meanwhile, it is read again, the change is made again, and the
replace is tried again. Each local copy lies beside a lock file its process
holds; one whose lock can be taken belongs to no live process and is removed.

#### What else it keeps

The store also keeps the cut key that erasure cuts are matched under
([erasure.md](erasure.md)) and the locked vocabularies
([tokenizer.md](tokenizer.md)): a machine without access to the store
matches no cut and uses no locked vocabulary. A locked vocabulary is a file
of each dataset built with it; a build finds it again there when
`cache/files/` does not hold it.

A named run's checkpoint lies beside its model: `<id>/models/<name>/checkpoint.json`
is a record that names the state's file, which lies in the same folder
([training.md](training.md), "Checkpoints and resuming").

The store also holds the record of every run of `wpt dataset download`,
`check --fetch`, `manifest`, `build`, `build-tokenizer` and `contamination`, `wpt train`,
`wpt eval` and `wpt manager`: one folder per run under
`<id>/runs/`, written by name, each file compressed
(a local store adds a record file's lines to one plain file while the
record runs, and compresses it into one part when the record ends;
[observability.md](observability.md)).

## The service layer

`packages/wpt-dataset/src/wpt_dataset/repo.py` is the only code that reads
or writes the repository's data: the
sources and their patches, the chat, task, rating and probe files inside them, the
model, tokenizer and mixture recipes, the image Dockerfiles and their
pools of tools (`tools.json`), the two
signer lists (`allowed-signers`, `allowed-gpg-keys.asc`), and the
license texts (`license-texts/<id>/`, [licenses.md](licenses.md),
"License texts"). Every command
that uses this data goes through it: `wpt dataset` (`check`, `migrate`,
`build`, `check-commits` and the rest), `wpt train`, the
image commands of `wpt agent`, and `wpt web`. It hands out documents, bytes
and versions, never a path under `data/`.

- Every file has a version: the sha256 of its bytes, or `""` for a file
  that does not exist yet. A source's files share one version, over its
  `source.json` and every patch. A save names the version it started from
  and is refused when the file changed since (a stale save).
- Every write applies the rules of `wpt dataset check` to the new content
  first, and replaces the file in one step (a temporary file, then a
  rename), so a refused or interrupted save leaves the old file. Text files
  are written with `\n` line endings.
- One lock per repository holds each version check together with the write
  it allows, across processes as well as threads: two saves of one version,
  from two tabs of `wpt web`, a double click, or `wpt web` and
  `wpt dataset` at the same moment, cannot both pass; the second is a stale
  save. Between processes it is a lock the operating system holds on the
  file `wpt-data.lock` in the repository's git folder (`.git`, or the
  folder a worktree's `.git` file names; `.wpt-data.lock` at the root of a
  folder that is not a git working tree), with `fcntl.flock` on Linux and
  macOS and `msvcrt.locking` on Windows. The system releases it when the
  process ends, a crash included, so a lock is never left behind to block
  the next save. Inside a process, the threads share one lock per
  repository (`wpt web` serves each request on its own thread).
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
  person writes the text later (`wpt dataset save source`, or the "Sources"
  page). Every reader of the sources (`wpt dataset build`, with its
  readers of chat and task files, `check --fetch`, `check --changed`) leaves such a source out with a note naming it, instead of
  failing, and `wpt dataset check` names it too, as a note that is not a
  failure. A source that is admitted needs its text, and a source with an
  empty text takes no files until it is written.
- A save of a source that changes a license entry's license, place or
  expected texts, without deciding that entry anew, sends the entry back
  to `under-review`: its decision then reads `by` `machine:wpt dataset`,
  with the date and the reason. The source's decision and its other
  entries stay. A commit holding such a change needs an allowed signature,
  as every decision change does.
- JSON is written with two spaces of indent, characters outside ASCII kept
  as they are, except U+2028, U+2029 and U+0085, always written as escapes,
  even in values a save keeps unchanged, so that no line splitter cuts a
  line. A save over an existing JSON file keeps its bytes when its meaning
  is the same (key order and those escapes aside), and otherwise rewrites
  only the values that changed, the rest keeping its layout and key order;
  a new key goes after the key the saved document puts before it, and a new
  key or item is written with the indent and spacing of the last member of
  its object or array.
- Conversations, tasks, ratings and probes are written only in an `empty`
  source, text written for WPT ([chat-format.md](chat-format.md), "Writing
  a chat file"; [training.md](training.md), "Tasks";
  [evaluation.md](evaluation.md), "People's ratings" and "Probes").
- A source's text changes only through a patch: the layer unpacks the
  source into a temporary folder, puts the change in, and records the
  difference as the next patch, kept only when the source then gives
  exactly those files and passes the checks of `wpt dataset check --fetch`
  ([Patches](#patches)). A new chat file or task folder gets its own format
  entry in `source.json` when none covers it.
- Generated files (a tokenizer's `.lock.json`, an image's `lock.json` and
  `licenses.json`) are readable everywhere and written only by the command
  that makes them (`wpt dataset build-tokenizer`, `wpt agent lock-image`),
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
package or its command script) is neither `repo.py` nor a test fails with
`DataAccessRefused` before it happens, in every command and in the test
suite. Inside `shutil.rmtree`, a relative name is a child of the folder being
removed, not of the current folder, so only the folder `rmtree` was given is
checked. A test is a
`conftest.py` or a file under a `tests` folder, both inside the project's
checkout: a folder named `tests` above the checkout, or a `conftest.py`
outside it, does not make its code a test. The tests are left out of both:
they build the repositories they check and read what a command wrote.

### Two interfaces

Every kind of data has two interfaces, both thin layers over the service
layer: the HTTP API of `wpt web`,
described in OpenAPI 3.1 at `/api/openapi.json` (the server checks every
request body against the same description), and `wpt dataset` commands.

| Kind | `wpt dataset` | `wpt web` |
| --- | --- | --- |
| any kind of file: list, read, write | `kinds`, `list KIND`, `show KIND NAME`, `save KIND NAME FILE --version V` | `/api/kinds`, `/api/documents/{kind}[/{name}]`, `/api/schemas/{kind}`; pages "Recipes", "Images", "Signers" |
| a new empty source | `new-source ORIGIN [ITEM] NAME ...` (without `--license-text`, a person writes the license text later; an empty `--license-text` file is an error; `--format FORMAT=PATTERN` declares a format entry with no file) | `POST /api/sources`; page "New source" |
| a source's description | `show source ID`, `save source ID FILE --version V` | `/api/sources/{id}`; page "Sources" |
| a source's files | `files ID`, `read ID PATH`, `write ID PATH FILE`, `remove ID PATH`, `tags ID PATH [TAG ...]` | `/api/sources/{id}/files[/{path}]`, `/api/sources/{id}/tags/{path}` |
| conversations | `conversations ID PATH`, `conversation ID PATH LINE`, `save-conversation ID PATH LINE` (or `new`) | `/api/sources/{id}/conversations/{path}[/{line}]`, `POST /api/drafts`; page "Conversations" |
| tasks | `tasks ID`, `task ID FOLDER`, `save-task ID FOLDER` | `/api/sources/{id}/tasks[/{folder}]`; page "Tasks" |
| people's ratings of attempts ([evaluation.md](evaluation.md), "People's ratings") | `ratings ID PATH`, `save-rating ID PATH LINE` (or `new`) | `/api/sources/{id}/ratings/{path}[/{line}]`, and the attempts of `wpt eval` runs from the store, `/api/runs[/{folder}/attempts]`; page "Ratings" |
| probes of the `probes` test ([evaluation.md](evaluation.md), "Probes") | `probes ID PATH`, `save-probe ID PATH LINE` (or `new`) | `/api/sources/{id}/probes/{path}[/{line}]`; page "Probes" |
| git's status of `data/` | `status` | `/api/status`, on every page |
| every other job of the command line: a `wpt dataset` command that reads or writes no person's text, `store`, `agent`, `train`, `eval`, `scaling`, `manager`, and `chat --about` | `wpt SUBCOMMAND ARGS...` | `POST /api/jobs/{job}` (for example `/api/jobs/dataset/download`, `/api/jobs/train`), then `/api/jobs[/{id}[/lines/{first}\|/records]]`; page "Jobs" |

The HTTP API answers 404 when the thing does not exist, 409 for a stale
save, 422 for content the rules refuse (with their messages) and 503 when
a tool or the store is missing on this machine. Both interfaces write files
and show git's status; commits stay in git.

A job route starts the subcommand as a child process of the server,
`python -m wpt SUBCOMMAND ARGS...`, in the repository's folder, with the
arguments the request gives and no standard input, and the server tells it
the job's id (the `WPT_JOB` setting), which the subcommand's run record
names (`job` in `start.json`, [observability.md](observability.md)). Any
number of jobs run at once. Following a job gives the lines it printed to
standard output and standard error (the last 20000) and its run record,
where it writes one; a job lives as long as its process, and a stopped
server forgets its jobs (it interrupts those still running first) while
their run records stay in the store. No route runs a `wpt dataset`
command that reads or writes a person's text (`read`, `write`,
`conversation`, `save-conversation`, `task`, `save-task`, `save-rating`,
`save-probe`), `new-source`'s options that do (`--file`, `--license-text`),
`wpt chat` but for `--about` (it reads what a person types), or `wpt web`.

`wpt web` listens only on 127.0.0.1 and checks neither the `Host` header
nor a request's `Content-Type`: like
the commands, it asks for no login and no header. What that leaves open
while it runs: a web page open in the same browser can send it a
cross-origin `POST` with `Content-Type: text/plain`, which the browser
sends without asking first, so such a page can start sources
(`POST /api/sources`), ask for drafts (`POST /api/drafts`) or start any
job (`POST /api/jobs/...`: a download, a build, a training run); and a site
that makes its own name resolve to 127.0.0.1 (DNS rebinding) can reach
every route, reading people's texts, changing data files in the working
tree and starting jobs. Nothing is committed by `wpt web`, so every change shows in
`git status` and in the page's list of changed files before a commit.

## Packages

| Package | Role |
| --- | --- |
| `wpt-dataset` | all data rules: sources, manifests, cuts, patches, pools, builds, the checks CI runs, migrations; the service layer (`repo.py`), the only code that reads or writes `data/`; a source as read (`source_data.py`); the protocols (`protocols/`), the download step (`download.py`), the license check (`license_check.py`), the shared checks (`checks.py`, `erasure.py`), the build (`build.py`, `kept.py`); the contamination check (`contamination.py`); the task builder (`builder.py`, run by `scripts/build_tasks.py`) |
| `wpt-store` | the store (its kinds: a local folder or S3), its compression, its layout of sessions with their sources' tars and records, datasets, models and run folders, its results and index, and the token shard format with its checked reader, so training and evaluation read data without depending on the data rules; what the index says (`listing.py`), and a kept model folder copied out, checked against its sha256s (`models.py`) |
| `wpt-telemetry` | the record of one command run in the store, sent live to an OpenTelemetry backend when one is set ([observability.md](observability.md)) |
| `wpt-tokenizer` | the byte-level BPE tokenizer, Rust inside |
| `wpt-model` | the WPT network ([model.md](model.md)) and its drafter, its safetensors model folders, and their license documents (`LICENSE`, `NOTICE`, model card) |
| `wpt-training` | the training loop (global batch, evaluation, checkpoints, resuming) that runs one training step, with its plug points: strategies (how the hardware is used), and the parts of a step ([training.md](training.md), "Steps"): methods (how the model learns from an example), environments (where the examples come from), optimizers (how an update changes the weights) and the steps that name one of each, with the model parts they lock |
| `wpt-inference` | sampling through the model's generation cache: several rows from one prompt, the model's next chat message (`next_message`, which `wpt chat --chat` and the text measures of `wpt eval` both write with), speculative decoding with a drafter and its scheduler, the scheduler's pass times (measured, recorded, replayed); training a model's drafter (`drafting.py`), which `wpt train` runs at the end of every run, so it uses `wpt-model` |
| `wpt-utils` | helpers several packages share, one submodule each: `wpt_utils.device` picks the best GPU, else the CPU; `wpt_utils.programs` starts the outside programs WPT runs (`program(name)`, with a clear error when one is missing), identifies the code (`code()`: the checkout's commit and a hash of its uncommitted changes, the one place that reads the commit, which records and a training step's fingerprint hold; `code_name()`: the same as one line, which a stored result of the data pipeline keeps as what made it) and gives that commit, and whether there are changes, in the shape the records name them (`repository()`), without importing PyTorch; `wpt_utils.files.sha256_file` is the one way a file is hashed, by the store, the model folders and the run records alike; `wpt_utils.files.lock_file` is the lock the operating system holds on a file for one process, which the service layer and a local store's replace of a file share; `wpt_utils.settings.typed` types the settings of a training step and of its method, environment and optimizer from the defaults each file declares; `wpt_utils.parallel` is the one rule that sizes parallel work (at most 75% of the cores; before each new item it measures again and starts it only while a quarter of the memory, of the cores and, where the items write to a disk, of the free disk stays free, one item's memory being what running items are measured taking, never a given figure; it never stops a running item) and runs items that way with their results in the items' order, an error an item may fail with becoming that item's result; `wpt_utils.lines` writes every printed line, with its time and level |
| `wpt-agent` | the agent harness of agentic training: the tools (the four built-in ones, `ask`, and those a task or its image's pool declares, answered by programs in the image) and each attempt's choice of them, one Docker container per attempt, the images of `data/images/` and their locks, the agent loop and the task's check, and the playing of a task's attempts (`wpt_agent.play`), which the `agent-tasks` environment of the `finish-tasks` and `finish-tasks-with-less-writing` steps and the agent measures of `wpt eval` share, so `wpt eval` does not depend on `wpt train`; an image's lock and build (`images.py`) |
| `wpt-worker` | the gpu worker, which runs the GPU work of a training on its machine, the link to it from the wpt manager's side, and the gradients exchanged between gpu workers in different places ([machines.md](machines.md)) |
| `wpt` (`apps/wpt`) | the app: the `wpt` command and its web server, two controllers over the services above ([The app](#the-app)) |

The libraries hold services only, and no command. Those that run without
the repository, `wpt-worker` (on a gpu worker's machine), `wpt-model` and
`wpt-inference` (a model folder and the text it writes), install alone.

## The app

One project, `wpt` (`apps/wpt`; the workspace's root project is
`wpt-workspace`), holds the one command, `wpt`, and its two controllers
over the same services:

- the subcommands: `wpt dataset`, `wpt store`, `wpt agent`, `wpt train`,
  `wpt eval`, `wpt scaling`, `wpt chat`, `wpt manager`, `wpt doctor` and `wpt web`, each
  with its own options (`wpt SUBCOMMAND --help`). The module `wpt`
  (`apps/wpt/src/wpt/__init__.py`) names the module whose `main` runs each
  one and imports it only when it runs, so a light subcommand never loads
  PyTorch; `python -m wpt` is the same command. The controllers of the
  data rules, the store and the agent harness are `wpt/dataset.py`,
  `wpt/store.py` and `wpt/agent.py`; the jobs are the subpackages
  `wpt.train`, `wpt.eval` (with `wpt.eval.scaling`), `wpt.chat` and
  `wpt.manager`.
- the web server, `wpt web` (`wpt/web/`: `server.py`, one file per page
  in `pages/`, `ui.py`, which builds the interface of `apps/wpt/ui` with
  Bun, and the services only the server uses: `drafts.py`, a model
  folder's draft of the next assistant message, `attempts.py`, the
  attempts of the store's `wpt eval` runs for the rating page, and
  `jobs.py`, the subcommands it runs as child processes). It listens on
  127.0.0.1 only; another machine reaches it through an SSH tunnel or
  Tailscale. Its job routes start and follow the subcommands ([Two
  interfaces](#two-interfaces)).

The wpt manager runs each of its steps as a subcommand of the same
command (`python -m wpt dataset download`, and so on). The app's tests
live in `apps/wpt/tests/`, one folder per controller or job, and share
their setup through `apps/wpt/tests/conftest.py`.

## Checks

`uv run scripts/check.py` runs every check, the same way on any machine
and in CI (`.github/workflows/ci.yml`, on every pull request and push), so
a green run before pushing means a green CI. It compares with the upstream
branch, as a push would.
`uv run scripts/check.py --working-tree` checks the files as they are, so
work needs no commit to be checked. It runs every check below but the
commits, and the image build tests when the files as they are, new ones git
does not track yet included, differ in image code from the upstream branch.
It says that it left the commits out, and a push still needs a run without it.

The checks:

- `uv.lock` matches every `pyproject.toml`.
- The tests of every package, and the Rust tests. None needs a server: the
  S3 test runs against moto, which answers S3 requests inside the test
  process. The tests run in parallel (pytest-xdist), each process limited
  to one PyTorch thread and kept off the GPU, as in CI, since a GPU that
  shares the machine's memory gives every process its own share of it; a
  plain `uv run pytest` still uses the GPU. A test passes in both. One whose
  claim a GPU cannot hold keeps its programs off the GPU in both (the
  `on_the_cpu` fixture of the root `conftest.py`): a comparison of two
  trainings bit for bit, since two runs on one GPU differ
  ([training.md](training.md), "What is exactly repeatable"), and several
  processes of a training on one machine, since NCCL refuses two processes on
  one GPU. A limit in seconds on what a test waits for stands far above what
  that takes, so other work on the machine does not reach it. The number of
  processes is
  fixed before they start, so it comes from `wpt_utils.parallel.limit`,
  the rule's form for a count fixed in advance. The check first runs a
  sample of the tests (every 40th, in collection order) alone, and
  measures the memory that run held at most (`wpt_utils.parallel.peak`: the
  program and every program it starts); one test process is taken to hold
  that much, and the count is the smaller of 75% of the cores and 75% of the
  memory available divided by it, at least 1. Where the memory of the sample
  cannot be measured, or the available memory cannot be read, the cores
  alone decide. The check prints the number, the sample's result and both
  inputs.
- The image build tests (marked `image_build`, in
  `apps/wpt/tests/agent/test_images.py`, and the test of the agent
  image's tools and check library in
  `packages/wpt-agent/tests/test_agent_image.py`) lock and build
  real images, which takes minutes. They run only when the push changes
  image code: `packages/wpt-agent/src/wpt_agent/images.py`,
  `apps/wpt/tests/agent/test_images.py`,
  `packages/wpt-agent/tests/test_agent_image.py`, the root `conftest.py`
  (the `image` fixture),
  `packages/wpt-dataset/src/wpt_dataset/repo.py` (its `Repo.image_files`
  decides which files an image's tag covers), the schemas
  `image-lock.schema.json` and `image-licenses.schema.json` in
  `packages/wpt-dataset/src/wpt_dataset/schemas/`, or anything under
  `data/images/`. Otherwise the check leaves it out and says so; a plain
  `uv run pytest` runs it with every other test.
- The two task step tests that run attempts in a real image
  (`apps/wpt/tests/train/test_steps.py`) keep its lock in
  `.pytest_cache/` and the built image in Docker, for as long as the
  image's files and `wpt_agent/images.py` stay the same. Only the first
  check on a machine, and every CI run (which starts empty), pays for
  locking and building it; when those files change, the next check locks
  and builds again and removes the old image.
- `wpt dataset check`: every file matches its schema and current shape,
  licenses are on the allowed list (with their text), coverage patterns
  have an allowed shape, each license entry expects license texts that
  exist and takes the entry itself as its place only where that is
  allowed, no machine admits or refuses at any level, each license text's
  id is that of its text, patches are numbered without gaps, mixture
  weights add up to 1 and pool expressions parse, recipes name existing
  recipes.
- `wpt dataset check --fetch
  standardebooks-joseph-conrad-ford-madox-ford-the-nature-of-a-crime-5366b0f3`:
  one short book is downloaded (or found in the store), and its cuts,
  patches, license map and text are checked, which proves the fetch and
  its checks work. The sources a pull
  request adds or changes are not fetched here: the maintainer runs
  `wpt dataset check --fetch` on them before merging.
- `wpt dataset check-commits <range>`: DCO sign-off on every commit, and a
  signature from an allowed key on every commit that changes a decision, at
  any of its three levels (a source's, a license entry's, a license
  text's), or what an admitted or refused entry or text holds, compared by
  content rather than by position.

It runs every check even after one failed, then lists which passed. There
is no local hook: run it before pushing.

## Migrations

No data file carries a shape number. A migration step recognizes an old
shape by its fields: it says whether a document has the old shape and
rewrites it. When a shape changes, the change comes with a step in
`packages/wpt-dataset/src/wpt_dataset/migrate.py`, in the list of the kind
of file it changes, and `wpt dataset migrate` rewrites every file a step
applies to, then checks every file against its schema. A step needs a field
difference to find, so a change to a shape leaves one (a field that an old
file lacks, or has and a current one does not). Until `migrate` ran, a
file with an old shape is refused: the check and every read name the old
shape and say to run it. Nobody edits the files by hand for it, except
where a new field holds something only a person can write, which the step
leaves empty and the check then names each file to fill in. The steps today:

- `source.json`: a license entry without `attribution` gets it empty; a
  source without `formats` gets an empty list; a license entry without
  `text` gets the text of a `LicenseRef-<name>.txt` file next to it (the
  file is then removed) or an empty one; a refusal whose reason is "under
  review" becomes `under-review` without that reason, and `retired` becomes
  `refused`, keeping its reason; the tags `sft`, `grpo` and `token-efficiency`
  become `answer-as-assistant`, `finish-tasks` and
  `finish-tasks-with-less-writing` in every tag list; a git source without
  `revision` gets the commit it was reviewed at: the last `revision` its
  `source.json` held in the repository's git history, else the commit its
  last download in the store got, else an empty one for a person to
  fill in. Only the service layer reads that history.
- model recipes: the architecture of an earlier model (`n_embd`, gated MLA)
  becomes the current one, its numbers a person fills in; the window length
  in the architecture and the `training` steps and batch size become one
  stage; a recipe without a `drafter` section gets one with the values every
  recipe states; `training` and `posttrain` become one list of `steps`.
- task files: a task with `replies` and no `answered_in` gets it empty. A
  repository pointer's `checksum` becomes `manifest`, its `sha256:`
  prefix dropped.
- tokenizer recipes: a recipe without `version` gets 1. A recipe's
  `mixture` gives way to `sample_bytes`, 10 GB, in the recipe and in the
  copy of it its lock keeps, so the lock stays valid. A lock without the
  list of its documents cannot be converted: delete it and build again.
- every kind: a `schema` field is dropped.

A file `migrate` cannot rewrite, such as a source's `chat` file (upstream's
bytes), goes through `migrate.upgrade` where it is read.
