---
name: wpt-sources
description: How to gather new training data for WPT, from finding candidate works to the maintainer's admission. Use it when asked to find, add or review sources (books, collections, repositories, datasets), to check whether a work is in the public domain, or to admit sources. It covers public-domain proof, when to ask the maintainer, writing the source files, the checks to run, and the explicit yes that admission needs.
---

# Gathering sources

Read the `wpt` skill first: it holds the repository's rules (allowed
licenses, robots.txt, formats, no personal data). This skill is the order
of work and the points where the maintainer decides.

## 1. Candidates

The maintainer names what to look for (an author, a collection, a kind of
text). Prefer editions that publish exact, versioned files with their own
statement of rights: a git repository pinned to a commit beats a web page.
Note for each candidate the work, its author, its translator or editor,
and the edition.

## 2. Prove the license, and ask about every doubt

For public domain (docs/licenses.md, "Public domain"), the work must be in
the public domain both in the United States and in its country of origin,
for every person whose words are in the text: author, translator, reviser,
annotator, editor of notes. Find for each one:

- the death year (life plus 70 years is the usual rule in the country of
  origin; check that country's rule);
- the first publication year and country of the text used (in the United
  States, works published before 1931 are in the public domain in 2026; the
  limit moves one year every 1 January);
- the edition's own statement for what it added (markup, corrections).

Cite where each fact comes from. When a fact cannot be found, or the
evidence disagrees (an unknown death year, a revision by a later person, an
unconfirmed country), **do not settle it yourself in either direction**:
ask the maintainer about that candidate right away, with the doubt, the
evidence found and what would remove it. Without their answer the
candidate stays out ("when in doubt, refuse"). If they admit it despite the
doubt, the admission records why in `reason` (step 6).

Candidates without doubt need no question yet; they wait for step 6.

## 3. Write the source files

For each candidate, following CONTRIBUTING.md, "Adding a source":

- **Id**: `uv run wpt-dataset source-id ORIGIN [ITEM] ADDRESS`.
- **Address and revision**: the exact commit for git and Mercurial; for an
  archive, the fetch date, with the checksum pinning the bytes. On GitHub,
  write the repository address without `.git`: GitHub's robots.txt forbids
  `/*.git` addresses to AI training crawlers, and the plain address serves
  the same repository.
- **extract.exclude**: everything that is not the work's text: images,
  styles, packaging and metadata files, and the edition's boilerplate pages
  (colophon, imprint, title pages, rights page). The work's own front and
  back matter (preface, notes, introduction) stays.
- **licenses**: `LicenseRef-public-domain` (or the license) on its paths,
  with `attribution` built from the edition's own metadata (author,
  translator, title, edition, producer, link), read by a script, never
  written from memory.
- **`LicenseRef-public-domain.txt`** next to `source.json`: the people and
  their death years, the first publication, why it holds in the United
  States and in the country of origin, the edition's own dedication, and
  the sources of these facts.
- **formats** and **tags**: drafted from the source's own files and shown
  to the maintainer as one pattern for the batch (one full `source.json`
  as the example), which they approve before the files are written.
- **decision**: `refused`, by the maintainer's name, today's date, reason
  "under review".

Then `uv run wpt-dataset manifest ID` gives the checksum to fill in.

## 4. Check every source

```sh
uv run wpt-dataset check --fetch ID [ID ...]   # checksum, cuts, patches, license and format maps
uv run wpt-dataset markdown-check ID [ID ...]  # notes complete, no leftover tags
```

Look at the start of a few documents' Markdown, at least one per kind of
text (verse, dialogue, prose with notes, plain text): headings, paragraphs,
line breaks, notes at the end, no stray escapes. Report what each source
gives (entries, documents) to the maintainer.

A robots.txt refusal on one address moves to the next mirror; a source
whose every address fails stops. Never download it another way.

## 5. Commit the sources as refused

Commit the new sources with their `refused` decisions (the `wpt` skill's
rules apply: `scripts/check.py`, sign-off, no personal paths).

## 6. Admission needs the maintainer's explicit yes

Never set `"decision": "admitted"` without it. Ask once for the batch: the
question lists every source with its one-line public-domain reason, and the
sources the maintainer already discussed in step 2 with what was decided.
An answer that approves only some admits only those.

Then set the approved decisions to `admitted`, by the maintainer, with
today's date, and no `reason` unless the admission is an exception (such
as a doubt the maintainer settled, or an explicit permission for content
without a license). The commit that changes decisions must be signed by an
allowed key, so it is the maintainer's commit: show them its full message
first.

## Known source kinds

- **Standard Ebooks** (`github.com/standardebooks/<repo>`): public-domain
  books with CC0 markup. Revision: the current commit of `master`. Exclude
  `images/**`, `LICENSE.md`, `production-notes.md`, `se-lint-ignore.xml`,
  `src/META-INF/**`, `src/mimetype`, `src/epub/content.opf`,
  `src/epub/toc.xhtml`, `src/epub/css/**`, `src/epub/images/**`, and the
  `colophon`, `imprint`, `uncopyright`, `titlepage` and `halftitlepage`
  files of `src/epub/text/`. Format `xhtml` on `**/*.xhtml`. Credits come
  from `src/epub/content.opf`. Its site's robots.txt refuses only assistant
  fetchers, not training crawlers.
- **eBible.org** read-aloud archives (`eBible.org/Scriptures/<id>_readaloud.zip`):
  type `http-archive`, format `text`; exclude the copyright page, the
  signing key and the empty front file. eBible rebuilds archives often, so
  a later fetch can give another checksum: fetch again and update the
  revision and checksum together.
