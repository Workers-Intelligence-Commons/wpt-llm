---
name: wpt-sources
description: How to gather new training data for WPT, from finding candidate works to the maintainer's admission. Use it when asked to find, add or review sources (books, collections, repositories, datasets), to check whether a work is in the public domain, or to admit sources. It covers where the rules live, when to ask the maintainer, writing the source files, the checks to run, and the explicit yes that admission needs.
---

# Gathering sources

This skill is the order of work. It holds no rule about what a source may
contain: every rule lives in the documentation, and the documentation is
the only place to take one from.

## 0. Read the documentation first

Before each batch, read, in full:

- `docs/licenses.md`: "What the corpus accepts" (the closed license list
  and what to check for each), "Rules" (every rule), "Who decides";
- `CONTRIBUTING.md`, "Adding a source": the files to write and the
  commands;
- `docs/data-selection.md` (documents, tags) and `docs/chat-format.md`
  (conversations);
- the `wpt` skill, for the tools.

Write each source from these documents. A case they do not settle is a
question for the maintainer, never a rule you make up: no exclusion, no
filter, no default that the documentation does not state. When the
maintainer answers, write the answer into the document that covers it, in
the same change, as a general rule about the current state only: no
date, no history, and no source named (CONTRIBUTING.md, "Writing about
the project").

## 1. Candidates

The maintainer names what to look for (an author, a collection, a kind of
text, a list of repositories). Note for each candidate what the
documentation asks you to prove (for public domain: every person whose
words are in the text, the edition, the first publication).

## 2. Prove the license, and ask about every doubt

Prove the license from the source's own statements, as `docs/licenses.md`
says for that license. Cite where each fact comes from. When a fact cannot
be found, or the evidence disagrees, **do not settle it yourself in either
direction**: ask the maintainer about that candidate right away, with the
doubt, the evidence found and what would remove it. Without their answer
the candidate stays out (a doubt stays under review and is not used); `docs/licenses.md` says
what happens to a fact still missing after one more research pass, and
where a question the maintainer cannot settle alone goes.

Candidates without doubt need no question yet; they wait for step 6.

## 3. Write the source files

Follow `CONTRIBUTING.md`, "Adding a source". Everything the documentation
lets in comes in: leave a file out only for a reason a document gives (a
binary file, a license that is not allowed, a doubt, a maintainer's
decision), and say which in `extract.note`. Draft licenses, credits,
formats and tags from the source's own files with a script, never from
memory, and show the maintainer one full `source.json` as the pattern for
the batch before writing the rest. Each license entry names the place of
its statement and the license texts expected there, with coverage
patterns of the allowed shapes (`docs/licenses.md`, "License entries"); a
license text not yet in `data/license-texts/` becomes a new item, under
review, with `wpt-dataset add-license-text ID --file PATH` (or `--page
ADDRESS`) once the source is downloaded. A new
source, each of its entries and each new license text start
`under-review`, as `CONTRIBUTING.md` says.

## 4. Check every source

```sh
uv run wpt-dataset check --fetch ID [ID ...]   # cuts, patches, license and format maps, text, statements
uv run wpt-dataset markdown-check ID [ID ...]  # notes complete, no leftover tags
```

Look at the start of a few documents' Markdown, at least one per kind of
file. Report what each source gives (entries, documents) to the
maintainer. A robots.txt refusal on one address moves to the next mirror;
a source whose every address fails stops. Never download it another way.

## 5. Commit the sources under review

Commit the new sources with their `under-review` decisions (the `wpt` skill's
rules apply: `scripts/check.py`, sign-off, no personal paths).

## 6. Admission needs the maintainer's explicit yes

Never set `"decision": "admitted"` without it, at any level: a source, a
license entry or a license text (`docs/licenses.md`, "Who decides"). Ask
once for the batch: the question lists every source with its one-line
license reason, its entries and the new license texts they expect, and
the sources the maintainer already discussed in step 2 with what was
decided. An answer that approves only some admits only those.

Then set the approved decisions to `admitted`, by the maintainer, with
today's date, and no `reason` unless the admission is an exception (such
as a doubt the maintainer settled, or an explicit permission for content
without a license). The commit that changes decisions must be signed by an
allowed key, so it is the maintainer's commit: show them its full message
first.
