# Erasure requests

**Every erasure request is honored**, including a copyright holder's request
about content that is properly licensed.

## How to ask

- A public GitHub issue on this repository, or
- privately, through GitHub's private vulnerability reporting: the
  repository's **Security** tab, then **Report a vulnerability**. It needs a
  GitHub account. Use it when the request itself should not be public.

Requests under the GDPR are answered within one month.

## What happens

An erasure is a **cut**: a passage removed from one file of a source's
upstream content, recorded in that source's `source.json`. A cut names the
passage by its length and two keyed hashes, never by its text, since the
repository is public:

```json
"cuts": [ { "path": "chapter-3.txt", "length": 1167,
            "hmac": "<HMAC-SHA256 of the passage under the cut key>",
            "rolling": 123456789 } ]
```

- `wpt-dataset cut ID PATH START END --version V` makes a cut from bytes
  `[START, END)` of the file as the cuts before it left it, read from the
  store. A person checks the range on the stored file first.
- `hmac` is the HMAC-SHA256 of the passage, and `rolling` its rolling hash,
  whose parameters come from the same key. The key is a secret kept in the
  store (`wpt_dataset/erasure.py`), so a guess at a short passage, such as
  a name or a phone number, cannot be confirmed from the repository. The
  first cut makes the key. Making a cut needs store access, and so does
  matching one.
- The build slides the keyed rolling hash over the file and confirms each
  position where it matches by the HMAC; every confirmed run is removed,
  wherever it is in the file. Several cuts on one file apply in the listed
  order, each to the result of the one before.
- A cut whose passage is no longer in the file, or a store without the
  key, takes the whole source out of every build until a person makes the
  cut again.
- Cuts apply before patches; patches are written against the cut text, and
  a patch that no longer applies takes the source out the same way.
- The removed text disappears, with no marker.
- Erasures never go through patches: a patch would publish the erased text
  and keep it in git history.

The cut key must be kept and backed up with the store. Without it, no cut
matches, and every source with a cut stays out of the builds until its
cuts are made again.

A person who asks to be erased, including from a copyright notice that a
license requires keeping, gets the notice **and everything they made**
removed, so that no license is left broken.

## Text inside the repository

Sources of type `empty` (text written for WPT) live in this repository as
patches. Erasing from them takes a new patch that removes the text, which
stops future builds from using it, but the text stays in the repository's
history, clones and forks. Such text is committed only by its own author,
who accepts that.

## What it covers

Erasure applies to **future builds and models only**. Models already trained
and published are not changed.

A locked vocabulary is not retrained when a source changes (see
[tokenizer.md](tokenizer.md)), so merges learned from erased text, such as
pieces of a rare name, stay in it until someone raises the tokenizer
recipe's `version`. When an erasure must reach the vocabulary too, raise
the version in the same commit as the cut.

## History rewrites

Projects almost never rewrite git history to remove licensed content: they
add a new commit. Debian, for example, never commits excluded upstream files
(`Files-Excluded` in `debian/copyright`, with a mandatory reason). Rewriting
history is for accidents such as a leaked secret, and it is incomplete:
clones, forks and cached views keep the data until GitHub Support purges it.
This project rewrites history only for accidents.
