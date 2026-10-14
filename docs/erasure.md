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

An erasure is a **cut**: a range of bytes removed from one file of a
source's upstream content, recorded in that source's `source.json`:

```json
"cuts": [ { "path": "chapter-3.txt", "start": 1043, "end": 2210,
            "sha256_after": "<sha256 of the whole file after the cut>" } ]
```

- The range refers to the upstream file as published, which the source's
  checksum pins, so it is exact forever. Several cuts on one file apply in
  the listed order, each to the result of the one before.
- Cuts apply before patches; patches are written against the cut text, and
  CI catches a patch that no longer applies.
- The check is the sha256 of the whole file after the cut. It reveals only
  the position and length of what was removed. (A hash of the removed bytes
  would let someone confirm a guess of the text.)
- The removed text disappears, with no marker.
- Erasures never go through patches: a patch would publish the erased text
  and keep it in git history.

A person who asks to be erased, including from a copyright notice that a
license requires keeping, gets the notice **and everything they made**
removed, so that no license is left broken.

## What it covers

Erasure applies to **future builds and models only**. Models already trained
and published are not changed.

## History rewrites

Projects almost never rewrite git history to remove licensed content: they
add a new commit. Debian, for example, never commits excluded upstream files
(`Files-Excluded` in `debian/copyright`, with a mandatory reason). Rewriting
history is for accidents such as a leaked secret, and it is incomplete:
clones, forks and cached views keep the data until GitHub Support purges it.
This project rewrites history only for accidents.
