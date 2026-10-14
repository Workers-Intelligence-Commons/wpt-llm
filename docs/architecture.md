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

One folder per source: `data/sources/<id>/` holds `source.json`, its patches
`0001-<name>.patch`, `0002-<name>.patch`... applied in that order, and the
text of any `LicenseRef-<name>` license as `LicenseRef-<name>.txt`.

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
  "schema": 1,
  "type": "http-archive",
  "address": "https://www.gutenberg.org/cache/epub/1342/pg1342-h.zip",
  "revision": "2026-03-01",
  "checksum": "sha256:...",
  "extract": { "strip": 1, "exclude": ["**/*.jpg"], "note": "" },
  "licenses": [ { "license": "LicenseRef-public-domain", "paths": ["**"], "note": "" } ],
  "tags": [ { "tags": ["en", "literature"], "paths": ["**"], "note": "" } ],
  "mirrors": [ { "address": "https://...", "note": "" } ],
  "cuts": [ { "path": "...", "start": 0, "end": 0, "sha256_after": "...", "note": "" } ],
  "decision": { "decision": "admitted", "by": "...", "date": "2026-10-14", "reason": "...", "note": "" },
  "note": ""
}
```

- `schema`: the shape version of the file, for migrations.
- `type`: `http-file`, `http-archive` (tar, tar.gz, tar.bz2, tar.xz or zip),
  `git` or `mercurial`.
- `revision`: the full commit hash for git and mercurial; the fetch date
  (`YYYY-MM-DD`) for http, where the checksum is what pins the content.
- `checksum`: the sha256 of the canonical manifest (below).
- `extract`: `strip` removes leading path components (entries left with no
  path are dropped), `exclude` leaves paths out, and `name` gives an
  `http-file` its path.
- `licenses`: SPDX identifiers mapped to paths by glob patterns. Every path
  must match exactly one entry, or the check fails; a new upstream file that
  matches no entry blocks the update until someone assigns it.
- `tags`: describe paths (language, domain); pools select documents by tags.
- `mirrors`: other addresses expected to give the same checksum.
- `cuts`: erasure cuts; see [erasure.md](erasure.md).
- `decision`: the human decision on the source (required): `admitted`,
  `refused` or `retired`, by whom, when, and why.
- Every object may hold a free-text `note`. It is data, so it survives
  migrations (comments would not).

Globs: `*` matches within one folder, `**` across any number of folders.

The schema is `packages/wpt-dataset/src/wpt_dataset/schemas/source.schema.json`.
Mixture, dataset recipe, tokenizer recipe and lock files have schemas there
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

Removal patches are allowed even though their `-` lines show the removed
text: this repository is the software that downloads and treats the data,
and the project is what comes out of it. Removed material is not used, not
endorsed, and explicitly removed; the removal is itself the record that a
license was respected.

## Fingerprints and the store

Every result is stored under the fingerprint of what produced it: the sha256
of its inputs, including WPT's own code version.

| Result | Fingerprint covers |
| --- | --- |
| upstream tree | the source checksum |
| final source tree | checksum, cuts, each patch's sha256, wpt-dataset version |
| vocabulary | tokenizer version, tokenizer recipe, its mixture, the selected documents' sha256 |
| dataset | dataset recipe, mixture, vocabulary, every document's id, sha256 and tags |

So "already done" is a lookup: nothing is downloaded or applied twice, and
changing one patch rebuilds only that source.

The store is set by one setting, `WPT_STORE`, in the environment or in
`.env` at the repository root:

```
WPT_STORE=file:///home/me/wpt-store      a local folder
WPT_STORE=s3://bucket/prefix             an S3-compatible bucket
```

For S3, credentials and a non-AWS endpoint come from the usual
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_ENDPOINT_URL`
settings. Objects are named by their sha256 and checked on every read.

## Packages

| Package | Role |
| --- | --- |
| `wpt-dataset` | all data rules: sources, manifests, cuts, patches, pools, builds, the checks CI runs, migrations; command `wpt-dataset` |
| `wpt-store` | the content-addressed store (local or S3) and the token shard format, so training reads data without depending on the data rules |
| `wpt-tokenizer` | the byte-level BPE tokenizer, Rust inside |
| `wpt-model` | the TinyGPT network and its safetensors model folders |
| `wpt-training` | the training loop and choosing the best checkpoint |
| `wpt-inference` | token-by-token sampling |
| `wpt-device` | picks the best GPU, else the CPU |
| `wpt-train`, `wpt-chat` | the applications |

## Checks

CI runs on every pull request and push (`.github/workflows/ci.yml`):

- `wpt-dataset check`: every file matches its schema and current shape,
  source ids match their address, licenses are on the allowed list (with
  their text for a LicenseRef), patches are numbered without gaps, mixture
  weights add up to 1 and pool expressions parse, recipes name existing
  recipes.
- `wpt-dataset check --changed <base>`: changed sources are fetched; their
  checksum, cuts, patches, license map and text are checked.
- `wpt-dataset check-commits <range>`: DCO sign-off on every commit, and a
  signature from an allowed key on every commit that changes a decision.
- The tests of every package, and the Rust tests.

There is no local hook: run the same commands before pushing if you want an
early answer.

## Migrations

Each data file carries its shape version (`schema`). When a shape changes,
the change comes with a migration step in
`packages/wpt-dataset/src/wpt_dataset/migrate.py`, and `wpt-dataset migrate`
rewrites every file that is behind. Nobody edits the files by hand for it.
