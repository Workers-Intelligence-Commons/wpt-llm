# WPT data architecture: discussion plan

Working notes from the `causons` discussion, 2026-10-14. This file lives in
the repo so it survives the session; apart from it, nothing in the repo or
Notion changes until the plan is complete and the user starts the work step.
Source pages read: Notion "WPT : Workers'
Pre-trained Transformer" and subpages "Architecture de sourcing", "Notice de
sortie et attribution", "Charte (brouillon)", "README (brouillon)".

## Settled decisions

### Repository and documentation
- Every decision (Notion ones + this discussion) is written in `docs/` in the
  repo, in English. Notion keeps very few decisions and points at the repo.
- Data tools (fetch, verify, patch, build) are Python, inside the uv workspace.

### Licenses
- In doubt, refuse. Grey-zone licenses (CC-BY-4.0, CC-BY-SA-4.0, ODbL) stay
  refused until a lawyer confirms them. The user is a software engineer, not a
  lawyer; a CC/FSF compatibility statement is not enough on its own.
  Verified facts for the lawyer file: CC declared GPLv3 a BY-SA-compatible
  license for BY-SA 4.0 on 2015-10-08, one-way
  (https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/);
  FSF reportedly lists CC BY 4.0 as compatible with all GPL versions and BY-SA
  4.0 with GPLv3 only (secondary source; gnu.org blocked automated reads);
  Wikipedia text is CC BY-SA 4.0 per current Wikimedia Terms of Use.
- A source can hold several licenses, each attached to a set of paths:
  in source.json, a list of `{license, paths}` entries where `paths` are glob
  patterns; every manifest path must match exactly 1 entry or the check fails
  (a new upstream file matching no glob blocks the update until assigned).
- A model may only FLAG and sort cases; a human decides every source
  (Notion rule kept). A small model trained on the user's manual decisions may
  later assist with flagging only.
- CONTRIBUTING.md: a new source needs a clearly compatible license (or one
  backed by case law); every source JSON states its license(s). No automatic
  filtering or tagging; review is manual.
- No personal-data sources are accepted.
- Public domain = public domain in BOTH the US and the work's country of
  origin (the Wikimedia Commons rule). Accepted cost: a work may still be
  protected in the country the user publishes from.
- "No AI training" wishes are honored even when the license would allow use:
  a README statement, robots.txt, or the machine-readable TDM reservation of
  EU Directive 2019/790 article 4 excludes the content. Review looks for them
  on every source.
- Contributors declare with DCO: every commit carries `Signed-off-by`
  (`git commit -s`), certifying the right to submit; CI checks it. Grants no
  relicensing rights.

### Sources and patches
- Source id = folder name `<origin>-<item>-<hash>` (example
  `gutenberg-1342-3fa9c1`), or `<origin>-<hash>` when there is no natural item;
  lowercase `a-z 0-9 -`, never reused. `<hash>` is 8 hex characters of the
  source's address, so it stays stable when the source updates in place.
  When upstream moves to a new address, that is a NEW source with a new id;
  the old one is retired, mixtures naming it are edited, and its erasure cuts
  move to the new folder.
- One directory per source: `data/sources/<id>/` with one JSON file + its
  patches in order (fields: see "source.json shape").
- Checksum = sha256 of a canonical MANIFEST: one line per extracted file
  (format below), sorted; the manifest's own sha256 is the source checksum. The manifest doubles as the per-file inventory used by
  licenses-by-path. The manifest records everything: symlinks (as their
  target) and the executable bit too, so a code repo is captured exactly.
  Known cost: Windows handles symlinks and permissions differently, so the
  same repo would yield different manifests on Linux and Windows if read from
  disk, so the manifest is computed from the DOWNLOADED DATA itself (git's
  object tree, archive file headers), one reader per source type. Remaining
  cost: patches run on disk files, so a patch touching a symlink behaves
  differently on Windows.
- Manifest line format, git-style: `<mode> <sha256> <path>`, mode 100644
  (file), 100755 (executable) or 120000 (symlink; sha256 of the target text).
  Not checkable with plain `sha256sum -c`; WPT provides the check command.
- Paths: UTF-8 with `/`, sorted by bytes; contents hashed as-is (line endings
  untouched). A path that is not valid UTF-8, or 2 paths differing only by
  letter case, make the source fail; an extraction rule or removal patch must
  exclude them first.
- Empty folders are ignored, so a git checkout and an archive of the same
  release give the same checksum and can be mirrors of each other.
- New upstream revision: the source updates IN PLACE (`data/sources/<id>/` moves
  to the new revision; its cuts and patches are redone in the same commit).
  Cost: rebuilding an old recipe needs an old commit of the repo, and every
  erasure cut is re-pinned by hand before the update lands.
- Patches use git's patch format; git becomes a dependency of data processing.
- Removal patches ARE allowed even though their `-` lines republish removed
  text. Position: the repo is the download-and-treatment software; the project
  is what comes out of it; docs state plainly that removed material is not
  used, not endorsed, and explicitly removed (removal = proof of respecting
  licenses). Pushback was given once; user's call stands.
- Setup is incremental and idempotent: never re-download or re-apply what is
  already done; smaller/partial patch sets possible. Design idea (mechanics):
  name every intermediate result by the fingerprint of what produced it
  (source checksum + each patch checksum + tool version), Nix-style, so
  "already done" is a lookup and changing one patch rebuilds only that source.

### source.json shape (approved)
File `data/sources/gutenberg-1342-3fa9c1/source.json`; patches
`0001-*.patch`, `0002-*.patch`... sit next to it, applied in order.
```json
{
  "schema": 1,
  "type": "http-archive",
  "address": "https://www.gutenberg.org/cache/epub/1342/pg1342-h.zip",
  "revision": "2026-03-01",
  "checksum": "sha256:...",
  "extract": { "strip": 1, "exclude": ["**/*.jpg"], "note": "" },
  "licenses": [ { "license": "<SPDX id>", "paths": ["**"], "note": "" } ],
  "tags": [ { "tags": ["en", "literature"], "paths": ["**"], "note": "" } ],
  "mirrors": [ { "address": "https://...", "note": "" } ],
  "cuts": [ { "path": "...", "start": 0, "end": 0, "sha256_after": "...", "note": "" } ],
  "decision": { "decision": "admitted", "by": "...", "date": "2026-10-14", "reason": "...", "note": "" },
  "note": ""
}
```
- `schema`: shape version, for migrations. `type`: http-file, http-archive,
  git, mercurial. `revision`: full commit hash for git/mercurial, fetch date
  for http. `checksum`: sha256 of the canonical manifest.
- Every object has a free-text `note`.
- `decision` (required): `decision` = admitted | refused | retired, `by`,
  `date`, `reason`. The 3 checks are not fields: the DCO sign-off certifies
  them.
- Who may change a `decision`: CI requires every commit that changes one to
  carry a cryptographic signature (git commit signed with an SSH or GPG key)
  from a key in an allowed list kept in the repo; changes to that list need a
  signature from a key already in it. A `Signed-off-by` line alone proves
  nothing (anyone can type it).
- Validated by a JSON Schema in CI.
- Licenses are SPDX identifiers; anything not on the SPDX list is
  `LicenseRef-<name>` with its full text stored in the source folder. CI
  checks every license against the allowed list.
- Erasure cuts live in source.json (`cuts`).
- Responsibility goes through the DCO:
  signing off a source commit certifies that the license was read, that no
  "no AI" wish was found, and that the source has no personal data; the
  committer takes responsibility. CONTRIBUTING.md states this.
- `wpt-dataset` includes a migration tool: each file carries its shape
  version (`schema`), and migrations rewrite every source and recipe from one
  shape to the next and add rules, so no file is edited by hand.

### Erasure
- Every erasure request is honored, including a copyright holder's request on
  properly licensed content.
- Erasures never go through patches (a patch would publish the erased text and
  keep it in git history). Mechanism: FINGERPRINT CUT = `{path, start byte,
  end byte}` on the extracted upstream (pinned by its checksum, so the range is
  exact forever), applied BEFORE patches; patches are written against the
  already-cut text; CI catches patches that no longer apply.
- Cut check: sha256 of the whole file AFTER the cut (reveals only position and
  length; a hash of the removed bytes could be confirmed by guessing).
- Cut text disappears; no marker.
- A person asking to be erased (including the name in a copyright notice that
  a license requires keeping) gets the notice AND everything they made
  removed, so no license is left broken. Otherwise notices stay as licensed.
- Channels: public GitHub issue (easy, automatable later) AND GitHub private
  vulnerability reporting (must be enabled on the repo; reporter needs a GitHub
  account; labeled "security"). GDPR deadline: 1 month.
- Erasure applies to future builds and models only; docs say so plainly.
- Background facts: projects almost never rewrite history for license removals
  (new commit); Debian never commits excluded upstream files (`Files-Excluded`
  in debian/copyright with mandatory reason, `+dfsg` repack); history rewrites
  are for leaked secrets and are incomplete (clones, forks, cached views keep
  data; GitHub Support must purge) - used here for accidents only.

### Storage
- The repo holds pointers + recipes + the code that reads them; WPT does not
  host data. Each machine points ONE setting in `.env` at its store:
  `WPT_STORE=file:///...` or `s3://bucket/prefix` (S3-compatible). Contents are
  content-addressed by sha256 and verified.

### Data selection
- Tags + mixtures: tags live in each source's JSON (a `tags` field,
  optionally per path like licenses); each pool is a small expression over
  tags with `&&` (and), `||` (or), `!` (not) and parentheses, as in common
  programming languages. `!` is allowed only directly before an opening
  parenthesis, to keep parsing simple: `fr && !(poetry)` is valid, `fr && !poetry`
  is not; mixtures
  combine pools with weights. Cost: retagging many sources edits many files.
- File format: plain JSON everywhere (sources, mixtures, recipes, lock files).
  JSONC was chosen, then dropped: its parser discards comments, so a
  migration rewriting files would delete them. Instead every JSON object has
  a free-text `note` field, which survives migrations because it is data.
- Human decisions are a structured object with required fields, so they are
  trackable across a huge list of sources (see the source.json shape).
- A document matching several pools of one mixture goes to the FIRST
  matching pool in the mixture's order. Accepted cost: reordering pools in
  the file changes the dataset (the dataset id changes with it, so it is
  visible in the fingerprint).
- Weight = share of the dataset's total tokens. The total is set by the
  DATASET RECIPE (mixture, tokenizer recipe, total tokens, held-out share);
  that recipe is the only one enforced. No tokens-per-parameter rule: a model
  may be data-starved or oversized for its data. A small pool repeats to fill its
  share, capped at 4 epochs by default (Muennighoff et al. 2023: up to ~4
  passes is about as good as fresh data). Beyond the cap the run refuses to
  start unless the mixture explicitly allows more.
- Languages: as many as possible, plus translation as a capability.
- Model size is chosen when training starts (a command-line option or an
  environment variable), not in a recipe. The repo is recipes, scripts and environment
  setups usable for any size. Flags/features let a smaller model run on a
  subset (example: computer-use pre/post-training).

### Mixture and dataset recipe shapes (derived from the decisions above)
`data/recipes/mixtures/<name>.json`:
```json
{
  "schema": 1,
  "pools": [
    { "name": "fr", "select": "fr && !(literature)", "weight": 0.5, "max_epochs": 4, "note": "" },
    { "name": "literature", "select": "literature", "weight": 0.5, "note": "" }
  ],
  "note": ""
}
```
- Weights sum to 1. `max_epochs` is optional (default 4); a higher value is
  the explicit permission to repeat more.
`data/recipes/datasets/<name>.json`:
```json
{ "schema": 1, "mixture": "<mixture name>", "tokenizer": "<tokenizer recipe name>",
  "total_tokens": 2000000000, "held_out": 0.001, "note": "" }
```
- The dataset id printed by `wpt-dataset build` is the fingerprint of this
  recipe, the mixture, the tokenizer vocabulary and every source checksum.

### Tokenizer
- Own byte-level BPE (every byte has a token; today's tokenizer silently drops
  unseen characters in `encode`), hot loop in Rust, as a WORKSPACE package
  (PyO3 + maturin, uv-buildable), with its own version number.
- Trained once per tokenizer recipe as a shared, versioned artifact; every run
  and model size uses it.
- The vocabulary (merge list) is a build result named by the fingerprint of
  its inputs: tokenizer code version + tokenizer recipe (data, vocabulary
  size) + checksums of that data. A dataset recipe names its tokenizer
  recipe; the build reuses or rebuilds the vocabulary from the fingerprint.
- BPE training must be deterministic: ties between equally frequent pairs
  broken by a fixed rule, result independent of thread count.
- The build records the vocabulary's own sha256 the first time; a later
  rebuild that differs fails. It is kept in a committed lock file next to
  the recipe (`<recipe>.lock.json`, like `uv.lock`); the first build is
  followed by a commit.

### Repeatability
- DATA is exactly repeatable, for audits: sources, cuts, patches, vocabulary
  and token content. Shards are checked by hashing their token content, not
  the Parquet file bytes (pyarrow upgrades change bytes, not tokens).
- Tool versions: the repo pins a reference ("most known") version of each
  tool (uv.lock, .python-version, a Rust toolchain pin). Each build records
  the versions it actually used and reports any difference from the
  reference; it never refuses to run because of one. The data hashes catch
  any real change. Background: most upgrades change nothing; Unicode-table
  changes only affect newly added characters in Unicode-dependent steps.
- Training is NOT bit-exact (GPU arithmetic differs across GPU, driver and
  PyTorch version). Around it, everything possible is checked: before
  training, every input is verified against its hash (shards, vocabulary,
  dataset recipe); the model folder records what produced it (dataset id,
  options, repo commit,
  input hashes, tool versions, device, seed).

### Translation
- Parallel text comes from public-domain pairs (old works and public-domain
  translations, Gutenberg, Wikisource; needs a sentence-alignment tool) and
  from contributed translations under DCO. Europarl/Tatoeba stay refused
  (CC BY) until a lawyer confirms.

### Self-distillation
- No model takes part in training in any way for now; there is no plan for
  it. If it ever happens (WPT's own bigger model producing data or a
  post-training role for smaller ones), it is stated explicitly and must be
  judgeable: CONTRIBUTING.md requires saying exactly what the bigger model
  did. Generated data
  is a build result fingerprinted by model, prompts and seed. Open for the
  lawyer: whether generated text inherits its training data's licenses.

### Deduplication
- Exact duplicate documents (same hash) are removed; the copy kept belongs to
  the first source by id. Never judges content. Near-duplicates stay.

### Evaluation
- Held-out text: a fixed share of every source's documents (for example 1 in
  1,000), chosen by a hash of the document id, never enters training.
- External benchmarks are NOT part of this repo or the model; WPT works
  without them. Training on a benchmark follows the normal license rule.
- Contamination (evaluation text also in training data, checked by overlap of
  long word runs, GPT-3 used 13 words) is REPORTED only; training data never
  depends on evaluation sets. Cost: inflated scores, read the report.

### Repo layout and checks
- Data descriptions live under `data/`: `data/sources/<id>/` (source.json +
  patches), `data/recipes/mixtures/`, `data/recipes/tokenizers/` (+ lock
  files), `data/recipes/datasets/`. Next to `docs/`, `packages/`, `apps/`,
  `scripts/`. (`data/` holds descriptions only; WPT never hosts data.)
- All dataset logic (store, sources, build, the checks CI runs) lives in a
  dedicated dataset package that can grow its own rules. The training package
  is only training: multi-GPU distribution, choosing the best checkpoint...
- Packages: `wpt-dataset` (sources, cuts, patches, manifests, build,
  dedup, held-out split, tokens, Parquet, CI checks; command
  `wpt-dataset build <dataset recipe>`) and a small
  separate store package (local path or S3, content-addressed, verified), so
  training reads the store without depending on the dataset rules.
- `wpt-dataset build <dataset recipe>` prints a built dataset id; `wpt-train`
  takes that BUILT DATASET ID plus model size and every training setting
  (learning rate, steps, batch size...) as command-line options or
  environment variables; there is no training recipe file; it never builds, only reads
  and verifies. Build and training fail separately. The model folder records
  the options used.
- Checks run in CI only (GitHub, every pull request): schema, license map
  covers every path, DCO sign-off, tests, and for changed sources: fetch,
  confirm checksum, patches apply. No local hook.

### Built shards
- Parquet (chosen by the user's criterion: most open and most widely used for
  datasets). Columns include tokens, source id, document id. Cost: pyarrow
  dependency; training loader reads row groups sequentially.

### Post-training
- Both recorded human demonstrations and interactive environments.
- Record format: JSONL, 1 record per line; media (screenshots...) are separate
  files named by sha256. Each record carries provenance: contributor, DCO
  commit, authorization reference, license. Converted before training.
- Environments follow the corpus rules: no fake web/apps; recorded content
  when the user has authorization (user manages it). Scores come from programs
  or humans, never another company's model (own-model judging not chosen).

### This plan
- `docs/plan.md` holds the plan during the discussion. Settled decisions
  become `docs/` pages at the end, once the plan is complete.
- Git tracks this file until publication; it is deleted before the repo goes
  public (cost: git history still holds it unless that history is rewritten;
  it names the private data files and the user's positions).

## Open questions
None.

## Execution steps (the later work step)
- Write `docs/` (English) from Notion + this plan; reduce Notion to 1 pointer
  page (the main page); the 4 subpages go.
- Write CONTRIBUTING.md: license rule, SPDX ids, manual review, DCO sign-off
  and what it certifies, signed commits for decisions, erasure channels,
  declaring any model's role.
- Create `data/` (sources, recipes/mixtures, recipes/tokenizers,
  recipes/datasets) and the JSON Schemas.
- New packages: the store package (local path or S3) and `wpt-dataset`
  (manifest readers per source type, cuts, patches, tags and pool
  expressions, dedup, held-out split, Parquet shards, migrations, CI checks;
  command `wpt-dataset build`).
- Replace the pure-Python `wpt-tokenizer` with the byte-level BPE in Rust
  (PyO3 + maturin), deterministic training, vocabulary lock files.
- Change `wpt-train` to take a built dataset id and command-line/environment
  settings, verify inputs, and record provenance in the model folder.
- CI on GitHub: schema, license map, allowed licenses, DCO, signed decision
  commits, tests, changed sources fetched and checked.
- Before publication: enable private vulnerability reporting; delete
  `docs/plan.md` and decide about its history.
