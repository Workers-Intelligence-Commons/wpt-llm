# Licenses

## The project's license

The code and documents of this repository are under AGPL-3.0-or-later: it
also covers use as a network service, and it accepts the most compatible
inputs (MIT, BSD, CC0, Apache-2.0, GPLv3...). "Or later" avoids being stuck
with one version. Compatibility runs one way: permissive content can enter
copyleft, never the reverse.

Still open:

- whether published weights and built data get a license of their own (for
  example AGPL for the code, and another license for data and weights);
- the legal effect of a license on weights is not settled; a lawyer reviews
  it before weights are published, and the commitment is written in a
  manifesto next to the license.

## What the corpus accepts

**When in doubt, refuse.** A doubtful content is left out rather than used.

The allowed list (SPDX identifiers), in
`packages/wpt-dataset/src/wpt_dataset/licenses.py`:

- CC0-1.0, Unlicense, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Apache-2.0
- GPL-3.0-or-later, LGPL-3.0-or-later, AGPL-3.0-or-later
- `LicenseRef-public-domain`: public domain, with a text next to source.json
  stating why

Refused by default, the grey zone: CC-BY-4.0, CC-BY-SA-4.0, ODbL-1.0. They
stay refused until a lawyer confirms them; a compatibility statement from
Creative Commons or the FSF is not enough on its own. Facts gathered for that
review:

- Creative Commons declared GPLv3 a BY-SA-compatible license for BY-SA 4.0 on
  2015-10-08, one way only
  (<https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/>).
- The FSF reportedly lists CC BY 4.0 as compatible with all GPL versions, and
  CC BY-SA 4.0 with GPLv3 only (secondary source, to confirm on gnu.org).
- Wikipedia text is CC BY-SA 4.0 under the current Wikimedia Terms of Use.
- Common Pile v0.1 includes CC BY-SA content; WPT's rule is stricter.

Refused: any NC (no commercial use) or ND (no derivatives) clause,
GPL-2.0-only, CC-BY-SA-3.0, anything without a license or with all rights
reserved.

Permissive does not mean without obligations: MIT, BSD and Apache require
keeping attribution, so the copyright notices of a source stay in its tree.

## Rules

- A source can hold several licenses, each mapped to its paths; every path
  matches exactly one.
- Checking is per file, not per repository: a license can be applied by
  mistake, and public domain depends on the country.
- **Public domain** means public domain both in the United States and in the
  work's country of origin (the Wikimedia Commons rule). A work can still be
  protected in the country someone publishes from.
- **"No AI training" wishes are honored** even when the license would allow
  use: a statement in a README, robots.txt, or a machine-readable
  text-and-data-mining reservation (EU Directive 2019/790, article 4) leaves
  the content out. Review looks for them on every source.
- **No personal data**: sources made of personal data are not accepted.
- A machine can check that a license is written and allowed, not that it is
  true. If the source page lies, only human review catches it.

## Who decides

A human decides every source; the decision is recorded in `source.json`
(`decision`) and its commit must be signed by an allowed key (see
[CONTRIBUTING.md](../CONTRIBUTING.md)). There is no automatic filtering or
tagging.

Tools may help reviewers by flagging, never by deciding:

- [ScanCode Toolkit](https://github.com/aboutcode-org/scancode-toolkit)
  detects license texts file by file, which helps on large repositories. It
  is optional and not part of the build.
- A small model trained on past human decisions may later flag doubtful
  cases. It never decides.

## No distillation from another company's model

No output of another company's language model enters the data or the
training, including "synthetic" datasets whose human origin is not proven.
