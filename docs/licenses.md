# Licenses

## The project's license

The code and documents of this repository are under AGPL-3.0-or-later: it
also covers use as a network service, and it accepts the most compatible
inputs (MIT, BSD, CC0, Apache-2.0, GPLv3...). "Or later" avoids being stuck
with one version. Compatibility runs one way: permissive content can enter
copyleft, never the reverse.

The weights of WPT models are under AGPL-3.0-or-later too. Each model folder
carries the license text (`LICENSE`), a `NOTICE` naming every source the
model learned from with the credit it asks for, and a model card; the weight
files' headers name the licenses (see [training.md](training.md)).

Still open:

- whether built data (token shards, vocabularies) gets a license of its own;
- the legal effect of a license on weights is not settled; a lawyer reviews
  it before weights are published, and the commitment is written in a
  manifesto next to the license.

## Dependencies

Every package `uv.lock` and `packages/wpt-tokenizer/rust/Cargo.lock` can
install, with the license it declares, read from its metadata
(`License-Expression`, `License` or license classifiers) and, where those
are vague or missing, from its license files. All of them were checked for
the versions below.

**When a version changes** in `uv.lock` or `Cargo.lock`, check that row
again (the license can change between versions) and update its version
here; a new package gets a row, a removed one loses its row. The required
packages must stay compatible with AGPL-3.0-or-later.

### Every install (the `dev` group: apps, libraries, tests)

| Package | Version | License | Note |
| --- | --- | --- | --- |
| attrs | 26.1.0 | MIT | |
| boto3 | 1.43.113 | Apache-2.0 | |
| botocore | 1.43.113 | Apache-2.0 | |
| certifi | 2026.7.22 | MPL-2.0 | tests, through moto (requests) |
| cffi | 2.1.1 | MIT-0 | tests, through moto (cryptography) |
| charset-normalizer | 3.5.2 | MIT | tests, through moto (requests) |
| colorama | 0.4.6 | BSD-3-Clause | Windows only, through pytest |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | tests, through moto; its compiled part includes OpenSSL (Apache-2.0), statically linked |
| idna | 3.20 | BSD-3-Clause | tests, through moto (requests) |
| iniconfig | 2.3.1 | MIT | through pytest |
| jmespath | 1.1.0 | MIT | |
| jsonschema | 4.26.0 | MIT | |
| jsonschema-specifications | 2025.9.1 | MIT | |
| lxml | 6.1.3 | BSD-3-Clause | reads XHTML for the Markdown conversion; its compiled part bundles zlib (Zlib), libiconv (LGPL-2.1), libxml2, libxslt and libexslt (MIT) |
| moto | 5.2.4 | Apache-2.0 | tests: answers the S3 test's requests inside the test process |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | through pytest |
| pluggy | 1.6.0 | MIT | through pytest |
| protego | 0.7.0 | BSD-3-Clause | reads robots.txt files |
| py-partiql-parser | 0.6.3 | MIT | tests, through moto |
| pyarrow | 26.0.0 | Apache-2.0 | |
| pycparser | 3.11 | BSD-3-Clause | tests, through moto (cffi) |
| pygments | 2.21.0 | BSD-2-Clause | through pytest |
| pytest | 9.1.1 | MIT | tests |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause | newer code is also under Apache-2.0; all of it is under BSD-3-Clause |
| pyyaml | 6.0.3 | MIT | tests, through moto |
| referencing | 0.37.0 | MIT | |
| requests | 2.34.2 | Apache-2.0 | tests, through moto |
| responses | 0.26.3 | Apache-2.0 | tests, through moto |
| rpds-py | 2026.9.1 | MIT | |
| s3transfer | 0.19.2 | Apache-2.0 | |
| safetensors | 0.8.0 | Apache-2.0 | |
| six | 1.17.0 | MIT | |
| urllib3 | 2.8.0 | MIT | |
| werkzeug | 3.1.9 | BSD-3-Clause | tests, through moto |
| xmltodict | 1.0.4 | MIT | tests, through moto |

### PyTorch (one build per machine, picked by `scripts/setup.py`)

Every build declares Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND
BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT.

| Package | Version | Build |
| --- | --- | --- |
| torch | 2.14.1 | CPU, macOS |
| torch | 2.14.1+cpu | CPU, Linux and Windows |
| torch | 2.14.1+cu130 | NVIDIA (CUDA 13.0) |
| torch | 2.14.1+rocm7.2 | AMD (ROCm 7.2), Linux only |
| torch | 2.14.1+xpu | Intel GPU |

Two builds carry vendor libraries inside the PyTorch package itself, under
the vendor's terms rather than PyTorch's:

- the CUDA build on Windows: NVIDIA's CUDA runtime, cuBLAS, cuDNN, cuFFT,
  cuRAND, cuSOLVER, cuSPARSE, NVRTC, nvJitLink, CUPTI and NVTX libraries
  (on Linux they come as the separate packages listed below);
- the ROCm build: AMD's ROCm libraries (HIP, hipBLASLt, hipSPARSELt and
  others). PyTorch's license files do not cover them, and their terms have
  not been checked one by one.

### Every PyTorch build (its own dependencies)

| Package | Version | License | Note |
| --- | --- | --- | --- |
| filelock | 4.1.1 | MIT | |
| fsspec | 2026.9.0 | BSD-3-Clause | |
| jinja2 | 3.1.6 | BSD-3-Clause | |
| markupsafe | 3.0.4 | BSD-3-Clause | |
| mpmath | 1.3.0 | BSD-3-Clause | |
| networkx | 3.7 | BSD-3-Clause | |
| setuptools | 84.0.0 | MIT | |
| sympy | 1.14.0 | BSD-3-Clause | |
| typing-extensions | 4.16.0 | PSF-2.0 | |

### CUDA build, Linux only

"NVIDIA proprietary" means `LicenseRef-NVIDIA-Proprietary` or NVIDIA's
license agreements (the CUDA Toolkit end user license agreement, the license
agreement for NVIDIA software development kits).

| Package | Version | License | Note |
| --- | --- | --- | --- |
| cuda-bindings | 13.4.3 | Apache-2.0 | Python below 3.15 |
| cuda-pathfinder | 1.8.3 | Apache-2.0 | Python below 3.15 |
| cuda-toolkit | 13.0.3.0 | none stated | holds only metadata listing the NVIDIA packages, no code |
| nvidia-cublas | 13.1.1.3 | NVIDIA proprietary | |
| nvidia-cuda-cupti | 13.0.85 | NVIDIA proprietary | |
| nvidia-cuda-nvrtc | 13.0.88 | NVIDIA proprietary | |
| nvidia-cuda-runtime | 13.0.96 | NVIDIA proprietary | CUDA Toolkit license agreement, from its license file |
| nvidia-cudnn-cu13 | 9.24.0.43 | NVIDIA proprietary | software development kit agreement, from its license file |
| nvidia-cufft | 12.0.0.61 | NVIDIA proprietary | |
| nvidia-cufile | 1.15.1.6 | NVIDIA proprietary | |
| nvidia-curand | 10.4.0.35 | NVIDIA proprietary | |
| nvidia-cusolver | 12.0.4.66 | NVIDIA proprietary | |
| nvidia-cusparse | 12.6.3.3 | NVIDIA proprietary | |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA proprietary | |
| nvidia-nccl-cu13 | 2.30.7 | BSD-3-Clause | from its license file |
| nvidia-nvjitlink | 13.4.92 | NVIDIA proprietary | |
| nvidia-nvshmem-cu13 | 3.4.5 | NVIDIA proprietary | CUDA Toolkit license agreement, from its license file |
| nvidia-nvtx | 13.0.85 | Apache-2.0 | from its license file (its metadata also says proprietary) |
| triton | 3.8.0 | MIT | Python below 3.15 |

### ROCm build, Linux only

| Package | Version | License |
| --- | --- | --- |
| triton-rocm | 3.8.0 | MIT |

### Intel GPU build

"Intel EULA" is the Intel End User License Agreement for Developer Tools;
"Intel SSL" is the Intel Simplified Software License. Both are proprietary.

| Package | Version | License | Note |
| --- | --- | --- | --- |
| dpcpp-cpp-rt | 2026.1.0 | Intel EULA | |
| impi-rt | 2021.18.1 | Intel SSL | Linux x86-64 only |
| intel-cmplr-lib-rt | 2026.1.0 | Intel EULA | |
| intel-cmplr-lib-ur | 2026.1.0 | Intel EULA | |
| intel-cmplr-lic-rt | 2026.1.0 | Intel EULA | |
| intel-opencl-rt | 2026.1.0 | Intel EULA | |
| intel-openmp | 2026.1.0 | Intel EULA | |
| intel-pti | 1.0.1 | MIT | from its license file (its metadata also says proprietary) |
| intel-sycl-rt | 2026.1.0 | Intel EULA | |
| mkl | 2026.1.0 | Intel SSL | |
| oneccl | 2022.1.1 | Intel SSL | Linux x86-64 only |
| oneccl-devel | 2022.1.1 | Intel SSL | Linux x86-64 only |
| onemkl-license | 2026.1.0 | Intel SSL | |
| onemkl-sycl-blas | 2026.1.0 | Intel SSL | |
| onemkl-sycl-dft | 2026.1.0 | Intel SSL | |
| onemkl-sycl-lapack | 2026.1.0 | Intel SSL | |
| onemkl-sycl-rng | 2026.1.0 | Intel SSL | |
| onemkl-sycl-sparse | 2026.1.0 | Intel SSL | |
| pyelftools | 0.33 | public domain | |
| pyzes | 0.1.2 | MIT | Linux x86-64 only |
| tbb | 2023.1.0 | Intel SSL | |
| tcmlib | 1.5.0 | Intel SSL | |
| triton-xpu | 3.8.0 | MIT | from the PyTorch package index |
| umf | 1.1.0 | Apache-2.0 WITH LLVM-exception | |

### Rust crates of the tokenizer (`Cargo.lock`)

| Crate | Version | License |
| --- | --- | --- |
| heck | 0.5.0 | MIT OR Apache-2.0 |
| libc | 0.2.190 | MIT OR Apache-2.0 |
| once_cell | 1.21.4 | MIT OR Apache-2.0 |
| portable-atomic | 1.15.0 | Apache-2.0 OR MIT |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 |
| pyo3 | 0.29.3 | MIT OR Apache-2.0 |
| pyo3-build-config | 0.29.3 | MIT OR Apache-2.0 |
| pyo3-ffi | 0.29.3 | MIT OR Apache-2.0 |
| pyo3-macros | 0.29.3 | MIT OR Apache-2.0 |
| pyo3-macros-backend | 0.29.3 | MIT OR Apache-2.0 |
| quote | 1.0.47 | MIT OR Apache-2.0 |
| syn | 2.0.119 | MIT OR Apache-2.0 |
| target-lexicon | 0.13.5 | Apache-2.0 WITH LLVM-exception |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 |

### Tools that build and check WPT (not part of it)

They run unmodified next to WPT and are not linked into it.

| Tool | Version | License | Role |
| --- | --- | --- | --- |
| maturin | >=1.15,<2 | MIT OR Apache-2.0 | builds the tokenizer package |
| uv_build | >=0.13.0,<0.14.0 | MIT OR Apache-2.0 | builds the other packages |
| Rust toolchain | 1.98.0 | MIT OR Apache-2.0 | compiles the tokenizer |
| git | any | GPL-2.0-only | fetches sources, applies patches, checks commits |
| mercurial | any | GPL-2.0-or-later | fetches mercurial sources |
| GnuPG | any | GNU GPL | checks GPG-signed decisions |
| actions/checkout | v4 | MIT | CI |
| astral-sh/setup-uv | v6 | MIT | CI |

### Distribution

WPT does not distribute any of these packages: installing a build downloads
them on the user's machine, under their own terms. Distributing WPT
together with them (a container image, an installer) needs its own license
check first, above all for the NVIDIA, Intel and AMD runtimes.

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
  use: a statement by the owner in a README or a machine-readable
  text-and-data-mining reservation (EU Directive 2019/790, article 4) leaves
  the content out. Review looks for them on every source.
- **robots.txt is enforced in the code**, with no setting to turn it off:
  before any download from an http(s) address, `wpt-dataset` reads the
  site's robots.txt (RFC 9309) and does not download from that address when
  it is forbidden to every crawler (`*`), to `wpt-dataset` itself, or to an
  AI training crawler. Those are the crawlers whose operator documents that
  they gather training data, listed with the link to that statement in
  `packages/wpt-dataset/src/wpt_dataset/robots.py`. Crawlers that only fetch
  a page for a person using an AI assistant (Claude-User, ChatGPT-User...)
  are not on the list: a site that refuses only them has not refused
  training.
- **A robots.txt speaks for its own site.** Whoever runs a site writes its
  robots.txt, and that is not always the owner of the content: a hosting
  platform can block every AI crawler while the author's own site allows
  them. So a refused address is skipped and the source's next mirror is
  tried, read with its own robots.txt; the source fails (`CrawlForbidden`)
  only when no address works and one of them refused. A statement by the
  owner (above) still leaves the content out wherever it is hosted.
- **Only a missing robots.txt means no rules**: an answer 404 (not found)
  or 410 (gone), since some sites publish no policy. Any other answer
  (401, 403, 429, a server error, a timeout, an unreachable site) leaves the
  policy unknown, so that address is not used and the next mirror is tried.
- **No personal data**: sources made of personal data are not accepted.
- A machine can check that a license is written and allowed, not that it is
  true. If the source page lies, only human review catches it.

## Who decides

A human decides every source; the decision is recorded in `source.json`
(`decision`, with a reason for a refusal, a retirement or an exceptional
admission) and its commit must be signed by an allowed key (see
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
