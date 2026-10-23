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
files' headers name the licenses (see [training.md](training.md)). A
model's drafter (`drafter/`, [model.md](model.md), "The drafter") is part of
its weights: the same license, the same header in its weight file. Its
prompts are the beginnings of the model's own training documents, whose
sources the same `NOTICE` already credits, and it learns only from answers
WPT's own model writes, never from another company's model (below).

Data built by WPT's tools (token shards, vocabularies, dataset records) is
under AGPL-3.0-or-later too, by the same logic as code: every source is
treated as a dependency, and every allowed source permits AGPL use. The
content of each source inside keeps its own license wherever it is shared,
with its credit and license text; the AGPL covers what WPT adds to it, the
format, the structure and everything the pipeline puts around the content
(maintainer's decision, 2026-10-16).

Still open:

- the legal effect of a license on weights is not settled; a lawyer reviews
  it before weights are published, and the commitment is written in a
  manifesto next to the license.

## Dependencies

Every package `uv.lock` and `packages/wpt-tokenizer/rust/Cargo.lock` can
install, with the license it declares, read from its metadata
(`License-Expression`, `License` or license classifiers) and, where those
are vague or missing, from its license files. All of them were checked for
the versions below.

**When a version changes** in `uv.lock`, `Cargo.lock` or
`packages/wpt-web/ui/bun.lock`, check that row
again (the license can change between versions) and update its version
here; a new package gets a row, a removed one loses its row. The required
packages must stay compatible with AGPL-3.0-or-later.

### Every install (the `dev` group: apps, libraries, tests)

| Package | Version | License | Note |
| --- | --- | --- | --- |
| attrs | 26.1.0 | MIT | |
| boto3 | 1.43.113 | Apache-2.0 | |
| botocore | 1.43.113 | Apache-2.0 | |
| certifi | 2026.7.22 | MPL-2.0 | through requests |
| cffi | 2.1.1 | MIT-0 | tests, through moto (cryptography) |
| charset-normalizer | 3.5.2 | MIT | through requests |
| colorama | 0.4.6 | BSD-3-Clause | Windows only, through pytest |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause | tests, through moto; its compiled part includes OpenSSL (Apache-2.0), statically linked |
| einops | 0.8.2 | MIT | through fla-core |
| execnet | 2.1.2 | MIT | tests, through pytest-xdist |
| fla-core | 0.5.2 | MIT | KDA's GPU kernel (flash-linear-attention), used where Triton runs |
| googleapis-common-protos | 1.75.5 | Apache-2.0 | through opentelemetry-exporter-otlp-proto-http |
| idna | 3.20 | BSD-3-Clause | through requests |
| iniconfig | 2.3.1 | MIT | through pytest |
| jmespath | 1.1.0 | MIT | |
| jsonschema | 4.26.0 | MIT | |
| jsonschema-specifications | 2025.9.1 | MIT | |
| lxml | 6.1.3 | BSD-3-Clause | reads XHTML for the Markdown conversion; its compiled part bundles zlib (Zlib), libiconv (LGPL-2.1), libxml2, libxslt and libexslt (MIT) |
| moto | 5.2.4 | Apache-2.0 | tests: answers the S3 test's requests inside the test process |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | |
| opentelemetry-api | 1.45.1 | Apache-2.0 | each command run's record and live view (wpt-telemetry) |
| opentelemetry-exporter-http-transport | 0.66b1 | Apache-2.0 | through opentelemetry-exporter-otlp-proto-http |
| opentelemetry-exporter-otlp-common | 0.66b1 | Apache-2.0 | through opentelemetry-exporter-otlp-proto-http |
| opentelemetry-exporter-otlp-proto-common | 1.45.1 | Apache-2.0 | through opentelemetry-exporter-otlp-proto-http |
| opentelemetry-exporter-otlp-proto-http | 1.45.1 | Apache-2.0 | sends the live view to an OpenTelemetry backend over OTLP/HTTP |
| opentelemetry-proto | 1.45.1 | Apache-2.0 | through opentelemetry-exporter-otlp-proto-http |
| opentelemetry-sdk | 1.45.1 | Apache-2.0 | |
| opentelemetry-semantic-conventions | 0.66b1 | Apache-2.0 | through opentelemetry-sdk |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | through pytest |
| pluggy | 1.6.0 | MIT | through pytest |
| protego | 0.7.0 | BSD-3-Clause | reads robots.txt files |
| protobuf | 7.36.2 | BSD-3-Clause | through opentelemetry-proto; its metadata says "3-Clause BSD License" |
| py-partiql-parser | 0.6.3 | MIT | tests, through moto |
| pyarrow | 26.0.0 | Apache-2.0 | |
| pycparser | 3.11 | BSD-3-Clause | tests, through moto (cffi) |
| pygments | 2.21.0 | BSD-2-Clause | through pytest |
| pytest | 9.1.1 | MIT | tests |
| pytest-xdist | 3.8.0 | MIT | tests: runs them in parallel (scripts/check.py) |
| python-dateutil | 2.9.0.post0 | Apache-2.0 AND BSD-3-Clause | newer code is also under Apache-2.0; all of it is under BSD-3-Clause |
| pyyaml | 6.0.3 | MIT | tests, through moto |
| referencing | 0.37.0 | MIT | |
| requests | 2.34.2 | Apache-2.0 | through opentelemetry-exporter-otlp-proto-http, and moto in tests |
| responses | 0.26.3 | Apache-2.0 | tests, through moto |
| rpds-py | 2026.9.1 | MIT | |
| s3transfer | 0.19.2 | Apache-2.0 | |
| safetensors | 0.8.0 | Apache-2.0 | |
| six | 1.17.0 | MIT | |
| typing-extensions | 4.16.0 | PSF-2.0 | through opentelemetry; PyTorch needs it too |
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

### CUDA build, Linux only

"NVIDIA proprietary" means `LicenseRef-NVIDIA-Proprietary` or NVIDIA's
license agreements (the CUDA Toolkit end user license agreement, the license
agreement for NVIDIA software development kits).

| Package | Version | License | Note |
| --- | --- | --- | --- |
| apache-tvm-ffi | 0.1.9 | Apache-2.0 | through tilelang; held below 0.1.10, whose later versions break tilelang 0.1.8 on Python 3.14 |
| cloudpickle | 3.1.2 | BSD-3-Clause | through tilelang |
| cuda-bindings | 13.4.3 | Apache-2.0 | Python below 3.15 |
| cuda-pathfinder | 1.8.3 | Apache-2.0 | Python below 3.15 |
| cuda-toolkit | 13.0.3.0 | none stated | holds only metadata listing the NVIDIA packages, no code |
| ml-dtypes | 0.6.0 | Apache-2.0 | through tilelang |
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
| psutil | 7.2.2 | BSD-3-Clause | through tilelang |
| tilelang | 0.1.8 | MIT | compiles the attention kernels, DeepSeek's among them ("Code copied from other projects", below); its compiled part bundles TVM (Apache-2.0) and the headers of CUTLASS (BSD-3-Clause) and Composable Kernel (MIT), whose license files the wheel does not carry |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | through tilelang |
| triton | 3.8.0 | MIT | Python below 3.15 |
| z3-solver | 4.15.4.0 | MIT | through tilelang |

### CUDA build, Windows only

| Package | Version | License | Note |
| --- | --- | --- | --- |
| triton-windows | 3.8.0.post29 | MIT | compiles the Triton attention kernels and flash-linear-attention's KDA kernel. It bundles, with no license file of theirs, NVIDIA's GPU compiler `ptxas.exe` (NVIDIA proprietary, CUDA Toolkit license agreement) and the TinyCC C compiler (`tcc.exe`, `libtcc.dll`; LGPL-2.1-or-later) |

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
| Docker Engine and its CLI | any | Apache-2.0 | builds the images and runs the containers of agentic post-training |
| Docker Desktop | any | proprietary (Docker Subscription Service Agreement) | runs Docker Engine on Windows and macOS (the way WPT uses Docker on Windows) |
| actions/checkout | v4 | MIT | CI |
| astral-sh/setup-uv | v6 | MIT | CI |

### The live view's backend (`deploy/otel-lgtm/`, not part of WPT)

`deploy/otel-lgtm/compose.yaml` runs Grafana's `grafana/otel-lgtm` image
next to WPT, unmodified, as a separate program that only receives data
([observability.md](observability.md)); WPT neither links nor
distributes it. Its license, from the image's own labels
(`org.opencontainers.image.licenses`), and those of what it bundles, at
the versions its Dockerfile pins for that release (GitHub's reading of
each project's `LICENSE` file at that tag):

| Part | Version | License |
| --- | --- | --- |
| grafana/otel-lgtm (the image's scripts and configuration) | 0.37.0 | Apache-2.0 |
| Grafana | v13.2.3 | AGPL-3.0 |
| Loki | v3.7.8 | AGPL-3.0 |
| Tempo | v3.1.0 | AGPL-3.0 |
| Pyroscope | v2.4.0 | AGPL-3.0 |
| Prometheus | v3.15.0 | Apache-2.0 |
| OpenTelemetry Collector | v0.162.0 | Apache-2.0 |
| OpenTelemetry eBPF Instrumentation | v0.14.0 | Apache-2.0 |
| Red Hat UBI 9 micro (the base image) | 9.8 | Red Hat's UBI terms (`com.redhat.license_terms`: https://www.redhat.com/en/about/red-hat-end-user-license-agreements#UBI) |

A new image version needs this table checked again.

### Container images of agentic tasks (`data/images/`)

The `grpo` and `token-efficiency` post-training steps run the model's
commands in containers built on each machine from the Dockerfiles in
`data/images/`
([post-training.md](post-training.md), "Images"). Locking an image
(`wpt-agent lock-image`) records every package installed in it, with its
version and its licenses, in `licenses.json` next to the image's
`lock.json`. No limit applies to which licenses an image may hold: the
record says what is there, and it is not a list of allowed licenses
(maintainer's decision, 2026-10-17). The images run the tasks on the
machine that builds them, and WPT does not distribute them. What a tool
result shows of an image's files is context the model reads during an
attempt; the loss covers only the tokens the model wrote.

A package's licenses come from its `/usr/share/doc/<package>/copyright`
file. When that file is machine-readable (Debian's copyright format, which
starts with a `Format:` field), the record lists the licenses its
`License:` fields name. When it is not, the record holds the file's full
text instead (maintainer's decision, 2026-10-17). A package with no such
file in the image has a record with its version only.

### Word lists of recipe names (`scripts/names/`)

`scripts/recipe_name.py` draws model recipe names from two lists taken
from Princeton WordNet 3.0, under its own license (the WordNet License,
kept in `scripts/names/LICENSE` as it asks):

- `adjectives.txt`: the adjectives that are one lowercase ASCII word and
  appear at least once in WordNet's tagged text (its sense counts), minus
  those WordNet marks as an ethnic slur, a disparagement or an obscenity:
  4,619 words.
- `animals.txt`: the nouns under `animal.n.01` that are one lowercase
  ASCII word, minus the same marks: 2,497 words.

### Code copied from other projects

- `packages/wpt-model/src/wpt_model/deepseek_kernels.py`: DeepSeek's
  sparse attention kernel for TileLang, copied unchanged from
  `inference/kernel.py` of DeepSeek-V4-Flash
  (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash, revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`), without its other kernels.
  MIT (Copyright (c) 2023 DeepSeek); the file keeps the license text, as
  the license asks.
- `packages/wpt-model/src/wpt_model/deepseek_index_kernel.py`: DeepSeek's
  FP8 lightning-indexer kernel for TileLang (`fp8_index_kernel` and
  `fp8_index`), copied unchanged from `inference/kernel.py` of
  DeepSeek-V3.2 (https://huggingface.co/deepseek-ai/DeepSeek-V3.2,
  revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`), without its other
  kernels; its compiler settings come from the file above, since the ones
  it names no longer exist in TileLang 0.1.8. MIT (Copyright (c) 2023
  DeepSeek); the file keeps the license text.
- `packages/wpt-model/src/wpt_model/triton_kernels.py` translates both
  kernels to Triton; it keeps DeepSeek's MIT notice for the translated
  parts. `tilelang_kernels.py` adds WPT's own backward kernels, written in
  the shape of DeepSeek's.

### Distribution

WPT does not distribute any of these packages: installing a build downloads
them on the user's machine, under their own terms. Distributing WPT
together with them (a container image, an installer) needs its own license
check first, above all for the NVIDIA, Intel and AMD runtimes.

### The web interface (`packages/wpt-web/ui/bun.lock`)

Bun installs these when `wpt-web` builds its interface. Only React,
React DOM and scheduler end up in the built files the browser loads; the
others build them. lightningcss's MPL-2.0 names the AGPL as a Secondary
License (its section 1.12).

| Package | Version | License | Note |
| --- | --- | --- | --- |
| react | 19.3.1 | MIT | in the built files |
| react-dom | 19.3.1 | MIT | in the built files |
| scheduler | 0.28.0 | MIT | in the built files, through react-dom |
| vite | 8.3.4 | MIT | builds the interface |
| rolldown | 1.2.14 | MIT | through vite |
| @rolldown/binding-* | 1.2.14 | MIT | one per platform (15), through rolldown |
| @rolldown/pluginutils | 1.0.1 | MIT | through vite |
| @oxc-project/types | 0.154.0 | MIT | through rolldown |
| lightningcss | 1.33.0 | MPL-2.0 | through vite |
| lightningcss-* | 1.33.0 | MPL-2.0 | one per platform (11), through lightningcss |
| detect-libc | 2.1.2 | Apache-2.0 | through lightningcss |
| postcss | 8.5.30 | MIT | through vite |
| nanoid | 3.3.20 | MIT | through postcss |
| picocolors | 1.1.1 | ISC | through postcss |
| source-map-js | 1.2.2 | BSD-3-Clause | through postcss |
| tinyglobby | 0.2.17 | MIT | through vite |
| fdir | 6.5.0 | MIT | through tinyglobby |
| picomatch | 4.0.7 | MIT | through tinyglobby |
| fsevents | 2.3.3 | MIT | macOS only, through vite |
| typescript | 7.0.2 | Apache-2.0 | checks the types |
| @typescript/typescript-* | 7.0.2 | Apache-2.0 | one per platform (20), through typescript |
| @types/react | 19.3.0 | MIT | types only |
| @types/react-dom | 19.3.0 | MIT | types only |
| csstype | 3.2.3 | MIT | types only, through @types/react |

## What the corpus accepts

**When in doubt, refuse.** A doubtful content is left out rather than used.

**The linking test.** A license is allowed only if, were the content code,
a program under the project's license (AGPL-3.0-or-later) could statically
link it and be distributed. This holds inside every family below: a license
that fails it is refused, whatever its family (maintainer's decision,
2026-10-21).

The allowed list below is closed: `packages/wpt-dataset/src/wpt_dataset/licenses.py`
holds it, and these are the licenses we recommend to contributors. Each one
lets its content end up inside AGPL-3.0-or-later weights: it is permissive,
its own text lets the content move to the GPLv3 or the AGPL (AGPL section 13
accepts GPLv3 works, each part keeping its license), or it asks only for
credit. The "What to check" column says where licenses of that kind hide a
trap; a person still reads the license of every path.

The evidence was gathered on 2026-10-15 from the FSF license list
(<https://www.gnu.org/licenses/license-list.html>), the license texts, the
Open Definition list (<https://opendefinition.org/licenses/>), the Blue Oak
Council list (<https://blueoakcouncil.org/list>), Creative Commons' pages on
compatibility and on AI training, and the publishers named below. A choice
that rests on the maintainer's decision alone, without an outside ruling,
says so in its row.

### Public domain and permissive

| License (SPDX) | Why it is allowed | What to check |
| --- | --- | --- |
| `LicenseRef-public-domain` | no rights left | public domain under the law of the country of first publication alone (Rules, "Public domain"), for every person whose words are in the text (author, translator, editor of notes); the entry's `text` gives the proof and its sources |
| `CC0-1.0`, `Unlicense` | public-domain dedications; FSF: compatible | a dedication by the right person: a site cannot dedicate what it does not own |
| `MIT`, `MIT-0`, `0BSD`, `BSD-1-Clause`, `BSD-2-Clause`, `BSD-2-Clause-Patent`, `BSD-3-Clause`, `ISC`, `X11`, `NCSA`, `PostgreSQL`, `UPL-1.0`, `HPND`, `W3C`, `curl` | FSF: GPL-compatible (BSD-2-Clause-Patent: Blue Oak Gold) | the copyright line of the source's own license file goes into `text` with the license |
| `Apache-2.0` | FSF: compatible with GPLv3 (not v2) | the source's `NOTICE` file, if any: its content follows the license in `text` |
| `Zlib`, `BSL-1.0`, `WTFPL` | FSF: compatible | Zlib: changed versions must be marked as changed |
| `Python-2.0`, `PSF-2.0` | FSF: compatible from Python 2.0.1 and 2.1.1 | Python 1.6b1 to 2.1 licenses are not compatible |
| `Unicode-3.0`, `Unicode-DFS-2016` | FSF: compatible with every GPL version | |
| `Artistic-2.0` | FSF: compatible through its section 4(c)(ii) | Artistic-1.0 (old Perl) is refused; Perl's "Artistic or GPL" is written as an expression (below) |
| `BlueOak-1.0.0` | Blue Oak Council's model permissive license | no FSF ruling; its text sets no condition the AGPL forbids |

### GPLv3 family

| License (SPDX) | Why it is allowed | What to check |
| --- | --- | --- |
| `GPL-3.0-only`, `GPL-3.0-or-later`, `LGPL-3.0-only`, `LGPL-3.0-or-later` | AGPL section 13 lets an AGPL work combine with GPLv3 works | a "-only" part can never move to a later AGPL version; the version comes from the notices in the files: a notice naming version 3 only gives "-only", and "any later version", or no notice at all, gives "-or-later" (section 14: a Program that specifies no version lets any version be chosen; maintainer's decision, 2026-10-16), the same for the AGPL |
| `AGPL-3.0-only`, `AGPL-3.0-or-later` | the weights' own license | |

### Copyleft that moves to the GPLv3 or the AGPL

| License (SPDX) | Why it is allowed | What to check |
| --- | --- | --- |
| `GPL-2.0-or-later` | "or later" lets the recipient choose the GPLv3 | the notice in each file: a notice naming version 2 only (GPL-2.0-only, the Linux kernel) is refused; "any later version", or no notice at all, gives "-or-later" (section 9: a Program that specifies no version lets any version be chosen; maintainer's decision, 2026-10-16) |
| `LGPL-2.1-only`, `LGPL-2.1-or-later` | LGPL 2.1 section 3: the ordinary GPL may apply instead, "if a newer version than version 2 ... has appeared, then you can specify that version" | a notice naming version 2.1 only gives "-only"; "any later version", or no notice at all, gives "-or-later" (section 14, as for the GPL; maintainer's decision, 2026-10-16) |
| `MPL-2.0` | section 1.12 names "the GNU Affero General Public License, Version 3.0, or any later versions" a Secondary License, and section 3.3 allows distribution under it | files marked "Incompatible With Secondary Licenses" (Exhibit B, SPDX `MPL-2.0-no-copyleft-exception`) are refused |
| `EUPL-1.2` | its appendix lists the AGPL; the European Commission: "you may distribute under the AGPL a larger derivative work integrating components covered by the EUPL" | EUPL-1.1 is refused (FSF: incompatible) |
| `CECILL-2.1` | FSF: "explicitly compatible with the GNU GPL" | CeCILL-B and CeCILL-C are refused |
| `EPL-2.0 OR GPL-2.0-or-later`, with or without `WITH Classpath-exception-2.0` | the GPL "secondary license" the file designates | EPL-2.0 alone is refused (FSF: incompatible without that designation) |

### Credit only: Creative Commons Attribution 4.0

| License (SPDX) | Why it is allowed | What to check |
| --- | --- | --- |
| `CC-BY-4.0` | FSF: "compatible with all versions of the GNU GPL"; credit may be given "in any reasonable manner", including a link (section 3(a)(2)); Creative Commons: for training, credit "could be a simple link to the source of the dataset" | NC and ND variants are refused; the credit names the creator, the copyright notice, the license, a link and whether the work was changed; a label that names no version (peS2o's "CCBY") is CC BY, written `CC-BY-4.0` with a note in `text` that no version is named (maintainer's decision, 2026-10-16, kept on 2026-10-21 on that decision alone: its first reason, that every version permits the use, fell when the versions before 4.0 failed the linking test) |

### Several licenses on one file

A license entry may be an SPDX expression of alternatives, such as
`EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0`. It passes when
one alternative is allowed, and WPT uses the content under that
alternative. `WITH` names an exception from a second closed list,
`EXCEPTIONS` in `licenses.py`, which holds only exceptions that add
permissions: `Classpath-exception-2.0` (Java), `LLVM-exception` (LLVM,
after Apache-2.0) and `GCC-exception-3.1` (GCC's runtime libraries, after
GPL-3.0-or-later). `AND` joins licenses that apply at once, such as
`MIT AND Apache-2.0` for a repository with both license files and no
statement that one may choose: the alternative passes when every license
in it is allowed, and its `text` holds every one of their texts
(maintainer's decision, 2026-10-16). When the repository states the choice
(its README or `Cargo.toml` says "MIT OR Apache-2.0", "MIT/Apache-2.0" or
"at your option"), the entry is `MIT OR Apache-2.0`. `AND` binds before `OR`, as in SPDX;
parentheses are not accepted.

### License texts

Every license entry carries `text`: the license as the source gives it, with
its copyright line, followed for Apache-2.0 by the source's own `NOTICE`
content; for `LicenseRef-public-domain`, the proof. MIT and BSD ask that
their notice go with copies, and Apache asks for its license and the
`NOTICE` content, so every model's `NOTICE` ends its list of sources with
each distinct text once.

### Still refused

The grey zone is CC-BY-SA-4.0 alone, refused until a lawyer confirms it
(maintainer's decision, 2026-10-16). Facts gathered for that review:

- Creative Commons declared GPLv3 a BY-SA-compatible license for BY-SA 4.0 on
  2015-10-08, one way only
  (<https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/>);
  the AGPL is not on that list.
- Creative Commons' guidance on training: "If AI models or outputs are based
  on ShareAlike content and they will be shared publicly, following the
  ShareAlike condition would require AI developers to use the same CC
  license as the original works"
  (<https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>),
  which contradicts AGPL weights.
- Wikipedia text is CC BY-SA 4.0 under the current Wikimedia Terms of Use.
- Common Pile v0.1 includes CC BY-SA content; WPT's rule is stricter.

Refused: any NC (no commercial use) or ND (no derivatives) clause,
GPL-2.0-only, CC-BY-SA-3.0, ODbL-1.0 (FSF: "incompatible with the GNU
GPL"), GFDL (FSF: not GPL-compatible; maintainer's decision, 2026-10-16:
plainly refused, not a grey zone), the Free Art
License, CDDL-1.0, EPL-1.0, EPL-2.0 alone, EUPL-1.1,
MPL-2.0-no-copyleft-exception, CeCILL-B, CeCILL-C, Artistic-1.0,
BSD-4-Clause, the old OpenSSL license, "responsible AI" and model licenses
(RAIL, Llama), and anything without a license or with all rights reserved.

Creative Commons Attribution before 4.0 fails the linking test: the FSF
rules only on 4.0 ("compatible with all versions of the GNU GPL"), and
Creative Commons states that its licenses "are currently not compatible
with the major software licenses"
(<https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software>).
Refused: `CC-BY-1.0`, `CC-BY-2.0`, `CC-BY-2.5`, `CC-BY-2.5-AU`,
`CC-BY-3.0`, `CC-BY-3.0-AT`, `CC-BY-3.0-AU`, `CC-BY-3.0-DE`,
`CC-BY-3.0-IGO`, `CC-BY-3.0-NL`, `CC-BY-3.0-US` and
`LicenseRef-CC-BY-2.0-FR`; a file under one of them is left out
(maintainer's decision, 2026-10-21).

### Waiting for the linking test

The data and public-sector licenses below were allowed before the linking
test. None has a ruling on linking with GPL or AGPL code, so each one is a
doubt and stays refused until it is checked against the test, one by one
(maintainer's decision, 2026-10-21). The facts gathered for each are kept
for that check.

| License (SPDX) | Why it was allowed | What to check |
| --- | --- | --- |
| `PDDL-1.0` | public-domain dedication for databases; Open Definition: recommended | |
| `CDLA-Permissive-2.0` | section 3.1: no restriction on Results, which section 5.4 says include machine learning models | sharing the data itself needs the license text, which WPT never does (it holds pointers) |
| `CDLA-Sharing-1.0` | section 3.5: no obligation on Results (maintainer's decision) | Results may hold no more than a de minimis part of the data (section 1.11): measure memorization (`wpt-eval FOLDER memorization`) before publishing weights trained on it |
| `ODC-By-1.0` | Open Definition: recommended | it covers a database, not the documents in it: each document needs its own license |
| `OGL-UK-3.0` | the licence states it is "compatible with the Creative Commons Attribution License 4.0 and the Open Data Commons Attribution License" | it excludes personal data, third-party rights and logos; default credit: "Contains public sector information licensed under the Open Government Licence v3.0." |
| `OGL-Canada-2.0` | Open Definition: conformant | |
| `etalab-2.0` | Licence Ouverte 2.0, built to be compatible with CC BY (2.0 and later), the OGL and ODC-BY | credit: the source (at least the licensor's name) and the date of its last update |
| `DL-DE-BY-2.0`, `DL-DE-ZERO-2.0` | Open Definition: conformant; allows change, merging and any purpose | BY: the provider's name, "dl-de/by-2-0" with its link, the dataset's link; changes are marked as such |
| `NLOD-2.0` | its section 9 states compatibility with the OGL | |
| `LicenseRef-IODL-2.0` | Italian Open Data License 2.0, a variant of the UK OGL | not in SPDX: its text comes from the publisher |
| `LicenseRef-PDL-1.0` | Japan's Public Data License 1.0; the Japanese government states it is compatible with CC BY 4.0 | each dataset may add limits in an "Important note" section, often only in Japanese: read it |
| `LicenseRef-OPL-UK-3.0` | the UK Open Parliament Licence v3.0, modelled on `OGL-UK-3.0` with the same terms: free use with attribution (maintainer's decision, 2026-10-16) | it excludes third-party material, personal data and the Parliament's logos; not in SPDX: its text comes from the publisher; credit: "Contains Parliamentary information licensed under the Open Parliament Licence v3.0." |

## Rules

- A source can hold several licenses, each mapped to its paths; every path
  matches exactly one. Each entry carries its credit (`attribution`) and its
  text (`text`).
- Checking is per file, not per repository: a license can be applied by
  mistake, and public domain depends on the country.
- **Per record** in a `records` file (a dataset holding many documents,
  each with its own license field): the license entries covering the file
  list in `values` the exact license values they cover, and a record enters
  only when its value is listed. Any other value (vague ones such as
  "Various open data", ShareAlike, unknown) leaves the record out, so a
  value nobody reviewed never enters. Each kept record's credit fields go
  into `CREDITS/<source id>.jsonl` in every model folder trained on it,
  which `NOTICE` names. The values are the dataset's own labels: its
  publisher checked each document, not WPT, and a label can be wrong.
  When a label is known to be wrong for part of the file, the format
  entry's `keep` rules leave that part out by another field: each names a
  field and the values it may hold, and a record enters only when every
  rule holds: UK Hansard keeps the records of the UK Parliament, which its
  licence covers, and leaves out those of the devolved parliaments and
  London; regulations keeps the rules and proposed rules agencies issue,
  and leaves out the notices and supporting material that also hold
  companies' filings (maintainer's decision, 2026-10-16).
- **A git history** (`history` in a git source) starts at the commit from
  which the repository's root license file holds the license the source
  lists: earlier commits were published without a license or under
  another one. Its diffs show every file the source keeps, third-party
  folders included, so its license entry is the AND of every license in
  the source, with every credit (maintainer's decision, 2026-10-16). Commit
  authors' names and emails stay as git wrote them: the repository
  publishes them under its license. A path under `.git/history/` is
  counted only against license entries whose patterns start with
  `.git/history/`, so a tree entry written `**` means every file of the
  tree and does not match the history files (maintainer's
  decision, 2026-10-20); a pattern starting with `.git/history/` matches
  no file of the tree. The order file, `.git/history/order` (the commits
  newest first, merges marked), is part of the history and gets its
  license entry by the same rule, from the patterns that start with
  `.git/history/` (maintainer's decision, 2026-10-22).
- **Every file whose license allows it comes in.** Only binary files (the
  repository reads text: UTF-8, or the encoding a file states itself,
  maintainer's decision, 2026-10-23), license refusals and doubts leave a
  file out; configuration, build files, license files and edition pages are
  data like any other (maintainer's decision, 2026-10-16).
- **Third-party code** (a copied library, a dependency folder) inside a
  source is used when its license is allowed, as its own license entry:
  its license file, or else the license its package metadata declares
  (`package.json`, `Cargo.toml`, `composer.json`, `pyproject.toml`), with
  that license's official text in `text`. A folder that declares no
  license stays out (maintainer's decision, 2026-10-16).
- **A license notice without the license's text** (a `LICENSE` that only
  says, for example, "Licensed under the Apache License, Version 2.0" and
  links to it) counts like a license declared in package metadata: the
  entry's `text` holds the notice, then that license's official text
  (maintainer's decision, 2026-10-21).
- **A git submodule** is another repository with its own license. Its
  files enter at the commit the parent pins, with a license entry of
  their own when that license is on the allowed list (a parent under
  MIT and a submodule under Apache-2.0 give two entries); a submodule
  under another license, or one that cannot be fetched, is excluded
  (maintainer's decision, 2026-10-16).
- **Public domain** means public domain under the law of the country where
  the work was first published, and that law alone: a Russian book under
  Russian law, a French book under French law, whatever another country
  says (maintainer's decision, 2026-10-22). A text is in the public domain
  in one of two ways: every person whose words are in it has died and the
  terms below have ended, or its authors placed it in the public domain
  themselves (a dedication, below) (maintainer's decision, 2026-10-22).
  A term runs to the end of the calendar year in which it ends: life + 70
  years for a person who died in 1955 ends on 31 December 2025. How to
  decide, for every person whose words are in the text (author,
  translator, reviser, editor of notes):
  - **The country of first publication is known**: that country's term
    and its own rules decide: France (wartime extensions of about 6 to 8
    years for works published before 1948, more for an author "mort pour
    la France"), Spain (life + 80 for authors who died before 1987),
    Colombia (life + 80), Mexico (life + 100), Russia (4 more years for
    people who worked or fought in the Second World War). A country whose
    law counts a work's term from its publication rather than from a
    death counts it so, and a condition that law set, such as a renewal
    of the copyright it required, counts too: the edition's own statement
    that the text is in the public domain in that country is proof that
    it was met (maintainer's decisions, 2026-10-22).
  - **The country of first publication is unknown**: the work is free only
    when every country where it could first have appeared has ended its
    term. Everyone lived until 1925 or earlier: public domain in every
    country (the longest term, Mexico's life + 100 years, ended by 2026).
    Everyone lived until 1955 or earlier: public domain in every country
    whose term is life + 70 years or shorter, which covers the
    English-speaking countries and most of Europe; a country with a longer
    term (above) that it could have appeared in still needs its own check.
    Someone lived until 1956 or later: the country must be found.
  - **First published in two countries within 30 days**: its country is the
    one whose term is shorter (Berne Convention, article 5(4)), which must
    be named.
  - **A translation is two layers**: the original is settled by its own
    author (an author who lived until 1925 or earlier, or an anonymous or
    traditional work written before 1800, frees the original everywhere,
    whatever its country), and the translation by its translator and
    editors, with the countries where that translation could first have
    appeared.
  - **A dedication by the authors themselves** to the public domain (the
    Python Enhancement Proposals, whose authors placed them in the public
    domain) is `LicenseRef-public-domain`, with the dedication quoted in
    `text` (maintainer's decision, 2026-10-15).
  - **A translation project's own dedication** counts as its authors':
    the project states in its own words that the text is in the public
    domain (or under CC0), and nothing from the same holder contradicts it,
    such as a notice reserving rights, a conflicting page or a condition on
    copying or changing the text. The Berean Standard Bible stays refused
    for a conflicting page of its holder. A request about the name alone
    does not contradict it (the World English Bible: "if you CHANGE the
    actual text ... you not call the result the World English Bible"), nor
    does a copyright line beside a release of every right (the
    Robinson-Pierpont Greek New Testament: "This compilation is Copyright
    ... All rights to this text are released to everyone"). A label written
    by a distributor counts only when the distributor is the project, as
    eBible.org's owner is the World English Bible's editor (maintainer's
    decisions, 2026-10-16). Exception: Robert Adam Boyd's three editions
    on eBible.org (`ebible-engasvbt`, `ebible-englxxup`, `ebible-grctr`)
    are admitted on eBible.org's label alone, as no statement in Boyd's
    own words was found (maintainer's decision alone, 2026-10-21).
  - **A text revised by an unnamed committee or reviser** and published
    before 1926 is an anonymous work, in the public domain everywhere; the
    named people (the original translator) are still checked
    (maintainer's decision, 2026-10-16).
  - An unknown death year, an unnamed reviser or committee of a text
    published in 1926 or later, or facts that disagree make the work a
    doubt for the maintainer, never a guess. A fact that one more research
    pass cannot find leaves the source refused, with that fact as its
    `reason` (maintainer's decision, 2026-10-16).
- **Collections the maintainer chose**: GitHub repositories with 10,000
  stars or more, forks and archived ones included, whose license is
  allowed, whether GitHub recognizes the license file or the file itself
  is identified (maintainer's decisions, 2026-10-15 and 2026-10-16); the
  build removes exact duplicates with Common Pile's code.
- **A distributor's own pages** (a copyright page, a note on the file set)
  enter only when the distributor states their license: eBible.org states
  none for its own pages, so they stay out, while its signing key, not a
  written work, enters. A dataset card enters when its license is stated:
  OpenAssistant's under Apache-2.0, not Common Pile's, which state none
  (maintainer's decisions, 2026-10-16).
- **A distributor's date is not proof of the edition**: compare a passage
  with the edition it names; modern spelling, section headings or
  introductions mean a later edition, whose editors must be found.
  Exception: the Arabic Van Dyck Bible (`ebible-arb-vd`), whose vowel
  marks, absent from the 1867 printing, are by unnamed editors, is
  admitted because they change no word (maintainer's decision alone,
  2026-10-21).
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
(`decision`, with a reason for a refusal or an exceptional admission) and
its commit must be signed by an allowed key (see
[CONTRIBUTING.md](../CONTRIBUTING.md)). A source has exactly three states
(maintainer's decision, 2026-10-22): `under-review` (not yet decided, where
every new source starts; a reason is optional), `admitted` and `refused`;
only an admitted source enters a build. A source replaced because its
address moved is refused, its reason naming the new source's id. A
refused source stays in the repository as a reference: it is never
deleted (maintainer's decision, 2026-10-22). There is no automatic
filtering or tagging.

A license question the maintainer cannot settle alone goes into
[LEGAL-QUESTIONS.html](../LEGAL-QUESTIONS.html), the page for legal
review, in English and French: the current choice, the reasons and
evidence, and the question. Until a lawyer answers, the current choice
applies. An answer is written under its question with its date and who
gave it, and the rule it changes is written into this document
(maintainer's decision, 2026-10-16).

Tools may help reviewers by flagging, never by deciding:

- [ScanCode Toolkit](https://github.com/aboutcode-org/scancode-toolkit)
  detects license texts file by file, which helps on large repositories. It
  is optional and not part of the build.
- A small model trained on past human decisions may later flag doubtful
  cases. It never decides.

## No distillation from another company's model

No output of another company's language model enters the data or the
training, including "synthetic" datasets whose human origin is not proven.
This covers text a model produced: generated datasets (smoltalk, Tulu,
OpenCodeInstruct), machine transcripts (Whisper transcripts of videos),
OCR corrected by a model, and conversations marked as synthetic. It does
not cover work that people published under an allowed license, such as a
repository's code, even when its authors used an AI assistant: the license
of the person who published it decides, and repositories are not filtered
for signs of such tools (maintainer's decision, 2026-10-15).
