"""One git source per GitHub repository with at least `stars` stars under a license key that can map to an allowed license.

Its candidates are GitHub's search for repositories, forks and archived
ones included, under each of KEYS (the license GitHub detects), as it answers
on the `listed` date. Each draft reads the repository at the commit its
default branch held on that date, by these rules:

- every UTF-8 text file comes in; only binary files are excluded;
- the root license file gives the main license entry;
- third-party code (a folder with its own license file, a folder whose
  package metadata declares a license, a submodule) is its own license
  entry when its license is allowed; a third-party folder declaring
  nothing, or under a license that is not allowed, is excluded; a license
  file no rule identifies is listed in the notes for a person to read (the
  folder stays excluded until then);
- the history entry is the AND of every license in the source;
- tags: "code" only.

A GPL-family license (GPL, LGPL, AGPL, any version) counts as "-or-later"
unless the license file, the package metadata or the repository's files
state "only" for it (ONLY): an explicit "only" wins over every other
answer, a review included, and a version stated nowhere gives "-or-later".

A license file is identified by its review in the reviewed license texts
(Repo.reviewed_license_texts, licenses.reviewed), else by the license key
GitHub gives the repository (its root only), else by its text matched
against the SPDX license list's official texts (Matcher). A repository
whose README may refuse AI training, whose root license nothing
identifies, or whose facts cannot be read needs a person.

The listing, each repository's commit and README statements, and the
SPDX texts (release `spdx` of spdx/license-list-data) are site facts. The
GitHub API is asked through context.fetch, with the token GITHUB_TOKEN
(the environment, else .env) when it is set: without one, GitHub answers
only a few requests an hour. The repositories are read with git, after
their robots.txt.
"""

import difflib
import functools
import glob as globmod
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib
import urllib.error
from urllib.parse import quote, urlencode

from wpt_store import read_env_file

from wpt_dataset import licenses, retry, robots
from wpt_dataset.errors import CrawlForbidden, RuleViolation
from wpt_dataset.licenses import ALLOWED, check_license
from wpt_dataset.repo import DRAFT_CHECKSUM, source_id
from wpt_dataset.source_drafter import Candidate, Drafted, NeedsPerson, Skipped
from wpt_dataset.sources import USER_AGENT

# stars: the search's minimum; listed: the date (YYYY-MM-DD) of the listing and of each repository's commit,
# "" for the run's date; spdx: the release of spdx/license-list-data whose texts the drafts read
SETTINGS = {"stars": 10000, "listed": "", "spdx": "v3.29.0"}

API = "https://api.github.com"
SPDX_TEXTS = "https://raw.githubusercontent.com/spdx/license-list-data/{release}/text/{id}.txt"
SEARCH_PAUSE = 2.5  # seconds between two search requests, below GitHub's limit for them
# GitHub's license keys searched: those that can map to an allowed license
KEYS = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "0bsd", "unlicense", "mpl-2.0", "gpl-3.0",
        "agpl-3.0", "lgpl-3.0", "lgpl-2.1", "gpl-2.0", "cc0-1.0", "bsl-1.0", "zlib", "artistic-2.0", "upl-1.0",
        "wtfpl", "postgresql", "ncsa", "eupl-1.2", "cecill-2.1", "bsd-2-clause-patent", "mit-0"]
NO_AI = re.compile(r"(not|no|never|prohibit\w*|forbid\w*|disallow\w*)\W+(\w+\W+){0,6}(AI|machine learning|LLM|train\w*)",
                   re.I)
# the official texts a license file is matched with (Matcher), and that a notice or a declaration gives
MATCHED = """0BSD AGPL-3.0-only AGPL-3.0-or-later Apache-2.0 Artistic-2.0 BlueOak-1.0.0 BSD-1-Clause BSD-2-Clause
BSD-2-Clause-Patent BSD-3-Clause BSL-1.0 CC0-1.0 CC-BY-1.0 CC-BY-2.0 CC-BY-2.5 CC-BY-2.5-AU CC-BY-3.0 CC-BY-3.0-AT
CC-BY-3.0-AU CC-BY-3.0-DE CC-BY-3.0-IGO CC-BY-3.0-NL CC-BY-3.0-US CC-BY-4.0 CC-BY-NC-1.0 CC-BY-NC-2.0 CC-BY-NC-2.5
CC-BY-NC-3.0 CC-BY-NC-3.0-DE CC-BY-NC-3.0-IGO CC-BY-NC-4.0 CC-BY-NC-ND-1.0 CC-BY-NC-ND-2.0 CC-BY-NC-ND-2.5
CC-BY-NC-ND-3.0 CC-BY-NC-ND-3.0-DE CC-BY-NC-ND-3.0-IGO CC-BY-NC-ND-4.0 CC-BY-NC-SA-1.0 CC-BY-NC-SA-2.0
CC-BY-NC-SA-2.0-DE CC-BY-NC-SA-2.0-FR CC-BY-NC-SA-2.0-UK CC-BY-NC-SA-2.5 CC-BY-NC-SA-3.0 CC-BY-NC-SA-3.0-DE
CC-BY-NC-SA-3.0-IGO CC-BY-NC-SA-4.0 CC-BY-ND-1.0 CC-BY-ND-2.0 CC-BY-ND-2.5 CC-BY-ND-3.0 CC-BY-ND-3.0-DE CC-BY-ND-4.0
CC-BY-SA-1.0 CC-BY-SA-2.0 CC-BY-SA-2.0-UK CC-BY-SA-2.1-JP CC-BY-SA-2.5 CC-BY-SA-3.0 CC-BY-SA-3.0-AT CC-BY-SA-3.0-DE
CC-BY-SA-3.0-IGO CC-BY-SA-4.0 CDLA-Permissive-2.0 CDLA-Sharing-1.0 CECILL-2.1 curl DL-DE-BY-2.0 DL-DE-ZERO-2.0
etalab-2.0 EUPL-1.2 GPL-2.0-or-later GPL-3.0-only GPL-3.0-or-later HPND ISC LGPL-2.1-only LGPL-2.1-or-later
LGPL-3.0-only LGPL-3.0-or-later MIT MIT-0 MPL-2.0 NCSA NLOD-2.0 ODC-By-1.0 OGL-Canada-2.0 OGL-UK-3.0 PDDL-1.0
PostgreSQL PSF-2.0 Python-2.0 Unicode-3.0 Unicode-DFS-2016 Unlicense UPL-1.0 W3C WTFPL X11 Zlib""".split()

SPDX = {"mit": "MIT", "apache-2.0": "Apache-2.0", "bsd-3-clause": "BSD-3-Clause", "bsd-2-clause": "BSD-2-Clause",
        "isc": "ISC", "0bsd": "0BSD", "unlicense": "Unlicense", "mpl-2.0": "MPL-2.0", "cc0-1.0": "CC0-1.0",
        "bsl-1.0": "BSL-1.0", "zlib": "Zlib", "artistic-2.0": "Artistic-2.0", "upl-1.0": "UPL-1.0", "wtfpl": "WTFPL",
        "postgresql": "PostgreSQL", "ncsa": "NCSA", "eupl-1.2": "EUPL-1.2", "cecill-2.1": "CECILL-2.1",
        "bsd-2-clause-patent": "BSD-2-Clause-Patent", "mit-0": "MIT-0", "cc-by-4.0": "CC-BY-4.0"}
FAMILY = {"gpl-3.0": "GPL-3.0", "agpl-3.0": "AGPL-3.0", "lgpl-3.0": "LGPL-3.0", "lgpl-2.1": "LGPL-2.1", "gpl-2.0": "GPL-2.0"}
LATER = re.compile(r"any later version|or[ -]later|\b(?:A|L)?GPL-?v?[23](?:\.[01])?\+", re.I)
ONLY = re.compile(r"\b(?:A|L)?GPL-[23]\.[01]-only\b|version [23](?:\.[01])? only|"
                  r"SPDX-License-Identifier:\s*(?:A|L)?GPL-[23]\.[01]\s*$|"
                  r"Free Software Foundation,? version [23](?:\.[01])?(?![^\n]{0,60}later)", re.I | re.M)
# a GPL-family term of a license expression: its family (GPL-3.0, LGPL-2.1, AGPL-3.0...) and its "-only" or
# "-or-later" suffix, when it has one
GPL_TERM = re.compile(r"(?<![\w.-])((?:A|L)?GPL-[23]\.[01])(-only|-or-later)?(?![\w.+-])")
LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|COPYRIGHT|UNLICENSE)([.\-_][^/]*)?$", re.I)
NOTICE_FILE = re.compile(r"^NOTICE(\.(txt|md))?$", re.I)
MIT_FILE = re.compile(r"^LICEN[CS]E[-._]MIT(\.(md|txt))?$", re.I)
APACHE_FILE = re.compile(r"^LICEN[CS]E[-._]APACHE(-2\.0)?(\.(md|txt))?$", re.I)
CHOICE = re.compile(r"MIT\s*(?:OR|/)\s*Apache-2\.0|Apache-2\.0\s*(?:OR|/)\s*MIT|at your option", re.I)
CONDITIONS = re.compile(r"commons clause|\bnon-?commercial\b(?! distribution and only if you received)"
                        r"(?!ly)|not be used for|machine learning|artificial intelligence", re.I)
# a folder holding other projects' code: what it holds is third-party unless a folder inside declares a license
VENDORED = re.compile(r"^(vendor|vendored|third[_-]?party|3rdparty|external|externals|deps|node_modules|bundled|"
                      r"extern|contrib)$", re.I)
METADATA = ("package.json", "composer.json", "Cargo.toml", "pyproject.toml")
BLOB_LIMIT = 1024 * 1024
# files the draft reads whole (license, notice, package metadata, README, Cargo.toml); of every other text file it
# reads only the first 60 lines, and only when they mention the GPL (version_of)
WHOLE = re.compile(r"^(LICEN[CS]E|COPYING|COPYRIGHT|UNLICENSE)([.\-_][^/]*)?$|^NOTICE(\.(txt|md))?$|"
                   r"^(package\.json|composer\.json|Cargo\.toml|pyproject\.toml)$|^README(\.\w+)?$", re.I)
GPL_HEAD = re.compile(r"General Public License|\bA?L?GPL")
BINARY = set("""png jpg jpeg gif bmp ico icns webp tif tiff psd xcf ai eps heic avif cur ani
ttf otf woff woff2 eot pfb pfm
zip gz tgz bz2 xz 7z rar jar war ear tar zst lz lzma lz4 cab deb rpm apk ipa dmg iso msi nupkg whl egg crx xpi vsix
exe dll so dylib a lib o obj ko class pyc pyo pyd wasm bin dat elf out node
mp3 mp4 m4a m4v wav ogg oga ogv opus flac aac wma avi mov mkv webm flv wmv mpg mpeg 3gp mid midi
pdf doc docx xls xlsx ppt pptx odt ods odp rtf epub mobi
sqlite sqlite3 db mdb accdb npy npz pkl pickle pt pth onnx h5 hdf5 tflite pb ckpt safetensors parquet feather arrow
gguf ggml keystore jks p12 pfx der crt cer gpg kdbx blend fbx glb 3ds max dds ktx tga exr hdr unitypackage swf fla""".split())
LANGUAGES = {
    "c": "c", "h": "c", "cc": "cpp", "cpp": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp", "hxx": "cpp", "ipp": "cpp",
    "cs": "csharp", "fs": "fsharp", "java": "java", "kt": "kotlin", "kts": "kotlin", "scala": "scala", "groovy": "groovy",
    "go": "go", "rs": "rust", "py": "python", "pyi": "python", "rb": "ruby", "php": "php", "pl": "perl", "pm": "perl",
    "lua": "lua", "js": "javascript", "mjs": "javascript", "cjs": "javascript", "jsx": "jsx", "ts": "typescript",
    "mts": "typescript", "cts": "typescript", "tsx": "tsx", "vue": "vue", "svelte": "svelte", "swift": "swift",
    "m": "objective-c", "mm": "objective-cpp", "dart": "dart", "ex": "elixir", "exs": "elixir", "erl": "erlang",
    "hrl": "erlang", "hs": "haskell", "ml": "ocaml", "mli": "ocaml", "clj": "clojure", "cljs": "clojure",
    "cljc": "clojure", "el": "elisp", "lisp": "lisp", "scm": "scheme", "rkt": "racket", "jl": "julia", "r": "r",
    "nim": "nim", "zig": "zig", "v": "v", "d": "d", "cr": "crystal", "sh": "bash", "bash": "bash", "zsh": "zsh",
    "fish": "fish", "ps1": "powershell", "psm1": "powershell", "sql": "sql", "css": "css", "scss": "scss",
    "sass": "sass", "less": "less", "html": "html", "htm": "html", "proto": "protobuf", "graphql": "graphql",
    "gql": "graphql", "tf": "hcl", "hcl": "hcl", "nix": "nix", "cmake": "cmake", "vim": "vim", "sol": "solidity",
    "asm": "asm", "s": "asm", "f90": "fortran", "f": "fortran", "pas": "pascal", "ada": "ada", "adb": "ada",
    "ads": "ada", "cu": "cuda", "cuh": "cuda", "glsl": "glsl", "hlsl": "hlsl", "wgsl": "wgsl", "metal": "metal",
    "gd": "gdscript", "elm": "elm", "purs": "purescript", "re": "reason", "res": "rescript", "coffee": "coffeescript",
    "tcl": "tcl", "awk": "awk", "bat": "batch", "cmd": "batch",
    # configuration, build and data files
    "json": "json", "jsonc": "json", "json5": "json5", "yaml": "yaml", "yml": "yaml", "toml": "toml", "xml": "xml",
    "xsd": "xml", "xsl": "xml", "xslt": "xml", "plist": "xml", "csproj": "xml", "vcxproj": "xml", "props": "xml",
    "targets": "xml", "resx": "xml", "xaml": "xml", "svg": "svg", "ini": "ini", "cfg": "ini", "conf": "ini",
    "properties": "properties", "gradle": "groovy", "mk": "makefile", "make": "makefile", "dockerfile": "dockerfile",
    "bzl": "starlark", "star": "starlark", "lock": "text", "diff": "diff", "patch": "diff", "csv": "csv", "tsv": "tsv",
    "tex": "latex", "sty": "latex", "bib": "bibtex", "rst": "rst", "adoc": "asciidoc", "org": "org",
    "ipynb": "json", "sln": "text", "pbxproj": "text", "gemspec": "ruby", "podspec": "ruby", "rake": "ruby",
    "erb": "erb", "ejs": "ejs", "hbs": "handlebars", "mustache": "mustache", "jinja": "jinja", "j2": "jinja",
    "twig": "twig", "liquid": "liquid", "pug": "pug", "astro": "astro", "mdx": "mdx", "styl": "stylus",
}
MARKDOWN = {"md", "markdown"}
NAMES = {"dockerfile": "dockerfile", "containerfile": "dockerfile", "makefile": "makefile", "gnumakefile": "makefile",
         "cmakelists.txt": "cmake", "gemfile": "ruby", "rakefile": "ruby", "podfile": "ruby", "vagrantfile": "ruby",
         "brewfile": "ruby", "fastfile": "ruby", "jenkinsfile": "groovy", "build": "starlark", "build.bazel": "starlark",
         "workspace": "starlark", "workspace.bazel": "starlark", "module.bazel": "starlark", "justfile": "just",
         "procfile": "text", "cargo.lock": "toml", "go.mod": "go-mod", "go.sum": "text", "meson.build": "meson"}

# an SSH-only address fails within 30 s instead of waiting for a prompt or a host that never answers
GIT_ENV = {**os.environ, "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o ConnectTimeout=30", "GIT_TERMINAL_PROMPT": "0"}
GIT_SECONDS = 3600  # a git call running longer counts as a failure, so a hang never stops a worker


def check(settings):
    if settings["listed"] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", settings["listed"]):
        raise ValueError(f"listed {settings['listed']!r}: a date, YYYY-MM-DD, or empty for the run's date")
    if settings["stars"] < 1:
        raise ValueError(f"stars {settings['stars']}: at least 1")
    if not re.fullmatch(r"v\d+\.\d+(\.\d+)?", settings["spdx"]):
        raise ValueError(f"spdx {settings['spdx']!r}: a release of spdx/license-list-data, such as v3.29.0")


# -- the site's facts ----------------------------------------------------------------------

def _item(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _address(name: str) -> str:
    return f"https://github.com/{name}"


def _api(context, path: str, raw=False) -> bytes:
    """What the GitHub API answers at path, with the token when one is set."""
    headers = {"Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or read_env_file(context.repo.root / ".env").get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return context.fetch(f"{API}/{path}", headers)


def _search(context, query: str) -> tuple[list, int]:
    """(the repositories the search gives, at most the 1,000 GitHub serves; its total count)."""
    found, page = [], 1
    while True:
        data = json.loads(_api(context, "search/repositories?" + urlencode(
            {"q": query, "per_page": 100, "page": page})))
        found += data["items"]
        if len(data["items"]) < 100 or page == 10:
            return found, data["total_count"]
        page += 1
        time.sleep(SEARCH_PAUSE)


def _listing(context) -> dict:
    """{full name: what the search says of it} of every repository with at
    least `stars` stars under one of KEYS; a range of stars holding more
    than the search serves is split in two until each holds less."""
    repos = {}
    for key in KEYS:
        ranges = [(context.settings["stars"], None)]
        while ranges:
            low, high = ranges.pop()
            stars = f"stars:>={low}" if high is None else f"stars:{low}..{high}"
            items, total = _search(context, f"{stars} license:{key}")
            time.sleep(SEARCH_PAUSE)
            if total > 1000:
                top = high or 500000
                middle = (low + top) // 2
                ranges += [(low, middle), (middle + 1, high)]
                continue
            for item in items:
                repos[item["full_name"]] = {"full_name": item["full_name"], "stars": item["stargazers_count"],
                                            "license": (item.get("license") or {}).get("spdx_id"), "key": key,
                                            "default_branch": item["default_branch"], "size_kb": item["size"],
                                            "language": item.get("language"), "description": item.get("description")}
        context.telemetry.log(f"license key {key}: {len(repos)} repositories listed")
    return repos


def _gathered(context, item: dict) -> dict:
    """A repository's facts: the commit its default branch holds, and what
    its README says that may refuse AI training (NO_AI); {"error": why}
    when GitHub knows no such branch."""
    name, branch = item["full_name"], item["default_branch"]
    try:
        commit = json.loads(_api(context, f"repos/{name}/branches/{quote(branch)}"))["commit"]["sha"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        return {"error": f"repos/{name}/branches/{branch}: not found"}
    try:
        readme = _api(context, f"repos/{name}/readme", raw=True).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        readme = ""  # no README
    return {"commit": commit, "readme_ai_statements": [m[0] for m in NO_AI.finditer(readme)][:10]}


def _listed(context) -> str:
    return context.settings["listed"] or context.today


def candidates(context):
    listing = context.fact(f"search/{context.settings['stars']}/{_listed(context)}", lambda: _listing(context))
    for name, item in sorted(listing.items()):
        yield Candidate(name, source_id("github", _item(name), _address(name)), item)


_MATCHERS = {}  # spdx release -> its Matcher, built once in each process
_REVIEWS = {}  # repository root -> its reviewed license texts, read once in each process


def draft(candidate, context):
    name = candidate.name
    facts = {**candidate.facts, **context.fact(f"repository/{_listed(context)}/{name}",
                                               lambda: _gathered(context, candidate.facts))}
    if "error" in facts:
        return NeedsPerson("facts: " + facts["error"], facts)
    release = context.settings["spdx"]
    if release not in _MATCHERS:
        _MATCHERS[release] = Matcher({spdx: context.fact(f"spdx/{release}/{spdx}", lambda spdx=spdx: context.fetch(
            SPDX_TEXTS.format(release=release, id=spdx)).decode("utf-8")) for spdx in MATCHED})
    root = str(context.repo.root)
    if root not in _REVIEWS:
        _REVIEWS[root] = context.repo.reviewed_license_texts()
    try:
        robots.check(_address(name), "git", USER_AGENT, context.robots)
    except CrawlForbidden as error:
        return Skipped(f"robots.txt: {error}")
    try:
        document, reason, notes = draft_document(facts, context.work, _MATCHERS[release], _REVIEWS[root], context)
    except (subprocess.CalledProcessError, UnicodeDecodeError, ValueError) as error:
        return NeedsPerson(f"draft failed: {error}", facts)
    if document is None:
        return NeedsPerson(reason, facts)
    return Drafted("github", _item(name), document, notes if any(notes.values()) else None)


# -- reading a repository ------------------------------------------------------------------

def git(repo, *args, text=True, check=True):
    return subprocess.run(["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo), *args],
                          capture_output=True, text=text, check=check, env=GIT_ENV, timeout=GIT_SECONDS).stdout


def fetch(address, commit, repo, history=False):
    """A bare repository holding `commit` with blobs up to BLOB_LIMIT (or, with
    history, every commit without contents)."""
    shutil.rmtree(repo, ignore_errors=True)
    subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True, capture_output=True)
    git(repo, "remote", "add", "origin", address)
    git(repo, "config", "remote.origin.promisor", "true")
    git(repo, "config", "remote.origin.partialclonefilter", "blob:none")
    args = ["--filter=blob:none"] if history else ["--depth", "1", f"--filter=blob:limit={BLOB_LIMIT}"]
    retry.call(lambda: git(repo, "fetch", "-q", "--no-tags", *args, "origin", commit, text=False), address)


def _stderr(error) -> str:
    return error.stderr.decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""


def tree(repo, commit):
    """{path: (mode, oid)} of a commit, submodules included (mode 160000)."""
    out = {}
    for entry in git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            mode, _, oid = meta.split()
            out[path] = (mode, oid)
    return out


def contents(repo, oids, keep):
    """{oid: (binary, kept)} for the blobs the repository holds (larger ones were
    not fetched): whether the blob is binary (is_binary), and keep(oid, content),
    what the draft needs of it later (None for nothing). git's output is read one
    blob at a time, so no more than one blob is held whole."""
    if not oids:
        return {}
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def feed():
        try:
            for o in oids:
                proc.stdin.write((o + "\n").encode())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    errors = []
    writer = threading.Thread(target=feed, daemon=True)
    reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
    writer.start()
    reader.start()
    out = {}
    while header := proc.stdout.readline():
        header = header.split()
        if header[1] == b"missing":
            continue
        size = int(header[2])
        content = proc.stdout.read(size)
        proc.stdout.read(1)
        oid = header[0].decode()
        out[oid] = (content_is_binary(content), keep(oid, content))
    writer.join()
    reader.join()
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, proc.args, stderr=b"".join(errors))
    return out


def extension(path):
    leaf = path.rsplit("/", 1)[-1]
    return leaf.rsplit(".", 1)[-1].lower() if "." in leaf.lstrip(".") else None


def is_binary(path, content):
    if content is None:  # too large to fetch here: its extension decides, and the checksum run reads it
        return extension(path) in BINARY
    return content_is_binary(content)


def content_is_binary(content):
    if b"\0" in content:
        return True
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def kept(content, what):
    """What the draft keeps of a blob: all of it, or (what == "head") the
    first 60 lines of its text when they mention the GPL, else nothing."""
    if what == "whole":
        return content
    if content_is_binary(content):
        return None
    head = "\n".join(text_of(content).splitlines()[:60])
    return head.encode("utf-8") if GPL_HEAD.search(head) else None


def review_of(reviews, content):
    """The review of a file's bytes in the reviewed license texts
    (licenses.reviewed), or None."""
    if content is None or not reviews:
        return None
    return licenses.reviewed(reviews, content)


def text_of(content):
    return content.decode("utf-8", errors="replace").removeprefix("﻿")


def normalized(text):
    lines = [l for l in text.splitlines() if not re.match(r"\s*(copyright|\(c\)|©|all rights reserved)", l, re.I)]
    return re.sub(r"[^a-z0-9 ]+", " ", " ".join(lines).lower()).split()


def cc_title(words):
    """The Creative Commons license a text names in its title (its first 30 words), as an SPDX id."""
    found = re.search(r"attribution( noncommercial)?( sharealike| noderivatives| noderivs)? (\d) (\d)", " ".join(words[:30]))
    if not found:
        return None
    kind = {" sharealike": "-SA", " noderivatives": "-ND", " noderivs": "-ND"}.get(found[2] or "", "")
    return f"CC-BY{'-NC' if found[1] else ''}{kind}-{found[3]}.{found[4]}"


class Matcher:
    """Identifies a license file's text among the official texts of the
    SPDX license list (MATCHED): one license whose text it matches almost
    word for word, clearly better than any other. official(spdx) is that
    text as the list gives it."""

    def __init__(self, texts: dict[str, str]):
        self.official_texts = texts
        self.texts = {spdx: normalized(text) for spdx, text in sorted(texts.items())}

    def official(self, spdx):
        return self.official_texts.get(spdx)

    def identify(self, text):
        words = normalized(text)
        if len(words) < 20:
            return None
        scores = []
        for spdx, ref in self.texts.items():
            ratio = difflib.SequenceMatcher(None, words, ref, autojunk=False).ratio()
            scores.append((ratio, spdx))
        scores.sort(reverse=True)
        best, spdx = scores[0]
        # texts identical between ids (GPL-3.0-only and -or-later) give the family; the notices decide the version
        same = [s for r, s in scores if r == best]
        if best < 0.93:
            return None
        if len(same) > 1:
            family = {re.sub(r"-(only|or-later)$", "", s) for s in same}
            return family.pop() + "?" if len(family) == 1 else None
        close = {s for r, s in scores if best - r < 0.02}
        if len(close) > 1 and all(s.startswith("CC-BY") for s in close):  # CC texts differ mostly in their title
            named = cc_title(words)
            return named if named in close else None
        if len(scores) > 1 and best - scores[1][0] < 0.02 and re.sub(r"-(only|or-later)$", "", scores[1][1]) != re.sub(r"-(only|or-later)$", "", spdx):
            return None
        if spdx.endswith(("-only", "-or-later")):
            return re.sub(r"-(only|or-later)$", "", spdx) + "?"
        return spdx


def declared(path, content):
    """The license expression a package metadata file declares, or None."""
    name = path.rsplit("/", 1)[-1]
    try:
        if name in ("package.json", "composer.json"):
            data = json.loads(text_of(content))
            value = data.get("license") or data.get("licenses")
            if isinstance(value, dict):
                value = value.get("type")
            if isinstance(value, list):
                value = " OR ".join(v.get("type") if isinstance(v, dict) else v for v in value)
        elif name == "Cargo.toml":
            value = tomllib.loads(text_of(content)).get("package", {}).get("license")
        else:
            value = tomllib.loads(text_of(content)).get("project", {}).get("license")
            if isinstance(value, dict):
                value = None  # {text = ...} or {file = ...}: the license file decides
    except (ValueError, AttributeError, TypeError):
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    value = re.sub(r"\s*/\s*", " OR ", value)  # Cargo's old "MIT/Apache-2.0"
    known = {a.lower(): a for a in ALLOWED} | {"gpl-3.0": "GPL-3.0", "agpl-3.0": "AGPL-3.0", "lgpl-3.0": "LGPL-3.0",
                                               "lgpl-2.1": "LGPL-2.1", "gpl-2.0": "GPL-2.0", "gpl-2.0+": "GPL-2.0-or-later",
                                               "gpl-3.0+": "GPL-3.0-or-later", "lgpl-2.1+": "LGPL-2.1-or-later",
                                               "apache 2.0": "Apache-2.0", "apache-2": "Apache-2.0", "apache2": "Apache-2.0",
                                               "apache license 2.0": "Apache-2.0", "bsd-3": "BSD-3-Clause",
                                               "bsd-2": "BSD-2-Clause", "unlicensed": None}
    terms = []
    for alternative in value.split(" OR "):
        parts = []
        for term in alternative.split(" AND "):
            mapped = known.get(term.strip().lower(), "?")
            if mapped is None:
                return None
            parts.append(mapped if mapped != "?" else "UNKNOWN:" + term.strip())
        terms.append(" AND ".join(parts))
    return " OR ".join(terms)


def allowed(expression):
    try:
        check_license(expression, "draft")
        return True
    except RuleViolation:
        return False


def version_of(family, files, scope_of, scope, readme=None):
    """(identifier, why) from the version notices of a scope's files (first 60
    lines) and its README: "-only" when one of them states "only", else
    "-or-later" (a notice, or none, allowing any later version). Identifier
    None for GPL-2.0 with an "only" notice, which is not allowed."""
    later, only = [], []
    for path, (content, *_) in sorted(files.items()):
        if scope_of.get(path) != scope or content is None:
            continue
        text = text_of(content)
        is_readme = path == readme
        head = text if is_readme else "\n".join(text.splitlines()[:60])
        if not re.search(r"General Public License|\bA?L?GPL", head):
            continue
        if found := ONLY.search(head):
            only.append((path, found[0].strip()))
        elif found := LATER.search(head):
            later.append((path, found[0].strip()))
    if only:
        path, words = only[0]
        why = f"{len(only)} files state one version only, such as {path} (\"{words}\")."
        return (None, f"GPL-2.0 with a version-only notice: {why}") if family == "GPL-2.0" else (f"{family}-only", why)
    if later:
        path, words = later[0]
        return f"{family}-or-later", f"{len(later)} files allow any later version, such as {path} (\"{words}\")."
    section, version = {"GPL-2.0": ("9", "2"), "LGPL-2.1": ("14", "2.1")}.get(family, ("14", "3"))
    return (f"{family}-or-later", f"No file states a version of the License; its section {section} lets any "
            f"version published by the Free Software Foundation be chosen, so it reads as version {version} or "
            "any later version.")


def credit_lines(text):
    lines = [line.strip() for line in text.splitlines()
             if re.match(r"\s*(?:Copyright|COPYRIGHT|(?:\([cC]\)|©)\s*[0-9])", line)
             and "Free Software Foundation" not in line
             and not re.search(r"\[(yyyy|year)|<year>|\{yyyy\}|copyright owner|copyright holder", line, re.I)]
    return list(dict.fromkeys(lines))[:5]


def metadata_author(path, content):
    try:
        if path.endswith(".json"):
            data = json.loads(text_of(content))
            author = data.get("author") or data.get("authors")
            name = data.get("name")
        else:
            data = tomllib.loads(text_of(content))
            section = data.get("package") or data.get("project") or {}
            author, name = section.get("authors"), section.get("name")
    except (ValueError, AttributeError, TypeError):
        return None, None
    if isinstance(author, list):
        author = ", ".join(a.get("name", "") if isinstance(a, dict) else str(a) for a in author)
    elif isinstance(author, dict):
        author = author.get("name")
    return (author or None), (name if isinstance(name, str) else None)


@functools.cache
def glob_match(glob):
    """PurePosixPath(path).full_match(glob) as one compiled pattern per glob (full_match compiles it again
    whenever the 512 patterns re caches are full, which takes hours on large repositories)."""
    return re.compile(globmod.translate(glob, recursive=True, include_hidden=True, seps="/")).match


def literal(path):
    """A path as a glob matching only itself: [, ], * and ? each in brackets."""
    return "".join(f"[{c}]" if c in "[]*?" else c for c in path)


def cover(paths, owner, scope):
    """The fewest globs that match exactly the paths whose owner is `scope`."""
    root = {}
    for path in paths:
        node = root
        for part in path.split("/"):
            node = node.setdefault(part, {})
        node[None] = path

    def owners(node):
        found = set()
        for key, child in node.items():
            found |= {owner[child]} if key is None else owners(child)
        return found

    globs = []

    def walk(node, prefix):
        found = owners(node)
        if found == {scope}:
            globs.append(f"{literal(prefix)}/**" if prefix else "**")
            return
        if scope not in found:
            return
        for key, child in sorted(node.items(), key=lambda i: (i[0] is None, i[0] or "")):
            if key is None:
                continue
            path = f"{prefix}/{key}" if prefix else key
            if None in child and len(child) == 1:
                if owner[child[None]] == scope:
                    globs.append(literal(path))
            else:
                walk(child, path)

    walk(root, "")
    return globs


def normalize_url(url, parent):
    url = url.strip()
    if url.startswith(("./", "../")):
        base = parent.rstrip("/").split("/")
        for part in url.split("/"):
            if part == "..":
                base.pop()
            elif part not in (".", ""):
                base.append(part)
        url = "/".join(base)
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    url = re.sub(r"^(ssh://git@|git://|http://)github\.com/", "https://github.com/", url)
    if "github.com/" in url:
        url = re.sub(r"\.git$", "", url.rstrip("/"))
    return url


class Repository:
    """Files of a repository and of its submodules, with their scopes."""

    def __init__(self, name, commit, work, robots_read):
        self.name, self.commit, self.work = name, commit, work
        self.robots_read = robots_read  # robots.check's cache: the context's
        self.files = {}         # path -> (what the draft reads of it, or None; mode; binary)
        self.submodules = {}    # path -> address
        self.blocked = {}       # submodule path -> why
        self.roots = {""}       # repository roots: "" and each fetched submodule
        self.count = 0

    def read(self, address, commit, prefix=""):
        repo = self.work / f"repo-{self.count}"
        self.count += 1
        fetch(address, commit, repo)
        entries = tree(repo, commit)
        wanted = {}  # oid -> "whole" or "head": what the draft reads of it (see kept)
        for path, (mode, oid) in entries.items():
            if mode != "160000":
                whole = mode == "120000" or WHOLE.match(path.rsplit("/", 1)[-1])  # a link's blob is its target
                wanted[oid] = "whole" if whole or wanted.get(oid) == "whole" else "head"
        blobs = contents(repo, list(wanted), lambda oid, content: kept(content, wanted[oid]))
        urls = {}
        if ".gitmodules" in entries:
            out = git(repo, "config", "--blob", f"{commit}:.gitmodules", "--get-regexp", r"^submodule\..*\.(path|url)$",
                      check=False)
            names = {}
            for line in out.splitlines():
                key, _, value = line.partition(" ")
                section, field = key.rsplit(".", 1)
                names.setdefault(section, {})[field] = value
            urls = {v["path"]: v["url"] for v in names.values() if "path" in v and "url" in v}
        for path, (mode, oid) in entries.items():
            full = prefix + path
            if mode == "160000":
                url = urls.get(path)
                if not url:
                    self.blocked[full] = "no address in .gitmodules"
                    continue
                sub = normalize_url(url, address)
                try:  # the check wpt-dataset makes before it fetches a submodule
                    robots.check(sub, "git", USER_AGENT, self.robots_read)
                except (CrawlForbidden, robots.RobotsUnreachable) as error:
                    self.blocked[full] = f"robots.txt: {error}"[:300]
                    continue
                try:
                    self.read(sub, oid, full + "/")
                    self.submodules[full] = sub
                    self.roots.add(full)
                except subprocess.CalledProcessError as error:
                    self.blocked[full] = f"cannot be fetched from {sub}: {_stderr(error).strip()[-160:]}"
                except subprocess.TimeoutExpired:
                    self.blocked[full] = f"cannot be fetched from {sub}: no answer within {GIT_SECONDS} s"
                continue
            if mode == "120000":  # a link: text, its target
                self.files[full] = (blobs.get(oid, (False, None))[1], mode, False)
            elif oid in blobs:
                binary, content = blobs[oid]
                self.files[full] = (content, mode, binary)
            else:  # too large to fetch here: its extension decides, and the checksum run reads it
                self.files[full] = (None, mode, is_binary(full, None))
        shutil.rmtree(repo, ignore_errors=True)


# -- the draft -------------------------------------------------------------------------------

def draft_document(facts, work, matcher, reviews, context):
    """(source.json without its decision, None, notes for a person), or
    (None, why a person must read the repository, {})."""
    name, commit, key = facts["full_name"], facts["commit"], facts["key"]
    address = _address(name)
    repo = Repository(name, commit, work, context.robots)
    repo.read(address, commit)
    files = repo.files
    if facts.get("readme_ai_statements"):
        readmes = [v[0] for p, v in files.items() if "/" not in p and re.match(r"README(\.\w+)?$", p, re.I)]
        if not any((review_of(reviews, c) or {}).get("kind") == "readme-ok" for c in readmes):
            return None, "the README may refuse AI training: " + " | ".join(facts["readme_ai_statements"]), {}
    text_files = {p: v for p, v in files.items() if v[1] == "120000" or not v[2]}
    binary = sorted(set(files) - set(text_files))

    # -- scopes: folders that declare their own license -------------------------------------------
    by_folder = {}  # folder -> its own text files, in text_files' order
    for p in text_files:
        by_folder.setdefault(p.rsplit("/", 1)[0] if "/" in p else "", []).append(p)
    scopes, manual, out = {}, {}, {}  # folder -> {"license", "text", "credit", "how"}; folder -> why; folder -> why
    for folder in sorted(set(by_folder) | repo.roots, key=lambda f: (f.count("/"), f)):
        here = {p.rsplit("/", 1)[-1]: p for p in by_folder.get(folder, [])}
        license_files = sorted(p for leaf, p in here.items() if LICENSE_FILE.match(leaf))
        metas = [here[m] for m in METADATA if m in here]
        if not license_files and not metas and folder not in repo.roots:
            continue
        result = resolve(folder, here, license_files, metas, text_files, matcher, reviews,
                         key if folder == "" else None, name, folder in repo.roots)
        if result is None:
            continue
        if "manual" in result:
            manual[folder] = result["manual"]
        elif "out" in result:
            out[folder] = result["out"]
        else:
            scopes[folder] = result

    if "" not in scopes:
        why = manual.get("", "no root license identified")
        return None, f"root license: {why}", {}

    # -- each file's owner: the nearest folder that declares a license -----------------------------
    def owner_of(path):
        parts = path.split("/")[:-1]
        for i in range(len(parts), -1, -1):
            folder = "/".join(parts[:i])
            if folder in scopes or folder in manual or folder in out:
                inner_vendored = any(VENDORED.match(p) for p in parts[i:])
                if folder in manual or folder in out:
                    return None
                if inner_vendored:
                    return None  # third-party code below its declared folder, declaring nothing itself
                return folder
        return None

    # a package folder declaring, in its metadata only, the license of the code around it is that code
    for folder in sorted(scopes, key=lambda f: -f.count("/")):
        info = scopes[folder]
        if folder and info["how"].startswith("declared") and not any(VENDORED.match(p) for p in folder.split("/")) \
                and folder not in repo.roots:
            parts = folder.split("/")
            around = next(("/".join(parts[:i]) for i in range(len(parts) - 1, -1, -1) if "/".join(parts[:i]) in scopes), None)
            if around is not None and scopes[around]["license"] == info["license"]:
                del scopes[folder]
    owner = {p: owner_of(p) for p in text_files}
    # a GPL-family license is "-or-later" unless the folder's own files or the package metadata state "only"
    for folder, info in list(scopes.items()):
        expression = info["license"]
        families = list(dict.fromkeys(m[1] for m in GPL_TERM.finditer(expression)))
        if not families:
            continue
        readme = next((p for p in text_files if owner.get(p) == folder and re.match(
            r"README(\.\w+)?$", p.rsplit("/", 1)[-1], re.I) and p.count("/") == (folder.count("/") + 1 if folder else 0)), None)
        found = {family: (f"{family}-only", f"{info['how']} states version {family.rsplit('-', 1)[1]} only.")
                 if family in info.get("only", ()) else version_of(family, text_files, owner, folder, readme)
                 for family in families}
        if GPL_TERM.fullmatch(expression) and found[families[0]][0] is None:  # GPL-2.0 with an "only" notice
            why = found[families[0]][1]
            if folder == "":
                return None, why, {}
            del scopes[folder]
            out[folder] = why
            continue
        spdx = GPL_TERM.sub(lambda m: found[m[1]][0] or f"{m[1]}-only", expression)
        info["license"] = spdx
        if spdx != expression:
            info["text"] += "\n\nVersion: " + " ".join(found[family][1] for family in families)
    for folder, info in list(scopes.items()):
        if not allowed(info["license"]):
            del scopes[folder]
            info["refused"] = True
    owner = {p: (o if o in scopes else None) for p, o in owner.items()}
    if not any(o == "" for o in owner.values()):
        return None, "no file of the repository's own code left", {}

    # -- exclusions, formats, licenses -------------------------------------------------------------
    excluded = set(binary) | {p for p, o in owner.items() if o is None}
    excluded |= {f"{p}" for p in repo.blocked}
    # paths a source cannot hold (docs/architecture.md): a backslash, or two that differ only by letter case
    folded = {}
    for p in files:
        if p not in excluded:
            folded.setdefault(p.casefold(), []).append(p)
    excluded |= {p for group in folded.values() if len(group) > 1 for p in group} | {p for p in files if "\\" in p}
    kept_paths = sorted(p for p in text_files if p not in excluded)
    exclude = cover(sorted(files) + sorted(repo.blocked), {**{p: ("x" if p in excluded else "k") for p in files},
                                                            **{p: "x" for p in repo.blocked}}, "x")
    formats = {}
    for path in kept_paths:
        if text_files[path][1] == "120000":
            continue
        leaf = path.rsplit("/", 1)[-1]
        ext = extension(path)
        if leaf.lower() in NAMES:
            language, glob = NAMES[leaf.lower()], f"**/{literal(leaf)}"
        elif ext in MARKDOWN:
            language, glob = None, f"**/*.{literal(path.rsplit('.', 1)[-1])}"
        elif ext in LANGUAGES:
            language, glob = LANGUAGES[ext], f"**/*.{literal(path.rsplit('.', 1)[-1])}"
        elif ext:
            language, glob = "text", f"**/*.{literal(path.rsplit('.', 1)[-1])}"
        else:
            language, glob = "text", f"**/{literal(leaf)}"
        formats.setdefault((language, glob), []).append(path)
    # a file named by NAMES whose extension also has an entry (CMakeLists.txt and *.txt) gives two matches:
    # those extension entries list their paths one by one
    named = {p for (lang, glob), ps in formats.items() if not glob.startswith("**/*.") for p in ps}
    entries, checks = [], []  # checks: each entry's paths listed one by one, and its real globs
    for (language, glob), paths in sorted(formats.items(), key=lambda i: (str(i[0][0]), i[0][1])):
        matching = glob.startswith("**/*.") and any(glob_match(glob)(p) for p in named)
        use = sorted(literal(p) for p in paths) if matching else [glob]
        checks.append((set(paths), []) if matching else (set(), [glob]))
        if language is None:
            entries.append({"format": "markdown", "paths": use})
        elif language == "text":
            entries.append({"format": "text", "paths": use})
        else:
            entries.append({"format": "code", "language": language, "paths": use})
    # globs of one format may still match a file of another (a hidden ".json" config named by NAMES): check
    for path in kept_paths:
        if text_files[path][1] == "120000":
            continue
        # a literal() glob matches only its own path: a set lookup, not one compiled pattern per path
        hits = [e for e, (exact, globs) in zip(entries, checks)
                if path in exact or any(glob_match(g)(path) for g in globs)]
        if len(hits) != 1:
            return None, f"format map: {path} matches {len(hits)} entries", {}

    license_entries = []
    all_paths = sorted(files)
    tree_owner = {p: (owner.get(p) if p in owner and p not in excluded else None) for p in all_paths}
    for folder, info in sorted(scopes.items()):
        paths = cover(all_paths, {p: (o if o is not None else "x") for p, o in tree_owner.items()}, folder)
        if not paths:
            continue
        where = address + (f"/tree/{commit}/{folder}" if folder else "")
        license_entries.append({"license": info["license"], "paths": paths, "text": info["text"],
                                "attribution": f"{info['credit']}, {info['license']} license ({info['how']}), {where}."})
    if not license_entries:
        return None, "no license entry left", {}

    # -- history ------------------------------------------------------------------------------------
    root = scopes[""]
    history = work / "history"
    fetch(address, commit, history, history=True)
    since = since_of(history, commit, root["file"], root["file_text"]) if root.get("file") else None
    if since is None:
        return None, f"no commit found where {root.get('file')} starts holding today's text", {}
    others = root.get("others", []) + ([(root["pair_other"], root["pair_other_text"])] if root.get("pair_other") else [])
    for other_file, other_text in others:
        other = since_of(history, commit, other_file, other_text)
        if other is None:
            return None, f"no commit found where {other_file} starts holding today's text", {}
        if subprocess.run(["git", "-C", str(history), "merge-base", "--is-ancestor", since, other]).returncode == 0:
            since = other
    shutil.rmtree(history, ignore_errors=True)
    used = list(license_entries)
    terms = list(dict.fromkeys(e["license"] for e in used))
    history_license = " AND ".join(terms) if len(terms) > 1 else terms[0]
    if " OR " in history_license and len(terms) > 1:
        # an OR inside an AND needs parentheses, which WPT does not read: keep each entry's first allowed alternative
        terms = [next(a for a in t.split(" OR ") if allowed(a)) for t in terms]
        history_license = " AND ".join(dict.fromkeys(terms))
    texts = list(dict.fromkeys(e["text"] for e in used))
    license_entries.insert(0, {"license": history_license, "paths": [".git/history/**"],
                               "text": "\n\n".join(texts),
                               "attribution": "Commit history of " + address + ": "
                               + "; ".join(e["attribution"].rstrip(".") for e in used) + "."})
    entries.insert(0, {"format": "code", "language": "patch", "paths": [".git/history/**"]})

    document = {
        "type": "git", "address": address, "revision": commit,
        "checksum": DRAFT_CHECKSUM,
        "extract": {"exclude": exclude} if exclude else {},
        "history": {"since": since},
        "licenses": license_entries,
        "formats": entries,
        "tags": [{"tags": ["code"], "paths": ["**"]}],
    }
    if not document["extract"]:
        del document["extract"]
    if repo.submodules:
        kept_subs = {p: a for p, a in repo.submodules.items()
                     if not any(glob_match(g)(p) for g in exclude)}
        if kept_subs:
            document["submodules"] = dict(sorted(kept_subs.items()))
    notes = {"manual": manual, "out": out, "blocked": repo.blocked,
             "refused_scopes": {f: i["license"] for f, i in scopes.items() if i.get("refused")}}
    return document, None, notes


def resolve(folder, here, license_files, metas, files, matcher, reviews, key, name, is_root):
    """A folder's license: {"license", "text", "credit", "how", ...}, {"manual": why}, or None (declares nothing)."""
    label = folder or "the repository root"
    content = lambda p: files[p][0]  # noqa: E731
    info = None
    reviewed = {p: review_of(reviews, content(p)) for p in license_files}
    license_files = [p for p in license_files if (reviewed[p] or {}).get("kind") != "not-a-license"]
    refused = [(p, reviewed[p]["why"]) for p in license_files if (reviewed[p] or {}).get("kind") == "out"]
    if refused:
        why = f"{refused[0][0]}: reviewed, {refused[0][1]}"
        return {"manual": why} if folder == "" else {"out": why}
    if license_files and all(reviewed[p] for p in license_files):
        terms, parts = [], []
        for p in license_files:
            o, text = reviewed[p], text_of(content(p))
            if o["license"] not in terms:
                terms.append(o["license"])
            if o["kind"] == "notice":
                officials = [matcher.official(t.split(" WITH ")[0]) for alt in o["license"].split(" OR ")
                             for t in alt.split(" AND ")]
                if any(x is None for x in officials):
                    return {"manual": f"{p} names {o['license']}; no official text for it"}
                text += (f"\n\n{p.rsplit('/', 1)[-1]} names the license \"{o['license']}\" without its text, so this is "
                         f"the license's official text (SPDX license list):\n\n" + "\n\n".join(dict.fromkeys(officials)))
            parts.append((p, text))
        if len(terms) > 1 and any(" OR " in t for t in terms):
            return {"manual": f"several license files in {label}, one of them a choice: {terms}"}
        stated = [p for leaf, p in here.items() if re.match(r"(README(\.\w+)?|Cargo\.toml)$", leaf, re.I)
                  and content(p) is not None and CHOICE.search(text_of(content(p)))] if len(terms) > 1 else []
        spdx = (" OR " if stated else " AND ").join(terms)
        text = parts[0][1] if len(parts) == 1 else "\n\n".join(f"{p}:\n\n{t}" for p, t in parts)
        if len(terms) > 1:
            text += ("\n\nVersion: " + (f"{stated[0]} states the choice between the licenses, so either applies."
                                          if stated else "No file states a choice between the licenses, so all apply."))
        info = {"license": spdx, "text": text, "how": " and ".join(p.rsplit("/", 1)[-1] for p in license_files),
                "file": license_files[0], "file_text": text_of(content(license_files[0])),
                "others": [(p, text_of(content(p))) for p in license_files[1:]]}
    elif license_files:
        missing = [p for p in license_files if content(p) is None]
        if missing:
            return {"manual": f"license file too large to read here: {missing}"}
        leaves = [p.rsplit("/", 1)[-1] for p in license_files]
        if len(license_files) == 2 and any(MIT_FILE.match(l) for l in leaves) and any(APACHE_FILE.match(l) for l in leaves):
            mit = next(p for p in license_files if MIT_FILE.match(p.rsplit("/", 1)[-1]))
            apache = next(p for p in license_files if APACHE_FILE.match(p.rsplit("/", 1)[-1]))
            stated = [p for leaf, p in here.items() if re.match(r"(README(\.\w+)?|Cargo\.toml)$", leaf, re.I)
                      and content(p) is not None and CHOICE.search(text_of(content(p)))]
            spdx = "MIT OR Apache-2.0" if stated else "MIT AND Apache-2.0"
            note = (f"{stated[0]} states the choice between the two licenses, so either applies." if stated
                    else "No file states a choice between the two licenses, so both apply.")
            text = f"{mit}:\n\n{text_of(content(mit))}\n\n{apache}:\n\n{text_of(content(apache))}\n\nVersion: {note}"
            info = {"license": spdx, "text": text, "how": f"{mit.rsplit('/', 1)[-1]} and {apache.rsplit('/', 1)[-1]}",
                    "file": apache, "file_text": text_of(content(apache)), "pair_other": mit,
                    "pair_other_text": text_of(content(mit))}
        else:
            if len(license_files) != 1:
                return {"manual": f"several license files in {label}: {license_files}"}
            path = license_files[0]
            text = text_of(content(path))
            spdx = SPDX.get(key) or FAMILY.get(key) if key else None  # a bare family: GPL_TERMS gives its version
            if not spdx:
                found = matcher.identify(text)
                if found is None:
                    return {"manual": f"{path}: the text matches no allowed license"}
                spdx = found.removesuffix("?")
            if CONDITIONS.search(text):
                return {"manual": f"{path} may add conditions to the license text"}
            info = {"license": spdx, "text": text, "how": path.rsplit("/", 1)[-1], "file": path, "file_text": text}
    if info is not None:
        notices = [p for leaf, p in here.items() if NOTICE_FILE.match(leaf) and content(p) is not None]
        if "Apache-2.0" in (info["license"] or ""):
            for notice in notices:
                info["text"] += f"\n\n{notice.rsplit('/', 1)[-1]}:\n\n{text_of(content(notice))}"
        credit = "; ".join(credit_lines(info["text"]))
        if not credit:
            for meta in [here[m] for m in METADATA if m in here and content(here[m]) is not None]:
                author, package = metadata_author(meta, content(meta))
                if author:
                    credit = author
                    break
        info["credit"] = credit or f"the authors of {name}" + (f" ({folder})" if folder else "") + " (the license file names no copyright line)"
        return info
    # no license file: package metadata
    for meta in metas:
        if content(meta) is None:
            continue
        expression = declared(meta, content(meta))
        if expression is None:
            continue
        if "UNKNOWN:" in expression or "?" in expression:
            o = review_of(reviews, content(meta)) or {}
            if o.get("kind") in ("license", "notice"):
                expression = o["license"]
            elif o.get("kind") == "out":
                why = f"{meta}: reviewed, {o['why']}"
                return {"manual": why} if folder == "" else {"out": why}
            else:
                return {"manual": f"{meta} declares \"{expression}\", which needs a person to map"}
        texts = []
        for term in dict.fromkeys(t.split(" WITH ")[0] for alt in expression.split(" OR ") for t in alt.split(" AND ")):
            # a GPL-family term naming no suffix reads "-or-later", whose official text is the family's
            official = matcher.official(GPL_TERM.sub(lambda m: m[0] + ("" if m[2] else "-or-later"), term))
            if official is None:
                return {"manual": f"{meta} declares {expression}; no official text for {term}"}
            texts.append(official)
        author, package = metadata_author(meta, content(meta))
        leaf = meta.rsplit("/", 1)[-1]
        text = (f"{leaf} declares the license \"{expression}\" and the folder holds no license file, so this is "
                f"the license's official text (SPDX license list):\n\n" + "\n\n".join(texts))
        credit = author or f"the authors of {package or folder or name}"
        stated = {m[1] for m in GPL_TERM.finditer(expression) if m[2] == "-only"}  # the metadata states "only"
        return {"license": expression, "text": text, "how": f"declared in {leaf}", "credit": credit, "only": stated}
    if folder == "":
        return {"manual": "no license declared at the root"}
    if is_root:
        return {"out": "a submodule declaring no license"}
    return None  # a folder of the code around it


def since_of(history, commit, license_path, current):
    since = None
    for c in git(history, "log", "--format=%H", commit, "--", license_path).split():
        try:
            text = git(history, "show", f"{c}:{license_path}", text=False).decode("utf-8", "replace")
        except subprocess.CalledProcessError:
            break
        if difflib.SequenceMatcher(None, normalized(text), normalized(current), autojunk=False).ratio() < 0.9:
            break
        since = c
    return since
