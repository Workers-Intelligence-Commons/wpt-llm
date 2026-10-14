"""From a source description to its final tree of documents.

    download (address, else each mirror) -> upstream tree, checked against `checksum`
                                            (an "empty" source has no files and downloads nothing)
    erasure cuts, in the order listed     -> each checked by sha256_after
    patches, in file-name order           -> applied with `git apply`

Every result is stored under the fingerprint of what produced it, so a step
that already ran is a lookup: nothing is downloaded or applied twice, and
changing one patch rebuilds only that source.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import manifest
from .errors import ChecksumMismatch, CutFailed, FetchFailed, LicenseMapError, NotText, PatchFailed
from .manifest import FILE, SYMLINK, Entry, Extractor, glob_match
from .programs import program
from .repo import Source

VERSION = importlib.metadata.version("wpt-dataset")
USER_AGENT = f"wpt-dataset/{VERSION} (+https://github.com/inayayousfi/wpt-llm)"


def fingerprint(inputs) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- upstream -----------------------------------------------------------------

def upstream_tree(source: Source, store) -> dict[str, Entry]:
    expected = source.data["checksum"]
    known = store.get_ref("upstream", expected.removeprefix("sha256:"))
    if known:
        return manifest.parse_manifest(store.get_bytes(known))
    addresses = [source.data["address"]] + [m["address"] for m in source.data.get("mirrors", [])]
    unreachable, mismatched = [], []
    for address in addresses:
        try:
            tree = _fetch(source.data, address, store)
        except (OSError, urllib.error.URLError, subprocess.CalledProcessError) as error:
            detail = error.stderr.decode(errors="replace").strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else error
            unreachable.append(f"{address}: {detail}")
            continue
        actual = manifest.checksum(tree)
        if actual == expected:
            store.set_ref("upstream", expected.removeprefix("sha256:"), store.put_bytes(manifest.manifest_bytes(tree)))
            return tree
        mismatched.append(f"{address}: {actual}")
    if mismatched:
        raise ChecksumMismatch(f"{source.id}: expected {expected}, got " + "; ".join(mismatched)
                               + ("".join(f"; unreachable {u}" for u in unreachable)))
    raise FetchFailed(f"{source.id}: " + "; ".join(unreachable))


def compute_upstream(data: dict, address: str, store) -> dict[str, Entry]:
    """The upstream tree of a description, without checking its checksum
    (used to fill in `checksum` for a new source)."""
    return _fetch(data, address, store)


def _fetch(data: dict, address: str, store) -> dict[str, Entry]:
    extractor = Extractor(store, data.get("extract", {}))
    kind, revision = data["type"], data["revision"]
    with tempfile.TemporaryDirectory(prefix="wpt-fetch-") as tmp:
        tmp = Path(tmp)
        if kind == "http-file":
            name = data.get("extract", {}).get("name") or unquote(urlparse(address).path.rsplit("/", 1)[-1])
            if not name:
                raise OSError(f"{address}: no file name in the address; set extract.name")
            extractor.add(name, FILE, _download(address, tmp / "download").read_bytes())
        elif kind == "http-archive":
            manifest.read_archive(_download(address, tmp / "download"), extractor)
        elif kind == "git":
            _git_fetch(address, revision, tmp / "repo")
            manifest.read_git(tmp / "repo", revision, extractor)
        elif kind == "mercurial":
            subprocess.run([program("hg"), "clone", "-q", "-U", "-r", revision, address, str(tmp / "repo")],
                           capture_output=True, check=True)
            manifest.read_mercurial(tmp / "repo", revision, extractor)
        elif kind == "empty":
            pass  # its patches hold every file
        else:
            raise ValueError(f"unknown source type {kind!r}")
    return extractor.finish()


def _download(address: str, target: Path) -> Path:
    if urlparse(address).scheme not in ("http", "https"):
        raise OSError(f"{address}: http sources need an http:// or https:// address")
    request = urllib.request.Request(address, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    return target


def _git_fetch(address: str, revision: str, repo: Path):
    subprocess.run(["git", "init", "-q", "--bare", str(repo)], capture_output=True, check=True)
    shallow = subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", address, revision],
                             capture_output=True)
    if shallow.returncode != 0:  # some servers only serve branch and tag tips
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", address,
                        "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*"], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True, check=True)


# -- cuts and patches -----------------------------------------------------------

def apply_cuts(source_id: str, tree: dict[str, Entry], cuts: list[dict], store) -> dict[str, Entry]:
    """Each cut removes bytes [start, end) from the file as the cuts before
    it left it, and must give a file whose sha256 is sha256_after."""
    tree = dict(tree)
    for cut in cuts:
        path = cut["path"]
        entry = tree.get(path)
        if entry is None or entry.mode == SYMLINK:
            raise CutFailed(f"{source_id}: cut on {path}: no such file")
        content = store.get_bytes(entry.sha256)
        if not 0 <= cut["start"] < cut["end"] <= len(content):
            raise CutFailed(f"{source_id}: cut on {path}: range {cut['start']}-{cut['end']} outside {len(content)} bytes")
        after = content[:cut["start"]] + content[cut["end"]:]
        if hashlib.sha256(after).hexdigest() != cut["sha256_after"]:
            raise CutFailed(f"{source_id}: cut on {path}: the result does not match sha256_after")
        tree[path] = Entry(entry.mode, store.put_bytes(after))
    return tree


def apply_patches(source_id: str, tree: dict[str, Entry], patches: list[Path], store) -> dict[str, Entry]:
    """Apply git-format patches to the tree inside a temporary git index:
    no files are written out, so modes and symlinks behave the same on every
    operating system."""
    if not patches:
        return tree
    with tempfile.TemporaryDirectory(prefix="wpt-patch-") as tmp:
        git, env = temporary_git(tmp)
        load_index(git, tree, store)
        for patch in patches:
            try:
                git("apply", "--cached", "--whitespace=nowarn", str(patch.resolve()))
            except subprocess.CalledProcessError as error:
                raise PatchFailed(f"{source_id}: {patch.name}: {error.stderr.decode(errors='replace').strip()}") from error
        listing = [r for r in git("ls-files", "-s", "-z").split(b"\0") if r]
        records = []
        for record in listing:
            meta, path = record.split(b"\t", 1)
            mode, object_id, _stage = meta.decode().split()
            records.append((path.decode(), mode, object_id))
        with contextlib.closing(manifest._git_blobs(env["GIT_DIR"], [r[2] for r in records])) as contents:
            patched = {path: Entry(mode, store.put_bytes(content)) for (path, mode, _), content in zip(records, contents)}
    manifest.check_paths(patched)
    return patched


def temporary_git(tmp):
    """(git, env): a function running git on an empty bare repository and
    index inside tmp, isolated from the user's settings and repositories."""
    env = {**os.environ, "GIT_DIR": str(Path(tmp) / "git"), "GIT_INDEX_FILE": str(Path(tmp) / "index"),
           "GIT_CEILING_DIRECTORIES": str(tmp), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

    def git(*args, input=None):
        return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.quotepath=off", *args], input=input,
                              cwd=tmp, env=env, capture_output=True, check=True).stdout

    git("init", "-q", "--bare", env["GIT_DIR"])
    return git, env


def load_index(git, tree: dict[str, Entry], store):
    """Make the temporary index hold exactly this tree."""
    git("read-tree", "--empty")
    paths = sorted(tree)
    if not paths:
        return
    objects = git("hash-object", "-w", "--no-filters", "--stdin-paths",
                  input="".join(f"{store.path(tree[p].sha256)}\n" for p in paths).encode()).decode().split()
    git("update-index", "-z", "--index-info",
        input=b"".join(f"{tree[p].mode} {o}\t{p}".encode() + b"\0" for p, o in zip(paths, objects)))


# -- the final tree -------------------------------------------------------------

def source_tree(source: Source, store) -> tuple[dict[str, Entry], str]:
    """The source's final tree and the sha256 of its manifest object."""
    inputs = {
        "kind": "source-tree",
        "wpt_dataset": VERSION,
        "checksum": source.data["checksum"],
        "cuts": source.data.get("cuts", []),
        "patches": [[p.name, hashlib.sha256(p.read_bytes()).hexdigest()] for p in source.patches],
    }
    key = fingerprint(inputs)
    known = store.get_ref("source-trees", key)
    if known:
        tree = manifest.parse_manifest(store.get_bytes(known))
    else:
        tree = upstream_tree(source, store)
        tree = apply_cuts(source.id, tree, source.data.get("cuts", []), store)
        tree = apply_patches(source.id, tree, source.patches, store)
        for path, entry in tree.items():
            if entry.mode != SYMLINK:
                try:
                    store.get_bytes(entry.sha256).decode("utf-8")
                except UnicodeDecodeError as error:
                    raise NotText(f"{source.id}: {path} is not UTF-8 text; exclude it in extract") from error
        known = store.put_bytes(manifest.manifest_bytes(tree))
        store.set_ref("source-trees", key, known)
    check_license_map(source, tree)
    return tree, known


def check_license_map(source: Source, tree: dict[str, Entry]):
    for path in tree:
        found = [e["license"] for e in source.data["licenses"] if any(glob_match(path, g) for g in e["paths"])]
        if len(found) != 1:
            what = "no license entry" if not found else f"{len(found)} license entries ({', '.join(found)})"
            raise LicenseMapError(f"{source.id}: {path} matches {what}; every path must match exactly 1")


def tags_of(source: Source, path: str) -> frozenset[str]:
    return frozenset(t for e in source.data.get("tags", []) if any(glob_match(path, g) for g in e["paths"])
                     for t in e["tags"])
