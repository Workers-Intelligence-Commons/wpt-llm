"""From a source description to its final tree of documents.

    download (address, else each mirror) -> upstream tree, checked against `checksum`
                                            (an "empty" source has no files and downloads nothing;
                                            a "huggingface" source is a dataset pinned to a commit,
                                            see huggingface.py;
                                            an address whose robots.txt forbids it, or cannot be
                                            read, is skipped for the next mirror, see robots.py)
    erasure cuts, in the order listed     -> each checked by sha256_after
    patches, in file-name order           -> applied with `git apply`

Every result is stored under the fingerprint of what produced it, so a step
that already ran is a lookup: nothing is downloaded or applied twice, and
changing one patch rebuilds only that source.
"""

import contextlib
import hashlib
import importlib.metadata
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from wpt_telemetry import Telemetry, set_ref
from wpt_utils.programs import program

from . import huggingface, manifest, retry, robots
from .errors import (ChecksumMismatch, CrawlForbidden, CutFailed, FetchFailed, LicenseMapError, NotText, PatchFailed,
                     UnsupportedEntry)
from .manifest import FILE, SYMLINK, Entry, Extractor, glob_match_any
from .repo import Patch, Source

VERSION = importlib.metadata.version("wpt-dataset")
USER_AGENT = f"wpt-dataset/{VERSION} (+https://github.com/Workers-Intelligence-Commons/wpt-llm)"


def fingerprint(inputs) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- upstream -----------------------------------------------------------------

def upstream_tree(source: Source, store, telemetry=None) -> dict[str, Entry]:
    """The source's upstream tree, fetched once and kept under the ref
    upstream/<checksum>; with a telemetry, the ref is recorded before it is
    set (wpt_telemetry.set_ref)."""
    expected = source.data["checksum"]
    known = store.get_ref("upstream", expected.removeprefix("sha256:"))
    if known:
        return manifest.parse_manifest(store.get_bytes(known))
    addresses = [source.data["address"]] + [m["address"] for m in source.data.get("mirrors", [])]
    unreachable, forbidden, mismatched = [], [], []
    for address in addresses:
        try:
            tree = _fetch(source.data, address, store)
        except CrawlForbidden as error:  # that site's wish only; another site may allow it
            forbidden.append(str(error))
            continue
        except UNREACHABLE as error:
            unreachable.append(f"{address}: {_reason(error)}")
            continue
        actual = manifest.checksum(tree)
        if actual == expected:
            set_ref(store, "upstream", expected.removeprefix("sha256:"), store.put_bytes(manifest.manifest_bytes(tree)),
                    telemetry or Telemetry.off())
            return tree
        mismatched.append(f"{address}: {actual}")
    others = "".join(f"; refused {f}" for f in forbidden) + "".join(f"; unreachable {u}" for u in unreachable)
    if mismatched:
        raise ChecksumMismatch(f"{source.id}: expected {expected}, got " + "; ".join(mismatched) + others)
    if forbidden:
        raise CrawlForbidden(f"{source.id}: every address failed{others}")
    raise FetchFailed(f"{source.id}: " + "; ".join(unreachable))


UNREACHABLE = (OSError, urllib.error.URLError, subprocess.CalledProcessError)  # what a failed download raises


def _reason(error) -> str:
    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        return error.stderr.decode(errors="replace").strip()
    return str(error)


def compute_upstream(data: dict, address: str, store) -> dict[str, Entry]:
    """The upstream tree of a description, without checking its checksum
    (used to fill in `checksum` for a new source). For a huggingface source
    it comes from the Hub's listing, and large files are not downloaded: the
    tree is only good for its manifest."""
    return _fetch(data, address, store, listing_only=True)


def fill_checksum(source: Source, store, telemetry=None) -> tuple[dict, list[str]]:
    """For a draft source (checksum all zeros): (its description with the
    checksum filled in, the paths it now excludes). One fetch of the main
    address gives the upstream tree; every file of the final tree that is
    not UTF-8 text, as source_tree finds them, is excluded by its exact
    path (exact_pattern), and the tree is read again from what that fetch
    downloaded, since an exclusion also changes the history files. The tree
    is kept under upstream/<checksum> as upstream_tree keeps it, so checking
    the source fetches nothing more. A huggingface source's checksum comes
    from the Hub's listing (compute_upstream) and nothing is excluded: its
    check downloads it, as before."""
    data = json.loads(json.dumps(source.data))
    address = data["address"]
    with tempfile.TemporaryDirectory(prefix="wpt-fetch-") as tmp:
        try:
            tree = _fetch(data, address, store, listing_only=True, tmp=Path(tmp))
        except UNREACHABLE as error:
            raise FetchFailed(f"{source.id}: {address}: {_reason(error)}") from error
        if data["type"] == "huggingface":
            data["checksum"] = manifest.checksum(tree)
            return data, []
        try:
            final = apply_patches(source.id, apply_cuts(source.id, tree, data.get("cuts", []), store), source.patches,
                                  store)
        except (CutFailed, PatchFailed):
            final = {}  # the check names that failure, before any text is read
        found = not_text(source, final, store)
        if made := [path for path in found if path not in tree]:
            raise NotText(f"{source.id}: {made[0]} is not UTF-8 text and comes from a patch, not upstream; "
                          f"extract cannot leave it out")
        if found:
            data.setdefault("extract", {}).setdefault("exclude", []).extend(map(exact_pattern, found))
            tree = _fetch(data, address, store, tmp=Path(tmp), again=True)
    data["checksum"] = manifest.checksum(tree)
    set_ref(store, "upstream", data["checksum"].removeprefix("sha256:"), store.put_bytes(manifest.manifest_bytes(tree)),
            telemetry or Telemetry.off())
    return data, found


def exact_pattern(path: str) -> str:
    """An extract pattern matching only that path: each `*`, `?` and `[`
    written as a class of that one letter."""
    return manifest.WILDCARD.sub(lambda letter: f"[{letter[0]}]", path)


def _fetch(data: dict, address: str, store, listing_only=False, tmp: Path | None = None, again=False) \
        -> dict[str, Entry]:
    """The upstream tree, downloaded into tmp (default: a temporary folder
    of its own). again: tmp holds what an earlier call downloaded, and it
    is read from there, with no robots.txt or download (a git source still
    asks for any object it needs and lacks)."""
    if tmp is None:
        with tempfile.TemporaryDirectory(prefix="wpt-fetch-") as folder:
            return _fetch(data, address, store, listing_only, Path(folder))
    extractor = Extractor(store, data.get("extract", {}))
    kind, revision = data["type"], data["revision"]
    if kind != "empty" and not again:
        robots.check(address, kind, USER_AGENT)  # before anything is downloaded
    if kind == "http-file":
        name = data.get("extract", {}).get("name") or unquote(urlparse(address).path.rsplit("/", 1)[-1])
        if not name:
            raise OSError(f"{address}: no file name in the address; set extract.name")
        download = tmp / "download" if again else _download(address, tmp / "download")
        extractor.add(name, FILE, download.read_bytes())
    elif kind == "http-archive":
        manifest.read_archive(tmp / "download" if again else _download(address, tmp / "download"), extractor)
    elif kind == "git":
        _read_git_source(data, address, tmp, extractor, again)
    elif kind == "huggingface":
        huggingface.read(address, revision, extractor, tmp, USER_AGENT, listing_only)
    elif kind == "mercurial":
        hg = program("hg")

        def clone():
            shutil.rmtree(tmp / "repo", ignore_errors=True)  # a cut clone leaves a folder hg refuses
            subprocess.run([hg, "clone", "-q", "-U", "-r", revision, address, str(tmp / "repo")],
                           capture_output=True, check=True)

        if not again:
            retry.call(clone, address)
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

    def download():
        with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)

    retry.call(download, address)
    return target


def _read_git_source(data: dict, address: str, tmp: Path, extractor: Extractor, again=False):
    """The commit, its history when asked, and each kept submodule at the
    commit it pins, from the address `submodules` lists for it, recursively.
    Only the contents of kept files are downloaded. again: each repository
    is in tmp already (_fetch)."""
    since = data["history"]["since"] if data.get("history") else None
    listed, found = data.get("submodules", {}), set()
    pending = [("", address, data["revision"], since)]
    for number in itertools.count():
        if not pending:
            break
        prefix, address, revision, since = pending.pop(0)
        repo = tmp / f"repo-{number}"
        if not again:
            _git_fetch(address, revision, repo, full=since is not None)
        _git_fetch_blobs(address, repo, manifest.git_blobs_needed(repo, revision, extractor, since, prefix))
        submodules = manifest.read_git(repo, revision, extractor, prefix)
        if since:
            manifest.read_git_history(repo, since, revision, extractor)
        for path, commit in submodules.items():
            if path not in listed:
                raise UnsupportedEntry(f"{path}: a submodule with no address in `submodules`; list it or exclude it")
            if not again:
                robots.check(listed[path], "git", USER_AGENT)
            found.add(path)
            pending.append((path + "/", listed[path], commit, None))
    if unused := sorted(set(listed) - found):
        raise UnsupportedEntry(f"`submodules` lists {', '.join(unused)}, not a kept submodule of the revision")


def _git_fetch(address: str, revision: str, repo: Path, full=False):
    """The commit `revision` without file contents (git's partial fetch); with
    `full`, its whole history too. A server that does not offer partial
    fetches sends the contents as well."""
    subprocess.run(["git", "init", "-q", "--bare", str(repo)], capture_output=True, check=True)
    git = ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo)]
    for setting in (["remote", "add", "origin", address], ["config", "remote.origin.promisor", "true"],
                    ["config", "remote.origin.partialclonefilter", "blob:none"]):
        subprocess.run([*git, *setting], capture_output=True, check=True)
    depth = [] if full else ["--depth", "1"]

    def fetch(*what):
        retry.call(lambda: subprocess.run([*git, "fetch", "-q", "--no-tags", "--filter=blob:none", *what],
                                          capture_output=True, check=True), address)

    try:
        fetch(*depth, "origin", revision)
    except subprocess.CalledProcessError:  # some servers only serve branch and tag tips
        fetch("origin", "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*")
    subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True, check=True)


def _git_fetch_blobs(address: str, repo: Path, object_ids: list[str]):
    """The file contents a partial fetch left out, asked for by id in one request."""
    missing = manifest.missing_objects(repo, object_ids)
    if not missing:
        return
    command = ["git", "-c", f"http.userAgent={USER_AGENT}", "-C", str(repo), "fetch", "-q", "--no-tags",
               "--no-write-fetch-head", "--filter=blob:none", "--stdin", "origin"]
    wanted = "".join(object_id + "\n" for object_id in missing).encode()
    retry.call(lambda: subprocess.run(command, input=wanted, capture_output=True, check=True), address)


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


def apply_patches(source_id: str, tree: dict[str, Entry], patches: list[Patch], store) -> dict[str, Entry]:
    """Apply git-format patches to the tree inside a temporary git index:
    no files are written out, so modes and symlinks behave the same on every
    operating system. The paths git's index refuses (any .git/ path, such
    as a history's files) are set aside and come back exactly as they
    were; a patch that names one fails, as git refuses it (maintainer's
    decision, 2026-10-22)."""
    if not patches:
        return tree
    with tempfile.TemporaryDirectory(prefix="wpt-patch-") as tmp:
        git, env = temporary_git(tmp)
        load_index(git, tree, store)
        held = {path for path, _, _ in _index(git)}
        aside = {path: entry for path, entry in tree.items() if path not in held}
        for patch in patches:
            try:
                git("apply", "--cached", "--whitespace=nowarn", "-", input=patch.data)
            except subprocess.CalledProcessError as error:
                raise PatchFailed(f"{source_id}: {patch.name}: {error.stderr.decode(errors='replace').strip()}") from error
        records = _index(git)
        with contextlib.closing(manifest._git_blobs(env["GIT_DIR"], [r[2] for r in records])) as contents:
            patched = {path: Entry(mode, store.put_bytes(content)) for (path, mode, _), content in zip(records, contents)}
    patched = aside | patched
    manifest.check_paths(patched)
    return patched


def _index(git) -> list[tuple[str, str, str]]:
    """(path, mode, object id) of every entry of the temporary index."""
    records = []
    for record in filter(None, git("ls-files", "-s", "-z").split(b"\0")):
        meta, path = record.split(b"\t", 1)
        mode, object_id, _stage = meta.decode().split()
        records.append((path.decode(), mode, object_id))
    return records


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
    """Make the temporary index hold this tree minus the paths git refuses:
    git silently drops any .git/ path (such as a history's files), so a
    caller that needs those must handle them itself."""
    git("read-tree", "--empty")
    paths = sorted(tree)
    if not paths:
        return
    objects = git("hash-object", "-w", "--no-filters", "--stdin-paths",
                  input="".join(f"{store.path(tree[p].sha256)}\n" for p in paths).encode()).decode().split()
    git("update-index", "-z", "--index-info",
        input=b"".join(f"{tree[p].mode} {o}\t{p}".encode() + b"\0" for p, o in zip(paths, objects)))


# -- the final tree -------------------------------------------------------------

def tree_key(source: Source) -> str:
    """The fingerprint of a source's final tree: its checksum, cuts and
    patches, and this wpt-dataset's version."""
    return fingerprint({
        "kind": "source-tree",
        "wpt_dataset": VERSION,
        "checksum": source.data["checksum"],
        "cuts": source.data.get("cuts", []),
        "patches": [[p.name, hashlib.sha256(p.data).hexdigest()] for p in source.patches],
    })


def source_tree(source: Source, store, telemetry=None) -> tuple[dict[str, Entry], str]:
    """The source's final tree and the sha256 of its manifest object. With
    a telemetry, every ref it sets is recorded first (wpt_telemetry.set_ref)."""
    key = tree_key(source)
    known = store.get_ref("source-trees", key)
    if known:
        tree = manifest.parse_manifest(store.get_bytes(known))
    else:
        tree = upstream_tree(source, store, telemetry)
        tree = apply_cuts(source.id, tree, source.data.get("cuts", []), store)
        tree = apply_patches(source.id, tree, source.patches, store)
        if found := not_text(source, tree, store, first=True):
            raise NotText(f"{source.id}: {found[0]} is not UTF-8 text; exclude it in extract")
        known = store.put_bytes(manifest.manifest_bytes(tree))
        set_ref(store, "source-trees", key, known, telemetry or Telemetry.off())
    check_license_map(source, tree)
    return tree, known


def not_text(source: Source, tree: dict[str, Entry], store, first=False) -> list[str]:
    """The files of a final tree that must be UTF-8 text and are not, in
    the tree's order (symlinks, records files and message trees are read
    otherwise); with first, only the first one."""
    found = []
    for path, entry in tree.items():
        if entry.mode != SYMLINK and not is_records(source, path) and not is_message_tree(source, path):
            try:
                store.get_bytes(entry.sha256).decode("utf-8")
            except UnicodeDecodeError:
                found.append(path)
                if first:
                    break
    return found


def is_records(source: Source, path: str) -> bool:
    """Whether a path is in the records format (see records.py)."""
    return any(e["format"] == "records" and glob_match_any(path, e["paths"])
               for e in source.data["formats"])


def is_message_tree(source: Source, path: str) -> bool:
    """Whether a path is a chat file read as a message tree, which is read
    like a records file (JSON Lines, compressed or not, or Parquet)."""
    return any(e["format"] == "chat" and "tree" in e and glob_match_any(path, e["paths"])
               for e in source.data["formats"])


def check_license_map(source: Source, tree: dict[str, Entry]):
    """Every path matches exactly one license entry, without `values`; a
    records file instead matches one or more, each listing the license
    values of the records it covers, with no value listed twice."""
    for path in tree:
        found = manifest.license_entries(source.data["licenses"], path)
        names = ", ".join(e["license"] for e in found)
        if not is_records(source, path):
            if len(found) != 1:
                what = "no license entry" if not found else f"{len(found)} license entries ({names})"
                raise LicenseMapError(f"{source.id}: {path} matches {what}; every path must match exactly 1")
            if "values" in found[0]:
                raise LicenseMapError(f"{source.id}: {path} is not a records file, but its license entry lists `values`")
            continue
        if not found or any("values" not in e for e in found):
            raise LicenseMapError(f"{source.id}: {path} is a records file: every license entry covering it "
                                  f"lists the `values` it covers (found: {names or 'none'})")
        values = [v for e in found for v in e["values"]]
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise LicenseMapError(f"{source.id}: {path}: license values listed by several entries: {repeated}")


def tags_of(source: Source, path: str) -> frozenset[str]:
    return frozenset(t for e in source.data.get("tags", []) if glob_match_any(path, e["paths"])
                     for t in e["tags"])
