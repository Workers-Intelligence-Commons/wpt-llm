"""From a source description to its final tree of documents.

    upstream tree (in the store)          -> put there by the download step (download.py), which
                                             fetches it (fetch: an "empty" source has no files
                                             and downloads nothing; a "huggingface" source is a
                                             dataset pinned to a commit, see huggingface.py; a
                                             source with `addresses` downloads each into its own
                                             folder of the tree; an address whose robots.txt
                                             forbids it, or cannot be read, is skipped for the
                                             next mirror, see robots.py) and checks it against
                                             `checksum`
    erasure cuts, in the order listed     -> each checked by sha256_after
    patches, in file-name order           -> applied with `git apply`

Nothing here downloads a source: a source whose upstream tree is not in
the store is NotDownloaded. Every result is stored under the fingerprint
of what produced it, so a step that already ran is a lookup: nothing is
applied twice, and changing one patch rebuilds only that source.

The store names an upstream tree by its checksum: the ref
upstream/<checksum> points at its manifest, and upstream-files/<checksum>
at its file list (wpt_store.sessions), which says where each downloaded
file lies in its session's bundles. A tree downloaded before bundles has
no file list: its files are store objects.
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
from pathlib import Path
from urllib.parse import unquote, urlparse

from wpt_store import parse_file_list
from wpt_telemetry import Telemetry, set_ref
from wpt_utils.programs import program

from . import archives, encoding, huggingface, manifest, records, retry, robots
from .errors import (ChecksumMismatch, CutFailed, FetchFailed, LicenseMapError, NotDownloaded, NotText, PatchFailed,
                     UnsupportedEntry)
from .manifest import FILE, SYMLINK, Entry, Extractor, glob_match_any
from .repo import Patch, Source

VERSION = importlib.metadata.version("wpt-dataset")
USER_AGENT = f"wpt-dataset/{VERSION} (+https://github.com/Workers-Intelligence-Commons/wpt-llm)"
UPSTREAM, UPSTREAM_FILES = "upstream", "upstream-files"  # the kinds of the refs naming an upstream tree by its checksum


def fingerprint(inputs) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- upstream -----------------------------------------------------------------

def upstream_tree(source: Source, store) -> dict[str, Entry]:
    """The source's upstream tree as the store holds it, each downloaded
    file placed in its bundle. An empty source's is empty, and needs no
    download. NotDownloaded when the store does not hold it."""
    if source.data["type"] == "empty":
        if manifest.checksum({}) != source.data["checksum"]:
            raise ChecksumMismatch(f"{source.id}: expected {source.data['checksum']}, got {manifest.checksum({})}")
        return {}
    tree = stored_upstream(source.data["checksum"], store)
    if tree is None:
        raise NotDownloaded(f"{source.id}: its upstream files are not in {store}; `wpt-dataset download {source.id}` "
                            f"fetches them")
    return tree


def stored_upstream(checksum: str, store) -> dict[str, Entry] | None:
    """The upstream tree the store holds under a checksum, each downloaded
    file placed in its bundle, or None."""
    known = store.get_ref(UPSTREAM, checksum.removeprefix("sha256:"))
    if not known:
        return None
    return manifest.placed_tree(manifest.parse_manifest(store.get_bytes(known)), stored_files(checksum, store))


def stored_files(checksum: str, store) -> dict:
    """The file list (wpt_store.parse_file_list) of the upstream tree the
    store holds under a checksum; {} for one downloaded before bundles."""
    listed = store.get_ref(UPSTREAM_FILES, checksum.removeprefix("sha256:"))
    return parse_file_list(store.get_bytes(listed)) if listed else {}


UNREACHABLE = (OSError, urllib.error.URLError, subprocess.CalledProcessError)  # what a failed download raises


def _reason(error) -> str:
    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        return error.stderr.decode(errors="replace").strip()
    return str(error)


def exact_pattern(path: str) -> str:
    """An extract pattern matching only that path: each `*`, `?` and `[`
    written as a class of that one letter."""
    return manifest.WILDCARD.sub(lambda letter: f"[{letter[0]}]", path)


def fetch(data: dict, address: str, store, tmp: Path, again=False) -> dict[str, Entry]:
    """The upstream tree at one address, without checking its checksum:
    downloaded into tmp, each kept file's content put into store (what a
    download stores into: manifest.Extractor). An http-archive source's
    archive is read while it downloads (archives.py). again: a git source
    is read from the repositories tmp holds already, with no robots.txt,
    asking only for any object it needs and lacks. A source with
    `addresses` fetches each into its own folder of the one tree, and
    into its own folder of tmp. A download
    that fails raises FetchFailed, naming the address it was for."""
    extractor = Extractor(store, data.get("extract", {}))
    parts = [(p["folder"], p["address"]) for p in data["addresses"]] if "addresses" in data else [("", address)]
    for folder, part_address in parts:
        (tmp / folder).mkdir(exist_ok=True)
        try:
            _fetch_into(data, part_address, extractor.within(folder), tmp / folder, again)
        except UNREACHABLE as error:  # of several addresses, the one that failed is named here
            raise FetchFailed(f"{part_address}: {_reason(error)}") from error
    return extractor.finish()


def _fetch_into(data: dict, address: str, extractor: Extractor, tmp: Path, again: bool):
    """Add what one address holds to the extractor's tree (see fetch)."""
    kind, revision = data["type"], data["revision"]
    if kind != "empty" and not (again and kind == "git"):
        robots.check(address, kind, USER_AGENT)  # before anything is downloaded
    if kind == "http-file":
        name = data.get("extract", {}).get("name") or unquote(urlparse(address).path.rsplit("/", 1)[-1])
        if not name:
            raise OSError(f"{address}: no file name in the address; set extract.name")
        download = _download(address, tmp / "download")
        if (path := extractor.kept(name)) is not None:
            extractor.add_file(path, FILE, download)  # never read into memory: a file may hold gigabytes
    elif kind == "http-archive":
        archives.read(address, extractor, tmp, USER_AGENT)
    elif kind == "git":
        _read_git_source(data, address, tmp, extractor, again)
    elif kind == "huggingface":
        huggingface.read(address, revision, extractor, tmp, USER_AGENT)
    elif kind == "mercurial":
        hg = program("hg")

        def clone():
            shutil.rmtree(tmp / "repo", ignore_errors=True)  # a cut clone leaves a folder hg refuses
            subprocess.run([hg, "clone", "-q", "-U", "-r", revision, address, str(tmp / "repo")],
                           capture_output=True, check=True)

        retry.call(clone, address)
        manifest.read_mercurial(tmp / "repo", revision, extractor)
    elif kind == "empty":
        pass  # its patches hold every file
    else:
        raise ValueError(f"unknown source type {kind!r}")


def _download(address: str, target: Path) -> Path:
    def download():
        with archives.get(address, USER_AGENT) as response, open(target, "wb") as out:
            while block := response.read(1 << 20):
                out.write(block)

    retry.call(download, address)
    return target


def _read_git_source(data: dict, address: str, tmp: Path, extractor: Extractor, again=False):
    """The commit, its history when asked, and each kept submodule at the
    commit it pins, from the address `submodules` lists for it, recursively.
    Only the contents of kept files are downloaded. again: each repository
    is in tmp already (fetch)."""
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
        content = manifest.content(store, entry)
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
    were; a patch that names one fails, as git refuses it. A file the
    patches leave as it was keeps its entry, and so where it lies."""
    if not patches:
        return tree
    with tempfile.TemporaryDirectory(prefix="wpt-patch-") as tmp:
        git, env = temporary_git(tmp)
        loaded = load_index(git, tree, store)
        held = {path for path, _, _ in _index(git)}
        aside = {path: entry for path, entry in tree.items() if path not in held}
        for patch in patches:
            try:
                git("apply", "--cached", "--whitespace=nowarn", "-", input=patch.data)
            except subprocess.CalledProcessError as error:
                raise PatchFailed(f"{source_id}: {patch.name}: {error.stderr.decode(errors='replace').strip()}") from error
        records = _index(git)
        patched = {path: tree[path] for path, mode, object_id in records
                   if loaded.get(path) == object_id and tree[path].mode == mode}
        changed = [record for record in records if record[0] not in patched]
        with contextlib.closing(manifest._git_blobs(env["GIT_DIR"], [r[2] for r in changed])) as contents:
            patched |= {path: Entry(mode, store.put_bytes(content)) for (path, mode, _), content in zip(changed, contents)}
    patched = aside | dict(sorted(patched.items()))
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


def load_index(git, tree: dict[str, Entry], store) -> dict[str, str]:
    """Make the temporary index hold this tree minus the paths git refuses:
    git silently drops any .git/ path (such as a history's files), so a
    caller that needs those must handle them itself. Returns {path: git
    object id} of the tree's files."""
    git("read-tree", "--empty")
    paths = sorted(tree)
    if not paths:
        return {}
    objects = git("hash-object", "-w", "--no-filters", "--stdin-paths",
                  input="".join(f"{manifest.local_file(store, tree[p])}\n" for p in paths).encode()).decode().split()
    git("update-index", "-z", "--index-info",
        input=b"".join(f"{tree[p].mode} {o}\t{p}".encode() + b"\0" for p, o in zip(paths, objects)))
    return dict(zip(paths, objects))


# -- the final tree -------------------------------------------------------------

def tree_key(source: Source) -> str:
    """The fingerprint of a source's final tree and of the check that its
    files are text: its checksum, cuts, patches and formats (which decide
    each file's encoding), and this wpt-dataset's version."""
    return fingerprint({
        "kind": "source-tree",
        "wpt_dataset": VERSION,
        "checksum": source.data["checksum"],
        "formats": source.data["formats"],
        "cuts": source.data.get("cuts", []),
        "patches": [[p.name, hashlib.sha256(p.data).hexdigest()] for p in source.patches],
    })


def source_tree(source: Source, store, telemetry=None) -> tuple[dict[str, Entry], str]:
    """The source's final tree and the sha256 of its manifest object, each
    file its patches left as it was placed where the download put it.
    NotDownloaded when the store does not hold its upstream files. With a
    telemetry, every ref it sets is recorded first (wpt_telemetry.set_ref)."""
    key = tree_key(source)
    known = store.get_ref("source-trees", key)
    if known:
        tree = manifest.parse_manifest(store.get_bytes(known))
        if source.data["type"] != "empty":
            tree = manifest.placed_tree(tree, stored_files(source.data["checksum"], store))
    else:
        tree = upstream_tree(source, store)
        tree = apply_cuts(source.id, tree, source.data.get("cuts", []), store)
        tree = apply_patches(source.id, tree, source.patches, store)
        if found := not_text(source, tree, store, first=True):
            raise NotText(f"{source.id}: {next(iter(found.values()))}; exclude it in extract")
        known = store.put_bytes(manifest.manifest_bytes(tree))
        set_ref(store, "source-trees", key, known, telemetry or Telemetry.off())
    check_license_map(source, tree)
    return tree, known


def not_text(source: Source, tree: dict[str, Entry], store, first=False) -> dict[str, str]:
    """{path: why} of the files of a final tree that must be text and are
    not, under the encoding each states (encoding.py), in the tree's order
    (symlinks, records files and message trees are read otherwise); with
    first, only the first one."""
    found = {}
    for path, entry in tree.items():
        if entry.mode != SYMLINK and not is_records(source, path) and not is_message_tree(source, path):
            try:
                encoding.decode(manifest.content(store, entry), encoding.entry_of(source, path), path)
            except NotText as error:
                found[path] = str(error)
                if first:
                    break
    return found


def is_records(source: Source, path: str) -> bool:
    """Whether a path is in the records format (see records.py), a file a
    records entry joins included."""
    return records_entry(source, path) is not None


def records_entry(source: Source, path: str) -> dict | None:
    """The records format entry covering a path, or None."""
    return next((e for e in manifest.format_entries(source.data["formats"], path) if e["format"] == "records"), None)


def is_message_tree(source: Source, path: str) -> bool:
    """Whether a path is a chat file read as a message tree, which is read
    like a records file (JSON Lines, compressed or not, or Parquet)."""
    return any(e["format"] == "chat" and "tree" in e and glob_match_any(path, e["paths"])
               for e in source.data["formats"])


def check_license_map(source: Source, tree: dict[str, Entry]):
    """Every path matches exactly one license entry, without `values`; a
    records file whose format names a `license` field instead matches one
    or more, each listing the license values of the records it covers, with
    no value listed twice. A records file whose format names none is like
    any other file: its records take its one entry."""
    for path in tree:
        found = manifest.license_entries(source.data["licenses"], path)
        names = ", ".join(e["license"] for e in found)
        entry = records_entry(source, path)
        if entry is None or "license" not in entry:
            if len(found) != 1:
                what = "no license entry" if not found else f"{len(found)} license entries ({names})"
                raise LicenseMapError(f"{source.id}: {path} matches {what}; every path must match exactly 1")
            if "values" in found[0]:
                what = "is not a records file" if entry is None else "is a records file with no `license` field"
                raise LicenseMapError(f"{source.id}: {path} {what}, but its license entry lists `values`")
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


def held_out_group(source: Source, path: str) -> str | None:
    """The held-out group of the source's document at path (its path in the
    source, as markdown.documents gives it), which the held-out hash reads
    instead of the document's id (build.is_held_out): where the source's
    `held_out_by` pattern is found in path, the pattern's groups joined by
    one space, or its whole match when it has none; otherwise None, and the
    id decides. Documents of any source with the same group are held out
    together: eBible's read-aloud sources hold their chapters out this way,
    by book and chapter in every translation at once."""
    rule = source.data.get("held_out_by")
    return records.key_of(rule["pattern"], path) if rule else None


def in_store(source: Source, store) -> bool:
    """Whether the store holds the source's upstream files, so its tree
    can be read."""
    return source.data["type"] == "empty" or store.get_ref(UPSTREAM, source.data["checksum"].removeprefix("sha256:")) \
        is not None
