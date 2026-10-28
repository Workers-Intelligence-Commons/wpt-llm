"""From a source description to its final tree of documents.

    upstream tree (in the store)          -> put there by the download step (download.py), which
                                             fetches it through the source's protocol (protocols/);
                                             an "empty" source has no files and downloads nothing
    erasure cuts, in the order listed     -> every run of the passage each names removed, found by its
                                             keyed hashes (erasure.py)
    patches, in file-name order           -> applied with `git apply`

Nothing here downloads a source: a source whose upstream tree is not in
the store is NotDownloaded. Every result is stored under the fingerprint
of what produced it, so a step that already ran is a lookup: nothing is
applied twice, and changing one patch rebuilds only that source.

The store names a source's upstream tree by upstream_key, the fingerprint
of the source's id and of what its download reads (its type, which names
its protocol, its addresses, extract rules, history and submodules, and
the outside license pages its entries declare): the ref upstream/<key>
points at its manifest, an object named by its content, and
upstream-files/<key> at its file list (wpt_store.sessions), which says
where each downloaded file lies in its session's bundles. The license
pages downloaded with it are named by upstream-pages/<key>, a file list
whose paths are the pages' addresses (stored_pages, license_page). A tree
downloaded before bundles has no file list: its files are store objects. Which license entry covers each
file is the license check's answer (license_check.py).
"""

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wpt_store import parse_file_list
from wpt_telemetry import Telemetry, set_ref
from wpt_utils.parallel import in_order
from wpt_utils.programs import code

from . import checks, erasure, license_check, manifest, records, workers
from .errors import CutFailed, NotDownloaded, NotText, PatchFailed
from .manifest import SYMLINK, Entry
from .repo import Patch, Source

# the kinds of the refs naming an upstream tree and the license pages downloaded with it (upstream_key)
UPSTREAM, UPSTREAM_FILES, UPSTREAM_PAGES = "upstream", "upstream-files", "upstream-pages"
# what a download reads from a source's description, besides its id and its license pages: a change to any of
# them is a new download
FETCHED = ("type", "address", "addresses", "extract", "history", "submodules")


def fingerprint(inputs) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- upstream -----------------------------------------------------------------

def upstream_key(source: Source) -> str:
    """The name the store gives the source's upstream tree: the fingerprint
    of its id and of what its download reads (FETCHED, and its license
    pages when it declares any)."""
    pages = license_pages(source)
    return fingerprint({"kind": "upstream", "source": source.id, **({"pages": pages} if pages else {}),
                        **{key: source.data[key] for key in FETCHED if key in source.data}})


def license_pages(source: Source) -> list[str]:
    """The outside addresses the source's license entries give as their
    place ({"page": address}), each once, in order: what its download
    stores with its files."""
    return sorted({entry["place"]["page"] for entry in source.data.get("licenses", [])
                   if isinstance(entry.get("place"), dict) and "page" in entry["place"]})


def stored_pages(source: Source, store) -> dict:
    """{address: wpt_store.Placed} of the license pages the store holds with
    the source's upstream tree; {} when it holds none."""
    listed = store.get_ref(UPSTREAM_PAGES, upstream_key(source))
    return {address: placed for address, (_, placed) in parse_file_list(store.get_bytes(listed)).items()} \
        if listed else {}


def license_page(source: Source, store, address: str) -> bytes | None:
    """The bytes of a license page downloaded with the source, checked
    against their sha256; None when the store holds no such page for it
    (what the license check reads at a page place, license_check.statement)."""
    placed = stored_pages(source, store).get(address)
    return store.read_file(placed) if placed is not None else None


def upstream_tree(source: Source, store) -> dict[str, Entry]:
    """The source's upstream tree as the store holds it, each downloaded
    file placed in its bundle. An empty source's is empty, and needs no
    download. NotDownloaded when the store does not hold it."""
    if source.data["type"] == "empty":
        return {}
    tree = stored_upstream(upstream_key(source), store)
    if tree is None:
        raise _not_downloaded(source, store)
    return tree


def _not_downloaded(source: Source, store) -> NotDownloaded:
    return NotDownloaded(f"{source.id}: its upstream files are not in {store}; `wpt-dataset download {source.id}` "
                         f"fetches them")


def stored_upstream(key: str, store) -> dict[str, Entry] | None:
    """The upstream tree the store holds under a key (upstream_key), each
    downloaded file placed in its bundle, or None."""
    known = store.get_ref(UPSTREAM, key)
    if not known:
        return None
    return manifest.placed_tree(manifest.parse_manifest(store.get_bytes(known)), stored_files(key, store))


def stored_files(key: str, store) -> dict:
    """The file list (wpt_store.parse_file_list) of the upstream tree the
    store holds under a key; {} for one downloaded before bundles."""
    listed = store.get_ref(UPSTREAM_FILES, key)
    return parse_file_list(store.get_bytes(listed)) if listed else {}


def exact_pattern(path: str) -> str:
    """An extract pattern matching only that path: each `*`, `?` and `[`
    written as a class of that one letter."""
    return manifest.WILDCARD.sub(lambda letter: f"[{letter[0]}]", path)


# -- cuts and patches -----------------------------------------------------------

def apply_cuts(source_id: str, tree: dict[str, Entry], cuts: list[dict], store) -> dict[str, Entry]:
    """Each cut, in the order listed, removes from its file, as the cuts
    before it left it, every run its keyed hashes name (erasure.apply,
    under the store's cut key). CutFailed when its file is not in the tree
    or none of its runs is found, CutKeyMissing when the store keeps no
    cut key: the whole source is out until a person acts."""
    if not cuts:
        return tree
    secret = erasure.key(store)
    tree = dict(tree)
    for cut in cuts:
        path = cut["path"]
        entry = tree.get(path)
        where = f"{source_id}: cut on {path}"
        if entry is None or entry.mode == SYMLINK:
            raise CutFailed(f"{where}: no such file")
        tree[path] = Entry(entry.mode, store.put_bytes(erasure.apply(manifest.content(store, entry), cut, secret,
                                                                     where)))
    return tree


def new_cut(source: Source, store, path: str, start: int, end: int, note: str | None = None) -> dict:
    """The cut entry that removes bytes [start, end) of an upstream file as
    the source's cuts so far left it: its path, the passage's length and its
    keyed hashes under the store's cut key (erasure.cut; the key is made on
    the first cut). The passage itself goes nowhere. NotDownloaded when the
    store does not hold the source, CutFailed for no such file or a range
    outside it."""
    tree = apply_cuts(source.id, upstream_tree(source, store), source.data.get("cuts", []), store)
    entry, where = tree.get(path), f"{source.id}: cut on {path}"
    if entry is None or entry.mode == SYMLINK:
        raise CutFailed(f"{where}: no such file")
    content = manifest.content(store, entry)
    if not 0 <= start < end <= len(content):
        raise CutFailed(f"{where}: range {start}-{end} outside its {len(content)} bytes")
    return {"path": path, **erasure.cut(content[start:end], erasure.made_key(store)), **({"note": note} if note else {})}


def apply_patches(source_id: str, tree: dict[str, Entry], patches: list[Patch], store) -> dict[str, Entry]:
    """Apply git-format patches to the tree inside a temporary git index:
    no files are written out, so modes and symlinks behave the same on every
    operating system. The paths git's index refuses (any .git/ path, such
    as a history's files) are set aside and come back exactly as they
    were; a patch that names one fails, as git refuses it. Only the files
    the patches name go into the index (git apply reads no other); every
    other file is set aside the same way, never read. A file the patches
    leave as it was keeps its entry, and so where it lies."""
    if not patches:
        return tree
    with tempfile.TemporaryDirectory(prefix="wpt-patch-") as tmp:
        git, env = temporary_git(tmp)
        touched = _touched(git, patches)
        loaded = load_index(git, {path: entry for path, entry in tree.items() if path in touched}, store,
                            in_memory=True)
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
    patched = dict(sorted((aside | patched).items()))  # the order a tree read back from its manifest has
    manifest.check_paths(patched)
    return patched


def _touched(git, patches: list[Patch]) -> set[str]:
    """Every path the patches name (git apply --numstat), each side of a
    rename: the only files git reads while it applies them."""
    found = set()
    for patch in patches:
        try:
            listed = git("apply", "--numstat", "-z", "-", input=patch.data)
        except subprocess.CalledProcessError:
            continue  # git apply --cached names what is wrong with it
        fields = listed.split(b"\0")
        while fields and fields[0]:
            *_, path = fields.pop(0).split(b"\t", 2)
            if path:
                found.add(path.decode())
            else:  # a rename: its old path, then its new one
                found |= {fields.pop(0).decode(), fields.pop(0).decode()}
    return found


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


def load_index(git, tree: dict[str, Entry], store, in_memory=False) -> dict[str, str]:
    """Make the temporary index hold this tree minus the paths git refuses:
    git silently drops any .git/ path (such as a history's files), so a
    caller that needs those must handle them itself. Returns {path: git
    object id} of the tree's files. With in_memory (for a few files), each
    file's bytes go to git from memory, with no local copy of it."""
    git("read-tree", "--empty")
    paths = sorted(tree)
    if not paths:
        return {}
    if in_memory:
        objects = [git("hash-object", "-w", "--no-filters", "--stdin", input=manifest.content(store, tree[p]))
                   .decode().strip() for p in paths]
    else:
        objects = git("hash-object", "-w", "--no-filters", "--stdin-paths",
                      input="".join(f"{manifest.local_file(store, tree[p])}\n" for p in paths).encode()).decode().split()
    git("update-index", "-z", "--index-info",
        input=b"".join(f"{tree[p].mode} {o}\t{p}".encode() + b"\0" for p, o in zip(paths, objects)))
    return dict(zip(paths, objects))


def opener(store, copies=None):
    """copies, or without it a function whose every call gives a context
    manager of store itself: what each_read gives each item a store from."""
    return copies or (lambda: contextlib.nullcontext(store))


def each_read(function, items, repo, store, copies=None, telemetry=None, what="sources read at once", args=(),
              disk=None) -> list:
    """[function(item, its store, *args, telemetry=...)] in the items'
    order, each item in a worker process (workers.py), as many at once as
    wpt_utils.parallel allows, disk(item) (when given) the bytes it may
    copy to the disk, which the conversion states beforehand (copy_size). An
    item's store is store, or with copies the store copies() gives that
    item alone (a context manager, such as a streamed build's temporary
    folder of local copies), ended as soon as the item is read, so the
    local copies it made are gone before later items need room. An item's
    error is raised at its turn, and no new item starts after it. With a
    telemetry, the item records under the caller's current span."""
    items = list(items)
    if not items:
        return []
    telemetry = telemetry or Telemetry.off()
    parent = telemetry.current()
    opened = opener(store, copies)
    with workers.Pool(repo, what, disk=disk, folder=tempfile.gettempdir() if copies else store.cache) as pool:
        def run(item):
            with telemetry.within(parent), opened() as here:
                return pool.call(function, item, here, *args, telemetry=telemetry)

        with ThreadPoolExecutor(pool.sizing.limit) as executor:
            return list(in_order(run, items, executor, pool.sizing))


def held_manifest(source: Source, store) -> str:
    """The source's manifest as the records and tasks name it: the sha256
    of the manifest of its upstream tree as the store holds it
    (upstream_manifest), or the empty tree's (manifest.digest of no
    files) for an empty source; what its download got, for the record
    only; nothing decides on it. Read from the store's name for the tree,
    never computed again. NotDownloaded when the store does not hold it."""
    if source.data["type"] == "empty":
        return manifest.digest({})
    known = upstream_manifest(source, store)
    if known is None:
        raise _not_downloaded(source, store)
    return known


def copy_size(source: Source, store, wanted) -> int:
    """The bytes a conversion will copy to the disk, stated beforehand: the
    size, from the source's file list, of its upstream files whose path
    wanted accepts, each content once. The conversion says which files it
    copies (copied_as_records, copied_as_trees, copied_for_repository),
    and each_read reserves that on the disk its copies go to. 0 for an
    empty source and for one whose file list the store does not hold."""
    if source.data["type"] == "empty":
        return 0
    return sum({placed.sha256: placed.length for path, (_, placed) in
                stored_files(upstream_key(source), store).items() if wanted(path)}.values())


def copied_as_records(source: Source):
    """Which paths the conversion of documents copies: the Parquet files
    read as records (a records file is otherwise read as a stream from the
    store, never copied: manifest.Stored)."""
    return lambda path: is_records(source, path) and _seeks(path)


def _seeks(path: str) -> bool:
    """Whether a records file is read from a local copy (Parquet, whose
    reader seeks) rather than as a stream."""
    return path.lower().endswith(".parquet")


def copied_by_pointers(source: Source, paths):
    """Which paths the reading of the documents tasks point into copies
    (markdown.chosen reads only the files the pointed paths, a file's or
    <file>#<record id>, name): those of them read as records from a local
    copy (Parquet)."""
    pointed = list(paths)
    return lambda path: is_records(source, path) and _seeks(path) and any(p == path or p.startswith(f"{path}#")
                                                                          for p in pointed)


def copied_as_trees(source: Source):
    """Which paths the reading of conversations copies: the chat files read
    as message trees from a local copy (Parquet; conversations.of_file
    reads the others as streams)."""
    return lambda path: is_message_tree(source, path) and _seeks(path)


def copied_for_repository(source: Source):
    """Which paths a task's repository copies (tasks.repository loads the
    source's files into a git index from copies): every file but the
    history's (`.git/`)."""
    return lambda path: not path.startswith(".git/")


# -- the final tree -------------------------------------------------------------

def upstream_manifest(source: Source, store) -> str | None:
    """The sha256 of the manifest object the store holds as the source's
    upstream tree, None for an empty source (no upstream) and for one the
    store does not hold."""
    return None if source.data["type"] == "empty" else store.get_ref(UPSTREAM, upstream_key(source))


def tree_key(source: Source, store) -> str:
    """The fingerprint of a source's final tree: its upstream tree as the
    store holds it (its manifest's sha256) and the license pages downloaded
    with it, cuts (and the cut key they are matched under), patches,
    formats (which decide each file's encoding), and the code
    (wpt_utils.programs.code: the WPT repository's whole commit, every
    uncommitted change under data/ included), so after any commit every
    result keyed by it is computed again from the stored bytes."""
    cuts = source.data.get("cuts", [])
    return fingerprint({
        "kind": "source-tree",
        "code": code(),
        "upstream": upstream_manifest(source, store),
        **({"pages": store.get_ref(UPSTREAM_PAGES, upstream_key(source))} if license_pages(source) else {}),
        "formats": source.data["formats"],
        "cuts": cuts,
        **({"cut_key": store.get_ref(erasure.KEY_KIND, erasure.KEY_NAME)} if cuts else {}),
        "patches": [[p.name, hashlib.sha256(p.data).hexdigest()] for p in source.patches],
    })


def final_tree(source: Source, store, telemetry=None) -> tuple[dict[str, Entry], str]:
    """The source's final tree, its upstream tree with its cuts and patches
    applied, and the sha256 of its manifest object, each file its patches
    left as it was placed where the download put it; nothing checked.
    NotDownloaded when the store does not hold its upstream files,
    CutFailed (CutKeyMissing among them) or PatchFailed when a cut or a
    patch does not apply. Kept under the ref source-trees/<tree_key>. With
    a telemetry, every ref it sets is recorded first (wpt_telemetry.set_ref)."""
    key = tree_key(source, store)
    known = store.get_ref("source-trees", key)
    if known:
        tree = manifest.parse_manifest(store.get_bytes(known))
        if source.data["type"] != "empty":
            tree = manifest.placed_tree(tree, stored_files(upstream_key(source), store))
        return tree, known
    tree = upstream_tree(source, store)
    tree = apply_cuts(source.id, tree, source.data.get("cuts", []), store)
    tree = apply_patches(source.id, tree, source.patches, store)
    known = store.put_bytes(manifest.manifest_bytes(tree))
    set_ref(store, "source-trees", key, known, telemetry or Telemetry.off())
    return tree, known


def source_tree(source: Source, store, telemetry=None) -> tuple[dict[str, Entry], str]:
    """final_tree, refused whole for the commands that hold a source to
    every rule (`check --fetch`, the writes of the service layer, `unpack`):
    NotText for its first file that is not text, LicenseMapError for its
    first file no license entry covers. The build leaves such files out
    one by one instead (kept.py)."""
    tree, known = final_tree(source, store, telemetry)
    if found := not_text(source, tree, known, store, telemetry):
        raise NotText(f"{source.id}: {next(iter(found.values()))}; exclude it in extract")
    check_license_map(source, tree)
    return tree, known


def not_text(source: Source, tree: dict[str, Entry], known: str, store, telemetry=None) -> dict[str, str]:
    """{path: why} of the files of a final tree (known: its manifest
    object) that the text check (checks.py) finds are not text, in the
    tree's order. Kept under the ref source-not-text/<fingerprint of the
    tree, the formats and the code>, so a tree is read for it once."""
    key = fingerprint({"kind": "source-not-text", "code": code(), "tree": known, "formats": source.data["formats"]})
    held = store.get_ref("source-not-text", key)
    if held:
        return json.loads(store.get_bytes(held))
    found = {}
    for path, entry in tree.items():
        if checks.asks_text(source, path, entry):
            if (why := checks.not_text(source, path, manifest.content(store, entry))) is not None:
                found[path] = why
    set_ref(store, "source-not-text", key, store.put_bytes(json.dumps(found, ensure_ascii=False).encode()),
            telemetry or Telemetry.off())
    return found


def is_records(source: Source, path: str) -> bool:
    """Whether a path is in the records format (see records.py), a file a
    records entry joins included."""
    return records_entry(source, path) is not None


def records_entry(source: Source, path: str) -> dict | None:
    """The records format entry covering a path, or None."""
    return next((e for e in manifest.format_entries(source, path) if e["format"] == "records"), None)


def is_message_tree(source: Source, path: str) -> bool:
    """Whether a path is a chat file read as a message tree, which is read
    like a records file (JSON Lines, compressed or not, or Parquet)."""
    return any(e["format"] == "chat" and "tree" in e for e in manifest.format_entries(source, path))


def check_license_map(source: Source, tree: dict[str, Entry]):
    """LicenseMapError unless the license check covers every path of the
    tree (license_check.coverage): each file by exactly one entry, a records
    file's records by entries with the same field place."""
    license_check.check_coverage(source.id, license_check.coverage(source.data["licenses"], tree,
                                                                   lambda path: is_records(source, path)))


def tags_of(source: Source, path: str) -> frozenset[str]:
    """The tags of a source's path: those of every tag entry covering it,
    found by the source's one matcher (manifest.source_paths)."""
    return manifest.source_paths(source).tags(path)


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
    return source.data["type"] == "empty" or store.get_ref(UPSTREAM, upstream_key(source)) is not None
