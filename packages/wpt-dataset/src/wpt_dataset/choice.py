"""A build's first two steps, which read no text: the index and the choice.

The index is built from what the store's records already say: each source's
final tree, with each file's tar, position, size and sha256, and the
source's formats, licenses and tags. It holds every file that can give a
document: no symbolic link, no chat, task, rating or probe file, no git
history's order file, no file a records entry joins to another, and no
file its license entries refuse without reading anything (no entry covers
it, several do, or its entry is not admitted). Files with the same sha256
are one document, and the index keeps the first in source id order. A
source whose format map fails, or that a final tree cannot be made for, is
named in `failed`.

What a source gives the index is kept in the store as the result
source-index/<index_key: its source.json, the fingerprint of its final tree,
its upstream file list and INDEX_VERSION>, so that a build computes only the
sources where one of them changed, and the duplicates are taken out of the
parts on every build. A source that fails keeps nothing: it is computed on
every build, so it stays failed, with its own error, for as long as it fails.

The choice draws a mixture's documents from the index. Each row goes to the
first pool, in mixture order, whose expression matches its tags. Inside a
pool, rows are grouped by the frame their file starts in (the tar and
start // wpt_store.blocks.FRAME), the unit the store fetches, so that a
build reads about the bytes it chooses; a row with no tar is a group of
its own. Each pool holds its groups in one seeded order, from a seed fixed
by the pool's name and the group's own name (the tar's name and the
frame number, or the row's sha256), so a source added later moves no group
before another; a group's rows keep their order in the tar. take() gives
the next whole groups of each pool, by file bytes, so a pool's share may be
passed by one group at most, and a build calls it in rounds until every pool
has its share: the first round guesses BYTES_PER_TOKEN, each later one uses
the bytes per token the rounds before it measured, with a MARGIN. A
vocabulary's sample is the start of each pool's chosen rows, up to the
pool's weight of the sample's bytes, ending on whole groups.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from wpt_store import CorruptFile, MissingFile, blocks
from wpt_telemetry import Telemetry, get_result, set_result
from wpt_utils.lines import say
from wpt_utils.parallel import Sizing, each

from . import license_check, markdown, pools, records, workers
from .errors import DatasetError
from .manifest import ORDER, SYMLINK
from .sources import final_tree, fingerprint, held_source, tags_of, tree_key

INDEX_VERSION = 2  # raised by hand when a change to the code changes what a source gives the index
BYTES_PER_TOKEN = 4  # the first round's guess of the file bytes one token takes
MARGIN = 1.1  # how much more a later round reads than its measure asks, so that few rounds are needed
NO_DOCUMENT = ("chat", "task", "rating", "probe")  # formats whose files give no document
SOURCE_INDEX = "source-index"  # the stage of what a source gives the index
LOOKUPS = 24  # sources whose saved parts are looked for at once: each only waits on the store's answers
FAILURES = (DatasetError, CorruptFile, MissingFile)  # what takes one source out of the index
ROW = np.dtype([("source", "<i4"), ("position", "<i4"), ("sha256", "S32"), ("bytes", "<i8"), ("tags", "<i4"),
                ("tar", "<i4"), ("start", "<i8")])  # position: in the final tree; tar -1: a file outside a tar


def gives_document(source, path: str, found: dict) -> bool:
    """Whether a file whose format entry is found can give a document."""
    if found["format"] in NO_DOCUMENT or (path == ORDER and source.data["type"] == "git"):
        return False
    return not (found["format"] == "records" and records.joined_by(found, path))


def _refused(licenses: list[dict], covered: license_check.Covered) -> bool:
    """Whether a file's license entries leave it out whatever is read."""
    return covered.problem is not None or (covered.field is None and not license_check.admitted(licenses[covered.entry]))


def index_key(source, store) -> str:
    """The fingerprint of what a source gives the index: everything its
    computation reads, the source's whole source.json (its type, licenses,
    formats, tags and cuts), its final tree (sources.tree_key: the upstream
    manifest, the cuts and their key, the patches), the sha256 of the
    source's tar, which alone says where each file lies, and INDEX_VERSION.
    The sizes of files outside a tar follow from their sha256."""
    return fingerprint({"kind": SOURCE_INDEX, "version": INDEX_VERSION, "source": source.id, "data": source.data,
                        "tree": tree_key(source, store), "tar": found.sha256 if (found := held_source(source, store)) else None})


def _part_bytes(rows: np.ndarray, tag_sets: list[frozenset[str]], tars: list[str]) -> bytes:
    """A source's part of the index as it is kept: its numbers on one line
    of JSON, then its rows."""
    numbers = {"tag_sets": [sorted(tags) for tags in tag_sets], "tars": tars}
    return json.dumps(numbers, ensure_ascii=False).encode() + b"\n" + rows.tobytes()


def _parsed_part(data: bytes) -> tuple[np.ndarray, list[frozenset[str]], list[str]]:
    """The part _part_bytes wrote, its rows writable."""
    line, _, rows = data.partition(b"\n")
    numbers = json.loads(line)
    return (np.frombuffer(rows, dtype=ROW).copy(), [frozenset(tags) for tags in numbers["tag_sets"]],
            numbers["tars"])


def _indexed(source, store, key, telemetry):
    """(rows, the tag sets their `tags` number, the tars their `tar`
    numbers) of one source, source numbered 0, in a worker process, kept
    under key."""
    tree, _ = final_tree(source, store, telemetry)
    licenses = source.data["licenses"]
    coverage = license_check.Coverage(licenses)
    found, tag_sets, tars = [], {}, {}
    for position, (path, entry) in enumerate(tree.items()):
        if entry.mode == SYMLINK:
            continue
        kind = markdown.format_of(source, path)
        if not gives_document(source, path, kind) or \
                _refused(licenses, coverage.covered(path, kind["format"] == "records")):
            continue
        placed = entry.placed
        found.append((0, position, bytes.fromhex(entry.sha256),
                      placed.length if placed is not None else store.size(entry.sha256),
                      tag_sets.setdefault(tags_of(source, path), len(tag_sets)),
                      tars.setdefault(source.id, len(tars)) if placed is not None else -1,
                      placed.start if placed is not None else 0))
    part = np.array(found, dtype=ROW), list(tag_sets), list(tars)
    set_result(store, SOURCE_INDEX, key, _part_bytes(*part), telemetry)
    return part


@dataclass
class Index:
    rows: np.ndarray  # of ROW, duplicates gone, in source and tree order
    sources: list  # the sources `source` numbers
    tag_sets: list[frozenset[str]]  # what `tags` numbers
    tars: list[str]  # the names of the tars `tar` numbers
    duplicates: int  # the rows the same sha256 took out
    failed: list = field(default_factory=list)  # [(source, error)], each source no row comes from


def _saved(admitted, store, telemetry) -> tuple[dict, dict]:
    """({source id: the Done of its saved part, or of the error finding it
    failed with}, {source id: its index_key}), several sources at once."""
    found, keys = {}, {}
    with telemetry.progress("index, saved parts", len(admitted), "sources") as progress:
        def look(source):
            try:
                keys[source.id] = index_key(source, store)
                held = get_result(store, SOURCE_INDEX, keys[source.id], telemetry)
                return _parsed_part(held) if held is not None else None
            finally:
                progress.advance()

        for done in each(look, admitted, FAILURES, Sizing("saved parts looked for at once", network=LOOKUPS)):
            if done.value is not None or done.error is not None:
                found[done.item.id] = done
    return found, keys


def _computed(sources, keys, repo, store, telemetry) -> dict:
    """{source id: the Done of its part}, several sources at once, in worker
    processes."""
    found = {}
    with (workers.Pool(repo, "sources indexed at once") as pool,
          telemetry.progress("index", len(sources), "sources") as progress):
        parent = telemetry.current()

        def run(source):
            try:
                with telemetry.within(parent):
                    return pool.call(_indexed, source, store, keys[source.id], telemetry=telemetry)
            finally:
                progress.advance()

        for done in each(run, sources, FAILURES, pool.sizing):
            found[done.item.id] = done
    return found


def build(repo, store, telemetry=None, log=say) -> Index:
    """The Index of every admitted source, from what the store keeps of each
    (see the module's docstring), the others computed in worker processes,
    none of whose texts is read."""
    telemetry = telemetry or Telemetry.off()
    admitted = repo.admitted(telemetry, log)
    parts, sources, tag_sets, tars, failed = [], [], {}, {}, []
    with telemetry.span("index", sources=len(admitted)) as span:
        results, keys = _saved(admitted, store, telemetry)
        reused = sum(done.error is None for done in results.values())
        missing = [source for source in admitted if source.id not in results]
        telemetry.log(f"index: {reused:,} of {len(admitted):,} sources saved, {len(missing):,} to compute")
        if missing:
            results |= _computed(missing, keys, repo, store, telemetry)
        for source in admitted:
            done = results[source.id]
            if done.error is not None:
                failed.append((source, done.error))
                continue
            rows, tags, names = done.value
            if len(rows):
                rows["source"] = len(sources)
                rows["tags"] = np.array([tag_sets.setdefault(found, len(tag_sets)) for found in tags])[rows["tags"]]
                numbers = np.array([tars.setdefault(name, len(tars)) for name in names] + [-1])
                rows["tar"] = numbers[rows["tar"]]
                parts.append(rows)
            sources.append(source)
        rows = np.concatenate(parts) if parts else np.empty(0, dtype=ROW)
        first = np.sort(np.unique(rows["sha256"], return_index=True)[1])
        span.set(files=len(rows), documents=len(first), reused=reused, failed=[source.id for source, _ in failed])
    return Index(rows[first], sources, list(tag_sets), list(tars), len(rows) - len(first), failed)


def _seed(name: str) -> np.uint64:
    return np.uint64(int.from_bytes(hashlib.sha256(f"pool {name}".encode()).digest()[:8]))


def _mixed(values: np.ndarray) -> np.ndarray:
    """values (uint64) mixed by SplitMix64's finishing steps."""
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return values ^ (values >> np.uint64(31))


def _ordered(rows: np.ndarray, tars: list[str], seed: np.uint64) -> tuple[np.ndarray, np.ndarray]:
    """(the rows' positions in their seeded order, the number of the group
    each of them is in, which never falls): the rows that start in one frame
    of one tar are a group, and so is each row of no tar. Groups come
    in the order of their name mixed with the seed, a group's rows in the
    order of their start."""
    names = np.array([int.from_bytes(hashlib.sha256(name.encode()).digest()[:8]) for name in tars] + [0], dtype="<u8")
    frames = (rows["start"] // blocks.FRAME).astype("<u8")
    own = np.ascontiguousarray(rows["sha256"]).view("u1").reshape(-1, 32)[:, :8].copy().view("<u8").ravel()
    named = np.where(rows["tar"] >= 0, _mixed(names[rows["tar"]] + frames * np.uint64(0x9e3779b97f4a7c15)), own)
    ranks = _mixed(named ^ seed)
    order = np.lexsort((rows["start"], frames, rows["tar"], ranks))
    ranks, tar, frames = ranks[order], rows["tar"][order], frames[order]
    starts = np.ones(len(order), dtype=bool)
    starts[1:] = (ranks[1:] != ranks[:-1]) | (tar[1:] != tar[:-1]) | (frames[1:] != frames[:-1]) | (tar[1:] < 0)
    return order, np.cumsum(starts)


class Choice:
    """A mixture's documents drawn from an index, round after round."""

    def __init__(self, index: Index, mixture: dict):
        self.index = index
        self.names = [pool["name"] for pool in mixture["pools"]]
        trees = [pools.parse(pool["select"]) for pool in mixture["pools"]]
        first = [next((number for number, tree in enumerate(trees) if pools.matches(tree, tags)), -1)
                 for tags in index.tag_sets]
        of_row = np.array(first + [-1], dtype="<i4")[index.rows["tags"]]
        self.orders, self.groups = {}, {}  # per pool: its rows' positions in order, and each one's group number
        for number, name in enumerate(self.names):
            rows = np.flatnonzero(of_row == number)
            order, self.groups[name] = _ordered(index.rows[rows], index.tars, _seed(name))
            self.orders[name] = rows[order]
        self.taken = dict.fromkeys(self.names, 0)  # how many rows of each pool's order the rounds took

    def take(self, wanted: dict[str, int]) -> np.ndarray:
        """The next whole groups of each pool until they hold its wanted
        bytes, or none is left."""
        found = []
        for name, size in wanted.items():
            start, order, groups = self.taken[name], self.orders[name], self.groups[name][self.taken[name]:]
            sizes = np.cumsum(self.index.rows["bytes"][order[start:]])
            count = min(int(np.searchsorted(sizes, size)) + 1, len(sizes)) if size > 0 else 0
            if count:
                count = int(np.searchsorted(groups, groups[count - 1], side="right"))
            found.append(order[start:start + count])
            self.taken[name] = start + count
        return self.index.rows[np.concatenate(found)] if found else self.index.rows[:0]

    def bytes_of(self, name: str) -> tuple[int, int]:
        """(file bytes taken, file bytes the index holds) of a pool."""
        sizes = self.index.rows["bytes"][self.orders[name]]
        return int(sizes[:self.taken[name]].sum()), int(sizes.sum())

    def sample_of_pools(self, mixture: dict, sample_bytes: int) -> dict[str, np.ndarray]:
        """{pool: its rows of a vocabulary's sample} out of the rows taken:
        the start of each pool's rows, up to its weight of sample_bytes and
        ending on whole groups, at least one group."""
        found = {}
        for pool in mixture["pools"]:
            order = self.orders[pool["name"]][:self.taken[pool["name"]]]
            sizes = np.cumsum(self.index.rows["bytes"][order])
            groups = self.groups[pool["name"]][:len(order)]
            count = int(np.searchsorted(sizes, pool["weight"] * sample_bytes, side="right"))
            if 0 < count < len(order):
                count = int(np.searchsorted(groups, groups[count], side="left"))  # back to the start of the group it cuts
            if count == 0 and len(order):
                count = int(np.searchsorted(groups, groups[0], side="right"))
            found[pool["name"]] = self.index.rows[order[:count]]
        return found

    def sample(self, mixture: dict, sample_bytes: int) -> np.ndarray:
        """The rows of a vocabulary's sample (sample_of_pools)."""
        return np.concatenate(list(self.sample_of_pools(mixture, sample_bytes).values()))


def first_wanted(mixture: dict, recipe: dict) -> dict[str, int]:
    """{pool: file bytes} of the first round of a dataset recipe: enough, at
    BYTES_PER_TOKEN, for the pool's share once its held-out part is gone."""
    return {pool["name"]: math.ceil(pool["weight"] * recipe["total_tokens"] * BYTES_PER_TOKEN / (1 - recipe["held_out"]))
            for pool in mixture["pools"]}


def next_wanted(mixture: dict, recipe: dict, read: dict[str, int], tokens: dict[str, int]) -> dict[str, int]:
    """{pool: file bytes} of the next round, only for the pools short of
    their share: the training tokens missing, at the bytes per training
    token measured so far (read and tokens, per pool), with MARGIN; a pool
    that gave no training token asks twice what it read."""
    wanted = {}
    for pool in mixture["pools"]:
        name = pool["name"]
        missing = round(pool["weight"] * recipe["total_tokens"]) - tokens[name]
        if missing > 0:
            wanted[name] = max(math.ceil(missing * read[name] / tokens[name] * MARGIN), 1) if tokens[name] \
                else 2 * read[name] + 1
    return wanted


def by_source(index: Index, rows: np.ndarray, without=frozenset()) -> list:
    """[(source, its rows in tar order)] in tar order, of the rows of
    the sources whose ids are not in without."""
    if not len(rows):
        return []
    rows = rows[np.lexsort((rows["start"], rows["tar"]))]
    numbers, first = np.unique(rows["source"], return_index=True)
    grouped = rows[np.argsort(rows["source"], kind="stable")]
    parts = np.split(grouped, np.searchsorted(grouped["source"], numbers)[1:])
    return [(index.sources[numbers[at]], parts[at]) for at in np.argsort(first)
            if index.sources[numbers[at]].id not in without]


def in_numbers(rows: np.ndarray) -> dict[str, int]:
    """How much the rows hold, for a plan: {files, bytes, sources, tars}."""
    return {"files": len(rows), "bytes": int(rows["bytes"].sum()), "sources": len(np.unique(rows["source"])),
            "tars": len(np.unique(rows["tar"][rows["tar"] >= 0]))}
