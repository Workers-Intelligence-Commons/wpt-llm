"""`wpt dataset contamination`: the runs of NGRAM words that a dataset's
held-out documents share with its training documents.

The check only reports: it removes nothing, and nothing but this command
reads its report. A build never runs it, and build.py does not import this
module, so the dataset id (which holds the code of build.py and of what it
imports) depends neither on this code nor on a report.

The documents come from the dataset's shards, the one place a dataset id
leads to their tokens: the held-out shards hold every held-out document, the
training shards every training document the pools drew, repeated ones once,
in the order of their first row. Each shard is copied to a temporary folder,
checked against its token hash and its copy removed at once, so one shard at
a time is on this machine, and the folder goes when the command ends,
also when it fails. The held-out runs are listed once; batches of training
documents are searched in worker processes (workers.py), as many at once as
wpt_utils.parallel allows, and reach them with their tokens.

The report is saved as an object under the ref REPORT/<dataset id>, and a
run on a dataset with a report prints that report and checks nothing. A
report inside the dataset's own record, under "contamination", counts as
saved too: such a record comes from a build that ran the check itself, over
every training document, the ones its pools did not draw included.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from wpt_store.shards import documents
from wpt_telemetry import Telemetry, set_ref, short
from wpt_utils.parallel import each

from . import reading, workers
from .build import current_record
from .errors import DatasetError, NotFound
from .repo import Repo

NGRAM = 13  # runs of 13 words, as in the GPT-3 paper
BATCH_TOKENS = 1 << 22  # tokens of the training documents one worker checks in one call (a longer document alone)
REPORT = "dataset-contamination"  # the kind of the ref, named by the dataset id, of a dataset's report


def run(repo: Repo, dataset_id: str, store, arguments) -> dict:
    """`wpt dataset contamination`: the dataset's saved report, or else its
    check, saved; prints it either way, recorded in the store
    (wpt_telemetry: records/<UTC>-wpt-dataset-contamination-<id>-.../).
    Returns the report. Overlap is a finding and raises nothing. NotFound
    for a dataset the store does not hold, DatasetError for a record of
    another shape, RecordError when the record cannot be written."""
    record = _record(store, dataset_id)
    with Telemetry.start(store, "wpt dataset contamination", arguments, short(dataset_id),
                         inputs={"dataset": dataset_id}, env_file=repo.root / ".env") as telemetry:
        report = saved(store, record)
        if report is not None:
            telemetry.log(f"dataset {dataset_id}: its saved report, nothing checked")
        else:
            report = _checked(repo, store, record, telemetry)
            with telemetry.span("contamination.report"):
                written = store.put_bytes(json.dumps(report, indent=2).encode() + b"\n")
                telemetry.output(contamination=written)
                set_ref(store, REPORT, dataset_id, written, telemetry)
        for line in summary(report):
            telemetry.log(line)
    return report


def saved(store, record: dict) -> dict | None:
    """The report saved for the dataset of this record; None when it has none."""
    known = store.get_ref(REPORT, record["id"])
    return record.get("contamination") if known is None else json.loads(store.get_bytes(known))


def summary(report: dict) -> list[str]:
    """What the command prints of a report: one line, then one per example."""
    return [f"contamination: {report['overlapping']} held-out documents share a run of {report['ngram_words']} "
            "words with training documents (reported only)",
            *(f"{example['held_out']} shares a run with {example['training']}" for example in report["examples"])]


def _record(store, dataset_id: str) -> dict:
    """The record of a dataset the store holds, of the shape this code reads."""
    known = store.get_ref("datasets", dataset_id)
    if known is None:
        raise NotFound(f"no dataset {dataset_id} in {store}")
    record = json.loads(store.get_bytes(known))
    if not current_record(record):
        raise DatasetError(f"dataset {dataset_id} has a record of another shape than this code reads; build its "
                           "recipe again")
    return record


def _checked(repo: Repo, store, record: dict, telemetry) -> dict:
    """overlap() of the dataset's shards, the training side searched in worker processes. Each takes a batch of
    training documents (BATCH_TOKENS tokens), decodes them and gives back which held-out runs each holds. The
    held-out runs reach the workers as one file, which each worker maps once."""
    vocabulary = record["tokenizer"]["vocabulary"]
    tokenizer = reading.tokenizer_of(vocabulary, store)
    tokens = sum(shard["tokens"] for shard in record["held_out_shards"] + record["shards"])
    with (tempfile.TemporaryDirectory(prefix="wpt-contamination-") as tmp,
          telemetry.span("contamination") as span,
          telemetry.progress("contamination", tokens, "tokens") as progress,
          workers.Pool(repo, "contamination batches at once") as pool):
        copies = store.copies_in(Path(tmp) / "copies")

        def search(runs):
            path = Path(tmp) / "held-out-runs.npy"
            np.save(path, runs)

            def look(batch):
                return pool.call(_runs_found, batch, vocabulary, str(path), store, telemetry=telemetry)

            # no error is an item's result: any stops the check
            for done in each(look, _batches(_read(copies, record["shards"], progress)), (), pool.sizing):
                yield from zip((document for document, _ in done.item), done.value)

        held_out = ((document, tokenizer.decode(found.tolist()))
                    for document, found in _read(copies, record["held_out_shards"], progress))
        report = overlap(held_out, search)
        span.set(**{key: value for key, value in report.items() if key != "examples"})
    return report


def _read(store, shards: list, progress):
    """(document id, tokens) of each document of these shards, once, in the order of its first row."""
    seen = set()
    for shard in shards:
        for document, _, tokens in documents(store, [shard]):
            if document not in seen:
                seen.add(document)
                yield document, tokens
        progress.advance(shard["tokens"])


def _batches(found):
    """The (document id, tokens) pairs in runs of BATCH_TOKENS tokens at least (the last may hold fewer), in order."""
    batch, size = [], 0
    for document, tokens in found:
        batch.append((document, tokens))
        size += len(tokens)
        if size >= BATCH_TOKENS:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


_HELD_OUT_RUNS = {}  # in a worker process: the held-out runs file it mapped, by path


def _runs_found(batch, vocabulary: str, path: str, store, telemetry) -> list[np.ndarray]:
    """In a worker process: for each document of batch, (document id, tokens) pairs, the positions in the held-out
    runs (the file at path) of the runs its text holds."""
    if path not in _HELD_OUT_RUNS:
        _HELD_OUT_RUNS.clear()
        _HELD_OUT_RUNS[path] = np.load(path, mmap_mode="r")
    runs = _HELD_OUT_RUNS[path]
    tokenizer = reading.tokenizer_of(vocabulary, store)
    found = []
    for _, tokens in batch:
        here = runs_of(tokenizer.decode(tokens.tolist()))
        at = np.minimum(np.searchsorted(runs, here), len(runs) - 1)
        found.append(at[runs[at] == here])
    return found


def runs_of(text: bytes) -> np.ndarray:
    """The distinct runs of NGRAM words of a text, each as the 8 bytes of its hash read as a uint64."""
    words = text.split()
    return np.frombuffer(b"".join({hashlib.blake2b(b" ".join(words[i:i + NGRAM]), digest_size=8).digest()
                                   for i in range(len(words) - NGRAM + 1)}), dtype="<u8")


def overlap(held_out, search) -> dict:
    """Held-out documents sharing a run of NGRAM words with training text.
    held_out yields (document id, Markdown). search(runs) gives (document id,
    positions in runs) for each training document in order, runs being the
    sorted array of every run the held-out documents hold, and positions
    those of the runs its text holds. Each held-out document is paired with
    the first training document, in order, that shares a run with it."""
    names, pieces = [], []
    for document, text in held_out:
        names.append(document)
        pieces.append(runs_of(text))
    everything = np.concatenate(pieces) if pieces else np.empty(0, dtype="<u8")
    owners = np.concatenate([np.full(len(piece), i, dtype=np.int64) for i, piece in enumerate(pieces)]) \
        if pieces else np.empty(0, dtype=np.int64)
    runs, first = np.unique(everything, return_index=True)  # first: where each run first occurs, in held-out order
    owners = owners[first]
    found = {}
    if len(runs):
        for document, positions in search(runs):
            for owner in np.unique(owners[positions]):
                found.setdefault(names[owner], document)
    return {"ngram_words": NGRAM, "held_out_documents": len(names), "overlapping": len(found),
            "examples": [{"held_out": h, "training": t} for h, t in sorted(found.items())[:20]]}
