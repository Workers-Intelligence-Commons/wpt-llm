"""Held-out loss: the model's next-token loss on every held-out token, and
by kind of text.

The held-out documents (context.documents("held_out"), each once, in shard
order) are read as one stream, as the training windows read the dataset:
each document after the end-of-document token that ends the one before it,
the first after one end-of-document token. The stream is cut into windows
of the model's block_size + 1 tokens, each window starting at the last
token of the one before, so every token of every document is predicted
exactly once, with the earlier tokens of its window before it
(context.model.losses). Each token's loss, in nats, goes to its document.

The result gives the loss per token over every held-out token and over
each source's; with context.kind_by (wpt eval's --kind-by), over each kind
of text by that field of the sources (wpt.eval.kinds): a kind's loss is the
mean over the tokens of its documents, and a document of several kinds
counts in each.
"""

from collections import defaultdict

import numpy as np

from wpt_dataset.repo import Repo
from wpt.eval import Result
from wpt.eval.kinds import FIELDS, Sources, kinds_of
from wpt_tokenizer import END

CALL_TOKENS = 1 << 20  # the tokens of the windows sent to the model in one call, at most (one window when longer)


def windows(documents, length, seen: list):
    """(window tokens, the index in seen of the document each of its tokens
    after the first belongs to): the windows of length + 1 tokens that
    cover the stream of documents, the last one shorter; each document's
    (id, source) is appended to seen as it is read."""
    pending, owners = np.array([END], dtype=np.uint32), np.array([-1], dtype=np.int64)
    for document in documents:
        index = len(seen)
        seen.append((document.id, document.source))
        pending = np.concatenate([pending, document.tokens.astype(np.uint32)])
        owners = np.concatenate([owners, np.full(len(document.tokens), index, dtype=np.int64)])
        while len(pending) > length:
            yield pending[:length + 1], owners[1:length + 1]
            pending, owners = pending[length:], owners[length:]
    if len(pending) > 1:
        yield pending, owners[1:]


def losses_by_document(model, documents) -> tuple[list, np.ndarray, np.ndarray]:
    """([(id, source)] of each document, the sum of its tokens' losses, its tokens)."""
    seen = []
    sums, counts = defaultdict(float), defaultdict(int)
    batch, batch_owners, size = [], [], 0

    def flush():
        for found, owners in zip(model.losses([window.tolist() for window in batch]), batch_owners):
            kept, place = np.unique(owners, return_inverse=True)
            for index, total, count in zip(kept.tolist(), np.bincount(place, weights=found).tolist(),
                                           np.bincount(place).tolist()):
                sums[index] += total
                counts[index] += count
        batch.clear()
        batch_owners.clear()

    for window, owners in windows(documents, model.config.block_size, seen):
        if batch and size + len(window) > CALL_TOKENS:
            flush()
            size = 0
        batch.append(window)
        batch_owners.append(owners)
        size += len(window)
    if batch:
        flush()
    count = len(seen)
    return (seen, np.array([sums[i] for i in range(count)], dtype=np.float64),
            np.array([counts[i] for i in range(count)], dtype=np.int64))


def grouped(names_of, totals, counts) -> dict:
    """{group: {"documents", "tokens", "loss"}}, each document's tokens
    counted in every group names_of(index) gives, largest group first."""
    sums, tokens, documents = defaultdict(float), defaultdict(int), defaultdict(int)
    for index, (total, count) in enumerate(zip(totals.tolist(), counts.tolist())):
        for name in names_of(index):
            sums[name] += total
            tokens[name] += count
            documents[name] += 1
    order = sorted(sums, key=lambda name: (-tokens[name], name))
    return {name: {"documents": documents[name], "tokens": tokens[name],
                   "loss": sums[name] / tokens[name] if tokens[name] else None} for name in order}


def run(context):
    length = context.model.config.block_size
    found, totals, counts = losses_by_document(context.model, context.documents("held_out"))
    tokens = int(counts.sum())
    data = {"window": length, "documents": len(found), "tokens": tokens,
            "loss": float(totals.sum()) / tokens if tokens else None, "kind_by": context.kind_by,
            "by_source": grouped(lambda index: (found[index][1],), totals, counts), "by_kind": None}
    if not tokens:
        return Result(data, "Not run: the dataset holds no held-out token.\n")
    lines = [f"Loss on every held-out token: {data['loss']:.4f} nats per token, over {tokens:,} tokens of "
             f"{len(found):,} documents, read in windows of {length:,} tokens."]
    if context.kind_by is None:
        lines.append(f"Not grouped by kind of text: --kind-by FIELD ({', '.join(FIELDS)}) groups it.")
    else:
        sources = Sources(Repo.find())
        data["by_kind"] = grouped(lambda index: sorted(kinds_of(context.kind_by, sources, context.store,
                                                                found[index][0])), totals, counts)
        lines += ["", f"By kind of text ({context.kind_by}):"]
        lines += [f"- {name}: {group['loss']:.4f} over {group['tokens']:,} tokens of {group['documents']:,} "
                  f"documents" for name, group in data["by_kind"].items()]
    return Result(data, "\n".join(lines) + "\n")
