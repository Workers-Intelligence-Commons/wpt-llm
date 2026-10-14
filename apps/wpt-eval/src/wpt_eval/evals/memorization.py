"""Memorization: does the model reproduce its training documents word for word?

For up to DOCUMENTS documents of each part, the model reads the first PREFIX
tokens of the document (after the end-of-document token that precedes every
document in training) and writes CONTINUATION tokens, always taking its most
likely token. A reproduction is exact when they equal the document's next
CONTINUATION tokens. Held-out documents, never trained on, give the baseline:
text that is easy to predict (repeated lines, license headers) is continued
exactly without any memorization. Exact reproductions of training documents
well above the baseline mean the model can output pieces of its sources,
which may carry attribution obligations.

The documents are the same on every run: those whose id has the smallest
sha256, among documents long enough.
"""

import hashlib
import heapq
from collections import Counter

import torch

from wpt_eval import Result
from wpt_tokenizer import Tokenizer

PREFIX, CONTINUATION, DOCUMENTS, BATCH = 32, 32, 500, 64
PARTS = {"training": "training documents", "held_out": "held-out documents"}


@torch.no_grad()
def greedy(model, prompts, steps):
    """The most likely next token, steps times, for each row of prompts."""
    ids = prompts
    for _ in range(steps):
        logits = model(ids[:, -model.config.block_size:])[:, -1, :]
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[:, prompts.size(1):]


def measure(context, part):
    counts = Counter()

    def counted():
        for d in context.documents(part):
            counts["in_part"] += 1
            if len(d.tokens) >= PREFIX + CONTINUATION:
                counts["long_enough"] += 1
                yield d

    chosen = heapq.nsmallest(DOCUMENTS, counted(), key=lambda d: hashlib.sha256(d.id.encode()).digest())
    matched, exact = [], Counter()
    for start in range(0, len(chosen), BATCH):
        batch = chosen[start:start + BATCH]
        prompts = torch.tensor([[Tokenizer.END, *d.tokens[:PREFIX].tolist()] for d in batch], device=context.device)
        truth = torch.tensor([d.tokens[PREFIX:PREFIX + CONTINUATION].tolist() for d in batch], device=context.device)
        same = greedy(context.model, prompts, CONTINUATION) == truth
        lengths = same.int().cumprod(dim=1).sum(dim=1).tolist()  # tokens equal before the first difference
        matched += lengths
        exact.update(d.source for d, n in zip(batch, lengths) if n == CONTINUATION)
    tested = len(chosen)
    return {"documents": tested, "in_part": counts["in_part"], "long_enough": counts["long_enough"],
            "exact": sum(exact.values()),
            "exact_rate": sum(exact.values()) / tested if tested else None,
            "mean_matched_tokens": sum(matched) / tested if tested else None,
            "exact_by_source": dict(sorted(exact.items()))}


def run(context):
    data = {"prefix_tokens": PREFIX, "continuation_tokens": CONTINUATION, "documents_per_part": DOCUMENTS}
    lines = [f"Prefix of {PREFIX} tokens from the start of a document, then {CONTINUATION} tokens chosen "
             "greedily, compared with the document's next tokens.", ""]
    for part, label in PARTS.items():
        found = data[part] = measure(context, part)
        if not found["in_part"]:
            lines.append(f"- {label}: none in this dataset")
            continue
        if not found["documents"]:
            lines.append(f"- {label}: none has {PREFIX + CONTINUATION} tokens or more")
            continue
        lines.append(f"- {label}: {found['exact']} of {found['documents']} reproduced exactly "
                     f"({found['exact_rate']:.1%}); {found['mean_matched_tokens']:.1f} tokens equal on average "
                     "before the first difference")
    by_source = data["training"]["exact_by_source"]
    if by_source:
        lines += ["", "Exact reproductions of training documents, by source: "
                  + ", ".join(f"{sid} {count}" for sid, count in by_source.items())]
    return Result(data, "\n".join(lines) + "\n")
