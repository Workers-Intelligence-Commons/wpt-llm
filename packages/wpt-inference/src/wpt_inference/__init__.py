"""Autoregressive sampling: feed the tokens so far, pick the next one, repeat.

Works with any model that maps (B, T) token ids to (B, T, vocab) logits.
The loop has no idea what the tokens mean; the caller decides when to stop
by breaking out of the iteration.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_tokens(model, ids, *, context_size, max_new_tokens, temperature=0.8, top_k=40):
    """Yield up to max_new_tokens sampled token ids, one at a time.

    ids: the starting token ids (a list of ints, at least one).
    context_size: how many trailing tokens the model sees at each step.
    temperature: >1 = more inventive, <1 = more predictable.
    top_k: sample only among the k highest-scoring tokens (None = all).
    """
    device = next(model.parameters()).device
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float("-inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, next_id], dim=1)
        yield int(next_id)
