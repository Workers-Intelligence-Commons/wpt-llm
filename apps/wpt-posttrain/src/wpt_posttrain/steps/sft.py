"""Supervised fine-tuning: teach the model to answer as the assistant.

Reads the conversations of every admitted source whose `chat` files are
tagged `sft` (docs/chat-format.md), turns each one into tokens with the chat
template (wpt_tokenizer.chat), and trains the model to predict the tokens of
the assistant's messages only: thinking, text, tool calls and the closing
<|im_end|>. The other turns are context the model reads and does not learn
to write. The CSA indexers keep learning by their own loss, as in
pre-training (docs/model.md, "Attention"): its gradient joins the step's,
and its value is shown beside the loss.

Each conversation must fit the model's context (block_size + 1 tokens, as
in pre-training); a longer one stops the step, since cutting it would drop
the answers it holds. The settings below are the step's whole configuration;
to change them, copy this file and give its path to wpt-posttrain.
"""

import math

import torch
import torch.nn.functional as F

from wpt_posttrain import Outcome
from wpt_tokenizer import chat

EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
BATCH = 8  # conversations per update
SEED = 0  # the order of the conversations in each epoch
IGNORED = -100  # target of a token the model does not learn


def run(context):
    found = context.conversations()
    if not found:
        return Outcome.failure("no conversations: no admitted source has chat files tagged sft")
    limit = context.model.config.block_size + 1
    examples = []
    for conversation_id, conversation in found:
        ids, learn = chat.encode(context.tokenizer, conversation)
        if len(ids) > limit:
            return Outcome.failure(f"{conversation_id}: {len(ids)} tokens, more than the model's context "
                                   f"({limit} with the first token)")
        if not any(learn[1:]):
            return Outcome.failure(f"{conversation_id}: no assistant message to learn from")
        examples.append((ids, learn))

    model, device = context.model, context.device
    context.telemetry.event("sft.settings", epochs=EPOCHS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                            batch=BATCH, seed=SEED, conversations=len(examples),
                            tokens=sum(len(ids) for ids, _ in examples))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order = torch.Generator().manual_seed(SEED)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses, indexer_losses = [], []
        permutation = torch.randperm(len(examples), generator=order).tolist()
        for start in range(0, len(examples), BATCH):
            inputs, targets = _batch([examples[i] for i in permutation[start:start + BATCH]], device)
            logits, indexers = model(inputs, indexer_loss=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORED)
            if not math.isfinite(loss.item()):
                model.eval()
                return Outcome.failure(f"epoch {epoch}: the loss is {loss.item()}; lower LEARNING_RATE")
            optimizer.zero_grad(set_to_none=True)
            (loss + (indexers - indexers.detach())).backward()  # the indexers' gradient, not their value
            optimizer.step()
            losses.append(loss.item())
            indexer_losses.append(indexers.item())
            context.telemetry.event("sft.update", values={"loss": loss.item(), "indexer_loss": indexer_losses[-1]},
                                    epoch=epoch, update=len(losses), conversations=len(inputs))
        context.telemetry.log(f"sft epoch {epoch}/{EPOCHS}: mean loss {sum(losses) / len(losses):.4f}, indexer loss "
                              f"{sum(indexer_losses) / len(indexer_losses):.4f} over {len(examples)} conversations")
    model.eval()
    return Outcome.success()


def _batch(examples, device):
    """(inputs, targets) padded to the longest example: each target is the
    next token where the model learns it, IGNORED elsewhere. Padding sits
    after every real token, so causal attention never lets it change them."""
    width = max(len(ids) for ids, _ in examples) - 1
    inputs = torch.zeros(len(examples), width, dtype=torch.long)
    targets = torch.full((len(examples), width), IGNORED, dtype=torch.long)
    for row, (ids, learn) in enumerate(examples):
        inputs[row, :len(ids) - 1] = torch.tensor(ids[:-1])
        targets[row, :len(ids) - 1] = torch.tensor([t if keep else IGNORED for t, keep in zip(ids[1:], learn[1:])])
    return inputs.to(device), targets.to(device)
