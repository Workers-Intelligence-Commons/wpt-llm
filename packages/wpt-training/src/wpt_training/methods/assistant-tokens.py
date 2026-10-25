"""Learning to write the assistant's tokens of conversations (docs/training.md,
"The answer-as-assistant step").

Each conversation is read whole, and the loss is the next-token loss of
the tokens the assistant wrote only (thinking, text, tool calls and the
closing <|im_end|>): the other turns are context the model reads and does
not learn to write. The loss of an update is the mean over every such
token of its conversations, on every process. The CSA indexers keep
learning by their own loss, as in read-text (docs/model.md, "Attention"):
its gradient joins the update's, and its value is shown beside the loss.
The multi-token prediction block is not used, so it does not change.

A loss that is not finite stops the step (StepFailed): lower the
optimizer's lr.
"""

import math

import torch.nn.functional as F

from wpt_model import MULTI_TOKEN_PREDICTION
from wpt_training import Method, StepFailed

READS = "conversations"


class AssistantTokens(Method):
    unreached = (MULTI_TOKEN_PREDICTION,)

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        """examples: (inputs, targets, learned, weight): targets -100 where
        nothing is learned, learned the number of tokens to learn in the
        whole update over every process, weight 0 for a pass that learns
        nothing (its process had no conversation left), else 1."""
        inputs, targets, learned, weight = examples
        with quiet(True):
            logits, indexers = wrapped(inputs, indexer_loss=True)
            share = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                    reduction="sum") / learned
            # the processes' gradients are averaged: each process's share counts world_size times
            (share * strategy.world_size + weight * (indexers - indexers.detach()) / passes).backward()
        return share.detach(), (indexers.detach() if weight else None)

    def update_loss(self, losses, indexers, strategy):
        loss, indexer = super().update_loss(losses, indexers, strategy)
        if not math.isfinite(loss):  # the same in every process, which all stop
            raise StepFailed(f"the loss is {loss}; lower the optimizer's lr")
        return loss, indexer


def start(settings):
    return AssistantTokens()
