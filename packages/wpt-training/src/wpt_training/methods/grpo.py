"""Group relative policy optimization on scored attempts (docs/training.md,
"The finish-tasks step").

The loss is minus each attempt's advantage times the log-probability of
each token the model wrote (never the instructions, the tool results or
the template's own tokens), over every token the model wrote in the
update's attempts. There is no frozen reference model and no KL penalty:
each update learns from attempts the current model just made. The routed
expert layers keep, during the loss pass, the experts their routers chose
while the model wrote the attempt (R3: wpt_model.Routes), so the loss
scores each token through the experts that wrote it; hash-routed layers
need nothing. The CSA indexers learn by their own loss, as in read-text
(docs/model.md, "Attention"). The multi-token prediction block is not
used, so it does not change.

A loss that is not finite stops the step (StepFailed): lower the
optimizer's lr.
"""

import math

import torch.nn.functional as F

from wpt_model import MULTI_TOKEN_PREDICTION
from wpt_training import Method, StepFailed

READS = "attempts"


class GRPO(Method):
    unreached = (MULTI_TOKEN_PREDICTION,)

    def learn(self, model, wrapped, examples, passes, strategy, segment, quiet):
        """examples: one attempt, (inputs, targets, mask, routes, advantage,
        count, weight): mask 1 where the model wrote the target, count the
        tokens the model wrote in the update's attempts, weight 0 for a pass
        that learns nothing (its process had no attempt left), else 1."""
        inputs, targets, mask, routes, advantage, count, weight = examples
        with quiet(True):
            logits, indexer = wrapped(inputs, replay=routes, indexer_loss=True)
            chosen = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            loss = -weight * advantage * (chosen * mask).sum() / count
            # the processes' gradients are averaged: each attempt counts world_size times
            ((loss + weight * (indexer - indexer.detach())) * strategy.world_size).backward()
        return loss.detach(), (indexer.detach() if weight else None)

    def update_loss(self, losses, indexers, strategy):
        loss, indexer = super().update_loss(losses, indexers, strategy)
        if not math.isfinite(loss):  # the same in every process, which all stop
            raise StepFailed(f"the loss is {loss}; lower the optimizer's lr")
        return loss, indexer


def start(settings):
    return GRPO()
