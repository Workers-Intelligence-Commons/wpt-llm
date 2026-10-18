"""Several processes, one per GPU (or several on the CPU), that split each
window between them: DeepSeek-V4's contextual parallelism (docs/training.md,
"Long windows"). Every process holds a full copy of the model and takes the
same windows; process r trains the r-th of equal, consecutive slices of
each, and the model exchanges what crosses the slices (wpt_model's
sequence.Split). The processes' gradients add up to the whole window's,
so a window too long for one GPU trains exactly as it would on one. They
are started like ddp's:

    wpt-train RECIPE DATASET_ID --strategy split --processes 2 ...
    torchrun --nproc-per-node 2 -m wpt_train RECIPE DATASET_ID --strategy split ...

Each window's length must split into slices of a multiple of 128 tokens.
GPUs talk through NCCL (RCCL on AMD), the CPU through gloo.
"""

import contextlib
import os

import torch
import torch.distributed as dist

from wpt_training import PluginError
from wpt_training.strategies.ddp import DDP


class Split(DDP):
    """ddp's processes, without DistributedDataParallel: gradients are summed
    once the step's are complete."""

    splits_windows = True

    def wrap(self, model):
        return model

    def no_sync(self, wrapped):
        return contextlib.nullcontext()

    def synchronize(self, model):
        """Each gradient becomes the sum of every process's: the slices'
        shares of the whole window's. A weight that no process gave a
        gradient keeps none, as on one process."""
        parameters = list(model.parameters())
        found = torch.tensor([p.grad is not None for p in parameters], dtype=torch.uint8, device=self.device)
        dist.all_reduce(found, op=dist.ReduceOp.MAX)
        for p, has in zip(parameters, found.tolist()):
            if has:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)


def start():
    if "LOCAL_RANK" not in os.environ:
        raise PluginError("split: start several processes, with --processes N or torchrun")
    return Split()
