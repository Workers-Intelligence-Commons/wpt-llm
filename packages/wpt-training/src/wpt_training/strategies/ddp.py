"""Several processes, one per GPU (or several on the CPU), on one machine or
several; each holds a full copy of the model and their gradients are
averaged (PyTorch's DistributedDataParallel). They are started by
wpt-train's --processes on one machine, or by torchrun on one or several:

    wpt-train RECIPE DATASET_ID --strategy ddp --processes 2 ...
    torchrun --nproc-per-node 2 -m wpt_train RECIPE DATASET_ID --strategy ddp ...

GPUs talk through NCCL (RCCL on AMD), the CPU through gloo. Windows has
gloo only, and torchrun does not start there with PyTorch 2.14 (use
--processes).
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from wpt_training import PluginError, Strategy


class DDP(Strategy):
    def __init__(self):
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.device_count():  # not is_available(): with every GPU hidden, Windows still says yes
            device = torch.device("cuda", local_rank % torch.cuda.device_count())
            torch.cuda.set_device(device)
            backend = "nccl" if dist.is_nccl_available() else "gloo"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend)
        super().__init__(device, dist.get_rank(), dist.get_world_size(), local_rank)

    def wrap(self, model):
        return DistributedDataParallel(model, device_ids=[self.device.index] if self.device.type == "cuda" else None)

    def no_sync(self, wrapped):
        return wrapped.no_sync()

    def mean(self, value):
        total = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(total)
        return total.item() / self.world_size

    def reduce(self, tensor, op="sum"):
        dist.all_reduce(tensor, op={"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op])
        return tensor

    def broadcast(self, value):
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def barrier(self):
        dist.barrier()

    def close(self):
        dist.destroy_process_group()


def start():
    if "LOCAL_RANK" not in os.environ:
        raise PluginError("ddp: start several processes, with --processes N or torchrun")
    return DDP()
