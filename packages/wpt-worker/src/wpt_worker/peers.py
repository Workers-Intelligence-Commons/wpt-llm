"""The ddp strategy for a run whose gpu workers are in different places
(docs/machines.md, "Several machines"): the same synced training as one
machine (each update waits for every machine's gradients), with the
gradients exchanged directly between the gpu workers over TLS, through
each worker's own port and token, since NCCL's ports have no password and
never cross the public internet.

On each machine, the job's processes share their gradients as ddp does
(DistributedDataParallel over NCCL, or gloo on the CPU, on this machine
only). Then one process per machine, its first, exchanges them with the
other machines' first processes (Exchange): gpu worker 0's job is the
hub, and every other worker's job reaches it at its address, its
certificate pinned, with the run's own secret (the manager makes one per
run and gives it, with worker 0's address and pin, in each job's spec;
worker 0's token never leaves the manager). The hub adds the machines' values up in the order of their
numbers and sends the sum back to each; each machine then shares it with
its own processes. So a gradient is the mean over every process of every
machine, as on one machine with as many processes; only the order of the
additions differs, and with it the last bits.

What crosses between machines is a JSON header and bytes, a tensor's
bytes: for broadcast, the main process's value as a JSON tree and each of
its tensors' bytes (packed()), so no pickle crosses and nothing that
arrives is run.

The job sets, for each of its processes (wpt_worker.training):

    WPT_MACHINE, WPT_MACHINES   this machine's number in the run, and how many there are
    WPT_PEER_HUB                on every machine but number 0, gpu worker 0's key with the run's
                                secret in place of its token
    LOCAL_RANK, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT   this machine's own processes
                                (wpt_training.launch)
"""

import os
import threading

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from wpt_training import Strategy

from .link import PART, Key, LinkError, dial, receive, send
from .server import PEER_WAIT

MACHINE, MACHINES, HUB = "WPT_MACHINE", "WPT_MACHINES", "WPT_PEER_HUB"
OPS = ("sum", "max")


def _bytes(tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def _tensor(data, dtype) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(dtype)


class Exchange:
    """The link between the first processes of the run's machines: on
    machine 0 (the hub) one socket per other machine, in their order; on
    every other machine one socket to the hub. LinkError when a machine's
    job is gone or sends what was not asked."""

    def __init__(self, machine, machines, sockets):
        self.machine, self.machines, self.sockets = machine, machines, sockets

    @classmethod
    def open(cls, machine, machines, environ) -> "Exchange":
        if machine == 0:
            return cls(0, machines, _accepted(machines, environ))
        hub = Key.parse(environ[HUB])
        return cls(machine, machines, [dial(hub, peer=machine, answer=PEER_WAIT + 60)])

    def reduce(self, tensor, op="sum") -> torch.Tensor:
        """The elementwise sum or maximum of tensor over the machines, on the
        CPU, in the shape and type of tensor; part by part."""
        flat = tensor.detach().reshape(-1).cpu()
        out = torch.empty_like(flat)
        step = max(1, PART // flat.element_size())
        for start in range(0, flat.numel(), step):
            part = flat[start:start + step]
            out[start:start + step] = self._reduce(part, op)
        return out.reshape(tensor.shape)

    def _reduce(self, part, op):
        dtype, count = str(part.dtype).removeprefix("torch."), part.numel()
        if self.machine:
            send(self.sockets[0], {"kind": "reduce", "op": op, "dtype": dtype, "count": count}, _bytes(part))
            header, body = self._expect(self.sockets[0], "reduced", count)
            return _tensor(body, part.dtype)
        total = part.clone()
        for sock in self.sockets:  # machine 1, 2, ...: the same order at every update
            header, body = self._expect(sock, "reduce", count)
            if header.get("op") != op or header.get("dtype") != dtype:
                raise LinkError(f"a gpu worker's job asked {header.get('op')} of {header.get('dtype')} where this "
                                f"one does {op} of {dtype}: the machines' jobs differ")
            other = _tensor(body, part.dtype)
            total = total + other if op == "sum" else torch.maximum(total, other)
        data = _bytes(total)
        for sock in self.sockets:
            send(sock, {"kind": "reduced", "count": count}, data)
        return total

    def broadcast(self, value):
        """The hub's value, on every machine (value is ignored elsewhere):
        tensors, None, booleans, numbers, strings, lists, tuples and dicts
        of these; the tensors arrive on the CPU. It crosses as a JSON tree
        (packed()) and each tensor's bytes, one tensor at a time."""
        if self.machine:
            header, _ = self._expect(self.sockets[0], "broadcast")
            tensors = [self._tensor_in(*found) for found in header["tensors"]]
            return unpacked(header["value"], tensors)
        tensors = []
        tree = packed(value, tensors)
        listed = [[str(t.dtype).removeprefix("torch."), list(t.shape), t.numel() * t.element_size()] for t in tensors]
        for sock in self.sockets:
            send(sock, {"kind": "broadcast", "value": tree, "tensors": listed})
            for tensor in tensors:
                data = _bytes(tensor)
                for start in range(0, len(data), PART):
                    send(sock, {"kind": "part"}, data[start:start + PART])
        return value

    def _tensor_in(self, dtype, shape, size):
        data, filled = bytearray(size), 0
        while filled < size:
            _, body = self._expect(self.sockets[0], "part")
            if filled + len(body) > size:
                raise LinkError("a gpu worker's job sent more of a tensor than its size: the machines' jobs differ")
            data[filled:filled + len(body)] = body
            filled += len(body)
        dtype = getattr(torch, dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise LinkError("a gpu worker's job sent a tensor of no known type: the machines' jobs differ")
        return torch.frombuffer(data, dtype=torch.uint8).view(dtype).reshape(shape) if size else \
            torch.empty(shape, dtype=dtype)

    def barrier(self):
        self.reduce(torch.zeros(1))

    def close(self):
        for sock in self.sockets:
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _expect(sock, kind, count=None):
        header, body = receive(sock)
        if header.get("kind") != kind or (count is not None and header.get("count") != count):
            raise LinkError(f"a gpu worker's job sent {header.get('kind')!r} where {kind!r} was due: the machines' "
                            "jobs differ")
        return header, body


def packed(value, tensors: list):
    """value as a JSON tree that says each part's type, its tensors
    appended to tensors and named by their place there; TypeError for a
    part of another type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return ["v", value]
    if isinstance(value, torch.Tensor):
        tensors.append(value)
        return ["tensor", len(tensors) - 1]
    if isinstance(value, (list, tuple)):
        return ["list" if isinstance(value, list) else "tuple", [packed(item, tensors) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[packed(key, tensors), packed(item, tensors)] for key, item in value.items()]]
    raise TypeError(f"a {type(value).__name__} cannot cross between gpu workers: tensors, numbers, strings, lists, "
                    "tuples and dicts only")


def unpacked(tree, tensors: list):
    """The value packed() made tree of, with its tensors."""
    kind, content = tree
    if kind == "v":
        return content
    if kind == "tensor":
        return tensors[content]
    if kind == "list":
        return [unpacked(item, tensors) for item in content]
    if kind == "tuple":
        return tuple(unpacked(item, tensors) for item in content)
    if kind == "dict":
        return {unpacked(key, tensors): unpacked(item, tensors) for key, item in content}
    raise LinkError(f"a gpu worker's job sent a value of kind {kind!r}: the machines' jobs differ")


def _accepted(machines, environ) -> list:
    """On machine 0: the connection of every other machine's job, taken
    from this worker's server (peer.accept), in their order."""
    from .job import connect_socket

    found, errors = [None] * (machines - 1), []

    def take(number):
        try:
            sock = connect_socket(environ, {"kind": "peer.accept", "peer": number})
            sock.settimeout(None)
            found[number - 1] = sock
        except (OSError, LinkError) as error:
            errors.append(f"gpu worker {number}: {error}")

    threads = [threading.Thread(target=take, args=(number,), daemon=True) for number in range(1, machines)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        for sock in found:
            if sock is not None:
                sock.close()
        raise LinkError("the other gpu workers' jobs did not all reach this one: " + "; ".join(sorted(errors)))
    return found


class Bridged(Strategy):
    """ddp's processes on each machine, the machines bridged by Exchange;
    see the module's docstring."""

    def __init__(self, environ=None):
        environ = os.environ if environ is None else environ
        local_rank, local_world = int(environ["LOCAL_RANK"]), int(environ["LOCAL_WORLD_SIZE"])
        self.machine, self.machines = int(environ[MACHINE]), int(environ[MACHINES])
        if torch.cuda.device_count():
            device = torch.device("cuda", local_rank % torch.cuda.device_count())
            torch.cuda.set_device(device)
            backend = "nccl" if dist.is_nccl_available() else "gloo"
        else:
            device, backend = torch.device("cpu"), "gloo"
        # this machine's processes only: their own addresses, which wpt_training.launch gives
        dist.init_process_group(backend, init_method=f"tcp://{environ['MASTER_ADDR']}:{environ['MASTER_PORT']}",
                                rank=local_rank, world_size=local_world)
        super().__init__(device, self.machine * local_world + local_rank, self.machines * local_world, local_rank)
        self.exchange = Exchange.open(self.machine, self.machines, environ) if local_rank == 0 else None

    def wrap(self, model):
        return DistributedDataParallel(model, device_ids=[self.device.index] if self.device.type == "cuda" else None)

    def no_sync(self, wrapped):
        return wrapped.no_sync()

    def _across(self, tensor, op="sum") -> torch.Tensor:
        """tensor, the same in each of this machine's processes, combined
        over the machines; in every process."""
        if self.exchange is not None:
            tensor.copy_(self.exchange.reduce(tensor, op).to(tensor.device))
        dist.broadcast(tensor, src=0)
        return tensor

    def synchronize(self, model):
        """Each machine's mean gradient (DistributedDataParallel made it the
        same in its processes) becomes the mean over the machines. A weight
        no machine gave a gradient keeps none, as on one machine."""
        parameters = list(model.parameters())
        found = torch.tensor([p.grad is not None for p in parameters], dtype=torch.float32, device=self.device)
        self._across(found, "max")
        grads = []
        for p, has in zip(parameters, found.tolist()):
            if has:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                grads.append(p.grad)
        for dtype in sorted({grad.dtype for grad in grads}, key=str):
            group = [grad for grad in grads if grad.dtype == dtype]
            flat = torch.cat([grad.reshape(-1) for grad in group])
            self._across(flat, "sum")
            flat /= self.machines
            start = 0
            for grad in group:
                grad.copy_(flat[start:start + grad.numel()].view_as(grad))
                start += grad.numel()

    def mean(self, value):
        total = torch.tensor([value], dtype=torch.float64, device=self.device)
        dist.all_reduce(total)
        return self._across(total).item() / self.world_size

    def reduce(self, tensor, op="sum"):
        dist.all_reduce(tensor, op={"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op])
        return self._across(tensor, op)

    def broadcast(self, value):
        if self.exchange is not None:
            value = self.exchange.broadcast(value)
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def barrier(self):
        dist.barrier()
        if self.exchange is not None:
            self.exchange.barrier()
        dist.barrier()

    def close(self):
        dist.destroy_process_group()
        if self.exchange is not None:
            self.exchange.close()
