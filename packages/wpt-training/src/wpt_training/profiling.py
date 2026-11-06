"""The profile of one update, for `wpt train --profile-update N`: the update
runs under torch.profiler and the run's record receives `profile.txt`."""

import contextlib
import os
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from wpt_utils.lines import WARNING

FILE = "profile.txt"
ROWS = 40
NAME_WIDTH = 100
GIB = 1024**3


class _Profiling:
    """The profiler and the time stamps of one update. The stamps come from
    wrappers on the calls that split the update (the batch, each pass's
    learn, the optimizer's step), which stand on the objects only until
    `finish`; an update that is not profiled never meets them."""

    def __init__(self, device, calls):
        self.cuda = device.type == "cuda"
        self.calls, self.stamps, self.passes, self.kept = calls, [], 0, []
        activities = [ProfilerActivity.CPU, *([ProfilerActivity.CUDA] if self.cuda else [])]
        self.profiler = profile(activities=activities)

    def stamp(self, label):
        self.sync()
        self.stamps.append((label, time.monotonic()))

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def wrapper(self, original, label):
        def stamped(*args, **options):
            try:
                return original(*args, **options)
            finally:
                if label == "learn":
                    self.passes += 1
                self.stamp(f"learn {self.passes}" if label == "learn" else label)
        return stamped

    def start(self):
        for owner, name, label in self.calls:
            own = vars(owner).get(name)
            setattr(owner, name, self.wrapper(getattr(owner, name), label))
            self.kept.append((owner, name, own))
        self.sync()
        self.profiler.start()
        self.stamps.append(("start", time.monotonic()))

    def unwrap(self):
        for owner, name, own in self.kept:
            if own is None:
                delattr(owner, name)
            else:
                setattr(owner, name, own)

    def finish(self, number, done):
        """Stop, and give the text of the profile when the update `done`."""
        try:
            if done:
                self.stamp("end")
        finally:
            self.unwrap()
            self.profiler.stop()
        return self.text(number) if done else None

    def text(self, number):
        events = self.profiler.key_averages()
        device = [event for event in self.profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        seconds = self.stamps[-1][1] - self.stamps[0][1]
        parts = [f"update {number}: {seconds:.3f} s wall, "
                 f"{sum(event.self_device_time_total for event in events) / 1e6:.3f} s of self device time, "
                 f"{len(device)} device kernel entries", "", "seconds by part:"]
        # what comes after the last stamp of a call up to the next one belongs to the next call
        names = [label for label, _ in self.stamps[1:-1]] + ["everything after"]
        names = [{"data": "data (environment.batch)", "optimizer step": "optimizer step (with the gradient "
                  "exchange before it)"}.get(name, name) for name in names]
        parts += [f"  {name}: {end - begin:.3f}" for name, (_, begin), (_, end)
                  in zip(names, self.stamps, self.stamps[1:])]
        parts += ["", "machine:", *machine(), "", "memory:", *memory(self.cuda)]
        if self.cuda:
            parts += ["", f"top {ROWS} by self CUDA time:", self.table(events, "self_device_time_total")]
        parts += ["", f"top {ROWS} by self CPU time:", self.table(events, "self_cpu_time_total")]
        return "\n".join(parts) + "\n"

    @staticmethod
    def table(events, order):
        return events.table(sort_by=order, row_limit=ROWS, max_name_column_width=NAME_WIDTH)


def machine() -> list[str]:
    facts = {"os.cpu_count()": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        facts["cpus this process may use"] = len(os.sched_getaffinity(0))
    with contextlib.suppress(OSError):
        facts["/sys/fs/cgroup/cpu.max"] = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
    facts["torch"], facts["torch.version.cuda"] = torch.__version__, torch.version.cuda
    try:
        import triton
        facts["triton"] = triton.__version__
    except ImportError:
        facts["triton"] = "not installed"
    if torch.cuda.is_available():
        facts["gpu"] = torch.cuda.get_device_name()
        for name, read in (("clock (MHz)", torch.cuda.clock_rate), ("power draw (mW)", torch.cuda.power_draw)):
            with contextlib.suppress(Exception):  # these need a library the machine may not have
                facts[name] = read()
    return [f"  {name}: {value}" for name, value in facts.items()]


def memory(cuda) -> list[str]:
    if not cuda:
        return ["  no CUDA device"]
    stats = torch.cuda.memory_stats()
    return [f"  num_alloc_retries: {stats['num_alloc_retries']}", f"  num_ooms: {stats['num_ooms']}",
            f"  peak allocated: {stats['allocated_bytes.all.peak'] / GIB:.2f} GiB",
            f"  peak reserved: {stats['reserved_bytes.all.peak'] / GIB:.2f} GiB",
            f"  segment allocations (cudaMalloc): {stats['segment.all.allocated']}"]


@contextlib.contextmanager
def profiled(number, device, record, log, environment, method, optimizer):
    """Run the block, one update, under the profiler, and give the run's
    record `profile.txt`. A failure of the profiler is a warning, and the
    update goes on; the update's own failure goes on as itself. `record`:
    the training loop's guarded record."""
    run = None
    try:
        run = _Profiling(device, ((environment, "batch", "data"), (method, "learn", "learn"),
                                  (optimizer.optimizer, "step", "optimizer step")))
        run.start()
    except Exception as error:
        log(f"profile: update {number} is not profiled: {type(error).__name__}: {error}", WARNING)
        if run is not None:
            with contextlib.suppress(Exception):
                run.finish(number, False)
        run = None
    done = False
    try:
        yield
        done = True
    finally:
        text = None
        if run is not None:
            try:
                text = run.finish(number, done)
            except Exception as error:
                log(f"profile: update {number} is not profiled: {type(error).__name__}: {error}", WARNING)
    if text is not None:
        record.guard(record.telemetry.write_file)(FILE, text.encode("utf-8"))
