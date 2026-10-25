"""How much parallel work this machine takes, the one rule every WPT place
that runs items side by side sizes itself with.

The rule: at most the fewer of 75% of the cores and 75% of the available
memory divided by one item's measured memory, at least 1 (limit()). A
quarter of the machine stays free: before each new item, Sizing.may_start
measures again and lets it start only while the available memory and the
idle cores, once the item is counted, stay above a quarter of the
machine's. Items started too recently to show in the measures yet count
as taking their share already. Running items are never stopped, so the
work shrinks when something else takes the machine and grows back, up to
the limit, when it leaves. in_order runs items that way and gives their results in the
items' order, so a result never depends on how many ran at once. each runs
them on a pool of threads of its own and turns the errors an item may fail
with into that item's result (Done).
"""

import collections
import ctypes
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

SHARE = 0.75  # the share of the cores and of the memory parallel work may take; the rest stays free
SAMPLE = 0.1  # seconds: the shortest span idle cores are measured over, so two measures never come too close
POLL = 1.0  # seconds: how often waiting work measures again whether a new item may start
SETTLE = 2.0  # seconds: how long a started item counts as not yet showing in the measures (a new process and its imports)
_END = object()  # what next() gives once the items run out


def memory():
    """(available, total) bytes of memory, or None when this machine does
    not say. Available is what new processes can take: MemAvailable on
    Linux, free, inactive and speculative pages on macOS, ullAvailPhys on
    Windows."""
    try:
        if sys.platform == "linux":
            fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
            return int(fields["MemAvailable"].split()[0]) * 1024, int(fields["MemTotal"].split()[0]) * 1024
        if sys.platform == "darwin":
            from wpt_utils.programs import MissingTool, program
            try:
                lines = subprocess.run([program("vm_stat")], capture_output=True, text=True,
                                       check=True).stdout.splitlines()
                total = int(subprocess.run([program("sysctl"), "-n", "hw.memsize"], capture_output=True, text=True,
                                           check=True).stdout)
            except MissingTool:
                return None
            page = int(lines[0].split("page size of")[1].split()[0])
            pages = {name.strip(): int(count.strip(" .")) for name, count in (line.split(":", 1) for line in lines[1:])
                     if count.strip(" .").isdigit()}
            return page * sum(pages[name] for name in ("Pages free", "Pages inactive", "Pages speculative")), total
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            *((name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys",
                              "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                              "ullAvailExtendedVirtual"))]
            status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys, status.ullTotalPhys
    except (OSError, ValueError, IndexError, KeyError, subprocess.CalledProcessError):
        pass
    return None


def _busy_times():
    """(idle, all) time the processors spent since the machine started, in
    the system's own unit, or None where the system keeps no such count:
    /proc/stat on Linux (iowait counts as idle), GetSystemTimes on Windows."""
    try:
        if sys.platform == "linux":
            with open("/proc/stat", encoding="utf-8") as stat:
                ticks = [int(value) for value in stat.readline().split()[1:9]]  # user .. steal; guest is in user
            return ticks[3] + ticks[4], sum(ticks)
        if sys.platform == "win32":
            idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))  # a FILETIME is 64 bits
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return idle.value, kernel.value + user.value  # kernel time includes the idle time
    except (OSError, ValueError, IndexError):
        pass
    return None


class Machine:
    """What this machine has free now: its cores, its memory, and how many
    cores are idle. Its state is the last processor count read and the
    idle cores measured with it, so idle cores are measured over the span
    since the previous measure."""

    def __init__(self):
        self.cores = os.cpu_count() or 1
        self._last = None  # (monotonic time, _busy_times()) of the previous measure
        self._idle = None  # the idle cores that measure gave

    def memory(self):
        return memory()

    def idle_cores(self):
        """How many cores were idle, on average, since the previous measure
        (over SAMPLE seconds for the first, waiting them; a call less than
        SAMPLE after the previous measure gives that measure again); None
        where the system does not say. Elsewhere than Linux and Windows (macOS), the cores the
        one-minute load average leaves."""
        if sys.platform not in ("linux", "win32"):
            try:
                return max(0.0, self.cores - os.getloadavg()[0])
            except (AttributeError, OSError):
                return None
        if self._last is None:
            self._last = (time.monotonic(), _busy_times())
            time.sleep(SAMPLE)
        elif time.monotonic() - self._last[0] < SAMPLE:
            return self._idle
        now = (time.monotonic(), _busy_times())
        before, self._last = self._last[1], now
        if before is None or now[1] is None or now[1][1] <= before[1]:
            self._idle = None
        else:
            self._idle = self.cores * (now[1][0] - before[0]) / (now[1][1] - before[1])
        return self._idle


def limit(item_memory, what="workers", machine=None):
    """(count, the line that says why): the fewer of 75% of the cores and
    75% of the available memory divided by item_memory (bytes, one item's
    measured memory), at least 1; the cores alone when the memory is
    unknown. what names the items in the line."""
    machine = machine or Machine()
    by_cores = max(1, int(machine.cores * SHARE))
    measured = machine.memory()
    if measured is None:
        return by_cores, f"{by_cores} {what}: {SHARE:.0%} of {machine.cores} cores (the available memory is unknown here)"
    available = measured[0]
    by_memory = max(1, int(available * SHARE / item_memory))
    count = min(by_cores, by_memory)
    return count, (f"{count} {what}: the fewer of {SHARE:.0%} of {machine.cores} cores ({by_cores}) and {SHARE:.0%} of "
                   f"{available / 1e9:.1f} GB available at {item_memory / 1e9} GB each ({by_memory})")


class Sizing:
    """The rule for one kind of item: its limit, fixed when the work starts,
    and may_start, measured again before each new item."""

    def __init__(self, item_memory, what="workers", machine=None):
        self.item_memory = item_memory
        self.machine = machine or Machine()
        self.limit, self.why = limit(item_memory, what, self.machine)

    def may_start(self, running, unseen=0):
        """Whether one more item may start while running items run, unseen
        of them started too recently to show in the measures: always the
        first; never past the limit; otherwise only while the available
        memory less one item's for the new one and each unseen one stays at
        or above a quarter of the memory, and the idle cores less one each
        at or above a quarter of the cores. A measure this machine does not
        give leaves the limit to decide."""
        if running == 0:
            return True
        if running >= self.limit:
            return False
        taken = 1 + unseen
        measured = self.machine.memory()
        if measured is not None and measured[0] - taken * self.item_memory < measured[1] * (1 - SHARE):
            return False
        idle = self.machine.idle_cores()
        return idle is None or idle - taken >= self.machine.cores * (1 - SHARE)


def in_order(work, items, executor, sizing, poll=POLL, settle=SETTLE, group=None):
    """Run work(item) for each item on executor (a concurrent.futures
    executor of at least sizing.limit threads or processes, which the
    caller owns), starting each item only when sizing.may_start allows,
    with the running items started less than settle seconds ago as its
    unseen ones, and yield the results in the items' order. group, when
    given, gives an item's (group, the most items of that group running at
    once), such as a site and its limit: an item whose group is full waits,
    and the items after it may start meanwhile. While no item may start, it
    measures again every poll seconds or as soon as an item ends. An item's
    exception is raised when its turn to be yielded comes; no new item
    starts after that, and the items still running finish on the
    executor."""
    items = iter(items)
    slots = collections.deque()  # one per item taken, in the items' order: [its future, None while it waits]
    waiting = collections.deque()  # (slot, item, group) of the items whose group was full, in the items' order
    running = {}  # each running item's future: (the monotonic time it started, its group)
    more = True

    def fits(name, most):
        return name is None or sum(running_group == name for _, running_group in running.values()) < most

    while more or slots:
        running = {future: value for future, value in running.items() if not future.done()}
        while sizing.may_start(len(running), sum(time.monotonic() - start < settle for start, _ in running.values())):
            chosen = next((waited for waited in waiting if fits(*waited[2])), None)
            if chosen is not None:
                waiting.remove(chosen)
            else:
                item = next(items, _END) if more else _END
                if item is _END:
                    more = False
                    break
                slot = [None]
                slots.append(slot)
                chosen = (slot, item, group(item) if group else (None, 0))
                if not fits(*chosen[2]):
                    waiting.append(chosen)
                    continue
            slot, item, (name, _) = chosen
            slot[0] = executor.submit(work, item)
            running[slot[0]] = (time.monotonic(), name)
        while slots and slots[0][0] is not None and slots[0][0].done():
            yield slots.popleft()[0].result()
        if running:
            wait(running, timeout=poll, return_when=FIRST_COMPLETED)


@dataclass(frozen=True)
class Done:
    """One item's result from each: the value work gave for it, or the error
    it failed with."""

    item: object
    value: object = None
    error: Exception | None = None


def each(work, items, errors, sizing, group=None):
    """Done of work(item) for each item, in the items' order, run on a pool
    of sizing.limit threads as in_order allows (group as there). An
    exception of errors (a class, or a tuple of them) that work raises is
    its item's error; any other is raised at its item's turn, as in_order
    raises it, and no new item starts."""
    def run(item):
        try:
            return Done(item, work(item))
        except errors as error:
            return Done(item, error=error)

    with ThreadPoolExecutor(sizing.limit) as executor:
        yield from in_order(run, items, executor, sizing, group=group)
