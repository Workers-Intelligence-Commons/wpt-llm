"""How much parallel work this machine takes, the one rule every WPT place
that runs items side by side sizes itself with.

The rule (Sizing): at most 75% of the cores at once (its limit, fixed when
the work starts), and before each new item it measures the machine again
and lets the item start only while, once the item is counted,

- the available memory stays at or above a quarter of the memory, and the
  running items with the new one take at most 75% of the memory the work
  has (what is available now, plus what its running items take);
- the idle cores stay at or above a quarter of the cores;
- with a disk to count, the free disk of the folder the items write to
  stays at or above a quarter of what was free when the work started once
  the new item's size is taken from it, the free disk being the lesser of
  what is free now and what was free at the start less the sizes of the
  running items (so what they already wrote is counted once).

Work that waits on the network rather than on the processors (a Sizing
given a network limit) takes that limit instead of the cores' share and
leaves the idle cores out; the memory and disk rules hold the same.

One item's memory is measured live, never given: what one running item was
last measured taking (Threads for items on this process's threads,
Processes for items in worker processes; each counts the programs an item
starts too).
Until a first measure exists, items start one at a time. Items started too
recently to show in the measures yet (SETTLE) count as taking one item's
memory and one core already. Running items are never stopped, so the work
shrinks when something else takes the machine and grows back, up to the
limit, when it leaves.

in_order runs items that way and gives their results in the items' order,
so a result never depends on how many ran at once. each runs them on a
pool of threads of its own and turns the errors an item may fail with into
that item's result (Done). process_pool makes a pool of worker processes
whose memory a Sizing measures.

limit() is the same rule for work whose count must be fixed before it
starts (the test processes of scripts/check.py), from one item's memory
measured beforehand (peak() measures a program run alone).
"""

import collections
import ctypes
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

SHARE = 0.75  # the share of the cores, of the memory and of the free disk parallel work may take; the rest stays free
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


def _processes() -> dict[int, tuple[int, int]] | None:
    """{pid: (parent pid, resident bytes)} of every process this machine
    lets this one see, or None where it does not say: /proc on Linux, ps on
    macOS, a process snapshot on Windows."""
    try:
        if sys.platform == "linux":
            page, found = os.sysconf("SC_PAGE_SIZE"), {}
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    with open(f"/proc/{name}/stat", "rb") as stat:
                        fields = stat.read().rsplit(b")", 1)[1].split()  # after the program's name, which may hold spaces
                    found[int(name)] = (int(fields[1]), int(fields[21]) * page)  # ppid, rss in pages
                except (OSError, IndexError, ValueError):
                    continue  # it ended meanwhile
            return found
        if sys.platform == "darwin":
            from wpt_utils.programs import MissingTool, program
            try:
                lines = subprocess.run([program("ps"), "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True,
                                       check=True).stdout.splitlines()
            except MissingTool:
                return None
            return {int(pid): (int(ppid), int(rss) * 1024) for pid, ppid, rss in (line.split() for line in lines)}
        if sys.platform == "win32":
            return _windows_processes()
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    return None


def _windows_processes():
    kernel = ctypes.windll.kernel32

    class Entry(ctypes.Structure):  # PROCESSENTRY32W
        _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong), ("th32ProcessID", ctypes.c_ulong),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", ctypes.c_ulong),
                    ("cntThreads", ctypes.c_ulong), ("th32ParentProcessID", ctypes.c_ulong),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", ctypes.c_ulong),
                    ("szExeFile", ctypes.c_wchar * 260)]

    class Counters(ctypes.Structure):  # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    *((name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                      "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                      "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"))]

    kernel.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel.OpenProcess.restype = ctypes.c_void_p
    snapshot = kernel.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, ctypes.c_void_p(-1).value):
        return None
    parents = {}
    try:
        entry = Entry(dwSize=ctypes.sizeof(Entry))
        more = kernel.Process32FirstW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
        while more:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            more = kernel.Process32NextW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
    finally:
        kernel.CloseHandle(ctypes.c_void_p(snapshot))
    found = {}
    for pid, parent in parents.items():
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            continue
        try:
            counters = Counters(cb=ctypes.sizeof(Counters))
            if kernel.K32GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb):
                found[pid] = (parent, counters.WorkingSetSize)
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    return found


def resident(pids) -> dict[int, int] | None:
    """{pid: resident bytes of that process and of every process it started,
    and they started, still running} for each of pids (0 for one that
    ended), or None where this machine does not say."""
    found = _processes()
    if found is None:
        return None
    children = collections.defaultdict(list)
    for pid, (parent, _) in found.items():
        if parent != pid:
            children[parent].append(pid)
    totals = {}
    for pid in pids:
        total, pending, seen = 0, [pid], set()
        while pending:
            here = pending.pop()
            if here in seen or here not in found:
                continue
            seen.add(here)
            total += found[here][1]
            pending += children[here]
        totals[pid] = total
    return totals


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
    """What this machine has free now: its cores, its memory, how many cores
    are idle and a folder's free disk. Its state is the last processor
    count read and the idle cores measured with it, so idle cores are
    measured over the span since the previous measure."""

    def __init__(self):
        self.cores = os.cpu_count() or 1
        self._last = None  # (monotonic time, _busy_times()) of the previous measure
        self._idle = None  # the idle cores that measure gave

    def memory(self):
        return memory()

    def free_disk(self, folder):
        """The bytes free on the disk that holds folder (or will: its nearest
        folder that exists), or None when it cannot be read."""
        folder = Path(folder).absolute()
        while not folder.exists() and folder != folder.parent:
            folder = folder.parent
        try:
            return shutil.disk_usage(folder).free
        except OSError:
            return None

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


class Threads:
    """The memory one item run on this process's threads takes: how much
    this process and the programs it started grew, while items ran, since
    the last moment no item ran, shared out between the running items."""

    def __init__(self):
        self._base = None  # resident bytes when no item ran last

    def __call__(self, running):
        found = resident([os.getpid()])
        if found is None:
            return None
        now = found[os.getpid()]
        if self._base is None:
            self._base = now
            return None
        per = max(0, now - self._base) / max(running, 1)
        if running == 0:
            self._base = now
        return per


class Processes:
    """The memory one item run in a pool's worker processes takes: the most
    any of its workers holds, with the programs it started (a worker runs
    one item at a time, and keeps what its items left)."""

    def __init__(self, pool):
        self.pool = pool

    def __call__(self, running):
        pids = list(getattr(self.pool, "_processes", None) or {})
        found = resident(pids) if pids else None
        return max(found.values()) if found else None


def process_pool(sizing, count, initializer=None, initargs=()):
    """A pool of at most count worker processes, started as each is needed
    (by spawning: alike on every system), whose memory sizing measures from
    now on (Processes). The caller owns it."""
    pool = ProcessPoolExecutor(max(1, count), mp_context=multiprocessing.get_context("spawn"),
                               initializer=initializer, initargs=initargs)
    sizing.measure = Processes(pool)
    return pool


def peak(command, poll=0.1, **options):
    """(the exit code, the most resident bytes the program and every program
    it started held at once, or None where this machine does not say) of
    command run to its end; options go to subprocess.Popen. The memory is
    read every poll seconds, so a peak shorter than that may be missed."""
    most = None
    with subprocess.Popen(command, **options) as process:
        while process.poll() is None:
            found = resident([process.pid])
            if found is not None:
                most = max(most or 0, found[process.pid])
            time.sleep(poll)
    return process.returncode, most


def limit(item_memory, what="workers", machine=None):
    """(count, the line that says why): the fewer of 75% of the cores and
    75% of the available memory divided by item_memory (bytes, one item's
    memory measured beforehand), at least 1; the cores alone when the
    memory, or one item's, is unknown (None). what names the items in the
    line. Only for work whose count is fixed before it starts; Sizing
    measures as it goes."""
    machine = machine or Machine()
    by_cores = max(1, int(machine.cores * SHARE))
    measured = machine.memory()
    if measured is None or item_memory is None:
        unknown = "the available memory" if measured is None else f"the memory of one of the {what}"
        return by_cores, f"{by_cores} {what}: {SHARE:.0%} of {machine.cores} cores ({unknown} is unknown here)"
    available = measured[0]
    by_memory = max(1, int(available * SHARE / item_memory))
    count = min(by_cores, by_memory)
    return count, (f"{count} {what}: the fewer of {SHARE:.0%} of {machine.cores} cores ({by_cores}) and {SHARE:.0%} of "
                   f"{available / 1e9:.1f} GB available at {item_memory / 1e9} GB each ({by_memory})")


class Sizing:
    """The rule for one kind of item: its limit, fixed when the work
    starts, and may_start, measured again before each new item. measure
    gives one item's memory now from how many run (Threads by default;
    process_pool sets Processes). disk, when given, gives the bytes an
    item writes to the disk of folder (default: the temporary folder's).
    network, when given, makes it work that waits on the network: at most
    that many items, whatever the cores."""

    def __init__(self, what="workers", machine=None, measure=None, disk=None, folder=None, network=None):
        self.machine = machine or Machine()
        self.measure = measure or Threads()
        self.disk, self.folder = disk, (folder or _temporary()) if disk else None
        self.free_at_start = self.machine.free_disk(self.folder) if disk else None
        self.item_memory = None  # what one running item was last measured taking
        self._measured = None  # the monotonic time of the last memory measure
        self.network = network
        if network is None:
            self.limit = max(1, int(self.machine.cores * SHARE))
            self.why = (f"at most {self.limit} {what} ({SHARE:.0%} of {self.machine.cores} cores); each more only "
                        f"while a quarter of the memory{', of the free disk' if disk else ''} and of the cores stays "
                        "free, at the memory a running one is measured taking")
        else:
            self.limit = max(1, network)
            self.why = (f"at most {self.limit} {what} (network work); each more only while a quarter of the memory"
                        f"{' and of the free disk' if disk else ''} stays free, at the memory a running one is "
                        "measured taking")

    def need(self, item) -> int:
        """The bytes item writes to the counted disk: 0 without disk, and 0
        when finding them fails (the item's own work meets that failure,
        and it becomes that item's result)."""
        if not self.disk:
            return 0
        try:
            return self.disk(item)
        except Exception:
            return 0

    def may_start(self, running, unseen=0, need=0, needs=0):
        """Whether one more item, writing need bytes to the counted disk,
        may start while running items run, writing needs bytes together,
        unseen of them started too recently to show in the measures:
        always the first; never past the limit; one at a time until one
        item's memory was measured; otherwise only while the available
        memory less one item's for the new one and each unseen one stays at
        or above a quarter of the memory, the running items and the new
        one take at most 75% of the memory the work has, the idle cores less
        one each stay at or above a quarter of the cores, and the free disk
        (the lesser of what is free now and what was free when the work
        started less needs) less need stays at or above a quarter of the
        free disk when the work started. Network work leaves the idle cores
        out. A measure this machine does not give leaves the others to
        decide."""
        # every running item shows in the measures: one item's memory is known; at most every SAMPLE seconds
        if unseen == 0 and (self._measured is None or time.monotonic() - self._measured >= SAMPLE):
            self._measured = time.monotonic()
            measured = self.measure(running)
            if measured is not None:
                self.item_memory = measured
        if running == 0:
            return True
        if running >= self.limit or self.item_memory is None:
            return False
        taken = 1 + unseen
        found = self.machine.memory()
        if found is not None:
            available, total = found
            if available - taken * self.item_memory < total * (1 - SHARE):
                return False
            if self.item_memory * (running + 1) > SHARE * (available + self.item_memory * (running - unseen)):
                return False
        if self.disk is not None and self.free_at_start is not None:
            free = self.machine.free_disk(self.folder)
            room = self.free_at_start - needs if free is None else min(free, self.free_at_start - needs)
            if room - need < self.free_at_start * (1 - SHARE):
                return False
        if self.network is not None:
            return True
        idle = self.machine.idle_cores()
        return idle is None or idle - taken >= self.machine.cores * (1 - SHARE)


def _temporary():
    import tempfile
    return tempfile.gettempdir()


def in_order(work, items, executor, sizing, poll=POLL, settle=SETTLE, group=None):
    """Run work(item) for each item on executor (a concurrent.futures
    executor of at least sizing.limit threads or processes, which the
    caller owns), starting each item only when sizing.may_start allows,
    with the running items started less than settle seconds ago as its
    unseen ones and the disk each writes (sizing.need), and yield the
    results in the items' order. group, when given, gives an item's
    (group, the most items of that group running at once), such as a site
    and its limit: an item whose group is full waits, and the items after
    it may start meanwhile. While no item may start, it measures again
    every poll seconds or as soon as an item ends. An item's exception is
    raised when its turn to be yielded comes; no new item starts after
    that, and the items still running finish on the executor."""
    items = iter(items)
    slots = collections.deque()  # one per item taken, in the items' order: [its future, None while it waits]
    waiting = collections.deque()  # (slot, item, group, need) of the items whose group was full, in the items' order
    running = {}  # each running item's future: (the monotonic time it started, its group, its need)
    drawn = None  # the next item taken from items, not yet started nor waiting
    more = True

    def fits(name, most):
        return name is None or sum(value[1] == name for value in running.values()) < most

    while more or slots:
        running = {future: value for future, value in running.items() if not future.done()}
        while True:
            chosen = next((waited for waited in waiting if fits(*waited[2])), None)
            if chosen is None:
                if drawn is None:
                    item = next(items, _END) if more else _END
                    if item is _END:
                        more = False
                        break
                    slot = [None]
                    slots.append(slot)
                    drawn = (slot, item, group(item) if group else (None, 0), sizing.need(item))
                if not fits(*drawn[2]):
                    waiting.append(drawn)
                    drawn = None
                    continue
                chosen = drawn
            unseen = sum(time.monotonic() - value[0] < settle for value in running.values())
            if not sizing.may_start(len(running), unseen, chosen[3], sum(value[2] for value in running.values())):
                break
            if chosen is drawn:
                drawn = None
            else:
                waiting.remove(chosen)
            slot, item, (name, _), need = chosen
            slot[0] = executor.submit(work, item)
            running[slot[0]] = (time.monotonic(), name, need)
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
