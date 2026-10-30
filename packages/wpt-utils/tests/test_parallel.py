import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from wpt_utils.parallel import Done, Machine, Processes, Sizing, Threads, each, in_order, limit, peak, process_pool

GB = 1e9


class Fake:
    """A machine whose measures the test sets."""

    def __init__(self, cores=16, available=64 * GB, total=64 * GB, idle=16.0, free=None):
        self.cores, self.available, self.total, self.idle, self.free = cores, available, total, idle, free

    def memory(self):
        return None if self.available is None else (self.available, self.total)

    def idle_cores(self):
        return self.idle

    def free_disk(self, folder):
        return self.free


def sizing_of(machine, item_memory=1 * GB, **more):
    """A Sizing on machine whose every measure finds one item taking item_memory."""
    return Sizing("test items", machine, measure=lambda running: item_memory, **more)


def test_the_limit_is_the_fewer_of_75_percent_of_the_cores_and_of_the_available_memory_per_item():
    assert limit(2.5 * GB, machine=Fake(cores=16, available=100 * GB))[0] == 12  # the cores decide
    count, why = limit(2.5 * GB, "test workers", Fake(cores=16, available=19.1 * GB))
    assert count == 5  # 75% of 19.1 GB is 14.3 GB: 5 items of 2.5 GB
    assert why == ("5 test workers: the fewer of 75% of 16 cores (12) and 75% of 19.1 GB available at 2.5 GB each (5)")
    assert limit(2.5 * GB, machine=Fake(cores=2, available=1 * GB))[0] == 1  # at least 1
    count, why = limit(2.5 * GB, "test workers", Fake(cores=8, available=None))
    assert (count, why) == (6, "6 test workers: 75% of 8 cores (the available memory is unknown here)")


def test_the_limit_is_the_cores_alone_when_one_item_s_memory_could_not_be_measured():
    count, why = limit(None, "test workers", Fake(cores=8, available=100 * GB))
    assert (count, why) == (6, "6 test workers: 75% of 8 cores (the memory of one of the test workers is unknown here)")


def test_a_new_item_starts_only_while_a_quarter_of_the_machine_stays_free():
    machine = Fake(cores=16, available=64 * GB, total=64 * GB, idle=16.0)
    sizing = Sizing("test items", machine, measure=lambda running: 2 * GB)
    assert sizing.limit == 12
    assert sizing.may_start(0) and sizing.may_start(5)
    assert not sizing.may_start(12)  # never past the limit
    machine.available = 17 * GB  # 15 GB left after one more item, under a quarter of 64 GB
    assert not sizing.may_start(5)
    assert sizing.may_start(0)  # the first item always starts
    machine.available, machine.idle = 64 * GB, 4.5  # 3.5 idle cores after one more, under a quarter of 16
    assert not sizing.may_start(5)
    machine.idle = 5.0  # something else ended: the work grows back
    assert sizing.may_start(5)
    machine.available, machine.idle = None, None  # measures this machine does not give: the limit decides
    assert sizing.may_start(11) and not sizing.may_start(12)


def test_network_work_takes_its_own_limit_and_busy_cores_do_not_stop_it():
    # uploads wait on the link, not on the processors: a busy machine still sends as many as the link wants
    machine = Fake(cores=16, available=64 * GB, total=64 * GB, idle=0.0)
    sizing = Sizing("test parts", machine, measure=lambda running: 0.1 * GB, network=32)
    assert sizing.limit == 32
    assert sizing.may_start(31) and not sizing.may_start(32)
    machine.available = 16.05 * GB  # the memory rule holds the same: 15.95 GB left is under a quarter of 64 GB
    assert not sizing.may_start(5)


def run(sizing, items, work, settle=0.0):
    """in_order's results, and the most items that ran at once."""
    lock, now, peak = threading.Lock(), [0], [0]

    def counted(item):
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        try:
            return work(item)
        finally:
            with lock:
                now[0] -= 1

    with ThreadPoolExecutor(sizing.limit) as executor:
        return list(in_order(counted, items, executor, sizing, poll=0.01, settle=settle)), peak[0]


def slow_square(item):
    time.sleep(0.02 * (item % 4))  # later items often end first
    return item * item


def test_results_come_in_the_items_order_as_one_at_a_time_gives_them():
    sizing = sizing_of(Fake(cores=8))
    results, peak = run(sizing, range(20), slow_square)
    assert results == [slow_square(item) for item in range(20)]
    assert 1 < peak <= sizing.limit == 6


def test_a_group_never_runs_more_items_than_its_limit_and_the_others_go_on_meanwhile():
    # downloads from one site stay under that site's limit, while the other
    # sites' downloads start past the ones waiting; results stay in the items' order
    items = [("slow.example", n) for n in range(8)] + [("other.example", n) for n in range(8)]
    lock, running, peak, started = threading.Lock(), {}, {}, []

    def work(item):
        site, n = item
        with lock:
            running[site] = running.get(site, 0) + 1
            peak[site] = max(peak.get(site, 0), running[site])
            started.append(item)
        time.sleep(0.02)
        with lock:
            running[site] -= 1
        return item

    sizing = sizing_of(Fake(cores=16))
    with ThreadPoolExecutor(sizing.limit) as executor:
        results = list(in_order(work, items, executor, sizing, poll=0.01, settle=0.0,
                                group=lambda item: (item[0], 2 if item[0] == "slow.example" else 8)))

    assert results == items
    assert peak == {"slow.example": 2, "other.example": peak["other.example"]} and peak["other.example"] > 2
    assert started.index(("other.example", 0)) < started.index(("slow.example", 7))  # it did not wait for them


def test_a_busy_machine_runs_one_item_at_a_time_and_still_finishes():
    sizing = sizing_of(Fake(cores=8, idle=0.0))
    results, peak = run(sizing, range(6), slow_square)
    assert results == [item * item for item in range(6)]
    assert peak == 1


def test_items_just_started_count_as_taking_their_share_before_the_measures_show_them():
    # 30 GB available of 64 GB at 2 GB each: only 7 items leave 16 GB, a quarter, free;
    # the fake's measures never show the started items, as at start-up before they take their memory
    sizing = sizing_of(Fake(cores=16, available=30 * GB, total=64 * GB), 2 * GB)
    results, peak = run(sizing, range(20), lambda item: time.sleep(0.2) or item, settle=60.0)  # long enough to overlap
    assert results == list(range(20))
    assert peak == 7


def test_instant_items_are_not_slowed_by_the_measures():
    # every start measures again; measuring must not wait SAMPLE (0.1 s) each time, which would take 20 s here
    start = time.monotonic()
    results, _ = run(Sizing("instant items"), range(200), lambda item: item)
    assert results == list(range(200))
    assert time.monotonic() - start < 5.0


def test_an_item_s_failure_is_raised_at_its_turn_after_the_results_before_it():
    def work(item):
        if item == 3:
            raise ValueError("item 3")
        return item

    got = []
    with ThreadPoolExecutor(4) as executor, pytest.raises(ValueError, match="item 3"):
        for result in in_order(work, range(10), executor, sizing_of(Fake(cores=4)), poll=0.01):
            got.append(result)
    assert got == [0, 1, 2]



def test_each_turns_the_errors_named_into_their_item_s_result_and_any_other_stops():
    def work(item):
        if item % 3 == 1:
            raise KeyError(item)
        if item == 8:
            raise ValueError("item 8")
        return slow_square(item)

    sizing = sizing_of(Fake(cores=4))
    got = []
    with pytest.raises(ValueError, match="item 8"):
        for done in each(work, range(10), (KeyError, IndexError), sizing):
            got.append(done)
    assert [(d.item, d.value, type(d.error)) for d in got] == [
        (n, None, KeyError) if n % 3 == 1 else (n, n * n, type(None)) for n in range(8)]
    assert got[0] == Done(0, 0)

def test_this_machine_s_measures_are_in_range():
    machine = Machine()
    measured = machine.memory()
    if measured is not None:
        assert 0 < measured[0] <= measured[1]
    idle = machine.idle_cores()
    if idle is not None:
        assert 0 <= idle <= machine.cores + 0.5  # the processor counts and the clock are read apart


class Filling(Fake):
    """A machine of 64 GB whose available memory falls by what each running item takes."""

    def __init__(self, per_item, cores=16):
        super().__init__(cores=cores)
        self.per_item, self.running, self.lock = per_item, 0, threading.Lock()

    def memory(self):
        return self.total - self.running * self.per_item, self.total

    def measure(self, running):
        return self.per_item

    def work(self, item):
        with self.lock:
            self.running += 1
        time.sleep(0.4)
        with self.lock:
            self.running -= 1
        return item


@pytest.mark.parametrize("per_item, most", [(8 * GB, 6), (2 * GB, 12)])
def test_the_memory_a_running_item_is_measured_taking_decides_how_many_start(per_item, most):
    # no figure is given per item: 8 GB measured each on 64 GB leaves 16 GB free with 6 running;
    # at 2 GB each, the 12 the cores allow fit
    machine = Filling(per_item)
    # an item counts as unseen for 0.1 s, the time its thread takes to show in the fake's memory
    results, peak = run(Sizing("test items", machine, measure=machine.measure), range(24), machine.work, settle=0.1)
    assert results == list(range(24))
    assert peak == most


def test_items_start_one_at_a_time_until_one_item_s_memory_is_measured():
    results, peak = run(Sizing("test items", Fake(), measure=lambda running: None), range(10), slow_square)
    assert results == [item * item for item in range(10)]
    assert peak == 1


def test_a_new_item_starts_only_while_a_quarter_of_the_free_disk_stays_free(tmp_path):
    # 100 GB free when the work starts, 10 GB written by each item: 7 at once leave 30 GB, 8 would leave 20
    machine = Fake(free=100 * GB)
    sizing = sizing_of(machine, disk=lambda item: 10 * GB, folder=tmp_path)
    results, peak = run(sizing, range(20), lambda item: time.sleep(0.05) or item)
    assert results == list(range(20))
    assert peak == 7
    # an item larger than the share still runs, alone
    results, peak = run(sizing_of(machine, disk=lambda item: 90 * GB, folder=tmp_path), range(3),
                        lambda item: time.sleep(0.02) or item)
    assert results == [0, 1, 2] and peak == 1


def test_what_running_items_already_wrote_counts_once_against_the_free_disk(tmp_path):
    # 100 GB free at the start, one item of 60 GB running: room for 15 GB more before 25 GB are left
    machine = Fake(free=100 * GB)
    sizing = sizing_of(machine, disk=lambda item: 1, folder=tmp_path)
    assert sizing.may_start(1, need=15 * GB, needs=60 * GB)  # it wrote nothing yet: its 60 GB still count
    assert not sizing.may_start(1, need=16 * GB, needs=60 * GB)
    machine.free = 40 * GB  # it wrote its 60 GB: they count once, not again
    assert sizing.may_start(1, need=15 * GB, needs=60 * GB)
    machine.free = 30 * GB  # something else took 10 GB meanwhile
    assert sizing.may_start(1, need=5 * GB, needs=60 * GB) and not sizing.may_start(1, need=6 * GB, needs=60 * GB)


def test_an_item_whose_size_cannot_be_found_runs_and_fails_as_its_own_result(tmp_path):
    def size(item):
        if item == 2:
            raise KeyError(item)  # such as a file list the store cannot read
        return 1

    def work(item):
        if item == 2:
            raise KeyError(item)  # the item's own work meets the same failure
        return item

    sizing = sizing_of(Fake(cores=4, free=100 * GB), disk=size, folder=tmp_path)
    done = list(each(work, range(5), KeyError, sizing))
    assert [(d.item, d.value, type(d.error)) for d in done] == [
        (n, None, KeyError) if n == 2 else (n, n, type(None)) for n in range(5)]


def test_one_item_s_memory_follows_the_latest_measure_so_the_work_grows_back():
    found = [8 * GB]
    sizing = Sizing("test items", Fake(cores=16, available=20 * GB, total=64 * GB), measure=lambda running: found[0])
    assert not sizing.may_start(3)  # 8 GB each: one more leaves 12 GB, under a quarter of 64 GB
    found[0] = 1 * GB  # the running items now take less: one more leaves 19 GB
    time.sleep(0.11)  # measures come at most every 0.1 s
    assert sizing.may_start(3)


def test_threads_measure_what_this_process_grows_by_while_items_run():
    measure = Threads()
    measure(0)  # the first measure is the base: no item runs
    if measure(0) is None:
        pytest.skip("this machine does not say how much memory a process takes")
    held = bytearray(200 * 1024 * 1024)
    for i in range(0, len(held), 4096):
        held[i] = 1  # the pages are taken only once written
    assert measure(1) >= 150e6
    del held


HOLD = "(lambda held: (__import__('time').sleep(1.0), len(held))[1])(b'x' * (200 << 20))"  # 200 MB held for 1 s


def test_processes_measure_what_a_worker_holds():
    sizing = Sizing("test items")
    with process_pool(sizing, 1) as pool:
        assert isinstance(sizing.measure, Processes)
        future = pool.submit(eval, HOLD)  # a built-in: the spawned worker cannot import this test file
        time.sleep(0.6)
        measured = sizing.measure(1)
        assert future.result() == 200 * 1024 * 1024
    if measured is None:
        pytest.skip("this machine does not say how much memory a process takes")
    assert measured >= 150e6


def test_peak_is_the_most_memory_a_program_and_the_programs_it_started_held_at_once():
    child = "import time; held = b'x' * (200 << 20); time.sleep(1.0)"
    parent = (f"import subprocess, sys; subprocess.run([sys.executable, '-c', {child!r}])")
    code, most = peak([sys.executable, "-c", parent], poll=0.1)
    assert code == 0
    if most is None:
        pytest.skip("this machine does not say how much memory a process takes")
    assert most >= 150e6  # the child's, which the parent did not hold itself


def test_peak_gives_the_exit_code_of_a_program_that_fails():
    code, _ = peak([sys.executable, "-c", "raise SystemExit(3)"])
    assert code == 3
