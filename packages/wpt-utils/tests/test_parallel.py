import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from wpt_utils.parallel import Done, Machine, Sizing, each, in_order, limit

GB = 1e9


class Fake:
    """A machine whose measures the test sets."""

    def __init__(self, cores=16, available=64 * GB, total=64 * GB, idle=16.0):
        self.cores, self.available, self.total, self.idle = cores, available, total, idle

    def memory(self):
        return None if self.available is None else (self.available, self.total)

    def idle_cores(self):
        return self.idle


def test_the_limit_is_the_fewer_of_75_percent_of_the_cores_and_of_the_available_memory_per_item():
    assert limit(2.5 * GB, machine=Fake(cores=16, available=100 * GB))[0] == 12  # the cores decide
    count, why = limit(2.5 * GB, "test workers", Fake(cores=16, available=19.1 * GB))
    assert count == 5  # 75% of 19.1 GB is 14.3 GB: 5 items of 2.5 GB
    assert why == ("5 test workers: the fewer of 75% of 16 cores (12) and 75% of 19.1 GB available at 2.5 GB each (5)")
    assert limit(2.5 * GB, machine=Fake(cores=2, available=1 * GB))[0] == 1  # at least 1
    count, why = limit(2.5 * GB, "test workers", Fake(cores=8, available=None))
    assert (count, why) == (6, "6 test workers: 75% of 8 cores (the available memory is unknown here)")


def test_a_new_item_starts_only_while_a_quarter_of_the_machine_stays_free():
    machine = Fake(cores=16, available=64 * GB, total=64 * GB, idle=16.0)
    sizing = Sizing(2 * GB, machine=machine)
    assert sizing.limit == 12
    assert sizing.may_start(5)
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
    sizing = Sizing(1 * GB, machine=Fake(cores=8))
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

    sizing = Sizing(1 * GB, machine=Fake(cores=16))
    with ThreadPoolExecutor(sizing.limit) as executor:
        results = list(in_order(work, items, executor, sizing, poll=0.01, settle=0.0,
                                group=lambda item: (item[0], 2 if item[0] == "slow.example" else 8)))

    assert results == items
    assert peak == {"slow.example": 2, "other.example": peak["other.example"]} and peak["other.example"] > 2
    assert started.index(("other.example", 0)) < started.index(("slow.example", 7))  # it did not wait for them


def test_a_busy_machine_runs_one_item_at_a_time_and_still_finishes():
    sizing = Sizing(1 * GB, machine=Fake(cores=8, idle=0.0))
    results, peak = run(sizing, range(6), slow_square)
    assert results == [item * item for item in range(6)]
    assert peak == 1


def test_items_just_started_count_as_taking_their_share_before_the_measures_show_them():
    # 30 GB available of 64 GB at 2 GB each: the limit is 11, but only 7 items leave 16 GB, a quarter, free;
    # the fake's measures never show the started items, as at start-up before they take their memory
    sizing = Sizing(2 * GB, machine=Fake(cores=16, available=30 * GB, total=64 * GB))
    assert sizing.limit == 11
    results, peak = run(sizing, range(20), lambda item: time.sleep(0.2) or item, settle=60.0)  # long enough to overlap
    assert results == list(range(20))
    assert peak == 7


def test_instant_items_are_not_slowed_by_the_measures():
    # every start measures again; measuring must not wait SAMPLE (0.1 s) each time, which would take 20 s here
    start = time.monotonic()
    results, _ = run(Sizing(1e6), range(200), lambda item: item)
    assert results == list(range(200))
    assert time.monotonic() - start < 5.0


def test_an_item_s_failure_is_raised_at_its_turn_after_the_results_before_it():
    def work(item):
        if item == 3:
            raise ValueError("item 3")
        return item

    got = []
    with ThreadPoolExecutor(4) as executor, pytest.raises(ValueError, match="item 3"):
        for result in in_order(work, range(10), executor, Sizing(1 * GB, machine=Fake(cores=4)), poll=0.01):
            got.append(result)
    assert got == [0, 1, 2]



def test_each_turns_the_errors_named_into_their_item_s_result_and_any_other_stops():
    def work(item):
        if item % 3 == 1:
            raise KeyError(item)
        if item == 8:
            raise ValueError("item 8")
        return slow_square(item)

    sizing = Sizing(1 * GB, machine=Fake(cores=4))
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
