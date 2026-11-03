"""A stand-in for a worker function of wpt_dataset that makes the first
item end last, so the build's worker processes finish out of order.

A worker process imports a function by its module's name: the tests put
this folder on the search path (monkeypatch.syspath_prepend) before the
workers start, so they can import this module, which a monkeypatched
function of the test process itself cannot give them.
"""

import importlib
import time

WAIT = 3.0  # seconds: longer than the 2 s an item counts as not yet measured (wpt_utils.parallel.SETTLE)


class FirstEndsLast:
    """module.name(item, ...) after WAIT seconds for the item named first
    (a source id, or a source whose id it is), at once for the others;
    each item's end goes as one line, the item's name, into log."""

    def __init__(self, module, name, first, log):
        self.module, self.name, self.first, self.log = module, name, first, str(log)

    def __call__(self, item, *arguments, **named):
        key = getattr(item, "id", item)
        if key == self.first:
            time.sleep(WAIT)
        try:
            return getattr(importlib.import_module(self.module), self.name)(item, *arguments, **named)
        finally:
            with open(self.log, "a", encoding="utf-8", newline="\n") as out:
                out.write(f"{key}\n")


def ended_out_of_order(log, first) -> bool:
    """Whether some item of log ended before the first one did."""
    ends = open(log, encoding="utf-8").read().splitlines()
    return ends.index(first) > 0
