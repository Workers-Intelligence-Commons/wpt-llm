"""How politely wpt-dataset downloads from each site: the limits of
sites.txt, in this package.

Two kinds of line, each for a host and every host under it
(wikimedia.org covers dumps.wikimedia.org):

    <host> <number>                          at most that many sources download at once from the site
    <host> <number> per <seconds> seconds    at most that many requests reach the site in any span of
                                             that many seconds

The line `default <number>` gives how many at once for every site with
no line of the first kind. A site may have a line of each kind. Each
number comes from what the site publishes, named in the comment above
its line.

The limits per time hold for every HTTP request this process sends
(paced): one that would pass the limit waits until the oldest request of
the span is that many seconds old. Requests git or Mercurial send are
theirs and not counted.
"""

import re
import threading
import time
from collections import deque
from importlib import resources
from urllib.parse import urlparse

SITES = "sites.txt"  # the per-site limits, in this package
PER_TIME = re.compile(r"(\S+) (\d+) per (\d+) seconds")


def limits(text: str | None = None) -> tuple[dict[str, int], int, dict[str, tuple[int, float]]]:
    """({site: how many at once}, the default, {site: (requests, seconds)})
    from sites.txt (or text). Blank lines and # comments are left out.
    ValueError for a line in neither format, a site given twice the same
    kind of limit, or no default."""
    text = resources.files(__package__).joinpath(SITES).read_text(encoding="utf-8") if text is None else text
    at_once, per_time = {}, {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = " ".join(line.split())
        if not line or line.startswith("#"):
            continue
        timed, parts = PER_TIME.fullmatch(line), line.split()
        if timed and int(timed[2]) >= 1 and int(timed[3]) >= 1:
            site, found = timed[1].lower(), per_time
            value = (int(timed[2]), float(timed[3]))
        elif len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 1:
            site, found, value = parts[0].lower(), at_once, int(parts[1])
        else:
            raise ValueError(f"{SITES} line {number}: write `<host> <number of downloads at once>` or `<host> "
                             f"<number of requests> per <number> seconds`: {line!r}")
        if site in found:
            raise ValueError(f"{SITES} line {number}: {site} has a limit of this kind already")
        found[site] = value
    if "default" not in at_once:
        raise ValueError(f"{SITES}: no `default <number>` line")
    return at_once, at_once.pop("default"), per_time


def site_of(address: str, sites) -> str | None:
    """The site among sites an address belongs to (its host, or a host
    above it), or None."""
    host = (urlparse(address).hostname or "").lower()
    return next((site for site in sites if host == site or host.endswith("." + site)), None)


class Pace:
    """At most `count` calls of wait() in any span of `seconds`, across
    threads: a call that would pass it sleeps until it no longer does."""

    def __init__(self, count, seconds, clock=time.monotonic, sleep=time.sleep):
        self.count, self.seconds, self.clock, self.sleep = count, seconds, clock, sleep
        self._times, self._lock = deque(), threading.Lock()

    def wait(self):
        while True:
            with self._lock:
                moment = self.clock()
                while self._times and self._times[0] <= moment - self.seconds:
                    self._times.popleft()
                if len(self._times) < self.count:
                    self._times.append(moment)
                    return
                pause = self._times[0] + self.seconds - moment
            self.sleep(pause)


_paces, _paces_lock = None, threading.Lock()


def paced(address: str):
    """Before an HTTP request to address: wait while its site's limit per
    time (sites.txt) would be passed; at once for a site without one."""
    global _paces
    with _paces_lock:
        if _paces is None:
            _paces = {site: Pace(*limit) for site, limit in limits()[2].items()}
        paces = _paces
    site = site_of(address, paces)
    if site is not None:
        paces[site].wait()
