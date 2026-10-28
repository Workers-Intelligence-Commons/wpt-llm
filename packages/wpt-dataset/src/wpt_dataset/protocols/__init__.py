"""Protocols: each turns an address into files.

A protocol is a module offering

    fetch(address, data, extractor, tmp) -> what it got
        the files at address added to extractor (manifest.Extractor: the
        source's `extract` rules, and the store its contents go into), each
        a path and its bytes; data is the source description, of which it
        reads only its own options; tmp an empty folder of its own. It
        returns what it got, for the record only (the commit, for example);
        nothing decides on it.

and owns its network effects; robots rules, retries and request limits
(robots.py, retry.py, sites.py) are helpers it uses. Its failures are
named (errors.py), each naming the address: SourceGone, TooManyRequests,
CrawlForbidden (refused by robots rules), RobotsUnreadable,
HostUnreachable, ServerError (through every try), AccessRefused (HTTP 401,
403, 451, or a password asked for), Incomplete, and their base
FetchFailed, the general fetch failure, for a failure none of them is
(base.py); and for what the
files cannot be (UnsupportedEntry, BadPath), those errors.

A source's `type` names its protocol (PROTOCOLS): http (http.py), git
(git.py), mercurial (mercurial.py); an `empty` source has none and no
files. Another protocol is one more module and one more name here: the
download and the build know only protocols and files.
"""

from pathlib import Path
from typing import NamedTuple

from ..manifest import Entry, Extractor
from . import git, http, mercurial
from .base import UNREACHABLE, USER_AGENT, named, reason

PROTOCOLS = {"http": http, "git": git, "mercurial": mercurial}
EMPTY = "empty"  # the type of a source with no upstream: its patches hold every file


class Fetched(NamedTuple):
    """What one fetch gave: the tree (path -> manifest.Entry) and what it
    got, for the record (a protocol's answer; for several addresses, each
    one's by its folder)."""
    tree: dict[str, Entry]
    got: dict


def fetch(data: dict, address: str, store, tmp: Path) -> Fetched:
    """The files a source's protocol gives at one address (with `addresses`,
    each of them into its own folder of the one tree, and of tmp), each
    kept file's content put into store (what a download stores into:
    manifest.Extractor). A failure to reach upstream is a named protocol
    failure (base.named) whose message starts with the address it was
    for."""
    extractor = Extractor(store, data.get("extract", {}))
    if data["type"] == EMPTY:
        return Fetched(extractor.finish(), {})
    protocol = PROTOCOLS[data["type"]]
    parts = [(p["folder"], p["address"]) for p in data["addresses"]] if "addresses" in data else [("", address)]
    got = {}
    for folder, part_address in parts:
        (tmp / folder).mkdir(exist_ok=True)
        try:
            answer = protocol.fetch(part_address, data, extractor.within(folder), tmp / folder)
        except UNREACHABLE as error:
            raise named(error, part_address) from error
        if answer:
            got[folder] = answer
    return Fetched(extractor.finish(), got.get("", got))


__all__ = ["EMPTY", "PROTOCOLS", "USER_AGENT", "Fetched", "fetch", "named", "reason"]
