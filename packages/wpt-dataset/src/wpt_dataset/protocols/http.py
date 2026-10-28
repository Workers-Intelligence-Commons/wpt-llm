"""The HTTP protocol: one file per address, an archive unpacked by its kind.

    robots.txt of the address's site   -> read first; a refusal is CrawlForbidden, a file that cannot be read
                                          RobotsUnreadable (robots.py)
    the file's name (extract.name, or  -> its kind:
    the address's last path part)
        an archive (ARCHIVES)          -> its members are the files, read while it downloads (archives.py),
                                          after extract.strip; nothing of it lies on the disk when its
                                          server allows it
        anything else                  -> the one file, under that name, streamed to the disk, never held
                                          whole in memory

Each request waits for its site's limit per time (sites.py); a try that
fails for a reason that may pass is tried again (retry.py). page() gives
one outside page, such as a license page, as a file. It gets nothing to
record: what upstream served is the files.

Failures: what protocols.base.named makes of what urllib raises
(SourceGone, TooManyRequests, ServerError, AccessRefused, HostUnreachable,
Incomplete, FetchFailed); CrawlForbidden, RobotsUnreadable (robots.py);
an OSError for an address that is not http(s), or that names no file and
has no extract.name; UnsupportedEntry for an archive that is neither a zip
nor a tar; what the extractor raises for an entry (BadPath).
"""

from pathlib import Path
from urllib.parse import unquote, urlparse

from .. import archives, retry, robots
from ..manifest import FILE, Extractor
from .base import USER_AGENT

# the ends of a name that make it an archive, in any letter case
ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def name_of(address: str, data: dict) -> str:
    """The name of the one file at address: extract.name, else the last
    part of the address's path; OSError when it has neither."""
    name = data.get("extract", {}).get("name") or unquote(urlparse(address).path.rsplit("/", 1)[-1])
    if not name:
        raise OSError(f"{address}: no file name in the address; set extract.name")
    return name


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVES)


def fetch(address: str, data: dict, extractor: Extractor, tmp: Path) -> dict:
    """The files at address added to extractor (see the module's
    docstring); a file to save goes into the empty folder tmp. What it got,
    for the record: nothing."""
    robots.check(address, "http", USER_AGENT)  # before anything is downloaded
    name = name_of(address, data)
    if is_archive(name):
        archives.read(address, extractor, tmp, USER_AGENT)
    elif (path := extractor.kept(name)) is not None:
        extractor.add_file(path, FILE, download(address, tmp / "download"))
    return {}


def page(address: str, target: Path) -> Path:
    """The page at address saved to the file target, its robots.txt read
    first."""
    robots.check(address, "http", USER_AGENT)
    return download(address, target)


def download(address: str, target: Path) -> Path:
    """The answer to a GET of address, streamed into the file target;
    archives.EndedEarly when it ends before its Content-Length."""
    def attempt():
        with archives.get(address, USER_AGENT) as answer, open(target, "wb") as out:
            count = 0
            while block := answer.read(archives.BLOCK):
                out.write(block)
                count += len(block)
            archives.whole(answer, count, address)

    retry.call(attempt, address)
    return target
