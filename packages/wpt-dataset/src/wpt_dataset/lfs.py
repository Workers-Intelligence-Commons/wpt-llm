"""Git LFS: the large files a git repository keeps outside itself, read by
the LFS batch request of the git host (the Git LFS batch API, as GitHub,
GitLab and Hugging Face serve it), never through a site's own address.

In git, such a file's blob is a pointer: three lines naming the LFS version,
the file's sha256 (`oid sha256:<hex>`) and its size. download() asks the
repository's LFS endpoint (its git address with `.git` and `/info/lfs`) for
that object, streams it from the address the answer gives into a file, and
checks its sha256 and size; fetch() gives the same bytes in memory, for a
small file. Each request waits for its site's limit per time (sites.py),
and a try that fails for a reason that may pass is tried again
(retry.py).

Failures: SourceGone when the answer says the host does not hold the
object (its code 404 or 410), FetchFailed when it gives no download for
another reason,
Incomplete for bytes whose sha256 or size are not the pointer's,
archives.EndedEarly for an object that stopped short through every try;
what urllib raises when the host cannot be reached or refuses (the
protocol names those two, protocols/base.py).
"""

import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

from . import archives, retry, sites
from .errors import FetchFailed, Incomplete, SourceGone

POINTER = b"version https://git-lfs.github.com/spec/v1\n"  # how a pointer blob starts
POINTER_FIELDS = re.compile(rb"^oid sha256:([0-9a-f]{64})\n(?:.*\n)*?size (\d+)\n?", re.M)
MEDIA = "application/vnd.git-lfs+json"
TIMEOUT = 60
BLOCK = 1 << 20  # bytes read at once
GONE = (404, 410)  # the codes an LFS answer gives an object its host does not hold or no longer holds


def pointer(blob: bytes) -> tuple[str, int] | None:
    """(sha256, size) of the file a pointer blob names; None for a blob that
    is not a pointer (the file itself)."""
    if not blob.startswith(POINTER):
        return None
    found = POINTER_FIELDS.search(blob)
    return (found[1].decode(), int(found[2])) if found else None


def endpoint(address: str) -> str:
    """The LFS endpoint of a git address: the address with `.git` (added
    when it lacks it) and `/info/lfs`."""
    address = address.rstrip("/")
    return (address if address.endswith(".git") else address + ".git") + "/info/lfs"


def fetch(address: str, sha256: str, size: int, user_agent: str) -> bytes:
    """The bytes of the LFS object sha256 (of that size) of the git
    repository at address; see the module's failures."""
    buffer = io.BytesIO()
    retry.call(lambda: _transfer(address, sha256, size, buffer, user_agent), address)
    return buffer.getvalue()


def download(address: str, sha256: str, size: int, target: Path, user_agent: str) -> Path:
    """The LFS object sha256 (of that size) of the git repository at
    address, streamed into the file target, never held whole in memory;
    see the module's failures."""
    def attempt():
        with open(target, "wb") as out:
            _transfer(address, sha256, size, out, user_agent)

    retry.call(attempt, address)
    return target


def _transfer(address: str, sha256: str, size: int, out, user_agent: str):
    """One try: the batch request, then the object written to out (emptied
    first), checked against its pointer."""
    request = {"operation": "download", "transfers": ["basic"], "objects": [{"oid": sha256, "size": size}]}
    batch = urllib.request.Request(endpoint(address) + "/objects/batch", data=json.dumps(request).encode(),
                                   method="POST", headers={"Accept": MEDIA, "Content-Type": MEDIA,
                                                           "User-Agent": user_agent})
    sites.paced(address)
    with urllib.request.urlopen(batch, timeout=TIMEOUT) as answer:
        objects = json.loads(answer.read()).get("objects") or [{}]
    found = objects[0]
    link = (found.get("actions") or {}).get("download")
    if not link or not link.get("href"):
        error = found.get("error") or {}
        why = error.get("message", "no download in the answer")
        raise (SourceGone if error.get("code") in GONE else FetchFailed)(f"{address}: LFS object {sha256}: {why}")
    headers = {"User-Agent": user_agent, **link.get("header", {})}
    out.seek(0)
    out.truncate()
    digest, count = hashlib.sha256(), 0
    sites.paced(link["href"])
    with urllib.request.urlopen(urllib.request.Request(link["href"], headers=headers), timeout=TIMEOUT) as answer:
        while block := answer.read(BLOCK):
            digest.update(block)
            out.write(block)
            count += len(block)
    if count < size:  # cut short: a ConnectionError, so the try is made again
        raise archives.EndedEarly(f"{address}: LFS object {sha256}: the answer ended at byte {count} of {size}")
    if count != size or digest.hexdigest() != sha256:
        raise Incomplete(f"{address}: LFS object {sha256}: the bytes received are not the pointer's "
                         f"({count} bytes of {size}, sha256 {digest.hexdigest()})")
