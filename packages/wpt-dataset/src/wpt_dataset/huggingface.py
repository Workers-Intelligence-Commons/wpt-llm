"""Dataset repositories on huggingface.co, pinned to a commit.

The address is https://huggingface.co/datasets/<owner>/<name> and the
revision the repository's full commit hash. The file list comes from the
Hub's API at that commit; each file's content comes from its `resolve`
address at that commit.

The Hub stores large files with Git LFS, and its listing gives their
sha256. `read` fetches every kept file, streamed to disk, and checks each
one against the listing.

An archive the dataset holds (ARCHIVES, the kinds an http-archive source
reads) is unpacked: its members are the source's files, in the archive's
folder, as unpacking it where it lies would place them, and the manifest
covers them, not the archive. `extract.exclude` matching the archive's
own path leaves it out undownloaded, and matching a member's path leaves
that member out.

robots.txt is read once for the address's site before anything else
(sources.py); the API and resolve addresses are on the same site.
"""

import hashlib
import json
import posixpath
import re
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse

from . import retry
from .errors import UnsupportedEntry
from .manifest import FILE, read_archive

API = "https://huggingface.co/api/datasets"
ARCHIVES = (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".zip")  # by the end of the name, in any letter case
NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')
BLOCK = 1 << 20


def dataset_of(address: str) -> str:
    """'<owner>/<name>' from https://huggingface.co/datasets/<owner>/<name>."""
    parts = urlparse(address).path.strip("/").split("/")
    if urlparse(address).netloc != "huggingface.co" or len(parts) != 3 or parts[0] != "datasets":
        raise OSError(f"{address}: a huggingface source's address is https://huggingface.co/datasets/<owner>/<name>")
    return f"{parts[1]}/{parts[2]}"


def files(address: str, revision: str, user_agent: str) -> list[dict]:
    """Every file of the dataset at the revision, as the API lists it: path,
    size, and `lfs` with its sha256 for large files."""
    url = f"{API}/{dataset_of(address)}/tree/{revision}?recursive=true&expand=false"
    found = []
    while url:
        page, link = retry.call(lambda url=url: _page(url, user_agent), url)
        found += page
        url = link[1] if link else None
    for entry in found:
        if entry["type"] not in ("file", "directory"):
            raise UnsupportedEntry(f"{entry['path']}: Hub entry of type {entry['type']!r} is neither a file nor a folder")
    return [entry for entry in found if entry["type"] == "file"]


def _page(url, user_agent):
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response), NEXT.search(response.headers.get("Link") or "")


def resolve_url(address: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_of(address)}/resolve/{revision}/{quote(path)}"


def fetch(address: str, revision: str, path: str, target: Path, user_agent: str) -> str:
    """Stream one file to target; returns its sha256."""
    url = resolve_url(address, revision, path)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})

    def download():
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
            while block := response.read(BLOCK):
                digest.update(block)
                out.write(block)
        return digest.hexdigest()

    return retry.call(download, url)


def read(address: str, revision: str, extractor, tmp: Path, user_agent: str):
    """Add every file the extractor keeps, each archive's members in its
    place."""
    for entry in files(address, revision, user_agent):
        path = extractor.kept(entry["path"])
        if path is None:
            continue
        archive = path.lower().endswith(ARCHIVES)
        expected = entry.get("lfs", {}).get("oid")
        target = tmp / "download"
        actual = fetch(address, revision, entry["path"], target, user_agent)
        if expected and actual != expected:
            raise OSError(f"{address}: {entry['path']} downloaded as sha256 {actual}, the listing says {expected}")
        if archive:
            read_archive(target, extractor.within(posixpath.dirname(path), strip=0))
        else:
            extractor.add_file(path, FILE, target)
        target.unlink()
