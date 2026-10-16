"""Dataset repositories on huggingface.co, pinned to a commit.

The address is https://huggingface.co/datasets/<owner>/<name> and the
revision the repository's full commit hash. The file list comes from the
Hub's API at that commit; each file's content comes from its `resolve`
address at that commit.

The Hub stores large files with Git LFS, and its listing gives their sha256,
which is what a manifest records. So `listing` builds a source's manifest
from the listing alone, downloading only the small files it holds directly
(such as README.md): `wpt-dataset manifest` gives the checksum of a dataset
of hundreds of gigabytes without fetching it. `download` fetches every kept
file, streamed to disk, and checks each one against the listing.

robots.txt is read once for the address's site before anything else
(sources.py); the API and resolve addresses are on the same site.
"""

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse

from . import retry
from .errors import UnsupportedEntry
from .manifest import FILE

API = "https://huggingface.co/api/datasets"
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


def read(address: str, revision: str, extractor, tmp: Path, user_agent: str, listing_only: bool):
    """Add every file the extractor keeps. listing_only: large files are
    recorded by the sha256 the listing gives, without their content."""
    for entry in files(address, revision, user_agent):
        path = extractor.kept(entry["path"])
        if path is None:
            continue
        expected = entry.get("lfs", {}).get("oid")
        if listing_only and expected:
            extractor.add_known(path, FILE, expected)
            continue
        target = tmp / "download"
        actual = fetch(address, revision, entry["path"], target, user_agent)
        if expected and actual != expected:
            raise OSError(f"{address}: {entry['path']} downloaded as sha256 {actual}, the listing says {expected}")
        extractor.add_file(path, FILE, target)
        target.unlink()
