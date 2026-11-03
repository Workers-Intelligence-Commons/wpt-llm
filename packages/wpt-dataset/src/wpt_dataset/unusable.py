"""Every file of a source that its download or the build cannot use, found
in one pass and written as exact-path excludes, for a person to review
before the commit.

    find        fetches the source from upstream, as a download does, into a
                temporary folder that nothing outlives, going past each file
                whose bytes upstream does not give: a git file whose LFS
                object its host no longer holds, refuses to send (over its
                download allowance, say) or cannot be reached for. Then each
                file of the tree goes through the text check (checks.py). A
                failure of a whole address moves on to the next of its
                mirrors, and the failure of all of them stops it, as they
                stop a download. Only git files have this per-file path: the
                other protocols fail or give the address whole
    exclude     adds each file found to the source's `extract.exclude`,
                by its exact path, with its reason in `extract.note`

The store is neither read nor written. A changed `extract` means another
upstream key, so the next download fetches the source again.
"""

from pathlib import Path

from . import checks, download, manifest, protocols
from .errors import FetchFailed
from .repo import Repo
from .source_data import Source
from .sources import exact_pattern


def find(source: Source) -> dict[str, str]:
    """{path: why} of every file of the source upstream that cannot be used,
    sorted by path."""
    failures = []
    with download.work_folder() as tmp:
        for number, address in enumerate(download.addresses(source)):
            staging, folder = download.folders(Path(tmp) / f"address-{number}", None)
            found = {}
            try:
                fetched = protocols.fetch(source.data, address, staging, folder, unusable=found)
            except FetchFailed as error:  # this address only; another may work
                failures.append(error)
                continue
            for path, entry in fetched.tree.items():
                if checks.asks_text(source, path, entry):
                    if (why := checks.not_text(source, path, manifest.content(staging, entry))) is not None:
                        found[path] = why
            return dict(sorted(found.items()))
    raise download.failure(source, failures)


def exclude(repo: Repo, sid: str) -> dict[str, str]:
    """What find gives for the source sid, written into its source.json
    through the source rules (Repo.save_source)."""
    source = repo.source(sid)
    found = find(source)
    if found:
        document = repo.read("source", sid).document
        extract = document.get("extract", {})
        note = [extract["note"]] if extract.get("note") else []
        document["extract"] = {**extract, "exclude": [*extract.get("exclude", []), *map(exact_pattern, found)],
                               "note": " ".join(note + [f"{path}: {why}." for path, why in found.items()])}
        repo.save_source(sid, document, repo.version("source", sid))
    return found
