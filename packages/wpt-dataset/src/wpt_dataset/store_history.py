"""The download a dataset's record names for each source, read from the
sources' records."""

from wpt_store import held_all


def downloads_used(store, manifests: dict[str, str]) -> dict[str, dict | None]:
    """{source id: {"session": the session holding it, "time": UTC, and, when
    the last attempt at that download failed, "failed": that attempt's
    {"time", "kind", "message"}}
    of the download whose tree has the manifest given, or None when the
    sessions the store reads hold no such download}. A download whose last
    attempt failed is still the one a build read, when a good file of it
    remains."""
    used = {}
    for sid, manifest in sorted(manifests.items()):
        found = next((one for one in held_all(store, sid) if one.sha256 is not None and one.manifest == manifest), None)
        used[sid] = None if found is None else {
            "session": found.session, "time": found.time,
            **({"failed": {name: found.error[name] for name in ("time", "kind", "message")}} if found.error else {})}
    return used
