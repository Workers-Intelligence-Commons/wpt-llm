"""The download a dataset's record names for each source, read from the
store database (wpt_store.database) alone."""

from wpt_store.database import DOWNLOADED


def downloads_used(store, manifests: dict[str, str]) -> dict[str, dict | None]:
    """{source id: {"session": the folder of its run record, "time": UTC}
    of the last download that got the source's upstream tree whose manifest
    the store names by the sha256 given (the `manifest` its `downloaded`
    line's got records), or None when the database holds no such
    download}."""
    with store.database() as found:
        rows = found.execute("SELECT source, json_extract(got, '$.manifest') AS manifest, session, MAX(time)"
                             " FROM lines WHERE outcome = ? GROUP BY source, manifest", (DOWNLOADED,)).fetchall()
    kept = {(source, manifest): {"session": session, "time": time} for source, manifest, session, time in rows}
    return {sid: kept.get((sid, manifest)) for sid, manifest in sorted(manifests.items())}
