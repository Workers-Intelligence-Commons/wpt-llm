"""The download a dataset's record names for each source, read from the
store database (wpt_store.database) alone."""

from wpt_store.database import DOWNLOADED


def downloads_used(store, checksums: dict[str, str]) -> dict[str, dict | None]:
    """{source id: {"session": the folder of its run record, "time": UTC}
    of the last download that kept the source's files under the checksum
    given, or None when the database holds no such download}."""
    with store.database() as found:
        rows = found.execute("SELECT source, checksum, session, MAX(time) FROM lines WHERE outcome = ?"
                             " GROUP BY source, checksum", (DOWNLOADED,)).fetchall()
    kept = {(source, checksum): {"session": session, "time": time} for source, checksum, session, time in rows}
    return {sid: kept.get((sid, checksum)) for sid, checksum in sorted(checksums.items())}
