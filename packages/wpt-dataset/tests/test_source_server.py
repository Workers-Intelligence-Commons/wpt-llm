"""The always-on source server (wpt_dataset.source_server): a round runs the
source drafters, then downloads every source again and checks it, filling
in the drafts; rounds follow one another a week apart."""

import pytest

from wpt_dataset import source_server
from wpt_dataset.cli import main
from wpt_dataset.repo import DRAFT_CHECKSUM

# A source drafter that drafts one http-file source, the book at the address its `address` setting gives.
DRAFTER = '''
from wpt_dataset.source_drafter import Candidate, Drafted

SETTINGS = {"address": "", "attribution": "", "text": ""}


def candidates(context):
    yield Candidate("drafted")


def draft(candidate, context):
    return Drafted("example", "drafted", {
        "schema": 7, "type": "http-file", "address": context.settings["address"], "revision": "2026-10-25",
        "checksum": "sha256:" + "0" * 64,
        "licenses": [{"license": "MIT", "paths": ["**"], "attribution": context.settings["attribution"],
                      "text": context.settings["text"]}],
        "formats": [{"format": "text", "paths": ["**"]}]})
'''

# A source drafter whose site cannot be reached.
UNREACHABLE = '''
def candidates(context):
    raise ConnectionError("the site does not answer")


def draft(candidate, context):
    raise AssertionError("never asked")
'''


def test_a_round_drafts_then_downloads_every_source_again_and_fills_in_the_drafts(kit, repo, store, tmp_path,
                                                                                    http_folder, monkeypatch):
    # given a git source and a book already in the store, a book whose upstream changed since its checksum was
    # filled in, a draft, and two source drafters (one that drafts a new book, one whose site does not answer):
    # the round downloads the two in the store again, fills in both drafts (the new one too), fails the
    # changed book with its failure alone recorded, and names both failures
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    folder, base = http_folder
    for name in ("kept", "changed", "draft", "drafted"):
        (folder / f"{name}.txt").write_bytes(f"the {name} book\n".encode())
    address, commit = kit.git_project(tmp_path / "project")
    git = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address, "revision": commit})
    kept, changed = (kit.add_source(repo, store, "example", name, {
        "type": "http-file", "address": f"{base}/{name}.txt", "revision": "2026-10-25"}) for name in ("kept", "changed"))
    draft = kit.add_source(repo, store, "example", "draft", {
        "type": "http-file", "address": f"{base}/draft.txt", "revision": "2026-10-25",
        "decision": {"decision": "under-review", "by": "A. Person", "date": "2026-10-25"}}, compute_checksum=False)
    (folder / "changed.txt").write_bytes(b"the changed book, changed\n")
    pinned = repo.source(changed).data["checksum"]
    drafter, unreachable = tmp_path / "drafter.py", tmp_path / "unreachable.py"
    drafter.write_text(DRAFTER.replace('"address": ""', f'"address": "{base}/drafted.txt"')
                       .replace('"attribution": ""', f'"attribution": "{kit.ATTRIBUTION}"')
                       .replace('"text": ""', f'"text": {kit.LICENSE_TEXT!r}'))
    unreachable.write_text(UNREACHABLE)

    failed = source_server.one_round(repo, store, "T", "test-server", [], drafters=[str(drafter), str(unreachable)],
                                     log=lambda *_: None)

    drafted = next(sid for sid in repo.source_ids() if sid.startswith("example-drafted-"))
    assert [line.split(":")[0] for line in failed] == ["unreachable", changed]
    assert "ChecksumMismatch" in failed[1]
    for sid in (draft, drafted):
        assert repo.source(sid).data["checksum"] not in (DRAFT_CHECKSUM, None)
    assert repo.source(changed).data["checksum"] == pinned
    assert repo.source(changed).data["decision"]["decision"] == "admitted"
    with store.database() as found:
        [session] = [row[0] for row in found.execute("SELECT folder FROM sessions WHERE command = ?",
                                                     ("wpt-dataset-test-server",))]
        lines = dict(found.execute("SELECT source, outcome FROM lines WHERE session = ?", (session,)).fetchall())
    assert lines == {git: "downloaded", kept: "downloaded", changed: "failed", draft: "downloaded",
                     drafted: "downloaded"}


def test_rounds_come_a_week_apart_and_each_round_logs_its_failures(repo, store, tmp_path, monkeypatch):
    # given the server started with its default pause and a source drafter whose site does not answer: it sleeps
    # one week after each round, and before each sleep its log names the round's failure, so an operator who
    # reads only the server's output sees it
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    unreachable = tmp_path / "unreachable.py"
    unreachable.write_text(UNREACHABLE)
    slept, logged = [], []

    class Stop(Exception):
        pass

    def sleep(seconds):
        slept.append((seconds, "\n".join(logged)))
        logged.clear()
        if len(slept) == 2:
            raise Stop

    with pytest.raises(Stop):
        source_server.serve(repo, store, "T", "test-server", [], drafters=[str(unreachable)],
                            log=lambda text, *_: logged.append(text), sleep=sleep)
    assert [seconds for seconds, _ in slept] == [7 * 24 * 3600] * 2
    for _, log in slept:
        assert "round done, 1 failed\n  unreachable: " in log
    assert len([folder for folder in store.record_folders() if "-wpt-dataset-test-server-" in folder]) == 2


def test_the_command_serves_with_its_pause_from_the_setting(repo, monkeypatch):
    # given WPT_SOURCE_SERVER_PAUSE, `wpt-dataset source-server --by NAME` serves this checkout with that pause
    # and the drafts' `by`; --pause overrides the setting
    served = []
    monkeypatch.setattr(source_server, "serve", lambda repo, store, by, command, arguments, pause: served.append(
        (by, command, pause)))
    monkeypatch.setenv("WPT_STORE", (repo.root / "store").as_uri())
    monkeypatch.setenv("WPT_SOURCE_SERVER_PAUSE", "3600")

    main(["--root", str(repo.root), "source-server", "--by", "T"])
    main(["--root", str(repo.root), "source-server", "--by", "T", "--pause", "60"])

    assert served == [("T", "source-server", 3600), ("T", "source-server", 60)]
