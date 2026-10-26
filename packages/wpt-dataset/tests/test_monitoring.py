"""Always-on monitoring (wpt_dataset.monitoring): a round runs the source
drafters on every candidate, then downloads every admitted source again
and checks it, filling in the checksums the drafters set back to a
draft; rounds follow one another a week apart."""

import socket

import pytest

from wpt_dataset import manifest, monitoring, sources
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
        "type": "http-file", "address": context.settings["address"], "revision": "2026-10-25",
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


def test_a_round_drafts_then_downloads_every_admitted_source_again_and_fills_in_their_drafts(
        kit, repo, store, tmp_path, http_folder, monkeypatch):
    # given a git source and a book already in the store, a book whose upstream changed since its checksum was
    # filled in, an admitted source whose checksum a drafter set back to a draft, a draft under review, and two
    # source drafters (one that drafts a new book, one whose site does not answer): the round downloads the two
    # in the store again, fills in the admitted draft, leaves the sources under review (the new one too) as they
    # are, fails the changed book with its failure alone recorded, and names both failures
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    folder, base = http_folder
    for name in ("kept", "changed", "moved", "draft", "drafted"):
        (folder / f"{name}.txt").write_bytes(f"the {name} book\n".encode())
    address, commit = kit.git_project(tmp_path / "project")
    git = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address, "revision": commit})
    kept, changed = (kit.add_source(repo, store, "example", name, {
        "type": "http-file", "address": f"{base}/{name}.txt", "revision": "2026-10-25"}) for name in ("kept", "changed"))
    moved = kit.add_source(repo, store, "example", "moved", {
        "type": "http-file", "address": f"{base}/moved.txt", "revision": "2026-10-26"}, compute_checksum=False)
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

    failed = monitoring.one_round(repo, store, "T", "test-monitoring", [], drafters=[str(drafter), str(unreachable)],
                                  log=lambda *_: None)

    drafted = next(sid for sid in repo.source_ids() if sid.startswith("example-drafted-"))
    assert [line.split(":")[0] for line in failed] == ["unreachable", changed]
    assert "ChecksumMismatch" in failed[1]
    assert repo.source(moved).data["checksum"] != DRAFT_CHECKSUM
    for sid in (draft, drafted):  # under review: not downloaded, so still a draft
        assert repo.source(sid).data["checksum"] == DRAFT_CHECKSUM
    assert repo.source(changed).data["checksum"] == pinned
    assert repo.source(changed).data["decision"]["decision"] == "admitted"
    with store.database() as found:
        [session] = [row[0] for row in found.execute("SELECT folder FROM sessions WHERE command = ?",
                                                     ("wpt-dataset-test-monitoring",))]
        lines = dict(found.execute("SELECT source, outcome FROM lines WHERE session = ?", (session,)).fetchall())
    # a check against upstream that downloads again a tree the store holds is a download
    assert lines == {git: "downloaded", kept: "downloaded", changed: "failed", moved: "downloaded"}


def test_a_round_asks_a_moved_git_source_only_for_the_files_the_store_lacks(kit, repo, store, tmp_path, monkeypatch):
    # given a git source in the store whose revision a drafter moved to a commit changing one file (its checksum
    # set back to a draft): the round asks upstream for that one file alone, takes the sha256 of the others from
    # the store's map of git objects, and the tree it keeps reads back whole
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    address, commit = kit.git_project(tmp_path / "project")
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": address, "revision": commit})
    project = tmp_path / "project"
    kit.run("git", "checkout", "-q", "--", ".", cwd=project)  # the project's files, which git_project left out
    (project / "README.md").write_bytes(b"a changed first line\n")
    kit.run("git", "commit", "-q", "-am", "change one file", cwd=project)
    moved = kit.run("git", "rev-parse", "HEAD", cwd=project).decode().strip()
    source = repo.source(sid).data | {"revision": moved, "checksum": DRAFT_CHECKSUM}
    repo.save_source(sid, source, repo.version("source", sid))
    asked = []
    real = sources._git_fetch_blobs
    monkeypatch.setattr(sources, "_git_fetch_blobs",
                        lambda address, folder, ids: asked.append(sorted(ids)) or real(address, folder, ids))

    assert monitoring.one_round(repo, store, "T", "test-monitoring", [], drafters=[], log=lambda *_: None) == []

    readme = kit.run("git", "rev-parse", f"{moved}:README.md", cwd=project).decode().strip()
    assert asked == [[readme]]
    tree, _ = sources.source_tree(repo.source(sid), store)
    expected = kit.FILES | {"README.md": b"a changed first line\n", "latest.txt": b"docs/guide.txt"}
    assert {path: manifest.content(store, entry) for path, entry in tree.items()} == expected


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
        monitoring.serve(repo, store, "T", "test-monitoring", [], drafters=[str(unreachable)],
                         log=lambda text, *_: logged.append(text), sleep=sleep)
    assert [seconds for seconds, _ in slept] == [7 * 24 * 3600] * 2
    for _, log in slept:
        assert "round done, 1 failed\n  unreachable: " in log
    assert len([folder for folder in store.record_folders() if "-wpt-dataset-test-monitoring-" in folder]) == 2


def test_the_command_runs_with_its_pause_from_the_setting_and_a_machine_s_name_without_by(repo, monkeypatch, capsys):
    # given WPT_MONITORING_PAUSE, `wpt-dataset always-on-monitoring` runs on this checkout with that pause; --pause
    # overrides the setting; the drafts' `by` is the one given, or without --by this machine's host name marked
    # as a machine's; a name holding ':' is no person's and is refused
    served = []
    monkeypatch.setattr(monitoring, "serve", lambda repo, store, by, command, arguments, pause: served.append(
        (by, command, pause)))
    monkeypatch.setenv("WPT_STORE", (repo.root / "store").as_uri())
    monkeypatch.setenv("WPT_MONITORING_PAUSE", "3600")

    main(["--root", str(repo.root), "always-on-monitoring", "--by", "T"])
    main(["--root", str(repo.root), "always-on-monitoring", "--pause", "60"])
    assert main(["--root", str(repo.root), "always-on-monitoring", "--by", "A: Person"]) == 1

    assert served == [("T", "always-on-monitoring", 3600),
                      (f"machine:{socket.gethostname()}", "always-on-monitoring", 60)]
    assert "a person's name holds no ':'" in capsys.readouterr().err
