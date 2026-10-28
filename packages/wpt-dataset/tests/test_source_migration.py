"""The one-off migration of sources to license entries with their own place,
expected texts and decision, and of the reviews of license texts to license
texts of their own (wpt_dataset/migrate.py; wpt_dataset/upstream_files.py):
`wpt-dataset migrate` makes each review of data/reviewed-license-texts.json
a license text decided by its verdict, reads a source's file list and
license files without a full download, and only when it needs them, drops
`checksum` and `revision`, narrows `**`, and carries the source's admission
over to an entry whose found statement is an admitted text of its own
license (the standard SPDX text included), to an entry that selects
records, and to a public-domain entry, whose place is the entry itself; a
license file no text holds becomes a new text under review, and every
other entry goes under review."""

import hashlib
import json

import pytest

from wpt_dataset.cli import main
from wpt_dataset.errors import RuleViolation
from wpt_dataset.license_check import coverage, text_id
from wpt_dataset.errors import OutdatedSchema
from wpt_dataset.migrate import MigrationIncomplete, _source_protocol, migrate
from wpt_dataset.spdx import Matcher
from wpt_dataset.upstream_files import Files

UNDER_REVIEW = {"decision": "under-review", "by": "machine:wpt-dataset",
                "reason": "the migration found no admitted text of its own license at its place"}
# a standard text, long enough for the matcher (20 words at least)
STANDARD = ("Permission to use, copy, modify, and distribute this software for any purpose with or without fee is "
            "hereby granted, provided that the above notice and this permission notice appear in all copies.\n")


def texts(kit, repo, found):
    """A license text, admitted, for each (bytes, license)."""
    kit.write_texts(repo.root, {text_id(content): {**kit.TEXTS[kit.REVIEWED], "license": license}
                                for content, license in found})


def item(repo, text):
    return json.loads((repo.root / "data" / "license-texts" / text / "license-text.json").read_text())


def old_source(kit, repo, sid, document):
    """A source.json of the shape that pinned its upstream, written as it was."""
    folder = repo.root / "data" / "sources" / sid
    folder.mkdir()
    (folder / "source.json").write_text(kit.dump_json(document), encoding="utf-8", newline="\n")
    return folder / "source.json"


def test_migrate_gives_each_entry_its_place_and_carries_the_admission_where_the_plan_carries_it(
        kit, repo, store, tmp_path, monkeypatch, capsys):
    # given a git source fixed at a commit, admitted, with one entry over every file and a public-domain one
    # over its documents: through the command, its files are listed from upstream without their contents; the
    # first entry finds its own license in LICENSE and keeps the admission, the public-domain entry's place
    # becomes the entry itself and keeps it too; `**` becomes patterns covering the same files, and nothing
    # pins the upstream any more
    files = {"LICENSE": kit.LICENSE_TEXT.encode(), "a.py": b"print(1)\n", "b.py": b"print(2)\n",
             "src/main.py": b"main()\n", "docs/guide.md": b"A guide.\n"}
    address, commit = kit.git_project(tmp_path / "upstream", files=files, executable=set(), symlinks={})
    for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):  # partial fetches, as on GitHub
        kit.run("git", "config", setting, "true", cwd=address)
    licenses = [{"license": "MIT", "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT},
                {"license": "LicenseRef-public-domain", "paths": ["docs/**"], "attribution": "T",
                 "text": "Its author died in 1900.\n"}]
    sid = "example-project-0123abcd"
    path = old_source(kit, repo, sid, {"type": "git", "address": address, "revision": commit,
                                       "checksum": "sha256:" + "0" * 64, "licenses": licenses,
                                       "formats": [{"format": "text", "paths": ["**"]}], "decision": kit.DECISION})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())

    assert main(["--root", str(repo.root), "migrate"]) == 0
    assert capsys.readouterr().out.strip().endswith("1 files rewritten")

    document = json.loads(path.read_text())
    assert "revision" not in document and "checksum" not in document and document["decision"] == kit.DECISION
    first, second = document["licenses"]
    assert (first["place"], first["expected"], first["decision"]) == ({"file": "LICENSE"},
                                                                      [text_id(kit.LICENSE_TEXT.encode())],
                                                                      kit.DECISION)
    assert (second["place"], second["decision"]) == ({"entry": True}, kit.DECISION) and "expected" not in second
    covered = coverage(document["licenses"], files, lambda _: False)
    assert {name: found.entry for name, found in covered.items()} == {
        "LICENSE": 0, "a.py": 0, "b.py": 0, "src/main.py": 0, "docs/guide.md": 1}
    assert "**" not in first["paths"]
    assert repo.source(sid).data == document  # the current shape: the source reads again
    assert migrate(repo, log=lambda _: None) == []


def test_an_entry_whose_license_file_states_another_license_goes_under_review_and_records_keep_the_admission(
        kit, repo):
    # given admitted sources: one whose entry says MIT while its LICENSE is a text of Apache-2.0, which goes
    # under review with that file as its place; one whose records name their license in a field the format
    # gave, whose entries select records by that field and keep the admission, read without listing its
    # files (no `**`, no license file to find); and an empty source, whose files are its patches: its entry's
    # place is the entry itself, under review, `**` narrowed to the files the patches hold
    apache = b"Apache License\nVersion 2.0, January 2004\n"
    texts(kit, repo, [(apache, "Apache-2.0")])
    entry = {"license": "MIT", "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT}
    other = old_source(kit, repo, "example-other-0123abcd", {
        "type": "http-archive", "address": "https://example.org/other.zip", "checksum": "sha256:" + "0" * 64,
        "licenses": [entry], "formats": [{"format": "text", "paths": ["**"]}], "decision": kit.DECISION})
    pages = old_source(kit, repo, "example-pages-0123abcd", {
        "type": "http-file", "address": "https://example.org/pages.jsonl", "checksum": "sha256:" + "0" * 64,
        "licenses": [{**entry, "license": "CC-BY-SA-4.0", "paths": ["pages.jsonl"], "values": ["CC BY-SA 4.0"]},
                     {**entry, "license": "CC0-1.0", "paths": ["pages.jsonl"], "values": ["CC0"]}],
        "formats": [{"format": "records", "paths": ["pages.jsonl"], "text": "text", "license": "license"}],
        "decision": kit.DECISION})
    mine = old_source(kit, repo, "mine-notes-0123abcd", {
        "type": "empty", "address": "empty:notes", "revision": "2026-10-01", "checksum": "sha256:" + "0" * 64,
        "licenses": [{**entry, "license": "CC0-1.0"}], "formats": [{"format": "chat", "paths": ["**"]}],
        "decision": kit.DECISION})
    upstream = {"example-other-0123abcd": {"LICENSE": apache, "main.c": b"int main;\n"},
                "mine-notes-0123abcd": {"sft/talks.jsonl": b"", "sft/more.jsonl": b""}}
    listed = []

    def files(sid, document):
        listed.append(sid)
        return Files(sorted(upstream[sid]), upstream[sid].get)

    assert sorted(migrate(repo, log=lambda _: None, upstream=files)) == sorted(
        path.relative_to(repo.root).as_posix() for path in (other, pages, mine))
    assert sorted(listed) == ["example-other-0123abcd", "mine-notes-0123abcd"]
    [moved] = json.loads(other.read_text())["licenses"]
    assert (moved["place"], moved["expected"]) == ({"file": "LICENSE"}, [text_id(apache)])
    assert {**moved["decision"], "date": None} == {**UNDER_REVIEW, "date": None}
    document = json.loads(pages.read_text())
    assert [(e["place"], e["expected"], e["decision"]) for e in document["licenses"]] == [
        ({"field": "license"}, ["CC BY-SA 4.0"], kit.DECISION), ({"field": "license"}, ["CC0"], kit.DECISION)]
    assert "license" not in document["formats"][0] and document["decision"] == kit.DECISION
    # an archive and a file of the shape that named how a site serves them are both read by the http protocol
    assert json.loads(other.read_text())["type"] == document["type"] == "http"
    [written] = json.loads(mine.read_text())["licenses"]
    assert (written["paths"], written["place"], written["decision"]["decision"]) == (
        ["sft/**"], {"entry": True}, "under-review")


def test_a_license_file_no_text_holds_becomes_a_text_of_its_own_admitted_only_when_it_is_a_standard_text(kit, repo):
    # given admitted sources whose LICENSE no license text holds: one whose file is the standard SPDX text of
    # its own license (with its own copyright line), whose text is added admitted, by the person the
    # migration names, and whose entry keeps the admission; one whose file is no standard text, added under
    # review with its bytes, its entry under review too
    custom = b"You may read this.\n"
    standard = b"Copyright (c) 2026 Someone\n\n" + STANDARD.encode()
    entry = {"license": "ISC", "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": STANDARD}
    for name in ("standard", "custom"):
        old_source(kit, repo, f"example-{name}-0123abcd", {
            "type": "http-archive", "address": f"https://example.org/{name}.zip", "checksum": "sha256:" + "0" * 64,
            "licenses": [entry], "formats": [{"format": "text", "paths": ["**"]}], "decision": kit.DECISION})
    upstream = {"example-standard-0123abcd": {"LICENSE": standard, "main.c": b"int main;\n"},
                "example-custom-0123abcd": {"LICENSE": custom, "main.c": b"int main;\n"}}

    def files(sid, document):
        return Files(sorted(upstream[sid]), upstream[sid].get)

    migrate(repo, log=lambda _: None, upstream=files, by="Test Maintainer",
            standard=lambda: Matcher({"ISC": STANDARD, "MIT": kit.LICENSE_TEXT * 5}))

    [carried] = repo.source("example-standard-0123abcd").data["licenses"]
    assert (carried["place"], carried["expected"], carried["decision"]) == (
        {"file": "LICENSE"}, [text_id(standard)], kit.DECISION)
    added = item(repo, text_id(standard))
    assert {**added, "decision": {**added["decision"], "date": None}} == {
        "decision": {"decision": "admitted", "by": "Test Maintainer", "date": None}, "kind": "license",
        "license": "ISC", "found": {"source": "example-standard-0123abcd", "path": "LICENSE"},
        "files": [hashlib.sha256(standard).hexdigest()], "explanation": "standard text of ISC, SPDX release v3.29.0"}
    [sent] = repo.source("example-custom-0123abcd").data["licenses"]
    assert (sent["place"], sent["expected"], sent["decision"]["decision"]) == (
        {"file": "LICENSE"}, [text_id(custom)], "under-review")
    new = item(repo, text_id(custom))
    assert (new["decision"]["decision"], new["decision"]["by"], new["files"]) == (
        "under-review", "machine:wpt-dataset", [hashlib.sha256(custom).hexdigest()])
    assert repo.license_text_itself(text_id(custom)) == custom
    assert repo.check() == []


def test_the_reviews_of_license_texts_become_license_texts_decided_by_their_verdict(kit, repo):
    # given the earlier reviews of license texts: each review keyed by its group becomes a license text of
    # that id, admitted, refused or under review as its verdict says, signed by the person the migration
    # names; a review that holds for one exact file stays in the earlier file, named, and a source whose
    # license file is that exact file stays in its earlier shape until a person decides it
    def review(kind, verdict, **more):
        return {"kind": kind, **more, "answer": {"verdict": verdict, "spdx": more.get("license", ""),
                                                 "explanation": f"{verdict} text"}}

    exact = b"Copyright (c) 2026 Test Author, but its photo is not free.\n" + kit.LICENSE_TEXT.encode()
    reviews = {"a" * 64: review("license", "allowed-cosmetic", license="MIT"),
               "b" * 64: review("out", "not-allowed", why="non-commercial"),
               "c" * 64: review("license", "gpl-version-unclear", license="GPL-2.0-or-later", recheck="read it again"),
               "file:" + hashlib.sha256(exact).hexdigest(): review("out", "not-allowed", why="its photo")}
    earlier = repo.root / "data" / "reviewed-license-texts.json"
    earlier.write_text(json.dumps({"texts": reviews}))
    with pytest.raises(RuleViolation, match="--by NAME"):
        migrate(repo, log=lambda _: None)
    assert sorted(repo.license_text_ids()) == [kit.REVIEWED]  # nothing written without a signer

    entry = {"license": "MIT", "paths": ["main.c", "LICENSE"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT}
    left = old_source(kit, repo, "example-left-0123abcd", {
        "type": "http-archive", "address": "https://example.org/left.zip", "checksum": "sha256:" + "0" * 64,
        "licenses": [entry], "formats": [{"format": "text", "paths": ["**"]}], "decision": kit.DECISION})
    before = left.read_text()

    def files(sid, document):
        return Files(["LICENSE", "main.c"], {"LICENSE": exact, "main.c": b"int main;\n"}.get)

    with pytest.raises(MigrationIncomplete) as raised:
        migrate(repo, log=lambda _: None, upstream=files, by="Test Maintainer")
    message = str(raised.value)
    assert f"review file:{hashlib.sha256(exact).hexdigest()} stays" in message and "example-left-0123abcd" in message
    assert left.read_text() == before
    assert list(json.loads(earlier.read_text())["texts"]) == ["file:" + hashlib.sha256(exact).hexdigest()]
    decided = {text: (found["decision"]["decision"], found["decision"].get("reason"), found["decision"].get("note"),
                      found["kind"], found.get("license"))
               for text in ("a" * 64, "b" * 64, "c" * 64) for found in [item(repo, text)]}
    assert decided == {"a" * 64: ("admitted", None, None, "license", "MIT"),
                       "b" * 64: ("refused", "non-commercial", None, "license", None),
                       "c" * 64: ("under-review", "the review's verdict: gpl-version-unclear", "read it again",
                                  "license", "GPL-2.0-or-later")}
    assert {item(repo, "a" * 64)["decision"]["by"], item(repo, "b" * 64)["decision"]["by"]} == {"Test Maintainer"}


def test_a_source_with_no_license_file_stays_in_its_earlier_shape_and_is_named(kit, repo):
    # every entry has a place, and a file place names a license file: an entry whose source holds no file
    # whose name says it states a license cannot be written in the current shape, so its source stays as it
    # was and the migration names it, while every other source is rewritten
    entry = {"license": "MIT", "paths": ["main.c", "README.md"], "attribution": kit.ATTRIBUTION,
             "text": kit.LICENSE_TEXT}
    left = old_source(kit, repo, "example-left-0123abcd", {
        "type": "http-archive", "address": "https://example.org/left.zip", "checksum": "sha256:" + "0" * 64,
        "licenses": [entry], "formats": [{"format": "text", "paths": ["**"]}], "decision": kit.DECISION})
    before = left.read_text()

    def files(sid, document):
        return Files(["README.md", "main.c"], {"README.md": b"MIT licensed.\n", "main.c": b"int main;\n"}.get)

    with pytest.raises(MigrationIncomplete, match="example-left-0123abcd") as raised:
        migrate(repo, log=lambda _: None, upstream=files)
    assert "no file of the source whose name says it states a license holds a license text" in str(raised.value)
    assert left.read_text() == before


def test_a_source_type_becomes_the_protocol_that_reads_it_unless_the_file_s_name_would_change_its_reading():
    # http-file and http-archive become http, which unpacks a file only when its name says it is an archive;
    # huggingface becomes git at the same address. A file the http protocol would read otherwise stays as it
    # is, named, for a person to give its name
    dataset = {"type": "huggingface", "address": "https://huggingface.co/datasets/owner/name"}
    assert _source_protocol(dataset, None) == {**dataset, "type": "git"}
    assert _source_protocol({"type": "http-archive", "address": "https://example.org/a.tar.gz"}, None)["type"] == "http"
    assert _source_protocol({"type": "http-file", "address": "https://example.org/a.txt"}, None)["type"] == "http"
    for kind, address in (("http-archive", "https://example.org/download?id=7"),
                          ("http-file", "https://example.org/a.zip")):
        with pytest.raises(OutdatedSchema, match="extract.name"):
            _source_protocol({"type": kind, "address": address}, None)
