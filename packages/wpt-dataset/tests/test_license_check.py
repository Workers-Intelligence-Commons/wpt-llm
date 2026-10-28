"""The license check (wpt_dataset/license_check.py; docs/licenses.md): which
entry covers each file and each record, and whether the statement at each
entry's place is a license text the entry expects, with that text's
decision."""

import hashlib

import pytest

from wpt_dataset.errors import RuleViolation
from wpt_dataset.license_check import LicenseTexts, check_pattern, coverage, statements, text_id, trains

MIT = b"Copyright (c) 2026 Test Author\n\nPermission is hereby granted, free of charge, to any person.\n"
CC0 = b"CC0 1.0 Universal\n\nThe person who associated a work with this deed has dedicated it to the public domain.\n"


def entry(license, paths, **more):
    return {"license": license, "paths": paths, "decision": {"decision": "under-review", "by": "T", "date": "2026-10-27"},
            **more}


def test_each_file_has_its_one_entry_and_each_record_the_entry_expecting_its_value():
    # given a source of code, documents and a records file whose records carry their own license, every
    # path is answered by exactly one entry, or by none with the reason a person fixes
    licenses = [entry("MIT", ["src/**", "LICENSE"]),
                entry("CC0-1.0", ["**/*.md"]),
                entry("CC-BY-SA-4.0", ["pages.jsonl"], place={"field": "license"}, expected=["CC BY-SA 4.0"]),
                entry("CC0-1.0", ["pages.jsonl"], place={"field": "license"}, expected=["CC0"])]
    paths = ["LICENSE", "src/main.py", "docs/guide.md", "pages.jsonl", "notes.txt", "src/notes.md"]
    found = coverage(licenses, paths, lambda path: path.endswith(".jsonl"))

    assert (found["LICENSE"].entry, found["src/main.py"].entry, found["docs/guide.md"].entry) == (0, 0, 1)
    records = found["pages.jsonl"]
    assert records.problem is None and records.field == "license"
    assert (records.of_record("CC BY-SA 4.0"), records.of_record("CC0"), records.of_record("GFDL")) == (2, 3, None)
    assert found["notes.txt"].problem == "notes.txt matches no license entry; every path must match exactly 1"
    assert found["src/notes.md"].problem == ("src/notes.md matches 2 license entries (MIT, CC0-1.0); every path must "
                                             "match exactly 1")


def test_a_history_file_counts_only_against_patterns_of_the_history():
    licenses = [entry("MIT", [".git/history/**"]), entry("MIT", [".git/**", "LICENSE"])]
    found = coverage(licenses, [".git/history/0123abcd", ".git/config"], lambda path: False)
    assert (found[".git/history/0123abcd"].entry, found[".git/config"].entry) == (0, 1)


def item(decision, license, files=()):
    return {"decision": {"decision": decision, "by": "T", "date": "2026-10-27",
                         **({"reason": "test"} if decision == "refused" else {})},
            "kind": "license", "license": license, "files": list(files), "explanation": "test"}


def test_a_statement_matches_when_its_place_holds_a_license_text_the_entry_expects():
    # given the license texts and the bytes at each place, an entry's statement matches only when the text
    # found there is one it expects, found by the file's own sha256 first, then by its normalized text; a
    # field place selects records and has nothing to match, and the entry itself is its statement as it stands
    texts = LicenseTexts({text_id(MIT): item("admitted", "MIT"),
                          "c" * 64: item("admitted", "CC0-1.0", [hashlib.sha256(CC0).hexdigest()])})
    tree = {"LICENSE": MIT.replace(b"Test Author", b"Other Author"), "docs/COPYING": CC0, "README": b"Hello.\n"}
    licenses = [entry("MIT", ["src/**"], place={"file": "LICENSE"}, expected=[text_id(MIT)]),
                entry("MIT", ["docs/**"], place={"file": "docs/COPYING"}, expected=[text_id(MIT)]),
                entry("MIT", ["README"], place={"file": "README"}, expected=[text_id(MIT)]),
                entry("MIT", ["lib/**"], place={"file": "lib/LICENSE"}, expected=[text_id(MIT)]),
                entry("CC0-1.0", ["pages.jsonl"], place={"field": "license"}, expected=["CC0"]),
                entry("LicenseRef-public-domain", ["other/**"], place={"entry": True})]
    found = statements(licenses, lambda place: tree.get(place["file"]), texts)

    assert [s.matched for s in found] == [True, False, False, False, True, True]
    assert (found[0].found, found[0].decision) == (text_id(MIT), "admitted")  # another copyright holder, one text
    assert found[1].problem == f"the license text there ({'c' * 64}) is not one it expects"
    assert found[2].problem == (f"no license text holds the text there (its id would be {text_id(tree['README'])}; "
                                "`wpt-dataset add-license-text` adds it, under review)")
    assert found[3].problem == "the file lib/LICENSE is not there"


def test_a_file_trains_only_when_its_entry_and_the_text_at_its_place_are_admitted():
    # given one entry per combination of its own decision and the decision of the text its place holds, the
    # license check reports each text's decision, and only the admitted entry whose matched text is admitted
    # lets its files train; a field place and the entry itself have no text to admit
    under_review, refused = MIT + b"under review\n", MIT + b"refused\n"
    texts = LicenseTexts({text_id(MIT): item("admitted", "MIT"), text_id(under_review): item("under-review", "MIT"),
                          text_id(refused): item("refused", "MIT")})
    tree = {"a/LICENSE": MIT, "b/LICENSE": under_review, "c/LICENSE": refused}
    admitted = {"decision": "admitted", "by": "T", "date": "2026-10-27"}
    licenses = [entry("MIT", ["a/**"], place={"file": "a/LICENSE"}, expected=[text_id(MIT)], decision=admitted),
                entry("MIT", ["b/**"], place={"file": "b/LICENSE"}, expected=[text_id(under_review)],
                      decision=admitted),
                entry("MIT", ["c/**"], place={"file": "c/LICENSE"}, expected=[text_id(refused)], decision=admitted),
                entry("MIT", ["d/**"], place={"file": "a/LICENSE"}, expected=[text_id(MIT)]),
                entry("CC0-1.0", ["pages.jsonl"], place={"field": "license"}, expected=["CC0"], decision=admitted)]
    found = statements(licenses, lambda place: tree.get(place["file"]), texts)

    assert [s.decision for s in found] == ["admitted", "under-review", "refused", "admitted", None]
    assert [trains(e, s) for e, s in zip(licenses, found)] == [True, False, False, False, True]
    assert found[1].problem == f"the license text there ({text_id(under_review)}) is under-review"


@pytest.mark.parametrize("pattern", ["src/*", "src/**", "**/*.js", "docs/**/*.md", "LICENSE", "*.py"])
def test_a_pattern_naming_a_folder_or_a_kind_of_file_is_allowed(pattern):
    check_pattern(pattern, "here")


@pytest.mark.parametrize("pattern", ["**", "*", "**/*", "*/**", "*.*"])
def test_a_pattern_naming_neither_a_folder_nor_a_kind_of_file_is_refused(pattern):
    with pytest.raises(RuleViolation, match="names no folder and no kind of file"):
        check_pattern(pattern, "here")
