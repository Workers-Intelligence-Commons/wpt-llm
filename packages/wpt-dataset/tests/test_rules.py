import json
import re

import pytest

from wpt_dataset.cli import main
from wpt_dataset.errors import InvalidFile, OutdatedSchema
from wpt_dataset.migrate import migrate
from wpt_dataset.pools import matches, parse
from wpt_dataset.repo import address_hash, dump_json


def test_pool_expressions_follow_usual_precedence():
    tree = parse("fr && !(poetry) || en")  # (fr and not poetry) or en
    assert matches(tree, {"fr"}) and matches(tree, {"en", "poetry"})
    assert not matches(tree, {"fr", "poetry"}) and not matches(tree, {"de"})
    assert matches(parse("fr && (literature || poetry)"), {"fr", "poetry"})
    for bad in ("fr && !poetry", "fr &&", "(fr", "fr en", ""):
        with pytest.raises(ValueError):
            parse(bad)


def test_check_reports_each_broken_rule(kit, repo, store, git_upstream):
    good = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    assert repo.check() == []

    grey = kit.add_source(repo, store, "example", "grey", {
        "type": "git", "address": git_upstream[0] + "/.", "revision": git_upstream[1],
        "licenses": [{"license": "CC-BY-SA-4.0", "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT}]})
    unexplained = kit.add_source(repo, store, "example", "unexplained", {
        "type": "git", "address": git_upstream[0] + "/./", "revision": git_upstream[1],
        "decision": {"decision": "refused", "by": "Test Reviewer", "date": "2026-10-14"}})  # a refusal must say why
    wrong_id = repo.data / "sources" / "example-wrong-00000000"
    wrong_id.mkdir()
    (wrong_id / "source.json").write_text((repo.data / "sources" / good / "source.json").read_text())
    (repo.data / "sources" / good / "0002-skipped.patch").write_text("")
    kit.write_recipe(repo, "mixture", "half", {"pools": [{"name": "a", "select": "a", "weight": 0.5}]})

    problems = "\n".join(repo.check())
    assert f"{grey}: license CC-BY-SA-4.0 is not on the allowed list" in problems
    assert any(unexplained in line and "'reason' is a required property" in line for line in problems.splitlines())
    assert f"the id must end with {address_hash(git_upstream[0])}" in problems
    assert f"{good}: patches must be numbered 0001" in problems
    assert "pool weights add up to 0.5, not 1" in problems


def test_migrate_rewrites_files_that_are_behind(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.data / "sources" / sid / "source.json"
    # a made-up next shape: "type" renamed to "kind"
    step = {5: lambda doc, _path: {**{k: v for k, v in doc.items() if k != "type"}, "kind": doc["type"], "schema": 6}}
    current = {"source": 6, "mixture": 1, "dataset": 1, "tokenizer": 1, "lock": 1}

    changed = migrate(repo, migrations={"source": step, "mixture": {}, "dataset": {}, "tokenizer": {}, "lock": {}},
                      current=current, log=lambda _: None)
    assert changed == [path]
    document = json.loads(path.read_text())
    assert document["schema"] == 6 and document["kind"] == "git" and "type" not in document
    assert path.read_text() == dump_json(document)


def test_migrating_a_shape_1_source_asks_a_person_for_attribution_text_and_formats(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.data / "sources" / sid / "source.json"
    old = json.loads(path.read_text())
    old["schema"] = 1
    del old["licenses"][0]["attribution"]
    del old["licenses"][0]["text"]
    del old["formats"]
    path.write_text(dump_json(old))

    with pytest.raises(InvalidFile, match=re.escape(str(path))):
        migrate(repo, log=lambda _: None)
    document = json.loads(path.read_text())
    assert document["schema"] == 5
    assert document["licenses"] == [{"license": "MIT", "paths": ["**"], "attribution": "", "text": ""}]
    assert document["formats"] == []

    document["licenses"][0]["attribution"] = kit.ATTRIBUTION  # what the person writes
    document["licenses"][0]["text"] = kit.LICENSE_TEXT
    path.write_text(dump_json(document))
    with pytest.raises(InvalidFile, match=re.escape(f"{path}: at formats")):
        migrate(repo, log=lambda _: None)
    document["formats"] = [{"format": "text", "paths": ["**"]}]  # the person names each file's format
    path.write_text(dump_json(document))
    assert migrate(repo, log=lambda _: None) == [] and repo.check() == []


def test_a_license_expression_passes_when_one_alternative_is_allowed(kit, repo, store, git_upstream):
    # WPT uses dual-licensed content under the alternative it allows; an
    # exception after WITH must be on its own closed list
    def add(item, expression, suffix):
        return kit.add_source(repo, store, "example", item, {
            "type": "git", "address": git_upstream[0] + suffix, "revision": git_upstream[1],
            "licenses": [{"license": expression, "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT}]})

    add("eclipse", "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0", "")
    assert repo.check() == []
    neither = add("neither", "EPL-2.0 OR CC-BY-SA-4.0", "/.")
    unknown = add("unknown", "GPL-2.0-or-later WITH Some-exception", "/./")
    problems = repo.check()
    assert f"{neither}: license expression EPL-2.0 OR CC-BY-SA-4.0 is not on the allowed list" in problems
    assert f"{unknown}: license expression GPL-2.0-or-later WITH Some-exception is not on the allowed list" in problems
    assert len(problems) == 2


def test_migrating_to_shape_4_moves_a_licenseref_text_into_source_json(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "book", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "licenses": [{"license": "LicenseRef-public-domain", "paths": ["**"], "attribution": "A. Author, Book"}]},
        compute_checksum=False)
    folder = repo.data / "sources" / sid
    path = folder / "source.json"
    proof = "Public domain: the author died in 1850.\n"
    (folder / "LicenseRef-public-domain.txt").write_text(proof, encoding="utf-8", newline="\n")
    path.write_text(dump_json({**json.loads(path.read_text()), "schema": 3}))

    assert migrate(repo, log=lambda _: None) == [path]
    assert json.loads(path.read_text())["licenses"][0]["text"] == proof
    assert not (folder / "LicenseRef-public-domain.txt").exists()
    assert repo.check() == []


def test_command_line_builds_a_dataset_and_prints_its_id(kit, repo, store, git_upstream, monkeypatch, capsys):
    kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "tags": [{"tags": ["text"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "text", "select": "text", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    kit.write_recipe(repo, "dataset", "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())

    assert main(["--root", str(repo.root), "check", "--fetch"]) == 0
    assert main(["--root", str(repo.root), "build", "d"]) == 0
    dataset_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert store.get_ref("datasets", dataset_id)
    assert repo.lock_path("t").exists()


def test_migrating_shape_1_tokenizer_files(kit, repo):
    # recipes start at vocabulary version 1; an old lock lists no documents, so it cannot be kept
    recipe = kit.write_recipe(repo, "tokenizer", "t", {"mixture": "m", "vocab_size": 556})
    recipe.write_text(json.dumps({**json.loads(recipe.read_text()), "schema": 1}))
    lock = repo.lock_path("t")
    lock.write_text(json.dumps({"schema": 1, "fingerprint": "0" * 64, "vocabulary_sha256": "1" * 64}))
    with pytest.raises(OutdatedSchema, match=re.escape(f"{lock}: a shape 1 lock does not list the documents")):
        migrate(repo, log=lambda _: None)
    assert json.loads(recipe.read_text()) == {"schema": 2, "mixture": "m", "vocab_size": 556, "version": 1}
    lock.unlink()  # what the message asks; the next build writes a shape 2 lock
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "a", "select": "a", "weight": 1}]})
    assert migrate(repo, log=lambda _: None) == []
