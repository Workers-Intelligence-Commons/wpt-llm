import json
import re

import pytest

from wpt_dataset.cli import main
from wpt_dataset.errors import InvalidFile
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
        "licenses": [{"license": "CC-BY-4.0", "paths": ["**"], "attribution": kit.ATTRIBUTION}]})
    wrong_id = repo.data / "sources" / "example-wrong-00000000"
    wrong_id.mkdir()
    (wrong_id / "source.json").write_text((repo.data / "sources" / good / "source.json").read_text())
    (repo.data / "sources" / good / "0002-skipped.patch").write_text("")
    kit.write_recipe(repo, "mixture", "half", {"pools": [{"name": "a", "select": "a", "weight": 0.5}]})

    problems = "\n".join(repo.check())
    assert f"{grey}: license CC-BY-4.0 is not on the allowed list" in problems
    assert f"the id must end with {address_hash(git_upstream[0])}" in problems
    assert f"{good}: patches must be numbered 0001" in problems
    assert "pool weights add up to 0.5, not 1" in problems


def test_migrate_rewrites_files_that_are_behind(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.data / "sources" / sid / "source.json"
    # a made-up next shape: "type" renamed to "kind"
    step = {3: lambda doc: {**{k: v for k, v in doc.items() if k != "type"}, "kind": doc["type"], "schema": 4}}
    current = {"source": 4, "mixture": 1, "dataset": 1, "tokenizer": 1, "lock": 1}

    changed = migrate(repo, migrations={"source": step, "mixture": {}, "dataset": {}, "tokenizer": {}, "lock": {}},
                      current=current, log=lambda _: None)
    assert changed == [path]
    document = json.loads(path.read_text())
    assert document["schema"] == 4 and document["kind"] == "git" and "type" not in document
    assert path.read_text() == dump_json(document)


def test_migrating_a_shape_1_source_asks_a_person_for_attribution_and_formats(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.data / "sources" / sid / "source.json"
    old = json.loads(path.read_text())
    old["schema"] = 1
    del old["licenses"][0]["attribution"]
    del old["formats"]
    path.write_text(dump_json(old))

    with pytest.raises(InvalidFile, match=re.escape(str(path))):
        migrate(repo, log=lambda _: None)
    document = json.loads(path.read_text())
    assert document["schema"] == 3 and document["licenses"] == [{"license": "MIT", "paths": ["**"], "attribution": ""}]
    assert document["formats"] == []

    document["licenses"][0]["attribution"] = kit.ATTRIBUTION  # what the person writes
    path.write_text(dump_json(document))
    with pytest.raises(InvalidFile, match=re.escape(f"{path}: at formats")):
        migrate(repo, log=lambda _: None)
    document["formats"] = [{"format": "text", "paths": ["**"]}]  # the person names each file's format
    path.write_text(dump_json(document))
    assert migrate(repo, log=lambda _: None) == [] and repo.check() == []


def test_command_line_builds_a_dataset_and_prints_its_id(kit, repo, store, git_upstream, monkeypatch, capsys):
    kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "tags": [{"tags": ["text"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "text", "select": "text", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"mixture": "m", "vocab_size": 300})
    kit.write_recipe(repo, "dataset", "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())

    assert main(["--root", str(repo.root), "check", "--fetch"]) == 0
    assert main(["--root", str(repo.root), "build", "d"]) == 0
    dataset_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert store.get_ref("datasets", dataset_id)
    assert repo.lock_path("t").exists()
