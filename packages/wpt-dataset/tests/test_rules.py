import json
import re

import pytest

from wpt_dataset.cli import main
from wpt_dataset.errors import InvalidFile, OutdatedSchema
from wpt_dataset.migrate import NEW_ATTENTION, migrate
from wpt_dataset.pools import matches, parse
from wpt_dataset.repo import CURRENT_SCHEMA, address_hash, dump_json


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

    noncommercial = kit.add_source(repo, store, "example", "noncommercial", {
        "type": "git", "address": git_upstream[0] + "/.", "revision": git_upstream[1],
        "licenses": [{"license": "CC-BY-NC-4.0", "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT}]})
    unexplained = kit.add_source(repo, store, "example", "unexplained", {
        "type": "git", "address": git_upstream[0] + "/./", "revision": git_upstream[1],
        "decision": {"decision": "refused", "by": "Test Reviewer", "date": "2026-10-14"}})  # a refusal must say why
    wrong_id = repo.root / "data" / "sources" / "example-wrong-00000000"
    wrong_id.mkdir()
    (wrong_id / "source.json").write_text((repo.root / "data" / "sources" / good / "source.json").read_text())
    (repo.root / "data" / "sources" / good / "0002-skipped.patch").write_text("")
    kit.write_recipe(repo, "mixture", "half", {"pools": [{"name": "a", "select": "a", "weight": 0.5}]})

    problems = "\n".join(repo.check())
    assert f"{noncommercial}: license CC-BY-NC-4.0 is not on the allowed list" in problems
    assert any(unexplained in line and "'reason' is a required property" in line for line in problems.splitlines())
    assert f"the id must end with {address_hash(git_upstream[0])}" in problems
    assert f"{good}: patches must be numbered 0001" in problems
    assert "pool weights add up to 0.5, not 1" in problems


def test_migrate_rewrites_files_that_are_behind(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.root / "data" / "sources" / sid / "source.json"
    # a made-up next shape: "type" renamed to "kind"
    step = {7: lambda doc, _path: {**{k: v for k, v in doc.items() if k != "type"}, "kind": doc["type"], "schema": 8}}
    current = {"source": 8, "mixture": 1, "model": 1, "tokenizer": 1, "lock": 1}

    changed = migrate(repo, migrations={"source": step, "mixture": {}, "model": {}, "tokenizer": {}, "lock": {}},
                      current=current, log=lambda _: None)
    assert changed == [path.relative_to(repo.root).as_posix()]
    document = json.loads(path.read_text())
    assert document["schema"] == 8 and document["kind"] == "git" and "type" not in document
    assert path.read_text() == dump_json(document)


def test_migrating_a_shape_1_source_asks_a_person_for_attribution_text_and_formats(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.root / "data" / "sources" / sid / "source.json"
    old = json.loads(path.read_text())
    old["schema"] = 1
    del old["licenses"][0]["attribution"]
    del old["licenses"][0]["text"]
    del old["formats"]
    path.write_text(dump_json(old))

    with pytest.raises(InvalidFile, match=re.escape(str(path))):
        migrate(repo, log=lambda _: None)
    document = json.loads(path.read_text())
    assert document["schema"] == 7
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
    # WPT uses dual-licensed content under the alternative it allows; licenses
    # joined by AND all apply, so each must be allowed; an exception after WITH
    # must be on its own closed list
    def add(item, expression, suffix):
        return kit.add_source(repo, store, "example", item, {
            "type": "git", "address": git_upstream[0] + suffix, "revision": git_upstream[1],
            "licenses": [{"license": expression, "paths": ["**"], "attribution": kit.ATTRIBUTION, "text": kit.LICENSE_TEXT}]})

    add("eclipse", "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0", "")
    add("both", "MIT AND Apache-2.0", "//")
    add("either", "CC-BY-NC-4.0 AND MIT OR CC0-1.0", "///")  # AND binds first: CC0-1.0 alone is allowed
    assert repo.check() == []
    neither = add("neither", "EPL-2.0 OR CC-BY-NC-4.0", "/.")
    unknown = add("unknown", "GPL-2.0-or-later WITH Some-exception", "/./")
    half = add("half", "MIT AND CC-BY-NC-4.0", "/.//")
    problems = repo.check()
    assert f"{neither}: license expression EPL-2.0 OR CC-BY-NC-4.0 is not on the allowed list" in problems
    assert f"{unknown}: license expression GPL-2.0-or-later WITH Some-exception is not on the allowed list" in problems
    assert f"{half}: license expression MIT AND CC-BY-NC-4.0 is not on the allowed list" in problems
    assert len(problems) == 3


def test_the_check_reads_the_lock_and_licenses_of_each_image(repo):
    # data/images/<name>/ follows the data-file rules: its two written files
    # match their schemas, and every image folder has its Dockerfile
    folder = repo.root / "data" / "images" / "tiny"
    folder.mkdir(parents=True)
    (folder / "Dockerfile").write_text("ARG BASE=debian:trixie-slim\nFROM ${BASE}\n")
    lock = {"schema": 1, "base": "debian@sha256:" + "a" * 64, "snapshot": "20261017T120000Z",
            "packages": {"bash": "5.2.37-2", "hello": "2.10-5"}}
    licenses = {"schema": 1, "packages": {"bash": {"version": "5.2.37-2", "licenses": ["GPL-3+"]},
                                          "hello": {"version": "2.10-5", "text": "Copyright: the authors\n"}}}
    (folder / "lock.json").write_text(dump_json(lock))
    (folder / "licenses.json").write_text(dump_json(licenses))
    assert repo.check() == []

    (folder / "lock.json").write_text(dump_json({**lock, "snapshot": "2026-10-17"}))
    (repo.root / "data" / "images" / "other").mkdir()
    problems = repo.check()
    assert problems == ["no image 'other': data/images/other/Dockerfile does not exist",
                        f"{folder / 'lock.json'}: at snapshot: '2026-10-17' does not match '^[0-9]{{8}}T[0-9]{{6}}Z$'"]


def test_migrating_to_shape_4_moves_a_licenseref_text_into_source_json(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "book", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "licenses": [{"license": "LicenseRef-public-domain", "paths": ["**"], "attribution": "A. Author, Book"}]},
        compute_checksum=False)
    folder = repo.root / "data" / "sources" / sid
    path = folder / "source.json"
    proof = "Public domain: the author died in 1850.\n"
    (folder / "LicenseRef-public-domain.txt").write_text(proof, encoding="utf-8", newline="\n")
    path.write_text(dump_json({**json.loads(path.read_text()), "schema": 3}))

    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    assert json.loads(path.read_text())["licenses"][0]["text"] == proof
    assert not (folder / "LicenseRef-public-domain.txt").exists()
    assert repo.check() == []


def test_migrating_to_shape_6_gives_each_source_one_of_three_states(kit, repo, store, git_upstream):
    # under-review, admitted, refused; a
    # refusal "under review" was a review, a retired source is a refused one
    by = {"by": "Test Reviewer", "date": "2026-10-14"}
    old = {"review": {"decision": "refused", **by, "reason": "under review"},
           "moved": {"decision": "retired", **by, "reason": "replaced by example-new-0123abcd", "note": "kept"},
           "refused": {"decision": "refused", **by, "reason": "missing fact"},
           "admitted": {"decision": "admitted", **by}}
    paths = {}
    for suffix, (item, decision) in zip(("", "/.", "/./", "//"), old.items()):
        sid = kit.add_source(repo, store, "example", item, {
            "type": "git", "address": git_upstream[0] + suffix, "revision": git_upstream[1]})
        paths[item] = repo.root / "data" / "sources" / sid / "source.json"
        paths[item].write_text(dump_json({**json.loads(paths[item].read_text()), "schema": 5, "decision": decision}))

    assert sorted(migrate(repo, log=lambda _: None)) == sorted(p.relative_to(repo.root).as_posix() for p in paths.values())
    decisions = {item: json.loads(path.read_text())["decision"] for item, path in paths.items()}
    assert decisions == {"review": {"decision": "under-review", **by},
                         "moved": {"decision": "refused", **by, "reason": "replaced by example-new-0123abcd",
                                   "note": "kept"},
                         "refused": old["refused"], "admitted": old["admitted"]}
    assert repo.check() == []


def test_command_line_builds_a_dataset_and_prints_its_id(kit, repo, store, git_upstream, monkeypatch, capsys):
    kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "tags": [{"tags": ["text"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "text", "select": "text", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert main(["--root", str(repo.root), "check", "--fetch"]) == 0
    assert main(["--root", str(repo.root), "build", "d"]) == 0
    dataset_id = capsys.readouterr().out.strip().splitlines()[-1]
    assert store.get_ref("datasets", dataset_id)
    assert (repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json").exists()
    # the build's record names the dataset it wrote, which training records as its input,
    # and each source it read
    [folder] = store.record_folders()
    assert "-wpt-dataset-build-d-" in folder  # named after the recipe it built
    ids = json.loads(store.read_record(folder, "ids.json"))
    assert ids["outputs"] == {"dataset": dataset_id, "dataset_record": store.get_ref("datasets", dataset_id),
                              "dataset_documents": store.get_ref("dataset-documents", dataset_id)}
    assert list(ids["inputs"]["sources"]) == repo.source_ids()
    events = [json.loads(line) for line in store.read_record(folder, "events.jsonl").splitlines()]
    fetched = {e["attributes"]["source"] for e in events if e.get("name") == "source" and e["kind"] == "span_start"}
    assert fetched == set(repo.source_ids())
    [locked] = [e["attributes"] for e in events if e.get("name") == "tokenizer.lock"]  # the lock file it wrote
    assert locked == {"path": "data/recipes/tokenizers/t.lock.json",
                      "vocabulary": json.loads((repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json").read_text())["vocabulary_sha256"]}
    assert json.loads(store.read_record(folder, "end.json"))["status"] == "succeeded"


def test_a_build_whose_record_breaks_writes_no_lock_file(kit, repo, store, git_upstream, monkeypatch, capsys):
    # the lock changes the repository: it may change only once the record
    # says so, like every other step of a build
    from wpt_store import LocalStore

    kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "tags": [{"tags": ["text"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "text", "select": "text", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    real = LocalStore._raw_append_record

    def disk_full_at_the_lock(self, folder, name, data):
        if b'"name": "tokenizer.lock"' in data:
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full_at_the_lock)
    assert main(["--root", str(repo.root), "build", "d"]) == 3
    assert "No space left on device" in capsys.readouterr().err
    assert not (repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json").exists()


def test_migrating_shape_1_tokenizer_files(kit, repo):
    # recipes start at vocabulary version 1; an old lock lists no documents, so it cannot be kept
    recipe = kit.write_recipe(repo, "tokenizer", "t", {"mixture": "m", "vocab_size": 556})
    recipe.write_text(json.dumps({**json.loads(recipe.read_text()), "schema": 1}))
    lock = (repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json")
    lock.write_text(json.dumps({"schema": 1, "fingerprint": "0" * 64, "vocabulary_sha256": "1" * 64}))
    with pytest.raises(OutdatedSchema, match=re.escape(f"{lock}: a shape 1 lock does not list the documents")):
        migrate(repo, log=lambda _: None)
    assert json.loads(recipe.read_text()) == {"schema": 2, "mixture": "m", "vocab_size": 556, "version": 1}
    lock.unlink()  # what the message asks; the next build writes a shape 2 lock
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "a", "select": "a", "weight": 1}]})
    assert migrate(repo, log=lambda _: None) == []


def test_migrating_a_shape_2_model_recipe_asks_a_person_for_the_new_attention_numbers(kit, repo):
    # gated MLA's numbers go; CSA and HCA's, and which global layer is which,
    # are a person's choice, so the check names the file until they are filled in
    old = {"block_size": 16, **{key: value for key, value in kit.MODEL.items()
                                if key not in (*NEW_ATTENTION, "global_attention")}}
    old.update({"mla_heads": 2, "mla_head_dim": 8, "mla_kv_rank": 8})
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    training = {key: value for key, value in kit.TRAINING.items() if key != "stages"}
    path.write_text(dump_json({**json.loads(path.read_text()), "schema": 2, "model": old,
                               "training": {**training, "steps": 1, "batch_size": 4}}))

    with pytest.raises(InvalidFile, match=re.escape(f"{path}: at model/")):
        migrate(repo, log=lambda _: None)
    document = json.loads(path.read_text())
    assert document["schema"] == CURRENT_SCHEMA["model"]
    assert not any(key.startswith("mla_") for key in document["model"])
    assert {key: document["model"][key] for key in NEW_ATTENTION} == dict.fromkeys(NEW_ATTENTION, 0)
    assert document["model"]["global_attention"] == []

    document["model"].update({key: kit.MODEL[key] for key in (*NEW_ATTENTION, "global_attention")})  # the person's choice
    path.write_text(dump_json(document))
    assert migrate(repo, log=lambda _: None) == []


def test_migrating_a_shape_3_model_recipe_makes_its_window_and_steps_one_stage(kit, repo):
    # docs/training.md, "Stages": a recipe trained on one window length
    # becomes one stage of that length, its steps and its batch; nothing
    # else changes
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    new = json.loads(path.read_text())
    training = {key: value for key, value in kit.TRAINING.items() if key != "stages"}
    path.write_text(dump_json({**new, "schema": 3, "model": {"block_size": 16, **kit.MODEL},
                               "training": {**training, "steps": 1, "batch_size": 4}}))
    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    assert json.loads(path.read_text()) == new


def test_migrating_a_shape_4_model_recipe_adds_the_drafter_s_training_values_after_training(kit, repo):
    # docs/model.md, "The drafter": shape 5 adds the recipe's drafter
    # section, with the training values every recipe states, and nothing else changes
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    new = json.loads(path.read_text())
    path.write_text(dump_json({**{key: value for key, value in new.items() if key != "drafter"}, "schema": 4}))
    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    migrated = json.loads(path.read_text())
    assert migrated["drafter"] == {"prompts": 256, "prompt_tokens": 64, "answer_tokens": 256, "epochs": 10,
                                   "learning_rate": 6e-4, "held_out": 0.1, "anchors": 512, "batch": 8, "seed": 0}
    assert migrated == {**new, "drafter": migrated["drafter"]}
    assert list(migrated) == ["schema", "data", "model", "training", "drafter", "posttrain"]
