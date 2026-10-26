import json
import re

import pytest

from wpt_dataset.cli import main
from wpt_dataset.errors import InvalidFile, OutdatedSchema
from wpt_dataset.migrate import NEW_ATTENTION, Step, migrate, outdated, upgrade
from wpt_dataset.pools import matches, parse
from wpt_dataset.repo import address_hash, dump_json, validate


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
        "decision": {"decision": "refused", "by": "Test Reviewer", "date": "2026-10-14"}},  # a refusal must say why
        fetch=False)  # an invalid source.json: never downloaded
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


def test_a_machine_drafts_but_never_admits_or_refuses_and_a_person_s_name_holds_no_colon(kit, repo, store,
                                                                                       git_upstream):
    # always-on monitoring saves drafts under `machine:<host>`; the check refuses that name on an admission or
    # a refusal, and a colon in a person's name, so a machine's name is never taken for a person's
    def decided(item, decision, by, **more):
        return kit.add_source(repo, store, "example", item, {
            "type": "git", "address": git_upstream[0] + "/" * len(item), "revision": git_upstream[1],
            "decision": {"decision": decision, "by": by, "date": "2026-10-25", **more}}, fetch=False)

    drafted = decided("a", "under-review", "machine:monitor-1")
    assert repo.check() == []
    admitted = decided("bb", "admitted", "machine:monitor-1")
    refused = decided("ccc", "refused", "machine:monitor-1", reason="not public")
    colon = decided("dddd", "admitted", "A: Person")
    empty = decided("eeeee", "under-review", "machine:")
    problems = repo.check()
    assert len(problems) == 4 and not any(drafted in line for line in problems)
    for sid in (admitted, refused):
        assert any(line.startswith(f"{sid}: ") and "a machine never admits or refuses" in line for line in problems)
    assert any(line.startswith(f"{colon}: ") and "a person's name holds no ':'" in line for line in problems)
    assert any(line.startswith(f"{empty}: ") and "a machine's name is `machine:<host>`" in line for line in problems)


def test_migrate_rewrites_the_files_a_step_recognizes_by_their_fields(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.root / "data" / "sources" / sid / "source.json"
    # a made-up next shape: "type" renamed to "kind"
    rename = Step("a source with `type`", lambda doc: "type" in doc,
                  lambda doc, _where: {**{k: v for k, v in doc.items() if k != "type"}, "kind": doc["type"]})
    steps = {"source": [rename]}

    assert outdated(json.loads(path.read_text()), "source", steps) == ["a source with `type`"]
    changed = migrate(repo, steps=steps, log=lambda _: None)
    assert changed == [path.relative_to(repo.root).as_posix()]
    document = json.loads(path.read_text())
    assert document["kind"] == "git" and "type" not in document
    assert path.read_text() == dump_json(document)
    assert outdated(document, "source", steps) == [] and migrate(repo, steps=steps, log=lambda _: None) == []


def test_upgrade_brings_a_document_to_the_current_shape_in_memory(kit):
    old = {"schema": 3, "messages": []}
    assert outdated(old, "conversation") == ["a shape number (`schema`)"]
    assert upgrade(old, "conversation", "x") == {"messages": []} and old == {"schema": 3, "messages": []}
    current = {"messages": []}
    assert outdated(current, "conversation") == [] and upgrade(current, "conversation", "x") is current
    assert outdated([1], "conversation") == [] and upgrade("text", "conversation", "x") == "text"


def test_migrate_drops_the_shape_number_of_every_kind_of_file(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    source = repo.root / "data" / "sources" / sid / "source.json"
    mixture = kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "a", "select": "a", "weight": 1}]})
    paths = (source, mixture)
    for path in paths:
        path.write_text(dump_json({"schema": 1, **json.loads(path.read_text())}))
    with pytest.raises(OutdatedSchema, match="an earlier shape"):
        validate(json.loads(source.read_text()), "source", source)

    assert sorted(migrate(repo, log=lambda _: None)) == sorted(p.relative_to(repo.root).as_posix() for p in paths)
    assert not any("schema" in json.loads(path.read_text()) for path in paths) and repo.check() == []


def test_migrating_a_source_renames_the_old_training_step_tags(kit, repo, store, git_upstream):
    # docs/training.md, "Steps": a tag that picks a step's examples is the
    # step's name; sft, grpo and token-efficiency become the names of the
    # steps that do the same, in every tag list, and any other tag stays
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.root / "data" / "sources" / sid / "source.json"
    old = json.loads(path.read_text())
    old["tags"] = [{"tags": ["en", "sft"], "paths": ["a/**"]},
                   {"tags": ["grpo", "token-efficiency"], "paths": ["b/**"], "note": "tasks"},
                   {"tags": ["sft", "answer-as-assistant"], "paths": ["c/**"]}]
    path.write_text(dump_json(old))

    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    document = json.loads(path.read_text())
    assert document["tags"] == [{"tags": ["en", "answer-as-assistant"], "paths": ["a/**"]},
                                {"tags": ["finish-tasks", "finish-tasks-with-less-writing"], "paths": ["b/**"],
                                 "note": "tasks"},
                                {"tags": ["answer-as-assistant"], "paths": ["c/**"]}]
    assert migrate(repo, log=lambda _: None) == [] and repo.check() == []


def test_migrating_a_source_without_credit_text_or_formats_asks_a_person_for_attribution_text_and_formats(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1]})
    path = repo.root / "data" / "sources" / sid / "source.json"
    old = json.loads(path.read_text())
    del old["licenses"][0]["attribution"]
    del old["licenses"][0]["text"]
    del old["formats"]
    path.write_text(dump_json(old))

    with pytest.raises(InvalidFile, match=re.escape(str(path))):
        migrate(repo, log=lambda _: None)
    document = json.loads(path.read_text())
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
    lock = {"base": "debian@sha256:" + "a" * 64, "snapshot": "20261017T120000Z",
            "packages": {"bash": "5.2.37-2", "hello": "2.10-5"}}
    licenses = {"packages": {"bash": {"version": "5.2.37-2", "licenses": ["GPL-3+"]},
                             "hello": {"version": "2.10-5", "text": "Copyright: the authors\n"}}}
    (folder / "lock.json").write_text(dump_json(lock))
    (folder / "licenses.json").write_text(dump_json(licenses))
    assert repo.check() == []

    (folder / "lock.json").write_text(dump_json({**lock, "snapshot": "2026-10-17"}))
    (repo.root / "data" / "images" / "other").mkdir()
    problems = repo.check()
    assert problems == ["no image 'other': data/images/other/Dockerfile does not exist",
                        f"{folder / 'lock.json'}: at snapshot: '2026-10-17' does not match '^[0-9]{{8}}T[0-9]{{6}}Z$'"]


def test_migrating_moves_a_licenseref_text_into_source_json(kit, repo, store, git_upstream):
    sid = kit.add_source(repo, store, "example", "book", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "licenses": [{"license": "LicenseRef-public-domain", "paths": ["**"], "attribution": "A. Author, Book"}]},
        compute_checksum=False)
    folder = repo.root / "data" / "sources" / sid
    path = folder / "source.json"
    proof = "Public domain: the author died in 1850.\n"
    (folder / "LicenseRef-public-domain.txt").write_text(proof, encoding="utf-8", newline="\n")

    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    assert json.loads(path.read_text())["licenses"][0]["text"] == proof
    assert not (folder / "LicenseRef-public-domain.txt").exists()
    assert repo.check() == []


def test_migrating_gives_each_source_one_of_three_states(kit, repo, store, git_upstream):
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
        paths[item].write_text(dump_json({**json.loads(paths[item].read_text()), "decision": decision}))

    changed = sorted(migrate(repo, log=lambda _: None))  # a decision already of the current kinds stays as it is
    assert changed == sorted(paths[item].relative_to(repo.root).as_posix() for item in ("review", "moved"))
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
    # check --fetch's session and the build's record, named after the recipe it built
    folders = store.record_folders()
    [downloading] = [name for name in folders if "-wpt-dataset-check-" in name]
    [folder] = [name for name in folders if "-wpt-dataset-build-d-" in name]
    assert len(folders) == 2
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
    # the build is a session too: one `reused` line for each source whose tree it used; a command that only
    # reads sources (markdown-check, unpack) writes no session
    [sid] = repo.source_ids()
    assert main(["--root", str(repo.root), "markdown-check"]) == 0
    assert main(["--root", str(repo.root), "unpack", sid, str(repo.root.parent / "unpacked")]) == 0
    with store.database() as found:
        sessions = found.execute("SELECT folder, command, ended IS NOT NULL FROM sessions").fetchall()
        assert {(downloading, "wpt-dataset-check", 1), (folder, "wpt-dataset-build", 1)} <= set(sessions)
        assert len(sessions) == 3  # and the test's own download, which put the source in the store
        assert found.execute("SELECT source, outcome, checksum, size > 0 FROM lines WHERE session = ?",
                             (folder,)).fetchall() == [(sid, "reused", repo.source(sid).data["checksum"], 1)]


@pytest.mark.parametrize("stream", [False, True])
def test_building_a_tokenizer_is_a_session_with_a_reused_line_for_each_source_it_read(
        kit, repo, store, git_upstream, monkeypatch, stream):
    [sid] = [kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "revision": git_upstream[1],
        "tags": [{"tags": ["text"], "paths": ["**"]}]})]
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "text", "select": "text", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert main(["--root", str(repo.root), "build-tokenizer", "t", *(["--stream"] if stream else [])]) == 0
    [folder] = [name for name in store.record_folders() if "-wpt-dataset-build-tokenizer-t-" in name]
    with store.database() as found:
        assert found.execute("SELECT command, ended IS NOT NULL FROM sessions WHERE folder = ?",
                             (folder,)).fetchall() == [("wpt-dataset-build-tokenizer", 1)]
        assert found.execute("SELECT source, outcome, checksum FROM lines WHERE session = ?", (folder,)).fetchall() \
            == [(sid, "reused", repo.source(sid).data["checksum"])]
    lock = repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json"
    assert lock.exists()

    # the vocabulary is locked now: a second command reads no source, so it writes no line
    assert main(["--root", str(repo.root), "build-tokenizer", "t"]) == 0
    [_, again] = sorted(name for name in store.record_folders() if "-wpt-dataset-build-tokenizer-t-" in name)
    with store.database() as found:
        assert found.execute("SELECT count(*) FROM lines WHERE session = ?", (again,)).fetchone() == (0,)


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


def test_migrating_tokenizer_files(kit, repo):
    # recipes start at vocabulary version 1; a lock that lists no documents cannot be kept
    recipe = kit.write_recipe(repo, "tokenizer", "t", {"mixture": "m", "vocab_size": 556})
    lock = (repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json")
    lock.write_text(json.dumps({"fingerprint": "0" * 64, "vocabulary_sha256": "1" * 64}))
    with pytest.raises(OutdatedSchema, match=re.escape(f"{lock}: the lock does not list the documents")):
        migrate(repo, log=lambda _: None)
    assert json.loads(recipe.read_text()) == {"mixture": "m", "vocab_size": 556, "version": 1}
    lock.unlink()  # what the message asks; the next build writes a current lock
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "a", "select": "a", "weight": 1}]})
    assert migrate(repo, log=lambda _: None) == []


def test_migrating_a_lock_drops_the_tokenizer_code_s_version(kit, repo):
    # the vocabulary's fingerprint holds the tokenizer code's hash; the lock records no version of the code
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "a", "select": "a", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    lock = repo.root / "data" / "recipes" / "tokenizers" / "t.lock.json"
    current = {"recipe": {"version": 1}, "documents": [["a", "2" * 64]], "vocabulary_sha256": "1" * 64}
    lock.write_text(dump_json({**current, "wpt_tokenizer": "0.1.0"}))
    assert outdated(json.loads(lock.read_text()), "lock") == ["the tokenizer code's version"]
    assert migrate(repo, log=lambda _: None) == [lock.relative_to(repo.root).as_posix()]
    assert json.loads(lock.read_text()) == current


# the smallest training part of a recipe with `training` and `posttrain`, which becomes its read-text step (kit.STEPS)
TRAINING = {"method": "adamw", "settings": {}, "stages": [{"block_size": 16, "steps": 1, "batch_size": 4}],
              "eval_every": 1, "eval_batches": 1, "seed": 1337}


def with_training(document, training=TRAINING, posttrain=()):
    """A current recipe written as an older shape: `training` and `posttrain`
    where `seed` and `steps` are."""
    old = {}
    for key, value in document.items():
        if key == "seed":
            old["training"] = training
        elif key == "drafter":
            old[key], old["posttrain"] = value, list(posttrain)
        elif key != "steps":
            old[key] = value
    return old


def test_migrating_a_model_recipe_with_gated_mla_asks_a_person_for_the_new_attention_numbers(kit, repo):
    # gated MLA's numbers go; CSA and HCA's, and which global layer is which,
    # are a person's choice, so the check names the file until they are filled in
    old = {"block_size": 16, **{key: value for key, value in kit.MODEL.items()
                                if key not in (*NEW_ATTENTION, "global_attention")}}
    old.update({"mla_heads": 2, "mla_head_dim": 8, "mla_kv_rank": 8})
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    training = {key: value for key, value in TRAINING.items() if key != "stages"}
    path.write_text(dump_json({**with_training(json.loads(path.read_text()), {**training, "steps": 1, "batch_size": 4}),
                               "model": old}))

    with pytest.raises(InvalidFile, match=re.escape(f"{path}: at model/")):
        migrate(repo, log=lambda _: None)
    document = json.loads(path.read_text())
    assert "training" not in document
    assert not any(key.startswith("mla_") for key in document["model"])
    assert {key: document["model"][key] for key in NEW_ATTENTION} == dict.fromkeys(NEW_ATTENTION, 0)
    assert document["model"]["global_attention"] == []

    document["model"].update({key: kit.MODEL[key] for key in (*NEW_ATTENTION, "global_attention")})  # the person's choice
    path.write_text(dump_json(document))
    assert migrate(repo, log=lambda _: None) == []


def test_migrating_a_model_recipe_with_its_window_in_the_architecture_makes_its_window_and_steps_one_stage(kit, repo):
    # docs/training.md, "Stages": a recipe trained on one window length
    # becomes one stage of that length, its steps and its batch; nothing
    # else changes
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    new = json.loads(path.read_text())
    training = {key: value for key, value in TRAINING.items() if key != "stages"}
    path.write_text(dump_json({**with_training(new, {**training, "steps": 1, "batch_size": 4}),
                               "model": {"block_size": 16, **kit.MODEL}}))
    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    assert json.loads(path.read_text()) == new


def test_migrating_a_model_recipe_without_a_drafter_adds_the_drafter_s_training_values_after_training(kit, repo):
    # docs/model.md, "The drafter": a recipe states its drafter
    # section, with the training values every recipe states, and nothing else changes
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    new = json.loads(path.read_text())
    path.write_text(dump_json({key: value for key, value in with_training(new).items() if key != "drafter"}))
    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    migrated = json.loads(path.read_text())
    assert migrated["drafter"] == {"prompts": 256, "prompt_tokens": 64, "answer_tokens": 256, "epochs": 10,
                                   "learning_rate": 6e-4, "held_out": 0.1, "anchors": 512, "batch": 8, "seed": 0}
    assert migrated == {**new, "drafter": migrated["drafter"]}
    assert list(migrated) == ["data", "model", "seed", "steps", "drafter"]


def test_migrating_a_model_recipe_with_training_and_post_training_apart_makes_its_training_and_post_training_one_list_of_steps(kit, repo):
    # docs/training.md, "Steps": the training becomes read-text, its method
    # the step's optimizer and its loss settings the step's; each
    # post-training step follows under the name of the step that does the
    # same, its learning rate and weight decay given to adamw, a tag naming
    # an old step renamed with it
    path = kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    new = json.loads(path.read_text())
    training = {**TRAINING, "method": "muon", "settings": {"lr": 0.02, "mtp_weight": 0.5}}
    path.write_text(dump_json(with_training(new, training, [
        "sft", {"step": "grpo", "settings": {"learning_rate": 1e-6, "rounds": 2}},
        {"step": "token-efficiency", "settings": {"tag": "grpo", "weight_decay": 0.1}}])))
    assert migrate(repo, log=lambda _: None) == [path.relative_to(repo.root).as_posix()]
    migrated = json.loads(path.read_text())
    assert list(migrated) == ["data", "model", "seed", "steps", "drafter"]
    assert migrated["seed"] == 1337
    assert migrated["steps"] == [
        {"step": "read-text", "optimizer": {"name": "muon", "settings": {"lr": 0.02}},
         "settings": {"stages": TRAINING["stages"], "eval_every": 1, "eval_batches": 1, "mtp_weight": 0.5},
         "locked": []},
        {"step": "answer-as-assistant", "settings": {}},
        {"step": "finish-tasks", "settings": {"rounds": 2},
         "optimizer": {"name": "adamw", "settings": {"lr": 1e-6}}},
        {"step": "finish-tasks-with-less-writing", "settings": {"tag": "finish-tasks"},
         "optimizer": {"name": "adamw", "settings": {"weight_decay": 0.1}}}]
