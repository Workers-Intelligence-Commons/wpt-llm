"""What a command does with --session, WPT_SESSION and --from."""

import pytest

from wpt.dataset import main
from wpt.store import main as store_command
from wpt_store import find_dataset, latest_dataset
from wpt_utils.lines import bare


@pytest.fixture
def ready(kit, repo, store, git_upstream, monkeypatch):
    """A source downloaded into the session of store, and the recipe d."""
    kit.add_source(repo, store, "example", "ok", {
        "type": "git", "address": git_upstream[0], "tags": [{"tags": ["text"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "text", "select": "text", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"version": 1, "sample_bytes": 10**9, "vocab_size": 556})
    kit.write_model_recipe(repo, "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 50, "held_out": 0})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("WPT_SESSION", raising=False)
    monkeypatch.delenv("WPT_FROM", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    return ["--root", str(repo.root)]


def test_a_command_given_no_session_is_refused_by_name_and_writes_nothing(ready, store, capsys):
    assert main([*ready, "build", "d"]) == 1
    assert "SessionNeeded" in capsys.readouterr().err
    assert latest_dataset(store, "d") is None and store.record_folders() == []


def test_a_build_reads_the_session_it_is_given_and_writes_there(ready, store, capsys):
    assert main([*ready, "build", "d", "--session", store.session]) == 0
    dataset = bare(capsys.readouterr().out).strip().splitlines()[-1]
    assert latest_dataset(store, "d")["id"] == dataset


def test_a_build_from_another_session_writes_under_its_own_and_names_the_old_one(ready, kit, store, capsys):
    assert main([*ready, "build", "d", "--session", "next", "--from", store.session]) == 0
    dataset = bare(capsys.readouterr().out).strip().splitlines()[-1]
    view = kit.open_store(store.root.as_uri()).at("next", store.session)
    key, record = find_dataset(view, dataset)
    assert key.startswith("next/") and record["from"] == store.session
    assert latest_dataset(store, "d") is None  # the session it started from is left as it was


def test_a_dataset_another_session_built_is_found_by_id_and_stays_where_it_lies(ready, kit, store, capsys):
    assert main([*ready, "build", "d", "--session", store.session]) == 0
    dataset = bare(capsys.readouterr().out).strip().splitlines()[-1]
    assert main([*ready, "build", "d", "--session", "next", "--from", store.session]) == 0
    assert bare(capsys.readouterr().out).strip().splitlines()[-1] == dataset
    # the same content finds the same folder: nothing is built or copied under the new session
    assert kit.open_store(store.root.as_uri()).kind.list("next/datasets/") == {}
    view = kit.open_store(store.root.as_uri()).at("next", store.session)
    assert find_dataset(view, dataset)[0].startswith(f"{store.session}/datasets/d-{dataset[:8]}/")
    assert latest_dataset(view, "d")["id"] == dataset


def test_the_store_list_shows_the_session_named_and_no_other(ready, store, capsys):
    assert store_command(["list", "--session", store.session]) == 0
    assert store.session in capsys.readouterr().out
    assert store_command(["list", "--session", "other"]) == 0
    assert store.session not in capsys.readouterr().out


def test_the_store_commands_that_find_an_id_work_with_no_session(ready, store, capsys):
    assert main([*ready, "build", "d", "--session", store.session]) == 0
    capsys.readouterr()
    assert store_command(["list"]) == 0
    assert f"{store.session}/example-ok" in capsys.readouterr().out
    assert store_command(["runs"]) == 0
    assert f"{store.session}/" in capsys.readouterr().out
    assert store_command(["rebuild-index"]) == 0


def test_a_damaged_index_is_made_again_from_the_files_by_a_command(ready, store, capsys):
    store.kind.write("index", b"no index")
    assert store_command(["list", "--session", store.session]) == 2
    assert "CorruptFile" in capsys.readouterr().err
    assert store_command(["rebuild-index", "--session", "other"]) == 0
    capsys.readouterr()
    assert store_command(["list", "--session", store.session]) == 0
    assert f"{store.session}/example-ok" in capsys.readouterr().out


def test_a_copy_out_given_no_session_is_refused_by_name(ready, tmp_path, capsys):
    assert store_command(["copy-out", "fox", str(tmp_path / "copy")]) == 2
    assert "SessionNeeded" in capsys.readouterr().err
