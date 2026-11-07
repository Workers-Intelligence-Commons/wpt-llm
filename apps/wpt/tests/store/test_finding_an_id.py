"""`wpt store list`, `runs`, `run` and `rebuild-index` with and without an ID."""

from wpt_store import open_store
from wpt_telemetry import Telemetry

from wpt.store import main


def _record(store, message):
    with Telemetry.start(store, "wpt train", [], "x", env_file=store.root / ".env", environ={},
                         echo=lambda *_: None) as telemetry:
        telemetry.log(message)


def test_a_store_command_with_no_id_shows_every_session_and_an_id_narrows_it(built, monkeypatch, capsys):
    other = open_store(built.store.root.as_uri()).at("other-session")
    _record(other, "from the other session")
    _record(built.store, "from the built session")
    monkeypatch.delenv("WPT_SESSION")
    capsys.readouterr()

    assert main(["list"]) == 0
    everything = capsys.readouterr().out
    assert "test-session" in everything and f"test-session/{built.source_id}" in everything

    assert main(["runs"]) == 0
    runs = capsys.readouterr().out
    assert "test-session/" in runs and "other-session/" in runs
    assert main(["runs", "--session", "other-session"]) == 0
    narrowed = capsys.readouterr().out
    assert "other-session/" in narrowed and "test-session/" not in narrowed
    assert main(["run", "other-session/"]) == 0  # a part of the name only one folder holds, with no ID
    assert "wpt train" in capsys.readouterr().out

    assert main(["list", "--session", "other-session"]) == 0
    assert f"test-session/{built.source_id}" not in capsys.readouterr().out

    assert main(["rebuild-index"]) == 0
    assert "the index holds" in capsys.readouterr().out


def test_a_start_with_no_session_is_refused_and_a_command_that_needs_a_session_still_is(built, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("WPT_SESSION")
    assert main(["list", "--from", "old"]) == 2
    assert "StoreSettingError" in capsys.readouterr().err
    assert main(["copy-out", "model", str(tmp_path / "out")]) == 2
    assert "SessionNeeded" in capsys.readouterr().err
