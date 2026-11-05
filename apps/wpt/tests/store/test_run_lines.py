"""`wpt store run`, which prints the lines of a run's record."""

import re

import pytest
from wpt_store import open_store
from wpt_telemetry import Telemetry
from wpt_utils.lines import ERROR

from wpt.store import main


def test_a_run_s_lines_print_at_the_level_of_its_record(tmp_path, monkeypatch, capsys):
    store = open_store((tmp_path / "store").as_uri())
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    with pytest.raises(ValueError, match="boom"):
        with Telemetry.start(store, "wpt train", [], "failed", env_file=tmp_path / ".env", environ={},
                             echo=lambda *_: None) as telemetry:
            telemetry.log("all is well")
            telemetry.log("it broke", ERROR)
            raise ValueError("boom")

    assert main(["run", "wpt-train", "--follow"]) == 0
    printed = capsys.readouterr().out
    # each one followed and among the last events, and the failure among the errors too
    assert len(re.findall(r"^\S+ \S+ INFO .*all is well$", printed, re.MULTILINE)) == 2
    assert len(re.findall(r"^\S+ \S+ ERROR .*it broke$", printed, re.MULTILINE)) == 3
