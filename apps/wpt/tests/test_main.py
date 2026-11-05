import sys
from types import SimpleNamespace

import pytest

import wpt
from wpt_utils.lines import bare


def test_a_command_stopped_by_a_text_prints_it_as_an_error_line_and_ends_with_code_1(monkeypatch, capsys):
    # sys.exit(text) would print the text as it is, the one way out of a command with no time and level
    def refuses(argv):
        sys.exit("KernelUnavailable: no such kernels")

    monkeypatch.setitem(wpt.COMMANDS, "refusing", "a_refusing_command")
    monkeypatch.setitem(sys.modules, "a_refusing_command", SimpleNamespace(main=refuses))
    assert wpt.main(["refusing"]) == 1
    printed = capsys.readouterr().err
    assert " ERROR KernelUnavailable" in printed and bare(printed) == "KernelUnavailable: no such kernels\n"


def test_a_command_stopped_by_a_number_keeps_it(monkeypatch):
    def stops(argv):
        sys.exit(2)

    monkeypatch.setitem(wpt.COMMANDS, "stopping", "a_stopping_command")
    monkeypatch.setitem(sys.modules, "a_stopping_command", SimpleNamespace(main=stops))
    with pytest.raises(SystemExit) as stopped:
        wpt.main(["stopping"])
    assert stopped.value.code == 2
