import re
import sys
from datetime import UTC, datetime

from wpt_utils.lines import ERROR, INFO, WARNING, bare, line, say

START = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d"


def test_a_line_starts_with_its_utc_time_to_the_second_and_its_level():
    before = datetime.now(UTC).replace(microsecond=0)
    written = line("read, round 2: 2,273 of 3,636 parts", WARNING)
    after = datetime.now(UTC)
    stamp, level_and_text = written[:19], written[20:]
    assert level_and_text == "WARNING read, round 2: 2,273 of 3,636 parts"
    assert before <= datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC) <= after


def test_each_line_of_a_text_gets_the_start_a_blank_one_too():
    assert re.fullmatch(rf"{START} ERROR first\n{START} ERROR\n{START} ERROR third", line("first\n\nthird", ERROR))


def test_a_level_chooses_the_stream_and_a_line_can_name_its_own(capsys):
    say("plain")
    say("watch it", WARNING)
    say("broken", ERROR)
    say("a side note", INFO, sys.stderr)
    printed = capsys.readouterr()
    assert [bare(text) for text in printed.out.splitlines()] == ["plain"]
    assert [bare(text) for text in printed.err.splitlines()] == ["watch it", "broken", "a side note"]


def test_a_reader_gets_the_text_back_whatever_it_is():
    for text in ("abc123", "2026-11-05 15:27:29 is a time", "", "INFO", "spaces  stay  "):
        assert bare(line(text)) == text
        assert bare(line(text, ERROR)) == text
    assert bare("a line of another program\n") == "a line of another program\n"


def test_every_line_of_what_a_command_printed_loses_its_start():
    assert bare(line("first\nsecond") + "\n" + line("third", ERROR) + "\n") == "first\nsecond\nthird\n"
