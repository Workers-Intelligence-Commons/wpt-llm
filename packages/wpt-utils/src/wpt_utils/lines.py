"""The form of every line WPT prints, the one place that writes it:

    2026-11-05 15:27:29 INFO read, round 2: 2,273 of 3,636 parts

its UTC time to the second, its level (INFO, WARNING or ERROR, the levels a
run's record and OpenTelemetry carry too), then its text. A text of several
lines gets this start on each of them. A program that reads what another
WPT command printed takes the text back with bare().
"""

import re
import sys
from datetime import UTC, datetime

INFO, WARNING, ERROR = "INFO", "WARNING", "ERROR"
START = re.compile(rf"^\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d (?:{INFO}|{WARNING}|{ERROR})(?: |$)", re.MULTILINE)


def line(text, level=INFO) -> str:
    """text as it prints: each of its lines with this moment and level."""
    start = f"{datetime.now(UTC):%Y-%m-%d %H:%M:%S} {level}"
    return "\n".join(f"{start} {part}" if part else start for part in str(text).split("\n"))


def say(text, level=INFO, file=None):
    """Print text as line() gives it, at once and in one write, so that the
    lines of two threads do not mix: to file, by default standard output
    for INFO and standard error for the other levels."""
    stream = file or (sys.stdout if level == INFO else sys.stderr)
    stream.write(line(text, level) + "\n")
    stream.flush()


def bare(printed: str) -> str:
    """printed with the start of each of its lines taken off; a line that
    does not start as line() writes it stays as it is."""
    return START.sub("", printed)
