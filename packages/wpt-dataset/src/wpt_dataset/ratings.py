"""People's ratings of measure attempts (docs/evaluation.md, "People's ratings").

A file in the `rating` format is JSON Lines: one rating per line, each
matching schemas/rating.schema.json. A rating names the attempt it rates
as a wpt-eval run record's attempts.jsonl holds it (the task id, the
record folder, the attempt's index and whether it was plain), the model's
weights as the record's ids.json names them, and the person's verdict,
pass or fail, who rated and when. Ratings are stored as files of an
`empty` source so the measure is reproducible, and they are a measure
only, never a training signal: such a
file is never a pre-training document (markdown.py gives it no Markdown)
and no training step reads it.

A person rates an attempt once: their newer rating replaces their
earlier one. parse() refuses a file
holding two ratings by one person of one attempt (attempt_of), so every
way of saving keeps the rule (the check runs it on every patch), and
Repo.save_rating replaces the earlier line when it adds a newer one.
"""

import json
from datetime import date

from .errors import InvalidFile
from .repo import validate


def attempt_of(rating: dict) -> tuple:
    """What tells one rated attempt and its rater from another: the task,
    the run record, the attempt's index, whether it was plain, and who rated."""
    return rating["task"], rating["record"], rating["attempt"], rating["plain"], rating["by"]


def parse(source_id: str, path: str, data: bytes) -> list[dict]:
    """The ratings of one rating file; InvalidFile names the line at fault."""
    ratings, seen = [], {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        where = f"{source_id}/{path} line {number}"
        if not line.strip():
            raise InvalidFile(f"{where}: empty line; a rating file holds one rating per line")
        try:
            rating = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidFile(f"{where}: {error}") from error
        validate(rating, "rating", where)
        try:
            date.fromisoformat(rating["date"])
        except ValueError:
            raise InvalidFile(f"{where}: at date: {rating['date']!r} is not a day of the calendar") from None
        earlier = seen.setdefault(attempt_of(rating), number)
        if earlier != number:
            raise InvalidFile(f"{where}: {rating['by']} rated this attempt on line {earlier} already; a person's "
                              f"newer rating of an attempt replaces their earlier one")
        ratings.append(rating)
    return ratings
