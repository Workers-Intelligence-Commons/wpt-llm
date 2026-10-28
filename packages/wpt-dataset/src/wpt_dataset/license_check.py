"""The license check: which license entry covers each file and each record
of a source, and whether the statement at each entry's place matches a
license text the entry expects, with that text's decision. It changes
nothing: it is given a source's entries, its paths (or a record's value),
the bytes at each entry's place and the license texts, and answers.

A license entry says three things (docs/licenses.md):

    place      where its statement must be found: {"file": path} of the
               source's tree (the covered file itself too), {"field": name}
               of each record, {"page": address} outside, downloaded with
               the source, or {"entry": true}, the entry itself, only where
               no statement can exist in the content (the text WPT writes
               itself, in an empty source, and a public-domain proof)
    expected   what the statement must be: at a file or a page, ids of
               license texts (data/license-texts/<id>/); at a field, the
               values of a record's field the entry covers; none for the
               entry itself
    paths      the files it covers

Coverage. Every file is covered by exactly one entry. The records of a
records file whose entries name a field place are covered one by one: each
by the entry that expects its field's value; a record whose value no entry
expects is covered by none: the build leaves it out and reports its value
with its count (records.Skipped.unlisted, the dataset record's
`records.unlisted`), the proposal of an entry for that value, which a
person adds, under review. A record's value never sends an entry back. A history file
(.git/history/...) is covered only by patterns that start with
`.git/history/`.

Statement match. A file's or a page's bytes are looked up among the license
texts (LicenseTexts.find: by the sha256 of its bytes, then by its
normalized text, text_id); they match when the text found is one the entry
expects. A field place matches by its selection alone, and the entry
itself as it stands: its signed text is the statement.

Three levels of decision. A file trains only when its source, its entry
and the text found at its entry's place are all admitted (trains: the
entry and the text; the source's decision is its caller's); a field place
and the entry itself have no text to admit.

Coverage patterns are narrow enough that the person signing sees at a
glance what they cover (check_pattern): a named folder with a star
(`src/*`, `src/**`), a named kind of file in every folder (`**/*.js`) or in
the top folder (`*.js`), or an exact path; a pattern made only of wildcards
(`**` alone) is refused. pattern_for gives the pattern a write adds for a
new file of an empty source.
"""

import dataclasses
import hashlib
import re
from dataclasses import dataclass

from .errors import LicenseMapError, RuleViolation
from .manifest import HISTORY, WILDCARD, PatternIndex

COPYRIGHT_LINE = re.compile(r"copyright|\(c\)|©|all rights reserved")
NAMES_NOTHING = re.compile(r"[*?.]*")  # a pattern segment that names no folder and no kind of file


# -- the license texts --------------------------------------------------------------

def text_id(content: bytes) -> str:
    """The id of a license text (data/license-texts/<id>/): the sha256 of
    its normalized text (its first 200,000 bytes read as UTF-8, lowercased,
    without the lines naming a copyright, then every run of characters other
    than a-z and 0-9 made one space, trimmed), so one item serves every copy
    of the same license with other copyright holders."""
    text = content[:200000].decode("utf-8", "replace").lower()
    text = "\n".join(line for line in text.splitlines() if not COPYRIGHT_LINE.search(line))
    return hashlib.sha256(re.sub(r"[^a-z0-9]+", " ", text).strip().encode()).hexdigest()


class LicenseTexts:
    """The license-text items, {id: its license-text.json} (Repo.license_texts),
    and find(content): the id of the item a file's bytes are, the item that
    lists the sha256 of those exact bytes in its `files` first, then the item
    of their normalized text (text_id); None when no item is either."""

    def __init__(self, items: dict[str, dict]):
        self.items, self._files = {}, {}
        for text, item in items.items():
            self.add(text, item)

    def add(self, text: str, item: dict):
        """Count one more item in (the migration adds those it writes)."""
        self.items[text] = item
        for sha256 in item.get("files", ()):
            self._files.setdefault(sha256, text)

    def find(self, content: bytes) -> str | None:
        exact = self._files.get(hashlib.sha256(content).hexdigest())
        if exact is not None:
            return exact
        found = text_id(content)
        return found if found in self.items else None

    def decision(self, text: str) -> str:
        """The decision of an item: admitted, under-review or refused."""
        return self.items[text]["decision"]["decision"]


# -- the shape of an entry ---------------------------------------------------------------

def check_pattern(pattern: str, where: str):
    """RuleViolation for a coverage pattern that names neither a folder nor
    a kind of file: one whose every segment is made of wildcards (`**`,
    `*`, `**/*`, `*/**`, `*.*`)."""
    if all(NAMES_NOTHING.fullmatch(part) for part in pattern.split("/")):
        raise RuleViolation(f"{where}: coverage pattern {pattern!r} names no folder and no kind of file; name a "
                            f"folder (`src/**`) or a kind of file (`**/*.js`), or list the paths")


def pattern_for(path: str) -> str:
    """The allowed pattern (check_pattern) a write adds to cover a new file
    of an empty source: its top folder, `<folder>/**`, when it lies in one;
    else its kind, `*.<extension>`, when its name has one; else its exact
    path. Wildcard letters of the path are matched as themselves."""
    def exact(part):
        return WILDCARD.sub(lambda letter: f"[{letter[0]}]", part)

    folder, inside, _ = path.partition("/")
    if inside:
        return exact(folder) + "/**"
    stem, dot, extension = path.lstrip(".").rpartition(".")
    return f"*.{exact(extension)}" if dot and stem else exact(path)


def field_of(entry: dict) -> str | None:
    """The record field an entry's place names, or None."""
    return entry.get("place", {}).get("field")


def admitted(entry: dict) -> bool:
    """Whether an entry's license is accepted (its own decision)."""
    return entry["decision"]["decision"] == "admitted"


# -- coverage --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Covered:
    """What covers one path: the number of the one entry that covers the
    whole file (entry), or for a records file selected by a field, that
    field and the entry expecting each value (values); problem says why no
    entry covers it, None when one does."""
    entry: int | None = None
    field: str | None = None
    values: dict[str, int] = dataclasses.field(default_factory=dict)
    problem: str | None = None

    def of_record(self, value) -> int | None:
        """The entry covering a record whose field holds value (any value
        for a file covered whole), or None."""
        return self.entry if self.field is None else self.values.get(value)


class Coverage:
    """The entries of one source compiled once: covered(path, is_records)
    answers for each path in turn, every pattern of every entry matched
    together (manifest.PatternIndex)."""

    def __init__(self, licenses: list[dict]):
        self.licenses = licenses
        self.all = PatternIndex([entry["paths"] for entry in licenses])
        self.history = PatternIndex([[p for p in entry["paths"] if p.startswith(HISTORY)] for entry in licenses])

    def covered(self, path: str, is_records: bool) -> Covered:
        found = (self.history if path.startswith(HISTORY) else self.all).matching(path)
        if not found:
            return Covered(problem=f"{path} matches no license entry; every path must match exactly 1")
        fields = [field_of(self.licenses[number]) for number in found]
        names = ", ".join(self.licenses[number]["license"] for number in found)
        if not any(fields):
            if len(found) != 1:
                return Covered(problem=f"{path} matches {len(found)} license entries ({names}); every path must "
                                       f"match exactly 1")
            return Covered(entry=found[0])
        if not is_records:
            return Covered(problem=f"{path} is not a records file, but a license entry covering it has a field "
                                   f"place")
        if not all(fields) or len(set(fields)) != 1:
            return Covered(problem=f"{path} is a records file: every license entry covering it has the same field "
                                   f"place (found: {names})")
        values, repeated = {}, set()
        for number in found:
            for value in self.licenses[number]["expected"]:
                if value in values:
                    repeated.add(value)
                values.setdefault(value, number)
        if repeated:
            return Covered(problem=f"{path}: record values expected by several entries: {sorted(repeated)}")
        return Covered(field=fields[0], values=values)


def coverage(licenses: list[dict], paths, is_records) -> dict[str, Covered]:
    """{path: Covered} for each path; is_records(path) says whether it is a
    records file."""
    compiled = Coverage(licenses)
    return {path: compiled.covered(path, is_records(path)) for path in paths}


def check_coverage(source_id: str, covered: dict[str, Covered]):
    """LicenseMapError naming the first path no entry covers (in the order
    given), for the callers that refuse such a source whole."""
    for found in covered.values():
        if found.problem is not None:
            raise LicenseMapError(f"{source_id}: {found.problem}")


# -- the statements ------------------------------------------------------------------------

@dataclass(frozen=True)
class Statement:
    """The license check of one entry: its place, the id of the license text
    found there (None when the place holds no known text, for a field and
    for the entry itself), whether it matches one the entry expects, that
    text's decision (None where no text was found), and why the entry's
    files do not train at this level (problem)."""
    entry: int
    place: dict
    found: str | None
    matched: bool
    decision: str | None = None
    problem: str | None = None

    @property
    def admitted(self) -> bool:
        """Whether the statement lets the entry's files train: it matches,
        and at a file or a page, the text found there is admitted."""
        return self.matched and (self.found is None or self.decision == "admitted")


def statement(number: int, entry: dict, read, texts: LicenseTexts) -> Statement:
    """The statement check of one entry: read(place) gives the bytes at a
    file or page place (None when the source's tree or its stored pages do
    not hold it); texts are the license texts."""
    place = entry["place"]
    if "field" in place or "entry" in place:
        return Statement(number, place, None, True)
    content = read(place)
    if content is None:
        where = f"file {place['file']}" if "file" in place else f"page {place['page']}"
        return Statement(number, place, None, False, problem=f"the {where} is not there")
    found = texts.find(content)
    if found is None:
        return Statement(number, place, None, False,
                         problem=f"no license text holds the text there (its id would be {text_id(content)}; "
                                 f"`wpt-dataset add-license-text` adds it, under review)")
    decision = texts.decision(found)
    if found not in entry["expected"]:
        return Statement(number, place, found, False, decision,
                         f"the license text there ({found}) is not one it expects")
    if decision != "admitted":
        return Statement(number, place, found, True, decision, f"the license text there ({found}) is {decision}")
    return Statement(number, place, found, True, decision)


def statements(licenses: list[dict], read, texts: LicenseTexts) -> list[Statement]:
    """statement() of every entry, in order."""
    return [statement(number, entry, read, texts) for number, entry in enumerate(licenses)]


def trains(entry: dict, found: Statement) -> bool:
    """Whether an entry's files train as far as the entry and its text
    decide (their source must be admitted too): the entry is admitted and
    its statement (found) is admitted."""
    return admitted(entry) and found.admitted
