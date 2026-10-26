"""One http-archive source per Wikipedia of a run of the Wikimedia Enterprise HTML dumps.

Its candidates are the run's article dumps (namespace 0) whose database
name is a Wikipedia: the run's listing at dumps.wikimedia.org crossed with
noc.wikimedia.org's wikipedia.dblist. Each source reads the dump's ndjson
files in the records format, its records' HTML converted to Markdown, under
CC-BY-SA-4.0 for the records whose license field is one of VALUES (the
license text says why the CC BY-SA 3.0 value moves to 4.0, then gives the
official CC BY-SA 4.0 legal code). Its tags are the language its records
state and "encyclopedia".

The language and the wiki's address come from the statistics of the dump's
records (statistics(): the whole dump when it is at most SAMPLE_WHOLE
bytes, its first SAMPLE_PART bytes otherwise; no article text is kept):
the language most records state (in_language), and the address the
records give the wiki (is_part_of); when they state none, the ones its
database name gives. The listing, the database list, each dump's
statistics and the legal code are site facts, kept in the store.

A run's address differs from an older run's, so each run's source of a
wiki is a source of its own. A draft supersedes the same wiki's sources
from older runs (Drafted.supersedes): the framework puts the admitted ones
back under review. Refusing them stays a person's decision.
"""

import collections
import json
import re
import zlib
from datetime import date

from wpt_dataset.repo import DRAFT_CHECKSUM, source_id
from wpt_dataset.source_drafter import Candidate, Drafted

SETTINGS = {"run": "20250320"}  # the dump run, YYYYMMDD

DUMPS = "https://dumps.wikimedia.org/other/enterprise_html/runs"
DBLIST = "https://noc.wikimedia.org/conf/dblists/wikipedia.dblist"
LEGAL_CODE = "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt"  # the official CC BY-SA 4.0 legal code
DOMAIN = "encyclopedia"
SAMPLE_WHOLE = 20_000_000  # a dump this large or smaller is read whole for its statistics
SAMPLE_PART = 4 * 1024 * 1024  # of a larger dump, the statistics read this many bytes from its start
# the exact JSON text WPT reads for each license value the records carry (records._text)
VALUES = [
    json.dumps([{"name": "Creative Commons Attribution Share Alike 3.0 Unported", "identifier": "CC-BY-SA-3.0",
                 "url": "https://creativecommons.org/licenses/by-sa/3.0/"}], ensure_ascii=False),
    json.dumps([{"name": "Creative Commons Attribution-ShareAlike License 4.0", "identifier": "CC-BY-SA-4.0",
                 "url": "https://creativecommons.org/licenses/by-sa/4.0/"}], ensure_ascii=False),
]

STATEMENT = """Records whose license field reads exactly one of these values (the field is a list; WPT reads it as that JSON text):
{values}

The second value names CC BY-SA 4.0 itself. The first names CC BY-SA 3.0 Unported, and the owners of the text state that it is under CC BY-SA 4.0:

Wikimedia's license page for its dumps, https://dumps.wikimedia.org/legal.html, says: "Text: Except as discussed below, all original textual content is licensed under the GNU Free Documentation License (GFDL) and the Creative Commons Attribution-Share-Alike 4.0 License. Some text may be available only under the Creative Commons license; see our Terms of Use for details." and "Where this information conflicts with specific information in the Wikimedia Foundation Terms of Use, or with other information contained inside the dumps themselves, this description should be ignored. Those terms are controlling."

The Wikimedia Foundation Terms of Use, which every contributor accepts when publishing, section 7, "Licensing of Content", https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use, says: "Text to which you hold the copyright: When you submit text to which you hold the copyright, you agree to license it under: Creative Commons Attribution-ShareAlike 4.0 International License ("CC BY-SA 4.0"), and GNU Free Documentation License ("GFDL") (unversioned, with no invariant sections, front-cover texts, or back-cover texts). Reusers may comply with either license or both." and "Modifications or additions to material that you reuse: When modifying or making additions to text that you have obtained from a Project Website, you agree to license the modified or added content under CC BY-SA 4.0 or later". On attribution, it says that contributors agree to be attributed "Through hyperlink (where possible) or URL to the article to which you contributed (since each article has a history page that lists all contributors, authors and editors)".

Text contributed under CC BY-SA 3.0 Unported moves to the later version through that license's section 4(b), https://creativecommons.org/licenses/by-sa/3.0/legalcode: "You may Distribute or Publicly Perform an Adaptation only under: (i) the terms of this License; (ii) a later version of this License with the same License Elements as this License; (iii) a Creative Commons jurisdiction license (either this or a later license version) that contains the same License Elements as this License (e.g., Attribution-ShareAlike 3.0 Unported) ("Applicable License")."

"""

ATTRIBUTION = ("Wikipedia contributors, {host} ({db}), {url}. Each article's authors are listed in its history, "
               "reached from the article's address, which WPT's per-document credit gives with its title and the "
               "editor of its revision. From the Wikimedia Enterprise HTML dump of {day}, {address}. "
               "Changed: each article's HTML converted to Markdown.")


def check(settings):
    try:
        _run_date(settings["run"])
    except ValueError:
        raise ValueError(f"run {settings['run']!r}: the dump run's date, YYYYMMDD") from None


def _run_date(run: str) -> date:
    if not re.fullmatch(r"\d{8}", run):
        raise ValueError(run)
    return date(int(run[:4]), int(run[4:6]), int(run[6:]))


def _address(run: str, db: str) -> str:
    return f"{DUMPS}/{run}/{db}-NS0-{run}-ENTERPRISE-HTML.json.tar.gz"


def _older(context, run: str) -> dict[str, list[str]]:
    """{database name: ids of its existing sources from runs before run}."""
    pattern = re.compile(rf"{re.escape(DUMPS)}/(\d{{8}})/([a-z0-9_]+)-NS0-\1-ENTERPRISE-HTML\.json\.tar\.gz")
    found = {}
    for sid in context.repo.source_ids():
        if not sid.startswith("wikimedia-wikipedia-"):
            continue
        match = pattern.fullmatch(context.repo.source(sid).data.get("address", ""))
        if match and match[1] < run:
            found.setdefault(match[2], []).append(sid)
    return found


def candidates(context):
    run = context.settings["run"]

    def listing():  # {database name: the archive's size} of the run's article dumps
        page = context.fetch(f"{DUMPS}/{run}/").decode("utf-8")
        pattern = rf'<a href="([a-z0-9_]+)-NS0-{run}-ENTERPRISE-HTML\.json\.tar\.gz">[^<]*</a>\s+\S+\s+\S+\s+(\d+)'
        return {found[1]: int(found[2]) for found in re.finditer(pattern, page)}

    def wikipedias():
        text = context.fetch(DBLIST).decode("utf-8")
        return sorted(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))

    sizes = context.fact(f"listing/{run}", listing)
    names = set(context.fact(f"wikipedia.dblist/{run}", wikipedias))
    older = _older(context, run)
    for db in sorted(set(sizes) & names):
        yield Candidate(db, source_id("wikimedia", _item(db), _address(run, db)),
                        {"size": sizes[db], "older": older.get(db, [])})


def _item(db: str) -> str:
    return f"wikipedia-{db.replace('_', '-')}"


def draft(candidate, context):
    run, db = context.settings["run"], candidate.name
    address = _address(run, db)
    sample = context.fact(f"statistics/{run}/{db}", lambda: statistics(context, db, address, candidate.facts["size"]))
    legal_text = context.fact("legal/by-sa-4.0", lambda: context.fetch(LEGAL_CODE).decode("utf-8"))
    url = site(db, sample)
    day = _run_date(run)
    return Drafted("wikimedia", _item(db), {
        "type": "http-archive",
        "address": address,
        "revision": day.isoformat(),
        "checksum": DRAFT_CHECKSUM,
        "licenses": [{"license": "CC-BY-SA-4.0", "paths": ["*.ndjson"], "values": VALUES,
                      "attribution": ATTRIBUTION.format(host=url.removeprefix("https://"), db=db, url=url,
                                                        day=f"{day.day} {day:%B} {day.year}", address=address),
                      "text": STATEMENT.format(values="\n".join(VALUES)) + legal_text}],
        "formats": [{"format": "records", "paths": ["*.ndjson"], "text": "article_body.html", "as": "html",
                     "id": "identifier", "newest": "date_modified", "title": "name", "license": "license",
                     "credit": ["name", "url", "version.editor.name"]}],
        "tags": [{"tags": [language(db, sample), DOMAIN], "paths": ["**"]}],
    }, supersedes=tuple(candidate.facts.get("older", [])))


def language(db: str, sample: dict) -> str:
    """The language the records state (in_language), else the one in the db name."""
    found = collections.Counter({json.loads(k): v for k, v in sample["languages"].items()})
    found.pop(None, None)
    if found:
        return found.most_common(1)[0][0]
    return db.removesuffix("wiki").replace("_", "-")


def site(db: str, sample: dict) -> str:
    """The wiki's address as its records state it (is_part_of.url), else the one its db name gives."""
    parts = [json.loads(k) for k in sample["is_part_of"]]
    urls = {p.get("url") for p in parts if isinstance(p, dict) and p.get("identifier") == db}
    if len(urls) == 1 and None not in urls:
        return urls.pop()
    return "https://" + db.removesuffix("wiki").replace("_", "-") + ".wikipedia.org"


def statistics(context, db: str, address: str, size: int) -> dict:
    """The structure of a dump's records: license values (their exact JSON
    text), languages, wiki addresses, record and id counts; no article text.
    A dump larger than SAMPLE_WHOLE bytes is read from its first SAMPLE_PART
    bytes only, a stream cut short whose last record is counted broken."""
    whole = size <= SAMPLE_WHOLE
    data = context.fetch(address, None if whole else {"Range": f"bytes=0-{SAMPLE_PART - 1}"})
    licenses, languages, parts = collections.Counter(), collections.Counter(), collections.Counter()
    ids, versions, events = collections.Counter(), collections.Counter(), collections.Counter()
    records = broken = no_html = 0
    for raw in _lines(_gunzipped(data)):
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            broken += 1  # the last line of a cut stream
            continue
        if not isinstance(row, dict):
            broken += 1
            continue
        records += 1
        value = row.get("license")
        licenses[value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)] += 1
        languages[json.dumps((row.get("in_language") or {}).get("identifier"))] += 1
        parts[json.dumps(row.get("is_part_of"), ensure_ascii=False)] += 1
        ids[row.get("identifier")] += 1
        versions[(row.get("identifier"), (row.get("version") or {}).get("identifier"))] += 1
        events[(row.get("event") or {}).get("identifier")] += 1
        html = (row.get("article_body") or {}).get("html")
        no_html += not (isinstance(html, str) and html.strip())
    return {"db": db, "complete": whole, "records": records, "broken_lines": broken, "no_html": no_html,
            "licenses": dict(licenses), "languages": dict(languages), "is_part_of": dict(parts),
            "repeated_ids": sum(1 for c in ids.values() if c > 1), "distinct_ids": len(ids),
            "repeated_versions": sum(1 for c in versions.values() if c > 1),
            "repeated_events": sum(1 for c in events.values() if c > 1)}


def _gunzipped(data: bytes):
    """The bytes of a gzip stream, in parts, as far as they decode (a part
    of a dump ends anywhere)."""
    while data:
        reader = zlib.decompressobj(31)
        while not reader.eof:
            try:
                yield reader.decompress(data, 1 << 20)
                data = reader.unconsumed_tail
                if not data and not reader.eof:  # the stream is cut short: what it holds so far
                    yield reader.flush()
                    return
            except zlib.error:
                return
        data = reader.unused_data


class _Stream:
    """Bytes read from an iterator of parts, so many at a time."""

    def __init__(self, parts):
        self._parts, self._buffer = iter(parts), b""

    def take(self, count: int) -> bytes:
        while len(self._buffer) < count:
            more = next(self._parts, None)
            if more is None:
                break
            self._buffer += more
        found, self._buffer = self._buffer[:count], self._buffer[count:]
        return found


def _lines(parts):
    """The lines of the regular files of a tar stream, as far as it goes."""
    stream = _Stream(parts)
    while True:
        header = stream.take(512)
        if len(header) < 512 or header == bytes(512):
            return
        try:
            size = int(header[124:136].rstrip(b"\0 ").decode() or "0", 8)
        except ValueError:
            return
        regular, left, tail = header[156:157] in (b"0", b"\0"), size, b""
        while left:
            part = stream.take(min(left, 1 << 20))
            if not part:
                break
            left -= len(part)
            if regular:
                *lines, tail = (tail + part).split(b"\n")
                yield from lines
        if tail:
            yield tail
        if left:
            return
        stream.take(-size % 512)
