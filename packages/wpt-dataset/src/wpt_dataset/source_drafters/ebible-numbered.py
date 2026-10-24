"""The numbered twin of each admitted eBible.org read-aloud source: the same translation's verse-per-line archive.

Its candidates are the repository's sources whose address is an eBible.org
read-aloud archive (<code>_readaloud.zip); one that is not admitted is
skipped. The twin keeps the read-aloud source's license entries and
credit, with the description of the archive in their text replaced by the
verse-per-line archive's (added at its end when the text describes no
archive), and only <code>_vpl.txt: the SQL and XML copies of the same
text, eBible.org's description page, style sheet and signature are left
out. Its one tag is verse-numbers, which no mixture selects, so it serves
the parallel verses measure and never enters training.

Its revision is the revision setting, or the run's date when it is empty.
"""

import copy
import re

from wpt_dataset.repo import CURRENT_SCHEMA, DRAFT_CHECKSUM, source_id
from wpt_dataset.source_drafter import Candidate, Drafted, Skipped

SETTINGS = {"revision": ""}  # the fetch date the drafts give (YYYY-MM-DD); "": the run's date

ADDRESS = re.compile(r"https://eBible\.org/Scriptures/(.+)_readaloud\.zip$")
READALOUD = re.compile(r"The read-aloud archive holds.*?(?:\n\n|\n?$)", re.S)
VPL = ("The verse-per-line archive holds the Bible text in one file, one verse per line, each line starting "
       "with its reference (book, chapter:verse); its SQL and XML copies of the same text, eBible.org's "
       "description page, style sheet and signature are left out of the source.\n")


def check(settings):
    if settings["revision"] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", settings["revision"]):
        raise ValueError(f"revision {settings['revision']!r}: a date, YYYY-MM-DD, or empty for the run's date")


def _twin(rid: str, code: str) -> tuple[str, str]:
    """(item, address) of the numbered twin of the read-aloud source rid."""
    return rid[len("ebible-"):rid.rindex("-")] + "-vpl", f"https://eBible.org/Scriptures/{code}_vpl.zip"


def candidates(context):
    for sid in context.repo.source_ids():
        if not sid.startswith("ebible-"):
            continue
        source = context.repo.source(sid)
        found = ADDRESS.match(source.data["address"])
        if found:
            item, address = _twin(source.id, found[1])
            yield Candidate(source.id, source_id("ebible", item, address), {"code": found[1]})


def draft(candidate, context):
    rid, code = candidate.name, candidate.facts["code"]
    readaloud = context.repo.source(rid)
    if not readaloud.admitted:
        return Skipped(f"{rid} is not admitted")
    item, address = _twin(rid, code)
    licenses = copy.deepcopy(readaloud.data["licenses"])
    for entry in licenses:
        if READALOUD.search(entry["text"]):
            entry["text"] = READALOUD.sub(lambda m: VPL + ("\n" if m[0].endswith("\n\n") else ""), entry["text"],
                                          count=1)
        else:  # the text describes no archive: the numbered one's description goes at its end
            entry["text"] = entry["text"].rstrip("\n") + "\n\n" + VPL
    return Drafted("ebible", item, {
        "schema": CURRENT_SCHEMA["source"], "type": "http-archive", "address": address,
        "revision": context.settings["revision"] or context.today,
        "checksum": DRAFT_CHECKSUM,
        "extract": {"exclude": [f"{code}_about.htm", f"{code}_vpl.sql", f"{code}_vpl.xml", "haiola.css",
                                "signature.txt.asc"],
                    "note": "only the verse-per-line text is kept: the SQL and XML files repeat it, and the rest is "
                            "eBible.org's own pages and signature, for which eBible.org states no license"},
        "licenses": licenses,
        "formats": [{"format": "text", "paths": [f"{code}_vpl.txt"]}],  # any other file fails the format map
        "tags": [{"tags": ["verse-numbers"], "paths": ["**"]}],
        "note": (f"The verse-numbered twin of {rid}, for the parallel verses measure only (#35): "
                 "its tag is verse-numbers alone, which no mixture selects, so it never enters training. "
                 "eBible.org rebuilds this zip often; the checksum covers the files inside."),
    })
