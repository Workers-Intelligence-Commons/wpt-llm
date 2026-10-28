"""What a source keeps for training: its final tree (sources.final_tree:
its cuts and patches applied) less the files the build leaves out, each
alone, and the report saying which and why. The build's own decision: it
asks the shared checks (checks.py) and the license check
(license_check.py), and decides.

    uncovered    a file no license entry covers, or several do
                 (license_check.Coverage); a person adds the entry
    not-text     a file the text check finds is not text: a file that is
                 not text is listed in the source by its exact path, as an
                 `extract` exclusion, so the store never holds it; a person
                 adds the path
    entry        every file of a license entry that does not let its files
                 train: the entry is not admitted, or the statement at its
                 place, in the stored bytes (a file of the final tree, or a
                 license page downloaded with the source), does not match a
                 license text it expects, or matches one that is not
                 admitted (license_check.statements, license_check.trains).
                 A records file whose entries select its records by a field
                 stays: each record of an entry that does not train is left
                 out when it is read (markdown.documents, `trains`)

Nothing here asks upstream or changes the source. The result is kept in
the store under the ref source-kept/<fingerprint of the final tree, the
license pages, the license entries and the code (wpt_utils.programs.code:
the WPT repository's whole commit, data/ included, so the decisions of
the license texts too)>, so a source is checked once per commit.
"""

import dataclasses
import json
from dataclasses import dataclass

from wpt_telemetry import Telemetry, set_ref
from wpt_utils.programs import code

from . import license_check, manifest
from .manifest import Entry
from .repo import Source
from .sources import (UPSTREAM_PAGES, final_tree, fingerprint, is_records, license_page, license_pages, not_text,
                      upstream_key)

UNCOVERED, NOT_TEXT, ENTRY = "uncovered", "not-text", "entry"  # why something is left out


@dataclass(frozen=True)
class Kept:
    """A source's kept tree and its check result."""
    tree: dict[str, Entry]  # the final tree's files that are kept, symbolic links included
    final: str  # the sha256 of the final tree's manifest object
    trains: frozenset[int]  # the numbers of the license entries whose files and records train
    statements: list[dict]  # the license check of each entry, in order (license_check.Statement as a dict)
    left_out: list[dict]  # [{"why": UNCOVERED or NOT_TEXT, "path", "detail"}, {"why": ENTRY, "entry", "license",
    #                        "files", "detail"}], in the tree's order, then the entries'

    def result(self) -> dict:
        """What a dataset's record keeps of the check: the statements and
        what was left out."""
        return {"statements": self.statements, "left_out": self.left_out}


def kept(source: Source, store, texts: license_check.LicenseTexts, telemetry=None) -> Kept:
    """The source's Kept (see the module's docstring). What final_tree
    raises (NotDownloaded, CutFailed, PatchFailed) takes the whole source
    out; the caller decides what that means for its own work."""
    telemetry = telemetry or Telemetry.off()
    tree, known = final_tree(source, store, telemetry)
    licenses = source.data["licenses"]
    key = fingerprint({"kind": "source-kept", "code": code(), "tree": known, "licenses": licenses,
                       **({"pages": store.get_ref(UPSTREAM_PAGES, upstream_key(source))}
                          if license_pages(source) else {})})
    held = store.get_ref("source-kept", key)
    if held:
        found = json.loads(store.get_bytes(held))
    else:
        found = _checked(source, tree, known, store, texts, telemetry)
        set_ref(store, "source-kept", key, store.put_bytes(json.dumps(found, ensure_ascii=False).encode()),
                telemetry)
    out = set(found["out"])
    return Kept({path: entry for path, entry in tree.items() if path not in out}, known, frozenset(found["trains"]),
                found["statements"], found["left_out"])


def statements(source: Source, tree: dict[str, Entry], store,
               texts: license_check.LicenseTexts) -> list[license_check.Statement]:
    """The license check of each of the source's entries on its stored
    bytes (license_check.statements): a file place read from its final
    tree, a page place from the license pages downloaded with it."""
    def read(place):
        if "file" in place:
            entry = tree.get(place["file"])
            return manifest.content(store, entry) if entry is not None else None
        return license_page(source, store, place["page"])

    return license_check.statements(source.data["licenses"], read, texts)


def _checked(source, tree, known, store, texts, telemetry) -> dict:
    """{"out": the paths left out, "trains", "statements", "left_out"} of
    one final tree (known: its manifest object)."""
    licenses = source.data["licenses"]
    found = statements(source, tree, store, texts)
    trains = {number for number, entry in enumerate(licenses) if license_check.trains(entry, found[number])}
    coverage = license_check.Coverage(licenses)
    unreadable = not_text(source, tree, known, store, telemetry)
    out, left_out, by_entry = [], [], {}
    for path, entry in tree.items():
        covered = coverage.covered(path, is_records(source, path))
        if covered.problem is not None:
            out.append(path)
            left_out.append({"why": UNCOVERED, "path": path, "detail": covered.problem})
        elif path in unreadable:
            out.append(path)
            left_out.append({"why": NOT_TEXT, "path": path, "detail": unreadable[path]})
        elif covered.field is None and covered.entry not in trains:
            out.append(path)
            by_entry[covered.entry] = by_entry.get(covered.entry, 0) + 1
    for number, entry in enumerate(licenses):
        if number not in trains:
            left_out.append({"why": ENTRY, "entry": number, "license": entry["license"],
                             "files": by_entry.get(number, 0), "detail": _why(entry, found[number])})
    return {"out": out, "trains": sorted(trains), "statements": [dataclasses.asdict(s) for s in found],
            "left_out": left_out}


def _why(entry: dict, statement: license_check.Statement) -> str:
    """Why an entry's files do not train."""
    if not license_check.admitted(entry):
        return f"the entry is {entry['decision']['decision']}"
    return statement.problem or "its statement does not match"
