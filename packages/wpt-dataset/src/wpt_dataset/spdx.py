"""The standard license texts of one fixed release of the SPDX license list
(RELEASE, of spdx/license-list-data), and Matcher, which says which of them
a license file's text is: what the one-off migration of sources reads to
recognize a license file no license text holds yet (migrate.py).

Matcher compares a file's words (copyright lines left out) with each
standard text of MATCHED and names the one it matches almost word for word
(a ratio of at least 0.93), clearly better than any other. Texts that are
the same for several ids (GPL-3.0-only and GPL-3.0-or-later) give their
family, as the OR of those ids (license_of). MATCHED holds licenses that
are refused too, so a refused text is not taken for an allowed one close
to it.

texts(user_agent) fetches the standard texts: one request per id, to
raw.githubusercontent.com. Failures: what urllib raises when it cannot be
reached (the caller turns it into its own failure).
"""

import difflib
import re
import urllib.request

from . import sites

RELEASE = "v3.29.0"
ADDRESS = "https://raw.githubusercontent.com/spdx/license-list-data/{release}/text/{id}.txt"
TIMEOUT = 60
# the standard texts a license file is matched with
MATCHED = """0BSD AGPL-3.0-only AGPL-3.0-or-later Apache-2.0 Artistic-2.0 BlueOak-1.0.0 BSD-1-Clause BSD-2-Clause
BSD-2-Clause-Patent BSD-3-Clause BSL-1.0 CC0-1.0 CC-BY-1.0 CC-BY-2.0 CC-BY-2.5 CC-BY-2.5-AU CC-BY-3.0 CC-BY-3.0-AT
CC-BY-3.0-AU CC-BY-3.0-DE CC-BY-3.0-IGO CC-BY-3.0-NL CC-BY-3.0-US CC-BY-4.0 CC-BY-NC-1.0 CC-BY-NC-2.0 CC-BY-NC-2.5
CC-BY-NC-3.0 CC-BY-NC-3.0-DE CC-BY-NC-3.0-IGO CC-BY-NC-4.0 CC-BY-NC-ND-1.0 CC-BY-NC-ND-2.0 CC-BY-NC-ND-2.5
CC-BY-NC-ND-3.0 CC-BY-NC-ND-3.0-DE CC-BY-NC-ND-3.0-IGO CC-BY-NC-ND-4.0 CC-BY-NC-SA-1.0 CC-BY-NC-SA-2.0
CC-BY-NC-SA-2.0-DE CC-BY-NC-SA-2.0-FR CC-BY-NC-SA-2.0-UK CC-BY-NC-SA-2.5 CC-BY-NC-SA-3.0 CC-BY-NC-SA-3.0-DE
CC-BY-NC-SA-3.0-IGO CC-BY-NC-SA-4.0 CC-BY-ND-1.0 CC-BY-ND-2.0 CC-BY-ND-2.5 CC-BY-ND-3.0 CC-BY-ND-3.0-DE CC-BY-ND-4.0
CC-BY-SA-1.0 CC-BY-SA-2.0 CC-BY-SA-2.0-UK CC-BY-SA-2.1-JP CC-BY-SA-2.5 CC-BY-SA-3.0 CC-BY-SA-3.0-AT CC-BY-SA-3.0-DE
CC-BY-SA-3.0-IGO CC-BY-SA-4.0 CDLA-Permissive-2.0 CDLA-Sharing-1.0 CECILL-2.1 curl DL-DE-BY-2.0 DL-DE-ZERO-2.0
etalab-2.0 EUPL-1.2 GPL-2.0-or-later GPL-3.0-only GPL-3.0-or-later HPND ISC LGPL-2.1-only LGPL-2.1-or-later
LGPL-3.0-only LGPL-3.0-or-later MIT MIT-0 MPL-2.0 NCSA NLOD-2.0 ODC-By-1.0 OGL-Canada-2.0 OGL-UK-3.0 PDDL-1.0
PostgreSQL PSF-2.0 Python-2.0 Unicode-3.0 Unicode-DFS-2016 Unlicense UPL-1.0 W3C WTFPL X11 Zlib""".split()
SUFFIX = re.compile(r"-(only|or-later)$")


def texts(user_agent: str, release: str = RELEASE) -> dict[str, str]:
    """{id: its standard text} of every id of MATCHED, in that release."""
    found = {}
    for spdx in MATCHED:
        address = ADDRESS.format(release=release, id=spdx)
        sites.paced(address)
        request = urllib.request.Request(address, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            found[spdx] = answer.read().decode("utf-8")
    return found


def normalized(text: str) -> list[str]:
    """The words of a text, lowercased, its copyright lines left out."""
    lines = [l for l in text.splitlines() if not re.match(r"\s*(copyright|\(c\)|©|all rights reserved)", l, re.I)]
    return re.sub(r"[^a-z0-9 ]+", " ", " ".join(lines).lower()).split()


def cc_title(words: list[str]) -> str | None:
    """The Creative Commons license a text names in its title (its first 30
    words), as an SPDX id."""
    found = re.search(r"attribution( noncommercial)?( sharealike| noderivatives| noderivs)? (\d) (\d)",
                      " ".join(words[:30]))
    if not found:
        return None
    kind = {" sharealike": "-SA", " noderivatives": "-ND", " noderivs": "-ND"}.get(found[2] or "", "")
    return f"CC-BY{'-NC' if found[1] else ''}{kind}-{found[3]}.{found[4]}"


def license_of(identified: str) -> str:
    """The SPDX expression of what Matcher.identify names: the id itself, or
    for a family ("GPL-3.0?") the OR of its -only and -or-later ids."""
    if not identified.endswith("?"):
        return identified
    family = identified[:-1]
    return f"{family}-only OR {family}-or-later"


class Matcher:
    """Identifies a license file's text among the standard texts given
    ({id: text}): one license whose text it matches almost word for word,
    clearly better than any other."""

    BEST = 0.93  # the least ratio of a match
    CLOSE = 0.02  # how much better than any other the match must be

    def __init__(self, texts: dict[str, str]):
        self.texts = {spdx: normalized(text) for spdx, text in sorted(texts.items())}

    def identify(self, text: str) -> str | None:
        """The id the text is the standard text of, its family ("GPL-3.0?")
        when the text is the same for several of its ids, or None."""
        words = normalized(text)
        if len(words) < 20:
            return None
        scores = []
        for spdx, ref in self.texts.items():
            matcher = difflib.SequenceMatcher(None, words, ref, autojunk=False)
            # the quick bounds are never below the ratio: a text they put out of reach scores below every match
            bound = matcher.real_quick_ratio() >= self.BEST - self.CLOSE and matcher.quick_ratio() >= self.BEST - \
                self.CLOSE
            scores.append((matcher.ratio() if bound else 0.0, spdx))
        scores.sort(reverse=True)
        best, spdx = scores[0]
        # texts identical between ids (GPL-3.0-only and -or-later) give the family
        same = [s for r, s in scores if r == best]
        if best < self.BEST:
            return None
        if len(same) > 1:
            family = {SUFFIX.sub("", s) for s in same}
            return family.pop() + "?" if len(family) == 1 else None
        close = {s for r, s in scores if best - r < self.CLOSE}
        if len(close) > 1 and all(s.startswith("CC-BY") for s in close):  # CC texts differ mostly in their title
            named = cc_title(words)
            return named if named in close else None
        if len(scores) > 1 and best - scores[1][0] < self.CLOSE and \
                SUFFIX.sub("", scores[1][1]) != SUFFIX.sub("", spdx):
            return None
        if spdx.endswith(("-only", "-or-later")):
            return SUFFIX.sub("", spdx) + "?"
        return spdx
