"""The licenses a source may carry today. A license not listed is refused.

docs/licenses.md says why each one is here and what to check before using
it. A license is on the list only if, were the content code, an
AGPL-3.0-or-later program could statically link it (the linking test);
another license that passes the test can be added. CC-BY-SA-4.0 is a grey
zone, allowed until legal review says otherwise; ODbL, GFDL and CC BY
before 4.0 are refused; the data and public-sector licenses wait for the
linking test. A license outside the SPDX list is written LicenseRef-<name>.

A source's license may be an SPDX expression of alternatives, such as
"EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0": WPT uses the
content under one allowed alternative, so the expression passes when at
least one of them is allowed. An alternative may join licenses that all
apply at once with AND ("MIT AND Apache-2.0"): it is allowed when every
one of them is. AND binds before OR, as in SPDX; parentheses are not read.
"""

import hashlib
import re

from .errors import RuleViolation

ALLOWED = frozenset({
    # public domain and permissive: credit and the license text, or nothing
    "CC0-1.0", "Unlicense", "0BSD", "MIT", "MIT-0", "BSD-1-Clause", "BSD-2-Clause", "BSD-2-Clause-Patent",
    "BSD-3-Clause", "ISC", "Apache-2.0", "Zlib", "BSL-1.0", "X11", "NCSA", "PostgreSQL", "UPL-1.0",
    "Python-2.0", "PSF-2.0", "Unicode-3.0", "Unicode-DFS-2016", "Artistic-2.0", "WTFPL", "HPND", "W3C",
    "curl", "BlueOak-1.0.0",
    # GPLv3 family, which AGPL section 13 accepts
    "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
    "AGPL-3.0-only", "AGPL-3.0-or-later",
    # copyleft whose own text lets it move to GPLv3 or the AGPL
    "GPL-2.0-or-later", "LGPL-2.1-only", "LGPL-2.1-or-later", "MPL-2.0", "EUPL-1.2", "CECILL-2.1",
    # attribution only
    "CC-BY-4.0",
    # share-alike, a grey zone: BY-SA 4.0 content may move to the GPLv3, which
    # AGPL section 13 accepts
    "CC-BY-SA-4.0",
    # public domain under the law of the country of first publication; the
    # entry's text states why
    "LicenseRef-public-domain",
})

# exceptions that only add permissions, written after WITH
EXCEPTIONS = frozenset({"Classpath-exception-2.0", "LLVM-exception", "GCC-exception-3.1"})


def _allowed(term: str) -> bool:
    license_id, _, exception = term.partition(" WITH ")
    return license_id in ALLOWED and (not exception or exception in EXCEPTIONS)


def check_license(expression: str, source_id: str):
    if not any(all(_allowed(term) for term in alternative.split(" AND ")) for alternative in expression.split(" OR ")):
        what = "license" if all(word not in expression for word in (" OR ", " AND ", " WITH ")) else "license expression"
        raise RuleViolation(f"{source_id}: {what} {expression} is not on the allowed list")


COPYRIGHT_LINE = re.compile(r"copyright|\(c\)|©|all rights reserved")


def reviewed(texts: dict, content: bytes) -> dict | None:
    """The review of a file's bytes in the reviewed license texts (texts:
    their `texts`, Repo.reviewed_license_texts), or None: the review of that
    exact file first, then the review of its group. A group's key is the
    sha256 of the text without its copyright lines, case and punctuation,
    so the same license with other copyright holders shares one review; a
    removed line may hold a term, so such a file has a review of its own."""
    exact = texts.get("file:" + hashlib.sha256(content).hexdigest())
    if exact is not None:
        return exact
    text = content[:200000].decode("utf-8", "replace").lower()
    text = "\n".join(line for line in text.splitlines() if not COPYRIGHT_LINE.search(line))
    return texts.get(hashlib.sha256(re.sub(r"[^a-z0-9]+", " ", text).strip().encode()).hexdigest())
