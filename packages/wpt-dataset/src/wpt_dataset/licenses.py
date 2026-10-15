"""The closed list of licenses a source may carry. When in doubt, refuse.

docs/licenses.md says why each one is here and what to check before using
it. CC-BY-SA, ODbL and GFDL stay out until a lawyer confirms them. A
license outside the SPDX list is written LicenseRef-<name>.

A source's license may be an SPDX expression of alternatives, such as
"EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0": WPT uses the
content under one allowed alternative, so the expression passes when at
least one of them is allowed.
"""

from pathlib import Path

from .errors import RuleViolation

ALLOWED = frozenset({
    # public domain and permissive: credit and the license text, or nothing
    "CC0-1.0", "Unlicense", "0BSD", "MIT", "MIT-0", "BSD-1-Clause", "BSD-2-Clause", "BSD-2-Clause-Patent",
    "BSD-3-Clause", "ISC", "Apache-2.0", "Zlib", "BSL-1.0", "X11", "NCSA", "PostgreSQL", "UPL-1.0",
    "Python-2.0", "PSF-2.0", "Unicode-3.0", "Unicode-DFS-2016", "Artistic-2.0", "WTFPL", "HPND", "W3C",
    "curl", "BlueOak-1.0.0",
    # data: no duty on what is learned from it (Results)
    "PDDL-1.0", "CDLA-Permissive-2.0", "CDLA-Sharing-1.0",
    # GPLv3 family, which AGPL section 13 accepts
    "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
    "AGPL-3.0-only", "AGPL-3.0-or-later",
    # copyleft whose own text lets it move to GPLv3 or the AGPL
    "GPL-2.0-or-later", "LGPL-2.1-only", "LGPL-2.1-or-later", "MPL-2.0", "EUPL-1.2", "CECILL-2.1",
    # attribution only: Creative Commons and public-sector licenses
    "CC-BY-1.0", "CC-BY-2.0", "CC-BY-2.5", "CC-BY-2.5-AU", "CC-BY-3.0", "CC-BY-3.0-AT", "CC-BY-3.0-AU",
    "CC-BY-3.0-DE", "CC-BY-3.0-IGO", "CC-BY-3.0-NL", "CC-BY-3.0-US", "CC-BY-4.0", "LicenseRef-CC-BY-2.0-FR",
    "ODC-By-1.0", "OGL-UK-3.0", "OGL-Canada-2.0", "etalab-2.0", "DL-DE-BY-2.0", "DL-DE-ZERO-2.0",
    "NLOD-2.0", "LicenseRef-IODL-2.0", "LicenseRef-PDL-1.0",
    # public domain in both the US and the work's country of origin; the
    # entry's text states why
    "LicenseRef-public-domain",
})

# exceptions that only add permissions, written after WITH
EXCEPTIONS = frozenset({"Classpath-exception-2.0", "LLVM-exception", "GCC-exception-3.1"})


def _allowed(alternative: str) -> bool:
    license_id, _, exception = alternative.partition(" WITH ")
    return license_id in ALLOWED and (not exception or exception in EXCEPTIONS)


def check_license(expression: str, source_folder: Path):
    if not any(_allowed(alternative) for alternative in expression.split(" OR ")):
        what = "license" if " OR " not in expression and " WITH " not in expression else "license expression"
        raise RuleViolation(f"{source_folder.name}: {what} {expression} is not on the allowed list")
