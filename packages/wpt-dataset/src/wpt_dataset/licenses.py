"""The closed list of licenses a source may carry. When in doubt, refuse.

Grey-zone licenses (CC-BY-4.0, CC-BY-SA-4.0, ODbL-1.0) stay out until a
lawyer confirms them; see docs/licenses.md. A license outside the SPDX list
is written LicenseRef-<name>, and its full text sits next to source.json as
LicenseRef-<name>.txt.
"""

from pathlib import Path

from .errors import RuleViolation

ALLOWED = frozenset({
    "CC0-1.0",
    "Unlicense",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Apache-2.0",
    "GPL-3.0-or-later",
    "LGPL-3.0-or-later",
    "AGPL-3.0-or-later",
    # public domain in both the US and the work's country of origin; the
    # LicenseRef text states why
    "LicenseRef-public-domain",
})


def check_license(license_id: str, source_folder: Path):
    if license_id not in ALLOWED:
        raise RuleViolation(f"{source_folder.name}: license {license_id} is not on the allowed list")
    if license_id.startswith("LicenseRef-") and not (source_folder / f"{license_id}.txt").is_file():
        raise RuleViolation(f"{source_folder.name}: {license_id} needs its text in {license_id}.txt")
