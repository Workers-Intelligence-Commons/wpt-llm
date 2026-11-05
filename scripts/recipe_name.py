# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Draw a name for a new model recipe or model series: an adjective and an animal.

Usage:
    uv run scripts/recipe_name.py

Prints a name such as `fluffy-axolotl` that no model recipe or model
series uses yet (it asks `wpt dataset list model` and `list series`: only
the service layer reads data/). The
words come from scripts/names/ (docs/licenses.md, "Word lists of recipe
names"): 4,619 adjectives and 2,497 animals, about 11.5 million names.
"""

import os
import secrets
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "wpt-utils" / "src"))  # no library here
from wpt_utils.lines import bare, say

ROOT = Path(__file__).resolve().parent.parent
NAMES = ROOT / "scripts" / "names"
UV = os.environ.get("UV", "uv")  # uv tells the scripts it runs where it is
ENVIRON = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # the project's environment, not this script's


def words(name):
    return (NAMES / f"{name}.txt").read_text(encoding="utf-8").split()


def main():
    adjectives, animals = words("adjectives"), words("animals")
    taken = {bare(line) for kind in ("model", "series")
             for line in subprocess.run([UV, "run", "wpt", "dataset", "list", kind], cwd=ROOT, env=ENVIRON, check=True,
                                        capture_output=True, text=True).stdout.splitlines()}
    while True:
        name = f"{secrets.choice(adjectives)}-{secrets.choice(animals)}"
        if name not in taken:
            say(name)
            return


if __name__ == "__main__":
    main()
