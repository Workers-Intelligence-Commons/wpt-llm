# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Draw a name for a new model recipe: an adjective and an animal.

Usage:
    uv run scripts/recipe_name.py

Prints a name such as `fluffy-axolotl` that no file in data/recipes/models/
uses yet. The words come from scripts/names/ (docs/licenses.md, "Word lists
of recipe names"): 4,619 adjectives and 2,497 animals, about 11.5 million
names.
"""

import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ROOT / "scripts" / "names"


def words(name):
    return (NAMES / f"{name}.txt").read_text(encoding="utf-8").split()


def main():
    adjectives, animals = words("adjectives"), words("animals")
    taken = {path.stem for path in (ROOT / "data" / "recipes" / "models").glob("*.json")}
    while True:
        name = f"{secrets.choice(adjectives)}-{secrets.choice(animals)}"
        if name not in taken:
            print(name)
            return


if __name__ == "__main__":
    main()
