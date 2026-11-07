"""What every part of the store reads alike: .env files, and the time."""

from datetime import UTC, datetime
from pathlib import Path


def read_env_file(path) -> dict:
    """KEY=VALUE lines; blank lines and # comments are skipped."""
    pairs = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return pairs
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            pairs[key.strip()] = value.strip().strip("'\"")
    return pairs


def now() -> str:
    """This moment in UTC, as a store's records write times."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
