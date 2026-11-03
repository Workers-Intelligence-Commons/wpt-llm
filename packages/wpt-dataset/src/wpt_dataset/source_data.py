"""A source as the rest of WPT reads it: its source.json and its patches.

It stands apart from the service layer (repo.py), which reads and writes the
files, so a result computed from a source (sources.py, kept.py) holds the
code of this module in its key and not that of everything repo.py imports.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Patch:
    name: str  # NNNN-<name>.patch
    data: bytes


@dataclass(frozen=True)
class Source:
    id: str
    data: dict
    patches: list[Patch] = field(default_factory=list)  # in file-name order

    @property
    def admitted(self) -> bool:
        return self.data["decision"]["decision"] == "admitted"

    @property
    def unwritten(self) -> bool:
        """Whether a license entry has no text yet (the schema allows that
        only while the source is not admitted (under review or refused): a
        person writes it later)."""
        return any(entry["text"] == "" for entry in self.data["licenses"])

    @property
    def note(self) -> str:
        """The note a reader prints when it leaves this source out for that."""
        return f"{self.id}: left out, a license text is empty (a person writes it, `save source`)"
