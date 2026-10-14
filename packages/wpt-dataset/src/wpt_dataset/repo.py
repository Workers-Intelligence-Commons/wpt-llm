"""The data descriptions in a WPT repository, read and validated.

    data/sources/<id>/source.json         + NNNN-*.patch, LicenseRef-*.txt
    data/recipes/mixtures/<name>.json
    data/recipes/tokenizers/<name>.json   + <name>.lock.json (written by builds)
    data/recipes/datasets/<name>.json
    data/allowed-signers                  SSH keys allowed to sign decisions
    data/allowed-gpg-keys.asc             GPG keys allowed to sign decisions (optional)
"""

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

import jsonschema

from .errors import InvalidFile, OutdatedSchema, RuleViolation
from .licenses import check_license
from .pools import parse

CURRENT_SCHEMA = {"source": 3, "mixture": 1, "dataset": 1, "tokenizer": 2, "lock": 2}
ID = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*-[0-9a-f]{8}")
PATCH = re.compile(r"(\d{4})-[A-Za-z0-9._-]+\.patch")
NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")


def address_hash(address: str) -> str:
    """The 8 hex characters that end a source id."""
    return hashlib.sha256(address.encode()).hexdigest()[:8]


def source_id(origin: str, item: str | None, address: str) -> str:
    parts = [origin] + ([item] if item else []) + [address_hash(address)]
    sid = "-".join(parts)
    if not ID.fullmatch(sid):
        raise RuleViolation(f"{sid!r}: origin and item use lowercase a-z, 0-9 and '-' only")
    return sid


@cache
def schema(kind: str) -> dict:
    return json.loads(resources.files(__package__).joinpath(f"schemas/{kind}.schema.json").read_text())


def load_json(path: Path, kind: str) -> dict:
    """Read and validate a file of the given kind (source, mixture, ...)."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidFile(f"{path}: {error}") from error
    version = document.get("schema") if isinstance(document, dict) else None
    if isinstance(version, int) and version < CURRENT_SCHEMA[kind]:
        raise OutdatedSchema(f"{path}: shape {version}, current is {CURRENT_SCHEMA[kind]}; run `wpt-dataset migrate`")
    try:
        jsonschema.validate(document, schema(kind))
    except jsonschema.ValidationError as error:
        where = "/".join(map(str, error.absolute_path)) or "(top level)"
        raise InvalidFile(f"{path}: at {where}: {error.message}") from error
    return document


def dump_json(document) -> str:
    """How every data file is written: 2-space indent, UTF-8, final newline."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class Source:
    id: str
    folder: Path
    data: dict

    @property
    def patches(self) -> list[Path]:
        return sorted(p for p in self.folder.iterdir() if p.suffix == ".patch")

    @property
    def admitted(self) -> bool:
        return self.data["decision"]["decision"] == "admitted"


class Repo:
    def __init__(self, root):
        self.root = Path(root)
        self.data = self.root / "data"

    @classmethod
    def find(cls, start=None) -> "Repo":
        """The nearest folder at or above start (default: cwd) holding data/sources."""
        here = Path(start or Path.cwd()).resolve()
        for folder in (here, *here.parents):
            if (folder / "data" / "sources").is_dir():
                return cls(folder)
        raise RuleViolation(f"no data/sources folder at or above {here}")

    # -- reading -------------------------------------------------------------
    def source_ids(self) -> list[str]:
        folder = self.data / "sources"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def source(self, sid: str) -> Source:
        folder = self.data / "sources" / sid
        return Source(sid, folder, load_json(folder / "source.json", "source"))

    def sources(self) -> list[Source]:
        return [self.source(sid) for sid in self.source_ids()]

    def recipe_path(self, kind: str, name: str) -> Path:
        folder = {"mixture": "mixtures", "tokenizer": "tokenizers", "dataset": "datasets"}[kind]
        return self.data / "recipes" / folder / f"{name}.json"

    def recipe(self, kind: str, name: str) -> dict:
        path = self.recipe_path(kind, name)
        if not path.is_file():
            raise RuleViolation(f"no {kind} recipe named {name!r} ({path.relative_to(self.root)})")
        document = load_json(path, kind)
        if kind == "mixture":
            check_mixture(document, path)
        return document

    def recipe_names(self, kind: str) -> list[str]:
        folder = self.recipe_path(kind, "x").parent
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.json") if not p.name.endswith(".lock.json"))

    def lock_path(self, tokenizer_name: str) -> Path:
        return self.data / "recipes" / "tokenizers" / f"{tokenizer_name}.lock.json"

    # -- rules ---------------------------------------------------------------
    def check(self) -> list[str]:
        """Every rule that needs no download. Returns one line per problem."""
        problems = []

        def attempt(fn, *args):
            try:
                fn(*args)
            except (InvalidFile, OutdatedSchema, RuleViolation) as error:
                problems.append(str(error))

        for sid in self.source_ids():
            attempt(self._check_source, sid)
        for kind in ("mixture", "tokenizer", "dataset"):
            for name in self.recipe_names(kind):
                attempt(self._check_recipe, kind, name)
        for name in self.recipe_names("tokenizer"):
            if self.lock_path(name).exists():
                attempt(load_json, self.lock_path(name), "lock")
        return problems

    def _check_source(self, sid):
        if not ID.fullmatch(sid):
            raise RuleViolation(f"{sid}: a source id is <origin>-<item>-<hash> or <origin>-<hash>, lowercase a-z 0-9 '-'")
        source = self.source(sid)
        expected = address_hash(source.data["address"])
        if not sid.endswith("-" + expected):
            raise RuleViolation(f"{sid}: the id must end with {expected}, the hash of its address")
        for entry in source.data["licenses"]:
            check_license(entry["license"], source.folder)
        numbers = []
        for path in source.folder.iterdir():
            if path.name == "source.json" or (path.name.startswith("LicenseRef-") and path.suffix == ".txt"):
                continue
            match = PATCH.fullmatch(path.name)
            if not match:
                raise RuleViolation(f"{sid}: unexpected file {path.name} (patches are NNNN-name.patch)")
            numbers.append(int(match[1]))
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise RuleViolation(f"{sid}: patches must be numbered 0001, 0002, ... without gaps")
        for cut in source.data.get("cuts", []):
            if cut["end"] <= cut["start"]:
                raise RuleViolation(f"{sid}: cut on {cut['path']} ends at or before its start")

    def _check_recipe(self, kind, name):
        if not NAME.fullmatch(name):
            raise RuleViolation(f"recipe name {name!r}: lowercase a-z, 0-9, '.', '_' and '-' only")
        document = self.recipe(kind, name)
        if kind in ("tokenizer", "dataset"):
            self.recipe("mixture", document["mixture"])
        if kind == "dataset":
            self.recipe("tokenizer", document["tokenizer"])


def check_mixture(mixture: dict, path):
    names = [pool["name"] for pool in mixture["pools"]]
    if len(set(names)) != len(names):
        raise RuleViolation(f"{path}: pool names repeat")
    total = sum(pool["weight"] for pool in mixture["pools"])
    if abs(total - 1) > 1e-9:
        raise RuleViolation(f"{path}: pool weights add up to {total}, not 1")
    for pool in mixture["pools"]:
        try:
            parse(pool["select"])
        except ValueError as error:
            raise RuleViolation(f"{path}: pool {pool['name']}: {error}") from error
