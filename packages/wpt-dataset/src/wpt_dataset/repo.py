"""The service layer: the only code that reads or writes the repository's data.

Every file under data/ is opened here and nowhere else
(docs/architecture.md, "The service layer"):

    data/sources/<id>/source.json         + NNNN-*.patch
    data/recipes/mixtures/<name>.json
    data/recipes/tokenizers/<name>.json   + <name>.lock.json (written by `wpt-dataset build-tokenizer`)
    data/recipes/models/<name>.json       data, architecture and training steps of one model
    data/images/<name>/Dockerfile         a container image of agentic tasks (and any file it copies)
                      + lock.json, licenses.json (written by `wpt-agent lock-image`)
                      + tools.json, the shared pool of tools its programs answer (optional)
    data/allowed-signers                  SSH keys allowed to sign decisions
    data/allowed-gpg-keys.asc             GPG keys allowed to sign decisions (optional)
    data/license-texts/<id>/license-text.json
                                          a license text's decision, kind and license
                      + license-text.txt  the text itself (optional)

Every command reaches these files through a Repo, and so do wpt-web's HTTP
API and the `wpt-dataset` commands that read and write them, which are thin
layers over it. A test (packages/wpt-dataset/tests/test_data_access.py)
fails when other code names the data folder. A Repo hands out documents,
bytes and versions, never a path under data/; external programs get what
they need through it too (git reads patches on its standard input, docker
an image's files as a tar archive).

Each kind of file is one of KINDS. Reading gives a Document: its text, its
JSON (for a JSON kind), its version and the problems the check finds in
it. Every write applies the rules of `wpt-dataset check` to what it writes
and writes nothing when one fails; it is also refused, with StaleFile, when
the file changed since the caller read it. A version is the sha256 of the
file's bytes ("" for a file that does not exist yet); for a source's files,
of its folder (source.json and every patch, source_version). The text of a
source changes only through a patch made from unpacked files
(change_source_files, edit.py), never by writing a patch's text. Generated
files (tokenizer locks, image locks and image license lists) are read like
any other, and written only by the commands that make them, through
write_generated.

The rules every write follows:

- one lock per repository holds each version check and its write together,
  so two saves of the same version cannot both pass: across processes
  (wpt-web and wpt-dataset at the same moment) a lock the operating system
  holds on a file in the repository's git folder (fcntl.flock on Linux and
  macOS, msvcrt.locking on Windows), which it releases when the process
  ends, crash included, so a lock is never left behind; inside a process an
  RLock shared by every Repo of one root, for its threads;
- a write through a symbolic link (a data file that is one, or a folder
  between the repository and it) writes the link's target when that target
  is inside the repository, and is refused (RuleViolation) when it is not; a change of a source's file whose
  path is a link or lies under one is refused (removing the link itself is
  allowed);
- JSON is written as dump_json writes it, and U+2028, U+2029 and U+0085
  are always written as escapes, even in values a save keeps unchanged, so
  that no line splitter breaks a line of JSON; a save over an existing JSON file keeps its bytes when the
  meaning is the same (those escapes aside), and otherwise rewrites only
  the values that changed, keeping the layout and key order of the rest
  (keep_layout; where a new key goes and the indent of what it writes);
- an audit hook (sys.addaudithook), set in every process when its first
  Repo is made, refuses with
  DataAccessRefused any opening, listing, removal or renaming of a path
  under a Repo's data folder, symbolic links resolved, whose nearest caller
  in the project's own code is neither this file nor a test (under a tests/
  folder of the project, or a conftest.py): the runtime half of
  test_data_access.py's scan.
"""

import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import date
from functools import cache, wraps
from importlib import resources
from pathlib import Path

import jsonschema
from wpt_utils.files import lock_file, unlock_file
from wpt_utils.programs import program

from .errors import DataAccessRefused, InvalidFile, NotFound, OutdatedSchema, RuleViolation, StaleFile
from .licenses import check_license
from .license_check import Coverage, check_pattern, field_of, pattern_for, text_id
from .pools import parse

ID = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*-[0-9a-f]{8}")
PATCH = re.compile(r"(\d{4})-[A-Za-z0-9._-]+\.patch")
NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
# the start of a decision's `by` that names a machine, not a person: a machine never admits or refuses; a person's
# name holds no ':'
MACHINE = "machine:"
# the `by` of a license entry the service layer sends back to under-review (Repo.save), and of one a migration
# leaves under review: no person decided it
SERVICE = MACHINE + "wpt-dataset"
# what a license entry's decision is about: a save that changes one of them, without deciding the entry anew,
# sends it back to under-review
DECIDED_ON = ("license", "place", "expected")
# what a decided (admitted or refused) license entry's decision holds, beyond DECIDED_ON: its coverage, and its text
# when its place is the entry itself (the statement); a commit that changes any of it needs an allowed signature
# (Repo.signature_reasons)
ENTRY_SIGNED = DECIDED_ON + ("paths",)
# what a decided license text's decision holds besides its text (named by its id): the exact files it is found by
TEXT_SIGNED = ("files",)
# the license of a work in the public domain: its entry's text is the proof (death dates, publication law), and its
# place may be the entry itself
PUBLIC_DOMAIN = "LicenseRef-public-domain"
SOURCE_FILE = "source.json"
# a license text's id, the sha256 of its normalized text (license_check.text_id), and the files of its folder
TEXT_ID = re.compile(r"[0-9a-f]{64}")
TEXT_FILE, TEXT_ITSELF = "license-text.json", "license-text.txt"
# a key type of an allowed-signers line (ssh-keygen ALLOWED SIGNERS), followed by the key in base64
SIGNER_KEY = re.compile(r"(?:^|\s)(?:ssh-[a-z0-9-]+|ecdsa-sha2-[a-z0-9-]+|sk-[a-z0-9@.-]+)\s+[A-Za-z0-9+/]+=*(?:\s|$)")
GPG_BLOCK = re.compile(r"-----BEGIN PGP PUBLIC KEY BLOCK-----.+?-----END PGP PUBLIC KEY BLOCK-----", re.DOTALL)


@dataclass(frozen=True)
class Kind:
    """One kind of data file. form: "json" (a document validated by its
    schema) or "text". made_by: the command that writes a generated kind
    (None: people write it). single: the one file of a kind that has no
    names (its name is the file's name)."""
    name: str
    form: str
    description: str
    schema: str | None = None
    made_by: str | None = None
    single: str | None = None


KINDS = {kind.name: kind for kind in (
    Kind("source", "json", "a source's description, data/sources/<id>/source.json", "source"),
    Kind("mixture", "json", "a mixture recipe: pools and their weights", "mixture"),
    Kind("tokenizer", "json", "a tokenizer recipe", "tokenizer"),
    Kind("model", "json", "a model recipe: data, architecture and training steps", "model"),
    Kind("lock", "json", "a tokenizer recipe's lock: the vocabulary it froze", "lock", "wpt-dataset build-tokenizer"),
    Kind("dockerfile", "text", "the Dockerfile of a container image of agentic tasks"),
    Kind("image-lock", "json", "an image's lock: its base, archive date and packages", "image-lock",
         "wpt-agent lock-image"),
    Kind("image-licenses", "json", "an image's packages and their licenses", "image-licenses", "wpt-agent lock-image"),
    Kind("image-tools", "json", "an image's shared pool of tools, each answered by a program in the image",
         "image-tools"),
    Kind("allowed-signers", "text", "SSH keys allowed to sign decisions", single="allowed-signers"),
    Kind("allowed-gpg-keys", "text", "GPG keys allowed to sign decisions", single="allowed-gpg-keys.asc"),
    Kind("license-text", "json", "a license text's decision, kind and license, "
         "data/license-texts/<id>/license-text.json (<id>: the sha256 of its normalized text)", "license-text"),
)}
RECIPE_FOLDERS = {"mixture": "mixtures", "tokenizer": "tokenizers", "model": "models"}
IMAGE_FILES = {"dockerfile": "Dockerfile", "image-lock": "lock.json", "image-licenses": "licenses.json",
               "image-tools": "tools.json"}


def address_hash(address: str) -> str:
    """The 8 hex characters that end a new source's id: given once, when the
    source is created, and never derived from its address again."""
    return hashlib.sha256(address.encode()).hexdigest()[:8]


def source_id(origin: str, item: str | None, address: str) -> str:
    """The id a new source gets from its origin, item and first address."""
    parts = [origin] + ([item] if item else []) + [address_hash(address)]
    sid = "-".join(parts)
    if not ID.fullmatch(sid):
        raise RuleViolation(f"{sid!r}: origin and item use lowercase a-z, 0-9 and '-' only")
    return sid


def line_number(given: str) -> int | None:
    """The line a save of a conversation or rating names: a number from 1,
    or None for "new" (a line added at the end); NotFound for anything
    else."""
    if given == "new":
        return None
    if given.isdigit() and int(given) >= 1:
        return int(given)
    raise NotFound(f"no line {given!r}: a number from 1, or new")


@cache
def schema(kind: str) -> dict:
    return json.loads(resources.files(__package__).joinpath(f"schemas/{kind}.schema.json").read_text())


INSIDE_SOURCES = ("conversation", "task", "rating")  # a chat file's lines, a task.json, a rating file's lines


def schema_of(name: str) -> dict:
    """The JSON schema of a JSON kind of data file, or of a shape inside a
    source's files (INSIDE_SOURCES); NotFound for any other name."""
    kind = KINDS.get(name)
    found = kind.schema if kind is not None else name if name in INSIDE_SOURCES else None
    if found is None:
        raise NotFound(f"no JSON schema for {name!r}")
    return schema(found)


@cache
def _validator(kind: str):
    """One validator per kind: jsonschema.validate checks the schema itself
    on every call, which costs more than checking a small document."""
    document = schema(kind)
    jsonschema.validators.validator_for(document).check_schema(document)
    return jsonschema.validators.validator_for(document)(document)


def load_json(path: Path, kind: str) -> dict:
    """Read and validate a file of the given kind (source, mixture, ...)."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidFile(f"{path}: {error}") from error
    validate(document, kind, path)
    return document


def validate(document, kind: str, path):
    """Raise OutdatedSchema or InvalidFile unless the document is a valid
    file of the given kind in its current shape; path names it in messages.
    OutdatedSchema when a migration step recognizes an earlier shape by the
    document's fields."""
    from .migrate import outdated

    earlier = outdated(document, kind)
    if earlier:
        raise OutdatedSchema(f"{path}: an earlier shape ({'; '.join(earlier)}); run `wpt-dataset migrate`")
    try:
        error = jsonschema.exceptions.best_match(_validator(kind).iter_errors(document))
        if error is not None:
            raise error
    except jsonschema.ValidationError as error:
        where = "/".join(map(str, error.absolute_path)) or "(top level)"
        raise InvalidFile(f"{path}: at {where}: {error.message}") from error


# characters str.splitlines() splits on that JSON may hold raw inside a string
LINE_BREAKS = {"\u2028": "\\u2028", "\u2029": "\\u2029", "\x85": "\\u0085"}


def json_text(value, indent=None) -> str:
    """value as JSON, UTF-8 characters kept as they are except U+2028,
    U+2029 and U+0085, written as escapes: the same meaning, and no line
    splitter (str.splitlines) cuts it. indent None: on one line."""
    return _escape_breaks(json.dumps(value, indent=indent, ensure_ascii=False))


def _escape_breaks(text: str) -> str:
    """JSON text with U+2028, U+2029 and U+0085 written as escapes: the same
    meaning, since outside its strings JSON holds only ASCII."""
    for character, escape in LINE_BREAKS.items():
        text = text.replace(character, escape)
    return text


def dump_json(document) -> str:
    """How every data file is written: 2-space indent, UTF-8, final newline."""
    return json_text(document, indent=2) + "\n"


def keep_layout(old: str | None, document, fresh=dump_json) -> str:
    """The text of document written over old (a file's text, or None): old
    itself when its JSON means the same (key order aside), otherwise old
    with only the values that changed rewritten, the rest keeping its
    layout and key order (a new key goes after the key the document puts
    before it). fresh(document) is used when old is None or not plain JSON
    (it does not parse, or repeats a key). Either way U+2028, U+2029 and
    U+0085 come out as escapes, those old kept raw included."""
    if old is None:
        return fresh(document)
    try:
        json.loads(old)
        root = _Node.parse(old, 0)
    except (json.JSONDecodeError, ValueError, IndexError):
        return fresh(document)
    return _escape_breaks(old[:root.start] + root.render(old, document, _indent_before(old, root.start))
                          + old[root.end:])


_SPACE = re.compile(r"[ \t\n\r]*")


def _same(a, b) -> bool:
    """Whether two JSON values mean the same: equal, of the same types
    (true is not 1, 1 is not 1.0), key order aside."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _indent_before(text: str, index: int) -> str:
    """The spaces that start the line holding text[index]."""
    line = text[text.rfind("\n", 0, index) + 1:index]
    return line[:len(line) - len(line.lstrip(" \t"))]


def _fresh_value(value, indent: str, multiline: bool) -> str:
    if not multiline:
        return json_text(value)
    return json_text(value, indent=2).replace("\n", "\n" + indent)


class _Node:
    """One JSON value of a text, with where it starts and ends; an object's
    or array's members keep the text around them (lead: the spaces before a
    member, between: the key, colon and spaces, tail: the spaces before the
    closing bracket)."""

    def __init__(self, start, end, value, members=None, tail=""):
        self.start, self.end, self.value, self.members, self.tail = start, end, value, members, tail

    @classmethod
    def parse(cls, text: str, index: int) -> "_Node":
        index = _SPACE.match(text, index).end()
        opening = text[index]
        if opening not in "{[":
            value, end = json.JSONDecoder().raw_decode(text, index)
            return cls(index, end, value)
        closing, members, values = "}" if opening == "{" else "]", [], {} if opening == "{" else []
        lead_start, at = index + 1, _SPACE.match(text, index + 1).end()
        if text[at] == closing:
            return cls(index, at + 1, values, [], text[index + 1:at])
        while True:
            lead, key, between_start = text[lead_start:at], None, at
            if opening == "{":
                key, after = json.decoder.scanstring(text, at + 1)
                colon = _SPACE.match(text, after).end()
                if text[colon] != ":" or key in values:
                    raise ValueError("not plain JSON")  # a repeated key keeps no single meaning to rewrite
                at = _SPACE.match(text, colon + 1).end()
            child = cls.parse(text, at)
            members.append((lead, key, text[between_start:child.start], child))
            if opening == "{":
                values[key] = child.value
            else:
                values.append(child.value)
            at = _SPACE.match(text, child.end).end()
            if text[at] == ",":
                lead_start, at = at + 1, _SPACE.match(text, at + 1).end()
                continue
            if text[at] != closing:
                raise ValueError("not plain JSON")
            return cls(index, at + 1, values, members, text[child.end:at])

    def render(self, text: str, value, indent: str) -> str:
        """This value's text, rewritten to hold value (indent: the spaces
        that start its line)."""
        if _same(self.value, value):
            return text[self.start:self.end]
        if not self.members or type(self.value) is not type(value) or not isinstance(value, (dict, list)):
            return _fresh_value(value, indent, "\n" in text)  # a file on one line stays on one line
        last_lead, last_between = self.members[-1][0], self.members[-1][2]
        member_indent = _indent_of(last_lead, indent)
        parts = []
        if isinstance(value, list):
            for position, item in enumerate(value):
                if position < len(self.members):
                    lead, _, between, child = self.members[position]
                    parts.append(lead + child.render(text, item, _indent_of(lead, member_indent)))
                else:
                    parts.append(last_lead + _fresh_value(item, member_indent, "\n" in last_lead))
            return "[" + ",".join(parts) + self.tail + "]"
        kept = {key: (lead, between, child) for lead, key, between, child in self.members}
        order = [key for key in kept if key in value]
        for position, key in enumerate(value):  # a new key goes after the key the document puts before it
            if key not in kept:
                before = [k for k in list(value)[:position] if k in order]
                order.insert(order.index(before[-1]) + 1 if before else 0, key)
        for key in order:
            if key in kept:
                lead, between, child = kept[key]
                parts.append(lead + between + child.render(text, value[key], _indent_of(lead, member_indent)))
            else:
                colon = last_between[last_between.rindex(":"):] if ":" in last_between else ": "
                parts.append(last_lead + json_text(key) + colon
                             + _fresh_value(value[key], member_indent, "\n" in last_lead))
        return "{" + ",".join(parts) + self.tail + "}"


def _indent_of(lead: str, otherwise: str) -> str:
    return lead[lead.rindex("\n") + 1:] if "\n" in lead else otherwise


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


@dataclass(frozen=True)
class Document:
    """One data file as read: text (None when it is not UTF-8), document
    (its JSON, or None for a text kind or a file that does not parse),
    version, and the problems the check finds in it."""
    kind: str
    name: str
    text: str | None
    document: object
    version: str
    problems: list[str]


def check_person(name: str, where: str):
    """RuleViolation for a person's name holding ':', which MACHINE keeps
    for a machine's name."""
    if ":" in name:
        raise RuleViolation(f"{where}: {name!r}: a person's name holds no ':' (`{MACHINE}<host>` names a machine)")


def check_decision(decision: dict, where: str):
    """RuleViolation unless a decision's `by` (a source's, or a license
    entry's) is a person's name (check_person) or, under review only, a
    machine's (MACHINE<host>): a machine never admits or refuses."""
    by = decision["by"]
    if not by.startswith(MACHINE):
        check_person(by, where)
        return
    if decision["decision"] != "under-review":
        raise RuleViolation(f"{where}: {decision['decision']} by {by!r}: a machine never admits or refuses; a person "
                            f"decides, under their own name")
    host = by.removeprefix(MACHINE)
    if not host or ":" in host:
        raise RuleViolation(f"{where}: {by!r}: a machine's name is `{MACHINE}<host>`, its host name holding no ':'")


def empty_source(name: str, *, license: str, attribution: str, text: str, by: str, formats: dict[str, str],
                 tags=(), started: str | None = None) -> dict:
    """The source.json of a new `empty` source (CONTRIBUTING.md, "Adding your
    own texts"): text written for WPT, each file's format by its path, under
    one license entry that covers the same paths, whose place is the entry
    itself (no statement can exist in text WPT writes itself); the source
    and its entry under review ("under-review") until a maintainer decides."""
    today = started or date.today().isoformat()
    decision = {"decision": "under-review", "by": by, "date": today}
    document = {"type": "empty", "address": f"empty:{name}",
                "licenses": [{"license": license, "paths": list(formats), "attribution": attribution,
                              "place": {"entry": True}, "text": text, "decision": dict(decision)}],
                "formats": [{"format": kind, "paths": [path]} for path, kind in formats.items()],
                "decision": decision}
    if tags:
        document["tags"] = [{"tags": list(tags), "paths": ["**"]}]
    return document


LOCK_FILE = "wpt-data.lock"  # in the repository's git folder (beside its index), or .wpt-data.lock at its root


class _RepositoryLock:
    """The lock of one repository: an RLock for the threads of this process,
    and, while a thread holds it, a lock the operating system holds on a
    file for this process (fcntl.flock on Linux and macOS, msvcrt.locking
    on Windows), so another process waits too. The system releases a file
    lock when its process ends, crash included: no lock is left behind."""

    def __init__(self, root: Path):
        self._root = root
        self._threads = threading.RLock()
        self._depth = 0  # how many times the holding thread entered
        self._file = None

    def __enter__(self):
        self._threads.acquire()
        try:
            if self._depth == 0:
                self._file = lock_file(_lock_path(self._root))
            self._depth += 1
        except BaseException:
            self._threads.release()
            raise
        return self

    def __exit__(self, *_):
        try:
            self._depth -= 1
            if self._depth == 0:
                file, self._file = self._file, None
                unlock_file(file)
        finally:
            self._threads.release()


def _lock_path(root: Path) -> Path:
    """The lock file of a repository: in its git folder (.git, or the folder
    a worktree's .git file names), where git status never lists it, or
    .wpt-data.lock at the root of a folder that is not a git working tree."""
    git = root / ".git"
    if git.is_dir():
        return git / LOCK_FILE
    if git.is_file():
        text = git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            folder = Path(text.removeprefix("gitdir:").strip())
            folder = folder if folder.is_absolute() else root / folder
            if folder.is_dir():
                return folder / LOCK_FILE
    return root / f".{LOCK_FILE}"


_LOCKS: dict[Path, _RepositoryLock] = {}  # one per repository root, shared by its Repo objects
_LOCKS_LOCK = threading.Lock()

def _locked(method):
    """The method, run while holding its repository's lock: a version check
    and the write it allows happen together, in every thread and process."""
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return run


# -- the audit hook: data files are reached only from this file ---------------------------
# the data folders of every Repo made in this process, as written and with links resolved; replaced, never
# changed in place, so the hook reads a whole tuple while another thread adds a root
_GUARDED: tuple[str, ...] = ()
_GUARDING = threading.Lock()
_HOOKED = False
_HERE = os.path.normcase(os.path.abspath(__file__))
# the project's checkout: packages/wpt-dataset/src/wpt_dataset/repo.py lies four folders below it
_PROJECT = os.path.normcase(os.path.abspath(Path(__file__).parents[4]))
# which arguments of each audited event are paths it reads or changes (os.symlink's first is the link's text)
_EVENTS = {"open": (0,), "os.listdir": (0,), "os.scandir": (0,), "os.remove": (0,), "os.rename": (0, 1),
           "os.mkdir": (0,), "os.rmdir": (0,), "os.chmod": (0,), "os.symlink": (1,), "os.link": (0, 1),
           "os.truncate": (0,), "os.utime": (0,), "shutil.rmtree": (0,), "shutil.copyfile": (0, 1),
           "shutil.copytree": (0, 1), "shutil.move": (0, 1), "glob.glob": (0,), "glob.glob/2": (0, 2)}
_LIBRARIES = tuple(sorted({os.path.normcase(os.path.abspath(path)) + os.sep for name, path in sysconfig.get_paths().items()
                           if name in ("stdlib", "platstdlib", "purelib", "platlib", "scripts")}))
_RMTREE = shutil.rmtree.__code__


def _guard(data: Path):
    """Guard data (a Repo's data folder) with the audit hook, set once per
    process before the first Repo is handed out."""
    global _GUARDED, _HOOKED
    folders = {os.path.normcase(os.path.abspath(data)), os.path.normcase(os.path.realpath(data))}
    with _GUARDING:
        _GUARDED = tuple(dict.fromkeys((*_GUARDED, *sorted(folders))))
        if not _HOOKED:
            sys.addaudithook(_audit)
            _HOOKED = True


def _audit(event, args):
    paths = _EVENTS.get(event)
    if paths is None:
        return
    for index in paths:
        if index < len(args) and _in_data(args[index]) and not (
                event != "shutil.rmtree" and _inside_rmtree(args[index])):
            caller = _project_caller()
            if caller is not None and caller != _HERE and not _is_test(caller):
                raise DataAccessRefused(f"{caller} reaches {os.fsdecode(args[index])} ({event}): only the service "
                                        f"layer, wpt_dataset/repo.py, opens files under data/ "
                                        f"(docs/architecture.md, \"The service layer\")")


def _in_data(path) -> bool:
    """Whether path lies under a guarded data folder, as written or with its
    symbolic links resolved."""
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False  # a file descriptor
    guarded = _GUARDED
    try:
        written = os.path.abspath(os.fsdecode(path))
        if _under(os.path.normcase(written), guarded):
            return True
        return _under(os.path.normcase(os.path.realpath(written)), guarded)
    except (TypeError, ValueError, OSError):
        return False


def _inside_rmtree(path) -> bool:
    """Whether path is relative and named inside shutil.rmtree, which opens
    and removes each child by its name relative to its parent folder's
    descriptor (dir_fd), not to the current folder; rmtree's own event
    already checked the folder it removes."""
    if os.path.isabs(os.fsdecode(path)):
        return False
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code is _RMTREE:
            return True
        frame = frame.f_back
    return False


def _under(found: str, folders) -> bool:
    return any(found == folder or found.startswith(folder + os.sep) for folder in folders)


def _project_caller() -> str | None:
    """The file of the nearest caller that is the project's own code (not
    Python's library, not an installed package); None when there is none."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if not name.startswith("<"):
            name = os.path.normcase(os.path.abspath(name))
            if not name.startswith(_LIBRARIES):
                return name
        frame = frame.f_back
    return None


def _is_test(name: str) -> bool:
    """Whether a file is test code: a conftest.py, or a file under a tests
    folder, inside the project's checkout (a folder named tests above the
    checkout, or a conftest.py outside it, does not count)."""
    if not name.startswith(_PROJECT + os.sep):
        return False
    if os.path.basename(name) == "conftest.py":
        return True
    return "tests" in Path(name[len(_PROJECT) + 1:]).parts


class Located:
    """Where a file being migrated lives, for a migration step: str() names
    it in messages, and the step may read or remove a file beside it (the
    LicenseRef texts shape 3 kept next to source.json)."""

    def __init__(self, repo: "Repo", path: Path):
        self._repo, self._path = repo, path

    def __str__(self):
        return str(self._path)

    def sibling_text(self, name: str) -> str | None:
        found = self._path.parent / name
        return found.read_text(encoding="utf-8") if found.is_file() else None

    def remove_sibling(self, name: str):
        self._repo._target(self._path.parent / name).unlink()


class Repo:
    def __init__(self, root):
        self.root = Path(root)
        self._data = self.root / "data"
        _guard(self._data)
        key = self.root.resolve()
        with _LOCKS_LOCK:
            self._lock = _LOCKS.setdefault(key, _RepositoryLock(key))

    def data_folder(self) -> str:
        """The data folder's path from the repository's root, with /: what a
        copy of the repository's code leaves out (wpt_worker.manager)."""
        return self._data.relative_to(self.root).as_posix()

    @classmethod
    def find(cls, start=None) -> "Repo":
        """The nearest folder at or above start (default: cwd) holding data/sources."""
        here = Path(start or Path.cwd()).resolve()
        for folder in (here, *here.parents):
            if (folder / "data" / "sources").is_dir():
                return cls(folder)
        raise RuleViolation(f"no data/sources folder at or above {here}")

    # -- where each kind lives ---------------------------------------------------
    def _kind(self, kind: str) -> Kind:
        if kind not in KINDS:
            raise NotFound(f"no kind of data named {kind!r}; the kinds are {', '.join(KINDS)}")
        return KINDS[kind]

    def _path(self, kind: str, name: str) -> Path:
        """The file of that kind and name; RuleViolation for a name that is
        not one (so no name reaches outside its folder)."""
        found = self._kind(kind)
        if found.single:
            if name != found.single:
                raise NotFound(f"{kind} has one file, {found.single}, not {name!r}")
            return self._data / found.single
        pattern = ID if kind == "source" else TEXT_ID if kind == "license-text" else NAME
        if not pattern.fullmatch(name) and not (kind == "source" and name in self.source_ids()):
            what = ("a source id is <origin>-<item>-<hash> or <origin>-<hash>, lowercase a-z 0-9 '-'"
                    if kind == "source" else "a license text's id is the sha256 of its normalized text, 64 of "
                    "0-9 a-f" if kind == "license-text" else "a name is lowercase a-z, 0-9, '.', '_' and '-'")
            raise RuleViolation(f"no {kind} {name!r}: {what}")
        if kind == "source":
            return self._data / "sources" / name / SOURCE_FILE
        if kind == "license-text":
            return self._data / "license-texts" / name / TEXT_FILE
        if kind in RECIPE_FOLDERS:
            return self._data / "recipes" / RECIPE_FOLDERS[kind] / f"{name}.json"
        if kind == "lock":
            return self._data / "recipes" / "tokenizers" / f"{name}.lock.json"
        return self._data / "images" / name / IMAGE_FILES[kind]

    def relative(self, kind: str, name: str) -> str:
        """The file's place in the repository, for messages and records:
        data/... with forward slashes."""
        return self._path(kind, name).relative_to(self.root).as_posix()

    def _source_folder(self, sid: str) -> Path:
        return self._path("source", sid).parent

    # -- reading -------------------------------------------------------------
    def kinds(self) -> list[dict]:
        """Every kind of data file: name, form, description, schema (its name,
        or None), made_by (None: people write it) and single."""
        return [asdict(kind) for kind in KINDS.values()]

    def names(self, kind: str) -> list[str]:
        """The names of the files of a kind that exist, sorted."""
        found = self._kind(kind)
        if found.single:
            return [found.single] if (self._data / found.single).is_file() else []
        if kind == "source":
            return self.source_ids()
        if kind == "license-text":
            return self.license_text_ids()
        if kind in RECIPE_FOLDERS:
            folder = self._data / "recipes" / RECIPE_FOLDERS[kind]
            return sorted(p.stem for p in folder.glob("*.json") if not p.name.endswith(".lock.json")) \
                if folder.is_dir() else []
        if kind == "lock":
            folder = self._data / "recipes" / "tokenizers"
            return sorted(p.name.removesuffix(".lock.json") for p in folder.glob("*.lock.json")) \
                if folder.is_dir() else []
        if kind == "dockerfile":
            return self.image_names()
        return [name for name in self.image_names() if (self._data / "images" / name / IMAGE_FILES[kind]).is_file()]

    def exists(self, kind: str, name: str) -> bool:
        return self._path(kind, name).is_file()

    def read(self, kind: str, name: str) -> Document:
        """The file as it is, with its problems; NotFound when it does not exist."""
        path = self._path(kind, name)
        if not path.is_file():
            raise NotFound(f"{self.relative(kind, name)} does not exist")
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        document = None
        if KINDS[kind].form == "json" and text is not None:
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                pass
        return Document(kind, name, text, document, _sha256(data), self.problems(kind, name))

    def version(self, kind: str, name: str) -> str:
        """The sha256 of the file's bytes, "" when it does not exist."""
        path = self._path(kind, name)
        return _sha256(path.read_bytes()) if path.is_file() else ""

    def problems(self, kind: str, name: str) -> list[str]:
        """What `wpt-dataset check` finds in that one file: [] or its problems."""
        if kind == "source":
            return self.check_source(name)
        try:
            if kind == "license-text":
                self._check_license_text(name, load_json(self._path(kind, name), kind))
            elif kind in RECIPE_FOLDERS:
                self._check_recipe(kind, name)
            elif kind == "dockerfile":
                self._image_folder(name)
            elif kind == "image-tools":
                self.image_tools(name)
            elif KINDS[kind].single and KINDS[kind].form == "text":
                self._check_signers(kind, self._path(kind, name).read_text(encoding="utf-8"))
            else:
                load_json(self._path(kind, name), kind)
        except (InvalidFile, OutdatedSchema, RuleViolation, UnicodeDecodeError) as error:
            return [str(error)]
        return []

    def source_ids(self) -> list[str]:
        folder = self._data / "sources"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def source(self, sid: str) -> Source:
        patches = self.patches(sid)  # NotFound first
        return Source(sid, load_json(self._data / "sources" / sid / SOURCE_FILE, "source"), patches)

    def patches(self, sid: str) -> list[Patch]:
        """A source's patches, in file-name order, whatever the shape of its
        source.json (a migration reads them before the source can be read)."""
        folder = self._data / "sources" / sid
        if not ID.fullmatch(sid) and sid not in self.source_ids() or not folder.is_dir():
            raise NotFound(f"no source {sid}")
        return [Patch(p.name, p.read_bytes()) for p in sorted(folder.iterdir()) if p.suffix == ".patch"]

    def sources(self, note=None) -> list[Source]:
        """Every source. With a function as note, a source whose license text
        is empty (Source.unwritten) is left out and note(Source.note) is
        called for it: readers skip it instead of failing."""
        found = [self.source(sid) for sid in self.source_ids()]
        if note is None:
            return found
        for source in found:
            if source.unwritten:
                note(source.note)
        return [source for source in found if not source.unwritten]

    def recipe(self, kind: str, name: str) -> dict:
        path = self._path(kind, name)
        if not path.is_file():
            raise RuleViolation(f"no {kind} recipe named {name!r} ({path.relative_to(self.root).as_posix()})")
        document = load_json(path, kind)
        if kind == "mixture":
            check_mixture(document, path)
        return document

    def recipe_names(self, kind: str) -> list[str]:
        return self.names(kind)

    def generated(self, kind: str, name: str) -> dict | None:
        """A generated file (lock, image-lock, image-licenses), validated; None
        when it does not exist."""
        path = self._path(kind, name)
        return load_json(path, kind) if path.is_file() else None

    def license_text_ids(self) -> list[str]:
        """The ids of the license texts, sorted: the folders of
        data/license-texts/ that hold a license-text.json."""
        folder = self._data / "license-texts"
        return sorted(p.name for p in folder.iterdir() if (p / TEXT_FILE).is_file()) if folder.is_dir() else []

    def license_texts(self) -> dict[str, dict]:
        """{id: license-text.json} of every license text, validated
        (license_check.LicenseTexts finds a file's text among them)."""
        return {text: load_json(self._path("license-text", text), "license-text")
                for text in self.license_text_ids()}

    def license_text_itself(self, text: str) -> bytes | None:
        """The bytes of a license text's license-text.txt, or None when it
        has none."""
        path = self._path("license-text", text).parent / TEXT_ITSELF
        return path.read_bytes() if path.is_file() else None

    def image_names(self) -> list[str]:
        folder = self._data / "images"
        return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []

    def _image_folder(self, name: str) -> Path:
        """data/images/<name>/; RuleViolation unless the name is valid and
        the folder holds a Dockerfile."""
        if not NAME.fullmatch(name):
            raise RuleViolation(f"no image {name!r}: an image name is lowercase a-z, 0-9, '.', '_' and '-'")
        folder = self._data / "images" / name
        if not (folder / IMAGE_FILES["dockerfile"]).is_file():
            raise RuleViolation(f"no image {name!r}: data/images/{name}/Dockerfile does not exist")
        return folder

    def check_image(self, name: str):
        """RuleViolation unless data/images/<name>/ holds a Dockerfile."""
        self._image_folder(name)

    def image_tools(self, name: str) -> list[dict]:
        """The declarations of data/images/<name>/tools.json, the image's
        shared pool of tools ([] without the file), validated; RuleViolation
        as check_image, InvalidFile for a pool the check refuses."""
        self._image_folder(name)
        return self._check_image_tools(name) if self.exists("image-tools", name) else []

    def _check_image_tools(self, name) -> list[dict]:
        from .tasks import check_declared

        document = load_json(self._path("image-tools", name), "image-tools")
        check_declared(self.relative("image-tools", name), [tool["name"] for tool in document["tools"]])
        return document["tools"]

    def image_files(self, name: str) -> dict[str, tuple[bool, bytes]]:
        """{path in the image's folder: (executable, content)} of every file
        there, the Dockerfile, its lock and licenses included, but none git
        ignores (a __pycache__ a Python run left): the image, and its tag,
        come from the files the repository can hold; RuleViolation as
        check_image."""
        folder = self._image_folder(name)
        found = [p for p in sorted(folder.rglob("*")) if p.is_file()]
        ignored = self._ignored([p.relative_to(self.root).as_posix() for p in found])
        return {p.relative_to(folder).as_posix(): (bool(p.stat().st_mode & 0o100), p.read_bytes())
                for p in found if p.relative_to(self.root).as_posix() not in ignored}

    # -- the rules -------------------------------------------------------------
    def check(self) -> list[str]:
        """Every rule that needs no download. Returns one line per problem."""
        problems = []

        def attempt(fn, *args):
            try:
                fn(*args)
            except (InvalidFile, OutdatedSchema, RuleViolation) as error:
                problems.append(str(error))

        text_ids = set(self.license_text_ids())
        for sid in self.source_ids():
            problems += self.check_source(sid, text_ids)
        for text in sorted(text_ids):
            problems += self.problems("license-text", text)
        for kind in RECIPE_FOLDERS:
            for name in self.names(kind):
                attempt(self._check_recipe, kind, name)
        for name in self.names("lock"):
            if name in self.names("tokenizer"):
                attempt(load_json, self._path("lock", name), "lock")
        for name in self.image_names():
            attempt(self._check_image, name)
        for kind in ("allowed-signers", "allowed-gpg-keys"):
            for name in self.names(kind):
                problems += self.problems(kind, name)
        return problems

    def check_source(self, sid, text_ids=None) -> list[str]:
        """Every rule on one source that needs no download: [] or one problem."""
        try:
            self._check_source_document(sid, load_json(self._data / "sources" / sid / SOURCE_FILE, "source"),
                                        text_ids)
        except (InvalidFile, OutdatedSchema, RuleViolation) as error:
            return [str(error)]
        return []

    def check_fetched(self, sid: str, store, telemetry=None) -> tuple[int, int, dict, list]:
        """The rules on a fetched source, as `wpt-dataset check --fetch`
        applies them: its cuts, patches, license and format maps,
        and each file converted (chat and task files checked, the texts
        pointer tasks point to read from their sources and checked against
        their sha256), and the statement at each license entry's place
        (kept.statements). Returns (entries, Markdown documents, {records
        file: what it left out}, the license_check.Statement of each entry
        whose statement does not let its files train: not there, no
        license text or one it does not expect, or a text not admitted).
        Those are reported, not refused: a new source's texts start under
        review, and the build leaves only that entry's files out."""
        from .kept import statements
        from .license_check import LicenseTexts
        from .manifest import SYMLINK
        from .markdown import documents, format_of
        from .sources import source_tree
        from .tasks import fill, of_source

        source = self.source(sid)
        tree, _ = source_tree(source, store, telemetry)
        skipped = {}
        # the task folders are parsed once, by of_source below, not again by documents
        # a symbolic link is no document, so it needs no format entry
        others = {path for path, entry in tree.items()
                  if entry.mode != SYMLINK and format_of(source, path)["format"] != "task"}
        converted = sum(1 for _ in documents(source, tree, store, self, skipped, only=others))
        fill(list(of_source(source, tree, store, self).values()), self, store)
        found = statements(source, tree, store, LicenseTexts(self.license_texts()))
        return len(tree), converted, skipped, [statement for statement in found if not statement.admitted]

    def _check_license_text(self, text, document):
        """The rules on a license text beyond its schema: a machine never
        admits or refuses it, and its folder holds license-text.json and,
        optionally, license-text.txt, whose normalized text has its id and
        whose sha256 it lists."""
        where = self.relative("license-text", text)
        check_decision(document["decision"], where)
        folder = self._path("license-text", text).parent
        held = [p.name for p in folder.iterdir()] if folder.is_dir() else []
        if unexpected := sorted(name for name in held if name not in (TEXT_FILE, TEXT_ITSELF)):
            raise RuleViolation(f"{where}: unexpected file {unexpected[0]} beside it (a license text's folder holds "
                                f"{TEXT_FILE} and, optionally, {TEXT_ITSELF})")
        itself = folder / TEXT_ITSELF
        if itself.is_file():
            content = itself.read_bytes()
            if text_id(content) != text:
                raise RuleViolation(f"{where}: {TEXT_ITSELF}'s normalized text has the id {text_id(content)}, not "
                                    f"{text}")
            if _sha256(content) not in document["files"]:
                raise RuleViolation(f"{where}: `files` does not list the sha256 of {TEXT_ITSELF}")

    def _check_source_document(self, sid, document, text_ids=None):
        """The source rules that need no download; text_ids: the ids of the
        license texts (read here when not given)."""
        if not ID.fullmatch(sid):
            raise RuleViolation(f"{sid}: a source id is <origin>-<item>-<hash> or <origin>-<hash>, lowercase a-z 0-9 '-'")
        check_decision(document["decision"], sid)
        if "addresses" in document:
            if document["address"] != document["addresses"][0]["address"]:
                raise RuleViolation(f"{sid}: `address` must be the first of `addresses`")
            folders = [part["folder"] for part in document["addresses"]]
            if len({folder.casefold() for folder in folders}) != len(folders):
                raise RuleViolation(f"{sid}: each of `addresses` needs a folder of its own")
        for number, entry in enumerate(document["licenses"], start=1):
            where = f"{sid}: license entry {number}"
            check_license(entry["license"], sid)
            check_decision(entry["decision"], where)
            for pattern in entry["paths"]:
                check_pattern(pattern, where)
            if "entry" in entry["place"]:
                if document["type"] != "empty" and entry["license"] != PUBLIC_DOMAIN:
                    raise RuleViolation(f"{where}: the entry itself is its place only where no statement can exist "
                                        f"in the content: in an empty source (text WPT writes itself), or for "
                                        f"{PUBLIC_DOMAIN} (a public-domain proof); name the file, field or page "
                                        f"that states its license")
            elif field_of(entry) is None:
                text_ids = set(self.license_text_ids()) if text_ids is None else text_ids
                if unknown := [text for text in entry["expected"] if text not in text_ids]:
                    raise RuleViolation(f"{where}: expects {', '.join(unknown)}, not the id of a license text "
                                        f"(data/license-texts/)")
        numbers = []
        folder = self._data / "sources" / sid
        for path in folder.iterdir() if folder.is_dir() else ():
            if path.name == SOURCE_FILE:
                continue
            match = PATCH.fullmatch(path.name)
            if not match:
                raise RuleViolation(f"{sid}: unexpected file {path.name} (patches are NNNN-name.patch)")
            numbers.append(int(match[1]))
        if sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise RuleViolation(f"{sid}: patches must be numbered 0001, 0002, ... without gaps")
        if "held_out_by" in document:
            try:
                re.compile(document["held_out_by"]["pattern"])
            except re.error as error:
                raise RuleViolation(f"{sid}: held_out_by's pattern is not a regular expression: {error}") from None
        for entry in document["formats"]:
            for key in ("key", "joined_key") if "join" in entry else ():
                try:
                    re.compile(entry["join"][key])
                except re.error as error:
                    raise RuleViolation(f"{sid}: a join's {key} is not a regular expression: {error}") from None

    def _check_image(self, name):
        self._image_folder(name)
        for kind in ("image-lock", "image-licenses"):
            if self.exists(kind, name):
                load_json(self._path(kind, name), kind)
        self.image_tools(name)

    def _check_recipe(self, kind, name, document=None):
        if not NAME.fullmatch(name):
            raise RuleViolation(f"recipe name {name!r}: lowercase a-z, 0-9, '.', '_' and '-' only")
        if document is None:
            document = self.recipe(kind, name)
        if kind == "tokenizer":
            self.recipe("mixture", document["mixture"])
        if kind == "model":
            self.recipe("mixture", document["data"]["mixture"])
            self.recipe("tokenizer", document["data"]["tokenizer"])

    def _check_signers(self, kind, text):
        """allowed-signers: every line that is not blank or a # comment names
        a key (its type, then the key in base64); allowed-gpg-keys.asc:
        empty, or ASCII-armored public key blocks."""
        where = self.relative(kind, KINDS[kind].single)
        if kind == "allowed-signers":
            for number, line in enumerate(text.splitlines(), start=1):
                if line.strip() and not line.lstrip().startswith("#") and not SIGNER_KEY.search(line):
                    raise RuleViolation(f"{where} line {number}: no key (principals, options, a key type such as "
                                        f"ssh-ed25519, then the key)")
        elif text.strip() and not GPG_BLOCK.search(text):
            raise RuleViolation(f"{where}: no ASCII-armored public key block (gpg --armor --export)")

    def _rules(self, kind: str, name: str, content):
        """The check's rules on content about to be written as that file."""
        where = self.relative(kind, name)
        found = KINDS[kind]
        if found.form == "json":
            validate(content, found.schema, where)
        if kind == "source":
            self._check_source_document(name, content)
        elif kind == "license-text":
            self._check_license_text(name, content)
        elif kind == "mixture":
            check_mixture(content, where)
        elif kind == "image-tools":
            from .tasks import check_declared

            self._image_folder(name)
            check_declared(where, [tool["name"] for tool in content["tools"]])
        if kind in RECIPE_FOLDERS:
            self._check_recipe(kind, name, content)
        elif found.single and found.form == "text":
            self._check_signers(kind, content)

    # -- writing ---------------------------------------------------------------
    @_locked
    def save(self, kind: str, name: str, content, version: str) -> str:
        """Write a file a person writes: content is a JSON document for a
        JSON kind, text for a text kind. version is what the caller read
        ("" for a new file): StaleFile when the file is not at that version
        any more. InvalidFile, OutdatedSchema or RuleViolation say why the
        check refuses the content; then nothing is written. A source's
        license entry whose license, place or expected texts the save
        changes goes back to under-review (sent_back). Text is written
        as UTF-8 with \\n line ends; JSON over an existing file keeps its
        layout where the meaning did not change (keep_layout). Returns the
        new version."""
        found = self._kind(kind)
        path = self._path(kind, name)
        if found.made_by:
            raise RuleViolation(f"{self.relative(kind, name)} is written only by `{found.made_by}`")
        if found.form == "text" and not isinstance(content, str):
            raise InvalidFile(f"{self.relative(kind, name)}: send text")
        self._fresh(path, version)
        if kind == "source" and version:
            try:
                content = sent_back(json.loads(path.read_text(encoding="utf-8")), content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass  # nothing held to compare with: the save writes what it was given
        self._rules(kind, name, content)
        data = self._json_text(path, content) if found.form == "json" else content.replace("\r\n", "\n")
        return self._write_data(path, data.encode("utf-8"))

    def add_found_license_text(self, sid: str, place: dict, store) -> tuple[str, str | None]:
        """Add the license text found at a place of the source ({"file":
        path} of its final tree, or {"page": address} downloaded with it),
        as the license check reads it there, when no license text holds it
        yet: a new item under review by SERVICE, its id the text's
        (text_id), its file's sha256 in `files`, where it was found (the
        path, or the page's address), its bytes as license-text.txt. A
        person reads it and decides it. Returns (its id, its place in the
        repository; None when an item holds that text already, and nothing
        is written). NotFound when the source, its file or its stored page
        is not there; NotDownloaded when the store does not hold the
        source; RuleViolation for another kind of place."""
        from .sources import license_page

        if set(place) == {"file"}:
            content = self.read_source_file(sid, place["file"], store)[1]
            where = place["file"]
        elif set(place) == {"page"}:
            content, where = license_page(self.source(sid), store, place["page"]), place["page"]
            if content is None:
                raise NotFound(f"{sid}: the store holds no license page {where} for it; `wpt-dataset download "
                               f"{sid}` fetches it with the source")
        else:
            raise RuleViolation(f"{sid}: a license text is found at a file or a page, not at {place}")
        text = text_id(content)
        if self.exists("license-text", text):
            return text, None
        document = {"decision": {"decision": "under-review", "by": SERVICE, "date": date.today().isoformat(),
                                 "reason": "first met in a source: a person reads it"},
                    "kind": "license", "found": {"source": sid, "path": where}, "files": [_sha256(content)],
                    "explanation": ""}
        return text, self.add_license_text(text, document, content)

    @_locked
    def add_license_text(self, text: str, document: dict, itself: bytes | None) -> str:
        """Write a new license text (one first met in a source:
        add_found_license_text, and the migration's), with its text itself
        as license-text.txt when given, after the rules of `check`. StaleFile when it exists already; InvalidFile,
        OutdatedSchema or RuleViolation say why the check refuses it, and
        then nothing is written. Returns its place in the repository."""
        path = self._path("license-text", text)
        self._fresh(path, "")
        where = self.relative("license-text", text)
        validate(document, "license-text", where)
        if itself is not None and (text_id(itself) != text or _sha256(itself) not in document["files"]):
            raise RuleViolation(f"{where}: its text's normalized text has another id, or `files` does not list its "
                                f"sha256")
        self._check_license_text(text, document)
        if itself is not None:
            self._write_data(path.parent / TEXT_ITSELF, itself)
        self._write_data(path, self._json_text(path, document).encode("utf-8"))
        return where

    def save_source(self, sid, document, version) -> str:
        """save("source", ...): source.json, after the source rules."""
        return self.save("source", sid, document, version)

    @_locked
    def write_generated(self, kind: str, name: str, document: dict) -> str:
        """Write a generated file (lock, image-lock, image-licenses), after
        its schema: only the command that makes it calls this. Returns its
        place in the repository."""
        if not KINDS[kind].made_by:
            raise RuleViolation(f"{kind} is written by people, through save")
        if kind != "lock":
            self._image_folder(name)
        validate(document, KINDS[kind].schema, self.relative(kind, name))
        path = self._path(kind, name)
        self._write_data(path, self._json_text(path, document).encode("utf-8"))
        return self.relative(kind, name)

    def _fresh(self, path: Path, version: str):
        current = _sha256(path.read_bytes()) if path.is_file() else ""
        if current != version:
            what = "exists already" if not version else "does not exist any more" if not current else \
                "changed on disk since it was read"
            raise StaleFile(f"{path.relative_to(self.root).as_posix()}: {what}")

    def _json_text(self, path: Path, document) -> str:
        """document as the file's new text: over an existing UTF-8 file, its
        layout kept where the meaning did not change (keep_layout)."""
        try:
            old = path.read_text(encoding="utf-8") if path.is_file() else None
        except UnicodeDecodeError:
            old = None
        return keep_layout(old, document)

    def _target(self, path: Path) -> Path:
        """Where a write to path lands: path itself, or, when it or a folder
        between the repository and it is a symbolic link, the link's target
        (every link resolved). RuleViolation when that place is outside the
        repository: nothing is written there."""
        root = Path(os.path.realpath(self.root))
        target = Path(os.path.realpath(path))
        if not target.is_relative_to(root):
            try:
                where = path.relative_to(self.root).as_posix()
            except ValueError:
                where = str(path)
            raise RuleViolation(f"{where} leads to {target}, outside the repository (through a symbolic link): the "
                                f"service layer writes a link's target only inside the repository")
        return target

    def _write_data(self, path: Path, data: bytes) -> str:
        """Write data as path's bytes (as the bytes of the target of a link
        on the way, _target), unless they are already; returns their sha256."""
        target = self._target(path)
        if target.is_file() and target.read_bytes() == data:
            return _sha256(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.writing")
        temporary.unlink(missing_ok=True)  # a link left there would be written through
        temporary.write_bytes(data)
        os.replace(temporary, target)
        return _sha256(data)

    # -- a source's files: always through patches ------------------------------------
    def source_version(self, sid: str) -> str:
        """The sha256 of a source's folder: source.json and every patch, by
        name. A change to any of them changes it."""
        folder = self._data / "sources" / sid
        if not ID.fullmatch(sid) and sid not in self.source_ids() or not folder.is_dir():
            raise NotFound(f"no source {sid}")
        digest = hashlib.sha256()
        for path in sorted(folder.iterdir()):
            if path.is_file():
                digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
        return digest.hexdigest()

    def source_files(self, sid: str, store) -> tuple[str, dict[str, dict]]:
        """(version, {path: {mode, size, format (None when the format map
        does not give exactly one; "joined" for a file a records entry's
        `join` reads), tags}}) of the source's final files."""
        from . import manifest
        from .errors import FormatMapError
        from .markdown import format_of
        from .records import joined_by
        from .sources import source_tree, tags_of

        version, source = self.source_version(sid), self.source(sid)
        tree, _ = source_tree(source, store)
        files = {}
        for path in sorted(tree, key=str.encode):
            try:
                entry = format_of(source, path)
                kind = "joined" if joined_by(entry, path) else entry["format"]
            except FormatMapError:
                kind = None
            files[path] = {"mode": tree[path].mode, "size": manifest.size(store, tree[path]), "format": kind,
                           "tags": sorted(tags_of(source, path))}
        return version, files

    def read_source_file(self, sid: str, path: str, store) -> tuple[str, bytes]:
        """(version, content) of one final file of the source."""
        from . import manifest
        from .sources import source_tree

        version, source = self.source_version(sid), self.source(sid)
        tree, _ = source_tree(source, store)
        if path not in tree:
            raise NotFound(f"{sid} has no file {path}")
        return version, manifest.content(store, tree[path])

    @_locked
    def change_source_files(self, sid: str, changes: dict[str, bytes | None], patch: str, version: str, store,
                            formats=(), tags=()) -> str:
        """Change, add (bytes) or remove (None) final files of the source, as
        `wpt-dataset unpack`, an edit and `wpt-dataset patch` would: the
        difference becomes the next patch, NNNN-<patch>.patch, kept only if
        the source then gives exactly those files and passes the rules of
        `wpt-dataset check --fetch`. formats: format entries source.json gets
        first, for new files no entry covers yet; tags: tag entries it gets
        first, the same way. In an empty source, a new file no license entry
        covers yet gets its pattern (license_check.pattern_for: its top
        folder, its kind at the top, or its path) added to the source's one
        license entry (RuleViolation when it has several: a person says which
        covers it). All are undone when the patch is refused.
        version is the source_version the caller read.
        A path that is a symbolic link, or lies under one, is refused
        (RuleViolation), except for removing the link itself: a write never
        follows a link. Returns the new one."""
        from . import edit

        if self.source_version(sid) != version:
            raise StaleFile(f"{self.relative('source', sid).rsplit('/', 1)[0]}: changed on disk since it was read")
        source_path = self._path("source", sid)
        before = source_path.read_bytes()
        source = self.source(sid)
        added = [path for path, content in changes.items() if content is not None]
        covered = sorted({pattern_for(path) for path in self.uncovered(sid, added, source)}) \
            if source.data["type"] == "empty" else []
        try:
            if formats or tags or covered:
                document = json.loads(before)
                licenses = document["licenses"]
                if covered:
                    if len(licenses) != 1:
                        raise RuleViolation(f"{sid}: {', '.join(covered)}: no license entry covers it, and the "
                                            f"source has {len(licenses)}; add it to the one that covers it")
                    licenses = [{**licenses[0], "paths": licenses[0]["paths"] + list(covered)}]
                self.save("source", sid, {**document, "formats": document["formats"] + list(formats),
                                          "licenses": licenses,
                                          **({"tags": document.get("tags", []) + list(tags)} if tags else {})},
                          _sha256(before))
            with tempfile.TemporaryDirectory(prefix="wpt-change-") as tmp:
                folder = Path(tmp) / sid
                edit.unpack(self, self.source(sid), store, folder)
                top = Path(os.path.realpath(folder))
                for path, content in changes.items():
                    target = _landing(top, folder.joinpath(*_parts(path)), sid, path, removing=content is None)
                    if content is None:
                        if not target.is_file() and not target.is_symlink():
                            raise NotFound(f"{sid} has no file {path}")
                        target.unlink()
                        _remove_empty_parents(target.parent, top)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                edit.write_patch(self, self.source(sid), store, folder, patch)
        except BaseException:
            if source_path.read_bytes() != before:  # the lock is held: only this call changed it
                self._write_data(source_path, before)
            raise
        return self.source_version(sid)

    @_locked
    def add_patch(self, sid: str, name: str, data: bytes, expected: dict, store) -> str:
        """Write data as the source's next patch, NNNN-<name>.patch, and keep
        it only if the source then gives exactly the expected tree ({path:
        Entry}) and passes the rules of `wpt-dataset check --fetch`;
        otherwise remove it and raise why. Called by edit.write_patch, which
        makes data from a folder. Returns the patch's place in the
        repository."""
        from .errors import PatchFailed
        from .sources import source_tree

        folder = self._data / "sources" / sid
        number = len([p for p in folder.iterdir() if p.suffix == ".patch"]) + 1
        path = folder / f"{number:04d}-{name}.patch"
        if not PATCH.fullmatch(path.name):
            from .errors import EditError
            raise EditError(f"patch name {name!r}: use letters, digits, '.', '_' and '-'")
        place = path.relative_to(self.root).as_posix()
        path = self._target(path)
        path.write_bytes(data)
        try:
            result, _ = source_tree(self.source(sid), store)  # applies every patch and checks the license map
            if result != expected:
                raise PatchFailed(f"{sid}: {path.name} does not give the folder's files")
            problems = self.check_source(sid)
            if problems:
                raise RuleViolation(problems[0])
            self.check_fetched(sid, store)
        except BaseException:
            path.unlink()
            raise
        return place

    @_locked
    def create_source(self, origin: str, item: str | None, document: dict, files: dict[str, bytes], patch: str,
                      store) -> str:
        """A new source: its source.json (after the source rules) and, when
        files are given, its first patch holding them (as
        change_source_files). A license entry whose `text` is "" is left for
        a person to write, which the schema allows while the decision is
        refused: the source is made without files (readers leave it out with
        a note, Repo.sources). Nothing
        stays when anything is refused or fails: the folder is removed, and
        the error raised is the one that stopped the creation (a removal that
        fails too is added to it as a note). Returns its id."""
        sid = source_id(origin, item, document.get("address", "") if isinstance(document, dict) else "")
        if sid in self.source_ids():
            raise RuleViolation(f"{sid} exists already")
        licenses = document.get("licenses") if isinstance(document, dict) else None
        later = [entry for entry in licenses if isinstance(entry, dict) and entry.get("text") == ""] \
            if isinstance(licenses, list) else []
        if later and files:
            raise RuleViolation(f"{sid}: files need the license text first; make the source without them, have a "
                                f"person write the text, then add the files")
        folder = self._data / "sources" / sid
        try:
            self.save("source", sid, document, "")
            if files:
                self.change_source_files(sid, files, patch, self.source_version(sid), store)
        except BaseException as error:
            try:
                if os.path.lexists(folder):
                    shutil.rmtree(self._target(folder))
            except Exception as cleanup:  # the error that stopped the creation stays the one raised
                error.add_note(f"{sid}: its half-made folder could not be removed ({type(cleanup).__name__}: "
                               f"{cleanup}); remove {folder} by hand")
            raise
        return sid

    @_locked
    def set_tags(self, sid: str, path: str, tags: list[str], version: str) -> str:
        """Give exactly this path (not a pattern) its own tags entry in
        source.json, replacing any entry that names only this path; no tags
        removes it. Tags other entries give the path through their patterns
        stay. version: the version of source.json the caller read."""
        document = json.loads(self.read("source", sid).text)
        kept = [entry for entry in document.get("tags", []) if entry["paths"] != [path]]
        if tags:
            kept.append({"tags": list(tags), "paths": [path]})
        changed = {key: value for key, value in document.items() if key != "tags"}
        if kept:
            changed["tags"] = kept
        return self.save("source", sid, changed, version)

    # -- conversations: lines of a chat file ----------------------------------------
    def conversations(self, sid: str, path: str, store) -> tuple[str, list[dict]]:
        """(version, conversations) of a chat file of the source, each in the
        current shape; an absent file has none."""
        from .conversations import parse

        return self._lines(sid, path, "chat", parse, store)

    @_locked
    def save_conversation(self, sid: str, path: str, line: int | None, conversation: dict | None, patch: str,
                          version: str, store) -> str:
        """Replace the conversation on line `line` (from 1), add one at the
        end (line None), or remove one (conversation None), through a patch
        (change_source_files), in a chat file of an `empty` source (text
        written for WPT). Each model draft's `edited` is set here, from
        whether its message's content still gives the drafted sha256. A
        replaced line keeps its layout where the meaning did not change; a
        new chat file gets its own `chat` format entry when none covers it.
        The whole file must pass the chat rules (docs/chat-format.md), model
        drafts included. Returns the new version."""
        from .conversations import mark_drafts, parse

        return self._save_line(sid, path, line, None if conversation is None else mark_drafts(conversation), "chat",
                               parse, patch, version, store)

    # -- ratings: lines of a rating file ----------------------------------------------
    def ratings(self, sid: str, path: str, store) -> tuple[str, list[dict]]:
        """(version, ratings) of a rating file of the source (people's pass
        or fail on measure attempts; ratings.py); an absent file has none."""
        from .ratings import parse

        return self._lines(sid, path, "rating", parse, store)

    @_locked
    def save_rating(self, sid: str, path: str, line: int | None, rating: dict | None, patch: str, version: str,
                    store) -> str:
        """Replace the rating on line `line` (from 1), add one (line None),
        or remove one (rating None), through a patch (change_source_files),
        in a rating file of an `empty` source, as save_conversation does with
        conversations. A rating added by a person who rated the same attempt
        already replaces their earlier line (ratings.attempt_of); any other goes at the end. A new
        rating file gets its own `rating` format entry when none covers it.
        The whole file must pass the rating rules (ratings.py). Returns the
        new version."""
        from .ratings import attempt_of, parse

        return self._save_line(sid, path, line, rating, "rating", parse, patch, version, store, same=attempt_of)

    def _lines(self, sid, path, format, parse, store) -> tuple[str, list[dict]]:
        """(version, parse's documents) of a file of the source in format
        (chat, rating), one document per line; an absent file has none."""
        self._editable(sid, path, format)
        try:
            version, data = self.read_source_file(sid, path, store)
        except NotFound:
            return self.source_version(sid), []
        return version, parse(sid, path, data)

    def _save_line(self, sid, path, line, document, format, parse, patch, version, store, same=None) -> str:
        """Replace, add or remove one line of a file in format (chat,
        rating) of an `empty` source, as the next patch, keeping a replaced
        line's layout where its meaning did not change; the whole file must
        pass parse. same, when given, says what a document is the same as:
        a document added (line None) replaces the line whose document gives
        the same same(). A new file gets its own format entry when none
        covers it. The caller holds the lock (_locked)."""
        self._written_for_wpt(sid)
        self._editable(sid, path, format)
        try:
            _, data = self.read_source_file(sid, path, store)
            lines = data.decode("utf-8").splitlines()  # as parse numbers them
        except NotFound:
            lines = []
        except UnicodeDecodeError:
            raise InvalidFile(f"{sid}/{path}: not UTF-8 text") from None
        if line is not None and not 1 <= line <= len(lines):
            raise NotFound(f"{sid}/{path} has no line {line}")
        if line is None and document is not None and same is not None and lines:
            try:
                key = same(document)
            except (KeyError, TypeError):  # a document that is not whole: parse names what it lacks
                key = None
            found = [number for number, earlier in enumerate(parse(sid, path, data), start=1)
                     if key is not None and same(earlier) == key]
            line = found[0] if found else None
        if document is None:
            if line is None:
                raise RuleViolation("give the line to remove")
            del lines[line - 1]
        elif line is None:
            lines.append(json_text(document))
        else:
            lines[line - 1] = keep_layout(lines[line - 1], document, json_text)
        content = ("\n".join(lines) + "\n").encode("utf-8") if lines else None
        if content is not None:
            parse(sid, path, content)
        formats = [] if self._format_entry(sid, path) else [{"format": format, "paths": [path]}]
        return self.change_source_files(sid, {path: content}, patch, version, store, formats)

    def uncovered(self, sid, paths, source: Source | None = None) -> list[str]:
        """Those of paths no license entry of the source covers."""
        coverage = Coverage((source or self.source(sid)).data["licenses"])
        return [path for path in paths if not coverage.all.matching(path)]

    def _written_for_wpt(self, sid, source: Source | None = None):
        """RuleViolation unless the source (read here when not given) is an
        `empty` one: conversations, tasks and ratings are text written for
        WPT (CONTRIBUTING.md, "Adding your own texts"), never a change to
        upstream's files."""
        kind = (source or self.source(sid)).data["type"]
        if kind != "empty":
            raise RuleViolation(f"{sid} is a {kind} source: conversations, tasks and ratings are written in an "
                                f"empty source")

    def _editable(self, sid, path, format):
        entry = self._format_entry(sid, path)
        if entry is not None and entry["format"] != format:
            raise RuleViolation(f"{sid}/{path} is in the {entry['format']} format, not {format}")
        if entry is not None and "tree" in entry:
            raise RuleViolation(f"{sid}/{path} is read as a message tree (upstream's records): not editable here")

    def _format_entry(self, sid, path, source: Source | None = None) -> dict | None:
        """The one format entry of the source (read here when not given)
        that covers path, None when none does; FormatMapError when several
        do."""
        from .errors import FormatMapError
        from .markdown import format_of

        try:
            return format_of(source or self.source(sid), path)
        except FormatMapError as error:
            if "no format entry" in str(error):
                return None
            raise

    # -- tasks: task folders of task files ---------------------------------------------
    def _task_folders(self, sid: str, store, source: Source | None = None) \
            -> tuple[str, dict[str, dict[str, tuple[str, bytes]]]]:
        """(version, {task folder: {path in the folder: (manifest mode,
        content)}}) of the source's files in the task format; the source is
        read and checked here when not given, and only once."""
        from . import manifest
        from .sources import source_tree
        from .tasks import folders

        version, source = self.source_version(sid), source or self.source(sid)
        tree, _ = source_tree(source, store)
        paths = [p for p in tree if (self._format_entry(sid, p, source) or {}).get("format") == "task"]
        found = {}
        for folder, owned in sorted(folders(paths).items()):
            prefix = folder + "/" if folder else ""
            found[folder] = {p.removeprefix(prefix): (tree[p].mode, manifest.content(store, tree[p]))
                             for p in sorted(owned)}
        return version, found

    def tasks(self, sid: str, store) -> tuple[str, dict[str, dict]]:
        """(version, {task folder: {"task": task.json's JSON (its text when
        it does not parse), "check": the check's text, "files": {path under
        files/: text}, "kept": {path in the folder: why}}}) of the source's
        files in the task format. Every file is UTF-8 text (source_tree
        refuses a source with another; InvalidFile here too), except a
        symbolic link under files/, which is not given as text: it is listed
        in "kept", and save_task leaves it as it is. The folder "" is a task
        at the source's root, which save_task refuses to write. A task.json
        of an earlier shape is given in the current one (migrate.upgrade),
        so a save writes that shape."""
        from .migrate import upgrade
        from .tasks import CHECK, FILES, TASK

        version, folders = self._task_folders(sid, store)
        found = {}
        for folder, owned in folders.items():
            text, kept = {}, _kept(owned)
            for path, (mode, data) in owned.items():
                if path in kept:
                    continue
                try:
                    text[path] = data.decode("utf-8")
                except UnicodeDecodeError:
                    raise InvalidFile(f"{sid}/{folder}/{path}: not UTF-8 text") from None
            task = text.get(TASK, "")
            try:
                task = upgrade(json.loads(task), "task", f"{sid}/{folder}/{TASK}")
            except json.JSONDecodeError:
                pass
            found[folder] = {"task": task, "check": text.get(CHECK, ""),
                             "files": {p.removeprefix(FILES): t for p, t in text.items() if p.startswith(FILES)},
                             "kept": kept}
        return version, found

    @_locked
    def save_task(self, sid: str, folder: str, task: dict | None, check: str, files: dict[str, str], patch: str,
                  version: str, store) -> str:
        """Write a task folder (docs/training.md, "Tasks") of an `empty`
        source as exactly task.json (the task), check and files/<path>
        (files), removing its other files; task None removes the folder.
        Through a patch (change_source_files) that holds only the files
        whose bytes change: a file keeps its mode (its execute bit), and
        task.json its layout where the meaning did not change. A symbolic
        link (tasks() lists it in "kept") stays as it is, and a save that
        gives it as text is refused. A task
        at the source's root (folder "") is refused. A new folder gets a
        `task` format entry (<folder>/**) when none covers it. Returns the
        new version."""
        from .tasks import CHECK, FILES, TASK

        if not folder or any(part in ("", ".", "..") for part in folder.split("/")) or "\\" in folder:
            raise RuleViolation(f"task folder {folder!r}: a relative path below the source's root, such as "
                                f"tasks/fix-calc (a task at the root is changed with unpack and patch)")
        source = self.source(sid)  # read and checked once for the whole save
        self._written_for_wpt(sid, source)
        _, folders = self._task_folders(sid, store, source)
        old = folders.get(folder, {})
        if task is None and not old:
            raise NotFound(f"{sid} has no task folder {folder}")
        prefix = folder + "/"
        kept = _kept(old)
        new = {}
        if task is not None:
            validate(task, "task", f"{sid}/{prefix}{TASK}")
            try:
                self._image_folder(task["image"])
            except RuleViolation as error:
                raise InvalidFile(f"{sid}/{prefix}{TASK}: {error}") from None
            old_task = old.get(TASK, (None, None))[1]
            new[TASK] = keep_layout(old_task.decode("utf-8") if old_task is not None else None, task).encode("utf-8")
            new[CHECK] = check.replace("\r\n", "\n").encode("utf-8")
            for path, text in files.items():
                _parts(path)
                new[FILES + path] = text.encode("utf-8")
            given = sorted(p for p in set(new) & set(kept) if new[p] != old[p][1])
            if given:
                raise RuleViolation(f"{sid}/{prefix}{given[0]} is {kept[given[0]]}: it is not written as text here "
                                    f"(change it with unpack and patch)")
            new.update({path: old[path][1] for path in kept})
        changes = {prefix + path: None for path in set(old) - set(new)}
        changes |= {prefix + path: data for path, data in new.items() if path not in old or old[path][1] != data}
        if not changes:  # nothing to record; a stale save is still refused
            if self.source_version(sid) != version:
                raise StaleFile(f"{self.relative('source', sid).rsplit('/', 1)[0]}: changed on disk since it was read")
            return version
        formats = [] if task is None or self._format_entry(sid, prefix + TASK, source) else \
            [{"format": "task", "paths": [prefix + "**"]}]
        return self.change_source_files(sid, changes, patch, version, store, formats)

    # -- git -----------------------------------------------------------------------
    def _git(self, *args, check=True) -> subprocess.CompletedProcess:
        return subprocess.run([program("git"), "-C", str(self.root), *args], capture_output=True, check=check)

    def _ignored(self, paths) -> set[str]:
        """Those of the paths (relative to the root) that git ignores and does
        not track; none when the repository is not a git working tree, or git
        is not installed, so nothing tells what it ignores."""
        git = shutil.which("git")
        if not paths or git is None:
            return set()
        result = subprocess.run([git, "-C", str(self.root), "check-ignore", "-z", "--stdin"],
                                input="\0".join(paths).encode() + b"\0", capture_output=True, check=False)
        if result.returncode not in (0, 1):  # 1: none ignored; otherwise not a git working tree
            return set()
        return {path.decode() for path in result.stdout.split(b"\0") if path}

    def status(self) -> list[dict] | None:
        """[{path, state}] of every file under data/ that git sees changed or
        new (state: git's two status letters, such as " M" or "??"); None
        when the repository is not a git working tree."""
        result = self._git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "data", check=False)
        if result.returncode:
            return None
        entries, found = result.stdout.split(b"\0"), []
        index = 0
        while index < len(entries):
            entry = entries[index].decode("utf-8", errors="replace")
            index += 1
            if not entry:
                continue
            found.append({"path": entry[3:], "state": entry[:2]})
            if entry[0] in "RC":
                index += 1  # the path it was renamed or copied from
        return found

    def changed_since(self, revision: str) -> tuple[list[str], bool, list[str]]:
        """(ids of the existing sources whose folder changed since the git
        revision, whether anything under data/images/ changed, ids of the
        sources whose folder is gone)."""
        out = self._git("diff", "--name-only", revision, "--", "data/sources", "data/images").stdout.decode()
        changed = out.splitlines()
        ids = {p.split("/")[2] for p in changed if p.startswith("data/sources/") and p.count("/") >= 3}
        existing = set(self.source_ids())
        return sorted(ids & existing), any(p.startswith("data/images/") for p in changed), sorted(ids - existing)

    def _show(self, commit: str, path: str) -> bytes | None:
        result = self._git("show", f"{commit}:{path}", check=False)
        return result.stdout if result.returncode == 0 else None

    def committed_values(self, kind: str, names, key: str) -> dict[str, object]:
        """{name: the value of key in the newest version of that file, in
        the git history of this repository's HEAD, whose JSON object holds
        key} for each of the names of files of kind that has one: what a
        migration step reads where a field an earlier shape held is needed
        again. {} when the repository is not a git working tree."""
        places = {self.relative(kind, name): name for name in names}
        if not places:
            return {}
        top = posixpath.commonpath([posixpath.dirname(place) for place in places])
        log = self._git("-c", "core.quotePath=false", "log", "--format=%x00%H", "--name-only", "--no-renames",
                        "--", top, check=False)
        if log.returncode:
            return {}
        commits: dict[str, list[str]] = {}  # place -> the commits that changed it, newest first
        for chunk in log.stdout.decode("utf-8", errors="replace").split("\0")[1:]:
            commit, *changed = chunk.splitlines()
            for place in changed:
                if place in places:
                    commits.setdefault(place, []).append(commit)
        found, depth = {}, 0
        while asked := [(place, versions[depth]) for place, versions in commits.items()
                        if place not in found and depth < len(versions)]:
            for (place, _), document in zip(asked, self._committed_json(asked)):
                if isinstance(document, dict) and key in document:
                    found[place] = document[key]
            depth += 1
        return {places[place]: value for place, value in found.items()}

    def _committed_json(self, asked: list[tuple[str, str]]) -> list[object]:
        """The JSON of each (place, commit) as that commit holds it, in one
        git process; None for one the commit does not hold or that is not
        JSON."""
        wanted = "".join(f"{commit}:{place}\n" for place, commit in asked).encode()
        out = subprocess.run([program("git"), "-C", str(self.root), "cat-file", "--batch"], input=wanted,
                             capture_output=True, check=True).stdout
        documents, at = [], 0
        for _ in asked:
            end = out.index(b"\n", at)
            header, at = out[at:end].split(), end + 1
            if len(header) != 3:  # "<name> missing", or ambiguous: the commit does not hold it
                documents.append(None)
                continue
            size = int(header[2])
            body, at = out[at:at + size], at + size + 1
            try:
                documents.append(json.loads(body))
            except (UnicodeDecodeError, json.JSONDecodeError):
                documents.append(None)
        return documents

    def signer_lists_at(self, commit: str) -> tuple[bytes | None, bytes | None]:
        """(data/allowed-signers, data/allowed-gpg-keys.asc) as they stood at
        a commit; None for a file the commit does not hold."""
        return (self._show(commit, f"data/{KINDS['allowed-signers'].single}"),
                self._show(commit, f"data/{KINDS['allowed-gpg-keys'].single}"))

    def signature_reasons(self, base: str | None, commit: str, changed: list[str]) -> list[str]:
        """Why a commit that changes these paths (from base, None for a first
        commit) must be signed by an allowed key: the lists of keys it
        changes, each source.json whose source decision it changes, each
        whose decided license entries it changes, and each license text
        whose decision it changes (all three levels of decision are
        signed). A decided (admitted or refused) entry or license text is
        compared with what its decision holds (ENTRY_SIGNED, TEXT_SIGNED),
        whatever its position: a change to it, or a decision that moves to
        other content, needs a signature as a changed decision does."""
        lists = [f"data/{KINDS[kind].single}" for kind in ("allowed-signers", "allowed-gpg-keys")]
        reasons = [p for p in changed if p in lists]
        for path in changed:
            if path.startswith("data/sources/") and path.endswith(f"/{SOURCE_FILE}"):
                before, after = self._decisions_at(base, path), self._decisions_at(commit, path)
                if before[0] != after[0]:
                    reasons.append(f"decision in {path}")
                if before[1] != after[1]:
                    reasons.append(f"decided license entries in {path}")
            elif path.startswith("data/license-texts/") and path.endswith(f"/{TEXT_FILE}"):
                if self._decisions_at(base, path) != self._decisions_at(commit, path):
                    reasons.append(f"license text decision in {path}")
        return reasons

    def _decisions_at(self, commit, path):
        """(the decision, what decided items hold) of a source.json or a
        license-text.json at a commit: for a source, each decided license
        entry as what its decision holds (signed_entry), in a fixed order;
        for a license text, what its decision holds (TEXT_SIGNED) when it is
        decided, else None. (None, None) when the commit does not hold it,
        and "unreadable" for what cannot be read."""
        if commit is None:
            return None, None
        data = self._show(commit, path)
        if data is None:
            return None, None
        try:
            document = json.loads(data)
            decision = document.get("decision")
            if path.endswith(f"/{TEXT_FILE}"):
                return decision, [document.get(key) for key in TEXT_SIGNED] if _decided(decision) else None
            licenses = document.get("licenses")
            entries = sorted(json_text(signed_entry(entry)) for entry in licenses
                             if _decided(entry.get("decision"))) if isinstance(licenses, list) else None
            return decision, entries
        except (json.JSONDecodeError, AttributeError):
            return "unreadable", "unreadable"

    # -- migrations ------------------------------------------------------------------
    def versioned(self) -> list[tuple[str, str]]:
        """(kind, name) of every existing JSON file that carries its shape:
        sources, recipes, locks of existing tokenizer recipes, image files, the
        license texts."""
        found = [("source", sid) for sid in self.source_ids()]
        found += [(kind, name) for kind in RECIPE_FOLDERS for name in self.names(kind)]
        found += [("lock", name) for name in self.names("tokenizer") if self.exists("lock", name)]
        found += [(kind, name) for name in self.image_names()
                  for kind in ("image-lock", "image-licenses", "image-tools") if self.exists(kind, name)]
        found += [("license-text", text) for text in self.license_text_ids()]
        return found

    def raw(self, kind: str, name: str):
        """The file's JSON, whatever its shape; InvalidFile when it does not parse."""
        path = self._path(kind, name)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InvalidFile(f"{path}: {error}") from error

    def located(self, kind: str, name: str) -> Located:
        return Located(self, self._path(kind, name))

    @_locked
    def rewrite(self, kind: str, name: str, document) -> str:
        """Write a migrated file as its migration left it: `wpt-dataset
        migrate` alone calls this. A step leaves a field only a person can
        fill empty, so the result may not pass the check yet; migrate then
        validates every file and names those to fill in. Returns its place
        in the repository."""
        path = self._path(kind, name)
        self._write_data(path, self._json_text(path, document).encode("utf-8"))
        return self.relative(kind, name)

    def validate_file(self, kind: str, name: str):
        load_json(self._path(kind, name), kind)


def _parts(path: str) -> list[str]:
    """The parts of a relative path of a source's file; RuleViolation for one
    that leaves its folder."""
    parts = path.split("/")
    if not path or posixpath.isabs(path) or "\\" in path or any(p in ("", ".", "..") for p in parts):
        raise RuleViolation(f"{path!r}: a file of a source is a relative path with '/' between folders")
    return parts


def _landing(top: Path, target: Path, sid: str, path: str, removing: bool) -> Path:
    """Where a change to target (a source's file, in the unpacked folder
    whose resolved place is top) lands: the target of a symbolic link on the
    way (every link resolved), or, with removing, target itself when it is
    the link (a link is removed, never followed). RuleViolation when that
    place is outside the source: nothing is written there."""
    landing = Path(os.path.realpath(target.parent)) / target.name if removing else Path(os.path.realpath(target))
    if not landing.is_relative_to(top):
        raise RuleViolation(f"{sid}/{path} leads to {landing}, outside the source (through a symbolic link): a "
                            f"change writes a link's target only inside the source")
    return landing


def _kept(owned: dict[str, tuple[str, bytes]]) -> dict[str, str]:
    """{path: why} of a task folder's files that are not given as text: its
    symbolic links."""
    from .manifest import SYMLINK

    return {path: "a symbolic link" for path, (mode, _) in owned.items() if mode == SYMLINK}


def _remove_empty_parents(folder: Path, top: Path):
    while folder != top and folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def _entries_of(document) -> list | None:
    licenses = document.get("licenses") if isinstance(document, dict) else None
    return licenses if isinstance(licenses, list) and all(isinstance(e, dict) for e in licenses) else None


def _decided(decision) -> bool:
    """Whether a decision admits or refuses (anything but under-review)."""
    return not (isinstance(decision, dict) and decision.get("decision") == "under-review")


def signed_entry(entry: dict) -> dict:
    """What a license entry's decision holds: ENTRY_SIGNED, its text when its
    place is the entry itself, and the decision."""
    keys = ENTRY_SIGNED + (("text",) if entry.get("place") == {"entry": True} else ())
    return {**{key: entry.get(key) for key in keys}, "decision": entry.get("decision")}


def sent_back(old, new, today: str | None = None):
    """new, a source.json about to replace old, with each license entry the
    save changes sent back to under-review: an entry that old does not hold
    with the same license, place and expected texts (DECIDED_ON) and the
    same decision, and whose decision is the one old held at its position
    (the save does not decide it anew). Its decision becomes under-review
    by SERVICE, with today's date and why. An entry already under review,
    an entry old holds as it is (at any position), and an entry the save
    decides anew (its decision is not old's at its position) keep theirs;
    the signature of the commit that holds a decision is checked there
    (commits.py). What is not a source with license entries is new as it
    came: the check says what is wrong with it."""
    before, after = _entries_of(old), _entries_of(new)
    if before is None or after is None:
        return new
    held = [([entry.get(key) for key in DECIDED_ON], entry.get("decision")) for entry in before]
    entries = []
    for number, entry in enumerate(after):
        decision = entry.get("decision")
        unchanged = ([entry.get(key) for key in DECIDED_ON], decision) in held
        decided_anew = number >= len(before) or before[number].get("decision") != decision
        if unchanged or decided_anew or not isinstance(decision, dict) or decision.get("decision") == "under-review":
            entries.append(entry)
            continue
        entries.append({**entry, "decision": {"decision": "under-review", "by": SERVICE,
                                              "date": today or date.today().isoformat(),
                                              "reason": "its license, place or expected texts changed"}})
    return {**new, "licenses": entries}


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
