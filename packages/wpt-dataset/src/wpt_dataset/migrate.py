"""Move data files to their current shape, so no file is edited by hand.

No file carries a shape number. A step recognizes the old shape by its
fields: it says whether a document has the old shape (`applies`) and
rewrites it (`rewrite`). A document that matches the current schema needs
no step. To change a shape: update the schema file and register a step
here, in the list of its kind:

    STEPS["source"].append(Step("licenses without a credit",
                                lambda doc: any("attribution" not in entry for entry in doc["licenses"]),
                                lambda doc, where: {...}))

The change must leave a field difference for `applies` to find. A step gets
where the file lives too (Where: its name for messages, the text of a file
next to it, for the rare step that reads one, and for a source the file
list and license files of its upstream and the license texts, which the
step that gave license entries their place and decision reads and adds
to).

`wpt-dataset migrate` then rewrites every file a step applies to, and
checks the result against the schema. A file migrate cannot rewrite, such as
a source's chat file (upstream's bytes), is brought to the current shape in
memory by upgrade() each time it is read.

Before any step, migrate turns the reviews of data/reviewed-license-texts.json,
the earlier shape of the license texts, into license-text items
(_convert_reviews), so the sources' step finds them.
"""

import hashlib
import itertools
import posixpath
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Callable, NamedTuple

from . import spdx, upstream_files
from .errors import DatasetError, OutdatedSchema, RuleViolation
from .license_check import Coverage, LicenseTexts, check_pattern, text_id
from .licenses import check_license
from .manifest import HISTORY, PatternIndex, glob_match
from .repo import PUBLIC_DOMAIN, SERVICE, Repo, check_person
from .protocols.base import USER_AGENT, named
from .protocols.http import is_archive, name_of
from .sources import exact_pattern

KINDS_WITH_STEPS = ("source", "mixture", "model", "tokenizer", "lock", "conversation", "task", "rating",
                    "image-lock", "image-licenses", "image-tools", "license-text")


class Step(NamedTuple):
    """One rewrite. what: the old shape, as a sentence part ("licenses
    without a credit"). applies(document): whether the document has it.
    rewrite(document, where): the document in the shape after it. after(where,
    document): what a rewritten file leaves to do beside it (optional)."""
    what: str
    applies: Callable
    rewrite: Callable
    after: Callable | None = None


STEPS: dict[str, list[Step]] = {kind: [] for kind in KINDS_WITH_STEPS}


class MigrationIncomplete(DatasetError):
    """Some files could not be rewritten; each stays in its earlier shape,
    and the message names each with why. Every other file was rewritten."""


class Texts:
    """The license texts as one migration finds and adds them: known (a
    license_check.LicenseTexts of every item, those it adds included), by
    (the person whose signed migration commit carries the decisions it
    writes, or None), standard(content) (the SPDX id a license file is the
    standard text of, spdx.Matcher, or None), held_back (the sha256s of the
    exact files whose earlier review no item can hold), and add (an item
    written through the service layer, then known)."""

    def __init__(self, repo: Repo, by: str | None, matcher=None, held_back=frozenset()):
        self._repo, self.by, self._matcher, self._standard = repo, by, matcher, None
        self.known, self.held_back = LicenseTexts(repo.license_texts()), set(held_back)

    def standard(self, content: bytes) -> str | None:
        if self._matcher is None:
            return None
        if self._standard is None:
            try:
                self._standard = self._matcher()
            except OSError as error:
                raise named(error, f"the standard license texts of SPDX release {spdx.RELEASE}") from error
        return self._standard.identify(content.decode("utf-8", "replace").removeprefix("\ufeff"))

    def add(self, text: str, item: dict, itself: bytes):
        self._repo.add_license_text(text, item, itself)
        self.known.add(text, item)


class Where:
    """Where a file being migrated lives, for a migration step: str() names
    it in messages, name is its name (a source's id), sibling_text and
    remove_sibling read or remove a file beside it (repo.Located), files
    (for a source) gives its upstream's file list and bytes
    (upstream_files.files: the caller of migrate gives upstream), and
    texts the license texts (Texts)."""

    def __init__(self, repo: Repo, kind: str, name: str, upstream=None, texts: Texts | None = None):
        self._repo, self._located, self.name, self._upstream = repo, repo.located(kind, name), name, upstream
        self._texts = texts

    def __str__(self):
        return str(self._located)

    def sibling_text(self, name: str) -> str | None:
        return self._located.sibling_text(name)

    def remove_sibling(self, name: str):
        self._located.remove_sibling(name)

    def files(self, document):
        """The upstream_files.Files of this source, as described by document."""
        if self._upstream is None:
            raise OutdatedSchema("this step reads the source's upstream files: run `wpt-dataset migrate`, which "
                                 "reads them from the store or the network")
        return self._upstream(self.name, document)

    def texts(self) -> Texts:
        if self._texts is None:
            self._texts = Texts(self._repo, None)
        return self._texts


def _entries(document, key) -> list[dict]:
    """The dict items of the list a document holds under key ([] when it holds none)."""
    found = document.get(key) if isinstance(document, dict) else None
    return [entry for entry in found if isinstance(entry, dict)] if isinstance(found, list) else []


def _source_attribution(document, _where):
    """A license entry gets the credit it asks for. Only a person can write
    it, so the step leaves it empty and the check that follows names each
    file to fill in."""
    return {**document, "licenses": [{**entry, "attribution": entry.get("attribution", "")}
                                     for entry in document["licenses"]]}


def _source_formats(document, _where):
    """How each file becomes Markdown for training (`formats`). Only a person
    knows each file's format, so the step leaves the list empty and the
    check that follows names each file to fill in."""
    return {**document, "formats": []}


def _license_file(entry, where) -> tuple[str, str] | None:
    """(name, text) of the LicenseRef-<name>.txt next to source.json that an
    earlier shape kept for a LicenseRef license, or None."""
    for alternative in entry["license"].split(" OR "):
        name = f"{alternative.partition(' WITH ')[0]}.txt"
        text = where.sibling_text(name) if alternative.startswith("LicenseRef-") else None
        if text is not None:
            return name, text
    return None


def _source_text(document, where):
    """Every license's text is in its entry (`text`), so that it travels into
    each dataset record and model NOTICE. A LicenseRef text moves in from its
    file; any other text is only on the source's side, so the step leaves it
    empty and the check that follows names each file to fill in."""
    licenses = []
    for entry in document["licenses"]:
        found = _license_file(entry, where)
        licenses.append({**entry, "text": entry["text"] if "text" in entry else found[1] if found else ""})
    return {**document, "licenses": licenses}


def _drop_moved_license_files(where, document):
    """Remove the LicenseRef files whose text the written source.json now
    holds exactly; anything else stays for the check to report."""
    for entry in document["licenses"]:
        found = _license_file(entry, where)
        if found and found[1] == entry["text"]:
            where.remove_sibling(found[0])


def _old_decision(document) -> bool:
    decision = document.get("decision")
    return isinstance(decision, dict) and (decision.get("decision") == "retired" or (
        decision.get("decision") == "refused" and decision.get("reason") == "under review"))


def _source_decision(document, _where):
    """A source has exactly three states, "under-review", "admitted" and
    "refused": a refusal whose reason was "under review" becomes
    "under-review" without that reason, and "retired" becomes "refused",
    keeping its reason (which names the source that replaced it)."""
    decision = document["decision"]
    if decision["decision"] == "refused" and decision.get("reason") == "under review":
        decision = {key: value for key, value in decision.items() if key != "reason"} | {"decision": "under-review"}
    elif decision["decision"] == "retired":
        decision = {**decision, "decision": "refused"}
    return {**document, "decision": decision}


# the tags that named a training step, and the tags that name the step doing the same
TAGS_OLD = {"sft": "answer-as-assistant", "grpo": "finish-tasks", "token-efficiency": "finish-tasks-with-less-writing"}


def _old_tags(document) -> bool:
    return any(isinstance(tag, str) and tag in TAGS_OLD for entry in _entries(document, "tags")
               for tag in (entry.get("tags") if isinstance(entry.get("tags"), list) else ()))


def _source_tags(document, _where):
    """A tag that picks a training step's examples is the step's name: sft
    becomes answer-as-assistant, grpo finish-tasks and token-efficiency
    finish-tasks-with-less-writing (TAGS_OLD), in every tag list; a tag
    listed twice by the rename stays once."""
    return {**document, "tags": [{**entry, "tags": list(dict.fromkeys(TAGS_OLD.get(tag, tag) for tag in entry["tags"]))}
                                 for entry in document["tags"]]}


STEPS["source"] += [
    Step("a license entry without its credit", lambda d: any("attribution" not in e for e in _entries(d, "licenses")),
         _source_attribution),
    Step("no formats", lambda d: "formats" not in d, _source_formats),
    Step("a license entry without its text", lambda d: any("text" not in e for e in _entries(d, "licenses")),
         _source_text, _drop_moved_license_files),
    Step("a decision of an earlier kind", _old_decision, _source_decision),
    Step("a tag that names a step by its old name", _old_tags, _source_tags),
]

# a file whose name says it states a license: what the migration reads to find an entry's place
LICENSE_FILE = re.compile(r"(un)?licen[cs]e|copying|copyright|notice|legal", re.IGNORECASE)
# a license text's kind that states a license (`license`: its text; `notice`: its name without its text)
STATES_LICENSE = ("license", "notice")


def _pinned_shape(document) -> bool:
    """A source of the shape that pinned its upstream (`checksum`,
    `revision`) and gave decisions and places to no license entry."""
    return any(key in document for key in ("checksum", "revision")) or \
        any("decision" not in entry or "place" not in entry or "values" in entry
            for entry in _entries(document, "licenses")) or \
        any("license" in entry for entry in _entries(document, "formats") if entry.get("format") == "records")


def _shapes(paths: list[str], taken: set[str], prefix: str = "") -> list[str]:
    """Patterns covering exactly the areas of paths (all inside prefix), in
    the allowed shapes (license_check.check_pattern) and covering none of
    taken: a folder that holds none of taken as `<folder>/**`, files of one
    extension (two or more, the pattern matching none of taken) as
    `<prefix>*.<extension>`,
    any other file by its exact path; a folder holding a taken path is
    looked into the same way."""
    folders, here = {}, []
    for path in paths:
        rest = path[len(prefix):]
        if "/" in rest:
            folders.setdefault(rest.split("/", 1)[0], []).append(path)
        else:
            here.append(path)
    shapes = []
    for folder, inside in sorted(folders.items()):
        whole = exact_pattern(prefix + folder) + "/**"
        if not any(path in taken for path in inside) and _allowed(whole):
            shapes.append(whole)
        else:
            shapes += _shapes(inside, taken, prefix + folder + "/")
    kinds = {}  # extension ("" for none) -> [the paths here that have it], taken ones included
    for path in here:
        name = path[len(prefix):]
        kinds.setdefault(name.rsplit(".", 1)[1] if "." in name.lstrip(".") else "", []).append(path)
    for extension, named in sorted(kinds.items()):
        kind = exact_pattern(prefix) + "*." + exact_pattern(extension)
        if extension and len(named) > 1 and _allowed(kind) and \
                not any(glob_match(path, kind) for path in here if path in taken):
            shapes.append(kind)
        else:
            shapes += [exact_pattern(path) for path in named if path not in taken]
    return shapes


def _allowed(pattern: str) -> bool:
    try:
        check_pattern(pattern, "")
    except DatasetError:
        return False
    return True


def _narrowed(licenses: list[dict], number: int, paths: list[str]) -> list[str]:
    """The paths of entry number with `**` replaced by the shapes that cover
    the same files of the tree: every path that is not a history file
    (`**` never covered those), but those another entry covers (an entry
    selecting records by `values`, as this one does, shares its files)."""
    shares = "values" in licenses[number]
    others = PatternIndex([entry["paths"] if n != number and not (shares and "values" in entry) else []
                           for n, entry in enumerate(licenses)])
    wanted = sorted(path for path in paths if not path.startswith(HISTORY))
    taken = {path for path in wanted if others.matching(path)}
    kept = [pattern for pattern in licenses[number]["paths"] if pattern != "**"]
    return kept + [shape for shape in _shapes(wanted, taken) if shape not in kept]


class _Found(NamedTuple):
    """A license file that can be an entry's place: its path, the id of its
    license text, the license that text states (None when unknown), whether
    the text is admitted, and, for a text no item holds yet, the item to
    add and its bytes."""
    path: str
    text: str
    license: str | None
    admitted: bool
    new: dict | None = None
    content: bytes | None = None


def _own(entry: dict, license: str | None) -> bool:
    """Whether a text stating license states the entry's own license: the
    same expression, or one of the ids it is equally the text of."""
    return license is not None and (license == entry["license"] or entry["license"] in license.split(" OR "))


def _new_text(sid: str, path: str, content: bytes, texts: Texts) -> dict:
    """The license-text item of a license file no item holds: an admitted
    item when it is the standard text (texts.standard) of a license on the
    allowed list, by the person whose signed migration commit carries it;
    else under review by SERVICE, with the license it is the standard text
    of when there is one. An admitted one's `by` is None when the migration
    names nobody (_source_entries refuses to write it)."""
    today = date.today().isoformat()
    found = {"found": {"source": sid, "path": path}, "files": [hashlib.sha256(content).hexdigest()]}
    identified = texts.standard(content)
    if identified is None:
        return {"decision": {"decision": "under-review", "by": SERVICE, "date": today,
                             "reason": "first met in the migration: a person reads it"}, "kind": "license", **found,
                "explanation": ""}
    license = spdx.license_of(identified)
    standard = {"kind": "license", "license": license, **found,
                "explanation": f"standard text of {license}, SPDX release {spdx.RELEASE}"}
    try:
        check_license(license, sid)
    except RuleViolation:
        return {"decision": {"decision": "under-review", "by": SERVICE, "date": today,
                             "reason": "first met in the migration, the standard text of a license not on the "
                                       "allowed list: a person decides"}, **standard}
    return {"decision": {"decision": "admitted", "by": texts.by, "date": today}, **standard}


def _place(sid: str, entry: dict, covered: set[str], files, texts: Texts, read: dict, pending: dict) -> _Found | None:
    """The license file that is the entry's place, among the files whose
    name says they state a license: those whose license text (an item, one
    this source adds, pending, or a new one, _new_text) states a license
    (STATES_LICENSE), one stating the entry's own license first, then one
    the entry covers, then the shallowest, then the first by path (an entry
    covering no license file, such as one of the history, takes the
    source's). OutdatedSchema for a license file whose earlier review held
    for its exact bytes only (texts.held_back). read keeps each file's bytes
    and the new license text it would be."""
    found = []
    for path in files.paths:
        if not LICENSE_FILE.match(posixpath.basename(path)) or path.startswith(HISTORY):
            continue
        if path not in read:
            read[path] = files.read(path)
        content = read[path]
        if content is None:
            continue
        if hashlib.sha256(content).hexdigest() in texts.held_back:
            raise OutdatedSchema(f"{path} has a review of its own exact bytes in the earlier reviews of license "
                                 f"texts, which no license text can hold yet; a person decides it")
        text = texts.known.find(content)
        item = texts.known.items[text] if text is not None else pending.get(text_id(content), (None,))[0]
        if item is not None:
            candidate = _Found(path, text or text_id(content), item.get("license"),
                               item["decision"]["decision"] == "admitted")
            if item["kind"] not in STATES_LICENSE:
                continue
        else:
            if ("new", path) not in read:
                read["new", path] = _new_text(sid, path, content, texts)
            new = read["new", path]
            candidate = _Found(path, text_id(content), new.get("license"), new["decision"]["decision"] == "admitted",
                               new, content)
        found.append(((not _own(entry, candidate.license), path not in covered, path.count("/"), path), candidate))
    return min(found)[1] if found else None


def _source_entries(document, where):
    """A license entry gives its own place, expected texts and decision, and
    a source pins nothing (docs/licenses.md, docs/architecture.md):
    `checksum` and `revision` go; `**` is narrowed into the allowed shapes,
    covering the same files of the upstream tree (_narrowed); each entry
    gets its place:

        it selects records by `values`  -> the field of a record that the records formats' `license`
                                           named (the format loses it), its values the ones it expects
        a LicenseRef-public-domain one,  -> the entry itself, no expected text
        or one of an empty source
        any other                       -> the license file of its license text (_place), its expected
                                           text that one; a license file no license text holds becomes
                                           one, admitted when it is the standard SPDX text of an allowed
                                           license, else under review, with its text (_new_text)

    Each source's decision stays its source decision. These entries keep
    the source's signed admission (the maintainer's signed migration
    commit carries it): one that selects records (its values were in the
    signed source), a public-domain one, and one whose found license file
    is an admitted text of its own license. Every other entry goes under
    review, by SERVICE. The source's file list and license files come from
    where.files (no full download), read only when an entry needs them: a
    `**` to narrow, a license file to find, or records files whose formats
    name several license fields. The license texts the source adds are
    written last, once every entry has its place."""
    texts = where.texts()
    listing = []  # the upstream_files.Files of the source, once read

    def files():
        if not listing:
            listing.append(where.files(document))
        return listing[0]

    formats = _entries(document, "formats")
    named = sorted({entry["license"] for entry in formats if entry.get("format") == "records" and "license" in entry})
    licenses = [dict(entry) for entry in _entries(document, "licenses")]
    for number, entry in enumerate(licenses):
        if "**" in entry["paths"]:
            entry["paths"] = _narrowed(licenses, number, files().paths)
    covered = {}  # entry number -> the listed paths it covers, once listed

    def covered_by(number):
        if not covered:
            coverage = Coverage(licenses)
            covered.update({n: set() for n in range(len(licenses))})
            for path in files().paths:
                for found in (coverage.history if path.startswith(HISTORY) else coverage.all).matching(path):
                    covered[found].add(path)
        return covered[number]

    admitted = document["decision"]["decision"] == "admitted"
    read, pending, migrated = {}, {}, []  # pending: the license texts it adds, {id: (item, bytes)}
    for number, entry in enumerate(licenses):
        carried, why = False, "the migration found no admitted text of its own license at its place"
        if "values" in entry:
            field = named[0] if len(named) == 1 else _records_field(formats, covered_by(number), number)
            entry["place"], entry["expected"], carried = {"field": field}, entry.pop("values"), True
        elif "place" in entry:
            carried = True
        elif document["type"] == "empty":
            entry["place"], why = {"entry": True}, "text written for WPT: its entry is signed anew"
        elif entry["license"] == PUBLIC_DOMAIN:
            entry["place"], carried = {"entry": True}, True
        elif found := _place(where.name, entry, covered_by(number), files(), texts, read, pending):
            entry["place"], entry["expected"] = {"file": found.path}, [found.text]
            carried = found.admitted and _own(entry, found.license)
            if found.new is not None:
                if found.new["decision"]["by"] is None:
                    raise OutdatedSchema(f"{found.path} is the standard text of {found.license}, whose license text "
                                         f"the signed migration commit admits: run `wpt-dataset migrate --by NAME`, "
                                         f"NAME its signer")
                pending.setdefault(found.text, (found.new, found.content))
        else:
            raise OutdatedSchema(f"license entry {number + 1} ({entry['license']}): no file of the source whose "
                                 f"name says it states a license holds a license text; a person gives its place")
        if "decision" not in entry:
            entry["decision"] = dict(document["decision"]) if carried and admitted else \
                {"decision": "under-review", "by": SERVICE, "date": date.today().isoformat(),
                 "reason": why if admitted else "the source is not admitted"}
        migrated.append(_ordered(entry))
    for text, (item, content) in pending.items():
        texts.add(text, item, content)
    return {key: ([{k: v for k, v in entry.items() if not (k == "license" and entry.get("format") == "records")}
                   for entry in value] if key == "formats" else migrated if key == "licenses" else value)
            for key, value in document.items() if key not in ("checksum", "revision")}


def _records_field(formats, paths, number) -> str:
    """The one license field the records formats name for these listed
    paths (those an entry selecting records covers); OutdatedSchema when
    they name none or several."""
    fields = set()
    for entry in formats:
        if entry.get("format") == "records" and "license" in entry:
            index = PatternIndex([entry.get("paths", [])])
            if any(index.matching(path) for path in paths):
                fields.add(entry["license"])
    if len(fields) != 1:
        raise OutdatedSchema(f"license entry {number + 1} lists `values`, but the records files it covers name "
                             f"{len(fields)} license fields ({', '.join(sorted(fields)) or 'none'}); a person "
                             f"gives its field place")
    return fields.pop()

# the order of a license entry's keys
ENTRY_KEYS = ("license", "paths", "attribution", "place", "expected", "text", "decision", "note")


def _ordered(entry: dict) -> dict:
    return {key: entry[key] for key in ENTRY_KEYS if key in entry} | \
        {key: value for key, value in entry.items() if key not in ENTRY_KEYS}


STEPS["source"].append(Step("a pinned upstream and no license entry decisions", _pinned_shape, _source_entries))

# the source types that named how a site serves its files, and the protocol each is read with (protocols/)
TYPES_OLD = {"http-file": "http", "http-archive": "http", "huggingface": "git"}


def _source_protocol(document, _where):
    """A source's type names its protocol (protocols/): http-file and
    http-archive become http, which unpacks a file whose name says it is an
    archive (protocols.http.is_archive) and keeps any other as one file;
    huggingface becomes git, a Hugging Face dataset being a git repository
    at the same address. OutdatedSchema, the file left as it is, for an
    http-archive source whose file's name is not an archive's, or an
    http-file one whose is: under http it would be read otherwise, so a
    person gives the file's name (extract.name)."""
    kind = document["type"]
    if kind in ("http-file", "http-archive"):
        for address in [document["address"]] + [part["address"] for part in _entries(document, "addresses")] + \
                [mirror["address"] for mirror in _entries(document, "mirrors")]:
            try:
                name = name_of(address, document)
            except OSError as error:
                raise OutdatedSchema(f"{kind} source: {error}") from error
            if is_archive(name) != (kind == "http-archive"):
                raise OutdatedSchema(f"{kind} source: the name of {address} says it is "
                                     f"{'not ' if kind == 'http-archive' else ''}an archive, so the http protocol "
                                     f"would read it otherwise; a person gives its name (extract.name)")
    return {**document, "type": TYPES_OLD[kind]}


STEPS["source"].append(Step("a source type that names a site's way of serving rather than a protocol",
                            lambda d: d.get("type") in TYPES_OLD, _source_protocol))

# the numbers of the architecture an earlier shape did not hold
NEW_ATTENTION = ("attention_heads", "attention_head_dim", "query_rank", "output_groups", "output_rank", "index_heads",
                 "index_head_dim", "index_top_k")


def _architecture(document, key) -> bool:
    return isinstance(document.get("model"), dict) and key in document["model"]


def _model_heads(document, _where):
    """The architecture's numbers are those of docs/model.md. The window, the
    layers and the width carry over, and the old heads become both attention
    kinds' heads; the other numbers are a person's choice, so the step
    leaves them at 0 and the check that follows names each file to fill in."""
    old = document["model"]
    head_dim = old["n_embd"] // old["n_head"]
    model = {"block_size": old["block_size"], "n_layer": old["n_layer"], "d_model": old["n_embd"],
             "kda_heads": old["n_head"], "kda_head_dim": head_dim, "mla_heads": old["n_head"], "mla_head_dim": head_dim,
             "mla_kv_rank": 0, "n_experts": 0, "top_k": 0, "expert_width": 0, "latent_width": 0, "shared_width": 0,
             "engram_layers": [], "engram_order": 0, "engram_heads": 0, "engram_dim": 0, "engram_table_size": 0}
    return {**document, "model": model}


def _model_attention(document, _where):
    """The attention is DeepSeek-V4's CSA and HCA (docs/model.md,
    "Attention"), whose numbers are new: the MLA numbers go, and the new ones
    are a person's choice, as is which global layer is CSA and which HCA
    (global_attention), which gated MLA's numbers do not say. So the step
    leaves the numbers at 0 and the list empty, and the check that follows
    names each file to fill in. Weights trained with the old attention do
    not load into the new model."""
    model = {}
    for key, value in document["model"].items():
        if key == "mla_heads":
            model.update(dict.fromkeys(NEW_ATTENTION, 0))
        elif key not in ("mla_head_dim", "mla_kv_rank"):
            model[key] = value
        if key == "n_layer":
            model["global_attention"] = []
    return {**document, "model": model}


def _model_stages(document, _where):
    """The model trains in stages (docs/training.md, "Stages"), each with its
    window length, steps and batch: the model's block_size and the
    training's steps and batch_size become one stage, in the training part
    where steps were."""
    model = {key: value for key, value in document["model"].items() if key != "block_size"}
    old, training = document["training"], {}
    for key, value in old.items():
        if key == "steps":
            training["stages"] = [{"block_size": document["model"]["block_size"], "steps": value,
                                   "batch_size": old["batch_size"]}]
        elif key != "batch_size":
            training[key] = value
    return {**document, "model": model, "training": training}


# the drafter's training values every recipe states
DRAFTER_DEFAULTS = {"prompts": 256, "prompt_tokens": 64, "answer_tokens": 256, "epochs": 10, "learning_rate": 6e-4,
                    "held_out": 0.1, "anchors": 512, "batch": 8, "seed": 0}


def _model_drafter(document, _where):
    """The `drafter` section (docs/model.md, "The drafter"): the drafter that
    wpt-train trains at the end of every run. It goes in after `training`,
    with the training values every recipe states (DRAFTER_DEFAULTS); the
    settings it leaves out keep their defaults."""
    migrated = {}
    for key, value in document.items():
        migrated[key] = value
        if key == "training":
            migrated["drafter"] = dict(DRAFTER_DEFAULTS)
    return migrated


# the settings of the old `muon` that make the loss, which the read-text step's method holds
NEXT_TOKEN_OLD = ("mtp_weight", "mtp_late_weight", "mtp_switch")
# the old post-training steps and the training steps that do the same, and the settings of theirs that
# belong to the step's optimizer (adamw), under that optimizer's names
STEPS_OLD = {"sft": "answer-as-assistant", "grpo": "finish-tasks", "token-efficiency": "finish-tasks-with-less-writing"}
OPTIMIZER_OLD = {"learning_rate": "lr", "weight_decay": "weight_decay"}


def _model_steps(document, _where):
    """A model trains in one ordered list of steps (docs/training.md,
    "Steps"), in place of `training` and `posttrain`: the training becomes
    the first step, read-text, its method the step's optimizer (with its
    settings but those of the loss, NEXT_TOKEN_OLD, which become the step's),
    its stages, eval_every and eval_batches the step's settings, nothing
    locked; its seed becomes the recipe's `seed`. Each post-training step
    follows under its new name (STEPS_OLD), the settings of its learning
    rate and weight decay given to its optimizer, adamw, and a `tag` setting
    that names an old step renamed with it; a step given as a Python file
    keeps its name and settings, and the check that follows says what the
    file lacks. `steps` and `seed` go where `training` was, `posttrain`
    goes."""
    training = document["training"]
    method_settings = training["settings"]
    loss = {key: value for key, value in method_settings.items() if key in NEXT_TOKEN_OLD}
    first = {"step": "read-text",
             "optimizer": {"name": training["method"],
                           "settings": {key: value for key, value in method_settings.items() if key not in loss}},
             "settings": {"stages": training["stages"], "eval_every": training["eval_every"],
                          "eval_batches": training["eval_batches"], **loss},
             "locked": []}
    steps = [first]
    for entry in document["posttrain"]:
        name, settings = (entry, {}) if isinstance(entry, str) else (entry["step"], entry["settings"])
        if name not in STEPS_OLD:
            steps.append({"step": name, "settings": settings})
            continue
        optimizer = {OPTIMIZER_OLD[key]: value for key, value in settings.items() if key in OPTIMIZER_OLD}
        # a tag that names an old step names the step that does the same
        step = {"step": STEPS_OLD[name], "settings": {key: STEPS_OLD.get(value, value) if key == "tag" else value
                                                      for key, value in settings.items() if key not in OPTIMIZER_OLD}}
        steps.append({**step, **({"optimizer": {"name": "adamw", "settings": optimizer}} if optimizer else {})})
    migrated = {}
    for key, value in document.items():
        if key == "training":
            migrated["seed"], migrated["steps"] = training["seed"], steps
        elif key != "posttrain":
            migrated[key] = value
    return migrated


STEPS["model"] += [
    Step("an architecture with n_embd", lambda d: _architecture(d, "n_embd"), _model_heads),
    Step("an architecture with gated MLA", lambda d: _architecture(d, "mla_heads"), _model_attention),
    Step("a window length in the architecture", lambda d: _architecture(d, "block_size") and "training" in d,
         _model_stages),
    Step("no drafter section", lambda d: "training" in d and "drafter" not in d, _model_drafter),
    Step("training and post-training apart", lambda d: "training" in d and "posttrain" in d, _model_steps),
]


STEPS["task"] += [
    Step("replies without the paths their answers live in", lambda d: bool(d.get("replies")) and "answered_in" not in d,
         lambda d, _where: {**d, "answered_in": []}),
]
# `answered_in` is the paths where a task's answers live, required with replies. Only a person knows where they
# live, so the step leaves it empty, and the check that follows names each file to fill in.


def _repository_checksum(document) -> bool:
    return any(pointer.get("for") == "repository" and "checksum" in pointer for pointer in document.get("pointers", []))


def _task_manifest(document, _where):
    """A repository pointer's `checksum`, "sha256:" + its tree's manifest,
    becomes `manifest`, the plain hex digest, in the same place."""
    def field(key, value):
        return ("manifest", value.removeprefix("sha256:")) if key == "checksum" else (key, value)

    def renamed(pointer):
        return dict(itertools.starmap(field, pointer.items())) if pointer.get("for") == "repository" else pointer
    return {**document, "pointers": [renamed(pointer) for pointer in document["pointers"]]}


STEPS["task"].append(Step("a repository pointer that names its tree by a checksum", _repository_checksum,
                          _task_manifest))


STEPS["tokenizer"] += [
    Step("no vocabulary version", lambda d: "version" not in d, lambda d, _where: {**d, "version": 1}),
]


def _lock_without_documents(document, _where):
    """A lock that does not list the documents its vocabulary was trained on
    cannot be converted."""
    raise OutdatedSchema("the lock does not list the documents its vocabulary was trained on, "
                         "so it cannot be converted: delete it and build again")


STEPS["lock"] += [
    Step("no list of the documents its vocabulary was trained on", lambda d: "documents" not in d,
         _lock_without_documents),
    Step("the tokenizer code's version", lambda d: "wpt_tokenizer" in d,
         lambda d, _where: {key: value for key, value in d.items() if key != "wpt_tokenizer"}),
]

# an earlier shape carried a number in `schema`; no file does now
for _kind in KINDS_WITH_STEPS:
    STEPS[_kind].append(Step("a shape number (`schema`)", lambda d: "schema" in d,
                             lambda d, _where: {key: value for key, value in d.items() if key != "schema"}))


def _applied(document, kind: str, where, steps) -> tuple[object, list[Step]]:
    """(the document after every step that applies, those steps, in order);
    the document itself, and none, when it is not an object (the check then
    says what is wrong). where names it in errors and goes to each step."""
    applied = []
    for step in steps.get(kind, ()) if isinstance(document, dict) else ():
        if step.applies(document):
            try:
                document = step.rewrite(document, where)
            except DatasetError as error:
                raise type(error)(f"{where}: {error}") from None
            applied.append(step)
    return document, applied


def outdated(document, kind: str, steps=None) -> list[str]:
    """What is of an earlier shape in the document, as the steps of its kind
    name it ([]: nothing; also for what is not an object, which the check
    then reports)."""
    steps = STEPS if steps is None else steps
    if not isinstance(document, dict):
        return []
    return [step.what for step in steps.get(kind, ()) if step.applies(document)]


def upgrade(document, kind: str, path, steps=None):
    """The document brought to the current shape of its kind, step by step;
    the document itself when it is current, or not an object. path names it
    in errors and goes to each step."""
    return _applied(document, kind, path, STEPS if steps is None else steps)[0]


# the verdicts of the earlier reviews of license texts, and the decision each becomes
VERDICTS = {"allowed-cosmetic": "admitted", "allowed-notice-only": "admitted", "allowed-plus-exception": "admitted",
            "readme-no-refusal": "admitted", "not-allowed": "refused", "readme-refuses-ai-training": "refused",
            "not-a-license": "refused", "gpl-version-unclear": "under-review", "no-license-found": "under-review",
            "hand-checked": "under-review"}
# the kinds of the earlier reviews that are a license text's kinds as they are, and their new names
KINDS_KEPT = {"license": "license", "notice": "notice", "not-a-license": "not-a-license", "readme-ok": "readme"}


def _text_of_review(review: dict, by: str, today: str) -> dict:
    """The license-text item of an earlier review, without its text (the
    review held none): its decision by the review's verdict (VERDICTS), by
    the person whose signed migration commit carries it, a refusal's reason
    the review's `why` or its verdict, `recheck` the decision's note; its
    kind the review's (KINDS_KEPT), or for `out` (what it covered was left
    out) what its verdict says the text is; its license the review's, else
    the SPDX expression it found; its explanation the review's."""
    answer, verdict = review["answer"], review["answer"]["verdict"]
    license = review.get("license") or answer.get("spdx") or None
    kind = KINDS_KEPT.get(review["kind"]) or ("readme" if verdict == "readme-refuses-ai-training" else
                                              "not-a-license" if verdict in ("not-a-license", "no-license-found") else
                                              "notice" if verdict == "allowed-notice-only" and license else "license")
    decision = {"decision": VERDICTS[verdict], "by": by, "date": today}
    if decision["decision"] == "refused":
        decision["reason"] = review.get("why") or f"the review's verdict: {verdict}"
    elif decision["decision"] == "under-review":
        decision["reason"] = f"the review's verdict: {verdict}"
    if review.get("recheck"):
        decision["note"] = review["recheck"]
    return {"decision": decision, "kind": kind, **({"license": license} if license else {}), "files": [],
            "explanation": answer["explanation"]}


def _convert_reviews(repo: Repo, by: str | None, log) -> tuple[list[str], list[str], set[str]]:
    """Make each review of data/reviewed-license-texts.json a license text
    (_text_of_review), keyed by its group's key, the sha256 of its
    normalized text, which is the license text's id. A review keyed by one
    file's sha256 ("file:"), held for those exact bytes only, has no
    normalized text to give it an id, and one with a verdict VERDICTS does
    not name has no decision: each stays in the file, named. Returns (the
    places written, why each review left stays, the sha256s of the exact
    files those "file:" reviews hold back). RuleViolation, before anything
    is written, when there are reviews to convert and no person (by) whose
    signed migration commit carries their decisions."""
    reviews = repo.earlier_reviews()
    if reviews is None:
        return [], [], set()
    if by is None:
        raise RuleViolation("the earlier reviews of license texts (reviewed-license-texts.json) become license "
                            "texts whose decisions the signed migration commit carries: run `wpt-dataset migrate "
                            "--by NAME`, NAME its signer")
    check_person(by, "--by")
    today, existing = date.today().isoformat(), set(repo.license_text_ids())
    written, left, kept = [], [], {}
    for key, review in reviews.items():
        verdict = review.get("answer", {}).get("verdict") if isinstance(review, dict) else None
        if key.startswith("file:"):
            kept[key], why = review, "it holds for one file's exact bytes, and a license text's id is the sha256 " \
                                     "of a normalized text, which the review does not hold"
        elif verdict not in VERDICTS:
            kept[key], why = review, f"its verdict {verdict!r} names no decision"
        else:
            if key not in existing:
                written.append(repo.add_license_text(key, _text_of_review(review, by, today), None))
            continue
        left.append(f"the earlier reviews of license texts: review {key} stays: {why}; a person decides it")
    repo.rewrite_earlier_reviews(kept or None)
    log(f"{len(written)} reviews of license texts made license texts, {len(left)} left")
    return written, left, {key.removeprefix("file:") for key in kept if key.startswith("file:")}


def migrate(repo: Repo, steps=None, log=print, upstream=None, by: str | None = None, standard=None) -> list[str]:
    """Rewrite every file that is of an earlier shape, through the service
    layer (Repo.rewrite), after the earlier reviews of license texts became
    license texts (_convert_reviews). upstream(source id, document) gives a
    source's upstream_files.Files, for the step that reads them; by is the
    person whose signed migration commit carries the decisions it writes
    (license texts converted or admitted as standard texts); standard() gives
    the spdx.Matcher of the standard texts (None: no license file is
    compared with them). Returns the places of the files changed (data/...).
    A file a step cannot rewrite (a DatasetError) stays as it is, and the
    others are rewritten: MigrationIncomplete then names each with why,
    and each review left in its earlier file."""
    steps = STEPS if steps is None else steps
    changed, left, held_back = _convert_reviews(repo, by, log)
    texts = Texts(repo, by, standard, held_back)
    for kind, name in repo.versioned():
        where = Where(repo, kind, name, upstream, texts)
        try:
            migrated, applied = _applied(repo.raw(kind, name), kind, where, steps)
        except DatasetError as error:
            left.append(f"{type(error).__name__}: {error}")
            log(f"left in its earlier shape: {left[-1]}")  # the error names the file
            continue
        if not applied:
            continue
        changed.append(repo.rewrite(kind, name, migrated))
        for step in applied:
            if step.after is not None:
                step.after(where, migrated)
        log(f"{changed[-1]}: {'; '.join(step.what for step in applied)}")
    if left:
        raise MigrationIncomplete(f"{len(changed)} files rewritten, {len(left)} left in their earlier shape:\n"
                                  + "\n".join(left))
    if steps is STEPS:  # the real shapes: every file must now validate
        for kind, name in repo.versioned():
            repo.validate_file(kind, name)
    return changed


def run(repo: Repo, store, by: str | None = None, log=print) -> list[str]:
    """`wpt-dataset migrate`: migrate with the real steps, a source's file
    list and license files read from store, or else from its upstream
    (upstream_files.files, never a full download), each source's fetches in
    a temporary folder of its own, removed at the end; and the standard
    texts of one SPDX release (spdx.texts, fetched once, only when a license
    file needs them). Its result and failures are migrate's."""
    with tempfile.TemporaryDirectory(prefix="wpt-migrate-") as tmp:
        count = itertools.count()

        def upstream(sid, document):
            folder = Path(tmp) / str(next(count))
            folder.mkdir()
            return upstream_files.files(sid, document, repo.patches(sid), store, folder)

        return migrate(repo, log=log, upstream=upstream, by=by, standard=lambda: spdx.Matcher(spdx.texts(USER_AGENT)))
