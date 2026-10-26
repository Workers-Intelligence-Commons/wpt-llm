"""Source drafters: programs that write source.json drafts from what a site
publishes, and the runner that saves their drafts (docs/architecture.md,
"Source drafters").

A source drafter is one Python file: a built-in in
wpt_dataset/source_drafters/, or any path given to `wpt-dataset
source-drafter`. It defines

    SETTINGS = {name: default, ...}  # optional: what --set NAME=VALUE changes, each of its default's type
    def check(settings)  # optional: raises ValueError, saying why, on settings that cannot go together
    def candidates(context) -> iterable of Candidate  # what the site offers
    def draft(candidate, context) -> Drafted | Skipped | NeedsPerson

The runner does the rest. Each Drafted document is saved through the
service layer (wpt_dataset.repo): a new source is created, and a source
that exists already (the same id) is updated in place, as updated() says. A source
that has patches is never written: its outcome is "stopped", with a note,
since a person redoes its patches. The sources a draft supersedes are put
back under review once the draft is saved (superseded()). Every outcome is recorded, in the run's
record and in the record of what is done, so the same command, on any
machine sharing the store, drafts only what is left. A candidate whose
draft raised is "failed" and is not done: the next run drafts it again.

Everything a run keeps lives in the store (WPT_STORE):

- the record of what is done: one ref per candidate, under
  source-drafter-done/<done key>.<sha256 of its name>, pointing at its
  outcome; the done key is the source drafter's name and the sha256 of its
  settings, so other settings start again;
- site facts: what context.fact(key, compute) computed once, under
  source-drafter-facts/<name>.<sha256 of key>;
- fetched pages: each page context.fetch read whose answer gave an ETag,
  with that ETag, under source-drafter-pages/<sha256 of the request (its
  address and headers, without Authorization)>, so the next fetch of the
  same request asks the site whether it changed (If-None-Match) and a 304
  answer gives the kept page;
- the run's record, records/<folder>/ (wpt_telemetry), with candidates.json
  (how many candidates, chosen, done before, to draft) and outcomes.jsonl
  (one line per outcome: candidate, outcome, source, change, reason,
  notes).

How many drafts run at once: the rule of wpt_utils.parallel, each
draft's memory measured as it runs. Above one, the drafts run in worker processes; the
saves and the records stay in the running process, one at a time, in the
candidates' order, so a run's outcomes do not depend on how many drafted
at once. A worker process's own log lines, and the site facts and pages
it keeps, are not in the run's record (they are in the store all the
same).
"""

import contextlib
import hashlib
import importlib.util
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from wpt_store import as_record_name, store_from_settings
from wpt_telemetry import Telemetry, set_ref
from wpt_utils.files import sha256_file
from wpt_utils.parallel import Sizing, in_order, process_pool
from wpt_utils.settings import typed

from . import retry, robots, sites
from .errors import DatasetError, RuleViolation
from .repo import DRAFT_CHECKSUM, Repo, json_text, source_id
from .sources import USER_AGENT

BUILT_IN = Path(__file__).parent / "source_drafters"
COMMAND = "wpt-dataset-source-drafter"  # the run record's command, part of its folder's name
# the kinds of ref a run keeps in the store
DONE, FACTS, PAGES = "source-drafter-done", "source-drafter-facts", "source-drafter-pages"
CANDIDATES, OUTCOMES = "candidates.json", "outcomes.jsonl"  # files of the run's record
# every outcome; "stopped": a source with patches, which is not written; "failed": the draft or its save raised
KINDS = ("drafted", "skipped", "needs-a-person", "stopped", "failed")
# what a drafted source changes: a new source; one updated in place with its decision kept, or sent back
# under review because its license entries differ; or one the draft gives exactly as it is
CHANGES = ("new", "updated", "licenses-changed", "unchanged")


class SourceDrafterError(DatasetError):
    """A source drafter cannot run: it is not a built-in nor a Python file,
    defines no candidates(context) or draft(candidate, context), a setting
    is unknown or of another type or its check refuses its settings, or
    candidates() gave something other than Candidates with distinct names.
    Nothing has been drafted."""


@dataclass(frozen=True)
class Candidate:
    """One thing a site offers. name: unique among the candidates and the
    same from run to run, since the record of what is done keys on it;
    source: the id of the source it drafts, when known before drafting (a
    source with patches then stops before its draft, and --only may name
    it); facts: what draft needs of it, JSON values only (a worker process
    receives it)."""
    name: str
    source: str | None = None
    facts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Drafted:
    """A draft of one source: its source.json without its decision (the
    runner gives it), and the origin and item of its id
    (repo.source_id). notes: JSON for a person, recorded with the outcome.
    supersedes: the ids of the sources this draft replaces (such as the same
    site's source at an older address); each admitted one goes back under
    review when the draft is saved, and nothing else in it changes."""
    origin: str
    item: str | None
    document: dict
    notes: object = None
    supersedes: tuple = ()


@dataclass(frozen=True)
class Skipped:
    """Nothing to draft, and why."""
    reason: str


@dataclass(frozen=True)
class NeedsPerson:
    """The draft needs a person: why, and notes (JSON) for them."""
    reason: str
    notes: object = None


@dataclass
class Context:
    """What a source drafter's candidates() and draft() get. repo: the
    repository, read only (the runner writes the drafts); store: the store,
    for anything fetched or built; settings: SETTINGS with those given;
    name: the source drafter's name; by, today: the decision's `by` and
    date the runner gives a draft; work: an empty folder of its own for each
    draft() call (git clones, downloads), removed after it, None in
    candidates(); telemetry: the run's record (Telemetry.off() in a worker
    process); robots: the robots.txt files read so far, which fetch reads
    again from here (robots.check's cache), as may a program that checks
    other addresses itself."""
    repo: Repo
    store: object
    settings: dict
    name: str
    by: str
    today: str
    work: Path | None = None
    telemetry: Telemetry = field(default_factory=Telemetry.off)
    robots: dict = field(default_factory=dict, repr=False)  # each site's robots.txt, read once (robots.check)

    def fact(self, key: str, compute):
        """The site fact named key: compute()'s JSON value, computed once
        and kept in the store, so a later run, on any machine sharing it,
        reads it instead. A key names what the value depends on (a dump's
        run, a listing's date): another value needs another key."""
        ref = f"{as_record_name(self.name)}.{hashlib.sha256(key.encode()).hexdigest()}"
        found = self.store.get_ref(FACTS, ref)
        if found is not None:
            return json.loads(self.store.get_bytes(found))["value"]
        value = compute()
        set_ref(self.store, FACTS, ref, self.store.put_bytes(json_text({"key": key, "value": value}).encode()),
                self.telemetry)
        return value

    def fetch(self, address: str, headers: dict | None = None) -> bytes:
        """The bytes an http(s) address answers, as wpt-dataset fetches a
        source: refused (CrawlForbidden) when the site's robots.txt forbids
        it (robots.py), and tried again after a failure that may pass
        (retry.py). Each site's robots.txt is read once per Context. A page
        whose answer gave an ETag is kept in the store with it; the same
        request asks again with If-None-Match, and an answer 304 (not
        changed) gives the kept page, which GitHub does not count against
        its limit."""
        robots.check(address, "http", USER_AGENT, self.robots)
        headers = {"User-Agent": USER_AGENT, **(headers or {})}
        # the token changes who asks, not the page: it stays out of the key and of the store
        asked = {name: value for name, value in headers.items() if name.lower() != "authorization"}
        ref = hashlib.sha256(json.dumps({"address": address, "headers": asked}, sort_keys=True).encode()).hexdigest()
        found = self.store.get_ref(PAGES, ref)
        kept = json.loads(self.store.get_bytes(found)) if found is not None else None
        request = urllib.request.Request(address, headers={**headers, **({"If-None-Match": kept["etag"]} if kept
                                                                         else {})})

        def read():
            sites.paced(address)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return response.read(), response.headers.get("ETag")
            except urllib.error.HTTPError as error:
                if error.code == 304 and kept is not None:
                    return None, None
                raise
        data, etag = retry.call(read, address)
        if data is None:
            return self.store.get_bytes(kept["page"])
        if etag:
            page = {"address": address, "headers": asked, "etag": etag, "page": self.store.put_bytes(data)}
            set_ref(self.store, PAGES, ref, self.store.put_bytes(json_text(page).encode()), self.telemetry)
        return data


def built_in_names() -> list[str]:
    return sorted(path.stem for path in BUILT_IN.glob("*.py") if not path.name.startswith("_"))


def find(target: str) -> tuple[str, Path, str]:
    """(name, file, how the record names the file): a built-in by its
    package path, another file by its name only, so the record never says
    where it lives on this machine."""
    if target in built_in_names():
        return target, BUILT_IN / f"{target}.py", f"wpt_dataset/source_drafters/{target}.py"
    path = Path(target)
    if path.suffix != ".py" or not path.is_file():
        raise SourceDrafterError(f"{target}: not a built-in source drafter ({', '.join(built_in_names())}) nor a "
                                 f"Python file")
    return path.stem, path, path.name


def load(target: str, overrides: dict) -> tuple[str, Path, str, object, dict]:
    """(name, file, how the record names it, the loaded module, its
    settings): SETTINGS changed by overrides, each of its default's type
    (wpt_utils.settings.typed), then passed through the module's
    check(settings), when it has one."""
    name, path, file_name = find(target)
    spec = importlib.util.spec_from_file_location(f"wpt_dataset_source_drafter_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for function in ("candidates", "draft"):
        if not callable(getattr(module, function, None)):
            raise SourceDrafterError(f"{file_name}: defines no {function}()")
    try:
        settings = typed(getattr(module, "SETTINGS", {}), overrides, file_name)
        getattr(module, "check", lambda settings: None)(settings)
    except ValueError as error:
        raise SourceDrafterError(f"{file_name}: {error}" if not str(error).startswith(file_name) else str(error)) \
            from None
    return name, path, file_name, module, settings


def describe(target: str) -> dict:
    """{name, file, summary (the first line of its docstring), settings
    (SETTINGS)} of a source drafter, for the lists of the command and of
    wpt-web."""
    name, _, file_name, module, settings = load(target, {})
    summary = (module.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
    return {"name": name, "file": file_name, "summary": summary, "settings": settings}


def done_key(name: str, settings: dict) -> str:
    """The name of a source drafter's record of what is done, for these
    settings."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    return f"{as_record_name(name)}-{digest}"


def _done_ref(key: str, candidate: str) -> str:
    return f"{key}.{hashlib.sha256(candidate.encode()).hexdigest()}"


# -- saving a draft: update in place ------------------------------------------------------

def under_review(by: str, today: str) -> dict:
    return {"decision": "under-review", "by": by, "date": today}


def updated(existing: dict, drafted: dict, by: str, today: str) -> tuple[dict, str]:
    """(the source.json to write, its change) when a draft gives a source
    that exists: the existing file with the draft's revision, checksum,
    exclusions (extract), file-kind map (formats), submodule list
    (submodules) and history start (history); the rest stays as a person
    may have left it. The checksum stays when the revision and the
    exclusions do and the draft's is the placeholder (DRAFT_CHECKSUM): the
    same content is pinned already.
    When the license entries differ, the draft's replace them and the
    decision goes back to under-review (by, today): change
    "licenses-changed"; otherwise the decision stays."""
    document = dict(existing)
    document["revision"] = drafted["revision"]
    for key in ("extract", "formats", "submodules", "history"):
        if key in drafted:
            document[key] = drafted[key]
        else:
            document.pop(key, None)
    same_content = drafted["revision"] == existing["revision"] and drafted.get("extract") == existing.get("extract")
    if not (same_content and drafted["checksum"] == DRAFT_CHECKSUM):
        document["checksum"] = drafted["checksum"]
    if drafted["licenses"] != existing["licenses"]:
        document["licenses"] = drafted["licenses"]
        document["decision"] = under_review(by, today)
        return document, "licenses-changed"
    return document, "unchanged" if document == existing else "updated"


def save(repo: Repo, candidate: Candidate, drafted: Drafted, by: str, today: str) -> dict:
    """Save a draft through the service layer; its outcome (as an
    outcomes.jsonl line, without the candidate). A new source is created
    under review (by, today); an existing one is updated in place (updated),
    or stopped when it has patches. RuleViolation when the draft's id is
    not the candidate's source; the service layer's errors when the rules
    refuse it, and then nothing is written."""
    if not isinstance(drafted.document, dict):
        raise RuleViolation(f"{candidate.name}: the draft's document is not a JSON object")
    document = {**drafted.document, "decision": under_review(by, today)}
    sid = source_id(drafted.origin, drafted.item, document.get("address", ""))
    if candidate.source is not None and candidate.source != sid:
        raise RuleViolation(f"{candidate.name}: the draft gives the source {sid}, the candidate names "
                            f"{candidate.source}")
    if not repo.exists("source", sid):
        repo.create_source(drafted.origin, drafted.item, document, {}, "", None)
        return _superseding(repo, sid, drafted, by, today, _line("drafted", sid, "new", "", drafted.notes))
    stopped = _with_patches(repo, sid)
    if stopped is not None:
        return stopped
    version = repo.version("source", sid)
    written, change = updated(repo.source(sid).data, document, by, today)
    if change != "unchanged":
        repo.save_source(sid, written, version)
    return _superseding(repo, sid, drafted, by, today, _line("drafted", sid, change, "", drafted.notes))


def _superseding(repo: Repo, sid: str, drafted: Drafted, by: str, today: str, line: dict) -> dict:
    """line, with "superseded" (the ids put back under review) when the
    draft supersedes sources: each one that exists and is admitted or
    refused gets the decision under-review (by, today) and nothing else
    changes in it."""
    superseded = []
    for other in dict.fromkeys(drafted.supersedes):
        if other == sid or not repo.exists("source", other):
            continue
        if repo.source(other).data["decision"]["decision"] != "under-review":
            document = dict(repo.source(other).data)
            document["decision"] = under_review(by, today)
            repo.save_source(other, document, repo.version("source", other))
            superseded.append(other)
    return {**line, "superseded": superseded} if superseded else line


def _with_patches(repo: Repo, sid: str) -> dict | None:
    """The "stopped" outcome of an existing source that has patches, or None."""
    patches = repo.source(sid).patches
    if not patches:
        return None
    return _line("stopped", sid, None, f"{sid} has patches ({', '.join(p.name for p in patches)}); "
                                      f"a person redoes them on the new revision, so it is left as it is")


def _line(outcome, source=None, change=None, reason="", notes=None) -> dict:
    return {"outcome": outcome, "source": source, "change": change, "reason": reason, "notes": notes}


# -- drafting: in this process or in workers -------------------------------------------------

def _draft(module, candidate: Candidate, context: Context):
    """draft()'s outcome in its own work folder, or the line of a failure."""
    context.store.scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wpt-source-drafter-", dir=context.store.scratch))
    context.work = work
    try:
        outcome = module.draft(candidate, context)
    except Exception as error:  # one candidate's failure never stops the others
        return _line("failed", candidate.source, None, f"{type(error).__name__}: {error}")
    finally:
        context.work = None
        shutil.rmtree(work, ignore_errors=True)
    if not isinstance(outcome, (Drafted, Skipped, NeedsPerson)):
        return _line("failed", candidate.source, None,
                     f"draft() returned {type(outcome).__name__}, not Drafted, Skipped or NeedsPerson")
    return outcome


_WORKER = {}  # in a worker process: the module and the context, set once by _start_worker


def _start_worker(path: str, name: str, settings: dict, root: str, by: str, today: str):
    spec = importlib.util.spec_from_file_location(f"wpt_dataset_source_drafter_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = Repo(root)
    _WORKER.update(module=module, context=Context(repo, store_from_settings(repo.root / ".env"), settings, name, by,
                                                  today))


def _draft_in_worker(candidate: Candidate):
    return _draft(_WORKER["module"], candidate, _WORKER["context"])


def _drafts(module, path, candidates, context):
    """(candidate, draft()'s outcome) of each candidate, in the candidates'
    order, as many drafting at once as wpt_utils.parallel allows: in this
    process when that is one, otherwise in worker processes."""
    sizing = Sizing("drafts at once")
    context.telemetry.log(sizing.why)
    if sizing.limit <= 1:
        for candidate in candidates:
            yield candidate, _draft(module, candidate, context)
        return
    pool = process_pool(sizing, sizing.limit, _start_worker,
                        (str(path), context.name, context.settings, str(context.repo.root), context.by, context.today))
    try:
        yield from zip(candidates, in_order(_draft_in_worker, candidates, pool, sizing))
    except BaseException:  # a stop (Ctrl+C) ends the drafts under way at once; the next run drafts them
        pool.terminate_workers()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# -- a run ---------------------------------------------------------------------------------

@contextlib.contextmanager
def recorded(store, target: str, arguments: list[str], started_by: str, env_file):
    """The telemetry of one run of the source drafter target: its record,
    records/<UTC>-wpt-dataset-source-drafter-<its name>-<random>/, with
    who started it (started_by, as wpt-web finds its own runs by it) and
    every new try of a fetch in it (retry.reporting)."""
    with (Telemetry.start(store, COMMAND, arguments, find(target)[0],
                          inputs={"source_drafter": target, "started_by": started_by}, env_file=env_file) as telemetry,
          retry.reporting(telemetry.log)):
        yield telemetry


def run(repo: Repo, store, target: str, overrides: dict, by: str, telemetry: Telemetry, only=(), again=False,
        today=None) -> dict[str, int]:
    """Run a source drafter (a built-in's name or a file's path) with its
    settings changed by overrides: its candidates (those only names, by name
    or source id, when only is given), each one not done yet (every one with
    again) drafted and saved, every outcome recorded in telemetry's record
    and in the record of what is done. Returns {outcome: count}, with
    "done before". SourceDrafterError before anything is drafted."""
    if not by:
        raise SourceDrafterError("give who drafts (--by), the decision's `by` of every draft")
    name, path, file_name, module, settings = load(target, overrides)
    today = today or date.today().isoformat()
    key = done_key(name, settings)
    context = Context(repo, store, settings, name, by, today, telemetry=telemetry)
    telemetry.input(source_drafter=name, file=file_name, file_sha256=sha256_file(path), settings=settings,
                    done_key=key)
    with telemetry.span("source_drafter.candidates"):
        found = list(module.candidates(context))
    if not all(isinstance(candidate, Candidate) for candidate in found):
        raise SourceDrafterError(f"{file_name}: candidates() gave something other than Candidates")
    names = [candidate.name for candidate in found]
    if len(set(names)) != len(names):
        raise SourceDrafterError(f"{file_name}: candidates() gave two candidates of one name")
    chosen = [c for c in found if c.name in only or c.source in only] if only else found
    unknown = sorted(set(only) - {c.name for c in chosen} - {c.source for c in chosen})
    if unknown:
        telemetry.log(f"not among the candidates: {', '.join(unknown)}", error=True)
    left = chosen if again else [c for c in chosen if store.get_ref(DONE, _done_ref(key, c.name)) is None]
    counts = {kind: 0 for kind in KINDS} | {"done before": len(chosen) - len(left)}
    summary = {"candidates": len(found), "chosen": len(chosen), "done_before": counts["done before"],
               "to_draft": len(left)}
    telemetry.write_file(CANDIDATES, json_text(summary, indent=2).encode() + b"\n")
    telemetry.log(f"{name}: {len(found)} candidates, {len(chosen)} chosen, {counts['done before']} done before, "
                  f"{len(left)} to draft")
    # a candidate whose existing source has patches stops before its draft
    stopped = {c.name: line for c in left if c.source and repo.exists("source", c.source)
               and (line := _with_patches(repo, c.source)) is not None}
    for candidate in left:
        if candidate.name in stopped:
            _record(store, telemetry, key, candidate, stopped[candidate.name], counts)
    drafting = [c for c in left if c.name not in stopped]
    for candidate, outcome in _drafts(module, path, drafting, context):
        _record(store, telemetry, key, candidate, _outcome(repo, candidate, outcome, by, today), counts)
    telemetry.log(f"{name}: " + ", ".join(f"{count} {kind}" for kind, count in counts.items()))
    return counts


def _outcome(repo, candidate, outcome, by, today) -> dict:
    """The outcomes.jsonl line of a draft()'s outcome, the draft saved."""
    if isinstance(outcome, dict):  # a failure, already a line
        return outcome
    if isinstance(outcome, Skipped):
        return _line("skipped", candidate.source, None, outcome.reason)
    if isinstance(outcome, NeedsPerson):
        return _line("needs-a-person", candidate.source, None, outcome.reason, outcome.notes)
    try:
        return save(repo, candidate, outcome, by, today)
    except DatasetError as error:
        return _line("failed", candidate.source, None, f"{type(error).__name__}: {error}", outcome.notes)


def _record(store, telemetry, key, candidate, line, counts):
    """Record one outcome: in the run's record, then, unless it failed, in
    the record of what is done."""
    line = {"candidate": candidate.name, **line}
    data = json_text(line).encode()
    telemetry.append_file(OUTCOMES, data + b"\n")
    said = f" {line['source']}" if line["source"] else ""
    said += f" ({line['change']})" if line["change"] else ""
    telemetry.log(f"{candidate.name}: {line['outcome']}{said}" + (f": {line['reason']}" if line["reason"] else ""),
                  error=line["outcome"] == "failed")
    counts[line["outcome"]] += 1
    if line["outcome"] != "failed":
        set_ref(store, DONE, _done_ref(key, candidate.name), store.put_bytes(data), telemetry)


# -- reading runs back, for wpt-web --------------------------------------------------------

def runs(store) -> list[str]:
    """The record folders of the source drafter runs in the store, oldest
    first (their names start with their UTC time, to the second)."""
    return [folder for folder in store.record_folders() if f"-{COMMAND}-" in folder]


def report(store, folder: str) -> dict:
    """What a run's record says: {folder, started, arguments, inputs (with
    source_drafter, settings, done_key, started_by), candidates
    (candidates.json, or None before it is written), counts ({outcome:
    count}), outcomes (every outcomes.jsonl line; one cut short by a stop
    left out), end (end.json, or None while it runs or after a crash)}.
    MissingObject (wpt_store) when the folder is no run's record."""
    from wpt_store import MissingObject
    from wpt_telemetry import END, IDS, START

    if f"-{COMMAND}-" not in folder:
        raise MissingObject(f"{folder} is not the record of a source drafter run")
    files = store.record_files(folder)
    start = json.loads(store.read_record(folder, START))
    ids = json.loads(store.read_record(folder, IDS)) if IDS in files else {}
    lines = store.read_record(folder, OUTCOMES).decode("utf-8", errors="replace").split("\n") \
        if OUTCOMES in files else []
    outcomes = []
    for line in lines:
        try:
            outcomes.append(json.loads(line))
        except json.JSONDecodeError:  # empty, or the last line cut short by a stop
            pass
    counts = {kind: sum(1 for o in outcomes if o["outcome"] == kind) for kind in KINDS}
    return {"folder": folder, "started": start["started"], "arguments": start["arguments"], "inputs": {**start["inputs"], **ids.get("inputs", {})},
            "candidates": json.loads(store.read_record(folder, CANDIDATES)) if CANDIDATES in files else None,
            "counts": counts, "outcomes": outcomes,
            "end": json.loads(store.read_record(folder, END)) if END in files else None}


def to_resume(store, holder: str) -> list[dict]:
    """The reports of the runs holder started (their started_by: holder, a
    dash and a token) that did not finish (no end.json, or stopped: exit
    code 130), each the latest of those holder started with its arguments:
    what holder starts again. A record that cannot be read (cut short
    before its start, or not readable) is left out, never hiding the
    others."""
    from wpt_store import CorruptObject, MissingObject

    found = []
    for folder in runs(store):
        try:
            found.append(report(store, folder))
        except (OSError, ValueError, KeyError, MissingObject, CorruptObject):  # nothing to resume there
            continue
    latest = {}
    for run in sorted(found, key=lambda run: run["started"]):  # to the microsecond; folders, to the second
        if str(run["inputs"].get("started_by", "")).rpartition("-")[0] == holder:
            latest[tuple(rerun(run["arguments"]))] = run
    return [run for run in latest.values() if run["end"] is None or run["end"].get("exit_code") == 130]


def rerun(given: list[str]) -> list[str]:
    """The arguments that start a run again (as those of arguments()): its
    recorded ones from the command on, without what comes before it (such
    as --root) nor its --started-by."""
    given = given[given.index("source-drafter"):]
    if "--started-by" not in given:
        return given
    at = given.index("--started-by")
    return given[:at] + given[at + 2:]


def arguments(target: str, by: str, settings: dict, only=(), again=False, started_by="") -> list[str]:
    """The arguments of `wpt-dataset source-drafter` for a run."""
    found = ["source-drafter", target, "--by", by]
    for key, value in settings.items():
        found += ["--set", f"{key}={json.dumps(value) if not isinstance(value, str) else value}"]
    for name in only:
        found += ["--only", name]
    if again:
        found.append("--again")
    if started_by:
        found += ["--started-by", started_by]
    return found

