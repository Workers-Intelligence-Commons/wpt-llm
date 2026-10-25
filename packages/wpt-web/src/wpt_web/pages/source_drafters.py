"""The source drafter page's routes: the built-in source drafters and their
settings, a run started as a separate process (`wpt-dataset
source-drafter`, which saves every draft through the service layer), and
each run's progress read from its record in the store
(wpt_dataset.source_drafter.report).

A run started here is a process of this wpt-web: it stops when wpt-web
stops (Runs.stop), and Runs.resume, when wpt-web starts next, starts again
every run it started that did not finish, with the same arguments; the
record of what is done makes it draft only what is left. Each wpt-web
names its runs by a name of the checkout it holds while it runs
(Repo.hold_name), so it never starts again a run another wpt-web, here or
on another machine sharing the store, still runs."""

import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading

from wpt_dataset.errors import NotFound
from wpt_dataset.source_drafter import arguments, built_in_names, describe, load, report, rerun, runs, to_resume
from wpt_store import MissingObject

from ..server import Route
from .common import STRING, refusals

HOLDER = "wpt-web"  # the prefix of the name this wpt-web holds; a run's started_by is that name, a dash and a token
STOP_SECONDS = 30  # a run asked to stop that is still running after this long is killed
ERROR_BYTES = 20_000  # how much of the end of a run's error output the page shows when it left no record


class Runs:
    """The source drafter runs this wpt-web started: one process each, by
    the started_by its record holds."""

    def __init__(self, repo):
        self.repo = repo
        self._holder = None  # the name this wpt-web holds, taken at its first run
        self._processes: dict[str, tuple[subprocess.Popen, object]] = {}  # started_by: process, its error output
        self._lock = threading.Lock()

    def holder(self) -> str:
        with self._lock:
            if self._holder is None:
                self._holder = self.repo.hold_name(HOLDER)
            return self._holder

    def start(self, given: list[str]) -> str:
        """Start `wpt-dataset source-drafter` with the given arguments (those
        of source_drafter.arguments, without --started-by), in a process
        group of its own, with its worker processes; returns the started_by
        its record names it by."""
        started_by = f"{self.holder()}-{secrets.token_hex(4)}"
        errors = tempfile.TemporaryFile()  # every printed line is in the run's record, once it has one
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
            else {"start_new_session": True}
        process = subprocess.Popen([sys.executable, "-m", "wpt_dataset.cli", "--root", str(self.repo.root), *given,
                                    "--started-by", started_by], cwd=self.repo.root, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=errors, **group)
        with self._lock:
            self._processes[started_by] = (process, errors)
        return started_by

    def running(self) -> set[str]:
        """The started_by of each run whose process still runs."""
        with self._lock:
            return {name for name, (process, _) in self._processes.items() if process.poll() is None}

    def ended(self) -> dict[str, dict]:
        """{started_by: {exit_code, error (the end of its error output)}} of
        each run whose process has ended."""
        with self._lock:
            found = [(name, process, errors) for name, (process, errors) in self._processes.items()
                     if process.poll() is not None]
        ended = {}
        for name, process, errors in found:
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - ERROR_BYTES))
            ended[name] = {"exit_code": process.returncode,
                           "error": errors.read().decode("utf-8", errors="replace").strip()}
        return ended

    def resume(self, store) -> list[str]:
        """Start again every run started under this wpt-web's name that did
        not finish (source_drafter.to_resume); returns their record
        folders."""
        found = to_resume(store, self.holder())
        for run in found:
            self.start(rerun(run["arguments"]))
        return [run["folder"] for run in found]

    def stop(self):
        """Stop every run still running, with its worker processes, as Ctrl+C
        would (killed after STOP_SECONDS): its record ends as stopped, and
        resume starts it again."""
        with self._lock:
            processes = [p for p, _ in self._processes.values() if p.poll() is None]
        for process in processes:
            _signal_group(process, stop=True)
        for process in processes:
            try:
                process.wait(STOP_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_group(process, stop=False)
                process.wait()


def _signal_group(process, stop: bool):
    """Ask a run's process group to stop (Ctrl+C; Ctrl+Break on Windows), or
    kill it, worker processes included."""
    try:
        if os.name == "nt":
            if stop:
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
        else:
            os.killpg(process.pid, signal.SIGINT if stop else signal.SIGKILL)
    except ProcessLookupError:  # the group ended meanwhile
        pass


SUMMARY = {"type": "object", "required": ["folder", "inputs", "candidates", "counts", "end", "running"],
           "properties": {"folder": STRING, "inputs": {"type": "object"},
                          "candidates": {"type": ["object", "null"]},
                          "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                          "end": {"type": ["object", "null"]}, "running": {"type": "boolean"}}}

NOT_STARTED = {"type": "object", "required": ["started_by", "exit_code", "error"],
               "properties": {"started_by": STRING, "exit_code": {"type": "integer"}, "error": {"type": "string"}}}


def routes(context, runs_here: Runs):
    def listed(values, body):
        return 200, [describe(name) for name in built_in_names()]

    @refusals
    def start(values, body):
        name = values["name"]
        if name not in built_in_names():
            raise NotFound(f"no built-in source drafter {name!r}")
        settings = {key: str(value) if not isinstance(value, str) else value for key, value in body["settings"].items()}
        load(name, settings)  # an unknown setting, or one its check refuses, is refused before any run starts
        given = arguments(name, body["by"], body["settings"], body["only"], body["again"])
        return 200, {"started_by": runs_here.start(given)}

    def summary(found, running):
        return {key: found[key] for key in ("folder", "inputs", "candidates", "counts", "end")} \
            | {"running": found["inputs"].get("started_by") in running}

    @refusals
    def all_runs(values, body):
        running, ended, found = runs_here.running(), runs_here.ended(), []
        for folder in reversed(runs(context.store)):
            try:
                found.append(summary(report(context.store, folder), running))
            except (MissingObject, ValueError, KeyError):  # a record cut short before its start
                continue
        recorded = {run["inputs"].get("started_by") for run in found}
        return 200, {"runs": found, "not_started": [{"started_by": name, **end} for name, end in ended.items()
                                                    if name not in recorded]}

    @refusals
    def one_run(values, body):
        try:
            found = report(context.store, values["folder"])
        except (MissingObject, ValueError) as error:
            raise NotFound(str(error)) from None
        return 200, {**summary(found, runs_here.running()), "arguments": found["arguments"],
                     "outcomes": found["outcomes"]}

    return [
        Route("GET", "/api/source-drafters", listed,
              "the built-in source drafters: name, file, what each drafts (summary) and its settings with their "
              "defaults",
              answer={"type": "array", "items": {"type": "object", "required": ["name", "summary", "settings"]}}),
        Route("POST", "/api/source-drafters/{name}/runs", start,
              "start a run of a built-in source drafter as a separate process (`wpt-dataset source-drafter`): its "
              "settings, who drafts (the decision's `by`), only these candidates (by name or source id; none: "
              "every one), again (draft those done before too); it stops when wpt-web stops and starts again when it "
              "starts next",
              body={"type": "object", "required": ["by", "settings", "only", "again"],
                    "additionalProperties": False,
                    "properties": {"by": {"type": "string", "minLength": 1},
                                   "settings": {"type": "object",
                                                "additionalProperties": {"type": ["string", "number", "boolean"]}},
                                   "only": {"type": "array", "items": {"type": "string", "minLength": 1}},
                                   "again": {"type": "boolean"}}},
              answer={"type": "object", "required": ["started_by"], "properties": {"started_by": STRING}}),
        Route("GET", "/api/source-drafter-runs", all_runs,
              "every source drafter run in the store, newest first, from its record: its inputs (source drafter, "
              "settings, started_by), its candidates (how many, chosen, done before, to draft), the count of each "
              "outcome, how it ended (null while it runs or after a crash), and whether this wpt-web runs it; and "
              "each run this wpt-web started that ended without a record (not_started): its started_by, exit code "
              "and the end of its error output",
              answer={"type": "object", "required": ["runs", "not_started"],
                      "properties": {"runs": {"type": "array", "items": SUMMARY},
                                     "not_started": {"type": "array", "items": NOT_STARTED}}}),
        Route("GET", "/api/source-drafter-runs/{folder}", one_run,
              "one run, as in the list, with its arguments and every outcome (candidate, outcome, source, change, "
              "reason, notes)",
              answer={**SUMMARY, "required": [*SUMMARY["required"], "arguments", "outcomes"]}),
    ]
