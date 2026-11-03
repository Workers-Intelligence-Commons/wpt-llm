"""The jobs the web server runs: a job is one `wpt` subcommand run as a child
process of the server, `python -m wpt <words> <args...>`, in the repository's
folder, with the arguments a request gives. The server only starts it and
lets people follow it; the subcommand does its own work, records its own run
in the store, and ends with its own exit code.

A job lives as long as its process. The server keeps what a running or ended
job printed (stdout and stderr, line by line, the last LINES of them) until
it stops; a job started by another server, or before this one started, is
unknown here. What stays is the run record the subcommand wrote in the store:
the server tells the job to the subcommand through the WPT_JOB setting
(wpt_telemetry), which its start.json then names, so a job's records are
found by that id. Any number of jobs run at once; none reads standard input.
"""

import json
import os
import secrets
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import UTC, datetime

from wpt_dataset.errors import NotFound
from wpt_store import MissingObject
from wpt_telemetry import END, IDS, JOB, START

LINES = 20_000  # the printed lines a job keeps; an older line is dropped when a newer one does not fit
STOP_SECONDS = 20  # how long a stopping server waits for the jobs it interrupted before it kills them
STAMP = "%Y%m%dT%H%M%SZ"  # the start of every record folder's name (wpt_telemetry.Telemetry.start)


class Job:
    """One child process and what it printed. Read it through view() and
    lines(); the threads that read its two output streams are the only
    writers."""

    def __init__(self, id, words, args, process, started):
        self.id, self.words, self.args, self.process, self.started = id, tuple(words), list(args), process, started
        self.ended = None  # set once the process has ended and all it printed is read
        self._lines, self._count, self._lock = deque(maxlen=LINES), 0, threading.Lock()

    def _add(self, stream, text):
        with self._lock:
            self._lines.append({"stream": stream, "text": text})
            self._count += 1

    def lines(self, start: int) -> dict:
        """The printed lines from number start on (the first printed is 0): {"first": the number of the first
        line returned, which is later than start when the lines before it were dropped, "next": the number to ask
        for next, "lines": [{"stream": "stdout" or "stderr", "text"}]}."""
        with self._lock:
            kept = list(self._lines)
            count = self._count
        first = min(max(start, count - len(kept)), count)
        return {"first": first, "next": count, "lines": kept[first - (count - len(kept)):]}

    def view(self) -> dict:
        """What is known of the job: {"id", "command" (as typed), "started", "ended" (null while it runs), "state"
        ("running", "succeeded" or "failed"), "exit_code" (null while it runs; negative: ended by that signal)}."""
        code = self.process.poll() if self.ended is not None else None
        state = "running" if code is None else "succeeded" if code == 0 else "failed"
        return {"id": self.id, "command": " ".join(["wpt", *self.words, *self.args]),
                "started": _iso(self.started), "ended": _iso(self.ended) if self.ended else None,
                "state": state, "exit_code": code}


class Jobs:
    """The jobs of one server, in the order they started. command is the
    program that runs a subcommand (default: this Python's `-m wpt`); root
    the folder they run in (the repository, which holds the .env that names
    the store)."""

    def __init__(self, root, command=(sys.executable, "-m", "wpt")):
        self.root, self.command = root, tuple(command)
        self._jobs, self._lock = {}, threading.Lock()

    def start(self, words, args) -> Job:
        """A new job: the subcommand words (`("dataset", "download")`) with args. OSError when the process cannot
        start."""
        id = secrets.token_hex(4)
        started = datetime.now(UTC)
        # unbuffered, so a line shows as soon as the subcommand prints it; the id is how its record names the job
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", JOB: id}
        process = subprocess.Popen([*self.command, *words, *args], cwd=self.root, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace")
        job = Job(id, words, args, process, started)
        with self._lock:
            self._jobs[id] = job
        threading.Thread(target=self._follow, args=(job,), daemon=True).start()
        return job

    def _follow(self, job):
        """Read both streams until they end, then wait for the process."""
        readers = [threading.Thread(target=self._read, args=(job, name, stream), daemon=True)
                   for name, stream in (("stdout", job.process.stdout), ("stderr", job.process.stderr))]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        job.process.wait()
        job.ended = datetime.now(UTC)

    @staticmethod
    def _read(job, name, stream):
        for line in stream:
            job._add(name, line.rstrip("\r\n"))
        stream.close()

    def get(self, id) -> Job:
        """The job of that id; NotFound when this server started none."""
        with self._lock:
            if id not in self._jobs:
                raise NotFound(f"no job {id!r} on this server (a job is known only to the server that started it, "
                               "until that server stops)")
            return self._jobs[id]

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def records(self, job, store) -> list[dict]:
        """The run records in the store that name the job, oldest first: [{"folder", "start", "ids" (null if not
        written yet), "end" (null while the run goes on)}]. A subcommand that writes no record leaves none; one that
        starts others (`wpt manager`) leaves the records of those too."""
        stamp = job.started.strftime(STAMP)
        found = []
        for folder in store.record_folders():
            if folder[:len(stamp)] < stamp:  # opened before the job started
                continue
            try:
                start = json.loads(store.read_record(folder, START))
            except (MissingObject, ValueError):  # not written yet, or not a run's folder
                continue
            if start.get("job") != job.id:
                continue
            files = store.record_files(folder)
            found.append({"folder": folder, "start": start,
                          **{name: json.loads(store.read_record(folder, file)) if file in files else None
                             for name, file in (("ids", IDS), ("end", END))}})
        return found

    def close(self):
        """Interrupt the jobs still running, as Ctrl+C would (the subcommand then records how it ended), and kill
        those that have not ended after STOP_SECONDS."""
        running = [job for job in self.all() if job.process.poll() is None]
        for job in running:
            if os.name == "posix":
                job.process.send_signal(signal.SIGINT)
            else:
                job.process.terminate()
        for job in running:
            try:
                job.process.wait(STOP_SECONDS)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait()


def _iso(moment) -> str:
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")
