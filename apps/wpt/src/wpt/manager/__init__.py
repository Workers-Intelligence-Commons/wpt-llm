"""The wpt manager: one command that downloads the sources, builds the
dataset and trains a model recipe, then keeps the model in the store under
the run's name (docs/machines.md).

Usage:
    uv run wpt manager
    docker compose --env-file .env -f deploy/manager/compose.yaml up      (the same, in its container)
    uv run wpt manager stop [RUN]
    docker compose exec manager uv run --frozen --no-sync wpt manager stop [RUN]    (in its container)

Settings, from the environment, else from the .env file at the
repository's root (an empty value counts as unset):

    WPT_RECIPE        the model recipe, a file name in data/recipes/models/ without .json, or
                      SERIES:SIZE, a size of a series file of data/recipes/series/, or
                      SERIES:SIZE:RUN, another run of that size;
                      required: the manager refuses to start without it
    WPT_SESSION       the session the run works in; unset, the one of the run it goes on with
                      (the run's file, or RUN_FILE), else a new ID, which it prints
    WPT_FROM          another session, which every command of the run reads what the session
                      does not hold from
    WPT_RUN           the run's name (letters, digits, '.', '_', '-'); unset, the manager goes on
                      with the run RUN_FILE names (at the repository's root, git-ignored) when
                      it is a run of this recipe whose model is not in the store yet, else makes
                      a new name and writes it there, with the command to copy and paste that
                      copies its model out of the store
    WPT_EVALUATE      1: evaluate the model at the end (wpt eval); off by default
    WPT_MANAGER_FOLDER  where the model folders are written before they go into the store
                      (default runs/manager in the repository)
    WPT_MANAGER_COMMAND  how a command is run where the manager runs, for the lines it prints
                      (default "uv run"; the Compose file sets its own)
    WPT_MANAGER_STATE where the manager keeps its runs' gpu workers, tokens included, never in the
                      store, and the lock of each run it runs (default runs/manager-state in the
                      repository; machines.py)
    WPT_MANAGER_CONTAINER
                      1 (the Compose file sets it): the manager is the container's one process;
                      a run that ended, failed or not, or was refused, makes it exit with code 0
                      (the failure is in its output and its record), so that the Compose
                      file's restart: on-failure restarts it only after a crash or a Docker
                      restart; `docker compose up` again goes on with a failed run
    WPT_GPU_WORKERS   the run's gpu workers: their keys, or how many to rent; unset, a local one
                      (machines.py)
    WPT_RENTAL_PROVIDER, WPT_RENTAL_KIND, WPT_RENTAL_GPUS, WPT_RENTAL_WAIT
                      how the workers are rented (rental.py); a provider's own keys come from
                      the environment or .env too
    WPT_BUDGET, WPT_MAX_HOURS
                      the cost cap, in dollars, in hours, or both (rental.py): the first reached
                      stops the run as `wpt manager stop` does, over all its starts
    WPT_STOP_DEADLINE seconds a stop or a cap gives the command it stops, then the manager ends it
                      (default 600; rental.py)
    WPT_REPLACEMENTS  how many times a run resumes after a lost gpu worker (default 3)
    WPT_STORE, WPT_LINK_CHECKPOINTS and every WPT_ option of wpt train
                      as each command reads them: from the environment, else from the same
                      .env file

In a container (a /.dockerenv file), a file:// store must be on a mounted
volume or folder: one in the container's own layer would go with the
container, so the manager refuses it.

The steps, each its own command, in order, each stopping the run with its
exit code when it fails:

    1. wpt dataset download              every admitted source, into the session
    2. wpt dataset build RECIPE          the dataset; its id is the last line it prints
                                         (WPT_DATASET names a dataset the session holds: neither
                                         step runs, and the run trains on that one)
       wpt agent build-image NAME        each image the dataset's tasks name, where DOCKER_HOST
                                         reaches Docker (the sandboxes' machine); none without tasks
    3. wpt train RECIPE ID --run NAME --resume --out FOLDER/NAME
                                         through the run's gpu workers, rented first when
                                         WPT_GPU_WORKERS asks, with WPT_KERNELS=triton on rented
                                         ones unless the settings set WPT_KERNELS; the run's
                                         checkpoint is the store's, under NAME, so the same command
                                         with the same WPT_RUN continues it. A gpu worker lost
                                         (wpt train's exit code 4) that the manager rented is
                                         reported, given back and
                                         replaced, and the step runs again from the last
                                         checkpoint in the store; so does a crash of a training
                                         process on one of several workers (exit code 5), on the
                                         workers that answer; both count against the same limit,
                                         WPT_REPLACEMENTS times
    4. wpt eval FOLDER/NAME              with WPT_EVALUATE only, on the run's first gpu worker,
                                         with WPT_KERNELS=triton on a rented one unless the
                                         settings set WPT_KERNELS
    5. the model folder into the session, as models/NAME (wpt_store.models), recorded first;
       then the folder is removed: the store holds it

The rentals are stopped once the evaluation (or, without one, the training)
ends, whatever its outcome. A rental that cannot be stopped stays named in
its run's file; the manager tries to stop every rental still named in the
file of a run that ended at each start and, until none is left, every SWEEP
seconds, each try recorded (stop_rentals()).

`wpt manager stop` stops the run a manager runs now (RUN: that run, when
several managers share WPT_MANAGER_STATE; refused, exit code 2, when no
manager runs one). The manager passes the stop on to wpt train as a stop
signal: the gpu worker finishes its update, saves a checkpoint, even inside
a step, and sends it to the store, wpt train exits with code 130, and the
rentals are given back. The evaluation is stopped by the same signal.
A run that holds no rental yet waits for the download and the build, and
then rents nothing. A run that still holds rentals when it goes on after a
crash pays for them meanwhile, so those commands get the same signal, and
its rentals are given back. Renting is looked at too: the provider's own
rent() ends first, and the wait for a worker's address is cut short. The
cost cap stops a run the same way, and so does a SIGTERM to this process
(`docker compose stop`), after which no other run starts; a lost worker is
not replaced once a stop is under way. The run's exit code is 130, and the
same command goes on from the checkpoint, once a cap reached is raised.
A run a crash cut short that a SIGTERM reaches before it went on starts no
command: its rentals are given back, and one that holds none is left for
the next start. The manager's exit code is 130 when the SIGTERM stopped a
run or kept one from starting, not when it came too late to do either.

A command that has not ended WPT_STOP_DEADLINE seconds after the stop is
killed, and the rentals are given back: what it had not sent yet is lost,
which the log says. rent() is outside the deadline: a provider's launch
goes on after the command that asked for it is killed, so only its return
can be waited for.

The manager records its own run in the session (<id>/runs/, command
wpt manager), each step a span, and prints the line that copies the model
out of the store (wpt store copy-out).

When it starts, the manager first goes on with each run of its own
(WPT_MANAGER_STATE) that did not end: its file is not marked ended, no
other manager holds its lock, no model under its name is in the store,
and its last record there has no end (unfinished()). It reconnects to
that run's gpu workers, whose keys its own disk holds; a rented one that
is gone by then is replaced as a lost one. Then it starts the run the
settings ask for, unless that is the run it went on with, or the settings
name none (no WPT_RUN) and it went on with a run: it has done its work.
Refused (exit code 2) when another manager runs a run of that name now.

    rental.py    the interface of the providers that rent gpu workers
    skypilot.py  SkyPilot's
    machines.py  the runs' gpu workers, on the manager's disk
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_store import NAME as STORE_NAME
from wpt_store import (MissingFile, StoreError, find_dataset, given, new_id, read_env_file, started,
                       store_from_settings)
from wpt_store.sessions import SESSION
from wpt_store.models import has_model, keep_model, put_model
from wpt_store.names import as_record_name
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry
from wpt.train import CRASHED, LOST
from wpt_training.launch import STOPPED, ask_to_stop
from wpt_utils.lines import ERROR, WARNING, bare, say
from wpt_worker.manager import WORKERS

from . import machines, rental

REFUSED = 2
RUN_FILE = "wpt-run.txt"  # at the repository's root, git-ignored: the run name the manager made
DEFAULT_FOLDER = Path("runs") / "manager"
DEFAULT_STATE = Path("runs") / "manager-state"  # the runs' gpu workers, their tokens included (machines.py)
CONTAINER_SETTING = "WPT_MANAGER_CONTAINER"
KERNELS, RENTED_KERNELS = "WPT_KERNELS", "triton"  # wpt train's --kernels, on rented gpu workers
SWEEP = 300.0  # seconds between two tries to stop the rentals that could not be stopped
WATCH = 5.0  # seconds between two looks at whether a run must stop, while a command it runs goes on
RUN_LINE = re.compile(r"# run (\S+) of the recipe (\S+) in session (\S+);")  # the first line of RUN_FILE
COMMAND = ["-m", "wpt"]  # the wpt command, by the Python that runs it: each step is one of its subcommands
CONTAINER = Path("/.dockerenv")  # Docker's mark that this process runs in a container
MOUNTS = Path("/proc/self/mountinfo")


class Failed(Exception):
    """A step's command failed: code is its exit code."""

    def __init__(self, step, code):
        super().__init__(f"{step} failed with exit code {code}")
        self.step, self.code = step, code


def made_name(recipe, moment=None) -> str:
    """A new run's name: the recipe's and the UTC time."""
    moment = moment or datetime.now(UTC)
    return f"{as_record_name(recipe)}-{moment:%Y%m%dT%H%M%SZ}"


def copy_out_line(name, session, environ) -> str:
    """The command a person copies to have the run's model in runs/NAME of
    the repository, once the run is done."""
    command = environ.get("WPT_MANAGER_COMMAND") or "uv run"
    return f"{command} wpt store copy-out {name} runs/{name} --session {session}"


class StopSignal:
    """While active, a SIGTERM only sets `asked`, each time, so that the
    run stops as `wpt manager stop` stops it. Signals reach Python's main
    thread only: elsewhere this does nothing. An ignored SIGTERM stays
    ignored."""

    def __init__(self):
        self.asked, self.previous = False, None

    def __enter__(self):
        if (threading.current_thread() is threading.main_thread()
                and signal.getsignal(signal.SIGTERM) is not signal.SIG_IGN):
            self.previous = signal.signal(signal.SIGTERM, self._handle)
        return self

    def _handle(self, number, frame):
        self.asked = True

    def __exit__(self, *exc):
        if self.previous is not None:
            signal.signal(signal.SIGTERM, self.previous)
        return False


class Stops:
    """Whether the run must stop before its end: `wpt manager stop` asked,
    a SIGTERM came (signalled), or its rentals reached the cost cap. ends:
    the time.monotonic() at which the stop's deadline passes, once there is
    a stop."""

    def __init__(self, workers, cap, deadline, telemetry, signalled):
        self.workers, self.cap, self.deadline, self.telemetry = workers, cap, deadline, telemetry
        self.signalled, self.reason, self.ends = signalled, None, None

    def check(self) -> str | None:
        """What stops the run, recorded once; None while nothing does."""
        if self.reason is None:
            self.reason = ("wpt manager stop" if self.workers.stop_asked()
                           else "SIGTERM" if self.signalled.asked
                           else self.cap.reached(*self.workers.spent()))
            if self.reason is not None:
                self.ends = time.monotonic() + self.deadline
                self.telemetry.event("manager.stop", reason=self.reason)
                self.telemetry.log(f"the run stops: {self.reason}", WARNING)
        return self.reason

    def left(self) -> float | None:
        """The seconds left before the deadline; None without a stop."""
        return None if self.ends is None else max(self.ends - time.monotonic(), 0.0)

    def overdue(self):
        """Record that the deadline passed, with what that costs."""
        self.telemetry.event("manager.stop_deadline", deadline=self.deadline)
        self.telemetry.log(f"the stop's deadline of {self.deadline:.0f} s passed (WPT_STOP_DEADLINE): the manager ends "
                           "the command and gives the machines back; the updates since the last checkpoint sent "
                           "are lost", ERROR)

    def halt(self, step="workers"):
        """Failed, with STOPPED, once the run must stop."""
        if self.check():
            raise Failed(step, STOPPED)


def run_command(step, arguments, log, last_line=False, environ=None, stops=None):
    """Run a subcommand of the wpt command (arguments: its name and its
    arguments), what it prints shown as it comes; with last_line, the last
    non-empty line it printed; environ: settings it gets besides this
    process's environment; stops: a Stops that, once it says so, sends the
    command one stop signal, and kills it at the stop's deadline. Failed
    when it fails, with STOPPED when that stop ended it."""
    command = [sys.executable, *COMMAND, *arguments]
    log(f"{step}: wpt {' '.join(arguments)}")
    env = {**os.environ, **(environ or {})}
    # on Windows, the one way to reach it with a stop signal is a process group of its own
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if stops and os.name == "nt" else {}
    piped = {"stdout": subprocess.PIPE, "text": True, "encoding": "utf-8", "errors": "replace"} if last_line else {}
    lines = []
    with subprocess.Popen(command, env=env, **group, **piped) as process:
        reader = threading.Thread(target=shown, args=(process.stdout, lines), daemon=True) if last_line else None
        if reader is not None:
            reader.start()
        try:
            code = watched(process, stops)
        except BaseException:
            process.kill()
            raise
        finally:
            if reader is not None:
                reader.join(stops.left() if stops is not None else None)
                if reader.is_alive():  # a process the command left holds its output: closing that would wait for it
                    process.stdout = None
    if code:
        raise Failed(step, STOPPED if stops is not None and stops.reason else code)
    return bare(lines[-1]) if lines else None


def shown(stream, lines):
    """Print what stream gives as it comes, its non-empty lines kept in
    lines."""
    for line in stream:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.strip():
            lines.append(line.strip())


def watched(process, stops) -> int:
    """Wait for process to end, looking every WATCH seconds at whether it
    must stop: one signal, since a second one ends it at once, without
    its checkpoint, and at the stop's deadline, killed. Its exit code."""
    asked = False
    while True:
        try:
            return process.wait(WATCH)
        except subprocess.TimeoutExpired:
            if stops is None:
                continue
            if not asked and stops.check():
                asked = True
                ask_to_stop(process)
            elif asked and not stops.left():
                stops.overdue()
                process.kill()
                return process.wait()


def settings(root) -> dict:
    """The environment, and for what it leaves unset or empty, the .env
    file at root."""
    return {**read_env_file(root / ".env"), **{key: value for key, value in os.environ.items() if value}}


def mount_point(path, mountinfo: str) -> str:
    """The mount point, among those of mountinfo (/proc/self/mountinfo's
    text), that holds path: the longest one path is in."""
    points = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) > 4:  # the fifth field, with its spaces and other bytes written as \ooo
            points.append(re.sub(r"\\([0-7]{3})", lambda found: chr(int(found[1], 8)), fields[4]))
    path = Path(path)
    return max((point for point in points if path == Path(point) or Path(point) in path.parents),
               key=len, default="/")


def store_in_layer(url) -> str | None:
    """In a container, the folder of a file:// store that is in the
    container's own layer (no mount holds it, so it goes with the
    container); None otherwise."""
    if not url.startswith("file://") or not CONTAINER.exists() or not MOUNTS.is_file():
        return None
    folder = Path(url.removeprefix("file://")).absolute()
    return str(folder) if mount_point(folder, MOUNTS.read_text(encoding="utf-8")) == "/" else None


def main(argv=None):
    with StopSignal() as signalled:
        code = _main(argv, signalled)
    if code:
        sys.exit(code)


def _main(argv, signalled) -> int:
    """The command; its exit code."""
    try:
        root = Repo.find().root
    except DatasetError as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return REFUSED
    environ = settings(root)
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments[:1] == ["stop"]:
        return stop_command(arguments[1:], Path(environ.get("WPT_MANAGER_STATE") or root / DEFAULT_STATE))
    container = environ.get(CONTAINER_SETTING, "") in ("1", "true", "yes")

    def leave(code):
        """The exit code: in the container, 0 for a run that ended or was
        refused, whatever its outcome, so that restart: on-failure restarts
        only a crash."""
        if code and container:
            say(f"wpt manager stopped with exit code {code}; this container exits with 0, so Docker does not "
                "start it again: `docker compose up` goes on with the run", ERROR)
            return 0
        return code

    recipe = environ.get("WPT_RECIPE", "")
    if not recipe:
        say("WPT_RECIPE is not set: name the model recipe to train (a file name in data/recipes/models/, without "
            ".json, or SERIES:SIZE[:RUN], a size of a series of data/recipes/series/ or another run of it), in the "
            "environment or in .env", ERROR)
        return leave(REFUSED)
    try:
        Repo(root).model_recipe(recipe)  # refused now, not after the download
    except DatasetError as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return leave(REFUSED)
    name = environ.get("WPT_RUN") or None
    if name is not None and not STORE_NAME.fullmatch(name):
        say(f"WPT_RUN={name!r}: use letters, digits, '.', '_' and '-' only", ERROR)
        return leave(REFUSED)
    try:
        rental.kind(environ)  # refused now, not when the workers are rented
        rental.wait(environ)
        rental.cap(environ)
        rental.deadline(environ)
        machines.replacements(environ)
    except (rental.RentalError, machines.WorkersError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return leave(REFUSED)
    layer = store_in_layer(environ.get("WPT_STORE", ""))
    if layer is not None:
        say(f"WPT_STORE names {layer}, which is in this container's own files, deleted with the container: name "
            "a bucket, or leave WPT_STORE unset for the volume wpt-store, or set WPT_STORE_FOLDER to that folder "
            "on this machine to mount it at the same path", ERROR)
        return leave(REFUSED)
    folder = Path(environ.get("WPT_MANAGER_FOLDER") or root / DEFAULT_FOLDER)
    kept = Path(environ.get("WPT_MANAGER_STATE") or root / DEFAULT_STATE)
    try:
        whole = store_from_settings(root / ".env")
        named = given(environ=environ, env_file=root / ".env")
        start = started(environ=environ, env_file=root / ".env")
        previous = previous_run(root, recipe, whole, name, named, start)
        session = named or (previous and previous[1]) or (name and machines.session_of(kept, name))
        if not session:
            session = new_id()
            say(f"session {session}")
        store = whole.at(session, start)
    except (StoreError, ValueError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return leave(REFUSED)
    evaluate = environ.get("WPT_EVALUATE", "") in ("1", "true", "yes")
    code, continued, postponed = 0, [], False
    stop_rentals(store, kept, environ, root, arguments)  # those a run that ended could not stop
    for path in machines.files(kept):  # the runs a crash or a stop cut short go on first
        workers = machines.Workers.claim(path)
        if workers is None:  # another manager runs it now
            continue
        try:
            own = whole.at(workers.state["session"], start)
            if not unfinished(own, workers.run):
                workers.end()
            elif signalled.asked and not workers.groups():
                postponed = True  # it costs nothing: the next start goes on with it
            else:  # after a SIGTERM, one_run() starts no command and its rentals are given back
                say(f"run {workers.run} did not end: it goes on, reconnecting to its gpu workers")
                code = one_run(workers.state["recipe"], workers.run, folder, evaluate, own, environ, root,
                               workers, arguments, signalled)
                continued.append(workers.run)
        finally:
            workers.release()
    # with no WPT_RUN, a start that went on with runs a crash cut short has done its work: the container
    # Docker starts again after a crash does not start another run
    wanted = not continued if name is None else name not in continued
    if signalled.asked:  # no other run starts; a run that finished before the signal keeps its exit code
        if wanted or postponed:
            code = code or STOPPED
    elif wanted:
        if name is None and previous is not None:
            name = previous[0]
            say(f"run {name} (named in {root / RUN_FILE}, its model is not in the store yet)")
        elif name is None:
            name = made_name(recipe)
            (root / RUN_FILE).write_text(f"# run {name} of the recipe {recipe} in session {session}; once it is done, "
                                         f"this copies its model out of the store:\n"
                                         f"{copy_out_line(name, session, environ)}\n", encoding="utf-8")
            say(f"run {name} (written into {root / RUN_FILE})")
        workers = machines.Workers.new(kept, name, recipe, session)
        if workers is None:
            say(f"run {name} is another wpt manager's, which runs it now (its lock in {kept} is held): this "
                "one leaves it", ERROR)
            code = REFUSED
        else:
            try:
                code = one_run(recipe, name, folder, evaluate, store, environ, root, workers, arguments, signalled)
            finally:
                workers.release()
    left = stop_rentals(store, kept, environ, root, arguments)
    while left:  # money is spent while a rental runs: the manager stays, trying
        say(f"{left} rental(s) are still rented: the manager tries to stop them again in {SWEEP:.0f} s", WARNING,
            sys.stdout)
        time.sleep(SWEEP)
        left = stop_rentals(store, kept, environ, root, arguments)
    return leave(code)


def stop_command(arguments, kept) -> int:
    """`wpt manager stop [RUN]`; its exit code."""
    if len(arguments) > 1:
        say("wpt manager stop [RUN]: at most one run's name", ERROR)
        return REFUSED
    run = arguments[0] if arguments else None
    asked = machines.ask_to_stop(kept, run)
    if not asked:
        say(f"no wpt manager runs {'a run' if run is None else f'the run {run}'} now (in {kept})", ERROR)
        return REFUSED
    say(f"asked to stop: {', '.join(asked)}; the manager saves the checkpoint, then gives the machines back")
    return 0


def previous_run(root, recipe, store, name, session, start) -> tuple[str, str] | None:
    """(the run, its session) RUN_FILE names when it is a run of recipe
    whose model is not in that session: a run that did not end well, which
    the same command goes on with. name, session: what the settings name,
    which it must then be. start: the session the run starts from."""
    try:
        first = (root / RUN_FILE).read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    found = RUN_LINE.match(first)
    if (found and found[2] == recipe and STORE_NAME.fullmatch(found[1]) and name in (None, found[1])
            and session in (None, found[3]) and not has_model(store.at(found[3], start), found[1])):
        return found[1], found[3]
    return None


def stop_rentals(store, kept, environ, root, arguments) -> int:
    """Try to stop every rental still named in the file of a run that ended
    (machines.with_rentals): the manager could not stop them when the run
    ended. Each try is in a record of its own (command wpt manager). Returns
    how many rentals are still named afterwards; the files of runs another
    manager holds are left to it."""
    waiting = machines.with_rentals(kept)
    if not waiting:
        return 0
    left = 0
    try:
        with Telemetry.start(store, "wpt manager", arguments, "rentals",
                             inputs={"stop_rentals": [path.stem for path in waiting]},
                             env_file=root / ".env") as telemetry:
            for path in waiting:
                workers = machines.Workers.claim(path)
                if workers is None:
                    continue
                try:
                    try:
                        workers.stop(environ, root, telemetry)
                    except rental.RentalError as error:
                        telemetry.log(f"RentalError: {error}", ERROR)
                    left += len(workers.groups())
                finally:
                    workers.release()
    except RecordError as error:
        say(f"RecordError: {error}; the rentals are not stopped without a record of each try", ERROR)
        return len(waiting)
    return left


def unfinished(store, name) -> bool:
    """Whether the run name did not end: no model of that name in the
    store, and no record of a wpt manager run of it, or the last one has no
    end (a crash or a stop cut it short)."""
    if has_model(store, name):
        return False
    found = []
    for folder in store.record_folders():
        if "-wpt-manager-" not in folder:
            continue
        try:
            start = json.loads(store.read_record(folder, "start.json"))
        except (MissingFile, ValueError):
            continue
        if start.get("command") == "wpt manager" and start.get("inputs", {}).get("run") == name:
            found.append((start.get("started", ""), folder))
    return not found or "end.json" not in store.record_files(max(found)[1])


def one_run(recipe, name, folder, evaluate, store, environ, root, workers, arguments, signalled) -> int:
    """One run, recorded, its file then marked ended (whatever its
    outcome: a person starts it again with the same WPT_RUN); its exit
    code."""
    try:
        with Telemetry.start(store, "wpt manager", arguments, name, inputs={"recipe": recipe, "run": name},
                             env_file=root / ".env") as telemetry:
            code = _run(recipe, name, folder / name, evaluate, store, telemetry, environ, root, workers, signalled)
            if code:
                raise SystemExit(code)  # recorded as the run's exit code
        code = 0
    except SystemExit as ended:
        code = ended.code
    except RecordError as error:
        say(f"RecordError: {error}; stopped, nothing more was recorded", ERROR)
        return RECORD_FAILED  # no end recorded: the manager that comes back goes on with it
    workers.end()
    return code


def task_images(store, dataset) -> list[str]:
    """The images the dataset's tasks name, each once, in order."""
    _, record = find_dataset(store, dataset)
    part = record.get("tasks")
    lines = [] if part is None else store.get_bytes(part["file"]).decode("utf-8").splitlines()
    return sorted({json.loads(line)["image"] for line in lines if line.strip()})


def _run(recipe, name, out, evaluate, store, telemetry, environ, root, workers, signalled) -> int:
    log, in_session = telemetry.log, {SESSION: store.session}
    stops = Stops(workers, rental.cap(environ), rental.deadline(environ), telemetry, signalled)
    held = stops if workers.groups() else None  # rentals a crash left: the commands before training pay for them too
    try:
        if held is not None and signalled.asked:  # the run's file, marked ended, has its rentals given back
            held.halt("download")
        dataset = environ.get("WPT_DATASET") or None
        if dataset is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", dataset) or find_dataset(store, dataset) is None:
                log(f"WPT_DATASET: session {store.session} holds no dataset {dataset!r}", ERROR)
                return REFUSED
            log(f"dataset {dataset}, as WPT_DATASET names: nothing downloaded or built")
        else:
            with telemetry.span("manager.download"):
                run_command("download", ["dataset", "download"], log, environ=in_session, stops=held)
            with telemetry.span("manager.build", recipe=recipe) as span:
                dataset = run_command("build", ["dataset", "build", recipe], log, last_line=True, environ=in_session,
                                      stops=held)
                if dataset is None or not re.fullmatch(r"[0-9a-f]{64}", dataset):
                    log(f"build: wpt dataset build printed no dataset id last (it printed {dataset!r})", ERROR)
                    return REFUSED
                span.set(dataset=dataset)
        telemetry.input(dataset=dataset)
        images = task_images(store, dataset)
        if images:  # the sandboxes' images, built where the sandboxes run
            with telemetry.span("manager.images", images=images):
                for image in images:
                    run_command("images", ["agent", "build-image", image], log, environ=in_session, stops=held)
        try:
            _train(recipe, name, out, dataset, telemetry, {**environ, **in_session}, root, workers, stops)
            if evaluate:  # on the run's gpu workers, rented ones included, which are given back after it
                with telemetry.span("manager.evaluate", folder=str(out)):
                    run_command("evaluate", ["eval", str(out)], log,
                                environ={**in_session, WORKERS: workers.text(), **kernels(workers, environ)},
                                stops=stops)
        finally:
            _stop_workers(workers, environ, root, telemetry)
    except Failed as error:
        log(f"{error}: the run stops here; the same command goes on from what is kept", ERROR)
        return error.code
    with telemetry.span("manager.model", folder=str(out)) as span:
        files = put_model(store, name, out)
        telemetry.irreversible("store.model_kept", model=name, session=store.session, files=len(files))
        keep_model(store, name, files)
        span.set(files=len(files))
    telemetry.output(model=f"{store.session}/{name}")
    shutil.rmtree(out)  # the store holds it: copy-out gives it back
    log(f"model {name} kept in session {store.session} of {store}; copy it out with:")
    log(copy_out_line(name, store.session, environ))
    return 0


def _train(recipe, name, out, dataset, telemetry, environ, root, workers, stops):
    """wpt train through the run's gpu workers, rented when the settings
    ask, resumed when a gpu worker it rented is lost or a training process
    crashed on one of several workers, a lost rented worker replaced, up to
    WPT_REPLACEMENTS times, unless stops says the run stops. Failed when it
    fails, and with STOPPED when it must stop, before anything more is
    rented."""
    log = telemetry.log
    stops.halt("train")
    try:
        with telemetry.span("manager.workers"):
            workers.prepare(environ, root, telemetry, stops.halt)
    except (rental.RentalError, machines.WorkersError) as error:
        log(f"{type(error).__name__}: {error}", ERROR)
        raise Failed("workers", REFUSED) from None
    while True:
        try:
            with telemetry.span("manager.train", dataset=dataset, folder=str(out), workers=len(workers.keys())):
                run_command("train", ["train", recipe, dataset, "--run", name, "--resume", "--out", str(out)],
                            log, environ={SESSION: environ[SESSION], WORKERS: workers.text(),
                                          **kernels(workers, environ)}, stops=stops)
            return
        except Failed as error:
            if error.code not in (LOST, CRASHED) or stops.check():
                raise
            try:
                if not workers.recover(environ, root, telemetry, stops.halt):
                    raise
            except rental.RentalError as failed:
                log(f"RentalError: {failed}", ERROR)
                raise error from None
            log("the run resumes from the last checkpoint in the store")


def kernels(workers, environ) -> dict:
    """WPT_KERNELS=triton for a training or an evaluation on rented gpu
    workers, unless the settings name the attention's form themselves
    (docs/machines.md)."""
    return {KERNELS: RENTED_KERNELS} if workers.groups() and not environ.get(KERNELS) else {}


def _stop_workers(workers, environ, root, telemetry):
    """Stop the run's rentals; those that cannot be stopped stay named in
    its file, and stop_rentals() tries them again."""
    try:
        workers.stop(environ, root, telemetry)
    except rental.RentalError as error:
        telemetry.log(f"RentalError: {error}; they stay named in the run's file, and the manager tries to stop them "
                      f"again every {SWEEP:.0f} s", ERROR)
