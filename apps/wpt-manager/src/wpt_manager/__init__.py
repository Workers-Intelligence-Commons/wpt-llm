"""The wpt manager: one command that downloads the sources, builds the
dataset and trains a model recipe, then keeps the model in the store under
the run's name (docs/machines.md).

Usage:
    uv run wpt-manager
    docker compose --env-file .env -f deploy/manager/compose.yaml up      (the same, in its container)

Settings, from the environment, else from the .env file at the
repository's root (an empty value counts as unset):

    WPT_RECIPE        the model recipe, a file name in data/recipes/models/ without .json;
                      required: the manager refuses to start without it
    WPT_RUN           the run's name (letters, digits, '.', '_', '-'); unset, the manager goes on
                      with the run RUN_FILE names (at the repository's root, git-ignored) when
                      it is a run of this recipe whose model is not in the store yet, else makes
                      a new name and writes it there, with the command to copy and paste that
                      copies its model out of the store
    WPT_EVALUATE      1: evaluate the model at the end (wpt-eval); off by default
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
    WPT_REPLACEMENTS  how many times a run resumes after a lost gpu worker (default 3)
    WPT_STORE, WPT_LINK_CHECKPOINTS and every WPT_ option of wpt-train
                      as each command reads them: from the environment, else from the same
                      .env file

In a container (a /.dockerenv file), a file:// store must be on a mounted
volume or folder: one in the container's own layer would go with the
container, so the manager refuses it.

The steps, each its own command, in order, each stopping the run with its
exit code when it fails:

    1. wpt-dataset download              every admitted source, into the store
    2. wpt-dataset build RECIPE          the dataset; its id is the last line it prints
       wpt-agent build-image NAME        each image the dataset's tasks name, where DOCKER_HOST
                                         reaches Docker (the sandboxes' machine); none without tasks
    3. wpt-train RECIPE ID --run NAME --resume --out FOLDER/NAME
                                         through the run's gpu workers, rented first when
                                         WPT_GPU_WORKERS asks; the run's checkpoint is the store's,
                                         under NAME, so the same command with the same WPT_RUN
                                         continues it. A gpu worker lost (wpt-train's exit code 4)
                                         that the manager rented is reported, given back and
                                         replaced, and the step runs again from the last
                                         checkpoint in the store; so does a crash of a training
                                         process on one of several workers (exit code 5), on the
                                         workers that answer; both count against the same limit,
                                         WPT_REPLACEMENTS times
    4. wpt-eval FOLDER/NAME              with WPT_EVALUATE only, on the run's first gpu worker
    5. the model folder into the store, ref models/NAME (wpt_store.models), recorded first;
       then the folder is removed: the store holds it

The rentals are stopped once the evaluation (or, without one, the training)
ends, whatever its outcome. A rental that cannot be stopped stays named in
its run's file; the manager tries to stop every rental still named in the
file of a run that ended at each start and, until none is left, every SWEEP
seconds, each try recorded (stop_rentals()).

The manager records its own run in the store (records/, command
wpt-manager), each step a span, and prints the line that copies the model
out of the store (wpt-store copy-out).

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
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_store import NAME as STORE_NAME
from wpt_store import MissingObject, StoreError, read_env_file, store_from_settings
from wpt_store.models import KIND, put_model
from wpt_store.names import as_record_name
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry, set_ref
from wpt_train import CRASHED, LOST
from wpt_worker.manager import WORKERS

from . import machines, rental

REFUSED = 2
RUN_FILE = "wpt-run.txt"  # at the repository's root, git-ignored: the run name the manager made
DEFAULT_FOLDER = Path("runs") / "manager"
DEFAULT_STATE = Path("runs") / "manager-state"  # the runs' gpu workers, their tokens included (machines.py)
CONTAINER_SETTING = "WPT_MANAGER_CONTAINER"
SWEEP = 300.0  # seconds between two tries to stop the rentals that could not be stopped
RUN_LINE = re.compile(r"# run (\S+) of the recipe (\S+);")  # the first line of RUN_FILE
# each command the manager runs, by the Python that runs it
COMMANDS = {"wpt-dataset": ["-m", "wpt_dataset.cli"], "wpt-train": ["-m", "wpt_train"],
            "wpt-agent": ["-m", "wpt_agent.cli"],
            "wpt-eval": ["-c", "import sys, wpt_eval; sys.exit(wpt_eval.main())"]}
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


def copy_out_line(name, environ) -> str:
    """The command a person copies to have the run's model in runs/NAME of
    the repository, once the run is done."""
    command = environ.get("WPT_MANAGER_COMMAND") or "uv run"
    return f"{command} wpt-store copy-out {name} runs/{name}"


def run_command(step, arguments, log, last_line=False, environ=None):
    """Run a command of COMMANDS, what it prints shown as it comes; with
    last_line, the last non-empty line it printed; environ: settings it
    gets besides this process's environment. Failed when it fails."""
    name, *rest = arguments
    command = [sys.executable, *COMMANDS[name], *rest]
    log(f"{step}: {name} {' '.join(rest)}")
    env = {**os.environ, **(environ or {})}
    if not last_line:
        code = subprocess.run(command, env=env).returncode
        if code:
            raise Failed(step, code)
        return None
    found = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                          env=env) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip():
                found = line.strip()
    if process.returncode:
        raise Failed(step, process.returncode)
    return found


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
    code = _main(argv)
    if code:
        sys.exit(code)


def _main(argv) -> int:
    """The command; its exit code."""
    try:
        root = Repo.find().root
    except DatasetError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return REFUSED
    environ = settings(root)
    container = environ.get(CONTAINER_SETTING, "") in ("1", "true", "yes")

    def leave(code):
        """The exit code: in the container, 0 for a run that ended or was
        refused, whatever its outcome, so that restart: on-failure restarts
        only a crash."""
        if code and container:
            print(f"wpt-manager stopped with exit code {code}; this container exits with 0, so Docker does not "
                  "start it again: `docker compose up` goes on with the run", file=sys.stderr, flush=True)
            return 0
        return code

    recipe = environ.get("WPT_RECIPE", "")
    if not recipe:
        print("WPT_RECIPE is not set: name the model recipe to train (a file name in data/recipes/models/, without "
              ".json), in the environment or in .env", file=sys.stderr)
        return leave(REFUSED)
    try:
        Repo(root).recipe("model", recipe)  # refused now, not after the download
    except DatasetError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return leave(REFUSED)
    name = environ.get("WPT_RUN") or None
    if name is not None and not STORE_NAME.fullmatch(name):
        print(f"WPT_RUN={name!r}: use letters, digits, '.', '_' and '-' only", file=sys.stderr)
        return leave(REFUSED)
    try:
        rental.kind(environ)  # refused now, not when the workers are rented
        rental.wait(environ)
        machines.replacements(environ)
    except (rental.RentalError, machines.WorkersError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return leave(REFUSED)
    layer = store_in_layer(environ.get("WPT_STORE", ""))
    if layer is not None:
        print(f"WPT_STORE names {layer}, which is in this container's own files, deleted with the container: name "
              "a bucket, or leave WPT_STORE unset for the volume wpt-store, or set WPT_STORE_FOLDER to that folder "
              "on this machine to mount it at the same path", file=sys.stderr)
        return leave(REFUSED)
    try:
        store = store_from_settings(root / ".env")
    except StoreError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return leave(REFUSED)
    folder = Path(environ.get("WPT_MANAGER_FOLDER") or root / DEFAULT_FOLDER)
    kept = Path(environ.get("WPT_MANAGER_STATE") or root / DEFAULT_STATE)
    evaluate = environ.get("WPT_EVALUATE", "") in ("1", "true", "yes")
    arguments = list(argv or sys.argv[1:])
    code, continued = 0, []
    stop_rentals(store, kept, environ, root, arguments)  # those a run that ended could not stop
    for path in machines.files(kept):  # the runs a crash or a stop cut short go on first
        workers = machines.Workers.claim(path)
        if workers is None:  # another manager runs it now
            continue
        try:
            if unfinished(store, workers.run):
                print(f"run {workers.run} did not end: it goes on, reconnecting to its gpu workers", flush=True)
                code = one_run(workers.state["recipe"], workers.run, folder, evaluate, store, environ, root,
                               workers, arguments)
                continued.append(workers.run)
            else:
                workers.end()
        finally:
            workers.release()
    # with no WPT_RUN, a start that went on with runs a crash cut short has done its work: the container
    # Docker starts again after a crash does not start another run
    if name is None and not continued:
        name = previous_run(root, recipe, store)
        if name is not None:
            print(f"run {name} (named in {root / RUN_FILE}, its model is not in the store yet)", flush=True)
        else:
            name = made_name(recipe)
            (root / RUN_FILE).write_text(f"# run {name} of the recipe {recipe}; once it is done, this copies its "
                                         f"model out of the store:\n{copy_out_line(name, environ)}\n",
                                         encoding="utf-8")
            print(f"run {name} (written into {root / RUN_FILE})", flush=True)
    if name is not None and name not in continued:
        workers = machines.Workers.new(kept, name, recipe)
        if workers is None:
            print(f"run {name} is another wpt manager's, which runs it now (its lock in {kept} is held): this "
                  "one leaves it", file=sys.stderr)
            code = REFUSED
        else:
            try:
                code = one_run(recipe, name, folder, evaluate, store, environ, root, workers, arguments)
            finally:
                workers.release()
    left = stop_rentals(store, kept, environ, root, arguments)
    while left:  # money is spent while a rental runs: the manager stays, trying
        print(f"{left} rental(s) are still rented: the manager tries to stop them again in {SWEEP:.0f} s",
              flush=True)
        time.sleep(SWEEP)
        left = stop_rentals(store, kept, environ, root, arguments)
    return leave(code)


def previous_run(root, recipe, store) -> str | None:
    """The name RUN_FILE holds when it is a run of recipe whose model is not
    in the store: a run that did not end well, which the same command goes
    on with."""
    try:
        first = (root / RUN_FILE).read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    found = RUN_LINE.match(first)
    if found and found[2] == recipe and STORE_NAME.fullmatch(found[1]) and store.get_ref(KIND, found[1]) is None:
        return found[1]
    return None


def stop_rentals(store, kept, environ, root, arguments) -> int:
    """Try to stop every rental still named in the file of a run that ended
    (machines.with_rentals): the manager could not stop them when the run
    ended. Each try is in a record of its own (command wpt-manager). Returns
    how many rentals are still named afterwards; the files of runs another
    manager holds are left to it."""
    waiting = machines.with_rentals(kept)
    if not waiting:
        return 0
    left = 0
    try:
        with Telemetry.start(store, "wpt-manager", arguments, "rentals",
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
                        telemetry.log(f"RentalError: {error}", error=True)
                    left += len(workers.groups())
                finally:
                    workers.release()
    except RecordError as error:
        print(f"RecordError: {error}; the rentals are not stopped without a record of each try", file=sys.stderr)
        return len(waiting)
    return left


def unfinished(store, name) -> bool:
    """Whether the run name did not end: no model of that name in the
    store, and no record of a wpt-manager run of it, or the last one has no
    end (a crash or a stop cut it short)."""
    if store.get_ref(KIND, name) is not None:
        return False
    found = []
    for folder in store.record_folders():
        if "-wpt-manager-" not in folder:
            continue
        try:
            start = json.loads(store.read_record(folder, "start.json"))
        except (MissingObject, ValueError):
            continue
        if start.get("command") == "wpt-manager" and start.get("inputs", {}).get("run") == name:
            found.append((start.get("started", ""), folder))
    return not found or "end.json" not in store.record_files(max(found)[1])


def one_run(recipe, name, folder, evaluate, store, environ, root, workers, arguments) -> int:
    """One run, recorded, its file then marked ended (whatever its
    outcome: a person starts it again with the same WPT_RUN); its exit
    code."""
    try:
        with Telemetry.start(store, "wpt-manager", arguments, name, inputs={"recipe": recipe, "run": name},
                             env_file=root / ".env") as telemetry:
            code = _run(recipe, name, folder / name, evaluate, store, telemetry, environ, root, workers)
            if code:
                raise SystemExit(code)  # recorded as the run's exit code
        code = 0
    except SystemExit as ended:
        code = ended.code
    except RecordError as error:
        print(f"RecordError: {error}; stopped, nothing more was recorded", file=sys.stderr)
        return RECORD_FAILED  # no end recorded: the manager that comes back goes on with it
    workers.end()
    return code


def task_images(store, dataset) -> list[str]:
    """The images the dataset's tasks name, each once, in order."""
    record = json.loads(store.get_bytes(store.get_ref("datasets", dataset)))
    part = record.get("tasks")
    lines = [] if part is None else store.get_bytes(part["object"]).decode("utf-8").splitlines()
    return sorted({json.loads(line)["image"] for line in lines if line.strip()})


def _run(recipe, name, out, evaluate, store, telemetry, environ, root, workers) -> int:
    log = telemetry.log
    try:
        with telemetry.span("manager.download"):
            run_command("download", ["wpt-dataset", "download"], log)
        with telemetry.span("manager.build", recipe=recipe) as span:
            dataset = run_command("build", ["wpt-dataset", "build", recipe], log, last_line=True)
            if dataset is None or not re.fullmatch(r"[0-9a-f]{64}", dataset):
                log(f"build: wpt-dataset build printed no dataset id last (it printed {dataset!r})", error=True)
                return REFUSED
            span.set(dataset=dataset)
        telemetry.input(dataset=dataset)
        images = task_images(store, dataset)
        if images:  # the sandboxes' images, built where the sandboxes run
            with telemetry.span("manager.images", images=images):
                for image in images:
                    run_command("images", ["wpt-agent", "build-image", image], log)
        try:
            _train(recipe, name, out, dataset, telemetry, environ, root, workers)
            if evaluate:  # on the run's gpu workers, rented ones included, which are given back after it
                with telemetry.span("manager.evaluate", folder=str(out)):
                    run_command("evaluate", ["wpt-eval", str(out)], log, environ={WORKERS: workers.text()})
        finally:
            _stop_workers(workers, environ, root, telemetry)
    except Failed as error:
        log(f"{error}: the run stops here; the same command goes on from what is kept", error=True)
        return error.code
    with telemetry.span("manager.model", folder=str(out)) as span:
        manifest = put_model(store, out)
        set_ref(store, KIND, name, manifest, telemetry)
        span.set(manifest=manifest)
    telemetry.output(model=manifest)
    shutil.rmtree(out)  # the store holds it: copy-out gives it back
    log(f"model {name} kept in {store} (ref {KIND}/{name}); copy it out with:")
    log(copy_out_line(name, environ))
    return 0


def _train(recipe, name, out, dataset, telemetry, environ, root, workers):
    """wpt-train through the run's gpu workers, rented when the settings
    ask, resumed when a gpu worker it rented is lost or a training process
    crashed on one of several workers, a lost rented worker replaced, up to
    WPT_REPLACEMENTS times. Failed when it fails."""
    log = telemetry.log
    try:
        with telemetry.span("manager.workers"):
            workers.prepare(environ, root, telemetry)
    except (rental.RentalError, machines.WorkersError) as error:
        log(f"{type(error).__name__}: {error}", error=True)
        raise Failed("workers", REFUSED) from None
    while True:
        try:
            with telemetry.span("manager.train", dataset=dataset, folder=str(out), workers=len(workers.keys())):
                run_command("train", ["wpt-train", recipe, dataset, "--run", name, "--resume", "--out", str(out)],
                            log, environ={WORKERS: workers.text()})
            return
        except Failed as error:
            if error.code not in (LOST, CRASHED):
                raise
            try:
                if not workers.recover(environ, root, telemetry):
                    raise
            except rental.RentalError as failed:
                log(f"RentalError: {failed}", error=True)
                raise error from None
            log("the run resumes from the last checkpoint in the store")


def _stop_workers(workers, environ, root, telemetry):
    """Stop the run's rentals; those that cannot be stopped stay named in
    its file, and stop_rentals() tries them again."""
    try:
        workers.stop(environ, root, telemetry)
    except rental.RentalError as error:
        telemetry.log(f"RentalError: {error}; they stay named in the run's file, and the manager tries to stop them "
                      f"again every {SWEEP:.0f} s", error=True)
