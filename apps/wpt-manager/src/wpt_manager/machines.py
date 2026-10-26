"""The gpu workers of the manager's runs (docs/machines.md, "Rented gpu
workers"), kept on the manager's own disk, never in the store: one file
per run, FOLDER/RUN.json, readable by its owner only, which holds each
worker's key (its token included) and, for a worker the manager rents, its
provider and the rental's name. Workers rented as one cluster (rental.py)
share one rental: its name, and one token.

WPT_GPU_WORKERS (environment, else .env) lists the workers of a new run:

    wpt-worker://TOKEN@HOST:PORT#PIN ...   the keys of workers a person started
    N                                      rent N workers (rental.py)
    unset or empty                         a local gpu worker, which wpt-train starts

The file is written before each step that changes it: a rental's name
before the provider rents it, its key once it gives its address, so the
manager that comes back after a crash reconnects to its workers, and can
stop every rental it began. A run's file is marked "ended" once the run
ends, whatever its outcome; the manager that comes back reads the other
files only.

One manager at a time owns a run: it holds a lock on FOLDER/RUN.lock (the
operating system's, released when the process ends, crash included) for
as long as it runs it. Another manager sharing the folder leaves a run
whose lock is held alone, and refuses to start that run itself.

A rental still named in the file of a run that ended is one the manager
could not stop: with_rentals() lists those files, and the manager tries
again at each start and every few minutes (wpt_manager.stop_rentals).
"""

import json
import os
import re
import secrets
from pathlib import Path

from wpt_utils.files import try_lock_file, unlock_file
from wpt_worker.link import Key, probe
from wpt_worker.manager import WORKERS, keys

from . import rental

REPLACEMENTS = 3  # WPT_REPLACEMENTS: how many times a run resumes after a lost gpu worker


class WorkersError(Exception):
    """The settings name no usable gpu workers."""


def rental_name(run) -> str:
    """A new rental's name: wpt-, the run's name as a provider takes it
    (lower-case letters, digits and '-'), and 6 random hex digits."""
    slug = re.sub(r"[^a-z0-9]+", "-", run.lower()).strip("-")[:24].strip("-")
    return f"wpt-{slug}-{secrets.token_hex(3)}" if slug else f"wpt-{secrets.token_hex(3)}"


def replacements(settings) -> int:
    text = settings.get("WPT_REPLACEMENTS") or str(REPLACEMENTS)
    if not text.isdigit():
        raise WorkersError(f"WPT_REPLACEMENTS={text!r}: a number of times, 0 or more")
    return int(text)


class Workers:
    """One run's gpu workers and their file (path), the run's lock held;
    see the module's docstring. state: {"run", "recipe", "ended",
    "workers": [{"key", "rental": {"provider", "name", "kind"} or null}],
    "resumed"}."""

    def __init__(self, path, state, lock):
        self.path, self.state, self.lock = Path(path), state, lock

    @classmethod
    def claim(cls, path) -> "Workers | None":
        """The run of the file path, its lock taken; None when another
        manager holds it."""
        path = Path(path)
        lock = try_lock_file(path.with_suffix(".lock"))
        if lock is None:
            return None
        try:
            return cls(path, json.loads(path.read_text(encoding="utf-8")), lock)
        except BaseException:
            unlock_file(lock)
            raise

    @classmethod
    def new(cls, folder, run, recipe) -> "Workers | None":
        """The run named run, started (again), its lock taken; None when
        another manager holds it."""
        path = Path(folder) / f"{run}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = try_lock_file(path.with_suffix(".lock"))
        if lock is None:
            return None
        state = {"run": run, "recipe": recipe, "ended": False, "workers": [], "resumed": 0}
        if path.is_file():  # the same run started again (WPT_RUN): its workers are still its own
            state = {**json.loads(path.read_text(encoding="utf-8")), "recipe": recipe, "ended": False, "resumed": 0}
        found = cls(path, state, lock)
        found.save()
        return found

    def release(self):
        """Give the run's lock back: another manager may take the run."""
        if self.lock is not None:
            unlock_file(self.lock)
            self.lock = None

    def end(self):
        """Mark the run ended: the manager that comes back leaves it."""
        self.state["ended"] = True
        self.save()

    @property
    def run(self):
        return self.state["run"]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(f".{self.path.name}.partial")
        partial.unlink(missing_ok=True)
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # tokens: its owner reads it only
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(self.state, file, indent=2)
        os.replace(partial, self.path)

    def keys(self) -> list[Key]:
        return [Key.parse(entry["key"]) for entry in self.state["workers"] if entry.get("key")]

    def text(self) -> str:
        """WPT_GPU_WORKERS for wpt-train: every key, in order; empty for a
        local worker."""
        return " ".join(str(key) for key in self.keys())

    def groups(self) -> list[tuple[dict, list[int]]]:
        """The rentals of the run in order, each with the numbers of its
        workers (more than one for a cluster)."""
        found = {}
        for index, entry in enumerate(self.state["workers"]):
            rented = entry["rental"]
            if rented is not None:
                found.setdefault(rented["name"], (rented, []))[1].append(index)
        return list(found.values())

    def prepare(self, settings, root, telemetry):
        """The run's workers, when the file holds none yet: those the
        settings list, or as many rented as they ask (as one cluster, when
        the provider can). WorkersError for settings that name none
        usable; rental.RentalError when a rental fails (the rentals made
        stay in the file, for stop())."""
        if self.state["workers"]:
            for rented, indexes in self.groups():
                if any(not self.state["workers"][index].get("key") for index in indexes):
                    # a rental a crash cut short before its address: given back, rented again
                    self._stop(rented, rental.providers()[rented["provider"]](settings, root), telemetry)
                    self._rent(rental.provider(settings, root), rented["kind"], settings, telemetry, indexes)
            return
        text = settings.get(WORKERS, "").strip()
        if text.isdigit():
            count = int(text)
            if count < 1:
                raise WorkersError(f"{WORKERS}={text}: rent 1 gpu worker or more, or leave it unset for a local one")
            self._rent(rental.provider(settings, root), rental.kind(settings), settings, telemetry, [None] * count)
            return
        try:
            given = keys(settings)
        except ValueError as error:
            raise WorkersError(str(error)) from None
        self.state["workers"] = [{"key": str(key), "rental": None} for key in given]
        self.save()

    def _rent(self, chosen, kind, settings, telemetry, indexes):
        """Rent one worker for each of indexes (positions in the file, None
        for a new one at the end): all in one request, as one cluster, when
        there are several and the provider can, else one rental each."""
        if len(indexes) > 1 and not chosen.cluster:
            for index in indexes:
                self._rent(chosen, kind, settings, telemetry, [index])
            return
        name, token = rental_name(self.run), secrets.token_urlsafe(32)
        rented = {"provider": settings.get("WPT_RENTAL_PROVIDER") or "skypilot", "name": name, "kind": kind}
        entries = [{"key": None, "token": token, "rental": rented} for _ in indexes]
        for index, entry in zip(indexes, entries):
            if index is None:
                self.state["workers"].append(entry)
            else:
                self.state["workers"][index] = entry
        self.save()  # its name first: a rental a crash cuts short is still one to stop
        with telemetry.span("manager.rent", rental=name, kind=kind, workers=len(entries)) as span:
            chosen.rent(name, token, kind == rental.INTERRUPTIBLE, len(entries))
            found = chosen.addresses(name, len(entries)) if len(entries) > 1 else [chosen.address(name)]
            found = [Key(each.host, each.port, each.pin, token, each.place, each.private) for each in found]
            for entry, key in zip(entries, found):
                entry["key"] = str(key)
                del entry["token"]
            self.save()
            span.set(workers=[key.shown() for key in found])
        telemetry.log(f"gpu worker rented: {name}, {', '.join(key.shown() for key in found)}")

    def recover(self, settings, root, telemetry) -> bool:
        """After wpt-train lost a gpu worker, or crashed on one of several:
        replace each worker the manager rented that is not there any more,
        so the run resumes; True when it may. A rental of several workers
        (a cluster) is replaced whole when one of them is lost. False when
        no resumption is left (WPT_REPLACEMENTS) or a lost worker is one a
        person started, which only that person can start again."""
        limit = replacements(settings)
        if self.state["resumed"] >= limit:
            telemetry.log(f"the run resumed {limit} times already (WPT_REPLACEMENTS): it stops here", error=True)
            return False
        lost = [index for index, key in enumerate(self.keys()) if not probe(key)]
        for index in lost:
            entry = self.state["workers"][index]
            telemetry.event("manager.worker_lost", worker=Key.parse(entry["key"]).shown(),
                            rental=(entry["rental"] or {}).get("name"))
        given = [Key.parse(self.state["workers"][index]["key"]).shown() for index in lost
                 if self.state["workers"][index]["rental"] is None]
        if given:
            telemetry.log(f"lost gpu workers a person started: {', '.join(given)}; start them again (or others, "
                          f"in {WORKERS}), then run the same command with the same WPT_RUN", error=True)
            return False
        self.state["resumed"] += 1
        self.save()
        if not lost:
            telemetry.log("every gpu worker answers again: the run resumes on them")
        chosen = rental.provider(settings, root) if lost else None
        for old, indexes in self.groups():
            if not set(indexes) & set(lost):
                continue
            self._stop(old, rental.providers()[old["provider"]](settings, root), telemetry)
            self._rent(chosen, old["kind"], settings, telemetry, indexes)
            telemetry.event("manager.worker_replaced", rental=old["name"],
                            by=self.state["workers"][indexes[0]]["rental"]["name"], workers=len(indexes))
        return True

    def stop(self, settings, root, telemetry):
        """Stop every rental of the run and forget its workers: at the run's
        end, whatever its outcome. Every rental is tried; those whose stop
        failed stay in the file, and rental.RentalError then names each of
        them, still rented."""
        kept, failed = [], []
        for rented, indexes in self.groups():
            try:
                self._stop(rented, rental.providers()[rented["provider"]](settings, root), telemetry)
            except rental.RentalError as error:
                telemetry.event("manager.rental_stop_failed", rental=rented["name"], error=str(error))
                kept += [self.state["workers"][index] for index in indexes]
                failed.append(f"{rented['name']} ({error})")
        self.state["workers"] = kept
        self.save()
        if failed:
            raise rental.RentalError(f"{len(failed)} rental(s) could not be stopped and are still rented: "
                                     + "; ".join(failed))

    @staticmethod
    def _stop(found, chosen, telemetry):
        telemetry.irreversible("manager.rental_stopped", rental=found["name"], provider=found["provider"])
        chosen.stop(found["name"])
        telemetry.log(f"gpu worker stopped: {found['name']}")


def files(folder) -> list[Path]:
    """The files of the runs that did not end, oldest first."""
    folder = Path(folder)
    found = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime) if folder.is_dir() else []
    return [path for path in found if not json.loads(path.read_text(encoding="utf-8")).get("ended")]


def with_rentals(folder) -> list[Path]:
    """The files of the runs that ended and still name a rental, oldest
    first: rentals the manager could not stop."""
    folder = Path(folder)
    found = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime) if folder.is_dir() else []
    states = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in found]
    return [path for path, state in states if state.get("ended") and any(entry["rental"] for entry in state["workers"])]
