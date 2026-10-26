"""Gpu workers rented through SkyPilot (Apache-2.0), with its `sky`
command, from the task deploy/gpu-worker/skypilot.yaml (rental.py's
interface). One rental is one SkyPilot cluster: one machine per worker, or,
for several workers, one cluster of that many nodes rented in one request
(cluster = True):

    rent      sky launch -c NAME -y -d --down --gpus GPUS [--num-nodes N] [--use-spot]
                  --env WPT_WORKER_TOKEN ... [--env WPT_WORKER_PLACE=NAME]
              the token goes in the command's environment, never on its command line; --down
              gives the machines back once their workers end (WPT_WORKER_WAIT with no manager)
    address   the worker's key line in `sky logs --no-follow NAME` (its token hidden: the
              manager set it) for the pin, and `sky status --endpoint 8470 NAME` for the
              address the manager reaches it at
    addresses a cluster's nodes, each by its rank: its key line in the same logs (the pin, and
              the private address the worker printed, since start.sh gives it WPT_WORKER_PLACE
              and WPT_WORKER_PRIVATE), the private address of each rank (start.sh's `wpt-node
              RANK ADDRESS` line), and the address the manager reaches the node at (its Host
              entry in the SSH configuration SkyPilot writes under HOME: NAME for rank 0,
              NAME-worker1 for rank 1, and so on), at the worker's port
    stop      sky down -y NAME

A cluster is one place: its nodes reach each other on the private addresses
their keys give, so a run on them exchanges gradients through NCCL
(docs/machines.md, "Several machines"). Clusters of different rentals are
places of their own.

SkyPilot gets its own provider keys only (environment()): never the
store's settings or keys. It reads the clouds' keys from the environment
names of SKY_PROVIDERS, or as it always does, from its own files under HOME.
"""

import re
import shutil
import subprocess
import time
from pathlib import Path

from wpt_worker.link import Key
from wpt_worker.server import PORT

from .rental import Address, Provider, RentalError, wait

TASK = Path("deploy") / "gpu-worker" / "skypilot.yaml"  # from the repository's root, where sky runs
PASSED = ("WPT_WORKER_WAIT",)  # the worker's settings passed on when the manager's settings hold them
GONE = re.compile(r"not found|does not exist|no (existing |matching )?cluster", re.IGNORECASE)
ENDPOINT = re.compile(r"(?:\w+://)?(\[[0-9A-Fa-f:]+\]|[\w.\-]+):(\d+)/?$")
NODE = re.compile(r"wpt-node (\d+) (\S+)")  # start.sh: a node's rank and private address
# what sky runs with: the shell's basics, and the names of the clouds' keys (AWS_* included). The store's names
# (WPT_STORE and its keys, WPT_STORE_*) are not among them
BASICS = re.compile(r"PATH|HOME|USER|LOGNAME|SHELL|TERM|TZ|TMPDIR|LANG|LC_\w+|SSH_AUTH_SOCK|XDG_\w+|KUBECONFIG|"
                    r"(?i:http_proxy|https_proxy|no_proxy|all_proxy)")
SKY_PROVIDERS = re.compile(r"(SKYPILOT|SKY|AWS|AZURE|ARM|GOOGLE|GCP|CLOUDSDK|LAMBDA|RUNPOD|VAST|FLUIDSTACK|PAPERSPACE|"
                           r"CUDO|IBM|OCI|SCP|NEBIUS|DIGITALOCEAN|HYPERBOLIC|SHADEFORM|PRIMEINTELLECT|VSPHERE|"
                           r"SEEWEB|NOVITA|COREWEAVE)_\w+")


def environment(settings, extra=None) -> dict:
    """The environment sky runs with: the basics of the manager's settings,
    its own provider keys (SKY_PROVIDERS), WPT_WORKER_WAIT, which it passes
    on to the workers, and extra."""
    kept = {name: value for name, value in settings.items()
            if BASICS.fullmatch(name) or SKY_PROVIDERS.fullmatch(name) or name in PASSED}
    return {**kept, **(extra or {})}


class SkyPilot(Provider):
    POLL = 10.0  # seconds between two looks at a rented worker that does not listen yet
    cluster = True

    def _sky(self, *arguments, extra=None) -> subprocess.CompletedProcess:
        sky = shutil.which("sky", path=self.settings.get("PATH"))
        if sky is None:
            raise RentalError("sky is not installed (or not on the search path): SkyPilot rents the gpu workers "
                              "(WPT_RENTAL_PROVIDER=skypilot); the manager's image installs it")
        return subprocess.run([sky, *arguments], cwd=self.root, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=environment(self.settings, extra))

    def rent(self, name, token, interruptible, count=1):
        gpus = self.settings.get("WPT_RENTAL_GPUS")
        if not gpus:
            raise RentalError("WPT_RENTAL_GPUS is not set: name what each rented machine has, as SkyPilot writes it "
                              "(such as H100:8)")
        passed = [part for setting in PASSED if self.settings.get(setting) for part in ("--env", setting)]
        cluster = ["--num-nodes", str(count), "--env", f"WPT_WORKER_PLACE={name}"] if count > 1 else []
        done = self._sky("launch", "-c", name, "-y", "-d", "--down", "--gpus", gpus, *cluster,
                         *(["--use-spot"] if interruptible else []), "--env", "WPT_WORKER_TOKEN", *passed, str(TASK),
                         extra={"WPT_WORKER_TOKEN": token})
        if done.returncode:
            raise RentalError(f"sky launch {name} failed (exit code {done.returncode}): {_last(done)}")

    def address(self, name):
        return self._until(name, lambda: self._one(name))

    def addresses(self, name, count):
        return self._until(name, lambda: self._cluster(name, count))

    def stop(self, name):
        done = self._sky("down", "-y", name)
        if done.returncode and not GONE.search(done.stdout + done.stderr):
            raise RentalError(f"sky down {name} failed (exit code {done.returncode}): {_last(done)}")

    def _until(self, name, look):
        """What look() finds, once it finds something: it is called again
        every POLL seconds, until WPT_RENTAL_WAIT is over."""
        deadline = time.monotonic() + wait(self.settings)
        while True:
            found = look()
            if found:
                return found
            if time.monotonic() > deadline:
                raise RentalError(f"the gpu worker of {name} gave no address within {wait(self.settings):.0f} s "
                                  "(WPT_RENTAL_WAIT)")
            time.sleep(self.POLL)

    def _one(self, name) -> Address | None:
        pin = self._pin(name)
        found = self._endpoint(name) if pin else None
        return Address(found[0], found[1], pin) if found else None

    def _cluster(self, name, count) -> list[Address] | None:
        """The nodes of the cluster, by rank, once every one has printed its
        key and its private address and has an address to be reached at."""
        pins, privates = {}, {}
        for line in self._sky("logs", "--no-follow", name).stdout.splitlines():
            node = NODE.search(line)
            if node:
                privates[int(node[1])] = node[2]
            elif "wpt-worker://" in line:
                try:
                    key = Key.parse(line[line.index("wpt-worker://"):])
                except ValueError:
                    continue
                if key.private:
                    pins[key.private] = key.pin
        hosts = self._hosts(name, count)
        if hosts is None or any(privates.get(rank) not in pins for rank in range(count)):
            return None
        return [Address(hosts[rank], PORT, pins[privates[rank]], place=name, private=privates[rank])
                for rank in range(count)]

    def _hosts(self, name, count) -> list[str] | None:
        """The address of each of the cluster's nodes by rank, from the
        SSH configuration SkyPilot writes for it; None until every one is
        there."""
        path = Path(self.settings.get("HOME") or Path.home()) / ".sky" / "generated" / "ssh" / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        hosts, current = {}, None
        for line in text.splitlines():
            words = line.split()
            if len(words) >= 2 and words[0].lower() == "host":
                current = words[1]
            elif len(words) >= 2 and words[0].lower() == "hostname" and current is not None:
                hosts[current] = words[1]
        found = [hosts.get(name if rank == 0 else f"{name}-worker{rank}") for rank in range(count)]
        return None if None in found else found

    def _pin(self, name) -> str | None:
        """The pin of the key the worker printed, when it has printed it."""
        done = self._sky("logs", "--no-follow", name)
        for line in reversed(done.stdout.splitlines()):
            if "wpt-worker://" in line:
                try:
                    return Key.parse(line[line.index("wpt-worker://"):]).pin
                except ValueError:
                    continue
        return None

    def _endpoint(self, name) -> tuple[str, int] | None:
        done = self._sky("status", "--endpoint", str(PORT), name)
        lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        found = ENDPOINT.search(lines[-1]) if done.returncode == 0 and lines else None
        return (found[1].strip("[]"), int(found[2])) if found else None


def _last(done) -> str:
    lines = [line for line in (done.stderr + done.stdout).splitlines() if line.strip()]
    return lines[-1] if lines else "no output"
