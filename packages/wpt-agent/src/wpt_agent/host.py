"""The machine the sandboxes run on, as wpt_utils.parallel's sizing rule
reads it.

The sandboxes run wherever the docker command reaches its daemon: this
machine, or the Docker host DOCKER_HOST names (docs/machines.md). On this
machine the rule reads this machine (wpt_utils.parallel.Machine); on a
remote Docker host (DOCKER_HOST naming anything but a local socket), it is
measured there, each measure through a short-lived container of an image
the host holds, with no network: its cores (nproc), its memory
(/proc/meminfo) and its idle cores (/proc/stat, read twice).

    machine(image)     the Machine the sizing rule reads for sandboxes of image
    remote(environ)    whether DOCKER_HOST names a remote Docker host
"""

import os
import time

from wpt_utils import parallel

from . import docker

LOCAL = ("unix://", "npipe://")  # a socket of this machine's own daemon


def remote(environ=None) -> bool:
    host = (os.environ if environ is None else environ).get("DOCKER_HOST", "")
    return bool(host) and not host.startswith(LOCAL)


def machine(image, environ=None):
    """parallel.Machine() for this machine's daemon, DockerMachine(image)
    for a remote one."""
    return DockerMachine(image) if remote(environ) else parallel.Machine()


class DockerMachine:
    """A Docker host's cores, memory and idle cores, as parallel.Machine
    gives this machine's, each read in a container of image that starts,
    prints and is removed. A measure the container cannot give is None, as
    for a machine that does not say; DockerFailed when Docker cannot run
    the container at all."""

    def __init__(self, image):
        self.image = image
        self.cores = int(self._read("nproc").split()[0])
        self._last = None  # (monotonic time, (idle, all)) of the previous measure
        self._idle = None

    def _read(self, script) -> str:
        return docker.run("run", "--rm", "--network", "none", "--entrypoint", "sh", self.image, "-c",
                          script).decode("utf-8", errors="replace")

    def memory(self):
        try:
            fields = dict(line.split(":", 1) for line in self._read("cat /proc/meminfo").splitlines() if ":" in line)
            return int(fields["MemAvailable"].split()[0]) * 1024, int(fields["MemTotal"].split()[0]) * 1024
        except (KeyError, ValueError, IndexError):
            return None

    def _times(self, line):
        ticks = [int(value) for value in line.split()[1:9]]
        return ticks[3] + ticks[4], sum(ticks)

    def idle_cores(self):
        """The idle cores since the previous measure; the first measure
        reads /proc/stat twice, SAMPLE seconds apart, in one container."""
        try:
            if self._last is None:
                first, second = self._read(f"head -n 1 /proc/stat; sleep {parallel.SAMPLE}; "
                                           "head -n 1 /proc/stat").splitlines()[:2]
                before, now = self._times(first), self._times(second)
            elif time.monotonic() - self._last[0] < parallel.SAMPLE:
                return self._idle
            else:
                before, now = self._last[1], self._times(self._read("head -n 1 /proc/stat").splitlines()[0])
        except (ValueError, IndexError):
            self._last = self._idle = None
            return None
        self._last = (time.monotonic(), now)
        self._idle = None if now[1] <= before[1] else self.cores * (now[0] - before[0]) / (now[1] - before[1])
        return self._idle
