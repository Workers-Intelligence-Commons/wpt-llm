"""The Docker host the sandboxes run on, measured for the sizing rule
(wpt_agent.host; docs/machines.md)."""

import subprocess

from wpt_agent import host
from wpt_utils import parallel


def test_a_remote_docker_host_is_measured_through_a_container_and_this_machine_s_daemon_is_read_here(docker_runs):
    # this test's daemon is this machine's, so what a container measures is
    # what this machine has: the same cores and total memory
    assert not host.remote({}) and not host.remote({"DOCKER_HOST": "unix:///var/run/docker.sock"})
    assert host.remote({"DOCKER_HOST": "ssh://user@gpu-box"}) and host.remote({"DOCKER_HOST": "tcp://10.0.0.2:2376"})
    assert isinstance(host.machine("busybox", {}), parallel.Machine)

    if subprocess.run(["docker", "image", "inspect", "ubuntu:24.04"], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", "ubuntu:24.04"], capture_output=True, check=True)
    measured = host.machine("ubuntu:24.04", {"DOCKER_HOST": "ssh://user@gpu-box"})
    assert isinstance(measured, host.DockerMachine)
    here = parallel.Machine()
    available, total = measured.memory()
    assert total == here.memory()[1] and 0 < available <= total
    assert measured.cores >= 1
    assert 0 <= measured.idle_cores() <= measured.cores
    count, why = parallel.limit(10**9, "sandboxes", measured)
    assert count >= 1 and f"of {measured.cores} cores" in why
