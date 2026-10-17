"""Shared setup for the wpt-agent tests: a Debian-based image the sandbox
tests run in. The tests that need it are skipped, with the reason, when
Docker cannot run here (the docker_runs fixture of the root conftest.py)."""

import subprocess

import pytest

IMAGE = "ubuntu:24.04"  # bash, coreutils' timeout and dpkg, as task images have


@pytest.fixture(scope="session")
def image(docker_runs):
    """IMAGE, pulled when this machine does not have it."""
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
        subprocess.run(["docker", "pull", "-q", IMAGE], capture_output=True, check=True)
    return IMAGE
