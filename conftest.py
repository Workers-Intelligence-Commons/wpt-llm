"""Shared setup for the tests of every package and application: whether
Docker runs on this machine. Test files reach it as a fixture (they cannot
import this file under --import-mode=importlib)."""

import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def docker_runs():
    """Skips the test, with the reason, when Docker is not installed or its
    daemon does not answer."""
    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "info"], capture_output=True).returncode:
        pytest.skip("Docker is not installed or its daemon does not answer")
