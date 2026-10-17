"""Starting the docker command: the one place wpt-agent runs it.

Docker is found through wpt_dataset.programs.program. Every failure to run
it, or a non-zero exit, is DockerFailed; images.py and sandbox.py turn it
into their own error.
"""

import subprocess

from wpt_dataset.errors import MissingTool
from wpt_dataset.programs import program


class DockerFailed(Exception):
    """Docker is missing, could not start, or exited with an error; the
    message names the docker command and what it printed, or its exit code
    when it printed nothing."""


def failed(command: str, code: int, stderr: bytes) -> DockerFailed:
    """The failure of `docker COMMAND` that exited with code, having printed
    stderr: "docker exec: <stderr>", or "docker exec: exit 137"."""
    return DockerFailed(f"docker {command}: {stderr.decode(errors='replace').strip() or f'exit {code}'}")


def path() -> str:
    try:
        return program("docker")
    except MissingTool as error:
        raise DockerFailed(str(error)) from None


def run(*args, stdin: bytes = b"") -> bytes:
    """The standard output of `docker ARGS`, which must exit with 0."""
    try:
        done = subprocess.run([path(), *args], input=stdin, capture_output=True)
    except OSError as error:
        raise DockerFailed(f"docker {args[0]}: {error}") from None
    if done.returncode:
        raise failed(args[0], done.returncode, done.stderr)
    return done.stdout


def start(*args, stdin=None, stdout=None, stderr=None) -> subprocess.Popen:
    """`docker ARGS` started, for a caller that streams its input or
    output; the caller waits for it."""
    try:
        return subprocess.Popen([path(), *args], stdin=stdin, stdout=stdout, stderr=stderr)
    except OSError as error:
        raise DockerFailed(f"docker {args[0]}: {error}") from None
