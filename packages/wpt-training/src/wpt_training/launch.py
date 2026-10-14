"""Start several training processes on this machine, as torchrun does.

torchrun covers several machines but does not start on Windows with
PyTorch 2.14 (its process store needs libuv, which the Windows build
lacks); this covers one machine on every system.
"""

import os
import socket
import subprocess
import sys
import time


def launch(processes: int, arguments: list[str]) -> int:
    """Run `python ARGUMENTS` processes times at once, each told its place
    the way torchrun tells it (RANK, LOCAL_RANK, WORLD_SIZE,
    LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT). Returns 0 when all
    succeed; when one fails, stops the others and returns its exit code."""
    with socket.socket() as probe:  # a free port for the processes to meet on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    shared = {"WORLD_SIZE": str(processes), "LOCAL_WORLD_SIZE": str(processes),
              "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
    children = [subprocess.Popen([sys.executable, *arguments],
                                 env={**os.environ, **shared, "RANK": str(rank), "LOCAL_RANK": str(rank)})
                for rank in range(processes)]
    try:
        while True:
            codes = [child.poll() for child in children]
            failed = next((code for code in codes if code not in (None, 0)), None)
            if failed is not None:
                return failed
            if all(code == 0 for code in codes):
                return 0
            time.sleep(0.1)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
