"""Start several training processes on this machine, as torchrun does.

torchrun covers several machines but does not start on Windows with
PyTorch 2.14 (its process store needs libuv, which the Windows build
lacks); this covers one machine on every system.

When a process fails, the others are stopped and, up to max_restarts
times, all start again; each is told how many times through
TORCHELASTIC_RESTART_COUNT, as torchrun's --max-restarts does, so a
restarted run knows to continue from its checkpoint. Two exits are never
restarted: REFUSED (the run refused its input, and would again) and
STOPPED (it was asked to stop). A stop signal to this launcher (Ctrl+C,
SIGTERM, a closed terminal) is passed on to every process once: they
finish their step, save, and exit with STOPPED. A second one is passed on
too, and stops them at once. A signal ignored when the launcher starts
(nohup ignores "hang up") stays ignored.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time

REFUSED = 2
STOPPED = 130
GRACE = 30  # seconds a process gets to exit once its peers failed, before it is killed


def launch(processes: int, arguments: list[str], max_restarts: int = 0, log=None) -> int:
    """Run `python ARGUMENTS` processes times at once, each told its place
    the way torchrun tells it (RANK, LOCAL_RANK, WORLD_SIZE,
    LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT). Returns 0 when all
    succeed; otherwise the exit code of the first that failed, once no
    restart is left or allowed."""
    log = log or (lambda text: print(text, file=sys.stderr, flush=True))
    running, stopping = [], []

    def forward(number, frame):
        stopping.append(number)
        for child in running:
            if child.poll() is None:
                _ask_to_stop(child)

    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            # an ignored signal stays ignored: nohup (scripts/detach.sh) ignores "hang up" so a run outlives its session
            if hasattr(signal, name) and signal.getsignal(getattr(signal, name)) is not signal.SIG_IGN:
                previous[getattr(signal, name)] = signal.signal(getattr(signal, name), forward)
    try:
        for attempt in range(max_restarts + 1):
            code = _run_once(processes, arguments, attempt, running, stopping)
            if code == 0 or stopping or code in (REFUSED, STOPPED) or attempt == max_restarts:
                return STOPPED if stopping and code != 0 else code
            log(f"a training process failed (exit code {code}): starting every process again "
                f"({attempt + 1} of {max_restarts} restarts)")
        return code
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _run_once(processes, arguments, attempt, running, stopping) -> int:
    with socket.socket() as probe:  # a free port for the processes to meet on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    shared = {"WORLD_SIZE": str(processes), "LOCAL_WORLD_SIZE": str(processes),
              "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "TORCHELASTIC_RESTART_COUNT": str(attempt)}
    # in their own process group, so a Ctrl+C reaches them only once, through forward()
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"process_group": 0})
    running[:] = [subprocess.Popen([sys.executable, *arguments],
                                   env={**os.environ, **shared, "RANK": str(rank), "LOCAL_RANK": str(rank)}, **group)
                  for rank in range(processes)]
    try:
        while True:
            codes = [child.poll() for child in running]
            failed = next((code for code in codes if code not in (None, 0)), None)
            if failed is not None and not stopping:  # asked to stop, each saves and exits on its own
                return failed
            if None not in codes:
                return failed or 0
            time.sleep(0.1)
    finally:
        for child in running:
            if child.poll() is None:
                child.terminate()
        for child in running:
            try:
                child.wait(GRACE)
            except subprocess.TimeoutExpired:  # stuck waiting for a peer that is gone
                child.kill()
                child.wait()


def _ask_to_stop(child):
    if os.name == "nt":
        child.send_signal(signal.CTRL_BREAK_EVENT)  # what a process in its own group can receive
    else:
        child.send_signal(signal.SIGTERM)
