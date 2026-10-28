"""Downloads tried again after a failure that may not happen twice.

A timeout, a cut connection, an HTTP answer of 408, 425, 429 or 5xx, or a
git or Mercurial network error is tried again, up to ATTEMPTS times in all.
The wait doubles from BASE seconds up to CAP, each one drawn at random below
that limit (so many fetches failing together do not come back together),
and follows a server's Retry-After up to CAP. Anything else, such as a 404
or an LFS object whose bytes are not its pointer's, fails at once: trying
again gives the same answer.

Each new try is announced in one line, printed on stderr; inside a
reporting(log) block (a command that records its run passes its
telemetry.log), the line goes to log instead, which records and prints it.
"""

import contextlib
import http.client
import random
import subprocess
import sys
import time
import urllib.error

ATTEMPTS = 5
BASE = 4.0
CAP = 60.0
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}
# what git, Mercurial and curl print when the network, not the request, failed
TRANSIENT_MESSAGES = ("could not resolve host", "connection timed out", "connection reset", "failed to connect",
                      "operation timed out", "early eof", "rpc failed", "the remote end hung up unexpectedly",
                      "unexpected disconnect", "transfer closed", "temporary failure in name resolution",
                      "connection refused", "returned error: 5", "returned error: 429", "http/2 stream",
                      "gnutls", "ssl_read", "timed out")


def _print(text, error=True):
    print(text, file=sys.stderr)


_report = _print  # where call() announces a new try; set by reporting()


@contextlib.contextmanager
def reporting(log):
    """Announce every new try inside the block, from any thread, through
    log(text, error=True) instead of printing it."""
    global _report
    previous, _report = _report, log
    try:
        yield
    finally:
        _report = previous


def call(action, what: str, sleep=time.sleep):
    """action() once, and again while it fails for a reason that may pass."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return action()
        except Exception as error:
            if attempt == ATTEMPTS or not transient(error):
                raise
            wait = _retry_after(error)
            if wait is None:
                wait = random.uniform(0, min(CAP, BASE * 2 ** (attempt - 1)))
            _report(f"{what}: {_describe(error)}; trying again in {wait:.0f}s ({attempt + 1}/{ATTEMPTS})", error=True)
            sleep(wait)


def transient(error) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)):
        return True
    if isinstance(error, subprocess.CalledProcessError):
        text = (error.stderr or b"").decode(errors="replace").lower()
        return any(message in text for message in TRANSIENT_MESSAGES)
    return False


def _retry_after(error) -> float | None:
    if isinstance(error, urllib.error.HTTPError) and error.headers is not None:
        value = error.headers.get("Retry-After", "")
        if value.strip().isdigit():
            return min(float(value), CAP)
    return None


def _describe(error) -> str:
    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        return error.stderr.decode(errors="replace").strip().splitlines()[-1]
    return str(error)
