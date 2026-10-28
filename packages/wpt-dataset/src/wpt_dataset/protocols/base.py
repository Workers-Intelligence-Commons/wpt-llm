"""What every protocol shares: the name WPT's requests carry, and the
naming of a failed fetch.

named(error, address) turns what a fetch raised, once every try that may
pass was made (retry.py), into the protocol failure it is (errors.py), its
message starting with the address:

    a protocol failure already (FetchFailed and its kinds)    -> itself
    HTTP 404 or 410; git or Mercurial: no such repository     -> SourceGone
    HTTP 429; git or Mercurial: the server answered 429       -> TooManyRequests
    HTTP 5xx; git or Mercurial: the server answered 5xx       -> ServerError
    HTTP 401, 403 or 451; git or Mercurial: the server        -> AccessRefused
    answered one of them, or asked for a user name, a
    password or a key (they run so they never wait for one)
    no connection made (urllib's URLError, other than an      -> HostUnreachable
    HTTP answer); git or Mercurial: a host name that does not
    resolve, a connection refused or never made
    an answer that stopped before its end (archives.          -> Incomplete
    EndedEarly, an incomplete read, a connection cut or too
    slow once made); git: a transfer cut before its end
    anything else UNREACHABLE gives: the general fetch        -> FetchFailed
    failure (another HTTP answer such as 400, an answer
    http.client cannot read, a git or Mercurial message none
    of the above names, an OSError that is not the
    network's, such as a full disk)

The robots.txt failures are robots.py's own (CrawlForbidden,
RobotsUnreadable, HostUnreachable) and pass through as they are.
"""

import http.client
import subprocess
import urllib.error

from ..archives import EndedEarly
from ..errors import AccessRefused, FetchFailed, HostUnreachable, Incomplete, ServerError, SourceGone, TooManyRequests

USER_AGENT = "wpt-dataset (+https://github.com/Workers-Intelligence-Commons/wpt-llm)"
UNREACHABLE = (OSError, urllib.error.URLError, subprocess.CalledProcessError, http.client.HTTPException)
GONE_HTTP, REFUSED_HTTP, DENIED_HTTP = (404, 410), 429, (401, 403, 451)
# what git (through curl) and Mercurial print, lowercased, for each failure; read in this order
GIT_REFUSED = ("returned error: 429", "http error 429")
GIT_SERVER = ("returned error: 5", "http error 5")
GIT_DENIED = ("returned error: 401", "returned error: 403", "returned error: 451", "http error 401",
              "http error 403", "http error 451", "could not read username", "could not read password",
              "terminal prompts disabled", "authentication failed", "http authorization required",
              "permission denied (publickey")
GIT_UNREACHABLE = ("could not resolve host", "failed to connect", "couldn't connect to server", "connection refused",
                   "connection timed out", "name or service not known", "nodename nor servname",
                   "temporary failure in name resolution", "no route to host", "network is unreachable")
GIT_CUT = ("early eof", "unexpected disconnect", "transfer closed", "the remote end hung up unexpectedly",
           "invalid index-pack output", "premature end", "operation timed out", "connection reset")
GIT_GONE = ("repository not found", "not found", "does not appear to be a git repository", "returned error: 404",
            "returned error: 410", "http error 410", "does not exist")
GIT_KINDS = ((GIT_REFUSED, TooManyRequests), (GIT_SERVER, ServerError), (GIT_DENIED, AccessRefused),
             (GIT_UNREACHABLE, HostUnreachable), (GIT_CUT, Incomplete), (GIT_GONE, SourceGone))


def reason(error) -> str:
    """What a failure says: git's own message for a git command."""
    if isinstance(error, subprocess.CalledProcessError) and error.stderr:
        return error.stderr.decode(errors="replace").strip()
    return str(error)


def named(error: Exception, address: str) -> FetchFailed:
    """The protocol failure error is (see the module's docstring)."""
    if isinstance(error, FetchFailed):
        return error
    return kind(error)(f"{address}: {reason(error)}")


def kind(error: Exception) -> type[FetchFailed]:
    """The class of protocol failure error is (see the module's docstring)."""
    if isinstance(error, urllib.error.HTTPError):
        return (SourceGone if error.code in GONE_HTTP else TooManyRequests if error.code == REFUSED_HTTP else
                AccessRefused if error.code in DENIED_HTTP else ServerError if 500 <= error.code < 600 else
                FetchFailed)
    if isinstance(error, urllib.error.URLError):
        return HostUnreachable
    if isinstance(error, (EndedEarly, http.client.IncompleteRead, ConnectionError, TimeoutError)):
        return Incomplete
    if isinstance(error, subprocess.CalledProcessError):
        said = reason(error).lower()
        return next((found for texts, found in GIT_KINDS if any(text in said for text in texts)), FetchFailed)
    return FetchFailed
