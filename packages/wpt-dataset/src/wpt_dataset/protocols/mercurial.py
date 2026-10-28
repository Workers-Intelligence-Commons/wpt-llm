"""The Mercurial protocol: a repository at its tip, as its address serves
it when asked.

The repository is cloned without a working copy into the empty folder the
caller gives, then read from Mercurial's own manifest and objects
(manifest.read_mercurial). Mercurial must be installed (wpt_utils.programs
names it when it is not). Its requests are Mercurial's own: no robots.txt
is read, and no site limit per time counts them (sites.py).

What it got, for the record: {"commit": the changeset the tip was}.

Failures: what protocols.base.named makes of what hg raises (SourceGone,
TooManyRequests, ServerError, AccessRefused, HostUnreachable, FetchFailed);
MissingTool; UnsupportedEntry and BadPath for what the tree cannot hold.
"""

import os
import shutil
import subprocess
from pathlib import Path

from wpt_utils.programs import program

from .. import manifest, retry
from ..manifest import Extractor


def fetch(address: str, data: dict, extractor: Extractor, tmp: Path) -> dict:
    """The files of the tip of the repository at address, added to extractor."""
    hg, repo = program("hg"), tmp / "repo"

    def clone():
        shutil.rmtree(repo, ignore_errors=True)  # a cut clone leaves a folder hg refuses
        # its messages in English; never a prompt (--noninteractive): a server that asks for a password fails
        # at once, and protocols.base.named makes it AccessRefused
        subprocess.run([hg, "clone", "--noninteractive", "-q", "-U", address, str(repo)], capture_output=True,
                       check=True, env={**os.environ, "LC_ALL": "C", "LANGUAGE": "", "HGPLAIN": "1"})

    retry.call(clone, address)
    tip = subprocess.run([hg, "-R", str(repo), "log", "-r", "tip", "-T", "{node}"], capture_output=True,
                         check=True).stdout.decode().strip()
    manifest.read_mercurial(repo, tip, extractor)
    return {"commit": tip}
