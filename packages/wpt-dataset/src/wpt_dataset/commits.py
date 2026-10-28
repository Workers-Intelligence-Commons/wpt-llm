"""Checks on commits: DCO sign-off, and signed decisions.

Every commit (merge commits aside) must carry `Signed-off-by: <author>`: by
it the author certifies the Developer Certificate of Origin and, for a
source, that its license was read, that no "no AI" wish was found, and that
it holds no personal data (see CONTRIBUTING.md).

A commit that changes any decision (a source's, a license entry's, a
license text's: data/license-texts/<id>/license-text.json), or the lists
of keys allowed to sign decisions (data/allowed-signers for SSH keys, data/allowed-gpg-keys.asc
for GPG keys), must be signed by a key from those lists as they stood before
the commit; the very first commit is checked against its own lists. A
Signed-off-by line alone proves nothing: anyone can type it. The lists
and the decisions are read from the commits through the service layer
(Repo.signer_lists_at, Repo.signature_reasons).
"""

import os
import subprocess
import tempfile
from pathlib import Path

from wpt_utils.programs import program

from .repo import Repo


def _git(root, *args, check=True, env=None):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=check, env=env)


def _changed_paths(root, base, commit) -> list[str]:
    if base is None:
        out = _git(root, "ls-tree", "-r", "-z", "--name-only", commit).stdout
    else:
        out = _git(root, "diff", "--name-only", "-z", "--no-renames", base, commit).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def _signed_by_allowed(repo: Repo, commit, lists_at) -> bool:
    root = repo.root
    raw = _git(root, "cat-file", "commit", commit).stdout
    signers, keys = repo.signer_lists_at(lists_at)
    if b"-----BEGIN SSH SIGNATURE-----" in raw:
        if not signers:
            return False
        with tempfile.TemporaryDirectory() as tmp:
            allowed = Path(tmp) / "allowed-signers"
            allowed.write_bytes(signers)
            result = _git(root, "-c", "gpg.ssh.program=ssh-keygen", "-c", f"gpg.ssh.allowedSignersFile={allowed}",
                          "verify-commit", commit, check=False)
        return result.returncode == 0
    if b"-----BEGIN PGP SIGNATURE-----" in raw:
        if not keys:
            return False
        with tempfile.TemporaryDirectory() as home:
            env = {**os.environ, "GNUPGHOME": home}
            gpg = program("gpg")
            subprocess.run([gpg, "--batch", "--import"], input=keys, capture_output=True, env=env, check=True)
            result = _git(root, "-c", f"gpg.program={gpg}", "verify-commit", "--raw", commit, check=False, env=env)
        return result.returncode == 0 and b"[GNUPG:] VALIDSIG" in result.stderr
    return False


def check_commits(repo: Repo, revision_range: str) -> list[str]:
    """One line per problem in the commits of revision_range (e.g. origin/main..HEAD)."""
    root, problems = repo.root, []
    commits = _git(root, "rev-list", "--reverse", "--parents", revision_range).stdout.decode().split("\n")
    for line in filter(None, commits):
        commit, *parents = line.split()
        if len(parents) > 1:
            continue  # merge commits carry no changes of their own here
        base = parents[0] if parents else None
        short = commit[:12]
        author = _git(root, "show", "-s", "--format=%an <%ae>", commit).stdout.decode().strip()
        message = _git(root, "show", "-s", "--format=%B", commit).stdout.decode()
        if f"Signed-off-by: {author}" not in message.splitlines():
            problems.append(f"{short}: no 'Signed-off-by: {author}' line (commit with `git commit -s`)")
        changed = _changed_paths(root, base, commit)
        reasons = repo.signature_reasons(base, commit, changed)
        if reasons and not _signed_by_allowed(repo, commit, base or commit):
            problems.append(f"{short}: changes {', '.join(reasons)} but is not signed by an allowed key")
    return problems
