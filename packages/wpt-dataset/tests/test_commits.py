import json
import os
import shutil
import subprocess

import pytest

from wpt_dataset.commits import check_commits
from wpt_dataset.repo import Repo


@pytest.fixture
def wpt_git(kit, tmp_path):
    """A repository whose commits can be SSH-signed with a throwaway key."""
    root = tmp_path / "repo"
    (root / "data" / "sources" / "example-x-12345678").mkdir(parents=True)
    key = tmp_path / "key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    public = (tmp_path / "key.pub").read_text().strip()
    (root / "data" / "allowed-signers").write_text(f't@example.org namespaces="git" {public}\n')
    kit.run("git", "init", "-q", cwd=root)
    for setting, value in [("gpg.format", "ssh"), ("user.signingkey", str(key)), ("gpg.ssh.program", "ssh-keygen")]:
        kit.run("git", "config", setting, value, cwd=root)
    return root


def commit(kit, root, decision, *flags):
    path = root / "data" / "sources" / "example-x-12345678" / "source.json"
    path.write_text(json.dumps({"decision": {"decision": decision}}))
    kit.run("git", "add", "-A", cwd=root)
    kit.run("git", "commit", "-q", "-m", f"decide {decision}", *flags, cwd=root)


def test_signed_and_signed_off_decisions_pass(kit, wpt_git):
    commit(kit, wpt_git, "admitted", "-s", "-S")
    commit(kit, wpt_git, "retired", "-s", "-S")
    assert check_commits(Repo(wpt_git), "HEAD") == []


def test_unsigned_decision_and_missing_sign_off_are_reported(kit, wpt_git):
    commit(kit, wpt_git, "admitted", "-s", "-S")
    commit(kit, wpt_git, "refused", "-s")          # signed off, not signed
    commit(kit, wpt_git, "admitted", "-S")         # signed, not signed off
    problems = check_commits(Repo(wpt_git), "HEAD~2..HEAD")
    assert len(problems) == 2
    assert "is not signed by an allowed key" in problems[0]
    assert "no 'Signed-off-by: T <t@example.org>' line" in problems[1]


def test_a_key_can_only_be_added_by_an_allowed_key(kit, wpt_git, tmp_path):
    commit(kit, wpt_git, "admitted", "-s", "-S")
    other = tmp_path / "other"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(other)], check=True)
    signers = wpt_git / "data" / "allowed-signers"
    signers.write_text(signers.read_text() + f'o@example.org namespaces="git" {(tmp_path / "other.pub").read_text()}')
    kit.run("git", "add", "-A", cwd=wpt_git)
    # signed with the new key itself, which the list does not hold yet
    kit.run("git", "-c", f"user.signingkey={other}", "commit", "-q", "-s", "-S", "-m", "add my key", cwd=wpt_git)
    assert check_commits(Repo(wpt_git), "HEAD~1..HEAD") == [
        f"{kit.run('git', 'rev-parse', 'HEAD', cwd=wpt_git).decode()[:12]}: changes data/allowed-signers "
        "but is not signed by an allowed key"]


def test_gpg_signed_decisions_pass_with_an_allowed_gpg_key(kit, wpt_git, tmp_path):
    home = tmp_path / "gnupg"
    home.mkdir(mode=0o700)
    env = {**os.environ, "GNUPGHOME": str(home)}
    subprocess.run(["gpg", "--batch", "--passphrase", "", "--quick-gen-key", "T <t@example.org>", "ed25519", "sign", "never"],
                   env=env, capture_output=True, check=True)
    (wpt_git / "data" / "allowed-gpg-keys.asc").write_bytes(
        subprocess.run(["gpg", "--armor", "--export", "t@example.org"], env=env, capture_output=True, check=True).stdout)
    kit.run("git", "add", "-A", cwd=wpt_git)
    kit.run("git", "commit", "-q", "-s", "-S", "-m", "keys", cwd=wpt_git)  # SSH-signed by the allowed key
    path = wpt_git / "data" / "sources" / "example-x-12345678" / "source.json"
    path.write_text(json.dumps({"decision": {"decision": "admitted"}}))
    kit.run("git", "add", "-A", cwd=wpt_git)
    kit.run("git", "-c", "gpg.format=openpgp", "-c", f"gpg.program={shutil.which('gpg')}", "-c", "user.signingkey=t@example.org", "commit", "-q", "-s", "-S",
            "-m", "gpg decision", cwd=wpt_git, env={"GNUPGHOME": str(home)})
    assert check_commits(Repo(wpt_git), "HEAD~1..HEAD") == []
