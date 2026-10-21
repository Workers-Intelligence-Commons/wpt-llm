"""Container images for attempts, built on each machine from the repository.

    data/images/<name>/Dockerfile      written by a person (and any file it copies)
    data/images/<name>/lock.json       written by lock(): the inputs, pinned
    data/images/<name>/licenses.json   written by lock(): every installed package and its licenses

The Dockerfile names its base as `ARG BASE=<image>` before `FROM ${BASE}`,
and installs `${PACKAGES}` (`ARG PACKAGES="<names>"`) with apt: images are
Debian or Ubuntu, so dpkg lists what they hold and apt reads their dated
archive (snapshot.debian.org, snapshot.ubuntu.com). The supported bases
are Debian 13 (trixie) and Ubuntu 24.04 (noble) or later, whose apt reads
APT::Snapshot; an older apt ignores it and would read today's archive.

lock() pulls the base, pins it by digest, takes the current moment as the
archive's date (the snapshot), checks the snapshot layer (below), builds
the image with the Dockerfile's packages, and writes the version of every
installed package to lock.json and its licenses to licenses.json; it
refuses to replace a lock unless forced. For each package, licenses.json
holds the `License:` fields of its /usr/share/doc/<package>/copyright when
that file is machine-readable, the file's whole text when it is not, and
neither when the image has no such file. Commit both.

build() builds the image from the lock: the pinned base, the snapshot, and
every package as name=version; it refuses a result whose installed
packages differ from the lock. built() gives the tag of an image this
machine has built from the folder as it is now. Both builds start from the
base plus one layer that sets apt to the snapshot (SNAPSHOT): apt then
reads the archive as it was at that moment, so a lock keeps building for
years. Both first check that layer the same way (_check_snapshot), and
refuse it unless apt, after `apt-get update`, reads its package lists only
from the snapshot hosts; an update that fails, or succeeds but fetches no
package list (the archive's server misbehaving), is tried 3 times in all
before the layer is refused with apt's errors. apt fetches the archive over HTTPS without checking
the server's certificate (the base images hold none; apt checks the
archive's signatures and every package's checksum itself), and accepts
the dated archive's expired Release files.

A built image is tagged wpt-agent/<name>:<hash of the folder's files
except licenses.json>, so a changed Dockerfile or lock builds again.

The folder is read and the two files written through the service layer
(wpt_dataset.repo): docker gets the folder's files as a tar archive on its
standard input, never a path in the repository.
"""

import hashlib
import io
import json
import re
import tarfile
import time
from datetime import UTC, datetime

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo

from . import docker
from .docker import DockerFailed

BASE = re.compile(r"^ARG\s+BASE=(\S+)\s*$", re.MULTILINE)
LICENSE = re.compile(r"^License:[ \t]*(\S[^\n]*?)[ \t]*$", re.MULTILINE)
MACHINE_READABLE = re.compile(r"\s*Format:")  # the first field of a machine-readable copyright file
INSTALLED = r"${db:Status-Abbrev}\t${Package}\t${Version}\n"
SNAPSHOT = """APT::Snapshot "{snapshot}";
Acquire::Check-Valid-Until "false";
Acquire::https::snapshot.debian.org::Verify-Peer "false";
Acquire::https::snapshot.ubuntu.com::Verify-Peer "false";
"""
SNAPSHOT_FILE = "/etc/apt/apt.conf.d/50wpt-snapshot"
SNAPSHOT_HOSTS = ("https://snapshot.debian.org/", "https://snapshot.ubuntu.com/")
UPDATE_ATTEMPTS = 3  # `apt-get update` in the snapshot layer: the archive's server may refuse requests for a while
UPDATE_PAUSE = 10  # seconds between two attempts
UPDATE_FAILED = "apt-get update failed"  # first line the check prints when the update fails, then apt's errors
# The update counts as failed when it exits non-zero, and also when it exits 0 but leaves no remote package
# list (a misbehaving server): the policy then shows only dpkg's own status, an address starting with "/".
CHECK = (f"if apt-get update -q >/tmp/update.log 2>&1 && apt-cache policy >/tmp/policy.log "
         f"&& grep -Eq '^ +-?[0-9]+ [^/ ]' /tmp/policy.log; then cat /tmp/policy.log; "
         f"else echo '{UPDATE_FAILED}'; grep -E '^[EW]:' /tmp/update.log || tail -n 5 /tmp/update.log; fi")
READ_FROM = re.compile(r"^ +-?\d+ (\S+) ", re.MULTILINE)  # a package list's address in `apt-cache policy`


class ImageError(Exception):
    """An image cannot be locked, built or found: no such folder, a lock
    that exists (lock) or does not (build), a Dockerfile without
    `ARG BASE=`, docker missing or failing, an image that is not Debian or
    Ubuntu, a base whose apt does not read the snapshot, a build
    whose packages differ from the lock, or an image this
    machine has not built. Nothing in the repository has changed; docker
    may keep the untagged layers of a failed build."""


def files(root, name) -> dict[str, tuple[bool, bytes]]:
    """{path: (executable, content)} of every file in data/images/<name>/."""
    try:
        return Repo(root).image_files(name)
    except DatasetError as error:
        raise ImageError(str(error)) from None


def tag(root, name) -> str:
    """The local tag of the image built from the folder as it is now."""
    digest = hashlib.sha256()
    for path, (_, content) in sorted(files(root, name).items()):
        if path != "licenses.json":
            digest.update(path.encode() + b"\0" + content + b"\0")
    return f"wpt-agent/{name}:{digest.hexdigest()[:16]}"


def lock(root, name, *, force=False, log=print) -> str:
    """Pin the image's inputs: write lock.json and licenses.json; with
    force, replace earlier ones. Returns lock.json's place in the
    repository (data/images/<name>/lock.json)."""
    where, repo = files(root, name), Repo(root)
    locked, licensed = repo.relative("image-lock", name), repo.relative("image-licenses", name)
    if repo.exists("image-lock", name) and not force:
        raise ImageError(f"{locked} exists; pass --force to lock the image again and replace it")
    found = BASE.search(where["Dockerfile"][1].decode("utf-8"))
    if not found:
        raise ImageError(f"{repo.relative('dockerfile', name)}: no `ARG BASE=<image>` line naming its base")
    log(f"pulling {found[1]}")
    _docker("pull", "-q", found[1])
    digests = json.loads(_docker("image", "inspect", "--format", "{{json .RepoDigests}}", found[1]))
    if not digests:
        raise ImageError(f"{found[1]} has no registry digest to pin")
    base, snapshot = digests[0], datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log(f"building {name} on {base}, packages as on {snapshot}")
    image = _build(where, base, snapshot, [])
    try:
        installed = _installed(image)
        licenses = _licenses(image, installed)
    finally:
        _docker("rmi", image)
    repo.write_generated("image-lock", name, {"schema": 1, "base": base, "snapshot": snapshot, "packages": installed})
    repo.write_generated("image-licenses", name, {"schema": 1, "packages": licenses})
    log(f"{len(installed)} packages locked in {locked}, their licenses in {licensed}; "
        f"build the image with `wpt-agent build-image {name}`")
    return locked


def build(root, name, log=print) -> str:
    """Build the image from the folder and its lock, unless this machine
    has it already. Returns its tag."""
    where, wanted, repo = files(root, name), tag(root, name), Repo(root)
    locked = repo.relative("image-lock", name)
    try:
        pinned = repo.generated("image-lock", name)
    except DatasetError as error:
        raise ImageError(str(error)) from None
    if pinned is None:
        raise ImageError(f"{locked} does not exist: run `wpt-agent lock-image {name}` and commit it")
    if docker_has(wanted):
        log(f"{wanted} is built already")
        return wanted
    log(f"building {wanted} from its lock")
    packages = [f"{package}={version}" for package, version in sorted(pinned["packages"].items())]
    image = _build(where, pinned["base"], pinned["snapshot"], ["--build-arg", f"PACKAGES={' '.join(packages)}"])
    installed = _installed(image)
    if installed != pinned["packages"]:
        _docker("rmi", image)
        changed = sorted(set(installed.items()) ^ set(pinned["packages"].items()))
        raise ImageError(f"{name}: the build installed other packages than {locked} pins, such as "
                         f"{', '.join(f'{p}={v}' for p, v in changed[:5])}; lock it again or fix the Dockerfile")
    _docker("tag", image, wanted)
    return wanted


def built(root, name) -> str:
    """The tag of the image built from the folder as it is now; ImageError
    naming the command that builds it when this machine has not."""
    wanted = tag(root, name)
    if not docker_has(wanted):
        raise ImageError(f"image {name!r} is not built on this machine for its current files: "
                         f"run `uv run wpt-agent build-image {name}`")
    return wanted


def docker_has(image) -> bool:
    """Whether this machine has the image; ImageError when docker fails."""
    return bool(_docker("image", "ls", "-q", image).strip())


def _build(where, base, snapshot, arguments) -> str:
    """The id of the image the folder's Dockerfile builds, from its files
    `where` ({path: (executable, content)}, sent to docker as the build's
    context), on base with apt set to the snapshot; first ImageError unless
    apt on that layer reads its package lists only from the snapshot
    hosts."""
    layer = f"wpt-agent/snapshot:{hashlib.sha256(f'{base} {snapshot}'.encode()).hexdigest()[:16]}"
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as archive:
        for path, text in (("Dockerfile", f"ARG BASE\nFROM ${{BASE}}\nCOPY snapshot {SNAPSHOT_FILE}\n"),
                           ("snapshot", SNAPSHOT.format(snapshot=snapshot))):
            data = text.encode()
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    _docker("build", "-q", "-t", layer, "--build-arg", f"BASE={base}", "-", stdin=context.getvalue())
    try:
        _check_snapshot(layer, base)
        return _docker("build", "-q", "--build-arg", f"BASE={layer}", *arguments, "-",
                       stdin=_context(where)).strip()
    finally:
        _docker("rmi", layer)  # only the tag: the image built on it keeps the layer


def _context(where) -> bytes:
    """The folder's files as a tar archive, docker's build context."""
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as archive:
        for path, (executable, content) in sorted(where.items()):
            info = tarfile.TarInfo(path)
            info.size, info.mode = len(content), 0o755 if executable else 0o644
            archive.addfile(info, io.BytesIO(content))
    return context.getvalue()


def _check_snapshot(layer, base):
    """ImageError unless every package list apt reads, after an update,
    comes from a snapshot host: an apt older than APT::Snapshot ignores it
    and reads today's archive, which no lock pins. A failed update, or one
    that fetches no package list, is tried again, UPDATE_ATTEMPTS times in
    all, then refused with apt's errors."""
    for attempt in range(1, UPDATE_ATTEMPTS + 1):
        policy = _docker("run", "--rm", layer, "sh", "-c", CHECK)
        if not policy.startswith(UPDATE_FAILED):
            break
        if attempt == UPDATE_ATTEMPTS:
            errors = policy.removeprefix(UPDATE_FAILED).strip()
            raise ImageError(f"{base}: apt-get update failed in the snapshot layer after {attempt} attempts: {errors}")
        time.sleep(UPDATE_PAUSE)
    lists = [address for address in READ_FROM.findall(policy) if not address.startswith("/")]  # not dpkg's own status
    other = sorted({address for address in lists if not address.startswith(SNAPSHOT_HOSTS)})
    if other:  # CHECK passes only a policy with a remote list, so `lists` is not empty here
        raise ImageError(f"{base}: apt reads {', '.join(other)}, not the dated archive "
                         f"(snapshot.debian.org, snapshot.ubuntu.com), so its apt ignores APT::Snapshot; "
                         f"use a base of Debian 13 (trixie) or Ubuntu 24.04 (noble) or later")


def _installed(image) -> dict[str, str]:
    """{package: version} of every package dpkg says is installed."""
    try:
        listing = _docker("run", "--rm", "--network", "none", image, "dpkg-query", "-W", "-f", INSTALLED)
    except ImageError as error:
        raise ImageError(f"cannot list the image's packages with dpkg-query (images are Debian or Ubuntu): "
                         f"{error}") from None
    rows = [line.split("\t") for line in listing.splitlines() if line]
    return {package: version for status, package, version in rows if status.strip() == "ii"}


def _licenses(image, installed) -> dict[str, dict]:
    """{package: {version, and licenses or text}} from the copyright files in the image."""
    archive = _docker("run", "--rm", "--network", "none", image, "sh", "-c",
                      "cd / && tar -chf - usr/share/doc/*/copyright 2>/dev/null; exit 0", text=False)
    texts = {}
    if archive:
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            for member in files:
                if member.isfile() or member.islnk():  # a link: a copyright file another package shares
                    texts[member.name.split("/")[3]] = _decode(files.extractfile(member).read())
    found = {}
    for package, version in sorted(installed.items()):
        text = texts.get(package)
        if text is None:
            found[package] = {"version": version}
        elif MACHINE_READABLE.match(text):
            found[package] = {"version": version, "licenses": sorted(set(LICENSE.findall(text)))}
        else:
            found[package] = {"version": version, "text": text}
    return found


def _decode(data: bytes) -> str:
    """A copyright file's text: UTF-8, or Latin-1 for the older files
    written in it, which keeps every byte."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _docker(*args, stdin=b"", text=True):
    try:
        out = docker.run(*args, stdin=stdin)
    except DockerFailed as error:
        raise ImageError(str(error)) from None
    return out.decode() if text else out
