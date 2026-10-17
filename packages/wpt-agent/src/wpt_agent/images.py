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
from the snapshot hosts. apt fetches the archive over HTTPS without
checking the server's certificate (the base images hold none; apt checks
the archive's signatures and every package's checksum itself), and
accepts the dated archive's expired Release files.

A built image is tagged wpt-agent/<name>:<hash of the folder's files
except licenses.json>, so a changed Dockerfile or lock builds again.
"""

import hashlib
import io
import json
import re
import tarfile
from datetime import UTC, datetime

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo, dump_json, load_json

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
READ_FROM = re.compile(r"^ +-?\d+ (\S+) ", re.MULTILINE)  # a package list's address in `apt-cache policy`


class ImageError(Exception):
    """An image cannot be locked, built or found: no such folder, a lock
    that exists (lock) or does not (build), a Dockerfile without
    `ARG BASE=`, docker missing or failing, an image that is not Debian or
    Ubuntu, a base whose apt does not read the snapshot, a build
    whose packages differ from the lock, or an image this
    machine has not built. Nothing in the repository has changed; docker
    may keep the untagged layers of a failed build."""


def folder(root, name):
    try:
        return Repo(root).image_folder(name)
    except DatasetError as error:
        raise ImageError(str(error)) from None


def tag(root, name) -> str:
    """The local tag of the image built from the folder as it is now."""
    where, licenses = folder(root, name), Repo(root).image_file(name, "image-licenses")
    digest = hashlib.sha256()
    for path in sorted(p for p in where.rglob("*") if p.is_file() and p != licenses):
        digest.update(path.relative_to(where).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return f"wpt-agent/{name}:{digest.hexdigest()[:16]}"


def lock(root, name, *, force=False, log=print):
    """Pin the image's inputs: write lock.json and licenses.json; with
    force, replace earlier ones. Returns lock.json's path."""
    where, repo = folder(root, name), Repo(root)
    locked, licensed = repo.image_file(name, "image-lock"), repo.image_file(name, "image-licenses")
    if locked.exists() and not force:
        raise ImageError(f"{locked} exists; pass --force to lock the image again and replace it")
    found = BASE.search(repo.dockerfile(name).read_text(encoding="utf-8"))
    if not found:
        raise ImageError(f"{repo.dockerfile(name)}: no `ARG BASE=<image>` line naming its base")
    log(f"pulling {found[1]}")
    _docker("pull", "-q", found[1])
    digests = json.loads(_docker("image", "inspect", "--format", "{{json .RepoDigests}}", found[1]))
    if not digests:
        raise ImageError(f"{found[1]} has no registry digest to pin")
    base, snapshot = digests[0], datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log(f"building {name} on {base}, packages as on {snapshot}")
    image = _build(where, repo.dockerfile(name), base, snapshot, [])
    try:
        installed = _installed(image)
        licenses = _licenses(image, installed)
    finally:
        _docker("rmi", image)
    locked.write_text(dump_json({"schema": 1, "base": base, "snapshot": snapshot, "packages": installed}),
                      encoding="utf-8", newline="\n")
    licensed.write_text(dump_json({"schema": 1, "packages": licenses}), encoding="utf-8", newline="\n")
    log(f"{len(installed)} packages locked in {locked}, their licenses in {licensed}; "
        f"build the image with `wpt-agent build-image {name}`")
    return locked


def build(root, name, log=print) -> str:
    """Build the image from the folder and its lock, unless this machine
    has it already. Returns its tag."""
    where, wanted, locked = folder(root, name), tag(root, name), Repo(root).image_file(name, "image-lock")
    if not locked.exists():
        raise ImageError(f"{locked} does not exist: run `wpt-agent lock-image {name}` and commit it")
    try:
        pinned = load_json(locked, "image-lock")
    except DatasetError as error:
        raise ImageError(str(error)) from None
    if docker_has(wanted):
        log(f"{wanted} is built already")
        return wanted
    log(f"building {wanted} from its lock")
    packages = [f"{package}={version}" for package, version in sorted(pinned["packages"].items())]
    image = _build(where, Repo(root).dockerfile(name), pinned["base"], pinned["snapshot"],
                   ["--build-arg", f"PACKAGES={' '.join(packages)}"])
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


def _build(where, dockerfile, base, snapshot, arguments) -> str:
    """The id of the dockerfile's image, built in the folder `where` on
    base with apt set to the snapshot; first ImageError unless apt on that
    layer reads its package lists only from the snapshot hosts."""
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
        return _docker("build", "-q", "-f", str(dockerfile), "--build-arg", f"BASE={layer}", *arguments,
                       str(where)).strip()
    finally:
        _docker("rmi", layer)  # only the tag: the image built on it keeps the layer


def _check_snapshot(layer, base):
    """ImageError unless every package list apt reads, after an update,
    comes from a snapshot host: an apt older than APT::Snapshot ignores it
    and reads today's archive, which no lock pins."""
    policy = _docker("run", "--rm", layer, "sh", "-c", "apt-get update -qq >/dev/null && apt-cache policy")
    lists = [address for address in READ_FROM.findall(policy) if not address.startswith("/")]  # not dpkg's own status
    other = sorted({address for address in lists if not address.startswith(SNAPSHOT_HOSTS)})
    if not lists or other:
        raise ImageError(f"{base}: apt reads {', '.join(other) or 'no package list'}, not the dated archive "
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
