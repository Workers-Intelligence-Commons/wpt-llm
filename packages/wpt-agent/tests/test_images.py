import json
import os
import re
import subprocess
from contextlib import nullcontext

import pytest

from wpt_agent import ImageError
from wpt_agent import docker
from wpt_agent import images
from wpt_agent.cli import main
from wpt_dataset.repo import Repo

# the documented shape: the base as an argument, packages installed with apt
DOCKERFILE = """ARG BASE=ubuntu:24.04
FROM ${BASE}
ARG PACKAGES="hello"
RUN apt-get update && apt-get install -y --no-install-recommends ${PACKAGES} && rm -rf /var/lib/apt/lists/* \\
    && echo ${PACKAGES} > /packages.txt
"""


@pytest.fixture
def root(tmp_path, image):
    """A repository holding data/images/tiny/, a Dockerfile on IMAGE; the
    images the test tags are removed afterwards."""
    (tmp_path / "data" / "images" / "tiny").mkdir(parents=True)
    (tmp_path / "data" / "images" / "tiny" / "Dockerfile").write_text(DOCKERFILE)
    tags = []
    yield tmp_path, tags
    for tag in tags:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.mark.image_build
def test_a_lock_pins_the_base_the_archive_date_and_packages_and_every_machine_builds_from_it(root):
    # the lock records what the image holds, and apt reads the dated
    # archive, so a build from the lock installs exactly the pinned versions
    root, tags = root
    lock = root / images.lock(root, "tiny", log=lambda _: None)
    pinned = json.loads(lock.read_text())
    assert pinned["base"].startswith("ubuntu@sha256:") and re.fullmatch(r"\d{8}T\d{6}Z", pinned["snapshot"])
    assert pinned["packages"]["hello"] and pinned["packages"]["bash"]
    licenses = json.loads((lock.parent / "licenses.json").read_text())["packages"]
    assert sorted(licenses) == sorted(pinned["packages"])
    assert licenses["bash"]["version"] == pinned["packages"]["bash"]
    assert licenses["bash"]["licenses"]  # bash's copyright file is machine-readable
    assert Repo(root).check() == []  # both files match their schemas
    with pytest.raises(ImageError, match="lock.json exists; pass --force"):
        images.lock(root, "tiny", log=lambda _: None)

    tag = images.tag(root, "tiny")
    tags.append(tag)
    with pytest.raises(ImageError, match="not built on this machine.*run `uv run wpt-agent build-image tiny`"):
        images.built(root, "tiny")
    assert images.build(root, "tiny", log=lambda _: None) == tag == images.built(root, "tiny")
    run = ["docker", "run", "--rm", "--network", "none", tag, "sh", "-c",
           "cat /packages.txt; apt-config dump APT::Snapshot"]
    shown = subprocess.run(run, capture_output=True, text=True, check=True).stdout
    assert f"hello={pinned['packages']['hello']}" in shown.split()  # every package came in pinned by version
    assert f'APT::Snapshot "{pinned["snapshot"]}";' in shown

    del pinned["packages"]["bash"]  # the base installs it anyway
    lock.write_text(json.dumps(pinned))
    tags.append(images.tag(root, "tiny"))
    with pytest.raises(ImageError, match="the build installed other packages than"):
        images.build(root, "tiny", log=lambda _: None)


def test_an_image_without_a_lock_is_not_built(root):
    root, _ = root
    with pytest.raises(ImageError, match="lock.json does not exist: run `wpt-agent lock-image tiny`"):
        images.build(root, "tiny")
    with pytest.raises(ImageError, match="no image 'other'"):
        images.build(root, "other")


def test_the_tag_comes_from_the_files_git_does_not_ignore(tmp_path):
    # a __pycache__ that a Python run left in an image's folder is ignored
    # by git: it changes neither the tag nor what docker gets, so two
    # machines with the same commit build the same tag; a new file git
    # does not ignore changes it
    folder = tmp_path / "data" / "images" / "tiny"
    folder.mkdir(parents=True)
    (folder / "Dockerfile").write_text(DOCKERFILE)
    (folder / "check.py").write_text("print(1)\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    before = images.tag(tmp_path, "tiny")
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "check.cpython-314.pyc").write_bytes(b"\0compiled")
    assert images.tag(tmp_path, "tiny") == before and sorted(images.files(tmp_path, "tiny")) == ["Dockerfile",
                                                                                                 "check.py"]
    (folder / "notes.txt").write_text("new\n")
    assert images.tag(tmp_path, "tiny") != before


IGNORES_SNAPSHOT = (r"apt reads http://deb\.debian\.org/debian.*not the dated archive.*"
                    r"use a base of Debian 13 \(trixie\) or Ubuntu 24\.04")


def test_a_base_whose_apt_ignores_the_snapshot_is_neither_locked_nor_built(root):
    # Debian 12's apt predates APT::Snapshot and would read today's archive:
    # a lock of it would pin nothing, so lock refuses and writes nothing, and
    # build refuses such a base even when a lock names it
    root, tags = root
    dockerfile = root / "data" / "images" / "tiny" / "Dockerfile"
    dockerfile.write_text(DOCKERFILE.replace("ubuntu:24.04", "debian:bookworm-slim"))
    with pytest.raises(ImageError, match=IGNORES_SNAPSHOT):
        images.lock(root, "tiny", log=lambda _: None)
    assert not (dockerfile.parent / "lock.json").exists()

    base = subprocess.run(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}",
                           "debian:bookworm-slim"], capture_output=True, text=True, check=True).stdout.strip()
    (dockerfile.parent / "lock.json").write_text(json.dumps(
        {"schema": 1, "base": base, "snapshot": "20260101T000000Z", "packages": {"hello": "2.10-3"}}))
    tags.append(images.tag(root, "tiny"))
    with pytest.raises(ImageError, match=IGNORES_SNAPSHOT):
        images.build(root, "tiny", log=lambda _: None)


UPDATE_ERROR = "apt-get update failed\nE: Failed to fetch https://snapshot.debian.org/ 503  Service Unavailable\n"
GOOD_POLICY = " 500 https://snapshot.debian.org/archive/debian/20260101T000000Z trixie/main amd64 Packages\n"


def test_a_failed_apt_update_is_tried_again_and_then_told_as_it_is(monkeypatch):
    # the snapshot server refused requests for a while: that is no old apt,
    # so the message names apt's error, not the base
    runs = []
    answers = [UPDATE_ERROR, UPDATE_ERROR, GOOD_POLICY]
    monkeypatch.setattr(images, "PAUSES", (0, 0))
    monkeypatch.setattr(images, "_docker", lambda *args, **_: runs.append(args) or answers[len(runs) - 1])
    images._check_snapshot("layer", "debian:trixie")
    assert len(runs) == 3

    runs.clear()
    answers[:] = [UPDATE_ERROR] * 3
    with pytest.raises(ImageError, match=r"^debian:trixie: apt-get update failed in the snapshot layer.*"
                                         r"E: Failed to fetch https://snapshot\.debian\.org/ 503") as refused:
        images._check_snapshot("layer", "debian:trixie")
    assert len(runs) == 3
    assert "use a base of" not in str(refused.value)


BUILD_ERROR = b"""#5 [2/2] RUN echo line-from-run; echo err-from-run >&2; exit 100
------
 > [2/2] RUN echo line-from-run; echo err-from-run >&2; exit 100:
0.145 line-from-run
0.145 err-from-run
------
Dockerfile:2
--------------------
   1 |     FROM busybox
   2 | >>> RUN echo line-from-run; echo err-from-run >&2; exit 100
   3 |     
--------------------
ERROR: failed to build: failed to solve: process "/bin/sh -c ..." did not complete successfully: exit code: 100
"""


def fake_build(monkeypatch, tmp_path, answers):
    """Docker's `build` answers one by one: an image id, or the stderr of a failed build."""
    runs = []

    def run(*args, stdin=b""):
        runs.append(args)
        answer = answers[len(runs) - 1]
        if isinstance(answer, bytes):
            raise docker.failed("build", 1, answer)
        (tmp_path / "id").write_text(answer)
        return b""

    monkeypatch.setattr(images, "PAUSES", (0, 0))
    monkeypatch.setattr(images.docker, "run", run)
    return runs


def test_a_failed_docker_build_is_tried_again_and_then_told_with_the_step_that_failed(tmp_path, monkeypatch):
    runs = fake_build(monkeypatch, tmp_path, [BUILD_ERROR, BUILD_ERROR, "sha256:abc\n"])
    monkeypatch.setattr(images.tempfile, "TemporaryDirectory", lambda: nullcontext(str(tmp_path)))
    assert images._docker_build(["-t", "layer"], b"context") == "sha256:abc"
    assert len(runs) == 3
    assert runs[0][:3] == ("build", "--progress=plain", "--iidfile") and runs[0][-3:] == ("-t", "layer", "-")

    runs = fake_build(monkeypatch, tmp_path, [BUILD_ERROR] * 3)
    with pytest.raises(ImageError, match=r"^docker build failed after 3 attempts: ------\n > \[2/2\] RUN ") as refused:
        images._docker_build([], b"context")
    assert len(runs) == 3
    assert "0.145 line-from-run\n0.145 err-from-run\n" in str(refused.value)
    assert str(refused.value).endswith("exit code: 100")
    assert "#5" not in str(refused.value)


def test_a_failed_docker_build_without_a_step_keeps_the_last_20_lines(tmp_path, monkeypatch):
    fake_build(monkeypatch, tmp_path, ["\n".join(f"line {number}" for number in range(1, 31)).encode()] * 3)
    with pytest.raises(ImageError) as refused:
        images._docker_build([], b"context")
    assert str(refused.value).split(": ", 1)[1] == "\n".join(f"line {number}" for number in range(11, 31))


def test_a_docker_that_cannot_start_is_not_tried_again(monkeypatch):
    runs = []

    def run(*args, stdin=b""):
        runs.append(args)
        raise docker.DockerFailed("docker build: [Errno 2] No such file or directory")

    monkeypatch.setattr(images.docker, "run", run)
    with pytest.raises(ImageError, match="^docker build: .*No such file"):
        images._docker_build([], b"context")
    assert len(runs) == 1


NO_LIST = "apt-get update failed\nW: Failed to fetch https://snapshot.ubuntu.com/ Connection timed out\n"
OTHER_POLICY = " 500 http://deb.debian.org/debian bookworm/main amd64 Packages\n"
REMOTE = " 500 https://snapshot.debian.org/archive/debian/1 trixie/main amd64 Packages\n"
DPKG = " 100 /var/lib/dpkg/status\n"


def test_an_update_that_fetches_no_package_list_is_tried_again_like_a_failed_one(monkeypatch):
    # apt-get update exits 0 when the server misbehaves, and the check then
    # prints the same marker: that is no old apt either
    runs = []
    answers = [NO_LIST, GOOD_POLICY]
    monkeypatch.setattr(images, "PAUSES", (0, 0))
    monkeypatch.setattr(images, "_docker", lambda *args, **_: runs.append(args) or answers[len(runs) - 1])
    images._check_snapshot("layer", "ubuntu:noble")
    assert len(runs) == 2

    runs.clear()
    answers[:] = [NO_LIST] * 3
    with pytest.raises(ImageError, match=r"^ubuntu:noble: apt-get update failed in the snapshot layer after 3 attempts: "
                                         r"W: Failed to fetch https://snapshot\.ubuntu\.com/ Connection timed out$") as refused:
        images._check_snapshot("layer", "ubuntu:noble")
    assert len(runs) == 3
    assert "use a base of" not in str(refused.value)


def test_a_base_that_reads_other_hosts_is_refused_at_once(monkeypatch):
    runs = []
    monkeypatch.setattr(images, "_docker", lambda *args, **_: runs.append(args) or OTHER_POLICY)
    with pytest.raises(ImageError, match=r"^debian:bookworm: apt reads http://deb\.debian\.org/debian, not the dated archive"):
        images._check_snapshot("layer", "debian:bookworm")
    assert len(runs) == 1


@pytest.mark.skipif(os.name == "nt", reason="the check is a shell command")
@pytest.mark.parametrize("update_exits, policy, prints", [
    (0, REMOTE + DPKG, REMOTE + DPKG),
    (0, DPKG, "apt-get update failed\nW: a warning\n"),  # exit 0, yet no remote package list
    (100, REMOTE, "apt-get update failed\nW: a warning\n"),
])
def test_the_check_prints_the_policy_only_when_the_update_left_a_remote_list(tmp_path, monkeypatch, update_exits,
                                                                              policy, prints):
    stand_ins = tmp_path / "bin"
    stand_ins.mkdir()
    (stand_ins / "apt-get").write_text(f"#!/bin/sh\necho 'W: a warning'\nexit {update_exits}\n")
    (stand_ins / "apt-cache").write_text(f"#!/bin/sh\nprintf '%s' '{policy}'\n")
    for stand_in in stand_ins.iterdir():
        stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stand_ins}{os.pathsep}{os.environ['PATH']}")
    check = images.CHECK.replace("/tmp/", f"{tmp_path}/")  # the container's own /tmp, here the test's folder
    assert subprocess.run(["sh", "-c", check], capture_output=True, text=True).stdout == prints


def a_repository(tmp_path):
    for folder in ("data/sources", "data/images/tiny"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "data" / "images" / "tiny" / "Dockerfile").write_text(DOCKERFILE)
    return tmp_path


@pytest.mark.skipif(os.name == "nt", reason="the stand-in docker is a shell script")
def test_docker_down_is_told_apart_from_an_image_not_built(tmp_path, monkeypatch):
    # "build the image" would send a person to the wrong fix when the daemon is
    # what failed: the error is docker's own
    root = a_repository(tmp_path / "repo")
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "docker"
    fake.write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    with pytest.raises(ImageError, match="^docker image: Cannot connect to the Docker daemon$"):
        images.built(root, "tiny")


def test_an_image_name_that_is_a_path_is_refused_with_a_message(tmp_path, monkeypatch):
    monkeypatch.chdir(a_repository(tmp_path))
    with pytest.raises(SystemExit, match=r"^ImageError: no image '/etc': an image name is lowercase"):
        main(["build-image", "/etc"])
