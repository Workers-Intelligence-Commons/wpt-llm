import json
import os
import re
import subprocess

import pytest

from wpt_agent import ImageError
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


def test_a_lock_pins_the_base_the_archive_date_and_packages_and_every_machine_builds_from_it(root):
    # the lock records what the image holds, and apt reads the dated
    # archive, so a build from the lock installs exactly the pinned versions
    root, tags = root
    lock = images.lock(root, "tiny", log=lambda _: None)
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
