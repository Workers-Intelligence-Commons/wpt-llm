# The wpt manager's image (compose.yaml beside it): Python, uv, the docker
# command for the sandboxes, SkyPilot's sky command for the gpu workers it
# rents, and what the commands it runs need (git and mercurial for
# downloads, GnuPG for signed commits, rustup for the tokenizer, built from
# source with the toolchain rust-toolchain.toml names).
# The repository is not in the image: Compose mounts the checkout, and uv
# installs its libraries into the manager's volume when the container starts.
# The images are pinned by digest: a new version is a change to their
# lines, and their licenses are checked again (docs/licenses.md).
FROM docker:29.9.2-cli@sha256:5e6baf67325a79f64ed3bed6863954d5e05d0f3d4e57cbdb14809346814bbfdb AS docker

FROM python:3.14-slim-trixie@sha256:35caba83af2fe6da9a5d922f9b0156f21d1246a48278c58a08bd253303ba42e8
COPY --from=docker /usr/local/bin/docker /usr/local/bin/docker
RUN apt-get update \
    && apt-get install -y --no-install-recommends git mercurial gnupg rustup build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.5
# SkyPilot in an environment of its own, on the Python it supports, with every cloud it reaches
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    uv tool install --python 3.13 "skypilot[all]==0.14.1"
# git refuses a repository another user owns, as a bind-mounted checkout is
RUN git config --system --add safe.directory /repo
WORKDIR /repo
