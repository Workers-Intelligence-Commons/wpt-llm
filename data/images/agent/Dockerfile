# The first image of agentic tasks (docs/post-training.md, "The agent image"): Debian with bash, git
# and Python, the shared check library (wpt_check.py, which every task's check calls), and the
# programs that answer the tools tasks declare (search, fetch) and the image's pool (tools.json).
ARG BASE=debian:trixie-slim
FROM ${BASE}
ARG PACKAGES="bash coreutils git python3"
RUN apt-get update && apt-get install -y --no-install-recommends ${PACKAGES} && rm -rf /var/lib/apt/lists/*
# on the path of every python3, -I and -P included, never in a folder the model writes
COPY wpt_check.py wpt_tools.py /usr/lib/python3/dist-packages/
COPY tools/ /usr/local/lib/wpt-tools/
