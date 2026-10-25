# The first image of agentic tasks (docs/training.md, "The agent image"): Debian with bash, git
# and Python, the shared check library (wpt_check.py, which every task's check calls), and the
# programs that answer the tools tasks declare (search, fetch) and the image's pool (tools.json);
# and ondrik, the program of the read-the-manual tasks, compiled in a stage of its own so the image
# holds only the binary, which the model's user may run but not read, with its manual, which a task
# may remove (its `absent`).
ARG BASE=debian:trixie-slim
FROM ${BASE} AS ondrik
# the compiler, from the same dated archive as the image's packages; none of it reaches the image
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY ondrik/ondrik.c /src/
RUN gcc -O2 -std=c11 -o /src/ondrik /src/ondrik.c

FROM ${BASE}
ARG PACKAGES="bash coreutils git python3"
RUN apt-get update && apt-get install -y --no-install-recommends ${PACKAGES} && rm -rf /var/lib/apt/lists/*
# on the path of every python3, -I and -P included, never in a folder the model writes
COPY wpt_check.py wpt_tools.py /usr/lib/python3/dist-packages/
COPY tools/ /usr/local/lib/wpt-tools/
# owned by root, executable by every user, readable by none but root: the model can run it, not read it
COPY --from=ondrik --chmod=0711 /src/ondrik /usr/local/bin/ondrik
COPY ondrik/manual.txt /usr/share/doc/ondrik/
