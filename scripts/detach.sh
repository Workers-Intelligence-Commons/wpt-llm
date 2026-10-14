#!/bin/sh
# Run a command on this Linux or macOS machine so it keeps going after the
# terminal or SSH session closes.
#
# Usage, from anywhere in the repository:
#     scripts/detach.sh uv run wpt-train <dataset id>
#
# Closing a session sends its programs a "hang up" signal, which ends them.
# nohup makes the command ignore it, and the command runs in the background
# from the repository folder. Output goes to runs/logs/<date-time>.log, as
# with scripts/detach.ps1 on Windows.
set -eu
if [ "$#" -eq 0 ]; then
    echo 'usage: scripts/detach.sh <command> [arguments...]' >&2
    exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/runs/logs"
# Create the log only if no run started in the same second has it (noclobber
# makes `>` refuse an existing file); otherwise add -1, -2...
stamp="$root/runs/logs/$(date +%Y%m%d-%H%M%S)"
log="$stamp.log"
n=0
until (set -C; : > "$log") 2>/dev/null; do
    n=$((n + 1))
    log="$stamp-$n.log"
done

# PYTHONUNBUFFERED: Python writes each line to the log at once instead of in
# blocks. PYTHONIOENCODING: the log is UTF-8 whatever the locale says.
cd "$root"
# Ignore "hang up" here already: a session that closes right after this
# script returns would otherwise reach the command before nohup is running.
trap '' HUP
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 nohup "$@" > "$log" 2>&1 < /dev/null &
pid=$!

echo "started: $*"
echo "process: $pid"
echo "follow:  tail -f '$log'"
echo "stop:    kill $pid"
