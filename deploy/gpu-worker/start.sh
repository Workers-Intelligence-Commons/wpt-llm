#!/bin/sh
# The gpu worker's start (compose.yaml beside it, and skypilot.yaml): uv,
# then the worker, run by Python 3.14 with the only library the worker itself
# needs (cryptography, for its certificate); its jobs get their own
# environment. The server files are /start/start.sh and /start/wpt_worker.
set -e
if [ ! -f /start/wpt_worker/server.py ]; then
    echo "start.sh: /start/wpt_worker holds no gpu worker: copy packages/wpt-worker/src/wpt_worker from the repository beside compose.yaml (docs/machines.md)" >&2
    exit 2
fi
# a node of a cluster the manager rented in one request: its private address is its rank's line of the nodes'
# list, printed (with its rank) for the manager and given to the worker for its key
if [ "${SKYPILOT_NUM_NODES:-1}" -gt 1 ]; then
    rank="${SKYPILOT_NODE_RANK:-0}"
    WPT_WORKER_PRIVATE="$(printf '%s\n' "$SKYPILOT_NODE_IPS" | sed -n "$((rank + 1))p")"
    export WPT_WORKER_PRIVATE
    echo "wpt-node $rank $WPT_WORKER_PRIVATE"
fi
pip install --no-cache-dir --quiet uv==0.13.5
exec uv run --no-project --python 3.14 --with "cryptography>=46,<51" python -c \
    "import sys; sys.path.insert(0, '/start'); from wpt_worker.server import main; sys.exit(main())"
