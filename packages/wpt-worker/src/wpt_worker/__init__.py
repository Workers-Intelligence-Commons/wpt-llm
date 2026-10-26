"""The gpu worker and its link to the wpt manager (docs/machines.md).

A training run has three roles: the wpt manager, which holds the store,
the tokenizer, the run's record and the sandboxes; a gpu worker, which does
only the GPU work; sandboxes, the Docker containers attempts run in. The
worker deals only in token ids, holds no store credentials, and runs the
manager's own checkout, which the manager sends it at the start.

    link.py         the frames between the manager and a worker, over TLS with the worker's
                    pinned certificate and its secret token, and between a worker and its job
    certificate.py  the worker's own certificate
    server.py       the worker: python -m wpt_worker
    job.py          what every process of a worker's job starts with
    training.py     wpt-train's steps, as a worker's job
    manager.py      the manager's side: start_workers, the checkout it sends, the jobs it runs
    peers.py        the gradients of a run across places, exchanged between its gpu workers
"""
