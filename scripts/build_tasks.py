"""Build the program-built tasks of one kind into a task source.

Usage:
    uv run scripts/build_tasks.py KIND SOURCE --count N [--seed S] [--select EXPRESSION]

KIND is find-the-passage, evidence-removed, look-or-ask, commit-and-push or
read-the-manual (docs/post-training.md, "The task builder"). SOURCE is the id of an
`empty` source a person made for the tasks (`wpt-dataset new-source`, its
license text written, tagged `grpo`); everything under KIND/ in it becomes
the N tasks drawn with seed S (default 1) as one new patch, checked as
`wpt-dataset check --fetch` checks it. The texts come from the admitted
sources already in the store (WPT_STORE) whose documents the pool
expression selects (required for the three text kinds); commit-and-push
reads the git sources with a history, and read-the-manual no source (its
tasks ask for results of a program of the agent image). Rerunning with other arguments
adds, removes or varies tasks. The first committed task set is one run
per kind with --count 2500.

After writing, it reports what the task source holds: its tasks of each
kind, and the shares of those whose kind acts only (commit-and-push),
checks only (find-the-passage, evidence-removed, look-or-ask) or does
both (read-the-manual), which sum to 100% (maintainer's decision,
2026-10-23; builder.CLASSES).
"""

import argparse
import sys

from wpt_dataset.builder import KINDS, build, shares, write
from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_store import store_from_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("source", help="the empty source that holds the tasks")
    parser.add_argument("--count", type=int, required=True, help="tasks to build")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--select", help="a pool expression: the documents the texts come from")
    arguments = parser.parse_args()
    repo = Repo.find()
    store = store_from_settings(repo.root / ".env")
    try:
        built = build(repo, store, arguments.kind, arguments.count, arguments.seed, arguments.select,
                      log=lambda line: print(line, file=sys.stderr))
        write(repo, store, arguments.source, arguments.kind, built, repo.source_version(arguments.source))
        report = shares(repo, store, arguments.source)
    except DatasetError as error:
        sys.exit(f"build_tasks: {error}")
    print(f"{len(built)} {arguments.kind} tasks written into {arguments.source}")
    counts = ", ".join(f"{kind} {count:,}" for kind, count in report["tasks"].items())
    print(f"{arguments.source} holds {sum(report['tasks'].values()):,} built tasks ({counts}): "
          f"acting only {report['act']:.1%}, checking only {report['check']:.1%}, both {report['both']:.1%}")


if __name__ == "__main__":
    main()
