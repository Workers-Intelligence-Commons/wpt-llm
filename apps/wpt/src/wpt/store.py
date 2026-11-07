"""wpt store: what the store named by WPT_STORE (environment or .env)
holds.

    wpt store list                    every session of the store's index, or with an ID the session and
                                      the one it started from: what each holds, and each source of it
                                      with its time, address and size, or its error
    wpt store copy-out NAME FOLDER    the model folder the session keeps under the run's name NAME
                                      (what the wpt manager prints at the end of a run) into the new
                                      FOLDER, each file checked against its sha256
    wpt store runs                    every run recorded in the store, or with an ID in the session and the
                                      one it started from: its folder, command, start and how it ended
    wpt store run FOLDER [--follow]   one run (FOLDER, or a part of its name only that folder holds):
                                      its stage, its counts, its last events and its errors; --follow
                                      prints its lines as they reach the store until it ends
    wpt store rebuild-index           the store's index made again from the files of every session

Each takes --session ID (WPT_SESSION) and --from OLD_ID. `list`, `runs` and
`run` work with no ID: they show every session, which is how a person finds
an ID, and an ID narrows them to it. `rebuild-index` needs none and ignores
it, since it covers the whole store. `copy-out` is refused (SessionNeeded)
with no ID.
"""

import argparse
import sys
from pathlib import Path

from wpt import sessions
from wpt_store import StoreError, index, store_from_settings
from wpt_store.listing import listing
from wpt_store.models import copy_out
from wpt_telemetry import runs
from wpt_utils.lines import ERROR, say


def cmd_list(args, store):
    states, rows = listing(store)
    if store.session:
        shown = store.reading()
        states = [state for state in states if state.session in shown]
        rows = [row for row in rows if row.session in shown]
    say(f"{len(rows)} sessions:")
    for row in rows:
        say(f"  {row.session}: {_counts(row.held)}")
    say(f"{len(states)} sources:")
    for state in states:
        last = (f"failed: {state.error}" if state.error else
                f"from {state.address}{f', {state.size:,} bytes' if state.size is not None else ''}"
                f"{''.join(f', {key} {value}' for key, value in sorted((state.got or {}).items()))}")
        say(f"  {state.session}/{state.source}: {state.time} {last}")
    return 0


def cmd_copy_out(args, store):
    count = copy_out(store, args.name, args.folder)
    say(f"model {args.name}: {count} files in {args.folder}")
    return 0


def cmd_runs(args, store):
    found = runs.runs(store)
    for run in found:
        took = f" in {run.seconds:,.0f} s" if run.seconds is not None else ""
        say(f"{run.started or '?'} {run.command or '?'}: {run.status}{took}: {run.folder}")
    say(f"{len(found)} runs")
    return 0


def cmd_run(args, store):
    try:
        folder = runs.find(store, args.folder)
    except ValueError as error:
        say(f"wpt store run: {error}", ERROR)
        return 2
    if args.follow:
        for text, level in runs.follow(store, folder):
            say(text, level, sys.stdout)
    found = runs.summary(store, folder)
    say(f"{found.run.folder}: {found.run.command or '?'} {' '.join(found.arguments)}")
    say(f"started {found.run.started or '?'}; {found.run.status}"
        + (f" in {found.run.seconds:,.0f} s" if found.run.seconds is not None else ""))
    if found.error:
        say(f"error: {found.error}")
    say(f"stage: {' > '.join(found.stage) or '(no span open)'}")
    say(f"{found.lines:,} printed lines; spans: {_counts(found.spans)}; events: {_counts(found.events)}")
    for title, lines in (("last events", found.last), ("errors", found.errors)):
        if lines:
            say(f"{title}:")
            for text, level in lines:
                say(f"  {text}", level, sys.stdout)
    return 0


def cmd_rebuild_index(args, store):
    say(f"the index holds {index.rebuild(store):,} things")
    return 0


def _counts(counted: dict) -> str:
    return ", ".join(f"{name} {count:,}" for name, count in sorted(counted.items())) or "none"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wpt store", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="every session, or the one given, and what it holds").set_defaults(run=cmd_list)
    copying = commands.add_parser("copy-out", help="a model folder the store keeps, into a new folder")
    copying.add_argument("name", help="the run's name, which the wpt manager printed")
    copying.add_argument("folder", type=Path, help="the folder to write, which must not exist yet")
    copying.set_defaults(run=cmd_copy_out)
    commands.add_parser("runs", help="every run recorded in the store, or in the session given").set_defaults(run=cmd_runs)
    showing = commands.add_parser("run", help="one run: its stage, counts, last events and errors")
    showing.add_argument("folder", help="its record folder, or a part of the name only that folder holds")
    showing.add_argument("--follow", action="store_true", help="print its lines as they reach the store until it ends")
    showing.set_defaults(run=cmd_run)
    commands.add_parser("rebuild-index", help="the store's index made again from the files of every session"
                        ).set_defaults(run=cmd_rebuild_index)
    for command in commands.choices.values():
        sessions.options(command)
    args = parser.parse_args(argv)
    env_file = Path.cwd() / ".env"
    try:
        store = store_from_settings(env_file)
        if args.run is cmd_copy_out:
            store = sessions.opened(store, args, env_file)
        elif args.run is not cmd_rebuild_index:
            store = sessions.viewed(store, args, env_file)
        return args.run(args, store)
    except (StoreError, FileExistsError) as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return 2


if __name__ == "__main__":
    sys.exit(main())
