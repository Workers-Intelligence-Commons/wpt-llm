"""wpt-store: what the store named by WPT_STORE (environment or .env)
holds.

    wpt-store list                    every session (the commands that wrote the store) and every
                                      source the store database knows, with its last outcome
    wpt-store mark ID [ID ...]        mark sources for a new download: the next session that meets one
    wpt-store mark --all              downloads it in full, though the store holds it (a person's way
                                      to ask for fresh content); --all marks every source
    wpt-store copy-out NAME FOLDER    the model folder kept under the run's name NAME (what the wpt
                                      manager prints at the end of a run) into the new FOLDER, each
                                      file checked against its sha256
"""

import argparse
import sys
from pathlib import Path

from . import StoreError, store_from_settings
from .listing import listing
from .models import copy_out


def cmd_list(args, store):
    sources, sessions = listing(store)
    print(f"{len(sessions)} sessions:")
    for row in sessions:
        print(f"  {row.started} to {row.ended or '(not ended)'} {row.command} at {row.commit or '?'}"
              f"{' (changed)' if row.dirty else ''}: records/{row.folder}/")
    print(f"{len(sources)} sources:")
    for state in sources:
        last = (f"downloaded {state.downloaded} from {state.address} at {state.commit or '?'}"
                f"{f', {state.size:,} bytes' if state.size is not None else ''}"
                f"{''.join(f', {key} {value}' for key, value in sorted((state.got or {}).items()))}"
                if state.downloaded else "never downloaded")
        print(f"  {state.source}: {state.outcome}{'; marked for a new download' if state.marked else ''}; {last}")
    return 0


def cmd_mark(args, store):
    if bool(args.ids) == args.all:
        print("wpt-store mark: give source ids or --all, not both", file=sys.stderr)
        return 2
    store.mark(None if args.all else args.ids)
    print("every source marked for a new download" if args.all else
          f"{len(args.ids)} sources marked for a new download")
    return 0


def cmd_copy_out(args, store):
    count = copy_out(store, args.name, args.folder)
    print(f"model {args.name}: {count} files in {args.folder}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wpt-store", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the sessions and the sources the store database knows").set_defaults(
        run=cmd_list)
    marking = commands.add_parser("mark", help="mark sources for a new download in full")
    marking.add_argument("ids", nargs="*", help="the source ids")
    marking.add_argument("--all", action="store_true", help="every source")
    marking.set_defaults(run=cmd_mark)
    copying = commands.add_parser("copy-out", help="a model folder the store keeps, into a new folder")
    copying.add_argument("name", help="the run's name, which the wpt manager printed")
    copying.add_argument("folder", type=Path, help="the folder to write, which must not exist yet")
    copying.set_defaults(run=cmd_copy_out)
    args = parser.parse_args(argv)
    try:
        store = store_from_settings(Path.cwd() / ".env")
        return args.run(args, store)
    except (StoreError, FileExistsError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
