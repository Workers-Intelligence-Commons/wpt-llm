"""wpt-web [--port N] [--root FOLDER]: build the interface if needed, then
serve it on http://127.0.0.1:N/ until Ctrl+C. It starts again the source
drafter runs it started that did not finish, and stops those still running
when it stops."""

import argparse
import sys

from wpt_dataset.errors import RuleViolation
from wpt_dataset.repo import Repo
from wpt_store import StoreError, store_from_settings
from wpt_utils.programs import MissingTool

from . import pages, ui
from .pages.source_drafters import Runs
from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wpt-web", description=__doc__)
    parser.add_argument("--port", type=int, default=8410)
    parser.add_argument("--root", help="the repository (default: the one holding the current folder)")
    args = parser.parse_args(argv)
    try:
        repo = Repo.find(args.root)
        static = ui.built()
        runs = Runs(repo)
        server = make_server(pages.routes(repo, source_drafter_runs=runs), static, args.port)
    except (MissingTool, RuleViolation, ui.UiBuildFailed, OSError) as error:
        print(f"wpt-web: {error}", file=sys.stderr)
        return 1
    print(f"wpt-web: http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C stops it)", flush=True)
    try:
        try:
            for folder in runs.resume(store_from_settings(repo.root / ".env")):
                print(f"wpt-web: source drafter run {folder} started again", flush=True)
        except StoreError as error:  # no store here: no run to start again, and the page says why
            print(f"wpt-web: no source drafter run started again ({type(error).__name__}: {error})", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runs.stop()
        server.server_close()
    return 0
