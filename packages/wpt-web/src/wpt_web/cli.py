"""wpt-web [--port N] [--root FOLDER]: build the interface if needed, then
serve it on http://127.0.0.1:N/ until Ctrl+C."""

import argparse
import sys

from wpt_dataset.errors import MissingTool, RuleViolation
from wpt_dataset.repo import Repo

from . import pages, ui
from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wpt-web", description=__doc__)
    parser.add_argument("--port", type=int, default=8410)
    parser.add_argument("--root", help="the repository (default: the one holding the current folder)")
    args = parser.parse_args(argv)
    try:
        repo = Repo.find(args.root)
        static = ui.built()
        server = make_server(pages.routes(repo), static, args.port)
    except (MissingTool, RuleViolation, ui.UiBuildFailed, OSError) as error:
        print(f"wpt-web: {error}", file=sys.stderr)
        return 1
    print(f"wpt-web: http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C stops it)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
