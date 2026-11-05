"""wpt web [--port N] [--root FOLDER]: build the interface if needed, then
serve it on http://127.0.0.1:N/ until Ctrl+C.

The web server is the wpt app's second controller, beside its
subcommands:

    server.py        serves the built interface, the routes pages declare
                     and their OpenAPI description (/api/openapi.json), on
                     this machine only
    ui.py            builds the interface (apps/wpt/ui) with Bun when its
                     sources are newer than the last build
    pages/<name>.py  one page's routes; pages/__init__.py lists the pages
    drafts.py        a model folder's draft of the next assistant message,
                     for the conversation editor; each failure named
    attempts.py      the attempts of the store's `wpt eval` runs, for the
                     rating page
    jobs.py          the subcommands the server runs as child processes
                     (the jobs page), and what they printed

A page is one Python file here and one TypeScript file in ui/src/pages/.
A route is a controller: it reads its request, calls a service and turns
its result or its named failure into an answer, and holds no rule of its
own, so the interface and the `wpt dataset` subcommands cannot disagree.
Pages never read or write data files themselves: every route goes through
the service layer (wpt_dataset.repo.Repo, and wpt_dataset's other services
over it), which applies the same rules as `wpt dataset check` and refuses
a stale save, as the `wpt dataset` subcommands do. The job routes run a
subcommand of `wpt` as a child process instead, and the subcommand calls
the services itself. The interface writes files and shows git's status;
commits stay in git.
"""

import argparse

from wpt_dataset.errors import RuleViolation
from wpt_dataset.repo import Repo
from wpt_utils.lines import ERROR, say
from wpt_utils.programs import MissingTool

from . import pages, ui
from .jobs import Jobs
from .server import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wpt web", description=__doc__)
    parser.add_argument("--port", type=int, default=8410)
    parser.add_argument("--root", help="the repository (default: the one holding the current folder)")
    args = parser.parse_args(argv)
    try:
        repo = Repo.find(args.root)
        static = ui.built()
        jobs = Jobs(repo.root)
        server = make_server(pages.routes(repo, running=jobs), static, args.port)
    except (MissingTool, RuleViolation, ui.UiBuildFailed, OSError) as error:
        say(f"wpt web: {error}", ERROR)
        return 1
    say(f"wpt web: http://127.0.0.1:{server.server_address[1]}/ (Ctrl+C stops it)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        jobs.close()
    return 0
