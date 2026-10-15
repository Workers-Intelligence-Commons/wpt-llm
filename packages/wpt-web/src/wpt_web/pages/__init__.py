"""Every page's routes. A new page is a file here, listed below, and its
TypeScript file in ui/src/pages/ (listed in ui/src/pages/index.ts)."""

from . import sources


def routes(repo):
    return [*sources.routes(repo)]
