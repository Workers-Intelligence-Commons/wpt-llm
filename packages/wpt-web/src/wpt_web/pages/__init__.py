"""Every page's routes. A new page is a file here, listed below, and its
TypeScript file in ui/src/pages/ (listed in ui/src/pages/index.ts)."""

from ..drafts import Drafter
from . import conversations, documents, ratings, source_drafters, sources, tasks
from .common import Context


def routes(repo, store=None, drafter=None, source_drafter_runs=None):
    """The routes over repo; store: the store a source's files are read
    from (default: WPT_STORE, opened when first needed); drafter: what
    writes model drafts (default: a Drafter); source_drafter_runs: the
    source drafter runs started here (default: a source_drafters.Runs)."""
    context = Context(repo, store)
    return [*sources.routes(context), *documents.routes(context),
            *conversations.routes(context, drafter or Drafter()), *tasks.routes(context),
            *ratings.routes(context),
            *source_drafters.routes(context, source_drafter_runs or source_drafters.Runs(repo))]
