"""Every page's routes. A new page is a file here, listed below, and its
TypeScript file in ui/src/pages/ (listed in ui/src/pages/index.ts)."""

from ..drafts import Drafter
from . import conversations, documents, sources, tasks
from .common import Context


def routes(repo, store=None, drafter=None):
    """The routes over repo; store: the store a source's files are read
    from (default: WPT_STORE, opened when first needed); drafter: what
    writes model drafts (default: a Drafter)."""
    context = Context(repo, store)
    return [*sources.routes(context), *documents.routes(context),
            *conversations.routes(context, drafter or Drafter()), *tasks.routes(context)]
