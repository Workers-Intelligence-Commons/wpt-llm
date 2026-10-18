"""A local web interface to the repository, started with `uv run wpt-web`.

    server.py        serves the built interface, the routes pages declare
                     and their OpenAPI description (/api/openapi.json), on
                     this machine only
    ui.py            builds the interface (packages/wpt-web/ui) with Bun when
                     its sources are newer than the last build
    pages/<name>.py  one page's routes; pages/__init__.py lists the pages
    drafts.py        a model folder's draft of the next assistant message,
                     for the conversation editor
    cli.py           puts them together

A page is one Python file here and one TypeScript file in ui/src/pages/.
Pages never read or write data files themselves: every route goes through
the service layer (wpt_dataset.repo.Repo), which applies the same rules as
`wpt-dataset check` and refuses a stale save, as the `wpt-dataset` commands
do. The interface writes files and shows git's status; commits stay in git.
"""
