"""A local web interface to the repository, started with `uv run wpt-web`.

    server.py        serves the built interface and the routes pages declare,
                     on this machine only
    ui.py            builds the interface (packages/wpt-web/ui) with Bun when
                     its sources are newer than the last build
    pages/<name>.py  one page's routes; pages/__init__.py lists the pages
    cli.py           puts them together

A page is one Python file here and one TypeScript file in ui/src/pages/.
Pages never write data files themselves: they go through the library that
owns them (wpt_dataset.Repo.save_source for source.json), which applies the
same rules as `wpt-dataset check`.
"""
