"""Build the interface's files with Bun, only when they are stale.

The interface's sources live in packages/wpt-web/ui (a source checkout:
uv installs workspace members in place). The build writes ui/node_modules
and ui/dist, both ignored by git.
"""

import subprocess
from pathlib import Path

from wpt_dataset.programs import program

UI = Path(__file__).resolve().parents[2] / "ui"
STAMP = ".built"  # in dist/: written last, so a failed build stays stale


class UiBuildFailed(Exception):
    """`bun install` or `bun run build` failed; its output was shown. dist/
    may hold a partial build, which stays stale and is rebuilt next time."""


def _newest_source(ui: Path) -> float:
    skip = {"node_modules", "dist"}
    return max(p.stat().st_mtime for p in ui.rglob("*")
               if p.is_file() and not skip & set(p.relative_to(ui).parts))


def built(ui: Path = UI, log=print) -> Path:
    """The folder of built files, building it first when any source is newer
    than the last build. MissingTool if Bun is not installed."""
    dist, stamp = ui / "dist", ui / "dist" / STAMP
    if stamp.is_file() and stamp.stat().st_mtime >= _newest_source(ui):
        return dist
    bun = program("bun")
    log("building the interface (bun install, bun run build)")
    for command in ([bun, "install", "--frozen-lockfile"], [bun, "run", "build"]):
        if subprocess.run(command, cwd=ui).returncode != 0:
            raise UiBuildFailed(f"`{' '.join(command[1:])}` failed in {ui}")
    stamp.write_text("", encoding="utf-8")
    return dist
