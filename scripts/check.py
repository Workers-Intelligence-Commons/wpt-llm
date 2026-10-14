# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Run every check CI runs, the same way, so a green run here means a green CI.

Usage:
    uv run scripts/check.py [--base REV] [--range RANGE]

In order, each one run even when an earlier one failed:

    lock      uv.lock matches every pyproject.toml
    tests     the tests of every package
    rust      the tokenizer's Rust tests
    data      wpt-dataset check: the data rules
    sources   wpt-dataset check --changed BASE: changed sources fetched and checked
    commits   wpt-dataset check-commits RANGE: sign-offs and decision signatures

BASE defaults to where this branch left its upstream branch (for example
origin/main), RANGE to BASE..HEAD: what a push would send. CI passes both
from the push or pull request. The tools come from the machine, as in CI:
uv, git, cargo, and mercurial and GnuPG for the sources that need them.
Exits with 1 when a check failed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV = os.environ.get("UV", "uv")  # uv tells the scripts it runs where it is
# this script runs in its own environment; the checks run in the project's
ENVIRON = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def upstream_base():
    for command in (["git", "merge-base", "HEAD", "@{upstream}"], ["git", "merge-base", "HEAD", "origin/main"]):
        found = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if found.returncode == 0:
            return found.stdout.strip()
    sys.exit("no upstream branch to compare with: pass --base and --range")


def checks(base, range_):
    # inside the folder, so rustup applies rust-toolchain.toml; through uv, so the
    # Rust build finds the project's Python even where none is on the search path
    rust = ROOT / "packages" / "wpt-tokenizer" / "rust"
    return [
        ("lock", [[UV, "lock", "--check"]], ROOT),
        ("tests", [[UV, "run", "pytest", "-rs"]], ROOT),
        ("rust", [[UV, "run", "rustc", "--version"], [UV, "run", "cargo", "test"]], rust),
        ("data", [[UV, "run", "wpt-dataset", "check"]], ROOT),
        ("sources", [[UV, "run", "wpt-dataset", "check", "--changed", base]], ROOT),
        ("commits", [[UV, "run", "wpt-dataset", "check-commits", range_]], ROOT),
    ]


def run(commands, folder):
    for command in commands:
        print(f"$ {' '.join(command)}", flush=True)
        try:
            if subprocess.run(command, cwd=folder, env=ENVIRON).returncode != 0:
                return False
        except FileNotFoundError:
            print(f"{command[0]}: not found on this machine", flush=True)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="revision to compare sources with (default: where HEAD left its upstream)")
    parser.add_argument("--range", dest="range_", help="commits to check (default: BASE..HEAD)")
    args = parser.parse_args()
    base = args.base or upstream_base()
    range_ = args.range_ or f"{base}..HEAD"

    results = []
    for name, commands, folder in checks(base, range_):
        print(f"\n== {name}", flush=True)
        results.append((name, run(commands, folder)))
    print("\n" + "\n".join(f"{'passed' if ok else 'FAILED'}  {name}" for name, ok in results))
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
