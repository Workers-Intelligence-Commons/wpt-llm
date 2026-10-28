# /// script
# requires-python = ">=3.14"
# dependencies = ["wpt-utils"]
#
# [tool.uv.sources]
# wpt-utils = { path = "../packages/wpt-utils", editable = true }
# ///
"""Run every check CI runs, the same way, so a green run here means a green CI.

Usage:
    uv run scripts/check.py [--base REV] [--range RANGE]

In order, each one run even when an earlier one failed:

    lock      uv.lock matches every pyproject.toml
    tests     the tests of every package, in parallel with one PyTorch thread
              each and no GPU, after a sample of them (every SAMPLE_EVERY-th) ran alone
              and its memory was measured: on as many processes as 75% of the cores and
              75% of the available memory (that memory each) both allow; the image build
              test (marked image_build) only when BASE..HEAD changes
              IMAGE_CODE or an image's folder (plain `uv run pytest` runs
              every test, serially)
    rust      the tokenizer's Rust tests
    data      wpt-dataset check: the data rules
    sources   wpt-dataset check --fetch SMALL_SOURCE: one small source fetched and
              checked, to prove the fetch and its checks work; the sources a
              pull request adds or changes are fetched by the maintainer
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

from wpt_utils.parallel import limit, peak

ROOT = Path(__file__).resolve().parent.parent
UV = os.environ.get("UV", "uv")  # uv tells the scripts it runs where it is
# this script runs in its own environment; the checks run in the project's
ENVIRON = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
SAMPLE_EVERY = 40  # every this many-th collected test is in the sample whose memory sizes the test processes
# what the image build test covers: a change to any of these, or to an image's folder, runs it
IMAGE_CODE = ("packages/wpt-agent/src/wpt_agent/images.py", "packages/wpt-agent/tests/test_images.py",
              "packages/wpt-agent/tests/test_agent_image.py",
              "packages/wpt-agent/tests/conftest.py",
              "packages/wpt-dataset/src/wpt_dataset/repo.py",  # Repo.image_files: what the image's tag covers
              "packages/wpt-dataset/src/wpt_dataset/schemas/image-lock.schema.json",
              "packages/wpt-dataset/src/wpt_dataset/schemas/image-licenses.schema.json")
# whether an image's folder changed since the revision: only the service layer names the data folder
IMAGES_CHANGED = "import sys; from wpt_dataset.repo import Repo; print(Repo('.').changed_since(sys.argv[1])[1])"
# the one source fetched, a short book, small enough to download on every run: it proves
# the fetch and its checks work; the sources a pull request adds or changes are fetched by
# the maintainer
SMALL_SOURCE = "standardebooks-joseph-conrad-ford-madox-ford-the-nature-of-a-crime-5366b0f3"


def upstream_base():
    for command in (["git", "merge-base", "HEAD", "@{upstream}"], ["git", "merge-base", "HEAD", "origin/main"]):
        found = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if found.returncode == 0:
            return found.stdout.strip()
    sys.exit("no upstream branch to compare with: pass --base and --range")


def image_tests(base):
    """pytest's arguments for the image build test, and the line that says
    whether it runs and why: it runs only when base..HEAD changes a file of
    IMAGE_CODE or an image's folder."""
    changed = subprocess.run(["git", "diff", "--name-only", base, "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if changed.returncode:
        sys.exit(f"git diff --name-only {base} HEAD failed: {changed.stderr.strip()}")
    touched = [path for path in changed.stdout.splitlines() if path in IMAGE_CODE]
    folders = subprocess.run([UV, "run", "python", "-c", IMAGES_CHANGED, base], cwd=ROOT, env=ENVIRON,
                             capture_output=True, text=True)
    if folders.returncode:
        sys.exit(f"cannot tell whether an image's folder changed since {base}: {folders.stderr.strip()}")
    if folders.stdout.strip() == "True":
        touched.append("an image's folder under data/images/")
    if touched:
        return [], f"the image build test runs: this push changes {', '.join(touched)}"
    return ["-m", "not image_build"], "the image build test is left out: this push changes no image code"


def sample_of(test_ids, every=SAMPLE_EVERY):
    """The sample of test ids that runs alone first: every `every`-th, from
    the first, in collection order, so the same tests give the same sample."""
    return test_ids[::every]


def collected(deselect):
    """The ids of the tests pytest collects with these arguments, in order."""
    found = subprocess.run([UV, "run", "pytest", "--collect-only", "-q", "-n", "0", *deselect], cwd=ROOT,
                           env=ENVIRON, capture_output=True, text=True)
    if found.returncode:
        sys.exit(f"pytest --collect-only failed:\n{found.stdout}{found.stderr}")
    return [line for line in found.stdout.splitlines() if "::" in line]


def test_processes(deselect):
    """(how many test processes, the line that says why): a sample of the
    tests runs alone, its memory is measured, and limit() sizes the
    processes from it, one test process being taken to hold that much."""
    sample = sample_of(collected(deselect))
    print(f"$ pytest {len(sample)} sample tests alone, to measure their memory", flush=True)
    code, most = peak([UV, "run", "pytest", "-q", "-n", "0", *sample], cwd=ROOT, env=ENVIRON)
    count, why = limit(most, "test workers")
    held = "its memory could not be measured" if most is None else f"it held {most / 1e9:.1f} GB at most"
    return count, f"{len(sample)} sample tests ran alone ({'passed' if code == 0 else 'FAILED'}); {held}\n{why}"


def checks(base, range_):
    """(name, what to run) of every check: what to run is a function that
    gives (commands, folder, the line printed before them or None), called
    when the check starts."""
    # inside the folder, so rustup applies rust-toolchain.toml; through uv, so the
    # Rust build finds the project's Python even where none is on the search path
    rust = ROOT / "packages" / "wpt-tokenizer" / "rust"

    def tests():
        deselect, why = image_tests(base)
        count, sizing = test_processes(deselect)
        return ([[UV, "run", "pytest", "-rs", "-n", str(count), "--dist", "loadgroup", *deselect]], ROOT,
                f"{sizing}\n{why}")

    return [
        ("lock", lambda: ([[UV, "lock", "--check"]], ROOT, None)),
        ("tests", tests),
        ("rust", lambda: ([[UV, "run", "rustc", "--version"], [UV, "run", "cargo", "test"]], rust, None)),
        ("data", lambda: ([[UV, "run", "wpt-dataset", "check"]], ROOT, None)),
        ("sources", lambda: ([[UV, "run", "wpt-dataset", "check", "--fetch", SMALL_SOURCE]], ROOT, None)),
        ("commits", lambda: ([[UV, "run", "wpt-dataset", "check-commits", range_]], ROOT, None)),
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
    parser.add_argument("--base", help="revision the push starts from, which decides whether the image build test runs (default: where HEAD left its upstream)")
    parser.add_argument("--range", dest="range_", help="commits to check (default: BASE..HEAD)")
    args = parser.parse_args()
    base = args.base or upstream_base()
    range_ = args.range_ or f"{base}..HEAD"

    results = []
    for name, prepare in checks(base, range_):
        print(f"\n== {name}", flush=True)
        commands, folder, why = prepare()
        if why:
            print(why, flush=True)
        results.append((name, run(commands, folder)))
    print("\n" + "\n".join(f"{'passed' if ok else 'FAILED'}  {name}" for name, ok in results))
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
