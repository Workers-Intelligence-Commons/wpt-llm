# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Run every check CI runs, the same way, so a green run here means a green CI.

Usage:
    uv run scripts/check.py [--base REV] [--range RANGE]

In order, each one run even when an earlier one failed:

    lock      uv.lock matches every pyproject.toml
    tests     the tests of every package, in parallel with one PyTorch thread
              each, on as many processes as 75% of the cores and 75% of the
              available memory (2.5 GB each) both allow; the image build
              test (marked image_build) only when BASE..HEAD changes
              IMAGE_CODE or an image's folder (plain `uv run pytest` runs
              every test, serially)
    rust      the tokenizer's Rust tests
    data      wpt-dataset check: the data rules
    sources   wpt-dataset check --changed BASE: changed sources fetched and checked
              (every source with tasks too when data/images/ changed)
    commits   wpt-dataset check-commits RANGE: sign-offs and decision signatures

BASE defaults to where this branch left its upstream branch (for example
origin/main), RANGE to BASE..HEAD: what a push would send. CI passes both
from the push or pull request. The tools come from the machine, as in CI:
uv, git, cargo, and mercurial and GnuPG for the sources that need them.
Exits with 1 when a check failed.
"""

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UV = os.environ.get("UV", "uv")  # uv tells the scripts it runs where it is
# this script runs in its own environment; the checks run in the project's
ENVIRON = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
WORKER_MEMORY = 2.5e9  # bytes: the measured memory of one test worker (about 2.2 GB, rounded up)
SHARE = 0.75  # the share of the cores and of the available memory the test workers may take
# what the image build test covers: a change to any of these, or to an image's folder, runs it
IMAGE_CODE = ("packages/wpt-agent/src/wpt_agent/images.py", "packages/wpt-agent/tests/test_images.py",
              "packages/wpt-agent/tests/conftest.py",
              "packages/wpt-dataset/src/wpt_dataset/schemas/image-lock.schema.json",
              "packages/wpt-dataset/src/wpt_dataset/schemas/image-licenses.schema.json")
# whether an image's folder changed since the revision: only the service layer names the data folder
IMAGES_CHANGED = "import sys; from wpt_dataset.repo import Repo; print(Repo('.').changed_since(sys.argv[1])[1])"


def available_memory():
    """Bytes of memory available to new processes, or None when this
    machine does not say: MemAvailable on Linux, free, inactive and
    speculative pages on macOS, ullAvailPhys on Windows."""
    try:
        if sys.platform == "linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            lines = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout.splitlines()
            page = int(lines[0].split("page size of")[1].split()[0])
            pages = {name.strip(): int(count.strip(" .")) for name, count in (line.split(":", 1) for line in lines[1:])
                     if count.strip(" .").isdigit()}
            return page * sum(pages[name] for name in ("Pages free", "Pages inactive", "Pages speculative"))
        elif sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            *((name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys",
                              "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                              "ullAvailExtendedVirtual"))]
            status = MemoryStatus(dwLength=ctypes.sizeof(MemoryStatus))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullAvailPhys
    except (OSError, ValueError, IndexError, KeyError, subprocess.CalledProcessError):
        pass
    return None


def workers():
    """How many test processes to run, and the line that says why: the
    smaller of 75% of the cores and 75% of the available memory divided by
    WORKER_MEMORY, at least 1; the cores alone when the memory is unknown."""
    cores = os.cpu_count() or 1
    by_cores = max(1, int(cores * SHARE))
    memory = available_memory()
    if memory is None:
        return by_cores, f"{by_cores} test workers: {SHARE:.0%} of {cores} cores (the available memory is unknown here)"
    by_memory = max(1, int(memory * SHARE / WORKER_MEMORY))
    count = min(by_cores, by_memory)
    return count, (f"{count} test workers: the fewer of {SHARE:.0%} of {cores} cores ({by_cores}) and {SHARE:.0%} of "
                   f"{memory / 1e9:.1f} GB available at {WORKER_MEMORY / 1e9} GB each ({by_memory})")


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


def checks(base, range_):
    """(name, commands, folder, the line printed before them or None) of every check."""
    # inside the folder, so rustup applies rust-toolchain.toml; through uv, so the
    # Rust build finds the project's Python even where none is on the search path
    rust = ROOT / "packages" / "wpt-tokenizer" / "rust"
    deselect, why = image_tests(base)
    count, sizing = workers()
    return [
        ("lock", [[UV, "lock", "--check"]], ROOT, None),
        ("tests", [[UV, "run", "pytest", "-rs", "-n", str(count), "--dist", "loadgroup", *deselect]], ROOT,
         f"{sizing}\n{why}"),
        ("rust", [[UV, "run", "rustc", "--version"], [UV, "run", "cargo", "test"]], rust, None),
        ("data", [[UV, "run", "wpt-dataset", "check"]], ROOT, None),
        ("sources", [[UV, "run", "wpt-dataset", "check", "--changed", base]], ROOT, None),
        ("commits", [[UV, "run", "wpt-dataset", "check-commits", range_]], ROOT, None),
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
    for name, commands, folder, why in checks(base, range_):
        print(f"\n== {name}", flush=True)
        if why:
            print(why, flush=True)
        results.append((name, run(commands, folder)))
    print("\n" + "\n".join(f"{'passed' if ok else 'FAILED'}  {name}" for name, ok in results))
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
