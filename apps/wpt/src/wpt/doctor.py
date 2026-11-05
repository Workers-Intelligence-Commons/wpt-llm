"""wpt doctor: check this machine before work, in one pass, and print what
is missing and how to fix each thing. It changes nothing and writes no
record, so it runs on a machine whose store does not answer.

    store      WPT_STORE names a store, and it answers with these credentials; a file:// store's
               folder exists: opening a store creates what it lacks, so a mistyped path would pass
    PyTorch    a build is installed, and a GPU it can use is here; a machine that trains
               through the gpu workers WPT_GPU_WORKERS lists needs none of its own
    attention  what computes the attention here (wpt_model.sparse; WPT_KERNELS names it), and
               the CUDA compiler TileLang's kernels need, on an NVIDIA GPU; nothing to check when
               WPT_GPU_WORKERS lists the gpu workers, where it is decided
    Rust       rustc and cargo, which build the tokenizer
    disk       the free space of the store's local folder, beside the bytes of the sources the
               store holds, which the next build reads: two facts, with no verdict
    checkout   no uncommitted or untracked files, which change every key on the code

Exit codes: 0 nothing is missing, 1 something is, 2 not in a checkout of the
repository.
"""

import os
import shutil
import subprocess
import sys

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo
from wpt_store import local_folder, read_env_file, store_from_settings
from wpt_store.listing import listing
from wpt_utils.lines import ERROR, INFO, say
from wpt_utils.programs import MissingTool, program

MISSING, REFUSED = 1, 2
SETUP = "run `uv run scripts/setup.py`, which installs the build for this machine's GPU"


def check_store(root, environ) -> tuple[list, object]:
    """(its findings, the store, None when it does not answer)."""
    try:
        folder = local_folder(environ.get("WPT_STORE", ""))
        if folder is not None and not folder.is_dir():
            return [(False, f"store: {folder} does not exist",
                     "correct WPT_STORE, or create that folder for a new store")], None
        store = store_from_settings(root / ".env")
        store.get_ref("datasets", "wpt-doctor")  # a read: no such ref, but the store has to answer
    except Exception as error:  # noqa: BLE001: whatever the store fails with is what this command reports
        return [(False, f"store: {type(error).__name__}: {error}",
                 "set WPT_STORE and its keys (WPT_STORE_*) in the environment or in .env")], None
    return [(True, f"store: {store}", None)], store


def check_torch(environ) -> tuple[list, object]:
    """(its findings, the device PyTorch would train on here, None without
    PyTorch)."""
    try:
        import torch

        from wpt_utils.device import describe, select_device
    except ImportError:
        return [(False, "PyTorch: not installed", SETUP)], None
    device = select_device()
    built = f"PyTorch {torch.__version__}"
    if device.type != "cpu":
        try:
            (torch.zeros(1, device=device) + 1).item()
        except RuntimeError as error:
            return [(False, f"{built}: {describe(device)} does not run: {error}", SETUP)], None
        return [(True, f"{built}: {describe(device)}", None)], device
    if environ.get("WPT_GPU_WORKERS"):
        return [(True, f"{built}: no GPU here, and WPT_GPU_WORKERS names the gpu workers", None)], device
    return [(False, f"{built}: no GPU it can use", f"on a machine with a GPU, {SETUP}; or list gpu workers in "
             "WPT_GPU_WORKERS")], device


def check_attention(device, environ) -> list:
    import torch

    from wpt_model.sparse import KernelUnavailable, kernels, nvcc

    try:
        chosen = kernels(device, environ.get("WPT_KERNELS"))
    except (KernelUnavailable, ValueError) as error:
        return [(False, f"attention: {error}", "name other kernels with --kernels or WPT_KERNELS, or fix what the "
                 "line says (docs/model.md, \"Attention\")")]
    found = [(True, f"attention: {chosen.NAME + ' kernels' if chosen else 'PyTorch'}", None)]
    if device.type == "cuda" and torch.version.cuda:
        compiler = nvcc()
        found.append((compiler is not None, f"nvcc: {compiler or 'not found'}",
                      None if compiler else "TileLang's kernels need it: install the CUDA Toolkit and put nvcc on "
                      "PATH or set CUDA_HOME; the Triton kernels run without it"))
    return found


def check_rust() -> list:
    found = []
    for name in ("rustc", "cargo"):
        try:
            version = subprocess.run([program(name), "--version"], capture_output=True, text=True, check=True)
            found.append((True, version.stdout.strip(), None))
        except (MissingTool, OSError, subprocess.CalledProcessError):
            found.append((False, f"{name}: not found", "install a Rust toolchain (https://rustup.rs): the tokenizer "
                          "is built from source"))
    return found


def check_disk(root, store) -> list:
    folder = store.scratch.parent if store is not None and store.scratch.parent.exists() else root
    free = shutil.disk_usage(folder).free
    text = f"disk: {free / 10**9:,.1f} GB free in {folder}"
    if store is not None:
        sources, _ = listing(store)
        text += f"; the store holds {sum(source.size or 0 for source in sources) / 10**9:,.1f} GB of sources"
    return [(True, text, None)]


def check_checkout(root) -> list:
    try:
        listed = subprocess.run([program("git"), "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=True).stdout
    except (MissingTool, OSError, subprocess.CalledProcessError) as error:
        return [(False, f"checkout: {error}", "install git, and work in a clone of the repository")]
    lines = listed.splitlines()
    if not lines:
        return [(True, "checkout: no uncommitted or untracked files", None)]
    untracked = sum(line.startswith("??") for line in lines)
    return [(False, f"checkout: {len(lines) - untracked} uncommitted and {untracked} untracked files, which change "
             "every key on the code", "commit them, or remove them")]


def main(argv=None):
    code = _main(sys.argv[1:] if argv is None else list(argv))
    if code:
        sys.exit(code)


def _main(argv) -> int:
    if argv:
        print(__doc__)
        return 0 if argv[0] in ("-h", "--help") else REFUSED
    try:
        root = Repo.find().root
    except DatasetError as error:
        say(f"{type(error).__name__}: {error}", ERROR)
        return REFUSED
    environ = {**read_env_file(root / ".env"), **{key: value for key, value in os.environ.items() if value}}
    found, store = check_store(root, environ)
    torch_found, device = check_torch(environ)
    found += torch_found
    if device is not None and not environ.get("WPT_GPU_WORKERS"):
        found += check_attention(device, environ)
    found += check_rust() + check_disk(root, store) + check_checkout(root)
    for good, text, fix in found:
        level = INFO if good else ERROR
        say(f"{'ok' if good else 'MISSING'}  {text}", level, sys.stdout)
        if fix:
            say(f"    fix: {fix}", level, sys.stdout)
    missing = sum(not good for good, _, _ in found)
    say("nothing is missing" if not missing else f"{missing} thing(s) missing", ERROR if missing else INFO, sys.stdout)
    return MISSING if missing else 0
