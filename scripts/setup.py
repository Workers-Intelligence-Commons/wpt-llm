# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Install the workspace with the PyTorch build that fits this machine's GPU.

Usage:
    uv run scripts/setup.py                 detect the GPU and install
    uv run scripts/setup.py --backend cpu   force a build: cpu, cuda, rocm, xpu

Run it once per machine, and again after a plain `uv sync`, which removes
PyTorch because no build is part of the default install. uv runs this script
in its own empty environment, so it works before anything is installed.

On Windows it ends with a note: the folders to leave out of Microsoft
Defender's scanning, and the command that does it. It changes nothing itself.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
# PCI vendor ids, in order of preference when a machine has several GPUs
VENDORS = [("10de", "NVIDIA", "cuda"), ("1002", "AMD", "rocm"), ("8086", "Intel", "xpu")]


def gpu_vendors():
    """PCI vendor ids (lowercase hex) of the display adapters present."""
    if sys.platform == "win32":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | ForEach-Object { $_.PNPDeviceID }"],
            capture_output=True, text=True,
        ).stdout
        # PNPDeviceID looks like PCI\VEN_10DE&DEV_2C58&...
        return {line.split("VEN_")[1][:4].lower() for line in out.splitlines() if "VEN_" in line}
    vendors = set()
    for device in Path("/sys/bus/pci/devices").glob("*"):
        try:
            if (device / "class").read_text().startswith("0x03"):  # display controller
                vendors.add((device / "vendor").read_text().strip().removeprefix("0x").lower())
        except OSError:
            continue
    return vendors


def pick_backend(vendors):
    for vendor_id, name, backend in VENDORS:
        if vendor_id in vendors:
            if backend == "rocm" and sys.platform != "linux":
                print(f"{name} GPU found, but PyTorch's AMD build exists for Linux only")
                continue
            return name, backend
    return None, "cpu"


def store_folder():
    """The local folder WPT_STORE names (environment, else .env at the
    root), as wpt_store reads it; None when it names no local folder."""
    # mirrors wpt_store.store_from_settings, which this cannot import: it runs before anything is installed
    url = os.environ.get("WPT_STORE")
    if not url:
        try:
            lines = (ROOT / ".env").read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        for line in map(str.strip, lines):
            if line and not line.startswith("#") and line.split("=", 1)[0].strip() == "WPT_STORE":
                url = line.split("=", 1)[1].strip().strip("'\"")
    parsed = urlparse(url or "")
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    return path[1:] if re.match(r"/[A-Za-z]:", path) else path  # file:///C:/x -> C:/x


def defender_note():
    """Where fetches write many files on Windows, and how to leave those
    folders out of Microsoft Defender's scanning."""
    store = store_folder()
    folders = ([str(Path(store))] if store else []) + [tempfile.gettempdir()]
    named = ("the store (WPT_STORE) and " if store else "") + "the temporary folder where fetches clone"
    quoted = ",".join("'" + folder.replace("'", "''") + "'" for folder in folders)
    print(f"\nMicrosoft Defender scans every file and process the fetches create, which can slow them. To leave "
          f"{named} out of its scanning, run in a PowerShell opened as administrator:\n\n"
          f"    Add-MpPreference -ExclusionPath {quoted}\n\n"
          f"Defender then never scans the files in these folders. This script changed nothing.")
    if not store:
        print("WPT_STORE names no local folder (file://...), so only the temporary folder is listed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["cpu", "cuda", "rocm", "xpu"], help="skip detection")
    args = parser.parse_args()

    if args.backend:
        backend = args.backend
        print(f"PyTorch build: {backend} (forced)")
    else:
        name, backend = pick_backend(gpu_vendors())
        print(f"PyTorch build: {backend} ({f'{name} GPU found' if name else 'no supported GPU found'})")

    uv = os.environ.get("UV", "uv")  # uv tells the scripts it runs where it is
    code = subprocess.run([uv, "sync", "--group", backend], cwd=ROOT).returncode
    if sys.platform == "win32":
        defender_note()
    sys.exit(code)


if __name__ == "__main__":
    main()
