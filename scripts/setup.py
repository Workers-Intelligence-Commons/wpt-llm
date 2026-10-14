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
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    sys.exit(subprocess.run([uv, "sync", "--group", backend], cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
