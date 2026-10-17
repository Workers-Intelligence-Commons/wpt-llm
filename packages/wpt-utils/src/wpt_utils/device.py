"""Pick the most capable PyTorch device on this machine.

Order: an NVIDIA or AMD GPU (both appear as "cuda" in PyTorch), then an Intel
GPU ("xpu"), then the CPU. Among several GPUs of one kind, the one with the
most memory wins. Which of these PyTorch can use depends on the build that
scripts/setup.py installed; a missing build or driver falls back to the CPU.
"""

import os
import subprocess
import sys

import torch

OVERRIDE = "HSA_OVERRIDE_GFX_VERSION"
KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def select_device() -> torch.device:
    """Return the device to run on.

    Effect, AMD on Linux only: when no installed GPU is in the PyTorch build's
    supported chip list, sets HSA_OVERRIDE_GFX_VERSION (unless already set) so
    the GPU runs the code of the closest supported chip of its family. ROCm
    reads it when the GPU is first used, so this must run before any GPU work.
    Without an override, GPUs whose chip the build lacks are never picked.
    """
    usable = None  # GPU indices worth considering; None = all
    if torch.version.hip and OVERRIDE not in os.environ:
        supported = _supported_amd_chips()
        override = _amd_override(_installed_amd_chips(), supported)
        if override:
            os.environ[OVERRIDE] = override
        else:  # skip GPUs this build cannot run, such as an integrated one beside a supported card
            usable = lambda i: _chip(torch.cuda.get_device_properties(i).gcnArchName) in supported
    if torch.cuda.is_available():
        index = _largest(torch.cuda, usable)
        if index is not None:
            return torch.device("cuda", index)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", _largest(torch.xpu))
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    """'cuda:0 (AMD Radeon 860M)', or 'cpu'."""
    if device.type == "cpu":
        return "cpu"
    return f"{device} ({getattr(torch, device.type).get_device_name(device)})"


def _largest(backend, usable=None):
    """Index of the usable GPU with the most memory, or None."""
    indices = [i for i in range(backend.device_count()) if usable is None or usable(i)]
    if not indices:
        return None
    return max(indices, key=lambda i: backend.get_device_properties(i).total_memory)


# -- AMD chip support --------------------------------------------------------
# A chip is named gfx<major><minor><step>, the last two as one hex digit each
# (gfx1152 = 11.5.2, gfx90a = 9.0.10). A build only runs on the chips it was
# compiled for; the override makes ROCm treat the GPU as another chip.

def _amd_override(installed, supported):
    """Override version for the first installed chip with a supported relative,
    or None when no override is needed or none would help."""
    if not installed:
        return None
    if any(chip in supported for chip in installed):
        return None  # at least one GPU runs natively; never break it with an override
    for chip in installed:
        same_family = [s for s in supported if s[0] == chip[0]]
        if same_family:
            closest = min(same_family, key=lambda s: (abs(s[1] - chip[1]), abs(s[2] - chip[2])))
            return ".".join(map(str, closest))
    return None  # no relative in this build: leave it to fall back to the CPU


def _installed_amd_chips():
    """(major, minor, step) of every GPU the AMD kernel driver reports."""
    chips = []
    try:
        nodes = os.listdir(KFD_NODES)
    except OSError:
        return chips
    for node in nodes:
        try:
            with open(f"{KFD_NODES}/{node}/properties") as f:
                fields = dict(line.split() for line in f if line.strip())
        except OSError:
            continue
        version = int(fields.get("gfx_target_version", 0))
        if version:  # 0 = a CPU node
            chips.append((version // 10000, version // 100 % 100, version % 100))
    return chips


def _supported_amd_chips():
    """Chips the installed PyTorch build was compiled for.

    Asking PyTorch starts the GPU runtime, after which the override would come
    too late, so a short child process asks instead.
    """
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(' '.join(torch.cuda.get_arch_list()))"],
        capture_output=True, text=True,
    ).stdout.split()
    return [chip for chip in map(_chip, out) if chip]


def _chip(name):
    """'gfx1152' -> (11, 5, 2); 'gfx942:sramecc+' -> (9, 4, 2); else None."""
    name = name.split(":")[0]
    if not name.startswith("gfx") or len(name) < 6:
        return None
    return (int(name[3:-2]), int(name[-2], 16), int(name[-1], 16))
