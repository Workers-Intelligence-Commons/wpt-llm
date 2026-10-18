"""Pick the most capable PyTorch device on this machine.

Order: an NVIDIA or AMD GPU (both appear as "cuda" in PyTorch), then an Intel
GPU ("xpu"), then the CPU. Among several GPUs of one kind, the one with the
most memory wins. Which of these PyTorch can use depends on the build that
scripts/setup.py installed; a missing build or driver falls back to the CPU.
processor_name() names the CPU the same way on Linux, Windows and macOS.
"""

import os
import platform
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


# The names Arm publishes for the codes /proc/cpuinfo shows on Arm processors
# ("CPU implementer", "CPU part": the fields of the MIDR_EL1 register), written
# for WPT from these facts. Implementers: Arm's "Arm Architecture Registers"
# for A-profile (DDI 0601, release 2026-03), MIDR_EL1, Implementer field.
# Parts of Arm Limited (0x41): the MIDR_EL1 PartNum value in each core's
# Technical Reference Manual. Only the cores of common Arm machines that run
# PyTorch under Linux: Raspberry Pi 3 to 5 (Cortex-A53, A72, A76), many other
# boards (Cortex-A55), NVIDIA Jetson Orin (Cortex-A78AE) and Jetson Thor
# (Neoverse-V3AE), NVIDIA DGX Spark (Cortex-X925, Cortex-A725), NVIDIA Grace
# (Neoverse-V2), and the Neoverse servers of cloud providers (N1, V1, N2, V3).
# Any other part is named by its implementer and its code, such as "Arm
# Limited part 0xd0d"; an implementer Arm does not publish (Apple's 0x61) is
# named by its code, such as "implementer 0x61 part 0x22".
IMPLEMENTERS = {0x41: "Arm Limited", 0x42: "Broadcom Corporation", 0x43: "Cavium Inc.",
                0x44: "Digital Equipment Corporation", 0x46: "Fujitsu Ltd.", 0x49: "Infineon Technologies AG",
                0x4D: "Motorola or Freescale Semiconductor Inc.", 0x4E: "NVIDIA Corporation",
                0x50: "Applied Micro Circuits Corporation", 0x51: "Qualcomm Inc.", 0x56: "Marvell International Ltd.",
                0x69: "Intel Corporation", 0xC0: "Ampere Computing"}
PARTS = {0x41: {0xD03: "Cortex-A53", 0xD05: "Cortex-A55", 0xD08: "Cortex-A72", 0xD0B: "Cortex-A76",
                0xD0C: "Neoverse-N1", 0xD40: "Neoverse-V1", 0xD42: "Cortex-A78AE", 0xD49: "Neoverse-N2",
                0xD4F: "Neoverse-V2", 0xD83: "Neoverse-V3AE", 0xD84: "Neoverse-V3", 0xD85: "Cortex-X925",
                0xD87: "Cortex-A725"}}


def processor_name() -> str:
    """The processor's name as its maker writes it (such as "AMD Ryzen AI 7
    450 w/ Radeon 860M" or "Apple M2"), read from what the system itself
    reports on each: /proc/cpuinfo on Linux (cpuinfo_name), the registry's
    ProcessorNameString of the first processor on Windows, and `sysctl -n
    machdep.cpu.brand_string` on macOS, with runs of spaces made one.
    platform.processor() is not used: it is empty on many Linux systems and
    names only the architecture on Windows and macOS. Where the system names
    nothing (no sysctl on the search path, no /proc/cpuinfo, or one with
    neither a model name nor Arm's codes), it is the machine's architecture,
    such as "aarch64"; a missing sysctl never stops the run."""
    name = ""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        elif sys.platform == "darwin":
            from wpt_utils.programs import MissingTool, program
            try:
                name = subprocess.run([program("sysctl"), "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True, check=True).stdout
            except MissingTool:
                name = ""
        else:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as cpuinfo:
                name = cpuinfo_name(cpuinfo.read())
    except (OSError, subprocess.CalledProcessError):
        name = ""
    return " ".join(name.split()) or platform.machine()


def cpuinfo_name(text: str) -> str:
    """The processor's name in the text of a Linux /proc/cpuinfo: its first
    "model name" line; without one (Linux on most Arm processors), a name
    built from the "CPU implementer" and "CPU part" codes of every core, as
    lscpu does, with IMPLEMENTERS and PARTS: the implementer, then each
    distinct part in the order the cores give them, such as "Arm Limited
    Cortex-X925 + Cortex-A725"; without those codes either, "" (the caller
    then names the architecture)."""
    implementer, cores = None, []
    for line in text.splitlines():
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "model name" and value:
            return value
        try:
            if key == "CPU implementer":
                implementer = int(value, 16)
            elif key == "CPU part" and implementer is not None:
                core = (implementer, int(value, 16))
                if core not in cores:
                    cores.append(core)
        except ValueError:
            continue
    names = []
    for implementer, part in cores:
        maker = IMPLEMENTERS.get(implementer, f"implementer {implementer:#x}")
        names.append((maker, PARTS.get(implementer, {}).get(part, f"part {part:#x}")))
    if not names:
        return ""
    if len({maker for maker, _ in names}) == 1:
        return f"{names[0][0]} " + " + ".join(part for _, part in names)
    return " + ".join(f"{maker} {part}" for maker, part in names)


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
