import platform
import subprocess
import sys

import wpt_utils.programs
from wpt_utils.device import cpuinfo_name, processor_name

X86 = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen AI 7 450 w/ Radeon 860M
flags\t\t: fpu vme

processor\t: 1
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen AI 7 450 w/ Radeon 860M
"""

# an NVIDIA DGX Spark's two kinds of cores, as its /proc/cpuinfo gives them
BIG_LITTLE = "".join(f"""processor\t: {index}
BogoMIPS\t: 2000.00
Features\t: fp asimd evtstrm
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x0
CPU part\t: {part}
CPU revision\t: 1

""" for index, part in enumerate(["0xd85", "0xd85", "0xd87", "0xd87", "0xd85"]))


def board(*cores):
    return "".join(f"processor\t: {index}\nCPU implementer\t: {implementer}\nCPU part\t: {part}\n\n"
                   for index, (implementer, part) in enumerate(cores))


def test_a_linux_processor_is_named_by_its_model_name_line():
    assert cpuinfo_name(X86) == "AMD Ryzen AI 7 450 w/ Radeon 860M"


def test_an_arm_processor_without_a_model_name_is_named_from_its_codes():
    # as lscpu does, from Arm's published implementer and part names, each
    # kind of core once in the order the cores give them (maintainer's
    # decision, 2026-10-18)
    assert cpuinfo_name(BIG_LITTLE) == "Arm Limited Cortex-X925 + Cortex-A725"
    assert cpuinfo_name(board(("0x41", "0xd0b"))) == "Arm Limited Cortex-A76"


def test_an_unknown_arm_part_or_implementer_is_named_by_its_codes():
    assert cpuinfo_name(board(("0x41", "0xd0d"))) == "Arm Limited part 0xd0d"
    assert cpuinfo_name(board(("0x51", "0x001"))) == "Qualcomm Inc. part 0x1"
    assert cpuinfo_name(board(("0x61", "0x022"), ("0x61", "0x023"))) == "implementer 0x61 part 0x22 + part 0x23"
    assert (cpuinfo_name(board(("0x41", "0xd03"), ("0x4e", "0x004")))
            == "Arm Limited Cortex-A53 + NVIDIA Corporation part 0x4")


def test_without_a_model_name_or_codes_the_name_is_left_to_the_architecture():
    assert cpuinfo_name("processor\t: 0\nBogoMIPS\t: 108.00\nCPU part\t: 0xd03\n") == ""
    assert cpuinfo_name("CPU implementer\t: unknown\nCPU part\t: 0xd03\n") == ""


def test_a_missing_sysctl_on_macos_names_the_architecture(monkeypatch):
    # a missing sysctl never stops a run (maintainer's decision, 2026-10-18)
    def missing(name):
        raise wpt_utils.programs.MissingTool(f"{name} is not installed (or not on the search path)")

    def never(*args, **kwargs):
        raise AssertionError("sysctl was started")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(wpt_utils.programs, "program", missing)
    monkeypatch.setattr(subprocess, "run", never)
    assert processor_name() == platform.machine()
