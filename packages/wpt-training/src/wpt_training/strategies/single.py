"""One process on the best device of this machine (a GPU, else the CPU)."""

import os

from wpt_utils.device import select_device
from wpt_training import PluginError, Strategy


def start():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:  # each process would train alone and write the same folder
        raise PluginError("single trains in one process; with --processes or torchrun, use --strategy ddp")
    return Strategy(select_device())
