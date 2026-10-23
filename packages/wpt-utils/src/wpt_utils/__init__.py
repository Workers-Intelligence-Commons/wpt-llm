"""Small helpers that several WPT packages share, one submodule each:

    wpt_utils.device     the most capable PyTorch device on this machine,
                         and the processor's name (imports torch)
    wpt_utils.programs   the outside programs WPT runs, with a clear error
                         when one is missing, and the repository commit
    wpt_utils.files      the sha256 of a file, the one way WPT hashes one
    wpt_utils.settings   a plug-in's settings, typed from the defaults its file declares

Import the submodule you need: this package itself imports nothing, so
wpt_utils.programs works where PyTorch is not installed (wpt-dataset).
"""
