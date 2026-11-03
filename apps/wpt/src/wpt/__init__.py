"""wpt: every WPT job, each a subcommand with its own options (`wpt
SUBCOMMAND --help`).

    wpt dataset ...   sources, the data rules, vocabularies and dataset builds
    wpt store ...     what the store holds, sources marked for a new download, a kept model copied out
    wpt agent ...     the images of agentic tasks, locked and built
    wpt train ...     train a model recipe on the dataset built from it
    wpt eval ...      run tests on a trained model, into its model card
    wpt scaling ...   fit the scaling curve and predict a loss
    wpt chat ...      continue text with a trained model, or chat with one
    wpt manager       download, build and train a model recipe in one command
    wpt doctor        check this machine: what is missing and how to fix it
    wpt web ...       the local web server, on 127.0.0.1

The subcommands and the web server (`wpt web`) are the app's two
controllers over the same services, the libraries of packages/. A
subcommand's module is imported only when it runs, so a light one (`wpt
store`) never loads PyTorch.
"""

import importlib
import sys

# each subcommand: the module whose main(argv) runs it
COMMANDS = {"dataset": "wpt.dataset", "store": "wpt.store", "agent": "wpt.agent", "train": "wpt.train",
            "eval": "wpt.eval", "scaling": "wpt.eval.scaling", "chat": "wpt.chat", "manager": "wpt.manager",
            "doctor": "wpt.doctor", "web": "wpt.web"}


def main(argv=None):
    """Run the subcommand argv names with the rest of argv; its exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if not argv or argv[0] not in COMMANDS:
        known = ", ".join(COMMANDS)
        print(f"wpt: {f'unknown subcommand {argv[0]!r}' if argv else 'no subcommand'}; one of {known} "
              "(wpt --help)", file=sys.stderr)
        return 2
    name, *rest = argv
    return importlib.import_module(COMMANDS[name]).main(rest)
