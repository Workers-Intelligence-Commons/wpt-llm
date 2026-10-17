"""wpt-agent: the commands of the agent harness.

Usage:
    wpt-agent lock-image NAME [--force]
    wpt-agent build-image NAME

Both work on the image data/images/NAME/ in the repository holding the
current folder, and need Docker.

lock-image pins the image's inputs: it pulls the Dockerfile's base, takes
the current moment as the date of the package archive apt reads, builds
the image, and writes lock.json (the base by digest, that date, every
installed package's version) and licenses.json (every installed package
and its licenses). It refuses to replace an existing lock without --force.
Commit both files.

build-image builds the image from its lock on this machine, which the
post-training steps that run tasks need first. It does nothing when the
image is built already.

Both refuse a base whose apt, after `apt-get update`, reads a package list
from anywhere else than snapshot.debian.org or snapshot.ubuntu.com.
"""

import argparse
import sys

from wpt_dataset.errors import DatasetError
from wpt_dataset.repo import Repo

from . import images


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wpt-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock-image", help="pin an image's base, archive date and packages in its lock.json")
    lock.add_argument("name", help="a folder name in data/images/")
    lock.add_argument("--force", action="store_true", help="replace an existing lock.json and licenses.json")
    build = commands.add_parser("build-image", help="build an image from its lock.json on this machine")
    build.add_argument("name", help="a folder name in data/images/")
    args = parser.parse_args(argv)
    try:
        root = Repo.find().root
        if args.command == "lock-image":
            images.lock(root, args.name, force=args.force)
        else:
            print(images.build(root, args.name))
    except (images.ImageError, DatasetError) as error:
        sys.exit(f"{type(error).__name__}: {error}")
