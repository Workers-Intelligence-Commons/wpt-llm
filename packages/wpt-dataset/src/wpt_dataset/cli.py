"""wpt-dataset: check, fetch and build WPT data.

    wpt-dataset check                     every rule that needs no download
    wpt-dataset check --fetch [ID ...]    also fetch sources (all admitted, or the ids given):
                                          checksum, cuts, patches, license map, text
    wpt-dataset check --changed REV       fetch only sources changed since git revision REV
    wpt-dataset check-commits RANGE       DCO sign-off and signed decisions (e.g. origin/main..HEAD)
    wpt-dataset source-id ORIGIN [ITEM] ADDRESS
                                          the id a new source must use
    wpt-dataset manifest ID               fetch a source without checking its checksum and
                                          print its manifest and checksum (to fill in source.json)
    wpt-dataset unpack ID FOLDER          write a source's final files into a new folder
                                          outside the repository, to edit them
    wpt-dataset patch ID FOLDER NAME      record the folder's differences from the source's
                                          final files as its next patch, NNNN-NAME.patch
    wpt-dataset build-tokenizer NAME      train (or find) the vocabulary of a tokenizer recipe
    wpt-dataset build NAME                build (or find) a dataset recipe; prints the dataset id
    wpt-dataset migrate                   rewrite data files to the current shapes

The store comes from WPT_STORE (environment or .env at the repository root).
"""

import argparse
import subprocess
import sys

from wpt_store import StoreError, store_from_settings

from . import manifest
from .build import build_dataset, build_tokenizer
from .commits import check_commits
from .edit import unpack, write_patch
from .errors import DatasetError
from .migrate import migrate
from .repo import Repo, source_id
from .markdown import documents
from .sources import compute_upstream, source_tree


def _store(repo):
    return store_from_settings(repo.root / ".env")


def cmd_check(args, repo):
    problems = repo.check()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    ids = []
    if args.changed:
        out = subprocess.run(["git", "-C", str(repo.root), "diff", "--name-only", args.changed, "--", "data/sources"],
                             capture_output=True, text=True, check=True).stdout
        ids = sorted({p.split("/")[2] for p in out.splitlines() if p.count("/") >= 3} & set(repo.source_ids()))
    elif args.fetch is not None:
        ids = args.fetch or [s.id for s in repo.sources() if s.admitted]
    if ids:
        store = _store(repo)
        for sid in ids:
            source = repo.source(sid)
            tree, _ = source_tree(source, store)
            converted = documents(source, tree, store)
            print(f"{sid}: {len(tree)} entries, checksum, cuts, patches, license and format maps OK; "
                  f"{len(converted)} Markdown documents")
    print(f"checked {len(repo.source_ids())} sources and their recipes")
    return 0


def cmd_check_commits(args, repo):
    problems = check_commits(repo.root, args.range)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("commits OK")
    return 0


def cmd_source_id(args, repo):
    print(source_id(args.origin, args.item, args.address))
    return 0


def cmd_manifest(args, repo):
    source = repo.source(args.id)
    tree = compute_upstream(source.data, source.data["address"], _store(repo))
    sys.stdout.write(manifest.manifest_bytes(tree).decode())
    print(f"checksum: {manifest.checksum(tree)}", file=sys.stderr)
    return 0


def cmd_unpack(args, repo):
    count = unpack(repo, repo.source(args.id), _store(repo), args.folder)
    print(f"{args.folder}: {count} files of {args.id}; edit them, then `wpt-dataset patch {args.id} {args.folder} <name>`")
    return 0


def cmd_patch(args, repo):
    path = write_patch(repo, repo.source(args.id), _store(repo), args.folder, args.name)
    print(f"wrote {path.relative_to(repo.root)}; it applies and gives exactly the folder's files")
    return 0


def cmd_build_tokenizer(args, repo):
    _, vocabulary, key = build_tokenizer(repo, args.name, _store(repo))
    print(f"vocabulary {vocabulary} (fingerprint {key})")
    return 0


def cmd_build(args, repo):
    print(build_dataset(repo, args.name, _store(repo)))
    return 0


def cmd_migrate(args, repo):
    changed = migrate(repo)
    print(f"{len(changed)} files rewritten")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wpt-dataset", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="repository root (default: found from the current folder)")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    check.add_argument("--fetch", nargs="*", metavar="ID")
    check.add_argument("--changed", metavar="REV")
    check.set_defaults(run=cmd_check)
    commits = commands.add_parser("check-commits")
    commits.add_argument("range")
    commits.set_defaults(run=cmd_check_commits)
    new_id = commands.add_parser("source-id")
    new_id.add_argument("origin")
    new_id.add_argument("parts", nargs="+", metavar="[ITEM] ADDRESS")
    new_id.set_defaults(run=cmd_source_id)
    show = commands.add_parser("manifest")
    show.add_argument("id")
    show.set_defaults(run=cmd_manifest)
    unpack_ = commands.add_parser("unpack")
    unpack_.add_argument("id")
    unpack_.add_argument("folder")
    unpack_.set_defaults(run=cmd_unpack)
    patch = commands.add_parser("patch")
    patch.add_argument("id")
    patch.add_argument("folder")
    patch.add_argument("name")
    patch.set_defaults(run=cmd_patch)
    vocab = commands.add_parser("build-tokenizer")
    vocab.add_argument("name")
    vocab.set_defaults(run=cmd_build_tokenizer)
    build = commands.add_parser("build")
    build.add_argument("name")
    build.set_defaults(run=cmd_build)
    commands.add_parser("migrate").set_defaults(run=cmd_migrate)
    args = parser.parse_args(argv)
    if args.command == "source-id":
        if len(args.parts) > 2:
            parser.error("source-id takes ORIGIN [ITEM] ADDRESS")
        args.item, args.address = (args.parts[0], args.parts[1]) if len(args.parts) == 2 else (None, args.parts[0])
    try:
        repo = Repo(args.root) if args.root else Repo.find()
        return args.run(args, repo)
    except (DatasetError, StoreError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
