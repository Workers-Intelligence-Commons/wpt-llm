"""wpt-dataset: check, fetch and build WPT data.

    wpt-dataset check                     every rule that needs no download
    wpt-dataset check --fetch [ID ...]    also fetch sources (all admitted, or the ids given):
                                          checksum, cuts, patches, license map, text; every
                                          source is checked, and the failures listed at the end
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
    wpt-dataset markdown-check [ID ...]   check the Markdown made from text and xhtml files (every
                                          source, or the ids given): notes, unescaped tags
    wpt-dataset build-tokenizer NAME [--stream]
                                          train (or find) the vocabulary of a tokenizer recipe
    wpt-dataset tokenizer-check NAME      encode and decode every admitted document with that
                                          vocabulary: exact round trip, characters per token
    wpt-dataset build NAME [--stream] [--skip-failed]
                                          build (or find) the data of a model recipe; prints the
                                          dataset id
    wpt-dataset migrate                   rewrite data files to the current shapes

The store comes from WPT_STORE (environment or .env at the repository root).
--stream (or WPT_STREAM=1) fetches one source at a time and keeps only its
tokens in the store, not its files: for a corpus larger than the disk.
A failing source stops a build; --skip-failed (or WPT_SKIP_FAILED=1) leaves
it out and lists it, with its error, in the dataset record. A download that
fails for a reason that may pass (timeout, cut connection, server error) is
tried again, up to 5 times (see retry.py).
"""

import argparse
import os
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
from .markdown_check import check_markdown
from .sources import compute_upstream, source_tree
from .tokenizer_check import check_tokenizer


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
    failed = []
    if ids:
        store = _store(repo)
        for sid in ids:  # every source is checked, so one run names every failure
            source = repo.source(sid)
            try:
                tree, _ = source_tree(source, store)
                skipped = {}
                converted = sum(1 for _ in documents(source, tree, store, skipped))
            except (DatasetError, StoreError) as error:
                failed.append(f"{sid}: {type(error).__name__}: {error}")
                print(f"{sid}: FAILED ({type(error).__name__})", file=sys.stderr)
                continue
            print(f"{sid}: {len(tree)} entries, checksum, cuts, patches, license and format maps OK; "
                  f"{converted} Markdown documents")
            _print_skipped(skipped)
    print(f"checked {len(repo.source_ids())} sources and their recipes")
    if failed:
        print(f"{len(failed)} of {len(ids)} fetched sources failed:", file=sys.stderr)
        print("\n".join(failed), file=sys.stderr)
        return 1
    return 0


def _print_skipped(skipped):
    """The records each records file left out, by license value."""
    unlisted, empty, outside = {}, 0, {}
    for report in skipped.values():
        empty += report.empty
        for value, count in report.unlisted.items():
            unlisted[value] = unlisted.get(value, 0) + count
        for key, count in report.outside.items():
            outside[key] = outside.get(key, 0) + count
    for (name, value), count in sorted(outside.items(), key=lambda item: -item[1]):
        print(f"  left out: {count} records whose {name} is {value!r} (not kept by `keep`)")
    for value, count in sorted(unlisted.items(), key=lambda item: -item[1]):
        print(f"  left out: {count} records with license value {value!r} (listed by no license entry)")
    if empty:
        print(f"  left out: {empty} records with empty text")


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


def cmd_markdown_check(args, repo):
    store, failed = _store(repo), False
    for sid in args.ids or repo.source_ids():
        report = check_markdown(repo.source(sid), store)
        for problem in report.problems:
            print(f"{sid}: {problem}", file=sys.stderr)
        failed = failed or bool(report.problems)
        print(f"{sid}: {report.documents:,} documents checked, {report.notes:,} notes, "
              + (f"{len(report.problems):,} problems" if report.problems else "OK"))
    return 1 if failed else 0


def cmd_build_tokenizer(args, repo):
    _, vocabulary, key = build_tokenizer(repo, args.name, _store(repo), stream=args.stream)
    print(f"vocabulary {vocabulary} (fingerprint {key})")
    return 0


def cmd_tokenizer_check(args, repo):
    reports = check_tokenizer(repo, args.name, _store(repo))
    for r in reports:
        print(f"{r.source}: {r.documents:,} documents, {r.characters:,} characters, {r.tokens:,} tokens, "
              f"{r.characters_per_token:.2f} characters per token" + ("" if r.in_vocabulary else
                                                                      " (not in the vocabulary's training)"))
    characters, tokens = sum(r.characters for r in reports), sum(r.tokens for r in reports)
    print(f"every document decodes back exactly: {sum(r.documents for r in reports):,} documents, "
          f"{characters:,} characters, {tokens:,} tokens, {characters / max(tokens, 1):.2f} characters per token")
    return 0


def cmd_build(args, repo):
    print(build_dataset(repo, args.name, _store(repo), stream=args.stream, skip_failed=args.skip_failed))
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
    markdown_check = commands.add_parser("markdown-check")
    markdown_check.add_argument("ids", nargs="*", metavar="ID")
    markdown_check.set_defaults(run=cmd_markdown_check)
    stream = os.environ.get("WPT_STREAM", "") in ("1", "true", "yes")
    stream_help = "fetch one source at a time and keep only tokens in the store, not its files (WPT_STREAM)"
    vocab = commands.add_parser("build-tokenizer")
    vocab.add_argument("name")
    vocab.add_argument("--stream", action="store_true", default=stream, help=stream_help)
    vocab.set_defaults(run=cmd_build_tokenizer)
    vocab_check = commands.add_parser("tokenizer-check")
    vocab_check.add_argument("name")
    vocab_check.set_defaults(run=cmd_tokenizer_check)
    build = commands.add_parser("build")
    build.add_argument("name")
    build.add_argument("--stream", action="store_true", default=stream, help=stream_help)
    build.add_argument("--skip-failed", action="store_true",
                       default=os.environ.get("WPT_SKIP_FAILED", "") in ("1", "true", "yes"),
                       help="leave out admitted sources that fail and list them in the dataset record, instead of "
                            "stopping; a new vocabulary still needs every source (WPT_SKIP_FAILED)")
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
