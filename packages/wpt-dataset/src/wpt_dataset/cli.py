"""wpt-dataset: check, fetch and build WPT data.

    wpt-dataset check                     every rule that needs no download
    wpt-dataset check --fetch [ID ...]    also fetch sources (all admitted, or the ids given):
                                          checksum, cuts, patches, license map, text; every
                                          source is checked, and the failures listed at the end
    wpt-dataset check --changed REV       fetch only sources changed since git revision REV, and
                                          every source with task files when data/images/ changed
    wpt-dataset check-commits RANGE       DCO sign-off and signed decisions (e.g. origin/main..HEAD)
    wpt-dataset source-id ORIGIN [ITEM] ADDRESS
                                          the id a new source must use
    wpt-dataset manifest ID               fetch a source without checking its checksum and
                                          print its manifest and checksum (to fill in source.json)
    wpt-dataset fill-checksum ID [ID ...] for draft sources (checksum all zeros): fetch each once,
                                          exclude every file that is not text (UTF-8, or the
                                          encoding it states) by its exact path, write the
                                          checksum into source.json, and check the source as
                                          `check --fetch` does, without fetching it again;
                                          every source is filled, and the failures listed at the end.
                                          It checks only the named sources, not the repository-wide
                                          rules `check` runs: run `wpt-dataset check` too
    wpt-dataset unpack ID FOLDER          write a source's final files into a new folder
                                          outside the repository, to edit them
    wpt-dataset patch ID FOLDER NAME      record the folder's differences from the source's
                                          final files as its next patch, NNNN-NAME.patch
    wpt-dataset markdown-check [ID ...]   check the Markdown made from text, xhtml and html files
                                          (every source, or the ids given): notes, unescaped tags
    wpt-dataset build-tokenizer NAME [--stream]
                                          train (or find) the vocabulary of a tokenizer recipe
    wpt-dataset tokenizer-check NAME      encode and decode every admitted document with that
                                          vocabulary: exact round trip, characters per token
    wpt-dataset build NAME [--stream] [--skip-failed]
                                          build (or find) the data of a model recipe; prints the
                                          dataset id
    wpt-dataset migrate                   rewrite data files to the current shapes

Reading and writing the data files, through the service layer (wpt_dataset.repo;
wpt-web's HTTP API does the same). Every write applies the rules of `check` and
is refused when the file changed since VERSION, the version `show` (or `files`,
`conversations`, `tasks`, `ratings`) printed; VERSION '' writes a new file:

    wpt-dataset kinds                     the kinds of data files, and who writes each
    wpt-dataset list KIND                 the names of the files of a kind
    wpt-dataset show KIND NAME            print a file (its version and problems on stderr)
    wpt-dataset save KIND NAME FILE --version VERSION
                                          write a file from FILE ('-': standard input)
    wpt-dataset status                    git's status of every changed file under data/

A source's files change only through patches (as unpack, an edit and patch):

    wpt-dataset new-source ORIGIN [ITEM] NAME --license L --attribution TEXT [--license-text FILE *]
                --by NAME [--tag TAG ...] [--file PATH=FILE ... *] [--chat PATH ...] [--format FORMAT=PATTERN ...]
                [--patch NAME]
                                          a new empty source for text written for WPT, under review
                                          (decision "under-review"): its files (Markdown for .md, else
                                          text), and chat files to write with save-conversation;
                                          --format declares a format entry (markdown=**/*.md) with no
                                          file, for files added later; without --license-text, the
                                          license text is left for a person to write (`save source`),
                                          and files come after it
    wpt-dataset files ID                  its final files: mode, size, format, tags, path
    wpt-dataset read ID PATH              print one file *
    wpt-dataset write ID PATH FILE --patch NAME --version VERSION
                                          add or replace a file, as the next patch *
    wpt-dataset remove ID PATH --patch NAME --version VERSION
    wpt-dataset tags ID PATH [TAG ...] --version VERSION
                                          give exactly that path its own tags (none: remove them)
    wpt-dataset conversations ID PATH     the conversations of a chat file: messages, roles, drafts
    wpt-dataset conversation ID PATH LINE print one conversation as JSON *
    wpt-dataset save-conversation ID PATH LINE|new [FILE] [--remove] --patch NAME --version VERSION
                                          replace, add or remove a conversation *
    wpt-dataset tasks ID                  the task folders: image, limits, files
    wpt-dataset task ID FOLDER            print a task folder as JSON {task, check, files} *
    wpt-dataset save-task ID FOLDER [FILE] [--remove] --patch NAME --version VERSION
                                          write or remove a task folder from that JSON *
    wpt-dataset ratings ID PATH           the ratings of a rating file: the attempt each rates, who, when
    wpt-dataset save-rating ID PATH LINE|new [FILE] [--remove] --patch NAME --version VERSION
                                          replace, add or remove a person's pass or fail on an attempt *

* reads or writes a person's text. An AI assistant uses only the other
commands and options, for structure and metadata, never these: it may run
new-source without --file and --license-text (CONTRIBUTING.md, "Models and
generated text").

The store comes from WPT_STORE (environment or .env at the repository root).
--stream (or WPT_STREAM=1) fetches one source at a time and keeps only its
tokens in the store, not its files: for a corpus larger than the disk.
A failing source stops a build; --skip-failed (or WPT_SKIP_FAILED=1) leaves
it out and lists it, with its error, in the dataset record. A build is
recorded in the store, in its own folder under records/ (wpt_telemetry;
docs/observability.md); when that record cannot be written, it stops with
exit code 3. A download that fails for a reason that may pass (timeout, cut
connection, server error) is tried again, up to 5 times (see retry.py); a
build records each new try.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from wpt_store import StoreError, store_from_settings
from wpt_telemetry import RECORD_FAILED, RecordError, Telemetry
from wpt_utils.programs import MissingTool

from . import manifest, retry
from .build import build_dataset, build_tokenizer
from .commits import check_commits
from .edit import unpack, write_patch
from .errors import DatasetError, InvalidFile
from .migrate import migrate
from .repo import KINDS, Repo, empty_source, source_id
from .markdown_check import check_markdown
from .sources import compute_upstream, source_tree
from .tasks import has_tasks, pointed_sources
from .tokenizer_check import check_tokenizer


def _store(repo):
    return store_from_settings(repo.root / ".env")


def cmd_check(args, repo):
    problems = repo.check()
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    unwritten = {s.id: s.note for s in repo.sources() if s.unwritten}
    for note in unwritten.values():  # the check names them; they are not a failure
        print(note)
    ids = []
    if args.changed:
        ids, images_changed, removed = repo.changed_since(args.changed)
        if images_changed:  # a task may name an image that changed or went
            ids = sorted(set(ids) | {s.id for s in repo.sources() if has_tasks(s)})
        elif ids or removed:  # a pointer task breaks when the source it points into changed or went
            store = _store(repo)
            changed = set(ids) | set(removed)  # a task source that cannot be read is checked too: the check names why
            ids = sorted(set(ids) | {s.id for s in repo.sources() if has_tasks(s) and not s.unwritten
                                    and ((pointed := _points_into(repo, s, store)) is None or pointed & changed)})
    elif args.fetch is not None:
        ids = args.fetch or [s.id for s in repo.sources() if s.admitted]
    ids = [sid for sid in ids if sid not in unwritten]  # no text to fetch or check yet: named above
    failed = []
    if ids:
        store = _store(repo)
        for sid in ids:  # every source is checked, so one run names every failure
            _check_fetched(repo, sid, store, failed)
    print(f"checked {len(repo.source_ids())} sources and their recipes")
    return _failures(failed, len(ids))


def _points_into(repo, source, store) -> set[str] | None:
    """The sources the pointer tasks of a task source point into; None for
    a task source that cannot be read (also one whose pointers name a
    source that was deleted)."""
    try:
        return pointed_sources(source, source_tree(source, store)[0], store, repo)
    except (DatasetError, StoreError, MissingTool):
        return None


def _check_fetched(repo, sid, store, failed: list[str]):
    """Check one fetched source as `check --fetch` does; a failure is
    printed and added to failed."""
    try:
        entries, converted, skipped = repo.check_fetched(sid, store)
    except (DatasetError, StoreError, MissingTool) as error:
        _failed(sid, error, failed)
        return
    print(f"{sid}: {entries} entries, checksum, cuts, patches, license and format maps OK; "
          f"{converted} Markdown documents")
    _print_skipped(skipped)


def _failed(sid, error, failed: list[str]):
    failed.append(f"{sid}: {type(error).__name__}: {error}")
    print(f"{sid}: FAILED ({type(error).__name__})", file=sys.stderr)


def _failures(failed: list[str], count: int) -> int:
    """Exit code 1 and the list of failures, if any."""
    if failed:
        print(f"{len(failed)} of {count} fetched sources failed:", file=sys.stderr)
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
    problems = check_commits(repo, args.range)
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


def cmd_fill_checksum(args, repo):
    store, failed = _store(repo), []
    for sid in args.ids:  # every source is filled and checked, so one run names every failure
        try:
            checksum, excluded = repo.fill_checksum(sid, store)
        except (DatasetError, StoreError, MissingTool) as error:
            _failed(sid, error, failed)
            continue
        print(f"{sid}: checksum {checksum}; {len(excluded)} files that are not text excluded")
        for why in excluded.values():
            print(f"{sid}: excluded, {why}")
        source = repo.source(sid)
        if source.unwritten:  # as `check` does: no text to check yet
            print(source.note)
            continue
        _check_fetched(repo, sid, store, failed)
    return _failures(failed, len(args.ids))


def cmd_unpack(args, repo):
    count = unpack(repo, repo.source(args.id), _store(repo), args.folder)
    print(f"{args.folder}: {count} files of {args.id}; edit them, then `wpt-dataset patch {args.id} {args.folder} <name>`")
    return 0


def cmd_patch(args, repo):
    written = write_patch(repo, repo.source(args.id), _store(repo), args.folder, args.name)
    print(f"wrote {written}; it applies and gives exactly the folder's files")
    return 0


def cmd_markdown_check(args, repo):
    store, failed = _store(repo), False
    for sid in args.ids or repo.source_ids():
        report = check_markdown(repo, repo.source(sid), store)
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
    store = _store(repo)
    with (Telemetry.start(store, "wpt-dataset-build", args.arguments, args.name, inputs={"recipe": args.name},
                          env_file=repo.root / ".env") as telemetry,
          retry.reporting(telemetry.log)):  # a download tried again is part of the run's record
        telemetry.log(build_dataset(repo, args.name, store, log=telemetry.log, stream=args.stream,
                                    skip_failed=args.skip_failed, telemetry=telemetry))
    return 0


def cmd_migrate(args, repo):
    changed = migrate(repo)
    print(f"{len(changed)} files rewritten")
    return 0


# -- reading and writing data files ---------------------------------------------------

def _input(file) -> bytes:
    return sys.stdin.buffer.read() if file == "-" else Path(file).read_bytes()


def _json(data: bytes, where):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidFile(f"{where}: {error}") from None


def _versioned(version, problems=()):
    print(f"version: {version}", file=sys.stderr)
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)


def cmd_kinds(args, repo):
    for kind in KINDS.values():
        writer = f"written by `{kind.made_by}`" if kind.made_by else "written by people"
        print(f"{kind.name}: {kind.form}, {writer}; {kind.description}")
    return 0


def cmd_list(args, repo):
    for name in repo.names(args.kind):
        print(name)
    return 0


def cmd_show(args, repo):
    document = repo.read(args.kind, args.name)
    if document.text is None:
        raise InvalidFile(f"{repo.relative(args.kind, args.name)} is not UTF-8 text")
    sys.stdout.write(document.text)
    _versioned(document.version, document.problems)
    return 0


def cmd_save(args, repo):
    data = _input(args.file)
    json_kind = args.kind in KINDS and KINDS[args.kind].form == "json"  # save names an unknown kind itself
    content = _json(data, args.file) if json_kind else data.decode("utf-8")
    version = repo.save(args.kind, args.name, content, args.version)
    print(f"saved {repo.relative(args.kind, args.name)}, version {version}")
    return 0


def cmd_status(args, repo):
    found = repo.status()
    if found is None:
        print("not a git working tree", file=sys.stderr)
        return 1
    for entry in found:
        print(f"{entry['state']} {entry['path']}")
    return 0


def cmd_new_source(args, repo):
    files, formats = {}, {}
    for given in args.file:
        path, _, file = given.partition("=")
        if not file:
            raise InvalidFile(f"--file {given}: write PATH=FILE")
        files[path] = _input(file)
        formats[path] = "markdown" if path.endswith(".md") else "text"
    formats.update(dict.fromkeys(args.chat, "chat"))
    for given in args.format:  # metadata only: the entry, no file
        kind, _, pattern = given.partition("=")
        if not kind or not pattern:
            raise InvalidFile(f"--format {given}: write FORMAT=PATTERN, such as markdown=**/*.md")
        formats[pattern] = kind
    text = ""  # written later
    if args.license_text is not None:
        text = _input(args.license_text).decode("utf-8")
        if not text.strip():
            raise InvalidFile(f"--license-text {args.license_text}: the file is empty; give the text, or leave the "
                              f"option out for a person to write it later")
    document = empty_source(args.name, license=args.license, attribution=args.attribution, text=text, by=args.by,
                            formats=formats, tags=args.tag)
    sid = repo.create_source(args.origin, args.item, document, files, args.patch, _store(repo) if files else None)
    print(sid)
    return 0


def cmd_files(args, repo):
    version, files = repo.source_files(args.id, _store(repo))
    for path, file in files.items():
        print(f"{file['mode']} {file['size']:>9} {file['format'] or '?'} [{' '.join(file['tags'])}] {path}")
    _versioned(version)
    return 0


def cmd_read(args, repo):
    version, data = repo.read_source_file(args.id, args.path, _store(repo))
    sys.stdout.buffer.write(data)
    _versioned(version)
    return 0


def cmd_write(args, repo):
    version = repo.change_source_files(args.id, {args.path: _input(args.file)}, args.patch, args.version, _store(repo))
    print(f"recorded as the next patch of {args.id}, version {version}")
    return 0


def cmd_remove(args, repo):
    version = repo.change_source_files(args.id, {args.path: None}, args.patch, args.version, _store(repo))
    print(f"recorded as the next patch of {args.id}, version {version}")
    return 0


def cmd_tags(args, repo):
    version = repo.set_tags(args.id, args.path, args.tags, args.version)
    print(f"saved {repo.relative('source', args.id)}, version {version}")
    return 0


def cmd_conversations(args, repo):
    version, found = repo.conversations(args.id, args.path, _store(repo))
    for line, conversation in enumerate(found, start=1):
        messages = conversation["messages"]
        drafts = [m["metadata"]["draft"] for m in messages if "draft" in m.get("metadata", {})]
        roles = ", ".join(m["role"] for m in messages)
        print(f"line {line}: {len(messages)} messages ({roles}); {len(conversation.get('tools', []))} tools; "
              f"{len(drafts)} model drafts, {sum(d['edited'] for d in drafts)} of them changed; "
              f"metadata: {', '.join(sorted(conversation.get('metadata', {}))) or 'none'}")
    _versioned(version)
    return 0


def cmd_conversation(args, repo):
    version, found = repo.conversations(args.id, args.path, _store(repo))
    if not 1 <= args.line <= len(found):
        raise InvalidFile(f"{args.id}/{args.path} has no line {args.line}")
    print(json.dumps(found[args.line - 1], indent=2, ensure_ascii=False))
    _versioned(version)
    return 0


def cmd_save_conversation(args, repo):
    line = None if args.line == "new" else int(args.line)
    if args.remove == bool(args.file):
        raise InvalidFile("give FILE, or --remove")
    conversation = None if args.remove else _json(_input(args.file), args.file)
    version = repo.save_conversation(args.id, args.path, line, conversation, args.patch, args.version, _store(repo))
    print(f"recorded as the next patch of {args.id}, version {version}")
    return 0


def cmd_tasks(args, repo):
    version, found = repo.tasks(args.id, _store(repo))
    for folder, task in found.items():
        document = task["task"] if isinstance(task["task"], dict) else {}
        print(f"{folder}: image {document.get('image', '?')}, limits {json.dumps(document.get('limits'))}, "
              f"container {json.dumps(document.get('container', {}))}, {len(task['files'])} files, "
              f"{len(task['kept'])} links kept as they are, check {'present' if task['check'] else 'missing'}")
    _versioned(version)
    return 0


def cmd_task(args, repo):
    version, found = repo.tasks(args.id, _store(repo))
    if args.folder not in found:
        raise InvalidFile(f"{args.id} has no task folder {args.folder}")
    print(json.dumps(found[args.folder], indent=2, ensure_ascii=False))
    _versioned(version)
    return 0


def cmd_save_task(args, repo):
    if args.remove == bool(args.file):
        raise InvalidFile("give FILE, or --remove")
    given = {"task": None, "check": "", "files": {}} if args.remove else _json(_input(args.file), args.file)
    if not isinstance(given, dict) or not {"task", "check", "files"} <= set(given):
        raise InvalidFile(f"{args.file}: a JSON object {{task, check, files}}, as `wpt-dataset task` prints it")
    version = repo.save_task(args.id, args.folder, given["task"], given["check"], given["files"], args.patch,
                             args.version, _store(repo))
    print(f"recorded as the next patch of {args.id}, version {version}")
    return 0


def cmd_ratings(args, repo):
    version, found = repo.ratings(args.id, args.path, _store(repo))
    for line, rating in enumerate(found, start=1):  # the verdict is the person's: not shown here
        print(f"line {line}: {rating['task']}, attempt {rating['attempt']}{' (plain)' if rating['plain'] else ''} "
              f"of record {rating['record']}; by {rating['by']}, {rating['date']}")
    _versioned(version)
    return 0


def cmd_save_rating(args, repo):
    line = None if args.line == "new" else int(args.line)
    if args.remove == bool(args.file):
        raise InvalidFile("give FILE, or --remove")
    rating = None if args.remove else _json(_input(args.file), args.file)
    version = repo.save_rating(args.id, args.path, line, rating, args.patch, args.version, _store(repo))
    print(f"recorded as the next patch of {args.id}, version {version}")
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
    fill = commands.add_parser("fill-checksum")
    fill.add_argument("ids", nargs="+", metavar="ID")
    fill.set_defaults(run=cmd_fill_checksum)
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
    _data_commands(commands)
    args = parser.parse_args(argv)
    args.arguments = sys.argv[1:] if argv is None else list(argv)  # as the run's record names them
    if args.command == "source-id":
        if len(args.parts) > 2:
            parser.error("source-id takes ORIGIN [ITEM] ADDRESS")
        args.item, args.address = (args.parts[0], args.parts[1]) if len(args.parts) == 2 else (None, args.parts[0])
    if args.command == "new-source":
        if len(args.parts) > 2:
            parser.error("new-source takes ORIGIN [ITEM] NAME")
        args.item, args.name = (args.parts[0], args.parts[1]) if len(args.parts) == 2 else (None, args.parts[0])
    try:
        repo = Repo(args.root) if args.root else Repo.find()
        return args.run(args, repo)
    except (DatasetError, StoreError, MissingTool) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except RecordError as error:
        print(f"RecordError: {error}; stopped, nothing more was done", file=sys.stderr)
        return RECORD_FAILED


def _data_commands(commands):
    def command(name, run, *arguments, text=False):
        parser = commands.add_parser(name, help="reads or writes a person's text" if text else None)
        for argument in arguments:
            parser.add_argument(argument)
        parser.set_defaults(run=run)
        return parser

    def writes(parser, patch=True):
        if patch:
            parser.add_argument("--patch", required=True, help="the name of the patch that records the change")
        parser.add_argument("--version", required=True, help="the version read ('' for a new file)")
        return parser

    command("kinds", cmd_kinds)
    command("list", cmd_list, "kind")
    command("show", cmd_show, "kind", "name")
    writes(command("save", cmd_save, "kind", "name", "file"), patch=False)
    command("status", cmd_status)
    new = command("new-source", cmd_new_source, "origin")
    new.add_argument("parts", nargs="+", metavar="[ITEM] NAME")
    new.add_argument("--license", required=True)
    new.add_argument("--attribution", required=True)
    new.add_argument("--license-text", metavar="FILE",
                     help="* the license text (a person's text); without it a person writes it later")
    new.add_argument("--by", required=True, help="who started it (the decision's `by`, until a maintainer decides)")
    new.add_argument("--tag", action="append", default=[])
    new.add_argument("--file", action="append", default=[], metavar="PATH=FILE",
                     help="* a file of the source (a person's text)")
    new.add_argument("--format", action="append", default=[], metavar="FORMAT=PATTERN",
                     help="a format entry (the format and the path pattern it covers), without any file, such as "
                          "markdown=**/*.md or text=notes/*.txt")
    new.add_argument("--chat", action="append", default=[], metavar="PATH", help="a chat file to write later")
    new.add_argument("--patch", default="texts")
    command("files", cmd_files, "id")
    command("read", cmd_read, "id", "path", text=True)
    writes(command("write", cmd_write, "id", "path", "file", text=True))
    writes(command("remove", cmd_remove, "id", "path"))
    tags = writes(command("tags", cmd_tags, "id", "path"), patch=False)
    tags.add_argument("tags", nargs="*")
    command("conversations", cmd_conversations, "id", "path")
    conversation = command("conversation", cmd_conversation, "id", "path", text=True)
    conversation.add_argument("line", type=int)
    save = writes(command("save-conversation", cmd_save_conversation, "id", "path", "line", text=True))
    save.add_argument("file", nargs="?")
    save.add_argument("--remove", action="store_true")
    command("tasks", cmd_tasks, "id")
    command("task", cmd_task, "id", "folder", text=True)
    save = writes(command("save-task", cmd_save_task, "id", "folder", text=True))
    save.add_argument("file", nargs="?")
    save.add_argument("--remove", action="store_true")
    command("ratings", cmd_ratings, "id", "path")
    save = writes(command("save-rating", cmd_save_rating, "id", "path", "line", text=True))
    save.add_argument("file", nargs="?")
    save.add_argument("--remove", action="store_true")


if __name__ == "__main__":
    sys.exit(main())
