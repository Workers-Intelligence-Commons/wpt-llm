import re

import pytest

from wpt_dataset import manifest
from wpt_dataset.build import gather
from wpt_dataset.edit import unpack, write_patch
from wpt_dataset.errors import EditError, PatchFailed
from wpt_dataset.sources import compute_upstream, source_tree


def commit(kit, folder, message, files):
    for path, content in files.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_bytes(content)
    kit.run("git", "add", "-A", cwd=folder)
    kit.run("git", "commit", "-q", "-m", message, cwd=folder)
    return kit.run("git", "rev-parse", "HEAD", cwd=folder).decode().strip()


def test_a_git_history_gives_one_patch_per_commit_since_the_license_limited_to_kept_files(kit, repo, store, tmp_path):
    # a repository's history teaches how code changes; only the commits made
    # under the license WPT lists may enter, and a diff never shows a file the
    # source leaves out
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):  # partial fetches, as on GitHub
        kit.run("git", "config", setting, "true", cwd=folder)
    before = commit(kit, folder, "start without a license", {"main.py": b"print(1)\n"})
    since = commit(kit, folder, "add the license", {"LICENSE": kit.LICENSE_TEXT.encode()})
    change = commit(kit, folder, "print two", {"main.py": b"print(2)\n", "vendor/lib.py": b"x = 1\n"})
    vendored = commit(kit, folder, "update the vendored library", {"vendor/lib.py": b"x = 2\n"})
    sid = kit.add_source(repo, store, "example", "history", {
        "type": "git", "address": str(folder), "revision": vendored, "history": {"since": since},
        "extract": {"exclude": ["vendor/**"]}, "licenses": kit.WITH_HISTORY,
        "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                    {"format": "code", "language": "python", "paths": ["**/*.py"]},
                    {"format": "text", "paths": ["LICENSE"]}]})

    documents, _, _ = gather(repo, store, log=lambda _: None)

    ids = sorted(d.id for d in documents)
    # the order file stays in the stored tree but is not a document
    assert manifest.ORDER in source_tree(repo.source(sid), store)[0]
    assert ids == sorted([f"{sid}/.git/history/{since}", f"{sid}/.git/history/{change}", f"{sid}/LICENSE",
                          f"{sid}/main.py"])
    assert before not in " ".join(ids) and vendored not in " ".join(ids)
    patch = store.get_bytes(next(d.sha256 for d in documents if d.id.endswith(change))).decode()
    assert f"From {change} " in patch and "From: T <t@example.org>" in patch and "Subject: [PATCH] print two" in patch
    assert "-print(1)\n+print(2)" in patch and "vendor/lib.py" not in patch


def test_a_tree_entry_written_everywhere_leaves_the_history_files_to_their_own_entry(kit, repo, store, tmp_path):
    # a history file counts only against entries whose patterns start with
    # .git/history/, so `**` means every file of the tree; each file then maps to exactly one entry
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "add the license", {"LICENSE": kit.LICENSE_TEXT.encode(), "main.py": b"print(1)\n"})
    head = commit(kit, folder, "print two", {"main.py": b"print(2)\n"})
    sid = kit.add_source(repo, store, "example", "history", {
        "type": "git", "address": str(folder), "revision": head, "history": {"since": since},
        "licenses": kit.WITH_HISTORY,
        "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                    {"format": "text", "paths": ["LICENSE", "main.py"]}]})

    tree, _ = source_tree(repo.source(sid), store)  # the license map check passes

    # the order file is part of the history, under the same entry
    history = [f".git/history/{c}" for c in (head, since)] + [manifest.ORDER]
    entry_for_history, entry_for_tree = kit.WITH_HISTORY
    assert sorted(tree) == sorted([*history, "LICENSE", "main.py"])
    for path in tree:
        expected = entry_for_history if path in history else entry_for_tree
        assert manifest.license_entries(kit.WITH_HISTORY, path) == [expected]


def test_a_patched_git_source_keeps_its_history_and_order_and_no_patch_touches_them(kit, repo, store, tmp_path):
    # git's index silently drops .git/ paths, so patches are applied with the
    # history files and the order file set aside; they come back exactly, and a patch naming one is refused
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "add the license", {"LICENSE": kit.LICENSE_TEXT.encode(), "main.py": b"print(1)\n"})
    head = commit(kit, folder, "print two", {"main.py": b"print(2)\n"})
    data = {"type": "git", "address": str(folder), "revision": head, "history": {"since": since},
            "licenses": kit.WITH_HISTORY, "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                                                      {"format": "text", "paths": ["LICENSE", "main.py"]}]}
    sid = kit.add_source(repo, store, "example", "history", data)
    before, _ = source_tree(repo.source(sid), store)
    history = {path: entry for path, entry in before.items() if path.startswith(manifest.HISTORY)}
    assert sorted(history) == sorted([f".git/history/{since}", f".git/history/{head}", manifest.ORDER])

    edited = tmp_path / "edited"
    unpack(repo, repo.source(sid), store, edited)
    (edited / "main.py").write_bytes(b"print(3)\n")
    write_patch(repo, repo.source(sid), store, edited, "print-three")

    after, _ = source_tree(repo.source(sid), store)
    assert {path: entry for path, entry in after.items() if path.startswith(manifest.HISTORY)} == history
    assert store.get_bytes(after["main.py"].sha256) == b"print(3)\n" and after["LICENSE"] == before["LICENSE"]

    # an edited folder whose history files differ cannot be recorded as a patch, and leaves no patch
    (edited / ".git" / "history" / "order").write_bytes(b"")
    with pytest.raises(EditError, match=re.escape(f"{manifest.ORDER}: a patch cannot change")):
        write_patch(repo, repo.source(sid), store, edited, "empty-order")
    assert len(repo.source(sid).patches) == 1

    # a patch that removes a history file is refused
    order = store.get_bytes(before[manifest.ORDER].sha256).decode()
    lines = order.splitlines()
    removal = (f"diff --git a/{manifest.ORDER} b/{manifest.ORDER}\ndeleted file mode 100644\n--- a/{manifest.ORDER}\n"
               f"+++ /dev/null\n@@ -1,{len(lines)} +0,0 @@\n" + "".join(f"-{line}\n" for line in lines))
    other = kit.add_source(repo, store, "example", "history-removed", {**data, "address": str(folder) + "/"},
                           patches=[removal])
    with pytest.raises(PatchFailed, match=re.escape(manifest.ORDER)):
        source_tree(repo.source(other), store)


def test_a_submodule_replacing_a_folder_keeps_the_folder_s_excluded_file_out_of_the_patch(kit, store, tmp_path):
    # a commit replaces a folder's files by a submodule of the same name, one of those files excluded: git reads
    # the kept path `frontend` as a folder too, so a patch limited to the kept paths would also show the excluded
    # file, whose content a partial fetch never downloads (the fetch then fails)
    library, folder = tmp_path / "library", tmp_path / "upstream"
    for each in (library, folder):
        each.mkdir()
        kit.run("git", "init", "-q", cwd=each)
        for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):  # partial fetches, as on GitHub
            kit.run("git", "config", setting, "true", cwd=each)
    library_commit = plumbing_commit(kit, library, "the library", {"lib.js": b"export {}\n"})
    since = plumbing_commit(kit, folder, "start", {"main.go": b"package main\n", "frontend/app.js": b"go()\n",
                                                   "frontend/favicon.ico": b"\x00\xff icon"})
    head = plumbing_commit(kit, folder, "move the frontend to a submodule",
                           {"frontend/app.js": None, "frontend/favicon.ico": None, "main.go": b"package main\n\n"},
                           [since], gitlinks={"frontend": library_commit})
    for each, tip in ((library, library_commit), (folder, head)):
        kit.run("git", "update-ref", "refs/heads/main", tip, cwd=each)

    tree = compute_upstream({"type": "git", "revision": head, "history": {"since": since},
                             "extract": {"exclude": ["frontend/favicon.ico"]}, "submodules": {"frontend": str(library)}},
                            str(folder), store)

    patch = store.get_bytes(tree[f".git/history/{head}"].sha256).decode()
    assert re.findall(r"^diff --git a/(\S+) ", patch, re.M) == ["frontend", "frontend/app.js", "main.go"]
    assert f"+Subproject commit {library_commit}" in patch and "favicon" not in patch
    assert "frontend/lib.js" in tree


def plumbing_commit(kit, folder, message, changes, parents=(), gitlinks=()):
    """A commit made from git objects only, no file on a disk (so any path
    git accepts, Windows or not): the first parent's tree with changes
    applied, path -> content, or None to remove the path, and gitlinks,
    path -> the commit a submodule there points to."""
    env = {"GIT_INDEX_FILE": str(folder / ".git" / "plumbing-index")}
    kit.run("git", "read-tree", *(parents[:1] or ["--empty"]), cwd=folder, env=env)
    for path, content in changes.items():
        if content is None:
            kit.run("git", "update-index", "--force-remove", "--", path, cwd=folder, env=env)
        else:
            blob = kit.run("git", "hash-object", "-w", "--stdin", cwd=folder, input=content).decode().strip()
            kit.run("git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", cwd=folder, env=env)
    for path, commit in dict(gitlinks).items():
        kit.run("git", "update-index", "--add", "--cacheinfo", f"160000,{commit},{path}", cwd=folder, env=env)
    tree = kit.run("git", "write-tree", cwd=folder, env=env).decode().strip()
    return kit.run("git", "commit-tree", tree, *(a for p in parents for a in ("-p", p)), "-m", message,
                   cwd=folder).decode().strip()


def history_one_call_per_commit(repo, since, revision, extractor):
    """The history files as read_git_history wrote them with one
    `git format-patch -1` per commit, its kept paths on the command line."""
    files = {}
    commits = manifest._history_commits(repo, since, revision)
    for commit, changes in manifest._changed_paths(repo, commits).items():
        changed = [path for path, _ in changes]
        paths = [p for p in changed if extractor.kept(p) is not None]
        if not paths:
            continue
        limit = [] if len(paths) == len(changed) else ["--", *manifest._only(paths, changed)]
        text = manifest._git(repo, "format-patch", "-1", "--root", "--stdout", "--no-signature", "--no-renames",
                             "--no-binary", commit, *limit)
        files[manifest.HISTORY + commit] = text.decode("utf-8", errors="replace").encode("utf-8")
    return files


@pytest.mark.parametrize("group", [500, 7])
@pytest.mark.parametrize("quoted", [False, True])
def test_history_files_are_the_bytes_of_one_format_patch_per_commit(kit, store, tmp_path, monkeypatch, quoted, group):
    # one git process per group of commits writes every commit of the group whose paths are all kept, and is cut
    # into one file per commit; the files must stay byte for byte what one call per commit wrote, whatever the
    # commits hold. quoted: the newest message holds an older commit's first line, so a group holding both
    # cannot be cut safely and each of its commits gets its own call. group 7: many groups, and the two
    # commits fall into different ones
    monkeypatch.setattr(manifest, "PATCH_GROUP", group)
    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    binary = b"\x89PNG\r\n\x1a\n\x00\xff\xfe"
    root = plumbing_commit(kit, folder, "the root commit", {"src/a.py": b"x = 0\n", "logo.png": binary,
                                                            "vendor/lib.py": b"v = 0\n", "notes.txt": b"caf\xe9\n"})
    head, side = root, None
    for number in range(1, 41):
        changes = {"src/a.py": f"x = {number}\n".encode()}
        if number % 5 == 0:
            changes["vendor/lib.py"] = f"v = {number}\n".encode()  # limited to the kept paths
        if number % 7 == 0:
            changes = {"vendor/lib.py": f"v = {number}.5\n".encode()}  # no kept path: no file
        if number % 3 == 0:
            changes["logo.png"] = binary + bytes([number])  # "Binary files differ"
        if number == 10:
            changes["notes.txt"] = b"na\xefve, not UTF-8\n"
        head = plumbing_commit(kit, folder, f"change {number}\n\nFrom a line that looks like a header\n", changes,
                               [head])
        if number == 1:
            first = head
        if number == 20:
            side = plumbing_commit(kit, folder, "on a branch", {"side.txt": b"side\n"}, [head])
    head = merge = plumbing_commit(kit, folder, "merge the branch", {"side.txt": b"side\n"}, [head, side])  # left out
    head = plumbing_commit(kit, folder, "rename", {"src/a.py": None, "src/renamed.py": b"x = 40\n"}, [head])
    many = {f"long/{'folder-' * 12}{i:04d}.txt": b"line\n" for i in range(300)}  # 300 paths > 32,767 characters
    head = plumbing_commit(kit, folder, "many files", {**many, "vendor/many.py": b"m\n"}, [head])
    if quoted:
        head = plumbing_commit(kit, folder, f"quote\n\nFrom {first} Mon Sep 17 00:00:00 2001\n",
                               {"src/renamed.py": b"x = 41\n"}, [head])
    extract = {"exclude": ["vendor/**"]}
    extractor = manifest.Extractor(store, extract)

    manifest.read_git_history(folder, root, head, extractor)

    written = {path: store.get_bytes(entry.sha256) for path, entry in extractor.tree.items()}
    order = written.pop(manifest.ORDER).decode().splitlines()
    expected = history_one_call_per_commit(folder, root, head, manifest.Extractor(store, extract))
    assert written == expected
    # the order file lists every commit since root, the merge and the commits
    # that changed no kept path included, newest first, each before its parents, and marks the merge
    listed = [line.split(" ")[0] for line in order]
    every = kit.run("git", "rev-list", head, cwd=folder).decode().split()
    assert sorted(listed) == sorted(every) and listed[0] == head
    assert [line for line in order if " " in line] == [f"{merge} merge"]
    for commit in listed:
        parents = kit.run("git", "rev-list", "--parents", "-n", "1", commit, cwd=folder).decode().split()[1:]
        assert all(listed.index(parent) > listed.index(commit) for parent in parents)
    assert len(written) == 40 + quoted  # 36 of the 40 changes (4 change vendor/ only), root, side, rename, many
    commits = manifest._history_commits(folder, root, head)
    whole = [c for c, changes in manifest._changed_paths(folder, commits).items()
             if not any(path.startswith("vendor/") for path, _ in changes)]
    cut = sum(len(manifest._group_patches(folder, whole[i:i + group])) for i in range(0, len(whole), group))
    assert cut == (0 if quoted and group == 500 else len(whole))  # every group cut, or the one group not


def test_a_repository_pointer_takes_the_tags_of_the_files_its_commit_changes(kit, repo, store, tmp_path):
    # a task pointing to a commit is a code
    # task when a file the commit changes is tagged code; a file pointer takes
    # its own file's tags
    from wpt_dataset.tasks import document_tags

    folder = tmp_path / "upstream"
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    since = commit(kit, folder, "add the license", {"LICENSE": kit.LICENSE_TEXT.encode()})
    code = commit(kit, folder, "print two", {"src/main.py": b"print(2)\n"})
    notes = commit(kit, folder, "write the notes", {"docs/notes.txt": b"Two.\n"})
    sid = kit.add_source(repo, store, "example", "history", {
        "type": "git", "address": str(folder), "revision": notes, "history": {"since": since},
        "licenses": kit.WITH_HISTORY,
        "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                    {"format": "code", "language": "python", "paths": ["**/*.py"]},
                    {"format": "text", "paths": ["LICENSE", "docs/**"]}],
        "tags": [{"tags": ["code"], "paths": ["src/**"]}, {"tags": ["en"], "paths": ["docs/**"]}]})

    assert document_tags(repo, store, sid, f".git/history/{code}") == {"code"}
    assert document_tags(repo, store, sid, f".git/history/{notes}") == {"en"}
    assert document_tags(repo, store, sid, "docs/notes.txt") == {"en"}
    assert document_tags(repo, store, sid, "src/main.py#3") == {"code"}
