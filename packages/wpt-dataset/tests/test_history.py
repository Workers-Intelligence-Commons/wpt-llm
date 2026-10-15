from wpt_dataset.build import gather


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
    before = commit(kit, folder, "start without a license", {"main.py": b"print(1)\n"})
    since = commit(kit, folder, "add the license", {"LICENSE": kit.LICENSE_TEXT.encode()})
    change = commit(kit, folder, "print two", {"main.py": b"print(2)\n", "vendor/lib.py": b"x = 1\n"})
    vendored = commit(kit, folder, "update the vendored library", {"vendor/lib.py": b"x = 2\n"})
    sid = kit.add_source(repo, store, "example", "history", {
        "type": "git", "address": str(folder), "revision": vendored, "history": {"since": since},
        "extract": {"exclude": ["vendor/**"]},
        "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                    {"format": "code", "language": "python", "paths": ["**/*.py"]},
                    {"format": "text", "paths": ["LICENSE"]}]})

    documents, _, _ = gather(repo, store, log=lambda _: None)

    ids = sorted(d.id for d in documents)
    assert ids == sorted([f"{sid}/.git/history/{since}", f"{sid}/.git/history/{change}", f"{sid}/LICENSE",
                          f"{sid}/main.py"])
    assert before not in " ".join(ids) and vendored not in " ".join(ids)
    patch = store.get_bytes(next(d.sha256 for d in documents if d.id.endswith(change))).decode()
    assert f"From {change} " in patch and "From: T <t@example.org>" in patch and "Subject: [PATCH] print two" in patch
    assert "-print(1)\n+print(2)" in patch and "vendor/lib.py" not in patch
