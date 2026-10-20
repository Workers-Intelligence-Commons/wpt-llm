"""`wpt-dataset fill-checksum`: a draft source's checksum and the exclusion
of its files that are not UTF-8 text, from one fetch, with the same result
as `manifest`, then each such file excluded and `check --fetch` again."""

import json
import re

from wpt_dataset import sources
from wpt_dataset.cli import main
from wpt_dataset.repo import Repo
from wpt_dataset.sources import exact_pattern
from wpt_store import open_store

BINARY = b"\x89PNG\r\n\x1a\n\x00\xff\xfe"


def plumbing_commit(kit, folder, message, changes, parents=()):
    """A commit made from git objects only, so paths such as `odd*name?.dat`
    work on Windows too: the first parent's tree with changes applied."""
    env = {"GIT_INDEX_FILE": str(folder / ".git" / "plumbing-index")}
    kit.run("git", "read-tree", *(parents[:1] or ["--empty"]), cwd=folder, env=env)
    for path, content in changes.items():
        blob = kit.run("git", "hash-object", "-w", "--stdin", cwd=folder, input=content).decode().strip()
        kit.run("git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", cwd=folder, env=env)
    tree = kit.run("git", "write-tree", cwd=folder, env=env).decode().strip()
    return kit.run("git", "commit-tree", tree, *(a for p in parents for a in ("-p", p)), "-m", message,
                   cwd=folder).decode().strip()


def upstream_with_binaries(kit, folder):
    """(address, since, revision): a git project whose history changes text
    and binary files in the same commits, binary paths with glob letters."""
    folder.mkdir()
    kit.run("git", "init", "-q", cwd=folder)
    for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):  # partial fetches, as on GitHub
        kit.run("git", "config", setting, "true", cwd=folder)
    since = plumbing_commit(kit, folder, "start", {"README.md": b"a project\n", "src/main.py": b"print(1)\n",
                                                   "logo.png": BINARY, "data/img[1].bin": BINARY + b"1"})
    head = plumbing_commit(kit, folder, "more", {"src/main.py": b"print(2)\n", "odd*name?.dat": BINARY + b"2"},
                           [since])
    head = plumbing_commit(kit, folder, "redraw", {"logo.png": BINARY + b"3", "README.md": b"a project, drawn\n"},
                           [head])
    return str(folder), since, head


def draft(kit, repo, store, address, since, revision):
    return kit.add_source(repo, store, "example", "project", {
        "type": "git", "address": address, "revision": revision, "history": {"since": since},
        "licenses": kit.WITH_HISTORY,
        "formats": [{"format": "code", "language": "patch", "paths": [".git/history/**"]},
                    {"format": "text", "paths": ["*", "src/**", "data/**"]}]}, compute_checksum=False)


def test_fill_checksum_gives_what_manifest_then_each_exclusion_and_check_give_from_one_fetch(
        kit, store, tmp_path, monkeypatch, capsys):
    address, since, revision = upstream_with_binaries(kit, tmp_path / "upstream")
    results, real_fetch = {}, sources._git_fetch
    for way in ("one by one", "fill-checksum"):
        root = tmp_path / way
        for folder in ("sources", "recipes/mixtures", "recipes/tokenizers", "recipes/models"):
            (root / "data" / folder).mkdir(parents=True)
        repo, way_store = Repo(root), open_store((tmp_path / f"store {way}").as_uri())
        monkeypatch.setenv("WPT_STORE", way_store.root.as_uri())
        sid = draft(kit, repo, way_store, address, since, revision)
        fetches = []
        monkeypatch.setattr(sources, "_git_fetch", lambda *args, **kwargs: fetches.append(1) or real_fetch(*args, **kwargs))
        if way == "one by one":
            code = by_hand(repo, sid, capsys)
        else:
            code = main(["--root", str(root), "fill-checksum", sid])
            assert f"{sid}: checksum " in capsys.readouterr().out
            assert len(fetches) == 1  # the check found the upstream tree in the store
        checksum = repo.source(sid).data["checksum"]
        results[way] = (code, repo.read("source", sid).text, checksum,
                        way_store.get_bytes(way_store.get_ref("upstream", checksum.removeprefix("sha256:"))))

    assert results["fill-checksum"] == results["one by one"]
    code, text, _, listing = results["fill-checksum"]
    assert code == 0
    assert json.loads(text)["extract"]["exclude"] == ["data/img[[]1].bin", "logo.png", "odd[*]name[?].dat"]
    paths = [line.split(" ", 2)[2] for line in listing.decode().splitlines()]
    assert [p for p in paths if not p.startswith(".git/history/")] == ["README.md", "src/main.py"]


def by_hand(repo, sid, capsys) -> int:
    """`manifest`, its checksum written in; then `check --fetch`, and for each
    file it names as not UTF-8 text, that path excluded and both again."""
    root = ["--root", str(repo.root)]
    while True:
        assert main([*root, "manifest", sid]) == 0
        checksum = re.search(r"checksum: (\S+)", capsys.readouterr().err)[1]
        document = repo.read("source", sid)
        repo.save_source(sid, {**document.document, "checksum": checksum}, document.version)
        code = main([*root, "check", "--fetch", sid])
        named = re.search(rf"NotText: {sid}: (.+) is not UTF-8 text; exclude it in extract", capsys.readouterr().err)
        if not named:
            return code
        document = repo.read("source", sid)
        data = document.document
        data.setdefault("extract", {}).setdefault("exclude", []).append(exact_pattern(named[1]))
        repo.save_source(sid, data, document.version)


def test_fill_checksum_fills_every_draft_and_lists_the_failures_at_the_end(kit, repo, store, tmp_path, monkeypatch,
                                                                         capsys):
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    gone = draft(kit, repo, store, str(tmp_path / "gone"), "0" * 40, "0" * 40)
    address, since, revision = upstream_with_binaries(kit, tmp_path / "upstream")
    good = draft(kit, repo, store, address, since, revision)

    code = main(["--root", str(repo.root), "fill-checksum", gone, good])

    out, err = capsys.readouterr()
    assert code == 1
    assert f"{good}: checksum sha256:" in out and "3 files that are not UTF-8 text excluded" in out
    assert re.search(rf"{good}: \d+ entries, checksum, cuts, patches, license and format maps OK", out)
    assert err.split("1 of 2 fetched sources failed:\n")[1].startswith(f"{gone}: FetchFailed: {gone}: ")
    assert repo.source(gone).data["checksum"] == "sha256:" + "0" * 64  # nothing written for it
    assert main(["--root", str(repo.root), "fill-checksum", good]) == 1  # filled already: refused
