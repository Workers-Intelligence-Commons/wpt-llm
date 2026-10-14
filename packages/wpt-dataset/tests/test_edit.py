import pytest

from wpt_dataset.build import build_dataset
from wpt_dataset.cli import main
from wpt_dataset.edit import unpack, write_patch
from wpt_dataset.errors import EditError, LicenseMapError
from wpt_dataset.sources import source_tree

EMPTY = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def contents(store, tree):
    return {path: store.get_bytes(entry.sha256) for path, entry in tree.items()}


def own_texts(kit, repo, store):
    return kit.add_source(repo, store, "author", "texts", {
        "type": "empty", "address": "empty:author-texts", "revision": "2026-10-14", "checksum": EMPTY,
        "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Test Author"}],
        "tags": [{"tags": ["fr"], "paths": ["**"]}]}, compute_checksum=False)


def test_own_texts_grow_by_patches_made_from_plain_files(kit, repo, store, tmp_path, monkeypatch):
    # a source written for WPT starts empty; each edit of its files becomes the next
    # patch, and the source then gives exactly the edited files
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    sid = own_texts(kit, repo, store)
    root = ["--root", str(repo.root)]

    first = tmp_path / "edit-1"
    assert main([*root, "unpack", sid, str(first)]) == 0 and list(first.iterdir()) == []
    (first / "notes").mkdir()
    (first / "notes" / "travail.txt").write_bytes(("le travail est un bien commun, " * 20 + "\n").encode())
    (first / "poeme.txt").write_bytes(("la nuit tombe sur la ville et les machines se taisent\n" * 15).encode())
    assert main([*root, "patch", sid, str(first), "first-texts"]) == 0

    second = tmp_path / "edit-2"
    assert main([*root, "unpack", sid, str(second)]) == 0
    (second / "poeme.txt").unlink()
    (second / "notes" / "travail.txt").write_bytes(("le travail est notre bien commun, " * 20 + "\n").encode())
    (second / "lettre.txt").write_bytes(("chere camarade, nous avons fini la premiere machine\n" * 15).encode())
    assert main([*root, "patch", sid, str(second), "rework"]) == 0

    source = repo.source(sid)
    assert [p.name for p in source.patches] == ["0001-first-texts.patch", "0002-rework.patch"]
    tree, _ = source_tree(source, store)
    assert contents(store, tree) == {"lettre.txt": (second / "lettre.txt").read_bytes(),
                                     "notes/travail.txt": (second / "notes" / "travail.txt").read_bytes()}
    assert repo.check() == []

    kit.write_recipe(repo, "mixture", "m", {"pools": [{"name": "fr", "select": "fr", "weight": 1}]})
    kit.write_recipe(repo, "tokenizer", "t", {"mixture": "m", "vocab_size": 300})
    kit.write_recipe(repo, "dataset", "d", {"mixture": "m", "tokenizer": "t", "total_tokens": 200, "held_out": 0})
    dataset_id = build_dataset(repo, "d", store, log=lambda _: None)
    assert kit.json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))["documents"]["total"] == 2


def test_a_removal_patch_on_an_upstream_source(kit, repo, store, tmp_path):
    upstream = kit.git_project(tmp_path / "upstream", symlinks={})
    sid = kit.add_source(repo, store, "example", "project", {"type": "git", "address": upstream[0],
                                                            "revision": upstream[1]})
    folder = tmp_path / "edit"
    unpack(repo, repo.source(sid), store, folder)
    (folder / "docs" / "guide.txt").unlink()
    write_patch(repo, repo.source(sid), store, folder, "remove-guide")
    tree, _ = source_tree(repo.source(sid), store)
    assert set(tree) == {"README.md", "run.sh"} and tree["run.sh"].mode == "100755"


def test_patch_refuses_what_it_cannot_record_and_leaves_no_patch(kit, repo, store, tmp_path):
    sid = own_texts(kit, repo, store)
    folder = tmp_path / "edit"
    unpack(repo, repo.source(sid), store, folder)
    with pytest.raises(EditError, match="nothing to record"):
        write_patch(repo, repo.source(sid), store, folder, "nothing")
    with pytest.raises(EditError, match="inside the repository"):
        unpack(repo, repo.source(sid), store, repo.root / "unpacked")

    # a file no license entry covers: the license map fails and the patch is removed again
    source_json = repo.data / "sources" / sid / "source.json"
    document = kit.json.loads(source_json.read_text())
    document["licenses"][0]["paths"] = ["texts/**"]
    source_json.write_text(kit.dump_json(document))
    (folder / "elsewhere.txt").write_bytes(b"a text outside texts/\n")
    with pytest.raises(LicenseMapError, match="elsewhere.txt matches no license entry"):
        write_patch(repo, repo.source(sid), store, folder, "uncovered")
    assert repo.source(sid).patches == []
