"""A run's model folder in the store, and `wpt-store copy-out`, which
writes it back to a disk folder, each file checked against its sha256."""

import hashlib

from wpt_store import open_store
from wpt_store.cli import main
from wpt_store.models import put_model


def model_folder(root):
    (root / "drafter").mkdir(parents=True)
    files = {"config.json": b"{}\n", "model.safetensors": b"weights" * 1000, "drafter/model.safetensors": b"d" * 10}
    for path, content in files.items():
        (root / path).write_bytes(content)
    return files


def test_a_model_kept_under_its_run_s_name_is_copied_out_whole_and_a_damaged_file_is_refused(tmp_path, monkeypatch,
                                                                                            capsys):
    store = open_store((tmp_path / "store").as_uri())
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    files = model_folder(tmp_path / "model")
    store.set_ref("models", "fox-1", put_model(store, tmp_path / "model"))

    assert main(["copy-out", "fox-1", str(tmp_path / "copy")]) == 0
    assert {path.relative_to(tmp_path / "copy").as_posix(): path.read_bytes()
            for path in (tmp_path / "copy").rglob("*") if path.is_file()} == files
    assert main(["copy-out", "fox-1", str(tmp_path / "copy")]) == 2  # never into a folder that exists
    assert "exists" in capsys.readouterr().err

    sha = hashlib.sha256(files["model.safetensors"]).hexdigest()
    [stored] = (tmp_path / "store" / "objects" / sha[:2]).glob(f"{sha}*")  # as it is, or compressed
    stored.write_bytes(b"tampered\n")
    assert main(["copy-out", "fox-1", str(tmp_path / "damaged")]) == 2
    assert "CorruptObject" in capsys.readouterr().err
    assert not (tmp_path / "damaged").exists() and not list(tmp_path.glob(".damaged-*"))
    assert main(["copy-out", "fox-2", str(tmp_path / "other")]) == 2
    assert "no model fox-2" in capsys.readouterr().err
