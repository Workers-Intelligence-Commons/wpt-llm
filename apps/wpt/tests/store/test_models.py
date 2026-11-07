"""A run's model folder in the store, written back to a disk folder, each
file checked against its sha256."""

import pytest

from wpt_store import CorruptFile, MissingFile, open_store
from wpt_store.models import copy_out, keep_model, put_model


def model_folder(root):
    (root / "drafter").mkdir(parents=True)
    files = {"config.json": b"{}\n", "model.safetensors": b"weights" * 1000, "drafter/model.safetensors": b"d" * 10}
    for path, content in files.items():
        (root / path).write_bytes(content)
    return files


def test_a_model_kept_under_its_run_s_name_is_copied_out_whole_and_a_damaged_file_is_refused(tmp_path):
    store = open_store((tmp_path / "store").as_uri()).at("s1")
    files = model_folder(tmp_path / "model")
    keep_model(store, "fox-1", put_model(store, "fox-1", tmp_path / "model"))

    copy_out(store, "fox-1", tmp_path / "copy")
    assert {path.relative_to(tmp_path / "copy").as_posix(): path.read_bytes()
            for path in (tmp_path / "copy").rglob("*") if path.is_file()} == files
    with pytest.raises(FileExistsError):  # never into a folder that exists
        copy_out(store, "fox-1", tmp_path / "copy")

    [stored] = (tmp_path / "store" / "s1" / "models" / "fox-1").glob("model.safetensors*")  # as it is, or compressed
    stored.write_bytes(b"tampered\n")
    with pytest.raises(CorruptFile):
        copy_out(store, "fox-1", tmp_path / "damaged")
    assert not (tmp_path / "damaged").exists() and not list(tmp_path.glob(".damaged-*"))
    with pytest.raises(MissingFile):
        copy_out(store, "fox-2", tmp_path / "other")
