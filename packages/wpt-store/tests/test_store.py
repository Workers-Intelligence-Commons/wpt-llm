import os
import uuid

import pytest

from wpt_store import CorruptObject, MissingObject, open_store


def exercise(store):
    sha = store.put_bytes(b"some text\n")
    assert store.get_bytes(sha) == b"some text\n"
    assert store.put_bytes(b"some text\n") == sha  # same bytes, same object
    assert store.get_ref("builds", "abc") is None
    store.set_ref("builds", "abc", sha)
    assert store.get_ref("builds", "abc") == sha
    with pytest.raises(MissingObject):
        store.get_bytes("0" * 64)
    return sha


def test_local_store_round_trip_and_refuses_damaged_objects(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    sha = exercise(store)

    (tmp_path / "store" / "objects" / sha[:2] / sha).write_bytes(b"tampered\n")
    with pytest.raises(CorruptObject):
        store.get_bytes(sha)


@pytest.mark.skipif(not os.environ.get("WPT_TEST_S3_BUCKET"),
                    reason="set WPT_TEST_S3_BUCKET (plus AWS_* settings) to test against an S3-compatible server")
def test_s3_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    store = open_store(f"s3://{os.environ['WPT_TEST_S3_BUCKET']}/test-{uuid.uuid4().hex}")
    exercise(store)
