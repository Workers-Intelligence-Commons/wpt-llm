import shutil

import boto3
import pytest
from moto import mock_aws

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


def test_s3_store_round_trip(tmp_path, monkeypatch):
    # moto answers boto3's S3 requests inside this process: no server, no network.
    # The store's local cache follows XDG_CACHE_HOME, or LOCALAPPDATA on Windows.
    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/some/prefix")
        sha = exercise(store)

        shutil.rmtree(tmp_path / "cache")  # a fresh machine: the object comes from the bucket again
        assert store.get_bytes(sha) == b"some text\n"
