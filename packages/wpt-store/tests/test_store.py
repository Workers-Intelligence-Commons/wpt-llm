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


def test_a_token_file_holds_the_shards_tokens_and_is_rebuilt_when_changed(tmp_path):
    # training reads this file instead of the shards, so it must hold exactly
    # their tokens, and a copy changed on disk must never be used
    import numpy as np

    from wpt_store.shards import token_file, write_shard

    store = open_store((tmp_path / "store").as_uri())
    rows = [[([1, 2, 3], "s", "s/a", "p", 0), ([4, 5], "s", "s/b", "p", 0)], [([6, 7, 8, 9], "s", "s/c", "p", 0)]]
    shards = []
    for i, part in enumerate(rows):
        token_sha256 = write_shard(tmp_path / f"{i}.parquet", part)
        shards.append({"object": store.put_file(tmp_path / f"{i}.parquet"), "token_sha256": token_sha256})

    path = token_file(store, shards)
    assert np.fromfile(path, dtype="<u4").tolist() == list(range(1, 10))
    path.write_bytes(np.arange(9, dtype="<u4").tobytes())
    assert np.fromfile(token_file(store, shards), dtype="<u4").tolist() == list(range(1, 10))


def test_a_shard_stream_keeps_only_the_shards_in_use_on_disk(tmp_path, monkeypatch):
    # a machine whose disk is smaller than the dataset trains from a remote
    # store: each step reads the shards in use at its place in the dataset,
    # and only those (and the next, prepared ahead) stay on its disk
    import numpy as np

    from wpt_store.shards import ShardStream, write_shard

    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/stream")
        shards = []
        for i in range(4):
            path = tmp_path / f"{i}.parquet"
            token_sha256 = write_shard(path, [(list(range(10 * i, 10 * i + 10)), "s", f"s/{i}", "p", 0)])
            shards.append({"object": store.put_file(path), "token_sha256": token_sha256, "tokens": 10})
        stream = ShardStream(store, shards, window=2)

        def on_disk():
            return sorted((store.scratch / "tokens").glob("*.u32"))

        assert len(stream) == 40
        assert stream.at(0)[0:20].tolist() == list(range(0, 20))
        assert stream.at(25)[0:20].tolist() == list(range(20, 40))
        # after the last shard comes the first again; a window may span two shards
        assert stream.at(35)[5:15].tolist() == [35, 36, 37, 38, 39, 0, 1, 2, 3, 4]
        assert len(on_disk()) <= 3
        assert not any((store.cache / s["object"][:2] / s["object"]).exists() for s in shards)  # copies released
        stream.close()
        assert on_disk() == []
