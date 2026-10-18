import shutil

import boto3
import pytest
from moto import mock_aws

from wpt_store import CorruptObject, MissingObject, MixedRecord, open_store


def exercise(store):
    sha = store.put_bytes(b"some text\n")
    assert store.get_bytes(sha) == b"some text\n"
    assert store.put_bytes(b"some text\n") == sha  # same bytes, same object
    assert store.get_ref("builds", "abc") is None
    store.set_ref("builds", "abc", sha)
    assert store.get_ref("builds", "abc") == sha
    with pytest.raises(MissingObject):
        store.get_bytes("0" * 64)
    # a checkpoint replaced by a newer one is removed, or a long run fills the store
    old = store.put_bytes(b"checkpoint at step 500\n")
    store.set_ref("checkpoints", "run", old)
    store.get_bytes(old)  # leaves a local copy for S3
    store.delete(old)
    store.delete_ref("checkpoints", "run")
    assert not store.has(old) and store.get_ref("checkpoints", "run") is None
    store.delete(old)  # absent: nothing to do
    store.delete_ref("checkpoints", "run")
    assert store.get_bytes(sha) == b"some text\n"
    return sha


def exercise_records(store, reopen):
    # a command's record: files written whole, and a file added to line by
    # line, by one process and then by another (a restarted training process)
    assert store.record_folders() == [] and store.record_files("run-1") == []
    store.write_record("run-1", "start.json", b"{}\n")
    store.write_record("run-1", "start.json", b'{"command": "x"}\n')  # replaced whole
    for line in (b"one\n", b"two\n"):
        store.append_record("run-1", "events.jsonl", line)
    store.flush_records()
    later = reopen()
    later.append_record("run-1", "events.jsonl", b"three\n")
    later.flush_records()
    assert store.read_record("run-1", "start.json") == b'{"command": "x"}\n'
    assert store.read_record("run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    assert store.record_files("run-1") == ["events.jsonl", "start.json"]
    store.append_record("run-2", "events.jsonl", b"other\n")
    store.flush_records()
    assert store.record_folders() == ["run-1", "run-2"]
    with pytest.raises(MissingObject):
        store.read_record("run-1", "end.json")

    # a file is written whole or added to, never both, whichever process made it
    for writer in (store, reopen()):
        with pytest.raises(MixedRecord):
            writer.write_record("run-1", "events.jsonl", b"replaced\n")
        with pytest.raises(MixedRecord):
            writer.append_record("run-1", "start.json", b"added\n")
    assert store.read_record("run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    assert store.read_record("run-1", "start.json") == b'{"command": "x"}\n'

    # wpt-train's launcher and its training process add to one file in turn;
    # reading gives their lines in the order they were added, even when each
    # process sends its lines later (an S3 store sends them in groups)
    launcher, child = reopen(), reopen()
    for writer, line in ((launcher, b"launcher 1\n"), (child, b"child 1\n"), (launcher, b"launcher 2\n"),
                         (child, b"child 2\n"), (launcher, b"launcher 3\n")):
        writer.append_record("run-3", "events.jsonl", line)
    child.flush_records()
    launcher.flush_records()
    assert store.read_record("run-3", "events.jsonl") == b"launcher 1\nchild 1\nlauncher 2\nchild 2\nlauncher 3\n"


def test_local_store_round_trip_and_refuses_damaged_objects(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    sha = exercise(store)
    exercise_records(store, lambda: open_store((tmp_path / "store").as_uri()))

    # a power cut while a line was written leaves it cut; the next process to
    # add to the file ends it first, so its own lines stay whole and readable
    with open(tmp_path / "store" / "records" / "run-1" / "events.jsonl", "ab") as cut:
        cut.write(b'{"kind": "ev')
    open_store((tmp_path / "store").as_uri()).append_record("run-1", "events.jsonl", b"four\n")
    assert store.read_record("run-1", "events.jsonl").splitlines()[-2:] == [b'{"kind": "ev', b"four"]

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
        exercise_records(store, lambda: open_store("s3://wpt-test/some/prefix"))

        shutil.rmtree(tmp_path / "cache")  # a fresh machine: the object comes from the bucket again
        assert store.get_bytes(sha) == b"some text\n"

        # a flush interrupted once S3 had stored its group (Ctrl+C, or the
        # record's own interruption) sends it again later: each line once
        real = store.client.put_object

        def stored_then_interrupted(**request):
            real(**request)
            monkeypatch.setattr(store.client, "put_object", real)
            raise KeyboardInterrupt

        store.append_record("run-4", "events.jsonl", b"one\n")
        monkeypatch.setattr(store.client, "put_object", stored_then_interrupted)
        with pytest.raises(KeyboardInterrupt):
            store.flush_records()
        store.append_record("run-4", "events.jsonl", b"two\n")
        store.flush_records()
        assert open_store("s3://wpt-test/some/prefix").read_record("run-4", "events.jsonl") == b"one\ntwo\n"


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


def test_reading_documents_from_a_remote_store_keeps_at_most_one_shard_on_disk(tmp_path, monkeypatch):
    # the drafter reads the training shards once at the end of a run: each
    # shard's local copy goes once its rows are read, as a shard stream's do
    from wpt_store.shards import documents, write_shard

    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/documents")
        shards = []
        for i in range(3):
            path = tmp_path / f"{i}.parquet"
            rows = [([i, 0], "s", f"s/{i}", "p", 0), ([i, 0], "s", "s/again", "p", 1)]
            token_sha256 = write_shard(path, rows)
            shards.append({"object": store.put_file(path), "token_sha256": token_sha256, "tokens": 4})

        def copies():
            return [s["object"] for s in shards if (store.cache / s["object"][:2] / s["object"]).exists()]

        seen = []
        for document, source, tokens in documents(store, shards):
            assert copies() == []  # the shard being read is already released
            seen.append((document, source, tokens.tolist()))
        assert seen == [("s/0", "s", [0, 0]), ("s/again", "s", [0, 0]), ("s/1", "s", [1, 0]), ("s/2", "s", [2, 0])]
