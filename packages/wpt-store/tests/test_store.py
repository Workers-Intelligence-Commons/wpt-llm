import json
import os
import shutil
import threading
import time
from compression import zstd

import boto3
import pytest
from moto import mock_aws

from wpt_store import (CorruptFile, MissingFile, MixedRecord, StoreSettingError, blocks, open_store,
                       s3_settings)


def exercise(store):
    sha = store.put_bytes(b"some text\n")
    assert store.get_bytes(sha) == b"some text\n"
    assert store.put_bytes(b"some text\n") == sha  # same bytes, same file
    with pytest.raises(MissingFile):
        store.get_bytes("0" * 64)
    # a result is found from its inputs by its fingerprint, and says what made it at no further read
    assert store.get_result("builds", "abc") == (None, None) and not store.has_result("builds", "abc")
    store.put_result("builds", "abc", b"what the stage gave\n")
    assert store.get_result("builds", "abc") == (b"what the stage gave\n", None) and store.has_result("builds", "abc")
    store.put_result("results", "abc", b"another stage\n", made_by="the-commit")
    assert store.get_result("results", "abc") == (b"another stage\n", "the-commit")
    # a file a record names is read back through what the record says, and a replaced one is the new one
    file = store.put_named("s1/datasets/d/shard-0", b"first bytes\n")
    assert store.get_bytes(file) == b"first bytes\n"
    again = store.put_named("s1/datasets/d/shard-0", b"other bytes\n" * 1000)  # compressed now: the plain one goes
    assert store.get_bytes(again) == b"other bytes\n" * 1000
    assert store.get_json("s1/datasets/d/dataset.json") is None
    store.put_json("s1/datasets/d/dataset.json", {"id": "d", "shards": [file.key]})
    assert store.get_json("s1/datasets/d/dataset.json")["shards"] == [file.key]
    # a checkpoint replaced by a newer one is removed, or a long run fills the store
    old = store.put_bytes(b"checkpoint at step 500\n")
    store.get_bytes(old)  # leaves a local copy for S3
    store.delete(old)
    assert not store.has(old)
    store.delete(old)  # absent: nothing to do
    assert store.get_bytes(sha) == b"some text\n"
    return sha


def exercise_records(store, reopen):
    # a command's record: files written whole, and a file added to line by
    # line, by one process and then by another (a restarted training process)
    assert store.record_folders() == [] and store.record_files("s1/run-1") == []
    store.write_record("s1/run-1", "start.json", b"{}\n")
    store.write_record("s1/run-1", "start.json", b'{"command": "x"}\n')  # replaced whole
    for line in (b"one\n", b"two\n"):
        store.append_record("s1/run-1", "events.jsonl", line)
    store.flush_records()
    later = reopen()
    later.append_record("s1/run-1", "events.jsonl", b"three\n")
    later.flush_records()
    assert store.read_record("s1/run-1", "start.json") == b'{"command": "x"}\n'
    assert store.read_record("s1/run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    assert store.record_files("s1/run-1") == ["events.jsonl", "start.json"]
    store.append_record("s1/run-2", "events.jsonl", b"other\n")
    store.flush_records()
    assert store.record_folders() == ["s1/run-1", "s1/run-2"]
    with pytest.raises(MissingFile):
        store.read_record("s1/run-1", "end.json")

    # a file is written whole or added to, never both, whichever process made it
    for writer in (store, reopen()):
        with pytest.raises(MixedRecord):
            writer.write_record("s1/run-1", "events.jsonl", b"replaced\n")
        with pytest.raises(MixedRecord):
            writer.append_record("s1/run-1", "start.json", b"added\n")
    assert store.read_record("s1/run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    assert store.read_record("s1/run-1", "start.json") == b'{"command": "x"}\n'

    # wpt train's launcher and its training process add to one file in turn;
    # reading gives their lines in the order they were added, even when each
    # process sends its lines later (an S3 store sends them in groups)
    launcher, child = reopen(), reopen()
    for writer, line in ((launcher, b"launcher 1\n"), (child, b"child 1\n"), (launcher, b"launcher 2\n"),
                         (child, b"child 2\n"), (launcher, b"launcher 3\n")):
        writer.append_record("s1/run-3", "events.jsonl", line)
    child.flush_records()
    launcher.flush_records()
    assert store.read_record("s1/run-3", "events.jsonl") == b"launcher 1\nchild 1\nlauncher 2\nchild 2\nlauncher 3\n"


def test_local_store_round_trip_and_refuses_damaged_objects(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    sha = exercise(store)
    exercise_records(store, lambda: open_store((tmp_path / "store").as_uri()))

    # a process that left its last line cut (it stopped before the line's
    # end): the next process's lines start on their own, whole and readable
    cut = open_store((tmp_path / "store").as_uri())
    cut.append_record("s1/run-1", "events.jsonl", b'{"kind": "ev')
    cut.flush_records()
    after = open_store((tmp_path / "store").as_uri())
    after.append_record("s1/run-1", "events.jsonl", b"four\n")
    after.flush_records()
    assert store.read_record("s1/run-1", "events.jsonl").splitlines()[-2:] == [b'{"kind": "ev', b"four"]

    (tmp_path / "store" / "cache" / "files" / sha).write_bytes(b"tampered\n")
    with pytest.raises(CorruptFile):
        store.get_bytes(sha)

    # a result and a record each check themselves, whatever a damaged disk gives
    packed = tmp_path / "store" / "cache" / "builds" / "abc"
    packed.write_bytes(packed.read_bytes().replace(b"what", b"that"))
    with pytest.raises(CorruptFile):
        store.get_result("builds", "abc")
    [record] = (tmp_path / "store" / "s1" / "datasets" / "d").glob("dataset.json*")
    plain = store.get_json("s1/datasets/d/dataset.json")
    record.unlink()
    (tmp_path / "store" / "s1" / "datasets" / "d" / "dataset.json").write_text(
        json.dumps({**plain, "id": "another"}), encoding="utf-8")
    with pytest.raises(CorruptFile):
        store.get_json("s1/datasets/d/dataset.json")


def test_a_local_record_file_is_added_to_at_once_and_compressed_into_one_part_when_the_record_ends(tmp_path):
    # a local store appends each line to one plain file while the record runs, so a crash loses none; the end
    # of the record compresses that file into one part; whole files are compressed and on the disk at once
    root = tmp_path / "store"
    store = open_store(root.as_uri())
    text = b"".join(b'{"kind": "event", "number": %d}\n' % i for i in range(1000))
    store.write_record("s1/run-1", "start.json", text)
    store.append_record("s1/run-1", "events.jsonl", text[:500])
    store.append_record("s1/run-1", "events.jsonl", text[500:])
    # in the file already, before any flush, for this store object and for another process
    assert (root / "s1" / "runs" / "run-1" / "events.jsonl.open").read_bytes() == text
    assert store.record_files("s1/run-1") == ["events.jsonl", "start.json"]
    assert open_store(root.as_uri()).read_record("s1/run-1", "events.jsonl") == text
    store.flush_records()
    assert set(store.kind.list("s1/runs/run-1/")) == {"s1/runs/run-1/start.json.zst", "s1/runs/run-1/events.jsonl.open"}
    assert zstd.decompress((root / "s1" / "runs" / "run-1" / "start.json.zst").read_bytes()) == text

    store.end_records("s1/run-1")
    stored = store.kind.list("s1/runs/run-1/")
    [part] = [key for key in stored if key.startswith("s1/runs/run-1/events.jsonl/")]
    assert set(stored) == {"s1/runs/run-1/start.json.zst", part} and part.endswith(".zst")  # one part, no plain file
    assert stored[part] < len(text) / 4
    assert store.record_files("s1/run-1") == ["events.jsonl", "start.json"]
    assert open_store(root.as_uri()).read_record("s1/run-1", "events.jsonl") == text
    store.end_records("s1/run-1")  # nothing left to compress
    assert len(store.kind.list("s1/runs/run-1/events.jsonl/")) == 1


def test_lines_added_to_a_local_record_file_after_its_record_ended_read_after_the_part(tmp_path):
    root = tmp_path / "store"
    store = open_store(root.as_uri())
    store.append_record("s1/run-1", "events.jsonl", b"one\ntwo\n")
    store.end_records("s1/run-1")
    again = open_store(root.as_uri())  # a resumed run
    again.append_record("s1/run-1", "events.jsonl", b"three\n")
    assert store.read_record("s1/run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    again.end_records("s1/run-1")
    assert len(store.kind.list("s1/runs/run-1/events.jsonl/")) == 2
    assert store.read_record("s1/run-1", "events.jsonl") == b"one\ntwo\nthree\n"
    with pytest.raises(MixedRecord):
        store.write_record("s1/run-1", "events.jsonl", b"replaced\n")


def test_a_line_a_crash_left_cut_in_a_local_record_file_is_ended_by_the_next_writer(tmp_path):
    root = tmp_path / "store"
    first = open_store(root.as_uri())
    first.append_record("s1/run-1", "events.jsonl", b'{"kind": "ev')  # the process stops here
    second, third = open_store(root.as_uri()), open_store(root.as_uri())
    for writer, line in ((second, b"two\n"), (third, b"three\n"), (second, b"four\n")):
        writer.append_record("s1/run-1", "events.jsonl", line)
    assert first.read_record("s1/run-1", "events.jsonl") == b'{"kind": "ev\ntwo\nthree\nfour\n'  # ended once, not twice


def test_a_local_record_file_that_grew_while_it_was_compressed_stays_as_it_is(tmp_path, monkeypatch):
    root = tmp_path / "store"
    store = open_store(root.as_uri())
    store.append_record("s1/run-1", "events.jsonl", b"one\n")
    compressed = blocks.compressed

    def with_a_line_added_meanwhile(data):
        open_store(root.as_uri()).append_record("s1/run-1", "events.jsonl", b"two\n")
        return compressed(data)

    monkeypatch.setattr(blocks, "compressed", with_a_line_added_meanwhile)
    store.end_records("s1/run-1")
    monkeypatch.undo()
    assert list(store.kind.list("s1/runs/run-1/")) == ["s1/runs/run-1/events.jsonl.open"]  # no part, nothing lost
    assert store.read_record("s1/run-1", "events.jsonl") == b"one\ntwo\n"


def test_an_s3_record_file_is_sent_in_one_part_at_each_flush_and_ending_the_record_changes_nothing(tmp_path, monkeypatch):
    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/records")
        store.append_record("s1/run-1", "events.jsonl", b"one\n")
        store.append_record("s1/run-1", "events.jsonl", b"two\n")
        assert store.record_files("s1/run-1") == []  # held back until the flush
        store.flush_records()
        [part] = store.kind.list("s1/runs/run-1/events.jsonl/")
        store.end_records("s1/run-1")
        assert list(store.kind.list("s1/runs/run-1/")) == [part]  # the parts it keeps, none plain
        assert store.read_record("s1/run-1", "events.jsonl") == b"one\ntwo\n"


def test_a_one_level_listing_gives_the_files_and_folders_right_under_a_prefix(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    for key in ("a/x", "a/b/y", "a/b/c/z", "ab"):
        store.kind.write(key, b"1")
    assert store.kind.list("a/", deep=False) == {"a/x": 1, "a/b/": None}
    assert store.kind.list("a", deep=False) == {"a/": None, "ab": 1}
    assert sorted(store.kind.list("a/")) == ["a/b/c/z", "a/b/y", "a/x"]


def test_an_s3_store_reads_its_own_settings_and_no_aws_setting(tmp_path, monkeypatch):
    # the keys of a cloud provider that other programs use are never the store's: it reads the names of its own
    aws = {"AWS_ACCESS_KEY_ID": "other", "AWS_SECRET_ACCESS_KEY": "other", "AWS_ENDPOINT_URL": "https://other",
           "AWS_DEFAULT_REGION": "eu-west-1"}
    with pytest.raises(StoreSettingError, match="WPT_STORE_ACCESS_KEY_ID and WPT_STORE_SECRET_ACCESS_KEY"):
        s3_settings(aws)
    with pytest.raises(StoreSettingError, match="WPT_STORE_SECRET_ACCESS_KEY"):
        s3_settings({**aws, "WPT_STORE_ACCESS_KEY_ID": "id"})
    own = {"WPT_STORE_ACCESS_KEY_ID": "id", "WPT_STORE_SECRET_ACCESS_KEY": "secret"}
    assert s3_settings({**aws, **own}) == {"access_key_id": "id", "secret_access_key": "secret"}
    assert s3_settings({**own, "WPT_STORE_SESSION_TOKEN": "token", "WPT_STORE_ENDPOINT_URL": "https://r2",
                        "WPT_STORE_REGION": "auto"}) == {
        "access_key_id": "id", "secret_access_key": "secret", "session_token": "token",
        "endpoint_url": "https://r2", "region": "auto"}

    env_file = tmp_path / ".env"  # the environment first, then the file
    env_file.write_text("WPT_STORE_ACCESS_KEY_ID=from-file\nWPT_STORE_SECRET_ACCESS_KEY=from-file\n"
                        "WPT_STORE_ENDPOINT_URL=https://file\n")
    assert s3_settings({"WPT_STORE_ACCESS_KEY_ID": "from-environment"}, env_file) == {
        "access_key_id": "from-environment", "secret_access_key": "from-file", "endpoint_url": "https://file"}

    for name in aws:
        monkeypatch.setenv(name, aws[name])
    for name in own:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(StoreSettingError, match="reads no AWS_"):
        open_store("s3://bucket/prefix")


def test_an_s3_store_signs_its_requests_with_its_own_keys(tmp_path, monkeypatch):
    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "the-id", "WPT_STORE_SECRET_ACCESS_KEY": "the-secret",
                        "WPT_STORE_REGION": "eu-west-1", "AWS_ACCESS_KEY_ID": "other", "AWS_SECRET_ACCESS_KEY": "other",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3", region_name="eu-west-1").create_bucket(
            Bucket="wpt-test", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        # neither an AWS endpoint nor an AWS profile (missing here) reaches the store
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://other.example")
        monkeypatch.setenv("AWS_PROFILE", "missing")
        store = open_store("s3://wpt-test/store")
        credentials = store.kind.client._request_signer._credentials
        assert (credentials.access_key, credentials.secret_key) == ("the-id", "the-secret")
        assert store.kind.client.meta.region_name == "eu-west-1"
        assert store.kind.client.meta.endpoint_url == "https://s3.eu-west-1.amazonaws.com"
        store.kind.write("a", b"1")
        assert store.kind.read("a")[0] == b"1"


def test_s3_store_round_trip(tmp_path, monkeypatch):
    # moto answers boto3's S3 requests inside this process: no server, no network.
    # The store's local cache follows XDG_CACHE_HOME, or LOCALAPPDATA on Windows.
    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
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

        store.append_record("s1/run-4", "events.jsonl", b"one\n")
        monkeypatch.setattr(store.client, "put_object", stored_then_interrupted)
        with pytest.raises(KeyboardInterrupt):
            store.flush_records()
        store.append_record("s1/run-4", "events.jsonl", b"two\n")
        store.flush_records()
        assert open_store("s3://wpt-test/some/prefix").read_record("s1/run-4", "events.jsonl") == b"one\ntwo\n"


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
        file = store.put_named(f"s1/datasets/d/{i}.parquet", tmp_path / f"{i}.parquet")
        shards.append({"key": file.key, "sha256": file.sha256, "token_sha256": token_sha256})

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

    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
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
            file = store.put_named(f"s1/datasets/d/{i}.parquet", path)
            shards.append({"key": file.key, "sha256": file.sha256, "token_sha256": token_sha256, "tokens": 10})
        stream = ShardStream(store, shards, window=2)

        def on_disk():
            return sorted((store.scratch / "tokens").glob("*.u32"))

        assert len(stream) == 40
        assert stream.at(0)[0:20].tolist() == list(range(0, 20))
        assert stream.at(25)[0:20].tolist() == list(range(20, 40))
        # after the last shard comes the first again; a window may span two shards
        assert stream.at(35)[5:15].tolist() == [35, 36, 37, 38, 39, 0, 1, 2, 3, 4]
        assert len(on_disk()) <= 3
        assert not any((store.cache / s["sha256"][:2] / s["sha256"]).exists() for s in shards)  # copies released
        stream.close()
        assert on_disk() == []


def test_reading_documents_from_a_remote_store_keeps_at_most_one_shard_on_disk(tmp_path, monkeypatch):
    # the drafter reads the training shards once at the end of a run: each
    # shard's local copy goes once its rows are read, as a shard stream's do
    from wpt_store.shards import documents, write_shard

    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
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
            file = store.put_named(f"s1/datasets/d/{i}.parquet", path)
            shards.append({"key": file.key, "sha256": file.sha256, "token_sha256": token_sha256, "tokens": 4})

        def copies():
            return [s["key"] for s in shards if (store.cache / s["sha256"][:2] / s["sha256"]).exists()]

        seen = []
        for document, source, tokens in documents(store, shards):
            assert copies() == []  # the shard being read is already released
            seen.append((document, source, tokens.tolist()))
        assert seen == [("s/0", "s", [0, 0]), ("s/again", "s", [0, 0]), ("s/1", "s", [1, 0]), ("s/2", "s", [2, 0])]


def test_a_refused_replace_of_a_file_already_stored_counts_as_stored(tmp_path, monkeypatch):
    # on Windows, two processes storing the same file at once: replacing a file
    # another process has open is refused, and the name is the content's hash
    store = open_store((tmp_path / "store").as_uri())
    source = tmp_path / "data.bin"
    source.write_bytes(b"same bytes\n")
    sha = store.put_file(source)
    target = tmp_path / "store" / "cache" / "files" / sha

    def refuse(tmp, to):
        raise PermissionError(5, "Access denied")

    monkeypatch.setattr("wpt_store.kinds.os.replace", refuse)
    store._put(sha, source)  # what put_file does when both workers saw no file
    assert target.read_bytes() == b"same bytes\n"
    assert [p.name for p in target.parent.iterdir()] == [sha]

    # nothing stored under that name: the refusal is a real failure
    target.unlink()
    with pytest.raises(PermissionError):
        store.put_file(source)
    assert list(target.parent.iterdir()) == []


def test_a_replace_larger_than_one_part_goes_up_in_parts_under_its_condition(tmp_path, monkeypatch):
    # a single upload is refused above 5 GiB, which the index may outgrow: a replace in parts must still
    # only replace the version it read
    from wpt_store import kinds
    from wpt_store.kinds import Changed

    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
                        "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(11 << 20))
    with mock_aws():
        client = boto3.client("s3")
        client.create_bucket(Bucket="wpt-test")
        bucket = kinds.Bucket(client, "wpt-test", "store", part=5 << 20)
        first = bucket.replace("index", big, None)
        assert bucket.read("index") == (big.read_bytes(), first)
        assert first.strip('"').endswith("-3")  # three parts, not one upload
        with pytest.raises(Changed):
            bucket.replace("index", big, None)  # a file is there now
        big.write_bytes(os.urandom(11 << 20))
        second = bucket.replace("index", big, first)
        assert bucket.read("index") == (big.read_bytes(), second)


def test_uploads_in_parts_running_at_once_share_one_limit_of_parts_in_flight(tmp_path, monkeypatch):
    # tars stored at the same moment must not multiply the connections to S3: their parts share one limit
    from wpt_store import kinds

    monkeypatch.setattr(kinds, "UPLOADS", 3)
    monkeypatch.setattr(kinds, "SETTLE", 0.0)  # a started part shows in the measures at once
    monkeypatch.setattr(kinds, "_parts", None)  # this process's parts, made again with that limit
    # the limit alone decides here: on a busy machine the free memory would keep a second part from starting
    sizing = kinds._upload_parts().sizing
    monkeypatch.setattr(sizing, "measure", lambda running: 1)
    monkeypatch.setattr(sizing.machine, "memory", lambda: (1 << 40, 1 << 40))
    lock, running, peak, sent = threading.Lock(), [0], [0], {}

    class Client:
        def create_multipart_upload(self, Bucket, Key):
            return {"UploadId": Key}

        def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.3)
            with lock:
                running[0] -= 1
                sent.setdefault(Key, {})[PartNumber] = Body
            return {"ETag": f"{Key}-{PartNumber}"}

        def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
            sent[Key]["done"] = [part["PartNumber"] for part in MultipartUpload["Parts"]]
            return {"ETag": f"{Key}-done"}

        def abort_multipart_upload(self, Bucket, Key, UploadId):
            raise AssertionError("no upload fails here")

    bucket = kinds.Bucket(Client(), "bucket", "", part=10)
    files = {}
    for name in ("one", "two"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode() * 20)  # 60 bytes: 6 parts
    uploads = [threading.Thread(target=bucket.write, args=(name, path)) for name, path in files.items()]
    for upload in uploads:
        upload.start()
    for upload in uploads:
        upload.join(60)
    assert peak[0] == 3  # three at once in all, not three for each upload
    for name, path in files.items():
        parts = sent[name]
        assert parts.pop("done") == [1, 2, 3, 4, 5, 6]
        assert b"".join(parts[number] for number in sorted(parts)) == path.read_bytes()


def test_a_failed_part_stops_the_parts_of_its_upload_that_have_not_started(tmp_path, monkeypatch):
    # after a part fails, sending the parts still waiting only wastes the link before the upload is aborted
    from wpt_store import kinds

    parts = kinds._Parts()
    parts.sizing.may_start = lambda running, unseen: running == 0  # one part at a time: the others wait
    monkeypatch.setattr(kinds, "_parts", parts)
    sent, aborted = [], []

    class Client:
        def create_multipart_upload(self, Bucket, Key):
            return {"UploadId": Key}

        def upload_part(self, Bucket, Key, UploadId, PartNumber, Body):
            sent.append(PartNumber)
            raise ConnectionError("the link dropped")

        def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
            raise AssertionError("an upload with a failed part is never completed")

        def abort_multipart_upload(self, Bucket, Key, UploadId):
            aborted.append(Key)

    source = tmp_path / "file"
    source.write_bytes(b"x" * 60)  # 6 parts of 10 bytes
    with pytest.raises(ConnectionError, match="the link dropped"):
        kinds.Bucket(Client(), "bucket", "", part=10).write("file", source)
    assert len(sent) == 1
    assert aborted == ["file"]


class CutBodies:
    """A client whose answers to get_object have a body that breaks while it is read, the first `cuts` times."""

    def __init__(self, client, cuts):
        self.client, self.cuts, self.asked = client, cuts, []

    def get_object(self, **request):
        from botocore.exceptions import ResponseStreamingError

        self.asked.append(request)
        answer = self.client.get_object(**request)
        if len(self.asked) > self.cuts:
            return answer

        class Body:
            def read(self):
                raise ResponseStreamingError(error="Connection broken: IncompleteRead(3 bytes read, 5 more expected)")

        return {**answer, "Body": Body()}


@pytest.mark.parametrize("cuts, tries", [(3, 4), (5, 5)])
def test_a_ranged_read_whose_answer_is_cut_while_read_is_asked_again_up_to_the_fixed_number_of_tries(
        monkeypatch, cuts, tries):
    from botocore.exceptions import ResponseStreamingError

    from wpt_store import kinds

    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
                        "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr(kinds, "WAIT", 0)
    with mock_aws():
        client = boto3.client("s3")
        client.create_bucket(Bucket="wpt-test")
        etag = client.put_object(Bucket="wpt-test", Key="store/file", Body=b"0123456789")["ETag"]
        cut = CutBodies(client, cuts)
        bucket = kinds.Bucket(cut, "wpt-test", "store")
        assert kinds.TRIES == 5
        if cuts < kinds.TRIES:
            assert bucket.read("file", 2, 4) == (b"2345", etag)
        else:
            with pytest.raises(ResponseStreamingError):
                bucket.read("file", 2, 4)
        assert len(cut.asked) == tries
        assert {request["Range"] for request in cut.asked} == {"bytes=2-5"}


class SlowAnswers:
    """A client whose get_object answers after a delay, counting the requests in flight, and raising `fail` for a
    request that starts at or after the byte `fail_from`."""

    def __init__(self, client, delay=0.05, fail_from=None, fail=None):
        self.client, self.delay, self.fail_from, self.fail = client, delay, fail_from, fail
        self.lock, self.now, self.most = threading.Lock(), 0, 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def get_object(self, **request):
        start = request.get("Range", "bytes=0-").removeprefix("bytes=").split("-")[0]
        with self.lock:
            self.now += 1
            self.most = max(self.most, self.now)
        try:
            time.sleep(self.delay)
            if self.fail_from is not None and start.isdigit() and int(start) >= self.fail_from:
                raise self.fail
            return self.client.get_object(**request)
        finally:
            with self.lock:
                self.now -= 1


def stored_text(tmp_path, monkeypatch, **slow):
    """(a client over moto, a store over it reading with frames of one block each, the object's sha256, its bytes):
    a text of 12 frames stored through one store and read back through another with an empty cache."""
    import random

    from wpt_store import S3Store, ahead

    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test",
                        "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    raw = boto3.client("s3")
    raw.create_bucket(Bucket="wpt-test")
    words = [os.urandom(3).hex() for _ in range(2000)]
    generator = random.Random(1)
    text = " ".join(generator.choice(words) for _ in range(2_000_000)).encode()[:12 << 20]
    sha = S3Store("wpt-test", "p", tmp_path / "writer", client=raw).put_bytes(text)
    slow_client = SlowAnswers(raw, **slow)
    store = S3Store("wpt-test", "p", tmp_path / "reader", client=slow_client)
    store._ahead = ahead.Ahead(requests=4, limit=3 << 20, frames=1)
    return slow_client, store, sha, text


def test_a_stored_file_read_in_order_over_a_slow_link_gives_the_same_bytes_with_requests_in_flight_together(
        tmp_path, monkeypatch):
    # R2 answers each request slowly: reading a file in order must keep several requests in flight, and the
    # bytes ahead of the reader must stay within the limit, whatever the timing
    with mock_aws():
        client, store, sha, text = stored_text(tmp_path, monkeypatch)
        read = []
        for start in range(0, len(text), 256 << 10):
            read.append(store.read_range(sha, start, min(256 << 10, len(text) - start)))
            held = sum(size for _, size in store._ahead._held.values())
            assert held <= store._ahead.limit
        assert b"".join(read) == text
        assert 2 <= client.most <= 4  # more than one at once, never more than the fixed number


def test_an_error_on_a_range_fetched_ahead_is_raised_at_the_read_that_needs_it(tmp_path, monkeypatch):
    class LinkDown(Exception):
        pass

    with mock_aws():
        client, store, sha, text = stored_text(tmp_path, monkeypatch)
        table = store._table(store._found(store._file(sha)))
        client.fail_from, client.fail = table.stored[8], LinkDown("the link dropped")  # from the 9th frame on
        frame = table.original[1]  # the original bytes of one frame, about 1 MB
        for start in range(0, 6 * frame, frame):  # the reads before: fine, though blocks ahead already failed
            assert store.read_range(sha, start, frame) == text[start:start + frame]
        with pytest.raises(LinkDown):
            for start in range(6 * frame, len(text), frame):
                store.read_range(sha, start, frame)
        assert start + frame > table.original[8]  # the read that needed the failed block, not an earlier one
