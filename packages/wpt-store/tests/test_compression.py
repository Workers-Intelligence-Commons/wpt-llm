import io
import random
from compression import zstd

import boto3
import pytest
from moto import mock_aws

from wpt_store import CorruptObject, blocks, open_store

TEXT = b"".join(f"line {i} of a text that compresses well\n".encode() for i in range(150_000))  # about 6 MB
NOISE = random.Random(1).randbytes(11 << 20)  # does not compress; on S3 it goes up in 3 parts


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path, monkeypatch):
    """A local store, or an S3 store whose requests moto answers inside the test process."""
    if request.param == "local":
        yield open_store((tmp_path / "store").as_uri())
        return
    for name, value in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/compressed")
        store.kind.part = 5 << 20  # S3's smallest part, so NOISE goes up in parts, several at once
        yield store


def test_an_object_is_kept_compressed_only_when_that_saves_space_and_reads_back_alike(store, tmp_path):
    (tmp_path / "text").write_bytes(TEXT)
    (tmp_path / "noise").write_bytes(NOISE)
    text, noise = store.put_file(tmp_path / "text"), store.put_file(tmp_path / "noise")
    small = store.put_bytes(b"too short to shrink\n")
    stored = store.kind.list("objects/")
    assert stored[f"objects/{text[:2]}/{text}.zst"] < len(TEXT) / 5
    assert stored[f"objects/{noise[:2]}/{noise}"] == len(NOISE)
    assert stored[f"objects/{small[:2]}/{small}"] == 20
    assert len(stored) == 3

    # the compressed file is plain zstd: any zstd tool gives the bytes back
    assert zstd.decompress(store.kind.read(f"objects/{text[:2]}/{text}.zst")[0]) == TEXT

    # sizes are known without copying anything into the cache
    assert (store.size(text), store.size(noise), store.size(small)) == (len(TEXT), len(NOISE), 20)
    assert not any(store.cache.rglob(text))

    # a range spanning two frames, read without the rest, then the whole object
    middle = blocks.FRAME * 3 - 100
    assert store.read_range(text, middle, 300) == TEXT[middle:middle + 300]
    assert store.read_range(noise, 10, 5) == NOISE[10:15]
    assert store.get_bytes(text) == TEXT and store.get_bytes(noise) == NOISE
    assert store.path(text).read_bytes() == TEXT
    assert store.path(small).read_bytes() == b"too short to shrink\n"
    store.release(text)
    store.delete(text)
    assert not store.has(text)


def test_a_damaged_compressed_object_is_refused(tmp_path):
    store = open_store((tmp_path / "store").as_uri())
    sha = store.put_bytes(TEXT)
    packed = tmp_path / "store" / "objects" / sha[:2] / f"{sha}.zst"
    data = bytearray(packed.read_bytes())
    data[100] ^= 0xFF
    packed.write_bytes(bytes(data))
    with pytest.raises(CorruptObject):
        store.get_bytes(sha)
    with pytest.raises(CorruptObject):
        store.path(sha)


def test_frames_compressed_side_by_side_give_the_bytes_one_at_a_time_gives():
    alone, together = io.BytesIO(), io.BytesIO()
    writer = blocks.Writer(alone)
    writer.write(TEXT)
    writer.close()
    executor, sizing = blocks.parallel()
    with executor:
        writer = blocks.Writer(together, executor, sizing, batch=4)
        for at in range(0, len(TEXT), 300_000):  # written in pieces that do not match the frames
            writer.write(TEXT[at:at + 300_000])
        writer.close()
    assert together.getvalue() == alone.getvalue()


def test_each_object_is_looked_for_once_however_many_ranges_are_read(store):
    noise = store.put_bytes(NOISE[:1000])  # kept as it is: looked for under the compressed name first
    reader = open_store(store.root.as_uri() if hasattr(store, "root") else f"s3://{store.bucket}/{store.prefix}")
    probes = []
    real = reader.kind.read

    def counted(key, start=0, length=None):
        if length == 0:
            probes.append(key)
        return real(key, start, length)

    reader.kind.read = counted
    for at in range(0, 1000, 100):
        assert reader.read_range(noise, at, 100) == NOISE[at:at + 100]
    assert len(probes) == 2  # its compressed name, then its own, once
