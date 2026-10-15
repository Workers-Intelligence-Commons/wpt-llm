import gzip
import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wpt_dataset.build import gather
from wpt_dataset.errors import LicenseMapError
from wpt_dataset.sources import source_tree


def parquet_bytes(rows):
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buffer)
    return buffer.getvalue()


def records_source(kit, repo, store, http_folder, licenses):
    folder, base = http_folder
    lines = [{"id": "a", "text": "first words", "metadata": {"license": "CC-BY", "url": "https://example.org/a"}},
             {"id": "b", "text": "second words", "metadata": {"license": "Various open data", "url": "https://example.org/b"}},
             # some datasets store their metadata as a string holding JSON
             {"id": "c", "text": "third words", "metadata": json.dumps({"license": "CC0", "url": "https://example.org/c"})}]
    files = {"data/part-0.jsonl.gz": gzip.compress("".join(json.dumps(r) + "\n" for r in lines).encode()),
             "data/part-1.parquet": parquet_bytes([
                 {"id": "d", "text": "fourth words", "metadata": {"license": "CC0", "url": "https://example.org/d"}},
                 {"id": "e", "text": "", "metadata": {"license": "CC0", "url": "https://example.org/e"}}])}
    (folder / "dataset.tar.gz").write_bytes(kit.tar_bytes(top="dataset", files=files, executable=set(), symlinks={}))
    record = {"format": "records", "paths": ["data/**"], "text": "text", "id": "id", "license": "metadata.license",
              "credit": ["metadata.url"]}
    return kit.add_source(repo, store, "example", "records", {
        "type": "http-archive", "address": f"{base}/dataset.tar.gz", "revision": "2026-10-15",
        "extract": {"strip": 1}, "formats": [record], "licenses": licenses})


def test_only_records_whose_license_value_is_listed_become_documents_with_their_credit(kit, repo, store, http_folder):
    # each record of a large dataset carries its own license: WPT keeps a
    # record only when a license entry lists its value, and every model
    # trained on it can name each document's credit
    sid = records_source(kit, repo, store, http_folder, [
        {"license": "CC-BY-4.0", "paths": ["data/**"], "values": ["CC-BY"], "attribution": kit.ATTRIBUTION, "text": "CC BY 4.0"},
        {"license": "CC0-1.0", "paths": ["data/**"], "values": ["CC0"], "attribution": kit.ATTRIBUTION, "text": "CC0 1.0"}])

    documents, info, _ = gather(repo, store, log=lambda _: None)

    assert [d.id for d in documents] == [f"{sid}/data/part-0.jsonl.gz#a", f"{sid}/data/part-0.jsonl.gz#c",
                                         f"{sid}/data/part-1.parquet#d"]
    assert store.get_bytes(documents[0].sha256) == b"first words\n"
    credits = [json.loads(line) for line in store.get_bytes(info[sid]["credits"]).decode().splitlines()]
    assert credits == [
        {"document": f"{sid}/data/part-0.jsonl.gz#a", "license": "CC-BY-4.0", "credit": {"metadata.url": "https://example.org/a"}},
        {"document": f"{sid}/data/part-0.jsonl.gz#c", "license": "CC0-1.0", "credit": {"metadata.url": "https://example.org/c"}},
        {"document": f"{sid}/data/part-1.parquet#d", "license": "CC0-1.0", "credit": {"metadata.url": "https://example.org/d"}}]


def test_a_records_file_needs_license_entries_that_list_their_values(kit, repo, store, http_folder):
    # without `values`, an entry would let every record of the file in,
    # whatever license the record itself carries
    sid = records_source(kit, repo, store, http_folder, [
        {"license": "CC0-1.0", "paths": ["data/**"], "attribution": kit.ATTRIBUTION, "text": "CC0 1.0"}])

    with pytest.raises(LicenseMapError, match="every license entry covering it lists the `values`"):
        source_tree(repo.source(sid), store)
