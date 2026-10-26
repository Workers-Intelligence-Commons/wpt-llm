"""Shared setup for the application tests: a dataset built through
wpt-dataset from a real git project, and a tiny model trained on it through
wpt-train. Test files reach them as fixtures (they cannot import this file
under --import-mode=importlib)."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpt_dataset import download as downloading
from wpt_dataset import manifest, sources
from wpt_dataset.build import build_dataset
from wpt_dataset.repo import Repo, address_hash, dump_json
from wpt_store import open_store
from wpt_train import main as train

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "T",
           "GIT_COMMITTER_EMAIL": "t@example.org", "GIT_CONFIG_GLOBAL": os.devnull}
ATTRIBUTION = "Fox Stories, copyright (c) 2026 Ada Writer, https://example.org/fox"
LICENSE_TEXT = "Copyright (c) 2026 Ada Writer\n\nPermission is hereby granted, free of charge, to any person.\n"


TINY_MODEL = {"n_layer": 4, "global_attention": ["CSA"], "d_model": 16, "kda_heads": 2,
              "kda_head_dim": 8, "attention_heads": 2, "attention_head_dim": 128, "query_rank": 8, "output_groups": 1,
              "output_rank": 8, "index_heads": 2, "index_head_dim": 128, "index_top_k": 2, "n_experts": 4, "top_k": 2, "expert_width": 8, "latent_width": 8,
              "shared_width": 16, "engram_layers": [2], "engram_order": 3, "engram_heads": 2, "engram_dim": 4,
              "engram_table_size": 31}
# the read-text step of the recipe "d": the optimizer, its settings, the step's text windows, the recipe's seed
ONE_STEP = {"optimizer": "muon", "settings": {}, "stages": [{"block_size": 16, "steps": 1, "batch_size": 4}],
            "eval_every": 2, "eval_batches": 20, "seed": 1337}
# the drafter every run trains at its end, as small and short as still trains and calibrates one
TINY_DRAFTER = {"epochs": 1, "prompts": 4, "prompt_tokens": 8, "answer_tokens": 8, "batch": 2, "anchors": 4,
                "held_out": 0.25}


def checksum_of(data) -> str:
    """The checksum of what a source's address gives, kept nowhere."""
    with tempfile.TemporaryDirectory(prefix="wpt-test-staging-") as tmp:
        (Path(tmp) / "download").mkdir()
        return manifest.checksum(sources.fetch(data, data["address"], downloading.Staging(Path(tmp) / "files"),
                                               Path(tmp) / "download"))


def download(root, store, sid):
    """The download step on one source, as `wpt-dataset download` runs it,
    in a session of its own; its error raised."""
    with store.session(f"20261025T120000Z-test-download-{sid}", "test", "abc123", False) as session:
        [done] = downloading.download(Repo(root), [sid], store, session, log=lambda *_: None)
    if done.error is not None:
        raise done.error


@pytest.fixture
def fetching():
    """checksum_of and download, for tests that write sources of their own."""
    return SimpleNamespace(checksum_of=checksum_of, download=download)


@pytest.fixture
def built(tmp_path, monkeypatch):
    """A dataset of one git project of 16 documents, about half of them held
    out; its 3000 training tokens draw only some of the training documents.
    The working folder tmp_path is the repository, and the store is
    WPT_STORE. Gives store, records(command) (the record folders of that
    command's runs, oldest first), checksum_of(data) and download(source id)
    (for a test's own sources), dataset_id (built from the model recipe "d", a
    tiny model trained one step), source_id, attribution,
    recipe(name, model=..., training=..., drafter=..., steps=...), which
    writes a model recipe of the same data part, with those changes to "d"
    (training: to ONE_STEP, its read-text step; steps: the entries that
    follow it), and returns its name, source(upstream, files, formats, tags,
    attribution, text), which commits files into the new git project
    upstream, adds it as an admitted CC0-1.0 source of that attribution
    and license text and downloads it into the store, returning its id, and
    build(name), which builds the dataset of recipe name again from the
    sources there are now, returning its id."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    for i in range(16):
        (upstream / f"t{i}.txt").write_text(f"story {i}: the quick brown fox jumps over the lazy dog. " * 30)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=upstream, env=GIT_ENV, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True).stdout.strip()

    root = tmp_path
    repo = Repo(root)
    store = open_store((tmp_path / "store").as_uri())
    def admitted(name, upstream, commit, license, attribution, text, formats, tags, held_out_by=None):
        data = {"type": "git", "address": str(upstream), "revision": commit,
                "licenses": [{"license": license, "paths": ["**"], "attribution": attribution, "text": text}],
                "formats": formats, "tags": tags, **({"held_out_by": held_out_by} if held_out_by else {}),
                "decision": {"decision": "admitted", "by": "T", "date": "2026-10-14", "reason": "test"}}
        data["checksum"] = checksum_of(data)
        sid = f"example-{name}-{address_hash(str(upstream))}"
        (root / "data" / "sources" / sid).mkdir(parents=True)
        (root / "data" / "sources" / sid / "source.json").write_text(dump_json(data))
        download(root, store, sid)
        return sid

    source_id = admitted("fox", upstream, commit, "MIT", ATTRIBUTION, LICENSE_TEXT,
                         [{"format": "text", "paths": ["**"]}], [{"tags": ["en"], "paths": ["**"]}],
                         # the held-out hash reads the document's path, not its id: the id holds a hash of the
                         # address, which holds the temporary folder's name, so a test would hold other documents
                         # out in every run
                         held_out_by={"pattern": r"(t\d+)\.txt"})

    def source(upstream, files, formats, tags, attribution, text):
        for path, content in files.items():
            (upstream / path).parent.mkdir(parents=True, exist_ok=True)
            (upstream / path).write_bytes(content)
            if path.endswith("/check"):
                (upstream / path).chmod(0o755)
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
            subprocess.run(["git", *args], cwd=upstream, env=GIT_ENV, check=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True,
                                text=True).stdout.strip()
        return admitted(upstream.name, upstream, commit, "CC0-1.0", attribution, text, formats, tags)

    def write(kind, name, document):
        path = root / "data" / "recipes" / f"{kind}s" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
        return name

    def recipe(name="d", model=None, training=None, drafter=None, steps=()):
        text = {**ONE_STEP, **(training or {})}
        read_text = {"step": "read-text", "optimizer": {"name": text["optimizer"], "settings": text["settings"]},
                     "settings": {key: text[key] for key in ("stages", "eval_every", "eval_batches")}}
        return write("model", name, {"data": {"mixture": "m", "tokenizer": "t", "total_tokens": 3000, "held_out": 0.5},
                                     "model": {**TINY_MODEL, **(model or {})}, "seed": text["seed"],
                                     "steps": [read_text, *steps], "drafter": {**TINY_DRAFTER, **(drafter or {})}})

    write("mixture", "m", {"pools": [{"name": "en", "select": "en", "weight": 1}]})
    write("tokenizer", "t", {"version": 1, "mixture": "m", "vocab_size": 556})
    dataset_id = build_dataset(repo, recipe(), store, log=lambda _: None)
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())  # where each command run also writes its record
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)  # no live backend, whatever this machine sets
    monkeypatch.setenv("WPT_MAX_RESTARTS", "0")  # wpt-train runs in the test's own process, unless a test says
    monkeypatch.chdir(tmp_path)
    def records(command):
        found = []
        for folder in store.record_folders():
            start = json.loads(store.read_record(folder, "start.json"))
            if start["command"] == command:
                found.append(SimpleNamespace(
                    folder=folder, start=start, ids=json.loads(store.read_record(folder, "ids.json")),
                    events=[json.loads(line) for line in store.read_record(folder, "events.jsonl").splitlines()],
                    end=json.loads(store.read_record(folder, "end.json")) if "end.json" in store.record_files(folder)
                    else None))
        return sorted(found, key=lambda run: run.start["started"])

    return SimpleNamespace(store=store, dataset_id=dataset_id, source_id=source_id, attribution=ATTRIBUTION,
                           license_text=LICENSE_TEXT, recipe=recipe, records=records, checksum_of=checksum_of,
                           download=lambda sid: download(root, store, sid), source=source,
                           build=lambda name="d": build_dataset(repo, name, store, log=lambda _: None))


@pytest.fixture
def trained(built, tmp_path):
    """The built dataset plus tmp_path/model, trained by the recipe "d"."""
    train(["d", built.dataset_id, "--out", "model"])
    return SimpleNamespace(**vars(built), folder=tmp_path / "model")


@pytest.fixture
def s3(tmp_path, monkeypatch):
    """An S3 store that WPT_STORE now names (moto answers its requests inside
    the test process). Gives store, copy(local), which puts a local store's
    objects and refs into it (the built dataset, for a command that reads
    it), and refuse(marker), after which the bucket refuses every group of
    record lines that holds marker."""
    import boto3
    from moto import mock_aws

    from wpt_store import S3Store

    for name, value in {"WPT_STORE_ACCESS_KEY_ID": "test", "WPT_STORE_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "us-east-1",
                        "XDG_CACHE_HOME": str(tmp_path / "cache"), "LOCALAPPDATA": str(tmp_path / "cache")}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3").create_bucket(Bucket="wpt-test")
        store = open_store("s3://wpt-test/store")
        monkeypatch.setenv("WPT_STORE", "s3://wpt-test/store")

        def copy(local):
            for path in sorted((local.root / "objects").glob("*/*")):
                store.put_file(local.path(path.name.removesuffix(".zst")))  # a compressed one as its own bytes
            for path in sorted((local.root / "refs").glob("*/*")):
                store.set_ref(path.parent.name, path.name, path.read_text().strip())

        def refuse(marker):
            real = S3Store._raw_flush_records

            def refusing(self):
                if any(marker in data for additions in self._pending.values() for *_, data in additions):
                    raise OSError("the bucket refused the request")
                real(self)

            monkeypatch.setattr(S3Store, "_raw_flush_records", refusing)

        yield SimpleNamespace(store=store, copy=copy, refuse=refuse)
