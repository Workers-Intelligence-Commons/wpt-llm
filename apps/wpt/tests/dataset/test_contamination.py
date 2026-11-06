"""`wpt dataset contamination`: the check of a built dataset, run by hand,
through the command line, on a dataset built from a real git project."""

import json
import tempfile

import pytest

from wpt import train
from wpt.dataset import main
from wpt_dataset import build, contamination
from wpt_dataset.build import build_dataset
from wpt_store.shards import read_shard
from wpt_tokenizer import Tokenizer
from wpt_utils.lines import bare

PASSAGES = 5  # each document ends with one of them, so documents on both sides of the split share runs


@pytest.fixture
def dataset(kit, repo, store, tmp_path, monkeypatch):
    """The id of a dataset that `wpt dataset build` made, held-out documents sharing runs with training documents,
    every document drawn. The held-out hash reads the path, so the same documents are held out in every run."""
    files = {}
    for i in range(40):
        own = " ".join(f"own{i}x{k}" for k in range(30))
        files[f"t{i:02d}.txt"] = f"{own} {' '.join(f'p{i % PASSAGES}x{k}' for k in range(20))}\n".encode()
    folder, _ = kit.git_project(tmp_path / "many", files=files, executable=set(), symlinks={})
    kit.add_source(repo, store, "example", "many", {"type": "git", "address": folder,
                                                    "held_out_by": {"pattern": r"(t\d+)\.txt"},
                                                    "tags": [{"tags": ["all"], "paths": ["**"]}]})
    kit.write_recipe(repo, "mixture", "all", {"pools": [{"name": "all", "select": "all", "weight": 1,
                                                         "max_epochs": 100}]})
    kit.write_recipe(repo, "tokenizer", "all", {"version": 1, "sample_bytes": 10**9, "vocab_size": 556})
    kit.write_model_recipe(repo, "split", {"mixture": "all", "tokenizer": "all", "total_tokens": 200000,
                                           "held_out": 0.5})
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert command(repo, "build", "split") == 0
    return build.last_dataset(store, "split")


def command(repo, *words) -> int:
    return main(["--root", str(repo.root), *words])


def record_of(store, dataset_id):
    return json.loads(store.get_bytes(store.get_ref("datasets", dataset_id)))


def runs_of_command(store, name):
    """The events of each run of the command of that name, in no set order: runs of one second sort at random."""
    folders = [folder for folder in store.record_folders()
               if json.loads(store.read_record(folder, "start.json"))["command"] == name]
    return [[json.loads(line) for line in store.read_record(folder, "events.jsonl").splitlines()]
            for folder in folders]


def spans(events):
    return [event["name"] for event in events if event["kind"] == "span_start"]


def plain_report(store, record):
    """The check by its definition, on the texts of the shards: the first held-out document with each run of 13
    words, then the first training document, in the shards' order, with it."""
    tokenizer = Tokenizer.from_bytes(store.get_bytes(record["tokenizer"]["vocabulary"]))

    def texts(shards):
        found = {}
        for shard in shards:
            for tokens, _, document, _, _ in read_shard(store.path(shard["object"])):
                found.setdefault(document, tokenizer.decode(tokens.tolist()).split())
        return found

    def runs(words):
        return {b" ".join(words[i:i + 13]) for i in range(len(words) - 12)}

    held_out, owner, paired = texts(record["held_out_shards"]), {}, {}
    for document, words in held_out.items():
        for run in runs(words):
            owner.setdefault(run, document)
    for document, words in texts(record["shards"]).items():
        for run in runs(words):
            if run in owner:
                paired.setdefault(owner[run], document)
    return {"ngram_words": 13, "held_out_documents": len(held_out), "overlapping": len(paired),
            "examples": [{"held_out": h, "training": t} for h, t in sorted(paired.items())[:20]]}


def test_the_command_saves_and_prints_what_the_check_finds_and_a_second_run_prints_it_again(
        repo, store, dataset, monkeypatch, capsys):
    monkeypatch.setattr(contamination, "BATCH_TOKENS", 1)  # one document per batch, so batches end in any order
    capsys.readouterr()
    assert command(repo, "contamination", dataset) == 0  # an overlap is a finding, not a failure
    first = capsys.readouterr().out

    report = json.loads(store.get_bytes(store.get_ref(contamination.REPORT, dataset)))
    assert report == plain_report(store, record_of(store, dataset))
    assert report["overlapping"] >= 2 and 0 < report["held_out_documents"] < 40
    assert (f"contamination: {report['overlapping']} held-out documents share a run of 13 words with training "
            "documents (reported only)") in bare(first)
    example = report["examples"][0]
    assert f"{example['held_out']} shares a run with {example['training']}" in bare(first)

    assert command(repo, "contamination", dataset) == 0
    again = capsys.readouterr().out
    assert "its saved report, nothing checked" in again
    assert [line for line in bare(first).splitlines() if "contamination:" in line or "shares a run" in line] == \
           [line for line in bare(again).splitlines() if "contamination:" in line or "shares a run" in line]
    runs = runs_of_command(store, "wpt dataset contamination")
    assert sorted("contamination" in spans(events) for events in runs) == [False, True]  # one run checked


def test_a_document_several_shards_repeat_is_read_once_and_the_report_is_the_same_definition(repo, store, dataset):
    store.delete_ref("datasets", dataset)
    assert build_dataset(repo, "split", store, log=lambda *_: None, shard_tokens=20000) == dataset
    record = record_of(store, dataset)
    assert len(record["shards"]) > 2

    assert command(repo, "contamination", dataset) == 0
    report = json.loads(store.get_bytes(store.get_ref(contamination.REPORT, dataset)))
    assert report == plain_report(store, record) and report["overlapping"] >= 2


def test_a_check_that_fails_leaves_no_local_copy_and_no_report(repo, store, dataset, tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    record = record_of(store, dataset)
    broken = {**record, "shards": [{**record["shards"][0], "token_sha256": "0" * 64}]}
    store.set_ref("datasets", dataset, store.put_bytes(json.dumps(broken).encode()))

    assert command(repo, "contamination", dataset) == 1
    assert store.get_ref(contamination.REPORT, dataset) is None
    assert list(temporary.iterdir()) == []
    assert not (store.cache / broken["shards"][0]["object"][:2] / broken["shards"][0]["object"]).exists()


def test_a_dataset_the_store_does_not_hold_is_refused(repo, store, dataset, capsys):
    assert command(repo, "contamination", "0" * 64) == 1
    assert "NotFound" in capsys.readouterr().err
    assert runs_of_command(store, "wpt dataset contamination") == []


def test_a_build_neither_runs_the_check_nor_records_it(store, dataset):
    assert "contamination" not in record_of(store, dataset)
    [events] = runs_of_command(store, "wpt dataset build")
    assert "shards" in spans(events) and "contamination" not in spans(events)
    assert store.get_ref(contamination.REPORT, dataset) is None


def test_a_record_that_holds_a_report_stays_current_and_the_command_prints_that_report(
        repo, store, dataset, capsys):
    report = {"ngram_words": 13, "held_out_documents": 7, "overlapping": 1,
              "examples": [{"held_out": "s/held", "training": "s/trained"}]}
    written = store.put_bytes(json.dumps({**record_of(store, dataset), "contamination": report}).encode())
    store.set_ref("datasets", dataset, written)

    assert build_dataset(repo, "split", store, log=lambda *_: None) == dataset
    assert store.get_ref("datasets", dataset) == written  # the same code gives the same id: found, not built again
    record, known, _ = train.load_dataset(store, dataset, repo.model_recipe("split"))
    assert record["id"] == dataset and known == written

    capsys.readouterr()
    assert command(repo, "contamination", dataset) == 0
    printed = bare(capsys.readouterr().out)
    assert "contamination: 1 held-out documents share a run of 13 words" in printed
    assert "s/held shares a run with s/trained" in printed
    assert store.get_ref(contamination.REPORT, dataset) is None
    [events] = runs_of_command(store, "wpt dataset contamination")
    assert "contamination" not in spans(events)
