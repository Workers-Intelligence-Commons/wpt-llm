"""`cache/` holds only what can be rebuilt: emptying it leaves every dataset and every checkpoint usable."""

import json
import shutil

import torch

from wpt.train import StoreCheckpoint
from wpt.train.remote import named_files
from wpt_dataset.build import document_list
from wpt_store import open_store
from wpt_store.shards import documents
from wpt_telemetry import Telemetry
from wpt_tokenizer import Tokenizer

TALK = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}]}


def test_emptying_the_cache_leaves_a_dataset_readable_and_a_checkpoint_to_resume_from(built, tmp_path):
    sid = built.source(tmp_path / "talks", {"talks.jsonl": (json.dumps(TALK) + "\n").encode()},
                       [{"format": "chat", "paths": ["**"]}], [{"tags": ["answer-as-assistant"], "paths": ["**"]}],
                       "Talks, by Bo Speaker, 2026", "Bo Speaker dedicates these talks to the public domain.\n")
    dataset_id = built.build()
    store = built.store
    record = store.get_json(f"{store.session}/datasets/d-{dataset_id[:8]}/dataset.json")
    assert record["conversations"]["count"] == 1 and record["conversations"]["sources"] == [sid]

    # every file the record refers to lies in the dataset's folder, none in cache/
    files = named_files(record).values()
    assert files and all(file.key.startswith(f"{store.session}/datasets/d-{dataset_id[:8]}/") for file in files)

    checkpoint = StoreCheckpoint(store, "fox-1", Telemetry.off(), set())
    checkpoint.save({"step": 3, "weights": torch.arange(4)})
    state = checkpoint.state_file()
    assert state["key"].startswith(f"{store.session}/models/fox-1/")

    shutil.rmtree(store.root / "cache")
    shutil.rmtree(store.cache, ignore_errors=True)  # and the copies this machine made of the store's files
    fresh = open_store(store.root.as_uri()).at(store.session)

    assert Tokenizer.from_bytes(fresh.get_bytes(record["tokenizer"]["vocabulary"])).vocab_size > 0
    assert [entry["id"] for entry in document_list(fresh, dataset_id)]
    assert list(documents(fresh, record["shards"]))
    [line] = fresh.get_bytes(record["conversations"]["file"]).decode().splitlines()
    assert json.loads(line)["conversation"]

    resumed = StoreCheckpoint(fresh, "fox-1", Telemetry.off(), set())
    assert resumed.exists() and resumed.load()["step"] == 3
