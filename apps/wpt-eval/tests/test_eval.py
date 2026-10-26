import json
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import wpt_telemetry
from wpt_eval import main, open_context, run_eval
from wpt_store import LocalStore
from wpt_telemetry import FLUSH_THREAD
from wpt_tokenizer import Tokenizer
from wpt_worker.evaluation import Served

class Recall(torch.nn.Module):
    """A stand-in model that knows its training documents by heart: after
    any start of one of them it gives the next token, otherwise token 0."""

    def __init__(self, documents, vocab_size):
        super().__init__()
        self.config = SimpleNamespace(block_size=1000)
        self.weight = torch.nn.Parameter(torch.zeros(1))  # a device for the model
        self.vocab_size = vocab_size
        self.next = {tuple([Tokenizer.END, *d.tokens[:i].tolist()]): int(d.tokens[i])
                     for d in documents for i in range(len(d.tokens))}

    def forward(self, ids):
        logits = torch.zeros(ids.size(0), ids.size(1), self.vocab_size)
        for row, seen in enumerate(ids.tolist()):
            logits[row, -1, self.next.get(tuple(seen), 0)] = 1
        return logits


def test_memorization_counts_the_training_documents_a_model_reproduces(trained):
    # given a model that recalls every training document and no held-out one,
    # the test must report all training documents and no held-out document
    folder, sid = trained.folder, trained.source_id
    context = open_context(folder, trained.store)
    training, held_out = list(context.documents("training")), list(context.documents("held_out"))
    assert training and held_out
    context.model = Served(Recall(training, context.tokenizer.vocab_size))
    context.device = context.model.device

    data = run_eval(context, "memorization")["data"]
    assert data["training"]["exact"] == data["training"]["documents"] == len(training)
    assert data["training"]["exact_by_source"] == {sid: len(training)}
    assert data["held_out"]["exact"] == 0 and data["held_out"]["documents"] == len(held_out)


def test_a_folder_whose_drafter_was_trained_against_other_weights_is_refused(trained):
    # a model and its drafter are one unit (docs/model.md, "The drafter")
    provenance = trained.folder / "drafter" / "provenance.json"
    provenance.write_text(json.dumps({"target": {"weights": {"model.safetensors": "0" * 64}}}))
    with pytest.raises(SystemExit, match="EvalError: .*its drafter was trained against weights"):
        main([str(trained.folder)])
    assert not (trained.folder / "evals").exists()


def test_the_command_runs_every_built_in_test_into_the_model_card(trained, capsys):
    folder = trained.folder
    main([str(folder)])

    result = json.loads((folder / "evals" / "memorization.json").read_text())
    assert result["name"] == "memorization" and result["file"] == "wpt_eval/evals/memorization.py"
    counts = json.loads(trained.store.get_bytes(json.loads((folder / "provenance.json").read_text())["dataset_record"]))["documents"]
    assert result["data"]["held_out"]["documents"] == counts["held_out"]  # every one is long enough
    assert 0 < result["data"]["training"]["documents"] <= counts["training"]
    card = (folder / "README.md").read_text()
    assert "### memorization" in card and result["text"].rstrip() in card
    assert result["text"].rstrip() in capsys.readouterr().out


def test_any_python_file_can_be_a_test(trained, tmp_path):
    folder = trained.folder
    test = tmp_path / "count_tokens.py"
    test.write_text(
        "from wpt_eval import Result\n\n"
        "def run(context):\n"
        "    n = sum(len(d.tokens) for d in context.documents('held_out'))\n"
        "    return Result({'held_out_tokens': n}, f'{n} held-out tokens')\n")
    main([str(folder), str(test)])

    result = json.loads((folder / "evals" / "count_tokens.json").read_text())
    assert result["file"] == "count_tokens.py" and result["data"]["held_out_tokens"] > 0
    assert f"{result['data']['held_out_tokens']} held-out tokens" in (folder / "README.md").read_text()


def test_a_folder_holding_a_result_of_another_wpt_eval_version_is_evaluated_again(trained, tmp_path):
    # a result written by another version of wpt-eval stays in the model card beside the new ones
    folder = trained.folder
    (folder / "evals").mkdir()
    (folder / "evals" / "earlier.json").write_text(json.dumps({
        "name": "earlier", "file": "earlier.py", "file_sha256": "0" * 64, "wpt_eval": "0.1.0",
        "device": "cpu", "data": {}, "text": "an earlier result"}))
    test = tmp_path / "count_tokens.py"
    test.write_text("from wpt_eval import Result\n\ndef run(context):\n    return Result({}, 'a new result')\n")
    main([str(folder), str(test)])

    card = (folder / "README.md").read_text()
    assert "an earlier result" in card and "wpt-eval 0.1.0" in card and "a new result" in card
    assert "schema" not in json.loads((folder / "evals" / "count_tokens.json").read_text())


def test_a_test_that_returns_no_result_writes_nothing(trained, tmp_path):
    folder = trained.folder
    test = tmp_path / "broken.py"
    test.write_text("def run(context):\n    return 'all good'\n")
    with pytest.raises(SystemExit, match="returned str, not wpt_eval.Result"):
        main([str(folder), str(test)])
    assert not (folder / "evals" / "broken.json").exists()


def test_a_record_the_store_refuses_stops_the_evaluation_with_code_3(trained, monkeypatch, capsys):
    # every command stops without its record, with the one exit code
    # that says so, before writing the result it could not record
    real = LocalStore._raw_append_record

    def disk_full_at_the_result(self, folder, name, data):
        if b'"name": "eval.result"' in data:
            raise OSError(28, "No space left on device")
        real(self, folder, name, data)

    monkeypatch.setattr(LocalStore, "_raw_append_record", disk_full_at_the_result)
    with pytest.raises(SystemExit) as stopped:
        main([str(trained.folder)])
    assert stopped.value.code == 3 and "No space left on device" in capsys.readouterr().err
    assert not (trained.folder / "evals").exists()


@pytest.mark.parametrize("refused, result_written", [(b'"eval_constant"', False), (b'"model_card"', True)])
def test_on_s3_a_result_and_the_model_card_are_written_only_once_their_record_is_in_the_bucket(
        trained, s3, tmp_path, refused, result_written):
    # S3 holds record lines back: each file's sha256 must be in the bucket
    # before the file is written, so a refused send leaves the earlier files
    test = tmp_path / "constant.py"
    test.write_text("from wpt_eval import Result\n\ndef run(context):\n    return Result(data={'x': 1}, text='x is 1')\n")
    before = {path.name: path.read_bytes() for path in trained.folder.iterdir() if path.is_file()}
    s3.refuse(refused)
    with pytest.raises(SystemExit) as stopped:
        main([str(trained.folder), str(test)])
    assert stopped.value.code == 3
    assert {path.name: path.read_bytes() for path in trained.folder.iterdir() if path.is_file()} == before
    assert (trained.folder / "evals" / "constant.json").exists() == result_written


def test_a_send_refused_in_the_background_stops_the_evaluation_before_its_next_record_call(trained, tmp_path,
                                                                                           monkeypatch, capsys):
    # the command must stop when the send fails, not at its next record call,
    # and exit as a record failure (3), not as a stop asked for (130)
    real = LocalStore._raw_flush_records

    def refused_in_the_background(self):
        if threading.current_thread().name == FLUSH_THREAD:
            raise OSError(28, "No space left on device")
        real(self)

    monkeypatch.setattr(wpt_telemetry, "FLUSH_SECONDS", 0.02)
    monkeypatch.setattr(LocalStore, "_raw_flush_records", refused_in_the_background)
    test = tmp_path / "long.py"  # works for 30 seconds without recording anything
    test.write_text("import time\nfrom wpt_eval import Result\n\ndef run(context):\n"
                    "    started = time.monotonic()\n    while time.monotonic() - started < 30:\n        pass\n"
                    "    return Result(data={}, text='')\n")
    started = time.monotonic()
    with pytest.raises(SystemExit) as stopped:
        main([str(trained.folder), str(test)])
    assert stopped.value.code == 3 and time.monotonic() - started < 15
    assert "No space left on device" in capsys.readouterr().err
    assert not (trained.folder / "evals").exists()
