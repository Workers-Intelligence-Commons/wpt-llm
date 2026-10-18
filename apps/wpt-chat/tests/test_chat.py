import json
import subprocess
import sys

from wpt_chat import main
import pytest
import torch

from wpt_inference import COSTS_FILE, costs_file, costs_setup
from wpt_store import LocalStore

from wpt_model import (WPT, Drafter, DrafterConfig, ModelConfig, load_drafter, load_model, save_drafter, save_model,
                       tokenizer_identity, weight_hashes)
from wpt_tokenizer import Tokenizer


def model_folder(folder, notice=None):
    tokenizer = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, 516)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=2, global_attention=["CSA"],
                         d_model=8, kda_heads=1, kda_head_dim=8, attention_heads=1, attention_head_dim=128,
                         query_rank=4, output_groups=1, output_rank=8, index_heads=1, index_head_dim=128, index_top_k=2, n_experts=2, top_k=1,
                         expert_width=4, latent_width=4, shared_width=8, engram_layers=[1], engram_order=2,
                         engram_heads=1, engram_dim=4, engram_table_size=31)
    save_model(WPT(config), folder, tokenizer.to_bytes(), "fox version 1")
    # an untrained drafter, named as trained against these weights: a model folder always holds one
    save_drafter(Drafter(config, DrafterConfig.for_model(config)), folder, *tokenizer_identity(folder),
                 provenance={"target": {"weights": weight_hashes(folder)}})
    if notice is not None:
        (folder / "NOTICE").write_text(notice)
    return folder


def test_the_start_line_names_the_licenses_and_where_the_notice_is(tmp_path, monkeypatch, capsys):
    folder = model_folder(tmp_path / "model", notice="the notice\n")
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # an empty line quits
    main([str(folder)])
    assert (f"generated text: CC0-1.0; weights: AGPL-3.0-or-later; sources and notices: {folder / 'NOTICE'} "
            "(wpt-chat --about)") in capsys.readouterr().out


def test_a_folder_whose_drafter_is_missing_or_of_other_weights_is_refused(tmp_path, monkeypatch):
    # a model and its drafter are one unit: no option skips the check
    folder = model_folder(tmp_path / "model")
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    provenance = folder / "drafter" / "provenance.json"
    provenance.write_text('{"target": {"weights": {"model.safetensors": "0"}}}')
    with pytest.raises(SystemExit, match="DrafterMismatch: .*its drafter was trained against weights"):
        main([str(folder)])
    provenance.unlink()
    with pytest.raises(SystemExit, match="DrafterMismatch: .*no drafter"):
        main([str(folder)])


def test_about_prints_the_folder_notice(tmp_path, capsys):
    folder = model_folder(tmp_path / "model", notice="WPT model notice\n\nSources...\n")
    main([str(folder), "--about"])
    assert capsys.readouterr().out == "WPT model notice\n\nSources...\n"


def test_a_folder_without_notice_still_names_the_licenses(tmp_path, monkeypatch, capsys):
    folder = model_folder(tmp_path / "model")
    main([str(folder), "--about"])
    about = capsys.readouterr().out
    assert "AGPL-3.0-or-later" in about and "CC0-1.0" in about and "https://github.com/Workers-Intelligence-Commons/wpt-llm" in about
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    main([str(folder)])
    assert "this folder has no NOTICE; notices and sources: https://github.com/Workers-Intelligence-Commons/wpt-llm" in capsys.readouterr().out


def test_notices_reach_a_pipe_as_utf8(tmp_path):
    # a NOTICE names authors in any script; sent to a file or another program
    # (a pipe), it must arrive whole, on Windows too
    folder = model_folder(tmp_path / "model")
    (folder / "NOTICE").write_text("Ada Writer \u2192 Fox Stories, \u00e9t\u00e9\n", encoding="utf-8")
    out = subprocess.run([sys.executable, "-c", "from wpt_chat import main; main()", str(folder), "--about"],
                         capture_output=True, check=True).stdout
    assert out.replace(b"\r\n", b"\n") == "Ada Writer \u2192 Fox Stories, \u00e9t\u00e9\n".encode()


def test_the_scheduler_s_times_can_be_read_from_a_run_s_record_on_the_same_setup(tmp_path, monkeypatch, capsys):
    # maintainer's decision, 2026-10-18: --scheduler-costs reads a record's
    # times instead of measuring them; another setup is refused
    folder = model_folder(tmp_path / "model")
    store = LocalStore(tmp_path / "store")
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.chdir(tmp_path)
    config = load_model(folder).config
    block = load_drafter(folder).block
    store.write_record("run", COSTS_FILE, costs_file([1.0] * (block + 1), costs_setup(config, 1, block, "cpu")))
    store.write_record("other", COSTS_FILE, costs_file([1.0] * 5, costs_setup(config, 1, block - 1, "cpu")))
    monkeypatch.setattr("wpt_chat.measure_costs", lambda *args: pytest.fail("the times are read, not measured"))
    monkeypatch.setattr("wpt_chat.select_device", lambda: torch.device("cpu"))
    answers = iter(["the", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    main([str(folder), "--scheduler-costs", "run", "--max-new-tokens", "3"])
    assert capsys.readouterr().out.splitlines()[-1].startswith("the")
    with pytest.raises(SystemExit, match=f"ValueError: .*another setup: draft {block - 1}, here {block}"):
        main([str(folder), "--scheduler-costs", "other"])
    with pytest.raises(SystemExit, match="MissingObject"):
        main([str(folder), "--scheduler-costs", "none"])


def test_a_chat_is_a_conversation_in_the_current_shape(tmp_path, monkeypatch):
    # what --chat holds is a chat-format conversation that the data rules accept as it is
    from wpt_dataset.repo import validate

    folder = model_folder(tmp_path / "model")
    typed = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    asked = []
    monkeypatch.setattr("wpt_chat.answer", lambda model, tokenizer, conversation, *rest: (
        asked.append(json.loads(json.dumps(conversation))) or ([{"type": "text", "text": "hi"}], "hi")))
    main([str(folder), "--chat"])
    validate(asked[0], "conversation", "wpt-chat --chat")
