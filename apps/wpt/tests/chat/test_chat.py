import json
import subprocess
import sys

from wpt.chat import main
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
            "(wpt chat --about)") in capsys.readouterr().out


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
    assert "AGPL-3.0-or-later" in about and "CC0-1.0" in about and "https://github.com/Workers-Intelligence-Commons/wpt" in about
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    main([str(folder)])
    assert "this folder has no NOTICE; notices and sources: https://github.com/Workers-Intelligence-Commons/wpt" in capsys.readouterr().out


def test_notices_reach_a_pipe_as_utf8(tmp_path):
    # a NOTICE names authors in any script; sent to a file or another program
    # (a pipe), it must arrive whole, on Windows too
    folder = model_folder(tmp_path / "model")
    (folder / "NOTICE").write_text("Ada Writer \u2192 Fox Stories, \u00e9t\u00e9\n", encoding="utf-8")
    out = subprocess.run([sys.executable, "-c", "from wpt.chat import main; main()", str(folder), "--about"],
                         capture_output=True, check=True).stdout
    assert out.replace(b"\r\n", b"\n") == "Ada Writer \u2192 Fox Stories, \u00e9t\u00e9\n".encode()


def test_the_scheduler_s_times_can_be_read_from_a_run_s_record_on_the_same_setup(tmp_path, monkeypatch, capsys):
    # --scheduler-costs reads a record's
    # times instead of measuring them; another setup is refused
    folder = model_folder(tmp_path / "model")
    store = LocalStore(tmp_path / "store").at("test-session")
    monkeypatch.setenv("WPT_STORE", store.root.as_uri())
    monkeypatch.setenv("WPT_SESSION", store.session)
    monkeypatch.chdir(tmp_path)
    config = load_model(folder).config
    block = load_drafter(folder).block
    store.write_record("test-session/run", COSTS_FILE, costs_file([1.0] * (block + 1), costs_setup(config, 1, block, "cpu")))
    store.write_record("test-session/other", COSTS_FILE, costs_file([1.0] * 5, costs_setup(config, 1, block - 1, "cpu")))
    monkeypatch.setattr("wpt.chat.measure_costs", lambda *args: pytest.fail("the times are read, not measured"))
    monkeypatch.setattr("wpt.chat.select_device", lambda: torch.device("cpu"))
    answers = iter(["the", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    main([str(folder), "--scheduler-costs", "test-session/run", "--max-new-tokens", "3"])
    # the 3 tokens are random and may hold line breaks, so "the" starts a line
    # of the output but that line is not always the last one
    assert "\nthe" in capsys.readouterr().out
    with pytest.raises(SystemExit, match=f"ValueError: .*another setup: draft {block - 1}, here {block}"):
        main([str(folder), "--scheduler-costs", "test-session/other"])
    with pytest.raises(SystemExit, match="MissingFile"):
        main([str(folder), "--scheduler-costs", "test-session/none"])
    monkeypatch.delenv("WPT_SESSION")
    with pytest.raises(SystemExit, match="SessionNeeded"):
        main([str(folder), "--scheduler-costs", "test-session/run"])


def test_a_chat_is_a_conversation_in_the_current_shape(tmp_path, monkeypatch):
    # what --chat holds is a chat-format conversation that the data rules accept as it is
    from wpt_dataset.repo import validate

    folder = model_folder(tmp_path / "model")
    typed = iter(["hello", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed))
    asked = []
    monkeypatch.setattr("wpt.chat.answer", lambda model, tokenizer, conversation, *rest: (
        asked.append(json.loads(json.dumps(conversation))) or ([{"type": "text", "text": "hi"}], "hi")))
    main([str(folder), "--chat"])
    validate(asked[0], "conversation", "wpt chat --chat")


def test_the_kernels_named_reach_the_attention_before_the_model_loads_and_one_that_cannot_work_stops_the_command(
        tmp_path, monkeypatch, capsys):
    # docs/model.md, "Attention": --kernels and WPT_KERNELS name the form
    # for the device; a form that cannot work stops the command
    import wpt.chat

    folder = model_folder(tmp_path / "model")
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    asked = []
    real = wpt.chat.kernels

    def watching(device, form):
        asked.append(form)
        return real(device, form)

    monkeypatch.setattr(wpt.chat, "kernels", watching)
    monkeypatch.setattr(wpt.chat, "select_device", lambda: torch.device("cpu"))
    monkeypatch.delenv("WPT_KERNELS", raising=False)
    main([str(folder)])
    assert asked == []  # unnamed on the CPU: nothing to decide

    monkeypatch.setenv("WPT_KERNELS", "triton")  # the environment names it; here no GPU can run it
    with pytest.raises(SystemExit) as stopped:
        main([str(folder)])
    assert asked == ["triton"] and "KernelUnavailable: Triton kernels run on a GPU" in str(stopped.value)
    monkeypatch.delenv("WPT_KERNELS")
    with pytest.raises(SystemExit) as stopped:
        main([str(folder), "--kernels", "tilelang"])
    assert asked == ["triton", "tilelang"] and "KernelUnavailable: TileLang kernels run on a GPU" in str(stopped.value)
    main([str(folder), "--kernels", "pytorch"])  # the PyTorch form works anywhere
    assert asked == ["triton", "tilelang", "pytorch"]

    monkeypatch.setenv("WPT_KERNELS", "fast")
    with pytest.raises(SystemExit) as refused:
        main([str(folder)])
    assert refused.value.code == 2 and "WPT_KERNELS='fast': not one of tilelang, triton, pytorch" in \
        capsys.readouterr().err
