from wpt_chat import main
from wpt_model import ModelConfig, TinyGPT, save_model
from wpt_tokenizer import Tokenizer


def model_folder(folder, notice=None):
    tokenizer = Tokenizer.train([b"the quick brown fox jumps over the lazy dog"] * 4, 260)
    save_model(TinyGPT(ModelConfig(vocab_size=tokenizer.vocab_size, block_size=8, n_layer=1, n_head=1, n_embd=8)),
               folder)
    tokenizer.save(folder / "tokenizer.json")
    if notice is not None:
        (folder / "NOTICE").write_text(notice)
    return folder


def test_the_start_line_names_the_licenses_and_where_the_notice_is(tmp_path, monkeypatch, capsys):
    folder = model_folder(tmp_path / "model", notice="the notice\n")
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # an empty line quits
    main([str(folder)])
    assert (f"generated text: CC0-1.0; weights: AGPL-3.0-or-later; sources and notices: {folder / 'NOTICE'} "
            "(wpt-chat --about)") in capsys.readouterr().out


def test_about_prints_the_folder_notice(tmp_path, capsys):
    folder = model_folder(tmp_path / "model", notice="WPT model notice\n\nSources...\n")
    main([str(folder), "--about"])
    assert capsys.readouterr().out == "WPT model notice\n\nSources...\n"


def test_a_folder_without_notice_still_names_the_licenses(tmp_path, monkeypatch, capsys):
    folder = model_folder(tmp_path / "model")
    main([str(folder), "--about"])
    about = capsys.readouterr().out
    assert "AGPL-3.0-or-later" in about and "CC0-1.0" in about and "https://github.com/inayayousfi/wpt-llm" in about
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    main([str(folder)])
    assert "this folder has no NOTICE; notices and sources: https://github.com/inayayousfi/wpt-llm" in capsys.readouterr().out
