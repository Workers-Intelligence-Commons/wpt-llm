import hashlib
from pathlib import Path

import pytest
import torch

from wpt_model import ModelConfig, TinyGPT, TokenizerMismatch, load_model, load_tokenizer, save_model, tokenizer_identity
from wpt_model.notice import license_text, output_notice

VOCABULARY = b'{"format":"wpt-bpe","merges":[],"tokenizer_version":"0.2.0"}\n'


def test_split_model_loads_back_identical(tmp_path):
    # split saving never runs at today's model sizes, so force it with a tiny limit
    model = TinyGPT(ModelConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))
    save_model(model, tmp_path, VOCABULARY, "test version 1", max_shard_bytes=20_000)

    assert len(list(tmp_path.glob("model-*-of-*.safetensors"))) > 1
    loaded = load_model(tmp_path)
    assert loaded.config == model.config
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[k], restored[k]) for k in original)
    assert tokenizer_identity(tmp_path) == ("test version 1", hashlib.sha256(VOCABULARY).hexdigest())  # in every shard


def test_a_model_runs_only_with_the_vocabulary_its_weights_name(tmp_path):
    # another vocabulary would turn every token id into another piece of text
    save_model(TinyGPT(ModelConfig(vocab_size=50, block_size=16, n_layer=1, n_head=1, n_embd=8)), tmp_path,
               VOCABULARY, "test version 1")
    assert load_tokenizer(tmp_path) == VOCABULARY
    (tmp_path / "tokenizer.json").write_bytes(VOCABULARY.replace(b"0.2.0", b"0.3.0"))
    with pytest.raises(TokenizerMismatch, match="trained with tokenizer test version 1"):
        load_tokenizer(tmp_path)


def test_the_package_copies_of_the_license_files_match_the_repository():
    # model folders carry these copies; they must say what the repository says
    root = Path(__file__).resolve().parents[3]
    assert output_notice() == (root / "OUTPUT-NOTICE").read_text(encoding="utf-8")
    assert license_text() == (root / "LICENSE").read_text(encoding="utf-8")
