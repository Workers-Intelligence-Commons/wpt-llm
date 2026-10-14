import torch

from wpt_model import ModelConfig, TinyGPT, load_model, save_model


def test_split_model_loads_back_identical(tmp_path):
    # split saving never runs at today's model sizes, so force it with a tiny limit
    model = TinyGPT(ModelConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))
    save_model(model, tmp_path, max_shard_bytes=20_000)

    assert len(list(tmp_path.glob("model-*-of-*.safetensors"))) > 1
    loaded = load_model(tmp_path)
    assert loaded.config == model.config
    original, restored = model.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[k], restored[k]) for k in original)
