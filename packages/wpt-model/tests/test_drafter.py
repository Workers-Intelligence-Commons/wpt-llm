import hashlib
import json

import pytest
import torch
from safetensors import safe_open

from wpt_model import (WPT, Drafter, DrafterConfig, DrafterMismatch, ModelConfig, TokenizerMismatch, load_drafter,
                       save_drafter, save_model, weight_hashes)
from wpt_model.drafter import default_markov_rank, feature_layers
from wpt_model.notice import WEIGHT_METADATA, documents
from wpt_utils.device import select_device

VOCABULARY = b'{"format":"wpt-bpe","merges":[],"tokenizer_version":"0.2.0"}\n'
TINY = dict(vocab_size=50, block_size=16, n_layer=4, global_attention=["CSA"], d_model=16, kda_heads=2, kda_head_dim=8,
            attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1, output_rank=8, index_heads=2,
            index_head_dim=128, index_top_k=2, n_experts=4, top_k=2, expert_width=8, latent_width=8, shared_width=16,
            engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31)


def test_a_drafter_saved_with_its_model_folder_loads_back_identical(tmp_path):
    # its weights, its calibrated temperatures and routing biases, its
    # provenance; its weight file carries the model's licenses and
    # vocabulary, and a folder without one is refused
    config = ModelConfig(**TINY)
    save_model(WPT(config), tmp_path, VOCABULARY, "test version 1", metadata=WEIGHT_METADATA)
    with pytest.raises(DrafterMismatch, match="no drafter"):
        load_drafter(tmp_path)
    drafter = Drafter(config, DrafterConfig.for_model(config))
    drafter.temperatures.copy_(torch.tensor([0.5, 0.8, 1.0, 1.3, 2.0]))
    drafter.layers[1].experts.bias.fill_(0.25)
    provenance = {"schema": 2, "acceptance": {"verified": 3}, "target": {"weights": weight_hashes(tmp_path)}}
    save_drafter(drafter, tmp_path, "test version 1", hashlib.sha256(VOCABULARY).hexdigest(), metadata=WEIGHT_METADATA,
                 provenance=provenance)

    loaded = load_drafter(tmp_path)
    assert loaded.config == drafter.config and loaded.provenance == provenance
    original, restored = drafter.state_dict(), loaded.state_dict()
    assert original.keys() == restored.keys()
    assert all(torch.equal(original[k], restored[k]) for k in original)
    with safe_open(tmp_path / "drafter" / "model.safetensors", framework="pt") as weights:
        header = weights.metadata()
    assert header["license"] == WEIGHT_METADATA["license"] and header["tokenizer"] == "test version 1"


def test_a_drafter_of_another_vocabulary_is_refused(tmp_path):
    # its token ids would name other pieces of text than the model's
    config = ModelConfig(**TINY)
    save_model(WPT(config), tmp_path, VOCABULARY, "test version 1")
    save_drafter(Drafter(config, DrafterConfig.for_model(config)), tmp_path, "test version 2", "0" * 64,
                 provenance={"target": {"weights": weight_hashes(tmp_path)}})
    with pytest.raises(TokenizerMismatch, match="saved with tokenizer test version 2"):
        load_drafter(tmp_path)


def test_the_model_card_names_the_drafter_and_its_notice_adds_no_source(tmp_path):
    # the drafter's prompts are the beginnings of the dataset's training
    # documents: the NOTICE credits the model's sources alone, and the card
    # says what made the drafter and against which weights
    config = ModelConfig(**TINY)
    save_model(WPT(config), tmp_path, VOCABULARY, "test version 1")
    source = {"address": "https://example.org/talks", "revision": "abc",
              "licenses": [{"license": "CC0-1.0", "paths": ["**"], "attribution": "Talks, by Bo Speaker"}]}
    provenance = {"schema": 4, "dataset": "d" * 64, "parameters": 1, "recipe": {"training": {"steps": 1}},
                  "batch": {"windows": 1}, "method": {"name": "muon", "settings": {}}, "best_validation_loss": 1.0,
                  "device": "cpu", "sources": {}}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    provenance["sources"] = {"example-talks": source}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    save_drafter(Drafter(config, DrafterConfig.for_model(config)), tmp_path, "test version 1", "0" * 64, provenance={
        "trained_by": {"command": "wpt-posttrain", "recipe_name": "chat"}, "dataset": "d" * 64,
        "target": {"weights": {"model.safetensors": "e" * 64}},
        "acceptance": {"verified": 10, "accepted": 4, "rounds": 6, "tokens_per_round": 10 / 6},
        "sources": {"example-other": {**source, "address": "https://example.org/other"}}})  # never read
    texts = documents(tmp_path)
    assert "example-talks" in texts["NOTICE"] and "example-other" not in texts["NOTICE"]
    card = texts["README.md"]
    assert "## Drafter" in card and "`wpt-posttrain`, recipe `chat`" in card and "`dddddddddddd`" in card
    assert "`model.safetensors` (sha256 `eeeeeeeeeeee`)" in card and "4 of 10 drafts kept" in card
    assert "example-other" not in card


def test_the_markov_head_s_rank_follows_the_model_s_width():
    # 256 x width / 4096, DSpark's 256 at
    # a width of 4096; 8, 20 and 32 at the widths of 128, 320 and 512
    assert [default_markov_rank(width) for width in (128, 320, 512, 4096)] == [8, 20, 32, 256]
    assert default_markov_rank(8) == 1  # never 0
    assert DrafterConfig.for_model(ModelConfig(**TINY)).markov_rank == 1
    assert DrafterConfig.for_model(ModelConfig(**TINY), markov_rank=7).markov_rank == 7


def test_the_drafter_reads_layers_from_the_second_to_the_third_to_last():
    # DFlash's choice: 5 layers spread evenly, fewer in a short model
    assert feature_layers(4) == (2,)
    assert feature_layers(8) == (2, 3, 4, 5, 6)


@pytest.mark.skipif(select_device().type != "cuda", reason="needs a GPU")
def test_on_a_gpu_the_drafter_scores_as_on_the_cpu():
    # through the GPU's attention kernels in BF16, within BF16's rounding:
    # blocks at the start of a text and past the window
    torch.manual_seed(0)
    config = ModelConfig(**TINY)
    model, drafter = WPT(config).eval(), Drafter(config, DrafterConfig.for_model(config)).eval()
    context = torch.randn(3, 128, len(drafter.features) * config.d_model)
    context[0, :125] = 0  # 3 tokens read, zeros before the text
    lengths, anchors = torch.tensor([3, 200, 1000]), torch.tensor([4, 7, 9])
    with torch.no_grad():
        _, expected = drafter.backbone(model, context, lengths, anchors, drafter.block)
        gpu = select_device()
        _, found = drafter.to(gpu).backbone(model.to(gpu), context.to(gpu), lengths.to(gpu), anchors.to(gpu),
                                            drafter.block)
    assert (found.cpu() - expected).abs().max() <= 3e-2 * expected.abs().max()
