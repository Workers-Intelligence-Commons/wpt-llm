import pytest
import torch

from wpt_model import WPT, ModelConfig, Routes
from wpt_model.attention import CSA_RATIO, HCA_RATIO, Attention
from wpt_model.experts import Experts
from wpt_model.kda import _fla_chunk_kda, chunked_kda, recurrent_kda
from wpt_utils.device import select_device

TINY = dict(vocab_size=50, block_size=16, n_layer=4, global_attention=["CSA"], d_model=16, kda_heads=2, kda_head_dim=8,
            attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1, output_rank=8, index_heads=2,
            index_head_dim=128, index_top_k=2, n_experts=4, top_k=2, expert_width=8, latent_width=8, shared_width=16,
            engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31)


def kda_inputs(B=2, T=37, H=3, d=8, device="cpu", dtype=torch.float32):
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).to(device, dtype)

    q = torch.nn.functional.normalize(rand(B, T, H, d), dim=-1)
    k = torch.nn.functional.normalize(rand(B, T, H, d), dim=-1)
    g = (-5 * torch.sigmoid(rand(B, T, H, d).float() * 3)).to(device)  # decays down to the bound
    return q, k, rand(B, T, H, d), g, torch.sigmoid(rand(B, T, H))


def test_the_chunked_kda_gives_the_formula_s_result():
    # docs/model.md: the step-by-step formula is the reference, and the
    # faster form must agree with it, over several chunks and a partial one
    q, k, v, g, beta = kda_inputs()
    assert torch.allclose(chunked_kda(q, k, v, g, beta), recurrent_kda(q, k, v, g, beta), atol=1e-4)


@pytest.mark.skipif(select_device().type != "cuda" or not _fla_chunk_kda(),  # select_device: AMD GPUs need it
                    reason="needs a GPU where flash-linear-attention imports")
def test_the_gpu_kernel_gives_the_formula_s_result():
    q, k, v, g, beta = kda_inputs(T=300, H=2, d=64, device=select_device(), dtype=torch.bfloat16)
    out, _ = _fla_chunk_kda()(q, k, v, g, beta, scale=1.0, safe_gate=True, lower_bound=-5.0)
    assert torch.allclose(out.float(), recurrent_kda(q, k, v, g, beta), atol=3e-2)


def test_a_token_never_changes_the_scores_before_it():
    # a language model must not see ahead: changing token 150 leaves every
    # score at positions 0 .. 149 as it was, through KDA, CSA (its merged
    # entries, among them the one of tokens 148 .. 151, and its indexer's
    # choice), HCA, Engram, the experts and the streams; 300 tokens let the
    # window slide and every kind of entry merge
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "n_layer": 8, "global_attention": ["CSA", "HCA"], "block_size": 300})).eval()
    ids = torch.randint(50, (2, 300))
    changed = ids.clone()
    changed[:, 150] = (ids[:, 150] + 1) % 50
    with torch.no_grad():
        before, after = model(ids), model(changed)
    assert torch.allclose(before[:, :150], after[:, :150], atol=1e-5)
    assert not torch.allclose(before[:, 150], after[:, 150])


def test_quantile_balancing_gives_each_expert_its_share():
    # a router that favours some experts ends up sending each one about k/n
    # of the tokens, with no auxiliary loss
    torch.manual_seed(0)
    experts = Experts(ModelConfig(**{**TINY, "n_experts": 8}), hashed=False)
    with torch.no_grad():
        experts.router.weight[:3] += 0.5  # with inputs of positive mean, experts 0, 1 and 2 score higher
    x, ids = torch.randn(4, 256, 16) + 0.5, torch.zeros(4, 256, dtype=torch.long)

    def shares():
        chosen, _ = experts.route(x.reshape(-1, 16), ids.reshape(-1))
        return torch.bincount(chosen.reshape(-1), minlength=8) / chosen.numel()

    assert shares().max() > 0.3
    experts.train()
    for _ in range(40):  # the same batch each step: the biases settle
        experts(x, ids)
        experts.balance(lambda tensor, op: tensor)
    assert torch.allclose(shares(), torch.full((8,), 1 / 8), atol=0.02)


def test_a_replayed_pass_routes_each_token_through_the_experts_recorded_while_writing_it():
    # R3 (docs/post-training.md): routes recorded by growing prefixes, as
    # sampling reads them, route a later pass over the whole sequence the
    # same way, even when the routers would now choose other experts
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY)).eval()
    ids = torch.randint(50, (1, 12))
    routes = Routes()
    with torch.no_grad():
        for end in range(5, 13):  # a prompt of 5 tokens, then one more token at a time
            model(ids[:, :end], record=routes)
        written = model(ids)
        routed = [layer.experts for layer in model.layers if not layer.experts.hashed]
        for experts in routed:
            experts.bias.copy_(torch.tensor([0.0, 0.0, 10.0, 10.0]))  # experts 2 and 3 for every token now
        assert not torch.allclose(model(ids), written, atol=1e-4)
        assert torch.allclose(model(ids, replay=routes), written, atol=1e-5)
    assert [len(routes.chosen[experts]) for experts in routed] == [12]


def test_recomputing_layers_trains_the_same_and_counts_each_token_once():
    # --recompute only trades time for memory: the same gradients (the
    # indexers' too), the indexers' loss counted once, and the layers'
    # second run adds nothing to balancing
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 64})).train()
    ids, after = torch.randint(50, (2, 64)), torch.randint(50, (2, 64))
    results = []
    for recompute in (False, True):
        model.recompute = recompute
        model.zero_grad()
        logits, ahead, indexers = model(ids, next_tokens=after, indexer_loss=True)
        (logits.logsumexp(-1).mean() + ahead.logsumexp(-1).mean() + indexers).backward()
        experts = model.layers[3].experts
        results.append(([p.grad.clone() for p in model.parameters()], experts.histogram.clone(), indexers.detach()))
        experts.histogram = None
    (grads, counts, loss), (grads_again, counts_again, loss_again) = results
    assert loss > 0 and torch.allclose(loss, loss_again)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(grads, grads_again))
    assert torch.equal(counts, counts_again)


def test_the_recipe_names_the_kind_of_each_global_layer():
    # docs/model.md: every 4th layer and the last is global, CSA or HCA as
    # the recipe's list says, in order; a list that does not match is refused
    model = WPT(ModelConfig(**{**TINY, "n_layer": 9, "global_attention": ["HCA", "CSA", "CSA"]}))
    ratios = {n: layer.attention.ratio for n, layer in enumerate(model.layers, 1)
              if isinstance(layer.attention, Attention)}
    assert ratios == {4: HCA_RATIO, 8: CSA_RATIO, 9: CSA_RATIO}
    with pytest.raises(ValueError, match="global_attention"):
        ModelConfig(**{**TINY, "n_layer": 8, "global_attention": ["CSA"]})


def test_the_indexers_learn_only_when_the_caller_asks_for_their_loss():
    # the indexer's loss is returned by the pass that asks for it, and is
    # the only thing that trains the indexers; a pass that does not ask
    # returns the scores alone and leaves the indexers without a gradient
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "block_size": 64})).train()
    ids, after = torch.randint(50, (2, 64)), torch.randint(50, (2, 64))
    indexer = model.layers[3].attention.indexer
    logits, ahead = model(ids, next_tokens=after)
    (logits.logsumexp(-1).mean() + ahead.logsumexp(-1).mean()).backward()
    assert all(p.grad is None for p in indexer.parameters())

    logits, indexers = model(ids, indexer_loss=True)
    assert logits.shape == (2, 64, 50) and indexers > 0
    indexers.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in indexer.parameters())
