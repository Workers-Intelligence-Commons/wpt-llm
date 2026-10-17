import pytest
import torch

from wpt_model import WPT, ModelConfig
from wpt_model.experts import Experts
from wpt_model.kda import _fla_chunk_kda, chunked_kda, recurrent_kda
from wpt_utils.device import select_device

TINY = dict(vocab_size=50, block_size=16, n_layer=4, d_model=16, kda_heads=2, kda_head_dim=8, mla_heads=2,
            mla_head_dim=8, mla_kv_rank=8, n_experts=4, top_k=2, expert_width=8, latent_width=8, shared_width=16,
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
    # a language model must not see ahead: changing token 9 leaves every
    # score at positions 0 .. 8 as it was, through KDA, MLA, Engram, the
    # experts and the streams
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY)).eval()
    ids = torch.randint(50, (2, 16))
    changed = ids.clone()
    changed[:, 9] = (ids[:, 9] + 1) % 50
    with torch.no_grad():
        before, after = model(ids), model(changed)
    assert torch.allclose(before[:, :9], after[:, :9], atol=1e-5)
    assert not torch.allclose(before[:, 9], after[:, 9])


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


def test_recomputing_layers_trains_the_same_and_counts_each_token_once():
    # --recompute only trades time for memory: the same gradients, and the
    # layers' second run adds nothing to balancing or query-key records
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY)).train()
    model.track_query_keys(True)
    ids, after = torch.randint(50, (2, 16)), torch.randint(50, (2, 16))
    results = []
    for recompute in (False, True):
        model.recompute = recompute
        model.zero_grad()
        logits, ahead = model(ids, next_tokens=after)
        (logits.logsumexp(-1).mean() + ahead.logsumexp(-1).mean()).backward()
        experts, attention = model.layers[3].experts, model.layers[3].attention
        results.append(([p.grad.clone() for p in model.parameters()], experts.histogram.clone(),
                        attention.score_max.clone()))
        experts.histogram, attention.score_max = None, None
    (grads, counts, scores), (grads_again, counts_again, scores_again) = results
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(grads, grads_again))
    assert torch.equal(counts, counts_again) and torch.equal(scores, scores_again)
