import math

import torch

from wpt_model import WPT, ModelConfig
from wpt_training import Run, load_method
from wpt_training.plugins import start_method

TINY = dict(vocab_size=50, block_size=16, n_layer=4, d_model=16, kda_heads=2, kda_head_dim=8, mla_heads=2,
            mla_head_dim=8, mla_kv_rank=8, n_experts=4, top_k=2, expert_width=8, latent_width=8, shared_width=16,
            engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31)


def muon(model, steps=100, **settings):
    plugin, values = load_method("muon", settings)
    return start_method(plugin, model, values, Run(tokens=1000, parameters=1, steps=steps, log=lambda *_: None))


def test_the_learning_rate_warms_up_then_drops_twice():
    # docs/model.md: linear warm-up, then 31.6% of the maximum after 80% of
    # the steps and 10% after 90%; Engram tables at 5 times the rate
    method = muon(WPT(ModelConfig(**TINY)), steps=100, lr=0.01, warmup=10)
    seen = {}
    for step in (5, 50, 85, 95):
        method.before_update(None, step)
        seen[step] = sorted({round(g["lr"], 6) for g in method.optimizer.param_groups})
    assert seen == {5: [0.005, 0.025], 50: [0.01, 0.05], 85: [0.00316, 0.0158], 95: [0.001, 0.005]}


def test_each_head_moves_as_much_whatever_its_gradient_size():
    # per-head Muon: a head with tiny gradients is not swamped by a head with
    # large ones; each head's step has the size of an orthogonal matrix's
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY))
    method = muon(model, lr=0.01, warmup=1, weight_decay=0.0)
    weight = model.layers[0].attention.q.weight  # KDA, 2 heads of 8 rows
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    weight.grad = torch.randn_like(weight)
    weight.grad[8:] *= 1e-4
    before = weight.detach().clone()
    method.before_update(model, 1)
    method.optimizer.step()
    step = (weight.detach() - before) / (0.01 * 0.2 * math.sqrt(16))
    for head in (step[:8], step[8:]):
        singular = torch.linalg.svdvals(head)
        assert singular.min() > 0.5 and singular.max() < 1.5


def test_qk_clip_brings_a_head_back_to_the_limit():
    # a head whose largest query-key score passed tau has its query and key
    # weights scaled so the score becomes tau; the other head is left alone
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY)).train()
    attention = model.layers[3].attention  # the last layer: MLA
    with torch.no_grad():
        attention.q.weight[:8] *= 40
    ids = torch.randint(50, (2, 16))

    def largest():
        model.track_query_keys(True)
        with torch.no_grad():
            model(ids)
        found = attention.score_max.clone()
        attention.score_max = None
        return found

    high = largest()
    assert high[0] > 10 > high[1]
    untouched = attention.q.weight[8:].clone()
    model(ids)
    model.clip_query_keys(10.0, lambda tensor, op: tensor)
    after = largest()
    assert math.isclose(after[0].item(), 10.0, rel_tol=1e-3)
    assert torch.equal(attention.q.weight[8:], untouched)
