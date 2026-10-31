import json
import math
from pathlib import Path

import pytest
import torch

from wpt_model import PARTS, WPT, ModelConfig
from wpt_training import PluginError, Run, load_step

TINY = dict(vocab_size=50, block_size=16, n_layer=4, global_attention=["CSA"], d_model=16, kda_heads=2, kda_head_dim=8,
            attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1, output_rank=8, index_heads=2,
            index_head_dim=128, index_top_k=2, n_experts=4, top_k=2, expert_width=8, latent_width=8, shared_width=16,
            engram_layers=[2], engram_order=3, engram_heads=2, engram_dim=4, engram_table_size=31)


def read_text(**settings):
    """A read-text step under by-part with these settings (docs/training.md, "Steps")."""
    return load_step({"step": "read-text", "optimizer": {"name": "by-part", "settings": settings},
                      "settings": {"stages": [{"block_size": 128, "steps": 1, "batch_size": 4}]}}, PARTS)


def by_part(model, steps=100, **settings):
    return read_text(**settings).start_optimizer(model, Run(tokens=1000, parameters=1, steps=steps,
                                                            log=lambda *_: None))


def test_the_learning_rate_warms_up_then_drops_twice():
    # docs/model.md: linear warm-up, then 31.6% of the maximum after 80% of
    # the steps and 10% after 90%; Engram tables at 5 times the rate
    method = by_part(WPT(ModelConfig(**TINY)), steps=100, lr=0.01, warmup=10)
    seen = {}
    for step in (5, 50, 85, 95):
        method.before_update(step)
        seen[step] = sorted({round(g["lr"], 6) for g in method.optimizer.param_groups})
    assert seen == {5: [0.005, 0.025], 50: [0.01, 0.05], 85: [0.00316, 0.0158], 95: [0.001, 0.005]}


def test_a_constant_schedule_keeps_the_learning_rate_at_every_update():
    # docs/training.md, "Optimizers": no warm-up and no drops; Engram tables
    # still at 5 times the rate
    method = by_part(WPT(ModelConfig(**TINY)), steps=100, lr=0.01, schedule="constant")
    for step in (1, 50, 95):
        method.before_update(step)
        assert sorted({round(g["lr"], 6) for g in method.optimizer.param_groups}) == [0.01, 0.05]


def test_early_stop_is_off_unless_patience_turns_it_on():
    # docs/training.md, "Optimizers": without patience every new lowest loss
    # is the best and nothing stops; with it, a stage stops once the
    # smoothed loss has not improved by min_improve for patience
    # evaluations, and an improvement is the best
    losses = [3.0, 2.0, 2.0, 2.0, 2.0, 1.9]
    plain = by_part(WPT(ModelConfig(**TINY)))
    assert [(v.best, v.stop) for v in (plain.evaluated(i, loss) for i, loss in enumerate(losses))] == [
        (True, False), (True, False), (False, False), (False, False), (False, False), (True, False)]
    stopping = by_part(WPT(ModelConfig(**TINY)), patience=2, smoothing=1)  # 1000 tokens per parameter: 1%
    verdicts = [stopping.evaluated(i, loss) for i, loss in enumerate(losses[:4])]
    assert [(v.best, v.stop) for v in verdicts] == [(True, False), (True, False), (False, False), (False, True)]
    # a stage after the first starts counting again
    stopping.stage(1, 32)
    assert stopping.evaluated(5, 2.5).best
    with pytest.raises(PluginError, match=r"schedule 'cosine': one of steps, constant"):
        read_text(schedule="cosine")


@pytest.mark.parametrize("weight", [
    lambda m: m.layers[0].attention.q.weight,                     # KDA queries, 2 heads
    lambda m: m.layers[3].attention.q_up.weight,                  # CSA queries, 2 heads
    lambda m: m.layers[3].attention.indexer.q.weight,             # indexer queries, 2 heads
    lambda m: m.layers[3].attention.out_down.weight,              # 2 output groups
    lambda m: m.layers[3].attention.compressor.value.weight,      # C^b and C^a
    lambda m: m.layers[3].attention.indexer.keys.score.weight,    # the indexer's Z^b and Z^a
])
def test_each_head_moves_as_much_whatever_its_gradient_size(weight):
    # per-head Muon (K3, DeepSeek-V4.1), and V4's "logically independent"
    # matrices apart: a part with tiny gradients is not swamped by a part
    # with large ones; each part's step is an orthogonal matrix, whose
    # singular values V4's last two Newton-Schulz steps settle at 1
    torch.manual_seed(0)
    model = WPT(ModelConfig(**{**TINY, "output_groups": 2}))
    method = by_part(model, lr=0.01, warmup=1, weight_decay=0.0)
    weight = weight(model)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    rows = weight.shape[0] // 2
    weight.grad = torch.randn_like(weight)
    weight.grad[rows:] *= 1e-4
    before = weight.detach().clone()
    method.before_update(1)
    method.optimizer.step()
    step = (weight.detach() - before) / (0.01 * 0.18 * math.sqrt(max(rows, weight.shape[1])))
    for part in (step[:rows], step[rows:]):
        singular = torch.linalg.svdvals(part)
        assert singular.min() > 0.98 and singular.max() < 1.02


@pytest.mark.parametrize("shape", [(64, 256), (256, 64), (96, 96)])
def test_a_matrix_s_update_has_an_rms_of_0_18(shape):
    # DeepSeek-V4: O = O' sqrt(max(n, m)) 0.18, so AdamW's learning rates
    # carry over: before weight decay, a matrix moves by lr times an update
    # whose root mean square is 0.18, whatever its shape and gradient size
    torch.manual_seed(0)
    model = torch.nn.Linear(shape[1], shape[0], bias=False)
    model.optimizer_roles = lambda: [("weight", model.weight, "matrix", (1, *shape))]
    method = by_part(model, lr=0.01, warmup=1, weight_decay=0.0)
    model.weight.grad = torch.randn(shape) * 1e3
    before = model.weight.detach().clone()
    method.before_update(1)
    method.optimizer.step()
    rms = ((model.weight.detach() - before) / 0.01).pow(2).mean().sqrt()
    assert abs(rms.item() - 0.18) < 0.002


def test_every_parameter_takes_a_step():
    # the new attention's parameters (sinks, merging biases, indexers, the
    # MTP block's window attention) each belong to one optimizer group, once
    torch.manual_seed(0)
    model = WPT(ModelConfig(**TINY))
    method = by_part(model, lr=0.01, warmup=1)
    grouped = [id(p) for group in method.optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = [p.detach().clone() for p in model.parameters()]
    method.before_update(1)
    method.optimizer.step()
    still = [name for (name, p), old in zip(model.named_parameters(), before) if torch.equal(p, old)]
    assert still == []


def test_the_classics_test_recipe_trains_with_a_falling_loss():
    # the test recipe's architecture (data/recipes/models/classics-test.json),
    # on a repeating text it can learn in a few steps: the loss stays finite
    # and falls, through every part including CSA's indexer
    path = Path(__file__).parents[3] / "data/recipes/models/classics-test.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    model = WPT(ModelConfig(vocab_size=64, **{**recipe["model"], "block_size": 128})).train()
    method = by_part(model, steps=20, **{**recipe["steps"][0]["optimizer"]["settings"], "warmup": 2})
    loss_of = read_text().start_method()  # next-token, with its multi-token prediction schedule
    text = torch.arange(129).remainder(13)[None].expand(4, -1)  # a period of 13 tokens
    x, y = text[:, :-1], text[:, 1:]
    losses, indexer_losses = [], []
    for step in range(1, 13):
        method.optimizer.zero_grad()
        loss, indexer = loss_of.loss(model, x, y)
        loss.backward()
        method.before_update(step)
        method.optimizer.step()
        model.after_update(lambda tensor, op: tensor)
        method.after_update(step, lambda tensor, op: tensor)
        loss_of.after_update(step)
        losses.append(loss.item())
        indexer_losses.append(indexer.item())
    assert all(math.isfinite(value) for value in losses + indexer_losses)
    assert losses[-1] < 0.5 * losses[0]

