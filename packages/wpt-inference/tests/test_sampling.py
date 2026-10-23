import pytest
import torch

from wpt_inference import DRAFT, Writer, sample_tokens, schedule
from wpt_model import WPT, Drafter, DrafterConfig, ModelConfig
from wpt_model.drafter import training_blocks

CONFIG = ModelConfig(vocab_size=50, block_size=16, n_layer=4, global_attention=["CSA"], d_model=16, kda_heads=2,
                     kda_head_dim=8, attention_heads=2, attention_head_dim=128, query_rank=8, output_groups=1,
                     output_rank=8, index_heads=4, index_head_dim=128, index_top_k=4, n_experts=4, top_k=2,
                     expert_width=8, latent_width=8, shared_width=16, engram_layers=[2], engram_order=3,
                     engram_heads=2, engram_dim=4, engram_table_size=31)


def test_a_seeded_generator_draws_the_same_tokens_again():
    # post-training runs repeat their attempts: the same seed gives the same
    # tokens, whatever else drew from torch's global generator in between
    torch.manual_seed(0)
    model = torch.nn.Embedding(50, 50)  # next-token scores from the last token alone

    def draw(seed):
        generator = torch.Generator().manual_seed(seed)
        torch.rand(100)  # other randomness, from the global generator
        return list(sample_tokens(model, [1, 2, 3], max_new_tokens=12, temperature=1.0, top_k=None,
                                  generator=generator))

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)


class Uncached(torch.nn.Module):
    """The model without its cache: it reads each row's whole text again."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        return self.model(idx)


def wpt():
    torch.manual_seed(0)
    return WPT(CONFIG).eval()


def test_rows_written_through_the_cache_get_the_tokens_the_whole_text_gives(routing):
    # 3 attempts from one prompt, written together past block_size (16),
    # one of them reading a tool result midway: the same seed draws the
    # same tokens as reading each row's whole text again for every token.
    # The cache's routes may differ from the whole text's at near ties only
    #: a row's tokens are then the
    # same up to the token drawn from the first position that went to other
    # experts, and the same everywhere when no position did
    model = wpt()

    def written(model, **options):
        writer = Writer(model, [0, 5, 7, 9], 3, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(3),
                        **options)
        for step in range(60):
            inputs = {1: [11, 12, 13, 14, 15]} if step == 20 else {}
            writer.write(inputs, {0: 1, 1: 1, 2: 1} if step % 7 else {0: 1, 2: 1})
        return writer

    cached, whole = written(model, routes=True), written(Uncached(model))
    for row, (history, history_whole) in enumerate(zip(cached.histories, whole.histories, strict=True)):
        assert len(history) == len(history_whole)
        differ = next((i for i, (a, b) in enumerate(zip(history, history_whole)) if a != b), None)
        read = len(history) - 1 if differ is None else differ  # the tokens read before the first one that differs
        first = routing.first_difference(model, history[:read], cached.routes[row])
        assert differ is None or first is not None, "a token differs although every route is the whole text's"


class Uniform:
    """A drafter proposing tokens drawn uniformly, whatever came before."""
    features, window = (), 0

    def draft(self, model, histories, context, count, probabilities, generator):
        tokens = torch.randint(CONFIG.vocab_size, (len(histories), count), generator=generator)
        return tokens, torch.full((len(histories), count, CONFIG.vocab_size), 1 / CONFIG.vocab_size), None


class Itself:
    """The model as its own drafter: its drafts are always kept."""
    features, window = (), 0

    def draft(self, model, histories, context, count, probabilities, generator):
        tokens, distributions = [], []
        for history in histories:
            history = list(history)
            for _ in range(count):
                p = model(torch.tensor([history]))[0, -1].softmax(-1)
                history.append(int(torch.multinomial(p, 1, generator=generator)))
                distributions.append(p)
            tokens.append(history[-count:])
        return torch.tensor(tokens), torch.stack(distributions).view(len(histories), count, -1), None


def test_speculative_decoding_keeps_the_model_s_own_distribution():
    # with drafts drawn uniformly, most are rejected: the first written
    # token of 4000 rows from one prompt still follows the model's own
    # probabilities, and the cache holds what each row's text gives
    model = wpt()
    prompt = [0, 5, 7, 9]
    rows = 4000
    writer = Writer(model, prompt, rows, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(0),
                    drafter=Uniform())
    first = writer.write({}, {row: 5 for row in range(rows)})
    counts = torch.bincount(torch.tensor([tokens[0] for tokens in first.values()]), minlength=CONFIG.vocab_size)
    with torch.no_grad():
        expected = model(torch.tensor([prompt]))[0, -1].softmax(-1)
    assert (counts / rows - expected).abs().sum() / 2 < 0.05  # total variation distance
    for _ in range(5):
        writer.write({}, {row: 5 for row in range(3)})
    for row in range(3):
        history = writer.histories[row]
        assert writer.cache.lengths[row] == len(history) - 1  # every token but the last written one is read
        with torch.no_grad():
            scores, _ = model.extend(torch.tensor([[history[-1]]] * rows), writer.cache, [1 if r == row else 0
                                                                                       for r in range(rows)])
            whole = model(torch.tensor([history]))[0, -1]
        assert torch.allclose(scores[row, 0], whole, atol=1e-3)


def test_drafts_drawn_from_the_model_itself_are_all_kept():
    # each step writes every draft and one more token, up to the limit, and
    # stops after a stop token
    model = wpt()
    writer = Writer(model, [0, 5, 7], 2, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(0),
                    drafter=Itself())
    with torch.no_grad():
        written = writer.write({}, {0: 100, 1: 3})
    assert len(written[0]) == DRAFT + 1 and len(written[1]) == 3
    stop = written[0][1]
    writer = Writer(model, [0, 5, 7], 2, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(0),
                    drafter=Itself(), stops=[stop])
    with torch.no_grad():
        assert writer.write({}, {0: 100, 1: 3})[0] == written[0][:written[0].index(stop) + 1]


def dspark(model):
    torch.manual_seed(1)
    return Drafter(CONFIG, DrafterConfig.for_model(CONFIG)).eval()


def test_the_dspark_drafter_keeps_the_model_s_own_distribution():
    # 4000 rows from one prompt, the drafts' confidences scheduling how many
    # are verified under passes that cost more the wider they are: the first
    # and second written tokens still follow the model's own probabilities
    model = wpt()
    prompt = [0, 5, 7, 9]
    rows = 4000
    writer = Writer(model, prompt, rows, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(0),
                    drafter=dspark(model), costs=[1.0, 1.3, 1.6, 1.9, 2.2, 2.5])
    while needed := {row: 2 - (len(history) - len(prompt)) for row, history in enumerate(writer.histories)
                     if len(history) < len(prompt) + 2}:
        writer.write({}, needed)
    assert 0 < writer.accepted < writer.verified < 5 * writer.rounds * rows  # some drafts kept, some cut
    first = torch.tensor([history[len(prompt)] for history in writer.histories])
    second = torch.tensor([history[len(prompt) + 1] for history in writer.histories])
    with torch.no_grad():
        p_first = model(torch.tensor([prompt]))[0, -1].softmax(-1)
        after = model(torch.tensor([prompt + [token] for token in range(CONFIG.vocab_size)]))[:, -1].softmax(-1)
    p_second = p_first @ after  # the second token's probability, whatever the first
    for drawn, expected in ((first, p_first), (second, p_second)):
        counts = torch.bincount(drawn, minlength=CONFIG.vocab_size)
        assert (counts / rows - expected).abs().sum() / 2 < 0.05  # total variation distance


@pytest.mark.parametrize("prompt", [[0, 5, 7, 9], [0]])  # [0]: nothing read before the first draft
def test_the_dspark_drafter_reads_what_the_whole_text_gives(prompt):
    # rows at different places, one reading a tool result midway: the
    # drafter's context is the hidden states of each row's last tokens read,
    # as the model gives them over the whole text
    model = wpt()
    drafter = dspark(model)
    writer = Writer(model, prompt, 3, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(3),
                    drafter=drafter, costs=[1.0] * (DRAFT + 1))
    for step in range(40):
        writer.write({1: [11, 12, 13, 14]} if step == 10 else {}, {0: 3, 2: 5} if step % 3 else {0: 2, 1: 4, 2: 1})
    for row, history in enumerate(writer.histories):
        read = len(history) - 1
        assert writer.cache.lengths[row] == read
        with torch.no_grad():
            _, features = model(torch.tensor([history[:-1]]), features=drafter.features)
        shown = min(read, drafter.window)
        assert torch.allclose(writer.context[row, -shown:], features[0, -shown:], atol=1e-4)
        assert not writer.context[row, :-shown].any()  # before the text


def test_a_dspark_drafter_trains_and_its_drafts_are_verified():
    # a short training on answers the model wrote itself (DSpark's losses)
    # gives finite losses and changes every weight it reaches, and the
    # writer verifies the trained drafter's drafts; how often they are kept
    # is not asked: this random model is nearly uniform, so no drafter
    # beats uniform drafts on it
    model, prompt = wpt(), [0, 5, 7, 9]
    teacher = Writer(model, prompt, 8, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(1))
    for _ in range(48):
        teacher.write({}, {row: 1 for row in range(8)})
    drafter = dspark(model).train()
    with torch.no_grad():
        reads = [model(torch.tensor([history]), features=drafter.features) for history in teacher.histories]
    optimizer = torch.optim.AdamW(drafter.parameters(), lr=3e-3)
    anchors = torch.Generator().manual_seed(2)
    before = {name: p.detach().clone() for name, p in drafter.named_parameters()}
    reached = set()
    for _ in range(5):
        blocks = [training_blocks(features[0], torch.tensor(history), scores[0],
                                  torch.randperm(len(history) - len(prompt) + 1, generator=anchors)[:16]
                                  + len(prompt) - 1, drafter.block)
                  for history, (scores, features) in zip(teacher.histories, reads)]
        loss, _ = drafter.losses(model, *[torch.cat(part) for part in zip(*blocks)])
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        reached |= {name for name, p in drafter.named_parameters() if p.grad is not None}
        optimizer.step()
        drafter.after_update(lambda tensor, op: tensor)
    drafter.eval()
    assert reached and all(not torch.equal(p, before[name]) for name, p in drafter.named_parameters() if name in reached)

    writer = Writer(model, prompt, 4, temperature=1.0, top_k=None, generator=torch.Generator().manual_seed(4),
                    drafter=drafter, costs=[1.0] * (DRAFT + 1))
    for _ in range(4):
        writer.write({}, {row: DRAFT + 1 for row in range(4)})
    assert all(len(history) > len(prompt) for history in writer.histories)
    assert 0 < writer.verified and 0 <= writer.accepted <= writer.verified


def test_the_scheduler_verifies_drafts_while_they_pay_for_their_pass():
    # passes cost the same up to 3 tokens per row, then 20% more: every draft
    # that fits the paid width is verified, and the wider pass is paid only
    # when the drafts it adds are likely enough to be kept; a row never
    # verifies more drafts than its cap
    costs = [1.0, 1.0, 1.0, 1.2, 1.2, 1.2]
    assert schedule([[0.9, 0.9, 0.9, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, 0.5]], [5, 1], costs) == [2, 1]
    assert schedule([[0.99] * 5, [0.99] * 5], [5, 5], costs) == [5, 5]
    assert schedule([[0.9] * 5], [0], costs) == [0]
