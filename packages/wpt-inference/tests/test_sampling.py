import torch

from wpt_inference import sample_tokens


def test_a_seeded_generator_draws_the_same_tokens_again():
    # post-training runs repeat their attempts: the same seed gives the same
    # tokens, whatever else drew from torch's global generator in between
    torch.manual_seed(0)
    model = torch.nn.Embedding(50, 50)  # next-token scores from the last token alone

    def draw(seed):
        generator = torch.Generator().manual_seed(seed)
        torch.rand(100)  # other randomness, from the global generator
        return list(sample_tokens(model, [1, 2, 3], context_size=16, max_new_tokens=12, temperature=1.0, top_k=None,
                                  generator=generator))

    assert draw(7) == draw(7)
    assert draw(7) != draw(8)
