import torch

from wpt_model.formats import fp4, fp8, hadamard


def test_fp8_gives_the_native_fp8_values_with_v4_s_scales():
    # the simulation must give what a native FP8 cast gives (the kernels'):
    # per block of 64, the power of 2 at or above max / 448 as scale, then
    # E4M3 rounding; magnitudes from 1e-6 to 1e3 reach subnormals and ties
    torch.manual_seed(0)
    x = torch.randn(1000, 128) * torch.logspace(-6, 3, 1000)[:, None]
    blocks = x.view(1000, 2, 64)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448)))
    native = ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).view(1000, 128)
    assert torch.equal(fp8(x), native)


def test_fp4_rounds_to_e2m1_values_ties_to_even():
    # E2M1 holds 0, 0.5, 1, 1.5, 2, 3, 4 and 6 (times a power of 2 per block
    # of 32); halfway values go to the even one, as a hardware cast does
    halfway = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    x = torch.cat([halfway, -halfway, torch.zeros(16)])[None]
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0])
    assert torch.equal(fp4(x)[0, :16], torch.cat([expected, -expected]))
    assert torch.equal(fp4(x * 2.0**-10), fp4(x) * 2.0**-10)  # the block's scale follows its largest value


def test_the_gradient_passes_through_the_rounding():
    # V4's quantization-aware training: the straight-through estimator
    x = torch.randn(4, 64, requires_grad=True)
    weights = torch.randn(4, 64)
    (fp4(hadamard(x)) * weights).sum().backward()
    assert torch.allclose(x.grad, weights @ hadamard(torch.eye(64)).T, atol=1e-6)


def test_the_hadamard_rotation_keeps_dot_products():
    # an orthogonal matrix of +-1/sqrt(n): q . k is unchanged before FP4
    q, k = torch.randn(10, 128), torch.randn(10, 128)
    assert torch.allclose((hadamard(q) * hadamard(k)).sum(-1), (q * k).sum(-1), atol=1e-4)
    assert torch.allclose(hadamard(torch.eye(128)).abs(), torch.full((128, 128), 128 ** -0.5))
