"""The low-precision formats DeepSeek-V4 keeps attention entries in
(docs/model.md, "Attention"), simulated in PyTorch so that every device
computes the same values: each value is rounded to the format and turned
back, in FP32, as DeepSeek's kernels do in place (act_quant and
fp4_act_quant with inplace=True in V4-Flash's inference/kernel.py).

- FP8 (E4M3): each block of 64 values shares a scale, the power of 2 at or
  above the block's largest magnitude / 448 (V4's "ue8m0" scales).
- FP4 (E2M1): each block of 32 values shares a scale, the power of 2 at or
  above the block's largest magnitude / 6.
- hadamard: the Hadamard rotation V4 applies before FP4, which spreads each
  vector over all its dimensions and leaves dot products unchanged.
- fp8_parts: FP8 values as the kernels read them, E4M3 numbers and one
  scale per block, whose product is fp8's result.

The gradient passes through the rounding as if it were not there (the
straight-through estimator of V4's quantization-aware training).
"""

import functools

import torch

FP8_BLOCK, FP4_BLOCK = 64, 32
E4M3 = (448.0, 3, -6, 1e-4)        # largest value, mantissa bits, smallest normal exponent, smallest block maximum
E2M1 = (6.0, 1, 0, 6 * 2.0**-126)  # (the smallest block maxima are the kernels': no scale below them)


def fp8(x, block=FP8_BLOCK):
    """x with each block of its last dimension rounded to FP8 E4M3 and back."""
    return _simulated(x, block, *E4M3)


def fp4(x, block=FP4_BLOCK):
    """x with each block of its last dimension rounded to FP4 E2M1 and back."""
    return _simulated(x, block, *E2M1)


@torch.no_grad()
def fp8_parts(x, block=FP8_BLOCK):
    """(numbers, scales): x's FP8 form as E4M3 numbers (B..., n) in FP32 and
    each block's power-of-2 scale (B..., n / block), so that numbers times
    their block's scale is fp8(x). What an FP8 kernel reads."""
    return _parts(x.float(), block, *E4M3)


def hadamard(x):
    """x times the Hadamard matrix of its last dimension (a power of 2), in
    Sylvester's order, scaled by 1/sqrt(n) so that it is orthogonal (V4's
    rotate_activation)."""
    return (x.float() @ _hadamard_matrix(x.shape[-1], x.device)).to(x.dtype)


@functools.cache
def _hadamard_matrix(n, device):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    if h.shape[0] != n:
        raise ValueError(f"a Hadamard rotation needs a power of 2, not {n}")
    return (h * n ** -0.5).to(device)


def _simulated(x, block, *format):
    xf = x.float()
    with torch.no_grad():
        numbers, scales = _parts(xf, block, *format)
        rounded = (numbers.unflatten(-1, (-1, block)) * scales[..., None]).view_as(xf)
    return (xf + (rounded - xf).detach()).to(x.dtype)  # the rounded values, with x's gradient


def _parts(x, block, largest, mantissa, smallest_normal, smallest_max):
    """x (FP32) as the format's numbers and each block's scale."""
    v = x.reshape(*x.shape[:-1], x.shape[-1] // block, block)
    ratio = v.abs().amax(-1).clamp(min=smallest_max) * (1 / largest)  # in FP32, as the kernels
    fraction, exponent = torch.frexp(ratio)
    scale = torch.ldexp(torch.ones_like(ratio), exponent - (fraction == 0.5).int())  # 2^ceil(log2(ratio))
    numbers = _nearest(torch.clamp(v / scale[..., None], -largest, largest), mantissa, smallest_normal)
    return numbers.view_as(x), scale


def _nearest(v, mantissa, smallest_normal):
    """v (within the format's range) rounded to the nearest value of a format
    with these mantissa bits and smallest normal exponent, ties to even."""
    _, exponent = torch.frexp(v)  # |v| = f 2^e with f in [0.5, 1): floor(log2 |v|) = e - 1
    step = torch.ldexp(torch.ones_like(v), (exponent - 1).clamp(min=smallest_normal) - mantissa)
    return torch.round(v / step) * step
