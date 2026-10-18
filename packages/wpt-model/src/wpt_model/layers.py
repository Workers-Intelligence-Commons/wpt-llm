"""Small parts several sublayers share."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .carry import Tail

EPS = 1e-6
CONV_KERNEL = 4


class ShortConv(nn.Module):
    """A depthwise causal convolution over time: each channel sees its own
    last `kernel` values, `dilation` tokens apart. (B, T, C) -> (B, T, C).
    earlier: the `reach` inputs before x (B, reach, C), zeros when None
    (before the text). It passes on its last `reach` inputs (carry.py):
    "inputs", or "inputs s" for each stream s when one convolution serves
    `streams` streams (Engram's)."""

    def __init__(self, channels, kernel=CONV_KERNEL, dilation=1, streams=None):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, dilation=dilation, bias=False)
        self.streams = streams

    @property
    def reach(self):
        """How many inputs before a token its output reads."""
        return (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]

    def passes_on(self):
        if self.streams is None:
            return {"inputs": Tail(self.reach)}
        return {f"inputs {s}": Tail(self.reach) for s in range(self.streams)}

    def forward(self, x, earlier=None):
        if earlier is None:
            earlier = x.new_zeros(x.shape[0], self.reach, x.shape[2])
        return self.conv(torch.cat([earlier.to(x.dtype), x], 1).transpose(1, 2)).transpose(1, 2)


def situ_glu(gate, up):
    """SiTU-GLU (K3) of the two branches' pre-activations: each branch is
    capped by a scaled tanh, so one unit's output never exceeds 100."""
    return 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))


class FeedForward(nn.Module):
    """One SiTU-GLU feed-forward block: width -> hidden -> width."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(situ_glu(self.gate(x), self.up(x)))


def expert_forward(x, w_gate, w_up, w_down):
    """One routed expert, whose weights are slices of the stacked tensors."""
    return F.linear(situ_glu(F.linear(x, w_gate), F.linear(x, w_up)), w_down)
