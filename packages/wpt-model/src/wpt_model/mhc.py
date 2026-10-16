"""mHC residual streams (docs/model.md, "Residual streams: mHC"): n streams
instead of one residual sum; around each sublayer F,

    H_pre  = sigmoid(a_pre  * x' phi_pre  + b_pre)
    H_post = 2 sigmoid(a_post * x' phi_post + b_post)
    H_res  = SinkhornKnopp(a_res * mat(x' phi_res) + b_res)
    x_next = H_res x + H_post^T F(H_pre x)

computed in FP32 from x', the RMS-normalized, flattened streams. The papers
do not give the biases' starting values; WPT starts with H_pre averaging
the streams (b_pre = logit(1/n)), H_post adding the output to each stream
in full (b_post = 0) and H_res near the identity (b_res = 4 on the
diagonal).
"""

import math

import torch
import torch.nn as nn

from .layers import EPS

STREAMS = 4
SINKHORN_ITERATIONS = 20
GATING_START = 0.01


def sinkhorn(logits):
    """Alternate column and row normalization of exp(logits), so rows and
    columns of each n x n matrix sum to 1."""
    m = (logits - logits.amax((-1, -2), keepdim=True)).exp()
    for _ in range(SINKHORN_ITERATIONS):
        m = m / m.sum(-2, keepdim=True)
        m = m / m.sum(-1, keepdim=True)
    return m


class Connection(nn.Module):
    """mHC's mappings around one sublayer."""

    def __init__(self, width, n=STREAMS):
        super().__init__()
        self.n = n
        self.norm = nn.RMSNorm(n * width, eps=EPS, elementwise_affine=False)
        self.phi = nn.Linear(n * width, 2 * n + n * n, bias=False)  # phi_pre, phi_post, phi_res side by side
        self.b_pre = nn.Parameter(torch.full((n,), math.log(1 / (n - 1))))  # logit(1/n)
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(4 * torch.eye(n))
        self.a_pre, self.a_post, self.a_res = (nn.Parameter(torch.tensor(GATING_START)) for _ in range(3))

    def forward(self, streams, sublayer):
        """streams (B, T, n, D) -> the next streams, with sublayer applied
        to H_pre x."""
        B, T, n, D = streams.shape
        with torch.autocast(streams.device.type, enabled=False):
            x = streams.float()
            p = self.phi(self.norm(x.reshape(B, T, n * D)))
            pre = torch.sigmoid(self.a_pre * p[..., :n] + self.b_pre)
            post = 2 * torch.sigmoid(self.a_post * p[..., n:2 * n] + self.b_post)
            res = sinkhorn(self.a_res * p[..., 2 * n:].view(B, T, n, n) + self.b_res)
            inner = (pre[..., None] * x).sum(2)
        out = sublayer(inner)
        with torch.autocast(streams.device.type, enabled=False):
            return res @ x + post[..., None] * out.float()[:, :, None]
