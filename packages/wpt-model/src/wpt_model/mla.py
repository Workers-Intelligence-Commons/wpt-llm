"""Gated Multi-head Latent Attention (docs/model.md, "Attention"): each
token's keys and values come from one compressed vector c_t = W_c x_t; no
positional encoding (NoPE), so DeepSeek's rotary key part is left out; K3's
output gate y_t = W_o(sigmoid(W_g x_t) * o_t), with the attention output in
FP32.

For QK-Clip, a layer can record each head's largest query-key score of a
step (track = True) and scale the heads that passed a limit (clip).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EPS

SCORE_ROWS = 256  # queries per block when recording the largest scores, to bound memory


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        D, H, d, r = config.d_model, config.mla_heads, config.mla_head_dim, config.mla_kv_rank
        self.heads, self.head_dim = H, d
        self.q = nn.Linear(D, H * d, bias=False)
        self.kv_down = nn.Linear(D, r, bias=False)  # W_c
        self.kv_norm = nn.RMSNorm(r, eps=EPS)
        self.k_up = nn.Linear(r, H * d, bias=False)
        self.v_up = nn.Linear(r, H * d, bias=False)
        self.gate = nn.Linear(D, H * d, bias=False)
        self.out = nn.Linear(H * d, D, bias=False)
        self.track = False
        self.score_max = None  # (H,) largest query-key score since the last clip; not a buffer, which DDP would overwrite

    def forward(self, x, collect=True):
        B, T, _ = x.shape

        def heads(t):  # (B, T, H * d) -> (B, H, T, d)
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        c = self.kv_norm(self.kv_down(x))
        q, k, v = heads(self.q(x)), heads(self.k_up(c)), heads(self.v_up(c))
        gate = torch.sigmoid(self.gate(x))
        with torch.autocast(x.device.type, enabled=False):
            if self.track and self.training and collect:
                self._record(q.float(), k.float())
            o = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, -1) * gate.float()
        return self.out(o)

    @torch.no_grad()
    def _record(self, q, k):
        """Keep each head's largest score q_i . k_j / sqrt(d) over j <= i."""
        T, scale = q.shape[2], self.head_dim ** -0.5
        largest = q.new_full((self.heads,), float("-inf"))
        for start in range(0, T, SCORE_ROWS):
            scores = q[:, :, start:start + SCORE_ROWS] @ k.transpose(-1, -2) * scale
            rows = torch.arange(start, start + scores.shape[2], device=q.device)[:, None]
            scores = scores.masked_fill(torch.arange(T, device=q.device)[None] > rows, float("-inf"))
            largest = torch.maximum(largest, scores.amax(dim=(0, 2, 3)))
        self.score_max = largest if self.score_max is None else torch.maximum(self.score_max, largest)

    @torch.no_grad()
    def clip(self, tau, reduce):
        """QK-Clip: a head whose largest score since the last call, over
        every process (reduce), passed tau has its query and key weights
        each scaled by sqrt(tau / largest)."""
        if self.score_max is None:
            return
        largest, self.score_max = reduce(self.score_max, "max"), None
        factor = (tau / largest).clamp(max=1).sqrt()[:, None, None]
        self.q.weight.view(self.heads, self.head_dim, -1).mul_(factor.to(self.q.weight.dtype))
        self.k_up.weight.view(self.heads, self.head_dim, -1).mul_(factor.to(self.k_up.weight.dtype))
