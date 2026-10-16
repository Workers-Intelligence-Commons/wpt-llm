"""Gated Multi-head Latent Attention (docs/model.md, "Attention"): each
token's keys and values come from one compressed vector c_t = W_c x_t; no
positional encoding (NoPE), so DeepSeek's rotary key part is left out; K3's
output gate y_t = W_o(sigmoid(W_g x_t) * o_t), with the attention output in
FP32."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EPS


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

    def forward(self, x):
        B, T, _ = x.shape

        def heads(t):  # (B, T, H * d) -> (B, H, T, d)
            return t.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        c = self.kv_norm(self.kv_down(x))
        q, k, v = heads(self.q(x)), heads(self.k_up(c)), heads(self.v_up(c))
        gate = torch.sigmoid(self.gate(x))
        with torch.autocast(x.device.type, enabled=False):
            o = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, -1) * gate.float()
        return self.out(o)
