"""Kimi Delta Attention (docs/model.md, "Attention"): per head, a fixed-size
memory S updated token by token,

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

with alpha_t = exp(g_t) per channel and g_t in [-5, 0) (K3's bound).

Three forms compute it. recurrent_kda follows the formula token by token:
it is the reference. chunked_kda is the same result 16 tokens at a time in
PyTorch, for CPUs and for GPUs where flash-linear-attention does not import
(no Triton). On a GPU where flash-linear-attention imports, its
chunked kernel runs. Tests check that each agrees with the reference.

KDA passes on its memory S and each short convolution's last inputs
(carry.py). A pass that follows earlier tokens starts from what they left
(sequence.py): training's segments and slices through delta_attention,
generation (cache.py) through delta_step, where fewer than CHUNK new tokens
go through the formula token by token (flash-linear-attention's fused
recurrent kernel on a GPU where it imports), more through the chunked form
(its chunked kernel there). Undoing tokens (rejected draft tokens) replays
the kept tokens' inputs from the S before them (Kimi K3's way, carry.Memory),
through the same forms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .carry import Memory
from .layers import EPS, ShortConv

DECAY_BOUND = 5.0  # g = -5 sigmoid(...): every decay stays above exp(-5)
CHUNK = 16         # 16 tokens of at most -5 each keep exp(+-40) inside FP32 when factored at the middle

_fla = None


def _fla_kernels():
    """flash-linear-attention's (chunk_kda, fused_recurrent_kda), or False
    when it does not import here (no Triton, or no GPU support)."""
    global _fla
    if _fla is None:
        try:
            from fla.ops.kda import chunk_kda, fused_recurrent_kda
            _fla = chunk_kda, fused_recurrent_kda
        except Exception:  # ImportError, or a Triton that fails to start: either way the PyTorch form runs
            _fla = False
    return _fla


def _fla_chunk_kda():
    """flash-linear-attention's chunk_kda, or False when it does not import."""
    kernels = _fla_kernels()
    return kernels and kernels[0]


def recurrent_kda(q, k, v, g, beta, initial=None, final=False):
    """The formula, token by token. q, k, g: (B, T, H, dk); v: (B, T, H, dv);
    beta: (B, T, H). Returns o: (B, T, H, dv), in FP32; with final, (o, the
    last S). initial: the S before the first token (zeros by default)."""
    B, T, H, dk = q.shape
    S = q.new_zeros(B, H, dk, v.shape[-1], dtype=torch.float32) if initial is None else initial.float()
    out = []
    for t in range(T):
        kt = k[:, t].float()
        S = S * g[:, t].float().exp()[..., None]                       # Diag(alpha_t) S_{t-1}
        delta = beta[:, t].float()[..., None] * (v[:, t].float() - (kt[..., None] * S).sum(-2))
        S = S + kt[..., None] * delta[..., None, :]                    # + beta_t k_t (v_t - S^T k_t)^T
        out.append((q[:, t].float()[..., None] * S).sum(-2))           # S_t^T q_t
    out = torch.stack(out, 1) if out else q.new_zeros(B, 0, H, v.shape[-1], dtype=torch.float32)
    return (out, S) if final else out


def chunked_kda(q, k, v, g, beta, initial=None, chunk=CHUNK):
    """The same result as recurrent_kda, chunk tokens at a time: inside a
    chunk the updates are solved together (a triangular system), and only
    the memory passes from chunk to chunk. Decays are factored at each
    chunk's middle token, so no exponential leaves FP32's range. initial:
    the memory before the first token, zeros when None. Returns (o, the
    memory after the last token)."""
    with torch.autocast(q.device.type, enabled=False):
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        pad = (-T) % chunk  # padded tokens have beta 0 and decay 1: they change nothing before them
        if pad:
            q, k, v, g = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
            beta = F.pad(beta, (0, 0, 0, pad))
        N = (T + pad) // chunk

        def chunks(x):  # (B, T, H, d) -> (B, N, H, chunk, d)
            return x.float().reshape(B, N, chunk, H, -1).transpose(2, 3)

        q, k, v, g = map(chunks, (q, k, v, g))
        beta = beta.float().reshape(B, N, chunk, H).transpose(2, 3)
        G = g.cumsum(-2)                                   # log of the decay since the chunk's start
        middle = G[..., chunk // 2:chunk // 2 + 1, :]
        k_from = k * (middle - G).exp()                    # k_i / Gamma_i, factored at the middle
        before = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
        A = ((k * (G - middle).exp()) @ k_from.transpose(-1, -2) * beta[..., None]).masked_fill(~before, 0)
        M = ((q * (G - middle).exp()) @ k_from.transpose(-1, -2)).masked_fill(~before.logical_or(
            torch.eye(chunk, dtype=torch.bool, device=q.device)), 0)
        eye = torch.eye(chunk, device=q.device)
        solved = torch.linalg.solve_triangular(eye + A, beta[..., None] * torch.cat([v, k * G.exp()], -1),
                                               upper=False, unitriangular=True)
        U, Wk = solved[..., :dv], solved[..., dv:]         # updates: W = U - Wk S_0
        q_start = q * G.exp()
        end = G[..., -1:, :]
        k_end = k * (end - G).exp()
        decay_end = end.exp().squeeze(-2)[..., None]       # (B, N, H, dk, 1)
        S = q.new_zeros(B, H, dk, dv) if initial is None else initial.float()
        out = []
        for n in range(N):
            W = U[:, n] - Wk[:, n] @ S
            out.append(q_start[:, n] @ S + M[:, n] @ W)
            S = decay_end[:, n] * S + k_end[:, n].transpose(-1, -2) @ W
        return torch.stack(out, 1).transpose(2, 3).reshape(B, N * chunk, H, dv)[:, :T], S


def delta_attention(q, k, v, g, beta, initial=None):
    """KDA's output for (B, T, H, d) inputs, and the memory after the last
    token, starting from initial (B, H, dk, dv; zeros when None):
    flash-linear-attention's kernel on a GPU where it imports, else
    chunked_kda."""
    if q.is_cuda and (kernel := _fla_chunk_kda()):
        q, k, v, beta = (t.to(torch.bfloat16) for t in (q, k, v, beta))  # the kernel takes one format; g stays FP32
        return kernel(q, k, v, g.float(), beta, scale=1.0, safe_gate=True, lower_bound=-DECAY_BOUND,
                      initial_state=None if initial is None else initial.float().contiguous(), output_final_state=True)
    return chunked_kda(q, k, v, g, beta, initial)


def delta_step(q, k, v, g, beta, initial):
    """(KDA's output, the last S) for (B, n, H, d) inputs that follow the
    memory initial (B, H, dk, dv), in FP32. Fewer than CHUNK tokens go
    token by token, more chunk by chunk; on a GPU where flash-linear-attention
    imports, its fused recurrent or chunked kernel computes them."""
    few = q.shape[1] < CHUNK
    with torch.autocast(q.device.type, enabled=False):
        if q.is_cuda and (fla := _fla_kernels()):
            chunk, fused = fla
            q, k, v, beta = (t.to(torch.bfloat16).contiguous() for t in (q, k, v, beta))
            g, initial = g.float().contiguous(), initial.float().contiguous()
            if few:
                out, S = fused(q, k, v, g, beta, scale=1.0, initial_state=initial, output_final_state=True)
            else:
                out, S = chunk(q, k, v, g, beta, scale=1.0, initial_state=initial, output_final_state=True,
                               safe_gate=True, lower_bound=-DECAY_BOUND)
            return out.float(), S.float()
        if few:
            return recurrent_kda(q, k, v, g, beta, initial=initial, final=True)
        return chunked_kda(q, k, v, g, beta, initial=initial)


class KDA(nn.Module):
    """A KDA attention sublayer."""

    def __init__(self, config):
        super().__init__()
        D, H, d = config.d_model, config.kda_heads, config.kda_head_dim
        self.heads, self.head_dim = H, d
        self.q, self.k, self.v = (nn.Linear(D, H * d, bias=False) for _ in range(3))
        self.q_conv, self.k_conv, self.v_conv = (ShortConv(H * d) for _ in range(3))
        self.beta = nn.Linear(D, H, bias=False)
        self.decay_down = nn.Linear(D, d, bias=False)        # z_t: a projection of rank = head dimension
        self.decay_up = nn.Linear(d, H * d, bias=False)
        self.A = nn.Parameter(torch.zeros(H))                # A_h, starting at 0
        self.gate = nn.Linear(D, H * d, bias=False)
        self.norm = nn.RMSNorm(d, eps=EPS)
        self.out = nn.Linear(H * d, D, bias=False)

    def passes_on(self):
        return {"memory": Memory()}  # and each short convolution its last inputs (layers.ShortConv)

    def forward(self, x, sequence):
        """x (B, T, D); sequence: where these tokens sit (sequence.py: the
        whole window, a part of it, or a generation cache's reading), which
        gives the convolutions' earlier inputs and the memory's start."""
        B, T, _ = x.shape
        H, d = self.heads, self.head_dim

        def heads(projection, conv):
            p = projection(x)
            return F.silu(conv(p, sequence.before(conv, "inputs", p))).view(B, T, H, d)

        q = F.normalize(heads(self.q, self.q_conv), dim=-1, eps=EPS)
        k = F.normalize(heads(self.k, self.k_conv), dim=-1, eps=EPS)
        v = heads(self.v, self.v_conv)
        z = self.decay_up(self.decay_down(x)).view(B, T, H, d)
        beta = torch.sigmoid(self.beta(x))
        g = -DECAY_BOUND * torch.sigmoid(self.A.exp()[:, None] * z.float())
        o = self.norm(sequence.delta(self, q, k, v, g, beta).to(x.dtype))
        return self.out((torch.sigmoid(self.gate(x)).view(B, T, H, d) * o).reshape(B, T, H * d))
