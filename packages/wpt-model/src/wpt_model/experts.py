"""The feed-forward sublayer (docs/model.md, "Experts"): 2 shared SiTU-GLU
experts at full width, plus routed experts working in a narrower latent
space,

    z = W_down x;  u = sum over chosen experts i of p_i E_i(z)
    y = sum of the shared experts(x) + W_up RMSNorm(u)

chosen by a fixed hash of the token id in the first 3 layers, and by a
router balanced with Quantile Balancing after.

Quantile Balancing's quantile is read from a histogram of each expert's
s_j - alpha, which processes can add up: 4096 bins within 0.5 of the
expert's previous quantile (-b_j), with values outside counted in the edge
bins. A bin is about 0.00024 wide, and one step moves a bias by at most 0.5.

Every step's dispatch is in a fixed order: each token's k copies are sorted
by expert with a stable sort and put back by the inverse permutation, so
gradients are sums in a fixed order, and every expert runs, even on no
token, so every process has a gradient for every weight.
"""

import torch
import torch.nn as nn

from .layers import EPS, FeedForward, expert_forward

SHARED_EXPERTS = 2
HASH_MULTIPLIER = 2654435761  # h(t) = (t * 2654435761 mod 2^32) mod n
BINS, WINDOW = 4096, 0.5      # balancing histograms: 4096 bins within 0.5 of each expert's previous quantile


class Experts(nn.Module):
    def __init__(self, config, hashed):
        super().__init__()
        D, L, E, n = config.d_model, config.latent_width, config.expert_width, config.n_experts
        self.n, self.k, self.hashed = n, config.top_k, hashed
        self.shared = nn.ModuleList(FeedForward(D, config.shared_width) for _ in range(SHARED_EXPERTS))
        self.down = nn.Linear(D, L, bias=False)
        self.up = nn.Linear(L, D, bias=False)
        self.latent_norm = nn.RMSNorm(L, eps=EPS)
        # routed experts, stacked: expert j is w_gate[j], w_up[j], w_down[j]
        self.w_gate = nn.Parameter(torch.empty(n, E, L).uniform_(-L ** -0.5, L ** -0.5))
        self.w_up = nn.Parameter(torch.empty(n, E, L).uniform_(-L ** -0.5, L ** -0.5))
        self.w_down = nn.Parameter(torch.empty(n, L, E).uniform_(-E ** -0.5, E ** -0.5))
        if not hashed:
            self.router = nn.Linear(D, n, bias=False)
            self.register_buffer("bias", torch.zeros(n))  # Quantile Balancing's b; saved, frozen for inference
            self.histogram = None  # this step's counts so far; not a buffer, which DDP would overwrite

    def route(self, x, ids):
        """(chosen experts (m, k), their weights (m, k)) for m tokens."""
        if self.hashed:
            h = (ids * HASH_MULTIPLIER) % 2**32 % self.n
            spread = torch.arange(self.k, device=ids.device) * self.n // self.k
            return (h[:, None] + spread) % self.n, torch.full((len(ids), self.k), 1 / self.k, device=ids.device)
        with torch.autocast(x.device.type, enabled=False):
            s = torch.sigmoid(self.router(x.float()))  # routing scores in FP32
            top = torch.topk(s + self.bias, self.k + 1)
            chosen = top.indices[:, :self.k]
            weights = s.gather(1, chosen)
            if self.training:
                self._count(s.detach(), top.values[:, self.k].detach())
            return chosen, weights / weights.sum(-1, keepdim=True)

    def _count(self, s, cutoff):
        """Add this batch's s_j - alpha_i to each expert's histogram, whose
        bins are centred on the previous quantile -b_j; alpha_i is token i's
        (k+1)-th largest biased score."""
        offset = s - cutoff[:, None] + self.bias  # s_j - alpha_i minus the previous quantile -b_j
        bins = ((offset + WINDOW) * (BINS / (2 * WINDOW))).floor().long().clamp(0, BINS - 1)
        flat = (bins + torch.arange(self.n, device=s.device) * BINS).reshape(-1)
        counts = torch.bincount(flat, minlength=self.n * BINS).view(self.n, BINS)
        self.histogram = counts if self.histogram is None else self.histogram + counts

    @torch.no_grad()
    def balance(self, reduce):
        """After a step: each expert's bias becomes -quantile_{1-k/n} of its
        s_j - alpha over the step's tokens on every process, then the biases
        are shifted to mean 0. They route the next step."""
        if self.hashed or self.histogram is None:
            return
        counts = reduce(self.histogram, "sum").double()
        self.histogram = None
        below = counts.cumsum(-1)
        target = (1 - self.k / self.n) * below[:, -1:]
        index = (below < target).sum(-1, keepdim=True).clamp(max=BINS - 1)
        before = torch.where(index > 0, below.gather(1, (index - 1).clamp(min=0)), 0.0)
        inside = (target - before) / counts.gather(1, index).clamp(min=1)
        quantile = -self.bias + (-WINDOW + (index + inside.clamp(0, 1)) * (2 * WINDOW / BINS)).squeeze(1)
        bias = -quantile
        self.bias.copy_(bias - bias.mean())

    def forward(self, x, ids):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        y = sum(expert(flat) for expert in self.shared)
        chosen, weights = self.route(flat, ids.reshape(-1))
        m, k = chosen.shape
        z = self.down(flat)
        order = chosen.reshape(-1).argsort(stable=True)  # the m * k copies, grouped by expert
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(len(order), device=order.device)
        copies = z.unsqueeze(1).expand(m, k, z.shape[-1]).reshape(m * k, -1)[order]
        sizes = torch.bincount(chosen.reshape(-1), minlength=self.n).tolist()
        out = torch.cat([expert_forward(part, self.w_gate[j], self.w_up[j], self.w_down[j])
                         for j, part in enumerate(copies.split(sizes))])
        u = (out[inverse].view(m, k, -1) * weights[..., None].to(out.dtype)).sum(1)
        return (y + self.up(self.latent_norm(u))).view(B, T, D)
