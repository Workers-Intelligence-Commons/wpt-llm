"""The WPT model (docs/model.md): token ids -> next-token scores, and the
scores of the token after next from the multi-token prediction block.

The model only maps token ids to scores. Loss belongs to training, sampling
belongs to inference. On a GPU it computes in BF16, keeping FP32 where
docs/model.md says (the attention output, mHC's mappings, routing scores,
KDA's memory).

With recompute set, training keeps only each layer's input and runs the
layer again during the backward pass, instead of keeping everything it
computed: the same result with much less memory, for about a third more
time. The second run records no statistics (expert balancing, query-key
scores), so a step counts every token once.

record and replay (experts.Routes) keep the experts the routers chose for
one sequence and route a later pass the same way (R3).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .engram import Engram
from .experts import SHARED_EXPERTS, Experts, Routes
from .kda import KDA
from .layers import EPS, FeedForward
from .mhc import STREAMS, Connection
from .mla import MLA

MLA_EVERY = 4    # gated MLA on every 4th layer and the last; KDA elsewhere
HASH_LAYERS = 3  # the first 3 layers' experts are hash-routed


class Layer(nn.Module):
    def __init__(self, config, number):
        super().__init__()
        D = config.d_model
        self.global_attention = number % MLA_EVERY == 0 or number == config.n_layer
        self.attention = MLA(config) if self.global_attention else KDA(config)
        self.experts = Experts(config, hashed=number <= HASH_LAYERS)
        self.engram = Engram(config, STREAMS) if number in config.engram_layers else None
        self.attention_norm, self.experts_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)
        self.attention_streams, self.experts_streams = Connection(D), Connection(D)

    def forward(self, streams, ids, canonical, collect=True, record=None, replay=None):
        """collect: record training statistics (False when the layer runs again for the backward pass)."""
        if self.engram is not None:
            streams = self.engram(streams, canonical)
        streams = self.attention_streams(streams, lambda h: self.attention(self.attention_norm(h), collect))
        return self.experts_streams(streams, lambda h: self.experts(self.experts_norm(h), ids, collect, record,
                                                                     replay))


def _first_run_collects(layer, record, replay):
    """The layer as a function whose first call records statistics and
    whose later call, the run again for the backward pass, does not."""
    runs = []

    def run(streams, ids, canonical):
        runs.append(None)
        return layer(streams, ids, canonical, collect=len(runs) == 1, record=record, replay=replay)
    return run


class MultiTokenPrediction(nn.Module):
    """DeepSeek-V3's MTP module: joins the final hidden state and the next
    token's embedding, then one layer. WPT's layer: gated MLA and a dense
    SiTU-GLU as wide as the shared experts together, on one stream."""

    def __init__(self, config):
        super().__init__()
        D = config.d_model
        self.hidden_norm, self.embedding_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)
        self.join = nn.Linear(2 * D, D, bias=False)
        self.attention_norm, self.attention = nn.RMSNorm(D, eps=EPS), MLA(config)
        self.feed_forward_norm = nn.RMSNorm(D, eps=EPS)
        self.feed_forward = FeedForward(D, SHARED_EXPERTS * config.shared_width)
        self.final_norm = nn.RMSNorm(D, eps=EPS)

    def forward(self, hidden, next_embedding):
        h = self.join(torch.cat([self.hidden_norm(hidden), self.embedding_norm(next_embedding)], -1))
        h = h + self.attention(self.attention_norm(h))
        h = h + self.feed_forward(self.feed_forward_norm(h))
        return self.final_norm(h)


class WPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(Layer(config, number) for number in range(1, config.n_layer + 1))
        self.final_norm = nn.RMSNorm(config.d_model, eps=EPS)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp = MultiTokenPrediction(config)
        # Engram's canonical id of each token id, from the vocabulary (engram.canonical_ids); saved with the weights
        self.register_buffer("canonical", torch.arange(config.vocab_size))
        self.recompute = False  # how training uses memory, not part of the model: see the module's docstring

    def set_canonical_ids(self, ids: torch.Tensor):
        self.canonical.copy_(ids)

    def forward(self, idx, next_tokens=None, *, record: Routes | None = None, replay: Routes | None = None):
        """(B, T) token ids -> (B, T, vocab_size) next-token scores. With
        next_tokens, the (B, T) tokens that follow idx, also returns the
        multi-token prediction block's scores for the token after each next
        token: (scores, mtp scores). With record or replay (B = 1), the
        routed layers record their choices into it, or route as it holds."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            canonical = self.canonical[idx]
            streams = self.embedding(idx).float().unsqueeze(2).expand(-1, -1, STREAMS, -1)  # copied into each stream
            recompute = self.recompute and self.training and torch.is_grad_enabled()
            for layer in self.layers:
                streams = (checkpoint(_first_run_collects(layer, record, replay), streams, idx, canonical,
                                      use_reentrant=False)
                           if recompute else layer(streams, idx, canonical, record=record, replay=replay))
            hidden = streams.sum(2)
            logits = self.head(self.final_norm(hidden)).float()
            if next_tokens is None:
                return logits
            return logits, self.head(self.mtp(hidden, self.embedding(next_tokens))).float()

    def after_update(self, reduce):
        """After each optimizer step: Quantile Balancing sets the routers'
        biases from the step's scores, summed over processes by reduce."""
        for layer in self.layers:
            layer.experts.balance(reduce)

    def attention_layers(self):
        return [m for m in self.modules() if isinstance(m, MLA)]

    def track_query_keys(self, on: bool):
        """Whether MLA layers record their largest query-key scores in training, for clip_query_keys."""
        for layer in self.attention_layers():
            layer.track = on

    def clip_query_keys(self, tau, reduce):
        """QK-Clip on every MLA layer (KDA has no softmax scores)."""
        for layer in self.attention_layers():
            layer.clip(tau, reduce)

    def optimizer_roles(self):
        """(name, parameter, role, view) for every parameter. Role "matrix"
        (Muon): view is the (matrices, rows, columns) shape it is
        orthogonalized as, one matrix per head for the query, key and value
        projections and one per expert for routed experts. Role "vector"
        (AdamW): the embedding, the output head, norms, mHC's biases and
        gating factors, KDA's decay scalars and the convolutions. Role
        "table" (Engram's tables). view is None for the last two."""
        heads = {}
        for module in self.modules():
            if isinstance(module, KDA):
                heads.update({id(module.q.weight): module.heads, id(module.k.weight): module.heads,
                              id(module.v.weight): module.heads})
            elif isinstance(module, MLA):
                heads.update({id(module.q.weight): module.heads, id(module.k_up.weight): module.heads,
                              id(module.v_up.weight): module.heads})
        vectors = {id(self.embedding.weight), id(self.head.weight)}
        tables = set()
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                vectors.add(id(module.weight))
            elif isinstance(module, Connection):
                vectors.update(id(p) for name, p in module.named_parameters() if name != "phi.weight")
            elif isinstance(module, Engram):
                tables.update(id(table.weight) for table in module.tables)
        for name, p in self.named_parameters():
            if id(p) in tables:
                yield name, p, "table", None
            elif id(p) in vectors or p.ndim < 2:
                yield name, p, "vector", None
            elif p.ndim == 3:
                yield name, p, "matrix", tuple(p.shape)
            else:
                n = heads.get(id(p), 1)
                yield name, p, "matrix", (n, p.shape[0] // n, p.shape[1])
