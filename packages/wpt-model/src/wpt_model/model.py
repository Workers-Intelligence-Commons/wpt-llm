"""The WPT model (docs/model.md): token ids -> next-token scores, and the
scores of the token after next from the multi-token prediction block.

The model only maps token ids to scores. Loss belongs to training, sampling
belongs to inference; the one exception is the CSA indexers' loss, which
only the model can compute: it returns it when the caller asks
(WPT.forward), and training adds it. On a GPU
it computes in BF16, keeping FP32 where docs/model.md says (the attention
output, merging entries, mHC's mappings, routing scores, KDA's memory).

With recompute set, training keeps only each layer's input and runs the
layer again during the backward pass, instead of keeping everything it
computed: the same result with much less memory, for about a third more
time. The second run records no statistics (expert balancing), so a step
counts every token once.

record and replay (experts.Routes) keep the experts the routers chose for
one sequence and route a later pass the same way (R3).

A pass may be part of a longer window (sequence.py): a segment on one GPU,
carrying what earlier segments passed on (carried), or one process's slice
of a window split across processes, or of one part of a window run as
parts split across processes (split). Each gives exactly the whole
window's result.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import CSA_RATIO, HCA_RATIO, Attention, Compressor
from .config import ModelConfig, global_layers
from .engram import Engram
from .experts import SHARED_EXPERTS, Experts, Routes
from .kda import KDA
from .layers import EPS, FeedForward
from .mhc import STREAMS, Connection
from .sequence import WHOLE, Carried, Split

HASH_LAYERS = 3   # the first 3 layers' experts are hash-routed
RATIOS = {"CSA": CSA_RATIO, "HCA": HCA_RATIO}


def merging_ratio(number, config):
    """The merging ratio of layer number's attention (CSA_RATIO or
    HCA_RATIO), the kind the recipe names for that global layer, or None
    for KDA."""
    numbers = global_layers(config.n_layer)
    return RATIOS[config.global_attention[numbers.index(number)]] if number in numbers else None


class Layer(nn.Module):
    def __init__(self, config, number):
        super().__init__()
        D = config.d_model
        ratio = merging_ratio(number, config)
        self.attention = KDA(config) if ratio is None else Attention(config, ratio)
        self.experts = Experts(config, hashed=number <= HASH_LAYERS)
        self.engram = Engram(config, STREAMS) if number in config.engram_layers else None
        self.attention_norm, self.experts_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)
        self.attention_streams, self.experts_streams = Connection(D), Connection(D)

    def forward(self, streams, ids, canonical, indexer_loss=False, collect=True, record=None, replay=None,
                sequence=WHOLE):
        """streams -> (the next streams, the indexer's loss or None). The loss
        comes only from a CSA layer asked for it (indexer_loss). collect:
        record training statistics (False when the layer runs again for the
        backward pass). record, replay: the routed experts' choices (R3).
        sequence: where these tokens sit in the window."""
        if self.engram is not None:
            streams = self.engram(streams, canonical, sequence)
        found = [None]

        def attend(h):
            h = self.attention_norm(h)
            if isinstance(self.attention, KDA):
                return self.attention(h, sequence)
            out, found[0] = self.attention(h, indexer_loss, sequence)
            return out
        streams = self.attention_streams(streams, attend)
        return self.experts_streams(streams, lambda h: self.experts(self.experts_norm(h), ids, collect, record,
                                                                     replay)), found[0]


def _first_run_collects(layer, record, replay):
    """The layer as a function whose first call records statistics and
    whose later call, the run again for the backward pass, does not."""
    runs = []

    def run(streams, ids, canonical, indexer_loss, sequence):
        runs.append(None)
        return layer(streams, ids, canonical, indexer_loss, collect=len(runs) == 1 and sequence.collect, record=record,
                     replay=replay, sequence=sequence)
    return run


class MultiTokenPrediction(nn.Module):
    """DeepSeek-V3's MTP module: joins the final hidden state and the next
    token's embedding, then one layer. WPT's layer: attention over the last
    128 tokens alone (DeepSeek-V4's attention with no merged entries, as
    V4's and MiMo-V2-Flash's MTP blocks) and a dense SiTU-GLU as wide as
    the shared experts together, on one stream."""

    def __init__(self, config):
        super().__init__()
        D = config.d_model
        self.hidden_norm, self.embedding_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)
        self.join = nn.Linear(2 * D, D, bias=False)
        self.attention_norm, self.attention = nn.RMSNorm(D, eps=EPS), Attention(config, ratio=0)
        self.feed_forward_norm = nn.RMSNorm(D, eps=EPS)
        self.feed_forward = FeedForward(D, SHARED_EXPERTS * config.shared_width)
        self.final_norm = nn.RMSNorm(D, eps=EPS)

    def forward(self, hidden, next_embedding, sequence=WHOLE):
        h = self.join(torch.cat([self.hidden_norm(hidden), self.embedding_norm(next_embedding)], -1))
        h = h + self.attention(self.attention_norm(h), sequence=sequence)[0]
        h = h + self.feed_forward(self.feed_forward_norm(h))
        return self.final_norm(h)


class WPT(nn.Module):
    PART = HCA_RATIO  # a segment or a slice of a window holds a multiple of this many tokens

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

    def forward(self, idx, next_tokens=None, indexer_loss=False, *, record: Routes | None = None,
                replay: Routes | None = None, sequence=None):
        """(B, T) token ids -> (B, T, vocab_size) next-token scores. With
        next_tokens, the (B, T) tokens that follow idx, also returns the
        multi-token prediction block's scores for the token after each next
        token. With indexer_loss, also returns the CSA indexers' loss of this
        pass, summed over CSA layers (0 without one); only the indexers'
        parameters get a gradient from it. Returns the scores alone, or the
        tuple (scores, [mtp scores], [indexer loss]) of what was asked. With
        record or replay (B = 1), the routed layers record their choices into
        it, or route as it holds. sequence: where idx sits in a longer window
        (carried, split), the whole window when None; the indexer loss is
        then the mean over idx's queries."""
        sequence = WHOLE if sequence is None else sequence
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            canonical = self.canonical[idx]
            streams = self.embedding(idx).float().unsqueeze(2).expand(-1, -1, STREAMS, -1)  # copied into each stream
            recompute = self.recompute and self.training and torch.is_grad_enabled()
            total = streams.new_zeros(())
            for layer in self.layers:
                streams, alignment = (
                    checkpoint(_first_run_collects(layer, record, replay), streams, idx, canonical, indexer_loss,
                               sequence, use_reentrant=False)
                    if recompute else layer(streams, idx, canonical, indexer_loss, collect=sequence.collect,
                                            record=record, replay=replay, sequence=sequence))
                if alignment is not None:
                    total = total + alignment
            hidden = streams.sum(2)
            outputs = [self.head(self.final_norm(hidden)).float()]
            if next_tokens is not None:
                outputs.append(self.head(self.mtp(hidden, self.embedding(next_tokens), sequence)).float())
            if indexer_loss:
                outputs.append(total)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @staticmethod
    def carried(offset, carried, collect=True):
        """The place of a segment starting at token offset of its window, given
        what the segment before passed on (sequence.Carried)."""
        return Carried(offset, carried, collect)

    @staticmethod
    def split(rank, world, length, start=0, carried=None, collect=True):
        """The place of process rank's slice, of length tokens, of a window
        split across world processes, or of the part of it that starts at
        token start, given what the part before passed on (sequence.Split)."""
        return Split(rank, world, length, start, carried, collect)

    def after_update(self, reduce):
        """After each optimizer step: Quantile Balancing sets the routers'
        biases from the step's scores, summed over processes by reduce."""
        for layer in self.layers:
            layer.experts.balance(reduce)

    def optimizer_roles(self):
        """(name, parameter, role, view) for every parameter. Role "matrix"
        (Muon): view is the (matrices, rows, columns) shape it is
        orthogonalized as, one matrix per head for the query, key and value
        projections, one per expert for routed experts, and one per
        logically separate matrix elsewhere (DeepSeek-V4): each output
        group's projection, and a merging's two projection pairs. Role
        "vector" (AdamW): the embedding, the output head, norms, mHC's
        biases and gating factors, KDA's decay scalars, the convolutions,
        the attention sinks and the merging's positional biases. Role
        "table" (Engram's tables). view is None for the last two."""
        heads = {}
        for module in self.modules():
            if isinstance(module, KDA):
                heads.update({id(module.q.weight): module.heads, id(module.k.weight): module.heads,
                              id(module.v.weight): module.heads})
            elif isinstance(module, Attention):  # its one key-value head is one matrix
                heads.update({id(module.q_up.weight): module.heads, id(module.out_down.weight): module.groups})
                if module.indexer is not None:
                    heads[id(module.indexer.q.weight)] = module.indexer.heads
            elif isinstance(module, Compressor):
                heads.update({id(module.value.weight): module.parts, id(module.score.weight): module.parts})
        vectors = {id(self.embedding.weight), id(self.head.weight)}
        tables = set()
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                vectors.add(id(module.weight))
            elif isinstance(module, Connection):
                vectors.update(id(p) for name, p in module.named_parameters() if name != "phi.weight")
            elif isinstance(module, Engram):
                tables.update(id(table.weight) for table in module.tables)
            elif isinstance(module, Compressor):
                vectors.add(id(module.position))
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
