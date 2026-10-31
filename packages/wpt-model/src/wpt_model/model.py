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

Every layer says once what it passes on to later tokens (carry.py), and a
pass may follow earlier tokens (sequence.py): a segment on one GPU,
carrying what earlier segments passed on (carried), or one process's slice
of a window split across processes, or of one part of a window run as
parts split across processes (split). Each gives exactly the whole
window's result.

Generation reads tokens through a cache (cache.py) that keeps the same
description: new_cache, then extend with each call's new tokens, undo to
take back the last tokens of an undoable extend, and extend_mtp for the
multi-token prediction block. extend is forward's own pass through the
cache's place, so the same scores within rounding, and it has no length
limit.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import CSA_RATIO, HCA_RATIO, Attention
from .cache import Cache, Recorder
from .config import ModelConfig, global_layers
from .engram import Engram
from .experts import SHARED_EXPERTS, Experts, Routes
from .kda import KDA
from .layers import EPS, FeedForward
from .mhc import STREAMS, Connection
from .optimizer import roles
from .sequence import WHOLE, Carried, Split

HASH_LAYERS = 3   # the first 3 layers' experts are hash-routed
RATIOS = {"CSA": CSA_RATIO, "HCA": HCA_RATIO}

# the model's parts, by the names a training step locks them by (docs/model.md, "Parts"): every parameter
# belongs to exactly one, and BALANCING is the routers' balancing biases, which change after each update
EMBEDDING, ATTENTION, INDEXERS, ROUTERS, BALANCING = "embedding", "attention", "indexers", "routers", "balancing"
EXPERTS, ENGRAM, STREAMS_PART, FINAL_NORM = "experts", "engram", "streams", "final-norm"
OUTPUT_HEAD, MULTI_TOKEN_PREDICTION = "output-head", "multi-token-prediction"
PARTS = (EMBEDDING, ATTENTION, INDEXERS, ROUTERS, BALANCING, EXPERTS, ENGRAM, STREAMS_PART, FINAL_NORM, OUTPUT_HEAD,
         MULTI_TOKEN_PREDICTION)
# the first module names of a parameter's name, in order of precedence, and the part they make it
_PART_OF = ((("mtp",), MULTI_TOKEN_PREDICTION), (("embedding",), EMBEDDING), (("head",), OUTPUT_HEAD),
            (("final_norm",), FINAL_NORM), (("attention", "indexer"), INDEXERS),
            (("attention",), ATTENTION), (("attention_norm",), ATTENTION), (("experts", "router"), ROUTERS),
            (("experts",), EXPERTS), (("experts_norm",), EXPERTS), (("engram",), ENGRAM),
            (("attention_streams",), STREAMS_PART), (("experts_streams",), STREAMS_PART))


def part_of(name: str) -> str:
    """The part (PARTS) a parameter belongs to, by its name in the model
    (WPT.named_parameters): a layer's are read after `layers.<n>.`."""
    words = name.split(".")
    if words[0] == "layers":
        words = words[2:]
    for prefix, part in _PART_OF:
        if tuple(words[:len(prefix)]) == prefix:
            return part
    raise ValueError(f"{name}: a parameter of no part")


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


def _counts(ids, counts):
    """counts (B,) as a tensor on ids' device: every one of ids' n tokens when None."""
    B, n = ids.shape
    if counts is None:
        return torch.full((B,), n, dtype=torch.long, device=ids.device)
    return torch.as_tensor(counts, dtype=torch.long, device=ids.device)


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
                replay: Routes | None = None, sequence=None, features=()):
        """(B, T) token ids -> (B, T, vocab_size) next-token scores. With
        next_tokens, the (B, T) tokens that follow idx, also returns the
        multi-token prediction block's scores for the token after each next
        token. With indexer_loss, also returns the CSA indexers' loss of this
        pass, summed over CSA layers (0 without one); only the indexers'
        parameters get a gradient from it. Returns the scores alone, or the
        tuple (scores, [mtp scores], [indexer loss], [features]) of what was
        asked. With record or replay (B = 1), the routed layers record their
        choices into it, or route as it holds. sequence: where idx sits in a
        longer window (carried, split), the whole window when None; the
        indexer loss is then the mean over idx's queries. features: layer
        numbers (from 1) whose hidden states (each layer's output streams,
        summed) come back side by side, (B, T, len(features) * d_model),
        what a drafter reads (drafter.py)."""
        sequence = WHOLE if sequence is None else sequence
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=idx.is_cuda):
            hidden, total, found = self._hidden(idx, indexer_loss, record, replay, sequence, features)
            outputs = [self.head(self.final_norm(hidden)).float()]
            if next_tokens is not None:
                outputs.append(self.head(self.mtp(hidden, self.embedding(next_tokens), sequence)).float())
            if indexer_loss:
                outputs.append(total)
            if features:
                outputs.append(found)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _hidden(self, idx, indexer_loss, record, replay, sequence, features):
        """(the final hidden states, the indexers' loss, the features' hidden
        states side by side or None) of idx at its place, inside forward's
        autocast."""
        canonical = self.canonical[idx]
        streams = self.embedding(idx).float().unsqueeze(2).expand(-1, -1, STREAMS, -1)  # copied into each stream
        recompute = self.recompute and self.training and torch.is_grad_enabled()
        total = streams.new_zeros(())
        found = {}
        for number, layer in enumerate(self.layers, start=1):
            streams, alignment = (
                checkpoint(_first_run_collects(layer, record, replay), streams, idx, canonical, indexer_loss,
                           sequence, use_reentrant=False)
                if recompute else layer(streams, idx, canonical, indexer_loss, collect=sequence.collect,
                                        record=record, replay=replay, sequence=sequence))
            if alignment is not None:
                total = total + alignment
            if number in features:
                found[number] = streams.sum(2).float()
        return streams.sum(2), total, torch.cat([found[number] for number in features], -1) if features else None

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

    def new_cache(self, rows, routes=False) -> Cache:
        """An empty cache of `rows` rows (cache.py); with routes, it records
        each row's routed experts (R3) as its tokens are read."""
        return Cache(self, rows, routes)

    @torch.no_grad()
    def extend(self, ids, cache, counts=None, undoable=False, features=()):
        """Reads ids (B, n), the next tokens of the cache's B rows: the
        first counts[b] of row b (all n by default), the rest padding.
        Returns (B, n, vocab_size) next-token scores, the same as forward's
        over each row's whole text within rounding (padding's are
        meaningless), and the (B, n, D) final hidden states, which
        extend_mtp reads; with features (layer numbers), also those layers'
        hidden states as forward gives them. With undoable, undo can then
        take back the last tokens of this call."""
        counts = _counts(ids, counts)
        reading = cache.text.reading(counts, undoable)
        record = Recorder(cache.routes, counts.tolist()) if cache.routes is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            hidden, _, found = self._hidden(ids, False, record, None, reading, features)
            scores = self.head(self.final_norm(hidden)).float()
        reading.finish()
        return (scores, hidden, found) if features else (scores, hidden)

    @torch.no_grad()
    def undo(self, cache, kept):
        """After an undoable extend: keeps only its first kept[b] tokens of
        row b, as if the others had never been read (the routes recorded
        for them too)."""
        cache.text.undo(torch.as_tensor(kept, dtype=torch.long, device=cache.lengths.device))
        if cache.routes is not None:
            for routes, length in zip(cache.routes, cache.lengths.tolist()):
                routes.keep(length)

    @torch.no_grad()
    def extend_mtp(self, hidden, next_ids, cache, counts=None, undoable=False):
        """The multi-token prediction block's scores (B, n, vocab_size) for
        the token after next, through its window in the cache: hidden, the
        final hidden states extend returned for n positions, and next_ids,
        the token that follows each. The block reads these pairs in order,
        apart from extend; undo_mtp takes back those of an undoable call."""
        reading = cache.mtp.reading(_counts(next_ids, counts), undoable)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=next_ids.is_cuda):
            scores = self.head(self.mtp(hidden, self.embedding(next_ids), reading)).float()
        reading.finish()
        return scores

    @torch.no_grad()
    def undo_mtp(self, cache, kept):
        """undo, for the pairs of the last undoable extend_mtp."""
        cache.mtp.undo(torch.as_tensor(kept, dtype=torch.long, device=cache.lengths.device))

    def after_update(self, reduce, balance=True):
        """After each optimizer step: Quantile Balancing sets the routers'
        biases from the step's scores, summed over processes by reduce.
        Without balance (a step that locks BALANCING), the biases stay as
        they are and the step's scores are dropped."""
        for layer in self.layers:
            if balance:
                layer.experts.balance(reduce)
            else:
                layer.experts.histogram = None

    def parts(self) -> dict[str, list[torch.nn.Parameter]]:
        """{part: its parameters} for every name of PARTS, in the model's
        order (BALANCING has none: its biases are not trained by gradient);
        every parameter is in exactly one."""
        found = {part: [] for part in PARTS}
        for name, parameter in self.named_parameters():
            found[part_of(name)].append(parameter)
        return found

    def optimizer_roles(self):
        """(name, parameter, role, view) for every parameter, by the rule of
        optimizer.roles, with the embedding and the output head among the
        "vector" ones (AdamW)."""
        return roles(self, (self.embedding.weight, self.head.weight))
