"""The drafter (docs/model.md, "The drafter"): a small model, with its own
weights, that proposes the next tokens for speculative decoding, after
DeepSeek's DSpark (arXiv 2607.05147) as it serves DeepSeek-V4, on DFlash's
parallel backbone (arXiv 2602.06036). A model folder and its drafter are one
unit: load_model refuses a folder whose drafter is missing or was trained
against other weights (checkpoint.py).

One round drafts a block of up to BLOCK tokens after the anchor, the last
token the model wrote (and has not read yet):

    context   the hidden states of FEATURES target layers at each of the last
              WINDOW tokens the model read, side by side, projected to the
              model's width and RMS-normalized (DFlash's fused context)
    backbone  LAYERS layers, each window attention over the context and the
              block, then experts, joined to 4 streams by mHC (DSpark's
              production backbone: 3 MoE layers with mHC and a sliding
              window of 128). The block is the anchor's embedding then mask
              vectors, all read in one pass (DSpark's anchor-first block);
              the block's places see each other both ways (DFlash), and
              each sees the context entries within WINDOW tokens of it.
              The context enters every layer as keys and values, through
              that layer's own key-value projection (DFlash's KV injection).
    head      the target's output head, frozen, gives the base scores U_k
    Markov    p_k(v) = softmax(U_k + W1[x_{k-1}] W2)(v), sampled left to
              right, x_0 the anchor (DSpark's Markov head, of rank r:
              default_markov_rank: 256 at a width of 4096, in proportion to it)
    confidence c_k = sigmoid(w . [h_k ; W1[x_{k-1}]] / T_k): the chance
              that draft k is kept when every draft before it was, with
              each position's temperature T_k set after training on real
              kept and rejected drafts (DSpark's Sequential Temperature
              Scaling, calibrate)

The target's embedding and output head are shared and frozen: the drafter
reads them from the model it drafts for, and saves only its own weights,
in the model folder's `drafter/` folder (save_drafter, load_drafter), with
its provenance, which names the sha256 of the weights it was trained
against. Its settings come from the model recipe's `drafter` section
(DrafterConfig.for_model; its training's, wpt_inference.drafting).

The drafter only proposes: the model checks every draft it is given and
keeps it by the rule that leaves the model's own distribution unchanged
(wpt_inference), so a drafter changes how fast the model writes, never
what it writes.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .attention import WINDOW, Attention, frequencies, rotate, stored
from .checkpoint import TokenizerMismatch, check_drafter, tokenizer_identity
from .config import ModelConfig
from .experts import Experts
from .layers import EPS
from .mhc import STREAMS, Connection
from .notice import DRAFTER_NAME
from .optimizer import roles
from .sparse import sparse_attention

BLOCK = 5            # the most drafts per round: DSpark-5, the production drafter
LAYERS = 3           # the production backbone's MoE layers with mHC
MARKOV_RANK = 256    # the Markov head's rank r at a width of 4096 (DSpark's), in proportion to the width below
MARKOV_WIDTH = 4096
FEATURES = 5         # target layers read, uniformly from the 2nd to the third-to-last (DFlash)
LOSS_WEIGHTS = {"ce": 0.1, "tv": 0.9, "confidence": 1.0}  # DSpark's alpha_ce, alpha_tv, alpha_conf
CONFIG_NAME, WEIGHTS_NAME, PROVENANCE_NAME = "config.json", "model.safetensors", "provenance.json"


def feature_layers(n_layer, count=FEATURES):
    """The target layers (numbered from 1) a drafter reads: count of them,
    uniformly from the 2nd to the third-to-last, fewer when the model has
    fewer layers there (the 2nd alone in a model of 4 layers or fewer)."""
    first, last = 2, min(max(2, n_layer - 2), n_layer)
    if count == 1 or first == last:
        return (first,)
    return tuple(sorted({round(first + i * (last - first) / (count - 1)) for i in range(count)}))


def default_markov_rank(width):
    """The Markov head's default rank for a model of that width: 256 at
    4096, in proportion to it (as the indexer's sizes, docs/model.md), at
    least 1."""
    return max(1, MARKOV_RANK * width // MARKOV_WIDTH)


@dataclass(frozen=True)
class DrafterConfig:
    features: tuple            # target layers whose hidden states the drafter reads
    layers: int = LAYERS
    block: int = BLOCK
    markov_rank: int = MARKOV_RANK

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))
        if not self.features or self.layers < 1 or self.block < 1 or self.markov_rank < 1:
            raise ValueError("drafter config: features, layers, block and markov_rank must each be at least 1")

    @classmethod
    def for_model(cls, config: ModelConfig, features=None, layers=LAYERS, block=BLOCK, markov_rank=None):
        """The drafter of a model of that config, with the shape the model
        recipe's `drafter` section gives (what it leaves out: the feature
        layers of feature_layers, LAYERS, BLOCK, the Markov rank of the
        model's width). ValueError when a feature layer is not one of the
        model's (numbered from 1 to n_layer), so a run refuses it before it
        trains anything."""
        features = tuple(features) if features is not None else feature_layers(config.n_layer)
        outside = [n for n in features if not 1 <= n <= config.n_layer]
        if outside:
            raise ValueError(f"drafter config: features {list(features)}: {', '.join(map(str, outside))} "
                             f"{'is not a layer' if len(outside) == 1 else 'are not layers'} of a "
                             f"{config.n_layer}-layer model (layers 1 to {config.n_layer})")
        return cls(features, layers, block,
                   markov_rank if markov_rank is not None else default_markov_rank(config.d_model))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class DrafterLayer(nn.Module):
    """One backbone layer: window attention (V4's attention with no merged
    entries, as the multi-token prediction block's) over the context and
    the block, then routed and shared experts, each joined by mHC."""

    def __init__(self, config):
        super().__init__()
        D = config.d_model
        self.attention = Attention(config, ratio=0)
        self.experts = Experts(config, hashed=False)
        self.attention_norm, self.experts_norm = nn.RMSNorm(D, eps=EPS), nn.RMSNorm(D, eps=EPS)
        self.attention_streams, self.experts_streams = Connection(D), Connection(D)

    def forward(self, streams, ids, context, positions, context_positions, idx, collect):
        def attend(h):
            a = self.attention
            h = self.attention_norm(h)
            turning = frequencies(False, h.device)
            _, q, block = a._queries_and_window(h, positions, turning)
            injected = stored(rotate(a.kv_norm(a.kv(context.to(h.dtype))), context_positions, turning))
            table = torch.cat([block, injected.to(block.dtype)], 1)
            o, _ = sparse_attention(q, table, a.sink, idx, a.head_dim ** -0.5)
            return a._output(o, positions, turning, h.dtype)
        streams = self.attention_streams(streams, attend)
        return self.experts_streams(streams, lambda h: self.experts(self.experts_norm(h), ids, collect))


class Drafter(nn.Module):
    def __init__(self, model_config: ModelConfig, config: DrafterConfig):
        super().__init__()
        D, V = model_config.d_model, model_config.vocab_size
        if not all(1 <= n <= model_config.n_layer for n in config.features):
            raise ValueError(f"drafter config: features {config.features} are not layers of a "
                             f"{model_config.n_layer}-layer model")
        self.model_config, self.config = model_config, config
        self.features, self.window, self.block = config.features, WINDOW, config.block  # what wpt_inference reads
        self.fuse = nn.Linear(len(config.features) * D, D, bias=False)  # W_c
        self.fuse_norm = nn.RMSNorm(D, eps=EPS)
        self.mask = nn.Parameter(torch.randn(D))  # the masked places' input, at the embedding's scale
        self.layers = nn.ModuleList(DrafterLayer(model_config) for _ in range(config.layers))
        self.final_norm = nn.RMSNorm(D, eps=EPS)
        self.markov_in = nn.Embedding(V, config.markov_rank)  # W1
        self.markov_out = nn.Linear(config.markov_rank, V, bias=False)  # W2
        nn.init.zeros_(self.markov_out.weight)  # starts as the parallel backbone alone
        self.confidence = nn.Linear(D + config.markov_rank, 1)
        self.register_buffer("temperatures", torch.ones(config.block))  # STS's, one per position; 1 before calibrate
        self.provenance = {}  # what made it (provenance.json), filled by whoever trains or loads it

    def backbone(self, model, context, lengths, anchors, count, collect=False):
        """(h (R, count, D), U (R, count, vocab)) for R blocks of count
        places: context (R, WINDOW, features * D), the hidden states of the
        WINDOW tokens before each anchor (zeros before the text), lengths
        (R,) the anchors' positions, anchors (R,) their tokens."""
        if not 1 <= count <= self.block:
            raise ValueError(f"the drafter drafts 1 to {self.block} tokens, not {count}")
        R, device = len(anchors), anchors.device
        positions = lengths[:, None] + torch.arange(count, device=device)
        context_positions = lengths[:, None] - WINDOW + torch.arange(WINDOW, device=device)
        # each place sees the whole block, then the context entries within WINDOW tokens of it
        seen = (context_positions[:, None, :] >= 0) & (context_positions[:, None, :] > positions[..., None] - WINDOW)
        idx = torch.cat([torch.arange(count, device=device).expand(R, count, count),
                         torch.where(seen, count + torch.arange(WINDOW, device=device), -1)], -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=anchors.is_cuda):
            fused = self.fuse_norm(self.fuse(context))
            x = torch.cat([F.embedding(anchors, model.embedding.weight.detach())[:, None],
                           self.mask.expand(R, count - 1, -1).to(model.embedding.weight.dtype)], 1)
            ids = torch.cat([anchors[:, None], anchors.new_zeros(R, count - 1)], 1)
            streams = x.float().unsqueeze(2).expand(-1, -1, STREAMS, -1)
            for layer in self.layers:
                streams = layer(streams, ids, fused, positions, context_positions.clamp(min=0), idx, collect)
            h = self.final_norm(streams.sum(2))
            return h.float(), F.linear(h, model.head.weight.detach()).float()

    def _markov(self, previous):
        """(the Markov bias (..., vocab), W1[previous] (..., r))."""
        embedded = self.markov_in(previous)
        return self.markov_out(embedded).float(), embedded

    def _confidence_logits(self, h, embedded):
        return self.confidence(torch.cat([h, embedded.to(h.dtype)], -1)).squeeze(-1).float()

    @torch.no_grad()
    def draft(self, model, histories, context, count, probabilities, generator):
        """wpt_inference's drafter: count tokens after each of the R token
        lists (histories, each ending with its anchor), with context the
        (features, lengths) the writer keeps; probabilities turns scores
        into the distribution the writer samples from (its temperature and
        top k), and each draft is drawn from it. Returns (tokens (R, count),
        their distributions (R, count, vocab), confidences (R, count))."""
        features, lengths = context
        anchors = torch.tensor([history[-1] for history in histories], dtype=torch.long, device=lengths.device)
        h, U = self.backbone(model, features, lengths, anchors, count)
        tokens, distributions, confidences = [], [], []
        previous = anchors
        for k in range(count):
            bias, embedded = self._markov(previous)
            q = probabilities(U[:, k] + bias)
            previous = torch.multinomial(q, 1, generator=generator)[:, 0]
            confidences.append(torch.sigmoid(self._confidence_logits(h[:, k], embedded) / self.temperatures[k]))
            tokens.append(previous)
            distributions.append(q)
        return torch.stack(tokens, 1), torch.stack(distributions, 1), torch.stack(confidences, 1)

    def losses(self, model, context, lengths, anchors, following, target, valid):
        """DSpark's training losses for R blocks of self.block places (one
        anchor each, as backbone): following (R, block) the tokens that
        follow each anchor (the model's own answer), target (R, block, vocab)
        the model's distribution of each of them, valid (R, block) where a
        token follows. Each place k (from 0) weighs exp(-k / block), and
        each term is summed over places and averaged over blocks. Returns
        (total, {"ce", "tv", "confidence"}), with
        total = 0.1 ce + 0.9 tv + 1.0 confidence."""
        n = self.block
        h, U = self.backbone(model, context, lengths, anchors, n, collect=True)
        previous = torch.cat([anchors[:, None], following[:, :-1]], 1)  # x_{k-1}: the real tokens before each place
        bias, embedded = self._markov(previous)
        log_q = F.log_softmax(U + bias, -1)
        weights = torch.exp(-torch.arange(n, device=h.device) / n) * valid
        ce = -log_q.gather(-1, following[..., None]).squeeze(-1)
        tv = (log_q.exp() - target).abs().sum(-1)  # ||p_d - p_t||_1
        accepted = (1 - tv / 2).detach().clamp(0, 1)  # c*: the probability a draft here is kept
        confidence = F.binary_cross_entropy_with_logits(self._confidence_logits(h, embedded), accepted,
                                                        reduction="none")
        parts = {name: (value * weights).sum() / len(anchors) for name, value in
                 (("ce", ce), ("tv", tv), ("confidence", confidence))}
        return sum(LOSS_WEIGHTS[name] * value for name, value in parts.items()), parts

    def optimizer_roles(self):
        """(name, parameter, role, view) for every parameter, by the model's
        rule (optimizer.roles, WPT.optimizer_roles): the fusing projection
        and the attention and expert matrices take Muon, head by head and
        expert by expert; the mask vector, the norms, mHC's biases and
        gating factors, the Markov head's two matrices and the confidence
        head take AdamW, as the model's embedding and output head."""
        return roles(self, (self.mask, self.markov_in.weight, self.markov_out.weight, self.confidence.weight))

    def after_update(self, reduce):
        """After each optimizer step: Quantile Balancing sets the experts'
        routing biases, as the model's (WPT.after_update)."""
        for layer in self.layers:
            layer.experts.balance(reduce)


def training_blocks(features, sequence, scores, anchors, block):
    """The blocks drafter.losses reads from one sequence the model read
    whole: features (T, features * D) and scores (T, vocab) from its
    forward pass, sequence (T,) its tokens, anchors (R,) the positions whose
    token starts a block. Returns (context, lengths, anchors' tokens,
    following, target, valid)."""
    T, device = len(sequence), sequence.device
    window = anchors[:, None] - WINDOW + torch.arange(WINDOW, device=device)
    context = torch.where((window >= 0)[..., None], features[window.clamp(min=0)], 0.0)
    places = anchors[:, None] + torch.arange(block, device=device)  # each place's own position
    valid = places + 1 < T
    following = torch.where(valid, sequence[(places + 1).clamp(max=T - 1)], 0)
    target = F.softmax(scores[places.clamp(max=T - 1)].float(), -1)
    return context, anchors, sequence[anchors], following, target, valid.float()


@torch.no_grad()
def calibrate(logits, kept, valid, bins, grid):
    """DSpark's Sequential Temperature Scaling, on real outcomes: for each
    position k from the first, the temperature among grid that minimizes
    the expected calibration error (bins equal-width bins) of the product
    of the confidences up to k against whether drafts 0 to k were all kept,
    over the rounds that verified a draft at k, the earlier positions'
    temperatures fixed. logits: the confidence head's logits before its
    temperatures; kept: 1 where the model kept the draft, 0 where it
    rejected it or one before it; valid: 1 where the round verified a draft
    there; each (N, block), one row per round. Since kept is what happened,
    the product of kept up to k is 1 exactly when a prefix of k + 1 drafts
    was kept, and a calibrated product is that chance. Returns
    (temperatures (block,), the error per position before, and after)."""
    temperatures, before, after = [], [], []
    product = torch.ones(len(logits), device=logits.device)
    observed = torch.ones_like(product)
    for k in range(logits.shape[1]):
        keep = valid[:, k] > 0
        observed = observed * kept[:, k]
        if not keep.any():
            temperatures.append(1.0)
            before.append(None)
            after.append(None)
            continue
        errors = [_ece(product[keep] * torch.sigmoid(logits[keep, k] / t), observed[keep], bins) for t in grid]
        best = min(range(len(grid)), key=lambda i: errors[i])
        before.append(_ece(product[keep] * torch.sigmoid(logits[keep, k]), observed[keep], bins))
        after.append(errors[best])
        temperatures.append(float(grid[best]))
        product = product * torch.sigmoid(logits[:, k] / grid[best])
    return torch.tensor(temperatures), before, after


def _ece(predicted, observed, bins):
    """Expected calibration error: over equal-width bins of the predicted
    probabilities, the gap between the mean prediction and the mean observed
    value, weighed by the bin's share."""
    index = (predicted * bins).long().clamp(max=bins - 1)
    count = torch.bincount(index, minlength=bins).float()
    gap = (torch.bincount(index, predicted, minlength=bins) - torch.bincount(index, observed, minlength=bins)).abs()
    return float(gap.sum() / count.sum())


def temperature_grid(points=200, low=0.05, high=20.0):
    """The temperatures calibrate tries: points values, evenly spaced in
    log, from low to high."""
    return [math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (points - 1)) for i in range(points)]


def save_drafter(drafter: Drafter, folder, tokenizer_name: str, tokenizer_sha256: str, metadata=None, provenance=None):
    """Write folder/drafter/: config.json, model.safetensors (its header
    carries metadata, such as notice.WEIGHT_METADATA, and the tokenizer of
    the model folder, which the drafter shares) and provenance.json, which
    must name the weights it was trained against ({"target": {"weights":
    checkpoint.weight_hashes(folder)}}, checked by load_model). Replaces a
    drafter saved there."""
    directory = Path(folder) / DRAFTER_NAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_NAME).unlink(missing_ok=True)  # written last: a save that fails leaves no loadable drafter
    state = {k: v.detach().cpu().contiguous() for k, v in drafter.state_dict().items()}
    save_file(state, directory / WEIGHTS_NAME, metadata={**(metadata or {}), "tokenizer": tokenizer_name,
                                                         "tokenizer_sha256": tokenizer_sha256, "format": "pt"})
    (directory / PROVENANCE_NAME).write_text(json.dumps(provenance or {}, indent=2) + "\n", encoding="utf-8",
                                             newline="\n")
    drafter.config.save(directory / CONFIG_NAME)


def load_drafter(folder, device="cpu") -> Drafter:
    """The drafter saved in a model folder, on device, with its provenance
    (drafter.provenance). Raises checkpoint.DrafterMismatch when the folder
    has none or it was trained against other weights than the folder's,
    and checkpoint.TokenizerMismatch when its weights name another
    tokenizer than the model's."""
    folder = Path(folder)
    directory = folder / DRAFTER_NAME
    check_drafter(folder)
    with safe_open(directory / WEIGHTS_NAME, framework="pt") as weights:
        header = weights.metadata() or {}
    expected = tokenizer_identity(folder)
    if (header.get("tokenizer"), header.get("tokenizer_sha256")) != expected:
        raise TokenizerMismatch(f"{directory}: the drafter was saved with tokenizer {header.get('tokenizer')} "
                                f"({header.get('tokenizer_sha256')}), the model uses {expected[0]} ({expected[1]})")
    drafter = Drafter(ModelConfig.load(folder / "config.json"), DrafterConfig.load(directory / CONFIG_NAME))
    drafter.load_state_dict(load_file(directory / WEIGHTS_NAME))
    drafter.provenance = json.loads((directory / PROVENANCE_NAME).read_text(encoding="utf-8"))
    return drafter.to(device).eval()
