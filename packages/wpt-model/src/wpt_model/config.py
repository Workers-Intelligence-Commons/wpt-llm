"""The numbers that define a WPT model (docs/model.md). Saved as config.json
next to the weights, so any saved model can be rebuilt without knowing how
it was trained. What docs/model.md fixes for every size (4 mHC streams,
a global layer every 4th layer and the last, the window, merging ratios,
rotary dimensions and YaRN, the entries' FP8 and FP4 formats, hash routing
in the first 3 layers, 2 shared experts) is in the code, not here."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .attention import ROPE_DIMS
from .formats import FP8_BLOCK

GLOBAL_EVERY = 4  # a global layer (CSA or HCA) every 4th layer and the last; KDA elsewhere
KINDS = ("CSA", "HCA")


def global_layers(n_layer):
    """The numbers (from 1) of the global layers of an n_layer model."""
    return [n for n in range(1, n_layer + 1) if n % GLOBAL_EVERY == 0 or n == n_layer]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int         # tokens per training window (the model itself has no length limit)
    n_layer: int
    global_attention: tuple  # "CSA" or "HCA" for each global layer, in order
    d_model: int            # width of the residual streams
    kda_heads: int
    kda_head_dim: int       # keys and values of a KDA head
    attention_heads: int    # query heads of CSA and HCA layers and the MTP block
    attention_head_dim: int  # each query head and the one key-value head; the last 64 dimensions rotate,
                             # the others are stored in FP8 blocks of 64
    query_rank: int         # width of the query latent c^Q, shared with the indexer
    output_groups: int      # groups of query heads in the grouped output projection
    output_rank: int        # width of each group's intermediate output
    index_heads: int        # query heads of the CSA indexer
    index_head_dim: int     # each indexer head and key, a power of 2 (the Hadamard rotation); the last 64 rotate
    index_top_k: int        # merged entries each query attends to in a CSA layer
    n_experts: int          # routed experts per layer
    top_k: int              # routed experts per token
    expert_width: int       # hidden width of a routed expert, in the latent space
    latent_width: int       # width of the space routed experts work in
    shared_width: int       # hidden width of each shared expert, at full width
    engram_layers: tuple    # layers (numbered from 1) that get Engram memory
    engram_order: int       # largest N-gram, N; orders 2 .. N are read
    engram_heads: int       # hash heads per order
    engram_dim: int         # width of one table row
    engram_table_size: int  # each table has a distinct prime size, the smallest at or above this

    def __post_init__(self):
        object.__setattr__(self, "engram_layers", tuple(self.engram_layers))
        object.__setattr__(self, "global_attention", tuple(self.global_attention))
        problems = [f"{f.name} must be at least 1" for f in fields(self)
                    if f.name not in ("engram_layers", "global_attention") and getattr(self, f.name) < 1]
        if self.n_layer < 2:
            problems.append("n_layer must be at least 2: the last layer is CSA or HCA, the first KDA")
        if self.block_size < 2:
            problems.append("block_size must be at least 2: multi-token prediction reads the token after next")
        if not self.top_k < self.n_experts:
            problems.append("top_k must be smaller than n_experts: Quantile Balancing reads the (k+1)-th score")
        expected = len(global_layers(self.n_layer))
        if len(self.global_attention) != expected or not set(self.global_attention) <= set(KINDS):
            problems.append(f"global_attention must name CSA or HCA for each of the {expected} global layers "
                            f"(layers {', '.join(map(str, global_layers(self.n_layer)))})")
        if not (self.attention_head_dim > ROPE_DIMS and (self.attention_head_dim - ROPE_DIMS) % FP8_BLOCK == 0):
            problems.append(f"attention_head_dim must be {ROPE_DIMS} rotary dimensions plus a multiple of {FP8_BLOCK}, "
                            "the FP8 blocks of the others")
        c = self.index_head_dim
        if not (c > ROPE_DIMS and c & (c - 1) == 0):
            problems.append(f"index_head_dim must be a power of 2 above {ROPE_DIMS}, for the Hadamard rotation")
        if self.attention_heads % self.output_groups:
            problems.append("attention_heads must be a multiple of output_groups")
        if self.engram_order < 2:
            problems.append("engram_order must be at least 2")
        if not self.engram_table_size < 2**31:
            problems.append("engram_table_size must be below 2^31, so hashing stays inside 64-bit integers")
        problems += [f"engram layer {n} is not between 1 and n_layer" for n in self.engram_layers
                     if not 1 <= n <= self.n_layer]
        if problems:
            raise ValueError("model config: " + "; ".join(problems))

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
