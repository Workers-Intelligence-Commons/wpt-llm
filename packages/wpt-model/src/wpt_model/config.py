"""The numbers that define a WPT model (docs/model.md). Saved as config.json
next to the weights, so any saved model can be rebuilt without knowing how
it was trained. What docs/model.md fixes for every size (4 mHC streams,
gated MLA every 4th layer and the last, hash routing in the first 3 layers,
2 shared experts) is in the code, not here."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int         # tokens per training window (the model itself has no length limit)
    n_layer: int
    d_model: int            # width of the residual streams
    kda_heads: int
    kda_head_dim: int       # keys and values of a KDA head
    mla_heads: int
    mla_head_dim: int       # queries, keys and values of an MLA head
    mla_kv_rank: int        # width of the compressed key-value vector c_t
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
        problems = [f"{f.name} must be at least 1" for f in fields(self)
                    if f.name != "engram_layers" and getattr(self, f.name) < 1]
        if self.n_layer < 2:
            problems.append("n_layer must be at least 2: KDA layers carry the position information, and the last "
                            "layer is MLA")
        if self.block_size < 2:
            problems.append("block_size must be at least 2: multi-token prediction reads the token after next")
        if not self.top_k < self.n_experts:
            problems.append("top_k must be smaller than n_experts: Quantile Balancing reads the (k+1)-th score")
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
