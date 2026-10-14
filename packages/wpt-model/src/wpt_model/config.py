"""The sizes that define a TinyGPT. Saved as config.json next to the weights,
so any saved model can be rebuilt without knowing how it was trained."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 256  # how many tokens the model sees at once
    n_layer: int = 4       # transformer blocks
    n_head: int = 4        # attention heads per block
    n_embd: int = 128      # embedding width
    dropout: float = 0.1

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
