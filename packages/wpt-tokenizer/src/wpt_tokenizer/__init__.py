"""Byte-level BPE tokenizer: every byte has a token, so nothing is ever dropped.

Ids 0-255 are the bytes, END (256) marks the end of a document, 257-512 are
reserved for special tokens (SPECIAL names those in use; see chat.py), and
each learned merge adds the next id from 513 on. Encoding never produces END
or a special token, and decoding gives them no bytes. The hot loops live in
Rust (_native).

A vocabulary file is canonical JSON, so the same merges always give the same
bytes, and its sha256 identifies the vocabulary:

    {"format":"wpt-bpe","merges":[[a,b],...],"tokenizer_version":"0.3.0"}

A vocabulary file written by another tokenizer version is refused: its
ids would mean other tokens.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from ._native import END, FIRST_MERGE, RUSTC_VERSION, VERSION, Codec, Trainer  # RUSTC_VERSION: the compiler that built _native

__version__ = VERSION
FORMAT = "wpt-bpe"
BASE_SIZE = FIRST_MERGE  # 256 bytes, END and the 256 reserved ids
# The special tokens in use, with their fixed ids; an id once given never
# changes, and a new one takes the next free id up to 512.
SPECIAL = {name: END + 1 + i for i, name in enumerate([
    "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tools>", "</tools>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"])}


class Tokenizer:
    END = END

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self._codec = Codec(self.merges)

    @classmethod
    def train(cls, documents: Iterable[bytes], vocab_size: int) -> "Tokenizer":
        """Learn up to vocab_size - 513 merges from the documents.

        Fewer merges are learned when no pair of tokens occurs twice anymore.
        """
        if vocab_size < BASE_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_SIZE} (256 bytes, END and 256 reserved ids), "
                             f"got {vocab_size}")
        trainer = Trainer()
        for document in documents:
            trainer.add(document)
        return cls(trainer.train(vocab_size - BASE_SIZE))

    @property
    def vocab_size(self) -> int:
        return self._codec.vocab_size

    def encode(self, data: bytes) -> list[int]:
        return self._codec.encode(data)

    def decode(self, ids) -> bytes:
        """Bytes of the tokens; END and the special tokens decode to nothing."""
        return self._codec.decode(list(ids))

    def to_bytes(self) -> bytes:
        document = {"format": FORMAT, "merges": [list(m) for m in self.merges], "tokenizer_version": VERSION}
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    @classmethod
    def from_bytes(cls, data: bytes) -> "Tokenizer":
        document = json.loads(data)
        if document.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} vocabulary file")
        if document.get("tokenizer_version") != VERSION:
            raise ValueError(f"vocabulary written by tokenizer {document.get('tokenizer_version')}; this is tokenizer "
                             f"{VERSION}, whose ids differ: rebuild the vocabulary (raise the tokenizer recipe's "
                             "version) and train a new model on it")
        return cls(document["merges"])

    def save(self, path):
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path) -> "Tokenizer":
        return cls.from_bytes(Path(path).read_bytes())
