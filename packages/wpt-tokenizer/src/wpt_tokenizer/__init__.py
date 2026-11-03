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
import threading
from collections.abc import Iterable
from pathlib import Path

from ._native import END, FIRST_MERGE, RUSTC_VERSION, SLOTS, VERSION, Codec, Trainer  # RUSTC_VERSION: the compiler that built _native

__version__ = VERSION
FORMAT = "wpt-bpe"
BASE_SIZE = FIRST_MERGE  # 256 bytes, END and the 256 reserved ids
WATCH_SECONDS = 5  # how often train tells its watcher how many merges it learned
# The special tokens in use, with their fixed ids; an id once given never
# changes, and a new one takes the next free id up to 512.
SPECIAL = {name: END + 1 + i for i, name in enumerate([
    "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tools>", "</tools>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<json>", "</json>"])}


class Tokenizer:
    END = END

    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self._codec = Codec(self.merges)

    @classmethod
    def train(cls, documents: Iterable[bytes], vocab_size: int, slots: int = SLOTS, merged=None) -> "Tokenizer":
        """Learn up to vocab_size - 513 merges from the documents.

        Fewer merges are learned when no pair of tokens occurs twice anymore.
        Every document is read, but at most `slots` distinct chunks are
        counted at once (the Trainer's table): a corpus with more distinct
        chunks than that has the counts of its rarer chunks approximate, and
        the vocabulary then depends on the order of the documents. The default
        is a fixed number, so a vocabulary never depends on the machine.
        merged, when given, is called every few seconds while the merges are
        learned, with the number of merges learned so far.
        """
        if vocab_size < BASE_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_SIZE} (256 bytes, END and 256 reserved ids), "
                             f"got {vocab_size}")
        trainer = Trainer(slots)
        for document in documents:
            trainer.add(document)
        if merged is None:
            return cls(trainer.train(vocab_size - BASE_SIZE))
        done = threading.Event()

        def watch():
            while not done.wait(WATCH_SECONDS):
                merged(trainer.merged())

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            return cls(trainer.train(vocab_size - BASE_SIZE))
        finally:
            done.set()
            watcher.join()

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
