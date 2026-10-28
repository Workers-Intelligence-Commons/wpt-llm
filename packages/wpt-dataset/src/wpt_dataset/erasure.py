"""Erasure cuts (docs/erasure.md): a passage removed from one file of a
source's upstream content, named by its length and two keyed hashes, never
by its text, since the repository is public:

    length    the passage's length in bytes
    hmac      the HMAC-SHA256 of the passage under the cut key, in hex
    rolling   the passage's keyed rolling hash (below), an integer

The cut key is a secret kept in the store (the ref erasure-key/key, an
object of KEY_BYTES random bytes), so a guess at a short passage (a name, a
phone number) cannot be confirmed from the repository alone. Making a cut
needs it (cut: the key is made on the first cut); matching a cut needs it
too, and a store without it matches no cut (CutKeyMissing).

The rolling hash is a polynomial hash modulo the prime MODULUS whose
parameters come from the key: its base, and the number each byte value
stands for. For the bytes w0 ... w(L-1) of a run:

    rolling = sum of number(w_k) x base^(L-1-k), modulo MODULUS

Matching (runs) slides it over the whole file, every position at once,
part by part (numpy), and confirms each position where it equals the cut's
by the HMAC: every confirmed run is a place of the passage. apply removes
every one of them, left to right, a run overlapping one removed before it
staying; none confirmed is CutFailed, which takes the whole source out
until a person makes the cut again.

Nothing here reads or writes data/: cut gives the entry a person's
source.json holds, and the service layer writes it (cli.py).
"""

import hashlib
import hmac
import secrets

import numpy as np

from .errors import CutFailed, CutKeyMissing

KEY_KIND, KEY_NAME = "erasure-key", "key"  # the ref of the cut key in the store
KEY_BYTES = 32
MODULUS = (1 << 31) - 1  # a prime below 2^31, so every product of two residues fits in 63 bits
PART = 1 << 20  # positions of a file whose rolling hash is computed at once


def key(store) -> bytes:
    """The cut key the store keeps; CutKeyMissing when it keeps none."""
    held = store.get_ref(KEY_KIND, KEY_NAME)
    if held is None:
        raise CutKeyMissing(f"{store} keeps no cut key (the ref {KEY_KIND}/{KEY_NAME}): no cut can be matched; "
                            f"restore the key from its backup, or make every cut again")
    return store.get_bytes(held)


def made_key(store) -> bytes:
    """The store's cut key, made (KEY_BYTES random bytes) and kept when it
    has none: what the first cut does. Keep a backup of it: losing it makes
    every cut unmatchable."""
    if store.get_ref(KEY_KIND, KEY_NAME) is None:
        store.set_ref(KEY_KIND, KEY_NAME, store.put_bytes(secrets.token_bytes(KEY_BYTES)))
    return key(store)


def cut(passage: bytes, secret: bytes) -> dict:
    """The cut that names passage under the key secret: {length, hmac,
    rolling}. A passage is at least one byte."""
    if not passage:
        raise CutFailed("a cut removes at least one byte")
    return {"length": len(passage), "hmac": _hmac(secret, passage), "rolling": _Rolling(secret).of(passage)}


def runs(content: bytes, named: dict, secret: bytes) -> list[int]:
    """The start of every run of content that the cut names: each position
    where the keyed rolling hash of the next `length` bytes is the cut's,
    confirmed by its HMAC, in order (runs may overlap)."""
    length = named["length"]
    if length > len(content):
        return []
    rolling = _Rolling(secret)
    found = []
    for start in range(0, len(content) - length + 1, PART):
        part = content[start:start + PART + length - 1]
        for at in np.flatnonzero(rolling.each(part, length) == named["rolling"]):
            begin = start + int(at)
            if hmac.compare_digest(_hmac(secret, content[begin:begin + length]), named["hmac"]):
                found.append(begin)
    return found


def apply(content: bytes, named: dict, secret: bytes, where: str) -> bytes:
    """content with every run the cut names removed (runs), left to right,
    a run overlapping one removed before it staying. CutFailed naming where
    when no run is confirmed."""
    kept, end = [], 0
    for begin in runs(content, named, secret):
        if begin >= end:
            kept.append(content[end:begin])
            end = begin + named["length"]
    if not kept:
        raise CutFailed(f"{where}: the passage the cut names is not in the file")
    return b"".join(kept) + content[end:]


def _hmac(secret: bytes, data: bytes) -> str:
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


class _Rolling:
    """The rolling hash's parameters under one key: its base, and the number
    each byte value stands for, each derived from the key by HMAC."""

    def __init__(self, secret: bytes):
        def drawn(label: bytes, low: int) -> int:
            return low + int.from_bytes(hmac.new(secret, label, hashlib.sha256).digest()[:8]) % (MODULUS - low)

        self.base = drawn(b"rolling base", 2)
        self.numbers = np.array([drawn(b"rolling byte " + bytes([value]), 1) for value in range(256)], dtype=np.int64)
        self.inverse = pow(self.base, -1, MODULUS)

    def of(self, data: bytes) -> int:
        """The rolling hash of one run (Horner's rule)."""
        found = 0
        for value in data:
            found = (found * self.base + int(self.numbers[value])) % MODULUS
        return found

    def each(self, data: bytes, length: int) -> np.ndarray:
        """The rolling hash of the run of length bytes starting at each
        position of data that has one: with x the inverse of the base, the
        prefix sums S of number(d_j) x^j give each run's sum of
        number(d_(i+k)) x^(i+k) as S[i+length] - S[i], times base^(i+length-1)."""
        values = self.numbers[np.frombuffer(data, dtype=np.uint8)]
        count = len(values) - length + 1
        sums = np.zeros(len(values) + 1, dtype=np.int64)
        np.cumsum(values * _powers(self.inverse, len(values)) % MODULUS, out=sums[1:])
        sums %= MODULUS
        window = (sums[length:length + count] - sums[:count]) % MODULUS
        return window * _powers(self.base, count + length - 1)[length - 1:] % MODULUS


def _powers(base: int, count: int) -> np.ndarray:
    """base^0 ... base^(count-1) modulo MODULUS, doubling the list each step."""
    found = np.ones(max(count, 1), dtype=np.int64)
    done, step = 1, base
    while done < count:
        more = min(done, count - done)
        found[done:done + more] = found[:more] * step % MODULUS
        done += more
        step = step * step % MODULUS
    return found[:count]
