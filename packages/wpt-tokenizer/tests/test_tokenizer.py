import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from wpt_tokenizer import FIRST_MERGE, Tokenizer, Trainer

DOCUMENTS = [
    "the cat sat on the mat. the cat sat again.".encode(),
    "le chat est sur le tapis, le chat dort.".encode(),
    "the mat and the cat\n\n\tand the tapis".encode(),
]


def test_any_bytes_survive_encode_then_decode(tmp_path):
    # a vocabulary is reused on text it never saw: no byte may be dropped
    Tokenizer.train(DOCUMENTS, vocab_size=556).save(tmp_path / "vocab.json")
    tokenizer = Tokenizer.load(tmp_path / "vocab.json")

    unseen = "ünïcode 🙂 jamais vu\r\n".encode() + bytes(range(256))
    assert tokenizer.decode(tokenizer.encode(unseen)) == unseen
    seen = DOCUMENTS[0]
    ids = tokenizer.encode(seen)
    assert len(ids) < len(seen)  # learned merges shorten known text
    assert tokenizer.decode(ids) == seen


def test_training_gives_the_same_vocabulary_whatever_the_document_order():
    # the vocabulary's sha256 is locked, so training must be repeatable
    forward = Tokenizer.train(DOCUMENTS, vocab_size=576).to_bytes()
    backward = Tokenizer.train(reversed(DOCUMENTS), vocab_size=576).to_bytes()
    assert forward == backward
    assert Tokenizer.from_bytes(forward).to_bytes() == forward


def test_end_token_marks_documents_and_decodes_to_nothing():
    tokenizer = Tokenizer.train(DOCUMENTS, vocab_size=556)
    ids = tokenizer.encode(b"one") + [Tokenizer.END] + tokenizer.encode(b"two")
    assert tokenizer.decode(ids) == b"onetwo"


def test_threads_encoding_at_once_give_what_one_thread_gives():
    # the build encodes documents on several threads, each with its own cache;
    # a dataset's tokens must not depend on how many ran at once
    tokenizer = Tokenizer.train(DOCUMENTS, vocab_size=576)
    texts = [document * (i + 1) for i, document in enumerate(DOCUMENTS * 20)]
    alone = [Tokenizer.train(DOCUMENTS, vocab_size=576).encode(text) for text in texts]  # a fresh cache each
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(tokenizer.encode, texts)) == alone


# -- counting chunks in a bounded table ---------------------------------------------------------------------

SPACE = b" \t\n\r\x0b\x0c"


def reference_chunks(data: bytes) -> list[bytes]:
    """The chunks of a text, from the rule the tokenizer states: an optional single leading space followed by
    bytes that are not ASCII whitespace, or a run of whitespace (less the space that starts the next word)."""
    n, out, i = len(data), [], 0

    def word_start(j):
        return data[j] == 0x20 and j + 1 < n and data[j + 1] not in SPACE

    while i < n:
        start = i
        if word_start(i) or data[i] not in SPACE:
            i += 1
            while i < n and data[i] not in SPACE:
                i += 1
        else:
            while i < n and data[i] in SPACE and not word_start(i):
                i += 1
        out.append(data[start:i])
    return out


def reference_merges(counts: Counter, n_merges: int) -> list[tuple[int, int]]:
    """Byte-pair merges learned from exact chunk counts by recounting every pair after each merge: the most
    frequent pair, the smaller (left, right) on a tie, until no pair occurs twice."""
    words = {chunk: list(chunk) for chunk in counts}
    merges = []
    while len(merges) < n_merges:
        pairs = Counter()
        for chunk, word in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] += counts[chunk]
        if not pairs:
            break
        pair, best = min(pairs.items(), key=lambda item: (-item[1], item[0]))
        if best < 2:
            break
        new = FIRST_MERGE + len(merges)
        merges.append(pair)
        for chunk, word in words.items():
            merged, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                    merged.append(new)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[chunk] = merged
    return merges


def corpus() -> list[bytes]:
    """Documents of words drawn so that a few are frequent and most are rare."""
    draw = random.Random(7)
    words = ["".join(draw.choice("abcdefghij") for _ in range(draw.randint(1, 7))) for _ in range(300)]
    weights = [1 / (rank + 1) for rank in range(len(words))]
    return [" ".join(draw.choices(words, weights, k=60)).encode() + b"\n" for _ in range(40)]


def true_counts(documents) -> Counter:
    return Counter(chunk for document in documents for chunk in reference_chunks(document))


def test_with_enough_slots_the_counts_and_the_merges_are_the_exact_ones():
    documents = corpus()
    expected = true_counts(documents)
    for slots in (len(expected), len(expected) + 1000):
        trainer = Trainer(slots)
        for document in documents:
            trainer.add(document)
        assert trainer.entries() == sorted((chunk, count, 0) for chunk, count in expected.items())
        assert trainer.train(100) == reference_merges(expected, 100)
    assert [tuple(m) for m in Tokenizer.train(documents, FIRST_MERGE + 100).merges] == reference_merges(expected, 100)


def test_with_few_slots_every_document_is_read_and_each_count_is_within_its_bound():
    # the vocabulary is trained from approximate counts of rare chunks, so the bound must hold, the frequent
    # chunks must be kept, and a second run over the same order must give the same table and merges
    documents = corpus()
    expected = true_counts(documents)
    slots = len(expected) // 4

    def run():
        trainer = Trainer(slots)
        for document in documents:
            trainer.add(document)
        return trainer

    trainer = run()
    held = trainer.entries()
    assert len(held) == slots
    assert sum(count for _, count, _ in held) == sum(expected.values())  # every chunk of every document counted
    for chunk, count, error in held:
        assert count - error <= expected[chunk] <= count
    smallest = min(count for _, count, _ in held)
    kept = {chunk for chunk, _, _ in held}
    assert {chunk for chunk, true in expected.items() if true > smallest} <= kept  # nothing frequent is lost
    assert smallest < max(expected.values())
    again = run()
    assert again.entries() == held and again.train(100) == trainer.train(100)


def test_any_number_of_threads_learns_the_same_merges():
    documents = corpus()
    expected = reference_merges(true_counts(documents), 100)
    trainer = Trainer()
    for document in documents:
        trainer.add(document)
    assert trainer.train(100) == expected
    assert [trainer.train(100, threads=threads) for threads in (1, 2, 7)] == [expected] * 3
    with pytest.raises(ValueError, match="at least 1 thread"):
        trainer.train(100, threads=0)


def test_a_trainer_needs_a_slot():
    with pytest.raises(ValueError, match="at least 1 slot"):
        Trainer(0)
