from concurrent.futures import ThreadPoolExecutor

from wpt_tokenizer import Tokenizer

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
