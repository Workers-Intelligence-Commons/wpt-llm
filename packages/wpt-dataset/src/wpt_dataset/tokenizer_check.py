"""How a tokenizer recipe's vocabulary encodes every admitted document.

The vocabulary is the one the recipe's lock freezes (see build.py). Every
document of every admitted source is encoded, then decoded: it must give
back its exact bytes (RoundTripFailed otherwise). Any text encodes, since
every byte has a token; what changes is how compactly, which the report
gives per source as characters per token. A source that the vocabulary was
not trained on (added after it was locked) is marked, so a falling number
shows when raising the recipe's version would help.
"""

from dataclasses import dataclass

from .build import build_tokenizer, gather
from .errors import RoundTripFailed
from .repo import Repo, load_json


@dataclass
class SourceReport:
    source: str
    documents: int
    characters: int
    tokens: int
    in_vocabulary: bool  # some of its documents trained the vocabulary

    @property
    def characters_per_token(self) -> float:
        return self.characters / self.tokens if self.tokens else 0.0


def check_tokenizer(repo: Repo, name: str, store, log=print) -> list[SourceReport]:
    """One report per admitted source, in source order. Raises RoundTripFailed
    on the first document that does not decode back to its bytes."""
    tokenizer, _, _ = build_tokenizer(repo, name, store, log)
    trained = {doc_id.split("/", 1)[0] for doc_id, _ in load_json(repo.lock_path(name), "lock")["documents"]}
    documents, _, _ = gather(repo, store, log)
    reports = {}
    for document in documents:
        data = store.get_bytes(document.sha256)
        ids = tokenizer.encode(data)
        if tokenizer.decode(ids) != data:
            raise RoundTripFailed(f"tokenizer {name}: {document.id} does not decode back to its exact bytes")
        report = reports.setdefault(document.source, SourceReport(document.source, 0, 0, 0, document.source in trained))
        report.documents += 1
        report.characters += len(data.decode("utf-8"))
        report.tokens += len(ids)
    return list(reports.values())
