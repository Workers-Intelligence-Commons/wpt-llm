"""How a tokenizer recipe's vocabulary encodes every admitted document.

The vocabulary is the one the recipe's lock freezes (see build.py). Every
document of every admitted source is encoded, then decoded: it must give
back its exact bytes (RoundTripFailed otherwise). Any text encodes, since
every byte has a token; what changes is how compactly, which the report
gives per source as characters per token. A source that the vocabulary was
not trained on (added after it was locked) is marked, so a falling number
shows when raising the recipe's version would help.

Each source is read and encoded in a worker process, several at once, and
gives only its numbers back: the check writes no document to the store. A
document with the same Markdown as an earlier one counts once, for the
first source in id order, then the first path.
"""

import contextlib
import hashlib
from dataclasses import dataclass

from wpt_telemetry import Telemetry

from . import markdown, reading, workers
from . import trained as trained_documents
from .build import each_source, locked_vocabulary
from .errors import DatasetError, RoundTripFailed
from .kept import kept
from .repo import Repo
from .sources import copied_as_records, copy_size


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


def _measured(source, store, name: str, vocabulary: str, telemetry) -> list[tuple[bytes, int, int]]:
    """(sha256, characters, tokens) of each document of a source, each
    Markdown once, in a worker process. RoundTripFailed on the first that
    does not decode back to its bytes."""
    tokenizer, held = reading.tokenizer_of(vocabulary, store), kept(source, store, workers.license_texts(), telemetry)
    found, seen = [], set()
    for converted in markdown.documents(source, held.tree, store, workers.repo(), trains=held.trains):
        sha256 = hashlib.sha256(converted.markdown).digest()
        if sha256 not in seen:
            seen.add(sha256)
            ids = tokenizer.encode(converted.markdown)
            if tokenizer.decode(ids) != converted.markdown:
                raise RoundTripFailed(f"tokenizer {name}: {source.id}/{converted.path} does not decode back to its "
                                      f"exact bytes")
            found.append((sha256, len(converted.markdown.decode("utf-8")), len(ids)))
    return found


def check_tokenizer(repo: Repo, name: str, store, log=print) -> list[SourceReport]:
    """One report per admitted source that gives a document, in source
    order. Raises RoundTripFailed on a document that does not decode back to
    its bytes, the failure of any source that cannot be read, and
    DatasetError when the recipe has no lock: a dataset's build trains it."""
    locked = locked_vocabulary(repo, name, store)
    if locked is None:
        raise DatasetError(f"tokenizer {name} has no lock for its version: `wpt dataset build-tokenizer` on a dataset "
                           f"recipe that names it trains it")
    lock = repo.generated("lock", name)
    trained = {doc_id.split("/", 1)[0] for doc_id, _ in trained_documents.read(store, lock["documents_sha256"])}
    admitted = [source for source in repo.sources(log) if source.admitted]
    telemetry = Telemetry.off(echo=lambda text, error: log(text))
    reports, seen = {}, set()
    # closed at once when a source stops the others, so the sources still running end first
    with (telemetry.progress("tokenizer check", len(admitted), "sources") as progress,
          contextlib.closing(each_source(_measured, admitted, repo, store, telemetry, progress,
                                         lambda source: (name, locked[1]),
                                         lambda source: copy_size(source, store, copied_as_records(source)),
                                         store.cache)) as measured):
        for done, span in measured:
            with telemetry.finish(span):
                if done.error is not None:
                    raise done.error
                for sha256, characters, tokens in done.value:
                    if sha256 not in seen:
                        seen.add(sha256)
                        report = reports.setdefault(done.item.id, SourceReport(done.item.id, 0, 0, 0,
                                                                               done.item.id in trained))
                        report.documents += 1
                        report.characters += characters
                        report.tokens += tokens
    return list(reports.values())
