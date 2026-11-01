"""An archive the HTTP protocol unpacks (protocols/http.py) read while
it downloads, member by member, so no copy of the archive lies on the disk
when its server allows it.

    a GET of the address, its first bytes
    a zip's start (PK\\3\\4, or PK\\5\\6 for an empty zip)
        its size known                  -> the GET is closed; its last bytes asked for (a range request),
        (Content-Length)                   whether the server says it serves ranges (Accept-Ranges: bytes)
                                           or not:
            the range in answer         -> the table of contents, at the zip's end, then every member,
                                           read through range requests (Ranged)
            the whole file in answer    -> that answer saved to a file in tmp, read from there
        otherwise                       -> the answer saved to a file in tmp, read from there
    anything else                       -> read as a tar (compressed or not) while it arrives
        no tar header at its start      -> what came so far and the rest of the answer saved to a file in
                                           tmp, read as manifest.read_archive reads a file
        hard links to members not kept  -> the archive read again as it arrives, the members those links
        (manifest.LinkBack)                need held in memory: a tar read as it arrives cannot go back

A tar whose server gives the file's size (Content-Length), a validator
(ETag, else Last-Modified) and Accept-Ranges: bytes continues from the byte
it reached when its answer is cut (EndedEarly, a reset, a timeout), through
a range request tied to the file's version by If-Range (Cuts): the reader
sees one continuous stream. The continuing answer must be a 206 for exactly
those bytes (Content-Range); any other answer, such as the whole file
because it changed, fails the try, which then starts over from the first
byte as retry.call says. A cut uses up one of the tries only when it
brought no new bytes since the one before (or since the start); the waits
are retry.resume's. A zip read through range requests (Ranged) and a server
that gives no size, no validator or no ranges: the zip continues the same
way when it has a validator, and the tar starts over at each cut.

Each member goes to the extractor as it is read, whole in memory, as from
a file (manifest.read_tar_members, manifest.read_zip_members), so the tree and
its manifest are the ones a copy on the disk gives. Whatever
the store the extractor stores into (download.Bundled: the session's
bundles) is given a member keeps it; a try that fails leaves no entry in
the tree (Extractor.trying), and is tried again as retry.call says.

Failures: what urllib raises (a URLError for an answer that is not 2xx,
an OSError for a cut connection), Disagreed (a ConnectionError) for a
server whose ranges disagree with the file it serves, EndedEarly (a
ConnectionError) for an answer that ends before its Content-Length (urllib
gives the shorter answer without raising) and cannot continue,
an OSError for a tar that is not the same in its second read,
UnsupportedEntry for an answer that is neither a zip nor a tar, and
whatever the extractor raises for an entry.
"""

import http.client
import io
import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from . import manifest, retry, sites
from .errors import UnsupportedEntry

ZIP_STARTS = (b"PK\x03\x04", b"PK\x05\x06")
TIMEOUT = 120  # seconds an answer may stay silent
BLOCK = 1 << 20  # bytes read at once
# bytes read and dropped rather than asked for again when a zip's next member starts a little further on
SKIP = 1 << 20


def read(address: str, extractor: manifest.Extractor, tmp: Path, user_agent: str):
    """Every entry of the archive at address, added to extractor (see the
    module's docstring); a file it must save goes to tmp / "download"."""
    def attempt():
        with extractor.trying():
            _read(address, extractor, tmp / "download", user_agent)

    retry.call(attempt, address)


def get(address: str, user_agent: str, headers=None):
    """The answer to a GET of an http or https address (OSError for any
    other), urllib's, still to be read; sent once the site's limit per time
    allows (sites.paced)."""
    if urlparse(address).scheme not in ("http", "https"):
        raise OSError(f"{address}: http sources need an http:// or https:// address")
    request = urllib.request.Request(address, headers={"User-Agent": user_agent, **(headers or {})})
    sites.paced(address)
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def _read(address, extractor, target, user_agent):
    with get(address, user_agent) as answer:
        start = _first(answer, 4)
        if start not in ZIP_STARTS:
            return _read_tar(address, start, answer, extractor, target, user_agent)
        size = answer.headers.get("Content-Length", "")
        if not size.isdigit():  # a range request needs the size: Ranged checks each answer against it
            return manifest.read_archive(_save(start, answer, target, address), extractor)
        address, validator = answer.geturl(), _validator(answer.headers)
    with Ranged(address, int(size), user_agent, validator) as ranged:
        try:
            archive = zipfile.ZipFile(io.BufferedReader(ranged, BLOCK))
        except WholeAnswer as whole:  # the server serves no range after all: its answer is the whole file
            with whole.answer as answer:
                return manifest.read_archive(_save(b"", answer, target, address), extractor)
        except zipfile.BadZipFile:  # it only starts like a zip: the whole file, read as a copy on the disk is read
            with get(address, user_agent) as answer:
                return manifest.read_archive(_save(b"", answer, target, address), extractor)
        with archive:
            manifest.read_zip_members(archive, extractor)


def _read_tar(address, start, answer, extractor, target, user_agent):
    with Recorded(start, answer, address, user_agent) as body:
        try:
            archive = tarfile.open(fileobj=io.BufferedReader(body, BLOCK), mode="r|*")
        except tarfile.ReadError:  # perhaps a zip with bytes before it, as a file on the disk shows
            return manifest.read_archive(_save(body.recorded, body.answer, target, address), extractor)
        body.recorded = None
        try:
            _tar_members(archive, extractor, target)
        except manifest.LinkBack as links:
            body.close()
            with get(address, user_agent) as again, Recorded(b"", again, address, user_agent) as second:
                try:
                    archive = tarfile.open(fileobj=io.BufferedReader(second, BLOCK), mode="r|*")
                    _tar_members(archive, extractor, target, links.targets)
                except (tarfile.ReadError, UnsupportedEntry, manifest.LinkBack) as error:
                    raise OSError(f"{address}: the archive is not the one its first read gave") from error


def _tar_members(archive, extractor, target, held=frozenset()):
    """manifest.read_tar_members of a tar read as it arrives; a try that fails leaves no entry."""
    try:
        with extractor.trying(), archive:
            manifest.read_tar_members(archive, extractor, streamed=True, held=held)
    except tarfile.ReadError as error:
        raise UnsupportedEntry(f"{target.name}: neither a zip nor a tar archive") from error


def _first(answer, count) -> bytes:
    """The first count bytes of an answer (fewer when it is shorter)."""
    data = b""
    while len(data) < count and (block := answer.read(count - len(data))):
        data += block
    return data


def _save(start: bytes, answer, target: Path, address: str) -> Path:
    """start (the answer's bytes read already), then the rest of the
    answer, written to target; EndedEarly when it ends early."""
    with open(target, "wb") as out:
        out.write(start)
        shutil.copyfileobj(answer, out, BLOCK)
        whole(answer, out.tell(), address)
    return target


class EndedEarly(ConnectionError):
    """An answer ended before the length it stated: a ConnectionError, so
    retry.call tries it again."""


def whole(answer, count: int, address: str):
    """EndedEarly when count, the bytes an answer gave in all, falls short
    of its Content-Length."""
    size = answer.headers.get("Content-Length", "")
    if size.isdigit() and count < int(size):
        raise EndedEarly(f"{address}: the answer ended at byte {count} of {size}")


def _validator(headers) -> str | None:
    """What If-Range sends so that a range of another version of the file
    is never given: the strong ETag, else the Last-Modified date."""
    tag = headers.get("ETag")
    if tag and not tag.startswith("W/"):
        return tag
    return headers.get("Last-Modified")


class Cuts:
    """The cuts of one remote file's reading, which continue through range
    requests: a file of size bytes at address, of the version validator."""

    def __init__(self, address, size, user_agent, validator):
        self.address, self.size, self.user_agent, self.validator = address, size, user_agent, validator
        self.stalls = 0  # the cuts in a row that brought no new bytes

    def resume(self, error, position, since, sleep=time.sleep):
        """The answer continuing at position after error cut one that began
        at since: the same error when ATTEMPTS cuts in a row brought no new
        bytes; OSError (Disagreed) when the server's answer is not those
        bytes of this version."""
        while True:
            self.stalls = 0 if position > since else self.stalls + 1
            if self.stalls >= retry.ATTEMPTS:
                raise error
            retry.resume(self.address, error, position, self.stalls, sleep)
            try:
                return _ask_range(self.address, self.user_agent, position, self.size, self.validator)
            except Disagreed:
                raise
            except Exception as asked:
                if not retry.transient(asked):
                    raise
                error, since = asked, position


CUT = (OSError, http.client.HTTPException)  # what reading an answer raises when the connection is cut


class Recorded(io.RawIOBase):
    """An answer's bytes after start (the bytes read from it already),
    each byte read kept in `recorded` until it is set to None;
    EndedEarly when the answer ends before its Content-Length, unless the
    answer can continue (Cuts) and `recorded` is None (the answer is known
    to be a tar): the bytes go on from where it was cut. The
    answer in use is `answer`; closing closes it."""

    def __init__(self, start: bytes, answer, address: str, user_agent: str):
        self.start, self.answer, self.recorded = start, answer, bytearray(start)
        self.address, self.count, self.begun = address, len(start), 0
        size, headers = answer.headers.get("Content-Length", ""), answer.headers
        self.size = int(size) if size.isdigit() else None
        validator = _validator(headers)
        ranges = headers.get("Accept-Ranges", "").strip().lower() == "bytes"
        self.cuts = Cuts(answer.geturl(), self.size, user_agent, validator) \
            if self.size and validator and ranges and answer.status == 200 else None

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.start:
            count = min(len(buffer), len(self.start))
            buffer[:count], self.start = self.start[:count], self.start[count:]
            return count
        while True:
            try:
                count = self.answer.readinto(buffer)
                if count == 0 and len(buffer) and self.size is not None and self.count < self.size:
                    raise EndedEarly(f"{self.address}: the answer ended at byte {self.count} of {self.size}")
                break
            except CUT as error:
                if self.cuts is None or self.recorded is not None:  # not yet known to be a tar: _save reads on
                    raise
                self.answer.close()
                self.answer = self.cuts.resume(error, self.count, self.begun)
                self.begun = self.count
        self.count += count
        if self.recorded is not None:
            self.recorded += buffer[:count]
        return count

    def close(self):
        self.answer.close()
        super().close()


class WholeAnswer(Exception):
    """A range request answered with the whole file (200): its answer."""

    def __init__(self, answer):
        super().__init__("the server answered a range request with the whole file")
        self.answer = answer


class Disagreed(ConnectionError):
    """The answer to a range request is not those bytes of that file (a
    ConnectionError: a reader that continued a cut answer starts its try
    over)."""


def _ask_range(address, user_agent, position, size, validator, first=False):
    """The answer to a request for bytes position- of a file of size bytes
    (of the version validator, when given), checked to be exactly that
    (206 and its Content-Range): Disagreed for any other, and WholeAnswer
    for a 200 when first."""
    headers = {"Range": f"bytes={position}-"} | ({"If-Range": validator} if validator else {})
    answer = get(address, user_agent, headers)
    if answer.status == 200 and first:
        raise WholeAnswer(answer)
    expected = f"bytes {position}-{size - 1}/{size}"
    if answer.status != 206 or answer.headers.get("Content-Range", "").strip() != expected:
        answer.close()
        raise Disagreed(f"{address}: asked for bytes {position}- of {size}, got answer "
                        f"{answer.status} {answer.headers.get('Content-Range', '')}".strip())
    return answer


class Ranged(io.RawIOBase):
    """A remote file of size bytes read and sought as a local one, through
    range requests: one answer stays open while the reads go on from where
    it is, so reading in order asks once. Every answer must be the range
    asked of a file of that size and, with a validator, of that version:
    WholeAnswer when the first one is the whole file, Disagreed for any
    other answer. With a validator, a cut answer continues from the byte
    reached (Cuts)."""

    def __init__(self, address, size, user_agent, validator=None):
        self.address, self.size, self.user_agent, self.validator = address, size, user_agent, validator
        self.position, self.answer, self.at, self.begun, self.asked = 0, None, 0, 0, 0
        self.cuts = Cuts(address, size, user_agent, validator) if validator else None

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def readinto(self, buffer):
        if self.position >= self.size:
            return 0
        if self.answer is not None and self.at != self.position:
            if 0 < self.position - self.at <= SKIP:
                self.at += len(_first(self.answer, self.position - self.at))
            if self.at != self.position:
                self._close()
        if self.answer is None:
            self._ask()
        while True:
            try:
                count = self.answer.readinto(memoryview(buffer)[:self.size - self.position])
                if count == 0:
                    raise EndedEarly(f"{self.address}: the answer ended at byte {self.position} of {self.size}")
                break
            except CUT as error:  # the answer is cut: with a validator, it continues from this byte
                if self.cuts is None:
                    raise
                self._close()
                self.answer = self.cuts.resume(error, self.position, self.begun)
                self.at = self.begun = self.position
        self.position += count
        self.at = self.position
        return count

    def _ask(self):
        self.answer = _ask_range(self.address, self.user_agent, self.position, self.size, self.validator,
                                 first=self.asked == 0)
        self.asked += 1
        self.at = self.begun = self.position

    def _close(self):
        if self.answer is not None:
            self.answer.close()
            self.answer = None

    def close(self):
        self._close()
        super().close()
