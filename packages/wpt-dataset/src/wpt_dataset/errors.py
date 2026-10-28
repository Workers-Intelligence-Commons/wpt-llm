"""Every failure wpt-dataset reports. Callers tell them apart by class, never
by message; the message carries the details (which file, which path). A
program the work needs and does not find (mercurial, GnuPG) raises
wpt_utils.programs.MissingTool."""


class DatasetError(Exception):
    """Base class; nothing below leaves the store or the repository changed
    except objects added to the content-addressed store, which are harmless."""


class InvalidFile(DatasetError):
    """A JSON file is unreadable or does not match its schema."""


class RuleViolation(DatasetError):
    """A repository rule is broken (id, license list, patch names, recipes...)."""


class StaleFile(DatasetError):
    """A file changed on disk since the caller read it; nothing was written.
    Read it again and redo the change."""


class NotFound(DatasetError):
    """The repository holds no such file, source, conversation or task."""


class OutdatedSchema(DatasetError):
    """A file uses an older shape; `wpt-dataset migrate` rewrites it."""


class FetchFailed(DatasetError):
    """An address could not be turned into files (protocols/): raised for
    one address, and for a source when none of its addresses (main or
    mirror) gave its files (each address and its failure are listed). The
    common base of the protocol failures below, which name what happened;
    raised itself only for a failure none of them is (protocols/base.py
    lists those). A source whose addresses failed in different ways fails
    with the kind its main address failed with. Nothing of that download
    is kept."""


class SourceGone(FetchFailed):
    """Upstream says it does not hold what the address names: an HTTP
    answer 404 or 410, a git repository that is not there, an LFS object
    its host does not hold."""


class TooManyRequests(FetchFailed):
    """Upstream refused the requests as too many (HTTP 429), through every
    try."""


class Incomplete(FetchFailed):
    """What upstream sent stopped before its end (an answer shorter than
    its stated length, a cut git transfer), or did not arrive intact (an
    LFS object whose size or sha256 is not its pointer's)."""


class AccessRefused(FetchFailed):
    """Upstream refused access to what the address names: an HTTP answer
    401, 403 or 451, or git or Mercurial asking for a password (git and
    Mercurial run so they never wait for one)."""


class HostUnreachable(FetchFailed):
    """The address's host could not be reached, through every try: its
    name did not resolve, or no connection to it could be made."""


class ServerError(FetchFailed):
    """The address's server answered with an error of its own (HTTP 5xx),
    through every try when the error may pass."""


class RobotsUnreadable(FetchFailed):
    """The robots.txt of an address's site could not be read: its server
    answered something other than the file or "no such file" (404, 410),
    so the site's rules are unknown and the address is not used. Nothing
    was downloaded from it."""


class CrawlForbidden(FetchFailed):
    """The robots.txt of an address's site forbids it to every crawler, to
    wpt-dataset, or to an AI training crawler. Nothing was downloaded from
    it. Raised for one address, and for a source whose main address was
    refused this way."""


class NotDownloaded(DatasetError):
    """The store does not hold the source's upstream files: `wpt-dataset
    download` fetches them. Nothing is downloaded anywhere else."""


class BadPath(DatasetError):
    """A path is not UTF-8, or two paths differ only by letter case."""


class UnsupportedEntry(DatasetError):
    """The downloaded data holds something other than files and symlinks
    (a device, a git submodule...)."""


class CutFailed(DatasetError):
    """An erasure cut does not apply: its file is not in the tree, or the
    passage it names is not in the file (erasure.py). The whole source is
    out until a person makes the cut again."""


class CutKeyMissing(CutFailed):
    """The store keeps no cut key (erasure.py), so no cut can be matched:
    every source with a cut is out until the key is restored or the cuts
    are made again."""


class PatchFailed(DatasetError):
    """A patch does not apply to the cut tree."""


class EditError(DatasetError):
    """`unpack` or `patch` cannot run: the folder lies inside the repository,
    `unpack`'s folder is not empty, `patch`'s folder holds something other
    than files and symlinks, or matches the source already. Nothing was
    written."""


class LicenseMapError(DatasetError):
    """A path matches no license entry, or more than one, or the entries
    covering a records file do not select its records by one field
    (license_check.py)."""


class FormatMapError(DatasetError):
    """A path matches no format entry, or more than one."""


class NotText(DatasetError):
    """A file is not text: not UTF-8, nor in an encoding it states that
    Python knows and its bytes decode under (encoding.py); exclude it in
    `extract` (it is likely an image or another binary file)."""


class PoolError(DatasetError):
    """A pool is empty, or filling its share needs more passes than allowed."""


class TokenizerVersionNeeded(DatasetError):
    """A tokenizer recipe, or the tokenizer code, changed since its version
    was locked; raise the recipe's `version` to train a new vocabulary."""


class VocabularyMissing(DatasetError):
    """The locked vocabulary is not in the store: a locked vocabulary is
    shared through the store that holds it, never trained again; use that
    store, or raise the recipe's `version` to train a new one."""


class DocumentListMissing(DatasetError):
    """Reading pointer tasks for the measures needs the model's dataset's
    document list, and that dataset predates document lists (or no dataset
    is named): such a model is used with the repository at the commit that
    built it, with no fallback."""


class RoundTripFailed(DatasetError):
    """A document does not decode back to its exact bytes."""


class SignatureError(DatasetError):
    """A commit lacks its DCO sign-off, or changes a decision (or the list of
    allowed keys) without a signature from an allowed key."""


class DataAccessRefused(RuntimeError):
    """Code outside the service layer (wpt_dataset/repo.py) opened, listed,
    removed or renamed a path under the repository's data folder; the audit
    hook repo.py sets refuses it before it happens (docs/architecture.md,
    "The service layer"). Not a DatasetError: no command treats it as a
    refusal of the data, it is a fault of the code."""
