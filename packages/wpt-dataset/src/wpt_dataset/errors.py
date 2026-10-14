"""Every failure wpt-dataset reports. Callers tell them apart by class, never
by message; the message carries the details (which file, which path)."""


class DatasetError(Exception):
    """Base class; nothing below leaves the store or the repository changed
    except objects added to the content-addressed store, which are harmless."""


class InvalidFile(DatasetError):
    """A JSON file is unreadable or does not match its schema."""


class RuleViolation(DatasetError):
    """A repository rule is broken (id, license list, patch names, recipes...)."""


class OutdatedSchema(DatasetError):
    """A file uses an older shape; `wpt-dataset migrate` rewrites it."""


class FetchFailed(DatasetError):
    """No address (main or mirror) gave data with the expected checksum."""


class ChecksumMismatch(DatasetError):
    """The downloaded data has another manifest checksum than source.json says."""


class BadPath(DatasetError):
    """A path is not UTF-8, or two paths differ only by letter case."""


class UnsupportedEntry(DatasetError):
    """The downloaded data holds something other than files and symlinks
    (a device, a git submodule...)."""


class CutFailed(DatasetError):
    """An erasure cut does not apply: missing path, bad range, or wrong
    sha256 after the cut."""


class PatchFailed(DatasetError):
    """A patch does not apply to the cut tree."""


class EditError(DatasetError):
    """`unpack` or `patch` cannot run: the folder lies inside the repository,
    `unpack`'s folder is not empty, `patch`'s folder holds something other
    than files and symlinks, or matches the source already. Nothing was
    written."""


class LicenseMapError(DatasetError):
    """A path matches no license entry, or more than one."""


class NotText(DatasetError):
    """A document is not valid UTF-8; exclude it in `extract` (it is likely
    an image or another binary file)."""


class PoolError(DatasetError):
    """A pool is empty, or filling its share needs more passes than allowed."""


class LockMismatch(DatasetError):
    """A rebuilt vocabulary differs from the sha256 its lock file records."""


class MissingTool(DatasetError):
    """A program the work needs (mercurial for mercurial sources, GnuPG for
    GPG-signed commits) is not installed."""


class SignatureError(DatasetError):
    """A commit lacks its DCO sign-off, or changes a decision (or the list of
    allowed keys) without a signature from an allowed key."""
