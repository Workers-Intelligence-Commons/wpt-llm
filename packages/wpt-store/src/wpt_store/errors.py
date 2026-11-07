"""The store's failures (see wpt_store)."""


class StoreError(Exception):
    pass


class StoreSettingError(StoreError):
    pass


class SessionNeeded(StoreError):
    """A command that touches the store was given no session ID (--session,
    WPT_SESSION). Nothing was read or written."""


class MissingFile(StoreError):
    pass


class CorruptFile(StoreError):
    pass


class MixedRecord(StoreError):
    """A record file is written whole or added to line by line, never both."""
