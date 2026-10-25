"""The store's failures (see wpt_store)."""


class StoreError(Exception):
    pass


class StoreSettingError(StoreError):
    pass


class MissingObject(StoreError):
    pass


class CorruptObject(StoreError):
    pass


class MixedRecord(StoreError):
    """A record file is written whole or added to line by line, never both."""
