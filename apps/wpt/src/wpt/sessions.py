"""The options of every command that touches the store, and the store as
that command works in it (wpt_store.sessions).

    options(parser)                 adds --session ID and --from OLD_ID
    named(args, env_file)           (the session named, the one it starts from); SessionNeeded
                                    without a session
    optional(args, env_file)        the same, the session None when none is named
    opened(store, args, env_file)   the store at the session named; SessionNeeded without one
    viewed(store, args, env_file)   the same for a command that also works with no session, as the
                                    whole store: the store at the session named, else the store itself
    from_settings(store, env_file)  opened, from the settings alone
    fetching(store, args, env_file, log)
                                    the same for a command that downloads: at a new ID, which it
                                    prints, when none is named
"""

from wpt_store import StoreSettingError, given, needed, new_id, started
from wpt_utils.lines import say


def options(parser):
    parser.add_argument("--session", metavar="ID", help="the session this command works in (WPT_SESSION)")
    parser.add_argument("--from", dest="from_id", metavar="OLD_ID",
                        help="another session, which this one reads what it does not hold from; what it writes "
                             "stays in its own session and names OLD_ID (WPT_FROM)")


def _named(read, value, env_file):
    """What read (wpt_store's given, needed or started) finds; StoreSettingError
    for a name that cannot be an ID."""
    try:
        return read(value, env_file=env_file)
    except ValueError as error:
        raise StoreSettingError(str(error)) from None


def named(args, env_file):
    return _named(needed, args.session, env_file), _named(started, args.from_id, env_file)


def optional(args, env_file):
    """(the session named, the one it starts from), the session None when none
    is named; --from without a session is refused (StoreSettingError)."""
    session, start = _named(given, args.session, env_file), _named(started, args.from_id, env_file)
    if session is None and start is not None:
        raise StoreSettingError("--from OLD_ID needs a session: give --session ID or set WPT_SESSION")
    return session, start


def opened(store, args, env_file):
    return store.at(*named(args, env_file))


def viewed(store, args, env_file):
    """opened, except that no session named gives the store itself, which
    shows every session; --from without a session is refused (StoreSettingError),
    since it narrows nothing."""
    session, start = optional(args, env_file)
    return store if session is None else store.at(session, start)


def from_settings(store, env_file):
    """opened, for a program with no options of its own to read."""
    return store.at(_named(needed, None, env_file), _named(started, None, env_file))


def fetching(store, args, env_file, log=say):
    session = _named(given, args.session, env_file)
    if session is None:
        session = new_id()
        log(f"session {session}")
    return store.at(session, _named(started, args.from_id, env_file))
