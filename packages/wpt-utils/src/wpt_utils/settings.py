"""A plug-in's settings: the defaults its file declares (SETTINGS), changed
by those a recipe gives, each of its default's type. A training method's
come from the recipe's training.settings, a post-training step's from its
`posttrain` entry (docs/training.md, docs/post-training.md)."""


def typed(defaults: dict, overrides: dict, where: str) -> dict:
    """defaults with the overrides, each of its default's type (an int where
    a float is expected is taken; text is converted). ValueError, starting
    with where, names an unknown setting or one of another type."""
    settings = dict(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"{where}: no setting {key!r} (settings: {', '.join(defaults) or 'none'})")
        kind = type(defaults[key])
        if isinstance(value, str) and kind is not str:
            try:
                value = (value.lower() in ("1", "true", "yes")) if kind is bool else kind(value)
            except ValueError:
                raise ValueError(f"{where}: setting {key}={value!r} is not a {kind.__name__}") from None
        elif kind is float and type(value) is int:
            value = float(value)
        if type(value) is not kind:
            raise ValueError(f"{where}: setting {key}={value!r} is not a {kind.__name__}")
        settings[key] = value
    return settings
