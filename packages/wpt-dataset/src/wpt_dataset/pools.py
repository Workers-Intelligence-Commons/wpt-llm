"""Pool expressions: which documents a pool selects, by their tags.

    tag                  documents carrying that tag
    a && b               both
    a || b               either
    !( expr )            not; `!` must directly precede an opening parenthesis
    ( expr )             grouping

`!` binds tightest, then `&&`, then `||`. Tags use a-z, 0-9, '.', '_', '-'.
"""

import re

TOKEN = re.compile(r"\s*(?:(&&)|(\|\|)|(!\()|(\()|(\))|([a-z0-9][a-z0-9._-]*))")


def _tokens(text):
    pos, out = 0, []
    text = text.rstrip()
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if not match:
            raise ValueError(f"unexpected {text[pos:].strip()[:10]!r} at position {pos}"
                             + (" (`!` must be followed by `(`)" if text[pos:].lstrip().startswith("!") else ""))
        operator = match[1] or match[2] or match[3] or match[4] or match[5]
        out.append(operator if operator else ("tag", match[6]))
        pos = match.end()
    return out


def parse(text: str):
    """The expression as nested tuples: ("tag", t), ("and"/"or", a, b), ("not", a)."""
    tokens = _tokens(text)
    if not tokens:
        raise ValueError("empty expression")
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take(expected=None):
        nonlocal pos
        token = peek()
        if token is None or (expected and token != expected):
            raise ValueError(f"expected {expected or 'a tag'}, found {token or 'the end'}")
        pos += 1
        return token

    def parse_or():
        node = parse_and()
        while peek() == "||":
            take()
            node = ("or", node, parse_and())
        return node

    def parse_and():
        node = parse_unary()
        while peek() == "&&":
            take()
            node = ("and", node, parse_unary())
        return node

    def parse_unary():
        token = take()
        if token == "!(":
            node = parse_or()
            take(")")
            return ("not", node)
        if token == "(":
            node = parse_or()
            take(")")
            return node
        if isinstance(token, tuple):
            return token
        raise ValueError(f"expected a tag, found {token}")

    tree = parse_or()
    if pos != len(tokens):
        raise ValueError(f"unexpected {tokens[pos]} after a complete expression")
    return tree


def matches(tree, tags) -> bool:
    match tree:
        case ("tag", name):
            return name in tags
        case ("and", a, b):
            return matches(a, tags) and matches(b, tags)
        case ("or", a, b):
            return matches(a, tags) or matches(b, tags)
        case ("not", a):
            return not matches(a, tags)
    raise ValueError(f"not an expression tree: {tree!r}")
