"""Globs, the compiled pattern lists, decide every path as glob_match
(PurePosixPath.full_match) decides it, so no manifest changes."""

import itertools
import random

from wpt_dataset.manifest import Globs, PatternIndex, glob_match, glob_match_any

PARTS = ["a", "A", "b.txt", "B.TXT", ".hidden", "é", "ü.md", "x y", "[x]", "a-b", "**", "*", "?", "*.txt", "*.TXT",
         "a?", "[ab]", "[!a]", "[a-c]", "[[]x]", "[]]", "[*]", "[?]", "[", "a]", "*.*", ".*", "..", ".", "", "\\",
         "a*b*c", "[^a]", "[a-]", "[!]"]
PATHS = ["a", "A", "b.txt", "B.TXT", ".hidden", "é", "ü.md", "x y", "[x]", "x]", "a-b", "*", "?", "ab", "ba", "c",
         "abc", "aXbYc", "a.b", "]", "[", "*.txt", "\\", "..", "-", "^", "é"]


def _paths():
    found = set(PATHS)
    found.update("/".join(parts) for parts in itertools.product(PATHS[:12], repeat=2))
    found.update("/".join(parts) for parts in itertools.product(["a", ".hidden", "b.txt", "[x]"], repeat=3))
    return sorted(found | {"", ".", "a/", "/a", "//a", "a//b", "./a", "a/./b", "a/.", "/"})


def _patterns():
    random.seed(7)
    found = set(PARTS)
    found.update("/".join(parts) for parts in itertools.product(PARTS, repeat=2))
    found.update("/".join(random.choices(PARTS, k=random.randint(3, 5))) for _ in range(400))
    return sorted(found | {"/", "/a", "//a", "a/", "a//b", "./a"})


def test_globs_decide_every_path_as_full_match_does():
    paths, patterns = _paths(), _patterns()
    for pattern in patterns:  # one pattern alone: the set or the regular expression
        one = Globs([pattern])
        assert [p for p in paths if one.match(p)] == [p for p in paths if glob_match(p, pattern)], pattern
    random.seed(11)
    for _ in range(200):  # several joined: exact paths and wildcards together
        some = random.sample(patterns, random.randint(2, 40)) + random.sample(paths, 5)
        many = Globs(some)
        assert [p for p in paths if many.match(p)] == [p for p in paths if any(glob_match(p, g) for g in some)], some


def test_typical_source_rules():
    exclude = ["**/*.png", "docs/**", "vendor/", "README", "src/[[]gen]/*.c", "**/.*"]
    expected = {"a/b.png": True, "b.png": True, "a/b.PNG": False, "docs": False, "docs/x/y": True, "doc/x": False,
                "vendor": True, "vendor/x": False, "README": True, "readme": False, "src/[gen]/a.c": True,
                "src/g/a.c": False, ".git": True, "a/.env": True, "a.b/c": False}
    assert {p: glob_match_any(p, exclude) for p in expected} == expected
    assert {p: glob_match_any(p, []) for p in expected} == dict.fromkeys(expected, False)


def test_pattern_index_finds_every_list_that_matches():
    """The one matcher of a source's entries (formats, tags, license
    coverage) gives, for each path, exactly the lists glob_match_any
    matches, whatever shape their patterns have."""
    paths, patterns = _paths(), _patterns()
    paths += ["src/a.js", "src/x/a.js", "a.tar.gz", "x/a.tar.gz", "x/CMakeLists.txt", "CMakeLists.txt", ".md",
              "docs/.md", "a.", "src"]
    patterns += ["**/*.js", "src/*.js", "*.tar.gz", "**/*.gz", "**/CMakeLists.txt", "x/**/CMakeLists.txt", "**/*.md",
                 "*.", "src/**", "src/*", "**/*.[jt]s", "**/a*.js"]
    random.seed(13)
    lists = [random.sample(patterns, random.randint(1, 4)) for _ in range(300)]
    index = PatternIndex(lists)
    for path in paths:
        assert index.matching(path) == [n for n, found in enumerate(lists) if glob_match_any(path, found)], path
