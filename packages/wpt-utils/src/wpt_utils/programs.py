"""The outside programs WPT runs (mercurial, GnuPG, Bun, git), the commit
of the WPT repository a command runs from, and what identifies the code
itself.

program(name) is how every WPT package finds such a program: a missing one
raises MissingTool, which names it, instead of an error from deep inside a
command. code() is the one place that reads the repository commit: what
a run's record and a model's provenance name as the code that made
something, the commit of the checkout the code runs from and a hash of
its uncommitted changes. code_of(module) identifies only the code a
module runs: the sha256 of the source files of the WPT modules it
imports, found from its import statements (modules_code), which the
fingerprint of a result that module computes holds, so a change to a
document, a test, a module it does not import or a module that only
records and prints (RECORDING) leaves that fingerprint the same. repository() gives the commit, and whether there are changes,
in the shape provenance.json and each run's record name them.
"""

import functools
import shutil
import subprocess
from pathlib import Path


class MissingTool(Exception):
    """A program the work needs (mercurial for mercurial sources, GnuPG for
    GPG-signed commits, Bun for wpt web) is not installed."""


def program(name: str) -> str:
    """The absolute path of a program on the search path. Handing git an
    absolute path also matters on Windows, where Git for Windows would
    otherwise run its own bundled GnuPG, which rejects Windows paths."""
    path = shutil.which(name)
    if path is None:
        raise MissingTool(f"{name} is not installed (or not on the search path)")
    return path


def repository() -> dict:
    """{"repository_commit": the commit of the checkout this code runs
    from, "repository_dirty": whether it has uncommitted changes}, both
    from code(), so they never disagree with it; each None when the code
    is not in a git checkout."""
    found = code()
    if found["commit"] is None:
        return {"repository_commit": None, "repository_dirty": None}
    return {"repository_commit": found["commit"], "repository_dirty": found["changes"] is not None}


# code()'s answer, and code_of()'s for each module asked, found once per process, or given by the process that
# started this one (set_code), which then gives every module this one may ask
_CODE = None
_MODULES = {}
_GIVEN = False
WPT = "wpt_"  # how the top-level name of every module of WPT's libraries starts (none imports the app's `wpt`)
# The modules, and those inside a package named here, that modules_code neither hashes nor follows: they record and
# print, and compute no result, so a change to them must not make a result unknown.
RECORDING = ("wpt_telemetry", "wpt_utils.lines")


def code() -> dict:
    """{"commit": the commit of the WPT checkout this code runs from,
    "changes": the sha256 of its uncommitted changes (every tracked file's
    difference from that commit and every untracked file git does not
    ignore), None when there are none}: what a run's record and a model's
    provenance name as the code that made it, and what a fingerprint holds
    when any change to the checkout, its data included, must make a new
    one. Code that is not in a git checkout (installed as packages) has
    commit None and, as changes, the sha256 of every file of the installed
    wpt-* packages. Found once per process (set_code gives a worker process
    its starter's)."""
    global _CODE
    if _CODE is None:
        top = _checkout()
        commit = _git(top, "rev-parse", "HEAD").decode().strip() if top else ""
        _CODE = {"commit": commit, "changes": _changes(top)} if commit else \
            {"commit": None, "changes": _installed_code()}
    return dict(_CODE)


def code_of(module: str) -> str:
    """modules_code(module): the code a fingerprint holds for a result
    that module's code computes. Found once per process and module
    (set_code gives a worker process its starter's). In a process given
    its codes by set_code, a module they do not hold raises KeyError,
    since finding it there could disagree with the starter's."""
    if module not in _MODULES:
        if _GIVEN:
            raise KeyError(f"code_of({module!r}) was not given by the starting process (codes)")
        _MODULES[module] = modules_code(module)
    return _MODULES[module]


def codes(*modules: str) -> dict:
    """{"code": code(), "modules": {module: code_of(module)} of each of
    modules}: what set_code gives a worker process, modules being every
    module it may ask code_of for."""
    return {"code": code(), "modules": {module: code_of(module) for module in modules}}


def set_code(found: dict):
    """Use found (codes() of another process of the same run) as this
    process's code() and code_of(), so a worker process's fingerprints are
    its starter's, even when the files changed meanwhile."""
    global _CODE, _MODULES, _GIVEN
    _CODE, _MODULES, _GIVEN = dict(found["code"]), dict(found["modules"]), True


def fingerprint(inputs) -> str:
    """The sha256 of inputs as canonical JSON: the name of a saved result,
    inputs being everything that decides it."""
    import hashlib
    import json

    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_name(found=None) -> str:
    """code() as one line: the commit, then +<the first 12 hex digits of
    the changes> when there are some."""
    found = found or code()
    return (found["commit"] or "packages") + (f"+{found['changes'][:12]}" if found["changes"] else "")


def modules_code(module: str) -> str:
    """The sha256 of the code the WPT module named module runs, found from
    the import statements themselves: the source file (its bytes as they
    are on disk, each CRLF line end read as LF, so a Windows checkout gives
    Linux's code) of that module and of every WPT module (whose top-level
    name starts with WPT) it imports, directly or through another, the
    packages that hold them included, wherever in a module an import
    statement stands, except the modules of RECORDING (they record and
    print, and compute no result), which are neither hashed nor followed:
    what only they import leaves the key too. A compiled module among them enters by the files
    that build it (compiled_from), read as source files are; each other
    top-level module they import (not of the standard library) by the
    version of every installed distribution that provides it. The Python
    version (major.minor) enters too. Any other
    file, a document, a test or a module none of them imports, takes no
    part, so a change to it leaves this the same."""
    import hashlib
    import importlib.metadata
    import sys

    files, others = _imported(module)
    digest = hashlib.sha256(f"python {sys.version_info[0]}.{sys.version_info[1]}".encode())
    for name, path in sorted(files.items()):
        built = compiled_from(path) if not path.name.endswith(".py") else None
        for shown, file in built or [("", path)]:
            content = file.read_bytes()
            if built or path.name.endswith(".py"):
                content = content.replace(b"\r\n", b"\n")  # a source file: a Windows checkout's line ends are Linux's
            digest.update(f"\0{name}\0{shown}\0".encode() + hashlib.sha256(content).digest())
    providers = importlib.metadata.packages_distributions()
    for name in sorted(others - set(sys.stdlib_module_names)):
        versions = sorted(f"{d} {importlib.metadata.version(d)}" for d in set(providers.get(name, [])))
        digest.update(f"\0{name}\0{','.join(versions)}".encode())
    return digest.hexdigest()


def compiled_from(path: Path) -> list[tuple[str, Path]] | None:
    """The files that build the compiled module at path, each with its
    path from the project folder: those the file entries of the [tool.uv]
    cache-keys of the nearest pyproject.toml above it name (the files
    whose change makes uv build it again), in path order; None when it
    names none (a package installed outside its project), and the module
    then enters by its own bytes."""
    import tomllib

    for folder in path.parents:
        project = folder / "pyproject.toml"
        if project.is_file():
            settings = tomllib.loads(project.read_text(encoding="utf-8"))
            keys = settings.get("tool", {}).get("uv", {}).get("cache-keys", [])
            found = {file for key in keys if isinstance(key, dict) and "file" in key
                     for file in folder.glob(key["file"]) if file.is_file()}
            return sorted((file.relative_to(folder).as_posix(), file) for file in found) or None
    return None


def _imported(entry: str) -> tuple[dict[str, Path], set[str]]:
    """({name: its file} of the module entry and of every WPT module it
    imports, directly or through another, with the packages that hold
    them; the top-level names of the other modules they import)."""
    import ast

    found, others = {}, set()
    waiting = _with_packages(entry)
    while waiting:
        name = waiting.pop()
        if name in found:
            continue
        spec = _spec(name)
        if spec is None or not spec.has_location:
            continue  # a name a module defines, not a module of its own
        found[name] = path = Path(spec.origin)
        if not path.name.endswith(".py"):
            continue  # a compiled module: no import statements to read
        package = name if spec.submodule_search_locations is not None else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_bytes(), str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    held = package.split(".")[:len(package.split(".")) - node.level + 1]
                    base = ".".join([*held, *([node.module] if node.module else [])])
                names = [base, *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")]
            else:
                continue
            for imported in names:
                if imported.startswith(WPT):
                    if _recording(imported):
                        continue
                    waiting += _with_packages(imported)
                elif imported:
                    others.add(imported.split(".")[0])
    return found, others


def _recording(name: str) -> bool:
    """Whether the module name is one of RECORDING, or inside one."""
    return any(name == recording or name.startswith(recording + ".") for recording in RECORDING)


def _with_packages(name: str) -> list[str]:
    """A module's name and those of the packages that hold it, since
    importing it runs each of them."""
    parts = name.split(".")
    return [".".join(parts[:end]) for end in range(1, len(parts) + 1)]


def _spec(name: str):
    """Where the module name lies (its importlib spec), found without
    importing it or its packages; None when there is no such module."""
    import importlib.machinery
    import importlib.util

    top, *rest = name.split(".")
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return None
    for end in range(1, len(rest) + 1):
        if spec is None or spec.submodule_search_locations is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(".".join([top, *rest[:end]]),
                                                        list(spec.submodule_search_locations))
    return spec


def _checkout() -> Path | None:
    """The top folder of the git checkout this code runs from, or None."""
    top = _git(Path(__file__).parent, "rev-parse", "--show-toplevel").decode().strip()
    return Path(top) if top else None


def _changes(top) -> str | None:
    """The sha256 of the uncommitted changes of the git checkout at top,
    every tracked file's difference from its commit and every untracked
    file git does not ignore; None when there are none."""
    import hashlib

    top = Path(top)
    digest, changed = hashlib.sha256(), False
    difference = _git(top, "diff", "HEAD", "--binary", "--no-color", "--no-ext-diff", "--no-textconv")
    if difference:
        digest.update(difference)
        changed = True
    for path in sorted(filter(None, _git(top, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))):
        try:
            content = (top / path.decode()).read_bytes()
        except OSError:
            continue  # gone meanwhile, or a folder git lists
        digest.update(b"\0" + path + b"\0" + hashlib.sha256(content).digest())
        changed = True
    return digest.hexdigest() if changed else None


def _git(top, *command) -> bytes:
    """What a git command run in the folder top prints; b"" when it fails."""
    try:
        return subprocess.run([program("git"), "-C", str(top), *command], capture_output=True, check=True).stdout
    except (MissingTool, OSError, subprocess.CalledProcessError):
        return b""


@functools.cache
def _installed_code() -> str:
    import hashlib
    import importlib.metadata

    digest = hashlib.sha256()
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"] or ""):
        name = distribution.metadata["Name"] or ""
        if not name.startswith("wpt-"):
            continue
        for file in sorted(distribution.files or [], key=str):
            if file.suffix == ".pyc":
                continue
            try:
                content = file.locate().read_bytes()
            except OSError:
                continue
            digest.update(f"\0{name}/{file}\0".encode() + hashlib.sha256(content).digest())
    return digest.hexdigest()
