import importlib
import importlib.machinery
import sys

import pytest

from wpt_utils import programs

def _package(root, files):
    for path, content in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(content)
    importlib.invalidate_caches()


def test_a_module_s_code_changes_with_every_module_it_imports_and_with_nothing_else(tmp_path, monkeypatch):
    # a stage's key holds the code of the modules its module imports, found
    # from the imports themselves: a change to any of them (one imported
    # from a function, or through another module, or the package that holds
    # them) makes a new one; a document, a test, a module nobody imports or
    # an untracked file elsewhere leaves it the same
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    _package(tmp_path, {
        "src/wpt_keyed/__init__.py": b"",
        "src/wpt_keyed/stage.py": b"from . import helper\n\ndef run():\n    import wpt_keyed.lazy\n",
        "src/wpt_keyed/helper.py": b"from .deep import value\nimport json\n",
        "src/wpt_keyed/deep.py": b"value = 1\n",
        "src/wpt_keyed/lazy.py": b"value = 2\n",
        "src/wpt_keyed/unrelated.py": b"value = 3\n",
        "README.md": b"# about\n",
        "tests/test_stage.py": b"import wpt_keyed.stage\n",
    })
    first = programs.modules_code("wpt_keyed.stage")

    _package(tmp_path, {"README.md": b"# about it\n", "tests/test_stage.py": b"import wpt_keyed.stage  # x\n",
                        "src/wpt_keyed/unrelated.py": b"value = 4\n", "notes.txt": b"left here\n"})
    assert programs.modules_code("wpt_keyed.stage") == first

    for path in ("src/wpt_keyed/stage.py", "src/wpt_keyed/helper.py", "src/wpt_keyed/deep.py",
                 "src/wpt_keyed/lazy.py", "src/wpt_keyed/__init__.py"):
        before = programs.modules_code("wpt_keyed.stage")
        _package(tmp_path, {path: (tmp_path / path).read_bytes() + b"# changed\n"})
        assert programs.modules_code("wpt_keyed.stage") != before, path


def test_a_compiled_module_enters_by_the_files_that_build_it(tmp_path, monkeypatch):
    # a compiled module (the tokenizer's Rust extension) has no imports to
    # read: it enters by the sources its project builds it from (the
    # cache-keys of its pyproject.toml), so a change to them makes a new
    # code, and the same sources built again on another machine give the
    # same one
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    native = f"src/wpt_built/_native{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    _package(tmp_path, {
        "pyproject.toml": b'[tool.uv]\ncache-keys = [{ file = "pyproject.toml" }, { file = "rust/src/**/*.rs" }]\n',
        "rust/src/lib.rs": b"fn one() {}\n",
        "src/wpt_built/__init__.py": b"from ._native import encode\n",
        native: b"built once",
    })
    first = programs.modules_code("wpt_built")
    _package(tmp_path, {native: b"built again, elsewhere"})
    assert programs.modules_code("wpt_built") == first
    _package(tmp_path, {"rust/src/lib.rs": b"fn two() {}\n"})
    assert programs.modules_code("wpt_built") != first


def test_a_checkout_with_windows_line_ends_has_linux_s_code(tmp_path, monkeypatch):
    # git on Windows may check text files out with CRLF line ends: the same
    # commit must key its results as on Linux, so the store's results are
    # shared between the machines
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    native = f"src/wpt_ends/_native{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    linux = {
        "pyproject.toml": b'[tool.uv]\ncache-keys = [{ file = "rust/src/**/*.rs" }]\n',
        "rust/src/lib.rs": b"fn one() {}\nfn two() {}\n",
        "src/wpt_ends/__init__.py": b"from . import stage\nfrom ._native import encode\n",
        "src/wpt_ends/stage.py": b"value = 1\n\ndef run():\n    return value\n",
        native: b"built",
    }
    _package(tmp_path, linux)
    first = programs.modules_code("wpt_ends")
    _package(tmp_path, {path: content.replace(b"\n", b"\r\n") for path, content in linux.items() if path != native})
    assert programs.modules_code("wpt_ends") == first


def test_a_module_s_code_changes_with_the_python_minor_version_only(tmp_path, monkeypatch):
    # the same files run by another Python minor version can give another result
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    _package(tmp_path, {"src/wpt_pyver/__init__.py": b"value = 1\n"})
    this = sys.version_info
    first = programs.modules_code("wpt_pyver")
    monkeypatch.setattr(sys, "version_info", (this[0], this[1], this[2] + 1, *this[3:]))
    assert programs.modules_code("wpt_pyver") == first  # another patch release
    monkeypatch.setattr(sys, "version_info", (this[0], this[1] + 1, 0, "final", 0))
    assert programs.modules_code("wpt_pyver") != first


def test_a_worker_uses_the_codes_its_starter_gives_and_refuses_a_module_not_given(monkeypatch):
    # a worker's fingerprints are its starter's even when the files changed
    # meanwhile, so it never finds a module's code itself
    for name in ("_CODE", "_MODULES", "_GIVEN"):  # put back as they were when the test ends
        monkeypatch.setattr(programs, name, getattr(programs, name))
    given = programs.codes("wpt_utils.programs")
    programs.set_code(given)
    assert programs.code_of("wpt_utils.programs") == programs.modules_code("wpt_utils.programs")
    assert programs.codes("wpt_utils.programs") == given
    with pytest.raises(KeyError):
        programs.code_of("wpt_utils.files")
