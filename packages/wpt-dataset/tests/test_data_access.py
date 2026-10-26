"""Only the service layer (wpt_dataset/repo.py) reads or writes the
repository's data (docs/architecture.md, "The service layer").

The test reads the source of every package and application
(packages/*/src, apps/*/src) and of scripts/, and fails on any code outside
repo.py that names the data folder in a way that could open a file under
it:

- the name "data" as a part of a path: either side of `/`, or an argument
  of a call that builds or reaches a path (Path(), joinpath(),
  os.path.join(), os.walk(), os.listdir(), os.scandir(), glob(), open()
  and the like);
- a name bound to "data" (a constant such as DATA = "data"), and a
  comparison with "data" (such as p.name == "data");
- a string, or an f-string, that starts with "data/" or "data\\", or holds
  "/data/" or ":data/" (git show HEAD:data/...); a docstring, an argparse
  help= text, and a message (inside a raise, or given to print(), a log
  call or sys.exit()) name the folder for people and are left alone;
- a private member of the Repo class used on anything but self (its data
  folder and the methods that give paths under it), or a private name
  imported from repo.py.

A syntax scan cannot follow every way to build a path, so the service
layer adds a runtime check: its audit hook refuses any opening, listing,
removal or renaming under a Repo's data folder whose nearest caller in the
project's code is not repo.py or a test (test_the_audit_hook_refuses_...).
The tests themselves are left out of both: they build the repositories
they check and read the files a command wrote, to compare them.
"""

import ast
import os
from pathlib import Path

import pytest

from wpt_dataset.errors import DataAccessRefused
from wpt_dataset.repo import Repo

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "packages" / "wpt-dataset" / "src" / "wpt_dataset" / "repo.py"
# calls whose arguments are parts of a path, or a path they reach
PATH_CALLS = {"Path", "PurePath", "PurePosixPath", "PureWindowsPath", "joinpath", "join", "walk", "fwalk", "listdir",
              "scandir", "glob", "iglob", "rglob", "open", "exists", "isdir", "isfile", "stat", "lstat", "rmtree",
              "copytree", "copy", "copyfile", "move", "remove", "unlink", "rename", "replace", "mkdir", "makedirs"}
MESSAGE_CALLS = {"print", "exit", "log", "debug", "info", "warning", "warn", "error", "exception"}
SEEN_FROM_PATHS = ("/data/", ":data/", "\\data\\")


def code_files() -> list[Path]:
    found = [*ROOT.glob("packages/*/src/**/*.py"), *ROOT.glob("apps/*/src/**/*.py"), *ROOT.glob("scripts/*.py")]
    return sorted(path for path in found if path != SERVICE)


def private_members() -> set[str]:
    """The names of Repo's private methods and attributes."""
    tree = ast.parse(SERVICE.read_text(encoding="utf-8"))
    [repo] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Repo"]
    names = {node.name for node in repo.body if isinstance(node, ast.FunctionDef)}
    names |= {node.attr for node in ast.walk(repo) if isinstance(node, ast.Attribute)
              and isinstance(node.value, ast.Name) and node.value.id == "self"}
    return {name for name in names if name.startswith("_") and not name.endswith("__")}


def _call_name(node):
    return node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)


def findings(source: str, private: set[str]) -> list[tuple[int, str]]:
    """(line, what) of each way the source could reach a file under data/."""
    tree = ast.parse(source)
    spared = set()  # docstrings, help texts, messages
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            spared.add(id(node.body[0].value))
        elif isinstance(node, ast.keyword) and node.arg == "help" or isinstance(node, ast.Raise) \
                or isinstance(node, ast.Call) and _call_name(node) in MESSAGE_CALLS:
            spared |= {id(inner) for inner in ast.walk(node)}

    def data(node):
        return isinstance(node, ast.Constant) and node.value == "data"

    def names_a_path(text):
        return isinstance(text, str) and (text.startswith(("data/", "data\\")) or any(p in text for p in SEEN_FROM_PATHS))

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and (data(node.left) or data(node.right)):
            found.append((node.lineno, '"data" joined into a path'))
        elif isinstance(node, ast.Call) and _call_name(node) in PATH_CALLS and any(data(arg) for arg in node.args):
            found.append((node.lineno, f'"data" given to {_call_name(node)}()'))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and data(node.value):
            found.append((node.lineno, 'a name bound to "data"'))
        elif isinstance(node, ast.Compare) and any(data(part) for part in [node.left, *node.comparators]):
            found.append((node.lineno, 'a comparison with "data"'))
        elif isinstance(node, ast.JoinedStr) and id(node) not in spared:
            literal = "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in node.values)
            if names_a_path(literal):
                found.append((node.lineno, f"the path f{literal!r}"))
            spared |= {id(part) for part in node.values}
        elif isinstance(node, ast.Constant) and names_a_path(node.value) and id(node) not in spared:
            found.append((node.lineno, f"the path {node.value!r}"))
        elif isinstance(node, ast.Attribute) and node.attr in private \
                and not (isinstance(node.value, ast.Name) and node.value.id in ("self", "cls")):
            found.append((node.lineno, f"Repo's private member {node.attr}"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").endswith("repo") \
                and any(alias.name.startswith("_") for alias in node.names):
            found.append((node.lineno, "a private name imported from repo.py"))
    return sorted(found)


def test_only_the_service_layer_opens_files_under_data():
    private = private_members()
    assert "_data" in private and "_path" in private
    problems = [f"{path.relative_to(ROOT).as_posix()}:{line}: {what}"
                for path in code_files() for line, what in findings(path.read_text(encoding="utf-8"), private)]
    assert problems == [], "only wpt_dataset/repo.py may reach files under data/:\n" + "\n".join(problems)


def test_the_scan_finds_each_way_to_the_data_folder():
    # the rules above, each on a line that breaks it, and on lines that only name the folder for people
    private = private_members()
    code = '''"""Reads data/sources/ (a docstring)."""
import os
from pathlib import Path
from wpt_dataset.repo import Repo, _parts
DATA = "data"

def run(root, repo, name):
    """data/images/ in a docstring."""
    first = root / "data" / "sources"
    second = Path(root, "data")
    third = os.path.join(root, DATA)
    fourth = root.joinpath("data", "recipes")
    fifth = root / "data/allowed-signers"
    sixth = repo._data / "sources"
    seventh = repo._path("source", name)
    eighth = root / f"data/images/{name}/Dockerfile"
    ninth = open(f"{root}/data/allowed-signers")
    tenth = os.walk("data")
    eleventh = ["git", "show", f"HEAD:data/sources/{name}/source.json"]
    twelfth = [p for p in root.iterdir() if p.name == "data"]
    parser.add_argument("name", help="a folder of data/images/")
    print(f"data/images/{name} is missing")
    if not name:
        raise ValueError(f"data/images/{name} is missing")
    return document["data"], {"data": 1}
'''
    assert findings(code, private) == [
        (4, "a private name imported from repo.py"), (5, 'a name bound to "data"'), (9, '"data" joined into a path'),
        (10, '"data" given to Path()'), (12, '"data" given to joinpath()'), (13, "the path 'data/allowed-signers'"),
        (14, "Repo's private member _data"), (15, "Repo's private member _path"),
        (16, "the path f'data/images/{}/Dockerfile'"), (17, "the path f'{}/data/allowed-signers'"),
        (18, '"data" given to walk()'), (19, "the path f'HEAD:data/sources/{}/source.json'"),
        (20, 'a comparison with "data"')]


def test_the_audit_hook_refuses_data_access_from_code_outside_the_service_layer(tmp_path):
    # the runtime half: whatever way a path was built, code that is neither repo.py nor a test cannot open,
    # list or change a file under a Repo's data folder; through the service layer the same file reads
    root = tmp_path / "wpt"
    (root / "data" / "recipes" / "mixtures").mkdir(parents=True)
    signers = root / "data" / "allowed-signers"
    signers.write_text("# no key yet\n", encoding="utf-8")
    repo = Repo(root)
    elsewhere = str(tmp_path / "elsewhere" / "wpt_reader" / "reader.py")  # not a test, not repo.py
    for line in (f"open({str(signers)!r}).read()", f"__import__('os').listdir({str(root / 'data')!r})",
                 f"__import__('pathlib').Path({str(signers)!r}).write_text('x')"):
        with pytest.raises(DataAccessRefused, match="only the service layer"):
            exec(compile(line, elsewhere, "exec"), {})
    assert repo.read("allowed-signers", "allowed-signers").text == "# no key yet\n"
    exec(compile(f"open({str(tmp_path / 'other.txt')!r}, 'w').write('x')", elsewhere, "exec"), {})  # elsewhere: free


def test_the_audit_hook_lets_code_remove_a_folder_elsewhere_that_holds_a_data_folder(tmp_path, monkeypatch):
    # shutil.rmtree names each child relative to its parent folder, so a child named data is not the
    # checkout's data/ even when the current folder is the checkout; removing data/ itself stays refused
    root = tmp_path / "wpt"
    (root / "data").mkdir(parents=True)
    Repo(root)
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    (work / "data" / "f").write_text("x", encoding="utf-8")
    monkeypatch.chdir(root)
    elsewhere = str(tmp_path / "elsewhere" / "reader.py")
    exec(compile(f"__import__('shutil').rmtree({str(work)!r})", elsewhere, "exec"), {})
    assert not work.exists()
    with pytest.raises(DataAccessRefused, match="only the service layer"):
        exec(compile("__import__('shutil').rmtree('data')", elsewhere, "exec"), {})
    assert (root / "data").is_dir()


def test_the_audit_hook_follows_links_and_counts_only_the_project_s_own_tests(tmp_path, monkeypatch):
    # a link from elsewhere into data/ is refused like the path itself, and a checkout under a folder named
    # tests is not test code for being there: only a tests folder inside the checkout is
    from wpt_dataset import repo as service

    root = tmp_path / "wpt"
    (root / "data").mkdir(parents=True)
    signers = root / "data" / "allowed-signers"
    signers.write_text("# no key yet\n", encoding="utf-8")
    Repo(root)
    (tmp_path / "link").symlink_to(root / "data", target_is_directory=True)
    elsewhere = str(tmp_path / "elsewhere" / "reader.py")
    with pytest.raises(DataAccessRefused, match="only the service layer"):
        exec(compile(f"open({str(tmp_path / 'link' / 'allowed-signers')!r}).read()", elsewhere, "exec"), {})

    checkout = tmp_path / "tests" / "wpt"
    monkeypatch.setattr(service, "_PROJECT", os.path.normcase(str(checkout)))
    line = f"open({str(signers)!r}).read()"
    with pytest.raises(DataAccessRefused, match="only the service layer"):
        exec(compile(line, str(checkout / "packages" / "reader" / "src" / "reader.py"), "exec"), {})
    exec(compile(line, str(checkout / "packages" / "reader" / "tests" / "test_reader.py"), "exec"), {})
    exec(compile(line, str(checkout / "apps" / "conftest.py"), "exec"), {})  # a conftest.py in the checkout is test code
    with pytest.raises(DataAccessRefused, match="only the service layer"):  # one outside the checkout is not
        exec(compile(line, str(tmp_path / "elsewhere" / "conftest.py"), "exec"), {})
