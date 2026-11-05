import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

from wpt_utils.lines import bare, line

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "recipe_name.py"


def test_a_name_that_wpt_dataset_list_prints_is_not_drawn_again(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("wpt_recipe_name_script", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: SimpleNamespace(stdout=line("fluffy-axolotl") + "\n"))
    monkeypatch.setattr(script, "words", lambda name: [name])
    draws = iter(["fluffy", "axolotl", "brave", "axolotl"])
    monkeypatch.setattr(script.secrets, "choice", lambda _: next(draws))
    script.main()
    assert bare(capsys.readouterr().out) == "brave-axolotl\n"
