import hashlib
import json

import pytest
import torch

from wpt_model import load_model
from wpt_posttrain import main

ZERO_WEIGHTS = ("import torch\nfrom wpt_posttrain import Outcome\n\n"
             "def run(context):\n"
             "    with torch.no_grad():\n"
             "        for p in context.model.parameters():\n"
             "            p.zero_()\n"
             "    return Outcome.success()\n")


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}


def test_steps_that_succeed_give_a_new_folder_and_leave_the_base_untouched(trained, tmp_path):
    base_before = digest(trained.folder)
    step = tmp_path / "zero_weights.py"
    step.write_text(ZERO_WEIGHTS)
    check = tmp_path / "check_records.py"
    check.write_text("from wpt_posttrain import Outcome\n\ndef run(context):\n    return Outcome.success()\n")
    main([str(trained.folder), str(tmp_path / "tuned"), str(step), str(check)])

    tuned = tmp_path / "tuned"
    assert all(torch.count_nonzero(p) == 0 for p in load_model(tuned).parameters())  # the steps' change was saved
    assert digest(trained.folder) == base_before
    provenance = json.loads((tuned / "provenance.json").read_text())
    post = provenance["post_training"]
    assert [s["file"] for s in post["steps"]] == ["zero_weights.py", "check_records.py"]
    assert post["steps"][0]["file_sha256"] == hashlib.sha256(step.read_bytes()).hexdigest()
    assert post["base_weights"] == {"model.safetensors": base_before["model.safetensors"]}
    assert provenance["sources"] == json.loads((trained.folder / "provenance.json").read_text())["sources"]
    assert trained.attribution in (tuned / "NOTICE").read_text()
    card = (tuned / "README.md").read_text()
    assert "## Post-training" in card and "1. `zero_weights.py`" in card and "2. `check_records.py`" in card
    assert not (tuned / "evals").exists()


def test_a_failed_step_stops_the_run_with_its_reason_and_writes_nothing(trained, tmp_path, capsys):
    step = tmp_path / "zero_weights.py"
    step.write_text(ZERO_WEIGHTS)
    failing = tmp_path / "needs_records.py"
    failing.write_text("from wpt_posttrain import Outcome\n\n"
                       "def run(context):\n    return Outcome.failure('no demonstration records in the store')\n")
    never = tmp_path / "never.py"
    never.write_text("raise SystemExit('this step must not run')\n")
    with pytest.raises(SystemExit, match="step needs_records failed: no demonstration records in the store"):
        main([str(trained.folder), str(tmp_path / "tuned"), str(step), str(failing), str(never)])
    assert list(tmp_path.glob("*tuned*")) == []


def test_a_step_that_returns_no_outcome_writes_nothing(trained, tmp_path):
    step = tmp_path / "broken.py"
    step.write_text("def run(context):\n    return True\n")
    with pytest.raises(SystemExit, match="returned bool, not wpt_posttrain.Outcome"):
        main([str(trained.folder), str(tmp_path / "tuned"), str(step)])
    assert list(tmp_path.glob("*tuned*")) == []
