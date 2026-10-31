"""A recipe's step entry read into a wpt_training.Step (docs/training.md, "Steps")."""

import pytest

from wpt_model import PARTS
from wpt_training import PluginError, load_step


def test_a_step_s_settings_are_its_parts_defaults_then_its_file_s_then_the_recipe_s_each_typed():
    step = load_step({"step": "finish-tasks", "settings": {"rounds": 2, "weight_rule": 1}}, PARTS)
    assert (step.method.name, step.environment.name, step.optimizer.name) == ("grpo", "agent-tasks", "by-part")
    environment = step.environment.module.SETTINGS
    assert step.settings == {**environment, "terse": True, "rounds": 2, "weight_rule": 1.0}  # terse: the step file's
    assert step.environment_settings == step.settings and step.method_settings == {}
    assert step.optimizer_settings["lr"] == 1e-6 and step.optimizer_settings["weight_decay"] == 0.0
    assert step.locked == ("routers", "balancing")
    # the record names every file and setting, which the step's fingerprint covers
    record = step.record()
    assert record["settings"] == step.settings and record["locked"] == ["routers", "balancing"]
    assert record["optimizer"]["settings"] == step.optimizer_settings
    assert record["step"]["file"] == "wpt_training/steps/finish-tasks.py"


def test_a_step_s_optimizer_is_chosen_apart_from_its_method(tmp_path):
    # the step's own optimizer takes the step file's settings for it, then the
    # recipe's; another one takes its own defaults, then the recipe's
    own = load_step({"step": "answer-as-assistant", "optimizer": {"name": "by-part", "settings": {"lr": 3e-5}}},
                    PARTS)
    assert own.optimizer_settings["lr"] == 3e-5 and own.optimizer_settings["weight_decay"] == 0.0
    assert own.optimizer_settings["schedule"] == "constant" and own.optimizer_settings["patience"] == 10
    plain = tmp_path / "plain.py"
    plain.write_text("SETTINGS = {'lr': 0.1, 'warmup': 0}\n\ndef start(model, settings, run):\n    ...\n")
    other = load_step({"step": "answer-as-assistant", "optimizer": {"name": str(plain), "settings": {"warmup": 5}}},
                      PARTS)
    assert other.method.name == "assistant-tokens"
    assert other.optimizer_settings == {"lr": 0.1, "warmup": 5}
    # the read-text step's next-token loss settings are the step's, whatever its optimizer
    text = load_step({"step": "read-text", "optimizer": {"name": str(plain)},
                      "settings": {"stages": [{"block_size": 16, "steps": 1, "batch_size": 4}], "mtp_weight": 0.5}},
                     PARTS)
    assert text.method_settings["mtp_weight"] == 0.5 and "mtp_weight" not in text.optimizer_settings


def test_a_step_entry_that_cannot_be_is_refused_saying_why(tmp_path):
    cases = [
        ({"step": "read-twice"}, r"read-twice: not a built-in step \(answer-as-assistant, finish-tasks, "),
        ({"step": "finish-tasks", "settings": {"round": 3}}, r"finish-tasks.py: no setting 'round' \(settings: "),
        ({"step": "finish-tasks", "settings": {"rounds": "many"}}, r"setting rounds='many' is not a int"),
        ({"step": "finish-tasks", "settings": {"reward": "longer"}}, r"reward 'longer': one of weighed, shorter"),
        ({"step": "answer-as-assistant", "locked": ["routres"]},
         r"answer-as-assistant.py: no part of the model is named 'routres' \(parts: embedding, attention, "),
        ({"step": "answer-as-assistant", "optimizer": {"name": "sgd"}}, r"sgd: not a built-in optimizer"),
    ]
    mismatched = tmp_path / "talk-text.py"  # a method that reads text windows, given conversations
    mismatched.write_text('METHOD = "next-token"\nENVIRONMENT = "conversations"\nOPTIMIZER = {"name": "by-part"}\n')
    cases.append(({"step": str(mismatched)},
                  r"talk-text.py: its method next-token reads windows, and its environment conversations gives "
                  r"conversations"))
    for entry, why in cases:
        with pytest.raises(PluginError, match=why):
            load_step(entry, PARTS)
