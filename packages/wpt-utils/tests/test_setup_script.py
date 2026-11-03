import importlib.util
from pathlib import Path

SETUP = Path(__file__).parents[3] / "scripts" / "setup.py"


def setup_script():
    spec = importlib.util.spec_from_file_location("wpt_setup_script", SETUP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nodes(folder, *versions):
    for number, version in enumerate(versions):
        node = folder / str(number)
        node.mkdir()
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {version}\n", encoding="utf-8")
    return folder


def test_a_radeon_860m_is_found_and_what_to_set_is_said(tmp_path, capsys):
    setup = setup_script()
    setup.amd_note(setup.amd_chips(nodes(tmp_path, 0, 110502)))
    out = capsys.readouterr().out
    assert "HSA_OVERRIDE_GFX_VERSION=11.5.1" in out and "HIP_LAUNCH_BLOCKING=1" in out


def test_a_machine_without_such_a_chip_is_told_nothing(tmp_path, capsys):
    setup = setup_script()
    setup.amd_note(setup.amd_chips(nodes(tmp_path, 0, 110000)))
    setup.amd_note(setup.amd_chips(tmp_path / "absent"))
    assert capsys.readouterr().out == ""
