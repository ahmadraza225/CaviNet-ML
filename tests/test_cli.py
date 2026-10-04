import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cavinet_ml import __version__
from cavinet_ml.cli import main


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"cavinet-ml {__version__}"


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: cavinet-ml" in capsys.readouterr().out


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "cavinet_ml", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == f"cavinet-ml {__version__}"


def _console_script() -> str | None:
    beside_python = Path(sys.executable).with_name("cavinet-ml")
    return str(beside_python) if beside_python.exists() else shutil.which("cavinet-ml")


@pytest.mark.skipif(_console_script() is None, reason="package not installed")
def test_console_script():
    result = subprocess.run(
        [_console_script(), "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == f"cavinet-ml {__version__}"


TOOLKIT_COMMANDS = (
    "index",
    "preprocess",
    "split",
    "train",
    "calibrate",
    "export",
    "evaluate",
    "baseline",
    "shortcut-check",
    "compare",
    "synthetic-dataset",
)


def test_help_lists_every_toolkit_command(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for command in TOOLKIT_COMMANDS:
        assert command in out


def test_toolkit_errors_are_one_line_with_exit_code_2(tmp_path, capsys):
    from cavinet_ml.cli import EXIT_ERROR

    assert (
        main(["split", "--work", str(tmp_path), "--splits", str(tmp_path / "s.json")]) == EXIT_ERROR
    )
    assert capsys.readouterr().err.startswith("error: ")
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch: {sizes: 4}\n", encoding="utf-8")
    code = main(["train", "--fold", "0", "--config", str(bad), "--work", str(tmp_path)])
    assert code == EXIT_ERROR and "unknown setting" in capsys.readouterr().err


def test_default_training_config_is_found():
    from cavinet_ml.cli import default_config

    assert default_config().name == "train.yaml" and default_config().is_file()
