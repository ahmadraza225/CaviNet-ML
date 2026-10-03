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
