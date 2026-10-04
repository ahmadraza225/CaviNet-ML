"""Where a result came from: git commit, software versions and the machine (NFR-8)."""

import getpass
import os
import platform
import socket
import subprocess
from typing import Any


def git_commit() -> str:
    """The checked-out commit (with "+dirty" if files are modified), CAVINET_GIT_COMMIT,
    or "unknown"."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("CAVINET_GIT_COMMIT") or "unknown"


def environment() -> dict[str, Any]:
    import torch

    from cavinet_ml import __version__

    info: dict[str, Any] = {
        "toolkit_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "user": _user(),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"
