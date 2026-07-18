"""Helpers for optional third-party dependencies.

These utilities avoid importing heavy or fragile packages at module import time.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any, Optional


def optional_dependency_available(module_name: str) -> bool:
    """Check for an optional dependency without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


_HAVE_TORCH_SPEC = optional_dependency_available("torch")
_TORCH_IMPORT_ATTEMPTED = False
_TORCH_IMPORT_ERROR: Optional[Any] = None
_TORCH_PROBE_ATTEMPTED = False
_TORCH_PROBE_ERROR: Optional[str] = None
_TORCH_MODULE = None


def _should_probe_torch_import() -> bool:
    """Probe torch in a child process on Windows so DLL faults stay isolated."""
    return sys.platform.startswith("win")


def _probe_torch_import_in_subprocess() -> bool:
    """Return True when a child process can safely import torch."""
    global _TORCH_PROBE_ATTEMPTED, _TORCH_PROBE_ERROR

    if _TORCH_PROBE_ATTEMPTED:
        return _TORCH_PROBE_ERROR is None

    _TORCH_PROBE_ATTEMPTED = True
    probe_code = """
import importlib
importlib.import_module("torch")
"""

    # In a PyInstaller bundle sys.executable is the OpenIMC executable, not a
    # Python interpreter, so ``-c`` would relaunch the GUI instead of running
    # the probe. The frozen entry point handles this private command directly.
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--openimc-torch-probe"]
    else:
        command = [sys.executable, "-c", probe_code]

    child_environment = os.environ.copy()
    child_environment["OPENIMC_TORCH_PROBE"] = "1"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
            env=child_environment,
        )
    except Exception as exc:
        _TORCH_PROBE_ERROR = f"Subprocess torch probe failed: {exc}"
        return False

    if result.returncode == 0:
        _TORCH_PROBE_ERROR = None
        return True

    detail = (result.stderr or result.stdout or "").strip()
    if detail:
        detail = detail.splitlines()[-1].strip()
    else:
        detail = f"Torch probe subprocess exited with code {result.returncode}"
    _TORCH_PROBE_ERROR = detail
    return False


def get_torch_module():
    """Import torch lazily and cache the result."""
    global _TORCH_IMPORT_ATTEMPTED, _TORCH_IMPORT_ERROR, _HAVE_TORCH_SPEC, _TORCH_MODULE

    if _TORCH_MODULE is not None:
        return _TORCH_MODULE

    if _TORCH_IMPORT_ATTEMPTED:
        return None

    _TORCH_IMPORT_ATTEMPTED = True
    if not _HAVE_TORCH_SPEC:
        return None

    if _should_probe_torch_import() and not _probe_torch_import_in_subprocess():
        _TORCH_IMPORT_ERROR = _TORCH_PROBE_ERROR
        _HAVE_TORCH_SPEC = False
        return None

    try:
        _TORCH_MODULE = importlib.import_module("torch")
        _TORCH_IMPORT_ERROR = None
        return _TORCH_MODULE
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        _HAVE_TORCH_SPEC = False
        _TORCH_MODULE = None
        return None


def torch_import_error() -> Optional[Any]:
    """Return the cached torch import error, if any."""
    return _TORCH_IMPORT_ERROR or _TORCH_PROBE_ERROR
