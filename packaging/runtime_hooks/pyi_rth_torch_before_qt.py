"""Load PyTorch before PyInstaller's Qt runtime hook on Windows.

PyTorch 2.9 and later can fail or stall when its native libraries are loaded
after Qt on Windows. Project-supplied PyInstaller runtime hooks execute before
the installed PyQt hook, so this preserves the required Torch-before-Qt order
for Cellpose and CellSAM without importing Torch during normal source startup.
"""

import sys


if sys.platform == "win32":
    import torch  # noqa: F401
