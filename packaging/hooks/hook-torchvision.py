"""Include Torchvision's indirectly loaded native operator libraries.

Torchvision 0.29 uses ``_C_stable`` rather than ``_C``. These libraries are
loaded with torch.ops.load_library(), so importing the old extension name in
PyInstaller's stock hook does not collect the new operators. Collect native
files by type to support both layouts, including Windows .pyd libraries.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs


binaries = collect_dynamic_libs(
    "torchvision",
    search_patterns=["*.pyd", "*.dll", "*.so", "*.dylib"],
)

# Torchvision and CellSAM inspect source while preparing inference models.
module_collection_mode = "pyz+py"
