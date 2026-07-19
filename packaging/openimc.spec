# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the full OpenIMC desktop application."""

import os
import platform
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


PROJECT_ROOT = Path(SPECPATH).parent
APP_VERSION = os.environ.get("OPENIMC_VERSION", "0.1.0")
CONSOLE = os.environ.get("OPENIMC_CONSOLE", "0") == "1"

datas = [
    (str(PROJECT_ROOT / "LICENSE"), "."),
    (str(PROJECT_ROOT / "third_party" / "licenses"), "third_party/licenses"),
]
binaries = []
hiddenimports = []


def is_runtime_module(module_name):
    """Exclude dependency test and benchmark modules from the application."""
    parts = module_name.split(".")
    return not any(
        part in {"tests", "test", "testing", "benchmarks", "conftest"}
        or part.startswith("test_")
        or part.endswith("_test")
        for part in parts
    )

# These packages use runtime registration, lazy modules, or plugin discovery,
# which static import analysis cannot see. Their native libraries and package
# data must travel with the application as well.
COLLECT_PACKAGES = (
    "anndata",
    "cellpose",
    "cellSAM",
    "combat",
    "dask",
    "dask_image",
    "distributed",
    "harmonypy",
    "hdbscan",
    "igraph",
    "imagecodecs",
    "kornia",
    "leidenalg",
    "readimc",
    "scanpy",
    "segment_anything",
    "squidpy",
    "spatialdata",
    "umap",
)

for package_name in COLLECT_PACKAGES:
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name,
        include_py_files=True,
        filter_submodules=is_runtime_module,
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

# Preserve metadata used by importlib.metadata and runtime version checks.
for distribution_name in (
    "OpenIMC",
    "anndata",
    "cellpose",
    "cellSAM",
    "combat",
    "dask",
    "dask-image",
    "distributed",
    "harmonypy",
    "hdbscan",
    "imagecodecs",
    "kornia",
    "leidenalg",
    "matplotlib",
    "openai",
    "python-igraph",
    "readimc",
    "scanpy",
    "scikit-image",
    "scikit-learn",
    "squidpy",
    "spatialdata",
    "torch",
    "torchvision",
    "umap-learn",
):
    try:
        datas += copy_metadata(distribution_name, recursive=True)
    except Exception:
        # Some optional distributions (notably CellSAM) may be absent in a
        # reduced build. The application already disables those features.
        pass

analysis = Analysis(
    [str(PROJECT_ROOT / "openimc" / "gui_entry.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[str(PROJECT_ROOT / "packaging" / "hooks")],
    hooksconfig={},
    # Project runtime hooks execute before PyInstaller's installed PyQt hook.
    # Windows PyTorch must initialize first so Cellpose/CellSAM remain usable.
    runtime_hooks=[
        str(
            PROJECT_ROOT
            / "packaging"
            / "runtime_hooks"
            / "pyi_rth_torch_before_qt.py"
        ),
        str(
            PROJECT_ROOT
            / "packaging"
            / "runtime_hooks"
            / "pyi_rth_persistent_mpl_cache.py"
        ),
    ],
    excludes=(
        "IPython",
        "notebook",
        "pytest",
        "sphinx",
        "tkinter",
    ),
    noarchive=False,
    optimize=1,
)

# PyInstaller's stock Matplotlib hook creates and removes a fresh cache on every
# launch to support one-file bundles whose extraction path changes each time.
# OpenIMC deliberately ships as a stable application folder, so that behavior
# needlessly rebuilds the system font cache and adds roughly 15 seconds to every
# startup. Replace it with our persistent, per-user cache runtime hook above.
analysis.scripts = [
    entry for entry in analysis.scripts if entry[0] != "pyi_rth_mplconfig"
]

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="OpenIMC",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("OPENIMC_CODESIGN_IDENTITY") or None,
    entitlements_file=None,
    icon=str(
        PROJECT_ROOT
        / "openimc"
        / "ui"
        / "resources"
        / ("OpenIMC.ico" if platform.system() == "Windows" else "OpenIMC.icns")
    ),
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OpenIMC",
)

if platform.system() == "Darwin":
    app = BUNDLE(
        collection,
        name="OpenIMC.app",
        icon=str(PROJECT_ROOT / "openimc" / "ui" / "resources" / "OpenIMC.icns"),
        bundle_identifier="org.openimc.OpenIMC",
        version=APP_VERSION,
        info_plist={
            "CFBundleDisplayName": "OpenIMC",
            "CFBundleName": "OpenIMC",
            "CFBundleShortVersionString": APP_VERSION,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
        codesign_identity=os.environ.get("OPENIMC_CODESIGN_IDENTITY") or None,
        entitlements_file=None,
    )
