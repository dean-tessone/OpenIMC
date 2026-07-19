import builtins
import importlib
import subprocess
import sys

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name, blocked_imports",
    [
        ("openimc.ui.dialogs.batch_correction_dialog", ("harmonypy",)),
        ("openimc.ui.dialogs.gpu_selection_dialog", ("torch",)),
        ("openimc.ui.dialogs.segmentation_dialog", ("torch", "cellSAM")),
        (
            "openimc.ui.main_window",
            (
                "torch",
                "cellSAM",
                "harmonypy",
                "umap",
                "pynndescent",
                "openimc.processing.custom_cellsam",
                "openimc.ui.dialogs.clustering",
            ),
        ),
    ],
)
def test_ui_modules_defer_optional_runtime_imports(monkeypatch, module_name, blocked_imports):
    attempted_imports = []
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        for blocked in blocked_imports:
            if top_level == blocked or name == blocked or name.startswith(f"{blocked}."):
                attempted_imports.append(name)
                raise AssertionError(f"unexpected optional import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module(module_name)
    importlib.reload(module)

    assert attempted_imports == []


def test_gui_module_defers_cli_import_until_cli_is_called():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import openimc.__main__; "
                "assert 'openimc.cli' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
