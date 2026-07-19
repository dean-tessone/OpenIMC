# SPDX-License-Identifier: GPL-3.0-or-later
#
# Thin entry script to start the OpenIMC GUI for desktop bundles.
import multiprocessing
import importlib
import importlib.util
import os
import sys
from pathlib import Path


def _smoke_checkpoint(message: str) -> None:
    """Persist frozen-startup progress for CI diagnostics."""
    checkpoint_path = os.environ.get("OPENIMC_SMOKE_CHECKPOINT")
    if checkpoint_path:
        Path(checkpoint_path).write_text(message + "\n", encoding="utf-8")


def _run_bootstrap_command() -> bool:
    """Handle commands used to validate a frozen application.

    A frozen executable cannot interpret Python's ``-c`` option.  These small
    bootstrap commands give the application a safe way to probe Torch on
    Windows and let the build workflow test a bundle without opening a window.
    """
    if "--openimc-torch-probe" in sys.argv:
        import torch  # noqa: F401

        if sys.platform == "win32":
            os._exit(0)
        return True

    if "--openimc-squidpy-probe" in sys.argv:
        for module_name in ("squidpy", "scanpy", "anndata"):
            importlib.import_module(module_name)
        if sys.platform == "win32":
            os._exit(0)
        return True

    if "--openimc-bundle-smoke-test" in sys.argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        _smoke_checkpoint("entry point reached")
        for module_name in (
            "cellpose",
            "h5py",
            "igraph",
            "leidenalg",
            "readimc",
            "torch",
        ):
            _smoke_checkpoint(f"importing {module_name}")
            importlib.import_module(module_name)
            _smoke_checkpoint(f"imported {module_name}")

        if sys.platform == "win32":
            # OpenIMC intentionally probes the Squidpy stack in a disposable
            # child process on Windows instead of importing it into the GUI
            # process, where native-library faults could terminate the app.
            for module_name in ("squidpy", "scanpy", "anndata"):
                if importlib.util.find_spec(module_name) is None:
                    raise ImportError(f"Packaged module not found: {module_name}")
                _smoke_checkpoint(f"found packaged {module_name}")
        else:
            for module_name in ("scanpy", "squidpy"):
                _smoke_checkpoint(f"importing {module_name}")
                importlib.import_module(module_name)
                _smoke_checkpoint(f"imported {module_name}")

        _smoke_checkpoint("importing OpenIMC main window")
        from PyQt5 import QtWidgets
        from openimc.ui.main_window import MainWindow
        _smoke_checkpoint("imported OpenIMC main window")

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        if sys.platform == "win32":
            # Constructing and closing the complete MainWindow is already
            # covered by the Windows pytest job. In a windowed PyInstaller
            # process, that close path can wait indefinitely inside native Qt
            # teardown even after all imports have succeeded. Use a real Qt
            # top-level window here to validate the frozen Qt platform plugin
            # and event loop, while the MainWindow import above validates the
            # packaged OpenIMC UI module and its native dependencies.
            probe_window = QtWidgets.QMainWindow()
            probe_window.setWindowTitle("OpenIMC frozen bundle smoke test")
            probe_window.show()
            app.processEvents()
            _smoke_checkpoint("Windows Qt event loop passed")
            os._exit(0)

        window = MainWindow()
        toolbar_icon_size = window.nav_toolbar.iconSize()
        if (toolbar_icon_size.width(), toolbar_icon_size.height()) != (20, 20):
            raise AssertionError(
                "Unexpected main-viewer toolbar icon size: "
                f"{toolbar_icon_size.width()}x{toolbar_icon_size.height()}"
            )
        _smoke_checkpoint("main viewer toolbar uses 20x20 logical icons")
        window.show()
        app.processEvents()
        original_question = QtWidgets.QMessageBox.question
        QtWidgets.QMessageBox.question = lambda *args, **kwargs: QtWidgets.QMessageBox.No
        try:
            window.close()
        finally:
            QtWidgets.QMessageBox.question = original_question
        app.processEvents()
        app.quit()
        return True

    if "--openimc-functional-test" in sys.argv:
        command_index = sys.argv.index("--openimc-functional-test")
        arguments = sys.argv[command_index + 1 : command_index + 4]
        if len(arguments) != 3:
            raise SystemExit(
                "--openimc-functional-test requires INPUT_OME_TIFF MASK OUTPUT_DIRECTORY"
            )
        from openimc.release_validation import run_release_validation

        run_release_validation(*arguments)
        return True

    return False

if __name__ == "__main__":
    # Required for multiprocessing workers spawned from PyInstaller bundles.
    multiprocessing.freeze_support()

    if not _run_bootstrap_command():
        from openimc.__main__ import run_gui

        run_gui()
