# SPDX-License-Identifier: GPL-3.0-or-later
#
# Thin entry script to start the OpenIMC GUI for desktop bundles.
import multiprocessing
import os
import sys


def _run_bootstrap_command() -> bool:
    """Handle commands used to validate a frozen application.

    A frozen executable cannot interpret Python's ``-c`` option.  These small
    bootstrap commands give the application a safe way to probe Torch on
    Windows and let the build workflow test a bundle without opening a window.
    """
    if "--openimc-torch-probe" in sys.argv:
        import torch  # noqa: F401

        return True

    if "--openimc-bundle-smoke-test" in sys.argv:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        import cellpose  # noqa: F401
        import h5py  # noqa: F401
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        import readimc  # noqa: F401
        import scanpy  # noqa: F401
        import squidpy  # noqa: F401
        import torch  # noqa: F401
        from PyQt5 import QtWidgets
        from openimc.ui.main_window import MainWindow

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
            os._exit(0)

        window = MainWindow()
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
