# SPDX-License-Identifier: GPL-3.0-or-later
#
# OpenIMC – Interactive analysis toolkit for IMC data
#
# Copyright (C) 2025 University of Southern California
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Entry point for running OpenIMC CLI and GUI as a module: python -m openimc
"""

# CRITICAL: Configure dask at module level BEFORE any other imports
# This must be the very first thing that happens when this module is imported
import os
import warnings

# Suppress dask dataframe legacy implementation warning at module level
warnings.filterwarnings('ignore', category=FutureWarning, module='dask.dataframe')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*legacy.*Dask DataFrame.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*dataframe.query-planning.*')

# Set environment variable first (this is read when dask is imported)
os.environ['DASK_DATAFRAME__QUERY_PLANNING'] = 'True'

def run_gui():
    """Run the OpenIMC GUI application."""
    import sys
    import os
    import platform

    # CRITICAL: Configure dask BEFORE any imports that might trigger dask.dataframe
    # This must be done at the very start of the application, before importing dask itself
    # Set environment variable first (this is read when dask is imported)
    os.environ['DASK_DATAFRAME__QUERY_PLANNING'] = 'True'
    
    # Suppress warnings from dependencies
    import warnings
    # Suppress dask dataframe legacy implementation warning
    warnings.filterwarnings('ignore', category=FutureWarning, module='dask.dataframe')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*legacy.*Dask DataFrame.*')
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*dataframe.query-planning.*')
    # Suppress squidpy anndata __version__ deprecation warning
    warnings.filterwarnings('ignore', category=FutureWarning, message='.*__version__.*deprecated.*')

    from PyQt5 import QtCore, QtWidgets, QtGui
    from PyQt5.QtCore import Qt

    app = QtWidgets.QApplication(sys.argv)
    
    # Set application name and display name for system dock/taskbar
    app.setApplicationName("OpenIMC")
    app.setApplicationDisplayName("OpenIMC")

    from openimc.ui.dialogs.display_settings_dialog import (
        get_default_font_size,
        get_font_size_preference,
        get_methods_log_file_preference,
        get_theme_preference,
        save_methods_log_file_preference,
    )
    from openimc.utils.logger import get_default_log_file_path, set_log_file
    from openimc.ui.theme import apply_application_theme, palette_is_dark

    # OpenIMC defaults to light even when the operating system is dark. Users
    # can opt into Dark or System from Display Settings.
    selected_theme = get_theme_preference()
    apply_application_theme(app, selected_theme)
    dark_startup = palette_is_dark(app.palette())

    # Give immediate visual feedback before importing the scientific and
    # plotting stack. This is especially important on a first frozen launch,
    # when OS verification and Matplotlib setup can otherwise look like a hang.
    splash_pixmap = QtGui.QPixmap(520, 210)
    splash_pixmap.fill(QtGui.QColor("#18212b" if dark_startup else "#f4f6f8"))
    painter = QtGui.QPainter(splash_pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtGui.QColor("#ffffff" if dark_startup else "#17324d"))
    title_font = QtGui.QFont()
    title_font.setPointSize(28)
    title_font.setBold(True)
    painter.setFont(title_font)
    painter.drawText(QtCore.QRect(32, 34, 456, 54), Qt.AlignLeft | Qt.AlignVCenter, "OpenIMC")
    painter.setPen(QtGui.QColor("#b8c5d1" if dark_startup else "#526779"))
    status_font = QtGui.QFont()
    status_font.setPointSize(13)
    painter.setFont(status_font)
    painter.drawText(
        QtCore.QRect(34, 98, 452, 42),
        Qt.AlignLeft | Qt.AlignVCenter,
        "Starting scientific analysis tools…",
    )
    painter.setBrush(QtGui.QColor("#4fa3d1" if dark_startup else "#2f80b8"))
    painter.setPen(QtCore.Qt.NoPen)
    painter.drawRoundedRect(QtCore.QRect(34, 158, 452, 6), 3, 3)
    painter.end()
    splash = QtWidgets.QSplashScreen(splash_pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    splash.show()
    app.processEvents()

    # Application bundles are read-only after installation. Initialize the
    # methods logger in a per-user location, or a user-selected file remembered
    # from an earlier session, before any analysis action tries to write to it.
    preferred_log_file = get_methods_log_file_preference()
    try:
        set_log_file(preferred_log_file or str(get_default_log_file_path()))
    except OSError as exc:
        fallback_log_file = get_default_log_file_path()
        set_log_file(str(fallback_log_file))
        save_methods_log_file_preference(None)
        QtWidgets.QMessageBox.warning(
            None,
            "Methods Log Location Reset",
            "OpenIMC could not write to the selected methods log file and "
            "has returned to the default user-writable location:\n\n"
            f"{fallback_log_file}\n\nReason: {exc}",
        )
    
    # Load and set application icon
    icon = None
    icon_path = None
    try:
        # Try multiple methods to find the icon file
        # Method 1: Relative to __main__.py (works in development)
        icon_path = os.path.join(os.path.dirname(__file__), 'ui', 'resources', 'OpenIMC_Icon.png')
        if not os.path.exists(icon_path):
            # Method 2: Try using importlib.resources (works when installed)
            try:
                import importlib.resources
                with importlib.resources.path('openimc.ui.resources', 'OpenIMC_Icon.png') as p:
                    icon_path = str(p)
            except (ImportError, ModuleNotFoundError, FileNotFoundError):
                # Method 3: Try relative to openimc package
                import openimc
                icon_path = os.path.join(os.path.dirname(openimc.__file__), 'ui', 'resources', 'OpenIMC_Icon.png')
        
        # Convert to absolute path for better compatibility
        icon_path = os.path.abspath(icon_path)
        
        if os.path.exists(icon_path):
            # Create icon with explicit sizes for better Linux compatibility
            icon = QtGui.QIcon()
            pixmap = QtGui.QPixmap(icon_path)
            if not pixmap.isNull():
                # Add multiple sizes to the icon (Linux window managers often need this)
                icon.addPixmap(pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                # Also add scaled versions for different contexts
                for size in [16, 32, 48, 64, 128, 256]:
                    scaled = pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    icon.addPixmap(scaled, QtGui.QIcon.Normal, QtGui.QIcon.Off)
                
                if not icon.isNull():
                    app.setWindowIcon(icon)
                    # On Linux, also set the WM_CLASS hint for better dock integration
                    if platform.system() == 'Linux':
                        app.setDesktopFileName('openimc')
    except Exception as e:
        # If icon loading fails, continue without it
        # Enable debugging by setting environment variable: export OPENIMC_DEBUG_ICON=1
        if os.environ.get('OPENIMC_DEBUG_ICON'):
            print(f"Warning: Could not load application icon: {e}")
            print(f"  Attempted path: {icon_path}")
            if icon_path and os.path.exists(icon_path):
                print(f"  File exists: {icon_path}")
            else:
                print(f"  File does not exist: {icon_path}")
        pass

    # Load font size preference or use default
    saved_font_size = get_font_size_preference()
    default_font_size = get_default_font_size()
    font_size = saved_font_size if saved_font_size is not None else default_font_size
    
    # Set consistent font size across all windows for readability on small and large screens
    # Use pixel size for better cross-platform consistency, especially on Windows
    font = QtGui.QFont()
    if platform.system() == 'Windows':
        font.setPixelSize(font_size)  # Pixel size for Windows
    else:
        font.setPointSize(font_size)  # Point size for Mac/Linux
    app.setFont(font)

    # Import modular MainWindow wrapper
    from openimc.ui.main_window import MainWindow  # type: ignore

    win = MainWindow()
    # Ensure the main window also has the icon set BEFORE showing
    # This is critical - the icon must be set before show() on some Linux window managers
    if icon is not None:
        win.setWindowIcon(icon)
    elif not app.windowIcon().isNull():
        win.setWindowIcon(app.windowIcon())
    
    win.show()
    splash.finish(win)

    # The public Ubuntu and Windows bundles deliberately ship a compact
    # CPU-only Torch runtime. If the machine has an NVIDIA driver, offer the
    # separately downloaded, per-user CUDA runtime after the window responds.
    from openimc.ui.gpu_setup import maybe_offer_cuda_setup

    QtCore.QTimer.singleShot(0, lambda: maybe_offer_cuda_setup(win))

    sys.exit(app.exec_())


def cli():
    """Run the OpenIMC CLI application."""
    # Keep the large CLI dependency graph out of GUI startup. The desktop
    # entry point imports this module only to call ``run_gui``.
    from openimc.cli import main

    main()


if __name__ == '__main__':
    cli()
