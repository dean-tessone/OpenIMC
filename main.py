#!/usr/bin/env python3
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

import sys
import os
import platform

# CRITICAL: Configure dask BEFORE any imports that might trigger dask.dataframe
# This must be done at the very start of the application
os.environ.setdefault('DASK_DATAFRAME__QUERY_PLANNING', 'False')

# Suppress warnings from dependencies
import warnings
# Suppress dask dataframe legacy implementation warning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*legacy.*Dask DataFrame.*')
# Suppress xarray_schema pkg_resources deprecation warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources.*deprecated.*')
# Suppress squidpy anndata __version__ deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*__version__.*deprecated.*')

try:
    import dask
    dask.config.set({'dataframe.query-planning': False})
except (ImportError, AttributeError):
    pass

from PyQt5 import QtWidgets, QtGui


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Load font size preference or use default
    from openimc.ui.dialogs.display_settings_dialog import (
        get_font_size_preference,
        get_default_font_size
    )
    
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
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


