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
import platform
from PyQt5 import QtWidgets, QtGui


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Set consistent font size across all windows for readability on small and large screens
    # Use pixel size for better cross-platform consistency, especially on Windows
    font = QtGui.QFont()
    # Windows typically needs larger pixel size to match Mac/Linux point size appearance
    if platform.system() == 'Windows':
        font.setPixelSize(13)  # Slightly larger pixel size for Windows
    else:
        font.setPointSize(10)  # Point size works well on Mac/Linux
    app.setFont(font)

    # Import modular MainWindow wrapper
    from openimc.ui.main_window import MainWindow  # type: ignore

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


