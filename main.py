#!/usr/bin/env python3
import sys
from PyQt5 import QtWidgets, QtGui


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Set consistent font size across all windows for readability on small and large screens
    # 10pt is a good default that scales well with DPI
    font = QtGui.QFont()
    font.setPointSize(10)
    app.setFont(font)

    # Import modular MainWindow wrapper
    from openimc.ui.main_window import MainWindow  # type: ignore

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


