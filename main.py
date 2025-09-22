#!/usr/bin/env python3
import sys
from PyQt5 import QtWidgets


def main():
    app = QtWidgets.QApplication(sys.argv)

    # Import modular MainWindow wrapper
    from openmcd.ui.main_window import MainWindow  # type: ignore

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


