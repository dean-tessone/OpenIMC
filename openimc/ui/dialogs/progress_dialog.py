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

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

class ProgressDialog(QtWidgets.QDialog):
    def __init__(self, title: str = "Export Progress", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setFixedSize(450, 180)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.cancelled = False
        self._create_ui()

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Preparing export...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.details_label = QtWidgets.QLabel("")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setStyleSheet("QLabel { color: #666; }")
        layout.addWidget(self.details_label)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)

    def _cancel(self):
        self.cancelled = True
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)

    def update_progress(self, value: int, status: str = "", details: str = ""):
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
        if details:
            self.details_label.setText(details)
        QtWidgets.QApplication.processEvents()

    def set_maximum(self, maximum: int):
        self.progress_bar.setMaximum(maximum)

    def is_cancelled(self) -> bool:
        return self.cancelled

