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


class OMETIFFFormatDialog(QtWidgets.QDialog):
    """Dialog to ask user about OME-TIFF channel format."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OME-TIFF Channel Format")
        self.setModal(True)
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        
        self.selected_format = 'CHW'  # Default to CHW (matches export format)
        self._create_ui()
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Question label
        question_label = QtWidgets.QLabel(
            "What format are the channels stored in the OME-TIFF file?"
        )
        question_label.setWordWrap(True)
        layout.addWidget(question_label)
        
        # Radio buttons
        self.chw_radio = QtWidgets.QRadioButton("CHW (Channels, Height, Width) - Default")
        self.chw_radio.setChecked(True)  # Default selection
        self.chw_radio.toggled.connect(lambda checked: setattr(self, 'selected_format', 'CHW') if checked else None)
        layout.addWidget(self.chw_radio)
        
        self.hwc_radio = QtWidgets.QRadioButton("HWC (Height, Width, Channels)")
        self.hwc_radio.toggled.connect(lambda checked: setattr(self, 'selected_format', 'HWC') if checked else None)
        layout.addWidget(self.hwc_radio)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "Note: Files exported by this application use CHW format."
        )
        info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
    
    def get_format(self) -> str:
        """Get the selected format."""
        return self.selected_format
    
    def set_format(self, format_str: str):
        """Set the selected format."""
        if format_str == 'CHW':
            self.chw_radio.setChecked(True)
            self.selected_format = 'CHW'
        elif format_str == 'HWC':
            self.hwc_radio.setChecked(True)
            self.selected_format = 'HWC'

