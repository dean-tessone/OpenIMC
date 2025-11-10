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
Dialog for selecting and configuring spillover correction.
"""

from typing import Optional
import os

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False


class SpilloverCorrectionDialog(QtWidgets.QDialog):
    """Dialog for configuring spillover correction."""
    
    def __init__(self, parent=None, channels: Optional[list] = None):
        super().__init__(parent)
        self.channels = channels or []
        self.spillover_matrix = None
        self.setWindowTitle("Spillover Correction")
        self.setModal(True)
        self.resize(600, 400)
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Description
        desc_label = QtWidgets.QLabel(
            "Spillover correction compensates for signal bleed-through between channels.\n"
            "Select a spillover matrix CSV file where rows and columns are channel names."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("QLabel { color: #666; margin-bottom: 10px; }")
        layout.addWidget(desc_label)
        
        # Spillover matrix file selection
        file_group = QtWidgets.QGroupBox("Spillover Matrix")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(QtWidgets.QLabel("CSV file:"))
        self.spillover_file_edit = QtWidgets.QLineEdit()
        self.spillover_file_edit.setPlaceholderText("Select spillover matrix CSV file...")
        self.spillover_file_edit.setReadOnly(True)
        self.spillover_file_btn = QtWidgets.QPushButton("Browse...")
        self.spillover_file_btn.clicked.connect(self._select_spillover_file)
        file_row.addWidget(self.spillover_file_edit)
        file_row.addWidget(self.spillover_file_btn)
        file_layout.addLayout(file_row)
        
        # Method selection
        method_layout = QtWidgets.QHBoxLayout()
        method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(["pgd", "nnls"])
        self.method_combo.setToolTip(
            "pgd: Fast projected gradient descent (recommended)\n"
            "nnls: Non-negative least squares (requires SciPy, slower but more accurate)"
        )
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        file_layout.addLayout(method_layout)
        
        layout.addWidget(file_group)
        
        # Channel mapping (optional)
        self.mapping_group = QtWidgets.QGroupBox("Channel Mapping (Optional)")
        mapping_layout = QtWidgets.QVBoxLayout(self.mapping_group)
        
        mapping_info = QtWidgets.QLabel(
            "If your channel names in the data don't match the spillover matrix,\n"
            "you can map them here. Leave empty to use exact matches."
        )
        mapping_info.setWordWrap(True)
        mapping_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        mapping_layout.addWidget(mapping_info)
        
        # Channel mapping table (simplified - just show info for now)
        # For a full implementation, you could add a table widget here
        self.mapping_note = QtWidgets.QLabel(
            "Channel mapping will be done automatically based on name matching.\n"
            "Use exact channel names in the spillover matrix for best results."
        )
        self.mapping_note.setStyleSheet("QLabel { color: #666; font-style: italic; font-size: 9pt; }")
        self.mapping_note.setWordWrap(True)
        mapping_layout.addWidget(self.mapping_note)
        
        layout.addWidget(self.mapping_group)
        
        # Preview/validation info
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: #0066cc; font-size: 9pt; }")
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self._validate_and_accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
    
    def _select_spillover_file(self):
        """Open file dialog to select spillover matrix CSV."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Spillover Matrix CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.spillover_file_edit.setText(file_path)
            self._validate_spillover_file(file_path)
    
    def _validate_spillover_file(self, file_path: str):
        """Validate and preview the spillover matrix."""
        if not _HAVE_PANDAS:
            self.info_label.setText("Error: pandas is required for spillover correction")
            self.info_label.setStyleSheet("QLabel { color: #cc0000; font-size: 9pt; }")
            return
        
        try:
            from openimc.processing.spillover_correction import load_spillover
            
            # Load and validate the spillover matrix
            S = load_spillover(file_path)
            
            # Show matrix info
            n_channels = len(S)
            common_channels = []
            if self.channels:
                common_channels = [ch for ch in S.columns if ch in self.channels]
            
            info_text = (
                f"✓ Valid spillover matrix loaded\n"
                f"Matrix size: {n_channels}x{n_channels}\n"
                f"Channels in matrix: {', '.join(S.columns[:5])}"
                + (f"..." if len(S.columns) > 5 else "")
            )
            
            if self.channels:
                info_text += f"\nCommon channels: {len(common_channels)}/{len(self.channels)}"
                if len(common_channels) < len(self.channels):
                    missing = [ch for ch in self.channels if ch not in S.columns]
                    if missing:
                        info_text += f"\nMissing in matrix: {', '.join(missing[:5])}"
                        if len(missing) > 5:
                            info_text += "..."
            
            self.info_label.setText(info_text)
            self.info_label.setStyleSheet("QLabel { color: #0066cc; font-size: 9pt; }")
            self.spillover_matrix = S
            
        except Exception as e:
            self.info_label.setText(f"Error loading spillover matrix: {str(e)}")
            self.info_label.setStyleSheet("QLabel { color: #cc0000; font-size: 9pt; }")
            self.spillover_matrix = None
    
    def _validate_and_accept(self):
        """Validate inputs and accept the dialog."""
        file_path = self.spillover_file_edit.text().strip()
        
        if not file_path:
            QtWidgets.QMessageBox.warning(
                self,
                "No file selected",
                "Please select a spillover matrix CSV file."
            )
            return
        
        if not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(
                self,
                "File not found",
                f"The selected file does not exist:\n{file_path}"
            )
            return
        
        # Validate the file
        if not _HAVE_PANDAS:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing dependency",
                "pandas is required for spillover correction.\nPlease install it: pip install pandas"
            )
            return
        
        try:
            from openimc.processing.spillover_correction import load_spillover
            S = load_spillover(file_path)
            
            # Check for at least some channel overlap if channels are provided
            if self.channels:
                common = [ch for ch in S.columns if ch in self.channels]
                if len(common) == 0:
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "No channel overlap",
                        "No channels in the spillover matrix match the data channels.\n"
                        "Continue anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        return
            
            self.spillover_matrix = S
            self.accept()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid spillover matrix",
                f"Error loading spillover matrix:\n{str(e)}"
            )
            return
    
    def get_spillover_matrix(self):
        """Get the loaded spillover matrix."""
        return self.spillover_matrix
    
    def get_spillover_file_path(self):
        """Get the path to the spillover matrix file."""
        return self.spillover_file_edit.text().strip()
    
    def get_method(self):
        """Get the selected compensation method."""
        return self.method_combo.currentText()

