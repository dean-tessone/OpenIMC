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

from typing import List, Optional
import os
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from openimc.data.mcd_loader import AcquisitionInfo, MCDLoader


class DeconvolutionDialog(QtWidgets.QDialog):
    def __init__(self, acquisitions: List[AcquisitionInfo], current_acq_id: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("High Resolution IMC Deconvolution")
        self.setModal(True)
        
        # Set dialog size
        if parent:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.8)
            dialog_height = int(parent_size.height() * 0.7)
            self.resize(dialog_width, dialog_height)
        else:
            self.resize(700, 600)
        
        self.setMinimumSize(600, 500)
        self.acquisitions = acquisitions
        self.current_acq_id = current_acq_id
        self.output_directory = ""
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        # Main layout for the dialog
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Create scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create widget to hold the scrollable content
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)
        
        # Information note
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_label = QtWidgets.QLabel(
            "This deconvolution method is optimized for high resolution IMC images with step sizes of 333 nm and 500 nm. "
            "The deconvolution uses Richardson-Lucy deconvolution with a circular kernel optimized for IMC data.\n\n"
            "Works with both MCD files and OME-TIFF directories."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #0066cc; font-style: italic; }")
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Acquisition selection
        acq_group = QtWidgets.QGroupBox("Acquisition Selection")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        
        self.single_roi_radio = QtWidgets.QRadioButton("Single ROI (Current Acquisition)")
        self.whole_slide_radio = QtWidgets.QRadioButton("Whole Slide (All Acquisitions)")
        self.single_roi_radio.setChecked(True)
        
        acq_layout.addWidget(self.single_roi_radio)
        acq_layout.addWidget(self.whole_slide_radio)
        
        # Current acquisition info
        self.acq_info_label = QtWidgets.QLabel("")
        acq_layout.addWidget(self.acq_info_label)
        
        layout.addWidget(acq_group)
        
        # Output directory selection
        dir_group = QtWidgets.QGroupBox("Output Directory")
        dir_layout = QtWidgets.QVBoxLayout(dir_group)
        
        dir_row = QtWidgets.QHBoxLayout()
        self.dir_label = QtWidgets.QLabel("No directory selected")
        self.dir_label.setStyleSheet("QLabel { color: #666; }")
        dir_row.addWidget(self.dir_label)
        dir_row.addStretch()
        
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_directory)
        dir_row.addWidget(self.browse_btn)
        
        dir_layout.addLayout(dir_row)
        layout.addWidget(dir_group)
        
        # Deconvolution parameters
        params_group = QtWidgets.QGroupBox("Deconvolution Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        
        # x0 parameter
        x0_layout = QtWidgets.QHBoxLayout()
        x0_layout.addWidget(QtWidgets.QLabel("x0 parameter:"))
        self.x0_spin = QtWidgets.QDoubleSpinBox()
        self.x0_spin.setRange(1.0, 20.0)
        self.x0_spin.setValue(7.0)
        self.x0_spin.setDecimals(1)
        self.x0_spin.setSingleStep(0.5)
        x0_layout.addWidget(self.x0_spin)
        x0_layout.addStretch()
        params_layout.addLayout(x0_layout)
        
        # Iterations parameter
        iter_layout = QtWidgets.QHBoxLayout()
        iter_layout.addWidget(QtWidgets.QLabel("Iterations:"))
        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 20)
        self.iterations_spin.setValue(4)
        iter_layout.addWidget(self.iterations_spin)
        iter_layout.addStretch()
        params_layout.addLayout(iter_layout)
        
        layout.addWidget(params_group)
        
        # Output format
        format_group = QtWidgets.QGroupBox("Output Format")
        format_layout = QtWidgets.QVBoxLayout(format_group)
        
        self.float_radio = QtWidgets.QRadioButton("Float (32-bit, preferred)")
        self.uint16_radio = QtWidgets.QRadioButton("16-bit unsigned integer")
        self.float_radio.setChecked(True)
        
        format_layout.addWidget(self.float_radio)
        format_layout.addWidget(self.uint16_radio)
        
        layout.addWidget(format_group)
        
        # Set scroll content widget
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area, 1)
        
        # Buttons (outside scroll area)
        button_layout = QtWidgets.QHBoxLayout()
        self.deconvolve_btn = QtWidgets.QPushButton("Deconvolve")
        self.deconvolve_btn.setEnabled(False)  # Disabled until directory is selected
        self.deconvolve_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.deconvolve_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.single_roi_radio.toggled.connect(self._on_acq_type_changed)
        self.whole_slide_radio.toggled.connect(self._on_acq_type_changed)
        
        # Initialize the display
        self._on_acq_type_changed()
        
    def _browse_directory(self):
        """Browse for output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Deconvolved OME-TIFF Files", ""
        )
        if directory:
            self.output_directory = directory
            self.dir_label.setText(directory)
            self.dir_label.setStyleSheet("QLabel { color: black; }")
            self.deconvolve_btn.setEnabled(True)
    
    def _on_acq_type_changed(self):
        """Update UI when acquisition type changes."""
        if self.single_roi_radio.isChecked():
            if self.current_acq_id:
                # Find current acquisition info
                current_acq = next((acq for acq in self.acquisitions if acq.id == self.current_acq_id), None)
                if current_acq:
                    info_text = f"Will deconvolve: {current_acq.name}\n"
                    info_text += f"Channels: {len(current_acq.channels)}\n"
                    if current_acq.well:
                        info_text += f"Well: {current_acq.well}\n"
                    # Show source file if available (for multiple files)
                    if current_acq.source_file:
                        source_name = os.path.basename(current_acq.source_file)
                        info_text += f"Source: {source_name}"
                    self.acq_info_label.setText(info_text)
                else:
                    self.acq_info_label.setText("Will deconvolve only the currently selected acquisition.")
            else:
                self.acq_info_label.setText("Will deconvolve only the currently selected acquisition.")
        else:
            # Show more detailed information about what will be deconvolved
            total_channels = sum(len(acq.channels) for acq in self.acquisitions)
            
            # Count files if multiple files are loaded
            source_files = set()
            for acq in self.acquisitions:
                if acq.source_file:
                    source_files.add(acq.source_file)
            
            info_text = f"Will deconvolve all {len(self.acquisitions)} acquisition(s)"
            if len(source_files) > 1:
                info_text += f" from {len(source_files)} file(s)"
            info_text += ".\n"
            info_text += f"Total channels: {total_channels}\n"
            info_text += f"Acquisitions: {', '.join([acq.name for acq in self.acquisitions[:3]])}"
            if len(self.acquisitions) > 3:
                info_text += f" and {len(self.acquisitions) - 3} more..."
            self.acq_info_label.setText(info_text)
    
    def get_acq_type(self):
        """Get the selected acquisition type."""
        return "single" if self.single_roi_radio.isChecked() else "whole"
    
    def get_output_directory(self):
        """Get the selected output directory."""
        return self.output_directory
    
    def get_x0(self):
        """Get the x0 parameter."""
        return self.x0_spin.value()
    
    def get_iterations(self):
        """Get the iterations parameter."""
        return self.iterations_spin.value()
    
    def get_output_format(self):
        """Get the output format: 'float' or 'uint16'."""
        return "float" if self.float_radio.isChecked() else "uint16"

