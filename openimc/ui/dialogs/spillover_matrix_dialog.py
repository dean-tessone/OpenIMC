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
Dialog for generating spillover matrices from single-stain MCD control files.
"""

from typing import Optional, List, Dict
import os
import re
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from openimc.processing.spillover_matrix import build_spillover_from_comp_mcd
from openimc.core import generate_spillover_matrix
from openimc.ui.dialogs.progress_dialog import ProgressDialog
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.data.mcd_loader import MCDLoader
from openimc.utils.logger import get_logger

# Optional seaborn for better heatmap styling
try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False


class SpilloverMatrixWorker(QThread):
    """Worker thread for computing spillover matrix from MCD file."""
    
    finished = Signal(pd.DataFrame, pd.DataFrame)
    error = Signal(str)
    
    def __init__(self, mcd_path, donor_label_per_acq, **kwargs):
        super().__init__()
        self.mcd_path = mcd_path
        self.donor_label_per_acq = donor_label_per_acq
        self.kwargs = kwargs
    
    def run(self):
        try:
            # Use core.generate_spillover_matrix for unified behavior
            S, qc = generate_spillover_matrix(
                mcd_path=self.mcd_path,
                donor_label_per_acq=self.donor_label_per_acq,
                **self.kwargs
            )
            self.finished.emit(S, qc)
        except Exception as e:
            import traceback
            print(f"Error computing spillover matrix: {e}")
            traceback.print_exc()
            self.error.emit(str(e))


class GenerateSpilloverMatrixDialog(QtWidgets.QDialog):
    """Dialog for generating spillover matrices from single-stain MCD control files."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.spillover_matrix = None
        self.qc_dataframe = None
        self.worker = None
        self.mcd_loader = None
        self.acquisitions = []
        self.channels = []
        self._filtering_in_progress = False  # Guard flag to prevent recursion
        self.mcd_file_path = None  # Store MCD file path for logging
        self.last_computation_params = None  # Store parameters for logging
        self.last_donor_mapping = None  # Store mapping for logging
        
        self.setWindowTitle("Generate Spillover Matrix")
        self.setModal(True)
        self.resize(1200, 900)
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Main tab widget
        self.main_tabs = QtWidgets.QTabWidget()
        
        # Tab 1: Configuration
        config_tab = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_tab)
        
        # Description
        desc_label = QtWidgets.QLabel(
            "Generate a spillover matrix from a single-stain control MCD file.\n"
            "Each acquisition should contain a single-stain control for one channel.\n"
            "The tool will automatically detect which channel each acquisition represents, "
            "or you can manually map them."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("QLabel { color: #666; margin-bottom: 10px; }")
        config_layout.addWidget(desc_label)
        
        # MCD file selection
        file_group = QtWidgets.QGroupBox("MCD File")
        file_layout = QtWidgets.QVBoxLayout(file_group)
        
        file_row = QtWidgets.QHBoxLayout()
        file_row.addWidget(QtWidgets.QLabel("File:"))
        self.mcd_file_edit = QtWidgets.QLineEdit()
        self.mcd_file_edit.setPlaceholderText("Select single-stain control MCD file...")
        self.mcd_file_edit.setReadOnly(True)
        self.mcd_file_btn = QtWidgets.QPushButton("Browse...")
        self.mcd_file_btn.clicked.connect(self._select_mcd_file)
        file_row.addWidget(self.mcd_file_edit)
        file_row.addWidget(self.mcd_file_btn)
        file_layout.addLayout(file_row)
        
        config_layout.addWidget(file_group)
        
        # Acquisition to donor mapping
        mapping_group = QtWidgets.QGroupBox("Acquisition to Donor Channel Mapping")
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)
        
        mapping_info = QtWidgets.QLabel(
            "Map each acquisition to its donor channel. The tool will attempt to auto-detect "
            "based on well names (e.g., 'In113' matches channel with metal 'In'). "
            "You can manually change the selection in the dropdown if auto-detection is incorrect."
        )
        mapping_info.setWordWrap(True)
        mapping_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        mapping_layout.addWidget(mapping_info)
        
        # Auto-detect button
        auto_detect_layout = QtWidgets.QHBoxLayout()
        self.auto_detect_btn = QtWidgets.QPushButton("Auto-detect Donors")
        self.auto_detect_btn.clicked.connect(self._auto_detect_donors)
        self.auto_detect_btn.setEnabled(False)
        auto_detect_layout.addWidget(self.auto_detect_btn)
        auto_detect_layout.addStretch()
        mapping_layout.addLayout(auto_detect_layout)
        
        # Mapping table
        self.mapping_table = QtWidgets.QTableWidget()
        self.mapping_table.setColumnCount(3)
        self.mapping_table.setHorizontalHeaderLabels(["Acquisition", "Name", "Donor Channel"])
        self.mapping_table.setAlternatingRowColors(True)
        self.mapping_table.horizontalHeader().setStretchLastSection(True)
        self.mapping_table.setMinimumHeight(250)
        self.mapping_table.setMaximumHeight(350)
        mapping_layout.addWidget(self.mapping_table)
        
        config_layout.addWidget(mapping_group)
        
        # Computation parameters - horizontal layout
        params_group = QtWidgets.QGroupBox("Computation Parameters")
        params_main_layout = QtWidgets.QHBoxLayout(params_group)
        
        # Left column
        left_params = QtWidgets.QGroupBox("Spillover Settings")
        left_layout = QtWidgets.QVBoxLayout(left_params)
        
        cap_layout = QtWidgets.QHBoxLayout()
        cap_layout.addWidget(QtWidgets.QLabel("Max spillover:"))
        self.cap_spin = QtWidgets.QDoubleSpinBox()
        self.cap_spin.setRange(0.0, 1.0)
        self.cap_spin.setSingleStep(0.05)
        self.cap_spin.setValue(0.3)
        self.cap_spin.setToolTip("Maximum spillover coefficient (cap)")
        cap_layout.addWidget(self.cap_spin)
        cap_layout.addStretch()
        left_layout.addLayout(cap_layout)
        
        agg_layout = QtWidgets.QHBoxLayout()
        agg_layout.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.aggregate_combo = QtWidgets.QComboBox()
        self.aggregate_combo.addItems(["median", "mean"])
        self.aggregate_combo.setToolTip("How to aggregate when multiple acquisitions per donor")
        agg_layout.addWidget(self.aggregate_combo)
        agg_layout.addStretch()
        left_layout.addLayout(agg_layout)
        
        params_main_layout.addWidget(left_params)
        
        # Right column
        right_params = QtWidgets.QGroupBox("Foreground Selection")
        right_layout = QtWidgets.QVBoxLayout(right_params)
        
        p_low_layout = QtWidgets.QHBoxLayout()
        p_low_layout.addWidget(QtWidgets.QLabel("Percentile (low):"))
        self.p_low_spin = QtWidgets.QDoubleSpinBox()
        self.p_low_spin.setRange(0.0, 100.0)
        self.p_low_spin.setSingleStep(1.0)
        self.p_low_spin.setValue(90.0)
        self.p_low_spin.setToolTip("Lower percentile for foreground pixel selection")
        p_low_layout.addWidget(self.p_low_spin)
        p_low_layout.addStretch()
        right_layout.addLayout(p_low_layout)
        
        p_high_layout = QtWidgets.QHBoxLayout()
        p_high_layout.addWidget(QtWidgets.QLabel("Percentile (high):"))
        self.p_high_spin = QtWidgets.QDoubleSpinBox()
        self.p_high_spin.setRange(0.0, 100.0)
        self.p_high_spin.setSingleStep(0.1)
        self.p_high_spin.setValue(99.9)
        self.p_high_spin.setToolTip("Upper percentile for foreground pixel clipping")
        p_high_layout.addWidget(self.p_high_spin)
        p_high_layout.addStretch()
        right_layout.addLayout(p_high_layout)
        
        channel_field_layout = QtWidgets.QHBoxLayout()
        channel_field_layout.addWidget(QtWidgets.QLabel("Channel field:"))
        self.channel_field_combo = QtWidgets.QComboBox()
        self.channel_field_combo.addItems(["name", "fullname"])
        self.channel_field_combo.setToolTip("Which field to use for channel names from MCD")
        channel_field_layout.addWidget(self.channel_field_combo)
        channel_field_layout.addStretch()
        right_layout.addLayout(channel_field_layout)
        
        params_main_layout.addWidget(right_params)
        params_main_layout.addStretch()
        
        config_layout.addWidget(params_group)
        
        # Buttons at bottom of config tab
        button_layout = QtWidgets.QHBoxLayout()
        
        self.generate_btn = QtWidgets.QPushButton("Generate Matrix")
        self.generate_btn.clicked.connect(self._generate_matrix)
        self.generate_btn.setEnabled(False)
        button_layout.addWidget(self.generate_btn)
        
        self.save_btn = QtWidgets.QPushButton("Save Matrix...")
        self.save_btn.clicked.connect(self._save_matrix)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        self.cancel_btn = QtWidgets.QPushButton("Close")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        config_layout.addLayout(button_layout)
        config_layout.addStretch()
        
        self.main_tabs.addTab(config_tab, "Configuration")
        
        # Tab 2: Matrix Visualization
        viz_tab = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_tab)
        
        # Heatmap canvas
        self.matrix_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        viz_layout.addWidget(self.matrix_canvas)
        
        # Save button for visualization
        viz_btn_layout = QtWidgets.QHBoxLayout()
        self.save_plot_btn = QtWidgets.QPushButton("Save Plot...")
        self.save_plot_btn.clicked.connect(self._save_matrix_plot)
        self.save_plot_btn.setEnabled(False)
        viz_btn_layout.addWidget(self.save_plot_btn)
        viz_btn_layout.addStretch()
        viz_layout.addLayout(viz_btn_layout)
        
        self.main_tabs.addTab(viz_tab, "Matrix Visualization")
        
        # Tab 3: Matrix Table
        matrix_tab = QtWidgets.QWidget()
        matrix_layout = QtWidgets.QVBoxLayout(matrix_tab)
        
        # Info label
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { color: #0066cc; font-size: 9pt; margin-bottom: 5px; }")
        matrix_layout.addWidget(self.info_label)
        
        self.matrix_table = QtWidgets.QTableWidget()
        self.matrix_table.setAlternatingRowColors(True)
        matrix_layout.addWidget(self.matrix_table)
        
        self.main_tabs.addTab(matrix_tab, "Matrix Table")
        
        # Tab 4: QC Metrics
        qc_tab = QtWidgets.QWidget()
        qc_layout = QtWidgets.QVBoxLayout(qc_tab)
        
        qc_info = QtWidgets.QLabel(
            "Quality control metrics for spillover matrix computation:\n"
            "- n_acqs: Number of acquisitions used per donor\n"
            "- offdiag_sum: Sum of off-diagonal spillover values\n"
            "- offdiag_max: Maximum off-diagonal spillover value\n"
            "- pixels_used_median: Median number of pixels used for estimation"
        )
        qc_info.setWordWrap(True)
        qc_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; margin-bottom: 5px; }")
        qc_layout.addWidget(qc_info)
        
        self.qc_table = QtWidgets.QTableWidget()
        self.qc_table.setAlternatingRowColors(True)
        qc_layout.addWidget(self.qc_table)
        
        self.main_tabs.addTab(qc_tab, "QC Metrics")
        
        # Disable result tabs initially
        self.main_tabs.setTabEnabled(1, False)  # Matrix Visualization
        self.main_tabs.setTabEnabled(2, False)  # Matrix Table
        self.main_tabs.setTabEnabled(3, False)  # QC Metrics
        
        layout.addWidget(self.main_tabs)
    
    def _select_mcd_file(self):
        """Open file dialog to select MCD file."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Single-Stain Control MCD File",
            "",
            "MCD Files (*.mcd);;All Files (*)"
        )
        
        if file_path:
            self.mcd_file_edit.setText(file_path)
            self.mcd_file_path = file_path
            self._load_mcd_file(file_path)
    
    def _load_mcd_file(self, file_path: str):
        """Load MCD file and populate acquisition list."""
        try:
            self.mcd_loader = MCDLoader()
            self.mcd_loader.open(file_path)
            self.acquisitions = self.mcd_loader.list_acquisitions(source_file=file_path)
            
            if not self.acquisitions:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Acquisitions",
                    "No acquisitions found in the MCD file."
                )
                return
            
            # Get channels from first acquisition (they should be the same across acquisitions)
            if self.acquisitions:
                self.channels = self.acquisitions[0].channels
            
            # Populate mapping table
            self._populate_mapping_table()
            
            # Enable auto-detect and generate buttons
            self.auto_detect_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
            
            # Auto-detect donors
            self._auto_detect_donors()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Error loading MCD file:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _populate_mapping_table(self):
        """Populate the acquisition to donor mapping table."""
        self.mapping_table.setRowCount(len(self.acquisitions))
        
        for i, acq in enumerate(self.acquisitions):
            # Acquisition ID
            id_item = QtWidgets.QTableWidgetItem(acq.id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.mapping_table.setItem(i, 0, id_item)
            
            # Use well name if available, otherwise use acquisition name
            display_name = acq.well if acq.well else acq.name
            name_item = QtWidgets.QTableWidgetItem(display_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            # Store well in data role for detection
            name_item.setData(Qt.UserRole, acq.well)
            name_item.setData(Qt.UserRole + 1, acq.name)
            self.mapping_table.setItem(i, 1, name_item)
            
            # Donor channel (editable combobox with search)
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            combo.lineEdit().setPlaceholderText("Search or select channel...")
            combo.addItem("")  # Empty option
            combo.addItems(self.channels)
            
            # Enable search/filter functionality
            # Create a proper closure to avoid lambda capture issues
            def make_filter_callback(c):
                def callback(text):
                    if self._filtering_in_progress:
                        return
                    try:
                        self._filter_combo_channels(c, text)
                    except Exception:
                        pass  # Silently fail to avoid recursion
                return callback
            
            combo.lineEdit().textChanged.connect(make_filter_callback(combo))
            
            self.mapping_table.setCellWidget(i, 2, combo)
        
        self.mapping_table.resizeColumnsToContents()
    
    def _filter_combo_channels(self, combo: QtWidgets.QComboBox, filter_text: str):
        """Filter combobox items based on search text."""
        # Check guard flag
        if self._filtering_in_progress:
            return
        
        # Set guard flag
        self._filtering_in_progress = True
        
        try:
            # Block signals on both combo and line edit
            combo.blockSignals(True)
            line_edit = combo.lineEdit()
            if line_edit:
                line_edit.blockSignals(True)
            
            # Store current selection (but don't rely on it if it's not in channels)
            current_text = combo.currentText()
            
            # Only preserve selection if it's a valid channel
            preserve_selection = current_text in self.channels
            
            # Clear and add empty option
            combo.clear()
            combo.addItem("")
            
            # Add filtered channels
            if filter_text:
                filter_lower = filter_text.lower()
                matching_channels = [ch for ch in self.channels if filter_lower in ch.lower()]
                combo.addItems(matching_channels)
            else:
                combo.addItems(self.channels)
            
            # Restore selection if it still exists and we want to preserve it
            if preserve_selection:
                index = combo.findText(current_text)
                if index >= 0:
                    combo.setCurrentIndex(index)
                else:
                    combo.setCurrentIndex(0)
            else:
                # No valid selection to preserve, just show the filter text
                combo.setCurrentIndex(0)
                if filter_text and line_edit:
                    # Set the line edit text directly without triggering signals
                    line_edit.setText(filter_text)
        except Exception:
            pass  # Silently fail to avoid recursion
        finally:
            # Always unblock signals and clear guard
            combo.blockSignals(False)
            if line_edit:
                line_edit.blockSignals(False)
            self._filtering_in_progress = False
    
    def _extract_metal_from_name(self, name: str) -> Optional[str]:
        """Extract metal symbol from a name (e.g., 'In113' -> 'In', 'Yb176' -> 'Yb')."""
        if not name:
            return None
        
        # Try to match common metal patterns (1-2 letters followed by numbers)
        # Metals are typically 1-2 uppercase letters
        match = re.match(r'^([A-Z][a-z]?|[A-Z]{2})', name)
        if match:
            return match.group(1)
        
        # Try lowercase
        match = re.match(r'^([a-z][a-z]?|[a-z]{2})', name)
        if match:
            return match.group(1).capitalize()
        
        return None
    
    def _extract_metal_from_channel(self, channel: str, channel_metals: Optional[List[str]] = None, channel_idx: Optional[int] = None) -> Optional[str]:
        """Extract metal symbol from channel name."""
        # First try to use channel_metals if available (most reliable)
        if channel_metals and channel_idx is not None and channel_idx < len(channel_metals):
            metal = channel_metals[channel_idx]
            if metal:
                return metal
        
        # Channel might be "Label_Metal", "Metal", or "MetalNumber"
        # Try to extract metal from various patterns
        
        # Pattern 1: "Label_Metal" or "Label_MetalMass"
        parts = channel.split('_')
        if len(parts) > 1:
            # Second part might be "Metal" or "MetalMass"
            metal_part = parts[1]
            metal = self._extract_metal_from_name(metal_part)
            if metal:
                return metal
        
        # Pattern 2: Direct metal name at start (e.g., "Yb176", "In113")
        metal = self._extract_metal_from_name(channel)
        if metal:
            return metal
        
        return None
    
    def _auto_detect_donors(self):
        """Auto-detect donor channels from well/acquisition names."""
        for i, acq in enumerate(self.acquisitions):
            # Use well name if available, otherwise use acquisition name
            search_name = acq.well if acq.well else acq.name
            if not search_name:
                continue
            
            # Extract metal from well/acquisition name (e.g., "In113" -> "In")
            well_metal = self._extract_metal_from_name(search_name)
            
            donor_channel = None
            if well_metal:
                # Try to find channel with matching metal
                # Use channel_metals if available for more reliable matching
                for j, ch in enumerate(self.channels):
                    channel_metal = self._extract_metal_from_channel(
                        ch, 
                        channel_metals=acq.channel_metals if hasattr(acq, 'channel_metals') else None,
                        channel_idx=j
                    )
                    if channel_metal and channel_metal.lower() == well_metal.lower():
                        donor_channel = ch
                        break
            
            # Fallback: try substring matching if metal extraction didn't work
            if donor_channel is None:
                search_name_lower = search_name.lower()
                for j, ch in enumerate(self.channels):
                    # Try full channel name match
                    if ch.lower() in search_name_lower:
                        # Check if this is a unique match
                        other_matches = [c for c in self.channels if c != ch and c.lower() in search_name_lower]
                        if not other_matches:
                            donor_channel = ch
                            break
                    
                    # Try metal-only match
                    channel_metal = self._extract_metal_from_channel(
                        ch,
                        channel_metals=acq.channel_metals if hasattr(acq, 'channel_metals') else None,
                        channel_idx=j
                    )
                    if channel_metal and channel_metal.lower() in search_name_lower:
                        donor_channel = ch
                        break
            
            # Set in combobox
            if donor_channel:
                combo = self.mapping_table.cellWidget(i, 2)
                if combo:
                    index = combo.findText(donor_channel)
                    if index >= 0:
                        combo.setCurrentIndex(index)
        
    
    def _get_donor_mapping(self) -> Dict[str, str]:
        """Get donor channel mapping from table.
        
        Returns mapping that can be used by position/index since readimc
        may use different ID formats when reopening the MCD file.
        """
        # Build mapping by position/index since acquisitions are in the same order
        mapping_by_index = {}
        mapping_by_all_ids = {}
        
        for i in range(self.mapping_table.rowCount()):
            acq_id = self.mapping_table.item(i, 0).text()
            combo = self.mapping_table.cellWidget(i, 2)
            if combo:
                donor_ch = combo.currentText().strip()
                if donor_ch:
                    # Map by index position (most reliable since order is preserved)
                    mapping_by_index[i] = donor_ch
                    
                    # Also build comprehensive mapping by all possible IDs
                    mapping_by_all_ids[acq_id] = donor_ch
                    
                    # Map by acquisition name (well name) for fallback
                    name_item = self.mapping_table.item(i, 1)
                    if name_item:
                        well_name = name_item.data(Qt.UserRole)  # well name
                        acq_name = name_item.data(Qt.UserRole + 1)  # acquisition name
                        display_name = well_name if well_name else acq_name
                        if display_name:
                            mapping_by_all_ids[display_name] = donor_ch
                    
                    # Try to extract numeric ID from our format (e.g., "slide_0_acq_1" -> "1")
                    try:
                        parts = acq_id.split('_')
                        if len(parts) >= 3 and parts[0] == 'slide' and parts[2] == 'acq':
                            acq_num = parts[-1]  # Last part is the acquisition number
                            mapping_by_all_ids[acq_num] = donor_ch
                            mapping_by_all_ids[int(acq_num)] = donor_ch
                    except (ValueError, IndexError):
                        pass
                    
                    # Also try mapping by the actual acquisition object if available
                    if i < len(self.acquisitions):
                        acq = self.acquisitions[i]
                        if hasattr(acq, 'id'):
                            mapping_by_all_ids[str(acq.id)] = donor_ch
                        if hasattr(acq, 'name') and acq.name:
                            mapping_by_all_ids[acq.name] = donor_ch
        
        # Store both mappings - index mapping will be used as fallback
        return {
            '_by_index': mapping_by_index,
            '_by_id': mapping_by_all_ids
        }
    
    def _generate_matrix(self):
        """Generate the spillover matrix."""
        file_path = self.mcd_file_edit.text().strip()
        if not file_path or not os.path.exists(file_path):
            QtWidgets.QMessageBox.warning(
                self,
                "No File",
                "Please select an MCD file first."
            )
            return
        
        # Get donor mapping
        donor_mapping_dict = self._get_donor_mapping()
        mapping_by_index = donor_mapping_dict.get('_by_index', {})
        mapping_by_id = donor_mapping_dict.get('_by_id', {})
        
        if not mapping_by_index and not mapping_by_id:
            QtWidgets.QMessageBox.warning(
                self,
                "No Mapping",
                "Please map at least one acquisition to a donor channel."
            )
            return
        
        # Get parameters
        cap = self.cap_spin.value()
        aggregate = self.aggregate_combo.currentText()
        p_low = self.p_low_spin.value()
        p_high = self.p_high_spin.value()
        channel_field = self.channel_field_combo.currentText()
        
        # Store parameters and mapping for logging
        self.last_computation_params = {
            "cap": cap,
            "aggregate": aggregate,
            "p_low": p_low,
            "p_high": p_high,
            "channel_field": channel_field
        }
        # Store mapping with both index and ID mappings for logging
        # Keep original integer keys for index mapping
        self.last_donor_mapping = {
            "_by_index": mapping_by_index.copy(),
            "_by_id": {str(k): str(v) for k, v in mapping_by_id.items()}
        }
        
        # Show progress dialog
        progress = ProgressDialog("Computing spillover matrix from MCD file...", self)
        progress.show()
        
        # Create worker - pass both mappings, function will use index-based mapping
        self.worker = SpilloverMatrixWorker(
            file_path,
            {'_by_index': mapping_by_index, '_by_id': mapping_by_id},
            cap=cap,
            aggregate=aggregate,
            p_low=p_low,
            p_high_clip=p_high,
            channel_name_field=channel_field
        )
        self.worker.finished.connect(lambda S, qc: self._on_matrix_computed(S, qc, progress))
        self.worker.error.connect(lambda msg: self._on_matrix_error(msg, progress))
        
        self.generate_btn.setEnabled(False)
        self.worker.start()
    
    def _on_matrix_computed(self, S: pd.DataFrame, qc: pd.DataFrame, progress: ProgressDialog):
        """Handle successful matrix computation."""
        progress.close()
        self.spillover_matrix = S
        self.qc_dataframe = qc
        self._display_matrix(S)
        self._display_matrix_plot(S)
        self._display_qc(qc)
        self.save_btn.setEnabled(True)
        self.save_plot_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        
        # Enable result tabs
        self.main_tabs.setTabEnabled(1, True)  # Matrix Visualization
        self.main_tabs.setTabEnabled(2, True)  # Matrix Table
        self.main_tabs.setTabEnabled(3, True)  # QC Metrics
        
        # Update info
        n_channels = len(S)
        n_nonzero = (S.values != 0).sum() - n_channels  # Exclude diagonal
        self.info_label.setText(
            f"✓ Matrix computed successfully\n"
            f"Size: {n_channels}x{n_channels}\n"
            f"Non-zero off-diagonal entries: {n_nonzero}"
        )
        self.info_label.setStyleSheet("QLabel { color: #006600; font-size: 9pt; }")
        
        # Log to methods log
        if self.last_computation_params and self.last_donor_mapping:
            logger = get_logger()
            # Get acquisition IDs
            acquisition_ids = [acq.id for acq in self.acquisitions] if self.acquisitions else []
            # Get source file name
            source_file = os.path.basename(self.mcd_file_path) if self.mcd_file_path else None
            
            # Create a readable representation of the donor mapping
            # Show the mapping by index (most reliable) and include human-readable acquisition info
            readable_mapping = {}
            mapping_by_index = self.last_donor_mapping.get('_by_index', {})
            for i, acq in enumerate(self.acquisitions):
                # Check if this acquisition index has a donor mapping
                if i in mapping_by_index:
                    donor_ch = mapping_by_index[i]
                    acq_name = acq.well if acq.well else acq.name
                    acq_id = acq.id
                    readable_mapping[acq_id] = {
                        "acquisition_name": acq_name,
                        "donor_channel": donor_ch
                    }
            
            logger.log_spillover_matrix(
                parameters=self.last_computation_params,
                donor_mapping=readable_mapping,
                acquisitions=acquisition_ids,
                notes=f"Generated {n_channels}x{n_channels} spillover matrix with {n_nonzero} non-zero off-diagonal entries",
                source_file=source_file
            )
    
    def _on_matrix_error(self, msg: str, progress: ProgressDialog):
        """Handle matrix computation error."""
        progress.close()
        self.generate_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(
            self,
            "Error",
            f"Error computing spillover matrix:\n{msg}"
        )
    
    def _display_matrix(self, S: pd.DataFrame):
        """Display the spillover matrix in the table."""
        self.matrix_table.setRowCount(len(S))
        self.matrix_table.setColumnCount(len(S.columns))
        self.matrix_table.setHorizontalHeaderLabels(S.columns.tolist())
        self.matrix_table.setVerticalHeaderLabels(S.index.tolist())
        
        # Populate table (show all entries)
        for i in range(len(S)):
            for j in range(len(S.columns)):
                value = S.iloc[i, j]
                item = QtWidgets.QTableWidgetItem(f"{value:.6f}")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                
                # Color code: diagonal = green, off-diagonal > 0 = yellow/orange, zero = white
                if i == j:
                    item.setBackground(QtGui.QColor(230, 255, 230))  # Light green for diagonal
                elif value > 0:
                    # Gradient: higher values = more orange
                    intensity = min(255, int(value * 255 * 3))  # Scale for visibility
                    item.setBackground(QtGui.QColor(255, 255 - intensity, 230))
                
                self.matrix_table.setItem(i, j, item)
        
        self.matrix_table.resizeColumnsToContents()
    
    def _display_matrix_plot(self, S: pd.DataFrame):
        """Display the spillover matrix as a heatmap."""
        import matplotlib.pyplot as plt
        
        fig = self.matrix_canvas.figure
        fig.clear()
        
        ax = fig.add_subplot(111)
        
        # Prepare data for plotting
        matrix_data = S.values
        
        # Determine rotation and tick spacing based on number of channels
        n_channels = len(S.columns)
        if n_channels > 30:
            # Many channels: rotate 90 degrees, show every 3rd-5th label
            x_rotation = 90
            tick_interval = max(1, n_channels // 30)
            tick_fontsize = 7
        elif n_channels > 15:
            # Medium number: rotate 45 degrees, show every 2nd label
            x_rotation = 45
            tick_interval = 2
            tick_fontsize = 8
        else:
            # Few channels: rotate 45 degrees, show all labels
            x_rotation = 45
            tick_interval = 1
            tick_fontsize = 9
        
        # Prepare tick labels
        x_labels = S.columns.tolist()
        y_labels = S.index.tolist()
        
        # Thin out x-axis labels if needed
        if tick_interval > 1:
            x_labels_thinned = [label if i % tick_interval == 0 else '' for i, label in enumerate(x_labels)]
        else:
            x_labels_thinned = x_labels
        
        # Use seaborn if available for better styling
        if _HAVE_SEABORN:
            sns.heatmap(
                matrix_data,
                ax=ax,
                xticklabels=True,  # Let seaborn create all ticks
                yticklabels=True,
                cmap='YlOrRd',  # Yellow-Orange-Red (yellow=low, red=high spillover)
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Spillover Coefficient'},
                square=True,
                linewidths=0.5,
                linecolor='gray',
                annot=False,  # Don't annotate to avoid clutter
                fmt='.3f'
            )
            
            # Override x-axis labels with proper rotation and thinning
            if tick_interval > 1:
                # Set all ticks but show only every Nth label
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels_thinned, rotation=x_rotation, ha='right' if x_rotation == 45 else 'center', fontsize=tick_fontsize)
            else:
                # Set explicit labels with rotation
                ax.set_xticks(range(len(x_labels)))
                ax.set_xticklabels(x_labels, rotation=x_rotation, ha='right' if x_rotation == 45 else 'center', fontsize=tick_fontsize)
            
            # Set y-axis labels with proper font size
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, rotation=0, fontsize=tick_fontsize)
        else:
            # Fallback to matplotlib
            im = ax.imshow(matrix_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(len(S.columns)))
            ax.set_yticks(range(len(S.index)))
            ax.set_xticklabels(x_labels, rotation=x_rotation, ha='right' if x_rotation == 45 else 'center', fontsize=tick_fontsize)
            ax.set_yticklabels(y_labels, fontsize=tick_fontsize)
            fig.colorbar(im, ax=ax, label='Spillover Coefficient')
        
        ax.set_xlabel('Receiver Channel', fontsize=10)
        ax.set_ylabel('Donor Channel', fontsize=10)
        ax.set_title('Spillover Matrix', fontsize=12, fontweight='bold')
        
        # Adjust layout to accommodate rotated labels
        fig.tight_layout(rect=[0, 0.03, 1, 0.97] if x_rotation == 90 else None)
        self.matrix_canvas.draw()
    
    def _display_qc(self, qc: pd.DataFrame):
        """Display QC metrics in the table."""
        self.qc_table.setRowCount(len(qc))
        self.qc_table.setColumnCount(len(qc.columns))
        self.qc_table.setHorizontalHeaderLabels(qc.columns.tolist())
        self.qc_table.setVerticalHeaderLabels(qc.index.tolist())
        
        for i in range(len(qc)):
            for j in range(len(qc.columns)):
                value = qc.iloc[i, j]
                if isinstance(value, (int, np.integer)):
                    text = str(value)
                elif isinstance(value, float):
                    text = f"{value:.6f}"
                else:
                    text = str(value) if value is not None else ""
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.qc_table.setItem(i, j, item)
        
        self.qc_table.resizeColumnsToContents()
    
    def _save_matrix_plot(self):
        """Save the spillover matrix visualization."""
        if self.spillover_matrix is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Matrix",
                "No spillover matrix to save. Please generate one first."
            )
            return
        
        if save_figure_with_options(
            self.matrix_canvas.figure,
            "spillover_matrix.png",
            self
        ):
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                "Matrix visualization saved successfully."
            )
    
    def _save_matrix(self):
        """Save the spillover matrix to CSV."""
        if self.spillover_matrix is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Matrix",
                "No spillover matrix to save. Please generate one first."
            )
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Spillover Matrix",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.spillover_matrix.to_csv(file_path)
                
                # Also save QC if available
                if self.qc_dataframe is not None:
                    qc_path = file_path.replace('.csv', '_qc.csv')
                    self.qc_dataframe.to_csv(qc_path)
                    QtWidgets.QMessageBox.information(
                        self,
                        "Success",
                        f"Spillover matrix saved to:\n{file_path}\n\n"
                        f"QC metrics saved to:\n{qc_path}"
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Success",
                        f"Spillover matrix saved to:\n{file_path}"
                    )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Error saving spillover matrix:\n{str(e)}"
                )
    
    def get_spillover_matrix(self):
        """Get the generated spillover matrix."""
        return self.spillover_matrix
