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
Batch Correction Dialog for OpenIMC

This module provides batch correction capabilities using Combat or Harmony
to correct for batch effects in feature data from multiple files.
"""

from typing import Optional, Dict, List
import os
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from openimc.processing.batch_correction import (
    apply_combat_correction,
    apply_harmony_correction,
    validate_batch_correction_inputs
)

# Optional imports for batch correction methods
try:
    from combat.pycombat import pycombat
    _HAVE_COMBAT = True
except ImportError:
    _HAVE_COMBAT = False

try:
    from harmonypy import run_harmony
    _HAVE_HARMONY = True
except ImportError:
    _HAVE_HARMONY = False



class BatchCorrectionDialog(QtWidgets.QDialog):
    """Dialog for batch correction of feature data."""
    
    def __init__(self, feature_dataframe: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Correction")
        self.setModal(True)
        self.resize(900, 700)
        
        self.feature_dataframe = feature_dataframe
        self.corrected_dataframe: Optional[pd.DataFrame] = None
        
        self._create_ui()
        self._update_ui_state()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Create scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        scroll_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(8)
        
        # Information section
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)
        
        info_label = QtWidgets.QLabel(
            "Batch correction removes technical variation (batch effects) between different files or batches.\n"
            "This is useful when combining features from multiple .mcd files or uploaded feature files.\n\n"
            "You can load additional feature files extracted by this app, or use the currently loaded features.\n\n"
            "Note: All features will be preserved in their original state in the CSV if they are not batch corrected."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        info_layout.addWidget(info_label)
        
        content_layout.addWidget(info_group)
        
        # Data source section
        source_group = QtWidgets.QGroupBox("Data Source")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(4)
        
        # Radio buttons for data source
        self.use_current_radio = QtWidgets.QRadioButton("Use currently loaded features")
        self.use_current_radio.setChecked(True)
        self.use_current_radio.toggled.connect(self._on_source_changed)
        source_layout.addWidget(self.use_current_radio)
        
        self.load_files_radio = QtWidgets.QRadioButton("Load feature files (CSV)")
        self.load_files_radio.toggled.connect(self._on_source_changed)
        source_layout.addWidget(self.load_files_radio)
        
        # File list for loaded files (hidden by default)
        self.file_list_widget = QtWidgets.QWidget()
        file_list_layout = QtWidgets.QVBoxLayout(self.file_list_widget)
        file_list_layout.setContentsMargins(20, 4, 0, 4)
        
        file_list_label = QtWidgets.QLabel("Loaded feature files:")
        file_list_layout.addWidget(file_list_label)
        
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.setEnabled(False)
        file_list_layout.addWidget(self.file_list)
        
        file_buttons_layout = QtWidgets.QHBoxLayout()
        self.add_file_btn = QtWidgets.QPushButton("Add File...")
        self.add_file_btn.setEnabled(False)
        self.add_file_btn.clicked.connect(self._add_feature_file)
        file_buttons_layout.addWidget(self.add_file_btn)
        
        self.remove_file_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_file_btn.setEnabled(False)
        self.remove_file_btn.clicked.connect(self._remove_feature_file)
        file_buttons_layout.addWidget(self.remove_file_btn)
        file_buttons_layout.addStretch()
        file_list_layout.addLayout(file_buttons_layout)
        
        self.file_list_widget.setVisible(False)  # Hidden by default
        source_layout.addWidget(self.file_list_widget)
        
        content_layout.addWidget(source_group)
        
        # Batch correction method section
        method_group = QtWidgets.QGroupBox("Batch Correction Method")
        method_layout = QtWidgets.QVBoxLayout(method_group)
        method_layout.setContentsMargins(8, 8, 8, 8)
        method_layout.setSpacing(4)
        
        method_label = QtWidgets.QLabel("Select batch correction method:")
        method_layout.addWidget(method_label)
        
        self.method_combo = QtWidgets.QComboBox()
        # Add methods in priority order: Harmony, Combat
        if _HAVE_HARMONY:
            self.method_combo.addItem("Harmony")
        if _HAVE_COMBAT:
            self.method_combo.addItem("Combat")
        
        # Set default to Harmony if available, otherwise Combat
        if _HAVE_HARMONY:
            self.method_combo.setCurrentText("Harmony")
        elif _HAVE_COMBAT:
            self.method_combo.setCurrentText("Combat")
        
        if self.method_combo.count() == 0:
            self.method_combo.addItem("No methods available")
            self.method_combo.setEnabled(False)
            no_methods_label = QtWidgets.QLabel(
                "Please install batch correction libraries:\n"
                "  - Combat: pip install combat\n"
                "  - Harmony: pip install harmonypy"
            )
            no_methods_label.setStyleSheet("QLabel { color: #d9534f; font-size: 9pt; }")
            method_layout.addWidget(no_methods_label)
        
        method_layout.addWidget(self.method_combo)
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        
        # PCA variance threshold for Harmony (hidden by default, shown only for Harmony)
        self.pca_variance_layout = QtWidgets.QHBoxLayout()
        self.pca_variance_layout.addWidget(QtWidgets.QLabel("PCA variance to retain:"))
        self.pca_variance_spin = QtWidgets.QDoubleSpinBox()
        self.pca_variance_spin.setRange(0.1, 1.0)
        self.pca_variance_spin.setSingleStep(0.05)
        self.pca_variance_spin.setValue(0.9)
        self.pca_variance_spin.setDecimals(2)
        self.pca_variance_spin.setToolTip(
            "Proportion of variance to retain in PCA before applying Harmony.\n"
            "Higher values retain more information but may be slower. Default: 90%"
        )
        self.pca_variance_spin.valueChanged.connect(self._update_pca_variance_suffix)
        # Set initial suffix
        self._update_pca_variance_suffix(0.9)
        self.pca_variance_layout.addWidget(self.pca_variance_spin)
        self.pca_variance_layout.addStretch()
        self.pca_variance_widget = QtWidgets.QWidget()
        self.pca_variance_widget.setLayout(self.pca_variance_layout)
        self.pca_variance_widget.setVisible(False)  # Hidden by default
        method_layout.addWidget(self.pca_variance_widget)
        
        # Batch variable selection
        batch_var_layout = QtWidgets.QHBoxLayout()
        batch_var_layout.addWidget(QtWidgets.QLabel("Batch variable:"))
        self.batch_var_combo = QtWidgets.QComboBox()
        self.batch_var_combo.addItems(["source_file", "acquisition_id"])
        self.batch_var_combo.setToolTip(
            "Variable to use for batch identification.\n"
            "'source_file' groups by file name, 'acquisition_id' groups by acquisition."
        )
        batch_var_layout.addWidget(self.batch_var_combo)
        batch_var_layout.addStretch()
        method_layout.addLayout(batch_var_layout)
        
        content_layout.addWidget(method_group)
        
        # Feature selection section
        feature_group = QtWidgets.QGroupBox("Features to Correct")
        feature_layout = QtWidgets.QVBoxLayout(feature_group)
        feature_layout.setContentsMargins(8, 8, 8, 8)
        feature_layout.setSpacing(4)
        
        feature_info = QtWidgets.QLabel(
            "Select which features to apply batch correction to.\n"
            "Batch correction is typically applied to intensity features (marker expression),\n"
            "not morphological features (cell size, shape, etc.), as batch effects primarily\n"
            "affect staining and signal intensity rather than cell morphology.\n"
            "Non-feature columns (label, cell_id, acquisition_id, etc.) will be preserved."
        )
        feature_info.setWordWrap(True)
        feature_info.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        feature_layout.addWidget(feature_info)
        
        # Feature filter section
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter features:"))
        
        self.filter_mean_chk = QtWidgets.QCheckBox("_mean")
        self.filter_mean_chk.setChecked(True)
        self.filter_mean_chk.toggled.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_mean_chk)
        
        self.filter_median_chk = QtWidgets.QCheckBox("_median")
        self.filter_median_chk.setChecked(True)
        self.filter_median_chk.toggled.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_median_chk)
        
        self.filter_other_chk = QtWidgets.QCheckBox("Other features")
        self.filter_other_chk.setChecked(False)
        self.filter_other_chk.toggled.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_other_chk)
        
        filter_layout.addStretch()
        feature_layout.addLayout(filter_layout)
        
        self.feature_list = QtWidgets.QListWidget()
        self.feature_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.feature_list.setMaximumHeight(150)
        feature_layout.addWidget(self.feature_list)
        
        feature_buttons_layout = QtWidgets.QHBoxLayout()
        
        self.select_all_features_btn = QtWidgets.QPushButton("Select All")
        self.select_all_features_btn.clicked.connect(self._select_all_features)
        feature_buttons_layout.addWidget(self.select_all_features_btn)
        
        self.deselect_all_features_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_features_btn.clicked.connect(self._deselect_all_features)
        feature_buttons_layout.addWidget(self.deselect_all_features_btn)
        feature_buttons_layout.addStretch()
        feature_layout.addLayout(feature_buttons_layout)
        
        content_layout.addWidget(feature_group)
        
        # Output section
        output_group = QtWidgets.QGroupBox("Output")
        output_layout = QtWidgets.QVBoxLayout(output_group)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(4)
        
        save_layout = QtWidgets.QHBoxLayout()
        self.save_output_chk = QtWidgets.QCheckBox("Save corrected features to CSV")
        self.save_output_chk.setChecked(False)
        save_layout.addWidget(self.save_output_chk)
        save_layout.addStretch()
        output_layout.addLayout(save_layout)
        
        output_path_layout = QtWidgets.QHBoxLayout()
        output_path_layout.addWidget(QtWidgets.QLabel("Output path:"))
        self.output_path_edit = QtWidgets.QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output file...")
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setEnabled(False)
        self.output_path_btn = QtWidgets.QPushButton("Browse...")
        self.output_path_btn.setEnabled(False)
        self.output_path_btn.clicked.connect(self._select_output_path)
        output_path_layout.addWidget(self.output_path_edit)
        output_path_layout.addWidget(self.output_path_btn)
        output_layout.addLayout(output_path_layout)
        
        content_layout.addWidget(output_group)
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.correct_btn = QtWidgets.QPushButton("Apply Batch Correction")
        self.correct_btn.clicked.connect(self._apply_batch_correction)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.correct_btn)
        layout.addLayout(button_layout)
        
        # Store loaded files
        self.loaded_files: List[str] = []
    
    def _on_source_changed(self):
        """Handle data source radio button change."""
        use_current = self.use_current_radio.isChecked()
        # Show/hide file list widget based on selection
        self.file_list_widget.setVisible(not use_current)
        self.file_list.setEnabled(not use_current)
        self.add_file_btn.setEnabled(not use_current)
        self.remove_file_btn.setEnabled(not use_current)
        
        if use_current:
            self._populate_features()
        else:
            # Clear feature list until files are loaded
            self.feature_list.clear()
    
    def _on_method_changed(self, method_name: str):
        """Handle batch correction method change."""
        # Show PCA variance control only for Harmony
        self.pca_variance_widget.setVisible(method_name == "Harmony")
    
    def _update_pca_variance_suffix(self, value: float):
        """Update the suffix display for PCA variance spinbox."""
        percentage = int(value * 100)
        self.pca_variance_spin.setSuffix(f" ({percentage}%)")
    
    def _add_feature_file(self):
        """Add a feature file to the list."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Feature CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            # Validate that it's a valid feature file
            try:
                df = pd.read_csv(file_path)
                # Check for required columns
                if 'cell_id' not in df.columns:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid File",
                        f"The file does not appear to be a valid feature file.\n"
                        f"Missing required column: 'cell_id'"
                    )
                    return
                
                # Add to list if not already present
                if file_path not in self.loaded_files:
                    self.loaded_files.append(file_path)
                    self.file_list.addItem(os.path.basename(file_path))
                    self._populate_features()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load file:\n{str(e)}"
                )
    
    def _remove_feature_file(self):
        """Remove selected feature file from the list."""
        current_item = self.file_list.currentItem()
        if current_item:
            index = self.file_list.row(current_item)
            file_path = self.loaded_files[index]
            self.loaded_files.pop(index)
            self.file_list.takeItem(index)
            self._populate_features()
    
    def _on_filter_changed(self):
        """Handle filter checkbox changes - repopulate feature list."""
        self._populate_features()
    
    def _populate_features(self):
        """Populate the feature list based on current data source and filter settings."""
        self.feature_list.clear()
        
        # Get combined dataframe
        combined_df = self._get_combined_dataframe()
        if combined_df is None or combined_df.empty:
            return
        
        # Get filter settings
        show_mean = self.filter_mean_chk.isChecked() if hasattr(self, 'filter_mean_chk') else True
        show_median = self.filter_median_chk.isChecked() if hasattr(self, 'filter_median_chk') else True
        show_other = self.filter_other_chk.isChecked() if hasattr(self, 'filter_other_chk') else False
        
        # Identify feature columns (exclude metadata columns)
        exclude_cols = {
            'label', 'cell_id', 'acquisition_id', 'acquisition_name', 
            'well', 'cluster', 'source_file', 'source_file_acquisition_id'
        }
        
        feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
        
        # Separate intensity and morphology features
        # Morphology features (based on feature_selector_dialog.py)
        morpho_names = {
            'area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
            'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
            'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count',
            'centroid_x', 'centroid_y'
        }
        
        # Intensity features identified by suffixes
        intensity_suffixes = ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos']
        
        mean_features = []
        median_features = []
        other_intensity_features = []
        morphology_features = []
        other_features = []
        
        for col in sorted(feature_cols):
            if col in morpho_names:
                morphology_features.append(col)
            elif col.endswith('_mean'):
                mean_features.append(col)
            elif col.endswith('_median'):
                median_features.append(col)
            elif any(col.endswith(suffix) for suffix in intensity_suffixes):
                other_intensity_features.append(col)
            else:
                other_features.append(col)
        
        # Add features based on filter settings
        # Mean features
        if show_mean:
            for col in mean_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(True)  # Auto-select mean features
        
        # Median features
        if show_median:
            for col in median_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(True)  # Auto-select median features
        
        # Other intensity features (if showing other)
        if show_other:
            for col in other_intensity_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(True)  # Default: select intensity features
            
            # Add morphology features (not selected by default)
            for col in morphology_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(False)  # Default: don't select morphology features
            
            # Add other features (not selected by default)
            for col in other_features:
                item = QtWidgets.QListWidgetItem(col)
                self.feature_list.addItem(item)
                item.setSelected(False)  # Default: don't select other features
        
        # Explicitly select all mean and median features to ensure they are selected
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            col_name = item.text()
            if col_name.endswith('_mean') or col_name.endswith('_median'):
                item.setSelected(True)
    
    def _get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the combined dataframe from current source or loaded files."""
        if self.use_current_radio.isChecked():
            return self.feature_dataframe
        
        if not self.loaded_files:
            return None
        
        # Load and combine all files
        dfs = []
        for file_path in self.loaded_files:
            try:
                df = pd.read_csv(file_path)
                # Ensure source_file column exists
                if 'source_file' not in df.columns:
                    df['source_file'] = os.path.basename(file_path)
                dfs.append(df)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Failed to load {os.path.basename(file_path)}:\n{str(e)}"
                )
                continue
        
        if not dfs:
            return None
        
        # Combine dataframes
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    
    def _select_all_features(self):
        """Select all features in the list."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(True)
    
    def _deselect_all_features(self):
        """Deselect all features in the list."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(False)
    
    def _auto_generate_output_path(self):
        """Auto-generate output file path based on input data."""
        if not self.save_output_chk.isChecked():
            return
        
        # Get method name
        method = self.method_combo.currentText().lower() if self.method_combo.count() > 0 else "batch"
        
        # Get base directory from first loaded file or use current directory
        base_dir = ""
        if self.loaded_files:
            base_dir = os.path.dirname(self.loaded_files[0])
        elif hasattr(self, 'parent') and self.parent() and hasattr(self.parent(), 'current_file_path'):
            # Try to get directory from parent's current file
            try:
                base_dir = os.path.dirname(self.parent().current_file_path)
            except:
                pass
        
        if not base_dir:
            base_dir = os.getcwd()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_batch_corrected_{method}_{timestamp}.csv"
        file_path = os.path.join(base_dir, filename)
        
        self.output_path_edit.setText(file_path)
    
    def _select_output_path(self):
        """Select output file path."""
        # Start with auto-generated path if available
        initial_path = self.output_path_edit.text() if self.output_path_edit.text() else ""
        if not initial_path:
            self._auto_generate_output_path()
            initial_path = self.output_path_edit.text()
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Corrected Features",
            initial_path,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.csv'):
                file_path += '.csv'
            self.output_path_edit.setText(file_path)
    
    def _update_ui_state(self):
        """Update UI state based on available methods and data."""
        # Enable/disable save output controls
        def _on_save_output_toggled(checked):
            self.output_path_edit.setEnabled(checked)
            self.output_path_btn.setEnabled(checked)
            if checked and not self.output_path_edit.text():
                # Auto-generate filename if not set
                self._auto_generate_output_path()
        
        self.save_output_chk.toggled.connect(_on_save_output_toggled)
        
        # Set initial PCA variance widget visibility based on selected method
        if hasattr(self, 'method_combo'):
            current_method = self.method_combo.currentText()
            self._on_method_changed(current_method)
        
        # Populate features if we have current data
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            self._populate_features()
    
    def get_corrected_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the batch-corrected dataframe."""
        return self.corrected_dataframe
    
    def get_combined_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the combined dataframe (original or from loaded files) before correction."""
        return self._get_combined_dataframe()
    
    def get_output_path(self) -> Optional[str]:
        """Get the output file path if saving."""
        if self.save_output_chk.isChecked():
            path = self.output_path_edit.text().strip()
            return path if path else None
        return None
    
    def _apply_batch_correction(self):
        """Apply batch correction to the data."""
        # Get combined dataframe
        combined_df = self._get_combined_dataframe()
        if combined_df is None or combined_df.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data",
                "No feature data available. Please load features first."
            )
            return
        
        # Get selected features
        selected_features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.isSelected():
                selected_features.append(item.text())
        
        if not selected_features:
            QtWidgets.QMessageBox.warning(
                self,
                "No Features Selected",
                "Please select at least one feature to correct."
            )
            return
        
        # Get batch variable
        batch_var = self.batch_var_combo.currentText()
        
        # Validate inputs
        try:
            validate_batch_correction_inputs(combined_df, batch_var, selected_features)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Validation Error",
                f"Invalid inputs for batch correction:\n{str(e)}"
            )
            return
        
        # Get method
        method = self.method_combo.currentText()
        
        # Show progress
        progress = QtWidgets.QProgressDialog("Applying batch correction...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Apply correction based on method
            if method == "Combat":
                if not _HAVE_COMBAT:
                    raise ImportError("Combat is not installed. Install with: pip install combat")
                self.corrected_dataframe = apply_combat_correction(
                    combined_df,
                    batch_var,
                    selected_features
                )
            elif method == "Harmony":
                if not _HAVE_HARMONY:
                    raise ImportError("Harmony is not installed. Install with: pip install harmonypy")
                # Get PCA variance threshold (default 0.9 for 90%)
                pca_variance = self.pca_variance_spin.value() if hasattr(self, 'pca_variance_spin') else 0.9
                self.corrected_dataframe = apply_harmony_correction(
                    combined_df,
                    batch_var,
                    selected_features,
                    pca_variance=pca_variance
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            progress.close()
            
            # Validate output path if saving
            if self.save_output_chk.isChecked():
                output_path = self.output_path_edit.text().strip()
                if not output_path:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "No Output Path",
                        "Please specify an output path to save the corrected features."
                    )
                    return
            
            # Success - accept dialog
            self.accept()
            
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Batch Correction Error",
                f"Batch correction failed:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

