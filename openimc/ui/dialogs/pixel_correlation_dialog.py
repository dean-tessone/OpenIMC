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
Pixel-Level Correlation Analysis Dialog for OpenIMC

This module provides pixel-level correlation analysis capabilities:
- Compute Spearman correlation coefficients for marker pairs
- Option to analyze within cell masks or entire ROI
- Optional: Group ROIs by treatment conditions (conditions are optional)
- Aggregate to mean correlation values per condition (if used) or across all ROIs
"""

from typing import Optional, Dict, List, Tuple
import os
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.core import pixel_correlation
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.progress_dialog import ProgressDialog
from openimc.utils.logger import get_logger
import tifffile
import multiprocessing as mp


# Import worker function from processing module
from openimc.processing.pixel_correlation_worker import correlation_process_roi_worker as _correlation_process_roi_worker

def _correlation_compute_correlations_worker(img_stack: np.ndarray, channels: List[str], 
                                             mask: Optional[np.ndarray], acq_id: str) -> List[Dict]:
    """Compute Spearman correlations for all marker pairs (module-level for multiprocessing)."""
    correlations = []
    
    try:
        # Determine shape - loaders return HWC format (H, W, C)
        if img_stack.ndim == 3:
            height, width, n_channels = img_stack.shape
        elif img_stack.ndim == 2:
            # Single channel
            return correlations
        else:
            return correlations
        
        # Ensure we have the right number of channels
        if len(channels) != n_channels:
            # If channel count doesn't match, channels list should match the image stack
            # This can happen if selected_channels was used to filter
            if len(channels) > n_channels:
                channels = channels[:n_channels]
        
        # Flatten images and apply mask if provided
        pixel_data = {}
        for i, channel in enumerate(channels):
            if i >= n_channels:
                continue
            # Extract channel from HWC format: (H, W, C) -> (H, W)
            channel_img = img_stack[:, :, i] if img_stack.ndim == 3 else img_stack
            
            if mask is not None:
                # Only use pixels within cells
                # Ensure mask matches image dimensions
                if mask.shape == channel_img.shape:
                    cell_mask = mask > 0
                    pixels = channel_img[cell_mask]
                else:
                    # Mask dimensions don't match, skip mask
                    pixels = channel_img.flatten()
            else:
                # Use all pixels
                pixels = channel_img.flatten()
            
            # Remove NaN and infinite values
            pixels = pixels[~np.isnan(pixels) & ~np.isinf(pixels)]
            pixel_data[channel] = pixels
        
        # Compute pairwise correlations
        channel_list = list(pixel_data.keys())
        for i, ch1 in enumerate(channel_list):
            for j, ch2 in enumerate(channel_list):
                if i >= j:  # Only compute upper triangle
                    continue
                
                data1 = pixel_data[ch1]
                data2 = pixel_data[ch2]
                
                # Ensure same length (take minimum)
                min_len = min(len(data1), len(data2))
                if min_len < 3:  # Need at least 3 points for correlation
                    continue
                
                data1 = data1[:min_len]
                data2 = data2[:min_len]
                
                # Compute Spearman correlation
                try:
                    corr_coef, p_value = spearmanr(data1, data2)
                    
                    if not np.isnan(corr_coef) and not np.isinf(corr_coef):
                        correlations.append({
                            'marker1': ch1,
                            'marker2': ch2,
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'n_pixels': min_len
                        })
                except Exception as e:
                    print(f"Error computing correlation for {ch1}-{ch2}: {e}")
                    continue
        
        return correlations
    except Exception as e:
        print(f"Error in correlation computation: {e}")
        import traceback
        traceback.print_exc()
        return correlations


class NumericTableWidgetItem(QtWidgets.QTableWidgetItem):
    """Custom table widget item that sorts by numeric value stored in UserRole."""
    
    def __lt__(self, other):
        """Compare items using numeric value if available, otherwise use text."""
        # Get numeric values from UserRole
        self_val = self.data(QtCore.Qt.UserRole)
        other_val = other.data(QtCore.Qt.UserRole)
        
        # If both have numeric values, compare numerically
        if self_val is not None and other_val is not None:
            try:
                return float(self_val) < float(other_val)
            except (ValueError, TypeError):
                pass
        
        # Fall back to text comparison
        return super().__lt__(other)


class ConditionROIWidget(QtWidgets.QWidget):
    """Widget for selecting a condition name and associated ROIs/MCD files."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition_name = ""
        self.roi_items = []  # List of (acq_id, acq_name, file_path, loader_type)
        self._create_ui()
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Condition name
        name_layout = QtWidgets.QHBoxLayout()
        name_layout.addWidget(QtWidgets.QLabel("Condition Name:"))
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Control, Treatment A")
        self.name_edit.textChanged.connect(self._on_name_changed)
        name_layout.addWidget(self.name_edit, 1)
        layout.addLayout(name_layout)
        
        # ROI list
        roi_label = QtWidgets.QLabel("ROIs/MCD Files:")
        layout.addWidget(roi_label)
        
        # Search box
        search_layout = QtWidgets.QVBoxLayout()
        search_label = QtWidgets.QLabel("Search:")
        search_layout.addWidget(search_label)
        self.roi_search = QtWidgets.QLineEdit()
        self.roi_search.setPlaceholderText("Type to filter ROIs...")
        search_layout.addWidget(self.roi_search)
        layout.addLayout(search_layout)
        
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setMaximumHeight(100)  # Make shorter since it's scrollable
        
        # Connect search to filter function
        def filter_roi_list(text):
            search_text = text.lower()
            for i in range(self.roi_list.count()):
                item = self.roi_list.item(i)
                item_text = item.text().lower()
                item.setHidden(search_text not in item_text)
        
        self.roi_search.textChanged.connect(filter_roi_list)
        layout.addWidget(self.roi_list)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.add_roi_btn = QtWidgets.QPushButton("Add ROI...")
        self.add_roi_btn.clicked.connect(self._add_roi)
        btn_layout.addWidget(self.add_roi_btn)
        
        self.remove_roi_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_roi_btn.clicked.connect(self._remove_roi)
        self.remove_roi_btn.setEnabled(False)
        btn_layout.addWidget(self.remove_roi_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.roi_list.itemSelectionChanged.connect(self._on_selection_changed)
    
    def _on_name_changed(self, text):
        self.condition_name = text.strip()
    
    def _on_selection_changed(self):
        self.remove_roi_btn.setEnabled(len(self.roi_list.selectedItems()) > 0)
    
    def _add_roi(self):
        """Add a ROI from the parent window's loaded data."""
        # Find the parent dialog
        parent_dialog = self.parent()
        while parent_dialog and not isinstance(parent_dialog, PixelCorrelationDialog):
            parent_dialog = parent_dialog.parent()
        
        if not parent_dialog:
            return
        
        # Get available acquisitions from parent window
        parent_window = parent_dialog.parent_window if hasattr(parent_dialog, 'parent_window') else None
        if not parent_window:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load data in the main window first.")
            return
        
        # Get acquisitions
        acquisitions = []
        if hasattr(parent_window, 'acquisitions') and parent_window.acquisitions:
            acquisitions = parent_window.acquisitions
        
        if not acquisitions:
            QtWidgets.QMessageBox.warning(self, "No Acquisitions", "No acquisitions available. Please load data first.")
            return
        
        # Create dialog to select acquisition
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select ROI")
        dialog.setModal(True)
        dialog.resize(600, 500)  # Make dialog larger
        layout = QtWidgets.QVBoxLayout(dialog)
        
        label = QtWidgets.QLabel("Select one or more acquisitions/ROIs:")
        layout.addWidget(label)
        
        # Search box
        search_layout = QtWidgets.QVBoxLayout()
        search_label = QtWidgets.QLabel("Search:")
        search_layout.addWidget(search_label)
        search_edit = QtWidgets.QLineEdit()
        search_edit.setPlaceholderText("Type to filter ROIs...")
        search_layout.addWidget(search_edit)
        layout.addLayout(search_layout)
        
        acq_list = QtWidgets.QListWidget()
        acq_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # Enable multi-selection
        
        # Store all items for filtering
        all_acq_items = []
        for acq_info in acquisitions:
            # Build display name with well/ROI name
            display_name = acq_info.name
            if hasattr(acq_info, 'well') and acq_info.well:
                display_name = f"{acq_info.well} - {acq_info.name}"
            if hasattr(acq_info, 'source_file') and acq_info.source_file:
                display_name += f" ({os.path.basename(acq_info.source_file)})"
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, acq_info)
            acq_list.addItem(item)
            all_acq_items.append(item)
        
        # Connect search to filter function
        def filter_acq_list(text):
            search_text = text.lower()
            for item in all_acq_items:
                item_text = item.text().lower()
                item.setHidden(search_text not in item_text)
        
        search_edit.textChanged.connect(filter_acq_list)
        layout.addWidget(acq_list)
        
        # Selection buttons
        selection_btn_layout = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: acq_list.selectAll())
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: acq_list.clearSelection())
        selection_btn_layout.addWidget(select_all_btn)
        selection_btn_layout.addWidget(deselect_all_btn)
        selection_btn_layout.addStretch()
        layout.addLayout(selection_btn_layout)
        
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_items = acq_list.selectedItems()
            if selected_items:
                for selected_item in selected_items:
                    acq_info = selected_item.data(QtCore.Qt.UserRole)
                    
                    # Get file path and loader type
                    file_path = None
                    loader_type = None
                    
                    if hasattr(acq_info, 'source_file') and acq_info.source_file:
                        file_path = acq_info.source_file
                        if file_path.endswith('.mcd') or file_path.endswith('.mcdx'):
                            loader_type = "mcd"
                        else:
                            loader_type = "ometiff"
                    elif hasattr(parent_window, 'acq_to_file') and acq_info.id in parent_window.acq_to_file:
                        file_path = parent_window.acq_to_file[acq_info.id]
                        loader_type = "mcd"
                    elif hasattr(parent_window, 'current_path') and parent_window.current_path:
                        file_path = parent_window.current_path
                        if os.path.isdir(file_path):
                            loader_type = "ometiff"
                        else:
                            loader_type = "mcd"
                    
                    if file_path and loader_type:
                        # Add to list
                        roi_item = (acq_info.id, acq_info.name, file_path, loader_type)
                        if roi_item not in self.roi_items:
                            self.roi_items.append(roi_item)
                            # Build display text with well/ROI name
                            display_text = acq_info.name
                            if hasattr(acq_info, 'well') and acq_info.well:
                                display_text = f"{acq_info.well} - {acq_info.name}"
                            display_text += f" ({os.path.basename(file_path)})"
                            list_item = QtWidgets.QListWidgetItem(display_text)
                            list_item.setData(QtCore.Qt.UserRole, roi_item)
                            self.roi_list.addItem(list_item)
                
                # Update mask availability check after adding all selected items
                # Find parent dialog to call check method
                parent_dialog = self.parent()
                while parent_dialog and not isinstance(parent_dialog, PixelCorrelationDialog):
                    parent_dialog = parent_dialog.parent()
                if parent_dialog:
                    parent_dialog._check_masks_available()
    
    def _remove_roi(self):
        """Remove selected ROI from the list."""
        selected_items = self.roi_list.selectedItems()
        for item in selected_items:
            roi_item = item.data(QtCore.Qt.UserRole)
            if roi_item in self.roi_items:
                self.roi_items.remove(roi_item)
            row = self.roi_list.row(item)
            self.roi_list.takeItem(row)
    
    def get_condition_data(self) -> Tuple[str, List[Tuple[str, str, str, str]]]:
        """Get condition name and list of (acq_id, acq_name, file_path, loader_type)."""
        return self.condition_name, self.roi_items.copy()


class PixelCorrelationDialog(QtWidgets.QDialog):
    """Dialog for pixel-level correlation analysis."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel-Level Correlation Analysis")
        self.setMinimumSize(1000, 900)  # Make taller so analysis scope is visible
        self.resize(1000, 900)
        
        self.parent_window = parent
        
        # Store results
        self.correlation_results: Optional[pd.DataFrame] = None
        self.aggregated_results: Optional[pd.DataFrame] = None
        
        # Analysis settings - will be set by _check_masks_available based on mask availability
        self.analyze_within_masks = False
        
        self._create_ui()
        self._check_masks_available()  # This will set default to "Within Cell Masks" if masks exist
    
    def _on_use_conditions_changed(self, checked):
        """Handle use conditions checkbox change."""
        self.conditions_group.setVisible(checked)
        self.roi_group.setVisible(not checked)
        # Update mask availability check when switching modes
        self._check_masks_available()
    
    def _add_condition(self):
        """Add a new condition widget."""
        condition_widget = ConditionROIWidget(self)
        condition_widget.setParent(self.conditions_container)
        self.condition_widgets.append(condition_widget)
        self.conditions_layout.addWidget(condition_widget)
        # Update mask availability check when condition is added
        self._check_masks_available()
    
    def _on_roi_selection_changed(self):
        """Handle ROI selection change."""
        self.remove_roi_btn.setEnabled(len(self.roi_list.selectedItems()) > 0)
    
    def _filter_roi_list(self, text):
        """Filter ROI list based on search text."""
        search_text = text.lower()
        for i in range(self.roi_list.count()):
            item = self.roi_list.item(i)
            item_text = item.text().lower()
            item.setHidden(search_text not in item_text)
    
    def _add_roi(self):
        """Add a ROI from the parent window's loaded data."""
        if not self.parent_window:
            QtWidgets.QMessageBox.warning(self, "No Data", "Please load data in the main window first.")
            return
        
        # Get acquisitions
        acquisitions = []
        if hasattr(self.parent_window, 'acquisitions') and self.parent_window.acquisitions:
            acquisitions = self.parent_window.acquisitions
        
        if not acquisitions:
            QtWidgets.QMessageBox.warning(self, "No Acquisitions", "No acquisitions available. Please load data first.")
            return
        
        # Create dialog to select acquisition
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select ROI")
        dialog.setModal(True)
        dialog.resize(600, 500)  # Make dialog larger
        layout = QtWidgets.QVBoxLayout(dialog)
        
        label = QtWidgets.QLabel("Select one or more acquisitions/ROIs:")
        layout.addWidget(label)
        
        # Search box
        search_layout = QtWidgets.QVBoxLayout()
        search_label = QtWidgets.QLabel("Search:")
        search_layout.addWidget(search_label)
        search_edit = QtWidgets.QLineEdit()
        search_edit.setPlaceholderText("Type to filter ROIs...")
        search_layout.addWidget(search_edit)
        layout.addLayout(search_layout)
        
        acq_list = QtWidgets.QListWidget()
        acq_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  # Enable multi-selection
        
        # Store all items for filtering
        all_acq_items = []
        for acq_info in acquisitions:
            # Build display name with well/ROI name
            display_name = acq_info.name
            if hasattr(acq_info, 'well') and acq_info.well:
                display_name = f"{acq_info.well} - {acq_info.name}"
            if hasattr(acq_info, 'source_file') and acq_info.source_file:
                display_name += f" ({os.path.basename(acq_info.source_file)})"
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, acq_info)
            acq_list.addItem(item)
            all_acq_items.append(item)
        
        # Connect search to filter function
        def filter_acq_list(text):
            search_text = text.lower()
            for item in all_acq_items:
                item_text = item.text().lower()
                item.setHidden(search_text not in item_text)
        
        search_edit.textChanged.connect(filter_acq_list)
        layout.addWidget(acq_list)
        
        # Selection buttons
        selection_btn_layout = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: acq_list.selectAll())
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: acq_list.clearSelection())
        selection_btn_layout.addWidget(select_all_btn)
        selection_btn_layout.addWidget(deselect_all_btn)
        selection_btn_layout.addStretch()
        layout.addLayout(selection_btn_layout)
        
        btn_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_items = acq_list.selectedItems()
            if selected_items:
                for selected_item in selected_items:
                    acq_info = selected_item.data(QtCore.Qt.UserRole)
                    
                    # Get file path and loader type
                    file_path = None
                    loader_type = None
                    
                    if hasattr(acq_info, 'source_file') and acq_info.source_file:
                        file_path = acq_info.source_file
                        if file_path.endswith('.mcd') or file_path.endswith('.mcdx'):
                            loader_type = "mcd"
                        else:
                            loader_type = "ometiff"
                    elif hasattr(self.parent_window, 'acq_to_file') and acq_info.id in self.parent_window.acq_to_file:
                        file_path = self.parent_window.acq_to_file[acq_info.id]
                        loader_type = "mcd"
                    elif hasattr(self.parent_window, 'current_path') and self.parent_window.current_path:
                        file_path = self.parent_window.current_path
                        if os.path.isdir(file_path):
                            loader_type = "ometiff"
                        else:
                            loader_type = "mcd"
                    
                    if file_path and loader_type:
                        # Add to list
                        roi_item = (acq_info.id, acq_info.name, file_path, loader_type)
                        if roi_item not in self.roi_items:
                            self.roi_items.append(roi_item)
                            # Build display text with well/ROI name
                            display_text = acq_info.name
                            if hasattr(acq_info, 'well') and acq_info.well:
                                display_text = f"{acq_info.well} - {acq_info.name}"
                            display_text += f" ({os.path.basename(file_path)})"
                            list_item = QtWidgets.QListWidgetItem(display_text)
                            list_item.setData(QtCore.Qt.UserRole, roi_item)
                            self.roi_list.addItem(list_item)
                
                # Update mask availability check after adding all selected items
                self._check_masks_available()
    
    def _remove_roi(self):
        """Remove selected ROI from the list."""
        selected_items = self.roi_list.selectedItems()
        for item in selected_items:
            roi_item = item.data(QtCore.Qt.UserRole)
            if roi_item in self.roi_items:
                self.roi_items.remove(roi_item)
            row = self.roi_list.row(item)
            self.roi_list.takeItem(row)
        
        # Update mask availability check
        self._check_masks_available()
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Information section (top only)
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_label = QtWidgets.QLabel(
            "To investigate colocalization of proteins, we conduct pixel-level correlation analysis of markers.\n"
            "Spearman correlation coefficients are computed for each marker pair.\n\n"
            "P-values are computed using the Spearman rank correlation test (two-tailed). "
            "For multiple comparisons, p-values are adjusted using the Benjamini-Hochberg procedure "
            "to control the false discovery rate (FDR).\n\n"
            "You can optionally group ROIs by treatment conditions, or analyze all ROIs together."
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        layout.addWidget(info_group)
        
        # Store ROI items: list of (acq_id, acq_name, file_path, loader_type)
        self.roi_items = []
        # Store condition widgets
        self.condition_widgets: List['ConditionROIWidget'] = []
        
        # Analysis section with tabs
        results_group = QtWidgets.QGroupBox("Analysis")
        results_layout = QtWidgets.QVBoxLayout(results_group)
        
        # Tabs for analysis options and results
        self.tabs = QtWidgets.QTabWidget()
        
        # Analysis Options tab (first tab)
        options_tab = QtWidgets.QWidget()
        options_tab_layout = QtWidgets.QVBoxLayout(options_tab)
        options_tab_layout.setContentsMargins(10, 10, 10, 10)
        options_tab_layout.setSpacing(10)
        
        # Create scroll area for options tab
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(10)
        
        # ROI/Condition Selection
        roi_selection_group = QtWidgets.QGroupBox("ROI/Condition Selection")
        roi_selection_layout = QtWidgets.QVBoxLayout(roi_selection_group)
        roi_selection_layout.setContentsMargins(8, 8, 8, 8)
        
        # Use conditions checkbox
        self.use_conditions_chk = QtWidgets.QCheckBox("Group ROIs by treatment conditions")
        self.use_conditions_chk.setChecked(False)
        self.use_conditions_chk.toggled.connect(self._on_use_conditions_changed)
        roi_selection_layout.addWidget(self.use_conditions_chk)
        
        # Conditions section (hidden by default)
        conditions_group = QtWidgets.QGroupBox("Treatment Conditions")
        conditions_layout = QtWidgets.QVBoxLayout(conditions_group)
        
        # Add condition button
        add_condition_btn = QtWidgets.QPushButton("Add Condition")
        add_condition_btn.clicked.connect(self._add_condition)
        conditions_layout.addWidget(add_condition_btn)
        
        # Conditions container
        self.conditions_container = QtWidgets.QWidget()
        self.conditions_layout = QtWidgets.QVBoxLayout(self.conditions_container)
        self.conditions_layout.setContentsMargins(0, 0, 0, 0)
        conditions_layout.addWidget(self.conditions_container)
        
        self.conditions_group = conditions_group
        self.conditions_group.setVisible(False)
        roi_selection_layout.addWidget(conditions_group)
        
        # ROI selection section (shown when not using conditions)
        roi_group = QtWidgets.QGroupBox("ROIs/MCD Files")
        roi_layout = QtWidgets.QVBoxLayout(roi_group)
        
        # ROI list with search
        search_layout = QtWidgets.QVBoxLayout()
        search_label = QtWidgets.QLabel("Search ROIs:")
        search_layout.addWidget(search_label)
        self.roi_search = QtWidgets.QLineEdit()
        self.roi_search.setPlaceholderText("Type to filter ROIs...")
        self.roi_search.textChanged.connect(self._filter_roi_list)
        search_layout.addWidget(self.roi_search)
        roi_layout.addLayout(search_layout)
        
        # ROI list
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setMaximumHeight(120)  # Make shorter since it's scrollable
        roi_layout.addWidget(self.roi_list)
        
        # Buttons
        roi_btn_layout = QtWidgets.QHBoxLayout()
        self.add_roi_btn = QtWidgets.QPushButton("Add ROI...")
        self.add_roi_btn.clicked.connect(self._add_roi)
        roi_btn_layout.addWidget(self.add_roi_btn)
        
        self.remove_roi_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_roi_btn.clicked.connect(self._remove_roi)
        self.remove_roi_btn.setEnabled(False)
        roi_btn_layout.addWidget(self.remove_roi_btn)
        roi_btn_layout.addStretch()
        roi_layout.addLayout(roi_btn_layout)
        
        self.roi_list.itemSelectionChanged.connect(self._on_roi_selection_changed)
        
        self.roi_group = roi_group
        roi_selection_layout.addWidget(roi_group)
        
        scroll_layout.addWidget(roi_selection_group)
        
        # Analysis scope
        scope_group = QtWidgets.QGroupBox("Analysis Scope")
        scope_layout = QtWidgets.QVBoxLayout(scope_group)
        scope_layout.setContentsMargins(8, 8, 8, 8)
        
        scope_info = QtWidgets.QLabel("Select whether to analyze pixels within cell masks or the entire ROI.")
        scope_info.setWordWrap(True)
        scope_layout.addWidget(scope_info)
        
        scope_combo_layout = QtWidgets.QHBoxLayout()
        scope_combo_layout.addWidget(QtWidgets.QLabel("Scope:"))
        self.scope_combo = QtWidgets.QComboBox()
        self.scope_combo.addItems(["Within Cell Masks", "Entire ROI"])
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        scope_combo_layout.addWidget(self.scope_combo)
        scope_combo_layout.addStretch()
        scope_layout.addLayout(scope_combo_layout)
        scroll_layout.addWidget(scope_group)
        
        # Channel selection
        channel_group = QtWidgets.QGroupBox("Channel Selection")
        channel_group_layout = QtWidgets.QVBoxLayout(channel_group)
        channel_group_layout.setContentsMargins(8, 8, 8, 8)
        
        channel_info = QtWidgets.QLabel("Select which channels to include in the correlation analysis.")
        channel_info.setWordWrap(True)
        channel_group_layout.addWidget(channel_info)
        
        channel_combo_layout = QtWidgets.QHBoxLayout()
        channel_combo_layout.addWidget(QtWidgets.QLabel("Channels:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(["All Channels", "Select Channels..."])
        self.channel_combo.currentIndexChanged.connect(self._on_channel_selection_changed)
        channel_combo_layout.addWidget(self.channel_combo)
        channel_combo_layout.addStretch()
        channel_group_layout.addLayout(channel_combo_layout)
        
        # Selected channels (hidden by default)
        self.selected_channels_widget = QtWidgets.QWidget()
        selected_channels_layout = QtWidgets.QVBoxLayout(self.selected_channels_widget)
        selected_channels_layout.setContentsMargins(10, 5, 0, 5)
        
        selected_channels_label = QtWidgets.QLabel("Selected Channels:")
        selected_channels_layout.addWidget(selected_channels_label)
        
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setMaximumHeight(200)
        # Use checkable items instead of MultiSelection mode for better compatibility
        selected_channels_layout.addWidget(self.channel_list)
        
        # Select All / Deselect All buttons
        channel_btn_layout = QtWidgets.QHBoxLayout()
        select_all_channels_btn = QtWidgets.QPushButton("Select All")
        select_all_channels_btn.clicked.connect(self._select_all_channels)
        deselect_all_channels_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_channels_btn.clicked.connect(self._deselect_all_channels)
        channel_btn_layout.addWidget(select_all_channels_btn)
        channel_btn_layout.addWidget(deselect_all_channels_btn)
        channel_btn_layout.addStretch()
        selected_channels_layout.addLayout(channel_btn_layout)
        
        self.selected_channels_widget.setVisible(False)
        channel_group_layout.addWidget(self.selected_channels_widget)
        
        scroll_layout.addWidget(channel_group)
        
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        options_tab_layout.addWidget(scroll_area)
        
        # Run button (outside scroll area)
        run_layout = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Correlation Analysis")
        self.run_btn.setMinimumHeight(35)
        self.run_btn.clicked.connect(self._run_analysis)
        run_layout.addWidget(self.run_btn)
        run_layout.addStretch()
        options_tab_layout.addLayout(run_layout)
        
        self.tabs.addTab(options_tab, "Analysis Options")
        
        # Correlation matrix tab
        matrix_tab = QtWidgets.QWidget()
        matrix_layout = QtWidgets.QVBoxLayout(matrix_tab)
        
        self.correlation_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        matrix_layout.addWidget(self.correlation_canvas)
        
        matrix_btn_layout = QtWidgets.QHBoxLayout()
        self.save_matrix_btn = QtWidgets.QPushButton("Save Heatmap...")
        self.save_matrix_btn.clicked.connect(self._save_matrix)
        self.save_matrix_btn.setEnabled(False)
        matrix_btn_layout.addWidget(self.save_matrix_btn)
        matrix_btn_layout.addStretch()
        matrix_layout.addLayout(matrix_btn_layout)
        
        self.tabs.addTab(matrix_tab, "Correlation Heatmap")
        
        # Results table tab
        table_tab = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_tab)
        
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.results_table)
        
        table_btn_layout = QtWidgets.QHBoxLayout()
        self.export_table_btn = QtWidgets.QPushButton("Export Results...")
        self.export_table_btn.clicked.connect(self._export_results)
        self.export_table_btn.setEnabled(False)
        table_btn_layout.addWidget(self.export_table_btn)
        table_btn_layout.addStretch()
        table_layout.addLayout(table_btn_layout)
        
        self.tabs.addTab(table_tab, "Results Table")
        
        results_layout.addWidget(self.tabs)
        layout.addWidget(results_group)
    
    def _check_masks_available(self):
        """Check if masks are available for selected ROIs and update scope combo."""
        # Check if any masks are available for the selected ROIs
        has_masks = False
        
        if self.use_conditions_chk.isChecked():
            # Check masks for all ROIs in all conditions
            for widget in self.condition_widgets:
                name, roi_items = widget.get_condition_data()
                if roi_items:
                    for acq_id, acq_name, file_path, loader_type in roi_items:
                        if self._has_mask_for_roi(acq_id, acq_name, file_path):
                            has_masks = True
                            break
                    if has_masks:
                        break
        else:
            # Check masks for all selected ROIs
            for acq_id, acq_name, file_path, loader_type in self.roi_items:
                if self._has_mask_for_roi(acq_id, acq_name, file_path):
                    has_masks = True
                    break
        
        # Update scope combo
        if has_masks:
            # Enable "Within Cell Masks" option
            was_disabled = self.scope_combo.model().item(0).isEnabled() == False
            if was_disabled:
                self.scope_combo.model().item(0).setEnabled(True)
            # Set "Within Cell Masks" as default when masks become available
            if was_disabled or self.scope_combo.currentIndex() == 1:
                self.scope_combo.setCurrentIndex(0)  # Within Cell Masks
                self.analyze_within_masks = True
        else:
            # Disable "Within Cell Masks" option
            self.scope_combo.model().item(0).setEnabled(False)
            # If currently set to "Within Cell Masks", switch to "Entire ROI"
            if self.scope_combo.currentIndex() == 0:
                self.scope_combo.setCurrentIndex(1)  # Entire ROI
                self.analyze_within_masks = False
    
    def _has_mask_for_roi(self, acq_id: str, acq_name: str, file_path: str) -> bool:
        """Check if a mask exists for a given ROI."""
        # Check parent window masks first
        if (self.parent_window and 
            hasattr(self.parent_window, 'segmentation_masks') and
            acq_id in self.parent_window.segmentation_masks):
            return True
        
        # Check file system for mask files
        base_path = file_path.replace('.mcd', '').replace('.mcdx', '')
        possible_paths = [
            f"{base_path}_segmentation.tif",
            f"{base_path}_segmentation.npy",
            os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.tif"),
            os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.npy"),
        ]
        for mask_path in possible_paths:
            if os.path.exists(mask_path):
                return True
        
        return False
    
    def _on_scope_changed(self, index):
        """Handle analysis scope change."""
        self.analyze_within_masks = (index == 0)
    
    def _on_channel_selection_changed(self, index):
        """Handle channel selection mode change."""
        if index == 1:  # Select Channels
            self.selected_channels_widget.setVisible(True)
            self._populate_channel_list()
        else:  # All Channels
            self.selected_channels_widget.setVisible(False)
    
    def _populate_channel_list(self):
        """Populate channel list from available data."""
        self.channel_list.clear()
        
        # Get channels from all available ROIs
        channels = set()
        
        # Helper function to get channels from a single ROI
        def get_channels_from_roi(acq_id, acq_name, file_path, loader_type):
            try:
                # Map unique acquisition ID to original ID for multiple MCD files
                original_acq_id = acq_id
                if (self.parent_window and 
                    hasattr(self.parent_window, 'unique_acq_to_original') and 
                    acq_id in self.parent_window.unique_acq_to_original):
                    original_acq_id = self.parent_window.unique_acq_to_original[acq_id]
                
                if loader_type == "mcd":
                    # Check if we can use existing loader
                    if (self.parent_window and 
                        hasattr(self.parent_window, 'mcd_loaders') and 
                        file_path in self.parent_window.mcd_loaders):
                        loader = self.parent_window.mcd_loaders[file_path]
                        return loader.get_channels(original_acq_id)
                    else:
                        loader = MCDLoader()
                        loader.open(file_path)
                        channels_list = loader.get_channels(original_acq_id)
                        loader.close()
                        return channels_list
                elif loader_type == "ometiff":
                    loader = OMETIFFLoader(channel_format='CHW')
                    loader.open(file_path)
                    channels_list = loader.get_channels(original_acq_id)
                    loader.close()
                    return channels_list
            except Exception as e:
                print(f"Error getting channels for {acq_name}: {e}")
                return []
            return []
        
        # Get channels from all ROIs (union of all channels)
        # Check both main ROI list and condition ROIs
        if self.use_conditions_chk.isChecked():
            # Get channels from all condition ROIs
            for widget in self.condition_widgets:
                name, roi_items = widget.get_condition_data()
                for acq_id, acq_name, file_path, loader_type in roi_items:
                    channels_list = get_channels_from_roi(acq_id, acq_name, file_path, loader_type)
                    channels.update(channels_list)
        else:
            # Get channels from main ROI list
            for acq_id, acq_name, file_path, loader_type in self.roi_items:
                channels_list = get_channels_from_roi(acq_id, acq_name, file_path, loader_type)
                channels.update(channels_list)
        
        # Also try to get from parent window acquisitions if no ROIs selected yet
        if not channels and self.parent_window and hasattr(self.parent_window, 'acquisitions'):
            try:
                # Get channels from first acquisition
                if self.parent_window.acquisitions:
                    acq_info = self.parent_window.acquisitions[0]
                    if hasattr(acq_info, 'channels'):
                        channels.update(acq_info.channels)
            except Exception:
                pass
        
        if channels:
            for channel in sorted(channels):
                item = QtWidgets.QListWidgetItem(channel)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                self.channel_list.addItem(item)
        else:
            # Show message if no channels found
            item = QtWidgets.QListWidgetItem("No channels available. Please add ROIs first.")
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
            self.channel_list.addItem(item)
    
    def _select_all_channels(self):
        """Select all channels in the channel list."""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.Checked)
    
    def _deselect_all_channels(self):
        """Deselect all channels in the channel list."""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.Unchecked)
    
    def _get_selected_channels(self) -> Optional[List[str]]:
        """Get list of selected channels, or None if all channels should be used."""
        if self.channel_combo.currentIndex() == 0:  # All Channels
            return None
        
        selected = []
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                selected.append(item.text())
        return selected if selected else None
    
    def _get_source_files_for_logging(self) -> Optional[str]:
        """Get source file names for logging. Returns a string with all unique source files."""
        source_files = set()
        
        # Collect source files from ROI items
        if self.use_conditions_chk.isChecked():
            for widget in self.condition_widgets:
                name, roi_items = widget.get_condition_data()
                if roi_items:
                    for acq_id, acq_name, file_path, loader_type in roi_items:
                        if loader_type == "mcd":
                            # For MCD files, use the file basename
                            source_files.add(os.path.basename(file_path))
                        elif loader_type == "ometiff":
                            # For OME-TIFF, use the folder name (directory containing the file)
                            if os.path.isdir(file_path):
                                # file_path is already a folder
                                folder_path = file_path
                            else:
                                # file_path is a file, get its directory
                                folder_path = os.path.dirname(file_path) if os.path.dirname(file_path) else file_path
                            source_files.add(os.path.basename(folder_path))
        else:
            for acq_id, acq_name, file_path, loader_type in self.roi_items:
                if loader_type == "mcd":
                    # For MCD files, use the file basename
                    source_files.add(os.path.basename(file_path))
                elif loader_type == "ometiff":
                    # For OME-TIFF, use the folder name (directory containing the file)
                    if os.path.isdir(file_path):
                        # file_path is already a folder
                        folder_path = file_path
                    else:
                        # file_path is a file, get its directory
                        folder_path = os.path.dirname(file_path) if os.path.dirname(file_path) else file_path
                    source_files.add(os.path.basename(folder_path))
        
        if not source_files:
            return None
        
        if len(source_files) == 1:
            return list(source_files)[0]
        else:
            # Return comma-separated list of source files
            sorted_files = sorted(source_files)
            if len(sorted_files) <= 3:
                return ", ".join(sorted_files)
            else:
                return ", ".join(sorted_files[:3]) + f" and {len(sorted_files) - 3} more"
    
    def _run_analysis(self):
        """Run pixel-level correlation analysis with multiprocessing."""
        use_conditions = self.use_conditions_chk.isChecked()
        
        if use_conditions:
            # Validate conditions
            valid_conditions = []
            for widget in self.condition_widgets:
                name, roi_items = widget.get_condition_data()
                if name and roi_items:
                    valid_conditions.append((name, roi_items))
            
            if not valid_conditions:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Conditions",
                    "Please add at least one condition with a name and at least one ROI."
                )
                return
        else:
            # Validate ROIs
            if not self.roi_items:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No ROIs",
                    "Please add at least one ROI to analyze."
                )
                return
        
        # Get selected channels
        selected_channels = self._get_selected_channels()
        
        # Show progress dialog
        progress_dlg = ProgressDialog("Running Correlation Analysis...", self)
        progress_dlg.update_progress(0, "Correlating pixel intensities...", "Preparing analysis...")
        progress_dlg.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Prepare tasks for multiprocessing - one task per ROI
            tasks = []
            roi_info = []  # Store ROI info for adding metadata later
            
            if use_conditions:
                for condition_name, roi_items in valid_conditions:
                    for acq_id, acq_name, file_path, loader_type in roi_items:
                        # Map unique acquisition ID to original ID
                        original_acq_id = acq_id
                        if (self.parent_window and 
                            hasattr(self.parent_window, 'unique_acq_to_original') and 
                            acq_id in self.parent_window.unique_acq_to_original):
                            original_acq_id = self.parent_window.unique_acq_to_original[acq_id]
                        
                        # Handle mask path
                        mask_path = None
                        if self.analyze_within_masks:
                            # Try to get mask from parent window
                            if (self.parent_window and 
                                hasattr(self.parent_window, 'segmentation_masks') and
                                acq_id in self.parent_window.segmentation_masks):
                                # Save mask to temp file for worker
                                import tempfile
                                temp_dir = tempfile.gettempdir()
                                mask_path = os.path.join(temp_dir, f"corr_mask_{acq_id}.tif")
                                try:
                                    mask = self.parent_window.segmentation_masks[acq_id]
                                    tifffile.imwrite(mask_path, mask.astype(np.uint32))
                                except Exception:
                                    mask_path = None
                            else:
                                # Try to find mask file
                                base_path = file_path.replace('.mcd', '').replace('.mcdx', '')
                                possible_paths = [
                                    f"{base_path}_segmentation.tif",
                                    f"{base_path}_segmentation.npy",
                                    os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.tif"),
                                    os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.npy"),
                                ]
                                for path in possible_paths:
                                    if os.path.exists(path):
                                        mask_path = path
                                        break
                        
                        task = (acq_id, acq_name, file_path, loader_type, selected_channels,
                                original_acq_id, self.analyze_within_masks, mask_path)
                        tasks.append(task)
                        roi_info.append((condition_name, acq_name, acq_id))
            else:
                for acq_id, acq_name, file_path, loader_type in self.roi_items:
                    # Map unique acquisition ID to original ID
                    original_acq_id = acq_id
                    if (self.parent_window and 
                        hasattr(self.parent_window, 'unique_acq_to_original') and 
                        acq_id in self.parent_window.unique_acq_to_original):
                        original_acq_id = self.parent_window.unique_acq_to_original[acq_id]
                    
                    # Handle mask path
                    mask_path = None
                    if self.analyze_within_masks:
                        # Try to get mask from parent window
                        if (self.parent_window and 
                            hasattr(self.parent_window, 'segmentation_masks') and
                            acq_id in self.parent_window.segmentation_masks):
                            # Save mask to temp file for worker
                            import tempfile
                            temp_dir = tempfile.gettempdir()
                            mask_path = os.path.join(temp_dir, f"corr_mask_{acq_id}.tif")
                            try:
                                mask = self.parent_window.segmentation_masks[acq_id]
                                tifffile.imwrite(mask_path, mask.astype(np.uint32))
                            except Exception:
                                mask_path = None
                        else:
                            # Try to find mask file
                            base_path = file_path.replace('.mcd', '').replace('.mcdx', '')
                            possible_paths = [
                                f"{base_path}_segmentation.tif",
                                f"{base_path}_segmentation.npy",
                                os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.tif"),
                                os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.npy"),
                            ]
                            for path in possible_paths:
                                if os.path.exists(path):
                                    mask_path = path
                                    break
                    
                    task = (acq_id, acq_name, file_path, loader_type, selected_channels,
                            original_acq_id, self.analyze_within_masks, mask_path)
                    tasks.append(task)
                    roi_info.append((None, acq_name, acq_id))
            
            if not tasks:
                progress_dlg.close()
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Tasks",
                    "No ROIs to process."
                )
                return
            
            # Process with multiprocessing
            all_correlations = []
            total_tasks = len(tasks)
            progress_dlg.set_maximum(total_tasks)
            
            # Use ProcessPoolExecutor for true parallelization
            # Use spawn method to ensure isolation
            num_workers = min(mp.cpu_count(), total_tasks)
            if num_workers > 1 and total_tasks > 1:
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=num_workers) as pool:
                    # Submit all tasks (one per ROI)
                    futures = []
                    for task in tasks:
                        future = pool.apply_async(_correlation_process_roi_worker, (task,))
                        futures.append(future)
                    
                    # Collect results as they complete
                    completed = 0
                    processed_futures = set()
                    
                    while completed < total_tasks:
                        # Check each future to see if it's ready
                        for i, future in enumerate(futures):
                            if i in processed_futures:
                                continue
                            
                            # Check if ready (non-blocking)
                            if future.ready():
                                try:
                                    correlations = future.get(timeout=0.1)
                                    if correlations:
                                        # Add ROI metadata
                                        condition_name, acq_name, acq_id = roi_info[i]
                                        for corr_data in correlations:
                                            if condition_name:
                                                corr_data['condition'] = condition_name
                                            corr_data['roi'] = acq_name
                                            corr_data['roi_id'] = acq_id
                                        all_correlations.extend(correlations)
                                    completed += 1
                                    processed_futures.add(i)
                                    
                                    # Update progress immediately
                                    progress = int((completed / total_tasks) * 100)
                                    progress_dlg.update_progress(
                                        progress,
                                        "Correlating pixel intensities...",
                                        f"Processed {completed}/{total_tasks} ROIs"
                                    )
                                    QtWidgets.QApplication.processEvents()
                                except Exception as e:
                                    # Handle errors
                                    completed += 1
                                    processed_futures.add(i)
                                    print(f"Error processing ROI: {e}")
                                    progress = int((completed / total_tasks) * 100)
                                    progress_dlg.update_progress(
                                        progress,
                                        "Correlating pixel intensities...",
                                        f"Processed {completed}/{total_tasks} ROIs"
                                    )
                                    QtWidgets.QApplication.processEvents()
                        
                        # Small delay to avoid busy-waiting and allow UI to update
                        if completed < total_tasks:
                            QtCore.QThread.msleep(50)  # 50ms delay for smoother UI updates
                            QtWidgets.QApplication.processEvents()
                    
                    # Final update
                    progress_dlg.update_progress(
                        100,
                        "Correlating pixel intensities...",
                        f"Processed {completed}/{total_tasks} ROIs"
                    )
                    QtWidgets.QApplication.processEvents()
            else:
                # Single-threaded processing (fallback)
                for i, task in enumerate(tasks):
                    correlations = _correlation_process_roi_worker(task)
                    if correlations:
                        # Add ROI metadata
                        condition_name, acq_name, acq_id = roi_info[i]
                        for corr_data in correlations:
                            if condition_name:
                                corr_data['condition'] = condition_name
                            corr_data['roi'] = acq_name
                            corr_data['roi_id'] = acq_id
                        all_correlations.extend(correlations)
                    
                    progress_dlg.update_progress(
                        i + 1,
                        "Correlating pixel intensities...",
                        f"Processing ROI {i + 1} of {total_tasks}: {roi_info[i][1]}"
                    )
                    QtWidgets.QApplication.processEvents()
            
            # Clean up temp mask files
            import tempfile
            temp_dir = tempfile.gettempdir()
            for condition_name, acq_name, acq_id in roi_info:
                temp_mask_path = os.path.join(temp_dir, f"corr_mask_{acq_id}.tif")
                if os.path.exists(temp_mask_path):
                    try:
                        os.remove(temp_mask_path)
                    except Exception:
                        pass
            
            if not all_correlations:
                progress_dlg.close()
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Results",
                    "No correlation data was computed. Please check your data and settings."
                )
                return
            
            # Create results dataframe
            self.correlation_results = pd.DataFrame(all_correlations)
            
            # Aggregate results
            self._aggregate_results(use_conditions)
            
            # Update UI
            self._update_results_display(use_conditions)
            
            progress_dlg.close()
            
            # Log pixel correlation analysis
            logger = get_logger()
            # Collect acquisition IDs
            acquisitions = []
            if use_conditions:
                for widget in self.condition_widgets:
                    name, roi_items = widget.get_condition_data()
                    if roi_items:
                        for acq_id, acq_name, file_path, loader_type in roi_items:
                            acquisitions.append(acq_id)
            else:
                for acq_id, acq_name, file_path, loader_type in self.roi_items:
                    acquisitions.append(acq_id)
            
            params = {
                "scope": "within_cell_masks" if self.analyze_within_masks else "entire_roi",
                "selected_channels": self._get_selected_channels() is not None,
                "n_channels": len(self._get_selected_channels()) if self._get_selected_channels() else "all",
                "use_conditions": use_conditions
            }
            source_file = self._get_source_files_for_logging()
            
            logger._write_entry(
                entry_type="pixel_correlation",
                operation="spearman_correlation",
                parameters=params,
                acquisitions=acquisitions,
                notes=f"Pixel-level correlation analysis: {len(self.correlation_results)} marker pair correlations computed",
                source_file=source_file
            )
            
            QtWidgets.QMessageBox.information(
                self,
                "Analysis Complete",
                f"Correlation analysis completed successfully.\n"
                f"Computed {len(self.correlation_results)} marker pair correlations."
            )
            
        except Exception as e:
            progress_dlg.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Analysis Error",
                f"Error during analysis: {str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _process_roi(self, acq_id: str, acq_name: str, file_path: str, loader_type: str, selected_channels: Optional[List[str]]):
        """Process a single ROI and return correlations."""
        try:
            # For multiple MCD files, we need to use the existing loader and map the unique ID to original ID
            original_acq_id = acq_id
            use_existing_loader = False
            
            if loader_type == "mcd":
                # Check if we're dealing with multiple MCD files
                if (self.parent_window and 
                    hasattr(self.parent_window, 'mcd_loaders') and 
                    file_path in self.parent_window.mcd_loaders):
                    # Use existing loader from main window
                    loader = self.parent_window.mcd_loaders[file_path]
                    use_existing_loader = True
                    
                    # Map unique acquisition ID to original ID
                    if (hasattr(self.parent_window, 'unique_acq_to_original') and 
                        acq_id in self.parent_window.unique_acq_to_original):
                        original_acq_id = self.parent_window.unique_acq_to_original[acq_id]
                else:
                    # Single file or new loader needed
                    loader = MCDLoader()
                    loader.open(file_path)
            elif loader_type == "ometiff":
                loader = OMETIFFLoader(channel_format='CHW')
                loader.open(file_path)
            else:
                return []
            
            # Get channels using original acquisition ID
            channels = loader.get_channels(original_acq_id)
            if selected_channels:
                channels = [ch for ch in channels if ch in selected_channels]
            
            if len(channels) < 2:
                if not use_existing_loader:
                    loader.close()
                return []
            
            # Load image stack using original acquisition ID
            img_stack = loader.get_all_channels(original_acq_id)
            
            # Load mask if needed
            mask = None
            if self.analyze_within_masks:
                # Try to get mask from parent window
                if (self.parent_window and 
                    hasattr(self.parent_window, 'segmentation_masks') and
                    acq_id in self.parent_window.segmentation_masks):
                    mask = self.parent_window.segmentation_masks[acq_id]
                else:
                    # Try to load from file - check multiple possible locations
                    base_path = file_path.replace('.mcd', '').replace('.mcdx', '')
                    possible_paths = [
                        f"{base_path}_segmentation.tif",
                        f"{base_path}_segmentation.npy",
                        os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.tif"),
                        os.path.join(os.path.dirname(file_path), f"{acq_name}_segmentation.npy"),
                    ]
                    for mask_path in possible_paths:
                        if os.path.exists(mask_path):
                            if mask_path.endswith('.npy'):
                                mask = np.load(mask_path)
                            else:
                                mask = tifffile.imread(mask_path)
                            break
            
            # Compute correlations
            correlations = self._compute_correlations(img_stack, channels, mask, acq_id)
            
            # Only close loader if we created it (not if using existing loader from main window)
            if not use_existing_loader:
                loader.close()
            return correlations
        except Exception as e:
            print(f"Error processing ROI {acq_name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _compute_correlations(self, img_stack: np.ndarray, channels: List[str], 
                             mask: Optional[np.ndarray], acq_id: str) -> List[Dict]:
        """Compute Spearman correlations for all marker pairs."""
        correlations = []
        
        # Determine shape - loaders return HWC format (H, W, C)
        if img_stack.ndim == 3:
            height, width, n_channels = img_stack.shape
        elif img_stack.ndim == 2:
            # Single channel
            return correlations
        else:
            return correlations
        
        # Flatten images and apply mask if provided
        pixel_data = {}
        for i, channel in enumerate(channels):
            if i >= n_channels:
                continue
            # Extract channel from HWC format: (H, W, C) -> (H, W)
            channel_img = img_stack[:, :, i] if img_stack.ndim == 3 else img_stack
            
            if mask is not None:
                # Only use pixels within cells
                # Ensure mask matches image dimensions
                if mask.shape == channel_img.shape:
                    cell_mask = mask > 0
                    pixels = channel_img[cell_mask]
                else:
                    # Mask dimensions don't match, skip mask
                    pixels = channel_img.flatten()
            else:
                # Use all pixels
                pixels = channel_img.flatten()
            
            # Remove NaN and infinite values
            pixels = pixels[~np.isnan(pixels) & ~np.isinf(pixels)]
            pixel_data[channel] = pixels
        
        # Compute pairwise correlations
        channel_list = list(pixel_data.keys())
        for i, ch1 in enumerate(channel_list):
            for j, ch2 in enumerate(channel_list):
                if i >= j:  # Only compute upper triangle
                    continue
                
                data1 = pixel_data[ch1]
                data2 = pixel_data[ch2]
                
                # Ensure same length (take minimum)
                min_len = min(len(data1), len(data2))
                if min_len < 3:  # Need at least 3 points for correlation
                    continue
                
                data1 = data1[:min_len]
                data2 = data2[:min_len]
                
                # Compute Spearman correlation
                try:
                    corr_coef, p_value = spearmanr(data1, data2)
                    
                    if not np.isnan(corr_coef) and not np.isinf(corr_coef):
                        correlations.append({
                            'marker1': ch1,
                            'marker2': ch2,
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'n_pixels': min_len
                        })
                except Exception as e:
                    print(f"Error computing correlation for {ch1}-{ch2}: {e}")
                    continue
        
        return correlations
    
    def _aggregate_results(self, use_conditions: bool):
        """Aggregate correlation results to mean values per marker pair and apply Benjamini-Hochberg correction."""
        if self.correlation_results is None or self.correlation_results.empty:
            return
        
        if use_conditions and 'condition' in self.correlation_results.columns:
            # Group by condition and marker pair
            grouped = self.correlation_results.groupby(['condition', 'marker1', 'marker2']).agg({
                'correlation': 'mean',
                'p_value': 'mean',
                'n_pixels': 'sum'
            }).reset_index()
            
            # Apply Benjamini-Hochberg correction within each condition
            grouped['p_value_adjusted'] = np.nan
            for condition in grouped['condition'].unique():
                condition_mask = grouped['condition'] == condition
                p_values = grouped.loc[condition_mask, 'p_value'].values
                # Apply BH correction
                _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
                grouped.loc[condition_mask, 'p_value_adjusted'] = p_adjusted
        else:
            # Group by marker pair only
            grouped = self.correlation_results.groupby(['marker1', 'marker2']).agg({
                'correlation': 'mean',
                'p_value': 'mean',
                'n_pixels': 'sum'
            }).reset_index()
            
            # Apply Benjamini-Hochberg correction to all p-values
            p_values = grouped['p_value'].values
            _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
            grouped['p_value_adjusted'] = p_adjusted
        
        self.aggregated_results = grouped
    
    def _update_results_display(self, use_conditions: bool):
        """Update the results display with correlation data."""
        if self.aggregated_results is None or self.aggregated_results.empty:
            return
        
        # Update heatmap
        self._update_heatmap(use_conditions)
        
        # Update table
        self._update_table()
        
        # Enable buttons
        self.save_matrix_btn.setEnabled(True)
        self.export_table_btn.setEnabled(True)
    
    def _update_heatmap(self, use_conditions: bool):
        """Update correlation heatmap."""
        if self.aggregated_results is None or self.aggregated_results.empty:
            return
        
        # Get unique markers
        all_markers = set()
        all_markers.update(self.aggregated_results['marker1'].unique())
        all_markers.update(self.aggregated_results['marker2'].unique())
        markers = sorted(all_markers)
        
        if len(markers) == 0:
            return
        
        fig = self.correlation_canvas.figure
        fig.clear()
        
        if use_conditions and 'condition' in self.aggregated_results.columns:
            # Get unique conditions
            conditions = sorted(self.aggregated_results['condition'].unique())
            n_conditions = len(conditions)
            
            if n_conditions == 1:
                ax = fig.add_subplot(111)
                self._plot_condition_heatmap(conditions[0], markers, ax)
                fig.suptitle(f"Pixel-Level Correlation: {conditions[0]}", fontsize=12)
            else:
                # Multiple subplots
                n_cols = min(2, n_conditions)
                n_rows = (n_conditions + n_cols - 1) // n_cols
                
                for idx, condition in enumerate(conditions):
                    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
                    self._plot_condition_heatmap(condition, markers, ax)
                    ax.set_title(condition, fontsize=10)
                
                fig.suptitle("Pixel-Level Correlation by Condition", fontsize=12)
        else:
            # Single heatmap for all ROIs
            ax = fig.add_subplot(111)
            self._plot_single_heatmap(markers, ax)
            ax.set_title("Pixel-Level Correlation Matrix", fontsize=12)
        
        fig.tight_layout()
        self.correlation_canvas.draw()
    
    def _plot_single_heatmap(self, markers: List[str], ax):
        """Plot single heatmap for all ROIs."""
        n_markers = len(markers)
        corr_matrix = np.full((n_markers, n_markers), np.nan)
        
        marker_to_idx = {marker: idx for idx, marker in enumerate(markers)}
        
        for _, row in self.aggregated_results.iterrows():
            m1 = row['marker1']
            m2 = row['marker2']
            corr = row['correlation']
            
            if m1 in marker_to_idx and m2 in marker_to_idx:
                idx1 = marker_to_idx[m1]
                idx2 = marker_to_idx[m2]
                corr_matrix[idx1, idx2] = corr
                corr_matrix[idx2, idx1] = corr  # Make symmetric
        
        # Set diagonal to 1.0
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_markers))
        ax.set_yticks(range(n_markers))
        ax.set_xticklabels(markers, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(markers, fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Spearman Correlation')
    
    def _plot_condition_heatmap(self, condition: str, markers: List[str], ax):
        """Plot heatmap for a single condition."""
        # Create correlation matrix
        n_markers = len(markers)
        corr_matrix = np.full((n_markers, n_markers), np.nan)
        
        # Fill matrix
        condition_data = self.aggregated_results[
            self.aggregated_results['condition'] == condition
        ]
        
        marker_to_idx = {marker: idx for idx, marker in enumerate(markers)}
        
        for _, row in condition_data.iterrows():
            m1 = row['marker1']
            m2 = row['marker2']
            corr = row['correlation']
            
            if m1 in marker_to_idx and m2 in marker_to_idx:
                idx1 = marker_to_idx[m1]
                idx2 = marker_to_idx[m2]
                corr_matrix[idx1, idx2] = corr
                corr_matrix[idx2, idx1] = corr  # Make symmetric
        
        # Set diagonal to 1.0
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n_markers))
        ax.set_yticks(range(n_markers))
        ax.set_xticklabels(markers, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(markers, fontsize=8)
        ax.set_title(condition, fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Spearman Correlation')
    
    def _update_table(self):
        """Update results table with sortable columns."""
        if self.aggregated_results is None or self.aggregated_results.empty:
            return
        
        df = self.aggregated_results.copy()
        
        # Format columns
        df['correlation'] = df['correlation'].round(4)
        
        # Format p-values with scientific notation for very small values
        def format_p_value(p_val):
            """Format p-value, using scientific notation for very small values."""
            if pd.isna(p_val) or np.isnan(p_val):
                return 'N/A'
            
            # Check if value is effectively zero (very close to zero)
            if p_val == 0.0 or abs(p_val) < 1e-10:
                # For exactly zero or extremely small values, show in scientific notation
                if p_val == 0.0:
                    return '0.0'
                else:
                    return f"{p_val:.3e}"
            
            # Use scientific notation for values < 0.001
            if p_val < 0.001:
                return f"{p_val:.3e}"
            else:
                # For values >= 0.001, use decimal notation
                # But check if rounding to 6 decimals would result in 0.0
                rounded = round(p_val, 6)
                if rounded == 0.0 and p_val > 0:
                    # If rounding would show 0.0 but value is actually > 0, use scientific notation
                    return f"{p_val:.3e}"
                else:
                    return f"{p_val:.6f}"
        
        df['p_value'] = df['p_value'].apply(format_p_value)
        if 'p_value_adjusted' in df.columns:
            df['p_value_adjusted'] = df['p_value_adjusted'].apply(format_p_value)
        
        # Rename columns for better display
        column_mapping = {
            'p_value': 'P-value (raw)',
            'p_value_adjusted': 'P-value (BH adjusted)',
            'correlation': 'Correlation',
            'n_pixels': 'N Pixels'
        }
        df = df.rename(columns=column_mapping)
        
        # Store original numeric values for sorting
        original_df = self.aggregated_results.copy()
        
        # Set up table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        # Populate table
        for i, row in enumerate(df.itertuples(index=False)):
            for j, (col_name, value) in enumerate(zip(df.columns, row)):
                # Use NumericTableWidgetItem for proper numeric sorting
                item = NumericTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                
                # Store original numeric values for sorting (for p-value columns)
                if col_name in ['P-value (raw)', 'P-value (BH adjusted)']:
                    # Get original numeric value from original_df
                    if col_name == 'P-value (raw)':
                        orig_col = 'p_value'
                    else:
                        orig_col = 'p_value_adjusted'
                    
                    if orig_col in original_df.columns:
                        orig_value = original_df.iloc[i][orig_col]
                        if not pd.isna(orig_value) and not np.isnan(orig_value):
                            item.setData(QtCore.Qt.UserRole, float(orig_value))
                        else:
                            # Store a large value for NaN so they sort to the end
                            item.setData(QtCore.Qt.UserRole, float('inf'))
                    else:
                        item.setData(QtCore.Qt.UserRole, float('inf'))
                elif col_name == 'Correlation':
                    # Store correlation value for sorting
                    orig_value = original_df.iloc[i]['correlation']
                    if not pd.isna(orig_value) and not np.isnan(orig_value):
                        item.setData(QtCore.Qt.UserRole, float(orig_value))
                    else:
                        item.setData(QtCore.Qt.UserRole, float('inf'))
                elif col_name == 'N Pixels':
                    # Store n_pixels value for sorting
                    orig_value = original_df.iloc[i]['n_pixels']
                    if not pd.isna(orig_value) and not np.isnan(orig_value):
                        item.setData(QtCore.Qt.UserRole, float(orig_value))
                    else:
                        item.setData(QtCore.Qt.UserRole, float('inf'))
                else:
                    # For other columns (like marker names, condition), try to extract numeric value if possible
                    try:
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            item.setData(QtCore.Qt.UserRole, float(value))
                    except (ValueError, TypeError):
                        pass
                
                self.results_table.setItem(i, j, item)
        
        # Enable sorting
        self.results_table.setSortingEnabled(True)
        
        # Resize columns
        self.results_table.resizeColumnsToContents()
    
    def _save_matrix(self):
        """Save correlation heatmap."""
        if self.correlation_canvas.figure is None:
            return
        
        save_figure_with_options(self.correlation_canvas.figure, self)
    
    def _export_results(self):
        """Export results to CSV."""
        if self.aggregated_results is None or self.aggregated_results.empty:
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.aggregated_results.to_csv(file_path, index=False)
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Results exported to:\n{file_path}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting results: {str(e)}"
                )

