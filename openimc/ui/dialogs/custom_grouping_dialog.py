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
Custom Grouping Dialog for Batch Correction

This module provides a dialog for users to create custom groupings of acquisitions/ROIs
for batch correction. Users can filter acquisitions by substring and assign them to groups.
"""

from typing import Dict, List, Optional, Set
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class CustomGroupingDialog(QtWidgets.QDialog):
    """Dialog for creating custom groupings of acquisitions/ROIs."""
    
    def __init__(self, feature_dataframe: Optional[pd.DataFrame] = None, existing_grouping: Optional[Dict[str, str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Grouping for Batch Correction")
        self.setModal(True)
        self.resize(900, 700)
        
        self.feature_dataframe = feature_dataframe
        # Initialize with existing grouping if provided
        self.grouping: Dict[str, str] = existing_grouping.copy() if existing_grouping else {}
        
        # Determine which column to use for grouping (source_well preferred, acquisition_id as fallback)
        self.grouping_column = None
        if feature_dataframe is not None and not feature_dataframe.empty:
            if 'source_well' in feature_dataframe.columns:
                self.grouping_column = 'source_well'
            elif 'acquisition_id' in feature_dataframe.columns:
                self.grouping_column = 'acquisition_id'
        
        # Get unique acquisitions from dataframe
        self.acquisitions: List[Dict[str, str]] = []
        if feature_dataframe is not None and not feature_dataframe.empty and self.grouping_column:
            # Get unique acquisition IDs/wells and their names
            unique_acqs = feature_dataframe.groupby(self.grouping_column).first()
            for acq_id, row in unique_acqs.iterrows():
                # Use source_well as display name if available, otherwise use acquisition_name or id
                if self.grouping_column == 'source_well':
                    acq_name = str(acq_id)  # source_well is already descriptive
                else:
                    acq_name = row.get('acquisition_name', str(acq_id))
                self.acquisitions.append({
                    'id': str(acq_id),
                    'name': str(acq_name)
                })
        
        self._create_ui()
        self._populate_groups_from_grouping()
        self._populate_acquisitions()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Instructions
        info_label = QtWidgets.QLabel(
            "Create custom groups for batch correction by selecting acquisitions/ROIs and assigning them to groups.\n"
            "Use the filter box to quickly find acquisitions by substring (e.g., 'Patient1_', 'Patient2_')."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        layout.addWidget(info_label)
        
        # Main content area (split horizontally)
        main_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        
        # Left side: Acquisition list with filter
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Filter section
        filter_label = QtWidgets.QLabel("Filter acquisitions (substring):")
        left_layout.addWidget(filter_label)
        
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Type to filter by acquisition name...")
        self.filter_edit.textChanged.connect(self._on_filter_changed)
        left_layout.addWidget(self.filter_edit)
        
        # Acquisition list
        acq_label = QtWidgets.QLabel("Available acquisitions/ROIs:")
        left_layout.addWidget(acq_label)
        
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.acq_list.setMinimumWidth(300)
        left_layout.addWidget(self.acq_list)
        
        # Selection info
        self.selection_info = QtWidgets.QLabel("0 selected")
        self.selection_info.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        left_layout.addWidget(self.selection_info)
        
        self.acq_list.itemSelectionChanged.connect(self._on_selection_changed)
        
        main_splitter.addWidget(left_widget)
        
        # Right side: Group management
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Group management section
        group_label = QtWidgets.QLabel("Groups:")
        right_layout.addWidget(group_label)
        
        # Group list
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setMinimumWidth(200)
        self.group_list.itemDoubleClicked.connect(self._rename_group)
        right_layout.addWidget(self.group_list)
        
        # Group buttons
        group_buttons_layout = QtWidgets.QHBoxLayout()
        
        self.add_group_btn = QtWidgets.QPushButton("New Group")
        self.add_group_btn.clicked.connect(self._add_group)
        group_buttons_layout.addWidget(self.add_group_btn)
        
        self.rename_group_btn = QtWidgets.QPushButton("Rename")
        self.rename_group_btn.clicked.connect(self._rename_group)
        group_buttons_layout.addWidget(self.rename_group_btn)
        
        self.remove_group_btn = QtWidgets.QPushButton("Remove")
        self.remove_group_btn.clicked.connect(self._remove_group)
        group_buttons_layout.addWidget(self.remove_group_btn)
        
        right_layout.addLayout(group_buttons_layout)
        
        # Assign section
        assign_label = QtWidgets.QLabel("Assign selected acquisitions to group:")
        right_layout.addWidget(assign_label)
        
        self.assign_combo = QtWidgets.QComboBox()
        self.assign_combo.setEditable(True)  # Allow creating new group on the fly
        self.assign_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        right_layout.addWidget(self.assign_combo)
        
        self.assign_btn = QtWidgets.QPushButton("Assign to Group")
        self.assign_btn.clicked.connect(self._assign_to_group)
        self.assign_btn.setEnabled(False)
        right_layout.addWidget(self.assign_btn)
        
        # Current assignments display
        assignment_label = QtWidgets.QLabel("Current assignments:")
        right_layout.addWidget(assignment_label)
        
        self.assignment_list = QtWidgets.QListWidget()
        self.assignment_list.setMaximumHeight(150)
        right_layout.addWidget(self.assignment_list)
        
        # Remove assignment button
        self.remove_assignment_btn = QtWidgets.QPushButton("Remove Assignment")
        self.remove_assignment_btn.clicked.connect(self._remove_assignment)
        self.remove_assignment_btn.setEnabled(False)
        right_layout.addWidget(self.remove_assignment_btn)
        
        self.assignment_list.itemSelectionChanged.connect(self._on_assignment_selection_changed)
        
        right_layout.addStretch()
        
        main_splitter.addWidget(right_widget)
        
        # Set splitter proportions (60% left, 40% right)
        main_splitter.setSizes([600, 400])
        layout.addWidget(main_splitter)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.clear_all_btn = QtWidgets.QPushButton("Clear All Assignments")
        self.clear_all_btn.clicked.connect(self._clear_all_assignments)
        button_layout.addWidget(self.clear_all_btn)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self._validate_and_accept)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
    
    def _populate_groups_from_grouping(self):
        """Populate the group list from existing grouping."""
        if not self.grouping:
            return
        
        # Get unique group names
        unique_groups = set(self.grouping.values())
        for group_name in sorted(unique_groups):
            self.group_list.addItem(group_name)
        
        self._update_assign_combo()
    
    def _populate_acquisitions(self):
        """Populate the acquisition list."""
        self.acq_list.clear()
        
        for acq in self.acquisitions:
            item = QtWidgets.QListWidgetItem(acq['name'])
            item.setData(Qt.UserRole, acq['id'])  # Store acquisition ID
            item.setData(Qt.UserRole + 1, acq['name'])  # Store original name for filtering
            # Check if already assigned to a group
            if acq['id'] in self.grouping:
                group_name = self.grouping[acq['id']]
                item.setText(f"{acq['name']} → {group_name}")
                item.setForeground(Qt.darkGreen)
            self.acq_list.addItem(item)
        
        self._update_assignment_list()
    
    def _on_filter_changed(self, text: str):
        """Handle filter text change."""
        filter_text = text.lower()
        
        for i in range(self.acq_list.count()):
            item = self.acq_list.item(i)
            # Get original name from stored data or from item text (before →)
            original_name = item.data(Qt.UserRole + 1)
            if not original_name:
                original_name = item.text().split(' → ')[0]
            
            if filter_text in original_name.lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def _on_selection_changed(self):
        """Handle acquisition selection change."""
        selected_count = len(self.acq_list.selectedItems())
        self.selection_info.setText(f"{selected_count} selected")
        self.assign_btn.setEnabled(selected_count > 0 and self.assign_combo.count() > 0)
    
    def _add_group(self):
        """Add a new group."""
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "New Group",
            "Enter group name:",
            text="Group1"
        )
        
        if ok and name:
            name = name.strip()
            if not name:
                QtWidgets.QMessageBox.warning(self, "Invalid Name", "Group name cannot be empty.")
                return
            
            # Check if group already exists
            for i in range(self.group_list.count()):
                if self.group_list.item(i).text() == name:
                    QtWidgets.QMessageBox.warning(self, "Duplicate Group", f"Group '{name}' already exists.")
                    return
            
            # Add to group list
            self.group_list.addItem(name)
            self._update_assign_combo()
            self.assign_combo.setCurrentText(name)
    
    def _rename_group(self):
        """Rename the selected group."""
        current_item = self.group_list.currentItem()
        if not current_item:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a group to rename.")
            return
        
        old_name = current_item.text()
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename Group",
            "Enter new group name:",
            text=old_name
        )
        
        if ok and name:
            name = name.strip()
            if not name:
                QtWidgets.QMessageBox.warning(self, "Invalid Name", "Group name cannot be empty.")
                return
            
            # Check if new name already exists
            for i in range(self.group_list.count()):
                if self.group_list.item(i).text() == name and self.group_list.item(i) != current_item:
                    QtWidgets.QMessageBox.warning(self, "Duplicate Group", f"Group '{name}' already exists.")
                    return
            
            # Update group name in all assignments
            for acq_id in list(self.grouping.keys()):
                if self.grouping[acq_id] == old_name:
                    self.grouping[acq_id] = name
            
            # Update list item
            current_item.setText(name)
            self._update_assign_combo()
            self._populate_acquisitions()
            self._update_assignment_list()
    
    def _remove_group(self):
        """Remove the selected group."""
        current_item = self.group_list.currentItem()
        if not current_item:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a group to remove.")
            return
        
        group_name = current_item.text()
        
        # Confirm removal
        reply = QtWidgets.QMessageBox.question(
            self,
            "Remove Group",
            f"Remove group '{group_name}'? All assignments to this group will be cleared.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Remove all assignments to this group
            acq_ids_to_remove = [acq_id for acq_id, g in self.grouping.items() if g == group_name]
            for acq_id in acq_ids_to_remove:
                del self.grouping[acq_id]
            
            # Remove from list
            self.group_list.takeItem(self.group_list.row(current_item))
            self._update_assign_combo()
            self._populate_acquisitions()
            self._update_assignment_list()
    
    def _update_assign_combo(self):
        """Update the assign combo box with current groups."""
        current_text = self.assign_combo.currentText()
        self.assign_combo.clear()
        
        for i in range(self.group_list.count()):
            self.assign_combo.addItem(self.group_list.item(i).text())
        
        # Restore selection if possible
        if current_text:
            index = self.assign_combo.findText(current_text)
            if index >= 0:
                self.assign_combo.setCurrentIndex(index)
            else:
                self.assign_combo.setCurrentText(current_text)
    
    def _assign_to_group(self):
        """Assign selected acquisitions to the selected group."""
        selected_items = self.acq_list.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select acquisitions to assign.")
            return
        
        group_name = self.assign_combo.currentText().strip()
        if not group_name:
            QtWidgets.QMessageBox.warning(self, "No Group", "Please select or enter a group name.")
            return
        
        # If group doesn't exist, create it
        group_exists = False
        for i in range(self.group_list.count()):
            if self.group_list.item(i).text() == group_name:
                group_exists = True
                break
        
        if not group_exists:
            self.group_list.addItem(group_name)
            self._update_assign_combo()
        
        # Assign selected acquisitions to group
        for item in selected_items:
            acq_id = item.data(Qt.UserRole)
            self.grouping[acq_id] = group_name
        
        self._populate_acquisitions()
        self._update_assignment_list()
        self.acq_list.clearSelection()
    
    def _update_assignment_list(self):
        """Update the assignment list display."""
        self.assignment_list.clear()
        
        # Group assignments by group name
        groups_dict: Dict[str, List[str]] = {}
        for acq_id, group_name in self.grouping.items():
            if group_name not in groups_dict:
                groups_dict[group_name] = []
            # Find acquisition name
            acq_name = acq_id
            for acq in self.acquisitions:
                if acq['id'] == acq_id:
                    acq_name = acq['name']
                    break
            groups_dict[group_name].append(f"{acq_name} ({acq_id})")
        
        # Display grouped by group name
        for group_name in sorted(groups_dict.keys()):
            group_item = QtWidgets.QListWidgetItem(f"Group: {group_name}")
            group_item.setData(Qt.UserRole, group_name)
            group_item.setForeground(Qt.blue)
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsSelectable)  # Not selectable
            self.assignment_list.addItem(group_item)
            
            for acq_display in sorted(groups_dict[group_name]):
                acq_item = QtWidgets.QListWidgetItem(f"  • {acq_display}")
                acq_item.setData(Qt.UserRole, group_name)
                self.assignment_list.addItem(acq_item)
    
    def _on_assignment_selection_changed(self):
        """Handle assignment list selection change."""
        selected_items = self.assignment_list.selectedItems()
        # Only enable if a non-group header item is selected
        has_selectable = any(
            item.flags() & Qt.ItemIsSelectable 
            for item in selected_items
        )
        self.remove_assignment_btn.setEnabled(has_selectable)
    
    def _remove_assignment(self):
        """Remove assignment for selected acquisition."""
        selected_items = self.assignment_list.selectedItems()
        if not selected_items:
            return
        
        # Get acquisition IDs to remove
        acq_ids_to_remove = []
        for item in selected_items:
            if item.flags() & Qt.ItemIsSelectable:  # Only process selectable items
                # Extract acquisition ID from display text
                display_text = item.text().strip()
                if display_text.startswith("  • "):
                    # Format: "  • Acquisition Name (acq_id)"
                    if "(" in display_text and ")" in display_text:
                        acq_id = display_text.split("(")[-1].rstrip(")")
                        acq_ids_to_remove.append(acq_id)
        
        # Remove assignments
        for acq_id in acq_ids_to_remove:
            if acq_id in self.grouping:
                del self.grouping[acq_id]
        
        self._populate_acquisitions()
        self._update_assignment_list()
    
    def _clear_all_assignments(self):
        """Clear all assignments."""
        if not self.grouping:
            return
        
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear All",
            "Clear all group assignments?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.grouping.clear()
            self._populate_acquisitions()
            self._update_assignment_list()
    
    def _validate_and_accept(self):
        """Validate grouping and accept dialog."""
        if not self.grouping:
            reply = QtWidgets.QMessageBox.question(
                self,
                "No Assignments",
                "No acquisitions have been assigned to groups. Continue anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                return
        
        # Check that all acquisitions in grouping still exist
        valid_acq_ids = {acq['id'] for acq in self.acquisitions}
        invalid_acq_ids = set(self.grouping.keys()) - valid_acq_ids
        if invalid_acq_ids:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Assignments",
                f"Some assignments reference acquisitions that no longer exist:\n{', '.join(invalid_acq_ids)}\n"
                "These will be removed."
            )
            for acq_id in invalid_acq_ids:
                del self.grouping[acq_id]
        
        self.accept()
    
    def get_grouping(self) -> Dict[str, str]:
        """Get the grouping mapping (acquisition_id -> group_name)."""
        return self.grouping.copy()

