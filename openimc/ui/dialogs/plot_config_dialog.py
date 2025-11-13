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
Plot Configuration Dialog for Clustering Analysis

This module provides a dialog for configuring plot settings after clustering has been performed.
"""

from typing import Optional, Dict, List
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class PlotConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring plot settings after clustering."""
    
    def __init__(self, parent_dialog, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.resize(700, 800)
        
        self.parent_dialog = parent_dialog  # Reference to CellClusteringDialog
        
        # Get current view from parent dialog
        if hasattr(parent_dialog, 'view_combo'):
            self.current_view = parent_dialog.view_combo.currentText()
        else:
            self.current_view = 'Heatmap'
        
        # Set window title based on current view
        self.setWindowTitle(f"Configure {self.current_view} Plot")
        
        self._create_ui()
        self._populate_from_parent()
        self._update_visibility_for_view()
    
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
        
        # Color-by control (UMAP/t-SNE only) - multi-select for faceted plotting
        # NOT shown for Heatmap (uses patient annotation bar instead)
        self.color_by_group = QtWidgets.QGroupBox("Color By (UMAP/t-SNE)")
        color_by_layout = QtWidgets.QVBoxLayout()
        self.color_by_label = QtWidgets.QLabel("Color by (select multiple for faceted plots):")
        color_by_layout.addWidget(self.color_by_label)
        # Search/filter box for color-by options
        self.color_by_search = QtWidgets.QLineEdit()
        self.color_by_search.setPlaceholderText("Search/filter options...")
        self.color_by_search.textChanged.connect(self._filter_color_by_options)
        color_by_layout.addWidget(self.color_by_search)
        self.color_by_listwidget = QtWidgets.QListWidget()
        self.color_by_listwidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.color_by_listwidget.setMaximumHeight(150)
        self.color_by_listwidget.itemSelectionChanged.connect(self._on_color_by_changed)
        color_by_layout.addWidget(self.color_by_listwidget)
        self.color_by_group.setLayout(color_by_layout)
        content_layout.addWidget(self.color_by_group)
        
        # UMAP/t-SNE point style controls
        self.point_style_group = QtWidgets.QGroupBox("Point Style (UMAP/t-SNE)")
        point_style_layout = QtWidgets.QHBoxLayout()
        self.point_size_label = QtWidgets.QLabel("Point size:")
        point_style_layout.addWidget(self.point_size_label)
        self.point_size_spinbox = QtWidgets.QSpinBox()
        self.point_size_spinbox.setMinimum(1)
        self.point_size_spinbox.setMaximum(200)
        self.point_size_spinbox.setValue(8)
        self.point_size_spinbox.setToolTip("Size of points in scatter plot")
        point_style_layout.addWidget(self.point_size_spinbox)

        self.point_alpha_label = QtWidgets.QLabel("Point alpha:")
        point_style_layout.addWidget(self.point_alpha_label)
        self.point_alpha_spinbox = QtWidgets.QDoubleSpinBox()
        self.point_alpha_spinbox.setMinimum(0.0)
        self.point_alpha_spinbox.setMaximum(1.0)
        self.point_alpha_spinbox.setSingleStep(0.1)
        self.point_alpha_spinbox.setValue(0.8)
        self.point_alpha_spinbox.setDecimals(2)
        self.point_alpha_spinbox.setToolTip("Transparency of points (0.0 = transparent, 1.0 = opaque)")
        point_style_layout.addWidget(self.point_alpha_spinbox)
        point_style_layout.addStretch()
        self.point_style_group.setLayout(point_style_layout)
        content_layout.addWidget(self.point_style_group)

        # Remake UMAP button (UMAP only)
        self.remake_umap_btn = QtWidgets.QPushButton("Remake UMAP")
        self.remake_umap_btn.setToolTip("Regenerate UMAP with new parameters (features, scaling, n_neighbors)")
        self.remake_umap_btn.clicked.connect(self._remake_umap)
        content_layout.addWidget(self.remake_umap_btn)

        # Group-by for stacked bars (Stacked Bars only)
        self.group_by_widget = QtWidgets.QWidget()
        group_by_layout = QtWidgets.QHBoxLayout(self.group_by_widget)
        self.group_by_label = QtWidgets.QLabel("Group by:")
        group_by_layout.addWidget(self.group_by_label)
        self.group_by_combo = QtWidgets.QComboBox()
        group_by_layout.addWidget(self.group_by_combo)
        group_by_layout.addStretch()
        content_layout.addWidget(self.group_by_widget)

        # Colormap selector (for heatmaps and differential expression)
        self.colormap_widget = QtWidgets.QWidget()
        colormap_layout = QtWidgets.QHBoxLayout(self.colormap_widget)
        self.colormap_label = QtWidgets.QLabel("Colormap:")
        colormap_layout.addWidget(self.colormap_label)
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems([
            "RdBu_r (Red-White-Blue)",
            "viridis (Purple-Green-Yellow)", 
            "plasma (Purple-Pink-Yellow)",
            "inferno (Purple-Red-Yellow)",
            "Blues (Light-Dark Blue)",
            "Reds (Light-Dark Red)",
            "Greens (Light-Dark Green)",
            "Oranges (Light-Dark Orange)",
            "Purples (Light-Dark Purple)"
        ])
        self.colormap_combo.setCurrentText("RdBu_r (Red-White-Blue)")
        colormap_layout.addWidget(self.colormap_combo)
        colormap_layout.addStretch()
        content_layout.addWidget(self.colormap_widget)

        # Heatmap settings group
        self.heatmap_group = QtWidgets.QGroupBox("Heatmap Settings")
        heatmap_layout = QtWidgets.QVBoxLayout()
        
        # Heatmap source selector (Clusters vs Manual Gates)
        heatmap_source_layout = QtWidgets.QHBoxLayout()
        self.heatmap_source_label = QtWidgets.QLabel("Heatmap of:")
        heatmap_source_layout.addWidget(self.heatmap_source_label)
        self.heatmap_source_combo = QtWidgets.QComboBox()
        self.heatmap_source_combo.addItems(["Clusters", "Manual Gates"])
        heatmap_source_layout.addWidget(self.heatmap_source_combo)
        heatmap_source_layout.addStretch()
        heatmap_layout.addLayout(heatmap_source_layout)

        # Heatmap filter and scaling
        heatmap_options_layout = QtWidgets.QHBoxLayout()
        self.heatmap_filter_btn = QtWidgets.QPushButton("Filter…")
        self.heatmap_filter_btn.setToolTip("Filter which clusters/phenotypes appear in the heatmap")
        self.heatmap_filter_btn.clicked.connect(self._open_heatmap_filter_dialog)
        heatmap_options_layout.addWidget(self.heatmap_filter_btn)

        self.heatmap_scaling_label = QtWidgets.QLabel("Scaling:")
        heatmap_options_layout.addWidget(self.heatmap_scaling_label)
        self.heatmap_scaling_combo = QtWidgets.QComboBox()
        self.heatmap_scaling_combo.addItems(["Z-score", "MAD (Median Absolute Deviation)", "None (no scaling)"])
        self.heatmap_scaling_combo.setCurrentText("Z-score")
        heatmap_options_layout.addWidget(self.heatmap_scaling_combo)
        heatmap_options_layout.addStretch()
        heatmap_layout.addLayout(heatmap_options_layout)
        
        # Feature tick font size (for heatmap)
        feature_fontsize_layout = QtWidgets.QHBoxLayout()
        self.feature_fontsize_label = QtWidgets.QLabel("Feature label font size:")
        feature_fontsize_layout.addWidget(self.feature_fontsize_label)
        self.feature_fontsize_spinbox = QtWidgets.QSpinBox()
        self.feature_fontsize_spinbox.setMinimum(4)
        self.feature_fontsize_spinbox.setMaximum(20)
        self.feature_fontsize_spinbox.setValue(8)
        self.feature_fontsize_spinbox.setToolTip("Font size for feature labels on y-axis")
        feature_fontsize_layout.addWidget(self.feature_fontsize_spinbox)
        feature_fontsize_layout.addStretch()
        heatmap_layout.addLayout(feature_fontsize_layout)
        
        self.heatmap_group.setLayout(heatmap_layout)
        content_layout.addWidget(self.heatmap_group)

        # Patient annotation controls (for heatmap only)
        self.patient_annotation_group = QtWidgets.QGroupBox("Patient Annotation")
        patient_annotation_group_layout = QtWidgets.QVBoxLayout()
        
        patient_annotation_layout = QtWidgets.QHBoxLayout()
        self.patient_annotation_checkbox = QtWidgets.QCheckBox("Show patient annotation")
        self.patient_annotation_checkbox.setToolTip("Show patient/source file annotation bar above cell annotation in heatmap")
        self.patient_annotation_checkbox.stateChanged.connect(self._on_patient_annotation_changed)
        patient_annotation_layout.addWidget(self.patient_annotation_checkbox)
        
        # Column selector for patient annotation
        self.patient_annotation_column_label = QtWidgets.QLabel("Column:")
        self.patient_annotation_column_label.setEnabled(False)
        patient_annotation_layout.addWidget(self.patient_annotation_column_label)
        
        self.patient_annotation_column_combo = QtWidgets.QComboBox()
        self.patient_annotation_column_combo.setEnabled(False)
        self.patient_annotation_column_combo.setToolTip("Select column to use for patient/source annotation")
        patient_annotation_layout.addWidget(self.patient_annotation_column_combo)
        patient_annotation_layout.addStretch()
        patient_annotation_group_layout.addLayout(patient_annotation_layout)
        
        self.patient_annotate_btn = QtWidgets.QPushButton("Customize Patient Labels...")
        self.patient_annotate_btn.setToolTip("Customize labels for patient/source file annotation")
        self.patient_annotate_btn.clicked.connect(self._open_patient_annotation_dialog)
        self.patient_annotate_btn.setEnabled(False)
        patient_annotation_group_layout.addWidget(self.patient_annotate_btn)
        
        # Legend label customization
        legend_label_layout = QtWidgets.QHBoxLayout()
        self.legend_label_label = QtWidgets.QLabel("Legend label:")
        legend_label_layout.addWidget(self.legend_label_label)
        self.legend_label_edit = QtWidgets.QLineEdit()
        self.legend_label_edit.setToolTip("Custom label for the patient/source annotation legend (e.g., 'Patient', 'Sample', 'Source')")
        self.legend_label_edit.setEnabled(False)
        legend_label_layout.addWidget(self.legend_label_edit)
        legend_label_layout.addStretch()
        patient_annotation_group_layout.addLayout(legend_label_layout)
        
        self.patient_annotation_checkbox.stateChanged.connect(lambda state: self.legend_label_edit.setEnabled(state == 2))
        
        self.patient_annotation_group.setLayout(patient_annotation_group_layout)
        content_layout.addWidget(self.patient_annotation_group)
        
        # Feature label customization button (available for all views)
        self.feature_labels_btn = QtWidgets.QPushButton("Customize Feature Labels...")
        self.feature_labels_btn.setToolTip("Set custom display names for features in visualizations (e.g., 'Vimentin_mean' -> 'Mean Vimentin')")
        self.feature_labels_btn.clicked.connect(self._open_feature_labels_dialog)
        content_layout.addWidget(self.feature_labels_btn)

        # Differential Expression settings
        self.de_group = QtWidgets.QGroupBox("Differential Expression Settings")
        de_layout = QtWidgets.QHBoxLayout()
        self.top_n_label = QtWidgets.QLabel("Top N:")
        de_layout.addWidget(self.top_n_label)
        self.top_n_spinbox = QtWidgets.QSpinBox()
        self.top_n_spinbox.setMinimum(1)
        self.top_n_spinbox.setMaximum(20)
        self.top_n_spinbox.setValue(5)
        de_layout.addWidget(self.top_n_spinbox)
        de_layout.addStretch()
        self.de_group.setLayout(de_layout)
        content_layout.addWidget(self.de_group)

        # Boxplot/Violin Plot settings
        self.boxplot_group = QtWidgets.QGroupBox("Boxplot/Violin Plot Settings")
        boxplot_layout = QtWidgets.QVBoxLayout()
        
        marker_select_layout = QtWidgets.QHBoxLayout()
        self.marker_select_label = QtWidgets.QLabel("Markers:")
        marker_select_layout.addWidget(self.marker_select_label)
        self.marker_select_btn = QtWidgets.QPushButton("Select Markers...")
        self.marker_select_btn.setToolTip("Select markers to visualize")
        self.marker_select_btn.clicked.connect(self._open_marker_selection_dialog)
        marker_select_layout.addWidget(self.marker_select_btn)
        marker_select_layout.addStretch()
        boxplot_layout.addLayout(marker_select_layout)

        plot_type_layout = QtWidgets.QHBoxLayout()
        self.plot_type_label = QtWidgets.QLabel("Plot type:")
        plot_type_layout.addWidget(self.plot_type_label)
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(["Violin Plot", "Boxplot"])
        self.plot_type_combo.setCurrentText("Violin Plot")
        plot_type_layout.addWidget(self.plot_type_combo)
        plot_type_layout.addStretch()
        boxplot_layout.addLayout(plot_type_layout)

        # Statistical testing
        stats_group = QtWidgets.QGroupBox("Statistical Testing")
        stats_layout = QtWidgets.QVBoxLayout()
        self.stats_test_checkbox = QtWidgets.QCheckBox("Perform statistical testing")
        self.stats_test_checkbox.setToolTip("Compare groups using Mann-Whitney U test")
        stats_layout.addWidget(self.stats_test_checkbox)

        stats_mode_layout = QtWidgets.QHBoxLayout()
        self.stats_mode_label = QtWidgets.QLabel("Test mode:")
        stats_mode_layout.addWidget(self.stats_mode_label)
        self.stats_mode_combo = QtWidgets.QComboBox()
        self.stats_mode_combo.addItems(["Pairwise (all pairs)", "One vs Others"])
        stats_mode_layout.addWidget(self.stats_mode_combo)
        stats_mode_layout.addStretch()
        stats_layout.addLayout(stats_mode_layout)

        stats_cluster_layout = QtWidgets.QHBoxLayout()
        self.stats_cluster_label = QtWidgets.QLabel("Reference cluster:")
        stats_cluster_layout.addWidget(self.stats_cluster_label)
        self.stats_cluster_combo = QtWidgets.QComboBox()
        stats_cluster_layout.addWidget(self.stats_cluster_combo)
        stats_cluster_layout.addStretch()
        stats_layout.addLayout(stats_cluster_layout)

        self.stats_export_btn = QtWidgets.QPushButton("Export Statistical Results...")
        self.stats_export_btn.setToolTip("Export statistical test results to CSV")
        self.stats_export_btn.setEnabled(False)
        stats_layout.addWidget(self.stats_export_btn)
        stats_group.setLayout(stats_layout)
        boxplot_layout.addWidget(stats_group)
        
        self.boxplot_group.setLayout(boxplot_layout)
        content_layout.addWidget(self.boxplot_group)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_settings)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.apply_btn)
        layout.addLayout(button_layout)
    
    def _update_visibility_for_view(self):
        """Show/hide controls based on current view."""
        view = self.current_view
        
        # Color-by: Only for UMAP and t-SNE (NOT for Heatmap)
        show_color_by = view in ['UMAP', 't-SNE']
        self.color_by_group.setVisible(show_color_by)
        
        # Point style: Only for UMAP and t-SNE
        show_point_style = view in ['UMAP', 't-SNE']
        self.point_style_group.setVisible(show_point_style)
        
        # Remake UMAP: Only for UMAP
        self.remake_umap_btn.setVisible(view == 'UMAP')
        
        # Group-by: Only for Stacked Bars
        self.group_by_widget.setVisible(view == 'Stacked Bars')
        
        # Colormap: For Heatmap and Differential Expression
        show_colormap = view in ['Heatmap', 'Differential Expression']
        self.colormap_widget.setVisible(show_colormap)
        
        # Heatmap settings: Only for Heatmap
        self.heatmap_group.setVisible(view == 'Heatmap')
        
        # Patient annotation: Only for Heatmap
        self.patient_annotation_group.setVisible(view == 'Heatmap')
        
        # Differential Expression settings: Only for Differential Expression
        self.de_group.setVisible(view == 'Differential Expression')
        
        # Boxplot/Violin Plot settings: Only for Boxplot/Violin Plot
        self.boxplot_group.setVisible(view == 'Boxplot/Violin Plot')
        
        # Feature labels: Available for all views
        self.feature_labels_btn.setVisible(True)
    
    def _populate_from_parent(self):
        """Populate UI from parent dialog's current settings."""
        if not self.parent_dialog:
            return
        
        # Populate color-by options from parent
        # Temporarily block signals to prevent triggering dialogs
        self.color_by_listwidget.blockSignals(True)
        if hasattr(self.parent_dialog, '_populate_color_by_options'):
            self.parent_dialog._populate_color_by_options()
            # Copy items from parent to this dialog
            self.color_by_listwidget.clear()
            if hasattr(self.parent_dialog, 'color_by_listwidget'):
                selected_items = []
                for i in range(self.parent_dialog.color_by_listwidget.count()):
                    parent_item = self.parent_dialog.color_by_listwidget.item(i)
                    item = QtWidgets.QListWidgetItem(parent_item.text())
                    item.setData(Qt.UserRole, parent_item.data(Qt.UserRole))
                    self.color_by_listwidget.addItem(item)
                    if parent_item.isSelected():
                        item.setSelected(True)
                        selected_items.append(item.text())
        self.color_by_listwidget.blockSignals(False)
        
        # Point style
        if hasattr(self.parent_dialog, 'point_size_spinbox'):
            self.point_size_spinbox.setValue(self.parent_dialog.point_size_spinbox.value())
        if hasattr(self.parent_dialog, 'point_alpha_spinbox'):
            self.point_alpha_spinbox.setValue(self.parent_dialog.point_alpha_spinbox.value())
        
        # Group by
        if hasattr(self.parent_dialog, 'group_by_combo'):
            self.group_by_combo.clear()
            for i in range(self.parent_dialog.group_by_combo.count()):
                self.group_by_combo.addItem(self.parent_dialog.group_by_combo.itemText(i))
            self.group_by_combo.setCurrentText(self.parent_dialog.group_by_combo.currentText())
        
        # Colormap
        if hasattr(self.parent_dialog, 'colormap_combo'):
            self.colormap_combo.setCurrentText(self.parent_dialog.colormap_combo.currentText())
        
        # Heatmap settings
        if hasattr(self.parent_dialog, 'heatmap_source_combo'):
            self.heatmap_source_combo.setCurrentText(self.parent_dialog.heatmap_source_combo.currentText())
        if hasattr(self.parent_dialog, 'heatmap_scaling_combo'):
            self.heatmap_scaling_combo.setCurrentText(self.parent_dialog.heatmap_scaling_combo.currentText())
        # Feature tick font size
        if hasattr(self.parent_dialog, 'feature_tick_fontsize'):
            self.feature_fontsize_spinbox.setValue(self.parent_dialog.feature_tick_fontsize)
        
        # Patient annotation - populate column options
        # Block signals to prevent triggering dialogs during population
        self.patient_annotation_checkbox.blockSignals(True)
        self.patient_annotation_column_combo.blockSignals(True)
        
        # Populate column combo directly from dataframe
        self.patient_annotation_column_combo.clear()
        if hasattr(self.parent_dialog, 'feature_dataframe') and self.parent_dialog.feature_dataframe is not None:
            # Priority order: source_file, batch_group, source_well
            available_columns = []
            for col in ['source_file', 'batch_group', 'source_well']:
                if col in self.parent_dialog.feature_dataframe.columns:
                    available_columns.append(col)
            
            # Add available columns to combo
            for col in available_columns:
                self.patient_annotation_column_combo.addItem(col, col)
            
            # Set default selection (first available, or use stored value if exists)
            if available_columns:
                # Check if parent has a stored column selection
                stored_column = getattr(self.parent_dialog, 'patient_annotation_column', None)
                if stored_column and stored_column in available_columns:
                    index = available_columns.index(stored_column)
                    self.patient_annotation_column_combo.setCurrentIndex(index)
                else:
                    # Default to first available (source_file if present)
                    self.patient_annotation_column_combo.setCurrentIndex(0)
        
        # Set checkbox state and update enabled state
        # Get state from parent's enabled flag (since checkbox might not exist in parent)
        is_checked = getattr(self.parent_dialog, 'patient_annotation_enabled', False)
        if hasattr(self.parent_dialog, 'patient_annotation_checkbox'):
            is_checked = self.parent_dialog.patient_annotation_checkbox.isChecked()
        self.patient_annotation_checkbox.setChecked(is_checked)
        # Manually update enabled state without triggering signals
        self.patient_annotation_column_label.setEnabled(is_checked)
        self.patient_annotation_column_combo.setEnabled(is_checked)
        self.patient_annotate_btn.setEnabled(is_checked)
        self.legend_label_edit.setEnabled(is_checked)
        
        # Populate legend label
        if hasattr(self.parent_dialog, 'patient_legend_label'):
            self.legend_label_edit.setText(self.parent_dialog.patient_legend_label)
        
        self.patient_annotation_checkbox.blockSignals(False)
        self.patient_annotation_column_combo.blockSignals(False)
        
        # Top N
        if hasattr(self.parent_dialog, 'top_n_spinbox'):
            self.top_n_spinbox.setValue(self.parent_dialog.top_n_spinbox.value())
        
        # Plot type
        if hasattr(self.parent_dialog, 'plot_type_combo'):
            self.plot_type_combo.setCurrentText(self.parent_dialog.plot_type_combo.currentText())
        
        # Stats
        if hasattr(self.parent_dialog, 'stats_test_checkbox'):
            self.stats_test_checkbox.setChecked(self.parent_dialog.stats_test_checkbox.isChecked())
        if hasattr(self.parent_dialog, 'stats_mode_combo'):
            self.stats_mode_combo.setCurrentText(self.parent_dialog.stats_mode_combo.currentText())
        if hasattr(self.parent_dialog, 'stats_cluster_combo'):
            self.stats_cluster_combo.clear()
            for i in range(self.parent_dialog.stats_cluster_combo.count()):
                self.stats_cluster_combo.addItem(self.parent_dialog.stats_cluster_combo.itemText(i))
            self.stats_cluster_combo.setCurrentText(self.parent_dialog.stats_cluster_combo.currentText())
    
    def _filter_color_by_options(self, text: str):
        """Filter color-by options based on search text."""
        text_lower = text.lower()
        for i in range(self.color_by_listwidget.count()):
            item = self.color_by_listwidget.item(i)
            item.setHidden(text_lower not in item.text().lower())
    
    def _on_color_by_changed(self):
        """Handle color-by selection change."""
        # Delegate to parent dialog
        if hasattr(self.parent_dialog, '_on_color_by_changed'):
            self.parent_dialog._on_color_by_changed()
    
    def _remake_umap(self):
        """Remake UMAP."""
        if hasattr(self.parent_dialog, '_remake_umap'):
            self.parent_dialog._remake_umap()
    
    def _open_heatmap_filter_dialog(self):
        """Open heatmap filter dialog."""
        if hasattr(self.parent_dialog, '_open_heatmap_filter_dialog'):
            self.parent_dialog._open_heatmap_filter_dialog()
    
    def _on_patient_annotation_changed(self, state: int):
        """Handle patient annotation checkbox change."""
        is_checked = (state == 2)
        self.patient_annotation_column_label.setEnabled(is_checked)
        self.patient_annotation_column_combo.setEnabled(is_checked)
        self.patient_annotate_btn.setEnabled(is_checked)
    
    def _open_patient_annotation_dialog(self):
        """Open patient annotation dialog."""
        if hasattr(self.parent_dialog, '_open_patient_annotation_dialog'):
            self.parent_dialog._open_patient_annotation_dialog()
    
    def _open_feature_labels_dialog(self):
        """Open feature labels dialog."""
        if hasattr(self.parent_dialog, '_open_feature_labels_dialog'):
            self.parent_dialog._open_feature_labels_dialog()
    
    def _open_marker_selection_dialog(self):
        """Open marker selection dialog."""
        if hasattr(self.parent_dialog, '_open_marker_selection_dialog'):
            self.parent_dialog._open_marker_selection_dialog()
    
    def _apply_settings(self):
        """Apply settings to parent dialog and refresh plots."""
        if not self.parent_dialog:
            return
        
        # Apply all settings to parent dialog
        # Point style
        if hasattr(self.parent_dialog, 'point_size_spinbox'):
            self.parent_dialog.point_size_spinbox.setValue(self.point_size_spinbox.value())
        if hasattr(self.parent_dialog, 'point_alpha_spinbox'):
            self.parent_dialog.point_alpha_spinbox.setValue(self.point_alpha_spinbox.value())
        
        # Group by
        if hasattr(self.parent_dialog, 'group_by_combo'):
            self.parent_dialog.group_by_combo.setCurrentText(self.group_by_combo.currentText())
        
        # Colormap
        if hasattr(self.parent_dialog, 'colormap_combo'):
            self.parent_dialog.colormap_combo.setCurrentText(self.colormap_combo.currentText())
            if hasattr(self.parent_dialog, '_on_colormap_changed'):
                self.parent_dialog._on_colormap_changed(self.colormap_combo.currentText())
        
        # Heatmap settings
        if hasattr(self.parent_dialog, 'heatmap_source_combo'):
            self.parent_dialog.heatmap_source_combo.setCurrentText(self.heatmap_source_combo.currentText())
            if hasattr(self.parent_dialog, '_on_heatmap_source_changed'):
                self.parent_dialog._on_heatmap_source_changed(self.heatmap_source_combo.currentText())
        if hasattr(self.parent_dialog, 'heatmap_scaling_combo'):
            self.parent_dialog.heatmap_scaling_combo.setCurrentText(self.heatmap_scaling_combo.currentText())
            if hasattr(self.parent_dialog, '_on_heatmap_scaling_changed'):
                self.parent_dialog._on_heatmap_scaling_changed(self.heatmap_scaling_combo.currentText())
        # Feature tick font size
        if hasattr(self.parent_dialog, 'feature_tick_fontsize'):
            self.parent_dialog.feature_tick_fontsize = self.feature_fontsize_spinbox.value()
        
        # Patient annotation
        # Update the enabled state in parent dialog
        is_checked = self.patient_annotation_checkbox.isChecked()
        self.parent_dialog.patient_annotation_enabled = is_checked
        if hasattr(self.parent_dialog, 'patient_annotation_checkbox'):
            self.parent_dialog.patient_annotation_checkbox.setChecked(is_checked)
            if hasattr(self.parent_dialog, '_on_patient_annotation_changed'):
                self.parent_dialog._on_patient_annotation_changed(
                    self.parent_dialog.patient_annotation_checkbox.checkState()
                )
        # Save selected patient annotation column
        if self.patient_annotation_column_combo.count() > 0:
            selected_column = self.patient_annotation_column_combo.currentData()
            if selected_column:
                self.parent_dialog.patient_annotation_column = selected_column
        
        # Legend label
        if hasattr(self.parent_dialog, 'patient_legend_label'):
            self.parent_dialog.patient_legend_label = self.legend_label_edit.text().strip() or 'Patient/Source'
        
        # Top N
        if hasattr(self.parent_dialog, 'top_n_spinbox'):
            self.parent_dialog.top_n_spinbox.setValue(self.top_n_spinbox.value())
            if hasattr(self.parent_dialog, '_on_top_n_changed'):
                self.parent_dialog._on_top_n_changed(self.top_n_spinbox.value())
        
        # Plot type
        if hasattr(self.parent_dialog, 'plot_type_combo'):
            self.parent_dialog.plot_type_combo.setCurrentText(self.plot_type_combo.currentText())
            if hasattr(self.parent_dialog, '_on_plot_type_changed'):
                self.parent_dialog._on_plot_type_changed(self.plot_type_combo.currentText())
        
        # Stats
        if hasattr(self.parent_dialog, 'stats_test_checkbox'):
            self.parent_dialog.stats_test_checkbox.setChecked(self.stats_test_checkbox.isChecked())
            if hasattr(self.parent_dialog, '_on_stats_test_changed'):
                self.parent_dialog._on_stats_test_changed(self.parent_dialog.stats_test_checkbox.checkState())
        if hasattr(self.parent_dialog, 'stats_mode_combo'):
            self.parent_dialog.stats_mode_combo.setCurrentText(self.stats_mode_combo.currentText())
            if hasattr(self.parent_dialog, '_on_stats_mode_changed'):
                self.parent_dialog._on_stats_mode_changed(self.stats_mode_combo.currentText())
        if hasattr(self.parent_dialog, 'stats_cluster_combo'):
            self.parent_dialog.stats_cluster_combo.setCurrentText(self.stats_cluster_combo.currentText())
            if hasattr(self.parent_dialog, '_on_stats_cluster_changed'):
                self.parent_dialog._on_stats_cluster_changed(self.stats_cluster_combo.currentText())
        
        # Color-by - sync selections
        if hasattr(self.parent_dialog, 'color_by_listwidget'):
            # Clear parent selections
            for i in range(self.parent_dialog.color_by_listwidget.count()):
                self.parent_dialog.color_by_listwidget.item(i).setSelected(False)
            # Set selections based on this dialog
            for i in range(self.color_by_listwidget.count()):
                item = self.color_by_listwidget.item(i)
                if item.isSelected():
                    # Find matching item in parent
                    for j in range(self.parent_dialog.color_by_listwidget.count()):
                        parent_item = self.parent_dialog.color_by_listwidget.item(j)
                        if parent_item.text() == item.text():
                            parent_item.setSelected(True)
                            break
            if hasattr(self.parent_dialog, '_on_color_by_changed'):
                self.parent_dialog._on_color_by_changed()
        
        # Refresh the current view in parent dialog
        if hasattr(self.parent_dialog, 'view_combo'):
            current_view = self.parent_dialog.view_combo.currentText()
            if hasattr(self.parent_dialog, '_on_view_changed'):
                self.parent_dialog._on_view_changed(current_view)
        
        self.accept()

