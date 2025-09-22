
from typing import List

import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from openmcd.data.mcd_loader import AcquisitionInfo


class FeatureExtractionDialog(QtWidgets.QDialog):
    """Dialog for configuring feature extraction."""
    
    def __init__(self, parent, acquisitions: List[AcquisitionInfo], segmentation_masks):
        super().__init__(parent)
        self.acquisitions = acquisitions
        self.segmentation_masks = segmentation_masks
        self.setWindowTitle("Feature Extraction")
        self.setModal(True)
        self.resize(600, 500)
        
        self._create_ui()
        self._populate_acquisitions()
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Acquisition selection
        acq_group = QtWidgets.QGroupBox("Select Acquisitions")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        
        self.all_with_masks_chk = QtWidgets.QCheckBox("All acquisitions with segmentation masks")
        self.all_with_masks_chk.setChecked(True)
        self.all_with_masks_chk.toggled.connect(self._on_all_with_masks_toggled)
        acq_layout.addWidget(self.all_with_masks_chk)
        
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.acq_list.setEnabled(False)
        acq_layout.addWidget(self.acq_list)
        
        layout.addWidget(acq_group)
        
        # Feature selection
        feature_group = QtWidgets.QGroupBox("Select Features to Extract")
        feature_layout = QtWidgets.QVBoxLayout(feature_group)
        
        # Morphology features
        morph_group = QtWidgets.QGroupBox("Morphology Features (from mask)")
        morph_layout = QtWidgets.QVBoxLayout(morph_group)
        
        self.morph_features = {
            'area_um2': QtWidgets.QCheckBox("Area (μm²)"),
            'perimeter_um': QtWidgets.QCheckBox("Perimeter (μm)"),
            'equivalent_diameter_um': QtWidgets.QCheckBox("Equivalent diameter (μm)"),
            'eccentricity': QtWidgets.QCheckBox("Eccentricity"),
            'solidity': QtWidgets.QCheckBox("Solidity"),
            'extent': QtWidgets.QCheckBox("Extent"),
            'circularity': QtWidgets.QCheckBox("Circularity (4π·area/perimeter²)"),
            'major_axis_len_um': QtWidgets.QCheckBox("Major axis length (μm)"),
            'minor_axis_len_um': QtWidgets.QCheckBox("Minor axis length (μm)"),
            'aspect_ratio': QtWidgets.QCheckBox("Aspect ratio (major/minor)"),
            'bbox_area_um2': QtWidgets.QCheckBox("Bounding box area (μm²)"),
            'touches_border': QtWidgets.QCheckBox("Touches border (boolean)"),
            'holes_count': QtWidgets.QCheckBox("Number of holes")
        }
        
        # Set all morphology features as checked by default
        for checkbox in self.morph_features.values():
            checkbox.setChecked(True)
            morph_layout.addWidget(checkbox)
        
        feature_layout.addWidget(morph_group)
        
        # Intensity features
        intensity_group = QtWidgets.QGroupBox("Per-channel Intensity Features")
        intensity_layout = QtWidgets.QVBoxLayout(intensity_group)
        
        self.intensity_features = {
            'mean': QtWidgets.QCheckBox("Mean intensity"),
            'median': QtWidgets.QCheckBox("Median intensity"),
            'std': QtWidgets.QCheckBox("Standard deviation"),
            'mad': QtWidgets.QCheckBox("Median absolute deviation"),
            'p10': QtWidgets.QCheckBox("10th percentile"),
            'p90': QtWidgets.QCheckBox("90th percentile"),
            'integrated': QtWidgets.QCheckBox("Integrated intensity (mean·area)"),
            'frac_pos': QtWidgets.QCheckBox("Fraction positive pixels")
        }
        
        # Set all intensity features as checked by default
        for checkbox in self.intensity_features.values():
            checkbox.setChecked(True)
            intensity_layout.addWidget(checkbox)
        
        feature_layout.addWidget(intensity_group)
        layout.addWidget(feature_group)
        
        # Output directory
        output_group = QtWidgets.QGroupBox("Output Settings")
        output_layout = QtWidgets.QVBoxLayout(output_group)
        
        dir_layout = QtWidgets.QHBoxLayout()
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory for CSV file...")
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QtWidgets.QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self._select_output_directory)
        
        dir_layout.addWidget(QtWidgets.QLabel("Output directory:"))
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(self.output_dir_btn)
        output_layout.addLayout(dir_layout)
        
        self.filename_edit = QtWidgets.QLineEdit("cell_features.csv")
        output_layout.addWidget(QtWidgets.QLabel("Filename:"))
        output_layout.addWidget(self.filename_edit)
        
        layout.addWidget(output_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.extract_btn = QtWidgets.QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.extract_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
    
    def _populate_acquisitions(self):
        """Populate the acquisition list with acquisitions that have masks."""
        self.acq_list.clear()
        for acq in self.acquisitions:
            if acq.id in self.segmentation_masks:
                item = QtWidgets.QListWidgetItem(f"{acq.name} (Well: {acq.well})" if acq.well else acq.name)
                item.setData(Qt.UserRole, acq.id)
                self.acq_list.addItem(item)
    
    def _on_all_with_masks_toggled(self):
        """Handle toggle of 'all with masks' checkbox."""
        self.acq_list.setEnabled(not self.all_with_masks_chk.isChecked())
    
    def _select_output_directory(self):
        """Open directory selection dialog."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            "Select Output Directory for Feature CSV",
            "",  # Start from current directory
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.output_dir_edit.setText(directory)
    
    def get_selected_acquisitions(self):
        """Get list of selected acquisition IDs."""
        if self.all_with_masks_chk.isChecked():
            return [acq.id for acq in self.acquisitions if acq.id in self.segmentation_masks]
        else:
            selected_ids = []
            for i in range(self.acq_list.count()):
                item = self.acq_list.item(i)
                if item.isSelected():
                    selected_ids.append(item.data(Qt.UserRole))
            return selected_ids
    
    def get_selected_features(self):
        """Get dictionary of selected features."""
        features = {}
        
        # Morphology features
        for key, checkbox in self.morph_features.items():
            features[key] = checkbox.isChecked()
        
        # Intensity features
        for key, checkbox in self.intensity_features.items():
            features[key] = checkbox.isChecked()
        
        return features
    
    def get_output_path(self):
        """Get the full output path for the CSV file."""
        directory = self.output_dir_edit.text().strip()
        filename = self.filename_edit.text().strip()
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        if directory:
            return os.path.join(directory, filename)
        else:
            return filename
