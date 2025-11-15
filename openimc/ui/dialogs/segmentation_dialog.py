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

from typing import List
import json
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer

# Data types
from openimc.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
import os
from openimc.ui.utils import combine_channels
from openimc.ui.dialogs.preprocessing_dialog import PreprocessingDialog
from openimc.ui.dialogs.ilastik_segmentation_dialog import IlastikSegmentationDialog

# Optional GPU runtime
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

# Optional scikit-image for denoising
try:
    from skimage import morphology, filters
    from skimage.filters import gaussian, median
    from skimage.morphology import disk, footprint_rectangle
    from skimage.restoration import denoise_nl_means, estimate_sigma
    from scipy import ndimage as ndi
    try:
        from skimage.restoration import rolling_ball as _sk_rolling_ball  # type: ignore
        _HAVE_ROLLING_BALL = True
    except Exception:
        _HAVE_ROLLING_BALL = False
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False
    _HAVE_ROLLING_BALL = False

# Optional CellSAM
# Catch both ImportError and OSError (Windows DLL loading errors)
try:
    from cellSAM import get_model, cellsam_pipeline  # type: ignore
    _HAVE_CELLSAM = True
except (ImportError, OSError):
    _HAVE_CELLSAM = False


def _get_user_config_path() -> Path:
    """Get path to user configuration file for storing preferences."""
    # Use user's home directory with .openimc config folder
    home = Path.home()
    config_dir = home / ".openimc"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "user_preferences.json"


def _load_user_preferences() -> dict:
    """Load user preferences from config file."""
    config_path = _get_user_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_user_preferences(prefs: dict):
    """Save user preferences to config file."""
    config_path = _get_user_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(prefs, f, indent=2)
    except IOError:
        # Silently fail if we can't write to config file
        pass



class SegmentationDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Segmentation")
        self.setModal(True)
        
        # Set dialog size to 90% of parent window size
        if parent:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        else:
            self.resize(800, 700)  # Fallback size if no parent
        
        self.setMinimumSize(600, 500)
        self.channels = channels
        # Persist selections per MCD file (by path on parent window)
        self._per_file_channel_prefs = {}
        self.segmentation_result = None
        self.preprocessing_config = None
        
        # Create UI
        self._create_ui()
        
        # Load persisted channel selections for this MCD file
        self._load_persisted_selections()
    
    def _load_persisted_selections(self):
        """Load previously saved channel selections for the current MCD file."""
        mcd_key = getattr(self.parent(), 'current_path', None)
        if not mcd_key:
            return
            
        prefs = self._per_file_channel_prefs.get(mcd_key, {})
        if not prefs:
            return
            
        # Restore preprocessing config if available
        if 'preprocessing_config' in prefs:
            self.preprocessing_config = prefs['preprocessing_config']
            
        # Restore model selection if available
        if 'model' in prefs:
            model_pref = prefs['model']
            # Handle both old values (cyto3, nuclei) and new display text
            model_map = {
                'cyto3': 'Cellpose Cyto3',
                'nuclei': 'Cellpose Nuclei'
            }
            display_model = model_map.get(model_pref, model_pref)
            model_index = self.model_combo.findText(display_model)
            if model_index >= 0:
                self.model_combo.setCurrentIndex(model_index)
        
    def _save_persisted_selections(self):
        """Save current selections for the current MCD file."""
        mcd_key = getattr(self.parent(), 'current_path', None)
        if not mcd_key:
            return
            
        # Save current selections
        self._per_file_channel_prefs[mcd_key] = {
            'model': self.model_combo.currentText(),
            'preprocessing_config': self.preprocessing_config
        }
    
    def accept(self):
        """Override accept to save selections before closing."""
        self._save_persisted_selections()
        # Save API key to user preferences if DeepCell CellSAM is selected and key is provided
        if self.model_combo.currentText() == "DeepCell CellSAM":
            api_key = self.api_key_edit.text().strip()
            if api_key:
                # Save to user preferences
                user_prefs = _load_user_preferences()
                user_prefs["deepcell_api_key"] = api_key
                _save_user_preferences(user_prefs)
                # Also set in environment for current session
                os.environ["DEEPCELL_ACCESS_TOKEN"] = api_key
        super().accept()
        
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
        layout.setSpacing(6)  # Reduced spacing between sections
        
        # Model selection
        model_group = QtWidgets.QGroupBox("Segmentation Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        model_layout.setSpacing(4)
        model_layout.setContentsMargins(8, 8, 8, 8)
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["DeepCell CellSAM", "Cellpose Cyto3", "Cellpose Nuclei", "Ilastik", "Classical Watershed"])
        self.model_combo.setCurrentIndex(0)  # Set DeepCell CellSAM as default
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        
        # Model description
        self.model_desc = QtWidgets.QLabel("DeepCell CellSAM: DeepCell's CellSAM model for cell segmentation. Supports nuclear-only, cyto-only, or combined modes.")
        self.model_desc.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        model_layout.addWidget(self.model_desc)
        
        layout.addWidget(model_group)
        
        # Preprocessing and Denoising section
        preprocess_group = QtWidgets.QGroupBox("Image Preprocessing")
        preprocess_layout = QtWidgets.QVBoxLayout(preprocess_group)
        preprocess_layout.setSpacing(4)
        preprocess_layout.setContentsMargins(8, 8, 8, 8)
        
        # Preprocessing button
        preprocess_btn = QtWidgets.QPushButton("Select Segmentation Channels")
        preprocess_btn.clicked.connect(self._open_preprocessing_dialog)
        preprocess_layout.addWidget(preprocess_btn)
        
        self.preprocess_info_label = QtWidgets.QLabel("No channels selected")
        self.preprocess_info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        self.preprocess_info_label.setWordWrap(True)
        preprocess_layout.addWidget(self.preprocess_info_label)
        
        # Intensity Scaling section
        scaling_group = QtWidgets.QGroupBox("Intensity Scaling")
        scaling_layout = QtWidgets.QVBoxLayout(scaling_group)
        scaling_layout.setSpacing(4)
        scaling_layout.setContentsMargins(8, 8, 8, 8)
        
        norm_method_layout = QtWidgets.QHBoxLayout()
        norm_method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.norm_method_combo = QtWidgets.QComboBox()
        self.norm_method_combo.addItems(["channelwise_minmax", "None", "arcsinh", "percentile_clip"])
        self.norm_method_combo.currentTextChanged.connect(self._on_norm_method_changed)
        norm_method_layout.addWidget(self.norm_method_combo)
        norm_method_layout.addStretch()
        scaling_layout.addLayout(norm_method_layout)
        
        self.arcsinh_frame = QtWidgets.QFrame()
        arcsinh_layout = QtWidgets.QVBoxLayout(self.arcsinh_frame)
        cofactor_row = QtWidgets.QHBoxLayout()
        cofactor_row.addWidget(QtWidgets.QLabel("Cofactor:"))
        self.arcsinh_cofactor_spin = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor_spin.setRange(0.1, 100.0)
        self.arcsinh_cofactor_spin.setValue(10.0)
        self.arcsinh_cofactor_spin.setDecimals(1)
        cofactor_row.addWidget(self.arcsinh_cofactor_spin)
        cofactor_row.addStretch()
        arcsinh_layout.addLayout(cofactor_row)
        
        # Note about arcsinh in segmentation
        arcsinh_note = QtWidgets.QLabel(
            "Note: Arcsinh transform on the raw images can improve segmentation performance\n"
            "but is not saved on images."
        )
        arcsinh_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        arcsinh_note.setWordWrap(True)
        arcsinh_layout.addWidget(arcsinh_note)
        scaling_layout.addWidget(self.arcsinh_frame)
        
        self.percentile_frame = QtWidgets.QFrame()
        percentile_layout = QtWidgets.QHBoxLayout(self.percentile_frame)
        percentile_layout.addWidget(QtWidgets.QLabel("Low percentile:"))
        self.p_low_spin = QtWidgets.QDoubleSpinBox()
        self.p_low_spin.setRange(0.1, 50.0)
        self.p_low_spin.setValue(1.0)
        self.p_low_spin.setDecimals(1)
        percentile_layout.addWidget(self.p_low_spin)
        percentile_layout.addWidget(QtWidgets.QLabel("High percentile:"))
        self.p_high_spin = QtWidgets.QDoubleSpinBox()
        self.p_high_spin.setRange(50.0, 99.9)
        self.p_high_spin.setValue(99.0)
        self.p_high_spin.setDecimals(1)
        percentile_layout.addWidget(self.p_high_spin)
        percentile_layout.addStretch()
        scaling_layout.addWidget(self.percentile_frame)
        
        preprocess_layout.addWidget(scaling_group)
        
        # Denoising options
        denoise_frame = QtWidgets.QFrame()
        denoise_layout = QtWidgets.QVBoxLayout(denoise_frame)
        denoise_layout.setContentsMargins(0, 5, 0, 0)
        denoise_layout.setSpacing(4)
        
        # Denoising source selection
        denoise_source_layout = QtWidgets.QHBoxLayout()
        denoise_source_layout.addWidget(QtWidgets.QLabel("Denoising:"))
        self.denoise_source_combo = QtWidgets.QComboBox()
        self.denoise_source_combo.addItems(["None", "Viewer", "Custom"])
        self.denoise_source_combo.currentTextChanged.connect(self._on_denoise_source_changed)
        denoise_source_layout.addWidget(self.denoise_source_combo)
        denoise_source_layout.addStretch()
        denoise_layout.addLayout(denoise_source_layout)
        
        # Custom denoising frame (hidden by default)
        self.custom_denoise_frame = QtWidgets.QFrame()
        self.custom_denoise_frame.setFrameStyle(QtWidgets.QFrame.Box)
        self.custom_denoise_frame.setVisible(False)
        custom_denoise_layout = QtWidgets.QVBoxLayout(self.custom_denoise_frame)
        custom_denoise_layout.setSpacing(4)
        custom_denoise_layout.setContentsMargins(8, 8, 8, 8)
        
        # Channel selection
        denoise_channel_row = QtWidgets.QHBoxLayout()
        denoise_channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.denoise_channel_combo = QtWidgets.QComboBox()
        self.denoise_channel_combo.currentTextChanged.connect(self._on_denoise_channel_changed)
        denoise_channel_row.addWidget(self.denoise_channel_combo, 1)
        custom_denoise_layout.addLayout(denoise_channel_row)
        
        # Denoising controls in horizontal layout
        denoise_controls_layout = QtWidgets.QHBoxLayout()
        
        # Hot pixel removal
        hot_frame = QtWidgets.QFrame()
        hot_layout = QtWidgets.QVBoxLayout(hot_frame)
        hot_layout.setContentsMargins(0, 0, 0, 0)
        hot_layout.setSpacing(3)
        self.hot_pixel_chk = QtWidgets.QCheckBox("Hot pixel")
        self.hot_pixel_method_combo = QtWidgets.QComboBox()
        self.hot_pixel_method_combo.addItems(["Median 3x3", ">N SD"])
        self.hot_pixel_method_combo.currentTextChanged.connect(self._sync_hot_controls_visibility)
        self.hot_pixel_n_spin = QtWidgets.QDoubleSpinBox()
        self.hot_pixel_n_spin.setRange(0.5, 10.0)
        self.hot_pixel_n_spin.setDecimals(1)
        self.hot_pixel_n_spin.setValue(5.0)
        self.hot_pixel_n_spin.setMaximumWidth(60)
        hot_layout.addWidget(self.hot_pixel_chk)
        hot_layout.addWidget(self.hot_pixel_method_combo)
        hot_n_layout = QtWidgets.QHBoxLayout()
        self.hot_pixel_n_label = QtWidgets.QLabel("N:")
        hot_n_layout.addWidget(self.hot_pixel_n_label)
        hot_n_layout.addWidget(self.hot_pixel_n_spin)
        hot_n_layout.addStretch()
        hot_layout.addLayout(hot_n_layout)
        denoise_controls_layout.addWidget(hot_frame)
        
        # Speckle smoothing
        speckle_frame = QtWidgets.QFrame()
        speckle_layout = QtWidgets.QVBoxLayout(speckle_frame)
        speckle_layout.setContentsMargins(0, 0, 0, 0)
        speckle_layout.setSpacing(3)
        self.speckle_chk = QtWidgets.QCheckBox("Speckle")
        self.speckle_method_combo = QtWidgets.QComboBox()
        self.speckle_method_combo.addItems(["Gaussian", "NL-means"])
        self.gaussian_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.1, 5.0)
        self.gaussian_sigma_spin.setDecimals(2)
        self.gaussian_sigma_spin.setValue(0.8)
        self.gaussian_sigma_spin.setMaximumWidth(60)
        speckle_layout.addWidget(self.speckle_chk)
        speckle_layout.addWidget(self.speckle_method_combo)
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("σ:"))
        sigma_layout.addWidget(self.gaussian_sigma_spin)
        sigma_layout.addStretch()
        speckle_layout.addLayout(sigma_layout)
        denoise_controls_layout.addWidget(speckle_frame)
        
        # Background subtraction
        bg_frame = QtWidgets.QFrame()
        bg_layout = QtWidgets.QVBoxLayout(bg_frame)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(3)
        self.bg_subtract_chk = QtWidgets.QCheckBox("Background")
        self.bg_method_combo = QtWidgets.QComboBox()
        self.bg_method_combo.addItems(["White top-hat", "Black top-hat", "Rolling ball"])
        self.bg_radius_spin = QtWidgets.QSpinBox()
        self.bg_radius_spin.setRange(1, 100)
        self.bg_radius_spin.setValue(15)
        self.bg_radius_spin.setMaximumWidth(60)
        bg_layout.addWidget(self.bg_subtract_chk)
        bg_layout.addWidget(self.bg_method_combo)
        radius_layout = QtWidgets.QHBoxLayout()
        radius_layout.addWidget(QtWidgets.QLabel("R:"))
        radius_layout.addWidget(self.bg_radius_spin)
        radius_layout.addStretch()
        bg_layout.addLayout(radius_layout)
        denoise_controls_layout.addWidget(bg_frame)
        
        custom_denoise_layout.addLayout(denoise_controls_layout)
        
        # Apply to all channels button
        self.apply_all_channels_btn = QtWidgets.QPushButton("Apply to All Channels")
        self.apply_all_channels_btn.clicked.connect(self._apply_denoise_to_all_channels)
        custom_denoise_layout.addWidget(self.apply_all_channels_btn)
        
        # Initialize custom denoising settings storage
        self.custom_denoise_settings = {}
        
        # Disable custom denoising panel if scikit-image is missing
        if not _HAVE_SCIKIT_IMAGE:
            self.custom_denoise_frame.setEnabled(False)
            custom_denoise_layout.addWidget(QtWidgets.QLabel("scikit-image not available; install to enable custom denoising."))
        
        denoise_layout.addWidget(self.custom_denoise_frame)
        preprocess_layout.addWidget(denoise_frame)
        
        # Add preprocessing group to main layout
        layout.addWidget(preprocess_group)
        
        # Parameters (Cellpose-specific)
        self.params_group = QtWidgets.QGroupBox("Segmentation Parameters")
        params_layout = QtWidgets.QVBoxLayout(self.params_group)
        params_layout.setSpacing(4)
        params_layout.setContentsMargins(8, 8, 8, 8)
        
        # Diameter
        diameter_layout = QtWidgets.QHBoxLayout()
        diameter_layout.addWidget(QtWidgets.QLabel("Diameter (pixels):"))
        self.diameter_spinbox = QtWidgets.QSpinBox()
        self.diameter_spinbox.setRange(1, 200)
        self.diameter_spinbox.setValue(10)
        self.diameter_spinbox.setSuffix(" px")
        diameter_layout.addWidget(self.diameter_spinbox)
        
        self.auto_diameter_chk = QtWidgets.QCheckBox("Auto-estimate")
        self.auto_diameter_chk.setChecked(True)
        self.auto_diameter_chk.toggled.connect(self._on_auto_diameter_toggled)
        diameter_layout.addWidget(self.auto_diameter_chk)
        diameter_layout.addStretch()
        
        params_layout.addLayout(diameter_layout)
        
        # Flow threshold
        flow_layout = QtWidgets.QVBoxLayout()
        flow_row = QtWidgets.QHBoxLayout()
        flow_row.addWidget(QtWidgets.QLabel("Flow threshold:"))
        self.flow_spinbox = QtWidgets.QDoubleSpinBox()
        self.flow_spinbox.setRange(0.0, 10.0)
        self.flow_spinbox.setDecimals(2)
        self.flow_spinbox.setValue(0.4)
        self.flow_spinbox.setSingleStep(0.1)
        flow_row.addWidget(self.flow_spinbox)
        flow_row.addStretch()
        flow_layout.addLayout(flow_row)
        flow_note = QtWidgets.QLabel("Enforces consistency of pixel flow direction toward center. Higher values result in fewer, more confident cells but may miss weak ones. Typical range: 0.4-1.0")
        flow_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        flow_note.setWordWrap(True)
        flow_layout.addWidget(flow_note)
        
        params_layout.addLayout(flow_layout)
        
        # Cell probability threshold
        cellprob_layout = QtWidgets.QVBoxLayout()
        cellprob_row = QtWidgets.QHBoxLayout()
        cellprob_row.addWidget(QtWidgets.QLabel("Cell probability threshold:"))
        self.cellprob_spinbox = QtWidgets.QDoubleSpinBox()
        self.cellprob_spinbox.setRange(-6.0, 6.0)
        self.cellprob_spinbox.setDecimals(1)
        self.cellprob_spinbox.setValue(0.0)
        self.cellprob_spinbox.setSingleStep(0.5)
        cellprob_row.addWidget(self.cellprob_spinbox)
        cellprob_row.addStretch()
        cellprob_layout.addLayout(cellprob_row)
        cellprob_note = QtWidgets.QLabel("Minimum probability to consider a pixel as cell. Higher values remove low-probability regions but may split faint cells. Typical range: 0.0-0.2 for cytoplasm models, -2.0-0.0 for nuclei models")
        cellprob_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        cellprob_note.setWordWrap(True)
        cellprob_layout.addWidget(cellprob_note)
        
        params_layout.addLayout(cellprob_layout)
        
        layout.addWidget(self.params_group)
        
        # Watershed-specific parameters (hidden by default)
        self.watershed_group = QtWidgets.QGroupBox("Watershed Parameters")
        watershed_layout = QtWidgets.QVBoxLayout(self.watershed_group)
        watershed_layout.setSpacing(4)
        watershed_layout.setContentsMargins(8, 8, 8, 8)
        
        # Nuclear fusion method
        nuclear_fusion_layout = QtWidgets.QHBoxLayout()
        nuclear_fusion_layout.addWidget(QtWidgets.QLabel("Nuclear fusion method:"))
        self.nuclear_fusion_combo = QtWidgets.QComboBox()
        self.nuclear_fusion_combo.addItems(["mean", "weighted", "pca1"])
        nuclear_fusion_layout.addWidget(self.nuclear_fusion_combo)
        nuclear_fusion_layout.addStretch()
        watershed_layout.addLayout(nuclear_fusion_layout)
        
        # Seed threshold method
        seed_threshold_layout = QtWidgets.QHBoxLayout()
        seed_threshold_layout.addWidget(QtWidgets.QLabel("Seed threshold method:"))
        self.seed_threshold_combo = QtWidgets.QComboBox()
        self.seed_threshold_combo.addItems(["otsu", "percentile"])
        seed_threshold_layout.addWidget(self.seed_threshold_combo)
        seed_threshold_layout.addStretch()
        watershed_layout.addLayout(seed_threshold_layout)
        
        # Min seed area
        min_seed_layout = QtWidgets.QHBoxLayout()
        min_seed_layout.addWidget(QtWidgets.QLabel("Min seed area (pixels):"))
        self.min_seed_area_spin = QtWidgets.QSpinBox()
        self.min_seed_area_spin.setRange(1, 1000)
        self.min_seed_area_spin.setValue(3)  # Much smaller for 10-pixel cells
        self.min_seed_area_spin.setSuffix(" px")
        min_seed_layout.addWidget(self.min_seed_area_spin)
        min_seed_layout.addStretch()
        watershed_layout.addLayout(min_seed_layout)
        
        # Min distance between peaks
        min_distance_layout = QtWidgets.QHBoxLayout()
        min_distance_layout.addWidget(QtWidgets.QLabel("Min distance between peaks:"))
        self.min_distance_peaks_spin = QtWidgets.QSpinBox()
        self.min_distance_peaks_spin.setRange(1, 50)
        self.min_distance_peaks_spin.setValue(3)  # Smaller for closely packed small cells
        self.min_distance_peaks_spin.setSuffix(" px")
        min_distance_layout.addWidget(self.min_distance_peaks_spin)
        min_distance_layout.addStretch()
        watershed_layout.addLayout(min_distance_layout)
        
        # Boundary detection method
        boundary_layout = QtWidgets.QHBoxLayout()
        boundary_layout.addWidget(QtWidgets.QLabel("Boundary detection:"))
        self.boundary_method_combo = QtWidgets.QComboBox()
        self.boundary_method_combo.addItems(["sobel", "scharr", "membrane_channels"])
        boundary_layout.addWidget(self.boundary_method_combo)
        boundary_layout.addStretch()
        watershed_layout.addLayout(boundary_layout)
        
        # Boundary smoothing sigma
        boundary_sigma_layout = QtWidgets.QHBoxLayout()
        boundary_sigma_layout.addWidget(QtWidgets.QLabel("Boundary smoothing σ:"))
        self.boundary_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.boundary_sigma_spin.setRange(0.1, 5.0)
        self.boundary_sigma_spin.setDecimals(2)
        self.boundary_sigma_spin.setValue(0.5)  # Less smoothing for small cells
        self.boundary_sigma_spin.setSingleStep(0.1)
        boundary_sigma_layout.addWidget(self.boundary_sigma_spin)
        boundary_sigma_layout.addStretch()
        watershed_layout.addLayout(boundary_sigma_layout)
        
        # Watershed compactness
        compactness_layout = QtWidgets.QHBoxLayout()
        compactness_layout.addWidget(QtWidgets.QLabel("Watershed compactness:"))
        self.compactness_spin = QtWidgets.QDoubleSpinBox()
        self.compactness_spin.setRange(0.0, 0.1)
        self.compactness_spin.setDecimals(3)
        self.compactness_spin.setValue(0.001)  # Lower compactness for small cells
        self.compactness_spin.setSingleStep(0.001)
        compactness_layout.addWidget(self.compactness_spin)
        compactness_layout.addStretch()
        watershed_layout.addLayout(compactness_layout)
        
        # Min/Max cell area
        cell_area_layout = QtWidgets.QHBoxLayout()
        cell_area_layout.addWidget(QtWidgets.QLabel("Min cell area:"))
        self.min_cell_area_spin = QtWidgets.QSpinBox()
        self.min_cell_area_spin.setRange(1, 10000)
        self.min_cell_area_spin.setValue(20)  # Much smaller for 10-pixel cells
        self.min_cell_area_spin.setSuffix(" px")
        cell_area_layout.addWidget(self.min_cell_area_spin)
        
        cell_area_layout.addWidget(QtWidgets.QLabel("Max cell area:"))
        self.max_cell_area_spin = QtWidgets.QSpinBox()
        self.max_cell_area_spin.setRange(100, 100000)
        self.max_cell_area_spin.setValue(200)  # Much smaller for 10-pixel cells
        self.max_cell_area_spin.setSuffix(" px")
        cell_area_layout.addWidget(self.max_cell_area_spin)
        cell_area_layout.addStretch()
        watershed_layout.addLayout(cell_area_layout)
        
        # Tiling parameters
        tiling_layout = QtWidgets.QHBoxLayout()
        tiling_layout.addWidget(QtWidgets.QLabel("Tile size:"))
        self.tile_size_spin = QtWidgets.QSpinBox()
        self.tile_size_spin.setRange(128, 2048)  # Allow smaller tiles for small ROIs
        self.tile_size_spin.setValue(128)  # Even smaller tiles for small cells and ROIs
        self.tile_size_spin.setSuffix(" px")
        tiling_layout.addWidget(self.tile_size_spin)
        
        tiling_layout.addWidget(QtWidgets.QLabel("Overlap:"))
        self.tile_overlap_spin = QtWidgets.QSpinBox()
        self.tile_overlap_spin.setRange(16, 256)  # Allow smaller overlaps
        self.tile_overlap_spin.setValue(16)  # Smaller overlap for small tiles
        self.tile_overlap_spin.setSuffix(" px")
        tiling_layout.addWidget(self.tile_overlap_spin)
        tiling_layout.addStretch()
        watershed_layout.addLayout(tiling_layout)
        
        # RNG seed for deterministic results
        rng_seed_layout = QtWidgets.QHBoxLayout()
        rng_seed_layout.addWidget(QtWidgets.QLabel("RNG seed:"))
        self.rng_seed_spin = QtWidgets.QSpinBox()
        self.rng_seed_spin.setRange(0, 999999)
        self.rng_seed_spin.setValue(42)
        rng_seed_layout.addWidget(self.rng_seed_spin)
        rng_seed_layout.addStretch()
        watershed_layout.addLayout(rng_seed_layout)
        
        # Initially hide watershed parameters
        self.watershed_group.setVisible(False)
        layout.addWidget(self.watershed_group)
        
        # CellSAM-specific parameters (hidden by default)
        self.cellsam_group = QtWidgets.QGroupBox("DeepCell CellSAM Parameters")
        cellsam_layout = QtWidgets.QVBoxLayout(self.cellsam_group)
        cellsam_layout.setSpacing(4)
        cellsam_layout.setContentsMargins(8, 8, 8, 8)
        
        # API key section
        api_key_layout = QtWidgets.QVBoxLayout()
        api_key_label = QtWidgets.QLabel("DeepCell API Key:")
        api_key_label.setStyleSheet("QLabel { font-weight: bold; }")
        api_key_layout.addWidget(api_key_label)
        
        api_key_info = QtWidgets.QLabel(
            "Get your API key from https://users.deepcell.org/login/\n"
            "Your username is your registration email without the domain suffix.\n"
            "The API key is used to download the most up-to-date model weights."
        )
        api_key_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        api_key_info.setWordWrap(True)
        api_key_layout.addWidget(api_key_info)
        
        api_key_input_layout = QtWidgets.QHBoxLayout()
        self.api_key_edit = QtWidgets.QLineEdit()
        # Check for API key in environment variable first, then user preferences
        existing_key = os.environ.get("DEEPCELL_ACCESS_TOKEN", "")
        key_source = None
        if existing_key:
            key_source = "environment variable"
        else:
            # Try loading from user preferences
            user_prefs = _load_user_preferences()
            existing_key = user_prefs.get("deepcell_api_key", "")
            if existing_key:
                key_source = "saved preferences"
        
        if existing_key:
            self.api_key_edit.setText(existing_key)
            self.api_key_edit.setPlaceholderText(f"API key loaded from {key_source}")
        else:
            self.api_key_edit.setPlaceholderText("Enter DeepCell API key...")
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        api_key_input_layout.addWidget(self.api_key_edit)
        
        self.show_api_key_btn = QtWidgets.QPushButton("Show")
        self.show_api_key_btn.setCheckable(True)
        self.show_api_key_btn.toggled.connect(self._on_show_api_key_toggled)
        api_key_input_layout.addWidget(self.show_api_key_btn)
        api_key_layout.addLayout(api_key_input_layout)
        
        cellsam_layout.addLayout(api_key_layout)
        
        # bbox_threshold
        bbox_layout = QtWidgets.QHBoxLayout()
        bbox_layout.addWidget(QtWidgets.QLabel("Bbox threshold:"))
        self.bbox_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.bbox_threshold_spin.setRange(0.01, 1.0)
        self.bbox_threshold_spin.setDecimals(2)
        self.bbox_threshold_spin.setValue(0.4)
        self.bbox_threshold_spin.setSingleStep(0.05)
        bbox_layout.addWidget(self.bbox_threshold_spin)
        bbox_info = QtWidgets.QLabel("(Lower for faint cells/adjacent cells: 0.1-0.4)")
        bbox_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        bbox_layout.addWidget(bbox_info)
        bbox_layout.addStretch()
        cellsam_layout.addLayout(bbox_layout)
        bbox_note = QtWidgets.QLabel("Note: Setting bbox threshold too low will lead to oversegmentation.")
        bbox_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        cellsam_layout.addWidget(bbox_note)
        
        # use_wsi
        wsi_layout = QtWidgets.QVBoxLayout()
        wsi_check_layout = QtWidgets.QHBoxLayout()
        self.use_wsi_chk = QtWidgets.QCheckBox("Use WSI mode")
        self.use_wsi_chk.setChecked(False)
        wsi_info = QtWidgets.QLabel("(Enable for ROIs with >~3000 cells)")
        wsi_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        wsi_check_layout.addWidget(self.use_wsi_chk)
        wsi_check_layout.addWidget(wsi_info)
        wsi_check_layout.addStretch()
        wsi_layout.addLayout(wsi_check_layout)
        wsi_note = QtWidgets.QLabel("Note: WSI mode tiles the image into multiple pieces for segmentation, which will increase processing time.")
        wsi_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        wsi_note.setWordWrap(True)
        wsi_layout.addWidget(wsi_note)
        cellsam_layout.addLayout(wsi_layout)
        
        # low_contrast_enhancement
        contrast_layout = QtWidgets.QVBoxLayout()
        contrast_check_layout = QtWidgets.QHBoxLayout()
        self.low_contrast_enhancement_chk = QtWidgets.QCheckBox("Low contrast enhancement")
        self.low_contrast_enhancement_chk.setChecked(False)
        contrast_info = QtWidgets.QLabel("(Enable for poor contrast images)")
        contrast_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        contrast_check_layout.addWidget(self.low_contrast_enhancement_chk)
        contrast_check_layout.addWidget(contrast_info)
        contrast_check_layout.addStretch()
        contrast_layout.addLayout(contrast_check_layout)
        contrast_note = QtWidgets.QLabel("Note: Using both denoising (from Image Preprocessing) and low contrast enhancement together can sometimes lead to poorer segmentation masks.")
        contrast_note.setStyleSheet("QLabel { color: #d97706; font-size: 9pt; font-style: italic; }")
        contrast_note.setWordWrap(True)
        contrast_layout.addWidget(contrast_note)
        cellsam_layout.addLayout(contrast_layout)
        
        # gauge_cell_size
        gauge_layout = QtWidgets.QHBoxLayout()
        self.gauge_cell_size_chk = QtWidgets.QCheckBox("Gauge cell size")
        self.gauge_cell_size_chk.setChecked(False)
        gauge_info = QtWidgets.QLabel("(Runs twice: estimates error, then returns mask)")
        gauge_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        gauge_layout.addWidget(self.gauge_cell_size_chk)
        gauge_layout.addWidget(gauge_info)
        gauge_layout.addStretch()
        cellsam_layout.addLayout(gauge_layout)
        
        # GPU usage note
        gpu_note = QtWidgets.QLabel("Note: DeepCell CellSAM automatically uses CUDA (GPU) if available and falls back to CPU if not. GPU selection is handled internally by CellSAM and cannot be controlled from this interface.")
        gpu_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        gpu_note.setWordWrap(True)
        cellsam_layout.addWidget(gpu_note)
        
        # Initially hide CellSAM parameters
        self.cellsam_group.setVisible(False)
        layout.addWidget(self.cellsam_group)
        
        # GPU selection (Cellpose-specific)
        self.gpu_group = QtWidgets.QGroupBox("GPU Acceleration")
        gpu_layout = QtWidgets.QVBoxLayout(self.gpu_group)
        gpu_layout.setSpacing(4)
        gpu_layout.setContentsMargins(8, 8, 8, 8)
        
        gpu_row = QtWidgets.QHBoxLayout()
        gpu_row.addWidget(QtWidgets.QLabel("Device:"))
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.addItem("Auto-detect", "auto")
        self.gpu_combo.addItem("CPU only", None)
        gpu_row.addWidget(self.gpu_combo)
        gpu_row.addStretch()
        gpu_layout.addLayout(gpu_row)
        
        self.gpu_info_label = QtWidgets.QLabel("")
        self.gpu_info_label.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        self.gpu_info_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_info_label)
        
        layout.addWidget(self.gpu_group)
        
        # Options
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        options_layout.setSpacing(4)
        options_layout.setContentsMargins(8, 8, 8, 8)
        
        self.show_overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay")
        self.show_overlay_chk.setChecked(True)
        options_layout.addWidget(self.show_overlay_chk)
        
        self.save_masks_chk = QtWidgets.QCheckBox("Save segmentation masks")
        self.save_masks_chk.setChecked(False)
        self.save_masks_chk.toggled.connect(self._on_save_masks_toggled)
        options_layout.addWidget(self.save_masks_chk)
        
        # Directory selection for saving masks
        self.masks_dir_layout = QtWidgets.QHBoxLayout()
        self.masks_dir_label = QtWidgets.QLabel("Save directory:")
        self.masks_dir_edit = QtWidgets.QLineEdit()
        self.masks_dir_edit.setPlaceholderText("Select directory for saving masks...")
        self.masks_dir_edit.setReadOnly(True)
        self.masks_dir_btn = QtWidgets.QPushButton("Browse...")
        self.masks_dir_btn.clicked.connect(self._select_masks_directory)
        
        self.masks_dir_layout.addWidget(self.masks_dir_label)
        self.masks_dir_layout.addWidget(self.masks_dir_edit)
        self.masks_dir_layout.addWidget(self.masks_dir_btn)
        
        self.masks_dir_frame = QtWidgets.QFrame()
        self.masks_dir_frame.setLayout(self.masks_dir_layout)
        self.masks_dir_frame.setVisible(False)
        options_layout.addWidget(self.masks_dir_frame)
        
        self.segment_all_chk = QtWidgets.QCheckBox("Segment all acquisitions in .mcd file")
        self.segment_all_chk.setChecked(False)
        self.segment_all_chk.toggled.connect(self._on_segment_all_toggled)
        options_layout.addWidget(self.segment_all_chk)
        
        # Warning for batch segmentation
        self.batch_warning = QtWidgets.QLabel("⚠️ For batch segmentation, consider enabling 'Save segmentation masks' to preserve results")
        self.batch_warning.setStyleSheet("QLabel { color: #d97706; font-size: 9pt; font-weight: bold; }")
        self.batch_warning.setWordWrap(True)
        self.batch_warning.setVisible(False)
        options_layout.addWidget(self.batch_warning)
        
        # Info label for segment all
        self.segment_all_info = QtWidgets.QLabel("")
        self.segment_all_info.setStyleSheet("QLabel { color: #666; font-size: 9pt; }")
        self.segment_all_info.setWordWrap(True)
        options_layout.addWidget(self.segment_all_info)
        
        layout.addWidget(options_group)
        
        # Set scroll content widget
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area, 1)  # Stretch factor 1 to take available space
        
        # Buttons (outside scroll area)
        button_layout = QtWidgets.QHBoxLayout()
        self.segment_btn = QtWidgets.QPushButton("Run Segmentation")
        self.segment_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; } QPushButton:hover { background-color: #218838; } QPushButton:pressed { background-color: #1e7e34; }")
        self.segment_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; } QPushButton:hover { background-color: #c82333; } QPushButton:pressed { background-color: #bd2130; }")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.segment_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
        
        # Initialize
        self._on_model_changed()
        self._on_auto_diameter_toggled()
        self._detect_and_populate_gpus()
        self.gpu_combo.currentTextChanged.connect(self._on_gpu_selection_changed)
        self._on_segment_all_toggled()
        
        # Initialize denoising
        self._populate_denoise_channel_list()
        self._on_denoise_source_changed()
        self._sync_hot_controls_visibility()
        
        # Initialize normalization
        self._on_norm_method_changed()
        
    def _on_model_changed(self):
        """Update UI when model selection changes."""
        model = self.model_combo.currentText()
        if model == "Cellpose Nuclei":
            self.model_desc.setText("Nuclei: Segments cell nuclei using nuclear channel")
            # When using nuclei model, hide cytoplasm selection UI in preprocessing
            # (actual hiding is applied when dialog is opened)
            self.params_group.setVisible(True)
            self.gpu_group.setVisible(True)
            self.watershed_group.setVisible(False)
            self.cellsam_group.setVisible(False)
        elif model == "Classical Watershed":
            self.model_desc.setText("Classical Watershed: Marker-controlled watershed with nucleus-seeded, membrane-guided segmentation")
            self.params_group.setVisible(False)
            self.gpu_group.setVisible(False)
            self.watershed_group.setVisible(True)
            self.cellsam_group.setVisible(False)
        elif model == "Ilastik":
            self.model_desc.setText("Ilastik: Load and run inference with a trained Ilastik model (.ilp project file)")
            self.params_group.setVisible(False)
            self.gpu_group.setVisible(False)
            self.watershed_group.setVisible(False)
            self.cellsam_group.setVisible(False)
        elif model == "DeepCell CellSAM":
            self.model_desc.setText("DeepCell CellSAM: DeepCell's CellSAM model for cell segmentation. Supports nuclear-only, cyto-only, or combined modes.")
            self.params_group.setVisible(False)
            self.gpu_group.setVisible(False)
            self.watershed_group.setVisible(False)
            self.cellsam_group.setVisible(True)
        else:  # Cellpose Cyto3
            self.model_desc.setText("Cytoplasm: Segments whole cells using cytoplasm + nuclear channels")
            self.params_group.setVisible(True)
            self.gpu_group.setVisible(True)
            self.watershed_group.setVisible(False)
            self.cellsam_group.setVisible(False)
    
    def _on_auto_diameter_toggled(self):
        """Enable/disable diameter spinbox based on auto-estimate checkbox."""
        self.diameter_spinbox.setEnabled(not self.auto_diameter_chk.isChecked())
    
    def _on_show_api_key_toggled(self, checked):
        """Toggle API key visibility."""
        if checked:
            self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.show_api_key_btn.setText("Hide")
        else:
            self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
            self.show_api_key_btn.setText("Show")
    
    def get_model(self):
        """Get selected model (returns original values for compatibility)."""
        model = self.model_combo.currentText()
        # Map display text back to original values for compatibility
        if model == "Cellpose Cyto3":
            return "cyto3"
        elif model == "Cellpose Nuclei":
            return "nuclei"
        elif model == "DeepCell CellSAM":
            return "DeepCell CellSAM"  # Keep full name for main_window processing
        else:
            return model
    
    
    def get_diameter(self):
        """Get diameter value (None if auto-estimate)."""
        if self.auto_diameter_chk.isChecked():
            return None
        return self.diameter_spinbox.value()
    
    def get_flow_threshold(self):
        """Get flow threshold."""
        return self.flow_spinbox.value()
    
    def get_cellprob_threshold(self):
        """Get cell probability threshold."""
        return self.cellprob_spinbox.value()
    
    def get_show_overlay(self):
        """Get whether to show overlay."""
        return self.show_overlay_chk.isChecked()
    
    def get_save_masks(self):
        """Get whether to save masks."""
        return self.save_masks_chk.isChecked()
    
    def _detect_and_populate_gpus(self):
        """Detect available GPUs and populate the combo box."""
        if not _HAVE_TORCH:
            self.gpu_info_label.setText("PyTorch not available. Using CPU only.")
            return
        
        try:
            available_gpus = []
            
            # Check CUDA
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    available_gpus.append({
                        'id': i,
                        'name': f"{gpu_name} ({gpu_memory:.1f} GB)",
                        'type': 'CUDA'
                    })
            
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_gpus.append({
                    'id': 'mps',
                    'name': 'Apple Metal Performance Shaders (MPS)',
                    'type': 'MPS'
                })
            
            # Add GPU options to combo
            for gpu in available_gpus:
                self.gpu_combo.addItem(gpu['name'], gpu['id'])
            
            # Update info
            if available_gpus:
                self.gpu_info_label.setText(f"Found {len(available_gpus)} GPU(s) available for acceleration.")
            else:
                self.gpu_info_label.setText("No GPUs detected. Using CPU only.")
                
        except Exception as e:
            self.gpu_info_label.setText(f"Error detecting GPUs: {str(e)}")
    
    def _on_gpu_selection_changed(self):
        """Update info when GPU selection changes."""
        gpu_id = self.gpu_combo.currentData()
        
        if gpu_id is None:
            self.gpu_info_label.setText("Using CPU for segmentation. This will be slower but more compatible.")
        elif gpu_id == "auto":
            self.gpu_info_label.setText("Will automatically select the best available GPU.")
        else:
            gpu_name = self.gpu_combo.currentText()
            self.gpu_info_label.setText(f"Selected: {gpu_name}")
    
    def get_selected_gpu(self):
        """Get the selected GPU ID."""
        return self.gpu_combo.currentData()
    
    def _open_preprocessing_dialog(self):
        """Open the preprocessing configuration dialog."""
        dlg = PreprocessingDialog(self.channels, self)
        # Apply persisted selections for current MCD file if available
        mcd_key = getattr(self.parent(), 'current_path', None)
        prefs = self._per_file_channel_prefs.get(mcd_key, {}) if mcd_key else {}
        if 'nuclear_channels' in prefs:
            dlg.set_nuclear_channels(prefs['nuclear_channels'])
        if 'cyto_channels' in prefs:
            dlg.set_cyto_channels(prefs['cyto_channels'])
        # Hide cytoplasm section when using nuclei-only models
        # DeepCell CellSAM supports both nuclear and cyto, so show both sections
        is_nuclei_only = self.model_combo.currentText() == 'Cellpose Nuclei'
        is_cellsam = self.model_combo.currentText() == 'DeepCell CellSAM'
        # Show cytoplasm section if not nuclei-only OR if DeepCell CellSAM (which supports both)
        dlg.set_cytoplasm_visible(not is_nuclei_only or is_cellsam)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.preprocessing_config = {
                'normalization_method': self.get_normalization_method(),
                'arcsinh_cofactor': self.get_arcsinh_cofactor(),
                'percentile_params': self.get_percentile_params(),
                'nuclear_channels': dlg.get_nuclear_channels(),
                'cyto_channels': dlg.get_cyto_channels(),
                'nuclear_combo_method': dlg.get_nuclear_combo_method(),
                'cyto_combo_method': dlg.get_cyto_combo_method(),
                'nuclear_weights': dlg.get_nuclear_weights(),
                'cyto_weights': dlg.get_cyto_weights()
            }
            # Persist per-file preferences after successful configuration
            if mcd_key:
                self._per_file_channel_prefs[mcd_key] = {
                    'nuclear_channels': self.preprocessing_config.get('nuclear_channels', []),
                    'cyto_channels': self.preprocessing_config.get('cyto_channels', [])
                }
            self._update_preprocess_info()
    
    def _update_preprocess_info(self):
        """Update the preprocessing info label."""
        if not self.preprocessing_config:
            self.preprocess_info_label.setText("No channels selected")
            return
        
        config = self.preprocessing_config
        info_parts = []
        
        # Channel combination info
        if config.get('nuclear_channels'):
            nuclear_info = f"Nuclear: {config['nuclear_combo_method']}({len(config['nuclear_channels'])} channels)"
            info_parts.append(nuclear_info)
        
        if config.get('cyto_channels'):
            cyto_info = f"Cytoplasm: {config['cyto_combo_method']}({len(config['cyto_channels'])} channels)"
            info_parts.append(cyto_info)
        
        if info_parts:
            self.preprocess_info_label.setText(" | ".join(info_parts))
        else:
            self.preprocess_info_label.setText("No channels selected")
    
    def get_preprocessing_config(self):
        """Get the preprocessing configuration."""
        # Ensure normalization values are included from main dialog
        if self.preprocessing_config is None:
            self.preprocessing_config = {}
        self.preprocessing_config['normalization_method'] = self.get_normalization_method()
        self.preprocessing_config['arcsinh_cofactor'] = self.get_arcsinh_cofactor()
        self.preprocessing_config['percentile_params'] = self.get_percentile_params()
        return self.preprocessing_config
    
    def _on_norm_method_changed(self):
        """Handle changes to normalization method."""
        method = self.norm_method_combo.currentText()
        self.arcsinh_frame.setVisible(method == "arcsinh")
        self.percentile_frame.setVisible(method == "percentile_clip")
    
    def get_normalization_method(self) -> str:
        """Get the selected normalization method."""
        return self.norm_method_combo.currentText()
    
    def get_arcsinh_cofactor(self) -> float:
        """Get the arcsinh cofactor value."""
        return self.arcsinh_cofactor_spin.value()
    
    def get_percentile_params(self):
        """Get the percentile parameters."""
        return (self.p_low_spin.value(), self.p_high_spin.value())

    def set_use_viewer_denoising(self, enabled: bool):
        """Initialize the 'use viewer denoising' toggle state."""
        if enabled:
            self.denoise_source_combo.setCurrentText("Viewer")
        else:
            self.denoise_source_combo.setCurrentText("None")

    def get_use_viewer_denoising(self) -> bool:
        """Return whether to use viewer denoising during segmentation."""
        return self.denoise_source_combo.currentText() == "Viewer"
    
    def _on_segment_all_toggled(self):
        """Update UI when segment all checkbox is toggled."""
        if self.segment_all_chk.isChecked():
            # Get acquisition count from parent (MainWindow)
            parent = self.parent()
            if hasattr(parent, 'acquisitions'):
                acq_count = len(parent.acquisitions)
                self.segment_all_info.setText(f"Will segment all {acq_count} acquisitions in the .mcd file. This may take a while.")
            else:
                self.segment_all_info.setText("Will segment all acquisitions in the .mcd file. This may take a while.")
            
            # Show warning about saving masks
            self.batch_warning.setVisible(True)
        else:
            self.segment_all_info.setText("")
            self.batch_warning.setVisible(False)
    
    def get_segment_all(self):
        """Get whether to segment all acquisitions."""
        return self.segment_all_chk.isChecked()
    
    def _on_save_masks_toggled(self):
        """Update UI when save masks checkbox is toggled."""
        self.masks_dir_frame.setVisible(self.save_masks_chk.isChecked())
    
    def _select_masks_directory(self):
        """Open directory selection dialog for saving masks."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            "Select Directory for Saving Segmentation Masks",
            "",  # Start from current directory
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if directory:
            self.masks_dir_edit.setText(directory)
    
    def get_masks_directory(self):
        """Get the selected directory for saving masks."""
        if self.save_masks_chk.isChecked():
            directory = self.masks_dir_edit.text().strip()
            if directory and os.path.exists(directory):
                return directory
            else:
                # Fallback to .mcd file directory
                parent = self.parent()
                if hasattr(parent, 'current_path') and parent.current_path:
                    return os.path.dirname(parent.current_path)
        return None
    
    # ---------- Denoising Methods ----------
    def _populate_denoise_channel_list(self):
        """Populate the denoise channel combo with available channels."""
        self.denoise_channel_combo.blockSignals(True)
        self.denoise_channel_combo.clear()
        for ch in self.channels:
            self.denoise_channel_combo.addItem(ch)
        self.denoise_channel_combo.blockSignals(False)
        if self.channels:
            self.denoise_channel_combo.setCurrentIndex(0)
            self._load_denoise_settings()
    
    def _on_denoise_source_changed(self):
        """Handle changes to the denoising source selection."""
        source = self.denoise_source_combo.currentText()
        use_custom = source == "Custom"
        self.custom_denoise_frame.setVisible(use_custom)
        
        # Adjust dialog size when custom denoising is shown/hidden
        QTimer.singleShot(10, lambda: self._adjust_dialog_size())
    
    def _adjust_dialog_size(self):
        """Adjust dialog size based on whether custom denoising is visible."""
        if self.parent():
            parent_size = self.parent().size()
            use_custom = self.custom_denoise_frame.isVisible()
            if use_custom:
                # Use 90% of parent size for custom denoising
                dialog_width = int(parent_size.width() * 0.9)
                dialog_height = int(parent_size.height() * 0.9)
            else:
                # Use 80% of parent size for basic view
                dialog_width = int(parent_size.width() * 0.8)
                dialog_height = int(parent_size.height() * 0.8)
            self.resize(dialog_width, dialog_height)
    
    def _on_denoise_channel_changed(self):
        """Handle changes to the denoise channel selection."""
        self._load_denoise_settings()
    
    def _load_denoise_settings(self):
        """Load saved denoise settings for the currently selected denoise channel into the UI."""
        ch = self.denoise_channel_combo.currentText()
        if not ch:
            return
        cfg = self.custom_denoise_settings.get(ch, {})
        hot = cfg.get("hot")
        speckle = cfg.get("speckle")
        bg = cfg.get("background")
        
        # Block signals during UI update
        self.hot_pixel_chk.blockSignals(True)
        self.hot_pixel_method_combo.blockSignals(True)
        self.hot_pixel_n_spin.blockSignals(True)
        self.speckle_chk.blockSignals(True)
        self.speckle_method_combo.blockSignals(True)
        self.gaussian_sigma_spin.blockSignals(True)
        self.bg_subtract_chk.blockSignals(True)
        self.bg_method_combo.blockSignals(True)
        self.bg_radius_spin.blockSignals(True)
        
        try:
            if hot:
                self.hot_pixel_chk.setChecked(True)
                self.hot_pixel_method_combo.setCurrentIndex(0 if hot.get("method") == "median3" else 1)
                self.hot_pixel_n_spin.setValue(float(hot.get("n_sd", 5.0)))
            else:
                self.hot_pixel_chk.setChecked(False)
                self.hot_pixel_method_combo.setCurrentIndex(0)
                self.hot_pixel_n_spin.setValue(5.0)
                
            if speckle:
                self.speckle_chk.setChecked(True)
                self.speckle_method_combo.setCurrentIndex(0 if speckle.get("method") == "gaussian" else 1)
                self.gaussian_sigma_spin.setValue(float(speckle.get("sigma", 0.8)))
            else:
                self.speckle_chk.setChecked(False)
                self.speckle_method_combo.setCurrentIndex(0)
                self.gaussian_sigma_spin.setValue(0.8)
                
            if bg:
                self.bg_subtract_chk.setChecked(True)
                # 0 white_tophat, 1 black_tophat, 2 rolling_ball
                method = bg.get("method")
                if method == "white_tophat":
                    self.bg_method_combo.setCurrentIndex(0)
                elif method == "black_tophat":
                    self.bg_method_combo.setCurrentIndex(1)
                else:
                    self.bg_method_combo.setCurrentIndex(2)
                self.bg_radius_spin.setValue(int(bg.get("radius", 15)))
            else:
                self.bg_subtract_chk.setChecked(False)
                self.bg_method_combo.setCurrentIndex(0)
                self.bg_radius_spin.setValue(15)
        finally:
            # Unblock signals
            self.hot_pixel_chk.blockSignals(False)
            self.hot_pixel_method_combo.blockSignals(False)
            self.hot_pixel_n_spin.blockSignals(False)
            self.speckle_chk.blockSignals(False)
            self.speckle_method_combo.blockSignals(False)
            self.gaussian_sigma_spin.blockSignals(False)
            self.bg_subtract_chk.blockSignals(False)
            self.bg_method_combo.blockSignals(False)
            self.bg_radius_spin.blockSignals(False)
        
        self._sync_hot_controls_visibility()
    
    def _apply_denoise_to_all_channels(self):
        """Apply current denoising parameters to all channels."""
        try:
            # Build config from current UI settings
            cfg_hot = None
            if self.hot_pixel_chk.isChecked():
                cfg_hot = {
                    "method": "median3" if self.hot_pixel_method_combo.currentIndex() == 0 else "n_sd_local_median",
                    "n_sd": float(self.hot_pixel_n_spin.value()),
                }

            cfg_speckle = None
            if self.speckle_chk.isChecked():
                cfg_speckle = {
                    "method": "gaussian" if self.speckle_method_combo.currentIndex() == 0 else "nl_means",
                    "sigma": float(self.gaussian_sigma_spin.value()),
                }

            cfg_bg = None
            if self.bg_subtract_chk.isChecked():
                bg_idx = self.bg_method_combo.currentIndex()
                if bg_idx == 0:
                    bg_method = "white_tophat"
                elif bg_idx == 1:
                    bg_method = "black_tophat"
                else:
                    bg_method = "rolling_ball"
                cfg_bg = {
                    "method": bg_method,
                    "radius": int(self.bg_radius_spin.value()),
                }

            # Apply the same configuration to all channels
            for channel in self.channels:
                self.custom_denoise_settings.setdefault(channel, {})
                self.custom_denoise_settings[channel]["hot"] = cfg_hot
                self.custom_denoise_settings[channel]["speckle"] = cfg_speckle
                self.custom_denoise_settings[channel]["background"] = cfg_bg
            
            # Show visual confirmation
            self.apply_all_channels_btn.setText("✓ Applied to All Channels")
            self.apply_all_channels_btn.setStyleSheet("QPushButton { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }")
            
            # Reset button appearance after 2 seconds
            QTimer.singleShot(2000, self._reset_apply_all_button)
            
        except Exception as e:
            # Silently handle any errors to avoid disrupting the UI
            pass
    
    def _reset_apply_all_button(self):
        """Reset the apply all channels button to its original appearance."""
        self.apply_all_channels_btn.setText("Apply to All Channels")
        self.apply_all_channels_btn.setStyleSheet("")
    
    def _sync_hot_controls_visibility(self):
        """Show N only for '>N SD above local median' method."""
        is_threshold = self.hot_pixel_method_combo.currentIndex() == 1
        self.hot_pixel_n_spin.setVisible(is_threshold)
        self.hot_pixel_n_label.setVisible(is_threshold)
    
    def get_denoise_source(self):
        """Get the selected denoising source."""
        source = self.denoise_source_combo.currentText()
        if source == "Viewer":
            return "viewer"
        elif source == "Custom":
            return "custom"
        else:
            return "none"
    
    def get_custom_denoise_settings(self):
        """Get the custom denoising settings."""
        return self.custom_denoise_settings
    
    # Watershed parameter getters
    def get_nuclear_fusion_method(self):
        """Get nuclear fusion method."""
        return self.nuclear_fusion_combo.currentText()
    
    def get_seed_threshold_method(self):
        """Get seed threshold method."""
        return self.seed_threshold_combo.currentText()
    
    def get_min_seed_area(self):
        """Get minimum seed area in pixels."""
        return self.min_seed_area_spin.value()
    
    def get_min_distance_peaks(self):
        """Get minimum distance between peaks in pixels."""
        return self.min_distance_peaks_spin.value()
    
    def get_boundary_method(self):
        """Get boundary detection method."""
        return self.boundary_method_combo.currentText()
    
    def get_boundary_sigma(self):
        """Get boundary smoothing sigma."""
        return self.boundary_sigma_spin.value()
    
    def get_compactness(self):
        """Get watershed compactness parameter."""
        return self.compactness_spin.value()
    
    def get_min_cell_area(self):
        """Get minimum cell area in pixels."""
        return self.min_cell_area_spin.value()
    
    def get_max_cell_area(self):
        """Get maximum cell area in pixels."""
        return self.max_cell_area_spin.value()
    
    def get_tile_size(self):
        """Get tile size for tiling."""
        return self.tile_size_spin.value()
    
    def get_tile_overlap(self):
        """Get tile overlap in pixels."""
        return self.tile_overlap_spin.value()
    
    def get_rng_seed(self):
        """Get RNG seed for deterministic results."""
        return self.rng_seed_spin.value()
    
    def get_membrane_fusion_method(self):
        """Get membrane fusion method (same as nuclear for now)."""
        return self.nuclear_fusion_combo.currentText()
    
    # CellSAM parameter getters
    def get_cellsam_api_key(self):
        """Get the DeepCell API key from the field, environment variable, or saved preferences."""
        api_key = self.api_key_edit.text().strip()
        # If field is empty, check environment variable
        if not api_key:
            api_key = os.environ.get("DEEPCELL_ACCESS_TOKEN", "")
        # If still empty, check saved preferences
        if not api_key:
            user_prefs = _load_user_preferences()
            api_key = user_prefs.get("deepcell_api_key", "")
        return api_key
    
    def get_cellsam_bbox_threshold(self):
        """Get bbox threshold for CellSAM."""
        return self.bbox_threshold_spin.value()
    
    def get_cellsam_use_wsi(self):
        """Get use_wsi flag for CellSAM."""
        return self.use_wsi_chk.isChecked()
    
    def get_cellsam_low_contrast_enhancement(self):
        """Get low_contrast_enhancement flag for CellSAM."""
        return self.low_contrast_enhancement_chk.isChecked()
    
    def get_cellsam_gauge_cell_size(self):
        """Get gauge_cell_size flag for CellSAM."""
        return self.gauge_cell_size_chk.isChecked()