
from typing import List

import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer

from openimc.data.mcd_loader import AcquisitionInfo

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


class FeatureExtractionDialog(QtWidgets.QDialog):
    """Dialog for configuring feature extraction."""
    
    def __init__(self, parent, acquisitions: List[AcquisitionInfo], segmentation_masks):
        super().__init__(parent)
        self.acquisitions = acquisitions
        self.segmentation_masks = segmentation_masks
        self.setWindowTitle("Feature Extraction")
        self.setModal(True)
        self.resize(1100, 800)
        
        self._create_ui()
        self._populate_acquisitions()
        
        # Initialize denoising
        self._populate_denoise_channel_list()
        self._on_denoise_source_changed()
        self._sync_hot_controls_visibility()
        
        # Initialize spillover correction
        self._spillover_matrix = None
        self._spillover_method = None
        
        # Initialize channel exclusion list
        self._populate_channel_exclusion_list()
    
    def _create_ui(self):
        """Create the user interface."""
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # Create scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # Create scrollable content widget
        scroll_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(scroll_content)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Top row: Acquisition selection and Output settings side by side
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        
        # Acquisition selection (left side)
        acq_group = QtWidgets.QGroupBox("Select Acquisitions")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        acq_layout.setContentsMargins(8, 8, 8, 8)
        acq_layout.setSpacing(4)
        
        self.all_with_masks_chk = QtWidgets.QCheckBox("All acquisitions with segmentation masks")
        self.all_with_masks_chk.setChecked(True)
        self.all_with_masks_chk.toggled.connect(self._on_all_with_masks_toggled)
        acq_layout.addWidget(self.all_with_masks_chk)
        
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.acq_list.setEnabled(False)
        self.acq_list.setMaximumHeight(80)  # Reduced height
        acq_layout.addWidget(self.acq_list)
        
        top_row.addWidget(acq_group, 1)
        
        # Output settings (right side)
        output_group = QtWidgets.QGroupBox("Output Settings")
        output_layout = QtWidgets.QVBoxLayout(output_group)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setSpacing(4)
        
        dir_layout = QtWidgets.QHBoxLayout()
        dir_layout.setSpacing(4)
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QtWidgets.QPushButton("Browse...")
        self.output_dir_btn.clicked.connect(self._select_output_directory)
        
        dir_layout.addWidget(QtWidgets.QLabel("Directory:"))
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(self.output_dir_btn)
        output_layout.addLayout(dir_layout)
        
        filename_layout = QtWidgets.QHBoxLayout()
        filename_layout.setSpacing(4)
        filename_layout.addWidget(QtWidgets.QLabel("Filename:"))
        self.filename_edit = QtWidgets.QLineEdit("cell_features.csv")
        filename_layout.addWidget(self.filename_edit)
        output_layout.addLayout(filename_layout)
        
        top_row.addWidget(output_group, 1)
        layout.addLayout(top_row)
        
        # Middle section: Preprocessing options (separate boxes)
        preprocess_layout = QtWidgets.QHBoxLayout()
        preprocess_layout.setSpacing(6)
        
        # Denoising (left box - happens first)
        denoise_group = QtWidgets.QGroupBox("1. Denoising")
        denoise_layout = QtWidgets.QVBoxLayout(denoise_group)
        denoise_layout.setContentsMargins(8, 8, 8, 8)
        denoise_layout.setSpacing(4)
        
        denoise_source_layout = QtWidgets.QHBoxLayout()
        denoise_source_layout.setSpacing(4)
        denoise_source_layout.addWidget(QtWidgets.QLabel("Source:"))
        self.denoise_source_combo = QtWidgets.QComboBox()
        self.denoise_source_combo.addItems(["None", "Viewer", "Custom"])  # Remove ambiguous 'Segmentation' source
        self.denoise_source_combo.currentTextChanged.connect(self._on_denoise_source_changed)
        denoise_source_layout.addWidget(self.denoise_source_combo)
        denoise_source_layout.addStretch()
        denoise_layout.addLayout(denoise_source_layout)
        
        preprocess_layout.addWidget(denoise_group)
        
        # Spillover correction (middle box)
        spillover_group = QtWidgets.QGroupBox("2. Spillover Correction")
        spillover_layout = QtWidgets.QVBoxLayout(spillover_group)
        spillover_layout.setContentsMargins(8, 8, 8, 8)
        spillover_layout.setSpacing(4)
        
        self.spillover_chk = QtWidgets.QCheckBox("Enable spillover correction")
        self.spillover_chk.setChecked(False)
        self.spillover_chk.setToolTip("Apply spillover correction to extracted intensity features after feature extraction but before arcsinh scaling")
        self.spillover_chk.toggled.connect(self._on_spillover_toggled)
        spillover_layout.addWidget(self.spillover_chk)
        
        spillover_file_layout = QtWidgets.QHBoxLayout()
        spillover_file_layout.setSpacing(4)
        spillover_file_layout.addWidget(QtWidgets.QLabel("Matrix:"))
        self.spillover_file_edit = QtWidgets.QLineEdit()
        self.spillover_file_edit.setPlaceholderText("Select spillover matrix CSV...")
        self.spillover_file_edit.setReadOnly(True)
        self.spillover_file_edit.setEnabled(False)
        self.spillover_file_btn = QtWidgets.QPushButton("Browse...")
        self.spillover_file_btn.setEnabled(False)
        self.spillover_file_btn.clicked.connect(self._select_spillover_file)
        spillover_file_layout.addWidget(self.spillover_file_edit)
        spillover_file_layout.addWidget(self.spillover_file_btn)
        spillover_layout.addLayout(spillover_file_layout)
        
        preprocess_layout.addWidget(spillover_group)
        
        # Normalization (right box - arcsinh applied after feature extraction)
        norm_group = QtWidgets.QGroupBox("3. Arcsinh Scaling")
        norm_layout = QtWidgets.QVBoxLayout(norm_group)
        norm_layout.setContentsMargins(8, 8, 8, 8)
        norm_layout.setSpacing(4)
        
        self.normalize_chk = QtWidgets.QCheckBox("Enable arcsinh normalization")
        self.normalize_chk.setChecked(False)
        self.normalize_chk.setToolTip("Apply arcsinh transformation to extracted intensity features (mean, median, std, etc.), not to raw images")
        norm_layout.addWidget(self.normalize_chk)
        
        cofactor_layout = QtWidgets.QHBoxLayout()
        cofactor_layout.setSpacing(4)
        cofactor_layout.addWidget(QtWidgets.QLabel("Cofactor:"))
        self.arcsinh_cofactor_spin = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor_spin.setRange(0.1, 100.0)
        self.arcsinh_cofactor_spin.setDecimals(1)
        self.arcsinh_cofactor_spin.setValue(10.0)
        self.arcsinh_cofactor_spin.setSingleStep(0.1)
        cofactor_layout.addWidget(self.arcsinh_cofactor_spin)
        cofactor_layout.addStretch()
        norm_layout.addLayout(cofactor_layout)
        
        # Note about arcsinh being applied after feature extraction
        arcsinh_note = QtWidgets.QLabel(
            "Applied after feature extraction\n"
            "and spillover correction"
        )
        arcsinh_note.setStyleSheet("QLabel { color: #666; font-size: 8pt; font-style: italic; }")
        arcsinh_note.setWordWrap(True)
        norm_layout.addWidget(arcsinh_note)
        
        preprocess_layout.addWidget(norm_group)
        
        layout.addLayout(preprocess_layout)
        
        # Custom denoising frame (collapsible)
        self.custom_denoise_frame = QtWidgets.QFrame()
        self.custom_denoise_frame.setFrameStyle(QtWidgets.QFrame.Box)
        self.custom_denoise_frame.setVisible(False)
        custom_denoise_layout = QtWidgets.QVBoxLayout(self.custom_denoise_frame)
        custom_denoise_layout.setContentsMargins(8, 8, 8, 8)
        custom_denoise_layout.setSpacing(4)
        
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
        self.hot_pixel_chk = QtWidgets.QCheckBox("Hot pixel")
        self.hot_pixel_method_combo = QtWidgets.QComboBox()
        self.hot_pixel_method_combo.addItems(["Median 3x3", ">N SD"])
        self.hot_pixel_n_spin = QtWidgets.QDoubleSpinBox()
        self.hot_pixel_n_spin.setRange(0.5, 10.0)
        self.hot_pixel_n_spin.setDecimals(1)
        self.hot_pixel_n_spin.setValue(5.0)
        self.hot_pixel_n_spin.setMaximumWidth(60)
        hot_layout.addWidget(self.hot_pixel_chk)
        hot_layout.addWidget(self.hot_pixel_method_combo)
        hot_n_layout = QtWidgets.QHBoxLayout()
        hot_n_layout.addWidget(QtWidgets.QLabel("N:"))
        hot_n_layout.addWidget(self.hot_pixel_n_spin)
        hot_n_layout.addStretch()
        hot_layout.addLayout(hot_n_layout)
        self.hot_pixel_n_label = QtWidgets.QLabel("N:")
        denoise_controls_layout.addWidget(hot_frame)
        
        # Speckle smoothing
        speckle_frame = QtWidgets.QFrame()
        speckle_layout = QtWidgets.QVBoxLayout(speckle_frame)
        speckle_layout.setContentsMargins(0, 0, 0, 0)
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
        
        layout.addWidget(self.custom_denoise_frame)
        
        # Feature selection with tabs
        feature_group = QtWidgets.QGroupBox("Select Features to Extract")
        feature_layout = QtWidgets.QVBoxLayout(feature_group)
        feature_layout.setContentsMargins(8, 8, 8, 8)
        feature_layout.setSpacing(4)
        
        # Create tab widget for features
        self.feature_tabs = QtWidgets.QTabWidget()
        self.feature_tabs.setMaximumHeight(140)  # Limit height of feature tabs
        
        # Morphology features tab
        morph_tab = QtWidgets.QWidget()
        morph_layout = QtWidgets.QVBoxLayout(morph_tab)
        morph_layout.setContentsMargins(4, 4, 4, 4)
        morph_layout.setSpacing(2)
        
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
            'holes_count': QtWidgets.QCheckBox("Number of holes"),
            'centroid_x': QtWidgets.QCheckBox("Centroid X coordinate (pixels)"),
            'centroid_y': QtWidgets.QCheckBox("Centroid Y coordinate (pixels)")
        }
        
        # Create two columns for morphology features
        morph_cols = QtWidgets.QHBoxLayout()
        morph_cols.setSpacing(8)
        morph_left = QtWidgets.QVBoxLayout()
        morph_left.setSpacing(2)
        morph_right = QtWidgets.QVBoxLayout()
        morph_right.setSpacing(2)
        
        morph_keys = list(self.morph_features.keys())
        mid_point = len(morph_keys) // 2
        
        for i, (key, checkbox) in enumerate(self.morph_features.items()):
            checkbox.setChecked(True)
            if i < mid_point:
                morph_left.addWidget(checkbox)
            else:
                morph_right.addWidget(checkbox)
        
        morph_cols.addLayout(morph_left)
        morph_cols.addLayout(morph_right)
        morph_layout.addLayout(morph_cols)
        
        self.feature_tabs.addTab(morph_tab, "Morphology")
        
        # Intensity features tab
        intensity_tab = QtWidgets.QWidget()
        intensity_layout = QtWidgets.QVBoxLayout(intensity_tab)
        intensity_layout.setContentsMargins(4, 4, 4, 4)
        intensity_layout.setSpacing(2)
        
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
        
        # Create two columns for intensity features
        intensity_cols = QtWidgets.QHBoxLayout()
        intensity_cols.setSpacing(8)
        intensity_left = QtWidgets.QVBoxLayout()
        intensity_left.setSpacing(2)
        intensity_right = QtWidgets.QVBoxLayout()
        intensity_right.setSpacing(2)
        
        intensity_keys = list(self.intensity_features.keys())
        mid_point = len(intensity_keys) // 2
        
        for i, (key, checkbox) in enumerate(self.intensity_features.items()):
            checkbox.setChecked(True)
            if i < mid_point:
                intensity_left.addWidget(checkbox)
            else:
                intensity_right.addWidget(checkbox)
        
        intensity_cols.addLayout(intensity_left)
        intensity_cols.addLayout(intensity_right)
        intensity_layout.addLayout(intensity_cols)
        
        self.feature_tabs.addTab(intensity_tab, "Intensity")
        
        feature_layout.addWidget(self.feature_tabs)
        layout.addWidget(feature_group)
        
        # Channel exclusion section
        channel_exclusion_group = QtWidgets.QGroupBox("Channel Exclusion")
        channel_exclusion_layout = QtWidgets.QVBoxLayout(channel_exclusion_group)
        channel_exclusion_layout.setContentsMargins(8, 8, 8, 8)
        channel_exclusion_layout.setSpacing(4)
        
        info_label = QtWidgets.QLabel(
            "Channels with low variance across ROIs are suggested for exclusion.\n"
            "You can manually select additional channels to exclude from feature extraction."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: #666; font-size: 8pt; }")
        channel_exclusion_layout.addWidget(info_label)
        
        # List widget for channel selection
        self.channel_exclusion_list = QtWidgets.QListWidget()
        self.channel_exclusion_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.channel_exclusion_list.setMaximumHeight(100)  # Reduced height
        channel_exclusion_layout.addWidget(self.channel_exclusion_list)
        
        # Button to calculate variance and suggest channels
        self.calculate_variance_btn = QtWidgets.QPushButton("Calculate Variance & Suggest Channels")
        self.calculate_variance_btn.clicked.connect(self._calculate_variance_and_suggest)
        channel_exclusion_layout.addWidget(self.calculate_variance_btn)
        
        layout.addWidget(channel_exclusion_group)
        
        # Store excluded channels
        self.excluded_channels = set()
        
        # Set scroll content widget
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Buttons (outside scroll area)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(8)
        self.extract_btn = QtWidgets.QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.extract_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
    
    def _populate_acquisitions(self):
        """Populate the acquisition list with acquisitions that have masks."""
        self.acq_list.clear()
        for acq in self.acquisitions:
            if acq.id in self.segmentation_masks:
                # Show source file name if available
                import os
                file_name = os.path.basename(acq.source_file) if hasattr(acq, 'source_file') and acq.source_file else None
                label = f"{acq.name}"
                if acq.well:
                    label += f" (Well: {acq.well})"
                if file_name:
                    label += f" [{file_name}]"
                item = QtWidgets.QListWidgetItem(label)
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
    
    # ---------- Preprocessing Methods ----------
    def _populate_denoise_channel_list(self):
        """Populate the denoise channel combo with available channels."""
        # Get channels from the first acquisition that has masks
        channels = []
        for acq in self.acquisitions:
            if acq.id in self.segmentation_masks:
                # Get channels from parent (MainWindow)
                parent = self.parent()
                if hasattr(parent, '_get_loader_for_acquisition') and hasattr(parent, '_get_original_acq_id'):
                    # Get the correct loader for this acquisition
                    loader = parent._get_loader_for_acquisition(acq.id)
                    if loader is not None:
                        # Get original acquisition ID (needed for multi-file support)
                        original_acq_id = parent._get_original_acq_id(acq.id)
                        try:
                            channels = loader.get_channels(original_acq_id)
                            break
                        except Exception:
                            pass
                # Fallback to old method for backward compatibility
                elif hasattr(parent, 'loader') and parent.loader is not None:
                    try:
                        channels = parent.loader.get_channels(acq.id)
                        break
                    except Exception:
                        pass
        
        self.denoise_channel_combo.blockSignals(True)
        self.denoise_channel_combo.clear()
        for ch in channels:
            self.denoise_channel_combo.addItem(ch)
        self.denoise_channel_combo.blockSignals(False)
        if channels:
            self.denoise_channel_combo.setCurrentIndex(0)
            self._load_denoise_settings()
    
    def _on_denoise_source_changed(self):
        """Handle changes to the denoising source selection."""
        use_custom = self.denoise_source_combo.currentText() == "Custom"
        self.custom_denoise_frame.setVisible(use_custom)
    
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
                # 0 white_tophat, 1 black_tophat, 2 rolling_ball (approx)
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

            # Get all available channels
            channels = []
            for i in range(self.denoise_channel_combo.count()):
                channels.append(self.denoise_channel_combo.itemText(i))

            # Apply the same configuration to all channels
            for channel in channels:
                self.custom_denoise_settings.setdefault(channel, {})
                # Store copies per channel to avoid shared references
                self.custom_denoise_settings[channel]["hot"] = dict(cfg_hot) if cfg_hot else None
                self.custom_denoise_settings[channel]["speckle"] = dict(cfg_speckle) if cfg_speckle else None
                self.custom_denoise_settings[channel]["background"] = dict(cfg_bg) if cfg_bg else None
            
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
    
    def _on_spillover_toggled(self):
        """Handle spillover correction checkbox toggle."""
        enabled = self.spillover_chk.isChecked()
        self.spillover_file_edit.setEnabled(enabled)
        self.spillover_file_btn.setEnabled(enabled)
    
    def _select_spillover_file(self):
        """Open spillover correction dialog to select matrix file."""
        from openimc.ui.dialogs.spillover_correction_dialog import SpilloverCorrectionDialog
        
        # Get channels from first acquisition
        channels = []
        for acq in self.acquisitions:
            if acq.id in self.segmentation_masks:
                parent = self.parent()
                if hasattr(parent, '_get_loader_for_acquisition') and hasattr(parent, '_get_original_acq_id'):
                    # Get the correct loader for this acquisition
                    loader = parent._get_loader_for_acquisition(acq.id)
                    if loader is not None:
                        # Get original acquisition ID (needed for multi-file support)
                        original_acq_id = parent._get_original_acq_id(acq.id)
                        try:
                            channels = loader.get_channels(original_acq_id)
                            break
                        except Exception:
                            pass
                # Fallback to old method for backward compatibility
                elif hasattr(parent, 'loader') and parent.loader is not None:
                    try:
                        channels = parent.loader.get_channels(acq.id)
                        break
                    except Exception:
                        pass
        
        dlg = SpilloverCorrectionDialog(self, channels=channels)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            file_path = dlg.get_spillover_file_path()
            if file_path:
                self.spillover_file_edit.setText(file_path)
                self._spillover_matrix = dlg.get_spillover_matrix()
                self._spillover_method = dlg.get_method()
            else:
                self._spillover_matrix = None
                self._spillover_method = None
        else:
            # If dialog was cancelled, uncheck the checkbox if no file is set
            if not self.spillover_file_edit.text():
                self.spillover_chk.setChecked(False)
    
    # ---------- Getter Methods ----------
    def get_normalization_config(self):
        """Get the normalization configuration."""
        if self.normalize_chk.isChecked():
            return {
                'method': 'arcsinh',
                'cofactor': float(self.arcsinh_cofactor_spin.value())
            }
        return None
    
    def get_denoise_source(self):
        """Get the selected denoising source."""
        return self.denoise_source_combo.currentText()
    
    def get_custom_denoise_settings(self):
        """Get the custom denoising settings."""
        return self.custom_denoise_settings
    
    def get_spillover_config(self):
        """Get the spillover correction configuration."""
        if self.spillover_chk.isChecked() and self._spillover_matrix is not None:
            return {
                'matrix': self._spillover_matrix,
                'method': self._spillover_method or 'pgd'
            }
        return None
    
    def _populate_channel_exclusion_list(self):
        """Populate the channel exclusion list with available channels."""
        # Get channels from the first acquisition that has masks
        channels = []
        for acq in self.acquisitions:
            if acq.id in self.segmentation_masks:
                # Get channels from parent (MainWindow)
                parent = self.parent()
                if hasattr(parent, '_get_loader_for_acquisition') and hasattr(parent, '_get_original_acq_id'):
                    # Get the correct loader for this acquisition
                    loader = parent._get_loader_for_acquisition(acq.id)
                    if loader is not None:
                        # Get original acquisition ID (needed for multi-file support)
                        original_acq_id = parent._get_original_acq_id(acq.id)
                        try:
                            channels = loader.get_channels(original_acq_id)
                            break
                        except Exception:
                            pass
                # Fallback to old method for backward compatibility
                elif hasattr(parent, 'loader') and parent.loader is not None:
                    try:
                        channels = parent.loader.get_channels(acq.id)
                        break
                    except Exception:
                        pass
        
        self.channel_exclusion_list.clear()
        for ch in channels:
            item = QtWidgets.QListWidgetItem(ch)
            item.setData(Qt.UserRole, ch)
            self.channel_exclusion_list.addItem(item)
    
    def _calculate_variance_and_suggest(self):
        """Calculate variance across ROIs for each channel and suggest low-variance channels."""
        # Get the first acquisition that has masks
        first_acq = None
        first_acq_id = None
        for acq in self.acquisitions:
            if acq.id in self.segmentation_masks:
                first_acq = acq
                first_acq_id = acq.id
                break
        
        if not first_acq or not first_acq_id:
            QtWidgets.QMessageBox.warning(
                self,
                "No Acquisition",
                "No acquisition with segmentation mask found for variance calculation."
            )
            return
        
        # Get parent window to access loader
        parent = self.parent()
        loader = None
        original_acq_id_for_loader = None
        
        # Try to get loader using multi-file support method
        if hasattr(parent, '_get_loader_for_acquisition') and hasattr(parent, '_get_original_acq_id'):
            loader = parent._get_loader_for_acquisition(first_acq_id)
            if loader is not None:
                original_acq_id_for_loader = parent._get_original_acq_id(first_acq_id)
        # Fallback to old method for backward compatibility
        elif hasattr(parent, 'loader') and parent.loader is not None:
            loader = parent.loader
            original_acq_id_for_loader = first_acq_id
        
        if loader is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Loader",
                "Cannot access data loader for variance calculation."
            )
            return
        
        try:
            # Get channels and image data
            channels = loader.get_channels(original_acq_id_for_loader)
            mask = self.segmentation_masks[first_acq_id]
            
            # Get image stack (use first acquisition for variance calculation)
            img_stack = loader.get_all_channels(original_acq_id_for_loader)
            
            # Calculate variance across ROIs for each channel
            import numpy as np
            from skimage.measure import regionprops_table
            
            # Ensure mask is int labels
            label_image = mask.astype(np.int32, copy=False)
            unique_labels = np.unique(label_image[label_image > 0])
            
            if len(unique_labels) == 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No ROIs",
                    "No ROIs found in segmentation mask for variance calculation."
                )
                return
            
            # Calculate mean intensity per ROI per channel
            channel_variances = {}
            
            for idx, ch_name in enumerate(channels):
                ch_img = img_stack[..., idx]
                
                # Get mean intensity per ROI
                roi_means = []
                for lbl in unique_labels:
                    mask_lbl = (label_image == lbl)
                    pix = ch_img[mask_lbl]
                    if pix.size > 0:
                        roi_means.append(float(np.mean(pix)))
                
                if len(roi_means) > 1:
                    # Calculate coefficient of variation (CV = std/mean) as a measure of variance
                    roi_means_arr = np.array(roi_means)
                    mean_val = np.mean(roi_means_arr)
                    std_val = np.std(roi_means_arr)
                    cv = std_val / (mean_val + 1e-8)  # Coefficient of variation
                    channel_variances[ch_name] = {
                        'cv': cv,
                        'std': std_val,
                        'mean': mean_val
                    }
                else:
                    # If only one ROI, set variance to 0
                    channel_variances[ch_name] = {
                        'cv': 0.0,
                        'std': 0.0,
                        'mean': float(roi_means[0]) if roi_means else 0.0
                    }
            
            # Sort channels by CV (coefficient of variation)
            sorted_channels = sorted(channel_variances.items(), key=lambda x: x[1]['cv'])
            
            # Suggest bottom 20% of channels (lowest variance) for exclusion
            num_to_suggest = max(1, int(len(sorted_channels) * 0.2))
            suggested_channels = {ch for ch, _ in sorted_channels[:num_to_suggest]}
            
            # Update the list widget with variance information and suggestions
            self.channel_exclusion_list.clear()
            self.excluded_channels.clear()
            
            for ch_name, var_info in sorted_channels:
                cv = var_info['cv']
                item_text = f"{ch_name} (CV: {cv:.4f})"
                if ch_name in suggested_channels:
                    item_text += " [Suggested]"
                
                item = QtWidgets.QListWidgetItem(item_text)
                item.setData(Qt.UserRole, ch_name)
                
                # Pre-select suggested channels
                if ch_name in suggested_channels:
                    item.setSelected(True)
                    self.excluded_channels.add(ch_name)
                
                self.channel_exclusion_list.addItem(item)
            
            # Update button text to show completion
            self.calculate_variance_btn.setText("✓ Variance Calculated")
            self.calculate_variance_btn.setStyleSheet("QPushButton { background-color: #d4edda; color: #155724; }")
            
            # Show message with suggestion
            QtWidgets.QMessageBox.information(
                self,
                "Variance Calculated",
                f"Calculated variance across {len(unique_labels)} ROIs.\n"
                f"Suggested {len(suggested_channels)} channel(s) with low variance for exclusion:\n"
                f"{', '.join(sorted(suggested_channels))}"
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to calculate variance:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def get_excluded_channels(self):
        """Get the set of excluded channel names."""
        # Update excluded channels from current selection
        self.excluded_channels.clear()
        for i in range(self.channel_exclusion_list.count()):
            item = self.channel_exclusion_list.item(i)
            if item.isSelected():
                ch_name = item.data(Qt.UserRole)
                if ch_name:
                    self.excluded_channels.add(ch_name)
        return self.excluded_channels
