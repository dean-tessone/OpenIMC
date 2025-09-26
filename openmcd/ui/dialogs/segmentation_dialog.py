from typing import List

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer

# Data types
from openmcd.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
import os
from openmcd.ui.utils import combine_channels
from openmcd.ui.dialogs.preprocessing_dialog import PreprocessingDialog

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
            model_index = self.model_combo.findText(prefs['model'])
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
        super().accept()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Model selection
        model_group = QtWidgets.QGroupBox("Segmentation Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["cyto3", "nuclei"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(QtWidgets.QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        
        # Model description
        self.model_desc = QtWidgets.QLabel("Cytoplasm: Segments whole cells using cytoplasm + nuclear channels")
        self.model_desc.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        model_layout.addWidget(self.model_desc)
        
        layout.addWidget(model_group)
        
        # Preprocessing and Denoising section
        preprocess_group = QtWidgets.QGroupBox("Image Preprocessing")
        preprocess_layout = QtWidgets.QVBoxLayout(preprocess_group)
        
        # Preprocessing button
        preprocess_btn = QtWidgets.QPushButton("Configure Segmentation...")
        preprocess_btn.clicked.connect(self._open_preprocessing_dialog)
        preprocess_layout.addWidget(preprocess_btn)
        
        self.preprocess_info_label = QtWidgets.QLabel("No preprocessing configured")
        self.preprocess_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.preprocess_info_label.setWordWrap(True)
        preprocess_layout.addWidget(self.preprocess_info_label)
        
        # Denoising options
        denoise_frame = QtWidgets.QFrame()
        denoise_layout = QtWidgets.QVBoxLayout(denoise_frame)
        denoise_layout.setContentsMargins(0, 10, 0, 0)
        
        # Denoising source selection
        denoise_source_layout = QtWidgets.QHBoxLayout()
        denoise_source_layout.addWidget(QtWidgets.QLabel("Denoising:"))
        self.denoise_source_combo = QtWidgets.QComboBox()
        self.denoise_source_combo.addItems(["None", "Use viewer settings", "Use custom settings"])
        self.denoise_source_combo.currentTextChanged.connect(self._on_denoise_source_changed)
        denoise_source_layout.addWidget(self.denoise_source_combo)
        denoise_source_layout.addStretch()
        denoise_layout.addLayout(denoise_source_layout)
        
        # Custom denoising frame (hidden by default)
        self.custom_denoise_frame = QtWidgets.QFrame()
        self.custom_denoise_frame.setFrameStyle(QtWidgets.QFrame.Box)
        self.custom_denoise_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; }")
        custom_denoise_layout = QtWidgets.QVBoxLayout(self.custom_denoise_frame)
        custom_denoise_layout.setContentsMargins(10, 10, 10, 10)
        custom_denoise_layout.setSpacing(8)
        
        # Add a title label
        title_label = QtWidgets.QLabel("Custom Denoising Settings")
        title_label.setStyleSheet("QLabel { font-weight: bold; color: #495057; }")
        custom_denoise_layout.addWidget(title_label)
        
        custom_denoise_layout.addWidget(QtWidgets.QLabel("Configure denoising parameters for each channel:"))
        
        # Channel dropdown for custom denoising
        denoise_channel_row = QtWidgets.QHBoxLayout()
        denoise_channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.denoise_channel_combo = QtWidgets.QComboBox()
        self.denoise_channel_combo.currentTextChanged.connect(self._on_denoise_channel_changed)
        denoise_channel_row.addWidget(self.denoise_channel_combo, 1)
        custom_denoise_layout.addLayout(denoise_channel_row)
        
        # Hot pixel removal
        hot_group = QtWidgets.QGroupBox("Hot Pixel Removal")
        hot_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        hot_layout = QtWidgets.QVBoxLayout(hot_group)
        hot_layout.setContentsMargins(8, 8, 8, 8)
        
        self.hot_pixel_chk = QtWidgets.QCheckBox("Enable hot pixel removal")
        self.hot_pixel_method_combo = QtWidgets.QComboBox()
        self.hot_pixel_method_combo.addItems(["Median 3x3", ">N SD above local median"])
        self.hot_pixel_n_spin = QtWidgets.QDoubleSpinBox()
        self.hot_pixel_n_spin.setRange(0.5, 10.0)
        self.hot_pixel_n_spin.setDecimals(1)
        self.hot_pixel_n_spin.setValue(5.0)
        hot_row = QtWidgets.QHBoxLayout()
        hot_row.addWidget(self.hot_pixel_chk)
        hot_row.addWidget(self.hot_pixel_method_combo)
        self.hot_pixel_n_label = QtWidgets.QLabel("N:")
        hot_row.addWidget(self.hot_pixel_n_label)
        hot_row.addWidget(self.hot_pixel_n_spin)
        hot_row.addStretch()
        hot_layout.addLayout(hot_row)
        custom_denoise_layout.addWidget(hot_group)
        
        # Speckle smoothing
        speckle_group = QtWidgets.QGroupBox("Speckle Smoothing")
        speckle_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        speckle_layout = QtWidgets.QVBoxLayout(speckle_group)
        speckle_layout.setContentsMargins(8, 8, 8, 8)
        
        self.speckle_chk = QtWidgets.QCheckBox("Enable speckle smoothing")
        self.speckle_method_combo = QtWidgets.QComboBox()
        self.speckle_method_combo.addItems(["Gaussian", "Non-local means (slow)"])
        self.gaussian_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.1, 5.0)
        self.gaussian_sigma_spin.setDecimals(2)
        self.gaussian_sigma_spin.setValue(0.8)
        self.gaussian_sigma_spin.setSingleStep(0.1)
        speckle_row = QtWidgets.QHBoxLayout()
        speckle_row.addWidget(self.speckle_chk)
        speckle_row.addWidget(self.speckle_method_combo)
        speckle_row.addWidget(QtWidgets.QLabel("σ:"))
        speckle_row.addWidget(self.gaussian_sigma_spin)
        speckle_row.addStretch()
        speckle_layout.addLayout(speckle_row)
        custom_denoise_layout.addWidget(speckle_group)
        
        # Background subtraction
        bg_group = QtWidgets.QGroupBox("Background Subtraction")
        bg_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        bg_layout = QtWidgets.QVBoxLayout(bg_group)
        bg_layout.setContentsMargins(8, 8, 8, 8)
        
        self.bg_subtract_chk = QtWidgets.QCheckBox("Enable background subtraction")
        self.bg_method_combo = QtWidgets.QComboBox()
        self.bg_method_combo.addItems(["White top-hat", "Black top-hat", "Rolling ball (approx)"])
        self.bg_radius_spin = QtWidgets.QSpinBox()
        self.bg_radius_spin.setRange(1, 100)
        self.bg_radius_spin.setValue(15)
        bg_row = QtWidgets.QHBoxLayout()
        bg_row.addWidget(self.bg_subtract_chk)
        bg_row.addWidget(self.bg_method_combo)
        bg_row.addWidget(QtWidgets.QLabel("radius:"))
        bg_row.addWidget(self.bg_radius_spin)
        bg_row.addStretch()
        bg_layout.addLayout(bg_row)
        custom_denoise_layout.addWidget(bg_group)
        
        # Apply to all channels button
        self.apply_all_channels_btn = QtWidgets.QPushButton("Apply to All Channels")
        self.apply_all_channels_btn.setStyleSheet("QPushButton { background-color: #007bff; color: white; font-weight: bold; padding: 8px; border-radius: 4px; } QPushButton:hover { background-color: #0056b3; }")
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
        
        # Parameters
        params_group = QtWidgets.QGroupBox("Segmentation Parameters")
        params_layout = QtWidgets.QVBoxLayout(params_group)
        
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
        flow_layout = QtWidgets.QHBoxLayout()
        flow_layout.addWidget(QtWidgets.QLabel("Flow threshold:"))
        self.flow_spinbox = QtWidgets.QDoubleSpinBox()
        self.flow_spinbox.setRange(0.0, 10.0)
        self.flow_spinbox.setDecimals(2)
        self.flow_spinbox.setValue(0.4)
        self.flow_spinbox.setSingleStep(0.1)
        flow_layout.addWidget(self.flow_spinbox)
        flow_layout.addStretch()
        
        params_layout.addLayout(flow_layout)
        
        # Cell probability threshold
        cellprob_layout = QtWidgets.QHBoxLayout()
        cellprob_layout.addWidget(QtWidgets.QLabel("Cell probability threshold:"))
        self.cellprob_spinbox = QtWidgets.QDoubleSpinBox()
        self.cellprob_spinbox.setRange(-6.0, 6.0)
        self.cellprob_spinbox.setDecimals(1)
        self.cellprob_spinbox.setValue(0.0)
        self.cellprob_spinbox.setSingleStep(0.5)
        cellprob_layout.addWidget(self.cellprob_spinbox)
        cellprob_layout.addStretch()
        
        params_layout.addLayout(cellprob_layout)
        
        layout.addWidget(params_group)
        
        # GPU selection
        gpu_group = QtWidgets.QGroupBox("GPU Acceleration")
        gpu_layout = QtWidgets.QVBoxLayout(gpu_group)
        
        gpu_row = QtWidgets.QHBoxLayout()
        gpu_row.addWidget(QtWidgets.QLabel("Device:"))
        self.gpu_combo = QtWidgets.QComboBox()
        self.gpu_combo.addItem("Auto-detect", "auto")
        self.gpu_combo.addItem("CPU only", None)
        gpu_row.addWidget(self.gpu_combo)
        gpu_row.addStretch()
        gpu_layout.addLayout(gpu_row)
        
        self.gpu_info_label = QtWidgets.QLabel("")
        self.gpu_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.gpu_info_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_info_label)
        
        layout.addWidget(gpu_group)
        
        # Options
        options_group = QtWidgets.QGroupBox("Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
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
        self.batch_warning.setStyleSheet("QLabel { color: #d97706; font-size: 11px; font-weight: bold; }")
        self.batch_warning.setWordWrap(True)
        self.batch_warning.setVisible(False)
        options_layout.addWidget(self.batch_warning)
        
        # Info label for segment all
        self.segment_all_info = QtWidgets.QLabel("")
        self.segment_all_info.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.segment_all_info.setWordWrap(True)
        options_layout.addWidget(self.segment_all_info)
        
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.segment_btn = QtWidgets.QPushButton("Run Segmentation")
        self.segment_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; } QPushButton:hover { background-color: #218838; } QPushButton:pressed { background-color: #1e7e34; }")
        self.segment_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; } QPushButton:hover { background-color: #c82333; } QPushButton:pressed { background-color: #bd2130; }")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.segment_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
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
        
    def _on_model_changed(self):
        """Update UI when model selection changes."""
        model = self.model_combo.currentText()
        if model == "nuclei":
            self.model_desc.setText("Nuclei: Segments cell nuclei using nuclear channel")
            # When using nuclei model, hide cytoplasm selection UI in preprocessing
            # (actual hiding is applied when dialog is opened)
        else:  # cyto3
            self.model_desc.setText("Cytoplasm: Segments whole cells using cytoplasm + nuclear channels")
    
    def _on_auto_diameter_toggled(self):
        """Enable/disable diameter spinbox based on auto-estimate checkbox."""
        self.diameter_spinbox.setEnabled(not self.auto_diameter_chk.isChecked())
    
    def get_model(self):
        """Get selected model."""
        return self.model_combo.currentText()
    
    
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
        # Hide cytoplasm section when using nuclei-only model
        dlg.set_cytoplasm_visible(self.model_combo.currentText() != 'nuclei')
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.preprocessing_config = {
                'normalization_method': dlg.get_normalization_method(),
                'arcsinh_cofactor': dlg.get_arcsinh_cofactor(),
                'percentile_params': dlg.get_percentile_params(),
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
            self.preprocess_info_label.setText("No preprocessing configured")
            return
        
        config = self.preprocessing_config
        info_parts = []
        
        # Normalization info
        if config['normalization_method'] != 'None':
            if config['normalization_method'] == 'arcsinh':
                info_parts.append(f"Arcsinh (cofactor={config['arcsinh_cofactor']})")
            elif config['normalization_method'] == 'percentile_clip':
                p_low, p_high = config['percentile_params']
                info_parts.append(f"Percentile clip ({p_low}-{p_high}%)")
        
        # Channel combination info
        if config['nuclear_channels']:
            nuclear_info = f"Nuclear: {config['nuclear_combo_method']}({len(config['nuclear_channels'])} channels)"
            info_parts.append(nuclear_info)
        
        if config['cyto_channels']:
            cyto_info = f"Cytoplasm: {config['cyto_combo_method']}({len(config['cyto_channels'])} channels)"
            info_parts.append(cyto_info)
        
        if info_parts:
            self.preprocess_info_label.setText(" | ".join(info_parts))
        else:
            self.preprocess_info_label.setText("No preprocessing configured")
    
    def get_preprocessing_config(self):
        """Get the preprocessing configuration."""
        return self.preprocessing_config

    def set_use_viewer_denoising(self, enabled: bool):
        """Initialize the 'use viewer denoising' toggle state."""
        if enabled:
            self.denoise_source_combo.setCurrentText("Use viewer settings")
        else:
            self.denoise_source_combo.setCurrentText("None")

    def get_use_viewer_denoising(self) -> bool:
        """Return whether to use viewer denoising during segmentation."""
        return self.denoise_source_combo.currentText() == "Use viewer settings"
    
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
        use_custom = source == "Use custom settings"
        self.custom_denoise_frame.setVisible(use_custom)
        
        # Adjust dialog size when custom denoising is shown/hidden
        if self.parent():
            parent_size = self.parent().size()
            if use_custom:
                # Use 90% of parent size for custom denoising
                dialog_width = int(parent_size.width() * 0.9)
                dialog_height = int(parent_size.height() * 0.9)
                self.resize(dialog_width, dialog_height)
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
                cfg_bg = {
                    "method": "white_tophat" if self.bg_method_combo.currentIndex() == 0 else "rolling_ball",
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
        if source == "Use viewer settings":
            return "viewer"
        elif source == "Use custom settings":
            return "custom"
        else:
            return "none"
    
    def get_custom_denoise_settings(self):
        """Get the custom denoising settings."""
        return self.custom_denoise_settings