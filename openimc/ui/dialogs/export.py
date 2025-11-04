from typing import List, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer

from openimc.data.mcd_loader import AcquisitionInfo

# Optional scikit-image for denoising
try:
    from skimage import morphology
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

# --------------------------
# Export Dialog
# --------------------------
class ExportDialog(QtWidgets.QDialog):
    def __init__(self, acquisitions: List[AcquisitionInfo], current_acq_id: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export to OME-TIFF")
        self.setModal(True)
        
        # Set dialog size
        if parent:
            parent_size = parent.size()
            dialog_width = int(parent_size.width() * 0.9)
            dialog_height = int(parent_size.height() * 0.9)
            self.resize(dialog_width, dialog_height)
        else:
            self.resize(700, 600)
        
        self.setMinimumSize(600, 500)
        self.acquisitions = acquisitions
        self.current_acq_id = current_acq_id
        self.output_directory = ""
        
        # Initialize custom denoising settings storage
        self.custom_denoise_settings = {}
        
        # Create UI
        self._create_ui()
        
        # Initialize denoising
        self._populate_denoise_channel_list()
        self._on_denoise_source_changed()
        self._sync_hot_controls_visibility()
        
        # Initialize normalization
        self._on_norm_method_changed()
        
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
        
        # Export type selection
        type_group = QtWidgets.QGroupBox("Export Type")
        type_layout = QtWidgets.QVBoxLayout(type_group)
        
        self.single_roi_radio = QtWidgets.QRadioButton("Single ROI (Current Acquisition)")
        self.whole_slide_radio = QtWidgets.QRadioButton("Whole Slide (All Acquisitions)")
        self.single_roi_radio.setChecked(True)
        
        type_layout.addWidget(self.single_roi_radio)
        type_layout.addWidget(self.whole_slide_radio)
        layout.addWidget(type_group)
        
        # Current acquisition info
        self.acq_info_label = QtWidgets.QLabel("")
        layout.addWidget(self.acq_info_label)
        
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
        
        # Intensity Scaling section
        scaling_group = QtWidgets.QGroupBox("Intensity Scaling")
        scaling_layout = QtWidgets.QVBoxLayout(scaling_group)
        scaling_layout.setSpacing(4)
        scaling_layout.setContentsMargins(8, 8, 8, 8)
        
        norm_method_layout = QtWidgets.QHBoxLayout()
        norm_method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.norm_method_combo = QtWidgets.QComboBox()
        self.norm_method_combo.addItems(["None", "arcsinh", "percentile_clip"])
        self.norm_method_combo.currentTextChanged.connect(self._on_norm_method_changed)
        norm_method_layout.addWidget(self.norm_method_combo)
        norm_method_layout.addStretch()
        scaling_layout.addLayout(norm_method_layout)
        
        self.arcsinh_frame = QtWidgets.QFrame()
        arcsinh_layout = QtWidgets.QHBoxLayout(self.arcsinh_frame)
        arcsinh_layout.addWidget(QtWidgets.QLabel("Cofactor:"))
        self.arcsinh_cofactor_spin = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor_spin.setRange(0.1, 100.0)
        self.arcsinh_cofactor_spin.setValue(5.0)
        self.arcsinh_cofactor_spin.setDecimals(1)
        arcsinh_layout.addWidget(self.arcsinh_cofactor_spin)
        arcsinh_layout.addStretch()
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
        
        layout.addWidget(scaling_group)
        
        # Denoising options
        denoise_group = QtWidgets.QGroupBox("Image Denoising")
        denoise_layout = QtWidgets.QVBoxLayout(denoise_group)
        denoise_layout.setSpacing(4)
        denoise_layout.setContentsMargins(8, 8, 8, 8)
        
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
        self.hot_pixel_method_combo.currentTextChanged.connect(self._save_current_denoise_settings)
        self.hot_pixel_n_spin = QtWidgets.QDoubleSpinBox()
        self.hot_pixel_n_spin.setRange(0.5, 10.0)
        self.hot_pixel_n_spin.setDecimals(1)
        self.hot_pixel_n_spin.setValue(5.0)
        self.hot_pixel_n_spin.setMaximumWidth(60)
        self.hot_pixel_chk.toggled.connect(self._save_current_denoise_settings)
        self.hot_pixel_n_spin.valueChanged.connect(self._save_current_denoise_settings)
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
        self.speckle_method_combo.currentTextChanged.connect(self._save_current_denoise_settings)
        self.gaussian_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.1, 5.0)
        self.gaussian_sigma_spin.setDecimals(2)
        self.gaussian_sigma_spin.setValue(0.8)
        self.gaussian_sigma_spin.setMaximumWidth(60)
        self.speckle_chk.toggled.connect(self._save_current_denoise_settings)
        self.gaussian_sigma_spin.valueChanged.connect(self._save_current_denoise_settings)
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
        self.bg_method_combo.currentTextChanged.connect(self._save_current_denoise_settings)
        self.bg_radius_spin = QtWidgets.QSpinBox()
        self.bg_radius_spin.setRange(1, 100)
        self.bg_radius_spin.setValue(15)
        self.bg_radius_spin.setMaximumWidth(60)
        self.bg_subtract_chk.toggled.connect(self._save_current_denoise_settings)
        self.bg_radius_spin.valueChanged.connect(self._save_current_denoise_settings)
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
        
        # Disable custom denoising panel if scikit-image is missing
        if not _HAVE_SCIKIT_IMAGE:
            self.custom_denoise_frame.setEnabled(False)
            custom_denoise_layout.addWidget(QtWidgets.QLabel("scikit-image not available; install to enable custom denoising."))
        
        denoise_layout.addWidget(self.custom_denoise_frame)
        layout.addWidget(denoise_group)
        
        # Options
        options_group = QtWidgets.QGroupBox("Export Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.include_metadata_chk = QtWidgets.QCheckBox("Include metadata in OME-TIFF")
        self.include_metadata_chk.setChecked(True)
        options_layout.addWidget(self.include_metadata_chk)
        
        layout.addWidget(options_group)
        
        # Set scroll content widget
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area, 1)  # Stretch factor 1 to take available space
        
        # Buttons (outside scroll area)
        button_layout = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setEnabled(False)  # Disabled until directory is selected
        self.export_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.single_roi_radio.toggled.connect(self._on_export_type_changed)
        self.whole_slide_radio.toggled.connect(self._on_export_type_changed)
        
        # Initialize the display
        self._on_export_type_changed()
        
    def _browse_directory(self):
        """Browse for output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if directory:
            self.output_directory = directory
            self.dir_label.setText(directory)
            self.dir_label.setStyleSheet("QLabel { color: black; }")
            self.export_btn.setEnabled(True)
    
    def _on_export_type_changed(self):
        """Update UI when export type changes."""
        if self.single_roi_radio.isChecked():
            if self.current_acq_id:
                # Find current acquisition info
                current_acq = next((acq for acq in self.acquisitions if acq.id == self.current_acq_id), None)
                if current_acq:
                    info_text = f"Will export: {current_acq.name}\n"
                    info_text += f"Channels: {len(current_acq.channels)}\n"
                    if current_acq.well:
                        info_text += f"Well: {current_acq.well}"
                    self.acq_info_label.setText(info_text)
                else:
                    self.acq_info_label.setText("Will export only the currently selected acquisition.")
            else:
                self.acq_info_label.setText("Will export only the currently selected acquisition.")
        else:
            # Show more detailed information about what will be exported
            total_channels = sum(len(acq.channels) for acq in self.acquisitions)
            info_text = f"Will export all {len(self.acquisitions)} acquisitions from the slide.\n"
            info_text += f"Total channels: {total_channels}\n"
            info_text += f"Acquisitions: {', '.join([acq.name for acq in self.acquisitions[:3]])}"
            if len(self.acquisitions) > 3:
                info_text += f" and {len(self.acquisitions) - 3} more..."
            self.acq_info_label.setText(info_text)
    
    def get_export_type(self):
        """Get the selected export type."""
        return "single" if self.single_roi_radio.isChecked() else "whole"
    
    def get_output_directory(self):
        """Get the selected output directory."""
        return self.output_directory
    
    def get_include_metadata(self):
        """Get whether to include metadata."""
        return self.include_metadata_chk.isChecked()
    
    # ---------- Denoising Methods ----------
    def _populate_denoise_channel_list(self):
        """Populate the denoise channel combo with available channels."""
        # Get channels from the first acquisition or current acquisition
        channels = []
        if self.current_acq_id:
            # Get channels from parent (MainWindow)
            parent = self.parent()
            if hasattr(parent, 'loader') and hasattr(parent, 'current_acq_id'):
                # Temporarily set current acquisition to get channels
                original_acq_id = parent.current_acq_id
                parent.current_acq_id = self.current_acq_id
                try:
                    channels = parent.loader.get_channels(self.current_acq_id)
                finally:
                    parent.current_acq_id = original_acq_id
        elif self.acquisitions:
            # Use first acquisition
            parent = self.parent()
            if hasattr(parent, 'loader'):
                try:
                    channels = parent.loader.get_channels(self.acquisitions[0].id)
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
        # Save current settings before switching channels
        self._save_current_denoise_settings()
        # Load settings for the new channel
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
    
    def _save_current_denoise_settings(self):
        """Save current UI denoise settings for the selected channel."""
        ch = self.denoise_channel_combo.currentText()
        if not ch:
            return
        
        cfg = {}
        
        # Hot pixel
        if self.hot_pixel_chk.isChecked():
            cfg["hot"] = {
                "method": "median3" if self.hot_pixel_method_combo.currentIndex() == 0 else "n_sd_local_median",
                "n_sd": float(self.hot_pixel_n_spin.value()),
            }
        
        # Speckle
        if self.speckle_chk.isChecked():
            cfg["speckle"] = {
                "method": "gaussian" if self.speckle_method_combo.currentIndex() == 0 else "nl_means",
                "sigma": float(self.gaussian_sigma_spin.value()),
            }
        
        # Background
        if self.bg_subtract_chk.isChecked():
            bg_idx = self.bg_method_combo.currentIndex()
            if bg_idx == 0:
                bg_method = "white_tophat"
            elif bg_idx == 1:
                bg_method = "black_tophat"
            else:
                bg_method = "rolling_ball"
            cfg["background"] = {
                "method": bg_method,
                "radius": int(self.bg_radius_spin.value()),
            }
        
        self.custom_denoise_settings[ch] = cfg
    
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
    
    # ---------- Normalization Methods ----------
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
        # Save current settings before returning
        self._save_current_denoise_settings()
        return self.custom_denoise_settings
