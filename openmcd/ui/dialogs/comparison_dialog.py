from typing import List, Optional

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Configure matplotlib backend
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Data types
from openmcd.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
from openmcd.ui.utils import combine_channels, arcsinh_normalize
from openmcd.ui.mpl_canvas import MplCanvas

# Optional GPU runtime
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False


class DynamicComparisonDialog(QtWidgets.QDialog):
    def __init__(self, acqs: List[AcquisitionInfo], loader: MCDLoader, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dynamic Comparison Mode")
        
        # Set dialog size to 80% of screen size
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        
        self.acqs = acqs
        self.loader = loader
        self.selected_acquisitions = []
        self.current_channel = None
        
        # Cache for loaded images to avoid reloading
        self.image_cache = {}  # {(acquisition_id, channel): image}
        self.max_cache_size = 50  # Limit overall items in cache
        # Prefetch control (selected acquisitions up to 5)
        self._prefetch_limit = 5
        self._prefetch_inflight = set()
        
        # Store last selected channel for auto-selection
        self.last_selected_channel: Optional[str] = None
        
        # Store previous scaling channel to save values when switching
        self.previous_scaling_channel: Optional[str] = None
        
        # Store per-image scaling values for individual scaling
        self.image_scaling = {}  # legacy per-image scaling for single channel
        # Per-channel scaling storages
        self.channel_linked_scaling = {}  # {channel: {'min': v, 'max': v}}
        self.channel_per_image_scaling = {}  # {channel: {acq_id: {'min': v, 'max': v}}}
        
        # Per-image arcsinh state storage
        self.image_arcsinh_state = {}  # {acq_id: {'enabled': bool, 'cofactor': float}}
        
        self.setMinimumSize(1000, 700)
        
        # Create UI
        self._create_ui()
        
        # Start with empty selection - user must select acquisitions of interest

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Use splitter to prioritize image view area
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        # Control panel (left)
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        
        # Acquisition selection
        acq_group = QtWidgets.QGroupBox("Acquisitions")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        
        acq_layout.addWidget(QtWidgets.QLabel("Available:"))
        # Search box for acquisitions
        self.acq_search = QtWidgets.QLineEdit()
        self.acq_search.setPlaceholderText("Search acquisitions...")
        self.acq_search.textChanged.connect(self._filter_acquisitions)
        acq_layout.addWidget(self.acq_search)
        self.available_acq_list = QtWidgets.QListWidget()
        self.available_acq_list.setMaximumHeight(120)
        for ai in self.acqs:
            label = f"{ai.name}" + (f" ({ai.well})" if ai.well else "")
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, ai.id)
            self.available_acq_list.addItem(item)
        
        acq_buttons = QtWidgets.QHBoxLayout()
        self.add_acq_btn = QtWidgets.QPushButton("Add →")
        acq_buttons.addWidget(self.add_acq_btn)
        
        acq_layout.addWidget(self.available_acq_list)
        acq_layout.addLayout(acq_buttons)
        
        acq_layout.addWidget(QtWidgets.QLabel("Selected:"))
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setMaximumHeight(120)
        
        acq_layout.addWidget(self.acq_list)
        # Place the Remove button below the Selected list
        self.remove_acq_btn = QtWidgets.QPushButton("← Remove")
        acq_layout.addWidget(self.remove_acq_btn)
        control_layout.addWidget(acq_group)
        
        # Channel selection
        channel_group = QtWidgets.QGroupBox("Channel")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)
        
        # Single-channel selector
        self.channel_combo = QtWidgets.QComboBox()
        channel_layout.addWidget(QtWidgets.QLabel("Marker channel:"))
        channel_layout.addWidget(self.channel_combo)
        
        # RGB mode and assignments (mimic main window)
        self.rgb_mode_chk = QtWidgets.QCheckBox("RGB Mode")
        self.rgb_mode_chk.toggled.connect(self._on_rgb_mode_toggled)
        channel_layout.addWidget(self.rgb_mode_chk)
        
        self.rgb_frame = QtWidgets.QFrame()
        self.rgb_frame.setFrameStyle(QtWidgets.QFrame.Box)
        rgb_layout = QtWidgets.QVBoxLayout(self.rgb_frame)
        # Red block
        red_block = QtWidgets.QVBoxLayout()
        red_block.addWidget(QtWidgets.QLabel("Red:"))
        self.red_search = QtWidgets.QLineEdit()
        self.red_search.setPlaceholderText("Search red channels...")
        self.red_search.textChanged.connect(self._filter_red)
        red_block.addWidget(self.red_search)
        self.red_list = QtWidgets.QListWidget()
        self.red_list.setMaximumHeight(140)
        self.red_list.itemChanged.connect(lambda _i: (self._refresh_scaling_channel_options(), self._update_display()))
        red_block.addWidget(self.red_list)
        rgb_layout.addLayout(red_block)
        # Green block
        green_block = QtWidgets.QVBoxLayout()
        green_block.addWidget(QtWidgets.QLabel("Green:"))
        self.green_search = QtWidgets.QLineEdit()
        self.green_search.setPlaceholderText("Search green channels...")
        self.green_search.textChanged.connect(self._filter_green)
        green_block.addWidget(self.green_search)
        self.green_list = QtWidgets.QListWidget()
        self.green_list.setMaximumHeight(140)
        self.green_list.itemChanged.connect(lambda _i: (self._refresh_scaling_channel_options(), self._update_display()))
        green_block.addWidget(self.green_list)
        rgb_layout.addLayout(green_block)
        # Blue block
        blue_block = QtWidgets.QVBoxLayout()
        blue_block.addWidget(QtWidgets.QLabel("Blue:"))
        self.blue_search = QtWidgets.QLineEdit()
        self.blue_search.setPlaceholderText("Search blue channels...")
        self.blue_search.textChanged.connect(self._filter_blue)
        blue_block.addWidget(self.blue_search)
        self.blue_list = QtWidgets.QListWidget()
        self.blue_list.setMaximumHeight(140)
        self.blue_list.itemChanged.connect(lambda _i: (self._refresh_scaling_channel_options(), self._update_display()))
        blue_block.addWidget(self.blue_list)
        rgb_layout.addLayout(blue_block)
        self.rgb_frame.setVisible(False)
        channel_layout.addWidget(self.rgb_frame)
        
        # Display options
        options_group = QtWidgets.QGroupBox("Display Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.link_chk = QtWidgets.QCheckBox("Linked scaling (shared min/max)")
        self.link_chk.setChecked(True)
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        
        options_layout.addWidget(self.link_chk)
        options_layout.addWidget(self.grayscale_chk)
        
        # Custom scaling controls for comparison mode
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.custom_scaling_chk.toggled.connect(self._on_comparison_scaling_toggled)
        options_layout.addWidget(self.custom_scaling_chk)

        # Optional segmentation overlay (if masks exist in parent)
        self.overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay (when available)")
        self.overlay_chk.toggled.connect(self._update_display)
        options_layout.addWidget(self.overlay_chk)
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        # Channel selector for per-channel scaling
        self.scaling_channel_row = QtWidgets.QWidget()
        scaling_channel_layout = QtWidgets.QHBoxLayout(self.scaling_channel_row)
        scaling_channel_layout.setContentsMargins(0, 0, 0, 0)
        scaling_channel_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.scaling_channel_combo = QtWidgets.QComboBox()
        self.scaling_channel_combo.currentTextChanged.connect(self._on_scaling_channel_changed)
        scaling_channel_layout.addWidget(self.scaling_channel_combo)
        scaling_channel_layout.addStretch()
        scaling_layout.addWidget(self.scaling_channel_row)
        
        # Image selection for individual scaling (shown only when link is OFF)
        self.image_selection_row = QtWidgets.QWidget()
        image_selection_layout = QtWidgets.QHBoxLayout(self.image_selection_row)
        image_selection_layout.setContentsMargins(0, 0, 0, 0)
        image_selection_layout.addWidget(QtWidgets.QLabel("Image:"))
        self.image_combo = QtWidgets.QComboBox()
        self.image_combo.currentTextChanged.connect(self._on_image_selection_changed)
        self.image_combo.currentIndexChanged.connect(self._on_image_selection_changed)
        image_selection_layout.addWidget(self.image_combo)
        image_selection_layout.addStretch()
        scaling_layout.addWidget(self.image_selection_row)
        
        # Min/Max controls (auto-apply on change)
        minmax_layout = QtWidgets.QHBoxLayout()
        minmax_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(-999999, 999999)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        self.min_spinbox.valueChanged.connect(lambda _v: self._apply_comparison_scaling())
        minmax_layout.addWidget(self.min_spinbox)
        
        minmax_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(-999999, 999999)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(100.0)
        self.max_spinbox.valueChanged.connect(lambda _v: self._apply_comparison_scaling())
        minmax_layout.addWidget(self.max_spinbox)
        
        scaling_layout.addLayout(minmax_layout)
        
        # Normalization controls
        norm_layout = QtWidgets.QHBoxLayout()
        self.arcsinh_chk = QtWidgets.QCheckBox("Arcsinh")
        self.arcsinh_chk.toggled.connect(self._on_arcsinh_toggled)
        norm_layout.addWidget(self.arcsinh_chk)
        norm_layout.addWidget(QtWidgets.QLabel("cofactor:"))
        self.arcsinh_cofactor = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor.setRange(0.01, 1000.0)
        self.arcsinh_cofactor.setDecimals(2)
        self.arcsinh_cofactor.setSingleStep(0.25)
        self.arcsinh_cofactor.setValue(5.0)
        self.arcsinh_cofactor.valueChanged.connect(self._on_arcsinh_cofactor_changed)
        norm_layout.addWidget(self.arcsinh_cofactor)
        norm_layout.addStretch()
        scaling_layout.addLayout(norm_layout)

        # Range helper buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.default_range_btn = QtWidgets.QPushButton("Original Range")
        self.default_range_btn.clicked.connect(self._comparison_default_range)
        button_layout.addWidget(self.default_range_btn)
        button_layout.addStretch()
        scaling_layout.addLayout(button_layout)
        self.scaling_frame.setVisible(False)
        
        options_layout.addWidget(self.scaling_frame)

        control_layout.addWidget(channel_group)
        control_layout.addWidget(options_group)
        control_panel.setMaximumWidth(380)
        splitter.addWidget(control_panel)

        # Image display area (right)
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_widget = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QGridLayout(self.image_widget)
        self.image_scroll.setWidget(self.image_widget)
        self.image_scroll.setWidgetResizable(True)
        splitter.addWidget(self.image_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        # Connect signals
        self.add_acq_btn.clicked.connect(self._add_acquisition)
        self.remove_acq_btn.clicked.connect(self._remove_acquisition)
        self.channel_combo.currentTextChanged.connect(self._update_display)
        self.link_chk.toggled.connect(self._on_link_contrast_toggled)
        self.grayscale_chk.toggled.connect(self._update_display)
        self.acq_list.itemSelectionChanged.connect(self._update_display)
        
        # Initialize channel combo when acquisitions are added
        self._update_channel_combo()
        # Initial prefetch (no-op until user selects acquisitions)
        self._start_prefetch_selected()

    def closeEvent(self, event):
        """Handle dialog close event with proper cleanup."""
        self._clear_display()
        super().closeEvent(event)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self._clear_display()
        except Exception:
            pass

    def _filter_acquisitions(self):
        text = self.acq_search.text().lower()
        for i in range(self.available_acq_list.count()):
            item = self.available_acq_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _on_rgb_mode_toggled(self):
        # Toggle visibility of RGB assignments vs single-channel combo
        is_rgb = self.rgb_mode_chk.isChecked()
        self.rgb_frame.setVisible(is_rgb)
        self.channel_combo.setEnabled(not is_rgb)
        # Populate RGB lists based on first selected acquisition's channels
        self._populate_rgb_lists()
        self._update_display()

    def _populate_rgb_lists(self):
        # Populate with channels from first selected acquisition
        self.red_list.blockSignals(True)
        self.green_list.blockSignals(True)
        self.blue_list.blockSignals(True)
        self.red_list.clear()
        self.green_list.clear()
        self.blue_list.clear()
        if not self.selected_acquisitions:
            self.red_list.blockSignals(False)
            self.green_list.blockSignals(False)
            self.blue_list.blockSignals(False)
            return
        acq_id = self.selected_acquisitions[0]
        ai = next((a for a in self.acqs if a.id == acq_id), None)
        channels = ai.channels if ai else []
        for ch in channels:
            for lst in (self.red_list, self.green_list, self.blue_list):
                it = QtWidgets.QListWidgetItem(ch)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                it.setCheckState(Qt.Unchecked)
                lst.addItem(it)
        self.red_list.blockSignals(False)
        self.green_list.blockSignals(False)
        self.blue_list.blockSignals(False)

    def _filter_red(self):
        text = self.red_search.text().lower()
        for i in range(self.red_list.count()):
            item = self.red_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _filter_green(self):
        text = self.green_search.text().lower()
        for i in range(self.green_list.count()):
            item = self.green_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _filter_blue(self):
        text = self.blue_search.text().lower()
        for i in range(self.blue_list.count()):
            item = self.blue_list.item(i)
            item.setHidden(text not in item.text().lower())

    def _add_acquisition(self):
        current_item = self.available_acq_list.currentItem()
        if current_item:
            acq_id = current_item.data(Qt.UserRole)
            if acq_id not in self.selected_acquisitions:
                # Store current channel before updating
                current_channel = self.channel_combo.currentText()
                
                self.selected_acquisitions.append(acq_id)
                # Create a new item with the same data
                new_item = QtWidgets.QListWidgetItem(current_item.text())
                new_item.setData(Qt.UserRole, acq_id)
                self.acq_list.addItem(new_item)
                
                # Update channel combo (will preserve current channel if possible)
                self._update_channel_combo()
                
                # Start background prefetch for selected acquisitions
        self._start_prefetch_selected()
        if self.custom_scaling_chk.isChecked() and not self.link_chk.isChecked():
            self._update_image_combo()

    def _remove_acquisition(self):
        current_item = self.acq_list.currentItem()
        if current_item:
            acq_id = current_item.data(Qt.UserRole)
            self.selected_acquisitions.remove(acq_id)
            self.acq_list.takeItem(self.acq_list.row(current_item))
            self._update_channel_combo()
            # Update prefetch set and prune cache
            self._start_prefetch_selected()
            if self.custom_scaling_chk.isChecked() and not self.link_chk.isChecked():
                self._update_image_combo()

    def _update_display(self):
        """Update the image display with proper cleanup."""
        # Clear existing images with proper cleanup
        self._clear_display()
        
        if not self.selected_acquisitions:
            return
        
        is_rgb = self.rgb_mode_chk.isChecked() if hasattr(self, 'rgb_mode_chk') else False
        channel = self.channel_combo.currentText()
        self.last_selected_channel = channel
        grayscale = self.grayscale_chk.isChecked()
        link_contrast = self.link_chk.isChecked()
        
        # Load images
        images = []
        titles = []
        if is_rgb:
            # Build per-acquisition RGB composites
            def _checked(lst: QtWidgets.QListWidget) -> List[str]:
                vals = []
                for i in range(lst.count()):
                    it = lst.item(i)
                    if it.checkState() == Qt.Checked:
                        vals.append(it.text())
                return vals
            reds = _checked(self.red_list)
            greens = _checked(self.green_list)
            blues = _checked(self.blue_list)
            for acq_id in self.selected_acquisitions:
                try:
                    # Sum selected channels per color
                    def _get_cached(acq: str, ch_name: str):
                        key = (acq, ch_name)
                        if key in self.image_cache:
                            return self.image_cache[key]
                        arr = self.loader.get_image(acq, ch_name)
                        self.image_cache[key] = arr
                        self._manage_cache_size()
                        return arr
                    def _normalize_with_channel_settings(channel_name: str, arr: np.ndarray, acquisition_id: str) -> np.ndarray:
                        # Optional arcsinh - only apply to the selected scaling channel and selected image
                        a = arr.astype(np.float32, copy=False)
                        selected_scaling_channel = self.scaling_channel_combo.currentText()
                        # Check if arcsinh should be applied to this specific image
                        if link_contrast:
                            # In linked mode, use global arcsinh setting
                            should_apply_arcsinh = (self.arcsinh_chk.isChecked() and 
                                                   channel_name == selected_scaling_channel)
                        else:
                            # In unlinked mode, check per-image arcsinh state
                            image_arcsinh = self.image_arcsinh_state.get(acquisition_id, {})
                            should_apply_arcsinh = (image_arcsinh.get('enabled', False) and 
                                                   channel_name == selected_scaling_channel)
                        if should_apply_arcsinh:
                            # Get the cofactor for this image
                            if link_contrast:
                                cofactor = self.arcsinh_cofactor.value()
                            else:
                                image_arcsinh = self.image_arcsinh_state.get(acquisition_id, {})
                                cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
                            a = arcsinh_normalize(a, cofactor)
                        # Custom scaling per channel
                        if self.custom_scaling_chk.isChecked():
                            vmin = None
                            vmax = None
                            if self.link_chk.isChecked():
                                if channel_name in self.channel_linked_scaling:
                                    vals = self.channel_linked_scaling[channel_name]
                                    vmin, vmax = vals.get('min'), vals.get('max')
                                else:
                                    # compute global across selected acquisitions for this channel
                                    try:
                                        imgs = [self.loader.get_image(aq, channel_name) for aq in self.selected_acquisitions]
                                        selected_scaling_channel = self.scaling_channel_combo.currentText()
                                        # Only apply arcsinh if this is the selected scaling channel and arcsinh is enabled
                                        if self.arcsinh_chk.isChecked() and channel_name == selected_scaling_channel and not link_contrast:
                                            imgs = [arcsinh_normalize(im, self.arcsinh_cofactor.value()) for im in imgs]
                                        vmin = float(min(np.min(im) for im in imgs))
                                        vmax = float(max(np.max(im) for im in imgs))
                                        self.channel_linked_scaling[channel_name] = {'min': vmin, 'max': vmax}
                                    except Exception:
                                        vmin = float(np.min(a))
                                        vmax = float(np.max(a))
                            else:
                                per_img = self.channel_per_image_scaling.get(channel_name, {})
                                if acquisition_id in per_img:
                                    vals = per_img[acquisition_id]
                                    vmin, vmax = vals.get('min'), vals.get('max')
                                else:
                                    # Use the transformed range (after arcsinh if applied)
                                    vmin = float(np.min(a))
                                    vmax = float(np.max(a))
                                    per_img.setdefault(acquisition_id, {'min': vmin, 'max': vmax})
                                    self.channel_per_image_scaling[channel_name] = per_img
                            if vmin is not None and vmax is not None and vmax > vmin:
                                # Don't normalize to 0-1, just clip to the range
                                a = np.clip(a, vmin, vmax)
                        return a
                    def _sum_channels(names: List[str]):
                        if not names:
                            # Return zeros of first available channel size
                            base = _get_cached(acq_id, channel) if channel else None
                            if base is None:
                                return None
                            return np.zeros_like(base, dtype=np.float32)
                        acc = None
                        for ch in names:
                            arr = _get_cached(acq_id, ch)
                            arr_n = _normalize_with_channel_settings(ch, arr, acq_id)
                            acc = arr_n if acc is None else (acc + arr_n)
                        if acc is None:
                            return None
                        # Don't clip to 0-1, keep the actual summed values
                        return acc
                    r = _sum_channels(reds)
                    g = _sum_channels(greens)
                    b = _sum_channels(blues)
                    if r is None or g is None or b is None:
                        continue
                    
                    # Normalize each channel for RGB display
                    def _normalize_channel(channel_data, channel_name):
                        if channel_data is None:
                            return None
                        
                        # Determine if arcsinh should be applied for this channel and image
                        selected_scaling_channel = self.scaling_channel_combo.currentText()
                        # Check if arcsinh should be applied to this specific image
                        if link_contrast:
                            # In linked mode, use global arcsinh setting
                            should_apply_arcsinh = (self.arcsinh_chk.isChecked() and 
                                                   channel_name == selected_scaling_channel)
                        else:
                            # In unlinked mode, check per-image arcsinh state
                            image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                            should_apply_arcsinh = (image_arcsinh.get('enabled', False) and 
                                                   channel_name == selected_scaling_channel)
                        
                        # If custom scaling is enabled, use the custom range for this channel
                        if self.custom_scaling_chk.isChecked():
                            # Check if this is the currently selected scaling channel
                            if channel_name == selected_scaling_channel:
                                # In unlinked mode, only use spinbox values for the currently selected image
                                if not link_contrast and acq_id == self.image_combo.currentData():
                                    ch_min = self.min_spinbox.value()
                                    ch_max = self.max_spinbox.value()
                                    # If arcsinh is applied, transform the min/max values
                                    if should_apply_arcsinh:
                                        # Get the cofactor for this image
                                        if link_contrast:
                                            cofactor = self.arcsinh_cofactor.value()
                                        else:
                                            image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                                            cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
                                        ch_min = arcsinh_normalize(np.array([ch_min]), cofactor)[0]
                                        ch_max = arcsinh_normalize(np.array([ch_max]), cofactor)[0]
                                else:
                                    # For linked mode or non-selected images, use stored values
                                    if link_contrast:
                                        # In linked mode, use the current spinbox values for the selected scaling channel
                                        ch_min = self.min_spinbox.value()
                                        ch_max = self.max_spinbox.value()
                                        # If arcsinh is applied, transform the min/max values
                                        if should_apply_arcsinh:
                                            # Get the cofactor for this image
                                            if link_contrast:
                                                cofactor = self.arcsinh_cofactor.value()
                                            else:
                                                image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                                                cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
                                            ch_min = arcsinh_normalize(np.array([ch_min]), cofactor)[0]
                                            ch_max = arcsinh_normalize(np.array([ch_max]), cofactor)[0]
                                    else:
                                        per_img = self.channel_per_image_scaling.get(channel_name, {})
                                        if acq_id in per_img:
                                            vals = per_img[acq_id]
                                            ch_min, ch_max = vals.get('min'), vals.get('max')
                                        else:
                                            ch_min = float(np.min(channel_data))
                                            ch_max = float(np.max(channel_data))
                            else:
                                # For other channels, use stored values or calculate from data
                                if link_contrast:
                                    # Use linked scaling for this channel
                                    if channel_name in self.channel_linked_scaling:
                                        vals = self.channel_linked_scaling[channel_name]
                                        ch_min, ch_max = vals.get('min'), vals.get('max')
                                    else:
                                        # Calculate global range for this channel
                                        try:
                                            imgs = [self.loader.get_image(aq, channel_name) for aq in self.selected_acquisitions]
                                            # Don't apply arcsinh to non-selected channels
                                            ch_min = float(min(np.min(im) for im in imgs))
                                            ch_max = float(max(np.max(im) for im in imgs))
                                            self.channel_linked_scaling[channel_name] = {'min': ch_min, 'max': ch_max}
                                        except Exception:
                                            ch_min = float(np.min(channel_data))
                                            ch_max = float(np.max(channel_data))
                                else:
                                    # Use per-image scaling for this channel
                                    per_img = self.channel_per_image_scaling.get(channel_name, {})
                                    if acq_id in per_img:
                                        vals = per_img[acq_id]
                                        ch_min, ch_max = vals.get('min'), vals.get('max')
                                    else:
                                        ch_min = float(np.min(channel_data))
                                        ch_max = float(np.max(channel_data))
                                        per_img.setdefault(acq_id, {'min': ch_min, 'max': ch_max})
                                        self.channel_per_image_scaling[channel_name] = per_img
                        else:
                            # No custom scaling, use the channel's own range
                            ch_min = float(np.min(channel_data))
                            ch_max = float(np.max(channel_data))
                        
                        # Apply arcsinh transformation if needed
                        if should_apply_arcsinh:
                            # Get the cofactor for this image
                            if link_contrast:
                                cofactor = self.arcsinh_cofactor.value()
                            else:
                                image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                                cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
                            
                            # Apply arcsinh transformation
                            transformed_data = arcsinh_normalize(channel_data, cofactor)
                            # Calculate the range of the transformed data
                            trans_min = np.min(transformed_data)
                            trans_max = np.max(transformed_data)
                            # Normalize the transformed data to 0-1 range
                            if trans_max > trans_min:
                                return (transformed_data - trans_min) / (trans_max - trans_min)
                            else:
                                return np.zeros_like(transformed_data)
                        else:
                            # Normalize to [0,1] using the determined range
                            if ch_max > ch_min:
                                return (channel_data - ch_min) / (ch_max - ch_min)
                            else:
                                return np.zeros_like(channel_data)
                    
                    r_norm = _normalize_channel(r, reds[0] if reds else "red")
                    g_norm = _normalize_channel(g, greens[0] if greens else "green")
                    b_norm = _normalize_channel(b, blues[0] if blues else "blue")
                    
                    rgb = np.dstack([r_norm, g_norm, b_norm])
                    images.append(rgb)
                    titles.append(self._get_acquisition_subtitle(acq_id))
                except Exception as e:
                    print(f"Error building RGB for {acq_id}: {e}")
                    continue
        else:
            for acq_id in self.selected_acquisitions:
                try:
                    if not channel:
                        continue
                    key = (acq_id, channel)
                    if key in self.image_cache:
                        img = self.image_cache[key]
                    else:
                        img = self.loader.get_image(acq_id, channel)
                        self.image_cache[key] = img
                        self._manage_cache_size()
                    images.append(img)
                    titles.append(self._get_acquisition_subtitle(acq_id))
                except Exception as e:
                    print(f"Error loading image for {acq_id}: {e}")
                    continue
        
        if not images:
            return
        
        # Calculate grid layout
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        # Calculate scaling based on custom scaling and link contrast settings
        custom_scaling_enabled = self.custom_scaling_chk.isChecked()
        custom_min = self.min_spinbox.value() if custom_scaling_enabled else None
        custom_max = self.max_spinbox.value() if custom_scaling_enabled else None
        
        # Determine if arcsinh should be applied globally
        # Disable arcsinh when linked scaling is enabled
        selected_scaling_channel = self.scaling_channel_combo.currentText()
        global_arcsinh_applied = (self.arcsinh_chk.isChecked() and 
                                 channel == selected_scaling_channel and 
                                 not link_contrast)
        
        # Calculate global min/max if link contrast is enabled
        vmin = None
        vmax = None
        if link_contrast and len(images) > 1:
            if custom_scaling_enabled:
                # Use custom scaling for linked contrast
                vmin = custom_min
                vmax = custom_max
            else:
                # Use global min/max for fair comparison (after optional transform)
                def _transform_for_display_global(arr: np.ndarray) -> np.ndarray:
                    a = arr
                    # Apply arcsinh if it should be applied globally
                    if global_arcsinh_applied:
                        a = arcsinh_normalize(a, self.arcsinh_cofactor.value())
                    return a
                
                # Calculate global range across all images
                all_mins = []
                all_maxs = []
                for img in images:
                    transformed = _transform_for_display_global(img)
                    if is_rgb and not grayscale:
                        # For RGB color display, data is already properly scaled, no global scaling needed
                        continue
                    else:
                        all_mins.append(np.min(transformed))
                        all_maxs.append(np.max(transformed))
                
                if all_mins and all_maxs:
                    vmin = min(all_mins)
                    vmax = max(all_maxs)
        elif not link_contrast and custom_scaling_enabled:
            # For individual scaling with custom scaling enabled,
            # each image can have its own custom scaling range
            # We'll handle this per-image in the display loop
            pass
        
        # Display images
        for i, (img, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            
            # Create canvas with proper parent
            try:
                canvas = MplCanvas(width=4, height=4, dpi=100)
                canvas.setParent(self.image_widget)
            except Exception as e:
                print(f"Error creating canvas: {e}")
                continue
            
            # Determine scaling for this image
            img_vmin = vmin
            img_vmax = vmax
            
            if img_vmin is None or img_vmax is None:
                # Individual scaling - check for per-image custom scaling
                acq_id = self.selected_acquisitions[i]
                # Check if arcsinh should be applied to this specific image
                if link_contrast:
                    # In linked mode, use global arcsinh setting
                    is_arcsinh_applied = global_arcsinh_applied
                else:
                    # In unlinked mode, check per-image arcsinh state
                    image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                    is_arcsinh_applied = image_arcsinh.get('enabled', False)
                
                if custom_scaling_enabled and not link_contrast:
                    # Check for per-channel per-image scaling
                    ch = selected_scaling_channel or channel
                    per_img = self.channel_per_image_scaling.get(ch, {})
                    if acq_id in per_img:
                        # Use custom scaling for this specific image
                        stored_min = per_img[acq_id]['min']
                        stored_max = per_img[acq_id]['max']
                        
                        if is_arcsinh_applied:
                            # Apply arcsinh to the stored min/max values to get the transformed range
                            img_vmin = arcsinh_normalize(np.array([stored_min]), self.arcsinh_cofactor.value())[0]
                            img_vmax = arcsinh_normalize(np.array([stored_max]), self.arcsinh_cofactor.value())[0]
                        else:
                            img_vmin = stored_min
                            img_vmax = stored_max
                    else:
                        # Use image's own min/max range (after arcsinh if applicable)
                        if is_arcsinh_applied:
                            # Get the cofactor for this image
                            if link_contrast:
                                cofactor = self.arcsinh_cofactor.value()
                            else:
                                image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                                cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
                            
                            # Apply arcsinh to get the transformed range
                            transformed_img = arcsinh_normalize(img, cofactor)
                            img_vmin = np.min(transformed_img)
                            img_vmax = np.max(transformed_img)
                            # Normalize the transformed data to 0-1 range
                            if img_vmax > img_vmin:
                                img = (transformed_img - img_vmin) / (img_vmax - img_vmin)
                            else:
                                img = np.zeros_like(transformed_img)
                            # Set vmin/vmax to 0-1 for display
                            img_vmin = 0.0
                            img_vmax = 1.0
                        else:
                            img_vmin = np.min(img)
                            img_vmax = np.max(img)
                else:
                    # Use image's own min/max range (after arcsinh if applicable)
                    if is_arcsinh_applied:
                        # Get the cofactor for this image
                        if link_contrast:
                            cofactor = self.arcsinh_cofactor.value()
                        else:
                            image_arcsinh = self.image_arcsinh_state.get(acq_id, {})
                            cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
                        
                        # Apply arcsinh to get the transformed range
                        transformed_img = arcsinh_normalize(img, cofactor)
                        img_vmin = np.min(transformed_img)
                        img_vmax = np.max(transformed_img)
                        # Normalize the transformed data to 0-1 range
                        if img_vmax > img_vmin:
                            img = (transformed_img - img_vmin) / (img_vmax - img_vmin)
                        else:
                            img = np.zeros_like(transformed_img)
                        # Set vmin/vmax to 0-1 for display
                        img_vmin = 0.0
                        img_vmax = 1.0
                    else:
                        img_vmin = np.min(img)
                        img_vmax = np.max(img)
            
            # Display image with optional arcsinh transformation only
            def _transform_for_display(arr: np.ndarray, acq_id_for_scaling: Optional[str], channel_name: Optional[str]) -> np.ndarray:
                a = arr.astype(np.float32, copy=False)
                
                # Arcsinh transformation is now handled in the vmin/vmax calculation above
                # This function just returns the data as-is since it's already been processed
                return a
            if is_rgb and not grayscale:
                # RGB data is already properly scaled (either to [0,1] or using custom scaling)
                im = canvas.ax.imshow(img, interpolation="nearest")
            elif is_rgb and grayscale:
                # Show mean across channels as grayscale
                # img already normalized per-channel above
                gray = np.mean(img, axis=2)
                im = canvas.ax.imshow(gray, interpolation="nearest", cmap='gray', vmin=img_vmin, vmax=img_vmax)
            else:
                channel_name = self.channel_combo.currentText() if self.channel_combo.currentText() else None
                transformed = _transform_for_display(img, self.selected_acquisitions[i], channel_name)
                im = canvas.ax.imshow(transformed, interpolation="nearest", 
                                    cmap='gray' if grayscale else 'viridis',
                                    vmin=img_vmin, vmax=img_vmax)

            # Optional mask overlay
            if self.overlay_chk.isChecked():
                parent = self.parent()
                masks = getattr(parent, 'segmentation_masks', {}) if parent else {}
                acq_id_overlay = self.selected_acquisitions[i] if i < len(self.selected_acquisitions) else None
                if acq_id_overlay and acq_id_overlay in masks:
                    try:
                        mask = masks[acq_id_overlay]
                        mask_bool = mask.astype(bool)
                        if mask_bool.ndim == 2 and mask_bool.shape == img.shape[:2]:
                            canvas.ax.contour(mask_bool, levels=[0.5], colors='r', linewidths=0.6, alpha=0.7)
                    except Exception:
                        pass
            
            canvas.ax.set_title(title, fontsize=10)
            canvas.ax.axis("off")
            
            # Add colorbar
            # Add colorbar only for single-channel display
            if not is_rgb or grayscale:
                cbar = canvas.fig.colorbar(im, ax=canvas.ax, shrink=0.8, aspect=20)
            
            self.image_layout.addWidget(canvas, row, col)

    def _clear_display(self):
        """Clear all displayed images with proper matplotlib cleanup."""
        # Remove all widgets from the layout
        while self.image_layout.count():
            child = self.image_layout.takeAt(0)
            if child.widget():
                widget = child.widget()
                # Properly close matplotlib figures
                if hasattr(widget, 'fig'):
                    try:
                        widget.fig.clear()
                        plt.close(widget.fig)
                    except Exception:
                        pass
                widget.deleteLater()
        
        # Force update to ensure widgets are actually removed
        self.image_widget.update()
        
    def _update_channel_combo(self):
        """Update the channel combo box based on selected acquisitions."""
        old_channel = self.channel_combo.currentText()
        self.channel_combo.clear()
        self.scaling_channel_combo.clear()
        if self.selected_acquisitions:
            # Get channels from first selected acquisition
            ai = next(a for a in self.acqs if a.id == self.selected_acquisitions[0])
            self.channel_combo.addItems(ai.channels)
            # Populate scaling channel combo with currently selected channels
            self._refresh_scaling_channel_options()
            
            # Pre-select the last selected channel if it exists in this acquisition
            if self.last_selected_channel and self.last_selected_channel in ai.channels:
                index = ai.channels.index(self.last_selected_channel)
                self.channel_combo.setCurrentIndex(index)
            elif old_channel and old_channel in ai.channels:
                # Try to preserve the previously selected channel if it exists
                index = ai.channels.index(old_channel)
                self.channel_combo.setCurrentIndex(index)
                self.last_selected_channel = old_channel
            elif ai.channels:
                # Select first channel by default if no previous selection
                self.channel_combo.setCurrentIndex(0)
                self.last_selected_channel = ai.channels[0]
            
            # Clear cache if channel changed
            new_channel = self.channel_combo.currentText()
            if old_channel and old_channel != new_channel:
                self._clear_cache()
            
            # Auto-load images if channel is selected
            if self.channel_combo.currentText():
                self._update_display()
        else:
            self._clear_cache()

    def _clear_cache(self):
        """Clear the image cache."""
        self.image_cache.clear()
    
    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds max size."""
        if len(self.image_cache) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.image_cache.keys())[:len(self.image_cache) - self.max_cache_size]
            for key in keys_to_remove:
                del self.image_cache[key]
    
    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        """Get acquisition subtitle showing well/description instead of acquisition number."""
        acq_info = next((ai for ai in self.acqs if ai.id == acq_id), None)
        if not acq_info:
            return "Unknown"
        
        # Use well if available, otherwise use name (which might be more descriptive)
        if acq_info.well:
            return f"{acq_info.well}"
        else:
            return acq_info.name

    def _on_comparison_scaling_toggled(self):
        """Handle custom scaling checkbox toggle in comparison mode."""
        self.scaling_frame.setVisible(self.custom_scaling_chk.isChecked())
        if self.custom_scaling_chk.isChecked():
            if self.link_chk.isChecked():
                self._comparison_auto_range()
            else:
                self._update_image_combo()
            # Populate scaling channel options
            self._refresh_scaling_channel_options()
        
        # Update control states based on link contrast
        self._on_link_contrast_toggled()

    def _on_scaling_channel_changed(self):
        # When the scaling channel changes, save current values and reload appropriate min/max for current context
        if not self.custom_scaling_chk.isChecked():
            return
        
        # Save current min/max values for the previous channel (if any)
        self._save_current_channel_scaling()
        
        # Update the previous channel to the current one
        self.previous_scaling_channel = self.scaling_channel_combo.currentText()
        
        if self.link_chk.isChecked():
            ch = self.scaling_channel_combo.currentText()
            vals = self.channel_linked_scaling.get(ch)
            if vals:
                self.min_spinbox.setValue(vals.get('min', self.min_spinbox.value()))
                self.max_spinbox.setValue(vals.get('max', self.max_spinbox.value()))
            else:
                # compute from current selection
                try:
                    imgs = [self.loader.get_image(aq, ch) for aq in self.selected_acquisitions]
                    # Only apply arcsinh to the currently selected scaling channel
                    if self.arcsinh_chk.isChecked() and ch == self.scaling_channel_combo.currentText():
                        imgs = [arcsinh_normalize(im, self.arcsinh_cofactor.value()) for im in imgs]
                    vmin = float(min(np.min(im) for im in imgs))
                    vmax = float(max(np.max(im) for im in imgs))
                    self.channel_linked_scaling[ch] = {'min': vmin, 'max': vmax}
                    self.min_spinbox.setValue(vmin)
                    self.max_spinbox.setValue(vmax)
                except Exception:
                    pass
        else:
            self._load_image_scaling()
        
        # Update spinbox for arcsinh if needed
        self._update_spinbox_for_arcsinh()
        
        self._update_display()

    def _save_current_channel_scaling(self):
        """Save the current min/max values for the previous scaling channel."""
        if not self.previous_scaling_channel or not self.custom_scaling_chk.isChecked():
            return
        
        # Save current spinbox values for the previous channel
        current_min = self.min_spinbox.value()
        current_max = self.max_spinbox.value()
        
        if self.link_chk.isChecked():
            # Save to linked scaling storage
            self.channel_linked_scaling[self.previous_scaling_channel] = {
                'min': current_min, 
                'max': current_max
            }
        else:
            # Save to per-image scaling storage
            current_acq_id = self.image_combo.currentData()
            if current_acq_id:
                per_img = self.channel_per_image_scaling.get(self.previous_scaling_channel, {})
                per_img[current_acq_id] = {'min': current_min, 'max': current_max}
                self.channel_per_image_scaling[self.previous_scaling_channel] = per_img

    def _on_arcsinh_toggled(self):
        """Handle arcsinh checkbox toggle - clear scaling cache to force recalculation."""
        # Clear stored scaling values so they get recalculated with new transformation
        self.channel_linked_scaling.clear()
        self.channel_per_image_scaling.clear()
        self.image_scaling.clear()
        
        # Save arcsinh state for the currently selected image in unlinked mode
        if not self.link_chk.isChecked() and self.image_combo.currentData():
            current_acq_id = self.image_combo.currentData()
            self.image_arcsinh_state[current_acq_id] = {
                'enabled': self.arcsinh_chk.isChecked(),
                'cofactor': self.arcsinh_cofactor.value()
            }
        
        # Update spinbox values and make them uneditable when arcsinh is applied
        self._update_spinbox_for_arcsinh()
        
        self._update_display()

    def _on_arcsinh_cofactor_changed(self):
        """Handle arcsinh cofactor change - clear scaling cache to force recalculation."""
        # Clear stored scaling values so they get recalculated with new cofactor
        self.channel_linked_scaling.clear()
        self.channel_per_image_scaling.clear()
        self.image_scaling.clear()
        
        # Update arcsinh state for the currently selected image in unlinked mode
        if not self.link_chk.isChecked() and self.image_combo.currentData():
            current_acq_id = self.image_combo.currentData()
            if current_acq_id in self.image_arcsinh_state:
                self.image_arcsinh_state[current_acq_id]['cofactor'] = self.arcsinh_cofactor.value()
        
        # Update spinbox values and make them uneditable when arcsinh is applied
        self._update_spinbox_for_arcsinh()
        
        self._update_display()

    def _update_spinbox_for_arcsinh(self):
        """Update spinbox values and make them uneditable when arcsinh is applied."""
        if not self.arcsinh_chk.isChecked() or not self.custom_scaling_chk.isChecked():
            # Arcsinh is disabled or custom scaling is disabled, make spinboxes editable
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            
            # Restore default range when arcsinh is turned off
            if not self.arcsinh_chk.isChecked() and self.custom_scaling_chk.isChecked():
                self._restore_default_range()
            return
        
        # Arcsinh is enabled, update spinbox values and make them uneditable
        try:
            if self.link_chk.isChecked():
                # Linked mode - use global range
                channel = self.scaling_channel_combo.currentText()
                if not channel or not self.selected_acquisitions:
                    return
                
                # Get all images for this channel
                imgs = [self.loader.get_image(aq, channel) for aq in self.selected_acquisitions]
                # Apply arcsinh to get the transformed range
                transformed_imgs = [arcsinh_normalize(im, self.arcsinh_cofactor.value()) for im in imgs]
                vmin = float(min(np.min(im) for im in transformed_imgs))
                vmax = float(max(np.max(im) for im in transformed_imgs))
                
                # Update spinboxes
                self.min_spinbox.blockSignals(True)
                self.max_spinbox.blockSignals(True)
                self.min_spinbox.setValue(vmin)
                self.max_spinbox.setValue(vmax)
                self.min_spinbox.blockSignals(False)
                self.max_spinbox.blockSignals(False)
                
                # Make spinboxes uneditable
                self.min_spinbox.setEnabled(False)
                self.max_spinbox.setEnabled(False)
                
            else:
                # Unlinked mode - use range for currently selected image
                current_acq_id = self.image_combo.currentData()
                channel = self.scaling_channel_combo.currentText()
                if not current_acq_id or not channel:
                    return
                
                # Get image for this acquisition and channel
                img = self.loader.get_image(current_acq_id, channel)
                # Apply arcsinh to get the transformed range
                transformed_img = arcsinh_normalize(img, self.arcsinh_cofactor.value())
                vmin = float(np.min(transformed_img))
                vmax = float(np.max(transformed_img))
                
                # Update spinboxes
                self.min_spinbox.blockSignals(True)
                self.max_spinbox.blockSignals(True)
                self.min_spinbox.setValue(vmin)
                self.max_spinbox.setValue(vmax)
                self.min_spinbox.blockSignals(False)
                self.max_spinbox.blockSignals(False)
                
                # Make spinboxes uneditable
                self.min_spinbox.setEnabled(False)
                self.max_spinbox.setEnabled(False)
                
        except Exception as e:
            print(f"Error updating spinbox for arcsinh: {e}")
            # On error, make spinboxes editable
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)

    def _restore_default_range(self):
        """Restore the default range of the image when arcsinh is turned off."""
        try:
            if self.link_chk.isChecked():
                # Linked mode - use global range across all images
                channel = self.scaling_channel_combo.currentText()
                if not channel or not self.selected_acquisitions:
                    return
                
                # Get all images for this channel (without arcsinh transformation)
                imgs = [self.loader.get_image(aq, channel) for aq in self.selected_acquisitions]
                vmin = float(min(np.min(im) for im in imgs))
                vmax = float(max(np.max(im) for im in imgs))
                
                # Update spinboxes
                self.min_spinbox.blockSignals(True)
                self.max_spinbox.blockSignals(True)
                self.min_spinbox.setValue(vmin)
                self.max_spinbox.setValue(vmax)
                self.min_spinbox.blockSignals(False)
                self.max_spinbox.blockSignals(False)
                
            else:
                # Unlinked mode - use range for currently selected image
                current_acq_id = self.image_combo.currentData()
                channel = self.scaling_channel_combo.currentText()
                if not current_acq_id or not channel:
                    return
                
                # Get image for this acquisition and channel (without arcsinh transformation)
                img = self.loader.get_image(current_acq_id, channel)
                vmin = float(np.min(img))
                vmax = float(np.max(img))
                
                # Update spinboxes
                self.min_spinbox.blockSignals(True)
                self.max_spinbox.blockSignals(True)
                self.min_spinbox.setValue(vmin)
                self.max_spinbox.setValue(vmax)
                self.min_spinbox.blockSignals(False)
                self.max_spinbox.blockSignals(False)
                
        except Exception as e:
            print(f"Error restoring default range: {e}")

    def _refresh_scaling_channel_options(self):
        """Populate scaling channel combo with currently selected channels (RGB or single)."""
        self.scaling_channel_combo.blockSignals(True)
        self.scaling_channel_combo.clear()
        channels_for_scaling = []
        if self.rgb_mode_chk.isChecked():
            def _checked(lst: QtWidgets.QListWidget):
                vals = []
                for i in range(lst.count()):
                    it = lst.item(i)
                    if it.checkState() == Qt.Checked:
                        vals.append(it.text())
                return vals
            seen = set()
            for ch in _checked(self.red_list) + _checked(self.green_list) + _checked(self.blue_list):
                if ch not in seen:
                    channels_for_scaling.append(ch)
                    seen.add(ch)
        else:
            if self.channel_combo.currentText():
                channels_for_scaling = [self.channel_combo.currentText()]
        self.scaling_channel_combo.addItems(channels_for_scaling)
        self.scaling_channel_combo.blockSignals(False)
        
        # Initialize previous scaling channel if not set
        if not self.previous_scaling_channel and channels_for_scaling:
            self.previous_scaling_channel = channels_for_scaling[0]

    def _comparison_auto_range(self):
        """Set min/max values to the current images' range in comparison mode."""
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        try:
            # Get all images to determine range
            images = []
            for acq_id in self.selected_acquisitions:
                img = self.loader.get_image(acq_id, channel)
                images.append(img)
            
            if images:
                # Calculate global range across all images
                global_min = min(np.min(img) for img in images)
                global_max = max(np.max(img) for img in images)
                
                self.min_spinbox.setValue(float(global_min))
                self.max_spinbox.setValue(float(global_max))
        except Exception as e:
            print(f"Error getting comparison range: {e}")

    def _comparison_auto_contrast(self):
        """Set scaling to maximize contrast using percentile-based scaling in comparison mode."""
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        
        if self.link_chk.isChecked():
            # Linked scaling - use global percentiles across all images
            try:
                images = []
                for acq_id in self.selected_acquisitions:
                    img = self.loader.get_image(acq_id, channel)
                    images.append(img)
                
                if images:
                    all_pixels = np.concatenate([img.flatten() for img in images])
                    min_val = float(np.percentile(all_pixels, 1))
                    max_val = float(np.percentile(all_pixels, 99))
                    
                    self.min_spinbox.setValue(min_val)
                    self.max_spinbox.setValue(max_val)
            except Exception as e:
                print(f"Error in comparison auto contrast: {e}")
        else:
            # Individual scaling - use percentiles for selected image only
            current_acq_id = self.image_combo.currentData()
            if not current_acq_id:
                return
            
            try:
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.percentile(img, 1))
                max_val = float(np.percentile(img, 99))
                
                self.min_spinbox.setValue(min_val)
                self.max_spinbox.setValue(max_val)
            except Exception as e:
                print(f"Error in individual auto contrast: {e}")

    def _comparison_default_range(self):
        """Set scaling to the images' actual min/max range in comparison mode."""
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        
        if self.link_chk.isChecked():
            # Linked scaling - use global range across all images
            try:
                images = []
                for acq_id in self.selected_acquisitions:
                    img = self.loader.get_image(acq_id, channel)
                    images.append(img)
                
                if images:
                    global_min = min(np.min(img) for img in images)
                    global_max = max(np.max(img) for img in images)
                    
                    self.min_spinbox.setValue(float(global_min))
                    self.max_spinbox.setValue(float(global_max))
            except Exception as e:
                print(f"Error in comparison default range: {e}")
        else:
            # Individual scaling - use range for selected image only
            current_acq_id = self.image_combo.currentData()
            if not current_acq_id:
                return
            
            try:
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
                
                self.min_spinbox.setValue(min_val)
                self.max_spinbox.setValue(max_val)
            except Exception as e:
                print(f"Error in individual default range: {e}")

    def _apply_comparison_scaling(self):
        """Apply the current scaling settings and refresh display in comparison mode."""
        if not self.link_chk.isChecked():
            # For individual scaling, save the scaling for the selected image
            self._save_image_scaling()
        else:
            # For linked scaling, save the current values for the current channel
            current_channel = self.scaling_channel_combo.currentText()
            if current_channel:
                self.channel_linked_scaling[current_channel] = {
                    'min': self.min_spinbox.value(),
                    'max': self.max_spinbox.value()
                }
        self._update_display()

    def _on_link_contrast_toggled(self):
        """Handle link contrast checkbox toggle in comparison mode."""
        # Update custom scaling controls based on link contrast setting
        if self.link_chk.isChecked():
            # Link contrast ON - custom scaling controls are enabled
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.default_range_btn.setEnabled(True)
            self.image_combo.setEnabled(False)  # No image selection needed for linked scaling
            if hasattr(self, 'image_selection_row'):
                self.image_selection_row.setVisible(False)
            # In linked mode, scaling applies per channel globally
            self.scaling_channel_row.setVisible(True)
            # Disable arcsinh in linked mode
            self.arcsinh_chk.setEnabled(False)
            self.arcsinh_cofactor.setEnabled(False)
        else:
            # Link contrast OFF - individual scaling, enable image selection
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.default_range_btn.setEnabled(True)
            self.image_combo.setEnabled(True)  # Enable image selection for individual scaling
            self._update_image_combo()
            if hasattr(self, 'image_selection_row'):
                self.image_selection_row.setVisible(True)
            self.scaling_channel_row.setVisible(True)
            # Re-enable arcsinh in unlinked mode
            self.arcsinh_chk.setEnabled(True)
            self.arcsinh_cofactor.setEnabled(True)
        
        # Update spinbox for arcsinh if needed
        self._update_spinbox_for_arcsinh()
        
        # Refresh display
        self._update_display()

    def _update_image_combo(self):
        """Update the image selection combo box with selected acquisitions."""
        self.image_combo.clear()
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        for acq_id in self.selected_acquisitions:
            acq_info = next((ai for ai in self.acqs if ai.id == acq_id), None)
            if acq_info:
                label = f"{acq_info.name}" + (f" ({acq_info.well})" if acq_info.well else "")
                self.image_combo.addItem(label, acq_id)
        
        # Select first image if available
        if self.image_combo.count() > 0:
            self.image_combo.setCurrentIndex(0)
            self._load_image_scaling()

    def _on_image_selection_changed(self):
        """Handle changes to the image selection for individual scaling."""
        if not self.link_chk.isChecked() and self.custom_scaling_chk.isChecked():
            # Save the current image's scaling before switching
            self._save_image_scaling()
            
            # Load the new image's scaling
            self._load_image_scaling()
            
            # Load the arcsinh state for the new image
            self._load_image_arcsinh_state()
            
            # Update spinbox for arcsinh if needed
            self._update_spinbox_for_arcsinh()
            
            self._update_display()

    def _load_image_scaling(self):
        """Load scaling values for the currently selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id or not self.channel_combo.currentText():
            return
        
        # Determine which channel's scaling to load
        ch = self.scaling_channel_combo.currentText() or self.channel_combo.currentText()
        per_img = self.channel_per_image_scaling.get(ch, {})
        if current_acq_id in per_img:
            # Load saved values
            min_val = per_img[current_acq_id]['min']
            max_val = per_img[current_acq_id]['max']
        else:
            # Use default range (image's own min/max)
            try:
                channel = ch
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
            except Exception as e:
                print(f"Error loading image scaling: {e}")
                return
        
        # Update spinboxes
        self.min_spinbox.setValue(min_val)
        self.max_spinbox.setValue(max_val)

    def _load_image_arcsinh_state(self):
        """Load arcsinh state for the currently selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id:
            return
        
        # Load the arcsinh state for this image
        image_arcsinh = self.image_arcsinh_state.get(current_acq_id, {})
        is_enabled = image_arcsinh.get('enabled', False)
        cofactor = image_arcsinh.get('cofactor', self.arcsinh_cofactor.value())
        
        # Update the UI to reflect the image's arcsinh state
        self.arcsinh_chk.setChecked(is_enabled)
        self.arcsinh_cofactor.setValue(cofactor)

    def _save_image_scaling(self):
        """Save current scaling values for the selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id:
            return
        
        min_val = self.min_spinbox.value()
        max_val = self.max_spinbox.value()
        ch = self.scaling_channel_combo.currentText() or self.channel_combo.currentText()
        
        # Save the current values
        per_img = self.channel_per_image_scaling.get(ch, {})
        per_img[current_acq_id] = {'min': min_val, 'max': max_val}
        self.channel_per_image_scaling[ch] = per_img

    def _start_prefetch_selected(self):
        """Prefetch all channels for up to the first N selected acquisitions."""
        if not self.selected_acquisitions:
            return
        # Limit to first N acquisitions to manage memory
        target_acqs = self.selected_acquisitions[: self._prefetch_limit]
        # Remove cache entries for acquisitions no longer within the limit
        allowed = set(target_acqs)
        keys_to_remove = [k for k in self.image_cache.keys() if k[0] not in allowed]
        for k in keys_to_remove:
            try:
                del self.image_cache[k]
            except Exception:
                pass
        # Prefetch per acquisition
        for acq_id in target_acqs:
            if acq_id in self._prefetch_inflight:
                continue
            try:
                ai = next((a for a in self.acqs if a.id == acq_id), None)
                channels = ai.channels if ai else []
                # Load all channels at once for this acquisition
                stack = self.loader.get_all_channels(acq_id)
                for idx, ch_name in enumerate(channels):
                    try:
                        self.image_cache[(acq_id, ch_name)] = stack[..., idx]
                    except Exception:
                        continue
                self._manage_cache_size()
            except Exception as e:
                print(f"Prefetch error for {acq_id}: {e}")

    def _refresh_markers(self):
        """Legacy method - redirect to _update_channel_combo."""
        self._update_channel_combo()