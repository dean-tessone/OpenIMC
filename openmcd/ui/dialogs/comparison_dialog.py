from typing import List

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

# Data types
from openmcd.data.mcd_loader import AcquisitionInfo, MCDLoader  # noqa: F401
from openmcd.ui.utils import combine_channels
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
        self.max_cache_size = 50  # Limit cache size to prevent memory issues
        
        # Store last selected channel for auto-selection
        self.last_selected_channel: Optional[str] = None
        
        # Store per-image scaling values for individual scaling
        self.image_scaling = {}  # {acquisition_id: {'min': value, 'max': value}}
        
        self.setMinimumSize(1000, 700)
        
        # Create UI
        self._create_ui()
        
        # Start with empty selection - user must select acquisitions of interest

    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Control panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control_panel)
        
        # Acquisition selection
        acq_group = QtWidgets.QGroupBox("Acquisitions")
        acq_layout = QtWidgets.QVBoxLayout(acq_group)
        
        acq_layout.addWidget(QtWidgets.QLabel("Available:"))
        self.available_acq_list = QtWidgets.QListWidget()
        self.available_acq_list.setMaximumHeight(150)
        for ai in self.acqs:
            label = f"{ai.name}" + (f" ({ai.well})" if ai.well else "")
            item = QtWidgets.QListWidgetItem(label)
            item.setData(Qt.UserRole, ai.id)
            self.available_acq_list.addItem(item)
        
        acq_buttons = QtWidgets.QHBoxLayout()
        self.add_acq_btn = QtWidgets.QPushButton("Add →")
        self.remove_acq_btn = QtWidgets.QPushButton("← Remove")
        acq_buttons.addWidget(self.add_acq_btn)
        acq_buttons.addWidget(self.remove_acq_btn)
        
        acq_layout.addWidget(self.available_acq_list)
        acq_layout.addLayout(acq_buttons)
        
        acq_layout.addWidget(QtWidgets.QLabel("Selected:"))
        self.acq_list = QtWidgets.QListWidget()
        self.acq_list.setMaximumHeight(150)
        
        acq_layout.addWidget(self.acq_list)
        control_layout.addWidget(acq_group)
        
        # Channel selection
        channel_group = QtWidgets.QGroupBox("Channel")
        channel_layout = QtWidgets.QVBoxLayout(channel_group)
        
        self.channel_combo = QtWidgets.QComboBox()
        channel_layout.addWidget(QtWidgets.QLabel("Marker channel:"))
        channel_layout.addWidget(self.channel_combo)
        
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
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        
        # Image selection for individual scaling
        image_selection_layout = QtWidgets.QHBoxLayout()
        image_selection_layout.addWidget(QtWidgets.QLabel("Image:"))
        self.image_combo = QtWidgets.QComboBox()
        self.image_combo.currentTextChanged.connect(self._on_image_selection_changed)
        self.image_combo.currentIndexChanged.connect(self._on_image_selection_changed)
        image_selection_layout.addWidget(self.image_combo)
        image_selection_layout.addStretch()
        scaling_layout.addLayout(image_selection_layout)
        
        # Min/Max controls
        minmax_layout = QtWidgets.QHBoxLayout()
        minmax_layout.addWidget(QtWidgets.QLabel("Min:"))
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(-999999, 999999)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        minmax_layout.addWidget(self.min_spinbox)
        
        minmax_layout.addWidget(QtWidgets.QLabel("Max:"))
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(-999999, 999999)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(100.0)
        minmax_layout.addWidget(self.max_spinbox)
        
        scaling_layout.addLayout(minmax_layout)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.auto_contrast_btn = QtWidgets.QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self._comparison_auto_contrast)
        button_layout.addWidget(self.auto_contrast_btn)
        
        self.default_range_btn = QtWidgets.QPushButton("Original Range")
        self.default_range_btn.clicked.connect(self._comparison_default_range)
        button_layout.addWidget(self.default_range_btn)
        
        self.apply_scaling_btn = QtWidgets.QPushButton("Apply")
        self.apply_scaling_btn.clicked.connect(self._apply_comparison_scaling)
        self.apply_scaling_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.apply_scaling_btn)
        
        scaling_layout.addLayout(button_layout)
        self.scaling_frame.setVisible(False)
        
        options_layout.addWidget(self.scaling_frame)
        
        control_layout.addWidget(channel_group)
        control_layout.addWidget(options_group)
        
        layout.addWidget(control_panel)
        
        # Image display area
        self.image_scroll = QtWidgets.QScrollArea()
        self.image_widget = QtWidgets.QWidget()
        self.image_layout = QtWidgets.QGridLayout(self.image_widget)
        self.image_scroll.setWidget(self.image_widget)
        self.image_scroll.setWidgetResizable(True)
        layout.addWidget(self.image_scroll)
        
        # Connect signals
        self.add_acq_btn.clicked.connect(self._add_acquisition)
        self.remove_acq_btn.clicked.connect(self._remove_acquisition)
        self.channel_combo.currentTextChanged.connect(self._update_display)
        self.link_chk.toggled.connect(self._on_link_contrast_toggled)
        self.grayscale_chk.toggled.connect(self._update_display)
        self.acq_list.itemSelectionChanged.connect(self._update_display)
        
        # Initialize channel combo when acquisitions are added
        self._update_channel_combo()

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
                
                # Preload images for this acquisition in background
        self._preload_images(acq_id)
        if self.custom_scaling_chk.isChecked() and not self.link_chk.isChecked():
            self._update_image_combo()

    def _remove_acquisition(self):
        current_item = self.acq_list.currentItem()
        if current_item:
            acq_id = current_item.data(Qt.UserRole)
            self.selected_acquisitions.remove(acq_id)
            self.acq_list.takeItem(self.acq_list.row(current_item))
            self._update_channel_combo()
            if self.custom_scaling_chk.isChecked() and not self.link_chk.isChecked():
                self._update_image_combo()

    def _update_display(self):
        # Clear existing images
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().setParent(None)
        
        if not self.selected_acquisitions or not self.channel_combo.currentText():
            return
        
        channel = self.channel_combo.currentText()
        self.last_selected_channel = channel
        grayscale = self.grayscale_chk.isChecked()
        link_contrast = self.link_chk.isChecked()
        
        # Load images
        images = []
        titles = []
        for acq_id in self.selected_acquisitions:
            try:
                img = self.loader.get_image(acq_id, channel)
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
        
        # Calculate global min/max if link contrast is enabled
        vmin = None
        vmax = None
        if link_contrast and len(images) > 1:
            if custom_scaling_enabled:
                # Use custom scaling for linked contrast
                vmin = custom_min
                vmax = custom_max
            else:
                # Use global min/max for fair comparison
                vmin = min(np.min(img) for img in images)
                vmax = max(np.max(img) for img in images)
        elif not link_contrast and custom_scaling_enabled:
            # For individual scaling with custom scaling enabled,
            # each image can have its own custom scaling range
            # We'll handle this per-image in the display loop
            pass
        
        # Display images
        for i, (img, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            
            # Create canvas
            canvas = MplCanvas(width=4, height=4, dpi=100)
            
            # Determine scaling for this image
            img_vmin = vmin
            img_vmax = vmax
            
            if img_vmin is None or img_vmax is None:
                # Individual scaling - check for per-image custom scaling
                acq_id = self.selected_acquisitions[i]
                if custom_scaling_enabled and not link_contrast and acq_id in self.image_scaling:
                    # Use custom scaling for this specific image
                    img_vmin = self.image_scaling[acq_id]['min']
                    img_vmax = self.image_scaling[acq_id]['max']
                else:
                    # Use image's own min/max range
                    img_vmin = np.min(img)
                    img_vmax = np.max(img)
            
            # Display image with appropriate scaling
            im = canvas.ax.imshow(img, interpolation="nearest", 
                                cmap='gray' if grayscale else 'viridis',
                                vmin=img_vmin, vmax=img_vmax)
            
            canvas.ax.set_title(title, fontsize=10)
            canvas.ax.axis("off")
            
            # Add colorbar
            cbar = canvas.fig.colorbar(im, ax=canvas.ax, shrink=0.8, aspect=20)
            
            self.image_layout.addWidget(canvas, row, col)
        
    def _update_channel_combo(self):
        """Update the channel combo box based on selected acquisitions."""
        old_channel = self.channel_combo.currentText()
        self.channel_combo.clear()
        if self.selected_acquisitions:
            # Get channels from first selected acquisition
            ai = next(a for a in self.acqs if a.id == self.selected_acquisitions[0])
            self.channel_combo.addItems(ai.channels)
            
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
        
        # Update control states based on link contrast
        self._on_link_contrast_toggled()

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
        self._update_display()

    def _on_link_contrast_toggled(self):
        """Handle link contrast checkbox toggle in comparison mode."""
        # Update custom scaling controls based on link contrast setting
        if self.link_chk.isChecked():
            # Link contrast ON - custom scaling controls are enabled
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.auto_contrast_btn.setEnabled(True)
            self.default_range_btn.setEnabled(True)
            self.apply_scaling_btn.setEnabled(True)
            self.image_combo.setEnabled(False)  # No image selection needed for linked scaling
        else:
            # Link contrast OFF - individual scaling, enable image selection
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.auto_contrast_btn.setEnabled(True)
            self.default_range_btn.setEnabled(True)
            self.apply_scaling_btn.setEnabled(True)
            self.image_combo.setEnabled(True)  # Enable image selection for individual scaling
            self._update_image_combo()
        
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
            self._load_image_scaling()

    def _load_image_scaling(self):
        """Load scaling values for the currently selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id or not self.channel_combo.currentText():
            return
        
        if current_acq_id in self.image_scaling:
            # Load saved values
            min_val = self.image_scaling[current_acq_id]['min']
            max_val = self.image_scaling[current_acq_id]['max']
        else:
            # Use default range (image's own min/max)
            try:
                channel = self.channel_combo.currentText()
                img = self.loader.get_image(current_acq_id, channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
            except Exception as e:
                print(f"Error loading image scaling: {e}")
                return
        
        # Update spinboxes
        self.min_spinbox.setValue(min_val)
        self.max_spinbox.setValue(max_val)

    def _save_image_scaling(self):
        """Save current scaling values for the selected image."""
        current_acq_id = self.image_combo.currentData()
        if not current_acq_id:
            return
        
        min_val = self.min_spinbox.value()
        max_val = self.max_spinbox.value()
        
        self.image_scaling[current_acq_id] = {'min': min_val, 'max': max_val}

    def _preload_images(self, acq_id):
        """Preload images for an acquisition in the background."""
        if not self.channel_combo.currentText():
            return
            
        channel = self.channel_combo.currentText()
        cache_key = (acq_id, channel)
        
        if cache_key not in self.image_cache:
            try:
                img = self.loader.get_image(acq_id, channel)
                self.image_cache[cache_key] = img
                self._manage_cache_size()
            except Exception as e:
                print(f"Error preloading image for {acq_id}: {e}")

    def _refresh_markers(self):
        """Legacy method - redirect to _update_channel_combo."""
        self._update_channel_combo()