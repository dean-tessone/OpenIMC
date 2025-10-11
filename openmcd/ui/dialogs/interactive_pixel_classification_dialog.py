"""
Interactive UI for Interactive pixel level classification segmentation.

This dialog provides a brush tool for painting training labels,
live preview of probability maps, and conversion to instance segmentation.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox, QProgressBar, QMessageBox, QFileDialog, QScrollArea

from openmcd.data.mcd_loader import MCDLoader
from openmcd.processing.interactive_pixel_features import InteractivePixelFeatureComputer
from openmcd.processing.interactive_pixel_classifier import InteractivePixelClassifier
from openmcd.processing.interactive_pixel_inference import InteractivePixelInferencePipeline
from openmcd.processing.interactive_pixel_instances import InteractivePixelInstanceSegmenter, create_probability_overlay

# Cellpose imports
try:
    from cellpose import models
    _HAVE_CELLPOSE = True
except ImportError:
    _HAVE_CELLPOSE = False


class BrushWidget(QWidget):
    """Interactive widget for painting training labels."""
    
    # Signals
    label_painted = pyqtSignal(np.ndarray, np.ndarray)  # coordinates, labels
    
    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.image = image
        self.height, self.width = image.shape[:2]
        
        # Brush settings
        self.brush_size = 5
        self.current_label = 0  # 0=background, 1=nucleus, 2=cytoplasm
        self.is_painting = False
        self.is_erasing = False
        self._is_right_clicking = False
        
        # Label colors
        self.label_colors = {
            0: QColor(0, 255, 255, 128),  # Background - cyan (more visible than black)
            1: QColor(255, 0, 0, 128),    # Nucleus - red
            2: QColor(0, 255, 0, 128)     # Cytoplasm - green
        }
        
        # Initialize label maps for each class (0 = unlabeled, 1 = labeled)
        self.label_maps = {
            0: np.zeros((self.height, self.width), dtype=np.int32),  # Background
            1: np.zeros((self.height, self.width), dtype=np.int32),  # Nucleus
            2: np.zeros((self.height, self.width), dtype=np.int32)   # Cytoplasm
        }
        
        # Set up widget
        self.setFixedSize(self.width, self.height)
        self.setMouseTracking(True)
        
        # Create pixmap for display
        self.update_display()
    
    def set_zoom(self, zoom_factor: float):
        """Set zoom factor and update display."""
        self.zoom_factor = zoom_factor
        new_width = int(self.width * zoom_factor)
        new_height = int(self.height * zoom_factor)
        self.setFixedSize(new_width, new_height)
        self.update()
    
    def set_brush_size(self, size: int):
        """Set brush size."""
        self.brush_size = max(1, size)
        self.update()
    
    def set_current_label(self, label: int):
        """Set current label class."""
        self.current_label = label
        self.update()
    
    def update_label_display(self):
        """Update the display to show/hide labels based on current selection."""
        self.update()
    
    def clear_labels(self):
        """Clear all painted labels."""
        for label_id in self.label_maps:
            self.label_maps[label_id].fill(0)  # Reset to unlabeled state
        self.update_display()
        self.update()
    
    def get_label_maps(self) -> Dict[int, np.ndarray]:
        """Get current label maps for all classes."""
        return {label_id: label_map.copy() for label_id, label_map in self.label_maps.items()}
    
    def get_label_map(self) -> np.ndarray:
        """Get combined label map (for backward compatibility)."""
        # Create a combined label map with priority: nucleus > cytoplasm > background
        combined = np.full((self.height, self.width), -1, dtype=np.int32)
        for label_id in [1, 2, 0]:  # Nucleus, Cytoplasm, Background
            combined[self.label_maps[label_id] == 1] = label_id
        return combined
    
    def update_display(self):
        """Update the display pixmap."""
        # Create RGB image from grayscale
        if self.image.ndim == 2:
            rgb_image = np.stack([self.image] * 3, axis=2)
        else:
            rgb_image = self.image.copy()
        
        # Normalize to 0-255
        rgb_image = ((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8) * 255).astype(np.uint8)
        
        # Create QImage
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create pixmap
        self.pixmap = QPixmap.fromImage(q_image)
    
    def paintEvent(self, event):
        """Paint the widget."""
        painter = QPainter(self)
        
        # Get zoom factor
        zoom_factor = getattr(self, 'zoom_factor', 1.0)
        
        # Draw base image scaled
        if hasattr(self, 'pixmap') and self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_pixmap)
        
        # Draw labels scaled - show all labeled classes
        for y in range(self.height):
            for x in range(self.width):
                # Check each class
                for label_id in [1, 2, 0]:  # Nucleus, Cytoplasm, Background (in priority order)
                    if self.label_maps[label_id][y, x] == 1:  # This pixel is labeled for this class
                        # Show the label if it's the current selection or if showing all
                        if self.current_label == 0 or label_id == self.current_label:
                            color = self.label_colors[label_id]
                            painter.setPen(QPen(color, max(1, int(zoom_factor))))
                            painter.setBrush(QBrush(color))
                            # Scale coordinates
                            scaled_x = int(x * zoom_factor)
                            scaled_y = int(y * zoom_factor)
                            painter.drawPoint(scaled_x, scaled_y)
                        break  # Only show the first (highest priority) label for this pixel
        
        # Draw brush preview
        if self.underMouse():
            mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
            if 0 <= mouse_pos.x() < self.width and 0 <= mouse_pos.y() < self.height:
                # Show different colors for painting vs erasing
                if self.is_erasing or (hasattr(self, '_is_right_clicking') and self._is_right_clicking):
                    # Red color for erasing
                    color = QColor(255, 0, 0, 200)  # Red with transparency
                else:
                    # Normal label color for painting
                    color = self.label_colors[self.current_label]
                
                painter.setPen(QPen(color, max(2, int(2 * zoom_factor))))
                painter.setBrush(Qt.NoBrush)
                scaled_brush_size = int(self.brush_size * zoom_factor)
                painter.drawEllipse(mouse_pos.x() - scaled_brush_size//2, 
                                  mouse_pos.y() - scaled_brush_size//2,
                                  scaled_brush_size, scaled_brush_size)
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.is_painting = True
            self.is_erasing = False
            self._is_right_clicking = False
            self.paint_at_position(event.pos())
        elif event.button() == Qt.RightButton:
            self.is_erasing = True
            self.is_painting = False
            self._is_right_clicking = True
            self.erase_at_position(event.pos())
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.is_painting:
            self.paint_at_position(event.pos())
        elif self.is_erasing:
            self.erase_at_position(event.pos())
        self.update()  # Update brush preview
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.is_painting = False
        elif event.button() == Qt.RightButton:
            self.is_erasing = False
            self._is_right_clicking = False
    
    def contextMenuEvent(self, event):
        """Prevent context menu from appearing on right-click."""
        # Do nothing - we handle right-click for erasing
        pass
    
    def paint_at_position(self, pos):
        """Paint label at given position."""
        # Get zoom factor
        zoom_factor = getattr(self, 'zoom_factor', 1.0)
        
        # Convert screen coordinates to image coordinates
        x = int(pos.x() / zoom_factor)
        y = int(pos.y() / zoom_factor)
        
        # Paint in circular brush - add to current class without removing others
        for dy in range(-self.brush_size//2, self.brush_size//2 + 1):
            for dx in range(-self.brush_size//2, self.brush_size//2 + 1):
                px, py = x + dx, y + dy
                if (0 <= px < self.width and 0 <= py < self.height and
                    dx*dx + dy*dy <= (self.brush_size//2)**2):
                    # Set the current label (don't remove others)
                    self.label_maps[self.current_label][py, px] = 1
        
        self.update()
        
        # Emit signal with painted coordinates
        painted_coords = np.column_stack(np.where(self.label_maps[self.current_label] == 1))
        painted_labels = np.full(len(painted_coords), self.current_label)
        self.label_painted.emit(painted_coords, painted_labels)
    
    def erase_at_position(self, pos):
        """Erase labels at given position."""
        # Get zoom factor
        zoom_factor = getattr(self, 'zoom_factor', 1.0)
        
        # Convert screen coordinates to image coordinates
        x = int(pos.x() / zoom_factor)
        y = int(pos.y() / zoom_factor)
        
        # Erase in circular brush - remove from all classes
        for dy in range(-self.brush_size//2, self.brush_size//2 + 1):
            for dx in range(-self.brush_size//2, self.brush_size//2 + 1):
                px, py = x + dx, y + dy
                if (0 <= px < self.width and 0 <= py < self.height and
                    dx*dx + dy*dy <= (self.brush_size//2)**2):
                    # Remove from all classes
                    for label_id in self.label_maps:
                        self.label_maps[label_id][py, px] = 0
        
        self.update()
        
        # Emit signal with erased coordinates (empty arrays since we're erasing)
        self.label_painted.emit(np.array([]).reshape(0, 2), np.array([]))


class InteractivePixelClassificationDialog(QtWidgets.QDialog):
    """Main dialog for Interactive pixel level classification segmentation."""
    
    def __init__(self, img_stack: np.ndarray, channel_names: List[str], parent=None, preprocessing_config=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive pixel level classification Segmentation")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.img_stack = img_stack
        self.channel_names = channel_names
        self.height, self.width, self.n_channels = img_stack.shape
        self.preprocessing_config = preprocessing_config
        
        # Process channels based on preprocessing config
        self.nucleus_channel = None
        self.cytoplasm_channel = None
        self._process_preprocessing_config()
        
        # Initialize components
        self.feature_computer = InteractivePixelFeatureComputer()
        self.classifier = InteractivePixelClassifier()
        self.inference_pipeline = None
        
        # Current state
        self.current_features = None
        self.current_probability_maps = None
        self.current_instances = None
        
        # Create UI
        self._create_ui()
        
        # Initialize with first channel
        self._update_display_channel()
    
    def _process_preprocessing_config(self):
        """Process preprocessing configuration to create nucleus and cytoplasm channels."""
        if not self.preprocessing_config:
            return
        
        # Get channel lists
        nuclear_channels = self.preprocessing_config.get('nuclear_channels', [])
        cyto_channels = self.preprocessing_config.get('cyto_channels', [])
        
        # Get combination methods
        nuclear_combo_method = self.preprocessing_config.get('nuclear_combo_method', 'mean')
        cyto_combo_method = self.preprocessing_config.get('cyto_combo_method', 'mean')
        
        # Get weights
        nuclear_weights = self.preprocessing_config.get('nuclear_weights', {})
        cyto_weights = self.preprocessing_config.get('cyto_weights', {})
        
        # Create nucleus channel
        if nuclear_channels:
            self.nucleus_channel = self._combine_channels(nuclear_channels, nuclear_combo_method, nuclear_weights)
        
        # Create cytoplasm channel
        if cyto_channels:
            self.cytoplasm_channel = self._combine_channels(cyto_channels, cyto_combo_method, cyto_weights)
    
    def _combine_channels(self, channel_names: List[str], method: str, weights: Dict[str, float]) -> np.ndarray:
        """Combine multiple channels into a single channel with proper preprocessing."""
        # Get channel indices
        channel_indices = [self.channel_names.index(name) for name in channel_names if name in self.channel_names]
        
        if not channel_indices:
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        # Extract channel data
        channel_data = self.img_stack[:, :, channel_indices]
        
        # Apply preprocessing to each channel before combining
        processed_channels = []
        for i, channel_name in enumerate([name for name in channel_names if name in self.channel_names]):
            channel_img = channel_data[:, :, i]
            
            # Apply denoising first if configured
            if self.preprocessing_config:
                denoise_source = self.preprocessing_config.get('denoise_source', 'none')
                custom_denoise_settings = self.preprocessing_config.get('custom_denoise_settings', {})
                
                if denoise_source == 'custom' and custom_denoise_settings:
                    channel_img = self._apply_denoise_to_channel(channel_img, channel_name, custom_denoise_settings.get(channel_name, {}))
                elif denoise_source == 'viewer':
                    # Apply viewer denoising if available
                    channel_img = self._apply_viewer_denoise(channel_img, channel_name)
            
            # Apply normalization after denoising
            if self.preprocessing_config:
                channel_img = self._apply_normalization_to_channel(channel_img, channel_name)
            
            processed_channels.append(channel_img)
        
        # Stack processed channels
        processed_channel_data = np.stack(processed_channels, axis=2)
        
        # Combine channels using the specified method
        if method == 'mean':
            if weights:
                # Weighted average
                weight_values = np.array([weights.get(name, 1.0) for name in channel_names if name in self.channel_names])
                weight_values = weight_values / np.sum(weight_values)  # Normalize
                combined = np.average(processed_channel_data, axis=2, weights=weight_values)
            else:
                combined = np.mean(processed_channel_data, axis=2)
        elif method == 'weighted':
            if weights:
                weight_values = np.array([weights.get(name, 1.0) for name in channel_names if name in self.channel_names])
                weight_values = weight_values / np.sum(weight_values)  # Normalize
                combined = np.average(processed_channel_data, axis=2, weights=weight_values)
            else:
                combined = np.mean(processed_channel_data, axis=2)
        elif method == 'pca1':
            # Use first principal component
            reshaped = processed_channel_data.reshape(-1, processed_channel_data.shape[2])
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(reshaped)
            combined = pca_result.reshape(processed_channel_data.shape[:2])
        else:
            # Default to mean
            combined = np.mean(processed_channel_data, axis=2)
        
        return combined.astype(np.float32)
    
    def _apply_denoise_to_channel(self, channel_img: np.ndarray, channel_name: str, denoise_settings: dict) -> np.ndarray:
        """Apply denoising to a single channel based on settings."""
        if not denoise_settings:
            return channel_img
        
        try:
            from skimage import morphology, filters
            from skimage.filters import gaussian, median
            from skimage.morphology import disk
            from skimage.restoration import denoise_nl_means, estimate_sigma
            from scipy import ndimage as ndi
        except ImportError:
            # If scikit-image is not available, return original image
            return channel_img
        
        result = channel_img.copy().astype(np.float32)
        
        # Hot pixel removal
        hot_config = denoise_settings.get("hot")
        if hot_config:
            method = hot_config.get("method", "median3")
            n_sd = float(hot_config.get("n_sd", 5.0))
            if method == "median3":
                # 3x3 median filter
                result = median(result, disk(1))
            elif method == "n_sd_local_median":
                # Replace pixels above N*local_std over local median
                try:
                    local_median = median(result, disk(1))
                except Exception:
                    local_median = ndi.median_filter(result, size=3)
                diff = result - local_median
                local_var = ndi.uniform_filter(diff * diff, size=3)
                local_std = np.sqrt(np.maximum(local_var, 1e-8))
                mask_hot = diff > (n_sd * local_std)
                result = np.where(mask_hot, local_median, result)
        
        # Speckle noise reduction
        speckle_config = denoise_settings.get("speckle")
        if speckle_config:
            method = speckle_config.get("method", "gaussian")
            sigma = float(speckle_config.get("sigma", 0.8))
            if method == "gaussian":
                result = gaussian(result, sigma=sigma)
            elif method == "nl_means":
                est = estimate_sigma(result)
                result = denoise_nl_means(result, h=est * sigma)
        
        # Background subtraction
        bg_config = denoise_settings.get("background")
        if bg_config:
            method = bg_config.get("method", "white_tophat")
            radius = int(bg_config.get("radius", 15))
            if method == "white_tophat":
                selem = disk(radius)
                result = morphology.white_tophat(result, selem)
            elif method == "black_tophat":
                selem = disk(radius)
                result = morphology.black_tophat(result, selem)
        
        return result
    
    def _apply_viewer_denoise(self, channel_img: np.ndarray, channel_name: str) -> np.ndarray:
        """Apply viewer denoising to a channel (placeholder - would need viewer settings)."""
        # For now, return the original image
        # In a full implementation, this would apply the same denoising
        # that's configured in the viewer for this channel
        return channel_img
    
    def _apply_normalization_to_channel(self, channel_img: np.ndarray, channel_name: str) -> np.ndarray:
        """Apply normalization to a single channel based on preprocessing config."""
        if not self.preprocessing_config:
            return channel_img
        
        normalization_method = self.preprocessing_config.get('normalization_method', 'None')
        
        if normalization_method == 'arcsinh':
            cofactor = self.preprocessing_config.get('arcsinh_cofactor', 5.0)
            return self._arcsinh_normalize(channel_img, cofactor)
        elif normalization_method == 'percentile_clip':
            p_low, p_high = self.preprocessing_config.get('percentile_params', (1.0, 99.0))
            return self._percentile_clip_normalize(channel_img, p_low, p_high)
        else:
            return channel_img
    
    def _arcsinh_normalize(self, arr: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
        """Apply arcsinh normalization to an array."""
        a = arr.astype(np.float32, copy=False)
        return np.arcsinh(a / cofactor)
    
    def _percentile_clip_normalize(self, arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
        """Apply percentile clipping normalization to an array."""
        a = arr.astype(np.float32, copy=False)
        vmin = np.percentile(a, p_low)
        vmax = np.percentile(a, p_high)
        clipped = np.clip(a, vmin, vmax)
        if vmax > vmin:
            normalized = (clipped - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(clipped)
        return normalized
    
    def _get_available_channels(self) -> List[str]:
        """Get list of channels available for annotation based on preprocessing config."""
        available_channels = []
        
        # Add nucleus channel if available
        if self.nucleus_channel is not None:
            available_channels.append("Nucleus")
        
        # Add cytoplasm channel if available
        if self.cytoplasm_channel is not None:
            available_channels.append("Cytoplasm")
        
        # If no processed channels, fall back to original channels
        if not available_channels:
            return self.channel_names
        
        return available_channels
    
    def _create_ui(self):
        """Create the user interface."""
        layout = QtWidgets.QHBoxLayout(self)
        
        # Left panel - controls
        left_panel = self._create_control_panel()
        layout.addWidget(left_panel, 1)
        
        # Right panel - image display
        right_panel = self._create_display_panel()
        layout.addWidget(right_panel, 3)
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Channel selection
        channel_group = QGroupBox("Channel Selection")
        channel_layout = QVBoxLayout(channel_group)
        
        self.channel_combo = QComboBox()
        # Only show channels that are used in preprocessing
        available_channels = self._get_available_channels()
        self.channel_combo.addItems(available_channels)
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(QLabel("Display Channel:"))
        channel_layout.addWidget(self.channel_combo)
        
        layout.addWidget(channel_group)
        
        # Brush settings
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QVBoxLayout(brush_group)
        
        # Label selection
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.addItems(["Background", "Nucleus", "Cytoplasm"])
        self.label_combo.currentIndexChanged.connect(self._on_label_changed)
        label_layout.addWidget(self.label_combo)
        brush_layout.addLayout(label_layout)
        
        # Brush size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 50)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        size_layout.addWidget(self.brush_size_spin)
        brush_layout.addLayout(size_layout)
        
        # Clear button
        self.clear_btn = QPushButton("Clear Labels")
        self.clear_btn.clicked.connect(self._clear_labels)
        brush_layout.addWidget(self.clear_btn)
        
        # Auto-nuclei button
        self.auto_nuclei_btn = QPushButton("Auto-Detect Nuclei")
        self.auto_nuclei_btn.clicked.connect(self._auto_detect_nuclei)
        self.auto_nuclei_btn.setToolTip("Use Cellpose nuclei model to automatically detect and label nuclei")
        if not _HAVE_CELLPOSE:
            self.auto_nuclei_btn.setEnabled(False)
            self.auto_nuclei_btn.setToolTip("Cellpose not available - install cellpose package")
        brush_layout.addWidget(self.auto_nuclei_btn)
        
        layout.addWidget(brush_group)
        
        # Training
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout(training_group)
        
        # Instructions
        instructions_label = QLabel("Use 'Auto-Detect Nuclei' to label nuclei, then paint cytoplasm and background. Right-click to erase.")
        instructions_label.setStyleSheet("QLabel { color: #666; font-size: 11px; font-style: italic; }")
        instructions_label.setWordWrap(True)
        training_layout.addWidget(instructions_label)
        
        self.train_btn = QPushButton("Train Classifier")
        self.train_btn.clicked.connect(self._train_classifier)
        training_layout.addWidget(self.train_btn)
        
        # Training info
        self.training_info = QLabel("No training data")
        self.training_info.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        training_layout.addWidget(self.training_info)
        
        layout.addWidget(training_group)
        
        # Inference
        inference_group = QGroupBox("Inference")
        inference_layout = QVBoxLayout(inference_group)
        
        self.infer_btn = QPushButton("Run Inference")
        self.infer_btn.clicked.connect(self._run_inference)
        self.infer_btn.setEnabled(False)
        inference_layout.addWidget(self.infer_btn)
        
        # Instance segmentation
        self.instance_btn = QPushButton("Convert to Instances")
        self.instance_btn.clicked.connect(self._convert_to_instances)
        self.instance_btn.setEnabled(False)
        inference_layout.addWidget(self.instance_btn)
        
        layout.addWidget(inference_group)
        
        # Watershed parameters
        watershed_group = QGroupBox("Watershed Parameters")
        watershed_layout = QVBoxLayout(watershed_group)
        
        # Min distance between seeds
        min_dist_layout = QHBoxLayout()
        min_dist_layout.addWidget(QLabel("Min Distance:"))
        self.min_distance_spin = QSpinBox()
        self.min_distance_spin.setRange(5, 50)
        self.min_distance_spin.setValue(10)
        self.min_distance_spin.setToolTip("Minimum distance between seed points (pixels)")
        min_dist_layout.addWidget(self.min_distance_spin)
        min_dist_layout.addStretch()
        watershed_layout.addLayout(min_dist_layout)
        
        # Threshold for seed detection
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Seed Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.3)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setToolTip("Minimum probability threshold for seed detection")
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        watershed_layout.addLayout(threshold_layout)
        
        # Min cell area
        min_area_layout = QHBoxLayout()
        min_area_layout.addWidget(QLabel("Min Cell Area:"))
        self.min_cell_area_spin = QSpinBox()
        self.min_cell_area_spin.setRange(10, 500)
        self.min_cell_area_spin.setValue(50)
        self.min_cell_area_spin.setToolTip("Minimum area for valid cell regions (pixels)")
        min_area_layout.addWidget(self.min_cell_area_spin)
        min_area_layout.addStretch()
        watershed_layout.addLayout(min_area_layout)
        
        # Max cell area
        max_area_layout = QHBoxLayout()
        max_area_layout.addWidget(QLabel("Max Cell Area:"))
        self.max_cell_area_spin = QSpinBox()
        self.max_cell_area_spin.setRange(100, 10000)
        self.max_cell_area_spin.setValue(2000)
        self.max_cell_area_spin.setToolTip("Maximum area for valid cell regions (pixels)")
        max_area_layout.addWidget(self.max_cell_area_spin)
        max_area_layout.addStretch()
        watershed_layout.addLayout(max_area_layout)
        
        # Nucleus expansion (for nucleus-only segmentation)
        expansion_layout = QHBoxLayout()
        expansion_layout.addWidget(QLabel("Nucleus Expansion:"))
        self.nucleus_expansion_spin = QSpinBox()
        self.nucleus_expansion_spin.setRange(1, 20)
        self.nucleus_expansion_spin.setValue(5)
        self.nucleus_expansion_spin.setToolTip("Expansion radius for nucleus-only segmentation (pixels)")
        expansion_layout.addWidget(self.nucleus_expansion_spin)
        expansion_layout.addStretch()
        watershed_layout.addLayout(expansion_layout)
        
        layout.addWidget(watershed_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.show_probabilities = QCheckBox("Show Probability Maps")
        self.show_probabilities.toggled.connect(self._update_display)
        display_layout.addWidget(self.show_probabilities)
        
        self.show_instances = QCheckBox("Show Instance Labels")
        self.show_instances.toggled.connect(self._update_display)
        display_layout.addWidget(self.show_instances)
        
        layout.addWidget(display_group)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_model_btn = QPushButton("Save Model")
        self.save_model_btn.clicked.connect(self._save_model)
        self.save_model_btn.setEnabled(False)
        button_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self._load_model)
        button_layout.addWidget(self.load_model_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self._save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
        return panel
    
    def _create_display_panel(self) -> QWidget:
        """Create the display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 400)  # 25% to 400%
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)
        
        # Zoom buttons
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setMaximumWidth(30)
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setMaximumWidth(30)
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_fit_btn = QPushButton("Fit")
        self.zoom_fit_btn.setMaximumWidth(40)
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        zoom_layout.addWidget(self.zoom_fit_btn)
        
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # Scroll area for image display
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)  # We'll control sizing manually
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scroll_area)
        
        # Image display widget (will contain the brush widget)
        self.image_widget = QWidget()
        self.image_widget.setMinimumSize(400, 400)
        self.scroll_area.setWidget(self.image_widget)
        
        # Brush widget
        self.brush_widget = None  # Will be created when channel is selected
        self.current_zoom = 1.0
        self.original_image_size = None
        
        return panel
    
    def _on_channel_changed(self):
        """Handle channel selection change."""
        self._update_display_channel()
    
    def _update_display_channel(self):
        """Update display to show selected channel."""
        channel_name = self.channel_combo.currentText()
        
        # Get the appropriate channel image
        if channel_name == "Nucleus" and self.nucleus_channel is not None:
            channel_img = self.nucleus_channel
        elif channel_name == "Cytoplasm" and self.cytoplasm_channel is not None:
            channel_img = self.cytoplasm_channel
        else:
            # Fallback to original channel
            channel_idx = self.channel_names.index(channel_name)
            channel_img = self.img_stack[:, :, channel_idx]
        
        # Create or update brush widget
        if self.brush_widget is None:
            self.brush_widget = BrushWidget(channel_img, self)
            self.brush_widget.label_painted.connect(self._on_label_painted)
            
            # Store original size
            self.original_image_size = (self.brush_widget.width, self.brush_widget.height)
            
            # Add brush widget to the image widget
            layout = QVBoxLayout(self.image_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.brush_widget)
        else:
            self.brush_widget.image = channel_img
            self.brush_widget.update_display()
        
        # Update zoom
        self._update_zoom()
        self._update_display()
    
    def _on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        self.current_zoom = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self._update_zoom()
    
    def _zoom_in(self):
        """Zoom in by 25%."""
        current_value = self.zoom_slider.value()
        new_value = min(400, current_value + 25)
        self.zoom_slider.setValue(new_value)
    
    def _zoom_out(self):
        """Zoom out by 25%."""
        current_value = self.zoom_slider.value()
        new_value = max(25, current_value - 25)
        self.zoom_slider.setValue(new_value)
    
    def _zoom_fit(self):
        """Fit image to window."""
        self.zoom_slider.setValue(100)
    
    def _update_zoom(self):
        """Update the zoom level of the brush widget."""
        if self.brush_widget:
            # Set zoom on the brush widget
            self.brush_widget.set_zoom(self.current_zoom)
            
            # Update the image widget size to accommodate the zoomed image
            new_width = int(self.original_image_size[0] * self.current_zoom)
            new_height = int(self.original_image_size[1] * self.current_zoom)
            self.image_widget.setFixedSize(new_width, new_height)
            
            # Force update of the scroll area
            self.scroll_area.update()
    
    def _on_label_changed(self, index: int):
        """Handle label selection change."""
        if self.brush_widget:
            self.brush_widget.set_current_label(index)
            self.brush_widget.update_label_display()
    
    def _on_brush_size_changed(self, size: int):
        """Handle brush size change."""
        if self.brush_widget:
            self.brush_widget.set_brush_size(size)
    
    def _clear_labels(self):
        """Clear all painted labels."""
        if self.brush_widget:
            self.brush_widget.clear_labels()
            self._update_training_info()
    
    def _auto_detect_nuclei(self):
        """Use Cellpose nuclei model to automatically detect and label nuclei."""
        if not _HAVE_CELLPOSE:
            QMessageBox.warning(self, "Cellpose Not Available", 
                              "Cellpose is not installed. Please install it with: pip install cellpose")
            return
        
        if not self.brush_widget:
            return
        
        # Get the current nucleus channel for detection
        current_channel = self.channel_combo.currentText()
        if current_channel == "Nucleus" and self.nucleus_channel is not None:
            detection_image = self.nucleus_channel
        elif current_channel == "Cytoplasm" and self.cytoplasm_channel is not None:
            detection_image = self.cytoplasm_channel
        else:
            # Fall back to first available channel
            if self.nucleus_channel is not None:
                detection_image = self.nucleus_channel
            elif self.cytoplasm_channel is not None:
                detection_image = self.cytoplasm_channel
            else:
                QMessageBox.warning(self, "No Channel Available", 
                                  "No nucleus or cytoplasm channel available for detection.")
                return
        
        # Show progress dialog
        progress_dlg = QMessageBox(self)
        progress_dlg.setWindowTitle("Auto-Detecting Nuclei")
        progress_dlg.setText("Running Cellpose nuclei detection...")
        progress_dlg.setStandardButtons(QMessageBox.NoButton)
        progress_dlg.show()
        QApplication.processEvents()
        
        try:
            # Initialize Cellpose nuclei model
            model = models.Cellpose(gpu=False, model_type='nuclei')
            
            # Run detection
            masks, flows, styles, diams = model.eval(
                detection_image, 
                diameter=None,  # Auto-estimate diameter
                channels=[0, 0],  # Grayscale
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            
            # Convert masks to nucleus labels (label 1)
            nucleus_mask = (masks > 0).astype(np.int32)
            
            # Add to brush widget's nucleus label map
            self.brush_widget.label_maps[1] = nucleus_mask
            
            # Update display
            self.brush_widget.update()
            self._update_training_info()
            
            # Show success message
            num_nuclei = len(np.unique(masks)) - 1  # Subtract 1 for background
            progress_dlg.close()
            QMessageBox.information(self, "Auto-Detection Complete", 
                                  f"Successfully detected {num_nuclei} nuclei.\n"
                                  f"Now paint cytoplasm and background regions, then train.")
            
        except Exception as e:
            progress_dlg.close()
            QMessageBox.critical(self, "Auto-Detection Failed", 
                               f"Failed to detect nuclei: {str(e)}")
    
    def _on_label_painted(self, coords: np.ndarray, labels: np.ndarray):
        """Handle label painting."""
        self._update_training_info()
    
    def _update_training_info(self):
        """Update training information display."""
        if self.brush_widget:
            label_maps = self.brush_widget.get_label_maps()
            
            info_parts = []
            class_names = ["Background", "Nucleus", "Cytoplasm"]
            
            for label_id in [0, 1, 2]:  # Background, Nucleus, Cytoplasm
                count = np.sum(label_maps[label_id] == 1)
                if count > 0:
                    info_parts.append(f"{class_names[label_id]}: {count} pixels")
            
            if len(info_parts) == 0:
                self.training_info.setText("No training data")
            else:
                self.training_info.setText(" | ".join(info_parts))
    
    def _train_classifier(self):
        """Train the pixel classifier."""
        if not self.brush_widget:
            return
        
        # Get multi-label data
        label_maps = self.brush_widget.get_label_maps()
        
        # Check which classes have training data
        classes_with_data = []
        for label_id, label_map in label_maps.items():
            if np.any(label_map == 1):
                classes_with_data.append(label_id)
        
        if len(classes_with_data) < 2:
            QMessageBox.warning(self, "Training Error", "Need at least 2 classes for training")
            return
        
        print(f"[DEBUG] Training with classes: {classes_with_data}")
        print(f"[DEBUG] Classes: {['Background', 'Nucleus', 'Cytoplasm'][i] for i in classes_with_data}")
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        QApplication.processEvents()
        
        # Set up timeout timer (30 seconds)
        self.training_timeout = QtCore.QTimer()
        self.training_timeout.setSingleShot(True)
        self.training_timeout.timeout.connect(self._on_training_timeout)
        self.training_timeout.start(30000)  # 30 seconds
        
        try:
            print(f"[DEBUG] Starting training with {len(classes_with_data)} classes: {classes_with_data}")
            
            # Use processed channels instead of all raw channels
            if self.nucleus_channel is not None and self.cytoplasm_channel is not None:
                # Create a 2-channel image stack from processed channels
                processed_img_stack = np.stack([self.nucleus_channel, self.cytoplasm_channel], axis=2)
                processed_channel_names = ["Nucleus", "Cytoplasm"]
                print(f"[DEBUG] Using processed channels: {processed_img_stack.shape} (Nucleus + Cytoplasm)")
                print(f"[DEBUG] Original channels would have been: {self.img_stack.shape} ({len(self.channel_names)} channels)")
            else:
                # Fallback to original channels (shouldn't happen if preprocessing is configured)
                processed_img_stack = self.img_stack
                processed_channel_names = self.channel_names
                print(f"[DEBUG] WARNING: Using original channels: {processed_img_stack.shape} ({len(processed_channel_names)} channels)")
                print("[DEBUG] This may be slow - preprocessing config may not be properly set up")
            
            # Compute features
            self.progress_bar.setFormat("Computing features...")
            QApplication.processEvents()
            print("[DEBUG] Computing features...")
            
            self.current_features = self.feature_computer.compute_features_tiled(
                processed_img_stack, processed_channel_names
            )
            feature_names = self.feature_computer.get_feature_names(processed_channel_names)
            print(f"[DEBUG] Features computed: shape {self.current_features.shape}, {len(feature_names)} feature names")
            
            # Extract training data
            self.progress_bar.setFormat("Extracting training data...")
            QApplication.processEvents()
            print("[DEBUG] Extracting training data...")
            
            # Get coordinates of labeled pixels from all classes
            all_labeled_coords = []
            all_labels = []
            
            for label_id in classes_with_data:
                coords = np.column_stack(np.where(label_maps[label_id] == 1))
                if len(coords) > 0:
                    all_labeled_coords.append(coords)
                    all_labels.extend([label_id] * len(coords))
            
            if len(all_labeled_coords) == 0:
                QMessageBox.warning(self, "Training Error", "No labeled pixels found")
                return
            
            labeled_coords = np.vstack(all_labeled_coords)
            labeled_labels = np.array(all_labels)
            print(f"[DEBUG] Found {len(labeled_labels)} labeled pixels")
            
            # Extract features for labeled pixels
            training_features = self.current_features[labeled_coords[:, 0], labeled_coords[:, 1]]
            print(f"[DEBUG] Training features shape: {training_features.shape}")
            
            # Add to classifier
            print("[DEBUG] Adding training data to classifier...")
            self.classifier.add_training_data(training_features, labeled_labels, feature_names)
            
            # Train classifier
            self.progress_bar.setFormat("Training classifier...")
            QApplication.processEvents()
            print("[DEBUG] Training classifier...")
            
            metrics = self.classifier.train()
            print(f"[DEBUG] Training completed with metrics: {metrics}")
            
            # Enable inference and model saving
            self.infer_btn.setEnabled(True)
            self.save_model_btn.setEnabled(True)
            
            # Show training results
            accuracy = metrics.get('cv_accuracy_mean', 0)
            self.training_info.setText(f"Trained! CV Accuracy: {accuracy:.3f}")
            print(f"[DEBUG] Training successful! CV Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"[DEBUG] Training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Training Error", f"Training failed: {str(e)}")
        finally:
            # Stop timeout timer
            if hasattr(self, 'training_timeout'):
                self.training_timeout.stop()
            self.progress_bar.setVisible(False)
    
    def _on_training_timeout(self):
        """Handle training timeout."""
        print("[DEBUG] Training timed out after 30 seconds")
        QMessageBox.warning(self, "Training Timeout", "Training took too long and was cancelled. Try with fewer training samples or check your data.")
        self.progress_bar.setVisible(False)
    
    def _run_inference(self):
        """Run inference on the entire image."""
        if not self.classifier.is_trained:
            return
        
        # Disable buttons during inference
        self.infer_btn.setEnabled(False)
        self.instance_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Running inference...")
        QApplication.processEvents()
        
        try:
            # Create inference pipeline
            self.inference_pipeline = InteractivePixelInferencePipeline(self.classifier)
            
            # Run inference
            self.progress_bar.setFormat("Computing features for inference...")
            QApplication.processEvents()
            
            # Use processed channels for inference
            if self.nucleus_channel is not None and self.cytoplasm_channel is not None:
                processed_img_stack = np.stack([self.nucleus_channel, self.cytoplasm_channel], axis=2)
                processed_channel_names = ["Nucleus", "Cytoplasm"]
            else:
                processed_img_stack = self.img_stack
                processed_channel_names = self.channel_names
            
            self.progress_bar.setFormat("Running pixel classification...")
            QApplication.processEvents()
            
            results = self.inference_pipeline.process_image(
                processed_img_stack, 
                processed_channel_names,
                return_probabilities=True,
                return_instances=False
            )
            
            self.current_probability_maps = results['probability_maps']
            
            # Enable instance conversion and save
            self.instance_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
            # Update display
            self._update_display()
            
            # Show success message
            QMessageBox.information(self, "Inference Complete", "Pixel classification completed successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"Inference failed: {str(e)}")
        finally:
            # Re-enable inference button
            self.infer_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def _convert_to_instances(self):
        """Convert probability maps to instance segmentation."""
        if self.current_probability_maps is None:
            return
        
        # Disable buttons during conversion
        self.instance_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Converting to instances...")
        QApplication.processEvents()
        
        try:
            # Create instance segmenter with user parameters
            segmenter = InteractivePixelInstanceSegmenter(
                min_distance=self.min_distance_spin.value(),
                threshold_abs=self.threshold_spin.value(),
                threshold_rel=0.2,  # Fixed relative threshold
                min_seed_area=5,    # Fixed seed area
                min_cell_area=self.min_cell_area_spin.value(),
                max_cell_area=self.max_cell_area_spin.value(),
                compactness=0.1,    # Fixed compactness
                nucleus_expansion=self.nucleus_expansion_spin.value()
            )
            
            # Convert to instances
            self.progress_bar.setFormat("Finding local maxima...")
            QApplication.processEvents()
            
            self.current_instances = segmenter.segment_instances(self.current_probability_maps)
            
            # Get statistics
            self.progress_bar.setFormat("Computing statistics...")
            QApplication.processEvents()
            
            stats = segmenter.get_instance_statistics(self.current_instances)
            
            # Update display
            self._update_display()
            
            # Show results
            QMessageBox.information(
                self, 
                "Instance Segmentation Complete", 
                f"Found {stats['n_instances']} instances\n"
                f"Mean area: {stats['mean_area']:.1f} pixels"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Instance Error", f"Instance conversion failed: {str(e)}")
        finally:
            # Re-enable button
            self.instance_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def _save_model(self):
        """Save the trained model to a file."""
        if not self.classifier.is_trained:
            QMessageBox.warning(self, "No Model", "No trained model to save.")
            return
        
        # Get save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Model", 
            "interactive_pixel_classifier.pkl",
            "Pickle files (*.pkl);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.classifier.save_model(file_path)
                QMessageBox.information(self, "Model Saved", f"Model saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save model: {str(e)}")
    
    def _load_model(self):
        """Load a trained model from a file."""
        # Get load location
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Model", 
            "",
            "Pickle files (*.pkl);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.classifier.load_model(file_path)
                
                # Enable inference and model saving
                self.infer_btn.setEnabled(True)
                self.save_model_btn.setEnabled(True)
                
                # Update training info
                summary = self.classifier.get_training_summary()
                if summary['status'] == 'has_training_data':
                    self.training_info.setText(f"Model loaded: {summary['n_samples']} training samples")
                
                QMessageBox.information(self, "Model Loaded", f"Model loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load model: {str(e)}")
    
    def _update_display(self):
        """Update the image display."""
        if not self.brush_widget:
            return
        
        # Always start with a fresh copy of the base image
        # This ensures that toggling off overlays properly removes them
        self.brush_widget.update_display()  # Reset to base image
        display_img = self.brush_widget.pixmap.copy()
        
        # Create a painter for the display
        painter = QPainter(display_img)
        
        # Overlay probability maps if requested
        if self.show_probabilities.isChecked() and self.current_probability_maps is not None:
            overlay = create_probability_overlay(self.current_probability_maps)
            # Convert overlay to QPixmap and blend
            h, w, c = overlay.shape
            bytes_per_line = c * w
            q_image = QImage(overlay.data, w, h, bytes_per_line, QImage.Format_RGB888)
            overlay_pixmap = QPixmap.fromImage(q_image)
            
            # Blend with base image
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setOpacity(0.7)
            painter.drawPixmap(0, 0, overlay_pixmap)
        
        # Overlay instance labels if requested
        if self.show_instances.isChecked() and self.current_instances is not None:
            # Create colored instance overlay
            instance_overlay = self._create_instance_overlay(self.current_instances)
            h, w, c = instance_overlay.shape
            bytes_per_line = c * w
            q_image = QImage(instance_overlay.data, w, h, bytes_per_line, QImage.Format_RGB888)
            instance_pixmap = QPixmap.fromImage(q_image)
            
            # Blend with display
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, instance_pixmap)
        
        painter.end()
        
        # Update the brush widget's pixmap
        self.brush_widget.pixmap = display_img
        self.brush_widget.update()
    
    def _create_instance_overlay(self, instances: np.ndarray) -> np.ndarray:
        """Create colored overlay for instance labels."""
        # Create random colors for each instance
        unique_instances = np.unique(instances)
        unique_instances = unique_instances[unique_instances > 0]
        
        overlay = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        for instance_id in unique_instances:
            # Generate random color
            np.random.seed(instance_id)  # Deterministic colors
            color = np.random.rand(3)
            
            # Apply color to instance
            mask = instances == instance_id
            overlay[mask] = color
        
        return overlay
    
    def _save_results(self):
        """Save segmentation results."""
        if self.current_instances is None:
            QMessageBox.warning(self, "No Results", "No instance segmentation results to save.")
            return
        
        # Get save directory
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not directory:
            return
        
        try:
            # Save instance labels
            np.save(f"{directory}/instance_labels.npy", self.current_instances)
            
            # Save probability maps
            if self.current_probability_maps is not None:
                for class_name, prob_map in self.current_probability_maps.items():
                    np.save(f"{directory}/probability_{class_name}.npy", prob_map)
            
            # Save classifier
            if self.classifier.is_trained:
                self.classifier.save_model(f"{directory}/classifier.pkl")
            
            QMessageBox.information(self, "Save Complete", f"Results saved to {directory}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Save failed: {str(e)}")
    
    def get_results(self) -> Dict[str, np.ndarray]:
        """Get segmentation results."""
        results = {}
        
        if self.current_instances is not None:
            results['instance_labels'] = self.current_instances
        
        if self.current_probability_maps is not None:
            results['probability_maps'] = self.current_probability_maps
        
        return results
