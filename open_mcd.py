#!/usr/bin/env python3
"""
IMC .mcd File Viewer
A PyQt5-based viewer for IMC .mcd files using the readimc library.
Uses the reading method: f.read_acquisition(acquisition) with proper array transposition.
"""

import os
import sys
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# OME-TIFF export
try:
    import tifffile
    _HAVE_TIFFFILE = True
except ImportError:
    _HAVE_TIFFFILE = False

# Cellpose segmentation
try:
    from cellpose import models, io
    import skimage
    import torch
    from sklearn.decomposition import PCA
    _HAVE_CELLPOSE = True
    _HAVE_TORCH = True
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_CELLPOSE = False
    _HAVE_TORCH = False
    _HAVE_SKLEARN = False

# ---- UI ----
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

# ---- Plotting ----
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec

# ---- readimc (required) ----
_HAVE_READIMC = False
try:
    from readimc import MCDFile as McdFile  # pip install readimc
    _HAVE_READIMC = True
except Exception:
    _HAVE_READIMC = False


# --------------------------
# Data types / helpers
# --------------------------
@dataclass
class AcquisitionInfo:
    id: str
    name: str
    well: Optional[str]
    size: Tuple[Optional[int], Optional[int]]  # (H, W)
    channels: List[str]
    channel_metals: List[str]
    channel_labels: List[str]
    metadata: Dict


def robust_percentile_scale(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """Scale array to [0,1] using percentile normalization."""
    a = arr.astype(np.float32, copy=False)
    lo = np.percentile(a, low)
    hi = np.percentile(a, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return a


def arcsinh_normalize(arr: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    """Apply arcsinh normalization with configurable co-factor."""
    a = arr.astype(np.float32, copy=False)
    # Apply arcsinh transformation: arcsinh(x / cofactor)
    normalized = np.arcsinh(a / cofactor)
    return normalized


def percentile_clip_normalize(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Apply percentile clipping normalization."""
    a = arr.astype(np.float32, copy=False)
    vmin = np.percentile(a, p_low)
    vmax = np.percentile(a, p_high)
    # Clip and scale to [0, 1]
    clipped = np.clip(a, vmin, vmax)
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(clipped)
    return normalized


def combine_channels(images: List[np.ndarray], method: str, weights: List[float] = None) -> np.ndarray:
    """Combine multiple channel images using specified method."""
    if not images:
        raise ValueError("No images provided")
    
    if method == "single":
        return images[0]
    
    elif method == "mean":
        return np.mean(images, axis=0)
    
    elif method == "weighted":
        if weights is None:
            # Default weights proportional to SNR (use std as proxy)
            weights = [np.std(img) for img in images]
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        if len(weights) != len(images):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of images ({len(images)})")
        
        result = np.zeros_like(images[0])
        for img, weight in zip(images, weights):
            result += weight * img
        return result
    
    elif method == "max":
        return np.maximum.reduce(images)
    
    elif method == "pca1":
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn required for PCA method")
        
        # Flatten images and stack them
        flattened = np.array([img.flatten() for img in images]).T
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(flattened)
        return pca_result.reshape(images[0].shape)
    
    else:
        raise ValueError(f"Unknown combination method: {method}")


class PreprocessingCache:
    """Cache for preprocessing statistics to ensure identical batch runs."""
    def __init__(self):
        self.cache = {}  # {acq_id: {channel: {method: stats}}}
    
    def get_key(self, acq_id: str, channel: str, method: str, **params) -> str:
        """Generate cache key for parameters."""
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{acq_id}_{channel}_{method}_{param_str}"
    
    def get_stats(self, acq_id: str, channel: str, method: str, **params) -> dict:
        """Get cached statistics."""
        key = self.get_key(acq_id, channel, method, **params)
        return self.cache.get(key, {})
    
    def set_stats(self, acq_id: str, channel: str, method: str, stats: dict, **params):
        """Cache statistics."""
        key = self.get_key(acq_id, channel, method, **params)
        self.cache[key] = stats
    
    def clear(self):
        """Clear all cached statistics."""
        self.cache.clear()


def stack_to_rgb(stack: np.ndarray) -> np.ndarray:
    """
    Convert 1–3 channel stack to RGB for display.
    1 ch → gray ; 2 ch → (ch1, ch2, ch2) ; ≥3 ch → first 3 channels
    """
    H, W, C = stack.shape
    if C == 1:
        # Scale to 0-1 range
        g = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        return np.dstack([g, g, g])
    elif C == 2:
        # Scale each channel to 0-1 range
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        return np.dstack([r, g, g])
    else:
        # Scale each channel to 0-1 range
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        b = (stack[..., 2] - np.min(stack[..., 2])) / (np.max(stack[..., 2]) - np.min(stack[..., 2]) + 1e-8)
        return np.dstack([r, g, b])


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def show_image(self, img: np.ndarray, title: str = "", grayscale: bool = False, show_colorbar: bool = True, raw_img: np.ndarray = None, custom_min: float = None, custom_max: float = None):
        # Clear the entire figure
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        
        im = None
        vmin, vmax = None, None  # Store the actual vmin/vmax used
        
        if img.ndim == 2:
            if custom_min is not None and custom_max is not None:
                vmin, vmax = custom_min, custom_max
            else:
                vmin, vmax = np.min(img), np.max(img)
            
            if grayscale:
                im = self.ax.imshow(img, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            else:
                im = self.ax.imshow(img, interpolation="nearest", cmap='viridis', vmin=vmin, vmax=vmax)
        elif img.ndim == 3:
            if grayscale:
                # Show first channel in grayscale
                if custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img[..., 0]), np.max(img[..., 0])
                im = self.ax.imshow(img[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            else:
                if img.shape[-1] <= 3:
                    im = self.ax.imshow(stack_to_rgb(img), interpolation="nearest")
                else:
                    im = self.ax.imshow(stack_to_rgb(img[..., :3]), interpolation="nearest")
        else:
            self.ax.text(0.5, 0.5, "Unsupported image shape", ha="center", va="center")
        
        self.ax.set_title(title)
        self.ax.axis("off")
        
        # Add colorbar if requested and image is valid
        if show_colorbar and im is not None and img.ndim == 2 and vmin is not None and vmax is not None:
            cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8, aspect=20)
            # The colorbar automatically shows the vmin/vmax range from imshow
            # We just need to format the tick labels nicely
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
        
        self.draw()

    def show_grid(self, images: List[np.ndarray], titles: List[str], grayscale: bool = False, raw_images: List[np.ndarray] = None, custom_min: float = None, custom_max: float = None, channel_names: List[str] = None, channel_scaling: dict = None, custom_scaling_enabled: bool = False):
        """Show multiple images in a grid layout."""
        n_images = len(images)
        if n_images == 0:
            return
        
        # Calculate grid dimensions
        cols = min(3, n_images)  # Max 3 columns
        rows = (n_images + cols - 1) // cols
        
        # Clear and create subplots
        self.fig.clear()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            im = None
            vmin, vmax = None, None  # Store the actual vmin/vmax used
            
            # Get per-channel scaling if available
            channel_min = None
            channel_max = None
            if custom_scaling_enabled and channel_names and i < len(channel_names) and channel_scaling:
                channel_name = channel_names[i]
                if channel_name in channel_scaling:
                    channel_min = channel_scaling[channel_name]['min']
                    channel_max = channel_scaling[channel_name]['max']
            
            if img.ndim == 2:
                if channel_min is not None and channel_max is not None:
                    vmin, vmax = channel_min, channel_max
                elif custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img), np.max(img)
                
                if grayscale:
                    im = ax.imshow(img, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(img, interpolation="nearest", cmap='viridis', vmin=vmin, vmax=vmax)
            elif img.ndim == 3 and not grayscale:
                if img.shape[-1] <= 3:
                    im = ax.imshow(stack_to_rgb(img), interpolation="nearest")
                else:
                    im = ax.imshow(stack_to_rgb(img[..., :3]), interpolation="nearest")
            else:
                # Grayscale for multi-channel
                if channel_min is not None and channel_max is not None:
                    vmin, vmax = channel_min, channel_max
                elif custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img[..., 0]), np.max(img[..., 0])
                im = ax.imshow(img[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            
            # Add colorbar for each subplot
            if im is not None and img.ndim == 2 and vmin is not None and vmax is not None:
                cbar = self.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                # The colorbar automatically shows the vmin/vmax range from imshow
                # We just need to format the tick labels nicely
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
        
        self.fig.tight_layout()
        self.draw()


# --------------------------
# IMC loader using readimc
# --------------------------
class MCDLoader:
    """Loader for IMC .mcd files using the readimc library with f.read_acquisition() method."""
    
    def __init__(self):
        if not _HAVE_READIMC:
            raise RuntimeError(
                "readimc is not installed. Run: pip install readimc"
            )
        self.mcd: Optional[McdFile] = None
        self._acq_map: Dict[str, object] = {}
        self._acq_channels: Dict[str, List[str]] = {}
        self._acq_channel_metals: Dict[str, List[str]] = {}
        self._acq_channel_labels: Dict[str, List[str]] = {}
        self._acq_size: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self._acq_name: Dict[str, str] = {}
        self._acq_well: Dict[str, Optional[str]] = {}
        self._acq_metadata: Dict[str, Dict] = {}

    def open(self, path: str):
        """Open an .mcd file."""
        self.mcd = McdFile(path)
        if hasattr(self.mcd, "open"):
            self.mcd.open()
        self._index()

    def _index(self):
        """Index all acquisitions in the .mcd file."""
        self._acq_map.clear()
        self._acq_channels.clear()
        self._acq_channel_metals.clear()
        self._acq_channel_labels.clear()
        self._acq_size.clear()
        self._acq_name.clear()
        self._acq_well.clear()
        self._acq_metadata.clear()

        acq_counter = 0
        slides = getattr(self.mcd, "slides", [])
        
        if slides:
            for slide_idx, slide in enumerate(slides):
                for acq_idx, acq in enumerate(getattr(slide, "acquisitions", [])):
                    acq_id = f"slide_{slide_idx}_acq_{acq_idx}"
                    acq_counter += 1
                    
                    # Get acquisition name
                    name = getattr(acq, "name", f"Slide {slide_idx + 1} Acquisition {acq_idx + 1}")
                    
                    # Get well information
                    well = getattr(acq, "well", getattr(slide, "well", None))
                    if well is None and hasattr(acq, "metadata"):
                        metadata = acq.metadata
                        if isinstance(metadata, dict) and 'Description' in metadata:
                            well = metadata['Description']
                    
                    # Get channel information
                    channel_metals = getattr(acq, "channel_names", [])
                    channel_labels = getattr(acq, "channel_labels", [])
                    
                    # Create display names for channels
                    channels = []
                    for i, (metal, label) in enumerate(zip(channel_metals, channel_labels)):
                        if label and metal:
                            channels.append(f"{label}_{metal}")
                        elif label:
                            channels.append(label)
                        elif metal:
                            channels.append(metal)
                        else:
                            channels.append(f"Channel_{i+1}")
                    
                    # Get image dimensions
                    try:
                        # Try to get dimensions from acquisition metadata
                        H = getattr(acq, "height", None) or getattr(acq, "rows", None)
                        W = getattr(acq, "width", None) or getattr(acq, "cols", None)
                        size = (int(H), int(W)) if H and W else (None, None)
                    except Exception:
                        size = (None, None)
                    
                    # Get metadata
                    metadata = getattr(acq, "metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # Store acquisition info
                    self._acq_map[acq_id] = acq
                    self._acq_channels[acq_id] = channels
                    self._acq_channel_metals[acq_id] = channel_metals
                    self._acq_channel_labels[acq_id] = channel_labels
                    self._acq_size[acq_id] = size
                    self._acq_name[acq_id] = name
                    self._acq_well[acq_id] = well
                    self._acq_metadata[acq_id] = metadata

        if not self._acq_map:
            raise RuntimeError("No acquisitions found in this .mcd file.")

    def list_acquisitions(self) -> List[AcquisitionInfo]:
        """List all acquisitions in the .mcd file."""
        infos: List[AcquisitionInfo] = []
        for acq_id in self._acq_map:
            infos.append(
                AcquisitionInfo(
                    id=acq_id,
                    name=self._acq_name.get(acq_id, acq_id),
                    well=self._acq_well.get(acq_id),
                    size=self._acq_size.get(acq_id, (None, None)),
                    channels=self._acq_channels.get(acq_id, []),
                    channel_metals=self._acq_channel_metals.get(acq_id, []),
                    channel_labels=self._acq_channel_labels.get(acq_id, []),
                    metadata=self._acq_metadata.get(acq_id, {}),
                )
            )
        return infos

    def get_channels(self, acq_id: str) -> List[str]:
        """Get channel names for a specific acquisition."""
        return self._acq_channels[acq_id]

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        """Get image data for a specific acquisition and channel."""
        acq = self._acq_map[acq_id]
        channels = self._acq_channels[acq_id]
        
        if channel not in channels:
            raise ValueError(f"Channel '{channel}' not found in acquisition {acq_id}.")
        
        ch_idx = channels.index(channel)
        
        # Use the reading method from the notebook: f.read_acquisition(acquisition)
        with self.mcd as f:
            img = f.read_acquisition(acq)  # array, shape: (c, y, x), dtype: float32
            
            # Reorder to height, width, channel as shown in the notebook
            img = np.transpose(img, (1, 2, 0))
            
            # Return the specific channel
            return img[..., ch_idx]

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        """Get all channels for a specific acquisition as a 3D array (H, W, C)."""
        acq = self._acq_map[acq_id]
        
        # Use the reading method from the notebook: f.read_acquisition(acquisition)
        with self.mcd as f:
            img = f.read_acquisition(acq)  # array, shape: (c, y, x), dtype: float32
            
            # Reorder to height, width, channel as shown in the notebook
            img = np.transpose(img, (1, 2, 0))
            
            return img

    def close(self):
        """Close the .mcd file."""
        if self.mcd and hasattr(self.mcd, "close"):
            self.mcd.close()


# --------------------------
# Dynamic Comparison dialog
# --------------------------
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

    def _remove_acquisition(self):
        current_item = self.acq_list.currentItem()
        if current_item:
            acq_id = current_item.data(Qt.UserRole)
            self.selected_acquisitions.remove(acq_id)
            self.acq_list.takeItem(self.acq_list.row(current_item))
            self._update_channel_combo()

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


# --------------------------
# Progress Dialog
# --------------------------
class ProgressDialog(QtWidgets.QDialog):
    def __init__(self, title: str = "Export Progress", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)  # Non-modal so user can still interact with main window
        self.setFixedSize(450, 180)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.cancelled = False
        self.start_time = None
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Status label
        self.status_label = QtWidgets.QLabel("Preparing export...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Details label
        self.details_label = QtWidgets.QLabel("")
        self.details_label.setAlignment(Qt.AlignCenter)
        self.details_label.setStyleSheet("QLabel { color: #666; }")
        layout.addWidget(self.details_label)
        
        # Time remaining label
        self.time_label = QtWidgets.QLabel("")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        layout.addWidget(self.time_label)
        
        # Cancel button
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_btn)
        
    def _cancel(self):
        """Handle cancel button click."""
        self.cancelled = True
        self.status_label.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)
        
    def update_progress(self, value: int, status: str = "", details: str = ""):
        """Update progress bar and status."""
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
        if details:
            self.details_label.setText(details)
        
        # Calculate time remaining
        if value > 0:
            elapsed = time.time() - self.start_time
            if value < self.progress_bar.maximum():
                estimated_total = elapsed * self.progress_bar.maximum() / value
                remaining = estimated_total - elapsed
                if remaining > 0:
                    self.time_label.setText(f"Estimated time remaining: {remaining:.0f}s")
                else:
                    self.time_label.setText("Almost done...")
            else:
                self.time_label.setText("Complete!")
        
        QtWidgets.QApplication.processEvents()  # Keep UI responsive
        
    def set_maximum(self, maximum: int):
        """Set maximum value for progress bar."""
        self.progress_bar.setMaximum(maximum)
        
    def is_cancelled(self) -> bool:
        """Check if user cancelled the operation."""
        return self.cancelled


# --------------------------
# GPU Selection Dialog
# --------------------------
class GPUSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Selection")
        self.setModal(True)
        self.setFixedSize(400, 300)
        self.selected_gpu = None
        
        # Detect available GPUs
        self.available_gpus = self._detect_gpus()
        
        # Create UI
        self._create_ui()
        
    def _detect_gpus(self):
        """Detect available GPUs using torch."""
        gpus = []
        
        if not _HAVE_TORCH:
            return gpus
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    gpus.append({
                        'id': i,
                        'name': gpu_name,
                        'memory': gpu_memory,
                        'type': 'CUDA'
                    })
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpus.append({
                    'id': 'mps',
                    'name': 'Apple Metal Performance Shaders (MPS)',
                    'memory': None,
                    'type': 'MPS'
                })
        except Exception as e:
            print(f"Error detecting GPUs: {e}")
        
        return gpus
    
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Select GPU for Segmentation")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QtWidgets.QLabel("GPU acceleration can significantly speed up segmentation. Select a device:")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # GPU list
        self.gpu_list = QtWidgets.QListWidget()
        self.gpu_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        
        # Add CPU option
        cpu_item = QtWidgets.QListWidgetItem("CPU (No GPU acceleration)")
        cpu_item.setData(Qt.UserRole, None)
        self.gpu_list.addItem(cpu_item)
        
        # Add GPU options
        for gpu in self.available_gpus:
            if gpu['memory'] is not None:
                text = f"{gpu['name']} ({gpu['memory']:.1f} GB VRAM)"
            else:
                text = gpu['name']
            item = QtWidgets.QListWidgetItem(text)
            item.setData(Qt.UserRole, gpu['id'])
            self.gpu_list.addItem(item)
        
        # Select first item by default
        if self.gpu_list.count() > 0:
            self.gpu_list.setCurrentRow(0)
        
        layout.addWidget(self.gpu_list)
        
        # GPU info
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Connect selection change
        self.gpu_list.itemSelectionChanged.connect(self._on_gpu_selection_changed)
        
        # Initialize info
        self._on_gpu_selection_changed()
    
    def _on_gpu_selection_changed(self):
        """Update info when GPU selection changes."""
        current_item = self.gpu_list.currentItem()
        if not current_item:
            return
        
        gpu_id = current_item.data(Qt.UserRole)
        
        if gpu_id is None:
            self.info_label.setText("Using CPU for segmentation. This will be slower but more compatible.")
        else:
            gpu = next((g for g in self.available_gpus if g['id'] == gpu_id), None)
            if gpu:
                if gpu['type'] == 'CUDA':
                    self.info_label.setText(f"Using CUDA GPU: {gpu['name']}\nMemory: {gpu['memory']:.1f} GB")
                elif gpu['type'] == 'MPS':
                    self.info_label.setText(f"Using Apple Metal Performance Shaders\nOptimized for Apple Silicon Macs")
    
    def get_selected_gpu(self):
        """Get the selected GPU ID."""
        current_item = self.gpu_list.currentItem()
        if current_item:
            return current_item.data(Qt.UserRole)
        return None


# --------------------------
# Preprocessing Dialog
# --------------------------
class PreprocessingDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preprocessing")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.channels = channels
        
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Image Preprocessing for Segmentation")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QtWidgets.QLabel("Configure normalization and channel combination for segmentation.")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Normalization section
        norm_group = QtWidgets.QGroupBox("Normalization")
        norm_layout = QtWidgets.QVBoxLayout(norm_group)
        
        # Normalization method
        norm_method_layout = QtWidgets.QHBoxLayout()
        norm_method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.norm_method_combo = QtWidgets.QComboBox()
        self.norm_method_combo.addItems(["None", "arcsinh", "percentile_clip"])
        self.norm_method_combo.currentTextChanged.connect(self._on_norm_method_changed)
        norm_method_layout.addWidget(self.norm_method_combo)
        norm_method_layout.addStretch()
        norm_layout.addLayout(norm_method_layout)
        
        # Arcsinh parameters
        self.arcsinh_frame = QtWidgets.QFrame()
        arcsinh_layout = QtWidgets.QHBoxLayout(self.arcsinh_frame)
        arcsinh_layout.addWidget(QtWidgets.QLabel("Cofactor:"))
        self.arcsinh_cofactor_spin = QtWidgets.QDoubleSpinBox()
        self.arcsinh_cofactor_spin.setRange(0.1, 100.0)
        self.arcsinh_cofactor_spin.setValue(5.0)
        self.arcsinh_cofactor_spin.setDecimals(1)
        arcsinh_layout.addWidget(self.arcsinh_cofactor_spin)
        arcsinh_layout.addStretch()
        norm_layout.addWidget(self.arcsinh_frame)
        
        # Percentile parameters
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
        norm_layout.addWidget(self.percentile_frame)
        
        layout.addWidget(norm_group)
        
        # Channel combination section
        combo_group = QtWidgets.QGroupBox("Channel Combination")
        combo_layout = QtWidgets.QVBoxLayout(combo_group)
        
        # Nuclear channels
        nuclear_layout = QtWidgets.QHBoxLayout()
        nuclear_layout.addWidget(QtWidgets.QLabel("Nuclear channels:"))
        self.nuclear_list = QtWidgets.QListWidget()
        self.nuclear_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for channel in self.channels:
            self.nuclear_list.addItem(channel)
        nuclear_layout.addWidget(self.nuclear_list)
        combo_layout.addLayout(nuclear_layout)
        
        # Auto-selection info label
        self.nuclear_auto_info = QtWidgets.QLabel("")
        self.nuclear_auto_info.setStyleSheet("QLabel { color: #0066cc; font-style: italic; font-size: 11px; }")
        self.nuclear_auto_info.setWordWrap(True)
        combo_layout.addWidget(self.nuclear_auto_info)
        
        # Nuclear combination method
        nuclear_combo_layout = QtWidgets.QHBoxLayout()
        nuclear_combo_layout.addWidget(QtWidgets.QLabel("Nuclear combination:"))
        self.nuclear_combo_method = QtWidgets.QComboBox()
        self.nuclear_combo_method.addItems(["single", "mean", "weighted", "max", "pca1"])
        self.nuclear_combo_method.currentTextChanged.connect(self._on_nuclear_combo_changed)
        nuclear_combo_layout.addWidget(self.nuclear_combo_method)
        nuclear_combo_layout.addStretch()
        combo_layout.addLayout(nuclear_combo_layout)
        
        # Connect nuclear channel selection to update combo options
        self.nuclear_list.itemSelectionChanged.connect(self._on_nuclear_channels_changed)
        
        # Nuclear weights (for weighted method)
        self.nuclear_weights_frame = QtWidgets.QFrame()
        nuclear_weights_layout = QtWidgets.QVBoxLayout(self.nuclear_weights_frame)
        nuclear_weights_layout.addWidget(QtWidgets.QLabel("Nuclear channel weights (leave empty for auto):"))
        self.nuclear_weights_edit = QtWidgets.QLineEdit()
        self.nuclear_weights_edit.setPlaceholderText("e.g., 0.5,0.3,0.2")
        nuclear_weights_layout.addWidget(self.nuclear_weights_edit)
        combo_layout.addWidget(self.nuclear_weights_frame)
        
        # Cytoplasm channels
        cyto_layout = QtWidgets.QHBoxLayout()
        cyto_layout.addWidget(QtWidgets.QLabel("Cytoplasm channels:"))
        self.cyto_list = QtWidgets.QListWidget()
        self.cyto_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for channel in self.channels:
            self.cyto_list.addItem(channel)
        cyto_layout.addWidget(self.cyto_list)
        combo_layout.addLayout(cyto_layout)
        
        # Cytoplasm auto-selection info label
        self.cyto_auto_info = QtWidgets.QLabel("")
        self.cyto_auto_info.setStyleSheet("QLabel { color: #0066cc; font-style: italic; font-size: 11px; }")
        self.cyto_auto_info.setWordWrap(True)
        combo_layout.addWidget(self.cyto_auto_info)
        
        # Cytoplasm combination method
        cyto_combo_layout = QtWidgets.QHBoxLayout()
        cyto_combo_layout.addWidget(QtWidgets.QLabel("Cytoplasm combination:"))
        self.cyto_combo_method = QtWidgets.QComboBox()
        self.cyto_combo_method.addItems(["single", "mean", "weighted", "max", "pca1"])
        self.cyto_combo_method.currentTextChanged.connect(self._on_cyto_combo_changed)
        cyto_combo_layout.addWidget(self.cyto_combo_method)
        cyto_combo_layout.addStretch()
        combo_layout.addLayout(cyto_combo_layout)
        
        # Connect cytoplasm channel selection to update combo options
        self.cyto_list.itemSelectionChanged.connect(self._on_cyto_channels_changed)
        
        # Cytoplasm weights (for weighted method)
        self.cyto_weights_frame = QtWidgets.QFrame()
        cyto_weights_layout = QtWidgets.QVBoxLayout(self.cyto_weights_frame)
        cyto_weights_layout.addWidget(QtWidgets.QLabel("Cytoplasm channel weights (leave empty for auto):"))
        self.cyto_weights_edit = QtWidgets.QLineEdit()
        self.cyto_weights_edit.setPlaceholderText("e.g., 0.5,0.3,0.2")
        cyto_weights_layout.addWidget(self.cyto_weights_edit)
        combo_layout.addWidget(self.cyto_weights_frame)
        
        layout.addWidget(combo_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        # Initialize
        self._on_norm_method_changed()
        self._on_nuclear_combo_changed()
        self._on_cyto_combo_changed()
        
        # Auto-parse channels for DNA and set defaults
        self._auto_parse_channels()
    
    def _on_norm_method_changed(self):
        """Update UI when normalization method changes."""
        method = self.norm_method_combo.currentText()
        self.arcsinh_frame.setVisible(method == "arcsinh")
        self.percentile_frame.setVisible(method == "percentile_clip")
    
    def _on_nuclear_combo_changed(self):
        """Update UI when nuclear combination method changes."""
        method = self.nuclear_combo_method.currentText()
        self.nuclear_weights_frame.setVisible(method == "weighted")
    
    def _on_cyto_combo_changed(self):
        """Update UI when cytoplasm combination method changes."""
        method = self.cyto_combo_method.currentText()
        self.cyto_weights_frame.setVisible(method == "weighted")
        
        # Clear auto-selection info when user manually changes combination method
        if self.cyto_auto_info.text().startswith("✓ Auto-selected"):
            self.cyto_auto_info.setText("")
    
    def _on_nuclear_channels_changed(self):
        """Update nuclear combination options when channel selection changes."""
        selected_count = len(self.nuclear_list.selectedItems())
        self._update_combo_options(self.nuclear_combo_method, selected_count)
        
        # Clear auto-selection info when user manually changes selection
        if self.nuclear_auto_info.text().startswith("✓ Auto-selected"):
            self.nuclear_auto_info.setText("")
    
    def _on_cyto_channels_changed(self):
        """Update cytoplasm combination options when channel selection changes."""
        selected_count = len(self.cyto_list.selectedItems())
        self._update_combo_options(self.cyto_combo_method, selected_count)
        
        # Clear auto-selection info when user manually changes selection
        if self.cyto_auto_info.text().startswith("✓ Auto-selected"):
            self.cyto_auto_info.setText("")
    
    def _auto_parse_channels(self):
        """Auto-parse channels to select DNA channels for nuclear and set max as default for cytoplasm."""
        # Auto-select channels containing 'DNA' for nuclear channels
        dna_channels = []
        dna_channel_names = []
        for i in range(self.nuclear_list.count()):
            item = self.nuclear_list.item(i)
            if 'DNA' in item.text().upper():
                dna_channels.append(item)
                dna_channel_names.append(item.text())
        
        # Select DNA channels
        for item in dna_channels:
            item.setSelected(True)
        
        # Show auto-selection info
        if dna_channel_names:
            self.nuclear_auto_info.setText(f"✓ Auto-selected DNA channels: {', '.join(dna_channel_names)}")
            # Scroll to show the first selected DNA channel
            if dna_channels:
                self.nuclear_list.scrollToItem(dna_channels[0])
        else:
            self.nuclear_auto_info.setText("No DNA channels found for auto-selection")
        
        # Set max as default for cytoplasm combination method
        self.cyto_combo_method.setCurrentText("max")
        self.cyto_auto_info.setText("✓ Auto-selected 'max' as cytoplasm combination method")
        
        # Update combo options based on selections
        self._on_nuclear_channels_changed()
        self._on_cyto_channels_changed()
    
    def _update_combo_options(self, combo_box, selected_count):
        """Update combination method options based on number of selected channels."""
        current_text = combo_box.currentText()
        
        # Clear and repopulate options
        combo_box.clear()
        
        if selected_count <= 1:
            # Only one or no channels selected - only "single" makes sense
            combo_box.addItems(["single"])
            combo_box.setCurrentText("single")
        else:
            # Multiple channels selected - remove "single" option
            combo_box.addItems(["mean", "weighted", "max", "pca1"])
            # Try to keep the same method if it's still valid
            if current_text in ["mean", "weighted", "max", "pca1"]:
                combo_box.setCurrentText(current_text)
            else:
                combo_box.setCurrentText("mean")  # Default to mean
    
    def get_normalization_method(self):
        """Get selected normalization method."""
        return self.norm_method_combo.currentText()
    
    def get_arcsinh_cofactor(self):
        """Get arcsinh cofactor."""
        return self.arcsinh_cofactor_spin.value()
    
    def get_percentile_params(self):
        """Get percentile clipping parameters."""
        return self.p_low_spin.value(), self.p_high_spin.value()
    
    def get_nuclear_channels(self):
        """Get selected nuclear channels."""
        selected_items = self.nuclear_list.selectedItems()
        return [item.text() for item in selected_items]
    
    def get_cyto_channels(self):
        """Get selected cytoplasm channels."""
        selected_items = self.cyto_list.selectedItems()
        return [item.text() for item in selected_items]
    
    def get_nuclear_combo_method(self):
        """Get nuclear combination method."""
        return self.nuclear_combo_method.currentText()
    
    def get_cyto_combo_method(self):
        """Get cytoplasm combination method."""
        return self.cyto_combo_method.currentText()
    
    def get_nuclear_weights(self):
        """Get nuclear channel weights."""
        text = self.nuclear_weights_edit.text().strip()
        if not text:
            return None
        try:
            weights = [float(x.strip()) for x in text.split(',')]
            return weights
        except ValueError:
            return None
    
    def get_cyto_weights(self):
        """Get cytoplasm channel weights."""
        text = self.cyto_weights_edit.text().strip()
        if not text:
            return None
        try:
            weights = [float(x.strip()) for x in text.split(',')]
            return weights
        except ValueError:
            return None


# --------------------------
# Segmentation Dialog
# --------------------------
class SegmentationDialog(QtWidgets.QDialog):
    def __init__(self, channels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Segmentation")
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.channels = channels
        self.segmentation_result = None
        self.preprocessing_config = None
        
        # Create UI
        self._create_ui()
        
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
        
        # Preprocessing button
        preprocess_group = QtWidgets.QGroupBox("Image Preprocessing")
        preprocess_layout = QtWidgets.QVBoxLayout(preprocess_group)
        
        preprocess_btn = QtWidgets.QPushButton("Configure Preprocessing...")
        preprocess_btn.clicked.connect(self._open_preprocessing_dialog)
        preprocess_layout.addWidget(preprocess_btn)
        
        self.preprocess_info_label = QtWidgets.QLabel("No preprocessing configured")
        self.preprocess_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.preprocess_info_label.setWordWrap(True)
        preprocess_layout.addWidget(self.preprocess_info_label)
        
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
        self.segment_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
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
        
    def _on_model_changed(self):
        """Update UI when model selection changes."""
        model = self.model_combo.currentText()
        if model == "nuclei":
            self.model_desc.setText("Nuclei: Segments cell nuclei using nuclear channel")
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
                if hasattr(parent, 'mcd_path') and parent.mcd_path:
                    return os.path.dirname(parent.mcd_path)
        return None


# --------------------------
# Feature Extraction Dialog
# --------------------------

class FeatureExtractionDialog(QtWidgets.QDialog):
    """Dialog for configuring feature extraction."""
    
    def __init__(self, parent, acquisitions, segmentation_masks):
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


# --------------------------
# Export Dialog
# --------------------------
class ExportDialog(QtWidgets.QDialog):
    def __init__(self, acquisitions: List[AcquisitionInfo], current_acq_id: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export to OME-TIFF")
        self.setModal(True)
        self.acquisitions = acquisitions
        self.current_acq_id = current_acq_id
        self.output_directory = ""
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
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
        
        # Options
        options_group = QtWidgets.QGroupBox("Export Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        self.include_metadata_chk = QtWidgets.QCheckBox("Include metadata in OME-TIFF")
        self.include_metadata_chk.setChecked(True)
        options_layout.addWidget(self.include_metadata_chk)
        
        self.apply_normalization_chk = QtWidgets.QCheckBox("Apply current normalization settings")
        self.apply_normalization_chk.setChecked(False)
        options_layout.addWidget(self.apply_normalization_chk)
        
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setEnabled(False)  # Disabled until directory is selected
        self.export_btn.clicked.connect(self.accept)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
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
    
    def get_apply_normalization(self):
        """Get whether to apply normalization."""
        return self.apply_normalization_chk.isChecked()


# --------------------------
# Cell Clustering Dialog
# --------------------------
class CellClusteringDialog(QtWidgets.QDialog):
    def __init__(self, feature_dataframe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Clustering Analysis")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.feature_dataframe = feature_dataframe
        self.cluster_labels = None
        self.clustered_data = None
        
        self._create_ui()
        self._setup_plot()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cell Clustering Analysis")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; }")
        layout.addWidget(title_label)
        
        # Options panel
        options_group = QtWidgets.QGroupBox("Clustering Options")
        options_layout = QtWidgets.QHBoxLayout(options_group)
        
        # Aggregation method
        options_layout.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.agg_method = QtWidgets.QComboBox()
        self.agg_method.addItems(["mean", "median"])
        self.agg_method.setCurrentText("mean")
        options_layout.addWidget(self.agg_method)
        
        # Include morphometric features
        self.include_morpho = QtWidgets.QCheckBox("Include morphometric features")
        self.include_morpho.setChecked(True)
        options_layout.addWidget(self.include_morpho)
        
        # Number of clusters
        options_layout.addWidget(QtWidgets.QLabel("Number of clusters:"))
        self.n_clusters = QtWidgets.QSpinBox()
        self.n_clusters.setRange(2, 20)
        self.n_clusters.setValue(5)
        options_layout.addWidget(self.n_clusters)
        
        # Clustering method
        options_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.cluster_method = QtWidgets.QComboBox()
        self.cluster_method.addItems(["ward", "complete", "average", "single"])
        self.cluster_method.setCurrentText("ward")
        options_layout.addWidget(self.cluster_method)
        
        # Run clustering button
        self.run_btn = QtWidgets.QPushButton("Run Clustering")
        self.run_btn.clicked.connect(self._run_clustering)
        options_layout.addWidget(self.run_btn)
        
        options_layout.addStretch()
        layout.addWidget(options_group)
        
        # Plot area
        plot_group = QtWidgets.QGroupBox("Clustering Results")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        
        # Create matplotlib canvas
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.explore_btn = QtWidgets.QPushButton("Explore Clusters")
        self.explore_btn.clicked.connect(self._explore_clusters)
        self.explore_btn.setEnabled(False)
        button_layout.addWidget(self.explore_btn)
        
        self.save_btn = QtWidgets.QPushButton("Save Heatmap")
        self.save_btn.clicked.connect(self._save_heatmap)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        plot_layout.addLayout(button_layout)
        layout.addWidget(plot_group)
        
    def _setup_plot(self):
        """Setup the matplotlib plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title("Cell Clustering Analysis")
        ax.text(0.5, 0.5, "Click 'Run Clustering' to generate heatmap", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        self.canvas.draw()
        
    def _run_clustering(self):
        """Run the clustering analysis."""
        try:
            # Get options
            agg_method = self.agg_method.currentText()
            include_morpho = self.include_morpho.isChecked()
            n_clusters = self.n_clusters.value()
            cluster_method = self.cluster_method.currentText()
            
            # Prepare data
            data = self._prepare_clustering_data(agg_method, include_morpho)
            
            if data is None or data.empty:
                QtWidgets.QMessageBox.warning(self, "No Data", "No suitable data found for clustering.")
                return
            
            # Perform clustering
            self.clustered_data, self.cluster_labels = self._perform_clustering(data, n_clusters, cluster_method)
            
            # Create heatmap
            self._create_heatmap()
            
            # Enable buttons
            self.explore_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Clustering Error", f"Error during clustering: {str(e)}")
    
    def _prepare_clustering_data(self, agg_method, include_morpho):
        """Prepare data for clustering."""
        # Get marker columns (intensity features)
        marker_cols = [col for col in self.feature_dataframe.columns 
                      if any(suffix in col for suffix in ['_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos'])]
        
        # Get morphometric columns if requested
        morpho_cols = []
        if include_morpho:
            morpho_cols = [col for col in self.feature_dataframe.columns 
                          if col in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count']]
        
        # Combine all feature columns
        feature_cols = marker_cols + morpho_cols
        
        if not feature_cols:
            return None
        
        # Extract data
        data = self.feature_dataframe[feature_cols].copy()
        
        # Handle missing values
        data = data.fillna(data.median())
        
        # Normalize data
        data = (data - data.mean()) / data.std()
        
        return data
    
    def _perform_clustering(self, data, n_clusters, method):
        """Perform hierarchical clustering."""
        # Calculate distance matrix
        distances = pdist(data.values, metric='euclidean')
        
        # Perform linkage
        linkage_matrix = linkage(distances, method=method)
        
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Sort data by cluster
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        # Sort by cluster
        clustered_data = data_with_clusters.sort_values('cluster')
        
        return clustered_data, cluster_labels
    
    def _create_heatmap(self):
        """Create the heatmap visualization."""
        self.figure.clear()
        
        # Create subplots
        gs = self.figure.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.3, wspace=0.3)
        
        # Main heatmap
        ax_heatmap = self.figure.add_subplot(gs[1, 0])
        
        # Prepare data for heatmap (exclude cluster column)
        heatmap_data = self.clustered_data.drop('cluster', axis=1).values
        
        # Create heatmap
        im = ax_heatmap.imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Set labels
        ax_heatmap.set_xlabel('Cells')
        ax_heatmap.set_ylabel('Features')
        ax_heatmap.set_title('Cell Clustering Heatmap')
        
        # Add cluster boundaries
        cluster_boundaries = []
        current_cluster = self.clustered_data['cluster'].iloc[0]
        for i, cluster in enumerate(self.clustered_data['cluster']):
            if cluster != current_cluster:
                cluster_boundaries.append(i)
                current_cluster = cluster
        
        # Draw cluster boundaries
        for boundary in cluster_boundaries:
            ax_heatmap.axvline(x=boundary, color='red', linewidth=2, alpha=0.7)
        
        # Colorbar
        cbar = self.figure.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label('Normalized Feature Value')
        
        # Dendrogram
        ax_dendro = self.figure.add_subplot(gs[0, 0])
        distances = pdist(self.clustered_data.drop('cluster', axis=1).values, metric='euclidean')
        linkage_matrix = linkage(distances, method=self.cluster_method.currentText())
        dendrogram(linkage_matrix, ax=ax_dendro, orientation='top', color_threshold=0)
        ax_dendro.set_title('Hierarchical Clustering Dendrogram')
        ax_dendro.set_xlabel('')
        ax_dendro.set_ylabel('Distance')
        
        # Cluster size bar
        ax_cluster = self.figure.add_subplot(gs[1, 1])
        cluster_sizes = self.clustered_data['cluster'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        ax_cluster.barh(range(len(cluster_sizes)), cluster_sizes.values, color=colors)
        ax_cluster.set_yticks(range(len(cluster_sizes)))
        ax_cluster.set_yticklabels([f'Cluster {i+1}' for i in cluster_sizes.index])
        ax_cluster.set_xlabel('Number of Cells')
        ax_cluster.set_title('Cluster Sizes')
        
        # Feature labels
        ax_features = self.figure.add_subplot(gs[0, 1])
        feature_names = self.clustered_data.drop('cluster', axis=1).columns
        ax_features.text(0.5, 0.5, f'Features ({len(feature_names)}):\n' + 
                        '\n'.join(feature_names[:10]) + ('...' if len(feature_names) > 10 else ''),
                        ha='center', va='center', transform=ax_features.transAxes, fontsize=8)
        ax_features.set_title('Feature List')
        ax_features.axis('off')
        
        self.canvas.draw()
    
    def _explore_clusters(self):
        """Open cluster explorer window."""
        if self.clustered_data is None:
            return
        
        # Get cluster info
        cluster_info = []
        for cluster_id in sorted(self.clustered_data['cluster'].unique()):
            cluster_cells = self.clustered_data[self.clustered_data['cluster'] == cluster_id]
            cluster_info.append({
                'cluster_id': cluster_id,
                'size': len(cluster_cells),
                'cells': cluster_cells.index.tolist()
            })
        
        # Open explorer dialog
        explorer = ClusterExplorerDialog(cluster_info, self.feature_dataframe, self.parent())
        explorer.exec_()
    
    def _save_heatmap(self):
        """Save the heatmap as an image."""
        if self.figure is None:
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Heatmap", "cell_clustering_heatmap.png", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Success", f"Heatmap saved to: {file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Error saving heatmap: {str(e)}")


# --------------------------
# Cluster Explorer Dialog
# --------------------------
class ClusterExplorerDialog(QtWidgets.QDialog):
    def __init__(self, cluster_info, feature_dataframe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Explorer")
        self.setModal(True)
        self.setMinimumSize(1000, 700)
        self.cluster_info = cluster_info
        self.feature_dataframe = feature_dataframe
        self.current_cluster = None
        self.cell_images = []
        
        self._create_ui()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title_label = QtWidgets.QLabel("Cluster Explorer")
        title_label.setStyleSheet("QLabel { font-weight: bold; font-size: 16px; }")
        layout.addWidget(title_label)
        
        # Controls
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Cluster selection
        controls_layout.addWidget(QtWidgets.QLabel("Select Cluster:"))
        self.cluster_combo = QtWidgets.QComboBox()
        for info in self.cluster_info:
            self.cluster_combo.addItem(f"Cluster {info['cluster_id']} ({info['size']} cells)", info)
        self.cluster_combo.currentIndexChanged.connect(self._on_cluster_changed)
        controls_layout.addWidget(self.cluster_combo)
        
        # Channel selection
        controls_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        controls_layout.addWidget(self.channel_combo)
        
        # Load images button
        self.load_btn = QtWidgets.QPushButton("Load Cell Images")
        self.load_btn.clicked.connect(self._load_cell_images)
        controls_layout.addWidget(self.load_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Image grid
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        layout.addWidget(self.scroll_area)
        
        # Status
        self.status_label = QtWidgets.QLabel("Select a cluster and channel to view cell images")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize
        self._populate_channels()
        self._on_cluster_changed()
    
    def _populate_channels(self):
        """Populate channel combo box with available channels."""
        # Get channels from feature dataframe columns
        channels = set()
        for col in self.feature_dataframe.columns:
            if '_mean' in col:
                channel = col.replace('_mean', '')
                channels.add(channel)
        
        self.channel_combo.addItems(sorted(channels))
    
    def _on_cluster_changed(self):
        """Handle cluster selection change."""
        self.current_cluster = self.cluster_combo.currentData()
        if self.current_cluster:
            self.status_label.setText(f"Selected Cluster {self.current_cluster['cluster_id']} with {self.current_cluster['size']} cells")
    
    def _load_cell_images(self):
        """Load and display cell images for the selected cluster."""
        if not self.current_cluster:
            return
        
        channel = self.channel_combo.currentText()
        if not channel:
            QtWidgets.QMessageBox.warning(self, "No Channel", "Please select a channel.")
            return
        
        try:
            # Get parent window to access loader and segmentation masks
            parent_window = self.parent()
            if not hasattr(parent_window, 'loader') or not hasattr(parent_window, 'segmentation_masks'):
                QtWidgets.QMessageBox.warning(self, "No Data", "Cannot access image data. Please ensure segmentation masks are loaded.")
                return
            
            # Clear previous images
            self.cell_images = []
            
            # Create image grid
            grid_widget = QtWidgets.QWidget()
            grid_layout = QtWidgets.QGridLayout(grid_widget)
            
            # Get cell data for this cluster
            cluster_cells = self.current_cluster['cells']
            
            # Limit to first 20 cells for performance
            max_cells = min(20, len(cluster_cells))
            
            for i, cell_idx in enumerate(cluster_cells[:max_cells]):
                if i >= max_cells:
                    break
                
                # Get cell data
                cell_data = self.feature_dataframe.iloc[cell_idx]
                acq_id = cell_data['acquisition_id']
                
                # Get mask and image
                if acq_id in parent_window.segmentation_masks:
                    mask = parent_window.segmentation_masks[acq_id]
                    cell_id = int(cell_data['cell_id'])
                    
                    # Create cell mask
                    cell_mask = (mask == cell_id).astype(np.uint8)
                    
                    if np.any(cell_mask):
                        # Load channel image
                        try:
                            channel_img = parent_window.loader.get_image(acq_id, channel)
                            
                            # Apply mask to image
                            masked_img = channel_img * cell_mask
                            
                            # Create image widget
                            img_widget = self._create_image_widget(masked_img, f"Cell {cell_id}")
                            grid_layout.addWidget(img_widget, i // 4, i % 4)
                            
                            self.cell_images.append({
                                'cell_id': cell_id,
                                'acquisition_id': acq_id,
                                'image': masked_img
                            })
                            
                        except Exception as e:
                            print(f"Error loading image for cell {cell_id}: {e}")
                            continue
            
            self.scroll_area.setWidget(grid_widget)
            self.status_label.setText(f"Loaded {len(self.cell_images)} cell images for Cluster {self.current_cluster['cluster_id']}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error loading cell images: {str(e)}")
    
    def _create_image_widget(self, image, title):
        """Create a widget to display a cell image."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Create matplotlib figure
        fig = Figure(figsize=(2, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        
        layout.addWidget(canvas)
        return widget


# --------------------------
# Main Window
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMC .mcd File Viewer")
        
        # Set window size to full screen
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(screen.width(), screen.height())

        # State
        self.loader: Optional[MCDLoader] = None
        self.current_path: Optional[str] = None
        self.acquisitions: List[AcquisitionInfo] = []
        self.current_acq_id: Optional[str] = None

        # Per (acq_id, channel) annotations → label
        self.annotations: Dict[Tuple[str, str], str] = {}
        self.annotation_labels = ["Unlabeled", "High-quality", "Low-quality", "Artifact/Exclude"]
        
        # Store last selected channels for auto-selection
        self.last_selected_channels: List[str] = []

        # Widgets
        self.canvas = MplCanvas(width=6, height=6, dpi=100)
        self.open_btn = QtWidgets.QPushButton("Open .mcd")
        self.acq_combo = QtWidgets.QComboBox()
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect all")
        self.view_btn = QtWidgets.QPushButton("View selected")
        self.comparison_btn = QtWidgets.QPushButton("Comparison mode")
        self.segment_btn = QtWidgets.QPushButton("Cell Segmentation")
        self.load_masks_btn = QtWidgets.QPushButton("Load Masks")
        self.extract_features_btn = QtWidgets.QPushButton("Extract Features")
        self.clustering_btn = QtWidgets.QPushButton("Cell Clustering")
        self.export_btn = QtWidgets.QPushButton("Export to OME-TIFF")
        
        # Visualization options
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        self.grid_view_chk = QtWidgets.QCheckBox("Grid view for multiple channels")
        self.grid_view_chk.setChecked(True)
        self.segmentation_overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay")
        self.segmentation_overlay_chk.toggled.connect(self._on_segmentation_overlay_toggled)
        
        # Custom scaling controls
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.custom_scaling_chk.toggled.connect(self._on_custom_scaling_toggled)
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        
        # Channel selection for per-channel scaling
        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.scaling_channel_combo = QtWidgets.QComboBox()
        self.scaling_channel_combo.currentTextChanged.connect(self._on_scaling_channel_changed)
        channel_row.addWidget(self.scaling_channel_combo)
        channel_row.addStretch()
        scaling_layout.addLayout(channel_row)
        
        # Slider controls
        slider_layout = QtWidgets.QVBoxLayout()
        
        # Min slider
        min_row = QtWidgets.QHBoxLayout()
        min_row.addWidget(QtWidgets.QLabel("Min:"))
        self.min_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self._on_slider_changed)
        min_row.addWidget(self.min_slider)
        self.min_label = QtWidgets.QLabel("0.000")
        self.min_label.setMinimumWidth(60)
        min_row.addWidget(self.min_label)
        slider_layout.addLayout(min_row)
        
        # Max slider
        max_row = QtWidgets.QHBoxLayout()
        max_row.addWidget(QtWidgets.QLabel("Max:"))
        self.max_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.setValue(1000)
        self.max_slider.valueChanged.connect(self._on_slider_changed)
        max_row.addWidget(self.max_slider)
        self.max_label = QtWidgets.QLabel("1000.000")
        self.max_label.setMinimumWidth(60)
        max_row.addWidget(self.max_label)
        slider_layout.addLayout(max_row)
        
        scaling_layout.addLayout(slider_layout)
        
        # Arcsinh normalization controls
        arcsinh_layout = QtWidgets.QHBoxLayout()
        arcsinh_layout.addWidget(QtWidgets.QLabel("Arcsinh Co-factor:"))
        self.cofactor_spinbox = QtWidgets.QDoubleSpinBox()
        self.cofactor_spinbox.setRange(0.1, 100.0)
        self.cofactor_spinbox.setDecimals(1)
        self.cofactor_spinbox.setValue(5.0)
        self.cofactor_spinbox.setSingleStep(0.5)
        arcsinh_layout.addWidget(self.cofactor_spinbox)
        arcsinh_layout.addStretch()
        scaling_layout.addLayout(arcsinh_layout)
        
        # Control buttons
        button_row = QtWidgets.QHBoxLayout()
        self.percentile_btn = QtWidgets.QPushButton("Percentile Scaling")
        self.percentile_btn.clicked.connect(self._percentile_scaling)
        button_row.addWidget(self.percentile_btn)
        
        self.arcsinh_btn = QtWidgets.QPushButton("Arcsinh Normalization")
        self.arcsinh_btn.clicked.connect(self._arcsinh_normalization)
        button_row.addWidget(self.arcsinh_btn)
        
        self.default_range_btn = QtWidgets.QPushButton("Default Range")
        self.default_range_btn.clicked.connect(self._default_range)
        button_row.addWidget(self.default_range_btn)
        
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_scaling)
        self.apply_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_row.addWidget(self.apply_btn)
        
        scaling_layout.addLayout(button_row)
        self.scaling_frame.setVisible(False)
        
        # Store per-channel scaling values
        self.channel_scaling = {}  # {channel_name: {'min': value, 'max': value}}
        
        # Arcsinh normalization state
        self.arcsinh_enabled = False
        
        # Segmentation state
        self.segmentation_masks = {}  # {acq_id: mask_array}
        self.segmentation_colors = {}  # {acq_id: colors_array}
        self.segmentation_overlay = False
        self.preprocessing_cache = PreprocessingCache()
        
        # Feature extraction state
        self.feature_dataframe = None  # Store extracted features in memory
        
        # Color assignment for RGB composite
        self.color_assignment_frame = QtWidgets.QFrame()
        self.color_assignment_frame.setFrameStyle(QtWidgets.QFrame.Box)
        color_layout = QtWidgets.QVBoxLayout(self.color_assignment_frame)
        color_layout.addWidget(QtWidgets.QLabel("Color Assignment (for RGB composite):"))
        
        color_row_layout = QtWidgets.QHBoxLayout()
        color_row_layout.addWidget(QtWidgets.QLabel("Red:"))
        self.red_combo = QtWidgets.QComboBox()
        self.red_combo.setMaximumWidth(120)
        color_row_layout.addWidget(self.red_combo)
        
        color_row_layout.addWidget(QtWidgets.QLabel("Green:"))
        self.green_combo = QtWidgets.QComboBox()
        self.green_combo.setMaximumWidth(120)
        color_row_layout.addWidget(self.green_combo)
        
        color_row_layout.addWidget(QtWidgets.QLabel("Blue:"))
        self.blue_combo = QtWidgets.QComboBox()
        self.blue_combo.setMaximumWidth(120)
        color_row_layout.addWidget(self.blue_combo)
        
        color_layout.addLayout(color_row_layout)

        self.ann_combo = QtWidgets.QComboBox()
        self.ann_combo.addItems(self.annotation_labels)
        self.ann_apply_btn = QtWidgets.QPushButton("Apply label")
        self.ann_save_btn = QtWidgets.QPushButton("Save annotations CSV")
        self.ann_load_btn = QtWidgets.QPushButton("Load annotations CSV")

        # Metadata display
        self.metadata_text = QtWidgets.QTextEdit()
        self.metadata_text.setMaximumHeight(150)
        self.metadata_text.setReadOnly(True)

        # Left panel layout
        controls = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(controls)
        v.addWidget(self.open_btn)

        v.addWidget(QtWidgets.QLabel("Acquisition:"))
        v.addWidget(self.acq_combo)

        v.addWidget(QtWidgets.QLabel("Channels:"))
        v.addWidget(self.channel_list, 1)
        
        # Channel control buttons
        channel_btn_row = QtWidgets.QHBoxLayout()
        channel_btn_row.addWidget(self.deselect_all_btn)
        channel_btn_row.addStretch()
        v.addLayout(channel_btn_row)
        
        # Visualization options
        v.addWidget(self.grayscale_chk)
        v.addWidget(self.grid_view_chk)
        v.addWidget(self.segmentation_overlay_chk)
        v.addWidget(self.custom_scaling_chk)
        v.addWidget(self.scaling_frame)
        v.addWidget(self.color_assignment_frame)

        ann_row = QtWidgets.QHBoxLayout()
        ann_row.addWidget(QtWidgets.QLabel("Annotation:"))
        ann_row.addWidget(self.ann_combo, 1)
        v.addLayout(ann_row)
        v.addWidget(self.ann_apply_btn)

        v.addSpacing(8)
        v.addWidget(self.view_btn)
        v.addWidget(self.comparison_btn)
        v.addWidget(self.segment_btn)
        v.addWidget(self.load_masks_btn)
        v.addWidget(self.extract_features_btn)
        v.addWidget(self.clustering_btn)
        v.addWidget(self.export_btn)
        v.addSpacing(8)
        v.addWidget(self.ann_save_btn)
        v.addWidget(self.ann_load_btn)
        v.addSpacing(8)
        
        v.addWidget(QtWidgets.QLabel("Metadata:"))
        v.addWidget(self.metadata_text)
        v.addStretch(1)

        # Splitter
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        leftw = QtWidgets.QWidget()
        leftw.setLayout(v)
        splitter.addWidget(leftw)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        act_open = file_menu.addAction("Open .mcd…")
        act_open.triggered.connect(self._open_dialog)
        file_menu.addSeparator()
        act_export = file_menu.addAction("Export to OME-TIFF…")
        act_export.triggered.connect(self._export_ome_tiff)
        file_menu.addSeparator()
        act_load_masks = file_menu.addAction("Load Segmentation Masks…")
        act_load_masks.triggered.connect(self._load_segmentation_masks)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit")
        act_quit.triggered.connect(self.close)

        # Analysis menu
        analysis_menu = self.menuBar().addMenu("&Analysis")
        act_clustering = analysis_menu.addAction("Cell Clustering…")
        act_clustering.triggered.connect(self._open_clustering_dialog)

        # Signals
        self.open_btn.clicked.connect(self._open_dialog)
        self.acq_combo.currentIndexChanged.connect(self._on_acq_changed)
        self.deselect_all_btn.clicked.connect(self._deselect_all_channels)
        self.channel_list.itemChanged.connect(self._on_channel_selection_changed)
        self.view_btn.clicked.connect(self._view_selected)
        self.comparison_btn.clicked.connect(self._comparison)
        self.segment_btn.clicked.connect(self._run_segmentation)
        self.load_masks_btn.clicked.connect(self._load_segmentation_masks)
        self.extract_features_btn.clicked.connect(self._extract_features)
        self.clustering_btn.clicked.connect(self._open_clustering_dialog)
        self.export_btn.clicked.connect(self._export_ome_tiff)
        self.ann_apply_btn.clicked.connect(self._apply_annotation)
        self.ann_save_btn.clicked.connect(self._save_annotations)
        self.ann_load_btn.clicked.connect(self._load_annotations)

        # Loader
        try:
            self.loader = MCDLoader()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Missing dependency", str(e))

    # ---------- File open ----------
    def _open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open IMC .mcd file", "", "IMC MCD files (*.mcd);;All files (*.*)"
        )
        if path:
            self._load_mcd(path)

    def _load_mcd(self, path: str):
        if self.loader is None:
            try:
                self.loader = MCDLoader()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Dependency error", str(e))
                return
        try:
            self.loader.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Failed to open {path}\n\n{e}")
            return
        self.current_path = path
        self.acquisitions = self.loader.list_acquisitions()
        self.acq_combo.clear()
        for ai in self.acquisitions:
            label = ai.name + (f"({ai.well})" if ai.well else "")
            self.acq_combo.addItem(label, ai.id)
        if self.acquisitions:
            self._populate_channels(self.acquisitions[0].id)

    # ---------- Acquisition / channels ----------
    def _on_acq_changed(self, idx: int):
        acq_id = self.acq_combo.itemData(idx)
        if acq_id:
            self._populate_channels(acq_id)
            # Update scaling channel combo when acquisition changes
            if self.custom_scaling_chk.isChecked():
                self._update_scaling_channel_combo()

    def _populate_channels(self, acq_id: str):
        self.current_acq_id = acq_id
        self.channel_list.clear()
        try:
            chans = self.loader.get_channels(acq_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Channels error", str(e))
            return
        
        # Pre-select channels that were selected in the previous acquisition
        selected_channels = []
        for ch in chans:
            item = QtWidgets.QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            
            # Check if this channel was selected in the previous acquisition
            if ch in self.last_selected_channels:
                item.setCheckState(Qt.Checked)
                selected_channels.append(ch)
            else:
                item.setCheckState(Qt.Unchecked)
            
            self.channel_list.addItem(item)
        
        # Update color assignment dropdowns
        self._populate_color_assignments(chans)
        
        # Auto-load image if channels were pre-selected
        if selected_channels:
            self._auto_load_image(selected_channels)
        
        # Update metadata display
        ai = next(a for a in self.acquisitions if a.id == acq_id)
        metadata_text = f"Acquisition: {ai.name}\n"
        if ai.well:
            metadata_text += f"{ai.well}\n"
        if ai.size[0] and ai.size[1]:
            metadata_text += f"Size: {ai.size[1]} x {ai.size[0]} pixels\n"
        metadata_text += f"Channels: {len(ai.channels)}\n\n"
        
        # Add GPU info if available
        if _HAVE_TORCH:
            gpu_info = self._get_gpu_info()
            if gpu_info:
                metadata_text += f"GPU: {gpu_info}\n\n"
        
        if ai.metadata:
            metadata_text += "Metadata:\n"
            for key, value in ai.metadata.items():
                metadata_text += f"  {key}: {value}\n"
        
        self.metadata_text.setPlainText(metadata_text)

    def _on_channel_selection_changed(self):
        """Update color assignment dropdowns when channel selection changes."""
        selected_channels = self._selected_channels()
        self._populate_color_assignments(selected_channels)
        
        # Clear any color assignments that are no longer in the selected channels
        self._clear_invalid_color_assignments(selected_channels)

    def _populate_color_assignments(self, channels: List[str]):
        """Populate the color assignment dropdowns with selected channels only."""
        # Clear existing items
        self.red_combo.clear()
        self.green_combo.clear()
        self.blue_combo.clear()
        
        # Add "None" option
        self.red_combo.addItem("None", None)
        self.green_combo.addItem("None", None)
        self.blue_combo.addItem("None", None)
        
        # Add only selected channels
        for ch in channels:
            self.red_combo.addItem(ch, ch)
            self.green_combo.addItem(ch, ch)
            self.blue_combo.addItem(ch, ch)
        
        # Set default assignments to None
        self.red_combo.setCurrentIndex(0)  # "None" is at index 0
        self.green_combo.setCurrentIndex(0)  # "None" is at index 0
        self.blue_combo.setCurrentIndex(0)  # "None" is at index 0

    def _clear_invalid_color_assignments(self, selected_channels: List[str]):
        """Clear color assignments that are no longer in the selected channels."""
        # Check each color assignment and clear if not in selected channels
        red_channel = self.red_combo.currentData()
        green_channel = self.green_combo.currentData()
        blue_channel = self.blue_combo.currentData()
        
        if red_channel and red_channel not in selected_channels:
            # Find "None" option and select it
            for i in range(self.red_combo.count()):
                if self.red_combo.itemData(i) is None:
                    self.red_combo.setCurrentIndex(i)
                    break
        
        if green_channel and green_channel not in selected_channels:
            # Find "None" option and select it
            for i in range(self.green_combo.count()):
                if self.green_combo.itemData(i) is None:
                    self.green_combo.setCurrentIndex(i)
                    break
        
        if blue_channel and blue_channel not in selected_channels:
            # Find "None" option and select it
            for i in range(self.blue_combo.count()):
                if self.blue_combo.itemData(i) is None:
                    self.blue_combo.setCurrentIndex(i)
                    break

    def _deselect_all_channels(self):
        """Deselect all channels in the channel list."""
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.channel_list.clearSelection()
        
        # Clear all color assignments when deselecting all channels
        self._populate_color_assignments([])

    def _selected_channels(self) -> List[str]:
        chans: List[str] = []
        for i in range(self.channel_list.count()):
            it = self.channel_list.item(i)
            if it.checkState() == Qt.Checked or it.isSelected():
                chans.append(it.text())
        # unique, preserve order
        seen = set()
        uniq = []
        for c in chans:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        """Get acquisition subtitle showing well/description instead of acquisition number."""
        acq_info = next((ai for ai in self.acquisitions if ai.id == acq_id), None)
        if not acq_info:
            return "Unknown"
        
        # Use well if available, otherwise use name (which might be more descriptive)
        if acq_info.well:
            return f"{acq_info.well}"
        else:
            return acq_info.name

    def _on_custom_scaling_toggled(self):
        """Handle custom scaling checkbox toggle."""
        self.scaling_frame.setVisible(self.custom_scaling_chk.isChecked())
        if self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
            self._load_channel_scaling()

    def _update_scaling_channel_combo(self):
        """Update the scaling channel combo box with available channels."""
        self.scaling_channel_combo.clear()
        if self.current_acq_id is None:
            return
        
        # Get all available channels for current acquisition
        if hasattr(self, 'acquisitions') and self.acquisitions:
            acq_info = next((ai for ai in self.acquisitions if ai.id == self.current_acq_id), None)
            if acq_info and hasattr(acq_info, 'channels'):
                for channel in acq_info.channels:
                    self.scaling_channel_combo.addItem(channel)
        
        # Select first channel if available
        if self.scaling_channel_combo.count() > 0:
            self.scaling_channel_combo.setCurrentIndex(0)
            self._load_channel_scaling()

    def _on_scaling_channel_changed(self):
        """Handle changes to the scaling channel selection."""
        if self.custom_scaling_chk.isChecked():
            self._load_channel_scaling()

    def _on_slider_changed(self):
        """Handle changes to the min/max sliders."""
        if self.custom_scaling_chk.isChecked():
            self._update_slider_labels()
            # Don't auto-refresh display - user must click Apply

    def _update_slider_labels(self):
        """Update the min/max labels based on slider values."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel or self.current_acq_id is None:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            img_min, img_max = float(np.min(img)), float(np.max(img))
            
            # Convert slider values (0-1000) to actual image range
            min_val = img_min + (self.min_slider.value() / 1000.0) * (img_max - img_min)
            max_val = img_min + (self.max_slider.value() / 1000.0) * (img_max - img_min)
            
            self.min_label.setText(f"{min_val:.3f}")
            self.max_label.setText(f"{max_val:.3f}")
        except Exception as e:
            print(f"Error updating slider labels: {e}")

    def _load_channel_scaling(self):
        """Load scaling values for the currently selected channel."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        if current_channel in self.channel_scaling:
            # Load saved values
            min_val = self.channel_scaling[current_channel]['min']
            max_val = self.channel_scaling[current_channel]['max']
        else:
            # Use default range (full image range)
            if self.current_acq_id is None:
                return
            try:
                img = self.loader.get_image(self.current_acq_id, current_channel)
                min_val = float(np.min(img))
                max_val = float(np.max(img))
            except Exception as e:
                print(f"Error loading channel scaling: {e}")
                return
        
        # Update sliders based on actual values
        self._update_sliders_from_values(min_val, max_val)

    def _save_channel_scaling(self):
        """Save current scaling values for the selected channel."""
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel or self.current_acq_id is None:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            img_min, img_max = float(np.min(img)), float(np.max(img))
            
            # Convert slider values to actual image range
            min_val = img_min + (self.min_slider.value() / 1000.0) * (img_max - img_min)
            max_val = img_min + (self.max_slider.value() / 1000.0) * (img_max - img_min)
            
            self.channel_scaling[current_channel] = {'min': min_val, 'max': max_val}
        except Exception as e:
            print(f"Error saving channel scaling: {e}")

    def _update_sliders_from_values(self, min_val, max_val):
        """Update sliders based on actual min/max values."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            img_min, img_max = float(np.min(img)), float(np.max(img))
            
            # Convert actual values to slider positions (0-1000)
            if img_max > img_min:
                min_slider_val = int(((min_val - img_min) / (img_max - img_min)) * 1000)
                max_slider_val = int(((max_val - img_min) / (img_max - img_min)) * 1000)
            else:
                min_slider_val = 0
                max_slider_val = 1000
            
            # Update sliders without triggering valueChanged
            self.min_slider.blockSignals(True)
            self.max_slider.blockSignals(True)
            self.min_slider.setValue(min_slider_val)
            self.max_slider.setValue(max_slider_val)
            self.min_slider.blockSignals(False)
            self.max_slider.blockSignals(False)
            
            # Update labels
            self._update_slider_labels()
        except Exception as e:
            print(f"Error updating sliders from values: {e}")

    def _percentile_scaling(self):
        """Set scaling using robust percentile scaling (1st-99th percentiles)."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            # Use robust percentile scaling function
            scaled_img = robust_percentile_scale(img, low=1.0, high=99.0)
            
            # Get the actual min/max values that were used for scaling
            min_val = float(np.percentile(img, 1))
            max_val = float(np.percentile(img, 99))
            
            self._update_sliders_from_values(min_val, max_val)
            
            # Disable arcsinh normalization when using other scaling methods
            self.arcsinh_enabled = False
            # Don't auto-apply - user must click Apply button
        except Exception as e:
            print(f"Error in percentile scaling: {e}")

    def _arcsinh_normalization(self):
        """Apply arcsinh normalization with configurable co-factor."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            cofactor = self.cofactor_spinbox.value()
            
            # Apply arcsinh normalization
            normalized_img = arcsinh_normalize(img, cofactor=cofactor)
            
            # Get the min/max values of the normalized image for scaling
            min_val = float(np.min(normalized_img))
            max_val = float(np.max(normalized_img))
            
            self._update_sliders_from_values(min_val, max_val)
            
            # Enable arcsinh normalization for this channel
            self.arcsinh_enabled = True
            # Don't auto-apply - user must click Apply button
        except Exception as e:
            print(f"Error in arcsinh normalization: {e}")

    def _default_range(self):
        """Set scaling to the image's actual min/max range."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        try:
            img = self.loader.get_image(self.current_acq_id, current_channel)
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            
            self._update_sliders_from_values(min_val, max_val)
            
            # Disable arcsinh normalization when using other scaling methods
            self.arcsinh_enabled = False
            # Don't auto-apply - user must click Apply button
        except Exception as e:
            print(f"Error in default range: {e}")

    def _load_image_with_normalization(self, acq_id: str, channel: str) -> np.ndarray:
        """Load image and apply arcsinh normalization if enabled."""
        img = self.loader.get_image(acq_id, channel)
        
        if self.arcsinh_enabled:
            cofactor = self.cofactor_spinbox.value()
            img = arcsinh_normalize(img, cofactor=cofactor)
        
        return img

    def _apply_scaling(self):
        """Apply the current scaling settings to the selected channel and refresh display."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Save current scaling values
        self._save_channel_scaling()
        
        # Refresh display
        self._view_selected()


    # ---------- View ----------
    def _view_selected(self):
        if self.current_acq_id is None:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Open a .mcd and select an acquisition.")
            return
        chans = self._selected_channels()
        if not chans:
            QtWidgets.QMessageBox.information(self, "No channels", "Select one or more channels.")
            return
        
        # Store selected channels for auto-selection in next acquisition
        self.last_selected_channels = chans.copy()
        
        grayscale = self.grayscale_chk.isChecked()
        grid_view = self.grid_view_chk.isChecked()
        
        # Get custom scaling values if enabled
        # For single channel view, use that channel's scaling
        # For RGB/grid view, we'll handle per-channel scaling in the display methods
        custom_min = None
        custom_max = None
        if self.custom_scaling_chk.isChecked() and len(chans) == 1:
            # For single channel, use the scaling for that specific channel
            channel = chans[0]
            if channel in self.channel_scaling:
                custom_min = self.channel_scaling[channel]['min']
                custom_max = self.channel_scaling[channel]['max']
        
        try:
            if len(chans) == 1 and not grid_view:
                # Single channel view
                img = self._load_image_with_normalization(self.current_acq_id, chans[0])
                
                # Apply segmentation overlay if enabled
                if self.segmentation_overlay:
                    img = self._get_segmentation_overlay(img)
                
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                title = f"{chans[0]}\n{acq_subtitle}"
                if self.segmentation_overlay:
                    title += " (with segmentation overlay)"
                self.canvas.show_image(img, title, grayscale=grayscale, raw_img=img, custom_min=custom_min, custom_max=custom_max)
            elif len(chans) <= 3 and not grid_view:
                # RGB composite view using user-selected color assignments
                self._show_rgb_composite(chans, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True or >3 channels)
                images = [self._load_image_with_normalization(self.current_acq_id, c) for c in chans]
                
                # Apply segmentation overlay to all images if enabled
                if self.segmentation_overlay:
                    images = [self._get_segmentation_overlay(img) for img in images]
                
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                # Add acquisition subtitle to each channel title
                titles = [f"{ch}\n{acq_subtitle}" for ch in chans]
                if self.segmentation_overlay:
                    titles = [f"{ch}\n{acq_subtitle} (segmented)" for ch in chans]
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=chans, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "View error", str(e))

    def _auto_load_image(self, selected_channels: List[str]):
        """Automatically load and display image for pre-selected channels."""
        try:
            grayscale = self.grayscale_chk.isChecked()
            grid_view = self.grid_view_chk.isChecked()
            
            # Get custom scaling values if enabled
            custom_min = None
            custom_max = None
            if self.custom_scaling_chk.isChecked() and len(selected_channels) == 1:
                # For single channel, use the scaling for that specific channel
                channel = selected_channels[0]
                if channel in self.channel_scaling:
                    custom_min = self.channel_scaling[channel]['min']
                    custom_max = self.channel_scaling[channel]['max']
            
            if len(selected_channels) == 1 and not grid_view:
                # Single channel view
                img = self._load_image_with_normalization(self.current_acq_id, selected_channels[0])
                
                # Apply segmentation overlay if enabled
                if self.segmentation_overlay:
                    img = self._get_segmentation_overlay(img)
                
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                title = f"{selected_channels[0]}\n{acq_subtitle}"
                if self.segmentation_overlay:
                    title += " (with segmentation overlay)"
                self.canvas.show_image(img, title, grayscale=grayscale, raw_img=img, custom_min=custom_min, custom_max=custom_max)
            elif len(selected_channels) <= 3 and not grid_view:
                # RGB composite view using user-selected color assignments
                self._show_rgb_composite(selected_channels, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True or >3 channels)
                images = [self._load_image_with_normalization(self.current_acq_id, c) for c in selected_channels]
                
                # Apply segmentation overlay to all images if enabled
                if self.segmentation_overlay:
                    images = [self._get_segmentation_overlay(img) for img in images]
                
                # Get acquisition subtitle
                acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
                # Add acquisition subtitle to each channel title
                titles = [f"{ch}\n{acq_subtitle}" for ch in selected_channels]
                if self.segmentation_overlay:
                    titles = [f"{ch}\n{acq_subtitle} (segmented)" for ch in selected_channels]
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=selected_channels, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked())
        except Exception as e:
            print(f"Auto-load error: {e}")

    def _show_rgb_composite(self, selected_channels: List[str], grayscale: bool):
        """Show RGB composite using user-selected color assignments."""
        # Get user-selected color assignments
        red_channel = self.red_combo.currentData()
        green_channel = self.green_combo.currentData()
        blue_channel = self.blue_combo.currentData()
        
        # If currentData() returns None, try currentText()
        if red_channel is None:
            red_channel = self.red_combo.currentText() if self.red_combo.currentText() != "None" else None
        if green_channel is None:
            green_channel = self.green_combo.currentText() if self.green_combo.currentText() != "None" else None
        if blue_channel is None:
            blue_channel = self.blue_combo.currentText() if self.blue_combo.currentText() != "None" else None
        
        # If no color assignments are made, use the first 1-3 selected channels as default
        if not red_channel and not green_channel and not blue_channel:
            if len(selected_channels) >= 1:
                red_channel = selected_channels[0]
            if len(selected_channels) >= 2:
                green_channel = selected_channels[1]
            if len(selected_channels) >= 3:
                blue_channel = selected_channels[2]
        
        # Create RGB stack based on user selections
        rgb_channels = []
        rgb_titles = []
        raw_channels = []  # Store raw images for colorbar
        
        # Get the first selected channel to determine image size
        first_img = None
        if selected_channels:
            first_img = self._load_image_with_normalization(self.current_acq_id, selected_channels[0])
        
        if first_img is None:
            QtWidgets.QMessageBox.information(self, "No RGB channels", "Please select at least one channel for RGB composite.")
            return
        
        # Always create 3 channels (R, G, B) even if some are empty
        for color, channel, color_name in [(red_channel, red_channel, 'Red'), (green_channel, green_channel, 'Green'), (blue_channel, blue_channel, 'Blue')]:
            # Convert both to strings for comparison to handle any type mismatches
            channel_str = str(channel) if channel else None
            selected_channels_str = [str(c) for c in selected_channels]
            
            if channel and channel_str in selected_channels_str:
                img = self._load_image_with_normalization(self.current_acq_id, channel)
                rgb_channels.append(img)
                raw_channels.append(img)
                rgb_titles.append(f"{channel} ({color_name})")
            else:
                # Add zero-filled channel if not selected
                rgb_channels.append(np.zeros_like(first_img))
                raw_channels.append(np.zeros_like(first_img))
                rgb_titles.append(f"None ({color_name})")
        
        # Ensure we have exactly 3 channels
        while len(rgb_channels) < 3:
            rgb_channels.append(np.zeros_like(first_img))
            raw_channels.append(np.zeros_like(first_img))
            rgb_titles.append(f"None ({['Red', 'Green', 'Blue'][len(rgb_channels)-1]})")
        
        # Stack channels
        stack = np.dstack(rgb_channels)
        raw_stack = np.dstack(raw_channels)
        
        # Apply segmentation overlay if enabled
        if self.segmentation_overlay:
            # Apply overlay to each channel in the stack
            for i in range(stack.shape[2]):
                if not np.all(rgb_channels[i] == 0):  # Only apply to non-zero channels
                    stack[..., i] = self._get_segmentation_overlay(stack[..., i])
        
        # Get acquisition subtitle
        acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
        title = " + ".join(rgb_titles) + f"\n{acq_subtitle}"
        if self.segmentation_overlay:
            title += " (segmented)"
        
        # Clear canvas and show RGB composite with individual colorbars
        self.canvas.fig.clear()
        
        if grayscale:
            # Single grayscale image with colorbar
            ax = self.canvas.fig.add_subplot(111)
            
            # Get per-channel scaling for the first channel
            vmin, vmax = None, None
            if self.custom_scaling_chk.isChecked() and selected_channels and len(selected_channels) > 0:
                first_channel = selected_channels[0]
                if first_channel in self.channel_scaling:
                    vmin = self.channel_scaling[first_channel]['min']
                    vmax = self.channel_scaling[first_channel]['max']
            
            if vmin is None or vmax is None:
                vmin, vmax = np.min(stack[..., 0]), np.max(stack[..., 0])
            
            im = ax.imshow(stack[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            # Add colorbar for grayscale
            cbar = self.canvas.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
            ax.set_title(title)
            ax.axis("off")
        else:
            # RGB composite with individual channel colorbars
            # Create a 2x2 grid: main image (top), colorbars (bottom)
            gs = self.canvas.fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
            
            # Main RGB composite image (spans top row)
            ax_main = self.canvas.fig.add_subplot(gs[0, :])
            im = ax_main.imshow(stack_to_rgb(stack), interpolation="nearest")
            ax_main.set_title(title)
            ax_main.axis("off")
            
            # Individual channel colorbars (bottom row)
            for i, (channel_name, color_name) in enumerate(zip(rgb_titles, ['Red', 'Green', 'Blue'])):
                ax_cbar = self.canvas.fig.add_subplot(gs[1, i])
                
                # Create a colorbar for this channel
                if i < len(raw_channels) and raw_channels[i] is not None and not np.all(raw_channels[i] == 0):
                    # This channel has data
                    raw_min, raw_max = np.min(raw_channels[i]), np.max(raw_channels[i])
                    
                    # Check for per-channel scaling
                    if self.custom_scaling_chk.isChecked() and selected_channels and i < len(selected_channels):
                        channel = selected_channels[i]
                        if channel in self.channel_scaling:
                            raw_min = self.channel_scaling[channel]['min']
                            raw_max = self.channel_scaling[channel]['max']
                    
                    if raw_max > raw_min:  # Valid data range
                        # Create a gradient for the colorbar
                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                        ax_cbar.imshow(gradient, aspect='auto', cmap='gray' if grayscale else ['Reds', 'Greens', 'Blues'][i])
                        ax_cbar.set_xticks([0, 255])
                        ax_cbar.set_xticklabels([f'{raw_min:.1f}', f'{raw_max:.1f}'])
                        ax_cbar.set_yticks([])
                        ax_cbar.set_title(f"{channel_name}", fontsize=10)
                    else:
                        # No data
                        ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes)
                        ax_cbar.set_title(f"{channel_name}", fontsize=10)
                else:
                    # No data for this channel
                    ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes)
                    ax_cbar.set_title(f"{channel_name}", fontsize=10)
                
                ax_cbar.set_xlim(0, 255)
        
        self.canvas.draw()


    # ---------- Annotations ----------
    def _apply_annotation(self):
        if self.current_acq_id is None:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        chans = self._selected_channels()
        if not chans:
            QtWidgets.QMessageBox.information(self, "No channels", "Select channel(s) to annotate.")
            return
        label = self.ann_combo.currentText()
        for ch in chans:
            self.annotations[(self.current_acq_id, ch)] = label
        QtWidgets.QMessageBox.information(self, "Annotated", f"Labeled {len(chans)} channel(s) as '{label}'.")

    def _save_annotations(self):
        if not self.annotations:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "No annotations yet.")
            return
        rows = [
            {"acquisition_id": a, "channel": c, "label": lab}
            for (a, c), lab in self.annotations.items()
        ]
        df = pd.DataFrame(rows).sort_values(["acquisition_id", "channel"])
        base = "channel_annotations.csv"
        if self.current_path:
            stem = os.path.splitext(os.path.basename(self.current_path))[0]
            base = f"{stem}_annotations.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save annotations CSV", base, "CSV (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Annotations saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _load_annotations(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load annotations CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            if not {"acquisition_id", "channel", "label"}.issubset(df.columns):
                raise ValueError("CSV must have columns: acquisition_id, channel, label")
            for _, row in df.iterrows():
                self.annotations[(str(row["acquisition_id"]), str(row["channel"]))] = str(row["label"])
            QtWidgets.QMessageBox.information(self, "Loaded", f"Loaded {len(df)} annotations.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    # ---------- Comparison ----------
    def _comparison(self):
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        # Open the dynamic comparison dialog
        dlg = DynamicComparisonDialog(self.acquisitions, self.loader, self)
        dlg.exec_()

    # ---------- Export ----------
    def _export_ome_tiff(self):
        """Export acquisitions to OME-TIFF format."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        if not _HAVE_TIFFFILE:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency", 
                "tifffile library is required for OME-TIFF export.\n"
                "Install it with: pip install tifffile"
            )
            return
        
        # Open export dialog
        dlg = ExportDialog(self.acquisitions, self.current_acq_id, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        export_type = dlg.get_export_type()
        output_dir = dlg.get_output_directory()
        include_metadata = dlg.get_include_metadata()
        apply_normalization = dlg.get_apply_normalization()
        
        # Create and show progress dialog
        progress_dlg = ProgressDialog("Export to OME-TIFF", self)
        progress_dlg.show()
        
        try:
            if export_type == "single":
                success = self._export_single_acquisition(
                    output_dir, include_metadata, apply_normalization, progress_dlg
                )
            else:
                success = self._export_whole_slide(
                    output_dir, include_metadata, apply_normalization, progress_dlg
                )
            
            progress_dlg.close()
            
            if success and not progress_dlg.is_cancelled():
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", 
                    f"Successfully exported to:\n{output_dir}"
                )
            elif progress_dlg.is_cancelled():
                QtWidgets.QMessageBox.information(
                    self, "Export Cancelled", 
                    "Export was cancelled by user."
                )
        except Exception as e:
            progress_dlg.close()
            QtWidgets.QMessageBox.critical(
                self, "Export Failed", 
                f"Export failed with error:\n{str(e)}"
            )
    
    def _export_single_acquisition(self, output_dir: str, include_metadata: bool, 
                                 apply_normalization: bool, progress_dlg: ProgressDialog) -> bool:
        """Export the currently selected acquisition."""
        if not self.current_acq_id:
            raise ValueError("No acquisition selected")
        
        acq_info = next(ai for ai in self.acquisitions if ai.id == self.current_acq_id)
        
        # Get all channels for this acquisition
        all_channels = self.loader.get_channels(self.current_acq_id)
        if not all_channels:
            raise ValueError("No channels found for this acquisition")
        
        progress_dlg.set_maximum(len(all_channels) + 2)  # +2 for stacking and writing
        progress_dlg.update_progress(0, f"Exporting {acq_info.name}", "Loading channels...")
        
        # Load all channel data
        channel_data = []
        channel_names = []
        
        for i, channel in enumerate(all_channels):
            if progress_dlg.is_cancelled():
                return False
                
            progress_dlg.update_progress(
                i + 1, 
                f"Exporting {acq_info.name}", 
                f"Loading channel {i+1}/{len(all_channels)}: {channel}"
            )
            
            if apply_normalization:
                img = self._load_image_with_normalization(self.current_acq_id, channel)
            else:
                img = self.loader.get_image(self.current_acq_id, channel)
            channel_data.append(img)
            channel_names.append(channel)
        
        if progress_dlg.is_cancelled():
            return False
        
        # Stack channels (C, H, W) for OME-TIFF
        progress_dlg.update_progress(
            len(all_channels) + 1, 
            f"Exporting {acq_info.name}", 
            "Stacking channels..."
        )
        stack = np.stack(channel_data, axis=0)
        
        # Create filename from acquisition name
        safe_name = self._sanitize_filename(acq_info.name)
        if acq_info.well:
            safe_well = self._sanitize_filename(acq_info.well)
            filename = f"{safe_name}_{safe_well}.ome.tiff"
        else:
            filename = f"{safe_name}.ome.tiff"
        
        output_path = os.path.join(output_dir, filename)
        
        # Prepare comprehensive metadata
        metadata = self._create_ome_metadata(
            acq_info, channel_names, include_metadata, stack.shape
        )
        
        # Write OME-TIFF
        progress_dlg.update_progress(
            len(all_channels) + 2, 
            f"Exporting {acq_info.name}", 
            f"Writing {filename}..."
        )
        
        if progress_dlg.is_cancelled():
            return False
            
        tifffile.imwrite(
            output_path,
            stack,
            imagej=True,
            metadata=metadata,
            ome=True
        )
        
        return True
    
    def _export_whole_slide(self, output_dir: str, include_metadata: bool, 
                          apply_normalization: bool, progress_dlg: ProgressDialog) -> bool:
        """Export all acquisitions from the slide."""
        total_acquisitions = len(self.acquisitions)
        progress_dlg.set_maximum(total_acquisitions)
        
        for acq_idx, acq_info in enumerate(self.acquisitions):
            if progress_dlg.is_cancelled():
                return False
            
            progress_dlg.update_progress(
                acq_idx, 
                f"Exporting acquisition {acq_idx + 1}/{total_acquisitions}", 
                f"Processing {acq_info.name}..."
            )
            
            # Get all channels for this acquisition
            all_channels = self.loader.get_channels(acq_info.id)
            if not all_channels:
                print(f"Warning: No channels found for acquisition {acq_info.name}")
                continue
            
            # Load all channel data
            channel_data = []
            channel_names = []
            
            for channel in all_channels:
                if progress_dlg.is_cancelled():
                    return False
                    
                if apply_normalization:
                    img = self._load_image_with_normalization(acq_info.id, channel)
                else:
                    img = self.loader.get_image(acq_info.id, channel)
                channel_data.append(img)
                channel_names.append(channel)
            
            if progress_dlg.is_cancelled():
                return False
            
            # Stack channels (C, H, W) for OME-TIFF
            stack = np.stack(channel_data, axis=0)
            
            # Create filename from acquisition name
            safe_name = self._sanitize_filename(acq_info.name)
            if acq_info.well:
                safe_well = self._sanitize_filename(acq_info.well)
                filename = f"{safe_name}_{safe_well}.ome.tiff"
            else:
                filename = f"{safe_name}.ome.tiff"
            
            output_path = os.path.join(output_dir, filename)
            
            # Prepare comprehensive metadata
            metadata = self._create_ome_metadata(
                acq_info, channel_names, include_metadata, stack.shape
            )
            
            # Write OME-TIFF
            progress_dlg.update_progress(
                acq_idx + 1, 
                f"Exporting acquisition {acq_idx + 1}/{total_acquisitions}", 
                f"Writing {filename}..."
            )
            
            if progress_dlg.is_cancelled():
                return False
                
            tifffile.imwrite(
                output_path,
                stack,
                imagej=True,
                metadata=metadata,
                ome=True
            )
        
        return True
    
    def _create_ome_metadata(self, acq_info: AcquisitionInfo, channel_names: List[str], 
                            include_metadata: bool, stack_shape: Tuple[int, ...]) -> Dict:
        """Create comprehensive OME-TIFF metadata."""
        metadata = {}
        
        # Basic acquisition information
        metadata['AcquisitionID'] = acq_info.id
        metadata['AcquisitionName'] = acq_info.name
        if acq_info.well:
            metadata['Well'] = acq_info.well
        
        # Image dimensions
        if len(stack_shape) >= 3:
            metadata['SizeC'] = stack_shape[0]  # Number of channels
            metadata['SizeT'] = 1  # Time points
            metadata['SizeZ'] = 1  # Z slices
            metadata['SizeY'] = stack_shape[1]  # Height
            metadata['SizeX'] = stack_shape[2]  # Width
        
        # Channel information
        metadata['ChannelNames'] = channel_names
        
        # Get detailed channel information from acquisition
        acq_id = acq_info.id
        if hasattr(self.loader, '_acq_channel_metals') and acq_id in self.loader._acq_channel_metals:
            channel_metals = self.loader._acq_channel_metals[acq_id]
            channel_labels = self.loader._acq_channel_labels[acq_id]
            
            # Create detailed channel metadata
            channel_metadata = []
            for i, (metal, label, name) in enumerate(zip(channel_metals, channel_labels, channel_names)):
                channel_info = {
                    'ID': f"Channel:{i}",
                    'Name': name,
                    'Metal': metal if metal else f"Channel_{i+1}",
                    'Label': label if label else f"Channel_{i+1}"
                }
                channel_metadata.append(channel_info)
            
            metadata['Channels'] = channel_metadata
        
        # Pixel size information (try to extract from metadata)
        pixel_size_x = None
        pixel_size_y = None
        pixel_size_unit = "µm"  # Default unit
        
        if include_metadata and acq_info.metadata:
            # Look for common pixel size keys in metadata
            for key, value in acq_info.metadata.items():
                key_lower = key.lower()
                if 'pixel' in key_lower and 'size' in key_lower:
                    if 'x' in key_lower or 'width' in key_lower:
                        try:
                            pixel_size_x = float(value)
                        except (ValueError, TypeError):
                            pass
                    elif 'y' in key_lower or 'height' in key_lower:
                        try:
                            pixel_size_y = float(value)
                        except (ValueError, TypeError):
                            pass
                elif 'resolution' in key_lower:
                    try:
                        # Sometimes resolution is given as a single value
                        pixel_size_x = pixel_size_y = float(value)
                    except (ValueError, TypeError):
                        pass
                elif 'unit' in key_lower and 'pixel' in key_lower:
                    pixel_size_unit = str(value)
                elif 'microns' in key_lower or 'micrometers' in key_lower:
                    # Sometimes pixel size is given as "microns per pixel"
                    try:
                        pixel_size_x = pixel_size_y = float(value)
                        pixel_size_unit = "µm"
                    except (ValueError, TypeError):
                        pass
            
            # If we found pixel size information, add it to metadata
            if pixel_size_x is not None:
                metadata['PhysicalSizeX'] = pixel_size_x
                metadata['PhysicalSizeXUnit'] = pixel_size_unit
            if pixel_size_y is not None:
                metadata['PhysicalSizeY'] = pixel_size_y
                metadata['PhysicalSizeYUnit'] = pixel_size_unit
            
            # Add all original metadata
            metadata.update(acq_info.metadata)
        
        # OME-TIFF specific metadata
        metadata['ImageJ'] = '1.53c'  # ImageJ version
        metadata['hyperstack'] = 'true'
        metadata['mode'] = 'grayscale'
        metadata['unit'] = pixel_size_unit
        
        # Add acquisition timestamp if available
        if include_metadata and acq_info.metadata:
            for key, value in acq_info.metadata.items():
                if 'time' in key.lower() or 'date' in key.lower():
                    metadata['AcquisitionTime'] = str(value)
                    break
        
        return metadata

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage."""
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Ensure filename is not empty
        if not filename:
            filename = "unnamed"
        
        return filename

    # ---------- Segmentation ----------
    def _run_segmentation(self):
        """Run cell segmentation using Cellpose."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a .mcd first.")
            return
        
        if not _HAVE_CELLPOSE:
            QtWidgets.QMessageBox.critical(
                self, "Missing dependency", 
                "Cellpose library is required for segmentation.\n"
                "Install it with: pip install cellpose"
            )
            return
        
        if not self.current_acq_id:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        
        # Get available channels
        channels = self.loader.get_channels(self.current_acq_id)
        if not channels:
            QtWidgets.QMessageBox.information(self, "No channels", "No channels available for segmentation.")
            return
        
        # Open segmentation dialog
        dlg = SegmentationDialog(channels, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # Get segmentation parameters
        model = dlg.get_model()
        diameter = dlg.get_diameter()
        flow_threshold = dlg.get_flow_threshold()
        cellprob_threshold = dlg.get_cellprob_threshold()
        show_overlay = dlg.get_show_overlay()
        save_masks = dlg.get_save_masks()
        masks_directory = dlg.get_masks_directory()
        gpu_id = dlg.get_selected_gpu()
        preprocessing_config = dlg.get_preprocessing_config()
        segment_all = dlg.get_segment_all()
        
        # Validate preprocessing configuration
        if not preprocessing_config:
            QtWidgets.QMessageBox.warning(self, "No preprocessing configured", "Please configure preprocessing to select channels for segmentation.")
            return
        
        # Get channels from preprocessing config
        nuclear_channels = preprocessing_config.get('nuclear_channels', [])
        cyto_channels = preprocessing_config.get('cyto_channels', [])
        
        if not nuclear_channels:
            QtWidgets.QMessageBox.warning(self, "No nuclear channels", "Please select at least one nuclear channel in the preprocessing configuration.")
            return
        
        if model == "cyto" and not cyto_channels:
            QtWidgets.QMessageBox.warning(self, "No cytoplasm channels", "Please select at least one cytoplasm channel in the preprocessing configuration for whole-cell segmentation.")
            return
        
        try:
            if segment_all:
                # Run segmentation on all acquisitions
                self._perform_segmentation_all_acquisitions(
                    model, diameter, flow_threshold, cellprob_threshold, 
                    show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config
                )
            else:
                # Run segmentation on current acquisition only
                self._perform_segmentation(
                    model, diameter, flow_threshold, cellprob_threshold, 
                    show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Segmentation Failed", 
                f"Segmentation failed with error:\n{str(e)}"
            )
    
    def _perform_segmentation(self, model: str, diameter: int = None, flow_threshold: float = 0.4, 
                            cellprob_threshold: float = 0.0, show_overlay: bool = True, 
                            save_masks: bool = False, masks_directory: str = None, gpu_id = None, preprocessing_config = None):
        """Perform the actual segmentation using Cellpose."""
        # Create progress dialog
        progress_dlg = ProgressDialog("Cell Segmentation", self)
        progress_dlg.show()
        
        try:
            progress_dlg.update_progress(0, "Initializing Cellpose model", "Loading model...")
            
            # Determine GPU usage
            use_gpu = False
            gpu_device = None
            
            if gpu_id == "auto":
                # Auto-detect best GPU
                if _HAVE_TORCH and torch.cuda.is_available():
                    use_gpu = True
                    gpu_device = 0  # Use first CUDA device
                elif _HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    gpu_device = 'mps'
            elif gpu_id is not None:
                # Use specific GPU
                use_gpu = True
                gpu_device = gpu_id
            
            # Initialize Cellpose model
            if model == "nuclei":
                model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Set device if using GPU
            if use_gpu and gpu_device is not None:
                if gpu_device == 'mps':
                    progress_dlg.update_progress(5, "Initializing Cellpose model", "Using Apple Metal Performance Shaders...")
                else:
                    progress_dlg.update_progress(5, "Initializing Cellpose model", f"Using CUDA GPU {gpu_device}...")
            else:
                progress_dlg.update_progress(5, "Initializing Cellpose model", "Using CPU...")
            
            progress_dlg.update_progress(20, "Preprocessing images", "Loading and preprocessing channels...")
            
            # Preprocess and combine channels
            nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                preprocessing_config, progress_dlg
            )
            
            # Prepare input images
            if model == "nuclei":
                # For nuclei model, use only nuclear channel
                images = [nuclear_img]
                channels = [0, 0]  # [cytoplasm, nucleus] - both are nuclear channel
            else:  # cyto3
                # For cyto3 model, use both channels
                if cyto_img is None:
                    cyto_img = nuclear_img  # Fallback to nuclear channel
                images = [cyto_img, nuclear_img]
                channels = [0, 1]  # [cytoplasm, nucleus]
            
            progress_dlg.update_progress(60, "Running segmentation", "Processing with Cellpose...")
            
            # Run segmentation
            masks, flows, styles, diams = model_obj.eval(
                images, 
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                channels=channels
            )
            
            progress_dlg.update_progress(80, "Processing results", "Creating segmentation masks...")
            
            # Store segmentation results
            self.segmentation_masks[self.current_acq_id] = masks[0]  # First (and only) mask
            # Clear colors for this acquisition so they get regenerated
            if self.current_acq_id in self.segmentation_colors:
                del self.segmentation_colors[self.current_acq_id]
            self.segmentation_overlay = show_overlay
            
            # Save masks if requested
            if save_masks:
                self._save_segmentation_masks(masks_directory)
            
            progress_dlg.update_progress(100, "Segmentation complete", f"Found {len(np.unique(masks[0])) - 1} cells")
            
            # Update display if overlay is enabled
            if show_overlay:
                self.segmentation_overlay_chk.setChecked(True)
                self._update_display_with_segmentation()
            
            progress_dlg.close()
            
            # Get channel information from preprocessing config
            nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
            cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
            
            channel_info = ""
            if nuclear_channels:
                channel_info += f"Nuclear: {len(nuclear_channels)} channels"
            if cyto_channels:
                if channel_info:
                    channel_info += f" + Cytoplasm: {len(cyto_channels)} channels"
                else:
                    channel_info += f"Cytoplasm: {len(cyto_channels)} channels"
            
            QtWidgets.QMessageBox.information(
                self, "Segmentation Complete", 
                f"Successfully segmented {len(np.unique(masks[0])) - 1} cells.\n"
                f"Model: {model}\n"
                f"Channels: {channel_info if channel_info else 'Not specified'}"
            )
            
        except Exception as e:
            progress_dlg.close()
            raise e
    
    def _perform_segmentation_all_acquisitions(self, model: str, diameter: int = None, 
                                             flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, 
                                             show_overlay: bool = True, save_masks: bool = False, 
                                             masks_directory: str = None, gpu_id = None, preprocessing_config = None):
        """Perform efficient batch segmentation on all acquisitions."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions available for segmentation.")
            return
        
        # Create progress dialog for batch processing
        progress_dlg = ProgressDialog("Batch Cell Segmentation", self)
        progress_dlg.set_maximum(len(self.acquisitions))
        progress_dlg.show()
        
        try:
            # Estimate optimal batch size based on available memory
            batch_size = self._estimate_optimal_batch_size(preprocessing_config, gpu_id)
            progress_dlg.update_progress(0, "Initializing batch processing", f"Optimal batch size: {batch_size} (0/{total_acquisitions} completed)")
            
            # Initialize Cellpose model once
            progress_dlg.update_progress(0, "Initializing Cellpose model", f"Loading model... (0/{total_acquisitions} completed)")
            
            # Determine GPU usage
            use_gpu = False
            gpu_device = None
            
            if gpu_id == "auto":
                if _HAVE_TORCH and torch.cuda.is_available():
                    use_gpu = True
                    gpu_device = 0
                elif _HAVE_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    gpu_device = 'mps'
            elif gpu_id is not None:
                use_gpu = True
                gpu_device = gpu_id
            
            # Initialize model
            if model == "nuclei":
                model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Process acquisitions in batches
            successful_segmentations = 0
            total_acquisitions = len(self.acquisitions)
            
            for batch_start in range(0, total_acquisitions, batch_size):
                if progress_dlg.is_cancelled():
                    break
                
                batch_end = min(batch_start + batch_size, total_acquisitions)
                batch_acquisitions = self.acquisitions[batch_start:batch_end]
                
                progress_dlg.update_progress(
                    successful_segmentations, 
                    f"Processing batch {batch_start//batch_size + 1}", 
                    f"Loading {len(batch_acquisitions)} acquisitions... ({successful_segmentations}/{total_acquisitions} completed)"
                )
                
                try:
                    # Load and preprocess all acquisitions in this batch
                    batch_data = self._load_batch_acquisitions(
                        batch_acquisitions, preprocessing_config, progress_dlg
                    )
                    
                    if not batch_data:
                        continue
                    
                    # Run segmentation on the entire batch
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Segmenting batch {batch_start//batch_size + 1}",
                        f"Processing {len(batch_data['images'])} images... ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                    masks, flows, styles, diams = model_obj.eval(
                        batch_data['images'],
                        diameter=diameter,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        channels=batch_data['channels']
                    )
                    
                    # Store results using acquisition mapping to ensure correct order
                    acquisition_mapping = batch_data['acquisition_mapping']
                    processed_acquisitions = set()
                    
                    for i, mask in enumerate(masks):
                        if i < len(acquisition_mapping):
                            acq_id = acquisition_mapping[i]
                            
                            # Only process each acquisition once (use the first mask for each acquisition)
                            if acq_id not in processed_acquisitions:
                                self.segmentation_masks[acq_id] = mask
                                # Clear colors for this acquisition so they get regenerated
                                if acq_id in self.segmentation_colors:
                                    del self.segmentation_colors[acq_id]
                                successful_segmentations += 1
                                processed_acquisitions.add(acq_id)
                                
                                # Save masks if requested
                                if save_masks:
                                    self._save_segmentation_masks_for_acquisition(mask, acq_id, masks_directory)
                                
                                # Update progress after each acquisition
                                acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
                                progress_dlg.update_progress(
                                    successful_segmentations,
                                    f"Completed batch {batch_start//batch_size + 1}",
                                    f"Segmented {acq_info.name} ({successful_segmentations}/{total_acquisitions} completed)"
                                )
                    
                    # Clear batch data to free memory
                    del batch_data
                    if _HAVE_TORCH and use_gpu:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing batch {batch_start//batch_size + 1}: {e}")
                    # Fall back to individual processing for this batch
                    for acq in batch_acquisitions:
                        try:
                            self._process_single_acquisition_fallback(
                                acq, model_obj, model, diameter, flow_threshold, 
                                cellprob_threshold, preprocessing_config, save_masks, masks_directory
                            )
                            successful_segmentations += 1
                            
                            # Update progress after each fallback acquisition
                            progress_dlg.update_progress(
                                successful_segmentations,
                                f"Fallback processing",
                                f"Segmented {acq.name} ({successful_segmentations}/{total_acquisitions} completed)"
                            )
                        except Exception as e2:
                            print(f"Error segmenting acquisition {acq.name}: {e2}")
                            continue
            
            progress_dlg.update_progress(total_acquisitions, "Batch segmentation complete", 
                                       f"Successfully segmented {successful_segmentations}/{total_acquisitions} acquisitions")
            
            # Show completion message
            QtWidgets.QMessageBox.information(
                self, "Batch Segmentation Complete",
                f"Successfully segmented {successful_segmentations} out of {total_acquisitions} acquisitions.\n"
                f"Segmentation masks are available for overlay display."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Batch Segmentation Failed", 
                f"Batch segmentation failed with error:\n{str(e)}"
            )
        finally:
            progress_dlg.close()
    
    def _estimate_optimal_batch_size(self, preprocessing_config: dict, gpu_id) -> int:
        """Estimate optimal batch size based on available memory."""
        # Start with a conservative estimate
        base_batch_size = 4
        
        if not preprocessing_config:
            return base_batch_size
        
        # Get channel counts
        nuclear_channels = len(preprocessing_config.get('nuclear_channels', []))
        cyto_channels = len(preprocessing_config.get('cyto_channels', []))
        total_channels = nuclear_channels + cyto_channels
        
        # Estimate image size (use first acquisition as reference)
        if self.acquisitions:
            try:
                first_acq = self.acquisitions[0]
                sample_img = self.loader.get_image(first_acq.id, first_acq.channels[0])
                img_size_mb = sample_img.nbytes / (1024 * 1024)  # Size in MB
            except:
                img_size_mb = 50  # Default estimate
        else:
            img_size_mb = 50
        
        # Estimate memory requirements per acquisition
        # Each acquisition needs: nuclear_img + cyto_img + masks + intermediate processing
        memory_per_acq_mb = img_size_mb * total_channels * 3  # 3x for processing overhead
        
        # Get available memory
        available_memory_mb = self._get_available_memory()
        
        # Calculate batch size based on available memory
        # Use 70% of available memory to leave room for system
        usable_memory_mb = available_memory_mb * 0.7
        estimated_batch_size = max(1, int(usable_memory_mb / memory_per_acq_mb))
        
        # Apply limits
        min_batch_size = 1
        max_batch_size = 16  # Reasonable upper limit
        
        batch_size = max(min_batch_size, min(max_batch_size, estimated_batch_size))
        
        print(f"Memory estimation: {available_memory_mb:.0f}MB available, {memory_per_acq_mb:.0f}MB per acquisition, batch size: {batch_size}")
        
        return batch_size
    
    def _get_available_memory(self) -> float:
        """Get available system memory in MB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # Fallback estimation
            return 4000  # Assume 4GB available
    
    def _load_batch_acquisitions(self, acquisitions, preprocessing_config: dict, progress_dlg) -> dict:
        """Load and preprocess a batch of acquisitions efficiently."""
        batch_images = []
        batch_channels = []
        acquisition_mapping = []  # Track which images belong to which acquisition
        
        for acq in acquisitions:
            try:
                # Temporarily set current acquisition for preprocessing
                original_acq_id = self.current_acq_id
                self.current_acq_id = acq.id
                
                # Preprocess channels for this acquisition
                nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                    preprocessing_config, progress_dlg
                )
                
                # Prepare input images based on model type
                if nuclear_img is not None:
                    if cyto_img is not None:
                        # Both nuclear and cytoplasm available
                        batch_images.extend([cyto_img, nuclear_img])
                        batch_channels.extend([0, 1])  # cyto, nuclear
                        acquisition_mapping.extend([acq.id, acq.id])  # Both images belong to same acquisition
                    else:
                        # Only nuclear available
                        batch_images.extend([nuclear_img, nuclear_img])
                        batch_channels.extend([0, 0])  # nuclear, nuclear
                        acquisition_mapping.extend([acq.id, acq.id])  # Both images belong to same acquisition
                else:
                    print(f"Warning: No valid images for acquisition {acq.name}")
                    continue
                
                # Restore original acquisition
                self.current_acq_id = original_acq_id
                
            except Exception as e:
                print(f"Error preprocessing acquisition {acq.name}: {e}")
                continue
        
        if not batch_images:
            return None
        
        return {
            'images': batch_images,
            'channels': batch_channels,
            'acquisition_mapping': acquisition_mapping,
            'acquisition_count': len(acquisitions)
        }
    
    def _process_single_acquisition_fallback(self, acq, model_obj, model: str, diameter: int,
                                           flow_threshold: float, cellprob_threshold: float,
                                           preprocessing_config: dict, save_masks: bool, masks_directory: str = None):
        """Fallback method to process a single acquisition individually."""
        # Temporarily set current acquisition
        original_acq_id = self.current_acq_id
        self.current_acq_id = acq.id
        
        try:
            # Preprocess channels
            nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                preprocessing_config, None
            )
            
            # Prepare input images
            if model == "nuclei":
                images = [nuclear_img]
                channels = [0, 0]
            else:  # cyto3
                if cyto_img is None:
                    cyto_img = nuclear_img
                images = [cyto_img, nuclear_img]
                channels = [0, 1]
            
            # Run segmentation
            masks, flows, styles, diams = model_obj.eval(
                images,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                channels=channels
            )
            
            # Store results
            self.segmentation_masks[acq.id] = masks[0]
            # Clear colors for this acquisition so they get regenerated
            if acq.id in self.segmentation_colors:
                del self.segmentation_colors[acq.id]
            
            # Save masks if requested
            if save_masks:
                self._save_segmentation_masks_for_acquisition(masks[0], acq.id, masks_directory)
                
        finally:
            # Restore original acquisition
            self.current_acq_id = original_acq_id
    
    def _save_segmentation_masks_for_acquisition(self, masks: np.ndarray, acq_id: str, masks_directory: str = None):
        """Save segmentation masks for a specific acquisition."""
        acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
        filename = f"{acq_info.name}_segmentation_masks.tif"
        
        # Use provided directory or fallback to .mcd directory
        if masks_directory and os.path.exists(masks_directory):
            filepath = os.path.join(masks_directory, filename)
        else:
            filepath = os.path.join(os.path.dirname(self.mcd_path), filename)
        
        try:
            tifffile.imwrite(filepath, masks.astype(np.uint16))
            print(f"Segmentation masks saved: {filepath}")
        except Exception as e:
            print(f"Error saving segmentation masks: {e}")
    
    def _save_segmentation_masks(self, masks_directory: str = None):
        """Save segmentation masks to file."""
        if not self.current_acq_id or self.current_acq_id not in self.segmentation_masks:
            return
        
        acq_info = next(ai for ai in self.acquisitions if ai.id == self.current_acq_id)
        safe_name = self._sanitize_filename(acq_info.name)
        if acq_info.well:
            safe_well = self._sanitize_filename(acq_info.well)
            filename = f"{safe_name}_{safe_well}_segmentation.tiff"
        else:
            filename = f"{safe_name}_segmentation.tiff"
        
        # Use provided directory or ask user to select
        if masks_directory and os.path.exists(masks_directory):
            output_dir = masks_directory
        else:
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Segmentation Masks", ""
            )
            if not output_dir:
                return
        
        output_path = os.path.join(output_dir, filename)
        
        # Save mask as TIFF
        if _HAVE_TIFFFILE:
            tifffile.imwrite(output_path, self.segmentation_masks[self.current_acq_id].astype(np.uint16))
        else:
            # Fallback to numpy save
            np.save(output_path.replace('.tiff', '.npy'), self.segmentation_masks[self.current_acq_id])
    
    def _update_display_with_segmentation(self):
        """Update the current display to show segmentation overlay."""
        if not self.segmentation_overlay or self.current_acq_id not in self.segmentation_masks:
            return
        
        # Refresh the current view
        self._view_selected()
    
    def _get_segmentation_overlay(self, img: np.ndarray) -> np.ndarray:
        """Create segmentation overlay for display."""
        if not self.segmentation_overlay or self.current_acq_id not in self.segmentation_masks:
            return img
        
        mask = self.segmentation_masks[self.current_acq_id]
        
        # Create colored overlay
        overlay = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        
        # Get or generate colors for this acquisition
        unique_labels = np.unique(mask)
        if self.current_acq_id not in self.segmentation_colors:
            # Generate and store colors for this acquisition
            self.segmentation_colors[self.current_acq_id] = np.random.rand(len(unique_labels), 3)
        
        colors = self.segmentation_colors[self.current_acq_id]
        
        for i, label in enumerate(unique_labels):
            if label == 0:  # Background
                continue
            cell_mask = (mask == label)
            overlay[cell_mask] = colors[i]
        
        # Blend with original image
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img
        
        # Normalize images to [0, 1]
        img_norm = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
        overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        
        # Blend (50% original, 50% overlay)
        blended = 0.7 * img_norm + 0.3 * overlay_norm
        
        return blended

    def _get_gpu_info(self):
        """Get GPU information for display."""
        if not _HAVE_TORCH:
            return None
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                return f"CUDA ({gpu_count} GPU{'s' if gpu_count > 1 else ''}): {', '.join(gpu_names)}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "Apple Metal Performance Shaders (MPS)"
            else:
                return "CPU only"
        except Exception:
            return "GPU detection failed"
    
    def _preprocess_channels_for_segmentation(self, preprocessing_config: dict, progress_dlg) -> tuple:
        """Preprocess and combine channels for segmentation."""
        if not preprocessing_config:
            raise ValueError("Preprocessing configuration is required for segmentation")
        
        config = preprocessing_config
        
        # Get nuclear channels
        nuclear_channels = config.get('nuclear_channels', [])
        if not nuclear_channels:
            raise ValueError("No nuclear channels specified in preprocessing configuration")
        
        # Get cytoplasm channels
        cyto_channels = config.get('cyto_channels', [])
        
        # Load and normalize nuclear channels
        progress_dlg.update_progress(25, "Preprocessing images", "Loading nuclear channels...")
        nuclear_imgs = []
        for channel in nuclear_channels:
            img = self.loader.get_image(self.current_acq_id, channel)
            # Apply normalization if configured
            img = self._apply_normalization(img, config, self.current_acq_id, channel)
            nuclear_imgs.append(img)
        
        # Combine nuclear channels
        nuclear_combo_method = config.get('nuclear_combo_method', 'single')
        nuclear_weights = config.get('nuclear_weights')
        nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights)
        
        # Load and normalize cytoplasm channels
        cyto_img = None
        if cyto_channels:
            progress_dlg.update_progress(35, "Preprocessing images", "Loading cytoplasm channels...")
            cyto_imgs = []
            for channel in cyto_channels:
                img = self.loader.get_image(self.current_acq_id, channel)
                # Apply normalization if configured
                img = self._apply_normalization(img, config, self.current_acq_id, channel)
                cyto_imgs.append(img)
            
            # Combine cytoplasm channels
            cyto_combo_method = config.get('cyto_combo_method', 'single')
            cyto_weights = config.get('cyto_weights')
            cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights)
        
        return nuclear_img, cyto_img
    
    def _apply_normalization(self, img: np.ndarray, config: dict, acq_id: str, channel: str) -> np.ndarray:
        """Apply normalization to an image based on configuration."""
        norm_method = config.get('normalization_method', 'None')
        
        if norm_method == 'None':
            return img
        
        # Check cache first
        cache_key = f"{acq_id}_{channel}_{norm_method}"
        if norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 5.0)
            cache_key += f"_{cofactor}"
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            cache_key += f"_{p_low}_{p_high}"
        
        # Apply normalization
        if norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 5.0)
            return arcsinh_normalize(img, cofactor)
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            return percentile_clip_normalize(img, p_low, p_high)
        
        return img
    
    def _on_segmentation_overlay_toggled(self):
        """Handle segmentation overlay checkbox toggle."""
        self.segmentation_overlay = self.segmentation_overlay_chk.isChecked()
        
        # Update display if we have segmentation masks
        if self.current_acq_id in self.segmentation_masks:
            self._view_selected()
            
            # Update checkbox text to show cell count
            if self.segmentation_overlay:
                cell_count = len(np.unique(self.segmentation_masks[self.current_acq_id])) - 1
                self.segmentation_overlay_chk.setText(f"Show segmentation overlay ({cell_count} cells)")
            else:
                self.segmentation_overlay_chk.setText("Show segmentation overlay")
    
    def _extract_features(self):
        """Open feature extraction dialog and perform feature extraction."""
        if not self.segmentation_masks:
            QtWidgets.QMessageBox.warning(
                self, 
                "No segmentation masks", 
                "No segmentation masks found. Please run segmentation first."
            )
            return
        
        # Open feature extraction dialog
        dlg = FeatureExtractionDialog(self, self.acquisitions, self.segmentation_masks)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # Get extraction parameters
        selected_acquisitions = dlg.get_selected_acquisitions()
        selected_features = dlg.get_selected_features()
        output_path = dlg.get_output_path()
        
        if not selected_acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions selected", "Please select at least one acquisition.")
            return
        
        if not any(selected_features.values()):
            QtWidgets.QMessageBox.warning(self, "No features selected", "Please select at least one feature to extract.")
            return
        
        # Perform feature extraction
        try:
            self._perform_feature_extraction(selected_acquisitions, selected_features, output_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Feature Extraction Failed", 
                f"Feature extraction failed with error:\n{str(e)}"
            )
    
    def _extract_features_worker(self, args):
        """Worker function for multiprocessing feature extraction."""
        acq_id, mask, selected_features, acq_info, loader, arcsinh_enabled, cofactor = args
        
        try:
            # Get pixel size from metadata
            pixel_size_um = self._get_pixel_size_um(acq_id, acq_info)
            
            # Get all available channels
            channels = acq_info.channels  # channels are already strings, not objects
            
            # Initialize feature dictionary
            features = {
                'acquisition_id': acq_id,
                'acquisition_name': acq_info.name,
                'well': acq_info.well if acq_info.well else ''
            }
            
            # Get unique cell IDs (excluding background = 0)
            unique_cells = np.unique(mask)
            unique_cells = unique_cells[unique_cells > 0]  # Remove background
            
            if len(unique_cells) == 0:
                return None
            
            # Extract morphology features
            if any(selected_features[key] for key in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 
                                                     'eccentricity', 'solidity', 'extent', 'circularity',
                                                     'major_axis_len_um', 'minor_axis_len_um', 'aspect_ratio',
                                                     'bbox_area_um2', 'touches_border', 'holes_count']):
                morph_features = self._extract_morphology_features(mask, unique_cells, pixel_size_um, selected_features)
                features.update(morph_features)
            
            # Extract intensity features for each channel
            if any(selected_features[key] for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']):
                for channel in channels:
                    try:
                        # Load channel image
                        channel_img = loader.get_image(acq_id, channel)
                        
                        # Apply arcsinh normalization if enabled
                        if arcsinh_enabled:
                            channel_img = arcsinh_normalize(channel_img, cofactor=cofactor)
                        
                        intensity_features = self._extract_intensity_features(
                            channel_img, mask, unique_cells, channel, selected_features
                        )
                        features.update(intensity_features)
                    except Exception as e:
                        print(f"Warning: Could not load channel {channel}: {e}")
                        continue
            
            # Create DataFrame
            features_df = pd.DataFrame(features)
            
            return features_df
            
        except Exception as e:
            print(f"Error extracting features for acquisition {acq_id}: {e}")
            return None

    def _perform_feature_extraction(self, selected_acquisitions, selected_features, output_path):
        """Perform the actual feature extraction using multiprocessing."""
        # Create progress dialog
        progress_dlg = ProgressDialog("Feature Extraction", self)
        progress_dlg.set_maximum(len(selected_acquisitions))
        progress_dlg.show()
        
        try:
            # Prepare arguments for multiprocessing
            mp_args = []
            for acq_id in selected_acquisitions:
                try:
                    current_acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
                    mask = self.segmentation_masks[acq_id]
                    cofactor = self.cofactor_spinbox.value() if hasattr(self, 'cofactor_spinbox') else 5.0
                    
                    mp_args.append((
                        acq_id, 
                        mask, 
                        selected_features, 
                        current_acq_info, 
                        self.loader, 
                        self.arcsinh_enabled, 
                        cofactor
                    ))
                except StopIteration:
                    continue
            
            if not mp_args:
                QtWidgets.QMessageBox.warning(self, "No valid acquisitions", "No valid acquisitions found for feature extraction.")
                return
            
            # Use multiprocessing with maximum CPU count
            max_workers = mp.cpu_count()
            progress_dlg.update_progress(0, "Starting feature extraction", f"Using {max_workers} CPU cores")
            
            all_features = []
            try:
                with mp.Pool(max_workers) as pool:
                    # Process acquisitions in parallel
                    results = pool.map(self._extract_features_worker, mp_args)
                    
                    # Collect results and update progress
                    for i, result in enumerate(results):
                        if progress_dlg.is_cancelled():
                            break
                        
                        if result is not None and not result.empty:
                            all_features.append(result)
                        
                        # Update progress
                        progress_dlg.update_progress(
                            i + 1, 
                            f"Processed acquisition {i+1}/{len(results)}", 
                            f"Extracted features from {len(all_features)} acquisitions"
                        )
            except Exception as mp_error:
                print(f"Multiprocessing failed, falling back to single-threaded processing: {mp_error}")
                progress_dlg.update_progress(0, "Multiprocessing failed, using single-threaded processing", "Processing acquisitions sequentially")
                
                # Fallback to single-threaded processing
                for i, args in enumerate(mp_args):
                    if progress_dlg.is_cancelled():
                        break
                    
                    result = self._extract_features_worker(args)
                    if result is not None and not result.empty:
                        all_features.append(result)
                    
                    # Update progress
                    progress_dlg.update_progress(
                        i + 1, 
                        f"Processed acquisition {i+1}/{len(mp_args)}", 
                        f"Extracted features from {len(all_features)} acquisitions"
                    )
            
            if not all_features:
                QtWidgets.QMessageBox.warning(self, "No features extracted", "No features could be extracted from the selected acquisitions.")
                return
            
            # Combine all features
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # Store in memory
            self.feature_dataframe = combined_features
            
            # Save to CSV
            if output_path:
                combined_features.to_csv(output_path, index=False)
                progress_dlg.update_progress(
                    len(selected_acquisitions), 
                    "Feature extraction complete", 
                    f"Features saved to: {output_path}\nTotal cells: {len(combined_features)}"
                )
            else:
                progress_dlg.update_progress(
                    len(selected_acquisitions), 
                    "Feature extraction complete", 
                    f"Features stored in memory\nTotal cells: {len(combined_features)}"
                )
            
            # Show completion message
            QtWidgets.QMessageBox.information(
                self, 
                "Feature Extraction Complete",
                f"Successfully extracted features from {len(selected_acquisitions)} acquisitions.\n"
                f"Total cells: {len(combined_features)}\n"
                f"Features saved to: {output_path if output_path else 'memory only'}"
            )
            
        except Exception as e:
            progress_dlg.close()
            raise e
        finally:
            progress_dlg.close()
    
    def _extract_features_for_acquisition(self, acq_id, mask, selected_features, acq_info):
        """Extract features for a single acquisition."""
        try:
            # Get pixel size from metadata
            pixel_size_um = self._get_pixel_size_um(acq_id, acq_info)
            
            # Get all available channels
            channels = acq_info.channels  # channels are already strings, not objects
            
            # Initialize feature dictionary
            features = {
                'acquisition_id': acq_id,
                'acquisition_name': acq_info.name,
                'well': acq_info.well if acq_info.well else ''
            }
            
            # Get unique cell IDs (excluding background = 0)
            unique_cells = np.unique(mask)
            unique_cells = unique_cells[unique_cells > 0]  # Remove background
            
            if len(unique_cells) == 0:
                return None
            
            # Extract morphology features
            if any(selected_features[key] for key in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 
                                                     'eccentricity', 'solidity', 'extent', 'circularity',
                                                     'major_axis_len_um', 'minor_axis_len_um', 'aspect_ratio',
                                                     'bbox_area_um2', 'touches_border', 'holes_count']):
                morph_features = self._extract_morphology_features(mask, unique_cells, pixel_size_um, selected_features)
                features.update(morph_features)
            
            # Extract intensity features for each channel
            if any(selected_features[key] for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']):
                for channel in channels:
                    try:
                        # Load channel image
                        channel_img = self._load_image_with_normalization(acq_id, channel)
                        intensity_features = self._extract_intensity_features(
                            channel_img, mask, unique_cells, channel, selected_features
                        )
                        features.update(intensity_features)
                    except Exception as e:
                        print(f"Warning: Could not load channel {channel}: {e}")
                        continue
            
            # Create DataFrame
            features_df = pd.DataFrame(features)
            
            return features_df
            
        except Exception as e:
            print(f"Error extracting features for acquisition {acq_id}: {e}")
            return None
    
    def _get_pixel_size_um(self, acq_id, acq_info=None):
        """Get pixel size in micrometers from acquisition metadata."""
        try:
            # Use provided acq_info or look it up
            if acq_info is None:
                acq_info = next(ai for ai in self.acquisitions if ai.id == acq_id)
            
            # Try to get pixel size from metadata
            if hasattr(acq_info, 'metadata') and acq_info.metadata:
                # Look for common pixel size keys
                for key in ['pixel_size_x', 'pixel_size', 'PhysicalSizeX']:
                    if key in acq_info.metadata:
                        return float(acq_info.metadata[key])
            
            # Default pixel size (1 μm) if not found
            return 1.0
        except Exception as e:
            return 1.0
    
    def _extract_morphology_features(self, mask, unique_cells, pixel_size_um, selected_features):
        """Extract morphology features from segmentation mask."""
        features = {}
        
        # Get region properties - mask is already labeled, no need for label() function
        props = regionprops(mask)
        
        # Initialize feature arrays including cell_id
        features['cell_id'] = []
        for key in ['area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity', 
                   'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um', 
                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count']:
            if selected_features[key]:
                features[key] = []
        
        for prop in props:
            cell_id = prop.label
            
            # Add cell_id to the features dictionary
            features['cell_id'].append(cell_id)
            
            if selected_features['area_um2']:
                features['area_um2'].append(prop.area * (pixel_size_um ** 2))
            
            if selected_features['perimeter_um']:
                features['perimeter_um'].append(prop.perimeter * pixel_size_um)
            
            if selected_features['equivalent_diameter_um']:
                features['equivalent_diameter_um'].append(prop.equivalent_diameter * pixel_size_um)
            
            if selected_features['eccentricity']:
                features['eccentricity'].append(prop.eccentricity)
            
            if selected_features['solidity']:
                features['solidity'].append(prop.solidity)
            
            if selected_features['extent']:
                features['extent'].append(prop.extent)
            
            if selected_features['circularity']:
                perimeter = prop.perimeter
                area = prop.area
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                features['circularity'].append(circularity)
            
            if selected_features['major_axis_len_um']:
                features['major_axis_len_um'].append(prop.major_axis_length * pixel_size_um)
            
            if selected_features['minor_axis_len_um']:
                features['minor_axis_len_um'].append(prop.minor_axis_length * pixel_size_um)
            
            if selected_features['aspect_ratio']:
                aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 0
                features['aspect_ratio'].append(aspect_ratio)
            
            if selected_features['bbox_area_um2']:
                bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
                features['bbox_area_um2'].append(bbox_area * (pixel_size_um ** 2))
            
            if selected_features['touches_border']:
                # Check if cell touches image border
                touches = (prop.bbox[0] == 0 or prop.bbox[1] == 0 or 
                          prop.bbox[2] == mask.shape[0] or prop.bbox[3] == mask.shape[1])
                features['touches_border'].append(touches)
            
            if selected_features['holes_count']:
                # Count holes in the cell (simplified - count of background pixels in convex hull)
                # This is a simplified implementation
                features['holes_count'].append(0)  # Placeholder - would need more complex analysis
        
        return features
    
    def _extract_intensity_features(self, channel_img, mask, unique_cells, channel_name, selected_features):
        """Extract intensity features for a specific channel."""
        features = {}
        
        # Initialize feature arrays
        for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']:
            if selected_features[key]:
                features[f"{key}_{channel_name}"] = []
        
        for cell_id in unique_cells:
            # Get mask for this cell
            cell_mask = (mask == cell_id)
            cell_pixels = channel_img[cell_mask]
            
            if len(cell_pixels) == 0:
                # Fill with NaN if no pixels
                for key in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated', 'frac_pos']:
                    if selected_features[key]:
                        features[f"{key}_{channel_name}"].append(np.nan)
                continue
            
            if selected_features['mean']:
                features[f"mean_{channel_name}"].append(np.mean(cell_pixels))
            
            if selected_features['median']:
                features[f"median_{channel_name}"].append(np.median(cell_pixels))
            
            if selected_features['std']:
                features[f"std_{channel_name}"].append(np.std(cell_pixels))
            
            if selected_features['mad']:
                features[f"mad_{channel_name}"].append(stats.median_abs_deviation(cell_pixels))
            
            if selected_features['p10']:
                features[f"p10_{channel_name}"].append(np.percentile(cell_pixels, 10))
            
            if selected_features['p90']:
                features[f"p90_{channel_name}"].append(np.percentile(cell_pixels, 90))
            
            if selected_features['integrated']:
                mean_intensity = np.mean(cell_pixels)
                area = np.sum(cell_mask)
                features[f"integrated_{channel_name}"].append(mean_intensity * area)
            
            if selected_features['frac_pos']:
                # Use 95th percentile of ROI as threshold
                threshold = np.percentile(channel_img, 95)
                frac_pos = np.sum(cell_pixels > threshold) / len(cell_pixels)
                features[f"frac_pos_{channel_name}"].append(frac_pos)
        
        return features
    
    def _load_segmentation_masks(self):
        """Load previously saved segmentation masks from a directory."""
        if not self.current_acq_id:
            QtWidgets.QMessageBox.warning(self, "No acquisition selected", "Please select an acquisition first.")
            return
        
        # Ask user to select directory containing masks
        masks_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            "Select Directory Containing Segmentation Masks",
            "",  # Start from current directory
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not masks_dir:
            return
        
        # Look for mask files for the current acquisition
        acq_info = next(ai for ai in self.acquisitions if ai.id == self.current_acq_id)
        safe_name = self._sanitize_filename(acq_info.name)
        
        # Try different possible filenames
        possible_filenames = []
        if acq_info.well:
            safe_well = self._sanitize_filename(acq_info.well)
            possible_filenames.append(f"{safe_name}_{safe_well}_segmentation.tiff")
            possible_filenames.append(f"{safe_name}_{safe_well}_segmentation_masks.tif")
        possible_filenames.append(f"{safe_name}_segmentation.tiff")
        possible_filenames.append(f"{safe_name}_segmentation_masks.tif")
        
        # Find the first existing mask file
        mask_file = None
        for filename in possible_filenames:
            filepath = os.path.join(masks_dir, filename)
            if os.path.exists(filepath):
                mask_file = filepath
                break
        
        if not mask_file:
            QtWidgets.QMessageBox.warning(
                self, 
                "No mask file found", 
                f"No segmentation mask file found for acquisition '{acq_info.name}' in the selected directory.\n\n"
                f"Looking for files matching:\n" + "\n".join(f"• {f}" for f in possible_filenames)
            )
            return
        
        try:
            # Load the mask file
            if _HAVE_TIFFFILE:
                mask = tifffile.imread(mask_file)
            else:
                # Fallback to PIL if tifffile not available
                from PIL import Image
                mask = np.array(Image.open(mask_file))
            
            # Store the loaded mask
            self.segmentation_masks[self.current_acq_id] = mask
            # Clear colors for this acquisition so they get regenerated
            if self.current_acq_id in self.segmentation_colors:
                del self.segmentation_colors[self.current_acq_id]
            self.segmentation_overlay = True
            self.segmentation_overlay_chk.setChecked(True)
            
            # Update display
            self._view_selected()
            
            # Show success message
            cell_count = len(np.unique(mask)) - 1  # Subtract 1 for background
            QtWidgets.QMessageBox.information(
                self, 
                "Masks loaded successfully", 
                f"Loaded segmentation masks from:\n{mask_file}\n\n"
                f"Found {cell_count} cells. Overlay is now enabled."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Error loading masks", 
                f"Failed to load segmentation masks:\n{str(e)}"
            )

    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.loader:
            self.loader.close()
        event.accept()

    def _open_clustering_dialog(self):
        """Open the cell clustering analysis dialog."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(
                self, 
                "No Feature Data", 
                "No feature data available. Please extract features first using the 'Extract Features' button."
            )
            return
        
        # Open clustering dialog
        dlg = CellClusteringDialog(self.feature_dataframe, self)
        dlg.exec_()


# --------------------------
# Entrypoint
# --------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    # Use 'fork' on Unix systems (Linux, macOS) and 'spawn' on Windows
    import platform
    if platform.system() == "Windows":
        mp.set_start_method('spawn', force=True)
    else:
        mp.set_start_method('fork', force=True)
    main()
