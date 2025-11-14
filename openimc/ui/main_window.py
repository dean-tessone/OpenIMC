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

from typing import Dict, List, Optional, Tuple, Union
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy import stats
from skimage.measure import regionprops, regionprops_table
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer

from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.feature_worker import extract_features_for_acquisition, load_and_extract_features
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.spillover_correction import comp_image_counts, load_spillover
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CustomNavigationToolbar(NavigationToolbar):
    """Custom navigation toolbar with improved save functionality."""
    
    def __init__(self, canvas, parent, main_window=None):
        super().__init__(canvas, parent)
        self.main_window = main_window
    
    def save_figure(self, *args):
        """Override save_figure method to use custom save options dialog."""
        if self.main_window and hasattr(self.main_window, '_get_suggested_save_filename'):
            # Get suggested filename from main window
            suggested_filename = self.main_window._get_suggested_save_filename()
            # Use the suggested filename but show full options dialog
            save_figure_with_options(
                self.canvas.figure,
                suggested_filename,
                self.main_window
            )
            return
            
        # Fallback to default behavior only if main window method is not available
        # But still use our enhanced dialog
        save_figure_with_options(
            self.canvas.figure,
            "figure.png",
            self.parent()
        )
    
    def _save(self):
        """Override _save method to use custom save options dialog."""
        if self.main_window and hasattr(self.main_window, '_get_suggested_save_filename'):
            # Get suggested filename from main window
            suggested_filename = self.main_window._get_suggested_save_filename()
            # Use the suggested filename but show full options dialog
            save_figure_with_options(
                self.canvas.figure,
                suggested_filename,
                self.main_window
            )
            return
            
        # Fallback to default behavior only if main window method is not available
        # But still use our enhanced dialog
        save_figure_with_options(
            self.canvas.figure,
            "figure.png",
            self.parent()
        )
from openimc.ui.mpl_canvas import MplCanvas
from openimc.ui.utils import (
    PreprocessingCache,
    robust_percentile_scale,
    arcsinh_normalize,
    percentile_clip_normalize,
    channelwise_minmax_normalize,
    stack_to_rgb,
    combine_channels,
)
from openimc.ui.dialogs.progress_dialog import ProgressDialog
from openimc.ui.dialogs.gpu_selection_dialog import GPUSelectionDialog
from openimc.ui.dialogs.preprocessing_dialog import PreprocessingDialog
from openimc.ui.dialogs.segmentation_dialog import SegmentationDialog
from openimc.ui.dialogs.ilastik_segmentation_dialog import IlastikSegmentationDialog
from openimc.ui.dialogs.export import ExportDialog
from openimc.ui.dialogs.feature_extraction import FeatureExtractionDialog

# Optional runtime flags for GPU/TIFF
_HAVE_TORCH = False
try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

_HAVE_TIFFFILE = False
try:
    import tifffile  # type: ignore  # noqa: F401
    _HAVE_TIFFFILE = True
except Exception:
    _HAVE_TIFFFILE = False
from openimc.ui.dialogs.clustering import CellClusteringDialog, ClusterExplorerDialog
from openimc.ui.dialogs.spatial_analysis import SpatialAnalysisDialog
# Import dialog classes directly to avoid lazy loading issues
from openimc.ui.dialogs.simple_spatial_analysis import SimpleSpatialAnalysisDialog
try:
    from openimc.ui.dialogs.advanced_spatial_analysis import AdvancedSpatialAnalysisDialog
except (ImportError, RuntimeError):
    AdvancedSpatialAnalysisDialog = None
from openimc.ui.dialogs.comparison_dialog import DynamicComparisonDialog
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.qc_analysis_dialog import QCAnalysisDialog
from openimc.ui.dialogs.spillover_matrix_dialog import GenerateSpilloverMatrixDialog
from openimc.ui.dialogs.batch_correction_dialog import BatchCorrectionDialog
from openimc.ui.dialogs.ometiff_format_dialog import OMETIFFFormatDialog
from openimc.ui.dialogs.deconvolution_dialog import DeconvolutionDialog
from openimc.ui.dialogs.pixel_correlation_dialog import PixelCorrelationDialog
from openimc.processing.deconvolution_worker import deconvolve_acquisition
from openimc.utils.logger import get_logger




# Optional runtime flags for extra deps
_HAVE_CELLPOSE = False
try:
    from cellpose import models as _cp_models  # type: ignore  # noqa: F401
    import skimage  # type: ignore  # noqa: F401
    _HAVE_CELLPOSE = True
except Exception:
    _HAVE_CELLPOSE = False
else:
    # Import models under the expected name when available
    from cellpose import models  # type: ignore

# Optional CellSAM
_HAVE_CELLSAM = False
try:
    from cellSAM import get_model, cellsam_pipeline  # type: ignore
    _HAVE_CELLSAM = True
except ImportError:
    _HAVE_CELLSAM = False

# Optional image processing deps (scikit-image, scipy)
_HAVE_SCIKIT_IMAGE = False
try:
    from skimage import morphology
    from skimage.filters import median, gaussian
    from skimage.morphology import disk, footprint_rectangle
    from skimage.restoration import denoise_nl_means, estimate_sigma
    try:
        from skimage.restoration import rolling_ball as _sk_rolling_ball  # type: ignore
        _HAVE_ROLLING_BALL = True
    except Exception:
        _HAVE_ROLLING_BALL = False
    import scipy.ndimage as ndi  # type: ignore
    _HAVE_SCIKIT_IMAGE = True
except Exception:
    _HAVE_SCIKIT_IMAGE = False
    _HAVE_ROLLING_BALL = False

# --------------------------
# Module-level worker function for multiprocessing
# --------------------------
def _load_and_preprocess_acquisition_worker(task_data):
    """
    Worker function to load and preprocess a single acquisition for segmentation.
    This function is isolated at module level to be picklable for multiprocessing.
    """
    (original_acq_id, unique_acq_id, acq_name, file_path, loader_type, preprocessing_config, 
     denoise_source, custom_denoise_settings, source_file, channel_format) = task_data
    
    try:
        # Import inside function to ensure isolation
        import numpy as np
        from openimc.data.mcd_loader import MCDLoader
        from openimc.data.ometiff_loader import OMETIFFLoader
        from openimc.ui.utils import combine_channels, arcsinh_normalize, percentile_clip_normalize, channelwise_minmax_normalize
        
        # Recreate loader (can't pickle loader objects)
        loader = None
        if loader_type == "mcd":
            loader = MCDLoader()
            loader.open(file_path)
        elif loader_type == "ometiff":
            loader = OMETIFFLoader(channel_format=channel_format or 'CHW')
            loader.open(file_path)
        
        if not loader:
            return None
        
        config = preprocessing_config
        
        # Get nuclear channels
        nuclear_channels = config.get('nuclear_channels', [])
        if not nuclear_channels:
            loader.close()
            return None
        
        # Get cytoplasm channels
        cyto_channels = config.get('cyto_channels', [])
        
        # Helper function to apply normalization
        def apply_normalization(img, config, channel):
            norm_method = config.get('normalization_method', 'None')
            if norm_method == 'None':
                return img
            elif norm_method == 'channelwise_minmax':
                return channelwise_minmax_normalize(img)
            elif norm_method == 'arcsinh':
                cofactor = config.get('arcsinh_cofactor', 10.0)
                return arcsinh_normalize(img, cofactor)
            elif norm_method == 'percentile_clip':
                p_low, p_high = config.get('percentile_params', (1.0, 99.0))
                return percentile_clip_normalize(img, p_low, p_high)
            return img
        
        # Helper function to apply custom denoising
        def apply_custom_denoise(img, channel, custom_denoise_settings):
            # Check if scikit-image is available
            try:
                from skimage import morphology
                from skimage.filters import median, gaussian
                from skimage.morphology import disk, footprint_rectangle
                from skimage.restoration import denoise_nl_means, estimate_sigma
                try:
                    from skimage.restoration import rolling_ball as _sk_rolling_ball_worker
                    _HAVE_ROLLING_BALL_WORKER = True
                except Exception:
                    _HAVE_ROLLING_BALL_WORKER = False
                import scipy.ndimage as ndi_worker
                _HAVE_SCIKIT_IMAGE_WORKER = True
            except Exception:
                _HAVE_SCIKIT_IMAGE_WORKER = False
                _HAVE_ROLLING_BALL_WORKER = False
            
            if not _HAVE_SCIKIT_IMAGE_WORKER:
                return img
            cfg = custom_denoise_settings.get(channel) if custom_denoise_settings else None
            if not cfg:
                return img
            
            out = img.astype(np.float32, copy=False)
            
            # Hot pixel removal
            hot = cfg.get("hot")
            if hot:
                method = hot.get("method")
                if method == "median3":
                    try:
                        out = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
                    except Exception:
                        out = ndi_worker.median_filter(out, size=3)
                elif method == "n_sd_local_median":
                    n_sd = float(hot.get("n_sd", 5.0))
                    try:
                        local_median = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
                    except Exception:
                        local_median = ndi_worker.median_filter(out, size=3)
                    diff = out - local_median
                    local_var = ndi_worker.uniform_filter(diff * diff, size=3)
                    local_std = np.sqrt(np.maximum(local_var, 1e-8))
                    mask_hot = diff > (n_sd * local_std)
                    out = np.where(mask_hot, local_median, out)
            
            # Speckle smoothing
            speckle = cfg.get("speckle")
            if speckle:
                method = speckle.get("method")
                if method == "gaussian":
                    sigma = float(speckle.get("sigma", 0.8))
                    out = gaussian(out, sigma=sigma, preserve_range=True)
                elif method == "nl_means":
                    mn, mx = float(np.min(out)), float(np.max(out))
                    scale = mx - mn
                    scaled = (out - mn) / scale if scale > 0 else out
                    sigma_est = np.mean(estimate_sigma(scaled, channel_axis=None))
                    out = denoise_nl_means(
                        scaled,
                        h=1.15 * sigma_est,
                        fast_mode=True,
                        patch_size=5,
                        patch_distance=6,
                        channel_axis=None,
                    )
                    out = out * scale + mn
            
            # Background subtraction
            bg = cfg.get("background")
            if bg:
                method = bg.get("method")
                radius = int(bg.get("radius", 15))
                if method == "white_tophat":
                    se = disk(radius)
                    try:
                        out = morphology.white_tophat(out, selem=se)
                    except TypeError:
                        out = morphology.white_tophat(out, footprint=se)
                elif method == "black_tophat":
                    se = disk(radius)
                    try:
                        out = morphology.black_tophat(out, selem=se)
                    except TypeError:
                        out = morphology.black_tophat(out, footprint=se)
                elif method == "rolling_ball":
                    if _HAVE_ROLLING_BALL_WORKER:
                        background = _sk_rolling_ball_worker(out, radius=radius)
                        out = out - background
                        out = np.clip(out, 0, None)
                    else:
                        se = disk(radius)
                        try:
                            opened = morphology.opening(out, selem=se)
                        except TypeError:
                            opened = morphology.opening(out, footprint=se)
                        out = out - opened
                        out = np.clip(out, 0, None)
            
            # Rescale to preserve original max intensity
            try:
                orig_max = float(np.max(img))
                new_max = float(np.max(out))
                if new_max > 0 and orig_max > 0:
                    out = out * (orig_max / new_max)
            except Exception:
                pass
            
            # Clip to dtype range if integer
            if np.issubdtype(img.dtype, np.integer):
                info = np.iinfo(img.dtype)
                out = np.clip(out, info.min, info.max)
            else:
                out = np.clip(out, 0, None)
            return out.astype(img.dtype, copy=False)
        
        # Load and normalize nuclear channels
        nuclear_imgs = []
        for channel in nuclear_channels:
            # Use original_acq_id for loader calls
            img = loader.get_image(original_acq_id, channel)
            if img is None:
                continue
            
            # Apply denoising
            if denoise_source == "custom" and custom_denoise_settings:
                try:
                    img = apply_custom_denoise(img, channel, custom_denoise_settings)
                except Exception:
                    pass
            
            # Apply normalization
            img = apply_normalization(img, config, channel)
            nuclear_imgs.append(img)
        
        if not nuclear_imgs:
            loader.close()
            return None
        
        # Combine nuclear channels
        nuclear_combo_method = config.get('nuclear_combo_method', 'single')
        nuclear_weights = config.get('nuclear_weights')
        nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights)
        
        # Load and normalize cytoplasm channels
        cyto_img = None
        if cyto_channels:
            cyto_imgs = []
            for channel in cyto_channels:
                # Use original_acq_id for loader calls
                img = loader.get_image(original_acq_id, channel)
                if img is None:
                    continue
                
                # Apply denoising
                if denoise_source == "custom" and custom_denoise_settings:
                    try:
                        img = apply_custom_denoise(img, channel, custom_denoise_settings)
                    except Exception:
                        pass
                
                # Apply normalization
                img = apply_normalization(img, config, channel)
                cyto_imgs.append(img)
            
            if cyto_imgs:
                # Combine cytoplasm channels
                cyto_combo_method = config.get('cyto_combo_method', 'single')
                cyto_weights = config.get('cyto_weights')
                cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights)
        
        # Close loader
        loader.close()
        
        # Return result with acquisition info
        return {
            'acq_id': unique_acq_id,  # Return unique ID for lookup in main process
            'acq_name': acq_name,
            'nuclear_img': nuclear_img,
            'cyto_img': cyto_img,
            'source_file': source_file
        }
        
    except Exception as e:
        print(f"Error processing acquisition {acq_name} ({unique_acq_id}): {e}")
        return None


# --------------------------
# Main Window
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMC File Viewer")
        
        # Set window size to full screen with minimum size constraint
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        self.resize(screen.width(), screen.height())
        
        # Set minimum size for smaller screens
        self.setMinimumSize(1000, 700)

        # State
        self.loader: Optional[Union[MCDLoader, OMETIFFLoader]] = None
        self.current_path: Optional[str] = None
        self.ometiff_channel_format: str = 'CHW'  # Store channel format for OME-TIFF files
        # Multi-file support: track multiple MCD files and their loaders
        self.mcd_loaders: Dict[str, MCDLoader] = {}  # Maps file path to MCDLoader
        self.acq_to_file: Dict[str, str] = {}  # Maps acquisition ID to source file path
        self.unique_acq_to_original: Dict[str, str] = {}  # Maps unique acquisition ID to original ID
        self.acquisitions: List[AcquisitionInfo] = []
        self.current_acq_id: Optional[str] = None
        # Image cache and prefetching
        self.image_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._prefetch_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Store last selected channels for auto-selection
        self.last_selected_channels: List[str] = []
        # Store channel list from previous acquisition to detect mismatches
        self.previous_acq_channels: List[str] = []
        
        # Simple zoom preservation for specific operations
        self.saved_zoom_limits = None
        self.preserve_zoom = False  # Flag to indicate when to preserve zoom
        self.had_no_channels = False  # Flag to track when we had no channels selected
        

        # Widgets
        self.canvas = MplCanvas(width=6, height=6, dpi=100)
        self.open_btn = QtWidgets.QPushButton("Open File/Folder")
        self.acq_combo = QtWidgets.QComboBox()
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.deselect_all_btn = QtWidgets.QPushButton("Deselect all")
        # Removed 'View selected' button; auto-refresh is enabled
        self.view_btn = QtWidgets.QPushButton("View selected")
        self.view_btn.setVisible(False)
        self.comparison_btn = QtWidgets.QPushButton("Comparison mode")
        self.segment_btn = QtWidgets.QPushButton("Cell Segmentation")
        self.extract_features_btn = QtWidgets.QPushButton("Extract Features")
        self.batch_correction_btn = QtWidgets.QPushButton("Batch Correction")
        self.clustering_btn = QtWidgets.QPushButton("Cell Clustering")
        self.spatial_btn = QtWidgets.QPushButton("Spatial Analysis")
        self.reset_zoom_btn = QtWidgets.QPushButton("Reset Zoom")
        
        # Visualization options
        self.grayscale_chk = QtWidgets.QCheckBox("Grayscale mode")
        self.grid_view_chk = QtWidgets.QCheckBox("Grid view for multiple channels")
        # Auto-refresh on toggle
        self.grayscale_chk.toggled.connect(self._on_grayscale_toggled)
        self.grid_view_chk.toggled.connect(self._on_grid_view_toggled)
        self.grid_view_chk.setChecked(True)
        self.segmentation_overlay_chk = QtWidgets.QCheckBox("Show segmentation overlay")
        self.segmentation_overlay_chk.toggled.connect(self._on_segmentation_overlay_toggled)
        
        # Overlay mode selection (outline vs mask vs cluster)
        overlay_mode_layout = QtWidgets.QHBoxLayout()
        overlay_mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.segmentation_overlay_mode_combo = QtWidgets.QComboBox()
        self.segmentation_overlay_mode_combo.addItems(["Mask", "Outline", "Cluster"])
        self.segmentation_overlay_mode_combo.setCurrentText("Mask")
        self.segmentation_overlay_mode_combo.currentTextChanged.connect(self._on_segmentation_overlay_mode_changed)
        
        # Initially disable Cluster mode (will be enabled if cluster data is available)
        # This will be updated when features are loaded or acquisition changes
        overlay_mode_layout.addWidget(self.segmentation_overlay_mode_combo)
        overlay_mode_layout.addStretch()
        self.segmentation_overlay_mode_widget = QtWidgets.QWidget()
        self.segmentation_overlay_mode_widget.setLayout(overlay_mode_layout)
        self.segmentation_overlay_mode_widget.setVisible(False)  # Hidden by default until overlay is enabled
        
        # Scale bar controls
        self.scale_bar_chk = QtWidgets.QCheckBox("Show scale bar")
        self.scale_bar_chk.toggled.connect(self._on_scale_bar_toggled)
        scale_bar_layout = QtWidgets.QHBoxLayout()
        scale_bar_layout.addWidget(QtWidgets.QLabel("Length (μm):"))
        self.scale_bar_length_spin = QtWidgets.QDoubleSpinBox()
        self.scale_bar_length_spin.setRange(0.1, 10000.0)
        self.scale_bar_length_spin.setDecimals(1)
        self.scale_bar_length_spin.setValue(10.0)
        self.scale_bar_length_spin.setSingleStep(10.0)
        self.scale_bar_length_spin.valueChanged.connect(self._on_scale_bar_changed)
        scale_bar_layout.addWidget(self.scale_bar_length_spin)
        scale_bar_layout.addStretch()
        self.scale_bar_widget = QtWidgets.QWidget()
        self.scale_bar_widget.setLayout(scale_bar_layout)
        self.scale_bar_widget.setVisible(False)  # Hidden until scale bar is enabled
        
        # Show all channels button (only visible in grid view)
        self.show_all_channels_btn = QtWidgets.QPushButton("Show all channels")
        self.show_all_channels_btn.clicked.connect(self._show_all_channels)
        self.show_all_channels_btn.setVisible(False)  # Hidden by default


        # Denoising enable + options panel
        self.denoise_enable_chk = QtWidgets.QCheckBox("Enable denoising")
        self.denoise_enable_chk.toggled.connect(self._on_denoise_toggled)
        self.denoise_frame = QtWidgets.QFrame()
        self.denoise_frame.setFrameStyle(QtWidgets.QFrame.Box)
        denoise_layout = QtWidgets.QVBoxLayout(self.denoise_frame)
        denoise_layout.addWidget(QtWidgets.QLabel("Denoising (apply per selected channel):"))

        # Channel dropdown (apply per-channel like custom scaling)
        denoise_channel_row = QtWidgets.QHBoxLayout()
        denoise_channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.denoise_channel_combo = QtWidgets.QComboBox()
        self.denoise_channel_combo.currentTextChanged.connect(self._on_denoise_channel_changed)
        denoise_channel_row.addWidget(self.denoise_channel_combo, 1)
        denoise_layout.addLayout(denoise_channel_row)

        # Hot pixel removal
        self.hot_pixel_chk = QtWidgets.QCheckBox("Hot pixel removal")
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
        denoise_layout.addLayout(hot_row)

        # Speckle / background smoothing
        self.speckle_chk = QtWidgets.QCheckBox("Speckle smoothing")
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
        denoise_layout.addLayout(speckle_row)

        # Background subtraction
        self.bg_subtract_chk = QtWidgets.QCheckBox("Background subtraction")
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
        denoise_layout.addLayout(bg_row)

        # Preprocessing order controls
        order_frame = QtWidgets.QFrame()
        order_frame.setFrameStyle(QtWidgets.QFrame.Plain)
        order_layout = QtWidgets.QHBoxLayout(order_frame)
        order_layout.addWidget(QtWidgets.QLabel("Order:"))
        self.step_names = ["Hot pixel", "Speckle", "Background"]
        self.order_combo_1 = QtWidgets.QComboBox(); self.order_combo_1.addItems(self.step_names)
        self.order_combo_2 = QtWidgets.QComboBox(); self.order_combo_2.addItems(self.step_names)
        self.order_combo_3 = QtWidgets.QComboBox(); self.order_combo_3.addItems(self.step_names)
        order_layout.addWidget(self.order_combo_1)
        order_layout.addWidget(QtWidgets.QLabel("→"))
        order_layout.addWidget(self.order_combo_2)
        order_layout.addWidget(QtWidgets.QLabel("→"))
        order_layout.addWidget(self.order_combo_3)
        order_layout.addStretch()
        denoise_layout.addWidget(order_frame)

        # Default order: Hot → Speckle → Background
        self.order_combo_1.setCurrentText("Hot pixel")
        self.order_combo_2.setCurrentText("Speckle")
        self.order_combo_3.setCurrentText("Background")

        # Apply to all channels button
        self.apply_all_channels_btn = QtWidgets.QPushButton("Apply to All Channels")
        self.apply_all_channels_btn.clicked.connect(self._apply_denoise_to_all_channels)
        self.apply_all_channels_btn.setMinimumWidth(200)  # Wide enough for text
        denoise_layout.addWidget(self.apply_all_channels_btn)

        # Disable panel if scikit-image is missing
        if not _HAVE_SCIKIT_IMAGE:
            self.denoise_frame.setEnabled(False)
            denoise_layout.addWidget(QtWidgets.QLabel("scikit-image not available; install to enable denoising."))
        # Hidden by default until enabled
        self.denoise_frame.setVisible(False)
        # Ensure proper initial visibility of hot-pixel controls
        # (Show N only for threshold method)
        # Create after widgets exist
        QtWidgets.QApplication.processEvents()
        try:
            self._sync_hot_controls_visibility()
        except Exception:
            pass
        
        # Initialize RGB combination method early (before UI setup that uses it)
        self.rgb_combination_method = "raw_addition"  # Options: "raw_addition", "minmax_scaled_mean", "raw_mean", "max"
        
        # Custom scaling controls
        self.custom_scaling_chk = QtWidgets.QCheckBox("Custom scaling")
        self.custom_scaling_chk.toggled.connect(self._on_custom_scaling_toggled)
        
        self.scaling_frame = QtWidgets.QFrame()
        self.scaling_frame.setFrameStyle(QtWidgets.QFrame.Box)
        scaling_layout = QtWidgets.QVBoxLayout(self.scaling_frame)
        scaling_layout.addWidget(QtWidgets.QLabel("Custom Intensity Range:"))
        
        # Note about arcsinh transform
        arcsinh_note = QtWidgets.QLabel(
            "Note: Arcsinh transform should be applied on extracted\n"
            "intensity features, not on raw images during export."
        )
        arcsinh_note.setStyleSheet("QLabel { color: #666; font-size: 9pt; font-style: italic; }")
        arcsinh_note.setWordWrap(True)
        scaling_layout.addWidget(arcsinh_note)
        
        # Channel selection for per-channel scaling
        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.scaling_channel_combo = QtWidgets.QComboBox()
        self.scaling_channel_combo.currentTextChanged.connect(self._on_scaling_channel_changed)
        self.scaling_channel_combo.setMinimumWidth(200)  # Wide enough for channel names
        channel_row.addWidget(self.scaling_channel_combo)
        channel_row.addStretch()
        scaling_layout.addLayout(channel_row)
        
        # Number input controls
        input_layout = QtWidgets.QVBoxLayout()
        
        # Min input
        min_row = QtWidgets.QHBoxLayout()
        min_row.addWidget(QtWidgets.QLabel("Min:"))
        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setRange(0.0, 10000.0)
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setValue(0.0)
        self.min_spinbox.setSingleStep(0.1)
        self.min_spinbox.valueChanged.connect(self._on_scaling_changed)
        min_row.addWidget(self.min_spinbox)
        min_row.addStretch()
        input_layout.addLayout(min_row)
        
        # Max input
        max_row = QtWidgets.QHBoxLayout()
        max_row.addWidget(QtWidgets.QLabel("Max:"))
        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setRange(0.0, 10000.0)
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setValue(1000.0)
        self.max_spinbox.setSingleStep(0.1)
        self.max_spinbox.valueChanged.connect(self._on_scaling_changed)
        max_row.addWidget(self.max_spinbox)
        max_row.addStretch()
        input_layout.addLayout(max_row)
        
        scaling_layout.addLayout(input_layout)
        
        # RGB combination method selection (only visible in RGB mode)
        self.rgb_combination_frame = QtWidgets.QFrame()
        self.rgb_combination_frame.setFrameStyle(QtWidgets.QFrame.Box)
        rgb_combination_layout = QtWidgets.QVBoxLayout(self.rgb_combination_frame)
        rgb_combination_layout.addWidget(QtWidgets.QLabel("RGB Channel Combination:"))
        
        self.rgb_combination_combo = QtWidgets.QComboBox()
        self.rgb_combination_combo.addItems([
            "Raw Addition (sum per pixel)",
            "Min/Max Scaled Mean (normalize each channel then mean)",
            "Raw Mean (mean per pixel)",
            "Max (maximum per pixel)"
        ])
        # Map stored method to combo text
        method_to_text = {
            "raw_addition": "Raw Addition (sum per pixel)",
            "minmax_scaled_mean": "Min/Max Scaled Mean (normalize each channel then mean)",
            "raw_mean": "Raw Mean (mean per pixel)",
            "max": "Max (maximum per pixel)"
        }
        initial_text = method_to_text.get(self.rgb_combination_method, "Raw Addition (sum per pixel)")
        self.rgb_combination_combo.setCurrentText(initial_text)
        self.rgb_combination_combo.currentTextChanged.connect(self._on_rgb_combination_changed)
        rgb_combination_layout.addWidget(self.rgb_combination_combo)
        self.rgb_combination_frame.setVisible(False)  # Hidden by default, shown in RGB mode
        scaling_layout.addWidget(self.rgb_combination_frame)
        
        # Arcsinh normalization controls
        arcsinh_layout = QtWidgets.QHBoxLayout()
        arcsinh_layout.addWidget(QtWidgets.QLabel("Arcsinh Co-factor:"))
        self.cofactor_spinbox = QtWidgets.QDoubleSpinBox()
        self.cofactor_spinbox.setRange(0.1, 100.0)
        self.cofactor_spinbox.setDecimals(1)
        self.cofactor_spinbox.setValue(10.0)
        self.cofactor_spinbox.setSingleStep(0.5)
        arcsinh_layout.addWidget(self.cofactor_spinbox)
        arcsinh_layout.addStretch()
        scaling_layout.addLayout(arcsinh_layout)
        
        # Control buttons
        button_row = QtWidgets.QVBoxLayout()  # Changed to vertical for better button sizing
        self.arcsinh_btn = QtWidgets.QPushButton("Arcsinh Normalization")
        self.arcsinh_btn.clicked.connect(self._arcsinh_normalization)
        self.arcsinh_btn.setMinimumWidth(200)  # Wide enough for text
        button_row.addWidget(self.arcsinh_btn)
        
        self.default_range_btn = QtWidgets.QPushButton("Default Range")
        self.default_range_btn.clicked.connect(self._default_range)
        self.default_range_btn.setMinimumWidth(200)  # Wide enough for text
        button_row.addWidget(self.default_range_btn)
        
        # Remove Apply button; auto-apply scaling on change
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setVisible(False)
        
        scaling_layout.addLayout(button_row)
        self.scaling_frame.setVisible(False)
        
        # Spillover correction controls
        self.spillover_correction_chk = QtWidgets.QCheckBox("Spillover correction")
        self.spillover_correction_chk.toggled.connect(self._on_spillover_correction_toggled)
        
        self.spillover_frame = QtWidgets.QFrame()
        self.spillover_frame.setFrameStyle(QtWidgets.QFrame.Box)
        spillover_layout = QtWidgets.QVBoxLayout(self.spillover_frame)
        spillover_layout.addWidget(QtWidgets.QLabel("Spillover Matrix:"))
        
        # Matrix file selection
        matrix_file_row = QtWidgets.QHBoxLayout()
        matrix_file_row.addWidget(QtWidgets.QLabel("Matrix:"))
        self.spillover_matrix_file_edit = QtWidgets.QLineEdit()
        self.spillover_matrix_file_edit.setPlaceholderText("No matrix loaded...")
        self.spillover_matrix_file_edit.setReadOnly(True)
        self.spillover_matrix_file_btn = QtWidgets.QPushButton("Browse...")
        self.spillover_matrix_file_btn.clicked.connect(self._select_spillover_matrix)
        matrix_file_row.addWidget(self.spillover_matrix_file_edit)
        matrix_file_row.addWidget(self.spillover_matrix_file_btn)
        spillover_layout.addLayout(matrix_file_row)
        
        # Method selection
        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("Method:"))
        self.spillover_method_combo = QtWidgets.QComboBox()
        self.spillover_method_combo.addItems(["pgd", "nnls"])
        self.spillover_method_combo.setToolTip("pgd: Fast projected gradient descent\nnnls: Exact non-negative least squares (slower)")
        self.spillover_method_combo.currentTextChanged.connect(self._on_spillover_method_changed)
        method_row.addWidget(self.spillover_method_combo)
        method_row.addStretch()
        spillover_layout.addLayout(method_row)
        
        self.spillover_frame.setVisible(False)
        
        # Store spillover correction state
        self.spillover_matrix: Optional[pd.DataFrame] = None
        self.spillover_matrix_path: Optional[str] = None
        self.spillover_correction_enabled = False
        self.spillover_method = "pgd"
        self._spillover_warning_shown = False  # Track if we've shown the warning
        
        # Store per-channel scaling values
        self.channel_scaling = {}  # {channel_name: {'min': value, 'max': value}}
        
        # Store RGB color scaling values (for RGB mode)
        self.rgb_color_scaling = {}  # {'Red': {'min': value, 'max': value}, 'Green': {...}, 'Blue': {...}}
        
        # Arcsinh normalization state
        self.arcsinh_enabled = False
        # Per-channel normalization config: {channel: {"method": str, "cofactor": float}}
        self.channel_normalization: Dict[str, Dict[str, float or str]] = {}
        
        # Per-channel scaling method state
        self.current_scaling_method = "default"  # kept for backward compatibility
        self.channel_scaling_method: Dict[str, str] = {}  # {channel: "default"|"arcsinh"}
        
        # Segmentation state
        self.segmentation_masks = {}  # {acq_id: mask_array}
        self.segmentation_colors = {}  # {acq_id: colors_array}
        self.cluster_colors = {}  # {acq_id: {cluster_id: color_array}}
        self.cluster_color_map = {}  # {acq_id: {cluster_id: display_name}}
        self.segmentation_overlay = False
        self.segmentation_overlay_mode = "Mask"  # "Outline", "Mask", or "Cluster"
        self.preprocessing_cache = PreprocessingCache()
        # Per-channel denoise config {channel: {"hot": {...}, "speckle": {...}, "background": {...}}}
        self.channel_denoise: Dict[str, Dict[str, dict]] = {}
        
        # Feature extraction state
        self.feature_dataframe = None  # Store extracted features in memory
        self.batch_corrected_dataframe = None  # Store batch-corrected features in memory
        
        # Dialog instances for state retention
        self.clustering_dialog = None  # Store clustering dialog instance
        self.spatial_dialog = None  # Store spatial analysis dialog instance
        self.segmentation_dialog = None  # Store segmentation dialog instance
        
        # Color assignment for RGB composite
        self.color_assignment_frame = QtWidgets.QFrame()
        self.color_assignment_frame.setFrameStyle(QtWidgets.QFrame.Box)
        color_layout = QtWidgets.QVBoxLayout(self.color_assignment_frame)
        color_layout.addWidget(QtWidgets.QLabel("Color Assignment (for RGB composite):"))
        
        # Red channel selection with search
        red_block = QtWidgets.QVBoxLayout()
        red_block.addWidget(QtWidgets.QLabel("Red:"))
        self.red_search = QtWidgets.QLineEdit()
        self.red_search.setPlaceholderText("Search red channels...")
        self.red_search.textChanged.connect(self._filter_red_channels)
        red_block.addWidget(self.red_search)
        self.red_list = QtWidgets.QListWidget()
        self.red_list.setMaximumHeight(80)
        self.red_list.itemChanged.connect(lambda _i: self._on_rgb_list_changed())
        red_block.addWidget(self.red_list)
        color_layout.addLayout(red_block)
        
        # Green channel selection with search
        green_block = QtWidgets.QVBoxLayout()
        green_block.addWidget(QtWidgets.QLabel("Green:"))
        self.green_search = QtWidgets.QLineEdit()
        self.green_search.setPlaceholderText("Search green channels...")
        self.green_search.textChanged.connect(self._filter_green_channels)
        green_block.addWidget(self.green_search)
        self.green_list = QtWidgets.QListWidget()
        self.green_list.setMaximumHeight(80)
        self.green_list.itemChanged.connect(lambda _i: self._on_rgb_list_changed())
        green_block.addWidget(self.green_list)
        color_layout.addLayout(green_block)
        
        # Blue channel selection with search
        blue_block = QtWidgets.QVBoxLayout()
        blue_block.addWidget(QtWidgets.QLabel("Blue:"))
        self.blue_search = QtWidgets.QLineEdit()
        self.blue_search.setPlaceholderText("Search blue channels...")
        self.blue_search.textChanged.connect(self._filter_blue_channels)
        blue_block.addWidget(self.blue_search)
        self.blue_list = QtWidgets.QListWidget()
        self.blue_list.setMaximumHeight(80)
        self.blue_list.itemChanged.connect(lambda _i: self._on_rgb_list_changed())
        blue_block.addWidget(self.blue_list)
        color_layout.addLayout(blue_block)

        # Metadata display (more compact for smaller screens)
        self.metadata_text = QtWidgets.QTextEdit()
        self.metadata_text.setMaximumHeight(120)
        self.metadata_text.setReadOnly(True)

        # Left panel layout with scrolling for smaller screens
        controls = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(controls)
        v.setContentsMargins(5, 5, 5, 5)  # Reduce margins for more space
        v.addWidget(self.open_btn)

        v.addWidget(QtWidgets.QLabel("Acquisition:"))
        v.addWidget(self.acq_combo)

        v.addWidget(QtWidgets.QLabel("Channels:"))
        
        # Channel search box
        self.channel_search = QtWidgets.QLineEdit()
        self.channel_search.setPlaceholderText("Search channels...")
        self.channel_search.textChanged.connect(self._filter_channels)
        # Denoise controls auto-refresh
        self.hot_pixel_chk.toggled.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.hot_pixel_method_combo.currentIndexChanged.connect(lambda _: self._on_hot_method_changed())
        self.hot_pixel_n_spin.valueChanged.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.speckle_chk.toggled.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.speckle_method_combo.currentIndexChanged.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.gaussian_sigma_spin.valueChanged.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.bg_subtract_chk.toggled.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.bg_method_combo.currentIndexChanged.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.bg_radius_spin.valueChanged.connect(lambda _: self._apply_denoise_settings_and_refresh())
        self.denoise_channel_combo.currentTextChanged.connect(lambda _: self._load_denoise_settings())
        # Order change handlers
        self.order_combo_1.currentIndexChanged.connect(lambda _: self._on_order_changed())
        self.order_combo_2.currentIndexChanged.connect(lambda _: self._on_order_changed())
        self.order_combo_3.currentIndexChanged.connect(lambda _: self._on_order_changed())
        v.addWidget(self.channel_search)
        
        v.addWidget(self.channel_list, 1)
        
        # Channel control buttons
        channel_btn_row = QtWidgets.QHBoxLayout()
        channel_btn_row.addWidget(self.deselect_all_btn)
        channel_btn_row.addStretch()
        v.addLayout(channel_btn_row)
        
        # Visualization options
        v.addWidget(self.grayscale_chk)
        v.addWidget(self.grid_view_chk)
        v.addWidget(self.show_all_channels_btn)
        v.addWidget(self.segmentation_overlay_chk)
        v.addWidget(self.segmentation_overlay_mode_widget)
        v.addWidget(self.scale_bar_chk)
        v.addWidget(self.scale_bar_widget)
        v.addWidget(self.denoise_enable_chk)
        v.addWidget(self.denoise_frame)
        v.addWidget(self.custom_scaling_chk)
        v.addWidget(self.scaling_frame)
        v.addWidget(self.spillover_correction_chk)
        v.addWidget(self.spillover_frame)
        v.addWidget(self.color_assignment_frame)

        v.addSpacing(4)
        v.addWidget(self.view_btn)
        v.addWidget(self.reset_zoom_btn)
        v.addWidget(self.comparison_btn)
        v.addWidget(self.segment_btn)
        v.addWidget(self.extract_features_btn)
        v.addWidget(self.batch_correction_btn)
        v.addWidget(self.clustering_btn)
        v.addWidget(self.spatial_btn)
        v.addSpacing(4)
        
        v.addWidget(QtWidgets.QLabel("Metadata:"))
        v.addWidget(self.metadata_text)
        v.addStretch(1)

        # Splitter with scrollable left panel for smaller screens
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        
        # Create scrollable left panel with resizable width
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidget(controls)
        left_scroll.setWidgetResizable(True)
        # Set minimum width to 420px, but allow expansion up to 30% of window width
        left_scroll.setMinimumWidth(420)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Store references for resize handling
        self.left_scroll = left_scroll
        self.splitter = splitter
        self.sidebar_min_width = 420
        
        splitter.addWidget(left_scroll)
        # Right pane with toolbar + canvas
        rightw = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(rightw)
        self.nav_toolbar = CustomNavigationToolbar(self.canvas, self, self)
        right_layout.addWidget(self.nav_toolbar)
        right_layout.addWidget(self.canvas, 1)
        splitter.addWidget(rightw)
        splitter.setStretchFactor(1, 1)
        
        # Connect splitter moved signal to enforce maximum width constraint
        splitter.splitterMoved.connect(self._on_splitter_moved)
        
        self.setCentralWidget(splitter)
        
        # Set initial maximum width after window is set up
        QTimer.singleShot(0, self._update_sidebar_max_width)

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        act_about = file_menu.addAction("About…")
        act_about.triggered.connect(self._show_about_dialog)
        act_display_settings = file_menu.addAction("Display Settings…")
        act_display_settings.triggered.connect(self._show_display_settings)
        file_menu.addSeparator()
        act_open = file_menu.addAction("Open File/Folder…")
        act_open.triggered.connect(self._open_dialog)
        act_load_features = file_menu.addAction("Load Feature File…")
        act_load_features.triggered.connect(self._load_feature_file)
        file_menu.addSeparator()
        
        # Export submenu
        export_submenu = file_menu.addMenu("Export")
        act_export_tiff = export_submenu.addAction("Export to OME-TIFF…")
        act_export_tiff.triggered.connect(self._export_ome_tiff)
        
        # Masks submenu
        masks_submenu = file_menu.addMenu("Segmentation Masks")
        act_load_masks = masks_submenu.addAction("Load Masks…")
        act_load_masks.triggered.connect(self._load_segmentation_masks)
        act_save_masks = masks_submenu.addAction("Save Masks…")
        act_save_masks.triggered.connect(self._save_segmentation_masks)
        
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit")
        act_quit.triggered.connect(self.close)

        # Analysis menu
        analysis_menu = self.menuBar().addMenu("&Analysis")
        act_spillover_matrix = analysis_menu.addAction("Generate Spillover Matrix…")
        act_spillover_matrix.triggered.connect(self._open_spillover_matrix_dialog)
        act_batch_correction = analysis_menu.addAction("Batch Correction…")
        act_batch_correction.triggered.connect(self._open_batch_correction_dialog)
        act_clustering = analysis_menu.addAction("Cell Clustering…")
        act_clustering.triggered.connect(self._open_clustering_dialog)
        # Spatial Analysis submenu
        spatial_submenu = analysis_menu.addMenu("Spatial Analysis")
        act_simple_spatial = spatial_submenu.addAction("Simple Spatial Analysis…")
        act_simple_spatial.triggered.connect(self._open_simple_spatial_dialog)
        act_advanced_spatial = spatial_submenu.addAction("Advanced Spatial Analysis (Squidpy)…")
        act_advanced_spatial.triggered.connect(self._open_advanced_spatial_dialog)
        # Default action (for button) - opens Advanced if squidpy available, otherwise Simple
        # Note: The button uses _open_spatial_dialog which defaults to Advanced
        act_qc = analysis_menu.addAction("QC Analysis…")
        act_qc.triggered.connect(self._open_qc_dialog)
        act_pixel_correlation = analysis_menu.addAction("Pixel-Level Correlation…")
        act_pixel_correlation.triggered.connect(self._open_pixel_correlation_dialog)
        act_deconvolution = analysis_menu.addAction("High Resolution Deconvolution…")
        act_deconvolution.triggered.connect(self._open_deconvolution_dialog)

        # Signals
        self.open_btn.clicked.connect(self._open_dialog)
        self.acq_combo.currentIndexChanged.connect(self._on_acq_changed)
        self.deselect_all_btn.clicked.connect(self._deselect_all_channels)
        self.channel_list.itemChanged.connect(self._on_channel_selection_changed)
        self.channel_search.textChanged.connect(self._filter_channels)
        # Auto-refresh: no manual 'View selected' action
        try:
            self.view_btn.clicked.disconnect()
        except Exception:
            pass
        self.reset_zoom_btn.clicked.connect(self._reset_zoom)
        self.comparison_btn.clicked.connect(self._comparison)
        self.segment_btn.clicked.connect(self._run_segmentation)
        self.extract_features_btn.clicked.connect(self._extract_features)
        self.batch_correction_btn.clicked.connect(self._open_batch_correction_dialog)
        self.clustering_btn.clicked.connect(self._open_clustering_dialog)
        self.spatial_btn.clicked.connect(self._open_spatial_dialog)

        # Loader - will be initialized when data is loaded
        self.loader = None

        # Ensure RGB controls are hidden when grid view is enabled on startup
        try:
            self._update_rgb_controls_visibility()
            # Also ensure show all channels button visibility is correct
            if hasattr(self, 'show_all_channels_btn'):
                self.show_all_channels_btn.setVisible(self.grid_view_chk.isChecked())
        except Exception:
            pass
        
        # Initialize cluster mode availability (will be disabled initially)
        try:
            self._update_cluster_mode_availability()
        except Exception:
            pass

    # ---------- About dialog ----------
    def _show_about_dialog(self):
        """Show the About dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About OpenIMC",
            "OpenIMC © 2025 University of Southern California — Licensed under GPL v3 or later\n\n"
            "NO WARRANTY. See LICENSE for details."
        )
    
    # ---------- Display settings ----------
    def _show_display_settings(self):
        """Show the Display Settings dialog."""
        from openimc.ui.dialogs.display_settings_dialog import DisplaySettingsDialog
        dialog = DisplaySettingsDialog(self)
        dialog.exec_()

    # ---------- File open ----------
    def _open_dialog(self):
        # Allow user to choose between file or directory
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Select Import Type")
        msg_box.setText("How would you like to import data?")
        msg_box.addButton("Open .mcd File(s)", QtWidgets.QMessageBox.YesRole)
        msg_box.addButton("Open OME-TIFF Folder", QtWidgets.QMessageBox.NoRole)
        msg_box.addButton(QtWidgets.QMessageBox.Cancel)
        choice = msg_box.exec_()
        
        if choice == 0:  # YesRole - Open .mcd file(s)
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Open IMC .mcd file(s)", "", "IMC MCD files (*.mcd);;All files (*.*)"
            )
            if paths:
                if len(paths) == 1:
                    # Single file - use existing path
                    self._load_data(paths[0])
                else:
                    # Multiple files - load them all
                    self._load_multiple_mcd_files(paths)
        elif choice == 1:  # NoRole - Open OME-TIFF folder
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Folder with OME-TIFF files", ""
            )
            if path:
                self._load_data(path)
        # else: Cancel - do nothing

    def _load_multiple_mcd_files(self, paths: List[str]):
        """Load multiple .mcd files and combine their acquisitions."""
        # Close existing loaders
        self._close_all_loaders()
        
        # Clear image cache to avoid collisions with old cache entries
        with self._cache_lock:
            self.image_cache.clear()
        
        # Track all acquisitions and their source files
        all_acquisitions = []
        file_channel_sets = {}  # Maps file path to set of all channels in that file
        
        # Load each MCD file
        for path in paths:
            if not os.path.isfile(path):
                QtWidgets.QMessageBox.warning(self, "Invalid file", f"Skipping invalid path: {path}")
                continue
            
            if not path.lower().endswith('.mcd'):
                QtWidgets.QMessageBox.warning(self, "Invalid file", f"Skipping non-MCD file: {path}")
                continue
            
            try:
                loader = MCDLoader()
                loader.open(path)
                self.mcd_loaders[path] = loader
                
                # Get acquisitions for this file
                file_acqs = loader.list_acquisitions(source_file=path)
                
                # Create unique acquisition IDs by incorporating file identifier
                # Use a hash of the file path to create a short unique identifier
                import hashlib
                file_hash = hashlib.md5(path.encode()).hexdigest()[:8]
                file_id = f"file_{file_hash}"
                
                # Track channels for mismatch detection (union of all channels in this file)
                file_channels = set()
                for acq in file_acqs:
                    # Create unique acquisition ID by combining original ID with file identifier
                    unique_acq_id = f"{acq.id}__{file_id}"
                    
                    # Create new AcquisitionInfo with unique ID
                    from openimc.data.mcd_loader import AcquisitionInfo
                    unique_acq = AcquisitionInfo(
                        id=unique_acq_id,
                        name=acq.name,
                        well=acq.well,
                        size=acq.size,
                        channels=acq.channels,
                        channel_metals=acq.channel_metals,
                        channel_labels=acq.channel_labels,
                        metadata=acq.metadata,
                        source_file=acq.source_file
                    )
                    all_acquisitions.append(unique_acq)
                    
                    file_channels.update(acq.channels)
                    self.acq_to_file[unique_acq_id] = path
                    self.unique_acq_to_original[unique_acq_id] = acq.id  # Store mapping from unique to original ID
                file_channel_sets[path] = file_channels
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Load failed", f"Failed to load {os.path.basename(path)}:\n{e}")
                continue
        
        if not all_acquisitions:
            QtWidgets.QMessageBox.critical(self, "No acquisitions", "No acquisitions could be loaded from the selected files.")
            return
        
        # Check for channel mismatches (compare channels per file)
        channel_sets_list = list(file_channel_sets.values())
        self._check_channel_mismatches(channel_sets_list, list(file_channel_sets.keys()))
        
        # Update state
        self.acquisitions = all_acquisitions
        self.current_path = paths[0] if paths else None
        
        # Update window title
        if len(paths) == 1:
            stem = os.path.splitext(os.path.basename(paths[0]))[0]
            self.setWindowTitle(f"IMC File Viewer - {stem} (MCD)")
        else:
            self.setWindowTitle(f"IMC File Viewer - {len(paths)} MCD files")
        
        # Clear canvas completely before loading new files to ensure proper redraw
        self.canvas.fig.clear()
        self.canvas.draw()
        
        # Update acquisition combo box with file names
        self.acq_combo.clear()
        for ai in self.acquisitions:
            file_name = os.path.basename(ai.source_file) if ai.source_file else "Unknown"
            # Use well name if available, otherwise use acquisition name
            label = ai.well if ai.well else ai.name
            label += f" [{file_name}]"
            self.acq_combo.addItem(label, ai.id)
        
        if self.acquisitions:
            self._populate_channels(self.acquisitions[0].id)
            # For initial file load, if no channels were pre-selected, select DNA1 if it exists, otherwise first channel
            if not self._selected_channels() and self.channel_list.count() > 0:
                # Look for channel containing DNA1 first
                dna1_item = None
                for i in range(self.channel_list.count()):
                    item = self.channel_list.item(i)
                    if "DNA1" in item.text():
                        dna1_item = item
                        break
                
                # Select DNA1 if found, otherwise select first channel
                if dna1_item:
                    dna1_item.setCheckState(Qt.Checked)
                else:
                    item = self.channel_list.item(0)
                    item.setCheckState(Qt.Checked)
    
    def _check_channel_mismatches(self, channel_sets: List[set], paths: List[str]):
        """Check if MCD files have different channels and warn the user."""
        if len(paths) < 2:
            return
        
        # Group channels by file (each file may have multiple acquisitions with same channels)
        # We'll compare the union of all channels from each file
        file_channel_sets = {}
        for i, path in enumerate(paths):
            if path not in file_channel_sets:
                file_channel_sets[path] = set()
            # Add all channels from this file's acquisitions
            if i < len(channel_sets):
                file_channel_sets[path].update(channel_sets[i])
        
        # Check if all file channel sets are identical
        file_paths_list = list(file_channel_sets.keys())
        if len(file_paths_list) < 2:
            return
        
        first_file_set = file_channel_sets[file_paths_list[0]]
        all_same = all(file_channel_sets[p] == first_file_set for p in file_paths_list[1:])
        
        if not all_same:
            # Find differences
            all_channels = set()
            for ch_set in file_channel_sets.values():
                all_channels.update(ch_set)
            
            missing_info = []
            for path in file_paths_list:
                file_channels = file_channel_sets[path]
                missing = all_channels - file_channels
                if missing:
                    missing_info.append(f"  {os.path.basename(path)}: missing {', '.join(sorted(missing))}")
            
            if missing_info:
                warning_msg = (
                    "Warning: The selected MCD files have different channels.\n\n"
                    "This may affect downstream analysis. Missing channels:\n" +
                    "\n".join(missing_info) +
                    "\n\nYou may need to apply batch effect correction later."
                )
                QtWidgets.QMessageBox.warning(self, "Channel Mismatch", warning_msg)
    
    def _close_all_loaders(self):
        """Close all MCD loaders and clear state."""
        # Close single loader if exists
        if self.loader:
            try:
                self.loader.close()
            except Exception:
                pass
            self.loader = None
        
        # Close all MCD loaders
        for loader in self.mcd_loaders.values():
            try:
                loader.close()
            except Exception:
                pass
        self.mcd_loaders.clear()
        self.acq_to_file.clear()
        self.unique_acq_to_original.clear()
        
        # Clear image cache when switching files
        with self._cache_lock:
            self.image_cache.clear()
        
        # Reset channel tracking when loading a new file
        self.previous_acq_channels = []
        self.last_selected_channels = []
        
        # Reset segmentation dialog when switching files
        if self.segmentation_dialog is not None:
            try:
                # Clear channel preferences for the old file
                if hasattr(self.segmentation_dialog, '_per_file_channel_prefs'):
                    self.segmentation_dialog._per_file_channel_prefs.clear()
                # Close and reset the dialog
                self.segmentation_dialog.close()
                self.segmentation_dialog = None
            except Exception:
                # If dialog is already deleted or has issues, just set to None
                self.segmentation_dialog = None
    
    def _get_loader_for_acquisition(self, acq_id: str) -> Optional[Union[MCDLoader, OMETIFFLoader]]:
        """Get the appropriate loader for a given acquisition ID."""
        # If we have multiple MCD files, use the loader for the acquisition's source file
        if acq_id in self.acq_to_file:
            file_path = self.acq_to_file[acq_id]
            loader = self.mcd_loaders.get(file_path)
            return loader
        
        # Otherwise, use the single loader (for OME-TIFF or single MCD file)
        return self.loader
    
    def _get_original_acq_id(self, acq_id: str) -> str:
        """Get the original acquisition ID from a unique ID (for multi-file support)."""
        # If this is a unique ID (contains __file_), extract the original ID
        if acq_id in self.unique_acq_to_original:
            original_id = self.unique_acq_to_original[acq_id]
            return original_id
        # Otherwise, it's already the original ID
        return acq_id
    
    @property
    def current_loader(self) -> Optional[Union[MCDLoader, OMETIFFLoader]]:
        """Get the loader for the current acquisition."""
        if self.current_acq_id:
            return self._get_loader_for_acquisition(self.current_acq_id)
        return self.loader

    def _load_data(self, path: str):
        """Load data from either a .mcd file or a directory of OME-TIFF files."""
        # Close existing loaders
        self._close_all_loaders()
        # Delete LLM cache when switching files
        pass
        
        # Determine if path is a file or directory
        is_file = os.path.isfile(path)
        is_dir = os.path.isdir(path)
        
        if not is_file and not is_dir:
            QtWidgets.QMessageBox.critical(self, "Invalid path", f"Path does not exist: {path}")
            return
        
        # Choose appropriate loader
        if is_file:
            # Load .mcd file - treat as single file for backward compatibility
            if not path.lower().endswith('.mcd'):
                QtWidgets.QMessageBox.critical(self, "Invalid file", "Please select a .mcd file.")
                return
            try:
                self.loader = MCDLoader()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Dependency error", str(e))
                return
            loader_type = "MCD"
        else:
            # Load OME-TIFF directory
            # Ask user about channel format
            format_dialog = OMETIFFFormatDialog(self)
            if format_dialog.exec_() != QtWidgets.QDialog.Accepted:
                # User cancelled, don't load
                return
            channel_format = format_dialog.get_format()
            self.ometiff_channel_format = channel_format  # Store for use in worker functions
            
            try:
                self.loader = OMETIFFLoader(channel_format=channel_format)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Dependency error", str(e))
                return
            loader_type = "OME-TIFF"
        
        # Open the data source
        try:
            self.loader.open(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Failed to open {path}\n\n{e}")
            self.loader = None
            return
        
        self.current_path = path
        try:
            if is_file:
                stem = os.path.splitext(os.path.basename(path))[0]
                self.setWindowTitle(f"IMC File Viewer - {stem} ({loader_type})")
            else:
                dirname = os.path.basename(path) or path
                self.setWindowTitle(f"IMC File Viewer - {dirname} ({loader_type})")
        except Exception:
            # Fallback to default title if something goes wrong
            self.setWindowTitle(f"IMC File Viewer ({loader_type})")
        
        # Clear canvas completely before loading new file to ensure proper redraw
        self.canvas.fig.clear()
        self.canvas.draw()
        
        # Get acquisitions with source file info
        self.acquisitions = self.loader.list_acquisitions(source_file=path if is_file else None)
        self.acq_combo.clear()
        for ai in self.acquisitions:
            # Use well name if available, otherwise use acquisition name
            label = ai.well if ai.well else ai.name
            self.acq_combo.addItem(label, ai.id)
        if self.acquisitions:
            self._populate_channels(self.acquisitions[0].id)
            # For initial file load, if no channels were pre-selected, select DNA1 if it exists, otherwise first channel
            if not self._selected_channels() and self.channel_list.count() > 0:
                # Look for channel containing DNA1 first
                dna1_item = None
                for i in range(self.channel_list.count()):
                    item = self.channel_list.item(i)
                    if "DNA1" in item.text():
                        dna1_item = item
                        break
                
                # Select DNA1 if found, otherwise select first channel
                if dna1_item:
                    dna1_item.setCheckState(Qt.Checked)
                else:
                    item = self.channel_list.item(0)
                    item.setCheckState(Qt.Checked)

    # ---------- Acquisition / channels ----------
    def _on_acq_changed(self, idx: int):
        acq_id = self.acq_combo.itemData(idx)
        if acq_id:
            # Clear canvas completely before switching to ensure proper redraw
            self.canvas.fig.clear()
            self.canvas.draw()
            
            # Store current scaling state before changing acquisition
            preserve_scaling = self.custom_scaling_chk.isChecked()
            current_scaling_method = self.current_scaling_method
            
            self._populate_channels(acq_id)
            # Start background prefetch of all channels for the new acquisition
            self._start_prefetch_all_channels(acq_id)
            
            # Update scaling channel combo when acquisition changes
            if preserve_scaling:
                self._update_scaling_channel_combo()
                # Restore scaling method state
                self.current_scaling_method = current_scaling_method
                self._update_minmax_controls_state()

    def _populate_channels(self, acq_id: str):
        self.current_acq_id = acq_id
        self.channel_list.clear()
        
        # Update segmentation overlay text for new acquisition
        self._update_segmentation_overlay_text()
        
        # Update cluster mode availability for new acquisition
        self._update_cluster_mode_availability()
        try:
            loader = self._get_loader_for_acquisition(acq_id)
            if loader is None:
                QtWidgets.QMessageBox.critical(self, "Loader error", f"No loader found for acquisition {acq_id}")
                return
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_id)
            chans = loader.get_channels(original_acq_id)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Channels error", str(e))
            return
        
        # Check if channels match the previous acquisition
        # If channels differ, reset the selection
        channels_match = False
        if self.previous_acq_channels:
            # Compare channel lists (order-independent comparison)
            if set(chans) == set(self.previous_acq_channels):
                channels_match = True
            else:
                # Channels don't match, reset selection
                self.last_selected_channels = []
        
        # Update the previous acquisition's channel list for next comparison
        self.previous_acq_channels = chans.copy()
        
        # Pre-select channels that were selected in the previous acquisition
        # (only if channels match, otherwise last_selected_channels is empty)
        selected_channels = []
        for ch in chans:
            item = QtWidgets.QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            
            # Check if this channel was selected in the previous acquisition
            # (only preserved if channels matched)
            if ch in self.last_selected_channels:
                item.setCheckState(Qt.Checked)
                selected_channels.append(ch)
            else:
                item.setCheckState(Qt.Unchecked)
            
            self.channel_list.addItem(item)
        
        # Kick off prefetch if not already running for this acq
        self._start_prefetch_all_channels(acq_id)

        # Update RGB color assignment lists with only currently selected channels
        self._populate_color_assignments(selected_channels)
        # Update denoise channel list
        self._populate_denoise_channel_list(selected_channels)
        # Sync hot controls visibility
        self._sync_hot_controls_visibility()
        
        # Auto-load image if channels were pre-selected
        if selected_channels:
            self._auto_load_image(selected_channels)
        
        # Update metadata display
        ai = self._get_acquisition_info(acq_id)
        if ai is None:
            QtWidgets.QMessageBox.warning(self, "Acquisition not found", f"Could not find acquisition {acq_id}")
            return
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
        
        # Check if we're going from no channels to having channels
        has_last_channels = hasattr(self, 'last_selected_channels') and self.last_selected_channels
        # Only trigger if we had no channels selected AND now we have channels
        going_from_no_channels = selected_channels and self.had_no_channels
        
        
        # Set preserve zoom FIRST before any other operations (unless going from no channels)
        if not going_from_no_channels:
            self.preserve_zoom = True
        
        self._populate_color_assignments(selected_channels)
        self._populate_denoise_channel_list(selected_channels)
        
        # Clear any color assignments that are no longer in the selected channels
        self._clear_invalid_color_assignments(selected_channels)
        
        # Update scaling channel combo to reflect current selection
        if self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
        # Auto-refresh view when channels change (preserve zoom)
        self._view_selected()

        # Update RGB control visibility and selections on change
        self._update_rgb_controls_visibility()
        
        # Update the had_no_channels flag
        if not selected_channels:
            self.had_no_channels = True
        else:
            self.had_no_channels = False
        
        # If we went from no channels to having channels, click the reset zoom button
        if going_from_no_channels:
            self.reset_zoom_btn.click()

    def _populate_color_assignments(self, channels: List[str]):
        """Populate the color assignment dropdowns with selected channels only."""
        # Clear existing items
        # Preserve current checks
        prev_red = {self.red_list.item(i).text(): self.red_list.item(i).checkState() == Qt.Checked for i in range(self.red_list.count())}
        prev_green = {self.green_list.item(i).text(): self.green_list.item(i).checkState() == Qt.Checked for i in range(self.green_list.count())}
        prev_blue = {self.blue_list.item(i).text(): self.blue_list.item(i).checkState() == Qt.Checked for i in range(self.blue_list.count())}

        self.red_list.clear()
        self.green_list.clear()
        self.blue_list.clear()
        
        # Only add selected channels; if none, keep lists empty
        for ch in channels:
            for lst, prev in [(self.red_list, prev_red), (self.green_list, prev_green), (self.blue_list, prev_blue)]:
                it = QtWidgets.QListWidgetItem(ch)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                # Restore previous check state if present
                checked = prev.get(ch, False)
                it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                lst.addItem(it)

        # Do not auto-select by default; leave empty if no channels are selected

    def _populate_denoise_channel_list(self, channels: List[str]):
        """Populate the denoise channel combo with currently selected channels."""
        self.denoise_channel_combo.blockSignals(True)
        self.denoise_channel_combo.clear()
        for ch in channels:
            self.denoise_channel_combo.addItem(ch)
        self.denoise_channel_combo.blockSignals(False)
        if channels:
            self.denoise_channel_combo.setCurrentIndex(0)
            self._load_denoise_settings()

    def _clear_invalid_color_assignments(self, selected_channels: List[str]):
        """Clear color assignments that are no longer in the selected channels."""
        # For list-based multi-select, deselect any items not in current selection list
        def _prune_list(lst: QtWidgets.QListWidget):
            for i in range(lst.count()):
                item = lst.item(i)
                if item.text() not in selected_channels:
                    item.setCheckState(Qt.Unchecked)
    def _filter_red_channels(self):
        """Filter red channel list based on search text."""
        text = self.red_search.text().lower()
        for i in range(self.red_list.count()):
            item = self.red_list.item(i)
            item.setHidden(text not in item.text().lower())
    
    def _filter_green_channels(self):
        """Filter green channel list based on search text."""
        text = self.green_search.text().lower()
        for i in range(self.green_list.count()):
            item = self.green_list.item(i)
            item.setHidden(text not in item.text().lower())
    
    def _filter_blue_channels(self):
        """Filter blue channel list based on search text."""
        text = self.blue_search.text().lower()
        for i in range(self.blue_list.count()):
            item = self.blue_list.item(i)
            item.setHidden(text not in item.text().lower())
    
    def _on_rgb_list_changed(self):
        # Ensure lists only keep checks for currently selected channels
        selected_channels = self._selected_channels()
        def _prune(lst: QtWidgets.QListWidget):
            for i in range(lst.count()):
                item = lst.item(i)
                if item.text() not in selected_channels:
                    item.setCheckState(Qt.Unchecked)
        _prune(self.red_list)
        _prune(self.green_list)
        _prune(self.blue_list)
        
        # Update arcsinh button state based on new RGB assignments
        self._update_minmax_controls_state()
        
        # Only refresh view if we're not already preserving zoom (to avoid double calls)
        if not self.preserve_zoom:
            self.preserve_zoom = True
            self._view_selected()

    def _on_grid_view_toggled(self):
        self._update_rgb_controls_visibility()
        
        # Show/hide "Show all channels" button based on grid view state
        if hasattr(self, 'show_all_channels_btn'):
            self.show_all_channels_btn.setVisible(self.grid_view_chk.isChecked())
        
        # Update RGB combination frame visibility
        if hasattr(self, 'rgb_combination_frame'):
            self._update_rgb_combination_visibility()
        
        # Update scaling channel combo when switching modes
        if hasattr(self, 'custom_scaling_chk') and self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
            self._load_channel_scaling()
        
        # Handle arcsinh state when switching between RGB and grid view
        if self.grid_view_chk.isChecked():
            # Switching to grid view - arcsinh becomes available for all channels
            self._enable_auto_scaling_for_grid_view()
        else:
            # Switching to RGB view - revert channels that had arcsinh applied in grid view
            self._revert_auto_scaling_for_rgb_view()
        
        self.preserve_zoom = True
        self._view_selected()

    def _on_grayscale_toggled(self):
        """Handle grayscale checkbox toggle."""
        self.preserve_zoom = True
        self._view_selected()

    def _update_rgb_controls_visibility(self):
        """Show RGB assignment panel only when grid view is off."""
        # Guard if called before widgets are constructed
        if not hasattr(self, 'color_assignment_frame'):
            return
        show_rgb = not self.grid_view_chk.isChecked()
        self.color_assignment_frame.setVisible(show_rgb)

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

    def _get_acquisition_info(self, acq_id: str) -> Optional[AcquisitionInfo]:
        """Get acquisition info for a given acquisition ID, handling multi-file cases.
        
        When multiple MCD files are loaded, acquisition IDs may not be unique across files.
        This method uses acq_to_file mapping to ensure we get the correct acquisition from the correct file.
        """
        if acq_id in self.acq_to_file:
            # Filter by both acq_id and source_file to ensure we get the right one
            source_file = self.acq_to_file[acq_id]
            acq_info = next((ai for ai in self.acquisitions if ai.id == acq_id and ai.source_file == source_file), None)
        else:
            # Single file or OME-TIFF: just match by acq_id
            acq_info = next((ai for ai in self.acquisitions if ai.id == acq_id), None)
        return acq_info
    
    def _get_cache_key(self, acq_id: str, channel: str) -> Tuple[str, str, str]:
        """Get a unique cache key for an acquisition and channel.
        
        When multiple MCD files are loaded, acquisition IDs may not be unique across files.
        This method includes the source file path in the cache key to prevent collisions.
        """
        if acq_id in self.acq_to_file:
            # Include source file path to make cache key unique
            source_file = self.acq_to_file[acq_id]
            cache_key = (source_file, acq_id, channel)
            return cache_key
        else:
            # Single file or OME-TIFF: use acq_id and channel (no file path needed)
            cache_key = ("", acq_id, channel)
            return cache_key
    
    def _get_acquisition_subtitle(self, acq_id: str) -> str:
        """Get acquisition subtitle showing well/description instead of acquisition number."""
        acq_info = self._get_acquisition_info(acq_id)
        if not acq_info:
            return "Unknown"
        
        # Use well if available, otherwise use name (which might be more descriptive)
        subtitle = f"{acq_info.well}" if acq_info.well else acq_info.name
        return subtitle

    def _on_custom_scaling_toggled(self):
        """Handle custom scaling checkbox toggle."""
        # Set preserve zoom FIRST before any other operations
        self.preserve_zoom = True
        
        self.scaling_frame.setVisible(self.custom_scaling_chk.isChecked())
        if self.custom_scaling_chk.isChecked():
            self._update_scaling_channel_combo()
            self._load_channel_scaling()
            # Initialize controls state
            self._update_minmax_controls_state()
        # Update RGB combination frame visibility (always check, even when disabled)
        self._update_rgb_combination_visibility()
        # Auto-refresh when toggled (preserve zoom)
        self._view_selected()
    
    def _on_rgb_combination_changed(self):
        """Handle changes to RGB channel combination method."""
        method_text = self.rgb_combination_combo.currentText()
        method_map = {
            "Raw Addition (sum per pixel)": "raw_addition",
            "Min/Max Scaled Mean (normalize each channel then mean)": "minmax_scaled_mean",
            "Raw Mean (mean per pixel)": "raw_mean",
            "Max (maximum per pixel)": "max"
        }
        self.rgb_combination_method = method_map.get(method_text, "raw_addition")
        # Auto-refresh when changed
        self.preserve_zoom = True
        self._view_selected()
    
    def _update_rgb_combination_visibility(self):
        """Update visibility of RGB combination method selector based on mode."""
        is_rgb_mode = hasattr(self, 'grid_view_chk') and not self.grid_view_chk.isChecked()
        self.rgb_combination_frame.setVisible(
            self.custom_scaling_chk.isChecked() and is_rgb_mode
        )

    def _update_scaling_channel_combo(self):
        """Update the scaling channel combo box with selected channels only."""
        self.scaling_channel_combo.clear()
        if self.current_acq_id is None:
            return
        
        # Check if we're in RGB mode (grid view is off)
        is_rgb_mode = hasattr(self, 'grid_view_chk') and not self.grid_view_chk.isChecked()
        
        if is_rgb_mode:
            # In RGB mode, show color names instead of individual channels
            self.scaling_channel_combo.addItem("Red")
            self.scaling_channel_combo.addItem("Green")
            self.scaling_channel_combo.addItem("Blue")
        else:
            # Only show currently selected channels
            selected_channels = self._selected_channels()
            for channel in selected_channels:
                self.scaling_channel_combo.addItem(channel)
        
        # Select first channel if available
        if self.scaling_channel_combo.count() > 0:
            self.scaling_channel_combo.setCurrentIndex(0)
            self._load_channel_scaling()

    def _on_scaling_channel_changed(self):
        """Handle changes to the scaling channel selection."""
        if self.custom_scaling_chk.isChecked():
            self._load_channel_scaling()
            # Update controls state based on current scaling method
            self._update_minmax_controls_state()
        else:
            # Even if custom scaling is off, ensure controls reflect per-channel method
            self._update_minmax_controls_state()
        # Auto-refresh
        self._view_selected()

    def _on_scaling_changed(self):
        """Handle changes to the min/max spinboxes."""
        if self.custom_scaling_chk.isChecked():
            # Save current values
            self._save_channel_scaling()
            # Auto-refresh display (preserve zoom)
            self.preserve_zoom = True
            self._view_selected()
    
    def _on_spillover_correction_toggled(self):
        """Handle spillover correction checkbox toggle."""
        self.preserve_zoom = True
        self.spillover_frame.setVisible(self.spillover_correction_chk.isChecked())
        self.spillover_correction_enabled = self.spillover_correction_chk.isChecked()
        # Auto-refresh when toggled (will check for matrix during viewing)
        self._view_selected()
    
    def _select_spillover_matrix(self):
        """Open file dialog to select spillover matrix CSV."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Spillover Matrix CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.spillover_matrix = load_spillover(file_path)
                self.spillover_matrix_path = file_path
                self.spillover_matrix_file_edit.setText(os.path.basename(file_path))
                # Reset warning flag since matrix is now loaded
                self._spillover_warning_shown = False
                
                # Check if matrix matches current acquisition channels
                if self.current_acq_id:
                    selected_channels = self._selected_channels()
                    if selected_channels:
                        # Check overlap
                        matrix_channels = set(self.spillover_matrix.columns)
                        image_channels = set(selected_channels)
                        common = matrix_channels.intersection(image_channels)
                        if not common:
                            QtWidgets.QMessageBox.warning(
                                self,
                                "Channel Mismatch",
                                "No overlapping channels between the spillover matrix and selected channels."
                            )
                
                # Enable correction if checkbox is checked
                if self.spillover_correction_chk.isChecked():
                    self.preserve_zoom = True
                    self._view_selected()
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Error loading spillover matrix:\n{str(e)}"
                )
                self.spillover_matrix = None
                self.spillover_matrix_path = None
                self.spillover_matrix_file_edit.setText("")
    
    def _on_spillover_method_changed(self):
        """Handle changes to spillover correction method."""
        self.spillover_method = self.spillover_method_combo.currentText()
        if self.spillover_correction_enabled:
            self.preserve_zoom = True
            self._view_selected()
    
    def _filter_channels(self):
        """Filter channels based on search text."""
        search_text = self.channel_search.text().lower()
        
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            channel_name = item.text().lower()
            item.setHidden(search_text not in channel_name)

    # ---------- Denoising ----------
    def _on_denoise_toggled(self):
        self.denoise_frame.setVisible(self.denoise_enable_chk.isChecked())
        self.preserve_zoom = True
        self._view_selected()
        # Ensure N control visibility is synced with method
        self._sync_hot_controls_visibility()

    def _on_order_changed(self):
        # Enforce uniqueness: if duplicates, rotate to next available
        combos = [self.order_combo_1, self.order_combo_2, self.order_combo_3]
        chosen = []
        for i, c in enumerate(combos):
            t = c.currentText()
            if t in chosen:
                # pick first not chosen
                for name in self.step_names:
                    if name not in chosen:
                        c.setCurrentText(name)
                        t = name
                        break
            chosen.append(t)
        # Refresh view to apply new order
        self._view_selected()

    def _apply_denoise_settings_and_refresh(self):
        """Capture current denoise UI settings, assign to selected channels in denoise list, refresh view."""
        try:
            # Target single channel from combo
            target_channel = self.denoise_channel_combo.currentText()
            if not target_channel:
                return

            # Build config from UI
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

            self.channel_denoise.setdefault(target_channel, {})
            self.channel_denoise[target_channel]["hot"] = cfg_hot
            self.channel_denoise[target_channel]["speckle"] = cfg_speckle
            self.channel_denoise[target_channel]["background"] = cfg_bg

            # Refresh (preserve zoom)
            self.preserve_zoom = True
            self._view_selected()
        except Exception:
            pass

    def _apply_denoise(self, channel: str, img: np.ndarray) -> np.ndarray:
        """Apply configured denoise steps for a channel in raw domain."""
        if not _HAVE_SCIKIT_IMAGE or not self.denoise_enable_chk.isChecked():
            return img
        cfg = self.channel_denoise.get(channel)
        if not cfg:
            return img
        out = img.astype(np.float32, copy=False)

        # Build ordered steps based on UI order selection
        order_map = {"Hot pixel": "hot", "Speckle": "speckle", "Background": "background"}
        chosen_order = [self.order_combo_1.currentText(), self.order_combo_2.currentText(), self.order_combo_3.currentText()]
        seen = set()
        exec_steps = []
        for name in chosen_order:
            key = order_map.get(name)
            if key and key not in seen:
                exec_steps.append(key)
                seen.add(key)

        for step in exec_steps:
            if step == "hot":
                hot = cfg.get("hot")
                if hot:
                    method = hot.get("method")
                    if method == "median3":
                        try:
                            out = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
                        except Exception:
                            out = ndi.median_filter(out, size=3)
                    elif method == "n_sd_local_median":
                        n_sd = float(hot.get("n_sd", 5.0))
                        try:
                            local_median = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
                        except Exception:
                            local_median = ndi.median_filter(out, size=3)
                        diff = out - local_median
                        local_var = ndi.uniform_filter(diff * diff, size=3)
                        local_std = np.sqrt(np.maximum(local_var, 1e-8))
                        mask_hot = diff > (n_sd * local_std)
                        out = np.where(mask_hot, local_median, out)
            elif step == "speckle":
                speckle = cfg.get("speckle")
                if speckle:
                    method = speckle.get("method")
                    if method == "gaussian":
                        sigma = float(speckle.get("sigma", 0.8))
                        out = gaussian(out, sigma=sigma, preserve_range=True)
                    elif method == "nl_means":
                        mn, mx = float(np.min(out)), float(np.max(out))
                        scale = mx - mn
                        scaled = (out - mn) / scale if scale > 0 else out
                        sigma_est = np.mean(estimate_sigma(scaled, channel_axis=None))
                        out = denoise_nl_means(
                            scaled,
                            h=1.15 * sigma_est,
                            fast_mode=True,
                            patch_size=5,
                            patch_distance=6,
                            channel_axis=None,
                        )
                        out = out * scale + mn
            elif step == "background":
                bg = cfg.get("background")
                if bg:
                    method = bg.get("method")
                    radius = int(bg.get("radius", 15))
                    if method == "white_tophat":
                        se = disk(radius)
                        try:
                            out = morphology.white_tophat(out, selem=se)
                        except TypeError:
                            out = morphology.white_tophat(out, footprint=se)
                    elif method == "black_tophat":
                        se = disk(radius)
                        try:
                            out = morphology.black_tophat(out, selem=se)
                        except TypeError:
                            out = morphology.black_tophat(out, footprint=se)
                    elif method == "rolling_ball":
                        if _HAVE_ROLLING_BALL:
                            background = _sk_rolling_ball(out, radius=radius)
                            out = out - background
                            out = np.clip(out, 0, None)
                        else:
                            se = disk(radius)
                            try:
                                opened = morphology.opening(out, selem=se)
                            except TypeError:
                                opened = morphology.opening(out, footprint=se)
                            out = out - opened
                            out = np.clip(out, 0, None)
        # Rescale to preserve original max intensity of this channel
        try:
            orig_max = float(np.max(img))
            new_max = float(np.max(out))
            if new_max > 0 and orig_max > 0:
                out = out * (orig_max / new_max)
        except Exception:
            pass
        # Clip to dtype range if integer
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            out = np.clip(out, info.min, info.max)
        else:
            out = np.clip(out, 0, None)
        return out.astype(img.dtype, copy=False)

    def _apply_custom_denoise(self, channel: str, img: np.ndarray, custom_denoise_settings: dict) -> np.ndarray:
        """Apply custom denoise steps for a channel in raw domain."""
        if not _HAVE_SCIKIT_IMAGE:
            return img
        cfg = custom_denoise_settings.get(channel)
        if not cfg:
            return img
        out = img.astype(np.float32, copy=False)

        # Apply denoising steps in order: hot pixel -> speckle -> background
        # Hot pixel removal
        hot = cfg.get("hot")
        if hot:
            method = hot.get("method")
            if method == "median3":
                try:
                    out = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
                except Exception:
                    out = ndi.median_filter(out, size=3)
            elif method == "n_sd_local_median":
                n_sd = float(hot.get("n_sd", 5.0))
                try:
                    local_median = median(out, footprint=footprint_rectangle(3, 3).astype(bool))
                except Exception:
                    local_median = ndi.median_filter(out, size=3)
                diff = out - local_median
                local_var = ndi.uniform_filter(diff * diff, size=3)
                local_std = np.sqrt(np.maximum(local_var, 1e-8))
                mask_hot = diff > (n_sd * local_std)
                out = np.where(mask_hot, local_median, out)
        
        # Speckle smoothing
        speckle = cfg.get("speckle")
        if speckle:
            method = speckle.get("method")
            if method == "gaussian":
                sigma = float(speckle.get("sigma", 0.8))
                out = gaussian(out, sigma=sigma, preserve_range=True)
            elif method == "nl_means":
                mn, mx = float(np.min(out)), float(np.max(out))
                scale = mx - mn
                scaled = (out - mn) / scale if scale > 0 else out
                sigma_est = np.mean(estimate_sigma(scaled, channel_axis=None))
                out = denoise_nl_means(
                    scaled,
                    h=1.15 * sigma_est,
                    fast_mode=True,
                    patch_size=5,
                    patch_distance=6,
                    channel_axis=None,
                )
                out = out * scale + mn
        
        # Background subtraction
        bg = cfg.get("background")
        if bg:
            method = bg.get("method")
            radius = int(bg.get("radius", 15))
            if method == "white_tophat":
                se = disk(radius)
                try:
                    out = morphology.white_tophat(out, selem=se)
                except TypeError:
                    out = morphology.white_tophat(out, footprint=se)
            elif method == "black_tophat":
                se = disk(radius)
                try:
                    out = morphology.black_tophat(out, selem=se)
                except TypeError:
                    out = morphology.black_tophat(out, footprint=se)
            elif method == "rolling_ball":
                if _HAVE_ROLLING_BALL:
                    background = _sk_rolling_ball(out, radius=radius)
                    out = out - background
                    out = np.clip(out, 0, None)
                else:
                    se = disk(radius)
                    try:
                        opened = morphology.opening(out, selem=se)
                    except TypeError:
                        opened = morphology.opening(out, footprint=se)
                    out = out - opened
                    out = np.clip(out, 0, None)
        
        # Rescale to preserve original max intensity of this channel
        try:
            orig_max = float(np.max(img))
            new_max = float(np.max(out))
            if new_max > 0 and orig_max > 0:
                out = out * (orig_max / new_max)
        except Exception:
            pass
        
        # Clip to dtype range if integer
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            out = np.clip(out, info.min, info.max)
        else:
            out = np.clip(out, 0, None)
        return out.astype(img.dtype, copy=False)

    def _on_hot_method_changed(self):
        self._sync_hot_controls_visibility()
        self._apply_denoise_settings_and_refresh()

    def _sync_hot_controls_visibility(self):
        # Show N only for ">N SD above local median"
        is_threshold = self.hot_pixel_method_combo.currentIndex() == 1
        self.hot_pixel_n_spin.setVisible(is_threshold)
        self.hot_pixel_n_label.setVisible(is_threshold)

    def _on_denoise_channel_changed(self):
        """Handle changes to the denoise channel selection."""
        self._load_denoise_settings()

    def _apply_denoise_to_all_channels(self):
        """Apply current denoising parameters to all channels (selected and unselected)."""
        try:
            # Get all available channels from the channel list
            all_channels = []
            for i in range(self.channel_list.count()):
                item = self.channel_list.item(i)
                all_channels.append(item.text())
            
            if not all_channels:
                return
            
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
            for channel in all_channels:
                self.channel_denoise.setdefault(channel, {})
                self.channel_denoise[channel]["hot"] = cfg_hot
                self.channel_denoise[channel]["speckle"] = cfg_speckle
                self.channel_denoise[channel]["background"] = cfg_bg

            # Show visual confirmation
            self.apply_all_channels_btn.setText("✓ Applied to All Channels")
            self.apply_all_channels_btn.setStyleSheet("QPushButton { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }")
            
            # Reset button appearance after 2 seconds
            QTimer.singleShot(2000, self._reset_apply_all_button)

            # Refresh view to show the changes
            self.preserve_zoom = True
            self._view_selected()
            
        except Exception as e:
            # Silently handle any errors to avoid disrupting the UI
            pass
    
    def _reset_apply_all_button(self):
        """Reset the apply all channels button to its original appearance."""
        self.apply_all_channels_btn.setText("Apply to All Channels")
        self.apply_all_channels_btn.setStyleSheet("")

    def _load_denoise_settings(self):
        """Load saved denoise settings for the currently selected denoise channel into the UI."""
        ch = self.denoise_channel_combo.currentText()
        if not ch:
            return
        cfg = self.channel_denoise.get(ch, {})
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
            self.hot_pixel_chk.blockSignals(False)
            self.hot_pixel_method_combo.blockSignals(False)
            self.hot_pixel_n_spin.blockSignals(False)
            self.speckle_chk.blockSignals(False)
            self.speckle_method_combo.blockSignals(False)
            self.gaussian_sigma_spin.blockSignals(False)
            self.bg_subtract_chk.blockSignals(False)
            self.bg_method_combo.blockSignals(False)
            self.bg_radius_spin.blockSignals(False)
        # Sync N visibility
        self._sync_hot_controls_visibility()

    def _show_all_channels(self):
        """Show all channels in a scrollable grid at original size."""
        if self.current_acq_id is None:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        
        try:
            # Get all channels for current acquisition
            loader = self._get_loader_for_acquisition(self.current_acq_id)
            if loader is None:
                QtWidgets.QMessageBox.information(self, "No loader", "No loader found for current acquisition.")
                return
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(self.current_acq_id)
            all_channels = loader.get_channels(original_acq_id)
            if not all_channels:
                QtWidgets.QMessageBox.information(self, "No channels", "No channels available.")
                return
            
            # Create dialog window - make it larger and resizable
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(f"All Channels - {self._get_acquisition_subtitle(self.current_acq_id)}")
            dialog.setModal(True)
            
            # Set size to 90% of main window
            main_window_size = self.size()
            dialog_width = int(main_window_size.width() * 0.9)
            dialog_height = int(main_window_size.height() * 0.9)
            dialog.resize(dialog_width, dialog_height)
            
            # Set minimum size
            dialog.setMinimumSize(800, 600)
            
            # Create control panel at top
            control_panel = QtWidgets.QHBoxLayout()
            
            # Image scaling controls
            scale_label = QtWidgets.QLabel("Image Scale:")
            scale_spin = QtWidgets.QDoubleSpinBox()
            scale_spin.setRange(0.5, 5.0)
            scale_spin.setDecimals(1)
            scale_spin.setValue(1.0)
            scale_spin.setSingleStep(0.1)
            control_panel.addWidget(scale_label)
            control_panel.addWidget(scale_spin)
            
            # Arcsinh scaling button
            arcsinh_btn = QtWidgets.QPushButton("Apply Arcsinh Scaling")
            arcsinh_btn.setCheckable(True)
            arcsinh_btn.setChecked(False)
            control_panel.addWidget(arcsinh_btn)
            
            # Co-factor spinbox
            cofactor_label = QtWidgets.QLabel("Co-factor:")
            cofactor_spin = QtWidgets.QDoubleSpinBox()
            cofactor_spin.setRange(0.1, 100.0)
            cofactor_spin.setDecimals(1)
            cofactor_spin.setValue(10.0)
            cofactor_spin.setSingleStep(0.5)
            control_panel.addWidget(cofactor_label)
            control_panel.addWidget(cofactor_spin)
            
            control_panel.addStretch()
            
            # Close button
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            control_panel.addWidget(close_btn)
            
            # Create scroll area
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Create widget to hold the grid
            grid_widget = QtWidgets.QWidget()
            grid_layout = QtWidgets.QGridLayout(grid_widget)
            grid_layout.setSpacing(5)  # Reduced spacing between images
            
            # Calculate grid dimensions based on available width
            # We'll calculate this after we know the image sizes, so start with a reasonable default
            n_channels = len(all_channels)
            cols = 4  # Start with 4 columns, will be adjusted
            rows = max(1, int(np.ceil(n_channels / cols)))
            
            # Store references to canvases and images for scaling and arcsinh
            canvas_refs = []
            image_refs = []
            original_sizes = []  # Store original image dimensions
            
            # Load and display each channel
            for i, channel in enumerate(all_channels):
                try:
                    # Load image with normalization and denoising
                    img = self._load_image_with_normalization(self.current_acq_id, channel)
                    image_refs.append(img)  # Store original image
                    original_sizes.append((img.shape[1], img.shape[0]))  # Store (width, height)
                    
                    # Create matplotlib figure for this channel
                    fig = Figure(figsize=(3, 3), dpi=100)
                    ax = fig.add_subplot(111)
                    
                    # Display image
                    if self.grayscale_chk.isChecked():
                        im = ax.imshow(img, cmap='gray', interpolation='nearest')
                    else:
                        im = ax.imshow(img, cmap='viridis', interpolation='nearest')
                    
                    ax.set_title(channel, fontsize=9, pad=3)
                    ax.axis('off')
                    
                    # Create canvas
                    canvas = FigureCanvas(fig)
                    # Set initial size to actual image dimensions (but with reasonable limits)
                    max_size = 400  # Maximum size to prevent huge images
                    min_size = 100  # Minimum size for visibility
                    width = max(min_size, min(max_size, img.shape[1]))
                    height = max(min_size, min(max_size, img.shape[0]))
                    canvas.setFixedSize(width, height)
                    
                    # Store references
                    canvas_refs.append((canvas, ax, im, fig))
                    
                    # Add to grid
                    row = i // cols
                    col = i % cols
                    grid_layout.addWidget(canvas, row, col)
                    
                except Exception as e:
                    print(f"Error loading channel {channel}: {e}")
                    # Add placeholder for failed channel
                    label = QtWidgets.QLabel(f"Error loading\n{channel}")
                    label.setAlignment(Qt.AlignCenter)
                    label.setStyleSheet("QLabel { border: 1px solid red; color: red; }")
                    label.setFixedSize(150, 150)
                    row = i // cols
                    col = i % cols
                    grid_layout.addWidget(label, row, col)
                    canvas_refs.append(None)  # Placeholder for failed channel
                    image_refs.append(None)
                    original_sizes.append((150, 150))  # Default size for error placeholder
            
            # Set scroll area widget
            scroll_area.setWidget(grid_widget)
            
            # Function to calculate optimal number of columns based on available width
            def calculate_columns():
                # Get available width (accounting for scrollbar and margins)
                available_width = scroll_area.width() - 50  # Account for scrollbar and margins
                if available_width <= 0:
                    return 4  # Default fallback
                
                # Get current image width (assuming all images are similar size)
                if canvas_refs and canvas_refs[0] is not None:
                    current_width = canvas_refs[0][0].width()
                else:
                    current_width = 200  # Default width
                
                # Calculate how many columns fit
                cols = max(1, available_width // (current_width + 5))  # +5 for spacing
                return min(cols, n_channels)  # Don't exceed number of channels
            
            # Function to update image sizes based on scale
            def update_image_sizes():
                scale = scale_spin.value()
                for i, (canvas_ref, orig_size) in enumerate(zip(canvas_refs, original_sizes)):
                    if canvas_ref is None:
                        continue
                    canvas, ax, im, fig = canvas_ref
                    
                    # Calculate new size based on scale
                    orig_width, orig_height = orig_size
                    new_width = int(orig_width * scale)
                    new_height = int(orig_height * scale)
                    
                    # Apply reasonable limits
                    max_size = 600  # Increased maximum size
                    min_size = 50   # Reduced minimum size
                    new_width = max(min_size, min(max_size, new_width))
                    new_height = max(min_size, min(max_size, new_height))
                    
                    # Update canvas size
                    canvas.setFixedSize(new_width, new_height)
                
                # Recalculate grid layout after size changes
                update_grid_layout()
            
            # Function to update grid layout
            def update_grid_layout():
                # Calculate optimal number of columns
                optimal_cols = calculate_columns()
                
                # Clear current layout
                for i in reversed(range(grid_layout.count())):
                    grid_layout.itemAt(i).widget().setParent(None)
                
                # Re-add widgets with new column count
                for i, canvas_ref in enumerate(canvas_refs):
                    if canvas_ref is None:
                        # Handle error placeholders
                        label = QtWidgets.QLabel(f"Error loading\n{all_channels[i]}")
                        label.setAlignment(Qt.AlignCenter)
                        label.setStyleSheet("QLabel { border: 1px solid red; color: red; }")
                        label.setFixedSize(150, 150)
                        row = i // optimal_cols
                        col = i % optimal_cols
                        grid_layout.addWidget(label, row, col)
                    else:
                        canvas, ax, im, fig = canvas_ref
                        row = i // optimal_cols
                        col = i % optimal_cols
                        grid_layout.addWidget(canvas, row, col)
                
                # Force the grid widget to update its size
                grid_widget.adjustSize()
                grid_widget.updateGeometry()
                
                # Update the scroll area's widget size
                scroll_area.widget().adjustSize()
            
            # Function to apply arcsinh scaling
            def apply_arcsinh_scaling():
                cofactor = cofactor_spin.value()
                for i, (canvas_ref, img) in enumerate(zip(canvas_refs, image_refs)):
                    if canvas_ref is None or img is None:
                        continue
                    canvas, ax, im, fig = canvas_ref
                    
                    if arcsinh_btn.isChecked():
                        # Apply arcsinh scaling
                        scaled_img = arcsinh_normalize(img, cofactor=cofactor)
                    else:
                        # Use original image
                        scaled_img = img
                    
                    # Update the image
                    im.set_array(scaled_img)
                    im.set_clim(vmin=np.min(scaled_img), vmax=np.max(scaled_img))
                    canvas.draw()
            
            # Connect controls
            scale_spin.valueChanged.connect(update_image_sizes)
            arcsinh_btn.toggled.connect(apply_arcsinh_scaling)
            cofactor_spin.valueChanged.connect(apply_arcsinh_scaling)
            
            # Connect scroll area resize to update grid layout
            def on_scroll_area_resize(event):
                QtWidgets.QScrollArea.resizeEvent(scroll_area, event)
                update_grid_layout()
            
            scroll_area.resizeEvent = on_scroll_area_resize
            
            # Initial grid layout update
            update_grid_layout()
            
            # Create main layout
            main_layout = QtWidgets.QVBoxLayout(dialog)
            main_layout.addLayout(control_panel)
            main_layout.addWidget(scroll_area)
            
            # Show dialog
            dialog.exec_()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error showing all channels: {str(e)}")
    
    def _update_minmax_controls_state(self):
        """Enable/disable min/max controls based on scaling method."""
        # Determine current channel's method
        current_channel = self.scaling_channel_combo.currentText()
        method = self.channel_scaling_method.get(current_channel, "default")
        
        # Check if automatic scaling should be disabled for this channel
        auto_scaling_disabled = self._is_auto_scaling_disabled_for_channel(current_channel)
        
        if method in ["arcsinh"]:
            # Disable min/max controls for automatic scaling methods
            self.min_spinbox.setEnabled(False)
            self.max_spinbox.setEnabled(False)
            self.min_spinbox.setStyleSheet("QDoubleSpinBox { background-color: #f0f0f0; color: #666; }")
            self.max_spinbox.setStyleSheet("QDoubleSpinBox { background-color: #f0f0f0; color: #666; }")
        else:
            # Enable min/max controls for manual/default scaling
            self.min_spinbox.setEnabled(True)
            self.max_spinbox.setEnabled(True)
            self.min_spinbox.setStyleSheet("")
            self.max_spinbox.setStyleSheet("")
        
        # Update arcsinh button states
        if auto_scaling_disabled:
            self.arcsinh_btn.setEnabled(False)
            self.arcsinh_btn.setToolTip("Arcsinh disabled: Multiple markers assigned to same RGB color")
            self.cofactor_spinbox.setEnabled(False)
        else:
            self.arcsinh_btn.setEnabled(True)
            self.arcsinh_btn.setToolTip("Apply arcsinh normalization to current channel")
            self.cofactor_spinbox.setEnabled(True)

    def _is_auto_scaling_disabled_for_channel(self, channel: str) -> bool:
        """Check if automatic scaling (arcsinh) should be disabled for a channel because multiple markers are assigned to the same RGB color."""
        if not channel:
            return False
        
        # In grid view, all channels are displayed individually, so arcsinh is always available
        if self.grid_view_chk.isChecked():
            return False
        
        # Get current RGB color assignments
        def _checked(lst: QtWidgets.QListWidget) -> List[str]:
            vals: List[str] = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    vals.append(item.text())
            return vals
        
        red_selection = _checked(self.red_list)
        green_selection = _checked(self.green_list)
        blue_selection = _checked(self.blue_list)
        
        # Check if this channel is assigned to any RGB color that has multiple channels
        if channel in red_selection and len(red_selection) > 1:
            return True
        if channel in green_selection and len(green_selection) > 1:
            return True
        if channel in blue_selection and len(blue_selection) > 1:
            return True
        
        return False

    def _enable_auto_scaling_for_grid_view(self):
        """Enable arcsinh for all channels when switching to grid view."""
        # In grid view, all channels are displayed individually, so arcsinh is always available
        # No special action needed - the _update_minmax_controls_state will handle enabling the buttons
        pass

    def _revert_auto_scaling_for_rgb_view(self):
        """Revert channels to default range when switching back to RGB view if they had arcsinh applied in grid view."""
        # Get current RGB color assignments
        def _checked(lst: QtWidgets.QListWidget) -> List[str]:
            vals: List[str] = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    vals.append(item.text())
            return vals
        
        red_selection = _checked(self.red_list)
        green_selection = _checked(self.green_list)
        blue_selection = _checked(self.blue_list)
        
        # Find channels that have multiple assignments and had arcsinh applied
        channels_to_revert = []
        
        # Check red channels
        if len(red_selection) > 1:
            for channel in red_selection:
                if (channel in self.channel_scaling_method and 
                    self.channel_scaling_method[channel] == "arcsinh"):
                    channels_to_revert.append(channel)
        
        # Check green channels
        if len(green_selection) > 1:
            for channel in green_selection:
                if (channel in self.channel_scaling_method and 
                    self.channel_scaling_method[channel] == "arcsinh"):
                    channels_to_revert.append(channel)
        
        # Check blue channels
        if len(blue_selection) > 1:
            for channel in blue_selection:
                if (channel in self.channel_scaling_method and 
                    self.channel_scaling_method[channel] == "arcsinh"):
                    channels_to_revert.append(channel)
        
        # Revert each channel to default range
        for channel in channels_to_revert:
            try:
                if self.current_acq_id:
                    loader = self._get_loader_for_acquisition(self.current_acq_id)
                    if loader is None:
                        continue
                    img = loader.get_image(self.current_acq_id, channel)
                    min_val = float(np.min(img))
                    max_val = float(np.max(img))
                    
                    # Update scaling method to default
                    self.channel_scaling_method[channel] = "default"
                    
                    # Clear normalization settings
                    if channel in self.channel_normalization:
                        self.channel_normalization.pop(channel, None)
                    
                    # Update scaling values
                    self.channel_scaling[channel] = {'min': min_val, 'max': max_val}
                    
            except Exception as e:
                print(f"Error reverting channel {channel} to default range: {e}")
        
        # Update UI controls to reflect the reverted state
        self._update_minmax_controls_state()

    def _load_channel_scaling(self):
        """Load scaling values for the currently selected channel or RGB color."""
        current_selection = self.scaling_channel_combo.currentText()
        if not current_selection:
            return
        
        # Check if we're in RGB mode (selection is a color name)
        is_rgb_color = current_selection in ['Red', 'Green', 'Blue']
        
        if is_rgb_color:
            # Load RGB color scaling
            if current_selection in self.rgb_color_scaling:
                # Load saved values
                min_val = self.rgb_color_scaling[current_selection]['min']
                max_val = self.rgb_color_scaling[current_selection]['max']
            else:
                # Use default range - compute from actual RGB composite
                if self.current_acq_id is None:
                    return
                try:
                    # Get the actual RGB channel values to compute default range
                    def _checked(lst: QtWidgets.QListWidget) -> List[str]:
                        vals: List[str] = []
                        for i in range(lst.count()):
                            item = lst.item(i)
                            if item.checkState() == Qt.Checked:
                                vals.append(item.text())
                        return vals
                    
                    red_selection = _checked(self.red_list)
                    green_selection = _checked(self.green_list)
                    blue_selection = _checked(self.blue_list)
                    
                    color_selections = {'Red': red_selection, 'Green': green_selection, 'Blue': blue_selection}
                    selected_channels = color_selections.get(current_selection, [])
                    
                    if selected_channels:
                        # Compute combined channel to get range
                        loader = self._get_loader_for_acquisition(self.current_acq_id)
                        if loader is None:
                            return
                        
                        # Get first channel to determine image size
                        first_img = loader.get_image(self.current_acq_id, selected_channels[0])
                        # Apply spillover correction if enabled
                        corrected_images = {}
                        if self.spillover_correction_enabled and self.spillover_matrix is not None:
                            corrected_images = self._apply_spillover_correction_to_channels(self.current_acq_id, selected_channels)
                        combined = self._combine_channels_for_rgb(selected_channels, first_img, corrected_images)
                        min_val = float(np.min(combined))
                        max_val = float(np.max(combined))
                    else:
                        # No channels selected, use default
                        min_val = 0.0
                        max_val = 1000.0
                except Exception as e:
                    print(f"Error loading RGB color scaling: {e}")
                    min_val = 0.0
                    max_val = 1000.0
        else:
            # Load channel scaling (non-RGB mode)
            if current_selection in self.channel_scaling:
                # Load saved values
                min_val = self.channel_scaling[current_selection]['min']
                max_val = self.channel_scaling[current_selection]['max']
            else:
                # Use default range (full image range)
                if self.current_acq_id is None:
                    return
                try:
                    loader = self._get_loader_for_acquisition(self.current_acq_id)
                    if loader is None:
                        return
                    img = loader.get_image(self.current_acq_id, current_selection)
                    min_val = float(np.min(img))
                    max_val = float(np.max(img))
                except Exception as e:
                    print(f"Error loading channel scaling: {e}")
                    return
        
        # Update spinboxes based on actual values
        self._update_spinboxes_from_values(min_val, max_val)
        
        # Load per-channel arcsinh cofactor if available (only for non-RGB mode)
        if not is_rgb_color and current_selection in self.channel_normalization:
            norm_cfg = self.channel_normalization[current_selection]
            if norm_cfg.get("method") == "arcsinh":
                cofactor = norm_cfg.get("cofactor", 10.0)
                self.cofactor_spinbox.setValue(float(cofactor))

    def _save_channel_scaling(self):
        """Save current scaling values for the selected channel or RGB color."""
        current_selection = self.scaling_channel_combo.currentText()
        if not current_selection:
            return
        
        # Get values directly from spinboxes
        min_val = self.min_spinbox.value()
        max_val = self.max_spinbox.value()
        
        # Check if we're in RGB mode (selection is a color name)
        is_rgb_color = current_selection in ['Red', 'Green', 'Blue']
        
        if is_rgb_color:
            # Save RGB color scaling
            self.rgb_color_scaling[current_selection] = {'min': min_val, 'max': max_val}
        else:
            # Save channel scaling
            self.channel_scaling[current_selection] = {'min': min_val, 'max': max_val}

    def _update_spinboxes_from_values(self, min_val, max_val):
        """Update spinboxes based on actual min/max values."""
        # Update spinboxes without triggering valueChanged
        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(min_val)
        self.max_spinbox.setValue(max_val)
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)


    def _arcsinh_normalization(self):
        """Apply arcsinh normalization with configurable co-factor."""
        if self.current_acq_id is None:
            return
        
        current_channel = self.scaling_channel_combo.currentText()
        if not current_channel:
            return
        
        # Check if automatic scaling is disabled for this channel
        if self._is_auto_scaling_disabled_for_channel(current_channel):
            QtWidgets.QMessageBox.warning(self, "Arcsinh Disabled", 
                f"Arcsinh normalization is disabled for '{current_channel}' because multiple markers are assigned to the same RGB color.")
            return
        
        try:
            loader = self._get_loader_for_acquisition(self.current_acq_id)
            if loader is None:
                return
            img = loader.get_image(self.current_acq_id, current_channel)
            # Apply denoising if enabled/configured before normalization
            try:
                img = self._apply_denoise(current_channel, img)
            except Exception:
                pass
            cofactor = self.cofactor_spinbox.value()
            
            # Apply arcsinh normalization
            normalized_img = arcsinh_normalize(img, cofactor=cofactor)
            
            # Get the min/max values of the normalized image for scaling
            min_val = float(np.min(normalized_img))
            max_val = float(np.max(normalized_img))
            
            self._update_spinboxes_from_values(min_val, max_val)
            
            # Update scaling method state
            self.current_scaling_method = "arcsinh"
            self.channel_scaling_method[current_channel] = "arcsinh"
            # Only set normalization for the selected channel
            self.channel_normalization[current_channel] = {"method": "arcsinh", "cofactor": cofactor}
            self._update_minmax_controls_state()
            
            # Auto-apply the scaling
            self._save_channel_scaling()
            self._view_selected()
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
            loader = self._get_loader_for_acquisition(self.current_acq_id)
            if loader is None:
                return
            img = loader.get_image(self.current_acq_id, current_channel)
            min_val = float(np.min(img))
            max_val = float(np.max(img))
            
            self._update_spinboxes_from_values(min_val, max_val)
            
            # Update scaling method state
            self.current_scaling_method = "default"
            self.channel_scaling_method[current_channel] = "default"
            # Clear per-channel normalization for this channel
            if current_channel in self.channel_normalization:
                self.channel_normalization.pop(current_channel, None)
            self._update_minmax_controls_state()
            
            # Auto-apply the scaling and reload image in original range
            self._save_channel_scaling()
            self._view_selected()
        except Exception as e:
            print(f"Error in default range: {e}")

    def _load_image_with_normalization(self, acq_id: str, channel: str) -> np.ndarray:
        """Load image, apply denoising (per-channel) then normalization if enabled."""
        # Try cache first
        cache_key = self._get_cache_key(acq_id, channel)
        with self._cache_lock:
            img = self.image_cache.get(cache_key)
        if img is None:
            loader = self._get_loader_for_acquisition(acq_id)
            if loader is None:
                raise ValueError(f"No loader found for acquisition {acq_id}")
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_id)
            img = loader.get_image(original_acq_id, channel)
            with self._cache_lock:
                self.image_cache[cache_key] = img
        
        # Apply per-channel denoising first (operates in raw space)
        try:
            img = self._apply_denoise(channel, img)
        except Exception:
            pass

        # Apply per-channel normalization (if configured)
        norm_cfg = self.channel_normalization.get(channel)
        if norm_cfg and norm_cfg.get("method") == "arcsinh":
            cofactor = float(norm_cfg.get("cofactor", 10.0))
            img = arcsinh_normalize(img, cofactor=cofactor)
        
        return img
    
    def _load_image_with_denoising_only(self, acq_id: str, channel: str) -> np.ndarray:
        """Load image and apply denoising only (no normalization)."""
        # Try cache first
        cache_key = self._get_cache_key(acq_id, channel)
        with self._cache_lock:
            img = self.image_cache.get(cache_key)
        if img is None:
            loader = self._get_loader_for_acquisition(acq_id)
            if loader is None:
                raise ValueError(f"No loader found for acquisition {acq_id}")
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_id)
            img = loader.get_image(original_acq_id, channel)
            with self._cache_lock:
                self.image_cache[cache_key] = img
        
        # Apply per-channel denoising (operates in raw space)
        try:
            img = self._apply_denoise(channel, img)
        except Exception:
            pass
        
        return img
    
    def _apply_spillover_correction_to_channels(
        self, 
        acq_id: str, 
        channels: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Apply spillover correction to a set of channels.
        Order: 1) denoising, 2) spillover correction, 3) normalization/scaling
        
        Returns a dictionary of corrected channel images with normalization applied.
        """
        # Check if correction is enabled and matrix is loaded
        if not self.spillover_correction_enabled or self.spillover_matrix is None:
            # Show warning once if correction is enabled but no matrix
            if self.spillover_correction_enabled and self.spillover_matrix is None and not self._spillover_warning_shown:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Spillover Matrix",
                    "Spillover correction is enabled but no matrix is loaded. Images will be displayed without correction.\n\nPlease load a spillover matrix CSV file or disable spillover correction."
                )
                self._spillover_warning_shown = True
            # Return images with full normalization if correction not enabled
            return {ch: self._load_image_with_normalization(acq_id, ch) for ch in channels}
        
        # Step 1: Load all channels with denoising only (no normalization)
        channel_images_denoised = {}
        for ch in channels:
            channel_images_denoised[ch] = self._load_image_with_denoising_only(acq_id, ch)
        
        # Find channels that are in both the image and spillover matrix
        matrix_channels = set(self.spillover_matrix.columns)
        image_channels = set(channels)
        common_channels = sorted(list(matrix_channels.intersection(image_channels)), key=lambda x: channels.index(x) if x in channels else 999)
        
        if not common_channels:
            # No overlap, apply normalization to original images and return
            channel_images = {}
            for ch in channels:
                img = channel_images_denoised[ch]
                # Apply normalization if configured
                norm_cfg = self.channel_normalization.get(ch)
                if norm_cfg and norm_cfg.get("method") == "arcsinh":
                    cofactor = float(norm_cfg.get("cofactor", 10.0))
                    img = arcsinh_normalize(img, cofactor=cofactor)
                channel_images[ch] = img
            return channel_images
        
        # Step 2: Apply spillover correction
        # Get image dimensions from first channel
        first_img = channel_images_denoised[common_channels[0]]
        H, W = first_img.shape
        
        # Stack channels into 3D array (H x W x C)
        # Only include channels that are in the spillover matrix
        channel_stack = []
        channel_order = []
        for ch in common_channels:
            channel_stack.append(channel_images_denoised[ch])
            channel_order.append(ch)
        
        # Stack into 3D array
        img_stack = np.stack(channel_stack, axis=2)  # H x W x C
        
        # Adapt spillover matrix to channel order
        # Create a submatrix with only the channels we're processing, in the correct order
        sm_adapted = self.spillover_matrix.loc[channel_order, channel_order]
        
        # Apply spillover correction
        try:
            corrected_stack = comp_image_counts(
                img_stack,
                sm_adapted,
                method=self.spillover_method
            )
            
            # Update channel_images_denoised with corrected values
            for i, ch in enumerate(channel_order):
                channel_images_denoised[ch] = corrected_stack[:, :, i]
        except Exception as e:
            print(f"Error applying spillover correction: {e}")
            # Continue with original images on error
        
        # Step 3: Apply normalization/scaling to all channels
        channel_images = {}
        for ch in channels:
            img = channel_images_denoised[ch]
            # Apply normalization if configured
            norm_cfg = self.channel_normalization.get(ch)
            if norm_cfg and norm_cfg.get("method") == "arcsinh":
                cofactor = float(norm_cfg.get("cofactor", 10.0))
                img = arcsinh_normalize(img, cofactor=cofactor)
            channel_images[ch] = img
        
        return channel_images

    def _start_prefetch_all_channels(self, acq_id: str):
        """Prefetch all channels for the given acquisition in the background (non-blocking)."""
        if not acq_id:
            return
        
        loader = self._get_loader_for_acquisition(acq_id)
        if loader is None:
            return
        
        # If a previous prefetch is running, let it finish; avoid stacking tasks
        if self._prefetch_future and not self._prefetch_future.done():
            return

        channels = []
        try:
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_id)
            channels = loader.get_channels(original_acq_id)
        except Exception:
            return

        def _prefetch():
            try:
                # Get original acquisition ID if this is a unique ID
                original_acq_id = self._get_original_acq_id(acq_id)
                # Load the full stack once, then split into channels for faster access
                stack = loader.get_all_channels(original_acq_id)
                # Store in cache
                with self._cache_lock:
                    for i, ch in enumerate(channels):
                        try:
                            cache_key = self._get_cache_key(acq_id, ch)
                            self.image_cache[cache_key] = stack[..., i]
                        except Exception:
                            continue
            except Exception:
                # Swallow errors silently to avoid UI disruption
                return

        self._prefetch_future = self._executor.submit(_prefetch)

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
            # Silent no-op during auto-refresh before an acquisition is selected
            return
        chans = self._selected_channels()
        if not chans:
            # Check if we can show cluster mask only (no channel overlay)
            overlay_mode = getattr(self, 'segmentation_overlay_mode', 'Mask')
            if (self.segmentation_overlay and 
                overlay_mode == "Cluster" and 
                self.current_acq_id in self.segmentation_masks):
                # Show cluster mask only (no channel overlay)
                self._show_cluster_mask_only()
                return
            
            # Clear the canvas when no channels are selected
            self.canvas.fig.clear()
            self.canvas.draw()
            # Clear any saved zoom limits and reset preserve flag to prevent issues when channels are selected again
            self.saved_zoom_limits = None
            self.preserve_zoom = False
            return
        
        # Check if we're going from no channels to having channels (force fresh start)
        if not hasattr(self, 'last_selected_channels') or not self.last_selected_channels:
            # Force a fresh start - clear any zoom preservation
            self.saved_zoom_limits = None
            self.preserve_zoom = False
        
        # Store selected channels for auto-selection in next acquisition
        self.last_selected_channels = chans.copy()
        
        grayscale = self.grayscale_chk.isChecked()
        grid_view = self.grid_view_chk.isChecked()
        
        # Save zoom limits if we should preserve them
        if self.preserve_zoom:
            self._save_zoom_limits()
        
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
            if not grid_view:
                # RGB composite view using user-selected color assignments (supports single or multiple channels per RGB)
                self._show_rgb_composite(chans, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True)
                # Apply spillover correction if enabled
                if self.spillover_correction_enabled and self.spillover_matrix is not None:
                    corrected_images = self._apply_spillover_correction_to_channels(self.current_acq_id, chans)
                    images = [corrected_images[c] for c in chans]
                else:
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
                
                # Get scale bar parameters if enabled
                scale_bar_length_um = None
                pixel_size_um = None
                if self.scale_bar_chk.isChecked():
                    scale_bar_length_um = self.scale_bar_length_spin.value()
                    pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
                
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=chans, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked(),
                                    scale_bar_length_um=scale_bar_length_um, pixel_size_um=pixel_size_um)
                
                # Add cluster legend if cluster overlay mode is active
                self._add_cluster_legend()
            
            # Restore zoom limits if we preserved them
            if self.preserve_zoom:
                self._restore_zoom_limits()
                self.preserve_zoom = False  # Reset flag
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "View error", str(e))
    
    def _reset_zoom(self):
        """Reset zoom to original view by clearing saved limits and redrawing the whole canvas."""
        # Clear any saved zoom limits
        self.saved_zoom_limits = None
        # Don't preserve zoom - force reset
        self.preserve_zoom = False
        
        # Redraw the current view - this will reset everything to default
        # imshow automatically sets the axes limits to show the full image
        self._view_selected()
        
        # After redrawing, update the navigation toolbar's home view
        # This ensures the toolbar's zoom state is properly reset
        # The view_selected() call already sets the correct limits via imshow,
        # so we just need to tell the toolbar that this is the new "home" view
        try:
            if hasattr(self, 'nav_toolbar') and self.nav_toolbar:
                # Push the current view (full image) as the new home
                # This updates the toolbar's internal state
                self.nav_toolbar.push_current()
                # Navigate to home (which is now the full image view)
                self.nav_toolbar.home()
        except Exception:
            # If toolbar update fails, that's okay - the view is already reset
            pass
    
    def _save_zoom_limits(self):
        """Save current zoom limits."""
        try:
            # For grid view, save limits for all axes
            if hasattr(self.canvas, 'grid_axes') and self.canvas.grid_axes:
                self.saved_zoom_limits = []
                for ax in self.canvas.grid_axes:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    self.saved_zoom_limits.append((xlim, ylim))
            # For single view, save limits for main axis
            elif hasattr(self.canvas, 'ax') and self.canvas.ax:
                xlim = self.canvas.ax.get_xlim()
                ylim = self.canvas.ax.get_ylim()
                self.saved_zoom_limits = (xlim, ylim)
            else:
                self.saved_zoom_limits = None
        except Exception:
            self.saved_zoom_limits = None
    
    def _restore_zoom_limits(self):
        """Restore saved zoom limits."""
        if self.saved_zoom_limits is None:
            return
        
        try:
            # For grid view, restore limits for all axes
            if (hasattr(self.canvas, 'grid_axes') and self.canvas.grid_axes and 
                isinstance(self.saved_zoom_limits, list)):
                for i, (ax, (xlim, ylim)) in enumerate(zip(self.canvas.grid_axes, self.saved_zoom_limits)):
                    if i < len(self.saved_zoom_limits):
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
            # For single view, restore limits for main axis
            elif (hasattr(self.canvas, 'ax') and self.canvas.ax and 
                  isinstance(self.saved_zoom_limits, tuple) and len(self.saved_zoom_limits) == 2):
                xlim, ylim = self.saved_zoom_limits
                self.canvas.ax.set_xlim(xlim)
                self.canvas.ax.set_ylim(ylim)
            
            # Redraw the canvas
            self.canvas.draw()
        except Exception:
            pass
    

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
            
            if not grid_view:
                # RGB composite view using user-selected color assignments (supports single or multiple channels per RGB)
                self._show_rgb_composite(selected_channels, grayscale)
            else:
                # Grid view for multiple channels (when grid_view is True)
                # Apply spillover correction if enabled
                if self.spillover_correction_enabled and self.spillover_matrix is not None:
                    corrected_images = self._apply_spillover_correction_to_channels(self.current_acq_id, selected_channels)
                    images = [corrected_images[c] for c in selected_channels]
                else:
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
                
                # Get scale bar parameters if enabled
                scale_bar_length_um = None
                pixel_size_um = None
                if self.scale_bar_chk.isChecked():
                    scale_bar_length_um = self.scale_bar_length_spin.value()
                    pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
                
                self.canvas.show_grid(images, titles, grayscale=grayscale, raw_images=images, 
                                    channel_names=selected_channels, channel_scaling=self.channel_scaling, 
                                    custom_scaling_enabled=self.custom_scaling_chk.isChecked(),
                                    scale_bar_length_um=scale_bar_length_um, pixel_size_um=pixel_size_um)
                
                # Add cluster legend if cluster overlay mode is active
                self._add_cluster_legend()
        except Exception as e:
            print(f"Auto-load error: {e}")

    def _combine_channels_for_rgb(self, channel_names: List[str], first_img: np.ndarray, corrected_images: dict = None) -> np.ndarray:
        """Combine multiple channels into a single RGB color channel using the selected method.
        
        Args:
            channel_names: List of channel names to combine
            first_img: Reference image to determine output shape and dtype
            corrected_images: Dictionary of corrected images (from spillover correction)
        
        Returns:
            Combined channel image
        """
        if not channel_names:
            return np.zeros_like(first_img)
        
        # Load all channel images
        channel_images = []
        for ch_name in channel_names:
            try:
                if corrected_images and ch_name in corrected_images:
                    img = corrected_images[ch_name]
                else:
                    img = self._load_image_with_normalization(self.current_acq_id, ch_name)
                channel_images.append(img.astype(np.float32))
            except Exception:
                channel_images.append(np.zeros_like(first_img, dtype=np.float32))
        
        if not channel_images:
            return np.zeros_like(first_img)
        
        # Stack all channels
        stacked = np.stack(channel_images, axis=0)  # Shape: (n_channels, H, W)
        
        # Apply combination method
        if self.rgb_combination_method == "raw_addition":
            # Sum all channels per pixel
            combined = np.sum(stacked, axis=0)
        elif self.rgb_combination_method == "minmax_scaled_mean":
            # Normalize each channel to 0-1 range, then take mean
            normalized_channels = []
            for img in channel_images:
                img_min = np.min(img)
                img_max = np.max(img)
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                else:
                    normalized = img
                normalized_channels.append(normalized)
            normalized_stack = np.stack(normalized_channels, axis=0)
            combined = np.mean(normalized_stack, axis=0)
            # Scale back to original range (approximate)
            combined = combined * (np.max(stacked) - np.min(stacked)) + np.min(stacked)
        elif self.rgb_combination_method == "raw_mean":
            # Mean of all channels per pixel
            combined = np.mean(stacked, axis=0)
        elif self.rgb_combination_method == "max":
            # Maximum per pixel across all channels
            combined = np.max(stacked, axis=0)
        else:
            # Default to raw addition
            combined = np.sum(stacked, axis=0)
        
        # Clip to prevent overflow and convert back to original dtype
        combined = np.clip(combined, 0, np.max(combined))
        return combined.astype(first_img.dtype)

    def _show_rgb_composite(self, selected_channels: List[str], grayscale: bool):
        """Show RGB composite using user-selected color assignments."""
        # Get user-selected color assignments
        # Read multi-selections for each color
        def _checked(lst: QtWidgets.QListWidget) -> List[str]:
            vals: List[str] = []
            for i in range(lst.count()):
                item = lst.item(i)
                if item.checkState() == Qt.Checked:
                    vals.append(item.text())
            return vals
        red_selection = _checked(self.red_list)
        green_selection = _checked(self.green_list)
        blue_selection = _checked(self.blue_list)
        
        # If only one channel is selected and no RGB assignments are made, assign it to red
        if (len(selected_channels) == 1 and 
            not red_selection and not green_selection and not blue_selection):
            red_selection = selected_channels.copy()
        
        # Create RGB stack based on user selections
        rgb_channels = []
        rgb_titles = []
        raw_channels = []  # Store raw images for colorbar
        
        # Collect all channels that will be used in RGB composite
        all_rgb_channels = list(set(red_selection + green_selection + blue_selection))
        
        # Apply spillover correction if enabled (to all channels used in RGB)
        if self.spillover_correction_enabled and self.spillover_matrix is not None and all_rgb_channels:
            corrected_images = self._apply_spillover_correction_to_channels(self.current_acq_id, all_rgb_channels)
        else:
            corrected_images = {}
        
        # Get the first selected channel to determine image size
        first_img = None
        if selected_channels:
            if selected_channels[0] in corrected_images:
                first_img = corrected_images[selected_channels[0]]
            else:
                first_img = self._load_image_with_normalization(self.current_acq_id, selected_channels[0])
        
        if first_img is None:
            QtWidgets.QMessageBox.information(self, "No RGB channels", "Please select at least one channel for RGB composite.")
            return
        
        # Build R, G, B channels using the selected combination method
        r_img = self._combine_channels_for_rgb(red_selection, first_img, corrected_images)
        g_img = self._combine_channels_for_rgb(green_selection, first_img, corrected_images)
        b_img = self._combine_channels_for_rgb(blue_selection, first_img, corrected_images)

        rgb_channels.append(r_img)
        raw_channels.append(r_img)
        rgb_titles.append(f"{'+'.join(red_selection) if red_selection else 'None'} (Red)")

        rgb_channels.append(g_img)
        raw_channels.append(g_img)
        rgb_titles.append(f"{'+'.join(green_selection) if green_selection else 'None'} (Green)")

        rgb_channels.append(b_img)
        raw_channels.append(b_img)
        rgb_titles.append(f"{'+'.join(blue_selection) if blue_selection else 'None'} (Blue)")
        
        # Ensure we have exactly 3 channels
        while len(rgb_channels) < 3:
            rgb_channels.append(np.zeros_like(first_img))
            raw_channels.append(np.zeros_like(first_img))
            rgb_titles.append(f"None ({['Red', 'Green', 'Blue'][len(rgb_channels)-1]})")
        
        # Apply RGB color custom scaling before stacking (for RGB display)
        if self.custom_scaling_chk.isChecked():
            scaled_channels = []
            color_names = ['Red', 'Green', 'Blue']
            
            for i, ch_img in enumerate(rgb_channels):
                # Skip empty channels (all zeros)
                if np.all(ch_img == 0):
                    scaled_channels.append(ch_img)
                    continue
                
                # Get the color name for this RGB channel
                color_name = color_names[i] if i < len(color_names) else None
                
                # Use RGB color scaling if available
                if color_name and color_name in self.rgb_color_scaling:
                    vmin = self.rgb_color_scaling[color_name]['min']
                    vmax = self.rgb_color_scaling[color_name]['max']
                    if vmax <= vmin:
                        vmax = vmin + 1e-6
                    
                    # Apply scaling to the whole color (combined channels)
                    ch_img = np.clip((ch_img.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
                else:
                    # No custom scaling for this color, normalize to 0-1 based on actual range
                    actual_min = float(np.min(ch_img))
                    actual_max = float(np.max(ch_img))
                    if actual_max > actual_min:
                        ch_img = (ch_img.astype(np.float32) - actual_min) / (actual_max - actual_min)
                    else:
                        ch_img = np.zeros_like(ch_img)
                
                scaled_channels.append(ch_img)
            rgb_channels = scaled_channels

        # Stack channels
        stack = np.dstack(rgb_channels)
        raw_stack = np.dstack(raw_channels)
        
        # Note: Do NOT apply overlay per-channel (will be 3-channel) — apply to final RGB image below
        
        # Get acquisition subtitle
        acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
        title = " + ".join(rgb_titles) + f"\n{acq_subtitle}"
        if self.segmentation_overlay:
            title += " (segmented)"
        
        # Clear canvas and show RGB composite with individual colorbars
        self.canvas.fig.clear()
        
        if grayscale:
            # Grayscale background from assigned channels (mean of non-empty channels)
            ax = self.canvas.fig.add_subplot(111)

            nonzero_channels = [ch for ch in rgb_channels if not np.all(ch == 0)]
            if len(nonzero_channels) == 0:
                gray_base = stack[..., 0]
            elif len(nonzero_channels) == 1:
                gray_base = nonzero_channels[0]
            else:
                gray_base = np.mean(np.dstack(nonzero_channels), axis=2)

            if self.segmentation_overlay:
                # Apply colored overlay on top of grayscale background
                blended = self._get_segmentation_overlay(gray_base)
                ax.imshow(blended, interpolation="nearest")
                ax.set_title(title)
                ax.axis("off")
            else:
                # Show pure grayscale with colorbar
                vmin, vmax = np.min(gray_base), np.max(gray_base)
                im = ax.imshow(gray_base, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
                cbar = self.canvas.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
                ax.set_title(title)
                ax.axis("off")
            
            # Draw scale bar if enabled (for grayscale view)
            if self.scale_bar_chk.isChecked():
                pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
                if pixel_size_um > 0:
                    scale_bar_length_um = self.scale_bar_length_spin.value()
                    self.canvas._draw_scale_bar_on_axes(gray_base.shape, scale_bar_length_um, pixel_size_um, ax)
        else:
            # RGB composite with slimmer individual channel colorbars
            # Create a grid with a much shorter bottom row
            gs = self.canvas.fig.add_gridspec(2, 3, height_ratios=[10, 1], hspace=0.12, wspace=0.2)
            
            # Main RGB composite image (spans top row)
            ax_main = self.canvas.fig.add_subplot(gs[0, :])
            rgb_img = stack_to_rgb(stack)
            if self.segmentation_overlay:
                rgb_img = self._get_segmentation_overlay(rgb_img)
            im = ax_main.imshow(rgb_img, interpolation="nearest")
            ax_main.set_title(title)
            ax_main.axis("off")
            
            # Draw scale bar if enabled (for RGB view)
            if self.scale_bar_chk.isChecked():
                pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
                if pixel_size_um > 0:
                    scale_bar_length_um = self.scale_bar_length_spin.value()
                    # Get image shape from the RGB stack
                    img_shape = stack.shape[:2]  # (height, width)
                    self.canvas._draw_scale_bar_on_axes(img_shape, scale_bar_length_um, pixel_size_um, ax_main)
            
            # Individual channel colorbars (bottom row)
            for i, (channel_name, color_name) in enumerate(zip(rgb_titles, ['Red', 'Green', 'Blue'])):
                ax_cbar = self.canvas.fig.add_subplot(gs[1, i])
                
                # Create a colorbar for this channel
                if i < len(raw_channels) and raw_channels[i] is not None and not np.all(raw_channels[i] == 0):
                    # This channel has data
                    raw_min, raw_max = np.min(raw_channels[i]), np.max(raw_channels[i])
                    
                    # Check for RGB color scaling (in RGB mode)
                    if self.custom_scaling_chk.isChecked() and color_name in self.rgb_color_scaling:
                        raw_min = self.rgb_color_scaling[color_name]['min']
                        raw_max = self.rgb_color_scaling[color_name]['max']
                    
                    if raw_max > raw_min:  # Valid data range
                        # Create a gradient for the colorbar
                        gradient = np.linspace(0, 1, 256).reshape(1, -1)
                        ax_cbar.imshow(gradient, aspect='auto', cmap='gray' if grayscale else ['Reds', 'Greens', 'Blues'][i])
                        ax_cbar.set_xticks([0, 255])
                        ax_cbar.set_xticklabels([f'{raw_min:.1f}', f'{raw_max:.1f}'])
                        ax_cbar.tick_params(axis='x', labelsize=8, pad=1)
                        ax_cbar.set_yticks([])
                        ax_cbar.set_title(f"{channel_name}", fontsize=8, pad=2)
                    else:
                        # No data
                        ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes, fontsize=8)
                        ax_cbar.set_title(f"{channel_name}", fontsize=8, pad=2)
                else:
                    # No data for this channel
                    ax_cbar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_cbar.transAxes, fontsize=8)
                    ax_cbar.set_title(f"{channel_name}", fontsize=8, pad=2)
                
                ax_cbar.set_xlim(0, 255)
                for spine in ax_cbar.spines.values():
                    spine.set_visible(False)
        
        self.canvas.draw()
        
        # Add cluster legend if cluster overlay mode is active
        self._add_cluster_legend()


    # ---------- Comparison ----------
    def _comparison(self):
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a file or folder first.")
            return
        
        # Open the dynamic comparison dialog
        dlg = DynamicComparisonDialog(self.acquisitions, self.loader, self)
        dlg.exec_()

    # ---------- Image Saving ----------
    def _get_suggested_save_filename(self):
        """Generate a suggested filename for saving images based on acquisition and channels."""
        if not self.current_acq_id or not hasattr(self, 'current_path') or not self.current_path:
            return "figure.png"
            
        try:
            import re
            
            # Get filename without extension (handle both files and directories)
            if os.path.isdir(self.current_path):
                base_filename = os.path.basename(self.current_path) or os.path.basename(os.path.dirname(self.current_path))
            else:
                base_filename = os.path.splitext(os.path.basename(self.current_path))[0]
            
            # Get acquisition descriptor (subtitle) - this is the ROI
            acquisition_descriptor = self._get_acquisition_subtitle(self.current_acq_id)
            
            # Check if we're in RGB mode (grid view is off)
            is_rgb_mode = hasattr(self, 'grid_view_chk') and not self.grid_view_chk.isChecked()
            
            if is_rgb_mode and hasattr(self, 'red_list') and hasattr(self, 'green_list') and hasattr(self, 'blue_list'):
                # RGB mode: use format slide_roi_Red_channel(s)_Green_channel(s)_Blue_channel(s)
                def _checked(lst: QtWidgets.QListWidget) -> List[str]:
                    vals: List[str] = []
                    for i in range(lst.count()):
                        item = lst.item(i)
                        if item.checkState() == Qt.Checked:
                            vals.append(item.text())
                    return vals
                
                red_selection = _checked(self.red_list)
                green_selection = _checked(self.green_list)
                blue_selection = _checked(self.blue_list)
                
                # Build filename parts
                parts = [base_filename, acquisition_descriptor]
                
                # Add Red channel(s) if any are selected
                if red_selection:
                    # Sanitize channel names and join with underscores
                    red_str = "_".join([re.sub(r'[<>:"/\\|?*]', '_', ch) for ch in red_selection])
                    parts.append(f"Red_{red_str}")
                
                # Add Green channel(s) if any are selected
                if green_selection:
                    green_str = "_".join([re.sub(r'[<>:"/\\|?*]', '_', ch) for ch in green_selection])
                    parts.append(f"Green_{green_str}")
                
                # Add Blue channel(s) if any are selected
                if blue_selection:
                    blue_str = "_".join([re.sub(r'[<>:"/\\|?*]', '_', ch) for ch in blue_selection])
                    parts.append(f"Blue_{blue_str}")
                
                filename = "_".join(parts) + ".png"
            else:
                # Non-RGB mode: use original format
                selected_channels = self._selected_channels()
                
                # Create filename: filename_acquisition_descriptor_channels.png
                if selected_channels:
                    channels_str = "_".join(selected_channels)
                    filename = f"{base_filename}_{acquisition_descriptor}_{channels_str}.png"
                else:
                    filename = f"{base_filename}_{acquisition_descriptor}.png"
            
            # Clean filename (remove invalid characters) - do this after all joins
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            return filename
            
        except (StopIteration, AttributeError):
            return "figure.png"
    
    def get_save_filename(self):
        """Generate a custom filename for saving images based on acquisition and channels."""
        # This method is kept for backward compatibility but now uses the new dialog
        suggested_filename = self._get_suggested_save_filename()
        if self.canvas and self.canvas.figure:
            save_figure_with_options(
                self.canvas.figure,
                suggested_filename,
                self
            )
        return None

    # ---------- Export ----------
    def _export_ome_tiff(self):
        """Export acquisitions to OME-TIFF format."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a file or folder first.")
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
        
        # Get denoising and normalization settings
        denoise_source = dlg.get_denoise_source()
        custom_denoise_settings = dlg.get_custom_denoise_settings()
        normalization_method = dlg.get_normalization_method()
        arcsinh_cofactor = dlg.get_arcsinh_cofactor()
        percentile_params = dlg.get_percentile_params()
        
        # Create and show progress dialog
        progress_dlg = ProgressDialog("Export to OME-TIFF", self)
        progress_dlg.show()
        
        try:
            if export_type == "single":
                success = self._export_single_acquisition(
                    output_dir, include_metadata, progress_dlg,
                    denoise_source, custom_denoise_settings,
                    normalization_method, arcsinh_cofactor, percentile_params
                )
            else:
                success = self._export_whole_slide(
                    output_dir, include_metadata, progress_dlg,
                    denoise_source, custom_denoise_settings,
                    normalization_method, arcsinh_cofactor, percentile_params
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
                                 progress_dlg: ProgressDialog,
                                 denoise_source: str, custom_denoise_settings: dict,
                                 normalization_method: str, arcsinh_cofactor: float,
                                 percentile_params: Tuple[float, float]) -> bool:
        """Export the currently selected acquisition."""
        if not self.current_acq_id:
            raise ValueError("No acquisition selected")
        
        acq_info = self._get_acquisition_info(self.current_acq_id)
        if acq_info is None:
            raise ValueError(f"Acquisition {self.current_acq_id} not found")
        
        # Get all channels for this acquisition
        loader = self._get_loader_for_acquisition(self.current_acq_id)
        if loader is None:
            raise ValueError(f"No loader found for acquisition {self.current_acq_id}")
        # Get original acquisition ID if this is a unique ID
        original_acq_id = self._get_original_acq_id(self.current_acq_id)
        all_channels = loader.get_channels(original_acq_id)
        if not all_channels:
            raise ValueError("No channels found for this acquisition")
        
        progress_dlg.set_maximum(len(all_channels) + 3)  # +3 for loading, processing, stacking, writing
        progress_dlg.update_progress(0, f"Exporting {acq_info.name}", "Loading channels...")
        
        # Load all raw channel data first (sequential to avoid memory issues)
        raw_channel_data = []
        channel_names = []
        
        for i, channel in enumerate(all_channels):
            if progress_dlg.is_cancelled():
                return False
                
            progress_dlg.update_progress(
                i + 1, 
                f"Exporting {acq_info.name}", 
                f"Loading channel {i+1}/{len(all_channels)}: {channel}"
            )
            
            # Load raw image
            img = loader.get_image(original_acq_id, channel)
            raw_channel_data.append(img)
            channel_names.append(channel)
        
        if progress_dlg.is_cancelled():
            return False
        
        # Process channels in parallel using multiprocessing
        progress_dlg.update_progress(
            len(all_channels) + 1,
            f"Exporting {acq_info.name}",
            "Processing channels with denoising and normalization..."
        )
        
        # Apply viewer denoising first if needed (cannot be pickled)
        if denoise_source == "viewer":
            for i, (channel, img) in enumerate(zip(channel_names, raw_channel_data)):
                raw_channel_data[i] = self._apply_denoise(channel, img)
        
        # Use multiprocessing for custom denoising and normalization
        channel_data = []
        max_workers = min(mp.cpu_count(), len(all_channels))
        
        try:
            with mp.Pool(processes=max_workers) as pool:
                # Submit all channel processing tasks
                futures = []
                for channel, img in zip(channel_names, raw_channel_data):
                    # Skip denoising if already done (viewer denoising)
                    effective_denoise_source = "none" if denoise_source == "viewer" else denoise_source
                    future = pool.apply_async(
                        process_channel_for_export,
                        (img, channel, effective_denoise_source, custom_denoise_settings,
                         normalization_method, arcsinh_cofactor, percentile_params, None)
                    )
                    futures.append((channel, future))
                
                # Collect results
                for i, (channel, future) in enumerate(futures):
                    if progress_dlg.is_cancelled():
                        pool.terminate()
                        return False
                    try:
                        result = future.get(timeout=300)  # 5 minute timeout per channel
                        channel_data.append(result)
                    except Exception as e:
                        print(f"Channel processing failed for {channel}: {e}")
                        # Fallback to unprocessed image
                        channel_data.append(raw_channel_data[i])
                        
        except Exception as mp_error:
            print(f"Multiprocessing failed, falling back to sequential processing: {mp_error}")
            # Fallback to sequential processing
            channel_data = []
            for i, (channel, img) in enumerate(zip(channel_names, raw_channel_data)):
                if progress_dlg.is_cancelled():
                    return False
                try:
                    effective_denoise_source = "none" if denoise_source == "viewer" else denoise_source
                    result = process_channel_for_export(
                        img, channel, effective_denoise_source, custom_denoise_settings,
                        normalization_method, arcsinh_cofactor, percentile_params, None
                    )
                    channel_data.append(result)
                except Exception as e:
                    print(f"Channel processing failed for {channel}: {e}")
                    channel_data.append(img)  # Use unprocessed image as fallback
        
        if progress_dlg.is_cancelled():
            return False
        
        # Stack channels (C, H, W) for OME-TIFF
        progress_dlg.update_progress(
            len(all_channels) + 2, 
            f"Exporting {acq_info.name}", 
            "Stacking channels..."
        )
        stack = np.stack(channel_data, axis=0)
        
        # Create filename from source_file and acquisition ID
        # Get original acquisition ID (not the unique ID with __file_ prefix)
        original_acq_id_for_filename = self._get_original_acq_id(self.current_acq_id)
        
        # Extract source file basename (without extension)
        if acq_info.source_file:
            source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
            safe_source = self._sanitize_filename(source_basename)
            safe_acq_id = self._sanitize_filename(original_acq_id_for_filename)
            filename = f"{safe_source}_{safe_acq_id}.tif"
        else:
            # Fallback if no source_file available
            safe_acq_id = self._sanitize_filename(original_acq_id_for_filename)
            filename = f"{safe_acq_id}.tif"
        
        output_path = os.path.join(output_dir, filename)
        
        # Prepare comprehensive metadata
        metadata = self._create_ome_metadata(
            acq_info, channel_names, include_metadata, stack.shape
        )
        
        # Extract pixel size from metadata if available
        pixel_size = metadata.get('PhysicalSizeX')
        pixel_size_unit = metadata.get('PhysicalSizeXUnit', 'µm')
        
        # Create OME-XML with channel names
        ome_xml = self._create_ome_xml(
            channel_names, stack.shape, pixel_size, pixel_size_unit
        )
        
        # Write OME-TIFF
        progress_dlg.update_progress(
            len(all_channels) + 3, 
            f"Exporting {acq_info.name}", 
            f"Writing {filename}..."
        )
        
        if progress_dlg.is_cancelled():
            return False
        
        # Write OME-TIFF with channel names in metadata
        # tifffile generates OME-XML automatically when ome=True, but we need to modify it
        # to include channel names. We'll write the file and then modify the OME-XML.
        
        # Write to temporary file first
        temp_path = output_path + '.tmp'
        try:
            tifffile.imwrite(
                temp_path,
                stack,
                imagej=True,
                metadata=metadata,
                ome=True,
                photometric='minisblack'
            )
            
            # Now modify the OME-XML to include channel names
            self._add_channel_names_to_ometiff(temp_path, output_path, channel_names, stack.dtype)
            
            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            # Fallback: write without OME-XML modification
            print(f"Warning: Could not modify OME-XML ({e}), writing with basic OME-XML")
            tifffile.imwrite(
                output_path,
                stack,
                imagej=True,
                metadata=metadata,
                ome=True,
                photometric='minisblack'
            )
        
        return True
    
    def _export_whole_slide(self, output_dir: str, include_metadata: bool, 
                          progress_dlg: ProgressDialog,
                          denoise_source: str, custom_denoise_settings: dict,
                          normalization_method: str, arcsinh_cofactor: float,
                          percentile_params: Tuple[float, float]) -> bool:
        """Export all acquisitions from the slide."""
        total_acquisitions = len(self.acquisitions)
        
        # Process each acquisition
        for acq_idx, acq_info in enumerate(self.acquisitions):
            if progress_dlg.is_cancelled():
                return False
            
            progress_dlg.update_progress(
                acq_idx, 
                f"Exporting acquisition {acq_idx + 1}/{total_acquisitions}", 
                f"Processing {acq_info.name}..."
            )
            
            # Get all channels for this acquisition
            loader = self._get_loader_for_acquisition(acq_info.id)
            if loader is None:
                print(f"Warning: No loader found for acquisition {acq_info.name}")
                continue
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_info.id)
            all_channels = loader.get_channels(original_acq_id)
            if not all_channels:
                print(f"Warning: No channels found for acquisition {acq_info.name}")
                continue
            
            # Load all raw channel data first (sequential to avoid memory issues)
            raw_channel_data = []
            channel_names = []
            
            for channel in all_channels:
                if progress_dlg.is_cancelled():
                    return False
                    
                # Load raw image
                img = loader.get_image(original_acq_id, channel)
                raw_channel_data.append(img)
                channel_names.append(channel)
            
            if progress_dlg.is_cancelled():
                return False
            
            # Apply viewer denoising first if needed (cannot be pickled)
            if denoise_source == "viewer":
                for i, (channel, img) in enumerate(zip(channel_names, raw_channel_data)):
                    raw_channel_data[i] = self._apply_denoise(channel, img)
            
            # Use multiprocessing for custom denoising and normalization
            channel_data = []
            max_workers = min(mp.cpu_count(), len(all_channels))
            
            try:
                with mp.Pool(processes=max_workers) as pool:
                    # Submit all channel processing tasks
                    futures = []
                    for channel, img in zip(channel_names, raw_channel_data):
                        # Skip denoising if already done (viewer denoising)
                        effective_denoise_source = "none" if denoise_source == "viewer" else denoise_source
                        future = pool.apply_async(
                            process_channel_for_export,
                            (img, channel, effective_denoise_source, custom_denoise_settings,
                             normalization_method, arcsinh_cofactor, percentile_params, None)
                        )
                        futures.append((channel, future))
                    
                    # Collect results
                    for i, (channel, future) in enumerate(futures):
                        if progress_dlg.is_cancelled():
                            pool.terminate()
                            return False
                        try:
                            result = future.get(timeout=300)  # 5 minute timeout per channel
                            channel_data.append(result)
                        except Exception as e:
                            print(f"Channel processing failed for {channel}: {e}")
                            # Fallback to unprocessed image
                            channel_data.append(raw_channel_data[i])
                            
            except Exception as mp_error:
                print(f"Multiprocessing failed, falling back to sequential processing: {mp_error}")
                # Fallback to sequential processing
                channel_data = []
                for i, (channel, img) in enumerate(zip(channel_names, raw_channel_data)):
                    if progress_dlg.is_cancelled():
                        return False
                    try:
                        effective_denoise_source = "none" if denoise_source == "viewer" else denoise_source
                        result = process_channel_for_export(
                            img, channel, effective_denoise_source, custom_denoise_settings,
                            normalization_method, arcsinh_cofactor, percentile_params, None
                        )
                        channel_data.append(result)
                    except Exception as e:
                        print(f"Channel processing failed for {channel}: {e}")
                        channel_data.append(img)  # Use unprocessed image as fallback
            
            if progress_dlg.is_cancelled():
                return False
            
            # Stack channels (C, H, W) for OME-TIFF
            stack = np.stack(channel_data, axis=0)
            
            # Create filename from source_file and well name (or acquisition name if well doesn't exist)
            # Use well name if available, otherwise use acquisition name
            if acq_info.well:
                label_for_filename = acq_info.well
            else:
                # Get original acquisition ID (not the unique ID with __file_ prefix)
                original_acq_id_for_filename = self._get_original_acq_id(acq_info.id)
                label_for_filename = original_acq_id_for_filename
            
            # Extract source file basename (without extension)
            if acq_info.source_file:
                source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
                safe_source = self._sanitize_filename(source_basename)
                safe_label = self._sanitize_filename(label_for_filename)
                filename = f"{safe_source}_{safe_label}.ome.tiff"
            else:
                # Fallback if no source_file available
                safe_label = self._sanitize_filename(label_for_filename)
                filename = f"{safe_label}.ome.tiff"
            
            output_path = os.path.join(output_dir, filename)
            
            # Prepare comprehensive metadata
            metadata = self._create_ome_metadata(
                acq_info, channel_names, include_metadata, stack.shape
            )
            
            # Extract pixel size from metadata if available
            pixel_size = metadata.get('PhysicalSizeX')
            pixel_size_unit = metadata.get('PhysicalSizeXUnit', 'µm')
            
            # Create OME-XML with channel names
            ome_xml = self._create_ome_xml(
                channel_names, stack.shape, pixel_size, pixel_size_unit
            )
            
            # Write OME-TIFF
            progress_dlg.update_progress(
                acq_idx + 1, 
                f"Exporting acquisition {acq_idx + 1}/{total_acquisitions}", 
                f"Writing {filename}..."
            )
            
            if progress_dlg.is_cancelled():
                return False
            
            # Write OME-TIFF with channel names in metadata
            # tifffile generates OME-XML automatically when ome=True, but we need to modify it
            # to include channel names. We'll write the file and then modify the OME-XML.
            
            # Write to temporary file first
            temp_path = output_path + '.tmp'
            try:
                tifffile.imwrite(
                    temp_path,
                    stack,
                    imagej=True,
                    metadata=metadata,
                    ome=True,
                    photometric='minisblack'
                )
                
                # Now modify the OME-XML to include channel names
                self._add_channel_names_to_ometiff(temp_path, output_path, channel_names, stack.dtype)
                
                # Remove temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                # Fallback: write without OME-XML modification
                print(f"Warning: Could not modify OME-XML for {acq_info.name} ({e}), writing with basic OME-XML")
                tifffile.imwrite(
                    output_path,
                    stack,
                    imagej=True,
                    metadata=metadata,
                    ome=True,
                    photometric='minisblack'
                )
        
        return True
    
    def _apply_export_denoising(self, img: np.ndarray, channel: str, 
                                 denoise_source: str, custom_denoise_settings: dict) -> np.ndarray:
        """Apply denoising to an image for export."""
        if denoise_source == "viewer":
            # Use viewer denoising (same as display)
            return self._apply_denoise(channel, img)
        elif denoise_source == "custom":
            # Use custom denoising settings
            return self._apply_custom_denoise(channel, img, custom_denoise_settings)
        else:
            # No denoising
            return img
    
    def _apply_export_normalization(self, img: np.ndarray, 
                                   normalization_method: str, 
                                   arcsinh_cofactor: float,
                                   percentile_params: Tuple[float, float]) -> np.ndarray:
        """Apply normalization to an image for export."""
        if normalization_method == "channelwise_minmax":
            return channelwise_minmax_normalize(img)
        elif normalization_method == "arcsinh":
            return arcsinh_normalize(img, cofactor=arcsinh_cofactor)
        elif normalization_method == "percentile_clip":
            p_low, p_high = percentile_params
            return percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
        else:
            # No normalization
            return img
    
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
        
        # Channel information - ensure channel names are included
        metadata['ChannelNames'] = channel_names
        
        # Get detailed channel information from acquisition
        acq_id = acq_info.id
        channel_metals = []
        channel_labels = []
        
        loader = self._get_loader_for_acquisition(acq_id)
        if loader and hasattr(loader, '_acq_channel_metals') and acq_id in loader._acq_channel_metals:
            channel_metals = loader._acq_channel_metals[acq_id]
            channel_labels = loader._acq_channel_labels[acq_id]
        
        # Ensure we have the same number of metals/labels as channels
        while len(channel_metals) < len(channel_names):
            channel_metals.append("")
        while len(channel_labels) < len(channel_names):
            channel_labels.append("")
        
        # Create detailed channel metadata
        channel_metadata = []
        for i, name in enumerate(channel_names):
            metal = channel_metals[i] if i < len(channel_metals) else ""
            label = channel_labels[i] if i < len(channel_labels) else ""
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
    
    def _create_ome_xml(self, channel_names: List[str], stack_shape: Tuple[int, ...], 
                       pixel_size: Optional[float] = None, pixel_size_unit: str = "µm") -> str:
        """Create OME-XML string with proper channel names embedded.
        
        This ensures channel names are properly included in the OME-XML metadata
        that tifffile will embed in the OME-TIFF file.
        """
        import xml.etree.ElementTree as ET
        
        # Create OME root element
        ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
        
        # Create root element with proper namespace handling
        root = ET.Element("OME", attrib={
            "xmlns": ome_ns,
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": f"{ome_ns} {ome_ns}/ome.xsd"
        })
        
        # Create Image element
        image = ET.SubElement(root, "Image")
        image.set("ID", "Image:0")
        image.set("Name", "Image")
        
        # Create Pixels element
        if len(stack_shape) >= 3:
            size_c = stack_shape[0]
            size_y = stack_shape[1]
            size_x = stack_shape[2]
        else:
            size_c = 1
            size_y = stack_shape[0] if len(stack_shape) > 0 else 1
            size_x = stack_shape[1] if len(stack_shape) > 1 else 1
        
        pixels = ET.SubElement(image, "Pixels")
        pixels.set("ID", "Pixels:0")
        pixels.set("Type", "uint16")  # Default type, may vary
        pixels.set("SizeX", str(size_x))
        pixels.set("SizeY", str(size_y))
        pixels.set("SizeZ", "1")
        pixels.set("SizeC", str(size_c))
        pixels.set("SizeT", "1")
        pixels.set("DimensionOrder", "XYZCT")
        
        if pixel_size is not None:
            pixels.set("PhysicalSizeX", str(pixel_size))
            pixels.set("PhysicalSizeY", str(pixel_size))
            pixels.set("PhysicalSizeXUnit", pixel_size_unit)
            pixels.set("PhysicalSizeYUnit", pixel_size_unit)
        
        # Create Channel elements with names - this is critical for channel name preservation
        for i, channel_name in enumerate(channel_names):
            channel = ET.SubElement(pixels, "Channel")
            channel.set("ID", f"Channel:{i}:0")
            # Set Name attribute directly on Channel element (OME standard)
            channel.set("Name", channel_name)
            channel.set("SamplesPerPixel", "1")
        
        # Add TiffData element for the image data (required by OME spec)
        tiffdata = ET.SubElement(pixels, "TiffData")
        tiffdata.set("IFD", "0")
        tiffdata.set("PlaneCount", "1")
        
        # Convert to string - use method='xml' to get proper formatting
        # Remove XML declaration since tifffile may add its own
        xml_string = ET.tostring(root, encoding='utf-8', method='xml')
        # Remove XML declaration if present
        if xml_string.startswith(b'<?xml'):
            # Find the end of the XML declaration
            decl_end = xml_string.find(b'>') + 1
            xml_string = xml_string[decl_end:].lstrip()
        
        return xml_string.decode('utf-8')
    
    def _add_channel_names_to_ometiff(self, input_path: str, output_path: str, 
                                       channel_names: List[str], dtype: type):
        """Modify an existing OME-TIFF file to add channel names to the OME-XML.
        
        This function reads the OME-XML from an existing OME-TIFF file, adds channel
        names to the Channel elements, and writes a new file with the modified OME-XML.
        """
        import xml.etree.ElementTree as ET
        
        # Read the existing file
        with tifffile.TiffFile(input_path) as tif:
            # Get the image data
            img_data = tif.asarray()
            
            # Get existing OME-XML
            ome_xml = tif.ome_metadata
            if not ome_xml:
                # If no OME-XML exists, create basic one
                raise ValueError("No OME-XML found in file")
            
            # Parse the OME-XML
            ome_ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
            root = ET.fromstring(ome_xml)
            
            # Find all Channel elements and update them with names
            channels = root.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel')
            if len(channels) != len(channel_names):
                # If channel count doesn't match, create new channels
                # Find Pixels element
                pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                if pixels is not None:
                    # Remove existing channels
                    for ch in channels:
                        pixels.remove(ch)
                    # Create new channels with names
                    for i, channel_name in enumerate(channel_names):
                        channel = ET.SubElement(pixels, f"{{{ome_ns}}}Channel")
                        channel.set("ID", f"Channel:{i}:0")
                        channel.set("Name", channel_name)
                        channel.set("SamplesPerPixel", "1")
            else:
                # Update existing channels with names
                for i, (channel, channel_name) in enumerate(zip(channels, channel_names)):
                    channel.set("Name", channel_name)
                    if channel.get("ID", "").startswith("Channel:"):
                        # Ensure ID is correct
                        channel.set("ID", f"Channel:{i}:0")
            
            # Convert back to XML string
            modified_xml = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
            
            # Write new file with modified OME-XML in the description tag
            # OME-TIFF is just TIFF with OME-XML in ImageDescription tag
            # We write without ome=True to avoid the warning, but include valid OME-XML
            tifffile.imwrite(
                output_path,
                img_data,
                photometric='minisblack',
                description=modified_xml,  # OME-XML goes in ImageDescription tag
                # This creates a valid OME-TIFF file
            )
    
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
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a file or folder first.")
            return
        
        # Check dependencies - will be checked again after dialog if needed
        
        if not self.current_acq_id:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        
        # Get available channels
        loader = self._get_loader_for_acquisition(self.current_acq_id)
        if loader is None:
            QtWidgets.QMessageBox.critical(self, "Loader error", "No loader found for current acquisition.")
            return
        # Get original acquisition ID if this is a unique ID
        original_acq_id = self._get_original_acq_id(self.current_acq_id)
        channels = loader.get_channels(original_acq_id)
        if not channels:
            QtWidgets.QMessageBox.information(self, "No channels", "No channels available for segmentation.")
            return
        
        # Check if dialog already exists and is still valid
        if self.segmentation_dialog is not None:
            try:
                # Check if dialog still exists (hasn't been deleted)
                # Check if the current file path has changed - if so, reset the dialog
                current_file_path = self.current_path
                dialog_file_path = getattr(self.segmentation_dialog, '_last_file_path', None)
                
                # If file path changed, reset the dialog to clear old channel preferences
                if current_file_path != dialog_file_path:
                    # Clear old preferences and reset dialog
                    if hasattr(self.segmentation_dialog, '_per_file_channel_prefs'):
                        self.segmentation_dialog._per_file_channel_prefs.clear()
                    # Update channels in the dialog
                    self.segmentation_dialog.channels = channels
                    self.segmentation_dialog._populate_denoise_channel_list()
                    # Store the new file path
                    self.segmentation_dialog._last_file_path = current_file_path
                    # Reload persisted selections for the new file
                    self.segmentation_dialog._load_persisted_selections()
                
                # If dialog is visible, just bring it to front
                # If dialog is not visible, show it
                if self.segmentation_dialog.isVisible():
                    self.segmentation_dialog.raise_()
                    self.segmentation_dialog.activateWindow()
                else:
                    self.segmentation_dialog.show()
                    self.segmentation_dialog.raise_()
                    self.segmentation_dialog.activateWindow()
                # Update channels if they've changed (e.g., user switched acquisitions)
                # The dialog will handle channel updates through its persistence mechanism
                return
            except (RuntimeError, AttributeError):
                # Dialog was deleted, set to None
                self.segmentation_dialog = None
        
        # Create new segmentation dialog
        self.segmentation_dialog = SegmentationDialog(channels, self)
        # Store the current file path in the dialog for tracking file switches
        self.segmentation_dialog._last_file_path = self.current_path
        # Initialize with current viewer denoising toggle state
        try:
            self.segmentation_dialog.set_use_viewer_denoising(self.denoise_enable_chk.isChecked())
        except Exception:
            pass
        # Make dialog non-modal so it can be closed and reopened without losing state
        self.segmentation_dialog.setModal(False)
        # Prevent dialog from being deleted when closed, so we can reopen it
        self.segmentation_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        self.segmentation_dialog.show()
        
        # Connect to accepted signal to handle segmentation when user clicks "Run Segmentation"
        # Only connect if not already connected (to avoid duplicate connections)
        if not hasattr(self.segmentation_dialog, '_segmentation_accepted_connected'):
            def on_segmentation_accepted():
                """Handle segmentation when dialog is accepted."""
                dlg = self.segmentation_dialog
                if dlg is None:
                    return
                
                # Get segmentation parameters
                model = dlg.get_model()
                
                # Handle Ilastik segmentation
                if model == "Ilastik":
                    self._run_ilastik_segmentation(dlg)
                    return
                
                # Check dependencies based on selected model
                if model == "DeepCell CellSAM":
                    if not _HAVE_CELLSAM:
                        QtWidgets.QMessageBox.critical(
                            self, "Missing dependency", 
                            "CellSAM library is required for segmentation.\n"
                            "Install it with: pip install git+https://github.com/vanvalenlab/cellSAM.git"
                        )
                        return
                elif model != "Classical Watershed" and not _HAVE_CELLPOSE:
                    QtWidgets.QMessageBox.critical(
                        self, "Missing dependency", 
                        "Cellpose library is required for segmentation.\n"
                        "Install it with: pip install cellpose"
                    )
                    return
                
                diameter = dlg.get_diameter()
                flow_threshold = dlg.get_flow_threshold()
                cellprob_threshold = dlg.get_cellprob_threshold()
                show_overlay = dlg.get_show_overlay()
                save_masks = dlg.get_save_masks()
                masks_directory = dlg.get_masks_directory()
                gpu_id = dlg.get_selected_gpu()
                preprocessing_config = dlg.get_preprocessing_config()
                use_viewer_denoising = dlg.get_use_viewer_denoising()
                segment_all = dlg.get_segment_all()
                
                # Get denoising parameters
                denoise_source = dlg.get_denoise_source()
                custom_denoise_settings = dlg.get_custom_denoise_settings()
                
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
                
                if model == "cyto3" and not cyto_channels:
                    QtWidgets.QMessageBox.warning(self, "No cytoplasm channels", "Please select at least one cytoplasm channel in the preprocessing configuration for whole-cell segmentation.")
                    return
                
                if model == "Classical Watershed" and not cyto_channels:
                    QtWidgets.QMessageBox.warning(self, "No membrane channels", "Please select at least one membrane/cytoplasm channel in the preprocessing configuration for watershed segmentation.")
                    return
                
                # For DeepCell CellSAM, at least one channel (nuclear or cyto) must be selected
                if model == "DeepCell CellSAM":
                    if not nuclear_channels and not cyto_channels:
                        QtWidgets.QMessageBox.warning(self, "No channels selected", "Please select at least one nuclear or cytoplasm channel in the preprocessing configuration for CellSAM segmentation.")
                        return
                
                try:
                    if segment_all:
                        # Run segmentation on all acquisitions
                        self._perform_segmentation_all_acquisitions(
                            model, diameter, flow_threshold, cellprob_threshold, 
                            show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config,
                            denoise_source, custom_denoise_settings, dlg
                        )
                    else:
                        # Run segmentation on current acquisition only
                        self._perform_segmentation(
                            model, diameter, flow_threshold, cellprob_threshold, 
                            show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config, use_viewer_denoising,
                            denoise_source, custom_denoise_settings, dlg
                        )
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, "Segmentation Failed", 
                        f"Segmentation failed with error:\n{str(e)}"
                    )
            
            # Connect to accepted signal
            self.segmentation_dialog.accepted.connect(on_segmentation_accepted)
            # Mark as connected to avoid duplicate connections
            self.segmentation_dialog._segmentation_accepted_connected = True
        
        # Show dialog (non-modal, so it can be closed and reopened)
        # User can click "Run Segmentation" to trigger segmentation, or close the dialog
        # The dialog state (including channel selections) will be preserved
    
    def _run_ilastik_segmentation(self, seg_dlg):
        """Run Ilastik segmentation."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a file or folder first.")
            return
        
        if not self.current_acq_id:
            QtWidgets.QMessageBox.information(self, "No acquisition", "Select an acquisition first.")
            return
        
        # Get available channels
        loader = self._get_loader_for_acquisition(self.current_acq_id)
        if loader is None:
            QtWidgets.QMessageBox.critical(self, "Loader error", "No loader found for current acquisition.")
            return
        # Get original acquisition ID if this is a unique ID
        original_acq_id = self._get_original_acq_id(self.current_acq_id)
        channels = loader.get_channels(original_acq_id)
        if not channels:
            QtWidgets.QMessageBox.information(self, "No channels", "No channels available for segmentation.")
            return
        
        try:
            # Load image stack
            img_stack = loader.get_all_channels(original_acq_id)
            
            # Get preprocessing config from segmentation dialog (optional for Ilastik)
            preprocessing_config = seg_dlg.get_preprocessing_config()
            
            # Open Ilastik segmentation dialog
            dlg = IlastikSegmentationDialog(img_stack, channels, self, preprocessing_config)
            if dlg.exec_() == QtWidgets.QDialog.Accepted:
                # Get results
                results = dlg.get_results()
                
                if not results:
                    return
                
                # Store segmentation results
                if not hasattr(self, 'segmentation_masks'):
                    self.segmentation_masks = {}
                
                # Get labels from results
                if 'labels' in results:
                    labels = results['labels']
                    self.segmentation_masks[self.current_acq_id] = labels
                    
                    # Log segmentation operation
                    logger = get_logger()
                    n_cells = len(np.unique(labels)) - 1  # Exclude background
                    
                    # Get project path
                    project_path = dlg.get_project_path()
                    
                    # Get preprocessing config if available
                    nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                    cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
                    
                    params = {
                        "method": "ilastik",
                        "project_path": project_path,
                        "nuclear_channels": nuclear_channels,
                        "cyto_channels": cyto_channels,
                        "n_cells": int(n_cells)
                    }
                    
                    # Add denoising info if available
                    denoise_source = seg_dlg.get_denoise_source()
                    custom_denoise_settings = seg_dlg.get_custom_denoise_settings()
                    if denoise_source == "custom" and custom_denoise_settings:
                        params["denoise_settings"] = custom_denoise_settings
                    else:
                        params["denoise_source"] = denoise_source
                    
                    # Get source file name
                    source_file = os.path.basename(self.current_path) if self.current_path else None
                    
                    logger.log_segmentation(
                        method="ilastik",
                        parameters=params,
                        acquisitions=[self.current_acq_id],
                        notes=f"Ilastik segmented {n_cells} cells/regions",
                        source_file=source_file
                    )
                    
                    # Show overlay
                    self._show_segmentation_overlay(labels)
                    
                    QtWidgets.QMessageBox.information(
                        self, 
                        "Segmentation Complete", 
                        f"Ilastik segmentation completed successfully!\n"
                        f"Found {n_cells} cells/regions"
                    )
                elif 'probabilities' in results:
                    # If only probabilities were returned, convert to labels
                    # Take the class with maximum probability
                    prob_maps = results['probabilities']
                    if isinstance(prob_maps, dict):
                        # Stack probabilities and get argmax
                        prob_arrays = list(prob_maps.values())
                        prob_stack = np.stack(prob_arrays, axis=-1)
                        labels = np.argmax(prob_stack, axis=-1).astype(np.int32)
                        self.segmentation_masks[self.current_acq_id] = labels
                        
                        # Show overlay
                        self._show_segmentation_overlay(labels)
                        
                        QtWidgets.QMessageBox.information(
                            self, 
                            "Segmentation Complete", 
                            "Ilastik segmentation completed successfully!"
                        )
        
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ilastik Segmentation Failed", 
                f"Ilastik segmentation failed with error:\n{str(e)}"
            )
    
    def _perform_segmentation(self, model: str, diameter: int = None, flow_threshold: float = 0.4, 
                            cellprob_threshold: float = 0.0, show_overlay: bool = True, 
                            save_masks: bool = False, masks_directory: str = None, gpu_id = None, preprocessing_config = None, use_viewer_denoising: bool = False,
                            denoise_source: str = "Use viewer settings", custom_denoise_settings: dict = None, dlg = None):
        """Perform the actual segmentation using Cellpose, CellSAM, or Watershed."""
        # Create progress dialog
        progress_dlg = ProgressDialog("Cell Segmentation", self)
        progress_dlg.show()
        
        try:
            if model == "DeepCell CellSAM":
                progress_dlg.update_progress(0, "Initializing DeepCell CellSAM model", "Loading model...")
            else:
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
            
            # Initialize model based on type
            if model == "Classical Watershed":
                model_obj = None  # Watershed doesn't use Cellpose model
            elif model == "DeepCell CellSAM":
                model_obj = None  # CellSAM model is initialized separately
                # Set API key from dialog
                if dlg is not None:
                    api_key = dlg.get_cellsam_api_key()
                    if api_key:
                        os.environ.update({"DEEPCELL_ACCESS_TOKEN": api_key})
                    elif not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
                        QtWidgets.QMessageBox.critical(
                            self, "Missing API Key",
                            "DeepCell API key is required for CellSAM.\n"
                            "Please enter your API key in the CellSAM Parameters section.\n"
                            "Get your key from https://users.deepcell.org/login/"
                        )
                        progress_dlg.close()
                        return
                else:
                    if not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
                        QtWidgets.QMessageBox.critical(
                            self, "Missing API Key",
                            "DeepCell API key is required for CellSAM.\n"
                            "Please set DEEPCELL_ACCESS_TOKEN environment variable or enter it in the dialog."
                        )
                        progress_dlg.close()
                        return
                
                # Initialize CellSAM model and download weights
                progress_dlg.update_progress(5, "Initializing CellSAM model", "Downloading model weights...")
                try:
                    get_model()  # This downloads weights if not already present
                except Exception as e:
                    QtWidgets.QMessageBox.critical(
                        self, "CellSAM Initialization Failed",
                        f"Failed to initialize CellSAM model:\n{str(e)}\n\n"
                        "Please check your API key and internet connection."
                    )
                    progress_dlg.close()
                    return
                progress_dlg.update_progress(10, "CellSAM model ready", "Model initialized successfully")
            elif model == "nuclei":
                model_obj = models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Set device if using GPU
            if model == "Classical Watershed":
                progress_dlg.update_progress(5, "Initializing watershed segmentation", "Using classical watershed algorithm...")
            elif model == "DeepCell CellSAM":
                # Progress already updated above
                pass
            elif use_gpu and gpu_device is not None:
                if gpu_device == 'mps':
                    progress_dlg.update_progress(5, "Initializing Cellpose model", "Using Apple Metal Performance Shaders...")
                else:
                    progress_dlg.update_progress(5, "Initializing Cellpose model", f"Using CUDA GPU {gpu_device}...")
            else:
                progress_dlg.update_progress(5, "Initializing Cellpose model", "Using CPU...")
            
            progress_dlg.update_progress(20, "Preprocessing images", "Loading and preprocessing channels...")
            
            # Preprocess and combine channels
            nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                preprocessing_config, progress_dlg, use_viewer_denoising, denoise_source, custom_denoise_settings
            )
            
            # Prepare input images
            if model == "DeepCell CellSAM":
                # DeepCell CellSAM supports three modes:
                # 1. Nuclear only: H x W array
                # 2. Cyto only: H x W array
                # 3. Combined: H x W x C array where channel 0 is blank, channel 1 is nuclear, channel 2 is cyto
                # Images are already normalized to 0-1 range from preprocessing
                nuclear_channels_list = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                cyto_channels_list = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
                
                # Ensure images are in 0-1 range (should already be, but double-check)
                nuclear_img_normalized = self._ensure_0_1_range(nuclear_img)
                cyto_img_normalized = None
                if cyto_img is not None:
                    cyto_img_normalized = self._ensure_0_1_range(cyto_img)
                
                if nuclear_channels_list and cyto_channels_list:
                    # Combined mode: H x W x 3 array
                    h, w = nuclear_img_normalized.shape
                    cellsam_input = np.zeros((h, w, 3), dtype=np.float32)
                    cellsam_input[:, :, 1] = nuclear_img_normalized  # Channel 1 is nuclear
                    cellsam_input[:, :, 2] = cyto_img_normalized if cyto_img_normalized is not None else nuclear_img_normalized  # Channel 2 is cyto
                elif nuclear_channels_list:
                    # Nuclear only mode: H x W array
                    cellsam_input = nuclear_img_normalized
                elif cyto_channels_list:
                    # Cyto only mode: H x W array
                    cellsam_input = cyto_img_normalized if cyto_img_normalized is not None else nuclear_img_normalized
                else:
                    raise ValueError("At least one channel (nuclear or cyto) must be selected for CellSAM")
                
                images = None  # Not used for CellSAM
                channels = None  # Not used for CellSAM
            elif model == "nuclei":
                # For nuclei model, use only nuclear channel
                # Ensure 0-1 range
                nuclear_img_normalized = self._ensure_0_1_range(nuclear_img)
                images = [nuclear_img_normalized]
                channels = [0, 0]  # [cytoplasm, nucleus] - both are nuclear channel
            else:  # cyto3
                # For cyto3 model, use both channels
                # Ensure 0-1 range
                nuclear_img_normalized = self._ensure_0_1_range(nuclear_img)
                if cyto_img is None:
                    cyto_img_normalized = nuclear_img_normalized  # Fallback to nuclear channel
                else:
                    cyto_img_normalized = self._ensure_0_1_range(cyto_img)
                images = [cyto_img_normalized, nuclear_img_normalized]
                channels = [0, 1]  # [cytoplasm, nucleus]
            
            progress_dlg.update_progress(60, "Running segmentation", "Processing...")
            
            # Run segmentation
            if model == "DeepCell CellSAM":
                if dlg is None:
                    raise ValueError("Dialog object is required for DeepCell CellSAM segmentation")
                
                # Get DeepCell CellSAM parameters from dialog
                bbox_threshold = dlg.get_cellsam_bbox_threshold()
                use_wsi = dlg.get_cellsam_use_wsi()
                low_contrast_enhancement = dlg.get_cellsam_low_contrast_enhancement()
                gauge_cell_size = dlg.get_cellsam_gauge_cell_size()
                
                # Run DeepCell CellSAM pipeline
                progress_dlg.update_progress(65, "Running DeepCell CellSAM", "Processing image...")
                mask = cellsam_pipeline(
                    cellsam_input,
                    bbox_threshold=bbox_threshold,
                    use_wsi=use_wsi,
                    low_contrast_enhancement=low_contrast_enhancement,
                    gauge_cell_size=gauge_cell_size
                )
                # Use mask directly without any modifications - preserve exactly what CellSAM returns
                # Ensure mask is a numpy array and make a copy to avoid any potential modifications
                if isinstance(mask, np.ndarray):
                    mask = mask.copy()
                # Convert to list format for consistency with other segmentation methods
                masks = [mask]
                flows = [None]
                styles = [None]
                diams = [None]
            elif model == "Classical Watershed":
                if dlg is None:
                    raise ValueError("Dialog object is required for watershed segmentation")
                
                # Get channel information from preprocessing config
                nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
                
                # Get watershed parameters from dialog
                nuclear_fusion_method = dlg.get_nuclear_fusion_method()
                seed_threshold_method = dlg.get_seed_threshold_method()
                min_seed_area = dlg.get_min_seed_area()
                min_distance_peaks = dlg.get_min_distance_peaks()
                membrane_fusion_method = dlg.get_membrane_fusion_method()
                boundary_method = dlg.get_boundary_method()
                boundary_sigma = dlg.get_boundary_sigma()
                compactness = dlg.get_compactness()
                min_cell_area = dlg.get_min_cell_area()
                max_cell_area = dlg.get_max_cell_area()
                tile_size = dlg.get_tile_size()
                tile_overlap = dlg.get_tile_overlap()
                rng_seed = dlg.get_rng_seed()
                
                # Get nuclear and membrane channel weights from preprocessing config
                nuclear_weights = preprocessing_config.get('nuclear_weights', {})
                membrane_weights = preprocessing_config.get('cyto_weights', {})
                
                # Load full image stack for watershed
                loader = self._get_loader_for_acquisition(self.current_acq_id)
                if loader is None:
                    raise ValueError("No loader found for current acquisition")
                # Get original acquisition ID if this is a unique ID
                original_acq_id = self._get_original_acq_id(self.current_acq_id)
                img_stack = loader.get_all_channels(original_acq_id)
                channel_names = loader.get_channels(original_acq_id)
                
                # Run watershed segmentation
                masks = watershed_segmentation(
                    img_stack, channel_names, nuclear_channels, cyto_channels,
                    denoise_settings=custom_denoise_settings if denoise_source == "custom" else None,
                    nuclear_fusion_method=nuclear_fusion_method,
                    nuclear_weights=nuclear_weights,
                    seed_threshold_method=seed_threshold_method,
                    min_seed_area=min_seed_area,
                    min_distance_peaks=min_distance_peaks,
                    membrane_fusion_method=membrane_fusion_method,
                    membrane_weights=membrane_weights,
                    boundary_method=boundary_method,
                    boundary_sigma=boundary_sigma,
                    compactness=compactness,
                    min_cell_area=min_cell_area,
                    max_cell_area=max_cell_area,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    rng_seed=rng_seed
                )
                masks = [masks]  # Convert to list format for consistency
                flows = [None]
                styles = [None]
                diams = [None]
            else:
                # Run Cellpose segmentation
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
            if self.current_acq_id in self.cluster_colors:
                del self.cluster_colors[self.current_acq_id]
            if self.current_acq_id in self.cluster_color_map:
                del self.cluster_color_map[self.current_acq_id]
            self.segmentation_overlay = show_overlay
            
            # Save masks if requested
            if save_masks:
                self._save_segmentation_masks(masks_directory)
            
            progress_dlg.update_progress(100, "Segmentation complete", f"Found {len(np.unique(masks[0])) - 1} cells")
            
            # Update display if overlay is enabled
            if show_overlay:
                self.segmentation_overlay_chk.setChecked(True)
                self.segmentation_overlay_mode_widget.setVisible(True)
                self._update_display_with_segmentation()
            
            # Update overlay text with new cell count
            self._update_segmentation_overlay_text()
            
            progress_dlg.close()
            
            # Get channel information from preprocessing config (for display purposes)
            if model not in ["Classical Watershed", "DeepCell CellSAM"]:
                nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
            elif model == "DeepCell CellSAM":
                nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
            # For watershed, channels are already defined above
            
            # Log segmentation operation
            logger = get_logger()
            n_cells = len(np.unique(masks[0])) - 1
            
            if model == "DeepCell CellSAM":
                # Log DeepCell CellSAM segmentation
                params = {
                    "method": "cellsam",
                    "bbox_threshold": bbox_threshold,
                    "use_wsi": use_wsi,
                    "low_contrast_enhancement": low_contrast_enhancement,
                    "gauge_cell_size": gauge_cell_size,
                    "nuclear_channels": nuclear_channels,
                    "cyto_channels": cyto_channels,
                    "input_mode": "combined" if (nuclear_channels and cyto_channels) else ("nuclear" if nuclear_channels else "cyto"),
                    "n_cells": int(n_cells)
                }
                if denoise_source == "custom" and custom_denoise_settings:
                    params["denoise_settings"] = custom_denoise_settings
                else:
                    params["denoise_source"] = denoise_source
            elif model == "Classical Watershed":
                # Log watershed segmentation
                params = {
                    "method": "watershed",
                    "nuclear_channels": nuclear_channels,
                    "cyto_channels": cyto_channels,
                    "nuclear_fusion_method": nuclear_fusion_method,
                    "nuclear_weights": nuclear_weights,
                    "seed_threshold_method": seed_threshold_method,
                    "min_seed_area": min_seed_area,
                    "min_distance_peaks": min_distance_peaks,
                    "membrane_fusion_method": membrane_fusion_method,
                    "membrane_weights": membrane_weights,
                    "boundary_method": boundary_method,
                    "boundary_sigma": boundary_sigma,
                    "compactness": compactness,
                    "min_cell_area": min_cell_area,
                    "max_cell_area": max_cell_area,
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "rng_seed": rng_seed,
                    "n_cells": int(n_cells)
                }
                if denoise_source == "custom" and custom_denoise_settings:
                    params["denoise_settings"] = custom_denoise_settings
                else:
                    params["denoise_source"] = denoise_source
            else:
                # Log Cellpose segmentation
                params = {
                    "model_type": model,
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "use_gpu": use_gpu,
                    "gpu_device": str(gpu_device) if gpu_device is not None else None,
                    "nuclear_channels": nuclear_channels,
                    "cyto_channels": cyto_channels,
                    "n_cells": int(n_cells)
                }
                if denoise_source == "custom" and custom_denoise_settings:
                    params["denoise_settings"] = custom_denoise_settings
                else:
                    params["denoise_source"] = denoise_source
                # Handle diams - it can be a scalar or a list/array
                if diams is not None:
                    if isinstance(diams, (list, tuple, np.ndarray)) and len(diams) > 0:
                        if diams[0] is not None:
                            params["estimated_diameter"] = float(diams[0])
                    elif isinstance(diams, (int, float, np.integer, np.floating)):
                        params["estimated_diameter"] = float(diams)
            
            # Get source file name
            source_file = os.path.basename(self.current_path) if self.current_path else None
            
            method_name = "watershed" if model == "Classical Watershed" else ("cellsam" if model == "DeepCell CellSAM" else "cellpose")
            logger.log_segmentation(
                method=method_name,
                parameters=params,
                acquisitions=[self.current_acq_id],
                output_path=masks_directory if save_masks else None,
                notes=f"Segmented {n_cells} cells",
                source_file=source_file
            )
            
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
                f"Successfully segmented {n_cells} cells.\n"
                f"Model: {model}\n"
                f"Channels: {channel_info if channel_info else 'Not specified'}"
            )
            
        except Exception as e:
            progress_dlg.close()
            raise e
    
    def _perform_segmentation_all_acquisitions(self, model: str, diameter: int = None, 
                                             flow_threshold: float = 0.4, cellprob_threshold: float = 0.0, 
                                             show_overlay: bool = True, save_masks: bool = False, 
                                             masks_directory: str = None, gpu_id = None, preprocessing_config = None,
                                             denoise_source: str = "Use viewer settings", custom_denoise_settings: dict = None, dlg = None):
        """Perform efficient batch segmentation on all acquisitions."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions available for segmentation.")
            return
        
        # Create progress dialog for batch processing
        total_acquisitions = len(self.acquisitions)
        progress_dlg = ProgressDialog("Batch Cell Segmentation", self)
        progress_dlg.set_maximum(total_acquisitions)
        progress_dlg.show()
        
        # Watershed segmentation is not supported in batch mode yet
        if model == "Classical Watershed":
            progress_dlg.close()
            QtWidgets.QMessageBox.warning(
                self, "Batch Watershed Not Supported", 
                "Batch segmentation is not yet supported for Classical Watershed.\n"
                "Please run watershed segmentation on individual acquisitions."
            )
            return
        
        # CellSAM batch processing: sequential (one at a time) since CellSAM doesn't support batch processing
        if model == "DeepCell CellSAM":
            self._perform_cellsam_batch_segmentation(
                preprocessing_config, denoise_source, custom_denoise_settings, 
                save_masks, masks_directory, dlg, progress_dlg, total_acquisitions
            )
            return
        
        try:
            # Set fixed batch size
            batch_size = 16
            progress_dlg.update_progress(0, "Initializing batch processing", f"Batch size: {batch_size} (0/{total_acquisitions} completed)")
            
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
                        batch_acquisitions, preprocessing_config, progress_dlg, denoise_source, custom_denoise_settings
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
                    acquisition_mapping = batch_data['acquisition_mapping']  # Contains indices
                    acquisition_info_list = batch_data.get('acquisition_info_list', [])  # List of AcquisitionInfo objects
                    processed_acquisitions = set()
                    
                    for i, mask in enumerate(masks):
                        if i < len(acquisition_mapping):
                            acq_idx = acquisition_mapping[i]  # Index in acquisition_info_list
                            
                            # Only process each acquisition once (use the first mask for each acquisition)
                            if acq_idx not in processed_acquisitions and acq_idx < len(acquisition_info_list):
                                acq_info = acquisition_info_list[acq_idx]
                                acq_id = acq_info.id
                                
                                self.segmentation_masks[acq_id] = mask
                                # Clear colors for this acquisition so they get regenerated
                                if acq_id in self.segmentation_colors:
                                    del self.segmentation_colors[acq_id]
                                if acq_id in self.cluster_colors:
                                    del self.cluster_colors[acq_id]
                                if acq_id in self.cluster_color_map:
                                    del self.cluster_color_map[acq_id]
                                successful_segmentations += 1
                                processed_acquisitions.add(acq_idx)
                                
                                # Save masks if requested - use the AcquisitionInfo directly from the batch
                                if save_masks:
                                    self._save_segmentation_masks_for_acquisition_with_info(mask, acq_info, masks_directory)
                                
                                # Update progress after each acquisition
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
    
    
    def _perform_cellsam_batch_segmentation(self, preprocessing_config: dict, denoise_source: str, 
                                            custom_denoise_settings: dict, save_masks: bool, 
                                            masks_directory: str, dlg, progress_dlg, total_acquisitions: int):
        """Perform sequential batch segmentation for CellSAM (one acquisition at a time)."""
        if dlg is None:
            QtWidgets.QMessageBox.critical(
                self, "Missing Dialog", 
                "Dialog object is required for DeepCell CellSAM segmentation."
            )
            progress_dlg.close()
            return
        
        try:
            # Initialize CellSAM model once
            progress_dlg.update_progress(0, "Initializing DeepCell CellSAM", "Setting up model...")
            
            # Set API key from dialog
            api_key = dlg.get_cellsam_api_key()
            if api_key:
                os.environ.update({"DEEPCELL_ACCESS_TOKEN": api_key})
            elif not os.environ.get("DEEPCELL_ACCESS_TOKEN"):
                progress_dlg.close()
                QtWidgets.QMessageBox.critical(
                    self, "Missing API Key",
                    "DeepCell API key is required for CellSAM.\n"
                    "Please enter your API key in the CellSAM Parameters section.\n"
                    "Get your key from https://users.deepcell.org/login/"
                )
                return
            
            # Initialize CellSAM model and download weights
            progress_dlg.update_progress(0, "Initializing CellSAM model", "Downloading model weights...")
            try:
                get_model()  # This downloads weights if not already present
            except Exception as e:
                progress_dlg.close()
                QtWidgets.QMessageBox.critical(
                    self, "CellSAM Initialization Failed",
                    f"Failed to initialize CellSAM model:\n{str(e)}\n\n"
                    "Please check your API key and internet connection."
                )
                return
            
            # Get CellSAM parameters from dialog
            bbox_threshold = dlg.get_cellsam_bbox_threshold()
            use_wsi = dlg.get_cellsam_use_wsi()
            low_contrast_enhancement = dlg.get_cellsam_low_contrast_enhancement()
            gauge_cell_size = dlg.get_cellsam_gauge_cell_size()
            
            # Process each acquisition sequentially
            successful_segmentations = 0
            
            for acq_idx, acq in enumerate(self.acquisitions):
                if progress_dlg.is_cancelled():
                    break
                
                acq_id = acq.id
                acq_name = acq.name
                
                progress_dlg.update_progress(
                    successful_segmentations,
                    f"Processing acquisition {acq_idx + 1}/{total_acquisitions}",
                    f"Loading {acq_name}... ({successful_segmentations}/{total_acquisitions} completed)"
                )
                
                try:
                    # Preprocess this acquisition
                    nuclear_img, cyto_img = self._preprocess_acquisition_for_cellsam(
                        acq_id, preprocessing_config, progress_dlg, denoise_source, custom_denoise_settings
                    )
                    
                    # Prepare CellSAM input
                    nuclear_channels_list = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                    cyto_channels_list = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
                    
                    if nuclear_channels_list and cyto_channels_list:
                        # Combined mode: H x W x 3 array
                        if nuclear_img is None:
                            raise ValueError("Failed to load nuclear channels for combined mode")
                        if cyto_img is None:
                            raise ValueError("Failed to load cytoplasm channels for combined mode")
                        h, w = nuclear_img.shape
                        cellsam_input = np.zeros((h, w, 3), dtype=np.float32)
                        cellsam_input[:, :, 1] = nuclear_img  # Channel 1 is nuclear
                        cellsam_input[:, :, 2] = cyto_img  # Channel 2 is cyto
                    elif nuclear_channels_list:
                        # Nuclear only mode: H x W array
                        if nuclear_img is None:
                            raise ValueError("Failed to load nuclear channels")
                        cellsam_input = nuclear_img
                    elif cyto_channels_list:
                        # Cyto only mode: H x W array
                        if cyto_img is None:
                            raise ValueError("Failed to load cytoplasm channels")
                        cellsam_input = cyto_img
                    else:
                        raise ValueError("At least one channel (nuclear or cyto) must be selected for CellSAM")
                    
                    # Run CellSAM segmentation
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Segmenting {acq_name}",
                        f"Running CellSAM... ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                    mask = cellsam_pipeline(
                        cellsam_input,
                        bbox_threshold=bbox_threshold,
                        use_wsi=use_wsi,
                        low_contrast_enhancement=low_contrast_enhancement,
                        gauge_cell_size=gauge_cell_size
                    )
                    
                    # Store the mask
                    if isinstance(mask, np.ndarray):
                        mask = mask.copy()
                    self.segmentation_masks[acq_id] = mask
                    # Clear colors for this acquisition so they get regenerated
                    if acq_id in self.segmentation_colors:
                        del self.segmentation_colors[acq_id]
                    if acq_id in self.cluster_colors:
                        del self.cluster_colors[acq_id]
                    if acq_id in self.cluster_color_map:
                        del self.cluster_color_map[acq_id]
                    
                    successful_segmentations += 1
                    
                    # Save mask if requested
                    if save_masks:
                        self._save_segmentation_masks_for_acquisition(mask, acq_id, masks_directory)
                    
                    # Update progress
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Completed {acq_name}",
                        f"Segmented {acq_name} ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                except Exception as e:
                    print(f"Error segmenting acquisition {acq_name} ({acq_id}): {e}")
                    # Continue with next acquisition
                    continue
            
            progress_dlg.update_progress(
                total_acquisitions, 
                "Batch segmentation complete", 
                f"Successfully segmented {successful_segmentations}/{total_acquisitions} acquisitions"
            )
            
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
    
    
    def _preprocess_acquisition_for_cellsam(self, acq_id: str, preprocessing_config: dict, progress_dlg, 
                                           denoise_source: str, custom_denoise_settings: dict) -> tuple:
        """Preprocess a specific acquisition for CellSAM segmentation."""
        if not preprocessing_config:
            raise ValueError("Preprocessing configuration is required for segmentation")
        
        # Get the correct loader for this acquisition
        loader = self._get_loader_for_acquisition(acq_id)
        if loader is None:
            raise ValueError(f"No loader found for acquisition {acq_id}")
        
        config = preprocessing_config
        
        # Get nuclear channels
        nuclear_channels = config.get('nuclear_channels', [])
        
        # Get cytoplasm channels
        cyto_channels = config.get('cyto_channels', [])
        
        # For CellSAM, at least one channel type must be selected
        if not nuclear_channels and not cyto_channels:
            raise ValueError("At least one channel (nuclear or cyto) must be selected for CellSAM")
        
        # Get original acquisition ID if this is a unique ID
        original_acq_id = self._get_original_acq_id(acq_id)
        
        # Load and normalize nuclear channels
        nuclear_img = None
        if nuclear_channels:
            nuclear_imgs = []
            for channel in nuclear_channels:
                img = loader.get_image(original_acq_id, channel)
                if img is None:
                    continue
                
                # Apply denoising
                if denoise_source == "viewer":
                    # Viewer denoising requires viewer state, skip for batch processing
                    pass
                elif denoise_source == "custom" and custom_denoise_settings:
                    try:
                        img = self._apply_custom_denoise(channel, img, custom_denoise_settings)
                    except Exception:
                        pass
                
                # Apply normalization
                img = self._apply_normalization(img, config, acq_id, channel)
                # Ensure 0-1 range after denoising and normalization
                img = self._ensure_0_1_range(img)
                nuclear_imgs.append(img)
            
            if not nuclear_imgs:
                raise ValueError(f"Failed to load nuclear channels for acquisition {acq_id}")
            
            # Combine nuclear channels
            nuclear_combo_method = config.get('nuclear_combo_method', 'single')
            nuclear_weights = config.get('nuclear_weights')
            nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights)
            # Ensure combined image is in 0-1 range
            nuclear_img = self._ensure_0_1_range(nuclear_img)
        
        # Load and normalize cytoplasm channels
        cyto_img = None
        if cyto_channels:
            cyto_imgs = []
            for channel in cyto_channels:
                img = loader.get_image(original_acq_id, channel)
                if img is None:
                    continue
                
                # Apply denoising
                if denoise_source == "viewer":
                    # Viewer denoising requires viewer state, skip for batch processing
                    pass
                elif denoise_source == "custom" and custom_denoise_settings:
                    try:
                        img = self._apply_custom_denoise(channel, img, custom_denoise_settings)
                    except Exception:
                        pass
                
                # Apply normalization
                img = self._apply_normalization(img, config, acq_id, channel)
                # Ensure 0-1 range after denoising and normalization
                img = self._ensure_0_1_range(img)
                cyto_imgs.append(img)
            
            if cyto_imgs:
                # Combine cytoplasm channels
                cyto_combo_method = config.get('cyto_combo_method', 'single')
                cyto_weights = config.get('cyto_weights')
                cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights)
                # Ensure combined image is in 0-1 range
                cyto_img = self._ensure_0_1_range(cyto_img)
        
        return nuclear_img, cyto_img
    
    
    def _load_batch_acquisitions(self, acquisitions, preprocessing_config: dict, progress_dlg, denoise_source: str = "none", custom_denoise_settings: dict = None) -> dict:
        """Load and preprocess a batch of acquisitions efficiently using multiprocessing."""
        batch_images = []
        batch_channels = []
        acquisition_mapping = []  # Track which images belong to which acquisition (by index in acquisition_info_list)
        acquisition_info_list = []  # Store AcquisitionInfo objects for each successfully processed acquisition
        
        # Prepare arguments for multiprocessing
        mp_args = []
        acq_to_index_map = {}  # Map acquisition ID to index in original acquisitions list
        
        for idx, acq in enumerate(acquisitions):
            acq_to_index_map[acq.id] = idx
            
            # Get source file path and determine loader type
            source_file = acq.source_file if hasattr(acq, 'source_file') else None
            
            # Determine file path and loader type
            if acq.id in self.acq_to_file:
                file_path = self.acq_to_file[acq.id]
                loader_type = "mcd"
            elif self.current_path:
                if os.path.isdir(self.current_path):
                    file_path = self.current_path
                    loader_type = "ometiff"
                else:
                    file_path = self.current_path
                    loader_type = "mcd"
            else:
                if source_file:
                    file_path = source_file
                    loader_type = "mcd" if source_file.endswith('.mcd') else "ometiff"
                else:
                    print(f"Warning: Cannot determine file path for acquisition {acq.name}, skipping")
                    continue
            
            # Normalize denoise_source: convert "Use viewer settings" to "viewer" or "none"
            if denoise_source == "Use viewer settings":
                # For multiprocessing, viewer denoising is not supported (requires viewer state)
                # Use custom denoising if available, otherwise none
                if custom_denoise_settings:
                    denoise_source_worker = "custom"
                else:
                    denoise_source_worker = "none"
            elif denoise_source == "viewer":
                # Viewer denoising not supported in multiprocessing
                if custom_denoise_settings:
                    denoise_source_worker = "custom"
                else:
                    denoise_source_worker = "none"
            else:
                denoise_source_worker = denoise_source
            
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq.id)
            # Get channel format for OME-TIFF files
            channel_format = self.ometiff_channel_format if loader_type == "ometiff" else None
            mp_args.append((
                original_acq_id,  # Use original ID for loader
                acq.id,  # Pass unique ID for tracking
                acq.name,
                file_path,
                loader_type,
                preprocessing_config,
                denoise_source_worker,
                custom_denoise_settings,
                source_file,
                channel_format
            ))
        
        if not mp_args:
            return None
        
        # Use multiprocessing for parallel loading and preprocessing
        max_workers = min(mp.cpu_count(), len(mp_args))
        
        batch_images = []
        batch_channels = []
        acquisition_mapping = []
        acquisition_info_list = []
        acq_id_to_acq_info = {acq.id: acq for acq in acquisitions}
        
        try:
            with mp.Pool(processes=max_workers) as pool:
                # Submit all tasks - each worker loads and preprocesses one acquisition
                futures = []
                for task_data in mp_args:
                    future = pool.apply_async(_load_and_preprocess_acquisition_worker, (task_data,))
                    futures.append(future)
                
                # Collect results as they complete
                for future in futures:
                    if progress_dlg and progress_dlg.is_cancelled():
                        break
                    
                    try:
                        result = future.get(timeout=600)  # 10 minute timeout per acquisition
                        if result is None:
                            continue
                        
                        acq_id = result['acq_id']
                        acq_name = result['acq_name']
                        nuclear_img = result['nuclear_img']
                        cyto_img = result['cyto_img']
                        
                        # Get acquisition info
                        if acq_id not in acq_id_to_acq_info:
                            print(f"Warning: Acquisition info not found for {acq_id}")
                            continue
                        
                        acq = acq_id_to_acq_info[acq_id]
                        
                        # Prepare input images based on model type
                        if nuclear_img is not None:
                            # Store the acquisition info for this processed acquisition
                            acq_idx = len(acquisition_info_list)
                            acquisition_info_list.append(acq)
                            
                            if cyto_img is not None:
                                # Both nuclear and cytoplasm available
                                batch_images.extend([cyto_img, nuclear_img])
                                batch_channels.extend([0, 1])  # cyto, nuclear
                                acquisition_mapping.extend([acq_idx, acq_idx])  # Both images belong to same acquisition
                            else:
                                # Only nuclear available
                                batch_images.extend([nuclear_img, nuclear_img])
                                batch_channels.extend([0, 0])  # nuclear, nuclear
                                acquisition_mapping.extend([acq_idx, acq_idx])  # Both images belong to same acquisition
                        else:
                            print(f"Warning: No valid images for acquisition {acq_name}")
                    except Exception as e:
                        print(f"Error processing acquisition in multiprocessing: {e}")
                        continue
        
        except Exception as e:
            print(f"Error in multiprocessing batch loading: {e}")
            # Fall back to sequential processing
            return self._load_batch_acquisitions_sequential(
                acquisitions, preprocessing_config, progress_dlg, denoise_source, custom_denoise_settings
            )
        
        if not batch_images:
            return None
        
        return {
            'images': batch_images,
            'channels': batch_channels,
            'acquisition_mapping': acquisition_mapping,  # Contains indices into acquisition_info_list
            'acquisition_info_list': acquisition_info_list,  # List of AcquisitionInfo objects in processing order
            'acquisition_count': len(acquisitions)
        }
    
    def _load_batch_acquisitions_sequential(self, acquisitions, preprocessing_config: dict, progress_dlg, denoise_source: str = "none", custom_denoise_settings: dict = None) -> dict:
        """Load and preprocess a batch of acquisitions sequentially (fallback method)."""
        batch_images = []
        batch_channels = []
        acquisition_mapping = []  # Track which images belong to which acquisition (by index in acquisition_info_list)
        acquisition_info_list = []  # Store AcquisitionInfo objects for each successfully processed acquisition
        
        for acq in acquisitions:
            try:
                # Temporarily set current acquisition for preprocessing
                original_acq_id = self.current_acq_id
                self.current_acq_id = acq.id
                
                # Preprocess channels for this acquisition
                nuclear_img, cyto_img = self._preprocess_channels_for_segmentation(
                    preprocessing_config, progress_dlg, False, denoise_source, custom_denoise_settings
                )
                
                # Prepare input images based on model type
                if nuclear_img is not None:
                    # Store the acquisition info for this processed acquisition (before adding images)
                    acq_idx = len(acquisition_info_list)
                    acquisition_info_list.append(acq)
                    
                    if cyto_img is not None:
                        # Both nuclear and cytoplasm available
                        batch_images.extend([cyto_img, nuclear_img])
                        batch_channels.extend([0, 1])  # cyto, nuclear
                        acquisition_mapping.extend([acq_idx, acq_idx])  # Both images belong to same acquisition (by index in info list)
                    else:
                        # Only nuclear available
                        batch_images.extend([nuclear_img, nuclear_img])
                        batch_channels.extend([0, 0])  # nuclear, nuclear
                        acquisition_mapping.extend([acq_idx, acq_idx])  # Both images belong to same acquisition (by index in info list)
                else:
                    print(f"Warning: No valid images for acquisition {acq.name}")
                    # Don't add to acquisition_info_list since processing failed
                    continue
                
                # Restore original acquisition
                self.current_acq_id = original_acq_id
                
            except Exception as e:
                print(f"Error preprocessing acquisition {acq.name}: {e}")
                # Don't add to acquisition_info_list since processing failed
                continue
        
        if not batch_images:
            return None
        
        return {
            'images': batch_images,
            'channels': batch_channels,
            'acquisition_mapping': acquisition_mapping,  # Contains indices into acquisition_info_list
            'acquisition_info_list': acquisition_info_list,  # List of AcquisitionInfo objects in processing order
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
                preprocessing_config, None, False, "Use viewer settings", None
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
            if acq.id in self.cluster_colors:
                del self.cluster_colors[acq.id]
            if acq.id in self.cluster_color_map:
                del self.cluster_color_map[acq.id]
            
            # Save masks if requested
            if save_masks:
                self._save_segmentation_masks_for_acquisition(masks[0], acq.id, masks_directory)
                
        finally:
            # Restore original acquisition
            self.current_acq_id = original_acq_id
    
    def _save_segmentation_masks_for_acquisition_with_info(self, masks: np.ndarray, acq_info: AcquisitionInfo, masks_directory: str = None):
        """Save segmentation masks for a specific acquisition using the provided AcquisitionInfo."""
        # Use well name if available, otherwise use acquisition name
        if acq_info.well:
            label_for_filename = acq_info.well
        else:
            label_for_filename = acq_info.name
        
        safe_label = self._sanitize_filename(label_for_filename)
        # Include source file name in filename to ensure uniqueness across multiple MCD files
        if acq_info.source_file:
            # Use source file basename (without extension) to make filename unique
            source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
            safe_source = self._sanitize_filename(source_basename)
            filename = f"{safe_source}_{safe_label}_segmentation_masks.tif"
        else:
            filename = f"{safe_label}_segmentation_masks.tif"
        
        # Use provided directory or fallback to current file/folder directory
        if masks_directory and os.path.exists(masks_directory):
            filepath = os.path.join(masks_directory, filename)
        else:
            # Use source file directory if available, otherwise fallback to current_path
            if acq_info.source_file:
                base_dir = os.path.dirname(acq_info.source_file)
            else:
                base_dir = os.path.dirname(self.current_path) if self.current_path else "."
            filepath = os.path.join(base_dir, filename)
        
        try:
            tifffile.imwrite(filepath, masks.astype(np.uint16))
            print(f"Segmentation masks saved: {filepath}")
        except Exception as e:
            print(f"Error saving segmentation masks: {e}")
    
    def _save_segmentation_masks_for_acquisition(self, masks: np.ndarray, acq_id: str, masks_directory: str = None):
        """Save segmentation masks for a specific acquisition."""
        acq_info = self._get_acquisition_info(acq_id)
        if acq_info is None:
            print(f"Warning: Could not find acquisition {acq_id} for saving masks")
            return
        self._save_segmentation_masks_for_acquisition_with_info(masks, acq_info, masks_directory)
    
    def _save_segmentation_masks(self, masks_directory: str = None):
        """Save all segmentation masks to files."""
        if not self.segmentation_masks:
            QtWidgets.QMessageBox.information(
                self, "No Masks", 
                "No segmentation masks available to save."
            )
            return
        
        # Use provided directory or ask user to select
        if masks_directory and os.path.exists(masks_directory):
            output_dir = masks_directory
        else:
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Segmentation Masks", ""
            )
            if not output_dir:
                return
        
        # Save all masks
        saved_count = 0
        for acq_id, mask in self.segmentation_masks.items():
            try:
                acq_info = self._get_acquisition_info(acq_id)
                if acq_info is None:
                    print(f"Warning: Could not find acquisition {acq_id} for saving masks")
                    continue
                
                # Use well name if available, otherwise use acquisition name
                if acq_info.well:
                    label_for_filename = acq_info.well
                else:
                    label_for_filename = acq_info.name
                
                safe_label = self._sanitize_filename(label_for_filename)
                # Include source file name in filename to ensure uniqueness across multiple MCD files
                if acq_info.source_file:
                    source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
                    safe_source = self._sanitize_filename(source_basename)
                    filename = f"{safe_source}_{safe_label}_segmentation.tiff"
                else:
                    filename = f"{safe_label}_segmentation.tiff"
                
                output_path = os.path.join(output_dir, filename)
                
                # Save mask as TIFF
                if _HAVE_TIFFFILE:
                    tifffile.imwrite(output_path, mask.astype(np.uint16))
                else:
                    # Fallback to numpy save
                    np.save(output_path.replace('.tiff', '.npy'), mask)
                
                saved_count += 1
            except Exception as e:
                print(f"Error saving mask for {acq_id}: {e}")
                continue
        
        QtWidgets.QMessageBox.information(
            self, "Masks Saved", 
            f"Successfully saved {saved_count} segmentation mask(s) to:\n{output_dir}"
        )
    
    def _update_display_with_segmentation(self):
        """Update the current display to show segmentation overlay."""
        if not self.segmentation_overlay or self.current_acq_id not in self.segmentation_masks:
            return
        
        # Refresh the current view
        self._view_selected()
    
    def _get_source_well_for_acquisition(self, acq_id: str) -> Optional[str]:
        """Get the source_well value for a given acquisition ID.
        
        Constructs source_well in the same way as feature extraction:
        - If source_file and well: "{source_file_base}_{well}"
        - If only well: "{well}"
        - If only source_file: "{source_file_base}"
        """
        if not hasattr(self, 'acquisitions') or not self.acquisitions:
            return None
        
        # Find the acquisition info
        acq_info = None
        for acq in self.acquisitions:
            if acq.id == acq_id:
                acq_info = acq
                break
        
        if acq_info is None:
            return None
        
        # Get source file
        source_file = None
        if hasattr(self, 'acq_to_file') and acq_id in self.acq_to_file:
            source_file = self.acq_to_file[acq_id]
        
        # Get well name
        well_name = acq_info.well if hasattr(acq_info, 'well') else None
        
        # Construct source_well the same way as feature extraction
        import os
        source_filename = os.path.basename(source_file) if source_file else None
        source_well = None
        
        if source_filename and well_name:
            # Remove .tif, .mcd, or .ome.tif extensions
            source_base = source_filename
            for ext in ['.ome.tif', '.tif', '.mcd']:
                if source_base.lower().endswith(ext):
                    source_base = source_base[:-len(ext)]
                    break
            source_well = f"{source_base}_{well_name}"
        elif well_name:
            # If no source_file but we have well, just use well
            source_well = well_name
        elif source_filename:
            # If no well but we have source_file, use source_file base name
            source_base = source_filename
            for ext in ['.ome.tif', '.tif', '.mcd']:
                if source_base.lower().endswith(ext):
                    source_base = source_base[:-len(ext)]
                    break
            source_well = source_base
        
        return source_well
    
    def _has_cluster_data(self, acq_id: Optional[str] = None) -> bool:
        """Check if cluster data is available in the feature dataframe.
        
        Args:
            acq_id: Optional acquisition ID to check for specific acquisition. If None, checks all data.
        
        Returns:
            True if cluster data is available, False otherwise.
        """
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return False
        
        # Filter dataframe for specific acquisition if provided
        if acq_id is not None:
            # Try matching by source_well first (preferred method)
            if 'source_well' in self.feature_dataframe.columns:
                source_well = self._get_source_well_for_acquisition(acq_id)
                if source_well:
                    acq_df = self.feature_dataframe[self.feature_dataframe['source_well'] == source_well]
                else:
                    acq_df = pd.DataFrame()
            else:
                acq_df = pd.DataFrame()
            
            # Fallback to acquisition_id matching if source_well doesn't work
            if acq_df.empty and 'acquisition_id' in self.feature_dataframe.columns:
                # Try exact match first
                acq_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == acq_id]
                
                # If no exact match, try using original acquisition ID (for multiple MCD files)
                if acq_df.empty and hasattr(self, 'unique_acq_to_original') and acq_id in self.unique_acq_to_original:
                    original_acq_id = self.unique_acq_to_original[acq_id]
                    acq_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == original_acq_id]
        else:
            acq_df = self.feature_dataframe
        
        if acq_df.empty:
            return False
        
        # Check if cluster columns exist
        has_cluster_col = any(col in acq_df.columns for col in ['cluster_phenotype', 'cluster', 'cluster_id'])
        has_centroid_cols = 'centroid_x' in acq_df.columns and 'centroid_y' in acq_df.columns
        
        # Only require cluster column - centroids can be computed from mask if missing
        return has_cluster_col
    
    def _update_cluster_mode_availability(self):
        """Update the availability of Cluster mode in the overlay dropdown."""
        # Check for cluster data - first try current acquisition, then check all data
        current_acq_id = self.current_acq_id if hasattr(self, 'current_acq_id') else None
        has_cluster = self._has_cluster_data(current_acq_id)
        
        # If no cluster data for current acquisition, check if any cluster data exists at all
        if not has_cluster:
            has_cluster = self._has_cluster_data(None)  # Check all data
        
        # Find the Cluster item in the combo box
        cluster_index = -1
        for i in range(self.segmentation_overlay_mode_combo.count()):
            if self.segmentation_overlay_mode_combo.itemText(i) == "Cluster":
                cluster_index = i
                break
        
        if cluster_index >= 0:
            # Enable or disable the Cluster option
            model = self.segmentation_overlay_mode_combo.model()
            item = model.item(cluster_index)
            if item:
                item.setEnabled(has_cluster)
                # If Cluster is currently selected but no longer available, switch to Mask
                if (self.segmentation_overlay_mode_combo.currentText() == "Cluster" and not has_cluster):
                    self.segmentation_overlay_mode_combo.setCurrentText("Mask")
                    self.segmentation_overlay_mode = "Mask"
    
    def _get_cluster_assignments(self, acq_id: str) -> Optional[Dict[int, Union[int, str]]]:
        """Get cluster assignments for cells in the current acquisition by matching centroids.
        
        Matches mask cells to feature dataframe rows by finding the closest centroid coordinates.
        
        Returns:
            Dictionary mapping mask cell label (cell_id) to cluster_id/cluster_phenotype, or None if no cluster data available.
        """
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            return None
        
        if acq_id not in self.segmentation_masks:
            return None
        
        mask = self.segmentation_masks[acq_id]
        
        # Filter dataframe for current acquisition - use source_well (preferred)
        if 'source_well' in self.feature_dataframe.columns:
            source_well = self._get_source_well_for_acquisition(acq_id)
            if source_well:
                acq_df = self.feature_dataframe[self.feature_dataframe['source_well'] == source_well].copy()
            else:
                acq_df = pd.DataFrame()
        else:
            acq_df = pd.DataFrame()
        
        # Fallback to acquisition_id matching if source_well doesn't work
        if acq_df.empty and 'acquisition_id' in self.feature_dataframe.columns:
            # Try exact match first
            acq_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == acq_id].copy()
            
            # If no exact match, try using original acquisition ID (for multiple MCD files)
            if acq_df.empty and hasattr(self, 'unique_acq_to_original') and acq_id in self.unique_acq_to_original:
                original_acq_id = self.unique_acq_to_original[acq_id]
                acq_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == original_acq_id].copy()
            
            # If still no match, try matching by prefix
            if acq_df.empty:
                for df_acq_id in self.feature_dataframe['acquisition_id'].unique():
                    if acq_id.startswith(df_acq_id) or df_acq_id.startswith(acq_id):
                        acq_df = self.feature_dataframe[self.feature_dataframe['acquisition_id'] == df_acq_id].copy()
                        break
        elif acq_df.empty:
            # No source_well or acquisition_id columns
            acq_df = self.feature_dataframe.copy()
        
        if acq_df.empty:
            return None
        
        # Look for cluster columns - prefer numeric cluster ID over phenotype label for overlay
        # Use 'cluster' or 'cluster_id' for numeric IDs, not 'cluster_phenotype' which has custom labels
        cluster_col = None
        for col in ['cluster', 'cluster_id']:
            if col in acq_df.columns:
                cluster_col = col
                break
        
        # Fallback to cluster_phenotype only if numeric columns don't exist
        # In this case, we'll need to map phenotype labels back to numeric cluster IDs
        # by creating a reverse mapping from the dataframe
        if cluster_col is None and 'cluster_phenotype' in acq_df.columns:
            # Try to find the original cluster column - it should exist if clustering was done
            # If not, we'll create a mapping from unique phenotypes to numeric IDs
            if 'cluster' in acq_df.columns:
                cluster_col = 'cluster'
            else:
                # Create a mapping from phenotype to numeric ID for overlay purposes
                # This is a fallback - normally 'cluster' should exist
                cluster_col = 'cluster_phenotype'
        
        if cluster_col is None:
            return None
        
        # Check if centroids are in dataframe, otherwise we'll compute them from mask
        has_centroids = 'centroid_x' in acq_df.columns and 'centroid_y' in acq_df.columns
        
        # Compute centroids for each cell in the mask using regionprops
        props = regionprops(mask)
        
        # Create mapping from mask cell labels to their centroids
        mask_centroids = {}
        for prop in props:
            if prop.label == 0:  # Skip background
                continue
            # regionprops centroid is (row, col) = (y, x)
            mask_centroids[prop.label] = (prop.centroid[1], prop.centroid[0])  # (x, y)
        
        if not mask_centroids:
            return None
        
        # Create mapping from mask cell labels to clusters
        cluster_map = {}
        
        if has_centroids:
            # Match by centroids if available in dataframe
            # Remove rows with NaN centroids from feature dataframe
            valid_df = acq_df.dropna(subset=['centroid_x', 'centroid_y'])
            
            if not valid_df.empty:
                # For each mask cell, find the closest feature dataframe row by centroid distance
                for mask_label, (mask_x, mask_y) in mask_centroids.items():
                    # Calculate distances to all feature dataframe centroids
                    distances = np.sqrt(
                        (valid_df['centroid_x'].values - mask_x) ** 2 +
                        (valid_df['centroid_y'].values - mask_y) ** 2
                    )
                    
                    # Find the closest match (within reasonable tolerance, e.g., 5 pixels)
                    min_distance = np.min(distances)
                    if min_distance < 5.0:  # Tolerance: 5 pixels
                        closest_idx = np.argmin(distances)
                        closest_row = valid_df.iloc[closest_idx]
                        cluster_val = closest_row[cluster_col]
                        
                        # Handle NaN values
                        if pd.isna(cluster_val):
                            cluster_map[mask_label] = 'Unassigned'
                        else:
                            cluster_map[mask_label] = cluster_val
                    else:
                        # No close match found
                        cluster_map[mask_label] = 'Unassigned'
        else:
            # Centroids not in dataframe - match by cell_id as fallback
            if 'cell_id' in acq_df.columns:
                # Create mapping from cell_id to cluster
                cell_to_cluster = {}
                for _, row in acq_df.iterrows():
                    cell_id = int(row['cell_id'])
                    cluster_val = row[cluster_col]
                    if pd.isna(cluster_val):
                        cell_to_cluster[cell_id] = 'Unassigned'
                    else:
                        cell_to_cluster[cell_id] = cluster_val
                
                # Map mask labels (which are cell_ids) to clusters
                for mask_label in mask_centroids.keys():
                    cluster_map[mask_label] = cell_to_cluster.get(mask_label, 'Unassigned')
            else:
                # No way to match - return None
                return None
        
        return cluster_map
    
    def _get_segmentation_overlay(self, img: np.ndarray) -> np.ndarray:
        """Create segmentation overlay for display (outline, mask, or cluster mode)."""
        if not self.segmentation_overlay or self.current_acq_id not in self.segmentation_masks:
            return img
        
        mask = self.segmentation_masks[self.current_acq_id]
        
        # Create overlay
        overlay = np.zeros((*img.shape[:2], 3), dtype=np.float32)
        
        # Get current overlay mode
        overlay_mode = getattr(self, 'segmentation_overlay_mode', 'Mask')
        
        if overlay_mode == "Cluster":
            # Cluster coloring mode
            cluster_map = self._get_cluster_assignments(self.current_acq_id)
            if cluster_map is None or len(cluster_map) == 0:
                # No cluster data available, fall back to regular mask mode
                unique_labels = np.unique(mask)
                if self.current_acq_id not in self.segmentation_colors:
                    self.segmentation_colors[self.current_acq_id] = np.random.rand(len(unique_labels), 3)
                colors = self.segmentation_colors[self.current_acq_id]
                
                for i, label in enumerate(unique_labels):
                    if label == 0:
                        continue
                    cell_mask = (mask == label)
                    overlay[cell_mask, :] = colors[i]
            else:
                # Get unique clusters and generate colors
                unique_clusters = sorted(set(cluster_map.values()))
                
                # Initialize cluster colors if not already set
                if self.current_acq_id not in self.cluster_colors:
                    # Use a colormap to generate distinct colors
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors
                    
                    # Use tab20 colormap for up to 20 clusters, otherwise use a larger colormap
                    if len(unique_clusters) <= 20:
                        cmap = cm.get_cmap('tab20')
                    else:
                        cmap = cm.get_cmap('tab20c')  # Has more colors
                    
                    cluster_colors_dict = {}
                    cluster_color_map_dict = {}
                    
                    for idx, cluster in enumerate(unique_clusters):
                        # Get color from colormap
                        color = cmap(idx % cmap.N)
                        # Convert to RGB (0-1 range)
                        cluster_colors_dict[cluster] = np.array(color[:3])
                        # Store display name
                        cluster_color_map_dict[cluster] = str(cluster)
                    
                    self.cluster_colors[self.current_acq_id] = cluster_colors_dict
                    self.cluster_color_map[self.current_acq_id] = cluster_color_map_dict
                
                cluster_colors = self.cluster_colors[self.current_acq_id]
                
                # Color each cell by its cluster
                unique_labels = np.unique(mask)
                for label in unique_labels:
                    if label == 0:  # Background
                        continue
                    
                    # Get cluster for this cell
                    cluster = cluster_map.get(label, 'Unassigned')
                    if cluster not in cluster_colors:
                        # Use gray for unassigned
                        color = np.array([0.5, 0.5, 0.5])
                    else:
                        color = cluster_colors[cluster]
                    
                    # Fill cell mask with cluster color
                    cell_mask = (mask == label)
                    overlay[cell_mask, :] = color
        else:
            # Regular mask or outline mode
            # Get or generate colors for this acquisition
            unique_labels = np.unique(mask)
            if self.current_acq_id not in self.segmentation_colors:
                # Generate and store colors for this acquisition
                self.segmentation_colors[self.current_acq_id] = np.random.rand(len(unique_labels), 3)
            
            colors = self.segmentation_colors[self.current_acq_id]
            
            if overlay_mode == "Outline":
                # Use efficient edge detection instead of computing full contours
                # Compute all outlines at once by eroding the entire mask
                # Create binary mask (non-zero labels)
                binary_mask = (mask > 0)
                
                # Erode by 1 pixel to get interior regions
                interior = ndi.binary_erosion(binary_mask, structure=np.ones((3, 3)))
                
                # Outline is the difference (original - interior)
                outline_mask = binary_mask & ~interior
                
                # Color outline pixels with their corresponding label colors
                for i, label in enumerate(unique_labels):
                    if label == 0:  # Background
                        continue
                    # Find outline pixels that belong to this label
                    label_outline = outline_mask & (mask == label)
                    overlay[label_outline, :] = colors[i]
            else:  # Mask mode (filled)
                # Fill all pixels of each cell with colors
                for i, label in enumerate(unique_labels):
                    if label == 0:  # Background
                        continue
                    cell_mask = (mask == label)
                    overlay[cell_mask, :] = colors[i]
        
        # Blend with original image
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img
        
        # Normalize images to [0, 1]
        img_norm = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
        overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        
        # Blend (70% original, 30% overlay for subtle outline effect)
        blended = 0.7 * img_norm + 0.3 * overlay_norm
        
        return blended
    
    def _show_cluster_mask_only(self):
        """Show cluster-colored mask without any channel overlay (only for cluster mode)."""
        if (self.current_acq_id not in self.segmentation_masks or
            not self.segmentation_overlay):
            return
        
        mask = self.segmentation_masks[self.current_acq_id]
        
        # Get cluster assignments
        cluster_map = self._get_cluster_assignments(self.current_acq_id)
        if cluster_map is None or len(cluster_map) == 0:
            # No cluster data available, show empty black image
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            black_img = np.zeros((*mask.shape, 3), dtype=np.float32)
            ax.imshow(black_img, interpolation="nearest")
            acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
            ax.set_title(acq_subtitle)
            ax.axis("off")
            self.canvas.draw()
            return
        
        # Create cluster-colored overlay
        overlay = np.zeros((*mask.shape, 3), dtype=np.float32)
        
        # Get unique clusters and generate colors
        unique_clusters = sorted(set(cluster_map.values()))
        
        # Initialize cluster colors if not already set
        if self.current_acq_id not in self.cluster_colors:
            # Use a colormap to generate distinct colors
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            # Use tab20 colormap for up to 20 clusters, otherwise use a larger colormap
            if len(unique_clusters) <= 20:
                cmap = cm.get_cmap('tab20')
            else:
                cmap = cm.get_cmap('tab20c')  # Has more colors
            
            cluster_colors_dict = {}
            cluster_color_map_dict = {}
            
            for idx, cluster in enumerate(unique_clusters):
                # Get color from colormap
                color = cmap(idx % cmap.N)
                # Convert to RGB (0-1 range)
                cluster_colors_dict[cluster] = np.array(color[:3])
                # Store display name
                cluster_color_map_dict[cluster] = str(cluster)
            
            self.cluster_colors[self.current_acq_id] = cluster_colors_dict
            self.cluster_color_map[self.current_acq_id] = cluster_color_map_dict
        
        cluster_colors = self.cluster_colors[self.current_acq_id]
        
        # Color each cell by its cluster
        unique_labels = np.unique(mask)
        for label in unique_labels:
            if label == 0:  # Background
                continue
            
            # Get cluster for this cell
            cluster = cluster_map.get(label, 'Unassigned')
            if cluster not in cluster_colors:
                # Use gray for unassigned
                color = np.array([0.5, 0.5, 0.5])
            else:
                color = cluster_colors[cluster]
            
            # Fill cell mask with cluster color
            cell_mask = (mask == label)
            overlay[cell_mask, :] = color
        
        # Normalize overlay to [0, 1]
        overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
        
        # Display on black background (no channel overlay)
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.imshow(overlay_norm, interpolation="nearest")
        
        # Get acquisition subtitle (ROI title)
        acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
        ax.set_title(acq_subtitle)
        ax.axis("off")
        
        # Draw scale bar if enabled
        if self.scale_bar_chk.isChecked():
            pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
            if pixel_size_um > 0:
                scale_bar_length_um = self.scale_bar_length_spin.value()
                self.canvas._draw_scale_bar_on_axes(mask.shape, scale_bar_length_um, pixel_size_um, ax)
        
        self.canvas.draw()
        
        # Add cluster legend
        self._add_cluster_legend()
    
    def _add_cluster_legend(self):
        """Add cluster legend to the canvas if cluster overlay mode is active."""
        if (not self.segmentation_overlay or 
            self.segmentation_overlay_mode != "Cluster" or
            self.current_acq_id not in self.cluster_color_map):
            return
        
        cluster_color_map = self.cluster_color_map[self.current_acq_id]
        cluster_colors = self.cluster_colors[self.current_acq_id]
        
        if not cluster_color_map:
            return
        
        # Create legend elements
        import matplotlib.patches as mpatches
        legend_elements = []
        
        # Sort clusters for consistent legend order
        sorted_clusters = sorted(cluster_color_map.keys(), key=lambda x: (isinstance(x, str), str(x)))
        
        for cluster in sorted_clusters:
            color = cluster_colors.get(cluster, [0.5, 0.5, 0.5])
            label = cluster_color_map.get(cluster, str(cluster))
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=label)
            )
        
        if not legend_elements:
            return
        
        # Clear any existing legends first
        if hasattr(self.canvas, 'ax') and self.canvas.ax:
            self.canvas.ax.legend_ = None
        if hasattr(self.canvas, 'grid_axes') and self.canvas.grid_axes:
            for ax in self.canvas.grid_axes:
                ax.legend_ = None
        if hasattr(self.canvas, 'fig'):
            # Clear figure-level legends
            for ax in self.canvas.fig.axes:
                if hasattr(ax, 'legend_') and ax.legend_ is not None:
                    ax.legend_.remove()
        
        # Add legend to the right of the image
        # Try to find the main axes (for single image view) or grid axes
        if hasattr(self.canvas, 'ax') and self.canvas.ax:
            # Single image view (RGB composite or single channel)
            # Position legend to the right of the axes
            legend = self.canvas.ax.legend(handles=legend_elements, loc='center left', 
                                          bbox_to_anchor=(1.02, 0.5), frameon=True, 
                                          fontsize=8, title='Clusters', title_fontsize=9,
                                          framealpha=0.9)
            # Adjust subplot to make room for legend on the right
            try:
                self.canvas.fig.subplots_adjust(right=0.85)  # Leave 15% space on right for legend
            except Exception:
                pass
        elif hasattr(self.canvas, 'grid_axes') and self.canvas.grid_axes:
            # Grid view - add legend to the figure, positioned to the right
            if len(self.canvas.grid_axes) > 0:
                # Position legend to the right of the figure
                legend = self.canvas.fig.legend(handles=legend_elements, loc='center left',
                                                bbox_to_anchor=(1.02, 0.5), frameon=True,
                                                fontsize=8, title='Clusters', title_fontsize=9,
                                                framealpha=0.9)
                # Adjust subplot to make room for legend on the right
                try:
                    self.canvas.fig.subplots_adjust(right=0.85)  # Leave 15% space on right for legend
                except Exception:
                    pass
        elif hasattr(self.canvas, 'fig'):
            # Fallback: add legend to figure
            legend = self.canvas.fig.legend(handles=legend_elements, loc='center left',
                                           bbox_to_anchor=(1.02, 0.5), frameon=True,
                                           fontsize=8, title='Clusters', title_fontsize=9,
                                           framealpha=0.9)
            # Adjust subplot to make room for legend on the right
            try:
                self.canvas.fig.subplots_adjust(right=0.85)  # Leave 15% space on right for legend
            except Exception:
                pass
        
        self.canvas.draw()

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
    
    def _preprocess_channels_for_segmentation(self, preprocessing_config: dict, progress_dlg, use_viewer_denoising: bool = False, 
                                            denoise_source: str = "Use viewer settings", custom_denoise_settings: dict = None) -> tuple:
        """Preprocess and combine channels for segmentation."""
        if not preprocessing_config:
            raise ValueError("Preprocessing configuration is required for segmentation")
        
        if not self.current_acq_id:
            raise ValueError("No acquisition selected for segmentation")
        
        # Get the correct loader for the current acquisition
        loader = self._get_loader_for_acquisition(self.current_acq_id)
        if loader is None:
            raise ValueError(f"No loader found for acquisition {self.current_acq_id}")
        
        config = preprocessing_config
        
        # Get nuclear channels
        nuclear_channels = config.get('nuclear_channels', [])
        if not nuclear_channels:
            raise ValueError("No nuclear channels specified in preprocessing configuration")
        
        # Get cytoplasm channels
        cyto_channels = config.get('cyto_channels', [])
        
        # Load and normalize nuclear channels
        if progress_dlg:
            progress_dlg.update_progress(25, "Preprocessing images", "Loading nuclear channels...")
        nuclear_imgs = []
        # Get original acquisition ID if this is a unique ID
        original_acq_id = self._get_original_acq_id(self.current_acq_id)
        for channel in nuclear_channels:
            img = loader.get_image(original_acq_id, channel)
            # Apply denoising based on source selection (always from raw loader image)
            if denoise_source == "viewer" and use_viewer_denoising:
                try:
                    img = self._apply_denoise(channel, img)
                except Exception:
                    pass
            elif denoise_source == "custom" and custom_denoise_settings:
                try:
                    img = self._apply_custom_denoise(channel, img, custom_denoise_settings)
                except Exception:
                    pass
            # Apply normalization if configured
            img = self._apply_normalization(img, config, self.current_acq_id, channel)
            # Ensure 0-1 range after denoising and normalization
            img = self._ensure_0_1_range(img)
            nuclear_imgs.append(img)
        
        # Combine nuclear channels
        nuclear_combo_method = config.get('nuclear_combo_method', 'single')
        nuclear_weights = config.get('nuclear_weights')
        nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights)
        # Ensure combined image is in 0-1 range
        nuclear_img = self._ensure_0_1_range(nuclear_img)
        
        # Load and normalize cytoplasm channels
        cyto_img = None
        if cyto_channels:
            if progress_dlg:
                progress_dlg.update_progress(35, "Preprocessing images", "Loading cytoplasm channels...")
            cyto_imgs = []
            for channel in cyto_channels:
                img = loader.get_image(original_acq_id, channel)
                # Apply denoising based on source selection (always from raw loader image)
                if denoise_source == "viewer" and use_viewer_denoising:
                    try:
                        img = self._apply_denoise(channel, img)
                    except Exception:
                        pass
                elif denoise_source == "custom" and custom_denoise_settings:
                    try:
                        img = self._apply_custom_denoise(channel, img, custom_denoise_settings)
                    except Exception:
                        pass
                # Apply normalization if configured
                img = self._apply_normalization(img, config, self.current_acq_id, channel)
                # Ensure 0-1 range after denoising and normalization
                img = self._ensure_0_1_range(img)
                cyto_imgs.append(img)
            
            # Combine cytoplasm channels
            cyto_combo_method = config.get('cyto_combo_method', 'single')
            cyto_weights = config.get('cyto_weights')
            cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights)
            # Ensure combined image is in 0-1 range
            cyto_img = self._ensure_0_1_range(cyto_img)
        
        return nuclear_img, cyto_img
    
    def _apply_normalization(self, img: np.ndarray, config: dict, acq_id: str, channel: str) -> np.ndarray:
        """Apply normalization to an image based on configuration."""
        norm_method = config.get('normalization_method', 'None')
        
        if norm_method == 'None':
            return img
        
        # Check cache first
        cache_key = f"{acq_id}_{channel}_{norm_method}"
        if norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 10.0)
            cache_key += f"_{cofactor}"
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            cache_key += f"_{p_low}_{p_high}"
        # channelwise_minmax doesn't need extra cache key params
        
        # Apply normalization
        if norm_method == 'channelwise_minmax':
            return channelwise_minmax_normalize(img)
        elif norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 10.0)
            return arcsinh_normalize(img, cofactor)
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            return percentile_clip_normalize(img, p_low, p_high)
        
        return img
    
    def _ensure_0_1_range(self, img: np.ndarray) -> np.ndarray:
        """Ensure image is normalized to 0-1 range using min-max scaling."""
        img_float = img.astype(np.float32, copy=True)
        vmin = np.min(img_float)
        vmax = np.max(img_float)
        if vmax > vmin:
            normalized = (img_float - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(img_float)
        return normalized
    
    def _update_segmentation_overlay_text(self):
        """Update the segmentation overlay checkbox text with current cell count."""
        if self.segmentation_overlay and self.current_acq_id and self.current_acq_id in self.segmentation_masks:
            cell_count = len(np.unique(self.segmentation_masks[self.current_acq_id])) - 1
            self.segmentation_overlay_chk.setText(f"Show segmentation overlay ({cell_count} cells)")
        else:
            self.segmentation_overlay_chk.setText("Show segmentation overlay")
    
    def _on_segmentation_overlay_toggled(self):
        """Handle segmentation overlay checkbox toggle."""
        self.segmentation_overlay = self.segmentation_overlay_chk.isChecked()
        
        # Show/hide mode selector based on overlay state
        self.segmentation_overlay_mode_widget.setVisible(self.segmentation_overlay)
        
        # Update checkbox text
        self._update_segmentation_overlay_text()
        
        # Update cluster mode availability
        self._update_cluster_mode_availability()
        
        # Update display if we have segmentation masks
        if self.current_acq_id in self.segmentation_masks:
            self.preserve_zoom = True
            self._view_selected()
    
    def _on_segmentation_overlay_mode_changed(self, mode: str):
        """Handle segmentation overlay mode change (Outline vs Mask)."""
        self.segmentation_overlay_mode = mode
        
        # Update display if overlay is enabled and we have segmentation masks
        if self.segmentation_overlay and self.current_acq_id in self.segmentation_masks:
            self.preserve_zoom = True
            self._view_selected()
    
    def _on_scale_bar_toggled(self, checked):
        """Handle scale bar checkbox toggle."""
        self.scale_bar_widget.setVisible(checked)
        self._view_selected()
    
    def _on_scale_bar_changed(self):
        """Handle scale bar length change."""
        self._view_selected()
    
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
        
        # Get preprocessing parameters
        normalization_config = dlg.get_normalization_config()
        denoise_source = dlg.get_denoise_source()
        custom_denoise_settings = dlg.get_custom_denoise_settings()
        spillover_config = dlg.get_spillover_config()
        
        # Get excluded channels
        excluded_channels = dlg.get_excluded_channels()
        
        # Store the normalization configuration for later use in clustering
        self.feature_extraction_config = {
            'normalization_config': normalization_config,
            'denoise_source': denoise_source,
            'custom_denoise_settings': custom_denoise_settings,
            'spillover_config': spillover_config
        }
        
        if not selected_acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions selected", "Please select at least one acquisition.")
            return
        
        if not any(selected_features.values()):
            QtWidgets.QMessageBox.warning(self, "No features selected", "Please select at least one feature to extract.")
            return
        
        # Perform feature extraction
        try:
            self._perform_feature_extraction(selected_acquisitions, selected_features, output_path, normalization_config, denoise_source, custom_denoise_settings, spillover_config, excluded_channels)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Feature Extraction Failed", 
                f"Feature extraction failed with error:\n{str(e)}"
            )
    

    def _perform_feature_extraction(self, selected_acquisitions, selected_features, output_path, normalization_config, denoise_source, custom_denoise_settings, spillover_config=None, excluded_channels=None):
        """Perform the actual feature extraction using multiprocessing.
        
        This now parallelizes both image loading and feature extraction for better performance.
        """
        # Create progress dialog
        progress_dlg = ProgressDialog("Feature Extraction", self)
        progress_dlg.set_maximum(len(selected_acquisitions))  # One phase: loading + extraction combined
        progress_dlg.show()
        
        try:
            # Prepare arguments for multiprocessing
            mp_args = []
            
            for acq_id in selected_acquisitions:
                try:
                    current_acq_info = self._get_acquisition_info(acq_id)
                    if current_acq_info is None:
                        print(f"[main] Acquisition {acq_id} not found, skipping")
                        continue
                    mask = self.segmentation_masks[acq_id]
                    # Prepare preprocessing parameters
                    arcsinh_enabled = normalization_config is not None and normalization_config.get('method') == 'arcsinh'
                    cofactor = normalization_config.get('cofactor', 10.0) if normalization_config else 10.0
                    
                    # Convert AcquisitionInfo to dictionary for pickling
                    acq_info_dict = {
                        'channels': current_acq_info.channels,
                        'name': current_acq_info.name,
                        'well': current_acq_info.well,
                        'id': current_acq_info.id
                    }
                    
                    # Get source file path and determine loader type
                    source_file = current_acq_info.source_file if hasattr(current_acq_info, 'source_file') else None
                    
                    # Determine file path and loader type
                    if acq_id in self.acq_to_file:
                        file_path = self.acq_to_file[acq_id]
                        loader_type = "mcd"
                    elif self.current_path:
                        if os.path.isdir(self.current_path):
                            file_path = self.current_path
                            loader_type = "ometiff"
                        else:
                            file_path = self.current_path
                            loader_type = "mcd"
                    else:
                        print(f"[main] Cannot determine file path for acquisition {acq_id}, skipping")
                        continue
                    
                    # Get original acquisition ID if this is a unique ID
                    original_acq_id = self._get_original_acq_id(acq_id)
                    # Use well name for acquisition label if available, otherwise use acquisition name
                    acq_label = current_acq_info.well if current_acq_info.well else current_acq_info.name
                    mp_args.append((
                        original_acq_id,  # Use original ID for loader
                        mask, 
                        selected_features, 
                        acq_info_dict, 
                        acq_label,  # acq_label - use well name if available
                        file_path,  # file path for loading
                        loader_type,  # "mcd" or "ometiff"
                        arcsinh_enabled, 
                        cofactor,
                        denoise_source,
                        custom_denoise_settings,
                        spillover_config,  # spillover correction config
                        source_file,
                        excluded_channels  # excluded channels set
                    ))
                except Exception as e:
                    print(f"[main] Error preparing arguments for {acq_id}: {e}")
                    continue
            
            if not mp_args:
                QtWidgets.QMessageBox.warning(self, "No valid acquisitions", "No valid acquisitions found for feature extraction.")
                return
            
            # Use multiprocessing for parallel loading and feature extraction
            max_workers = min(mp.cpu_count(), len(mp_args))
            progress_dlg.update_progress(0, "Starting parallel loading and extraction", f"Using {max_workers} CPU cores")
            
            all_features = []
            try:
                with mp.Pool(processes=max_workers) as pool:
                    # Submit all tasks - each worker loads and extracts in parallel
                    futures = []
                    for (acq_id, mask, selected_features, acq_info_dict, acq_label, file_path, loader_type, arcsinh_enabled, cofactor, denoise_source, custom_denoise_settings, spillover_config, source_file, excluded_channels) in mp_args:
                        future = pool.apply_async(
                            load_and_extract_features,
                            (acq_id, mask, selected_features, acq_info_dict, acq_label, file_path, loader_type, arcsinh_enabled, cofactor, denoise_source, custom_denoise_settings, spillover_config, source_file, excluded_channels)
                        )
                        futures.append((acq_id, future))
                    
                    # Collect results as they complete
                    completed = 0
                    for acq_id, future in futures:
                        if progress_dlg.is_cancelled():
                            break
                        try:
                            result = future.get(timeout=600)  # 10 minute timeout per acquisition (includes loading)
                            if result is not None and not result.empty:
                                all_features.append(result)
                        except Exception as e:
                            print(f"Feature extraction failed for acquisition {acq_id}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                        
                        completed += 1
                        progress_dlg.update_progress(
                            completed,
                            f"Processed acquisition {completed}/{len(futures)}",
                            f"Extracted features from {len(all_features)} acquisitions"
                        )
                        
            except Exception as mp_error:
                print(f"Multiprocessing failed, falling back to sequential processing: {mp_error}")
                import traceback
                traceback.print_exc()
                progress_dlg.update_progress(0, "Multiprocessing failed, using sequential processing", "Processing acquisitions one by one")
                
                # Fallback to sequential processing
                for i, (acq_id, mask, selected_features, acq_info_dict, acq_label, file_path, loader_type, arcsinh_enabled, cofactor, denoise_source, custom_denoise_settings, spillover_config, source_file) in enumerate(mp_args):
                    if progress_dlg.is_cancelled():
                        break
                    
                    try:
                        result = load_and_extract_features(
                            acq_id, mask, selected_features, acq_info_dict, acq_label, file_path, loader_type, arcsinh_enabled, cofactor, denoise_source, custom_denoise_settings, spillover_config, source_file
                        )
                        
                        if result is not None and not result.empty:
                            all_features.append(result)
                            
                    except Exception as e:
                        print(f"Feature extraction failed for acquisition {acq_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
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
            
            # Update cluster mode availability after feature extraction
            self._update_cluster_mode_availability()
            
            # Log feature extraction operation
            logger = get_logger()
            features_extracted = [k for k, v in selected_features.items() if v]
            params = {
                "normalization_method": normalization_config.get('method') if normalization_config else None,
                "arcsinh_cofactor": normalization_config.get('cofactor') if normalization_config and normalization_config.get('method') == 'arcsinh' else None,
                "denoise_source": denoise_source
            }
            if denoise_source == "custom" and custom_denoise_settings:
                params["denoise_settings"] = custom_denoise_settings
            
            # Get source file name
            source_file = os.path.basename(self.current_path) if self.current_path else None
            
            logger.log_feature_extraction(
                parameters=params,
                acquisitions=selected_acquisitions,
                features_extracted=features_extracted,
                output_path=output_path,
                notes=f"Extracted features from {len(combined_features)} cells across {len(selected_acquisitions)} acquisitions",
                source_file=source_file
            )
            
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
                   'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count',
                   'centroid_x', 'centroid_y']:
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
            
            if selected_features['centroid_x']:
                # X coordinate (column) of centroid in pixels
                features['centroid_x'].append(prop.centroid[1])
            
            if selected_features['centroid_y']:
                # Y coordinate (row) of centroid in pixels
                features['centroid_y'].append(prop.centroid[0])
        
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
        """Load previously saved segmentation masks from a directory for all ROIs."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions available. Please load a file first.")
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
        
        # Load masks for all acquisitions
        loaded_count = 0
        failed_count = 0
        missing_acqs = []  # List of tuples: (acquisition_name, source_file_name)
        total_cells = 0
        
        for acq_info in self.acquisitions:
            # Only use new format with source file prefix
            if not acq_info.source_file:
                # Skip acquisitions without source_file (shouldn't happen in multi-file mode)
                source_file_name = "Unknown"
                missing_acqs.append((acq_info.name, source_file_name))
                continue
            
            source_basename = os.path.splitext(os.path.basename(acq_info.source_file))[0]
            safe_source = self._sanitize_filename(source_basename)
            
            # Try different possible filenames with source file prefix
            # First try well name, then fall back to acquisition name
            possible_filenames = []
            
            # Try well name first (if available)
            if acq_info.well:
                safe_well = self._sanitize_filename(acq_info.well)
                possible_filenames.append(f"{safe_source}_{safe_well}_segmentation.tiff")
                possible_filenames.append(f"{safe_source}_{safe_well}_segmentation.tif")
                possible_filenames.append(f"{safe_source}_{safe_well}_segmentation_masks.tiff")
                possible_filenames.append(f"{safe_source}_{safe_well}_segmentation_masks.tif")
            
            # Fall back to acquisition name
            safe_name = self._sanitize_filename(acq_info.name)
            possible_filenames.append(f"{safe_source}_{safe_name}_segmentation.tiff")
            possible_filenames.append(f"{safe_source}_{safe_name}_segmentation.tif")
            possible_filenames.append(f"{safe_source}_{safe_name}_segmentation_masks.tiff")
            possible_filenames.append(f"{safe_source}_{safe_name}_segmentation_masks.tif")
            
            # Find the first existing mask file
            mask_file = None
            for filename in possible_filenames:
                filepath = os.path.join(masks_dir, filename)
                if os.path.exists(filepath):
                    mask_file = filepath
                    break
            
            if not mask_file:
                # Store both acquisition name and source file for better reporting
                source_file_name = os.path.basename(acq_info.source_file)
                missing_acqs.append((acq_info.name, source_file_name))
                continue
            
            try:
                # Load the mask file
                if _HAVE_TIFFFILE:
                    mask = tifffile.imread(mask_file)
                else:
                    # Fallback to PIL if tifffile not available
                    from PIL import Image
                    mask = np.array(Image.open(mask_file))
                
                # Store the loaded mask
                self.segmentation_masks[acq_info.id] = mask
                # Clear colors for this acquisition so they get regenerated
                if acq_info.id in self.segmentation_colors:
                    del self.segmentation_colors[acq_info.id]
                if acq_info.id in self.cluster_colors:
                    del self.cluster_colors[acq_info.id]
                if acq_info.id in self.cluster_color_map:
                    del self.cluster_color_map[acq_info.id]
                
                loaded_count += 1
                cell_count = len(np.unique(mask)) - 1  # Subtract 1 for background
                total_cells += cell_count
                
            except Exception as e:
                failed_count += 1
                print(f"Error loading mask for {acq_info.name}: {e}")
                continue
        
        # Enable overlay if any masks were loaded
        if loaded_count > 0:
            self.segmentation_overlay = True
            self.segmentation_overlay_chk.setChecked(True)
            self.segmentation_overlay_mode_widget.setVisible(True)
            # Update overlay text with cell count for current acquisition
            self._update_segmentation_overlay_text()
            # Update display
            self._view_selected()
        
        # Show summary message
        message_parts = []
        if loaded_count > 0:
            message_parts.append(f"Successfully loaded {loaded_count} mask file(s).")
            message_parts.append(f"Total cells found: {total_cells}")
            message_parts.append(f"Overlay is now enabled.")
        else:
            message_parts.append("No mask files were loaded.")
        
        if missing_acqs:
            message_parts.append(f"\nNo mask files found for {len(missing_acqs)} acquisition(s).")
            if len(missing_acqs) <= 10:
                message_parts.append("Missing acquisitions:")
                for acq_info in missing_acqs:
                    if isinstance(acq_info, tuple):
                        acq_name, source_file = acq_info
                        message_parts.append(f"  • {acq_name} [{source_file}]")
                    else:
                        # Backward compatibility for old format
                        message_parts.append(f"  • {acq_info}")
            else:
                message_parts.append(f"Missing acquisitions (first 10):")
                for acq_info in missing_acqs[:10]:
                    if isinstance(acq_info, tuple):
                        acq_name, source_file = acq_info
                        message_parts.append(f"  • {acq_name} [{source_file}]")
                    else:
                        # Backward compatibility for old format
                        message_parts.append(f"  • {acq_info}")
                message_parts.append(f"  ... and {len(missing_acqs) - 10} more")
        
        if failed_count > 0:
            message_parts.append(f"\nFailed to load {failed_count} mask file(s) due to errors.")
        
        QtWidgets.QMessageBox.information(
            self, 
            "Mask Loading Complete", 
            "\n".join(message_parts)
        )

    def _update_sidebar_max_width(self):
        """Update the sidebar maximum width based on current window width."""
        if hasattr(self, 'left_scroll') and hasattr(self, 'splitter'):
            window_width = self.width()
            # Maximum width is 30% of window width, but at least the minimum width
            max_width = max(self.sidebar_min_width, int(window_width * 0.30))
            # Update maximum width constraint
            self.left_scroll.setMaximumWidth(max_width)
    
    def _on_splitter_moved(self, pos, index):
        """Handle splitter movement to enforce maximum width constraint."""
        if index == 0 and hasattr(self, 'left_scroll') and hasattr(self, 'splitter'):
            window_width = self.width()
            max_width = max(self.sidebar_min_width, int(window_width * 0.30))
            current_width = self.left_scroll.width()
            # If current width exceeds max, constrain it
            if current_width > max_width:
                sizes = self.splitter.sizes()
                if len(sizes) >= 2:
                    sizes[0] = max_width
                    sizes[1] = window_width - max_width - self.splitter.handleWidth()
                    self.splitter.setSizes(sizes)
    
    def resizeEvent(self, event):
        """Handle window resize to update sidebar maximum width constraint."""
        super().resizeEvent(event)
        if hasattr(self, 'left_scroll') and hasattr(self, 'splitter'):
            window_width = self.width()
            # Maximum width is 30% of window width, but at least the minimum width
            max_width = max(self.sidebar_min_width, int(window_width * 0.30))
            current_width = self.left_scroll.width()
            # If current width exceeds max, constrain it
            if current_width > max_width:
                # Get current splitter sizes
                sizes = self.splitter.sizes()
                if len(sizes) >= 2:
                    # Constrain the left panel to max_width
                    sizes[0] = max_width
                    sizes[1] = window_width - max_width - self.splitter.handleWidth()
                    self.splitter.setSizes(sizes)
            # Update maximum width constraint
            self.left_scroll.setMaximumWidth(max_width)
    
    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.loader:
            self.loader.close()
        # Delete LLM cache when closing the app
        pass
        event.accept()

    def _open_clustering_dialog(self):
        """Open the cell clustering analysis dialog."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(
                self, 
                "No Feature Data", 
                "No feature data available. Please:\n"
                "- Extract features using the 'Extract Features' button, or\n"
                "- Load features using 'Analysis > Load Feature File'"
            )
            return
        
        # Check if dialog already exists and is still valid
        if self.clustering_dialog is not None:
            try:
                # Check if dialog still exists (hasn't been deleted)
                # If dialog is visible, just bring it to front
                # If dialog is not visible, show it
                if self.clustering_dialog.isVisible():
                    self.clustering_dialog.raise_()
                    self.clustering_dialog.activateWindow()
                else:
                    self.clustering_dialog.show()
                    self.clustering_dialog.raise_()
                    self.clustering_dialog.activateWindow()
                return
            except (RuntimeError, AttributeError):
                # Dialog was deleted, set to None
                self.clustering_dialog = None
        
        # Get normalization configuration from feature extraction
        normalization_config = None
        if hasattr(self, 'feature_extraction_config') and self.feature_extraction_config:
            normalization_config = self.feature_extraction_config.get('normalization_config')
        
        # Create new clustering dialog with both original and batch-corrected features
        self.clustering_dialog = CellClusteringDialog(
            self.feature_dataframe, 
            normalization_config, 
            batch_corrected_dataframe=self.batch_corrected_dataframe,
            parent=self
        )
        # Make dialog non-modal so it can be closed and reopened without losing state
        self.clustering_dialog.setModal(False)
        # Prevent dialog from being deleted when closed, so we can reopen it
        self.clustering_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        self.clustering_dialog.show()
        
        # Connect to dialog's close event to update main window dataframes
        def on_dialog_closed():
            """Update main window dataframes when dialog is closed."""
            if hasattr(self.clustering_dialog, 'clustered_data') and self.clustering_dialog.clustered_data is not None:
                # Get all cluster-related columns to copy
                cluster_cols = [col for col in self.clustering_dialog.clustered_data.columns 
                              if col in ['cluster', 'cluster_id', 'cluster_phenotype'] 
                              or col.startswith('cluster_')]
                
                if cluster_cols:
                    # Helper function to update a dataframe with cluster assignments
                    def update_dataframe_with_clusters(target_df, source_df, cluster_cols):
                        """Update target dataframe with cluster columns from source."""
                        for col in cluster_cols:
                            if col not in target_df.columns:
                                # Initialize with default value (0 for numeric, empty string for string)
                                if col in ['cluster', 'cluster_id']:
                                    target_df[col] = 0
                                else:
                                    target_df[col] = ''
                        
                        # Match by index if possible
                        if source_df.index.isin(target_df.index).any():
                            matching_indices = source_df.index.intersection(target_df.index)
                            for col in cluster_cols:
                                target_df.loc[matching_indices, col] = source_df.loc[matching_indices, col].values
                        # Otherwise match by cell_id
                        elif 'cell_id' in source_df.columns and 'cell_id' in target_df.columns:
                            # Create mapping from cell_id to cluster values
                            for col in cluster_cols:
                                cluster_map = dict(zip(source_df['cell_id'], source_df[col]))
                                mask = target_df['cell_id'].isin(cluster_map.keys())
                                target_df.loc[mask, col] = target_df.loc[mask, 'cell_id'].map(cluster_map)
                    
                    # Update original feature dataframe
                    update_dataframe_with_clusters(self.feature_dataframe, self.clustering_dialog.clustered_data, cluster_cols)
                    
                    # Update batch-corrected dataframe if it exists
                    if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
                        update_dataframe_with_clusters(self.batch_corrected_dataframe, self.clustering_dialog.clustered_data, cluster_cols)
                    
                    # Update cluster mode availability after cluster data is added
                    # Use a small delay to ensure the update happens after the dialog is fully closed
                    QTimer.singleShot(100, self._update_cluster_mode_availability)
                    
                    # Notify spatial analysis dialogs that clusters have changed
                    # Use a small delay to ensure the update happens after the dialog is fully closed
                    QTimer.singleShot(200, self._notify_spatial_dialogs_clusters_changed)
        
        # Connect close event
        self.clustering_dialog.finished.connect(on_dialog_closed)

    def _open_spatial_dialog(self):
        """Open the spatial analysis dialog (defaults to Simple)."""
        # Default to Simple Spatial Analysis
        self._open_simple_spatial_dialog()
    
    def _notify_spatial_dialogs_clusters_changed(self):
        """Notify open spatial analysis dialogs that clusters have changed."""
        # Notify simple spatial dialog if it exists and is visible
        if hasattr(self, 'simple_spatial_dialog') and self.simple_spatial_dialog is not None:
            try:
                if self.simple_spatial_dialog.isVisible() and hasattr(self.simple_spatial_dialog, 'on_clusters_changed'):
                    self.simple_spatial_dialog.on_clusters_changed()
            except (RuntimeError, AttributeError):
                pass
        
        # Notify advanced spatial dialog if it exists and is visible
        if hasattr(self, 'advanced_spatial_dialog') and self.advanced_spatial_dialog is not None:
            try:
                if self.advanced_spatial_dialog.isVisible() and hasattr(self.advanced_spatial_dialog, 'on_clusters_changed'):
                    self.advanced_spatial_dialog.on_clusters_changed()
            except (RuntimeError, AttributeError):
                pass
    
    def _open_simple_spatial_dialog(self):
        """Open the simple spatial analysis dialog."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "No Feature Data",
                "No feature data available. Please:\n"
                "- Extract features using the 'Extract Features' button, or\n"
                "- Load features using 'Analysis > Load Feature File'"
            )
            return
        
        # Check if cluster column exists
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        if cluster_col is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Cluster Data",
                "Spatial analysis requires cluster assignments.\n\n"
                "Please perform clustering first using:\n"
                "- 'Analysis > Cell Clustering'"
            )
            return
        
        # Check if dialog already exists and is still valid
        if hasattr(self, 'simple_spatial_dialog') and self.simple_spatial_dialog is not None:
            try:
                if self.simple_spatial_dialog.isVisible():
                    self.simple_spatial_dialog.raise_()
                    self.simple_spatial_dialog.activateWindow()
                else:
                    self.simple_spatial_dialog.show()
                    self.simple_spatial_dialog.raise_()
                    self.simple_spatial_dialog.activateWindow()
                return
            except (RuntimeError, AttributeError):
                self.simple_spatial_dialog = None
        
        # Create new simple spatial analysis dialog
        self.simple_spatial_dialog = SimpleSpatialAnalysisDialog(
            self.feature_dataframe, 
            batch_corrected_dataframe=self.batch_corrected_dataframe,
            parent=self
        )
        self.simple_spatial_dialog.setModal(False)
        self.simple_spatial_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        self.simple_spatial_dialog.show()
    
    def _open_advanced_spatial_dialog(self):
        """Open the advanced spatial analysis dialog (squidpy)."""
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            QtWidgets.QMessageBox.warning(
                self,
                "No Feature Data",
                "No feature data available. Please:\n"
                "- Extract features using the 'Extract Features' button, or\n"
                "- Load features using 'Analysis > Load Feature File'"
            )
            return
        
        # Check if cluster column exists
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in self.feature_dataframe.columns:
                cluster_col = col
                break
        
        if cluster_col is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Cluster Data",
                "Spatial analysis requires cluster assignments.\n\n"
                "Please perform clustering first using:\n"
                "- 'Analysis > Cell Clustering'"
            )
            return
        
        # Check if AdvancedSpatialAnalysisDialog is available
        if AdvancedSpatialAnalysisDialog is None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Squidpy Not Available",
                "Advanced Spatial Analysis requires squidpy, which is not installed.\n\n"
                "Would you like to open Simple Spatial Analysis instead?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self._open_simple_spatial_dialog()
            return
        
        # Check if dialog already exists and is still valid
        if hasattr(self, 'advanced_spatial_dialog') and self.advanced_spatial_dialog is not None:
            try:
                if self.advanced_spatial_dialog.isVisible():
                    self.advanced_spatial_dialog.raise_()
                    self.advanced_spatial_dialog.activateWindow()
                else:
                    self.advanced_spatial_dialog.show()
                    self.advanced_spatial_dialog.raise_()
                    self.advanced_spatial_dialog.activateWindow()
                return
            except (RuntimeError, AttributeError):
                self.advanced_spatial_dialog = None
        
        # Create new advanced spatial analysis dialog
        self.advanced_spatial_dialog = AdvancedSpatialAnalysisDialog(
            self.feature_dataframe, 
            batch_corrected_dataframe=self.batch_corrected_dataframe,
            parent=self
        )
        self.advanced_spatial_dialog.setModal(False)
        self.advanced_spatial_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        self.advanced_spatial_dialog.show()
    
    def _open_qc_dialog(self):
        """Open the QC analysis dialog."""
        if self.loader is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load data first before running QC analysis."
            )
            return
        dlg = QCAnalysisDialog(self)
        dlg.exec_()
    
    def _open_pixel_correlation_dialog(self):
        """Open the pixel-level correlation analysis dialog."""
        # Check if data is loaded (either single file loader or multiple MCD files)
        has_data = (self.loader is not None) or (len(self.mcd_loaders) > 0) or (len(self.acquisitions) > 0)
        
        if not has_data:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load data first before running pixel-level correlation analysis."
            )
            return
        dlg = PixelCorrelationDialog(self)
        dlg.exec_()
    
    def _open_deconvolution_dialog(self):
        """Open the high resolution deconvolution dialog."""
        # Check if we have data loaded (MCD or OME-TIFF)
        has_data = False
        
        # Check for single loader (MCD or OME-TIFF)
        if isinstance(self.loader, (MCDLoader, OMETIFFLoader)):
            has_data = True
        # Check for multiple MCD files
        elif hasattr(self, 'mcd_loaders') and self.mcd_loaders:
            has_data = True
        # Check if current path is an MCD file or directory
        elif self.current_path:
            if os.path.isfile(self.current_path) and self.current_path.lower().endswith('.mcd'):
                has_data = True
            elif os.path.isdir(self.current_path):
                has_data = True
        
        if not has_data:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Loaded",
                "Deconvolution requires MCD files or OME-TIFF directories. Please load data first."
            )
            return
        
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(
                self,
                "No Acquisitions",
                "No acquisitions found. Please load data with acquisitions."
            )
            return
        
        # Open deconvolution dialog
        dlg = DeconvolutionDialog(self.acquisitions, self.current_acq_id, self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        acq_type = dlg.get_acq_type()
        output_dir = dlg.get_output_directory()
        x0 = dlg.get_x0()
        iterations = dlg.get_iterations()
        output_format = dlg.get_output_format()
        
        # Create and show progress dialog
        progress_dlg = ProgressDialog("High Resolution Deconvolution", self)
        progress_dlg.show()
        
        try:
            if acq_type == "single":
                success = self._deconvolve_single_acquisition(
                    output_dir, progress_dlg, x0, iterations, output_format
                )
            else:
                success = self._deconvolve_whole_slide(
                    output_dir, progress_dlg, x0, iterations, output_format
                )
            
            progress_dlg.close()
            
            if success and not progress_dlg.is_cancelled():
                # Ask user if they want to switch to the deconvolved images
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Deconvolution Complete",
                    f"Successfully deconvolved images saved to:\n{output_dir}\n\n"
                    "Would you like to load the deconvolved images now?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                
                if reply == QtWidgets.QMessageBox.Yes:
                    # Switch to the deconvolved OME-TIFF folder
                    self._load_deconvolved_images(output_dir)
            elif progress_dlg.is_cancelled():
                QtWidgets.QMessageBox.information(
                    self,
                    "Deconvolution Cancelled",
                    "Deconvolution was cancelled by user."
                )
        except Exception as e:
            progress_dlg.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Deconvolution Failed",
                f"Deconvolution failed with error:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _deconvolve_single_acquisition(self, output_dir: str, progress_dlg: ProgressDialog,
                                       x0: float, iterations: int, output_format: str) -> bool:
        """Deconvolve the currently selected acquisition."""
        if not self.current_acq_id:
            raise ValueError("No acquisition selected")
        
        acq_info = self._get_acquisition_info(self.current_acq_id)
        if acq_info is None:
            raise ValueError(f"Acquisition {self.current_acq_id} not found")
        
        # Get the loader and determine type
        loader = self._get_loader_for_acquisition(self.current_acq_id)
        if loader is None:
            raise ValueError(f"No loader found for acquisition {self.current_acq_id}")
        
        # Get original acquisition ID (for multiple files, this maps unique ID to original)
        original_acq_id = self._get_original_acq_id(self.current_acq_id)
        
        # Get channel names
        channel_names = loader.get_channels(original_acq_id)
        
        # Determine data path and loader type
        data_path = None
        loader_type = None
        source_file_path = None
        
        if isinstance(loader, MCDLoader):
            loader_type = "mcd"
            # Get the MCD file path - check acq_to_file mapping first (for multiple files)
            if self.current_acq_id in self.acq_to_file:
                data_path = self.acq_to_file[self.current_acq_id]
            elif acq_info.source_file:
                data_path = acq_info.source_file
            elif self.current_path and os.path.isfile(self.current_path) and self.current_path.lower().endswith('.mcd'):
                data_path = self.current_path
            else:
                # Try to get path from the MCD file object
                if hasattr(loader, 'mcd') and loader.mcd:
                    if hasattr(loader.mcd, 'path'):
                        data_path = loader.mcd.path
                    elif hasattr(loader.mcd, 'filename'):
                        data_path = loader.mcd.filename
            
            if not data_path or not os.path.isfile(data_path):
                raise ValueError(f"Could not determine MCD file path for acquisition {self.current_acq_id}")
            source_file_path = data_path
            
        elif isinstance(loader, OMETIFFLoader):
            loader_type = "ometiff"
            # For OME-TIFF, get the file path from the loader's _acq_map
            if hasattr(loader, '_acq_map') and original_acq_id in loader._acq_map:
                data_path = loader._acq_map[original_acq_id]
            elif acq_info.source_file:
                data_path = acq_info.source_file
            elif self.current_path and os.path.isdir(self.current_path):
                # Try to find the file in the directory
                import glob
                tiff_files = glob.glob(os.path.join(self.current_path, "*.ome.tif"))
                tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.ome.tiff")))
                tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.tif")))
                tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.tiff")))
                # Try to match by acquisition name or ID
                for tiff_file in tiff_files:
                    if acq_info.name in os.path.basename(tiff_file):
                        data_path = tiff_file
                        break
                if not data_path and tiff_files:
                    # Fallback to first file if we can't match
                    data_path = tiff_files[0]
            
            if not data_path or not os.path.isfile(data_path):
                raise ValueError(f"Could not determine OME-TIFF file path for acquisition {self.current_acq_id}")
            
            # For OME-TIFF, source_file_path is the directory containing the files
            if self.current_path and os.path.isdir(self.current_path):
                source_file_path = self.current_path
            else:
                source_file_path = os.path.dirname(data_path)
        
        if not data_path or loader_type is None:
            raise ValueError(f"Could not determine data path or loader type for acquisition {self.current_acq_id}")
        
        progress_dlg.set_maximum(100)
        progress_dlg.update_progress(0, f"Deconvolving {acq_info.name}", "Starting deconvolution...")
        
        try:
            # Deconvolve the acquisition
            progress_dlg.update_progress(10, f"Deconvolving {acq_info.name}", "Loading image data...")
            
            output_path = deconvolve_acquisition(
                data_path=data_path,
                acq_id=original_acq_id,
                output_dir=output_dir,
                x0=x0,
                iterations=iterations,
                output_format=output_format,
                channel_names=channel_names,
                source_file_path=source_file_path,  # Pass source file/dir for filename generation
                unique_acq_id=self.current_acq_id,  # Pass unique ID for filename
                loader_type=loader_type,
                channel_format=self.ometiff_channel_format if isinstance(loader, OMETIFFLoader) else 'CHW',
                well_name=acq_info.well  # Pass well name if available
            )
            
            progress_dlg.update_progress(100, "Complete", f"Saved to {os.path.basename(output_path)}")
            
            # Log deconvolution
            logger = get_logger()
            # Get source file name for logging
            source_file_name = None
            if isinstance(loader, MCDLoader):
                source_file_name = os.path.basename(source_file_path) if source_file_path else None
            elif isinstance(loader, OMETIFFLoader):
                # For OME-TIFF, use the folder name
                source_file_name = os.path.basename(source_file_path) if source_file_path else None
            
            params = {
                "x0": x0,
                "iterations": iterations,
                "output_format": output_format,
                "loader_type": loader_type
            }
            
            logger._write_entry(
                entry_type="deconvolution",
                operation="high_resolution_deconvolution",
                parameters=params,
                acquisitions=[self.current_acq_id],
                output_path=output_path,
                notes=f"High resolution deconvolution: {acq_info.name}",
                source_file=source_file_name
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed: {str(e)}"
            progress_dlg.update_progress(0, "Error", error_msg)
            import traceback
            print(f"Deconvolution error: {traceback.format_exc()}")
            raise
    
    def _deconvolve_whole_slide(self, output_dir: str, progress_dlg: ProgressDialog,
                                x0: float, iterations: int, output_format: str) -> bool:
        """Deconvolve all acquisitions."""
        if not self.acquisitions:
            raise ValueError("No acquisitions found")
        
        total_acqs = len(self.acquisitions)
        progress_dlg.set_maximum(total_acqs * 100)
        
        processed = 0
        for acq in self.acquisitions:
            if progress_dlg.is_cancelled():
                return False
            
            progress_dlg.update_progress(
                processed * 100,
                f"Deconvolving {acq.name}",
                f"Processing acquisition {processed + 1} of {total_acqs}"
            )
            
            try:
                # Get the loader and determine type
                loader = self._get_loader_for_acquisition(acq.id)
                if loader is None:
                    raise ValueError(f"No loader found for acquisition {acq.id}")
                
                # Get original acquisition ID (for multiple files)
                original_acq_id = self._get_original_acq_id(acq.id)
                
                # Get channel names
                channel_names = loader.get_channels(original_acq_id) if loader else acq.channels
                
                # Determine data path and loader type
                data_path = None
                loader_type = None
                source_file_path = None
                
                if isinstance(loader, MCDLoader):
                    loader_type = "mcd"
                    # Get the MCD file path
                    if acq.id in self.acq_to_file:
                        data_path = self.acq_to_file[acq.id]
                    elif acq.source_file:
                        data_path = acq.source_file
                    elif self.current_path and os.path.isfile(self.current_path) and self.current_path.lower().endswith('.mcd'):
                        data_path = self.current_path
                    
                    if not data_path or not os.path.isfile(data_path):
                        raise ValueError(f"Could not determine MCD file path for acquisition {acq.id}")
                    source_file_path = data_path
                    
                elif isinstance(loader, OMETIFFLoader):
                    loader_type = "ometiff"
                    # For OME-TIFF, get the file path from the loader's _acq_map
                    if hasattr(loader, '_acq_map') and original_acq_id in loader._acq_map:
                        data_path = loader._acq_map[original_acq_id]
                    elif acq.source_file:
                        data_path = acq.source_file
                    elif self.current_path and os.path.isdir(self.current_path):
                        # Try to find the file in the directory
                        import glob
                        tiff_files = glob.glob(os.path.join(self.current_path, "*.ome.tif"))
                        tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.ome.tiff")))
                        tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.tif")))
                        tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.tiff")))
                        # Try to match by acquisition name or ID
                        for tiff_file in tiff_files:
                            if acq.name in os.path.basename(tiff_file):
                                data_path = tiff_file
                                break
                        if not data_path and tiff_files:
                            # Fallback to first file if we can't match
                            data_path = tiff_files[0]
                    
                    if not data_path or not os.path.isfile(data_path):
                        raise ValueError(f"Could not determine OME-TIFF file path for acquisition {acq.id}")
                    
                    # For OME-TIFF, source_file_path is the directory containing the files
                    if self.current_path and os.path.isdir(self.current_path):
                        source_file_path = self.current_path
                    else:
                        source_file_path = os.path.dirname(data_path) if data_path else None
                
                if not data_path or loader_type is None:
                    raise ValueError(f"Could not determine data path or loader type for acquisition {acq.id}")
                
                # Deconvolve the acquisition
                deconvolve_acquisition(
                    data_path=data_path,
                    acq_id=original_acq_id,
                    output_dir=output_dir,
                    x0=x0,
                    iterations=iterations,
                    output_format=output_format,
                    channel_names=channel_names,
                    source_file_path=source_file_path,  # Pass source file/dir for filename generation
                    unique_acq_id=acq.id,  # Pass unique ID for filename
                    loader_type=loader_type,
                    channel_format=self.ometiff_channel_format if isinstance(loader, OMETIFFLoader) else 'CHW'
                )
                
                processed += 1
                progress_dlg.update_progress(
                    processed * 100,
                    f"Deconvolved {acq.name}",
                    f"Completed {processed} of {total_acqs} acquisitions"
                )
            except Exception as e:
                progress_dlg.update_progress(
                    processed * 100,
                    f"Error processing {acq.name}",
                    f"Error: {str(e)}"
                )
                # Continue with next acquisition
                processed += 1
        
        progress_dlg.update_progress(
            total_acqs * 100,
            "Complete",
            f"Successfully deconvolved {processed} acquisition(s)"
        )
        
        # Log deconvolution for all acquisitions
        logger = get_logger()
        # Collect unique source files
        source_files = set()
        for acq in self.acquisitions:
            loader = self._get_loader_for_acquisition(acq.id)
            if loader:
                if isinstance(loader, MCDLoader):
                    if acq.id in self.acq_to_file:
                        source_file = self.acq_to_file[acq.id]
                        source_files.add(os.path.basename(source_file))
                    elif acq.source_file:
                        source_files.add(os.path.basename(acq.source_file))
                elif isinstance(loader, OMETIFFLoader):
                    if self.current_path and os.path.isdir(self.current_path):
                        source_files.add(os.path.basename(self.current_path))
                    elif acq.source_file:
                        folder_path = os.path.dirname(acq.source_file) if os.path.dirname(acq.source_file) else acq.source_file
                        source_files.add(os.path.basename(folder_path))
        
        source_file_str = None
        if source_files:
            if len(source_files) == 1:
                source_file_str = list(source_files)[0]
            else:
                sorted_files = sorted(source_files)
                if len(sorted_files) <= 3:
                    source_file_str = ", ".join(sorted_files)
                else:
                    source_file_str = ", ".join(sorted_files[:3]) + f" and {len(sorted_files) - 3} more"
        
        params = {
            "x0": x0,
            "iterations": iterations,
            "output_format": output_format,
            "n_acquisitions": processed
        }
        
        logger._write_entry(
            entry_type="deconvolution",
            operation="high_resolution_deconvolution",
            parameters=params,
            acquisitions=[acq.id for acq in self.acquisitions],
            output_path=output_dir,
            notes=f"High resolution deconvolution: {processed} acquisition(s) deconvolved",
            source_file=source_file_str
        )
        
        return True
    
    def _load_deconvolved_images(self, output_dir: str):
        """Load the deconvolved OME-TIFF images and switch the loader."""
        try:
            # Close existing loaders
            self._close_all_loaders()
            
            # Ask user about channel format (default to CHW for exported images)
            format_dialog = OMETIFFFormatDialog(self)
            format_dialog.set_format('CHW')  # Set default to CHW
            if format_dialog.exec_() != QtWidgets.QDialog.Accepted:
                return
            
            channel_format = format_dialog.get_format()
            self.ometiff_channel_format = channel_format
            
            # Create OME-TIFF loader
            self.loader = OMETIFFLoader(channel_format=channel_format)
            self.loader.open(output_dir)
            
            # Update current path
            self.current_path = output_dir
            
            # Get acquisitions
            self.acquisitions = self.loader.list_acquisitions(source_file=output_dir)
            self.acq_combo.clear()
            for ai in self.acquisitions:
                # Use well name if available, otherwise use acquisition name
                label = ai.well if ai.well else ai.name
                self.acq_combo.addItem(label, ai.id)
            
            # Update window title
            dirname = os.path.basename(output_dir) or output_dir
            self.setWindowTitle(f"IMC File Viewer - {dirname} (Deconvolved OME-TIFF)")
            
            # Select first acquisition if available
            if self.acquisitions:
                self.acq_combo.setCurrentIndex(0)
                self.current_acq_id = self.acquisitions[0].id
                self._on_acq_changed(0)  # Pass the index parameter
            
            QtWidgets.QMessageBox.information(
                self,
                "Images Loaded",
                f"Successfully loaded deconvolved images from:\n{output_dir}\n\n"
                "All downstream viewing, segmentation, feature extraction, and analysis "
                "will now use the deconvolved images."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Failed",
                f"Failed to load deconvolved images:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _open_spillover_matrix_dialog(self):
        """Open the generate spillover matrix dialog."""
        dlg = GenerateSpilloverMatrixDialog(self)
        dlg.exec_()
    
    def _load_feature_file(self):
        """Load a feature file directly into memory."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Feature File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            df = pd.read_csv(file_path)
            
            # Validate that it's a valid feature file
            if 'cell_id' not in df.columns:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid File",
                    f"The file does not appear to be a valid feature file.\n"
                    f"Missing required column: 'cell_id'"
                )
                return
            
            # Ensure source_file column exists
            if 'source_file' not in df.columns:
                df['source_file'] = os.path.basename(file_path)
            
            # Store in memory
            self.feature_dataframe = df
            
            # Update cluster mode availability after loading features
            self._update_cluster_mode_availability()
            
            QtWidgets.QMessageBox.information(
                self,
                "Features Loaded",
                f"Successfully loaded {len(df)} cells with {len(df.columns)} columns from:\n{os.path.basename(file_path)}\n\n"
                f"Features are now available for clustering, spatial analysis, and batch correction."
            )
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load feature file:\n{str(e)}"
            )
    
    def _open_batch_correction_dialog(self):
        """Open the batch correction dialog."""
        # Check if we have feature data or allow loading files
        if self.feature_dataframe is None or self.feature_dataframe.empty:
            # Still allow opening dialog to load files
            pass
        
        dlg = BatchCorrectionDialog(self.feature_dataframe, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # Get combined dataframe (from loaded files or current)
            combined_df = dlg.get_combined_dataframe()
            if combined_df is not None and not combined_df.empty:
                # If features were loaded from files in the dialog, store them
                if not dlg.use_current_radio.isChecked() and dlg.loaded_files:
                    # Features were loaded from files - store them as main feature dataframe
                    self.feature_dataframe = combined_df
                    QtWidgets.QMessageBox.information(
                        self,
                        "Features Loaded",
                        f"Loaded {len(combined_df)} cells from {len(dlg.loaded_files)} file(s).\n"
                        f"Features are now available for clustering and spatial analysis."
                    )
            
            # Get corrected dataframe
            corrected_df = dlg.get_corrected_dataframe()
            if corrected_df is not None and not corrected_df.empty:
                # Store batch-corrected features separately (keep original)
                self.batch_corrected_dataframe = corrected_df
                
                # Save to file if requested
                output_path = dlg.get_output_path()
                if output_path:
                    try:
                        corrected_df.to_csv(output_path, index=False)
                        QtWidgets.QMessageBox.information(
                            self,
                            "Success",
                            f"Batch correction completed and saved to:\n{output_path}\n\n"
                            f"Both original and batch-corrected features are now available in memory."
                        )
                    except Exception as e:
                        QtWidgets.QMessageBox.critical(
                            self,
                            "Save Error",
                            f"Failed to save corrected features:\n{str(e)}"
                        )
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Success",
                        "Batch correction completed. Both original and batch-corrected features are now available in memory."
                    )





