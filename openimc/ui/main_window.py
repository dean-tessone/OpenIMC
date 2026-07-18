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

# CRITICAL: Configure dask BEFORE any imports that might trigger dask.dataframe
# This must be done at the very top, before any other imports
import os
import sys
import warnings
from pathlib import Path

# Suppress dask dataframe legacy implementation warning
warnings.filterwarnings('ignore', category=FutureWarning, module='dask.dataframe')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*legacy.*Dask DataFrame.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*dataframe.query-planning.*')

# Use direct assignment (not setdefault) to ensure it's set
os.environ['DASK_DATAFRAME__QUERY_PLANNING'] = 'True'

# Also configure dask directly if available (must be before any dask.dataframe import)
# Check if dask.dataframe was already imported
dask_dataframe_imported = 'dask.dataframe' in sys.modules
try:
    import dask
    # Configure before dask.dataframe can be imported
    dask.config.set({'dataframe.query-planning': True})
except (ImportError, AttributeError) as e:
    pass

from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial
import threading
import copy
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime

import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy import stats
from skimage.measure import regionprops, regionprops_table
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer

from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.feature_worker import (
    extract_features_for_acquisition,
    drop_excluded_channel_feature_columns
)
from openimc.core import extract_features, load_mcd
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.core import segment
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
    additive_blend_channels,
)
from openimc.ui.dialogs.progress_dialog import (
    ProgressDialog,
    close_progress_dialog,
    run_task_with_event_pump,
)
from openimc.ui.dialogs.gpu_selection_dialog import GPUSelectionDialog
from openimc.ui.dialogs.preprocessing_dialog import PreprocessingDialog
from openimc.ui.dialogs.segmentation_dialog import SegmentationDialog
from openimc.ui.dialogs.ilastik_segmentation_dialog import IlastikSegmentationDialog
from openimc.ui.dialogs.export import ExportDialog
from openimc.ui.dialogs.feature_extraction import FeatureExtractionDialog
from openimc.utils.optional_dependencies import (
    get_torch_module,
    optional_dependency_available,
)

# Optional runtime flags for GPU/TIFF
_HAVE_TORCH = optional_dependency_available("torch")

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
from openimc.ui.dialogs.pixel_correlation_dialog import PixelCorrelationDialog, ConditionROIWidget
from openimc.ui.cluster_utils import (
    build_cluster_annotation_map,
    canonicalize_cluster_id,
    get_cluster_display_name,
    normalize_cluster_annotation_map,
    sort_cluster_values,
)
from openimc.core import deconvolution
from openimc.utils.logger import get_logger
from openimc.ui.mask_manager import DynamicMaskManager
from openimc.ui.state_manager import StateManager
from openimc.ui.analysis_steps_exporter import AnalysisStepsExporter
from openimc.processing.denoising import (
    apply_channel_denoise,
    background_index_from_method,
    background_method_from_index,
)
from openimc.ui.dialogs.display_settings_dialog import (
    get_masks_directory_preference, save_masks_directory_preference
)




# Optional runtime flags for extra deps
_HAVE_CELLPOSE = optional_dependency_available("cellpose")
_CELLPOSE_IMPORT_ATTEMPTED = False
_CELLPOSE_IMPORT_ERROR: Optional[BaseException] = None
cellpose_models = None


def _load_cellpose_models():
    """Import Cellpose only when segmentation actually needs it."""
    global _HAVE_CELLPOSE, _CELLPOSE_IMPORT_ATTEMPTED, _CELLPOSE_IMPORT_ERROR
    global cellpose_models

    if cellpose_models is not None:
        return cellpose_models

    if _CELLPOSE_IMPORT_ATTEMPTED:
        return None

    _CELLPOSE_IMPORT_ATTEMPTED = True
    if not _HAVE_CELLPOSE:
        return None

    # Avoid importing Cellpose when torch itself is unavailable or unsafe.
    if get_torch_module() is None:
        _CELLPOSE_IMPORT_ERROR = torch_import_error()
        _HAVE_CELLPOSE = False
        return None

    try:
        from cellpose import models as imported_cellpose_models  # type: ignore
    except Exception as exc:
        _CELLPOSE_IMPORT_ERROR = exc
        _HAVE_CELLPOSE = False
        return None

    cellpose_models = imported_cellpose_models
    _CELLPOSE_IMPORT_ERROR = None
    return cellpose_models

# Optional CellSAM - import lazily so torch-backed dependencies do not run
# during module import in headless or partially provisioned environments.
_HAVE_CELLSAM = optional_dependency_available("cellSAM")
_CELLSAM_IMPORT_ATTEMPTED = False
_CELLSAM_IMPORT_ERROR: Optional[BaseException] = None
cellsam_pipeline = None
cellsam_pipeline_subprocess = None


def _load_cellsam_helpers() -> bool:
    """Import CellSAM helpers only when the feature is actually used."""
    global _HAVE_CELLSAM, _CELLSAM_IMPORT_ATTEMPTED, _CELLSAM_IMPORT_ERROR
    global cellsam_pipeline, cellsam_pipeline_subprocess

    if cellsam_pipeline is not None and cellsam_pipeline_subprocess is not None:
        return True

    if _CELLSAM_IMPORT_ATTEMPTED:
        return False

    _CELLSAM_IMPORT_ATTEMPTED = True
    if not _HAVE_CELLSAM:
        return False

    try:
        from openimc.processing.custom_cellsam import (
            cellsam_pipeline_custom,
            run_cellsam_pipeline_subprocess,
        )
    except Exception as exc:
        _HAVE_CELLSAM = False
        _CELLSAM_IMPORT_ERROR = exc
        cellsam_pipeline = None
        cellsam_pipeline_subprocess = None
        return False

    cellsam_pipeline = cellsam_pipeline_custom
    cellsam_pipeline_subprocess = run_cellsam_pipeline_subprocess
    _CELLSAM_IMPORT_ERROR = None
    return True

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
                cofactor = config.get('arcsinh_cofactor', 1.0)
                return arcsinh_normalize(img, cofactor)
            elif norm_method == 'percentile_clip':
                p_low, p_high = config.get('percentile_params', (1.0, 99.0))
                return percentile_clip_normalize(img, p_low, p_high)
            return img
        
        # Helper function to apply custom denoising
        def apply_custom_denoise(img, channel, custom_denoise_settings):
            cfg = custom_denoise_settings.get(channel) if custom_denoise_settings else None
            return apply_channel_denoise(img, cfg)
        
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


# Import worker function from processing module (now uses core.extract_features)
from openimc.processing.feature_worker import load_and_extract_features as _extract_features_worker


# --------------------------
# Mask Manager Dict Wrapper
# --------------------------
class MaskManagerDict:
    """Dict-like wrapper for DynamicMaskManager to maintain backward compatibility."""
    
    def __init__(self, mask_manager: DynamicMaskManager):
        self.mask_manager = mask_manager
        self._acq_info_cache = {}  # Cache acquisition info for mask loading
    
    def __getitem__(self, acq_id: str):
        """Get mask for acquisition, loading from disk if needed."""
        acq_info = self._acq_info_cache.get(acq_id)
        mask = self.mask_manager.get_mask(acq_id, acq_info)
        if mask is None:
            raise KeyError(f"No mask found for acquisition {acq_id}")
        return mask
    
    def __setitem__(self, acq_id: str, mask: np.ndarray):
        """Set mask for acquisition."""
        acq_info = self._acq_info_cache.get(acq_id)
        self.mask_manager.set_mask(acq_id, mask, acq_info=acq_info)
    
    def __contains__(self, acq_id: str) -> bool:
        """Check if mask exists for acquisition."""
        return self.mask_manager.has_mask(acq_id)
    
    def __delitem__(self, acq_id: str):
        """Remove mask from memory (does not delete disk files)."""
        self.mask_manager.remove_mask(acq_id)
    
    def __len__(self) -> int:
        """Return the number of masks available."""
        return len(self.mask_manager.get_all_mask_ids())
    
    def keys(self):
        """Get all acquisition IDs that have masks."""
        return self.mask_manager.get_all_mask_ids()
    
    def items(self):
        """Get all (acq_id, mask) pairs, loading masks as needed."""
        for acq_id in self.keys():
            acq_info = self._acq_info_cache.get(acq_id)
            mask = self.mask_manager.get_mask(acq_id, acq_info)
            if mask is not None:
                yield (acq_id, mask)
    
    def values(self):
        """Get all masks, loading as needed."""
        for acq_id in self.keys():
            acq_info = self._acq_info_cache.get(acq_id)
            mask = self.mask_manager.get_mask(acq_id, acq_info)
            if mask is not None:
                yield mask
    
    def get(self, acq_id: str, default=None):
        """Get mask with default value if not found."""
        acq_info = self._acq_info_cache.get(acq_id)
        mask = self.mask_manager.get_mask(acq_id, acq_info)
        return mask if mask is not None else default
    
    def set_acq_info(self, acq_id: str, acq_info):
        """Set acquisition info for better mask file resolution."""
        self._acq_info_cache[acq_id] = acq_info


# --------------------------
# Main Window
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("OpenIMC")
            
            # Set window size to full screen with minimum size constraint
            screen = QtWidgets.QApplication.desktop().screenGeometry()
            self.resize(screen.width(), screen.height())
            
            # Set minimum size for smaller screens
            self.setMinimumSize(1000, 700)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
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
        # Session tracking for analysis steps export
        self.session_start_time = datetime.now()
        # QC analysis results cache (persists until files change)
        self.qc_results_cache: Dict[str, Dict] = {}  # Maps file_set_id to QC results
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
        try:
            self.canvas = MplCanvas(width=6, height=6, dpi=100)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
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
        
        # Control buttons
        button_row = QtWidgets.QVBoxLayout()  # Changed to vertical for better button sizing
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
        self.rgb_color_scaling = {}  # {'Red': {'min': value, 'max': value}, 'Teal': {...}, 'Blue': {...}}
        
        # Arcsinh normalization state
        self.arcsinh_enabled = False
        # Per-channel normalization config: {channel: {"method": str, "cofactor": float}}
        self.channel_normalization: Dict[str, Dict[str, float or str]] = {}
        
        # Per-channel scaling method state
        self.current_scaling_method = "default"  # kept for backward compatibility
        self.channel_scaling_method: Dict[str, str] = {}  # {channel: "default"}
        
        # Segmentation state
        # Use dynamic mask manager for large datasets
        try:
            saved_masks_dir = get_masks_directory_preference()
            self.mask_manager = DynamicMaskManager(masks_directory=saved_masks_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        # Keep segmentation_masks as a property that uses mask_manager for backward compatibility
        self.segmentation_colors = {}  # {acq_id: colors_array}
        self.cluster_colors = {}  # {acq_id: {cluster_id: color_array}}
        self.cluster_color_map = {}  # {acq_id: {cluster_id: display_name}}
        self.segmentation_overlay = False
        self.segmentation_overlay_mode = "Mask"  # "Outline", "Mask", or "Cluster"
        self.preprocessing_cache = PreprocessingCache()
        # Per-channel denoise config {channel: {"hot": {...}, "speckle": {...}, "background": {...}}}
        self.channel_denoise: Dict[str, Dict[str, dict]] = {}
        # Acquisition info cache for tracking mask status
        self._acquisition_info_cache: Dict[str, Dict] = {}
        
        # Feature extraction state
        self.feature_dataframe = None  # Store extracted features in memory
        self.batch_corrected_dataframe = None  # Store batch-corrected features in memory
        self.analysis_feature_set_preference = None
        self._analysis_feature_sync_in_progress = False
        
        # Dialog instances for state retention
        self.clustering_dialog = None  # Store clustering dialog instance
        self.spatial_dialog = None  # Store spatial analysis dialog instance
        self.segmentation_dialog = None  # Store segmentation dialog instance
        
        # Color assignment for multi-channel composite (additive blending)
        self.color_assignment_frame = QtWidgets.QFrame()
        self.color_assignment_frame.setFrameStyle(QtWidgets.QFrame.Box)
        color_layout = QtWidgets.QVBoxLayout(self.color_assignment_frame)
        color_layout.addWidget(QtWidgets.QLabel("Channel Color Assignment (for multi-channel composite):"))
        
        # Mode toggle: RGB mode (default) vs Multicolor mode (opt-in)
        self.multicolor_mode_chk = QtWidgets.QCheckBox("Multicolor mode (check to enable)")
        self.multicolor_mode_chk.setChecked(False)  # Default to RGB mode
        self.multicolor_mode_chk.setToolTip("Unchecked: Traditional RGB mode with 3 colors (Red, Green, Blue) using channel stacking.\nChecked: Multicolor mode with 6 colors (Blue, Teal, Yellow, Magenta, Red, White) using additive blending.")
        self.multicolor_mode_chk.toggled.connect(self._on_multicolor_mode_toggled)
        color_layout.addWidget(self.multicolor_mode_chk)
        
        # Search box for filtering channels
        self.channel_color_search = QtWidgets.QLineEdit()
        self.channel_color_search.setPlaceholderText("Search channels...")
        self.channel_color_search.textChanged.connect(self._filter_channel_color_table)
        color_layout.addWidget(self.channel_color_search)
        
        # Create table for channel-to-color mapping
        self.channel_color_table = QtWidgets.QTableWidget()
        self.channel_color_table.setColumnCount(2)
        self.channel_color_table.setHorizontalHeaderLabels(["Channel", "Color"])
        self.channel_color_table.horizontalHeader().setStretchLastSection(True)
        # Set height to show at least 3 rows (header ~30px + 3 rows ~75px = ~105px, add padding)
        self.channel_color_table.setMinimumHeight(120)
        self.channel_color_table.setMaximumHeight(400)
        self.channel_color_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.channel_color_table.verticalHeader().setVisible(False)
        color_layout.addWidget(self.channel_color_table)
        
        # Store channel color assignments (channel_name -> color_name)
        self.channel_color_assignments = {}
        
        # Available colors for assignment (multicolor mode)
        self.available_colors_multicolor = ['Blue', 'Teal', 'Green', 'Yellow', 'Magenta', 'Red', 'White']
        # Available colors for RGB mode
        self.available_colors_rgb = ['Red', 'Green', 'Blue']
        # Current available colors (default to multicolor)
        self.available_colors = self.available_colors_multicolor
        
        # Legacy support: keep old lists for backward compatibility during transition
        # These will be populated from the table when needed
        self.red_list = QtWidgets.QListWidget()
        self.green_list = QtWidgets.QListWidget()
        self.blue_list = QtWidgets.QListWidget()
        self.red_search = QtWidgets.QLineEdit()
        self.green_search = QtWidgets.QLineEdit()
        self.blue_search = QtWidgets.QLineEdit()

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
        try:
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
            
            # Save/Load State
            act_save_state = file_menu.addAction("Save State…")
            act_save_state.triggered.connect(self._save_state)
            act_load_state = file_menu.addAction("Load State…")
            act_load_state.triggered.connect(self._load_state)
            act_export_steps = file_menu.addAction("Export Analysis Steps…")
            act_export_steps.triggered.connect(self._export_analysis_steps)
            file_menu.addSeparator()
            
            # Export submenu
            export_submenu = file_menu.addMenu("Export")
            act_export_tiff = export_submenu.addAction("Export to OME-TIFF…")
            act_export_tiff.triggered.connect(self._export_ome_tiff)
            act_export_panel = export_submenu.addAction("Export Panel CSV…")
            act_export_panel.triggered.connect(self._export_panel)
            
            # Masks submenu
            masks_submenu = file_menu.addMenu("Segmentation Masks")
            act_load_masks = masks_submenu.addAction("Load Masks…")
            act_load_masks.triggered.connect(self._load_segmentation_masks)
            act_save_masks = masks_submenu.addAction("Save Masks…")
            act_save_masks.triggered.connect(self._save_segmentation_masks)
            
            file_menu.addSeparator()
            act_quit = file_menu.addAction("Quit")
            act_quit.triggered.connect(self.close)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

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
        try:
            self.open_btn.clicked.connect(self._open_dialog)
            self.acq_combo.currentIndexChanged.connect(self._on_acq_changed)
            self.deselect_all_btn.clicked.connect(self._deselect_all_channels)
            self.channel_list.itemChanged.connect(self._on_channel_selection_changed)
            self.channel_search.textChanged.connect(self._filter_channels)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        # Auto-refresh: no manual 'View selected' action
        try:
            self.view_btn.clicked.disconnect()
        except Exception:
            pass
        
        try:
            self.reset_zoom_btn.clicked.connect(self._reset_zoom)
            self.comparison_btn.clicked.connect(self._comparison)
            self.segment_btn.clicked.connect(self._run_segmentation)
            self.extract_features_btn.clicked.connect(self._extract_features)
            self.batch_correction_btn.clicked.connect(self._open_batch_correction_dialog)
            self.clustering_btn.clicked.connect(self._open_clustering_dialog)
            self.spatial_btn.clicked.connect(self._open_spatial_dialog)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

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
        # Also initialize clustering button and spatial analysis button state
        try:
            self._update_cluster_mode_availability()
        except Exception:
            pass
    
    @property
    def segmentation_masks(self):
        """
        Property for backward compatibility with self.segmentation_masks.
        Returns a dict-like object that uses the mask manager.
        """
        return MaskManagerDict(self.mask_manager)
    
    @segmentation_masks.setter
    def segmentation_masks(self, value):
        """Setter for backward compatibility - converts dict to mask manager entries."""
        if isinstance(value, dict):
            for acq_id, mask in value.items():
                self.mask_manager.set_mask(acq_id, mask)

    # ---------- About dialog ----------
    def _show_about_dialog(self):
        """Show the About dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About OpenIMC",
            "OpenIMC © 2025 University of Southern California — Licensed under GPL v3 or later\n\n"
            "NO WARRANTY. See LICENSE for details.\n\n"
            "Source code: <a href='https://github.com/dean-tessone/OpenIMC'>https://github.com/dean-tessone/OpenIMC</a>\n\n"
            "Documentation: <a href='https://dean-tessone.github.io/OpenIMC/overview.html'>https://dean-tessone.github.io/OpenIMC/overview.html</a>"
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
            self.setWindowTitle(f"OpenIMC - {stem} (MCD)")
        else:
            self.setWindowTitle(f"OpenIMC - {len(paths)} MCD files")
        
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
        # Clear QC results cache when files are closed
        self.qc_results_cache.clear()
        
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
                self.setWindowTitle(f"OpenIMC - {stem} ({loader_type})")
            else:
                dirname = os.path.basename(path) or path
                self.setWindowTitle(f"OpenIMC - {dirname} ({loader_type})")
        except Exception:
            # Fallback to default title if something goes wrong
            self.setWindowTitle(f"OpenIMC ({loader_type})")
        
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
        """Populate the channel color assignment table with selected channels."""
        # Block signals to avoid triggering updates during population
        self.channel_color_table.blockSignals(True)
        
        # Determine which color set to use based on mode
        is_multicolor = self.multicolor_mode_chk.isChecked()
        if is_multicolor:
            self.available_colors = self.available_colors_multicolor
            default_color = 'Blue'
        else:
            self.available_colors = self.available_colors_rgb
            default_color = 'Red'
        
        # Preserve current color assignments
        prev_assignments = self.channel_color_assignments.copy()
        
        # Clear table
        self.channel_color_table.setRowCount(0)
        self.channel_color_assignments = {}
        
        # Populate table with selected channels
        for ch in channels:
            row = self.channel_color_table.rowCount()
            self.channel_color_table.insertRow(row)
            
            # Channel name (read-only)
            channel_item = QtWidgets.QTableWidgetItem(ch)
            channel_item.setFlags(channel_item.flags() & ~Qt.ItemIsEditable)
            self.channel_color_table.setItem(row, 0, channel_item)
            
            # Color selection combo box
            color_combo = QtWidgets.QComboBox()
            color_combo.addItems(self.available_colors)
            # Restore previous assignment if valid, otherwise use default
            prev_color = prev_assignments.get(ch, default_color)
            if prev_color in self.available_colors:
                initial_color = prev_color
            else:
                # Previous color not valid in current mode, use default
                initial_color = self.available_colors[0]
            
            # Block signals on combo box during setup to avoid triggering during initialization
            color_combo.blockSignals(True)
            # Set the current text
            color_combo.setCurrentText(initial_color)
            color_combo.blockSignals(False)
            
            # Connect signal after setting initial value and unblocking signals
            # Use a closure to properly capture the channel name
            def make_color_changed_handler(channel_name):
                def handler(text):
                    self._on_channel_color_changed(channel_name, text)
                return handler
            
            color_combo.currentTextChanged.connect(make_color_changed_handler(ch))
            self.channel_color_table.setCellWidget(row, 1, color_combo)
            
            # Store assignment with the actual current value
            self.channel_color_assignments[ch] = initial_color
        
        self.channel_color_table.blockSignals(False)
        
        # Legacy: also populate old lists for backward compatibility
        # Map colors to old RGB lists
        prev_red = {}
        prev_green = {}
        prev_blue = {}
        if self.red_list.count() > 0:
            prev_red = {self.red_list.item(i).text(): self.red_list.item(i).checkState() == Qt.Checked for i in range(self.red_list.count())}
        if self.green_list.count() > 0:
            prev_green = {self.green_list.item(i).text(): self.green_list.item(i).checkState() == Qt.Checked for i in range(self.green_list.count())}
        if self.blue_list.count() > 0:
            prev_blue = {self.blue_list.item(i).text(): self.blue_list.item(i).checkState() == Qt.Checked for i in range(self.blue_list.count())}
        
        self.red_list.clear()
        self.green_list.clear()
        self.blue_list.clear()
        
        # Map new color assignments to old RGB lists
        for ch in channels:
            color = self.channel_color_assignments.get(ch, 'Blue')
            # Map colors to RGB: Red->red_list, Green->green_list, Blue->blue_list
            # Yellow->red+green, Magenta->red+blue, White->all
            if color == 'Red':
                lst = self.red_list
            elif color == 'Teal':
                lst = self.green_list  # Legacy: green_list is used for teal
            elif color == 'Blue':
                lst = self.blue_list
            elif color == 'Yellow':
                # Add to both red and green
                for lst in [self.red_list, self.green_list]:
                    it = QtWidgets.QListWidgetItem(ch)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    checked = prev_red.get(ch, False) or prev_green.get(ch, False)
                    it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    lst.addItem(it)
                continue
            elif color == 'Magenta':
                # Add to both red and blue
                for lst in [self.red_list, self.blue_list]:
                    it = QtWidgets.QListWidgetItem(ch)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    checked = prev_red.get(ch, False) or prev_blue.get(ch, False)
                    it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    lst.addItem(it)
                continue
            elif color == 'White':
                # Add to all three
                for lst in [self.red_list, self.green_list, self.blue_list]:
                    it = QtWidgets.QListWidgetItem(ch)
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                    checked = prev_red.get(ch, False) or prev_green.get(ch, False) or prev_blue.get(ch, False)
                    it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                    lst.addItem(it)
                continue
            else:
                lst = self.blue_list  # Default
            
            it = QtWidgets.QListWidgetItem(ch)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checked = prev_red.get(ch, False) if lst == self.red_list else (prev_green.get(ch, False) if lst == self.green_list else prev_blue.get(ch, False))
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            lst.addItem(it)
    
    def _on_channel_color_changed(self, channel_name: str, color_name: str):
        """Handle color assignment change for a channel."""
        # Update the assignment
        self.channel_color_assignments[channel_name] = color_name
        
        # Also update the legacy RGB lists for backward compatibility
        is_multicolor = self.multicolor_mode_chk.isChecked()
        if not is_multicolor:
            # In RGB mode, update the legacy lists based on color assignment
            # Clear the channel from all lists first
            for lst in [self.red_list, self.green_list, self.blue_list]:
                for i in range(lst.count()):
                    item = lst.item(i)
                    if item.text() == channel_name:
                        lst.takeItem(i)
                        break
            
            # Add to the appropriate list based on color
            if color_name == 'Red':
                target_list = self.red_list
            elif color_name == 'Green':
                target_list = self.green_list
            elif color_name == 'Blue':
                target_list = self.blue_list
            else:
                target_list = self.red_list  # Default
            
            # Add to target list
            it = QtWidgets.QListWidgetItem(channel_name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked)
            target_list.addItem(it)
        
        # Update view if not in grid mode
        if not self.grid_view_chk.isChecked():
            if not self.preserve_zoom:
                self.preserve_zoom = True
                self._view_selected()
    
    def _on_multicolor_mode_toggled(self):
        """Handle toggle between RGB mode and multicolor mode."""
        # Clear invalid color assignments when switching modes
        is_multicolor = self.multicolor_mode_chk.isChecked()
        if is_multicolor:
            valid_colors = set(self.available_colors_multicolor)
        else:
            valid_colors = set(self.available_colors_rgb)
        
        # Remove assignments that aren't valid in the new mode
        invalid_channels = []
        for ch, color in list(self.channel_color_assignments.items()):
            if color not in valid_colors:
                invalid_channels.append(ch)
        for ch in invalid_channels:
            del self.channel_color_assignments[ch]
        
        # Repopulate the color assignment table with the correct colors
        selected_channels = self._selected_channels()
        if selected_channels:
            self._populate_color_assignments(selected_channels)
        # Update scaling combo
        self._update_scaling_channel_combo()
        # Refresh view
        if not self.grid_view_chk.isChecked():
            if not self.preserve_zoom:
                self.preserve_zoom = True
                self._view_selected()

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
    
    def _filter_channel_color_table(self):
        """Filter channel color table based on search text."""
        if not hasattr(self, 'channel_color_search') or not hasattr(self, 'channel_color_table'):
            return
        text = self.channel_color_search.text().lower()
        for i in range(self.channel_color_table.rowCount()):
            channel_item = self.channel_color_table.item(i, 0)
            if channel_item:
                channel_name = channel_item.text().lower()
                # Show row if search text matches channel name
                self.channel_color_table.setRowHidden(i, text not in channel_name)
    
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
            # Check if we're in multicolor mode or RGB mode
            is_multicolor = self.multicolor_mode_chk.isChecked()
            
            if is_multicolor:
                # Multicolor mode: show all available color names
                selected_channels = self._selected_channels()
                assigned_colors = set()
                for ch in selected_channels:
                    if ch in self.channel_color_assignments:
                        assigned_colors.add(self.channel_color_assignments[ch])
                    else:
                        # Fallback: check legacy RGB lists
                        def _is_checked(lst: QtWidgets.QListWidget, channel: str) -> bool:
                            for i in range(lst.count()):
                                item = lst.item(i)
                                if item.text() == channel and item.checkState() == Qt.Checked:
                                    return True
                            return False
                        in_red = _is_checked(self.red_list, ch)
                        in_green = _is_checked(self.green_list, ch)
                        in_blue = _is_checked(self.blue_list, ch)
                        if in_red and in_green and in_blue:
                            assigned_colors.add('White')
                        elif in_red and in_green:
                            assigned_colors.add('Yellow')
                        elif in_red and in_blue:
                            assigned_colors.add('Magenta')
                        elif in_green and in_blue:
                            assigned_colors.add('Teal')
                        elif in_red:
                            assigned_colors.add('Red')
                        elif in_green:
                            assigned_colors.add('Teal')
                        elif in_blue:
                            assigned_colors.add('Blue')
                
                # Add all available colors, but prioritize assigned ones
                all_colors = ['Blue', 'Teal', 'Green', 'Yellow', 'Magenta', 'Red', 'White']
                for color in all_colors:
                    if color in assigned_colors or not assigned_colors:  # Show all if none assigned
                        self.scaling_channel_combo.addItem(color)
            else:
                # RGB mode: show only Red, Green, Blue
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

    def _build_current_denoise_config(self) -> dict:
        """Capture the current denoise controls into a serializable config dict."""
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
                "method": background_method_from_index(self.bg_method_combo.currentIndex()),
                "radius": int(self.bg_radius_spin.value()),
            }

        return {
            "hot": cfg_hot,
            "speckle": cfg_speckle,
            "background": cfg_bg,
        }

    def _get_denoise_step_order(self):
        """Translate the UI order controls into pipeline step keys."""
        order_map = {"Hot pixel": "hot", "Speckle": "speckle", "Background": "background"}
        chosen_order = [
            self.order_combo_1.currentText(),
            self.order_combo_2.currentText(),
            self.order_combo_3.currentText(),
        ]
        seen = set()
        exec_steps = []
        for name in chosen_order:
            key = order_map.get(name)
            if key and key not in seen:
                exec_steps.append(key)
                seen.add(key)
        return exec_steps

    def _apply_denoise_settings_and_refresh(self):
        """Capture current denoise UI settings, assign to selected channels in denoise list, refresh view."""
        try:
            # Target single channel from combo
            target_channel = self.denoise_channel_combo.currentText()
            if not target_channel:
                return

            self.channel_denoise[target_channel] = self._build_current_denoise_config()

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
        return apply_channel_denoise(img, cfg, step_order=self._get_denoise_step_order())

    def _apply_custom_denoise(self, channel: str, img: np.ndarray, custom_denoise_settings: dict) -> np.ndarray:
        """Apply custom denoise steps for a channel in raw domain."""
        if not _HAVE_SCIKIT_IMAGE:
            return img
        cfg = custom_denoise_settings.get(channel)
        return apply_channel_denoise(img, cfg)

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

            # Apply the same configuration to all channels
            config = self._build_current_denoise_config()
            for channel in all_channels:
                self.channel_denoise[channel] = {
                    "hot": dict(config["hot"]) if config["hot"] else None,
                    "speckle": dict(config["speckle"]) if config["speckle"] else None,
                    "background": dict(config["background"]) if config["background"] else None,
                }

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
                self.bg_method_combo.setCurrentIndex(background_index_from_method(bg.get("method")))
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
            cofactor_spin.setValue(1.0)
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
        # Min/max controls are always enabled for manual scaling
        self.min_spinbox.setEnabled(True)
        self.max_spinbox.setEnabled(True)
        self.min_spinbox.setStyleSheet("")
        self.max_spinbox.setStyleSheet("")

    def _load_channel_scaling(self):
        """Load scaling values for the currently selected channel or RGB color."""
        current_selection = self.scaling_channel_combo.currentText()
        if not current_selection:
            return
        
        # Check if we're in RGB mode (selection is a color name)
        # Check mode first to determine which colors are valid
        is_rgb_mode = hasattr(self, 'grid_view_chk') and not self.grid_view_chk.isChecked()
        is_multicolor = self.multicolor_mode_chk.isChecked() if is_rgb_mode else False
        
        if is_rgb_mode and not is_multicolor:
            # RGB mode: only Red, Green, Blue
            is_rgb_color = current_selection in ['Red', 'Green', 'Blue']
        else:
            # Multicolor mode or grid mode
            is_rgb_color = current_selection in ['Red', 'Teal', 'Green', 'Blue', 'Yellow', 'Magenta', 'White']
        
        if is_rgb_color:
            # Load RGB color scaling
            if current_selection in self.rgb_color_scaling:
                # Load saved values
                min_val = self.rgb_color_scaling[current_selection]['min']
                max_val = self.rgb_color_scaling[current_selection]['max']
            else:
                # Use default range - compute from actual channels assigned to this color
                if self.current_acq_id is None:
                    return
                try:
                    # Get channels assigned to this color
                    selected_channels = []
                    
                    if is_rgb_mode and not is_multicolor:
                        # RGB mode: use legacy lists
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
                        
                        if current_selection == 'Red':
                            selected_channels = red_selection
                        elif current_selection == 'Green':
                            selected_channels = green_selection
                        elif current_selection == 'Blue':
                            selected_channels = blue_selection
                    else:
                        # Multicolor mode: use new color assignment system
                        for ch_name, assigned_color in self.channel_color_assignments.items():
                            if assigned_color == current_selection:
                                selected_channels.append(ch_name)
                        
                        # Fallback: check legacy RGB lists if no assignments found
                        if not selected_channels:
                            def _checked(lst: QtWidgets.QListWidget) -> List[str]:
                                vals: List[str] = []
                                for i in range(lst.count()):
                                    item = lst.item(i)
                                    if item.checkState() == Qt.Checked:
                                        vals.append(item.text())
                                return vals
                            
                            red_selection = _checked(self.red_list)
                            teal_selection = _checked(self.green_list)  # Legacy: green_list is teal
                            blue_selection = _checked(self.blue_list)
                            
                            # Map legacy selections to new colors
                            if current_selection == 'Red':
                                selected_channels = red_selection
                            elif current_selection == 'Teal':
                                selected_channels = teal_selection
                            elif current_selection == 'Blue':
                                selected_channels = blue_selection
                            elif current_selection == 'Yellow':
                                # Yellow uses both red and green (teal)
                                selected_channels = list(set(red_selection + teal_selection))
                            elif current_selection == 'Magenta':
                                # Magenta uses both red and blue
                                selected_channels = list(set(red_selection + blue_selection))
                            elif current_selection == 'White':
                                # White uses all three
                                selected_channels = list(set(red_selection + teal_selection + blue_selection))
                    
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

    def _save_channel_scaling(self):
        """Save current scaling values for the selected channel or RGB color."""
        current_selection = self.scaling_channel_combo.currentText()
        if not current_selection:
            return
        
        # Get values directly from spinboxes
        min_val = self.min_spinbox.value()
        max_val = self.max_spinbox.value()
        
        # Check if we're in RGB mode (selection is a color name)
        # Check mode first to determine which colors are valid
        is_rgb_mode = hasattr(self, 'grid_view_chk') and not self.grid_view_chk.isChecked()
        is_multicolor = self.multicolor_mode_chk.isChecked() if is_rgb_mode else False
        
        if is_rgb_mode and not is_multicolor:
            # RGB mode: only Red, Green, Blue
            is_rgb_color = current_selection in ['Red', 'Green', 'Blue']
        else:
            # Multicolor mode or grid mode
            is_rgb_color = current_selection in ['Red', 'Teal', 'Green', 'Blue', 'Yellow', 'Magenta', 'White']
        
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
            
            # Validate loader state (for MCDLoader, check if mcd is not None)
            if isinstance(loader, MCDLoader) and loader.mcd is None:
                raise RuntimeError(f"Loader for acquisition {acq_id} is closed or invalid")
            
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_id)
            try:
                img = loader.get_image(original_acq_id, channel)
            except (OSError, RuntimeError) as e:
                if isinstance(e, OSError) and e.errno == 9:
                    # Bad file descriptor - clear cache and re-raise with better message
                    with self._cache_lock:
                        # Clear cache entries for this acquisition
                        keys_to_remove = [k for k in self.image_cache.keys() if k[0] == acq_id]
                        for k in keys_to_remove:
                            self.image_cache.pop(k, None)
                    raise RuntimeError(
                        f"File descriptor error when loading {acq_id}, channel {channel}. "
                        "The file may have been closed. Please reload the file."
                    ) from e
                raise
            with self._cache_lock:
                self.image_cache[cache_key] = img
        
        # Apply per-channel denoising first (operates in raw space)
        try:
            img = self._apply_denoise(channel, img)
        except Exception:
            pass
        
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
            
            # Validate loader state (for MCDLoader, check if mcd is not None)
            if isinstance(loader, MCDLoader) and loader.mcd is None:
                raise RuntimeError(f"Loader for acquisition {acq_id} is closed or invalid")
            
            # Get original acquisition ID if this is a unique ID
            original_acq_id = self._get_original_acq_id(acq_id)
            try:
                img = loader.get_image(original_acq_id, channel)
            except (OSError, RuntimeError) as e:
                if isinstance(e, OSError) and e.errno == 9:
                    # Bad file descriptor - clear cache and re-raise with better message
                    with self._cache_lock:
                        # Clear cache entries for this acquisition
                        keys_to_remove = [k for k in self.image_cache.keys() if k[0] == acq_id]
                        for k in keys_to_remove:
                            self.image_cache.pop(k, None)
                    raise RuntimeError(
                        f"File descriptor error when loading {acq_id}, channel {channel}. "
                        "The file may have been closed. Please reload the file."
                    ) from e
                raise
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
            # No overlap, return original images
            return channel_images_denoised
        
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
        
        # Step 3: Return corrected images (no normalization applied)
        return channel_images_denoised

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
                # Validate loader state before prefetching
                if isinstance(loader, MCDLoader) and loader.mcd is None:
                    return
                
                # Get original acquisition ID if this is a unique ID
                original_acq_id = self._get_original_acq_id(acq_id)
                # Load the full stack once, then split into channels for faster access
                try:
                    stack = loader.get_all_channels(original_acq_id)
                except (OSError, RuntimeError) as e:
                    if isinstance(e, OSError) and e.errno == 9:
                        # Bad file descriptor - loader is invalid, skip prefetch
                        return
                    raise
                
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
            # Validate that we have a valid loader and acquisition
            if not self.current_acq_id:
                return
            
            loader = self._get_loader_for_acquisition(self.current_acq_id)
            if loader is None:
                return
            
            # Check if loader is valid (for MCDLoader, check if mcd is not None)
            if isinstance(loader, MCDLoader) and loader.mcd is None:
                return
            
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
        except (OSError, RuntimeError) as e:
            # Handle file descriptor errors and other runtime errors gracefully
            if isinstance(e, OSError) and e.errno == 9:
                # Bad file descriptor - loader is in invalid state
                # Silently fail to avoid spamming console with errors
                pass
            else:
                # Log other errors but don't disrupt the UI
                print(f"Auto-load error: {e}")
        except Exception as e:
            # Catch all other exceptions to prevent UI disruption
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
        """Show multi-channel composite using additive blending or RGB stacking based on mode."""
        if not selected_channels:
            QtWidgets.QMessageBox.information(self, "No channels", "Please select at least one channel for composite.")
            return
        
        # Check which mode we're in
        is_multicolor = self.multicolor_mode_chk.isChecked()
        
        if is_multicolor:
            # Multicolor mode: use additive blending
            # Get color assignments for selected channels
            channel_colors = []
            channels_to_blend = []
            
            for ch in selected_channels:
                # Get color assignment from new system
                if ch in self.channel_color_assignments:
                    color = self.channel_color_assignments[ch]
                    channels_to_blend.append(ch)
                    channel_colors.append(color)
                else:
                    # Fallback: check legacy RGB lists
                    def _is_checked(lst: QtWidgets.QListWidget, channel: str) -> bool:
                        for i in range(lst.count()):
                            item = lst.item(i)
                            if item.text() == channel and item.checkState() == Qt.Checked:
                                return True
                        return False
                    
                    # Determine color from legacy lists
                    in_red = _is_checked(self.red_list, ch)
                    in_green = _is_checked(self.green_list, ch)
                    in_blue = _is_checked(self.blue_list, ch)
                    
                    if in_red and in_green and in_blue:
                        color = 'White'
                    elif in_red and in_green:
                        color = 'Yellow'
                    elif in_red and in_blue:
                        color = 'Magenta'
                    elif in_green and in_blue:
                        color = 'Teal'
                    elif in_red:
                        color = 'Red'
                    elif in_green:
                        color = 'Teal'
                    elif in_blue:
                        color = 'Blue'
                    else:
                        color = 'Blue'
                    
                    channels_to_blend.append(ch)
                    channel_colors.append(color)
            
            # If no channels have color assignments, assign first channel to Blue
            if not channels_to_blend and selected_channels:
                channels_to_blend = [selected_channels[0]]
                channel_colors = ['Blue']
        else:
            # RGB mode: use old R/G/B channel assignment and stacking
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
            
            # Collect all channels that will be used in RGB composite
            all_rgb_channels = list(set(red_selection + green_selection + blue_selection))
            
            if not all_rgb_channels:
                QtWidgets.QMessageBox.information(self, "No RGB channels", "Please assign channels to Red, Green, or Blue.")
                return
            
            # Apply spillover correction if enabled
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
            
            rgb_channels = [r_img, g_img, b_img]
            rgb_titles = [
                f"{'+'.join(red_selection) if red_selection else 'None'} (Red)",
                f"{'+'.join(green_selection) if green_selection else 'None'} (Green)",
                f"{'+'.join(blue_selection) if blue_selection else 'None'} (Blue)"
            ]
            
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
            
            # Stack channels into RGB image
            rgb_img = np.dstack(rgb_channels)
            
            # Create title
            acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
            title = " + ".join(rgb_titles) + f"\n{acq_subtitle}"
            if self.segmentation_overlay:
                title += " (segmented)"
            
            # Display RGB composite (same display code as multicolor mode below)
            self.canvas.fig.clear()
            
            if grayscale:
                # Convert RGB to grayscale
                ax = self.canvas.fig.add_subplot(111)
                gray_base = np.mean(rgb_img, axis=2)
                
                if self.segmentation_overlay:
                    blended = self._get_segmentation_overlay(gray_base)
                    ax.imshow(blended, interpolation="nearest")
                    ax.set_title(title)
                    ax.axis("off")
                else:
                    vmin, vmax = np.min(gray_base), np.max(gray_base)
                    im = ax.imshow(gray_base, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
                    cbar = self.canvas.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                    cbar.set_ticks([vmin, vmax])
                    cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
                    ax.set_title(title)
                    ax.axis("off")
                
                if self.scale_bar_chk.isChecked():
                    pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
                    if pixel_size_um > 0:
                        scale_bar_length_um = self.scale_bar_length_spin.value()
                        self.canvas._draw_scale_bar_on_axes(gray_base.shape, scale_bar_length_um, pixel_size_um, ax)
            else:
                ax_main = self.canvas.fig.add_subplot(111)
                
                if self.segmentation_overlay:
                    rgb_img = self._get_segmentation_overlay(rgb_img)
                
                im = ax_main.imshow(stack_to_rgb(rgb_img), interpolation="nearest")
                ax_main.set_title(title)
                ax_main.axis("off")
                
                if self.scale_bar_chk.isChecked():
                    pixel_size_um = self._get_pixel_size_um(self.current_acq_id)
                    if pixel_size_um > 0:
                        scale_bar_length_um = self.scale_bar_length_spin.value()
                        img_shape = rgb_img.shape[:2]
                        self.canvas._draw_scale_bar_on_axes(img_shape, scale_bar_length_um, pixel_size_um, ax_main)
            
            self.canvas.draw()
            self._add_cluster_legend()
            return  # Early return for RGB mode
        
        # Multicolor mode continues here
        if not channels_to_blend:
            QtWidgets.QMessageBox.information(self, "No channels", "Please select at least one channel for composite.")
            return
        
        # Collect all channels that will be used
        all_channels = list(set(channels_to_blend))
        
        # Apply spillover correction if enabled
        if self.spillover_correction_enabled and self.spillover_matrix is not None and all_channels:
            corrected_images = self._apply_spillover_correction_to_channels(self.current_acq_id, all_channels)
        else:
            corrected_images = {}
        
        # Get the first channel to determine image size
        first_ch = channels_to_blend[0]
        if first_ch in corrected_images:
            first_img = corrected_images[first_ch]
        else:
            first_img = self._load_image_with_normalization(self.current_acq_id, first_ch)
        
        if first_img is None:
            QtWidgets.QMessageBox.information(self, "No channels", "Could not load channel images.")
            return
        
        # Load all channel images
        channel_images = []
        for ch_name in channels_to_blend:
            try:
                if ch_name in corrected_images:
                    img = corrected_images[ch_name]
                else:
                    img = self._load_image_with_normalization(self.current_acq_id, ch_name)
                channel_images.append(img.astype(np.float32))
            except Exception:
                channel_images.append(np.zeros_like(first_img, dtype=np.float32))
        
        # Apply custom scaling per color group if enabled
        if self.custom_scaling_chk.isChecked():
            scaled_images = []
            for i, (img, ch_name, color) in enumerate(zip(channel_images, channels_to_blend, channel_colors)):
                # First check if there's color-specific scaling
                if color in self.rgb_color_scaling:
                    vmin = self.rgb_color_scaling[color]['min']
                    vmax = self.rgb_color_scaling[color]['max']
                    if vmax > vmin:
                        img = np.clip((img.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
                    else:
                        img = np.zeros_like(img)
                # Fallback to per-channel scaling if no color scaling is set
                elif ch_name in self.channel_scaling:
                    vmin = self.channel_scaling[ch_name]['min']
                    vmax = self.channel_scaling[ch_name]['max']
                    if vmax > vmin:
                        img = np.clip((img.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
                    else:
                        img = np.zeros_like(img)
                scaled_images.append(img)
            channel_images = scaled_images
        
        # Use additive blending to create RGB composite
        # Alpha is set to 1.0 (hidden from users as requested)
        rgb_img = additive_blend_channels(channel_images, channel_colors, alpha=1.0, normalize_per_channel=True)
        
        # Create title showing channel assignments
        color_groups = {}
        for ch, color in zip(channels_to_blend, channel_colors):
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(ch)
        
        title_parts = []
        for color in ['Red', 'Teal', 'Green', 'Blue', 'Yellow', 'Magenta', 'White']:
            if color in color_groups:
                ch_list = '+'.join(color_groups[color])
                title_parts.append(f"{ch_list} ({color})")
        
        acq_subtitle = self._get_acquisition_subtitle(self.current_acq_id)
        title = " + ".join(title_parts) if title_parts else "Composite"
        title += f"\n{acq_subtitle}"
        if self.segmentation_overlay:
            title += " (segmented)"
        
        # Clear canvas and show composite
        self.canvas.fig.clear()
        
        if grayscale:
            # Convert RGB to grayscale
            ax = self.canvas.fig.add_subplot(111)
            gray_base = np.mean(rgb_img, axis=2)  # Mean across RGB channels
            
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
            # RGB composite display
            ax_main = self.canvas.fig.add_subplot(111)
            
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
                    img_shape = rgb_img.shape[:2]  # (height, width)
                    self.canvas._draw_scale_bar_on_axes(img_shape, scale_bar_length_um, pixel_size_um, ax_main)
        
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
        max_workers = max(1, min(mp.cpu_count() - 2, len(all_channels)))
        
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
        
        export_params = {
            "export_scope": "single_acquisition",
            "include_metadata": include_metadata,
            "denoise_source": denoise_source,
            "denoise_used": self._get_relevant_denoise_settings(
                denoise_source,
                custom_denoise_settings=custom_denoise_settings,
                relevant_channels=channel_names
            ) is not None,
            "normalization_method": normalization_method,
            "percentile_params": list(percentile_params) if normalization_method == "percentile_clip" else None,
            "channels_exported": channel_names,
            "exported_acquisitions": [self.current_acq_id],
            "source_files": self._get_source_files_for_logging([self.current_acq_id]),
            "output_format": "OME-TIFF",
        }
        denoise_settings = self._get_relevant_denoise_settings(
            denoise_source,
            custom_denoise_settings=custom_denoise_settings,
            relevant_channels=channel_names
        )
        if denoise_settings:
            export_params["denoise_settings"] = denoise_settings
        self._log_export_operation(
            "ome_tiff",
            export_params,
            output_path,
            acquisitions=[self.current_acq_id],
            notes=f"Exported {len(channel_names)} channels from {acq_info.name} to OME-TIFF"
        )
        
        return True
    
    def _export_whole_slide(self, output_dir: str, include_metadata: bool, 
                          progress_dlg: ProgressDialog,
                          denoise_source: str, custom_denoise_settings: dict,
                          normalization_method: str, arcsinh_cofactor: float,
                          percentile_params: Tuple[float, float]) -> bool:
        """Export all acquisitions from the slide."""
        total_acquisitions = len(self.acquisitions)
        exported_acq_ids = []
        channel_counts_by_acquisition = {}
        reference_channel_names = None
        
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
            max_workers = max(1, min(mp.cpu_count() - 2, len(all_channels)))
            
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
            
            exported_acq_ids.append(acq_info.id)
            channel_counts_by_acquisition[acq_info.id] = len(channel_names)
            if reference_channel_names is None:
                reference_channel_names = list(channel_names)
        
        if exported_acq_ids:
            export_params = {
                "export_scope": "whole_slide",
                "include_metadata": include_metadata,
                "denoise_source": denoise_source,
                "denoise_used": self._get_relevant_denoise_settings(
                    denoise_source,
                    custom_denoise_settings=custom_denoise_settings
                ) is not None,
                "normalization_method": normalization_method,
                "percentile_params": list(percentile_params) if normalization_method == "percentile_clip" else None,
                "n_exported_acquisitions": len(exported_acq_ids),
                "exported_acquisitions": exported_acq_ids,
                "channel_counts_by_acquisition": channel_counts_by_acquisition,
                "channels_exported": reference_channel_names or [],
                "source_files": self._get_source_files_for_logging(exported_acq_ids),
                "output_format": "OME-TIFF",
            }
            denoise_settings = self._get_relevant_denoise_settings(
                denoise_source,
                custom_denoise_settings=custom_denoise_settings
            )
            if denoise_settings:
                export_params["denoise_settings"] = denoise_settings
            self._log_export_operation(
                "ome_tiff",
                export_params,
                output_dir,
                acquisitions=exported_acq_ids,
                notes=f"Exported {len(exported_acq_ids)} acquisitions to OME-TIFF"
            )
        
        return True
    
    def _export_panel(self):
        """Export panel.csv file from current acquisition."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.information(self, "No acquisitions", "Open a file or folder first.")
            return
        
        if not self.current_acq_id:
            QtWidgets.QMessageBox.information(self, "No acquisition selected", "Please select an acquisition first.")
            return
        
        # Get current acquisition info
        acq_info = self._get_acquisition_info(self.current_acq_id)
        if acq_info is None:
            QtWidgets.QMessageBox.critical(self, "Error", "Could not find acquisition information.")
            return
        
        # Open file dialog to save panel.csv
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Panel CSV",
            "panel.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
        
        try:
            from openimc.core import get_panel
            
            # Generate panel.csv
            output_path = get_panel(acq_info, file_path)
            self._log_export_operation(
                "panel_csv",
                {
                    "export_scope": "single_acquisition",
                    "channels_exported": list(getattr(acq_info, 'channels', []) or []),
                    "exported_acquisitions": [self.current_acq_id],
                    "source_files": self._get_source_files_for_logging([self.current_acq_id]),
                    "output_format": "CSV",
                },
                output_path,
                acquisitions=[self.current_acq_id],
                notes=f"Exported panel.csv for {acq_info.name}"
            )
            
            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Panel CSV exported successfully to:\n{output_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Export Failed",
                f"Failed to export panel CSV:\n{str(e)}"
            )
    
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

    def _get_source_files_for_logging(self, acquisition_ids: Optional[List[str]] = None) -> List[str]:
        """Collect unique source file labels for the provided acquisitions."""
        source_files = []
        seen = set()

        def add_path(file_path: Optional[str]):
            if not file_path:
                return
            if os.path.isdir(file_path):
                label = os.path.basename(file_path.rstrip(os.sep)) or file_path
            else:
                label = os.path.basename(file_path)
            if label and label not in seen:
                seen.add(label)
                source_files.append(label)

        if acquisition_ids:
            for acq_id in acquisition_ids:
                acq_info = self._get_acquisition_info(acq_id)
                if acq_info is not None and getattr(acq_info, 'source_file', None):
                    add_path(acq_info.source_file)
                elif acq_id in self.acq_to_file:
                    add_path(self.acq_to_file[acq_id])
        elif self.acq_to_file:
            for file_path in self.acq_to_file.values():
                add_path(file_path)
        elif self.current_path:
            add_path(self.current_path)

        return source_files

    def _get_source_file_summary_for_logging(self, acquisition_ids: Optional[List[str]] = None) -> Optional[str]:
        """Return a compact source-file summary for logger headers."""
        source_files = self._get_source_files_for_logging(acquisition_ids)
        if not source_files:
            return None
        if len(source_files) == 1:
            return source_files[0]
        if len(source_files) <= 3:
            return ", ".join(source_files)
        return ", ".join(source_files[:3]) + f" and {len(source_files) - 3} more"

    def _get_relevant_denoise_settings(
        self,
        denoise_source: str,
        custom_denoise_settings: Optional[Dict[str, Dict[str, dict]]] = None,
        relevant_channels: Optional[List[str]] = None
    ) -> Optional[Dict[str, Dict[str, dict]]]:
        """Return denoise settings that were actually relevant to an operation."""
        normalized_source = (denoise_source or "none").lower()
        if "viewer" in normalized_source:
            normalized_source = "viewer"
        elif "custom" in normalized_source:
            normalized_source = "custom"
        else:
            normalized_source = "none"
        if normalized_source == "viewer":
            settings = getattr(self, 'channel_denoise', {}) or {}
        elif normalized_source == "custom":
            settings = custom_denoise_settings or {}
        else:
            return None

        if relevant_channels:
            channels = []
            seen = set()
            for channel in relevant_channels:
                if channel and channel not in seen:
                    seen.add(channel)
                    channels.append(channel)
        else:
            channels = list(settings.keys())

        filtered_settings = {}
        for channel in channels:
            channel_settings = settings.get(channel)
            if not isinstance(channel_settings, dict):
                continue
            cleaned = {
                step_name: step_config
                for step_name, step_config in channel_settings.items()
                if step_config
            }
            if cleaned:
                filtered_settings[channel] = cleaned

        return filtered_settings or None

    def _build_segmentation_log_params(
        self,
        preprocessing_config: Optional[dict],
        denoise_source: str,
        custom_denoise_settings: Optional[Dict[str, Dict[str, dict]]],
        acquisitions: Optional[List[str]] = None,
        *,
        show_overlay: bool = True,
        save_masks: bool = False,
        masks_directory: Optional[str] = None,
        segment_scope: str = "current_acquisition"
    ) -> Dict[str, Any]:
        """Build shared segmentation metadata for analysis-step logging."""
        config = preprocessing_config or {}
        nuclear_channels = list(config.get('nuclear_channels') or [])
        cyto_channels = list(config.get('cyto_channels') or [])
        relevant_channels = nuclear_channels + [ch for ch in cyto_channels if ch not in nuclear_channels]
        normalized_source = (denoise_source or "none").lower()
        if "viewer" in normalized_source:
            normalized_source = "viewer"
        elif "custom" in normalized_source:
            normalized_source = "custom"
        else:
            normalized_source = "none"
        denoise_settings = self._get_relevant_denoise_settings(
            normalized_source,
            custom_denoise_settings=custom_denoise_settings,
            relevant_channels=relevant_channels
        )

        params = {
            "segment_scope": segment_scope,
            "show_overlay": bool(show_overlay),
            "save_masks": bool(save_masks),
            "masks_directory": masks_directory if save_masks else None,
            "normalization_method": config.get('normalization_method'),
            "arcsinh_cofactor": config.get('arcsinh_cofactor') if config.get('normalization_method') == 'arcsinh' else None,
            "percentile_params": list(config.get('percentile_params')) if config.get('normalization_method') == 'percentile_clip' and config.get('percentile_params') is not None else None,
            "nuclear_channels": nuclear_channels,
            "cyto_channels": cyto_channels,
            "nuclear_combo_method": config.get('nuclear_combo_method'),
            "cyto_combo_method": config.get('cyto_combo_method'),
            "nuclear_weights": config.get('nuclear_weights'),
            "cyto_weights": config.get('cyto_weights'),
            "selected_channel_count": len(relevant_channels),
            "input_mode": "combined" if (nuclear_channels and cyto_channels) else ("nuclear" if nuclear_channels else ("cyto" if cyto_channels else "unknown")),
            "denoise_source": normalized_source,
            "denoise_used": denoise_settings is not None,
            "source_files": self._get_source_files_for_logging(acquisitions),
        }
        if denoise_settings:
            params["denoise_settings"] = denoise_settings
        return params

    def _log_export_operation(
        self,
        export_type: str,
        parameters: Dict[str, Any],
        output_path: str,
        acquisitions: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> None:
        """Write a structured export entry to the methods log."""
        logger = get_logger()
        logger.log_export(
            export_type=export_type,
            parameters=parameters,
            output_path=output_path,
            acquisitions=acquisitions or [],
            notes=notes,
            source_file=self._get_source_file_summary_for_logging(acquisitions)
        )

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
                    if not _load_cellsam_helpers():
                        detail = f"\n\nDetails: {_CELLSAM_IMPORT_ERROR}" if _CELLSAM_IMPORT_ERROR else ""
                        QtWidgets.QMessageBox.critical(
                            self, "Missing dependency", 
                            "CellSAM library is required for segmentation.\n"
                            "Install it with: pip install git+https://github.com/vanvalenlab/cellSAM.git"
                            f"{detail}"
                        )
                        return
                elif model != "Classical Watershed":
                    if _load_cellpose_models() is None:
                        detail = f"\n\nDetails: {_CELLPOSE_IMPORT_ERROR}" if _CELLPOSE_IMPORT_ERROR else ""
                        QtWidgets.QMessageBox.critical(
                            self, "Missing dependency",
                            "Cellpose library is required for segmentation.\n"
                            "Install it with: pip install cellpose"
                            f"{detail}"
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
                        # Check if dialog has a specific list of acquisitions to segment (from "Segment Missing Masks")
                        acquisitions_to_segment = None
                        if hasattr(dlg, '_acquisitions_to_segment'):
                            acquisitions_to_segment = dlg._acquisitions_to_segment
                        
                        # Get num_workers from dialog (for Cellpose)
                        num_workers = dlg.get_num_workers() if hasattr(dlg, 'get_num_workers') else None
                        
                        # Run segmentation on all acquisitions (or specific subset)
                        self._perform_segmentation_all_acquisitions(
                            model, diameter, flow_threshold, cellprob_threshold, 
                            show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config,
                            denoise_source, custom_denoise_settings, dlg,
                            acquisitions_to_segment=acquisitions_to_segment,
                            num_workers=num_workers
                        )
                    else:
                        # Run segmentation on current acquisition only
                        self._perform_segmentation(
                            model, diameter, flow_threshold, cellprob_threshold, 
                            show_overlay, save_masks, masks_directory, gpu_id, preprocessing_config, use_viewer_denoising,
                            denoise_source, custom_denoise_settings, dlg
                        )
                except Exception as e:
                    # Check if this is a CUDA memory error
                    try:
                        from openimc.processing.custom_cellsam import CUDAMemoryError
                    except Exception:
                        CUDAMemoryError = None
                    if CUDAMemoryError is not None and isinstance(e, CUDAMemoryError):
                        # Show user-friendly CUDA memory error message
                        QtWidgets.QMessageBox.critical(
                            self, "CUDA Out of Memory", 
                            f"{str(e)}\n\n"
                            f"Segmentation has been cancelled. Please reduce the batch size and try again."
                        )
                    else:
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
                    if isinstance(labels, np.ndarray):
                        labels = labels.copy()
                    
                    # Update acquisition info cache
                    acq_info = self._get_acquisition_info(self.current_acq_id)
                    if acq_info:
                        self.segmentation_masks.set_acq_info(self.current_acq_id, acq_info)
                    
                    # Store mask using mask manager (always in memory)
                    self.mask_manager.set_mask(
                        self.current_acq_id, labels,
                        save_to_disk=False,
                        acq_info=acq_info
                    )
                    
                    # Log segmentation operation
                    logger = get_logger()
                    n_cells = len(np.unique(labels)) - 1  # Exclude background
                    
                    # Get project path
                    project_path = dlg.get_project_path()
                    denoise_source = seg_dlg.get_denoise_source()
                    custom_denoise_settings = seg_dlg.get_custom_denoise_settings()
                    params = self._build_segmentation_log_params(
                        preprocessing_config,
                        denoise_source,
                        custom_denoise_settings,
                        acquisitions=[self.current_acq_id],
                        show_overlay=True,
                        save_masks=False,
                        masks_directory=None,
                        segment_scope="current_acquisition"
                    )
                    params.update({
                        "method": "ilastik",
                        "project_path": project_path,
                        "n_cells": int(n_cells),
                    })
                    
                    logger.log_segmentation(
                        method="ilastik",
                        parameters=params,
                        acquisitions=[self.current_acq_id],
                        notes=f"Ilastik segmented {n_cells} cells/regions",
                        source_file=self._get_source_file_summary_for_logging([self.current_acq_id])
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
                        
                        # Update acquisition info cache
                        acq_info = self._get_acquisition_info(self.current_acq_id)
                        if acq_info:
                            self.segmentation_masks.set_acq_info(self.current_acq_id, acq_info)
                        
                        # Store mask using mask manager (always in memory)
                        self.mask_manager.set_mask(
                            self.current_acq_id, labels,
                            save_to_disk=False,
                            acq_info=acq_info
                        )
                        
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
            # Get loader and acquisition info
            loader = self._get_loader_for_acquisition(self.current_acq_id)
            if loader is None:
                raise ValueError("No loader found for current acquisition")
            
            acq_info = self._get_acquisition_info(self.current_acq_id)
            if acq_info is None:
                raise ValueError(f"Acquisition {self.current_acq_id} not found")
            
            # Get channel information from preprocessing config
            nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
            cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
            nuclear_combo_method = preprocessing_config.get('nuclear_combo_method', 'mean') if preprocessing_config else 'mean'
            cyto_combo_method = preprocessing_config.get('cyto_combo_method', 'mean') if preprocessing_config else 'mean'
            nuclear_weights = preprocessing_config.get('nuclear_weights') if preprocessing_config else None
            cyto_weights = preprocessing_config.get('cyto_weights') if preprocessing_config else None
            
            # Convert denoise_source to denoise_settings format
            denoise_settings = None
            if denoise_source == "viewer" and use_viewer_denoising:
                # Use viewer denoising settings
                denoise_settings = self.channel_denoise.copy() if hasattr(self, 'channel_denoise') else {}
            elif denoise_source == "custom" and custom_denoise_settings:
                # Use custom denoise settings from dialog
                denoise_settings = custom_denoise_settings.copy()
            
            # Map GUI model names to core method names
            if model == "Classical Watershed":
                # Watershed uses worker function directly (keep existing implementation)
                progress_dlg.update_progress(5, "Initializing watershed segmentation", "Using classical watershed algorithm...")
                
                if dlg is None:
                    raise ValueError("Dialog object is required for watershed segmentation")
                
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
                
                # Get original acquisition ID if this is a unique ID
                original_acq_id = self._get_original_acq_id(self.current_acq_id)
                img_stack = loader.get_all_channels(original_acq_id)
                channel_names = loader.get_channels(original_acq_id)
                
                # Run watershed segmentation in a worker thread to keep UI responsive.
                progress_dlg.update_progress(20, "Running watershed segmentation", "Processing...")
                def _watershed_task():
                    return watershed_segmentation(
                        img_stack, channel_names, nuclear_channels, cyto_channels,
                        denoise_settings=denoise_settings,
                        nuclear_fusion_method=nuclear_fusion_method,
                        nuclear_weights=nuclear_weights,
                        seed_threshold_method=seed_threshold_method,
                        min_seed_area=min_seed_area,
                        min_distance_peaks=min_distance_peaks,
                        membrane_fusion_method=membrane_fusion_method,
                        membrane_weights=cyto_weights,
                        boundary_method=boundary_method,
                        boundary_sigma=boundary_sigma,
                        compactness=compactness,
                        min_cell_area=min_cell_area,
                        max_cell_area=max_cell_area,
                        tile_size=tile_size,
                        tile_overlap=tile_overlap,
                        rng_seed=rng_seed
                    )
                masks = run_task_with_event_pump(_watershed_task, poll_interval_ms=80)
                masks = [masks]  # Convert to list format for consistency
                flows = [None]
                styles = [None]
                diams = [None]
                
                # Log watershed segmentation
                logger = get_logger()
                n_cells = len(np.unique(masks[0])) - 1
                params = self._build_segmentation_log_params(
                    preprocessing_config,
                    denoise_source,
                    custom_denoise_settings,
                    acquisitions=[self.current_acq_id],
                    show_overlay=show_overlay,
                    save_masks=save_masks,
                    masks_directory=masks_directory,
                    segment_scope="current_acquisition"
                )
                params.update({
                    "method": "watershed",
                    "nuclear_fusion_method": nuclear_fusion_method,
                    "seed_threshold_method": seed_threshold_method,
                    "min_seed_area": min_seed_area,
                    "min_distance_peaks": min_distance_peaks,
                    "membrane_fusion_method": membrane_fusion_method,
                    "membrane_weights": cyto_weights,
                    "boundary_method": boundary_method,
                    "boundary_sigma": boundary_sigma,
                    "compactness": compactness,
                    "min_cell_area": min_cell_area,
                    "max_cell_area": max_cell_area,
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "rng_seed": rng_seed,
                    "n_cells": int(n_cells),
                })
                logger.log_segmentation(
                    method="watershed",
                    parameters=params,
                    acquisitions=[self.current_acq_id],
                    output_path=masks_directory if save_masks else None,
                    notes=f"Segmented {n_cells} cells",
                    source_file=self._get_source_file_summary_for_logging([self.current_acq_id])
                )
                
            else:
                # Use core.segment() for Cellpose and CellSAM
                if model == "DeepCell CellSAM":
                    progress_dlg.update_progress(0, "Initializing DeepCell CellSAM model", "Loading model...")
                    core_method = "cellsam"
                    
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
                    
                    # Get CellSAM parameters from dialog
                    bbox_threshold = dlg.get_cellsam_bbox_threshold() if dlg else 0.4
                    use_wsi = dlg.get_cellsam_use_wsi() if dlg else False
                    low_contrast_enhancement = dlg.get_cellsam_low_contrast_enhancement() if dlg else False
                    gauge_cell_size = dlg.get_cellsam_gauge_cell_size() if dlg else False
                    deepcell_api_key = os.environ.get("DEEPCELL_ACCESS_TOKEN")
                    
                elif model == "Cellpose Nuclei":
                    progress_dlg.update_progress(0, "Initializing Cellpose model", "Loading model...")
                    core_method = "cellpose"
                    cellpose_model = "nuclei"
                    
                    # Determine GPU usage
                    if gpu_id == "auto":
                        torch = get_torch_module()
                        if torch is not None and torch.cuda.is_available():
                            gpu_id = 0
                        elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            gpu_id = 'mps'
                        else:
                            gpu_id = None
                    
                else:  # Cellpose Cyto3
                    progress_dlg.update_progress(0, "Initializing Cellpose model", "Loading model...")
                    core_method = "cellpose"
                    cellpose_model = "cyto3"
                    
                    # Determine GPU usage
                    if gpu_id == "auto":
                        torch = get_torch_module()
                        if torch is not None and torch.cuda.is_available():
                            gpu_id = 0
                        elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                            gpu_id = 'mps'
                        else:
                            gpu_id = None
                
                progress_dlg.update_progress(20, "Running segmentation", "Processing...")
                
                # Get normalization settings from preprocessing_config
                norm_method = preprocessing_config.get('normalization_method', 'channelwise_minmax') if preprocessing_config else 'channelwise_minmax'
                arcsinh_cofactor = preprocessing_config.get('arcsinh_cofactor', 1.0) if preprocessing_config else 1.0
                percentile_params = preprocessing_config.get('percentile_params', (1.0, 99.0)) if preprocessing_config else (1.0, 99.0)
                
                # Call segmentation with UI-safe execution.
                if core_method == "cellsam":
                    if cellsam_pipeline_subprocess is None:
                        raise ImportError("CellSAM subprocess entry point is unavailable.")

                    progress_dlg.update_progress(25, "Preprocessing images", "Preparing CellSAM input...")
                    def _preprocess_cellsam_task():
                        return self._preprocess_channels_for_segmentation(
                            preprocessing_config,
                            None,
                            use_viewer_denoising=use_viewer_denoising,
                            denoise_source=denoise_source,
                            custom_denoise_settings=custom_denoise_settings
                        )

                    nuclear_img, cyto_img = run_task_with_event_pump(
                        _preprocess_cellsam_task,
                        poll_interval_ms=80,
                    )

                    if nuclear_channels and cyto_channels:
                        h, w = nuclear_img.shape
                        cellsam_input = np.zeros((h, w, 3), dtype=np.float32)
                        cellsam_input[:, :, 1] = nuclear_img
                        cellsam_input[:, :, 2] = cyto_img if cyto_img is not None else nuclear_img
                    elif nuclear_channels:
                        cellsam_input = nuclear_img
                    elif cyto_channels:
                        cellsam_input = cyto_img if cyto_img is not None else nuclear_img
                    else:
                        raise ValueError("At least one channel (nuclear or cyto) must be selected for CellSAM")

                    progress_dlg.update_progress(45, "Running CellSAM", "Segmenting cells...")
                    cellsam_task = partial(
                        cellsam_pipeline_subprocess,
                        cellsam_input,
                        bbox_threshold=bbox_threshold,
                        use_wsi=use_wsi,
                        low_contrast_enhancement=low_contrast_enhancement,
                        gauge_cell_size=gauge_cell_size,
                    )
                    mask = run_task_with_event_pump(
                        cellsam_task,
                        poll_interval_ms=80,
                        use_process=True,
                    )
                else:  # cellpose
                    def _cellpose_segment_task():
                        return segment(
                            loader=loader,
                            acquisition=acq_info,
                            method="cellpose",
                            nuclear_channels=nuclear_channels,
                            cyto_channels=cyto_channels,
                            output_dir=masks_directory if save_masks else None,
                            denoise_settings=denoise_settings,
                            normalization_method=norm_method,
                            arcsinh_cofactor=arcsinh_cofactor,
                            percentile_params=percentile_params,
                            nuclear_combo_method=nuclear_combo_method,
                            cyto_combo_method=cyto_combo_method,
                            nuclear_weights=nuclear_weights,
                            cyto_weights=cyto_weights,
                            cellpose_model=cellpose_model,
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            gpu_id=gpu_id
                        )
                    mask = run_task_with_event_pump(_cellpose_segment_task, poll_interval_ms=80)
                
                masks = [mask]
                flows = [None]
                styles = [None]
                diams = [None]
                
                # Log segmentation for Cellpose and CellSAM
                logger = get_logger()
                n_cells = len(np.unique(masks[0])) - 1
                params = self._build_segmentation_log_params(
                    preprocessing_config,
                    denoise_source,
                    custom_denoise_settings,
                    acquisitions=[self.current_acq_id],
                    show_overlay=show_overlay,
                    save_masks=save_masks,
                    masks_directory=masks_directory,
                    segment_scope="current_acquisition"
                )
                if core_method == "cellsam":
                    params.update({
                        "method": "cellsam",
                        "bbox_threshold": bbox_threshold,
                        "use_wsi": use_wsi,
                        "low_contrast_enhancement": low_contrast_enhancement,
                        "gauge_cell_size": gauge_cell_size,
                        "n_cells": int(n_cells),
                    })
                else:  # cellpose
                    params.update({
                        "method": "cellpose",
                        "model_type": cellpose_model,
                        "diameter": diameter,
                        "flow_threshold": flow_threshold,
                        "cellprob_threshold": cellprob_threshold,
                        "gpu_id": str(gpu_id) if gpu_id is not None else None,
                        "n_cells": int(n_cells),
                    })
                
                method_name = "cellsam" if core_method == "cellsam" else "cellpose"
                logger.log_segmentation(
                    method=method_name,
                    parameters=params,
                    acquisitions=[self.current_acq_id],
                    output_path=masks_directory if save_masks else None,
                    notes=f"Segmented {n_cells} cells",
                    source_file=self._get_source_file_summary_for_logging([self.current_acq_id])
                )
            
            progress_dlg.update_progress(80, "Processing results", "Creating segmentation masks...")
            
            # Store segmentation results using mask manager
            mask = masks[0]  # First (and only) mask
            if isinstance(mask, np.ndarray):
                mask = mask.copy()
            
            # Update acquisition info cache
            self.segmentation_masks.set_acq_info(self.current_acq_id, acq_info)
            
            # Store mask in memory (save to disk only if user explicitly requested)
            if save_masks and masks_directory:
                save_masks_directory_preference(masks_directory)
                self.mask_manager.set_masks_directory(masks_directory)
            
            self.mask_manager.set_mask(
                self.current_acq_id, mask,
                save_to_disk=save_masks,
                acq_info=acq_info,
                masks_directory=masks_directory
            )
            
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
            
            # Get channel information for display
            channel_info = ""
            if nuclear_channels:
                channel_info += f"Nuclear: {len(nuclear_channels)} channels"
            if cyto_channels:
                if channel_info:
                    channel_info += f" + Cytoplasm: {len(cyto_channels)} channels"
                else:
                    channel_info += f"Cytoplasm: {len(cyto_channels)} channels"
            
            n_cells = len(np.unique(masks[0])) - 1
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
                                             denoise_source: str = "Use viewer settings", custom_denoise_settings: dict = None, dlg = None,
                                             acquisitions_to_segment: List = None, num_workers: int = None):
        """Perform efficient batch segmentation on all acquisitions or a specific subset."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions available for segmentation.")
            return
        
        # Use specific acquisitions list if provided (from "Segment Missing Masks"), otherwise use all
        if acquisitions_to_segment is not None:
            acquisitions_to_process = acquisitions_to_segment
        else:
            acquisitions_to_process = self.acquisitions
        
        if not acquisitions_to_process:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions to segment.")
            return
        
        # Create progress dialog for batch processing
        total_acquisitions = len(acquisitions_to_process)
        progress_dlg = ProgressDialog("Batch Cell Segmentation", self)
        progress_dlg.set_maximum(total_acquisitions)
        progress_dlg.show()
        
        # Watershed batch processing: sequential (one at a time)
        if model == "Classical Watershed":
            self._perform_watershed_batch_segmentation(
                preprocessing_config, denoise_source, custom_denoise_settings, 
                save_masks, masks_directory, show_overlay, dlg, progress_dlg, total_acquisitions,
                acquisitions_to_process=acquisitions_to_process
            )
            return
        
        # CellSAM batch processing: sequential (one at a time) since CellSAM doesn't support batch processing
        if model == "DeepCell CellSAM":
            self._perform_cellsam_batch_segmentation(
                preprocessing_config, denoise_source, custom_denoise_settings, 
                save_masks, masks_directory, show_overlay, dlg, progress_dlg, total_acquisitions,
                acquisitions_to_process=acquisitions_to_process
            )
            return
        
        try:
            # Initialize Cellpose model once
            progress_dlg.update_progress(0, "Initializing Cellpose model", f"Loading model... (0/{total_acquisitions} completed)")
            
            # Determine GPU usage
            use_gpu = False
            gpu_device = None
            
            if gpu_id == "auto":
                torch = get_torch_module()
                if torch is not None and torch.cuda.is_available():
                    use_gpu = True
                    gpu_device = 0
                elif torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    use_gpu = True
                    gpu_device = 'mps'
            elif gpu_id is not None:
                use_gpu = True
                gpu_device = gpu_id
            
            # Initialize model
            loaded_cellpose_models = _load_cellpose_models()
            if loaded_cellpose_models is None:
                detail = f"\n\nDetails: {_CELLPOSE_IMPORT_ERROR}" if _CELLPOSE_IMPORT_ERROR else ""
                QtWidgets.QMessageBox.critical(
                    self,
                    "Missing dependency",
                    "Cellpose library is required for batch segmentation.\n"
                    "Install it with: pip install cellpose"
                    f"{detail}"
                )
                progress_dlg.close()
                return

            if model == "nuclei":
                model_obj = loaded_cellpose_models.Cellpose(gpu=use_gpu, model_type='nuclei')
            else:  # cyto3
                model_obj = loaded_cellpose_models.Cellpose(gpu=use_gpu, model_type='cyto3')
            
            # Process acquisitions sequentially
            successful_segmentations = 0
            successful_acq_ids = []
            failed_acq_ids = []
            cell_counts_by_acquisition = {}
            
            for acq_idx, acq in enumerate(acquisitions_to_process):
                if progress_dlg.is_cancelled():
                    break
                
                progress_dlg.update_progress(
                    successful_segmentations, 
                    f"Processing acquisition {acq_idx + 1}/{total_acquisitions}", 
                    f"Loading {acq.name}... ({successful_segmentations}/{total_acquisitions} completed)"
                )
                
                try:
                    # Process single acquisition
                    acq_id = acq.id
                    acq_name = acq.name
                    
                    # Get acquisition info
                    acq_info = self._get_acquisition_info(acq_id)
                    if acq_info is None:
                        raise ValueError(f"Acquisition {acq_id} not found")
                    
                    # Preprocess this acquisition
                    nuclear_img, cyto_img = self._preprocess_acquisition_for_cellpose(
                        acq_id, preprocessing_config, progress_dlg, denoise_source, custom_denoise_settings
                    )
                    
                    # Prepare Cellpose input
                    nuclear_channels_list = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
                    cyto_channels_list = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
                    
                    if nuclear_channels_list and cyto_channels_list:
                        # Combined mode: stack nuclear and cyto channels
                        if nuclear_img is None or cyto_img is None:
                            raise ValueError("Failed to load channels for combined mode")
                        cellpose_input = [np.stack([nuclear_img, cyto_img], axis=0)]
                        channels = [1, 2]
                    elif nuclear_channels_list:
                        if nuclear_img is None:
                            raise ValueError("Failed to load nuclear channels")
                        cellpose_input = [nuclear_img]
                        channels = [0, 0]
                    elif cyto_channels_list:
                        if cyto_img is None:
                            raise ValueError("Failed to load cytoplasm channels")
                        cellpose_input = [cyto_img]
                        channels = [0, 0]
                    else:
                        raise ValueError("At least one channel (nuclear or cyto) must be selected")
                    
                    # Run segmentation
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Segmenting {acq_name}",
                        f"Running Cellpose... ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                    # Run Cellpose eval in worker thread to keep GUI responsive.
                    # Note: Cellpose eval() does not support nproc parameter.
                    def _cellpose_eval_task():
                        return model_obj.eval(
                            cellpose_input,
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            channels=channels
                        )

                    masks, flows, styles, diams = run_task_with_event_pump(
                        _cellpose_eval_task,
                        poll_interval_ms=80,
                    )
                    
                    mask = masks[0] if len(masks) > 0 else np.zeros((1, 1), dtype=np.int32)
                    
                    # Explicitly release memory
                    del cellpose_input, nuclear_img, cyto_img
                    del masks, flows, styles, diams
                    import gc
                    gc.collect()
                    torch = get_torch_module()
                    if torch is not None and use_gpu and hasattr(torch, "cuda"):
                        torch.cuda.empty_cache()
                    
                    # Update acquisition info cache
                    if acq_info:
                        self.segmentation_masks.set_acq_info(acq_id, acq_info)
                    
                    # Store mask in memory (save to disk only if user requested)
                    if isinstance(mask, np.ndarray):
                        mask = mask.copy()
                    self.mask_manager.set_mask(
                        acq_id, mask,
                        save_to_disk=save_masks,
                        acq_info=acq_info,
                        masks_directory=masks_directory
                    )
                    
                    # Clear colors for this acquisition so they get regenerated
                    if acq_id in self.segmentation_colors:
                        del self.segmentation_colors[acq_id]
                    if acq_id in self.cluster_colors:
                        del self.cluster_colors[acq_id]
                    if acq_id in self.cluster_color_map:
                        del self.cluster_color_map[acq_id]
                    
                    n_cells = int(len(np.unique(mask)) - 1)
                    successful_acq_ids.append(acq_id)
                    cell_counts_by_acquisition[acq_id] = n_cells
                    successful_segmentations += 1
                    
                    # Update progress
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Completed {acq_name}",
                        f"Segmented {acq_name} ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                except Exception as e:
                    print(f"Error segmenting acquisition {acq.name}: {e}")
                    failed_acq_ids.append(acq.id)
                    # Continue with next acquisition
                    continue

            if successful_acq_ids:
                params = self._build_segmentation_log_params(
                    preprocessing_config,
                    denoise_source,
                    custom_denoise_settings,
                    acquisitions=successful_acq_ids,
                    show_overlay=show_overlay,
                    save_masks=save_masks,
                    masks_directory=masks_directory,
                    segment_scope="batch"
                )
                params.update({
                    "method": "cellpose",
                    "model_type": "nuclei" if model == "nuclei" else "cyto3",
                    "diameter": diameter,
                    "flow_threshold": flow_threshold,
                    "cellprob_threshold": cellprob_threshold,
                    "gpu_id": str(gpu_device) if gpu_device is not None else None,
                    "n_requested_acquisitions": total_acquisitions,
                    "n_successful_acquisitions": len(successful_acq_ids),
                    "successful_acquisitions": successful_acq_ids,
                    "failed_acquisitions": failed_acq_ids,
                    "cell_counts_by_acquisition": cell_counts_by_acquisition,
                })
                get_logger().log_segmentation(
                    method="cellpose",
                    parameters=params,
                    acquisitions=successful_acq_ids,
                    output_path=masks_directory if save_masks else None,
                    notes=f"Batch-segmented {len(successful_acq_ids)} of {total_acquisitions} acquisitions",
                    source_file=self._get_source_file_summary_for_logging(successful_acq_ids)
                )
            
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
                                            masks_directory: str, show_overlay: bool, dlg, progress_dlg, total_acquisitions: int,
                                            acquisitions_to_process: List = None):
        """Perform sequential batch segmentation for CellSAM (one acquisition at a time)."""
        if acquisitions_to_process is None:
            acquisitions_to_process = self.acquisitions
        if dlg is None:
            QtWidgets.QMessageBox.critical(
                self, "Missing Dialog", 
                "Dialog object is required for DeepCell CellSAM segmentation."
            )
            progress_dlg.close()
            return
        
        # Set up masks directory if saving to disk
        if save_masks and masks_directory:
            save_masks_directory_preference(masks_directory)
            self.mask_manager.set_masks_directory(masks_directory)
        
        try:
            # Initialize CellSAM model once
            progress_dlg.update_progress(0, "Initializing DeepCell CellSAM", "Setting up model...")
            QtWidgets.QApplication.processEvents()
            
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
            
            if not _load_cellsam_helpers():
                progress_dlg.close()
                detail = f"\n\nDetails: {_CELLSAM_IMPORT_ERROR}" if _CELLSAM_IMPORT_ERROR else ""
                QtWidgets.QMessageBox.critical(
                    self, "CellSAM Not Available",
                    "CellSAM is not installed or failed to load.\n\n"
                    "Please install with: pip install git+https://github.com/vanvalenlab/cellSAM.git"
                    f"{detail}"
                )
                return
            
            # Get CellSAM parameters from dialog
            bbox_threshold = dlg.get_cellsam_bbox_threshold()
            use_wsi = dlg.get_cellsam_use_wsi()
            low_contrast_enhancement = dlg.get_cellsam_low_contrast_enhancement()
            gauge_cell_size = dlg.get_cellsam_gauge_cell_size()
            
            # Pre-initialize the model to download weights if needed (prevents hanging during first segmentation)
            # This ensures the model is loaded and weights are downloaded before processing acquisitions
            progress_dlg.update_progress(5, "Initializing DeepCell CellSAM", "Loading model weights (this may take a moment on first use)...")
            QtWidgets.QApplication.processEvents()
            try:
                from openimc.processing.custom_cellsam import _get_cached_model
                # Pre-initialize the model in a worker thread - this can download/extract
                # large weights and would otherwise freeze the UI.
                def _cellsam_init_task():
                    return _get_cached_model(model_path=None, bbox_threshold=bbox_threshold)

                model = run_task_with_event_pump(_cellsam_init_task, poll_interval_ms=100)
                progress_dlg.update_progress(10, "Initializing DeepCell CellSAM", "Model loaded successfully")
                QtWidgets.QApplication.processEvents()
            except Exception as e:
                progress_dlg.close()
                QtWidgets.QMessageBox.critical(
                    self, "Model Initialization Failed",
                    f"Failed to initialize CellSAM model:\n{str(e)}\n\n"
                    "This may be due to:\n"
                    "- Invalid API key\n"
                    "- Network connectivity issues\n"
                    "- Insufficient disk space for model weights\n\n"
                    "Please check your API key and internet connection."
                )
                return
            
            # Process each acquisition sequentially
            successful_segmentations = 0
            successful_acq_ids = []
            failed_acq_ids = []
            cell_counts_by_acquisition = {}
            
            for acq_idx, acq in enumerate(acquisitions_to_process):
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
                    # Get acquisition info
                    acq_info = self._get_acquisition_info(acq_id)
                    if acq_info is None:
                        raise ValueError(f"Acquisition {acq_id} not found")
                    
                    # Preprocess this acquisition
                    nuclear_img, cyto_img = self._preprocess_acquisition_for_cellsam(
                        acq_id, preprocessing_config, progress_dlg, denoise_source, custom_denoise_settings
                    )
                    
                    # Clear loader image cache immediately after preprocessing to free memory
                    loader = self._get_loader_for_acquisition(acq_id)
                    if loader is not None and hasattr(loader, '_image_cache'):
                        cache_size_before = len(loader._image_cache)
                        loader._image_cache.clear()
                        import gc
                        gc.collect()
                    
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
                    
                    def _cellsam_segment_task():
                        return cellsam_pipeline(
                            cellsam_input,
                            bbox_threshold=bbox_threshold,
                            use_wsi=use_wsi,
                            low_contrast_enhancement=low_contrast_enhancement,
                            gauge_cell_size=gauge_cell_size
                        )

                    mask = run_task_with_event_pump(_cellsam_segment_task, poll_interval_ms=80)
                    
                    # Explicitly release input images memory immediately after segmentation
                    del cellsam_input
                    if 'nuclear_img' in locals() and nuclear_img is not None:
                        del nuclear_img
                    if 'cyto_img' in locals() and cyto_img is not None:
                        del cyto_img
                    import gc
                    gc.collect()
                    
                    # Update acquisition info cache
                    if acq_info:
                        self.segmentation_masks.set_acq_info(acq_id, acq_info)
                    
                    # Store mask in memory (save to disk only if user requested)
                    if isinstance(mask, np.ndarray):
                        mask = mask.copy()
                    self.mask_manager.set_mask(
                        acq_id, mask,
                        save_to_disk=save_masks,
                        acq_info=acq_info,
                        masks_directory=masks_directory
                    )
                    # mask_manager.set_mask already saves with _segmentation_masks.tif naming
                    # No need for redundant save
                    
                    # Clear colors for this acquisition so they get regenerated
                    if acq_id in self.segmentation_colors:
                        del self.segmentation_colors[acq_id]
                    if acq_id in self.cluster_colors:
                        del self.cluster_colors[acq_id]
                    if acq_id in self.cluster_color_map:
                        del self.cluster_color_map[acq_id]
                    
                    n_cells = int(len(np.unique(mask)) - 1)
                    successful_acq_ids.append(acq_id)
                    cell_counts_by_acquisition[acq_id] = n_cells
                    successful_segmentations += 1
                    
                    # Update progress
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Completed {acq_name}",
                        f"Segmented {acq_name} ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                except Exception as e:
                    print(f"Error segmenting acquisition {acq_name} ({acq_id}): {e}")
                    failed_acq_ids.append(acq_id)
                    # Continue with next acquisition
                    continue

            if successful_acq_ids:
                params = self._build_segmentation_log_params(
                    preprocessing_config,
                    denoise_source,
                    custom_denoise_settings,
                    acquisitions=successful_acq_ids,
                    show_overlay=show_overlay,
                    save_masks=save_masks,
                    masks_directory=masks_directory,
                    segment_scope="batch"
                )
                params.update({
                    "method": "cellsam",
                    "bbox_threshold": bbox_threshold,
                    "use_wsi": use_wsi,
                    "low_contrast_enhancement": low_contrast_enhancement,
                    "gauge_cell_size": gauge_cell_size,
                    "n_requested_acquisitions": total_acquisitions,
                    "n_successful_acquisitions": len(successful_acq_ids),
                    "successful_acquisitions": successful_acq_ids,
                    "failed_acquisitions": failed_acq_ids,
                    "cell_counts_by_acquisition": cell_counts_by_acquisition,
                })
                get_logger().log_segmentation(
                    method="cellsam",
                    parameters=params,
                    acquisitions=successful_acq_ids,
                    output_path=masks_directory if save_masks else None,
                    notes=f"Batch-segmented {len(successful_acq_ids)} of {total_acquisitions} acquisitions",
                    source_file=self._get_source_file_summary_for_logging(successful_acq_ids)
                )
            
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
            # Check if this is a CUDA memory error
            try:
                from openimc.processing.custom_cellsam import CUDAMemoryError
            except Exception:
                CUDAMemoryError = None
            if CUDAMemoryError is not None and isinstance(e, CUDAMemoryError):
                # Show user-friendly CUDA memory error message
                QtWidgets.QMessageBox.critical(
                    self, "CUDA Out of Memory", 
                    f"{str(e)}\n\n"
                    f"Segmentation has been cancelled. Please reduce the batch size and try again."
                )
            else:
                QtWidgets.QMessageBox.critical(
                    self, "Batch Segmentation Failed", 
                    f"Batch segmentation failed with error:\n{str(e)}"
                )
        finally:
            progress_dlg.close()
    
    
    def _perform_watershed_batch_segmentation(self, preprocessing_config: dict, denoise_source: str, 
                                               custom_denoise_settings: dict, save_masks: bool, 
                                               masks_directory: str, show_overlay: bool, dlg, progress_dlg, total_acquisitions: int,
                                               acquisitions_to_process: List = None):
        """Perform sequential batch segmentation for Classical Watershed (one acquisition at a time)."""
        if acquisitions_to_process is None:
            acquisitions_to_process = self.acquisitions
        if dlg is None:
            QtWidgets.QMessageBox.critical(
                self, "Missing Dialog", 
                "Dialog object is required for Classical Watershed segmentation."
            )
            progress_dlg.close()
            return
        
        # Set up masks directory if saving to disk
        if save_masks and masks_directory:
            save_masks_directory_preference(masks_directory)
            self.mask_manager.set_masks_directory(masks_directory)
        
        try:
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
            
            # Get channel information from preprocessing config
            nuclear_channels = preprocessing_config.get('nuclear_channels', []) if preprocessing_config else []
            cyto_channels = preprocessing_config.get('cyto_channels', []) if preprocessing_config else []
            nuclear_weights = preprocessing_config.get('nuclear_weights') if preprocessing_config else None
            cyto_weights = preprocessing_config.get('cyto_weights') if preprocessing_config else None
            
            # Convert denoise_source to denoise_settings format
            denoise_settings = None
            if denoise_source == "viewer" and hasattr(self, 'channel_denoise'):
                # Use viewer denoising settings
                denoise_settings = self.channel_denoise.copy() if hasattr(self, 'channel_denoise') else {}
            elif denoise_source == "custom" and custom_denoise_settings:
                # Use custom denoise settings from dialog
                denoise_settings = custom_denoise_settings.copy()
            
            # Process each acquisition sequentially
            successful_segmentations = 0
            successful_acq_ids = []
            failed_acq_ids = []
            cell_counts_by_acquisition = {}
            
            for acq_idx, acq in enumerate(acquisitions_to_process):
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
                    # Get loader and acquisition info
                    loader = self._get_loader_for_acquisition(acq_id)
                    if loader is None:
                        raise ValueError(f"No loader found for acquisition {acq_id}")
                    
                    acq_info = self._get_acquisition_info(acq_id)
                    if acq_info is None:
                        raise ValueError(f"Acquisition {acq_id} not found")
                    
                    # Get original acquisition ID if this is a unique ID
                    original_acq_id = self._get_original_acq_id(acq_id)
                    
                    # Load image stack
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Loading {acq_name}",
                        f"Loading image data... ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                    img_stack = loader.get_all_channels(original_acq_id)
                    channel_names = loader.get_channels(original_acq_id)
                    
                    # Run watershed segmentation
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Segmenting {acq_name}",
                        f"Running watershed segmentation... ({successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                    def _watershed_batch_task():
                        return watershed_segmentation(
                            img_stack, channel_names, nuclear_channels, cyto_channels,
                            denoise_settings=denoise_settings,
                            nuclear_fusion_method=nuclear_fusion_method,
                            nuclear_weights=nuclear_weights,
                            seed_threshold_method=seed_threshold_method,
                            min_seed_area=min_seed_area,
                            min_distance_peaks=min_distance_peaks,
                            membrane_fusion_method=membrane_fusion_method,
                            membrane_weights=cyto_weights,
                            boundary_method=boundary_method,
                            boundary_sigma=boundary_sigma,
                            compactness=compactness,
                            min_cell_area=min_cell_area,
                            max_cell_area=max_cell_area,
                            tile_size=tile_size,
                            tile_overlap=tile_overlap,
                            rng_seed=rng_seed
                        )

                    mask = run_task_with_event_pump(_watershed_batch_task, poll_interval_ms=80)
                    
                    # Explicitly release image stack memory immediately after segmentation
                    del img_stack
                    import gc
                    gc.collect()
                    
                    # Store the mask using mask manager
                    if isinstance(mask, np.ndarray):
                        mask = mask.copy()
                    
                    # Update acquisition info cache for better mask file resolution
                    self.segmentation_masks.set_acq_info(acq_id, acq_info)
                    
                    # Store mask in memory (save to disk only if user requested)
                    self.mask_manager.set_mask(
                        acq_id, mask, 
                        save_to_disk=save_masks,
                        acq_info=acq_info,
                        masks_directory=masks_directory
                    )
                    
                    # Clear colors for this acquisition so they get regenerated
                    if acq_id in self.segmentation_colors:
                        del self.segmentation_colors[acq_id]
                    if acq_id in self.cluster_colors:
                        del self.cluster_colors[acq_id]
                    if acq_id in self.cluster_color_map:
                        del self.cluster_color_map[acq_id]
                    
                    n_cells = int(len(np.unique(mask)) - 1)
                    successful_acq_ids.append(acq_id)
                    cell_counts_by_acquisition[acq_id] = n_cells
                    successful_segmentations += 1
                    
                    # Update progress
                    progress_dlg.update_progress(
                        successful_segmentations,
                        f"Completed {acq_name}",
                        f"Segmented {acq_name} ({n_cells} cells, {successful_segmentations}/{total_acquisitions} completed)"
                    )
                    
                except Exception as e:
                    print(f"Error segmenting acquisition {acq_name} ({acq_id}): {e}")
                    failed_acq_ids.append(acq_id)
                    # Continue with next acquisition
                    continue

            if successful_acq_ids:
                params = self._build_segmentation_log_params(
                    preprocessing_config,
                    denoise_source,
                    custom_denoise_settings,
                    acquisitions=successful_acq_ids,
                    show_overlay=show_overlay,
                    save_masks=save_masks,
                    masks_directory=masks_directory,
                    segment_scope="batch"
                )
                params.update({
                    "method": "watershed",
                    "nuclear_fusion_method": nuclear_fusion_method,
                    "seed_threshold_method": seed_threshold_method,
                    "min_seed_area": min_seed_area,
                    "min_distance_peaks": min_distance_peaks,
                    "membrane_fusion_method": membrane_fusion_method,
                    "membrane_weights": cyto_weights,
                    "boundary_method": boundary_method,
                    "boundary_sigma": boundary_sigma,
                    "compactness": compactness,
                    "min_cell_area": min_cell_area,
                    "max_cell_area": max_cell_area,
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "rng_seed": rng_seed,
                    "n_requested_acquisitions": total_acquisitions,
                    "n_successful_acquisitions": len(successful_acq_ids),
                    "successful_acquisitions": successful_acq_ids,
                    "failed_acquisitions": failed_acq_ids,
                    "cell_counts_by_acquisition": cell_counts_by_acquisition,
                })
                get_logger().log_segmentation(
                    method="watershed",
                    parameters=params,
                    acquisitions=successful_acq_ids,
                    output_path=masks_directory if save_masks else None,
                    notes=f"Batch-segmented {len(successful_acq_ids)} of {total_acquisitions} acquisitions",
                    source_file=self._get_source_file_summary_for_logging(successful_acq_ids)
                )
            
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
            # Release intermediate images immediately to free memory
            del nuclear_imgs
            import gc
            gc.collect()
        
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
                # Release intermediate images immediately to free memory
                del cyto_imgs
                gc.collect()
        
        # Clear loader cache after preprocessing to free memory
        if hasattr(loader, '_image_cache'):
            loader._image_cache.clear()
            gc.collect()
        
        return nuclear_img, cyto_img
    
    
    def _preprocess_acquisition_for_cellpose(self, acq_id: str, preprocessing_config: dict, progress_dlg, 
                                             denoise_source: str, custom_denoise_settings: dict) -> tuple:
        """Preprocess a specific acquisition for Cellpose segmentation."""
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
        
        # For Cellpose, at least one channel type must be selected
        if not nuclear_channels and not cyto_channels:
            raise ValueError("At least one channel (nuclear or cyto) must be selected for Cellpose")
        
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
                if denoise_source == "Use viewer settings":
                    # Get viewer denoise settings
                    viewer_settings = self._get_viewer_denoise_settings()
                    if viewer_settings and channel in viewer_settings:
                        try:
                            img = self._apply_custom_denoise(channel, img, viewer_settings)
                        except Exception:
                            pass
                elif denoise_source == "Custom" and custom_denoise_settings:
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
            # Release intermediate images immediately to free memory
            del nuclear_imgs
            import gc
            gc.collect()
        
        # Load and normalize cytoplasm channels
        cyto_img = None
        if cyto_channels:
            cyto_imgs = []
            for channel in cyto_channels:
                img = loader.get_image(original_acq_id, channel)
                if img is None:
                    continue
                
                # Apply denoising
                if denoise_source == "Use viewer settings":
                    # Get viewer denoise settings
                    viewer_settings = self._get_viewer_denoise_settings()
                    if viewer_settings and channel in viewer_settings:
                        try:
                            img = self._apply_custom_denoise(channel, img, viewer_settings)
                        except Exception:
                            pass
                elif denoise_source == "Custom" and custom_denoise_settings:
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
                # Release intermediate images immediately to free memory
                del cyto_imgs
                gc.collect()
        
        # Clear loader cache after preprocessing to free memory
        if hasattr(loader, '_image_cache'):
            loader._image_cache.clear()
            gc.collect()
        
        return nuclear_img, cyto_img
    
    
    def _load_batch_acquisitions(self, acquisitions, preprocessing_config: dict, progress_dlg, denoise_source: str = "none", custom_denoise_settings: dict = None, num_workers: int = None) -> dict:
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
        # Use num_workers if provided, otherwise default to cpu_count - 2
        if num_workers is not None:
            max_workers = max(1, min(num_workers, len(mp_args)))
        else:
            max_workers = max(1, min(mp.cpu_count() - 2, len(mp_args)))
        
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
            # Use uint16 format with compression to reduce file size
            tifffile.imwrite(filepath, masks.astype(np.uint16), compression='lzw')
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
        """Update the availability of Cluster mode in the overlay dropdown, clustering button, and spatial analysis button."""
        # Check for cluster data - first try current acquisition, then check all data
        current_acq_id = self.current_acq_id if hasattr(self, 'current_acq_id') else None
        has_cluster = self._has_cluster_data(current_acq_id)
        
        # If no cluster data for current acquisition, check if any cluster data exists at all
        if not has_cluster:
            has_cluster = self._has_cluster_data(None)  # Check all data
        
        # Enable/disable clustering button based on feature data availability (not cluster columns)
        # The button should work regardless of whether cluster columns exist
        if hasattr(self, 'clustering_btn'):
            has_feature_data = (self.feature_dataframe is not None and 
                              not self.feature_dataframe.empty)
            self.clustering_btn.setEnabled(has_feature_data)
        
        # Enable/disable spatial analysis button based on cluster column presence
        # Spatial analysis requires cluster columns to be available
        if hasattr(self, 'spatial_btn'):
            has_cluster_columns = False
            if self.feature_dataframe is not None and not self.feature_dataframe.empty:
                has_cluster_columns = any(col in self.feature_dataframe.columns 
                                       for col in ['cluster', 'cluster_id', 'cluster_phenotype'])
            self.spatial_btn.setEnabled(has_cluster_columns)
        
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

    def _get_cluster_annotation_map(self) -> Dict[Any, str]:
        """Return the best available cluster annotation map for display/legend labels."""
        annotation_map: Dict[Any, str] = {}
        if hasattr(self, 'clustering_dialog') and self.clustering_dialog is not None:
            annotation_map = normalize_cluster_annotation_map(
                getattr(self.clustering_dialog, 'cluster_annotation_map', {}) or {}
            )
        else:
            saved_state = getattr(self, '_saved_clustering_state', {}) or {}
            annotation_map = normalize_cluster_annotation_map(saved_state.get('cluster_annotation_map', {}) or {})

        return build_cluster_annotation_map(
            annotation_map,
            getattr(self, 'feature_dataframe', None),
            getattr(self, 'batch_corrected_dataframe', None),
            getattr(self, 'clustered_cells_dataframe', None),
        )

    def _get_cluster_display_name(self, cluster_id):
        """Return the current display name for a cluster id."""
        return get_cluster_display_name(cluster_id, annotation_map=self._get_cluster_annotation_map())

    def _ensure_cluster_overlay_palette(self, acq_id: str, unique_clusters) -> None:
        """Ensure cluster overlay colors and labels are initialized for an acquisition."""
        ordered_clusters = sort_cluster_values(
            unique_clusters,
            annotation_map=self._get_cluster_annotation_map(),
            canonical=True,
        )
        if not ordered_clusters:
            return

        if (
            acq_id not in self.cluster_colors
            or set(self.cluster_colors[acq_id].keys()) != set(ordered_clusters)
        ):
            import matplotlib.cm as cm

            cmap = cm.get_cmap('tab20' if len(ordered_clusters) <= 20 else 'tab20c')
            self.cluster_colors[acq_id] = {
                cluster: np.array(cmap(idx % cmap.N)[:3])
                for idx, cluster in enumerate(ordered_clusters)
            }

        self.cluster_color_map[acq_id] = {
            cluster: self._get_cluster_display_name(cluster)
            for cluster in ordered_clusters
        }

    def _get_cluster_assignments(self, acq_id: str) -> Optional[Dict[int, Union[int, str]]]:
        """Get cluster assignments for cells in the current acquisition by matching centroids.
        
        Matches mask cells to feature dataframe rows by finding the closest centroid coordinates.
        
        Returns:
            Dictionary mapping mask cell label (cell_id) to cluster_id/cluster_phenotype, or None if no cluster data available.
        """
        annotation_map = self._get_cluster_annotation_map()

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
                            canonical_cluster = canonicalize_cluster_id(cluster_val, annotation_map=annotation_map)
                            cluster_map[mask_label] = canonical_cluster if canonical_cluster is not None else 'Unassigned'
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
                        canonical_cluster = canonicalize_cluster_id(cluster_val, annotation_map=annotation_map)
                        cell_to_cluster[cell_id] = canonical_cluster if canonical_cluster is not None else 'Unassigned'
                
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
                unique_clusters = sort_cluster_values(
                    set(cluster_map.values()),
                    annotation_map=self._get_cluster_annotation_map(),
                    canonical=True,
                )
                self._ensure_cluster_overlay_palette(self.current_acq_id, unique_clusters)
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
        unique_clusters = sort_cluster_values(
            set(cluster_map.values()),
            annotation_map=self._get_cluster_annotation_map(),
            canonical=True,
        )
        self._ensure_cluster_overlay_palette(self.current_acq_id, unique_clusters)
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
        sorted_clusters = sort_cluster_values(
            cluster_color_map.keys(),
            annotation_map=self._get_cluster_annotation_map(),
            canonical=True,
        )
        
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
        torch = get_torch_module()
        if torch is None:
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
            cofactor = config.get('arcsinh_cofactor', 1.0)
            cache_key += f"_{cofactor}"
        elif norm_method == 'percentile_clip':
            p_low, p_high = config.get('percentile_params', (1.0, 99.0))
            cache_key += f"_{p_low}_{p_high}"
        # channelwise_minmax doesn't need extra cache key params
        
        # Apply normalization
        if norm_method == 'channelwise_minmax':
            return channelwise_minmax_normalize(img)
        elif norm_method == 'arcsinh':
            cofactor = config.get('arcsinh_cofactor', 1.0)
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
        spillover_matrix_path = dlg.get_spillover_file_path()
        
        # Get excluded channels
        excluded_channels = dlg.get_excluded_channels()
        
        # Store the normalization configuration for later use in clustering
        self.feature_extraction_config = {
            'normalization_config': normalization_config,
            'denoise_source': denoise_source,
            'custom_denoise_settings': custom_denoise_settings,
            'spillover_config': spillover_config,
            'spillover_matrix_path': spillover_matrix_path,
        }
        
        if not selected_acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions selected", "Please select at least one acquisition.")
            return
        
        if not any(selected_features.values()):
            QtWidgets.QMessageBox.warning(self, "No features selected", "Please select at least one feature to extract.")
            return
        
        # Perform feature extraction
        try:
            self._perform_feature_extraction(
                selected_acquisitions,
                selected_features,
                output_path,
                normalization_config,
                denoise_source,
                custom_denoise_settings,
                spillover_config,
                spillover_matrix_path,
                excluded_channels
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 
                "Feature Extraction Failed", 
                f"Feature extraction failed with error:\n{str(e)}"
            )
    

    def _perform_feature_extraction(self, selected_acquisitions, selected_features, output_path, normalization_config, denoise_source, custom_denoise_settings, spillover_config=None, spillover_matrix_path=None, excluded_channels=None):
        """Perform the actual feature extraction using multiprocessing.
        
        This now parallelizes both image loading and feature extraction for better performance.
        """
        # Clear image caches to prevent massive RAM usage during feature extraction
        self.image_cache.clear()
        
        # Clear loader image caches
        for acq_id in selected_acquisitions:
            loader = self._get_loader_for_acquisition(acq_id)
            if loader is not None and hasattr(loader, '_image_cache'):
                loader._image_cache.clear()
        
        # Create progress dialog
        progress_dlg = ProgressDialog("Feature Extraction", self)
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
                    
                    # Check if mask is on disk - if so, pass path instead of array for efficiency
                    mask_file_path = self.mask_manager.get_mask_file_path(acq_id, current_acq_info)
                    if mask_file_path:
                        # Mask is on disk - pass path instead of loading into memory
                        mask = None  # Will be loaded by worker
                        mask_path_for_worker = mask_file_path
                    else:
                        # Mask is in memory - pass array (will be written to temp file by worker)
                        mask = self.segmentation_masks[acq_id]
                        mask_path_for_worker = None
                    
                    # Prepare preprocessing parameters
                    arcsinh_enabled = normalization_config is not None and normalization_config.get('method') == 'arcsinh'
                    cofactor = normalization_config.get('cofactor', 1.0) if normalization_config else 1.0
                    
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
                        mask,  # Mask array (None if mask is on disk)
                        mask_path_for_worker,  # Mask file path (None if mask is in memory)
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
                close_progress_dialog(progress_dlg)
                QtWidgets.QMessageBox.warning(self, "No valid acquisitions", "No valid acquisitions found for feature extraction.")
                return

            total_acquisitions = len(mp_args)
            apply_arcsinh_at_end = normalization_config is not None and normalization_config.get('method') == 'arcsinh'
            finalization_steps = 2 + int(apply_arcsinh_at_end) + int(bool(output_path))
            progress_dlg.set_maximum(total_acquisitions + finalization_steps)
            
            # Group acquisitions by file path
            # IMPORTANT: Only group MCD files together. OME-TIFF files should be processed individually
            # (one file per worker) since each OME-TIFF file is typically a single acquisition.
            # MCD files need grouping to avoid file locking issues with readimc.
            from collections import defaultdict
            file_groups = defaultdict(list)
            ometiff_tasks = []  # OME-TIFF tasks processed individually
            
            for args in mp_args:
                file_path = args[6]  # file_path is at index 6
                loader_type = args[7]  # loader_type is at index 7
                
                # Only group MCD files. OME-TIFF files are processed individually
                if loader_type == "mcd":
                    file_groups[file_path].append(args)
                else:
                    # OME-TIFF: each file gets its own "group" (will be processed individually)
                    ometiff_tasks.append(args)
            
            # Use multiprocessing for parallel feature extraction within each file
            # Count MCD file groups + individual OME-TIFF files
            num_unique_files = len(file_groups) + len(ometiff_tasks)
            max_workers = max(1, min(mp.cpu_count() - 2, len(mp_args)))
            progress_dlg.update_progress(0, "Starting feature extraction", f"Processing {num_unique_files} files with up to {max_workers} workers")
            
            all_features = []
            import time
            start_time = time.time()
            worker_timeout = 300  # 5 minutes per worker
            
            try:
                # Use spawn context to avoid conflicts and hangs
                ctx = mp.get_context('spawn')
                
                # Process MCD files: each file sequentially, but all acquisitions from each file in parallel
                file_num = 0
                for file_path, file_args_list in file_groups.items():
                    file_num += 1
                    file_basename = os.path.basename(file_path)
                    print(f"[main] Processing MCD file {file_num}/{num_unique_files}: {file_basename} ({len(file_args_list)} acquisitions)")
                    
                    with ctx.Pool(processes=min(max_workers, len(file_args_list))) as pool:
                        # Submit all acquisitions from this file
                        futures = []
                        for args in file_args_list:
                            (acq_id, mask, mask_path, selected_features, acq_info_dict, acq_label, 
                             file_path_arg, loader_type, arcsinh_enabled, cofactor, denoise_source, 
                             custom_denoise_settings, spillover_config, source_file, excluded_channels) = args
                            future = pool.apply_async(
                                _extract_features_worker,
                                (acq_id, mask, mask_path, selected_features, acq_info_dict, acq_label, 
                                 file_path_arg, loader_type, arcsinh_enabled, cofactor, denoise_source, 
                                 custom_denoise_settings, spillover_config, source_file, excluded_channels)
                            )
                            futures.append((acq_id, future))
                        
                        # Collect results as they complete
                        completed_for_file = 0
                        for acq_idx, (acq_id, future) in enumerate(futures, 1):
                            if progress_dlg.is_cancelled():
                                pool.terminate()
                                pool.join()
                                break
                            
                            try:
                                result = future.get(timeout=worker_timeout)
                                if result is not None and not result.empty:
                                    all_features.append(result)
                                    completed_for_file += 1
                            except mp.TimeoutError:
                                print(f"[main] [ERROR] Feature extraction timed out for {acq_id} from {file_basename} after {worker_timeout}s")
                            except Exception as e:
                                print(f"[main] [ERROR] Feature extraction failed for {acq_id} from {file_basename}: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Update progress: count acquisitions processed (whether successful or not)
                            # Calculate how many acquisitions we've processed so far across all files
                            acquisitions_processed = sum(len(file_groups[f]) for f in list(file_groups.keys())[:file_num-1]) + acq_idx
                            progress_dlg.update_progress(
                                acquisitions_processed,
                                f"File {file_num}/{num_unique_files}: {completed_for_file}/{len(file_args_list)} acquisitions",
                                f"Processed {acquisitions_processed}/{total_acquisitions} acquisitions ({len(all_features)} successful)"
                            )
                            QtWidgets.QApplication.processEvents()
                    
                    print(f"[main] Completed MCD file {file_basename}: {completed_for_file}/{len(file_args_list)} acquisitions")
                
                # Process OME-TIFF files: each file individually (one file = one worker)
                # OME-TIFF files don't need grouping since each file is typically a single acquisition
                # and there are no file locking issues with OME-TIFF loaders
                if ometiff_tasks:
                    print(f"[main] Processing {len(ometiff_tasks)} OME-TIFF files individually")
                    with ctx.Pool(processes=min(max_workers, len(ometiff_tasks))) as pool:
                        futures = []
                        for args in ometiff_tasks:
                            (acq_id, mask, mask_path, selected_features, acq_info_dict, acq_label, 
                             file_path_arg, loader_type, arcsinh_enabled, cofactor, denoise_source, 
                             custom_denoise_settings, spillover_config, source_file, excluded_channels) = args
                            future = pool.apply_async(
                                _extract_features_worker,
                                (acq_id, mask, mask_path, selected_features, acq_info_dict, acq_label, 
                                 file_path_arg, loader_type, arcsinh_enabled, cofactor, denoise_source, 
                                 custom_denoise_settings, spillover_config, source_file, excluded_channels)
                            )
                            futures.append((acq_id, future))
                        
                        # Collect results as they complete
                        completed_ometiff = 0
                        mcd_acquisitions_processed = sum(len(args_list) for args_list in file_groups.values())
                        
                        for acq_idx, (acq_id, future) in enumerate(futures, 1):
                            if progress_dlg.is_cancelled():
                                pool.terminate()
                                pool.join()
                                break
                            
                            try:
                                result = future.get(timeout=worker_timeout)
                                if result is not None and not result.empty:
                                    all_features.append(result)
                                    completed_ometiff += 1
                            except mp.TimeoutError:
                                print(f"[main] [ERROR] Feature extraction timed out for OME-TIFF acquisition {acq_id} after {worker_timeout}s")
                            except Exception as e:
                                print(f"[main] [ERROR] Feature extraction failed for OME-TIFF acquisition {acq_id}: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Update progress
                            acquisitions_processed = mcd_acquisitions_processed + acq_idx
                            progress_dlg.update_progress(
                                acquisitions_processed,
                                f"Processing OME-TIFF files...",
                                f"Processed {acquisitions_processed}/{total_acquisitions} acquisitions ({len(all_features)} successful)"
                            )
                            QtWidgets.QApplication.processEvents()
                        
                        print(f"[main] Completed {completed_ometiff}/{len(ometiff_tasks)} OME-TIFF files")
                        
            except Exception as mp_error:
                print(f"Multiprocessing failed, falling back to sequential processing: {mp_error}")
                import traceback
                traceback.print_exc()
                progress_dlg.update_progress(0, "Multiprocessing failed, using sequential processing", "Processing acquisitions one by one")
                
                # Fallback to sequential processing
                for i, (acq_id, mask, mask_path, selected_features, acq_info_dict, acq_label, file_path, loader_type, arcsinh_enabled, cofactor, denoise_source, custom_denoise_settings, spillover_config, source_file, excluded_channels) in enumerate(mp_args):
                    if progress_dlg.is_cancelled():
                        break
                    
                    try:
                        result = _extract_features_worker(
                            acq_id, mask, mask_path, selected_features, acq_info_dict, acq_label, file_path, loader_type, arcsinh_enabled, cofactor, denoise_source, custom_denoise_settings, spillover_config, source_file, excluded_channels
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
                close_progress_dialog(progress_dlg)
                QtWidgets.QMessageBox.warning(self, "No features extracted", "No features could be extracted from the selected acquisitions.")
                return

            progress_value = total_acquisitions
            progress_dlg.set_cancel_enabled(False)
            progress_value += 1
            progress_dlg.update_progress(
                progress_value,
                "Finalizing feature table",
                (
                    f"All acquisitions processed. Combining results from {len(all_features)} acquisitions. "
                    "This may take a while for large datasets."
                ),
            )

            # Combine all features
            combined_features = pd.concat(all_features, ignore_index=True)
            # Enforce excluded-channel schema after concat so excluded channels
            # never appear as all-NaN columns in the final feature table.
            combined_features = drop_excluded_channel_feature_columns(combined_features, excluded_channels)
            
            # Apply arcsinh transformation at the end if enabled (more efficient - single pass on all data)
            if apply_arcsinh_at_end:
                progress_value += 1
                progress_dlg.update_progress(
                    progress_value,
                    "Applying arcsinh normalization",
                    f"Normalizing intensity features for {len(combined_features)} cells...",
                )
                cofactor = normalization_config.get('cofactor', 1.0)
                
                # Find all intensity feature columns (exclude frac_pos as it's a proportion)
                intensity_cols = [col for col in combined_features.columns 
                                 if any(col.endswith(f"_{ft}") for ft in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated'])
                                 and not col.endswith('_frac_pos')]
                
                if intensity_cols:
                    # Apply arcsinh to all intensity features at once
                    combined_features[intensity_cols] = arcsinh_normalize(combined_features[intensity_cols].values, cofactor=cofactor)
            
            # Store in memory
            self.feature_dataframe = combined_features
            
            # Update cluster mode availability after feature extraction
            self._update_cluster_mode_availability()
            
            # Log feature extraction operation
            logger = get_logger()
            features_extracted = [k for k, v in selected_features.items() if v]
            morphology_feature_names = {
                'area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
                'solidity', 'extent', 'circularity', 'major_axis_len_um',
                'minor_axis_len_um', 'aspect_ratio', 'bbox_area_um2',
                'touches_border', 'touches_edge', 'holes_count',
                'centroid_x', 'centroid_y'
            }
            selected_morphology_features = [name for name in features_extracted if name in morphology_feature_names]
            selected_intensity_features = [name for name in features_extracted if name not in morphology_feature_names]
            feature_categories = []
            if selected_morphology_features:
                feature_categories.append("morphology")
            if selected_intensity_features:
                feature_categories.append("intensity")
            denoise_settings = self._get_relevant_denoise_settings(
                denoise_source,
                custom_denoise_settings=custom_denoise_settings
            )
            params = {
                "normalization_method": normalization_config.get('method') if normalization_config else "none",
                "arcsinh_cofactor": normalization_config.get('cofactor') if normalization_config and normalization_config.get('method') == 'arcsinh' else None,
                "denoise_source": denoise_source,
                "denoise_used": denoise_settings is not None,
                "feature_categories": feature_categories,
                "selected_morphology_features": selected_morphology_features,
                "selected_intensity_features": selected_intensity_features,
                "excluded_channels": sorted(excluded_channels or []),
                "n_excluded_channels": len(excluded_channels or []),
                "spillover_correction": spillover_config is not None,
                "spillover_method": spillover_config.get('method') if spillover_config else None,
                "spillover_matrix_path": spillover_matrix_path if spillover_config else None,
                "spillover_matrix_size": list(spillover_config['matrix'].shape) if spillover_config and 'matrix' in spillover_config and hasattr(spillover_config['matrix'], 'shape') else None,
                "n_selected_acquisitions": len(selected_acquisitions),
                "source_files": self._get_source_files_for_logging(selected_acquisitions),
            }
            if denoise_settings:
                params["denoise_settings"] = denoise_settings
            
            source_file = self._get_source_file_summary_for_logging(selected_acquisitions)
            
            logger.log_feature_extraction(
                parameters=params,
                acquisitions=selected_acquisitions,
                features_extracted=features_extracted,
                output_path=output_path,
                notes=f"Extracted features from {len(combined_features)} cells across {len(selected_acquisitions)} acquisitions",
                source_file=source_file
            )
            
            if output_path:
                progress_value += 1
                progress_dlg.update_progress(
                    progress_value,
                    "Saving feature table",
                    f"Writing {os.path.basename(output_path)}...",
                )
                total_rows = len(combined_features)
                chunk_size = 50000

                if total_rows == 0:
                    combined_features.to_csv(output_path, index=False)
                else:
                    for start_idx in range(0, total_rows, chunk_size):
                        end_idx = min(start_idx + chunk_size, total_rows)
                        chunk = combined_features.iloc[start_idx:end_idx]
                        chunk.to_csv(
                            output_path,
                            mode='w' if start_idx == 0 else 'a',
                            header=(start_idx == 0),
                            index=False
                        )
                        progress_dlg.update_progress(
                            progress_value,
                            "Saving feature table",
                            (
                                f"Writing {os.path.basename(output_path)}...\n"
                                f"{end_idx:,}/{total_rows:,} rows written"
                            ),
                        )
                        QtWidgets.QApplication.processEvents()

            progress_value += 1
            if output_path:
                progress_dlg.update_progress(
                    progress_value,
                    "Feature extraction complete",
                    f"Features saved to: {output_path}\nTotal cells: {len(combined_features)}"
                )
            else:
                progress_dlg.update_progress(
                    progress_value,
                    "Feature extraction complete",
                    f"Features stored in memory\nTotal cells: {len(combined_features)}"
                )

            close_progress_dialog(progress_dlg)
            
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
        """Load previously saved segmentation masks from a directory for all ROIs (using dynamic loading)."""
        if not self.acquisitions:
            QtWidgets.QMessageBox.warning(self, "No acquisitions", "No acquisitions available. Please load a file first.")
            return
        
        # Ask user to select directory containing masks (use saved preference if available)
        saved_dir = get_masks_directory_preference()
        masks_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, 
            "Select Directory Containing Segmentation Masks",
            saved_dir or "",  # Start from saved directory if available
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        
        if not masks_dir:
            return
        
        # Save masks directory to preferences
        save_masks_directory_preference(masks_dir)
        
        # Clear all existing masks from memory before loading new ones
        # This ensures old masks are removed when loading a different set
        self.mask_manager.clear_all_masks()
        
        # Set the new masks directory (this will also clear masks if directory changed)
        self.mask_manager.set_masks_directory(masks_dir)
        
        # Masks are always stored in memory (no force disk storage)
        
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
                # Store mask file path in mask manager (dynamic loading)
                self.mask_manager._mask_file_paths[acq_info.id] = mask_file
                # Update acquisition info cache for better mask file resolution
                self.segmentation_masks.set_acq_info(acq_info.id, acq_info)
                
                # For small datasets, load mask into memory; for large datasets, keep on disk
                if len(self.acquisitions) <= 50:
                    # Load mask into memory for small datasets
                    if _HAVE_TIFFFILE:
                        mask = tifffile.imread(mask_file)
                    else:
                        # Fallback to PIL if tifffile not available
                        from PIL import Image
                        mask = np.array(Image.open(mask_file))
                    self.mask_manager.set_mask(acq_info.id, mask, save_to_disk=False)
                    cell_count = len(np.unique(mask)) - 1  # Subtract 1 for background
                else:
                    # For large datasets, just register the file path (load on demand)
                    # Try to get cell count by loading mask temporarily
                    try:
                        if _HAVE_TIFFFILE:
                            mask = tifffile.imread(mask_file)
                        else:
                            from PIL import Image
                            mask = np.array(Image.open(mask_file))
                        cell_count = len(np.unique(mask)) - 1
                        # Don't keep in memory for large datasets
                        del mask
                    except Exception:
                        cell_count = 0
                
                # Clear colors for this acquisition so they get regenerated
                if acq_info.id in self.segmentation_colors:
                    del self.segmentation_colors[acq_info.id]
                if acq_info.id in self.cluster_colors:
                    del self.cluster_colors[acq_info.id]
                if acq_info.id in self.cluster_color_map:
                    del self.cluster_color_map[acq_info.id]
                
                loaded_count += 1
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
        reply = QtWidgets.QMessageBox.question(
            self,
            "Save Analysis Steps",
            "Do you want to save an analysis steps text file before closing OpenIMC?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        )
        if reply == QtWidgets.QMessageBox.Yes:
            export_result = self._prompt_analysis_steps_export(
                dialog_title="Save Analysis Steps Before Closing",
                show_success_message=False,
                show_failure_message=True
            )
            if export_result is not True:
                event.ignore()
                return

        loaders_to_close = []
        if self.loader is not None:
            loaders_to_close.append(self.loader)
        for loader in self.mcd_loaders.values():
            if loader not in loaders_to_close:
                loaders_to_close.append(loader)
        for loader in loaders_to_close:
            try:
                loader.close()
            except Exception:
                pass
        event.accept()

    def _has_batch_corrected_feature_source(self) -> bool:
        """Return whether a batch-corrected feature table is available in memory."""
        return isinstance(self.batch_corrected_dataframe, pd.DataFrame) and not self.batch_corrected_dataframe.empty

    def _normalize_analysis_feature_set_preference(self, feature_set) -> str:
        """Normalize a feature-set label or key to the internal preference key."""
        if feature_set in ("batch_corrected", "Batch-Corrected Features"):
            return "batch_corrected"
        return "original"

    def _get_default_analysis_feature_set_preference(self) -> str:
        """Return the default feature-set preference for the current session."""
        return "batch_corrected" if self._has_batch_corrected_feature_source() else "original"

    def _get_effective_analysis_feature_set_preference(self, feature_set=None) -> str:
        """Return the requested feature-set preference with availability fallback applied."""
        if feature_set is None and getattr(self, 'analysis_feature_set_preference', None) is None:
            return self._get_default_analysis_feature_set_preference()

        normalized = self._normalize_analysis_feature_set_preference(
            feature_set if feature_set is not None else self.analysis_feature_set_preference
        )
        if normalized == "batch_corrected" and not self._has_batch_corrected_feature_source():
            return "original"
        return normalized

    def _apply_analysis_feature_set_to_dialog(self, dialog):
        """Refresh a dialog's feature sources and apply the shared selection."""
        if dialog is None:
            return

        try:
            if hasattr(dialog, 'refresh_dataframe'):
                dialog.refresh_dataframe()
            if hasattr(dialog, 'apply_feature_set_preference'):
                dialog.apply_feature_set_preference(self._get_effective_analysis_feature_set_preference())
        except (RuntimeError, AttributeError):
            pass

    def _sync_analysis_feature_source_dialogs(self, source_dialog=None):
        """Push the shared feature-set selection to any open analysis dialogs."""
        for attr in ('clustering_dialog', 'simple_spatial_dialog', 'advanced_spatial_dialog'):
            dialog = getattr(self, attr, None)
            if dialog is None or dialog is source_dialog:
                continue
            self._apply_analysis_feature_set_to_dialog(dialog)

    def _set_analysis_feature_set_preference(self, feature_set, source_dialog=None, sync_dialogs=True):
        """Update the shared analysis feature-set preference and propagate it."""
        self.analysis_feature_set_preference = self._get_effective_analysis_feature_set_preference(feature_set)
        if not sync_dialogs or self._analysis_feature_sync_in_progress:
            return

        self._analysis_feature_sync_in_progress = True
        try:
            self._sync_analysis_feature_source_dialogs(source_dialog=source_dialog)
        finally:
            self._analysis_feature_sync_in_progress = False

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
                self._apply_analysis_feature_set_to_dialog(self.clustering_dialog)
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
            clustered_cells_dataframe=getattr(self, 'clustered_cells_dataframe', None),
            parent=self
        )
        # Make dialog non-modal so it can be closed and reopened without losing state
        self.clustering_dialog.setModal(False)
        # Prevent dialog from being deleted when closed, so we can reopen it
        self.clustering_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Restore saved clustering state if available
        if hasattr(self, '_saved_clustering_state') and self._saved_clustering_state:
            self._restore_clustering_state(self.clustering_dialog, self._saved_clustering_state)

        self._apply_analysis_feature_set_to_dialog(self.clustering_dialog)
        
        # Ensure llm_phenotype_cache is always initialized as a dict
        if not hasattr(self.clustering_dialog, 'llm_phenotype_cache') or not isinstance(self.clustering_dialog.llm_phenotype_cache, dict):
            self.clustering_dialog.llm_phenotype_cache = {}
        
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
                self._apply_analysis_feature_set_to_dialog(self.simple_spatial_dialog)
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
            clustered_cells_dataframe=getattr(self, 'clustered_cells_dataframe', None),
            parent=self
        )
        self.simple_spatial_dialog.setModal(False)
        self.simple_spatial_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Restore saved spatial state if available
        if hasattr(self, '_saved_spatial_state') and self._saved_spatial_state:
            self._restore_spatial_state(self.simple_spatial_dialog, self._saved_spatial_state)

        self._apply_analysis_feature_set_to_dialog(self.simple_spatial_dialog)
        
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
                self._apply_analysis_feature_set_to_dialog(self.advanced_spatial_dialog)
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
        try:
            self.advanced_spatial_dialog = AdvancedSpatialAnalysisDialog(
                self.feature_dataframe, 
                batch_corrected_dataframe=self.batch_corrected_dataframe,
                clustered_cells_dataframe=getattr(self, 'clustered_cells_dataframe', None),
                parent=self
            )
        except RuntimeError as e:
            if "squidpy" in str(e).lower():
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Squidpy Not Available",
                    "Advanced Spatial Analysis requires squidpy, which is not installed.\n\n"
                    "Install with: pip install squidpy anndata\n\n"
                    "Would you like to open Simple Spatial Analysis instead?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    self._open_simple_spatial_dialog()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open Advanced Spatial Analysis: {str(e)}")
            return
        self.advanced_spatial_dialog.setModal(False)
        self.advanced_spatial_dialog.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Restore saved spatial state if available
        if hasattr(self, '_saved_spatial_state') and self._saved_spatial_state:
            self._restore_spatial_state(self.advanced_spatial_dialog, self._saved_spatial_state)

        self._apply_analysis_feature_set_to_dialog(self.advanced_spatial_dialog)
        
        self.advanced_spatial_dialog.show()
    
    def _get_qc_file_set_id(self) -> str:
        """Get a unique identifier for the current file set (for QC results caching).
        
        Returns a hash of all loaded file paths to uniquely identify the current data set.
        """
        import hashlib
        file_paths = sorted(set(self.acq_to_file.values()))
        if self.current_path and self.current_path not in file_paths:
            # For single file loaders (OME-TIFF or single MCD)
            file_paths.append(self.current_path)
        if not file_paths:
            return ""
        # Create a hash of all file paths
        file_set_str = "|".join(sorted(file_paths))
        return hashlib.md5(file_set_str.encode()).hexdigest()
    
    def _open_qc_dialog(self):
        """Open the QC analysis dialog."""
        # Check if data is loaded (either single file loader or multiple MCD files)
        has_data = (self.loader is not None) or (len(self.mcd_loaders) > 0) or (len(self.acquisitions) > 0)
        
        if not has_data:
            QtWidgets.QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load data first before running QC analysis."
            )
            return
        
        # Check if dialog already exists
        if hasattr(self, 'qc_dialog') and self.qc_dialog is not None:
            try:
                if self.qc_dialog.isVisible():
                    self.qc_dialog.raise_()
                    self.qc_dialog.activateWindow()
                else:
                    self.qc_dialog.show()
                    self.qc_dialog.raise_()
                    self.qc_dialog.activateWindow()
                return
            except (RuntimeError, AttributeError):
                self.qc_dialog = None
        
        dlg = QCAnalysisDialog(self)
        self.qc_dialog = dlg
        dlg.setModal(False)
        dlg.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Restore saved QC state if available (UI state is restored in dialog __init__)
        # Additional state restoration for results tables is done after dialog creation
        if hasattr(self, '_saved_qc_ui_state') and self._saved_qc_ui_state:
            self._restore_qc_ui_state(dlg, self._saved_qc_ui_state)
        dlg.show()
    
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
        
        # Check if dialog already exists
        if hasattr(self, 'pixel_correlation_dialog') and self.pixel_correlation_dialog is not None:
            try:
                if self.pixel_correlation_dialog.isVisible():
                    self.pixel_correlation_dialog.raise_()
                    self.pixel_correlation_dialog.activateWindow()
                else:
                    self.pixel_correlation_dialog.show()
                    self.pixel_correlation_dialog.raise_()
                    self.pixel_correlation_dialog.activateWindow()
                return
            except (RuntimeError, AttributeError):
                self.pixel_correlation_dialog = None
        
        dlg = PixelCorrelationDialog(self)
        self.pixel_correlation_dialog = dlg
        dlg.setModal(False)
        dlg.setAttribute(Qt.WA_DeleteOnClose, False)
        
        # Restore saved pixel correlation state if available
        if hasattr(self, '_saved_pixel_correlation_state') and self._saved_pixel_correlation_state:
            self._restore_pixel_correlation_state(dlg, self._saved_pixel_correlation_state)
        
        dlg.show()
    
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
        resolution = dlg.get_resolution()
        passes, contributions, psf_kernel, passes_arr, contribs_arr, kernel_dim, region_data_full, sigmoidal_params = dlg.get_deconv_passes_contributions()
        
        # Create and show progress dialog
        progress_dlg = ProgressDialog("High Resolution Deconvolution", self)
        progress_dlg.show()
        
        try:
            if acq_type == "single":
                success = self._deconvolve_single_acquisition(
                    output_dir, progress_dlg, x0, iterations, output_format, passes, contributions, psf_kernel,
                    passes_arr, contribs_arr, kernel_dim, region_data_full, None, resolution
                )
            else:
                success = self._deconvolve_whole_slide(
                    output_dir, progress_dlg, x0, iterations, output_format, passes, contributions, psf_kernel,
                    passes_arr, contribs_arr, kernel_dim, region_data_full, None, resolution
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
                                     x0: float, iterations: int, output_format: str,
                                     passes=None, contributions=None, kernel=None,
                                     passes_arr=None, contribs_arr=None, kernel_dim=None,
                                     region_data_full=None, I0=None, resolution=333) -> bool:
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
            # Clear image cache before deconvolution to ensure fresh data
            with self._cache_lock:
                self.image_cache.clear()
            
            # Deconvolve the acquisition using core function
            progress_dlg.update_progress(10, f"Deconvolving {acq_info.name}", "Loading image data...")
            
            # Create AcquisitionInfo with original ID for core function
            acq_info_for_core = AcquisitionInfo(
                id=original_acq_id,
                name=acq_info.name,
                well=acq_info.well,
                size=acq_info.size,
                channels=acq_info.channels,
                channel_metals=acq_info.channel_metals,
                channel_labels=acq_info.channel_labels,
                metadata=acq_info.metadata,
                source_file=acq_info.source_file
            )
            
            # Use core deconvolution function
            # Pass explicit paths since loader may not have file_path/directory attributes
            output_path = deconvolution(
                loader=loader,
                acquisition=acq_info_for_core,
                output_dir=output_dir,
                x0=x0,
                iterations=iterations,
                output_format=output_format,
                loader_path=data_path,
                source_file_path=source_file_path,
                unique_acq_id=self.current_acq_id,
                passes=passes,
                contributions=contributions,
                kernel=kernel,
                passes_arr=passes_arr,
                contribs_arr=contribs_arr,
                kernel_dim=kernel_dim,
                region_data_full=region_data_full,
                I0=None,  # Will be computed per channel from max intensity
                resolution=resolution
            )
            
            output_path = str(output_path)  # Convert Path to string for compatibility
            
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
                                x0: float, iterations: int, output_format: str,
                                passes=None, contributions=None, kernel=None,
                                passes_arr=None, contribs_arr=None, kernel_dim=None,
                                region_data_full=None, I0=None, resolution=333) -> bool:
        """Deconvolve all acquisitions. Uses multiprocessing for many ROIs."""
        if not self.acquisitions:
            raise ValueError("No acquisitions found")
        
        # Clear image cache before deconvolution to ensure fresh data
        with self._cache_lock:
            self.image_cache.clear()
        
        total_acqs = len(self.acquisitions)
        
        # Use multiprocessing if we have many acquisitions (>= 4)
        use_multiprocessing = total_acqs >= 4
        
        if use_multiprocessing:
            # Use multiprocessing for parallel deconvolution
            import multiprocessing as mp
            from functools import partial
            
            # Determine number of workers (use CPU count but cap at number of acquisitions)
            max_workers = min(mp.cpu_count(), total_acqs, 8)  # Cap at 8 to avoid too many processes
            
            progress_dlg.set_maximum(total_acqs * 100)
            progress_dlg.update_progress(0, "Preparing multiprocessing", f"Setting up {max_workers} workers for {total_acqs} acquisitions...")
            
            # Prepare arguments for each acquisition
            deconv_args = []
            for acq in self.acquisitions:
                # Get the loader and determine type
                loader = self._get_loader_for_acquisition(acq.id)
                if loader is None:
                    continue
                
                # Get original acquisition ID
                original_acq_id = self._get_original_acq_id(acq.id)
                
                # Determine data path and loader type
                data_path = None
                loader_type = None
                source_file_path = None
                
                if isinstance(loader, MCDLoader):
                    loader_type = "mcd"
                    if acq.id in self.acq_to_file:
                        data_path = self.acq_to_file[acq.id]
                    elif acq.source_file:
                        data_path = acq.source_file
                    elif self.current_path and os.path.isfile(self.current_path) and self.current_path.lower().endswith('.mcd'):
                        data_path = self.current_path
                    
                    if not data_path or not os.path.isfile(data_path):
                        continue
                    source_file_path = data_path
                    
                elif isinstance(loader, OMETIFFLoader):
                    loader_type = "ometiff"
                    if hasattr(loader, '_acq_map') and original_acq_id in loader._acq_map:
                        data_path = loader._acq_map[original_acq_id]
                    elif acq.source_file:
                        data_path = acq.source_file
                    elif self.current_path and os.path.isdir(self.current_path):
                        import glob
                        tiff_files = glob.glob(os.path.join(self.current_path, "*.ome.tif"))
                        tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.ome.tiff")))
                        tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.tif")))
                        tiff_files.extend(glob.glob(os.path.join(self.current_path, "*.tiff")))
                        for tiff_file in tiff_files:
                            if acq.name in os.path.basename(tiff_file):
                                data_path = tiff_file
                                break
                        if not data_path and tiff_files:
                            data_path = tiff_files[0]
                    
                    if not data_path or not os.path.isfile(data_path):
                        continue
                    
                    if self.current_path and os.path.isdir(self.current_path):
                        source_file_path = self.current_path
                    else:
                        source_file_path = os.path.dirname(data_path) if data_path else None
                
                if not data_path or loader_type is None:
                    continue
                
                # Convert AcquisitionInfo to dict for multiprocessing (not pickleable)
                acq_info_dict = {
                    'id': original_acq_id,
                    'name': acq.name,
                    'well': acq.well,
                    'size': acq.size,
                    'channels': acq.channels,
                    'channel_metals': acq.channel_metals,
                    'channel_labels': acq.channel_labels,
                    'metadata': acq.metadata,
                    'source_file': acq.source_file
                }
                
                deconv_args.append((
                    loader_type, acq_info_dict, data_path, source_file_path, acq.id,
                    output_dir, x0, iterations, output_format,
                    passes, contributions, kernel,
                    passes_arr, contribs_arr, kernel_dim, region_data_full, I0, resolution
                ))
            
            # Worker function for multiprocessing
            def deconv_worker(args):
                (loader_type, acq_info_dict, data_path, source_file_path, unique_acq_id,
                 output_dir, x0, iterations, output_format,
                 passes, contributions, kernel,
                 passes_arr, contribs_arr, kernel_dim, region_data_full, I0, resolution) = args
                
                try:
                    # Create loader in worker process (loaders are not pickleable)
                    if loader_type == "mcd":
                        from openimc.data.mcd_loader import MCDLoader
                        loader = MCDLoader()
                        loader.open(data_path)
                    elif loader_type == "ometiff":
                        from openimc.data.ometiff_loader import OMETIFFLoader
                        loader = OMETIFFLoader()
                        loader.open(source_file_path if os.path.isdir(source_file_path) else os.path.dirname(data_path))
                    else:
                        raise ValueError(f"Unknown loader type: {loader_type}")
                    
                    # Reconstruct AcquisitionInfo from dict
                    from openimc.data.mcd_loader import AcquisitionInfo
                    acq_info = AcquisitionInfo(**acq_info_dict)
                    
                    try:
                        from openimc.core import deconvolution
                        deconvolution(
                            loader=loader,
                            acquisition=acq_info,
                            output_dir=output_dir,
                            x0=x0,
                            iterations=iterations,
                            output_format=output_format,
                            loader_path=data_path,
                            source_file_path=source_file_path,
                            unique_acq_id=unique_acq_id,
                            passes=passes,
                            contributions=contributions,
                            kernel=kernel,
                            passes_arr=passes_arr,
                            contribs_arr=contribs_arr,
                            kernel_dim=kernel_dim,
                            region_data_full=region_data_full,
                            I0=I0,
                            resolution=resolution
                        )
                        return (unique_acq_id, True, None)
                    finally:
                        # Close loader in worker process
                        loader.close()
                except Exception as e:
                    import traceback
                    return (unique_acq_id, False, str(e))
            
            # Process in parallel
            processed = 0
            try:
                with mp.Pool(processes=max_workers) as pool:
                    results = []
                    for result in pool.imap(deconv_worker, deconv_args):
                        if progress_dlg.is_cancelled():
                            pool.terminate()
                            pool.join()
                            return False
                        
                        unique_acq_id, success, error = result
                        processed += 1
                        
                        if success:
                            progress_dlg.update_progress(
                                processed * 100,
                                f"Deconvolved {unique_acq_id}",
                                f"Completed {processed} of {total_acqs} acquisitions"
                            )
                        else:
                            progress_dlg.update_progress(
                                processed * 100,
                                f"Error processing {unique_acq_id}",
                                f"Error: {error}"
                            )
                        results.append((unique_acq_id, success, error))
                    
                    # Count successful deconvolutions
                    successful = sum(1 for _, success, _ in results if success)
                    progress_dlg.update_progress(
                        total_acqs * 100,
                        "Complete",
                        f"Successfully deconvolved {successful} of {total_acqs} acquisition(s)"
                    )
            except Exception as e:
                # Fall back to sequential processing if multiprocessing fails
                print(f"Multiprocessing failed, falling back to sequential: {e}")
                import traceback
                traceback.print_exc()
                use_multiprocessing = False
        
        if not use_multiprocessing:
            # Sequential processing (original code)
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
                
                # Create AcquisitionInfo with original ID for core function
                acq_info_for_core = AcquisitionInfo(
                    id=original_acq_id,
                    name=acq.name,
                    well=acq.well,
                    size=acq.size,
                    channels=acq.channels,
                    channel_metals=acq.channel_metals,
                    channel_labels=acq.channel_labels,
                    metadata=acq.metadata,
                    source_file=acq.source_file
                )
                
                # Use core deconvolution function
                deconvolution(
                    loader=loader,
                    acquisition=acq_info_for_core,
                    output_dir=output_dir,
                    x0=x0,
                    iterations=iterations,
                    output_format=output_format,
                    loader_path=data_path,
                    source_file_path=source_file_path,
                    unique_acq_id=acq.id,
                    passes=passes,
                    contributions=contributions,
                    kernel=kernel,
                    passes_arr=passes_arr,
                    contribs_arr=contribs_arr,
                    kernel_dim=kernel_dim,
                    region_data_full=region_data_full,
                    I0=I0,
                    resolution=resolution
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
            self.setWindowTitle(f"OpenIMC - {dirname} (Deconvolved OME-TIFF)")
            
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
            self._set_analysis_feature_set_preference(self.analysis_feature_set_preference)
            
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
        
        # If we have restored custom grouping and metadata files from a saved state,
        # populate the dialog with them
        if hasattr(self, 'batch_correction_custom_grouping') and self.batch_correction_custom_grouping:
            dlg.custom_grouping = copy.deepcopy(self.batch_correction_custom_grouping)
            # Update the custom grouping status label
            if hasattr(dlg, 'custom_grouping_status'):
                num_groups = len(set(self.batch_correction_custom_grouping.values()))
                dlg.custom_grouping_status.setText(
                    f"Restored: {len(self.batch_correction_custom_grouping)} acquisitions in {num_groups} groups"
                )
                dlg.custom_grouping_status.setVisible(True)
            if hasattr(dlg, 'custom_grouping_btn'):
                dlg.custom_grouping_btn.setVisible(True)
        
        if hasattr(self, 'batch_correction_metadata_files') and self.batch_correction_metadata_files:
            dlg.metadata_files = copy.deepcopy(self.batch_correction_metadata_files)
            # Update the metadata list UI if the dialog has this method
            if hasattr(dlg, '_update_metadata_list'):
                try:
                    dlg._update_metadata_list()
                except Exception as e:
                    print(f"Warning: Could not update metadata list in dialog: {e}")
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            had_batch_corrected = self._has_batch_corrected_feature_source()
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
                if not had_batch_corrected:
                    self._set_analysis_feature_set_preference("batch_corrected")
                else:
                    self._sync_analysis_feature_source_dialogs()
                
                # Store custom grouping and metadata files for state saving
                if hasattr(dlg, 'custom_grouping') and dlg.custom_grouping:
                    self.batch_correction_custom_grouping = copy.deepcopy(dlg.custom_grouping)
                else:
                    self.batch_correction_custom_grouping = None
                
                if hasattr(dlg, 'metadata_files') and dlg.metadata_files:
                    # Store metadata files (dictionary with file_path as key)
                    # Each entry contains 'filename_column' and 'dataframe'
                    self.batch_correction_metadata_files = copy.deepcopy(dlg.metadata_files)
                else:
                    self.batch_correction_metadata_files = None
                
                # Save to file if requested
                output_path = dlg.get_output_path()
                if output_path:
                    try:
                        self._save_dataframe_with_progress(
                            corrected_df,
                            output_path,
                            title="Saving Batch-Corrected Features",
                            status_prefix=(
                                "Batch correction is complete. Writing corrected features to disk.\n"
                                "Large files may take a few minutes."
                            )
                        )
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

    def _save_dataframe_with_progress(
        self,
        dataframe: pd.DataFrame,
        output_path: str,
        title: str = "Saving Data",
        status_prefix: str = "Saving data to disk...",
        chunk_size: int = 50000
    ) -> None:
        """Save a dataframe to CSV in chunks while keeping the UI responsive."""
        total_rows = len(dataframe)
        if total_rows == 0:
            dataframe.to_csv(output_path, index=False)
            return

        progress = QtWidgets.QProgressDialog(status_prefix, None, 0, total_rows, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            rows_written = 0
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = dataframe.iloc[start_idx:end_idx]
                chunk.to_csv(
                    output_path,
                    mode='w' if start_idx == 0 else 'a',
                    header=(start_idx == 0),
                    index=False
                )

                rows_written = end_idx
                progress.setLabelText(f"{status_prefix}\n{rows_written:,}/{total_rows:,} rows written")
                progress.setValue(rows_written)
                QtWidgets.QApplication.processEvents()
        finally:
            progress.close()
    
    def _save_state(self):
        """Save complete application state to a folder."""
        from PyQt5.QtWidgets import QFileDialog
        
        # Get save directory
        state_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save State",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not state_dir:
            return
        
        # Check if directory exists and is not empty
        state_path = Path(state_dir)
        if state_path.exists() and any(state_path.iterdir()):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Directory Not Empty",
                f"The selected directory is not empty.\n\n"
                f"Do you want to overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        
        # Collect state from main window
        state_data = self._collect_state()
        
        # Save state
        state_manager = StateManager()
        success = state_manager.save_state(state_path, state_data, overwrite=True)
        
        if success:
            QtWidgets.QMessageBox.information(
                self,
                "State Saved",
                f"Application state saved successfully to:\n{state_path}"
            )
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save application state to:\n{state_path}"
            )
    
    def _load_state(self):
        """Load complete application state from a folder."""
        from PyQt5.QtWidgets import QFileDialog
        
        # Get load directory
        state_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Load State",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not state_dir:
            return
        
        state_path = Path(state_dir)
        if not (state_path / "state.json").exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid State",
                f"The selected directory does not contain a valid state file."
            )
            return
        
        # Confirm loading (will replace current state)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Load State",
            f"Loading state will replace the current session.\n\n"
            f"Do you want to continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        # Load state
        state_manager = StateManager()
        loaded_state = state_manager.load_state(state_path)
        
        if loaded_state is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load application state from:\n{state_path}"
            )
            return
        
        # Restore state
        try:
            self._restore_state(loaded_state, state_path)
            QtWidgets.QMessageBox.information(
                self,
                "State Loaded",
                f"Application state loaded successfully from:\n{state_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Restore Error",
                f"Failed to restore application state:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def _collect_state(self) -> Dict[str, Any]:
        """Collect all state from the application for saving."""
        state = {
            "main_state": {},
            "images": {},
            "masks": {},
            "features": {},
            "analysis": {}
        }
        
        # Main window state
        main_state = {
            "current_path": self.current_path,
            "current_acq_id": self.current_acq_id,
            "ometiff_channel_format": self.ometiff_channel_format,
            "acq_to_file": self.acq_to_file,
            "unique_acq_to_original": self.unique_acq_to_original,
            "openimc_version": self._get_openimc_version(),
            "save_timestamp": datetime.now().isoformat(),
        }
        
        # Save acquisition info (serializable)
        if self.acquisitions:
            main_state["acquisitions"] = [
                {
                    "id": acq.id,
                    "name": acq.name,
                    "source_file": acq.source_file,
                    "well": acq.well if hasattr(acq, 'well') else None,
                    "channels": acq.channels if hasattr(acq, 'channels') else [],
                }
                for acq in self.acquisitions
            ]
        
        state["main_state"] = main_state
        
        # Collect masks from mask_manager
        if hasattr(self, 'mask_manager') and self.mask_manager:
            masks = {}
            acquisitions_info_dict = {}
            all_mask_ids = self.mask_manager.get_all_mask_ids()
            for acq_id in all_mask_ids:
                try:
                    mask = self.mask_manager.get_mask(acq_id)
                    if mask is not None:
                        masks[acq_id] = mask
                    
                    # Get acquisition info for proper naming
                    acq_info = self._get_acquisition_info(acq_id)
                    if acq_info:
                        acquisitions_info_dict[acq_id] = acq_info
                except Exception as e:
                    print(f"Warning: Could not get mask for {acq_id}: {e}")
                    continue
            state["masks"] = masks
            state["acquisitions_info"] = acquisitions_info_dict
        
        # Collect source files (.mcd files)
        source_files = []
        if self.current_path and os.path.exists(self.current_path):
            if self.current_path.endswith('.mcd'):
                source_files.append(self.current_path)
        
        # Also check acq_to_file for additional source files
        if hasattr(self, 'acq_to_file') and self.acq_to_file:
            for file_path in set(self.acq_to_file.values()):
                if file_path and os.path.exists(file_path) and file_path.endswith('.mcd'):
                    if file_path not in source_files:
                        source_files.append(file_path)
        
        state["source_files"] = source_files
        
        # Collect features
        features = {}
        if self.feature_dataframe is not None and not self.feature_dataframe.empty:
            features["original"] = self.feature_dataframe
        if self.batch_corrected_dataframe is not None and not self.batch_corrected_dataframe.empty:
            features["batch_corrected"] = self.batch_corrected_dataframe
        
        # Save clustered cells dataframe (only cells used in clustering, excluding filtered cells)
        # This is extracted from the clustering dialog if available
        if hasattr(self, 'clustering_dialog') and self.clustering_dialog is not None:
            dlg = self.clustering_dialog
            if hasattr(dlg, 'clustered_data') and dlg.clustered_data is not None:
                # Only include cells that were actually clustered (cluster != 0)
                clustered_cells = dlg.clustered_data[dlg.clustered_data['cluster'] != 0].copy()
                if not clustered_cells.empty:
                    features["clustered_cells"] = clustered_cells
                    print(f"[Save State] Saving clustered_cells dataframe with {len(clustered_cells)} cells")
        
        state["features"] = features
        
        # Collect analysis module states
        analysis_state = {}
        analysis_state["feature_set_preference"] = self._get_effective_analysis_feature_set_preference()
        
        # Feature extraction parameters
        if hasattr(self, 'feature_extraction_config') and self.feature_extraction_config:
            analysis_state["feature_extraction"] = {
                "config": self.feature_extraction_config,
                "normalization_config": self.feature_extraction_config.get('normalization_config'),
            }
        
        # QC Analysis state
        if hasattr(self, 'qc_results_cache') and self.qc_results_cache:
            analysis_state["qc_analysis"] = self.qc_results_cache
        
        # Also save QC dialog UI state if dialog exists
        if hasattr(self, 'qc_dialog') and self.qc_dialog is not None:
            qc_ui_state = {}
            dlg = self.qc_dialog
            if hasattr(dlg, 'mode_combo'):
                qc_ui_state["analysis_mode"] = dlg.mode_combo.currentText()
            if hasattr(dlg, 'acq_combo'):
                qc_ui_state["selected_acquisition"] = dlg.acq_combo.currentText()
            if hasattr(dlg, 'workers_spin'):
                qc_ui_state["num_workers"] = dlg.workers_spin.value()
            if hasattr(dlg, 'snr_threshold_spin'):
                qc_ui_state["snr_threshold"] = dlg.snr_threshold_spin.value()
            if hasattr(dlg, 'denoise_source_combo'):
                qc_ui_state["denoise_source"] = dlg.denoise_source_combo.currentText()
            if hasattr(dlg, 'custom_denoise_settings') and dlg.custom_denoise_settings:
                qc_ui_state["custom_denoise_settings"] = dlg.custom_denoise_settings
            if hasattr(dlg, 'get_cell_signal_method'):
                qc_ui_state["cell_signal_method"] = dlg.get_cell_signal_method()
            if hasattr(dlg, 'positive_threshold_sd_spin'):
                qc_ui_state["positive_threshold_sd"] = dlg.positive_threshold_sd_spin.value()
            if hasattr(dlg, 'upper_quantile_spin'):
                qc_ui_state["upper_quantile_percent"] = dlg.upper_quantile_spin.value()
            if qc_ui_state:
                if "qc_analysis" not in analysis_state:
                    analysis_state["qc_analysis"] = {}
                analysis_state["qc_analysis"]["ui_state"] = qc_ui_state
        
        # Clustering state
        if hasattr(self, 'clustering_dialog') and self.clustering_dialog is not None:
            clustering_state = self._collect_clustering_state()
            if clustering_state:
                analysis_state["clustering"] = clustering_state
        
        # Spatial Analysis state (check both simple and advanced)
        spatial_state = self._collect_spatial_state()
        if spatial_state:
            analysis_state["spatial"] = spatial_state
        
        # Batch Correction state
        batch_correction_state = self._collect_batch_correction_state()
        if batch_correction_state:
            analysis_state["batch_correction"] = batch_correction_state
        
        # Pixel Correlation state (if dialog exists and has results)
        if hasattr(self, 'pixel_correlation_dialog') and self.pixel_correlation_dialog is not None:
            pixel_correlation_state = self._collect_pixel_correlation_state()
            if pixel_correlation_state:
                analysis_state["pixel_correlation"] = pixel_correlation_state
        
        # Deconvolution state (if any)
        # Note: Deconvolution results are typically saved to files
        
        state["analysis"] = analysis_state
        
        return state
    
    def _collect_clustering_state(self) -> Optional[Dict[str, Any]]:
        """Collect clustering dialog state including all parameters."""
        if not hasattr(self, 'clustering_dialog') or self.clustering_dialog is None:
            return None
        
        dlg = self.clustering_dialog
        state = {}
        
        # Collect key state from clustering dialog
        if hasattr(dlg, 'cluster_labels') and dlg.cluster_labels is not None:
            state["cluster_labels"] = dlg.cluster_labels
        
        if hasattr(dlg, 'clustered_data') and dlg.clustered_data is not None:
            # Save clustered data as DataFrame reference (will be saved separately)
            state["has_clustered_data"] = True
        
        if hasattr(dlg, 'normalization_config') and dlg.normalization_config:
            state["normalization_config"] = dlg.normalization_config
        
        if hasattr(dlg, 'cluster_annotation_map') and dlg.cluster_annotation_map:
            state["cluster_annotation_map"] = dlg.cluster_annotation_map
        
        # Collect clustering method and parameters
        if hasattr(dlg, 'last_clustering_method'):
            state["clustering_method"] = dlg.last_clustering_method
        
        if hasattr(dlg, 'last_clustering_params'):
            state["clustering_parameters"] = dlg.last_clustering_params
        
        if hasattr(dlg, 'last_features_used'):
            state["features_used"] = dlg.last_features_used
        
        # Collect actual clustering method and dendrogram mode (what was actually used)
        if hasattr(dlg, 'actual_clustering_method') and dlg.actual_clustering_method:
            state["actual_clustering_method"] = dlg.actual_clustering_method
        
        if hasattr(dlg, 'actual_dendrogram_mode') and dlg.actual_dendrogram_mode:
            state["actual_dendrogram_mode"] = dlg.actual_dendrogram_mode
        
        # Collect filter settings (excluded cells, area filters, etc.)
        if hasattr(dlg, 'filter_settings') and dlg.filter_settings is not None:
            state["filter_settings"] = dlg.filter_settings
        
        # Collect scaling method (z-score/mad/none)
        if hasattr(dlg, 'clustering_scaling_method') and dlg.clustering_scaling_method:
            state["clustering_scaling_method"] = dlg.clustering_scaling_method
        
        # Collect custom names
        if hasattr(dlg, 'feature_label_map') and dlg.feature_label_map:
            state["feature_label_map"] = dlg.feature_label_map
        
        if hasattr(dlg, 'patient_annotation_map') and dlg.patient_annotation_map:
            state["patient_annotation_map"] = dlg.patient_annotation_map
        
        if hasattr(dlg, 'patient_cohort_map') and dlg.patient_cohort_map:
            state["patient_cohort_map"] = dlg.patient_cohort_map
        
        if hasattr(dlg, 'cluster_backend_names') and dlg.cluster_backend_names:
            state["cluster_backend_names"] = dlg.cluster_backend_names
        
        # Collect original cluster assignments (before merging)
        if hasattr(dlg, 'original_cluster_assignments') and dlg.original_cluster_assignments is not None:
            state["original_cluster_assignments"] = dlg.original_cluster_assignments
        
        # Collect plot settings
        if hasattr(dlg, 'cluster_map_orientation'):
            state["cluster_map_orientation"] = dlg.cluster_map_orientation
        
        if hasattr(dlg, 'cluster_map_dendrogram'):
            state["cluster_map_dendrogram"] = dlg.cluster_map_dendrogram
        
        if hasattr(dlg, 'cluster_map_zscore_method'):
            state["cluster_map_zscore_method"] = dlg.cluster_map_zscore_method
        
        if hasattr(dlg, 'cluster_map_cell_size'):
            state["cluster_map_cell_size"] = dlg.cluster_map_cell_size

        if hasattr(dlg, 'cluster_map_colorbar_width'):
            state["cluster_map_colorbar_width"] = dlg.cluster_map_colorbar_width

        if hasattr(dlg, 'cluster_map_colorbar_position'):
            state["cluster_map_colorbar_position"] = dlg.cluster_map_colorbar_position

        if hasattr(dlg, 'cluster_map_colorbar_orientation'):
            state["cluster_map_colorbar_orientation"] = dlg.cluster_map_colorbar_orientation
        
        # Collect heatmap scaling method
        if hasattr(dlg, 'heatmap_scaling_combo'):
            state["heatmap_scaling"] = dlg.heatmap_scaling_combo.currentText()
        
        # Collect patient annotation settings
        if hasattr(dlg, 'patient_annotation_column'):
            state["patient_annotation_column"] = dlg.patient_annotation_column
        
        if hasattr(dlg, 'patient_annotation_enabled'):
            state["patient_annotation_enabled"] = dlg.patient_annotation_enabled
        
        if hasattr(dlg, 'patient_legend_label'):
            state["patient_legend_label"] = dlg.patient_legend_label
        
        # Collect plot customization settings (font sizes, legend layout)
        if hasattr(dlg, 'x_tick_fontsize'):
            state["x_tick_fontsize"] = dlg.x_tick_fontsize
        
        if hasattr(dlg, 'y_tick_fontsize'):
            state["y_tick_fontsize"] = dlg.y_tick_fontsize
        
        if hasattr(dlg, 'legend_fontsize'):
            state["legend_fontsize"] = dlg.legend_fontsize
        
        if hasattr(dlg, 'legend_nrows'):
            state["legend_nrows"] = dlg.legend_nrows
        
        if hasattr(dlg, 'legend_ncols'):
            state["legend_ncols"] = dlg.legend_ncols
        
        # Collect LLM phenotype cache (always save it, even if empty, to preserve state)
        if hasattr(dlg, 'llm_phenotype_cache'):
            # Make a deep copy to ensure we capture the current state
            cache_to_save = copy.deepcopy(dlg.llm_phenotype_cache) if dlg.llm_phenotype_cache else {}
            state["llm_phenotype_cache"] = cache_to_save
        
        # Collect selected display features for heatmap (which features are shown in the heatmap)
        if hasattr(dlg, 'selected_display_features') and dlg.selected_display_features is not None:
            state["selected_display_features"] = list(dlg.selected_display_features)
        
        # Collect UMAP and t-SNE embeddings
        if hasattr(dlg, 'umap_embedding') and dlg.umap_embedding is not None:
            # Save as numpy array (will be serialized by state manager)
            state["umap_embedding"] = dlg.umap_embedding
            # Save the index that corresponds to the embedding
            if hasattr(dlg, 'umap_index') and dlg.umap_index is not None:
                state["umap_index"] = list(dlg.umap_index) if isinstance(dlg.umap_index, (list, pd.Index)) else dlg.umap_index.tolist()
        
        if hasattr(dlg, 'tsne_embedding') and dlg.tsne_embedding is not None:
            # Save as numpy array (will be serialized by state manager)
            state["tsne_embedding"] = dlg.tsne_embedding
            # Save the index that corresponds to the embedding
            if hasattr(dlg, 'tsne_index') and dlg.tsne_index is not None:
                state["tsne_index"] = list(dlg.tsne_index) if isinstance(dlg.tsne_index, (list, pd.Index)) else dlg.tsne_index.tolist()
        
        # Collect UI element states
        ui_state = {}
        
        # View selection
        if hasattr(dlg, 'view_combo'):
            ui_state["view"] = dlg.view_combo.currentText()
        
        # Feature set selection
        if hasattr(dlg, 'feature_set_combo'):
            ui_state["feature_set"] = dlg.feature_set_combo.currentText()
        
        # Clustering method and parameters
        if hasattr(dlg, 'clustering_type'):
            ui_state["clustering_type"] = dlg.clustering_type.currentText()
        
        if hasattr(dlg, 'n_clusters'):
            ui_state["n_clusters"] = dlg.n_clusters.value()
        
        if hasattr(dlg, 'seed_spinbox'):
            ui_state["seed"] = dlg.seed_spinbox.value()

        # PCA feature representation
        if hasattr(dlg, 'use_pca_checkbox'):
            ui_state["use_pca"] = dlg.use_pca_checkbox.isChecked()
        if hasattr(dlg, 'pca_mode_combo'):
            ui_state["pca_mode"] = dlg.pca_mode_combo.currentData() or dlg.pca_mode_combo.currentText()
        if hasattr(dlg, 'pca_variance_spinbox'):
            ui_state["pca_variance_percent"] = dlg.pca_variance_spinbox.value()
        if hasattr(dlg, 'pca_n_components_spinbox'):
            ui_state["pca_n_components"] = dlg.pca_n_components_spinbox.value()
        
        # Hierarchical clustering
        if hasattr(dlg, 'hierarchical_method'):
            ui_state["hierarchical_method"] = dlg.hierarchical_method.currentText()
        
        # Leiden clustering
        if hasattr(dlg, 'resolution_radio') and hasattr(dlg, 'modularity_radio'):
            ui_state["leiden_mode"] = "resolution" if dlg.resolution_radio.isChecked() else "modularity"
        
        if hasattr(dlg, 'n_neighbors_spinbox'):
            ui_state["n_neighbors"] = dlg.n_neighbors_spinbox.value()
        
        if hasattr(dlg, 'resolution_spinbox'):
            ui_state["resolution"] = dlg.resolution_spinbox.value()
        
        if hasattr(dlg, 'leiden_metric_combo'):
            ui_state["leiden_metric"] = dlg.leiden_metric_combo.currentText()
        
        if hasattr(dlg, 'jaccard_checkbox'):
            ui_state["jaccard_weighting"] = dlg.jaccard_checkbox.isChecked()
        
        # HDBSCAN clustering
        if hasattr(dlg, 'min_cluster_size_spinbox'):
            ui_state["min_cluster_size"] = dlg.min_cluster_size_spinbox.value()
        
        if hasattr(dlg, 'min_samples_spinbox'):
            ui_state["min_samples"] = dlg.min_samples_spinbox.value()
        
        if hasattr(dlg, 'cluster_selection_combo'):
            ui_state["cluster_selection_method"] = dlg.cluster_selection_combo.currentText()
        
        if hasattr(dlg, 'metric_combo'):
            ui_state["metric"] = dlg.metric_combo.currentText()
        
        # Dendrogram mode
        if hasattr(dlg, 'dendro_mode'):
            ui_state["dendro_mode"] = dlg.dendro_mode.currentText()
        
        # Visualization settings
        if hasattr(dlg, 'color_by_listwidget'):
            selected_items = [item.text() for item in dlg.color_by_listwidget.selectedItems()]
            ui_state["color_by"] = selected_items
        
        if hasattr(dlg, 'use_cohort_checkbox'):
            ui_state["use_cohort_coloring"] = dlg.use_cohort_checkbox.isChecked()
        
        if hasattr(dlg, 'point_size_spinbox'):
            ui_state["point_size"] = dlg.point_size_spinbox.value()
        
        if hasattr(dlg, 'point_alpha_spinbox'):
            ui_state["point_alpha"] = dlg.point_alpha_spinbox.value()
        
        if hasattr(dlg, 'show_legend_checkbox'):
            ui_state["show_legend"] = dlg.show_legend_checkbox.isChecked()
        
        if hasattr(dlg, 'group_by_combo'):
            ui_state["group_by"] = dlg.group_by_combo.currentText()
        
        if hasattr(dlg, 'stacked_bars_view_type_combo'):
            ui_state["stacked_bars_view_type"] = dlg.stacked_bars_view_type_combo.currentText()
        
        if hasattr(dlg, 'colormap_combo'):
            ui_state["colormap"] = dlg.colormap_combo.currentText()
        
        if hasattr(dlg, 'top_n_spinbox'):
            ui_state["top_n"] = dlg.top_n_spinbox.value()
        
        if hasattr(dlg, 'stacked_bars_filter_selection') and dlg.stacked_bars_filter_selection is not None:
            ui_state["stacked_bars_filter_selection"] = list(dlg.stacked_bars_filter_selection)
        
        if ui_state:
            state["ui_state"] = ui_state
        
        return state if state else None
    
    def _collect_spatial_state(self) -> Optional[Dict[str, Any]]:
        """Collect spatial analysis dialog state (both Simple and Advanced)."""
        # Check both simple and advanced spatial dialogs
        simple_dlg = getattr(self, 'simple_spatial_dialog', None)
        advanced_dlg = getattr(self, 'advanced_spatial_dialog', None)
        
        # Legacy check for spatial_dialog
        legacy_dlg = getattr(self, 'spatial_dialog', None)
        
        # Determine which dialog to use (prefer simple, then advanced, then legacy)
        dlg = simple_dlg or advanced_dlg or legacy_dlg
        
        if dlg is None:
            return None
        
        state = {}
        dialog_type = "simple" if simple_dlg else ("advanced" if advanced_dlg else "legacy")
        state["dialog_type"] = dialog_type
        
        # Common state for both Simple and Advanced
        if hasattr(dlg, 'cluster_annotation_map') and dlg.cluster_annotation_map:
            state["cluster_annotation_map"] = dlg.cluster_annotation_map
        if hasattr(dlg, 'feature_label_map') and dlg.feature_label_map:
            state["feature_label_map"] = dlg.feature_label_map
        
        # Simple Spatial Analysis specific state
        if dialog_type == "simple":
            # Graph data - save actual DataFrame
            if hasattr(dlg, 'edge_df') and dlg.edge_df is not None and not dlg.edge_df.empty:
                state["edge_df"] = dlg.edge_df
            
            # Graph metadata
            if hasattr(dlg, 'metadata') and dlg.metadata:
                state["metadata"] = dlg.metadata
            
            # Cell ID mappings (convert to serializable format)
            if hasattr(dlg, 'cell_id_to_gid') and dlg.cell_id_to_gid:
                state["cell_id_to_gid"] = {f"{k[0]}_{k[1]}": v for k, v in dlg.cell_id_to_gid.items()}
            
            if hasattr(dlg, 'gid_to_cell_id') and dlg.gid_to_cell_id:
                state["gid_to_cell_id"] = {str(k): list(v) for k, v in dlg.gid_to_cell_id.items()}
            
            # Analysis results DataFrames - save actual DataFrames
            if hasattr(dlg, 'cluster_summary_df') and dlg.cluster_summary_df is not None and not dlg.cluster_summary_df.empty:
                state["cluster_summary_df"] = dlg.cluster_summary_df
            
            if hasattr(dlg, 'enrichment_df') and dlg.enrichment_df is not None and not dlg.enrichment_df.empty:
                state["enrichment_df"] = dlg.enrichment_df
            
            if hasattr(dlg, 'distance_df') and dlg.distance_df is not None and not dlg.distance_df.empty:
                state["distance_df"] = dlg.distance_df
            
            # Analysis flags
            if hasattr(dlg, 'enrichment_analysis_run'):
                state["enrichment_analysis_run"] = dlg.enrichment_analysis_run
            
            if hasattr(dlg, 'distance_analysis_run'):
                state["distance_analysis_run"] = dlg.distance_analysis_run
            
            if hasattr(dlg, 'spatial_viz_run'):
                state["spatial_viz_run"] = dlg.spatial_viz_run
            
            if hasattr(dlg, 'community_analysis_run'):
                state["community_analysis_run"] = dlg.community_analysis_run
            
            # Source file filters
            if hasattr(dlg, 'selected_source_files') and dlg.selected_source_files:
                state["selected_source_files"] = list(dlg.selected_source_files)
            
            if hasattr(dlg, 'available_source_files') and dlg.available_source_files:
                state["available_source_files"] = list(dlg.available_source_files)
            
            # Random seed for reproducibility
            if hasattr(dlg, 'rng_seed'):
                state["rng_seed"] = dlg.rng_seed
            
            # Spatial visualization cache - save actual cache data
            if hasattr(dlg, 'spatial_viz_cache') and dlg.spatial_viz_cache:
                state["spatial_viz_cache"] = dlg.spatial_viz_cache
        
        # Advanced Spatial Analysis specific state
        elif dialog_type == "advanced":
            # Analysis status
            if hasattr(dlg, 'analysis_status') and dlg.analysis_status:
                state["analysis_status"] = dlg.analysis_status
            
            # Processed ROIs info (without full AnnData objects)
            if hasattr(dlg, 'processed_rois') and dlg.processed_rois:
                state["processed_rois"] = {
                    roi_id: {
                        "graph_built": info.get("graph_built", False),
                        "analyses": info.get("analyses", [])
                    }
                    for roi_id, info in dlg.processed_rois.items()
                }
            
            # Aggregated results - save actual results (may contain DataFrames or TempAnnData objects)
            if hasattr(dlg, 'aggregated_results') and dlg.aggregated_results:
                # Extract data from TempAnnData objects for serialization
                serialized_aggregated = {}
                for key, value in dlg.aggregated_results.items():
                    # Check if it's a TempAnnData-like object (has uns, obs, _cluster_key attributes)
                    if hasattr(value, 'uns') and hasattr(value, 'obs') and hasattr(value, '_cluster_key'):
                        # Serialize TempAnnData object
                        serialized_aggregated[key] = {
                            "__type__": "TempAnnData",
                            "uns": value.uns if hasattr(value, 'uns') else {},
                            "obs": value.obs if hasattr(value, 'obs') else None,
                            "_cluster_key": value._cluster_key if hasattr(value, '_cluster_key') else None,
                            "_significant_counts": value._significant_counts if hasattr(value, '_significant_counts') else None
                        }
                    else:
                        # Regular object - let state manager handle it
                        serialized_aggregated[key] = value
                state["aggregated_results"] = serialized_aggregated
            
            # Graph built flag
            if hasattr(dlg, 'spatial_graph_built'):
                state["spatial_graph_built"] = dlg.spatial_graph_built
            
            # Save graph construction parameters so we can rebuild the graphs
            graph_params = {}
            if hasattr(dlg, 'graph_method_combo'):
                graph_params["method"] = dlg.graph_method_combo.currentText()
            if hasattr(dlg, 'graph_k_spin'):
                graph_params["k"] = dlg.graph_k_spin.value()
            if hasattr(dlg, 'graph_radius_spin'):
                graph_params["radius"] = dlg.graph_radius_spin.value()
            if hasattr(dlg, 'seed_spinbox'):
                graph_params["seed"] = dlg.seed_spinbox.value()
            if graph_params:
                state["graph_construction_params"] = graph_params
            
            # Save which ROIs had graphs built
            if hasattr(dlg, 'anndata_cache') and dlg.anndata_cache:
                state["has_anndata_cache"] = True
                state["anndata_cache_rois"] = list(dlg.anndata_cache.keys())
        
        return state if state else None
    
    def _collect_batch_correction_state(self) -> Optional[Dict[str, Any]]:
        """Collect batch correction state including custom grouping and metadata files."""
        # Batch correction dialog is typically modal and closed after use,
        # but we can save the state from the last correction if available
        state = {}
        
        # Save the actual batch corrected dataframe
        if hasattr(self, 'batch_corrected_dataframe') and self.batch_corrected_dataframe is not None:
            state["batch_corrected_dataframe"] = self.batch_corrected_dataframe
        
        # Save custom grouping (maps acquisition_id -> group_name)
        if hasattr(self, 'batch_correction_custom_grouping') and self.batch_correction_custom_grouping:
            state["custom_grouping"] = self.batch_correction_custom_grouping
        
        # Save metadata files (dictionary with file_path as key)
        # Each entry contains 'filename_column' and 'dataframe'
        if hasattr(self, 'batch_correction_metadata_files') and self.batch_correction_metadata_files:
            state["metadata_files"] = self.batch_correction_metadata_files
        
        return state if state else None
    
    def _restore_batch_correction_state(self, state: Dict[str, Any]):
        """Restore batch correction state including custom grouping and metadata files."""
        try:
            # Restore batch corrected dataframe
            if "batch_corrected_dataframe" in state and state["batch_corrected_dataframe"] is not None:
                self.batch_corrected_dataframe = state["batch_corrected_dataframe"]
                print(f"Restored batch corrected dataframe with {len(self.batch_corrected_dataframe)} cells")
            
            # Restore custom grouping
            if "custom_grouping" in state and state["custom_grouping"]:
                self.batch_correction_custom_grouping = state["custom_grouping"]
                print(f"Restored custom grouping with {len(self.batch_correction_custom_grouping)} groups")
            
            # Restore metadata files
            if "metadata_files" in state and state["metadata_files"]:
                self.batch_correction_metadata_files = state["metadata_files"]
                print(f"Restored {len(self.batch_correction_metadata_files)} metadata file(s)")
        except Exception as e:
            print(f"Warning: Could not fully restore batch correction state: {e}")
            import traceback
            traceback.print_exc()
    
    def _collect_pixel_correlation_state(self) -> Optional[Dict[str, Any]]:
        """Collect pixel correlation dialog state."""
        if not hasattr(self, 'pixel_correlation_dialog') or self.pixel_correlation_dialog is None:
            return None
        
        dlg = self.pixel_correlation_dialog
        state = {}
        
        # Save actual correlation results DataFrames
        if hasattr(dlg, 'correlation_results') and dlg.correlation_results is not None and not dlg.correlation_results.empty:
            state["correlation_results"] = dlg.correlation_results
        
        if hasattr(dlg, 'aggregated_results') and dlg.aggregated_results is not None and not dlg.aggregated_results.empty:
            state["aggregated_results"] = dlg.aggregated_results
        
        # Save analysis settings
        if hasattr(dlg, 'analyze_within_masks'):
            state["analyze_within_masks"] = dlg.analyze_within_masks
        if hasattr(dlg, 'use_conditions_chk'):
            state["use_conditions"] = bool(dlg.use_conditions_chk.isChecked())
        
        # Save ROI items (list of tuples: acq_id, acq_name, file_path, loader_type)
        if hasattr(dlg, 'roi_items') and dlg.roi_items:
            state["roi_items"] = dlg.roi_items
        
        # Save condition widgets data if using conditions
        if hasattr(dlg, 'condition_widgets') and dlg.condition_widgets:
            condition_data = []
            for widget in dlg.condition_widgets:
                if hasattr(widget, 'get_condition_data'):
                    condition_name, roi_items = widget.get_condition_data()
                    condition_data.append({
                        "condition_name": condition_name,
                        "roi_items": roi_items
                    })
            if condition_data:
                state["condition_data"] = condition_data
        
        # Save UI state
        ui_state = {}
        # PixelCorrelationDialog uses `scope_combo` and `channel_combo`
        if hasattr(dlg, 'scope_combo'):
            ui_state["scope"] = dlg.scope_combo.currentText()
        if hasattr(dlg, 'channel_combo'):
            ui_state["channel_selection"] = dlg.channel_combo.currentText()
        # Persist explicitly selected channels if using "Select Channels..."
        if hasattr(dlg, '_get_selected_channels'):
            ui_state["selected_channels"] = dlg._get_selected_channels()
        if hasattr(dlg, 'roi_list'):
            # Save selected ROI items
            selected_rois = []
            for i in range(dlg.roi_list.count()):
                item = dlg.roi_list.item(i)
                if item.isSelected():
                    selected_rois.append(item.text())
            ui_state["selected_rois"] = selected_rois
        
        if ui_state:
            state["ui_state"] = ui_state
        
        return state if state else None
    
    def _restore_pixel_correlation_state(self, dialog, state: Dict[str, Any]):
        """Restore pixel correlation dialog state."""
        try:
            # Restore results first (so any UI refresh uses complete state)
            if "correlation_results" in state and state["correlation_results"] is not None and hasattr(dialog, 'correlation_results'):
                dialog.correlation_results = state["correlation_results"]
            if "aggregated_results" in state and state["aggregated_results"] is not None and hasattr(dialog, 'aggregated_results'):
                dialog.aggregated_results = state["aggregated_results"]
            
            # Restore ROI items
            if "roi_items" in state and state["roi_items"]:
                if hasattr(dialog, 'roi_items'):
                    dialog.roi_items = state["roi_items"]
                    # Update ROI list in UI
                    if hasattr(dialog, 'roi_list'):
                        dialog.roi_list.clear()
                        for acq_id, acq_name, file_path, loader_type in dialog.roi_items:
                            # Keep display friendly; exact format isn't critical as long as list reflects content.
                            item_text = f"{acq_name} ({os.path.basename(file_path)})"
                            item = QtWidgets.QListWidgetItem(item_text)
                            item.setData(Qt.UserRole, (acq_id, acq_name, file_path, loader_type))
                            dialog.roi_list.addItem(item)
            
            # Restore condition widgets data
            if "condition_data" in state and state["condition_data"]:
                if hasattr(dialog, 'condition_widgets'):
                    # Clear existing conditions
                    for widget in dialog.condition_widgets[:]:
                        widget.setParent(None)
                        dialog.condition_widgets.remove(widget)
                    # Restore conditions
                    for cond_data in state["condition_data"]:
                        widget = ConditionROIWidget(dialog)
                        widget.setParent(dialog.conditions_container)
                        # Set condition name (ConditionROIWidget uses `name_edit`)
                        if hasattr(widget, 'name_edit'):
                            widget.name_edit.setText(cond_data.get("condition_name", "") or "")
                        # Add ROI items to widget
                        if hasattr(widget, 'roi_items'):
                            widget.roi_items = cond_data.get("roi_items", [])
                            # Update widget's ROI list if it has one
                            if hasattr(widget, 'roi_list'):
                                widget.roi_list.clear()
                                for acq_id, acq_name, file_path, loader_type in widget.roi_items:
                                    item_text = f"{acq_name} ({os.path.basename(file_path)})"
                                    item = QtWidgets.QListWidgetItem(item_text)
                                    item.setData(Qt.UserRole, (acq_id, acq_name, file_path, loader_type))
                                    widget.roi_list.addItem(item)
                        dialog.condition_widgets.append(widget)
                        if hasattr(dialog, 'conditions_layout'):
                            dialog.conditions_layout.addWidget(widget)
            
            # Restore whether conditions mode was enabled
            use_conditions = bool(state.get("use_conditions")) or bool(state.get("condition_data"))
            if hasattr(dialog, 'use_conditions_chk'):
                dialog.use_conditions_chk.setChecked(use_conditions)
                # Ensure correct visibility immediately
                if hasattr(dialog, 'conditions_group'):
                    dialog.conditions_group.setVisible(use_conditions)
                if hasattr(dialog, 'roi_group'):
                    dialog.roi_group.setVisible(not use_conditions)
            
            # Restore analysis settings
            if "analyze_within_masks" in state:
                if hasattr(dialog, 'analyze_within_masks'):
                    dialog.analyze_within_masks = state["analyze_within_masks"]
                    # Update UI to reflect the setting (`scope_combo` exists)
                    if hasattr(dialog, 'scope_combo'):
                        index = 0 if state["analyze_within_masks"] else 1
                        dialog.scope_combo.setCurrentIndex(index)
            
            # Restore UI state
            if "ui_state" in state:
                ui_state = state["ui_state"]
                if "scope" in ui_state and hasattr(dialog, 'scope_combo'):
                    index = dialog.scope_combo.findText(ui_state["scope"])
                    if index >= 0:
                        dialog.scope_combo.setCurrentIndex(index)
                if "channel_selection" in ui_state and hasattr(dialog, 'channel_combo'):
                    index = dialog.channel_combo.findText(ui_state["channel_selection"])
                    if index >= 0:
                        dialog.channel_combo.setCurrentIndex(index)
                
                # Restore selected channels checks if in "Select Channels..." mode
                if ui_state.get("selected_channels") is not None and hasattr(dialog, 'channel_list'):
                    # Ensure the channel list is populated before applying checks
                    if hasattr(dialog, '_populate_channel_list'):
                        try:
                            dialog._populate_channel_list()
                        except Exception:
                            pass
                    selected = set(ui_state.get("selected_channels") or [])
                    for i in range(dialog.channel_list.count()):
                        item = dialog.channel_list.item(i)
                        # Only toggle checkable items
                        if item.flags() & Qt.ItemIsUserCheckable:
                            item.setCheckState(Qt.Checked if item.text() in selected else Qt.Unchecked)
                if "selected_rois" in ui_state and hasattr(dialog, 'roi_list'):
                    # Restore selected ROIs
                    for i in range(dialog.roi_list.count()):
                        item = dialog.roi_list.item(i)
                        if item.text() in ui_state["selected_rois"]:
                            item.setSelected(True)
            
            # Recompute scope availability based on restored ROIs/masks
            if hasattr(dialog, '_check_masks_available'):
                try:
                    dialog._check_masks_available()
                except Exception:
                    pass
            
            # Refresh results UI if any results exist
            if hasattr(dialog, '_update_results_display') and getattr(dialog, 'aggregated_results', None) is not None:
                try:
                    dialog._update_results_display(use_conditions)
                except Exception as e:
                    print(f"Warning: Could not update pixel correlation results display: {e}")
        except Exception as e:
            print(f"Warning: Could not fully restore pixel correlation state: {e}")
            import traceback
            traceback.print_exc()
    
    def _restore_qc_ui_state(self, dialog, ui_state: Dict[str, Any]):
        """Restore QC analysis dialog UI state."""
        try:
            # Restore analysis mode (pixel/cell)
            if "analysis_mode" in ui_state and hasattr(dialog, 'mode_combo'):
                index = dialog.mode_combo.findText(ui_state["analysis_mode"])
                if index >= 0:
                    dialog.mode_combo.setCurrentIndex(index)
            
            # Restore selected acquisition
            if "selected_acquisition" in ui_state and hasattr(dialog, 'acq_combo'):
                index = dialog.acq_combo.findText(ui_state["selected_acquisition"])
                if index >= 0:
                    dialog.acq_combo.setCurrentIndex(index)
            
            # Restore number of workers
            if "num_workers" in ui_state and hasattr(dialog, 'workers_spin'):
                dialog.workers_spin.setValue(ui_state["num_workers"])
            if "snr_threshold" in ui_state and hasattr(dialog, 'snr_threshold_spin'):
                dialog.snr_threshold_spin.setValue(float(ui_state["snr_threshold"]))

            if "denoise_source" in ui_state and hasattr(dialog, 'denoise_source_combo'):
                index = dialog.denoise_source_combo.findText(ui_state["denoise_source"])
                if index >= 0:
                    dialog.denoise_source_combo.setCurrentIndex(index)

            if "custom_denoise_settings" in ui_state and hasattr(dialog, 'custom_denoise_settings'):
                dialog.custom_denoise_settings = dict(ui_state["custom_denoise_settings"] or {})
                if hasattr(dialog, '_populate_denoise_channel_list'):
                    dialog._populate_denoise_channel_list()
                if hasattr(dialog, '_load_denoise_settings'):
                    dialog._load_denoise_settings()
            if "cell_signal_method" in ui_state and hasattr(dialog, 'cell_signal_method_combo'):
                index = dialog.cell_signal_method_combo.findData(ui_state["cell_signal_method"])
                if index >= 0:
                    dialog.cell_signal_method_combo.setCurrentIndex(index)
            if "positive_threshold_sd" in ui_state and hasattr(dialog, 'positive_threshold_sd_spin'):
                dialog.positive_threshold_sd_spin.setValue(float(ui_state["positive_threshold_sd"]))
            if "upper_quantile_percent" in ui_state and hasattr(dialog, 'upper_quantile_spin'):
                dialog.upper_quantile_spin.setValue(float(ui_state["upper_quantile_percent"]))
            if hasattr(dialog, '_update_cell_signal_controls'):
                dialog._update_cell_signal_controls()
            if hasattr(dialog, '_update_settings_summary'):
                dialog._update_settings_summary()
            
            # Refresh QC results display if results are cached
            if hasattr(dialog, 'qc_results_aggregated') and dialog.qc_results_aggregated is not None:
                if hasattr(dialog, '_update_summary_table'):
                    try:
                        dialog._update_summary_table()
                    except Exception:
                        pass
                if hasattr(dialog, '_update_plots'):
                    try:
                        dialog._update_plots()
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: Could not fully restore QC UI state: {e}")
            import traceback
            traceback.print_exc()
    
    def _restore_state(self, loaded_state: Dict[str, Any], state_path: Path):
        """Restore application state from loaded data."""
        # Restore main window state
        main_state = loaded_state.get("main_window", {})
        
        # Load .mcd files from images folder if they exist
        source_files = main_state.get("source_files", [])
        images_dir = state_path / "images"
        
        if source_files and images_dir.exists():
            # Load all .mcd files found in the images folder
            mcd_files = []
            for filename in source_files:
                mcd_path = images_dir / filename
                if mcd_path.exists() and mcd_path.suffix.lower() == '.mcd':
                    mcd_files.append(str(mcd_path))
            
            if mcd_files:
                # Close existing loaders first
                self._close_all_loaders()
                
                # Clear canvas completely before loading to ensure proper redraw
                self.canvas.fig.clear()
                self.canvas.draw()
                
                # Load all MCD files using the same logic as _load_multiple_mcd_files
                # Track all acquisitions and their source files
                all_acquisitions = []
                file_channel_sets = {}  # Maps file path to set of all channels in that file
                
                # Clear image cache to avoid collisions with old cache entries
                with self._cache_lock:
                    self.image_cache.clear()
                
                # Load each MCD file
                for mcd_file in mcd_files:
                    if not os.path.isfile(mcd_file):
                        print(f"Warning: Skipping invalid path: {mcd_file}")
                        continue
                    
                    if not mcd_file.lower().endswith('.mcd'):
                        print(f"Warning: Skipping non-MCD file: {mcd_file}")
                        continue
                    
                    try:
                        loader = MCDLoader()
                        loader.open(mcd_file)
                        self.mcd_loaders[mcd_file] = loader
                        
                        # Get acquisitions for this file
                        file_acqs = loader.list_acquisitions(source_file=mcd_file)
                        
                        # Create unique acquisition IDs by incorporating file identifier
                        # Use a hash of the file path to create a short unique identifier
                        import hashlib
                        file_hash = hashlib.md5(mcd_file.encode()).hexdigest()[:8]
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
                            self.acq_to_file[unique_acq_id] = mcd_file
                            self.unique_acq_to_original[unique_acq_id] = acq.id  # Store mapping from unique to original ID
                        file_channel_sets[mcd_file] = file_channels
                        
                    except Exception as e:
                        print(f"Warning: Could not load MCD file {mcd_file}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if not all_acquisitions:
                    QtWidgets.QMessageBox.critical(self, "No acquisitions", "No acquisitions could be loaded from the saved state files.")
                    return
                
                # Check for channel mismatches (compare channels per file)
                channel_sets_list = list(file_channel_sets.values())
                self._check_channel_mismatches(channel_sets_list, list(file_channel_sets.keys()))
                
                # Update state
                self.acquisitions = all_acquisitions
                self.current_path = mcd_files[0] if mcd_files else None
                
                # Set main loader to first file for backward compatibility
                if mcd_files and mcd_files[0] in self.mcd_loaders:
                    self.loader = self.mcd_loaders[mcd_files[0]]
                
                # Update window title
                if len(mcd_files) == 1:
                    stem = os.path.splitext(os.path.basename(mcd_files[0]))[0]
                    self.setWindowTitle(f"OpenIMC - {stem} (MCD)")
                else:
                    self.setWindowTitle(f"OpenIMC - {len(mcd_files)} MCD files")
                
                # Clear canvas completely before loading new files to ensure proper redraw
                self.canvas.fig.clear()
                self.canvas.draw()
                
                # Update acquisition combo box with file names (same as _load_multiple_mcd_files)
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
                    
                    # Display the selected images
                    self._view_selected()
        
        # Restore file paths and acquisition info
        if "current_path" in main_state and not self.current_path:
            # Fallback if files weren't loaded above
            self.current_path = main_state["current_path"]
        
        # Restore masks (do this after acquisitions are loaded so we can match them properly)
        # Use the same matching logic as _load_segmentation_masks for consistency
        masks = loaded_state.get("masks", {})
        if masks and hasattr(self, 'mask_manager') and self.acquisitions:
            # Get saved acquisition info from state to help with matching
            saved_acquisitions = main_state.get("acquisitions", [])
            saved_acq_map = {acq.get("id"): acq for acq in saved_acquisitions}
            
            for saved_acq_id, mask in masks.items():
                # Try to find matching acquisition using multiple strategies
                acq_info = None
                restored_acq_id = None
                
                # Get saved acquisition info if available
                saved_acq_data = saved_acq_map.get(saved_acq_id, {})
                saved_well = saved_acq_data.get("well")
                saved_name = saved_acq_data.get("name")
                saved_source_file = saved_acq_data.get("source_file")
                
                # Strategy 1: Try exact match with saved acquisition ID (works if IDs match exactly)
                acq_info = self._get_acquisition_info(saved_acq_id)
                if acq_info:
                    restored_acq_id = saved_acq_id
                else:
                    # Strategy 2: Match by source file + well/name (same as base app logic)
                    # This is the most reliable for multiple MCD files
                    for ai in self.acquisitions:
                        if not ai.source_file:
                            continue
                        
                        # Match by source file first (most reliable for multi-file)
                        source_matches = False
                        if saved_source_file:
                            # Compare source files (handle both full paths and basenames)
                            saved_source_basename = os.path.basename(saved_source_file)
                            current_source_basename = os.path.basename(ai.source_file)
                            if saved_source_basename == current_source_basename:
                                source_matches = True
                        else:
                            # If no saved source file, try to match by checking if saved_acq_id references this file
                            if saved_acq_id in self.acq_to_file:
                                if self.acq_to_file[saved_acq_id] == ai.source_file:
                                    source_matches = True
                        
                        if source_matches:
                            # Now match by well or name
                            if saved_well and ai.well and saved_well == ai.well:
                                acq_info = ai
                                restored_acq_id = ai.id
                                break
                            elif saved_name and ai.name and saved_name == ai.name:
                                acq_info = ai
                                restored_acq_id = ai.id
                                break
                            elif not saved_well and not saved_name:
                                # If no well/name saved, match by original ID if available
                                if saved_acq_id in self.unique_acq_to_original:
                                    original_id = self.unique_acq_to_original.get(ai.id)
                                    if original_id and original_id == self.unique_acq_to_original[saved_acq_id]:
                                        acq_info = ai
                                        restored_acq_id = ai.id
                                        break
                    
                    # Strategy 3: Try matching by extracting base ID from unique ID format
                    if not acq_info and '__file_' in saved_acq_id:
                        base_id = saved_acq_id.split('__file_')[0]
                        # Try to find acquisition with matching original ID
                        for ai in self.acquisitions:
                            if ai.id in self.unique_acq_to_original:
                                original_id = self.unique_acq_to_original[ai.id]
                                if original_id == base_id:
                                    # Also verify source file matches if available
                                    if not saved_source_file or (ai.source_file and os.path.basename(ai.source_file) == os.path.basename(saved_source_file)):
                                        acq_info = ai
                                        restored_acq_id = ai.id
                                        break
                    
                    # Strategy 4: Try matching by well/name alone (fallback)
                    if not acq_info:
                        for ai in self.acquisitions:
                            if saved_well and ai.well and saved_well == ai.well:
                                acq_info = ai
                                restored_acq_id = ai.id
                                break
                            elif saved_name and ai.name and saved_name == ai.name:
                                acq_info = ai
                                restored_acq_id = ai.id
                                break
                
                # If we found acquisition info, set the mask
                if acq_info and restored_acq_id:
                    # Set acquisition info in the mask manager dict cache (same as base app)
                    self.segmentation_masks.set_acq_info(restored_acq_id, acq_info)
                    
                    # Set mask in mask_manager using the correct acquisition ID
                    self.mask_manager.set_mask(restored_acq_id, mask, save_to_disk=True, acq_info=acq_info)
                    
                    # Clear colors for this acquisition so they get regenerated (same as base app)
                    if restored_acq_id in self.segmentation_colors:
                        del self.segmentation_colors[restored_acq_id]
                else:
                    # Fallback: try to store with saved ID (might work if format matches)
                    print(f"Warning: Could not match mask for saved_acq_id={saved_acq_id} to any acquisition")
            
            # Update segmentation overlay text after restoring masks
            if self.current_acq_id:
                self._update_segmentation_overlay_text()
                
                # If masks exist for current acquisition and overlay is enabled, refresh display
                if self.current_acq_id in self.segmentation_masks:
                    # Make sure overlay mode widget is visible if overlay is enabled
                    if hasattr(self, 'segmentation_overlay_mode_widget'):
                        if self.segmentation_overlay:
                            self.segmentation_overlay_mode_widget.setVisible(True)
                    
                    # Refresh display if overlay is enabled to show the masks
                    if self.segmentation_overlay:
                        self.preserve_zoom = True
                        self._view_selected()
        
        # Restore features
        features = loaded_state.get("features", {})
        if "original" in features:
            self.feature_dataframe = features["original"]
        if "batch_corrected" in features:
            self.batch_corrected_dataframe = features["batch_corrected"]
        
        # Restore clustered cells dataframe (cells actually used in clustering)
        if "clustered_cells" in features:
            self.clustered_cells_dataframe = features["clustered_cells"]
            print(f"[Restore State] Restored clustered_cells dataframe with {len(self.clustered_cells_dataframe)} cells")
        else:
            self.clustered_cells_dataframe = None
        
        # Restore analysis module states
        analysis = loaded_state.get("analysis", {})
        restored_feature_set = analysis.get("feature_set_preference")
        if restored_feature_set is None:
            restored_feature_set = (
                analysis.get("clustering", {})
                .get("ui_state", {})
                .get("feature_set")
            )
        
        # Restore QC analysis state
        if "qc_analysis" in analysis:
            if not hasattr(self, 'qc_results_cache'):
                self.qc_results_cache = {}
            qc_state = analysis["qc_analysis"]
            # Separate UI state from cache data
            if "ui_state" in qc_state:
                self._saved_qc_ui_state = qc_state["ui_state"]
                # Remove UI state from cache data
                qc_cache = {k: v for k, v in qc_state.items() if k != "ui_state"}
                self.qc_results_cache.update(qc_cache)
            else:
                self.qc_results_cache.update(qc_state)
        
        # Restore clustering state
        if "clustering" in analysis:
            clustering_state = analysis["clustering"]
            # Note: Clustering dialog will be restored when opened
            # Store state for later restoration
            if not hasattr(self, '_saved_clustering_state'):
                self._saved_clustering_state = {}
            self._saved_clustering_state = clustering_state
        
        # Restore spatial analysis state
        if "spatial" in analysis:
            spatial_state = analysis["spatial"]
            # Store state for later restoration
            if not hasattr(self, '_saved_spatial_state'):
                self._saved_spatial_state = {}
            self._saved_spatial_state = spatial_state
        
        # Restore batch correction state
        if "batch_correction" in analysis:
            batch_correction_state = analysis["batch_correction"]
            self._restore_batch_correction_state(batch_correction_state)
        
        # Restore pixel correlation state (dialog restored when opened)
        if "pixel_correlation" in analysis:
            pixel_state = analysis["pixel_correlation"]
            if not hasattr(self, '_saved_pixel_correlation_state'):
                self._saved_pixel_correlation_state = {}
            self._saved_pixel_correlation_state = pixel_state

        self.analysis_feature_set_preference = self._get_effective_analysis_feature_set_preference(restored_feature_set)
        
        # Update UI to reflect loaded state
        if self.feature_dataframe is not None:
            # Enable buttons that depend on features
            if hasattr(self, 'clustering_btn'):
                self.clustering_btn.setEnabled(True)
            if hasattr(self, 'spatial_btn'):
                self.spatial_btn.setEnabled(True)
            if hasattr(self, 'batch_correction_btn'):
                self.batch_correction_btn.setEnabled(True)
    
    def _restore_clustering_state(self, dialog, state: Dict[str, Any]):
        """Restore clustering dialog state including all parameters."""
        try:
            # Restore clustered_data directly from saved clustered_cells dataframe
            # This is the cleanest approach - we saved exactly what cells were clustered
            if hasattr(self, 'clustered_cells_dataframe') and self.clustered_cells_dataframe is not None:
                print(f"[Restore Clustering] Using clustered_cells_dataframe with {len(self.clustered_cells_dataframe)} cells")
                dialog.clustered_data = self.clustered_cells_dataframe.copy()
                dialog.clustered_data = dialog.clustered_data.sort_values('cluster')
                dialog.clustered_data_unscaled = dialog.clustered_data.copy()
                
                # Extract cluster_labels for compatibility
                if 'cluster' in dialog.clustered_data.columns:
                    dialog.cluster_labels = dialog.clustered_data['cluster'].values
                    print(f"[Restore Clustering] Unique clusters: {sorted(dialog.clustered_data['cluster'].unique())}")
            else:
                # Fallback: Use old method if clustered_cells_dataframe not available (backward compatibility)
                print(f"[Restore Clustering] Fallback: clustered_cells_dataframe not found, using cluster_labels")
                if "cluster_labels" in state and state["cluster_labels"] is not None:
                    if hasattr(dialog, 'cluster_labels'):
                        dialog.cluster_labels = state["cluster_labels"]
                        # Apply cluster labels to feature_dataframe and create clustered_data
                        if (hasattr(dialog, 'feature_dataframe') and dialog.feature_dataframe is not None and
                            dialog.cluster_labels is not None):
                            
                            # Check if feature_dataframe already has a cluster column
                            if 'cluster' in dialog.feature_dataframe.columns:
                                print(f"[Restore Clustering] feature_dataframe already has cluster column")
                                # Use existing cluster column, filter out cluster 0
                                valid_cluster_mask = (dialog.feature_dataframe['cluster'] != 0) & (dialog.feature_dataframe['cluster'].notna())
                                dialog.clustered_data = dialog.feature_dataframe[valid_cluster_mask].copy()
                                print(f"[Restore Clustering] Created clustered_data with {len(dialog.clustered_data)} cells")
                                dialog.clustered_data = dialog.clustered_data.sort_values('cluster')
                                dialog.clustered_data_unscaled = dialog.clustered_data.copy()
                            else:
                                # Apply cluster_labels to feature_dataframe
                                if isinstance(dialog.cluster_labels, pd.Series):
                                    cluster_labels_array = dialog.cluster_labels.values
                                elif isinstance(dialog.cluster_labels, np.ndarray):
                                    cluster_labels_array = dialog.cluster_labels
                                else:
                                    cluster_labels_array = np.array(dialog.cluster_labels)
                                
                                if len(cluster_labels_array) == len(dialog.feature_dataframe):
                                    dialog.feature_dataframe['cluster'] = cluster_labels_array.astype(int)
                                    valid_cluster_mask = (dialog.feature_dataframe['cluster'] != 0) & (dialog.feature_dataframe['cluster'].notna())
                                    dialog.clustered_data = dialog.feature_dataframe[valid_cluster_mask].copy()
                                    dialog.clustered_data = dialog.clustered_data.sort_values('cluster')
                                    dialog.clustered_data_unscaled = dialog.clustered_data.copy()
                                elif hasattr(dialog, 'filter_settings') and dialog.filter_settings is not None:
                                    # Try with filter settings
                                    from openimc.ui.dialogs.clustering import CellClusteringDialog
                                    if isinstance(dialog, CellClusteringDialog):
                                        filtered_df = dialog._apply_filters(dialog.feature_dataframe.copy(), dialog.filter_settings)
                                        if len(cluster_labels_array) == len(filtered_df):
                                            dialog.feature_dataframe['cluster'] = 0
                                            dialog.feature_dataframe.loc[filtered_df.index, 'cluster'] = cluster_labels_array.astype(int)
                                            valid_cluster_mask = (dialog.feature_dataframe['cluster'] != 0) & (dialog.feature_dataframe['cluster'].notna())
                                            dialog.clustered_data = dialog.feature_dataframe[valid_cluster_mask].copy()
                                            dialog.clustered_data = dialog.clustered_data.sort_values('cluster')
                                            dialog.clustered_data_unscaled = dialog.clustered_data.copy()
            
            # Restore normalization config
            if "normalization_config" in state and state["normalization_config"]:
                if hasattr(dialog, 'normalization_config'):
                    dialog.normalization_config = state["normalization_config"]
            
            # Restore cluster annotation map
            if "cluster_annotation_map" in state and state["cluster_annotation_map"]:
                if hasattr(dialog, 'cluster_annotation_map'):
                    dialog.cluster_annotation_map = state["cluster_annotation_map"]
            if "feature_label_map" in state and state["feature_label_map"]:
                if hasattr(dialog, 'feature_label_map'):
                    dialog.feature_label_map = state["feature_label_map"]
            if hasattr(dialog, '_apply_cluster_annotations_to_dataframes'):
                try:
                    dialog._apply_cluster_annotations_to_dataframes()
                except Exception:
                    pass
            
            # Restore filter settings (excluded cells, area filters, etc.)
            if "filter_settings" in state and state["filter_settings"] is not None:
                if hasattr(dialog, 'filter_settings'):
                    dialog.filter_settings = state["filter_settings"]
            
            # Restore scaling method (z-score/mad/none) - both attribute and UI
            if "clustering_scaling_method" in state and state["clustering_scaling_method"]:
                dialog.clustering_scaling_method = state["clustering_scaling_method"]
                # Also set the UI combo box
                if hasattr(dialog, 'clustering_scaling_combo'):
                    dialog.clustering_scaling_combo.blockSignals(True)
                    dialog.clustering_scaling_combo.setCurrentText(state["clustering_scaling_method"])
                    dialog.clustering_scaling_combo.blockSignals(False)
                    print(f"[Restore Clustering] Restored clustering scaling method: {state['clustering_scaling_method']}")
            
            # Restore custom names
            if "feature_label_map" in state and state["feature_label_map"]:
                if hasattr(dialog, 'feature_label_map'):
                    dialog.feature_label_map = state["feature_label_map"]
            
            if "patient_annotation_map" in state and state["patient_annotation_map"]:
                if hasattr(dialog, 'patient_annotation_map'):
                    dialog.patient_annotation_map = state["patient_annotation_map"]
            
            if "patient_cohort_map" in state and state["patient_cohort_map"]:
                if hasattr(dialog, 'patient_cohort_map'):
                    dialog.patient_cohort_map = state["patient_cohort_map"]
            
            if "cluster_backend_names" in state and state["cluster_backend_names"]:
                if hasattr(dialog, 'cluster_backend_names'):
                    dialog.cluster_backend_names = state["cluster_backend_names"]
            
            # Restore original cluster assignments (before merging)
            if "original_cluster_assignments" in state and state["original_cluster_assignments"] is not None:
                if hasattr(dialog, 'original_cluster_assignments'):
                    dialog.original_cluster_assignments = state["original_cluster_assignments"]
            
            # Restore plot settings
            if "cluster_map_orientation" in state:
                if hasattr(dialog, 'cluster_map_orientation'):
                    dialog.cluster_map_orientation = state["cluster_map_orientation"]
            
            if "cluster_map_dendrogram" in state:
                if hasattr(dialog, 'cluster_map_dendrogram'):
                    dialog.cluster_map_dendrogram = state["cluster_map_dendrogram"]
            
            if "cluster_map_zscore_method" in state:
                if hasattr(dialog, 'cluster_map_zscore_method'):
                    dialog.cluster_map_zscore_method = state["cluster_map_zscore_method"]
            
            if "cluster_map_cell_size" in state:
                if hasattr(dialog, 'cluster_map_cell_size'):
                    dialog.cluster_map_cell_size = state["cluster_map_cell_size"]

            if "cluster_map_colorbar_width" in state:
                if hasattr(dialog, 'cluster_map_colorbar_width'):
                    dialog.cluster_map_colorbar_width = state["cluster_map_colorbar_width"]

            if "cluster_map_colorbar_position" in state:
                if hasattr(dialog, 'cluster_map_colorbar_position'):
                    dialog.cluster_map_colorbar_position = state["cluster_map_colorbar_position"]

            if "cluster_map_colorbar_orientation" in state:
                if hasattr(dialog, 'cluster_map_colorbar_orientation'):
                    dialog.cluster_map_colorbar_orientation = state["cluster_map_colorbar_orientation"]
            
            # Restore heatmap scaling
            if "heatmap_scaling" in state:
                if hasattr(dialog, 'heatmap_scaling_combo'):
                    dialog.heatmap_scaling_combo.blockSignals(True)
                    dialog.heatmap_scaling_combo.setCurrentText(state["heatmap_scaling"])
                    dialog.heatmap_scaling_combo.blockSignals(False)
            
            # Restore patient annotation settings
            if "patient_annotation_column" in state:
                if hasattr(dialog, 'patient_annotation_column'):
                    dialog.patient_annotation_column = state["patient_annotation_column"]
            
            if "patient_annotation_enabled" in state:
                if hasattr(dialog, 'patient_annotation_enabled'):
                    dialog.patient_annotation_enabled = state["patient_annotation_enabled"]
            
            if "patient_legend_label" in state:
                if hasattr(dialog, 'patient_legend_label'):
                    dialog.patient_legend_label = state["patient_legend_label"]
            
            # Restore plot customization settings (font sizes, legend layout)
            if "x_tick_fontsize" in state:
                dialog.x_tick_fontsize = state["x_tick_fontsize"]
            
            if "y_tick_fontsize" in state:
                dialog.y_tick_fontsize = state["y_tick_fontsize"]
            
            if "legend_fontsize" in state:
                dialog.legend_fontsize = state["legend_fontsize"]
            
            if "legend_nrows" in state:
                dialog.legend_nrows = state["legend_nrows"]
            
            if "legend_ncols" in state:
                dialog.legend_ncols = state["legend_ncols"]
            
            # Restore clustering method and parameters (for reference)
            if "clustering_method" in state:
                dialog.last_clustering_method = state["clustering_method"]
            
            if "clustering_parameters" in state:
                dialog.last_clustering_params = state["clustering_parameters"]
            
            if "features_used" in state:
                # Initialize attribute if it doesn't exist
                dialog.last_features_used = state["features_used"]
            
            # Restore actual clustering method and dendrogram mode (what was actually used)
            if "actual_clustering_method" in state:
                dialog.actual_clustering_method = state["actual_clustering_method"]
            
            if "actual_dendrogram_mode" in state:
                dialog.actual_dendrogram_mode = state["actual_dendrogram_mode"]
            
            # Restore LLM phenotype cache
            if "llm_phenotype_cache" in state:
                cached_data = state["llm_phenotype_cache"]
                if hasattr(dialog, 'llm_phenotype_cache'):
                    # Restore cache, ensuring it's a proper dict
                    if cached_data:
                        dialog.llm_phenotype_cache = copy.deepcopy(cached_data)
                    else:
                        dialog.llm_phenotype_cache = {}
            else:
                # Ensure cache is initialized even if not in saved state
                if hasattr(dialog, 'llm_phenotype_cache'):
                    if not isinstance(dialog.llm_phenotype_cache, dict):
                        dialog.llm_phenotype_cache = {}
            
            # Restore selected display features for heatmap
            if "selected_display_features" in state and state["selected_display_features"]:
                if hasattr(dialog, 'selected_display_features'):
                    # Convert to set if it's a list
                    if isinstance(state["selected_display_features"], list):
                        dialog.selected_display_features = set(state["selected_display_features"])
                    else:
                        dialog.selected_display_features = state["selected_display_features"]
            
            # Restore UMAP and t-SNE embeddings
            if "umap_embedding" in state and state["umap_embedding"] is not None:
                if hasattr(dialog, 'umap_embedding'):
                    dialog.umap_embedding = state["umap_embedding"]
                    # Restore corresponding index if available
                    if "umap_index" in state and state["umap_index"] is not None:
                        if hasattr(dialog, 'umap_index'):
                            dialog.umap_index = pd.Index(state["umap_index"])
                            # Safety check: ensure umap_index aligns with clustered_data if it exists
                            if hasattr(dialog, 'clustered_data') and dialog.clustered_data is not None:
                                # UMAP should only include cells that were actually clustered
                                # Filter umap_index to only include indices present in clustered_data
                                valid_indices = [idx for idx in dialog.umap_index if idx in dialog.clustered_data.index]
                                if len(valid_indices) != len(dialog.umap_index):
                                    # Some indices in UMAP are not in clustered_data
                                    # This can happen if cells were filtered out
                                    if len(valid_indices) == len(dialog.umap_embedding):
                                        # Perfect match - just update index
                                        dialog.umap_index = pd.Index(valid_indices)
                                    elif len(dialog.umap_embedding) == len(dialog.clustered_data):
                                        # Embedding matches clustered_data size, use its index
                                        dialog.umap_index = dialog.clustered_data.index
                                    else:
                                        # Can't reconcile - clear UMAP state to avoid errors
                                        print(f"Warning: UMAP state mismatch - umap_embedding size: {len(dialog.umap_embedding)}, "
                                              f"clustered_data size: {len(dialog.clustered_data)}, valid_indices: {len(valid_indices)}")
                                        dialog.umap_embedding = None
                                        dialog.umap_index = None
                                elif len(dialog.umap_embedding) != len(dialog.umap_index):
                                    # Embedding size doesn't match index size
                                    print(f"Warning: UMAP embedding size ({len(dialog.umap_embedding)}) != "
                                          f"index size ({len(dialog.umap_index)})")
                                    # Try to reconcile
                                    min_len = min(len(dialog.umap_embedding), len(dialog.umap_index))
                                    dialog.umap_embedding = dialog.umap_embedding[:min_len]
                                    dialog.umap_index = dialog.umap_index[:min_len]
            
            if "tsne_embedding" in state and state["tsne_embedding"] is not None:
                if hasattr(dialog, 'tsne_embedding'):
                    dialog.tsne_embedding = state["tsne_embedding"]
                    # Restore corresponding index if available
                    if "tsne_index" in state and state["tsne_index"] is not None:
                        if hasattr(dialog, 'tsne_index'):
                            dialog.tsne_index = pd.Index(state["tsne_index"])
            
            # Restore UI element states
            if "ui_state" in state:
                ui_state = state["ui_state"]
                
                # View selection
                if "view" in ui_state and hasattr(dialog, 'view_combo'):
                    index = dialog.view_combo.findText(ui_state["view"])
                    if index >= 0:
                        dialog.view_combo.blockSignals(True)
                        dialog.view_combo.setCurrentIndex(index)
                        dialog.view_combo.blockSignals(False)
                
                # Feature set selection
                if "feature_set" in ui_state and hasattr(dialog, 'feature_set_combo'):
                    dialog.feature_set_combo.blockSignals(True)
                    dialog.feature_set_combo.setCurrentText(ui_state["feature_set"])
                    dialog.feature_set_combo.blockSignals(False)
                
                # Clustering method and parameters
                if "clustering_type" in ui_state and hasattr(dialog, 'clustering_type'):
                    dialog.clustering_type.blockSignals(True)
                    dialog.clustering_type.setCurrentText(ui_state["clustering_type"])
                    dialog.clustering_type.blockSignals(False)
                
                if "n_clusters" in ui_state and hasattr(dialog, 'n_clusters'):
                    dialog.n_clusters.setValue(ui_state["n_clusters"])
                
                if "seed" in ui_state and hasattr(dialog, 'seed_spinbox'):
                    dialog.seed_spinbox.setValue(ui_state["seed"])

                # PCA feature representation
                if "use_pca" in ui_state and hasattr(dialog, 'use_pca_checkbox'):
                    dialog.use_pca_checkbox.setChecked(bool(ui_state["use_pca"]))

                if "pca_mode" in ui_state and hasattr(dialog, 'pca_mode_combo'):
                    mode = ui_state["pca_mode"]
                    index = dialog.pca_mode_combo.findData(mode)
                    if index < 0:
                        index = dialog.pca_mode_combo.findText(str(mode))
                    if index >= 0:
                        dialog.pca_mode_combo.setCurrentIndex(index)

                if "pca_variance_percent" in ui_state and hasattr(dialog, 'pca_variance_spinbox'):
                    dialog.pca_variance_spinbox.setValue(float(ui_state["pca_variance_percent"]))

                if "pca_n_components" in ui_state and hasattr(dialog, 'pca_n_components_spinbox'):
                    dialog.pca_n_components_spinbox.setValue(int(ui_state["pca_n_components"]))

                if hasattr(dialog, '_on_pca_controls_changed'):
                    dialog._on_pca_controls_changed()
                
                # Hierarchical clustering
                if "hierarchical_method" in ui_state and hasattr(dialog, 'hierarchical_method'):
                    dialog.hierarchical_method.setCurrentText(ui_state["hierarchical_method"])
                
                # Leiden clustering
                if "leiden_mode" in ui_state:
                    if ui_state["leiden_mode"] == "resolution" and hasattr(dialog, 'resolution_radio'):
                        dialog.resolution_radio.setChecked(True)
                    elif hasattr(dialog, 'modularity_radio'):
                        dialog.modularity_radio.setChecked(True)
                
                if "n_neighbors" in ui_state and hasattr(dialog, 'n_neighbors_spinbox'):
                    dialog.n_neighbors_spinbox.setValue(ui_state["n_neighbors"])
                
                if "resolution" in ui_state and hasattr(dialog, 'resolution_spinbox'):
                    dialog.resolution_spinbox.setValue(ui_state["resolution"])
                
                if "leiden_metric" in ui_state and hasattr(dialog, 'leiden_metric_combo'):
                    dialog.leiden_metric_combo.setCurrentText(ui_state["leiden_metric"])
                
                if "jaccard_weighting" in ui_state and hasattr(dialog, 'jaccard_checkbox'):
                    dialog.jaccard_checkbox.setChecked(ui_state["jaccard_weighting"])
                
                # HDBSCAN clustering
                if "min_cluster_size" in ui_state and hasattr(dialog, 'min_cluster_size_spinbox'):
                    dialog.min_cluster_size_spinbox.setValue(ui_state["min_cluster_size"])
                
                if "min_samples" in ui_state and hasattr(dialog, 'min_samples_spinbox'):
                    dialog.min_samples_spinbox.setValue(ui_state["min_samples"])
                
                if "cluster_selection_method" in ui_state and hasattr(dialog, 'cluster_selection_combo'):
                    dialog.cluster_selection_combo.setCurrentText(ui_state["cluster_selection_method"])
                
                if "metric" in ui_state and hasattr(dialog, 'metric_combo'):
                    dialog.metric_combo.setCurrentText(ui_state["metric"])
                
                # Dendrogram mode
                if "dendro_mode" in ui_state and hasattr(dialog, 'dendro_mode'):
                    dialog.dendro_mode.setCurrentText(ui_state["dendro_mode"])
                
                # Visualization settings
                if "color_by" in ui_state and hasattr(dialog, 'color_by_listwidget'):
                    dialog.color_by_listwidget.clearSelection()
                    for i in range(dialog.color_by_listwidget.count()):
                        item = dialog.color_by_listwidget.item(i)
                        if item.text() in ui_state["color_by"]:
                            item.setSelected(True)
                    # Trigger update
                    if hasattr(dialog, '_on_color_by_changed'):
                        dialog._on_color_by_changed()
                
                if "use_cohort_coloring" in ui_state and hasattr(dialog, 'use_cohort_checkbox'):
                    dialog.use_cohort_checkbox.setChecked(ui_state["use_cohort_coloring"])
                
                if "point_size" in ui_state and hasattr(dialog, 'point_size_spinbox'):
                    dialog.point_size_spinbox.setValue(ui_state["point_size"])
                
                if "point_alpha" in ui_state and hasattr(dialog, 'point_alpha_spinbox'):
                    dialog.point_alpha_spinbox.setValue(ui_state["point_alpha"])
                
                if "show_legend" in ui_state and hasattr(dialog, 'show_legend_checkbox'):
                    dialog.show_legend_checkbox.setChecked(ui_state["show_legend"])
                
                if "group_by" in ui_state and hasattr(dialog, 'group_by_combo'):
                    index = dialog.group_by_combo.findText(ui_state["group_by"])
                    if index >= 0:
                        dialog.group_by_combo.setCurrentIndex(index)
                
                if "stacked_bars_view_type" in ui_state and hasattr(dialog, 'stacked_bars_view_type_combo'):
                    dialog.stacked_bars_view_type_combo.setCurrentText(ui_state["stacked_bars_view_type"])
                
                if "colormap" in ui_state and hasattr(dialog, 'colormap_combo'):
                    dialog.colormap_combo.setCurrentText(ui_state["colormap"])
                
                if "top_n" in ui_state and hasattr(dialog, 'top_n_spinbox'):
                    dialog.top_n_spinbox.setValue(ui_state["top_n"])
                
                if "stacked_bars_filter_selection" in ui_state and hasattr(dialog, 'stacked_bars_filter_selection'):
                    dialog.stacked_bars_filter_selection = set(ui_state["stacked_bars_filter_selection"])
                
                # Update clustering method UI to show/hide appropriate controls
                if hasattr(dialog, '_on_clustering_type_changed'):
                    try:
                        dialog._on_clustering_type_changed()
                    except Exception as e:
                        print(f"Warning: Could not update clustering method UI: {e}")
                
                # Update Leiden mode UI if applicable
                if hasattr(dialog, '_on_leiden_mode_changed'):
                    try:
                        dialog._on_leiden_mode_changed()
                    except Exception as e:
                        print(f"Warning: Could not update Leiden mode UI: {e}")
            
            # After restoring state, ensure clustered_data is properly set up and redraw heatmap
            if (hasattr(dialog, 'cluster_labels') and dialog.cluster_labels is not None and
                hasattr(dialog, 'feature_dataframe') and dialog.feature_dataframe is not None):
                # Ensure clustered_data is set up from feature_dataframe with cluster column
                if 'cluster' in dialog.feature_dataframe.columns:
                    dialog.clustered_data = dialog.feature_dataframe.copy()
                    dialog.clustered_data = dialog.clustered_data.sort_values('cluster')
                    dialog.clustered_data_unscaled = dialog.clustered_data.copy()
                
                # Determine which view to restore
                restored_view = None
                if "ui_state" in state and "view" in state["ui_state"]:
                    restored_view = state["ui_state"]["view"]
                
                # Redraw appropriate view if we have the necessary data
                if hasattr(dialog, 'clustered_data') and dialog.clustered_data is not None:
                    # Set view if specified, otherwise default to Heatmap if features are selected
                    if restored_view and hasattr(dialog, 'view_combo'):
                        dialog.view_combo.blockSignals(True)
                        dialog.view_combo.setCurrentText(restored_view)
                        dialog.view_combo.blockSignals(False)
                    elif (hasattr(dialog, 'selected_display_features') and dialog.selected_display_features and
                          hasattr(dialog, 'view_combo')):
                        dialog.view_combo.blockSignals(True)
                        dialog.view_combo.setCurrentText('Heatmap')
                        dialog.view_combo.blockSignals(False)
                    
                    # Trigger appropriate view redraw
                    if restored_view == 'Heatmap' or (not restored_view and dialog.selected_display_features):
                        if hasattr(dialog, '_show_heatmap'):
                            try:
                                dialog._show_heatmap()
                            except Exception as e:
                                print(f"Warning: Could not redraw heatmap after state restoration: {e}")
                    elif restored_view == 'UMAP' and hasattr(dialog, 'umap_embedding') and dialog.umap_embedding is not None:
                        if hasattr(dialog, '_show_umap'):
                            try:
                                dialog._show_umap()
                            except Exception as e:
                                print(f"Warning: Could not redraw UMAP after state restoration: {e}")
                    elif restored_view == 't-SNE' and hasattr(dialog, 'tsne_embedding') and dialog.tsne_embedding is not None:
                        if hasattr(dialog, '_show_tsne'):
                            try:
                                dialog._show_tsne()
                            except Exception as e:
                                print(f"Warning: Could not redraw t-SNE after state restoration: {e}")
            
            # Note: Clustered data will be regenerated when needed
            # The cluster labels and all settings are the key state that needs to be restored
        except Exception as e:
            print(f"Warning: Could not fully restore clustering state: {e}")
            import traceback
            traceback.print_exc()
    
    def _rebuild_adj_matrices_for_simple_spatial(self, dialog):
        """Rebuild adjacency matrices from edge_df for simple spatial analysis."""
        try:
            # Import scipy.sparse if available
            try:
                from scipy import sparse as sp
                _HAVE_SPARSE = True
            except ImportError:
                _HAVE_SPARSE = False
                print("Warning: scipy.sparse not available, cannot rebuild adjacency matrices")
                return
            
            # Get filtered dataframe
            filtered_df = dialog.feature_dataframe
            roi_col = dialog._get_roi_column() if hasattr(dialog, '_get_roi_column') else None
            
            if not roi_col or roi_col not in filtered_df.columns:
                # No ROI grouping, treat as single ROI
                roi_groups = [(None, filtered_df)]
            else:
                roi_groups = list(filtered_df.groupby(roi_col))
            
            # Initialize adjacency matrices dict
            dialog.adj_matrices = {}
            global_id_counter = 0
            
            for roi_id, roi_df in roi_groups:
                roi_id_str = str(roi_id) if roi_id is not None else "global"
                roi_edges = dialog.edge_df[dialog.edge_df['roi_id'] == roi_id_str] if 'roi_id' in dialog.edge_df.columns else dialog.edge_df
                
                if roi_edges.empty:
                    continue
                
                # Get cell IDs for this ROI
                cell_ids = roi_df["cell_id"].astype(int).to_numpy() if 'cell_id' in roi_df.columns else roi_df.index.values
                n_cells = len(cell_ids)
                
                # Build local cell_id to global_id mapping for this ROI
                roi_cell_to_gid = {}
                for cell_id in cell_ids:
                    roi_cell_to_gid[cell_id] = global_id_counter
                    global_id_counter += 1
                
                # Build adjacency matrix from edges
                rows, cols, data = [], [], []
                for _, edge in roi_edges.iterrows():
                    src_cell_id = int(edge['cell_id_A'])
                    dst_cell_id = int(edge['cell_id_B'])
                    
                    if src_cell_id in roi_cell_to_gid and dst_cell_id in roi_cell_to_gid:
                        src_gid = roi_cell_to_gid[src_cell_id]
                        dst_gid = roi_cell_to_gid[dst_cell_id]
                        
                        # Convert global IDs to local indices for this ROI
                        src_local = src_gid - (global_id_counter - n_cells)
                        dst_local = dst_gid - (global_id_counter - n_cells)
                        
                        # Add both directions (undirected graph)
                        rows.extend([src_local, dst_local])
                        cols.extend([dst_local, src_local])
                        data.extend([1.0, 1.0])
                
                if rows:
                    adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
                    dialog.adj_matrices[roi_id_str] = adj_matrix.tocsr()
            
            print(f"Rebuilt {len(dialog.adj_matrices)} adjacency matrices from edge DataFrame")
            
        except Exception as e:
            print(f"Error rebuilding adjacency matrices: {e}")
            import traceback
            traceback.print_exc()
    
    def _restore_spatial_state(self, dialog, state: Dict[str, Any]):
        """Restore spatial analysis dialog state (both Simple and Advanced)."""
        try:
            dialog_type = state.get("dialog_type", "legacy")
            
            # Common state for both Simple and Advanced
            if "cluster_annotation_map" in state and state["cluster_annotation_map"]:
                if hasattr(dialog, 'cluster_annotation_map'):
                    dialog.cluster_annotation_map = state["cluster_annotation_map"]
            
            # Simple Spatial Analysis specific restoration
            if dialog_type == "simple":
                # Restore graph data - actual DataFrame
                if "edge_df" in state and state["edge_df"] is not None:
                    if hasattr(dialog, 'edge_df'):
                        dialog.edge_df = state["edge_df"]
                
                # Restore metadata
                if "metadata" in state and state["metadata"]:
                    if hasattr(dialog, 'metadata'):
                        dialog.metadata = state["metadata"]
                
                # Restore cell ID mappings
                if "cell_id_to_gid" in state and state["cell_id_to_gid"]:
                    if hasattr(dialog, 'cell_id_to_gid'):
                        # Convert back from serialized format
                        # Format: "roi_id_cell_id" -> (roi_id, cell_id)
                        try:
                            dialog.cell_id_to_gid = {}
                            for k, v in state["cell_id_to_gid"].items():
                                parts = k.split('_', 1)
                                if len(parts) == 2:
                                    dialog.cell_id_to_gid[(parts[0], int(parts[1]))] = v
                        except Exception as e:
                            print(f"Warning: Could not restore cell_id_to_gid: {e}")
                
                if "gid_to_cell_id" in state and state["gid_to_cell_id"]:
                    if hasattr(dialog, 'gid_to_cell_id'):
                        # Convert back from serialized format
                        # Format: "gid" -> [roi_id, cell_id]
                        try:
                            dialog.gid_to_cell_id = {}
                            for k, v in state["gid_to_cell_id"].items():
                                if isinstance(v, list) and len(v) == 2:
                                    dialog.gid_to_cell_id[int(k)] = (v[0], int(v[1]))
                        except Exception as e:
                            print(f"Warning: Could not restore gid_to_cell_id: {e}")
                
                # Rebuild adjacency matrices from edge_df
                # This is critical for enabling spatial analysis tabs
                if hasattr(dialog, 'edge_df') and dialog.edge_df is not None and not dialog.edge_df.empty:
                    try:
                        self._rebuild_adj_matrices_for_simple_spatial(dialog)
                    except Exception as e:
                        print(f"Warning: Could not rebuild adjacency matrices: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Restore analysis results DataFrames - actual DataFrames
                if "cluster_summary_df" in state and state["cluster_summary_df"] is not None:
                    if hasattr(dialog, 'cluster_summary_df'):
                        dialog.cluster_summary_df = state["cluster_summary_df"]
                        # Update UI if there's a method to display cluster summary
                        if hasattr(dialog, '_update_cluster_summary_display'):
                            try:
                                dialog._update_cluster_summary_display()
                            except Exception as e:
                                print(f"Warning: Could not update cluster summary display: {e}")
                
                if "enrichment_df" in state and state["enrichment_df"] is not None:
                    if hasattr(dialog, 'enrichment_df'):
                        dialog.enrichment_df = state["enrichment_df"]
                        # Update UI - call the correct method to display enrichment plot
                        if hasattr(dialog, '_update_enrichment_plot'):
                            try:
                                dialog._update_enrichment_plot()
                            except Exception as e:
                                print(f"Warning: Could not update enrichment plot: {e}")
                
                if "distance_df" in state and state["distance_df"] is not None:
                    if hasattr(dialog, 'distance_df'):
                        dialog.distance_df = state["distance_df"]
                        # Update UI - populate cluster list and update distance plot
                        if hasattr(dialog, '_populate_distance_cluster_list'):
                            try:
                                dialog._populate_distance_cluster_list()
                            except Exception as e:
                                print(f"Warning: Could not populate distance cluster list: {e}")
                        if hasattr(dialog, '_update_distance_plot'):
                            try:
                                dialog._update_distance_plot()
                            except Exception as e:
                                print(f"Warning: Could not update distance plot: {e}")
                
                # Restore spatial visualization cache
                if "spatial_viz_cache" in state and state["spatial_viz_cache"]:
                    if hasattr(dialog, 'spatial_viz_cache'):
                        dialog.spatial_viz_cache = state["spatial_viz_cache"]
                
                # Restore analysis flags
                if "enrichment_analysis_run" in state:
                    if hasattr(dialog, 'enrichment_analysis_run'):
                        dialog.enrichment_analysis_run = state["enrichment_analysis_run"]
                
                if "distance_analysis_run" in state:
                    if hasattr(dialog, 'distance_analysis_run'):
                        dialog.distance_analysis_run = state["distance_analysis_run"]
                
                if "spatial_viz_run" in state:
                    if hasattr(dialog, 'spatial_viz_run'):
                        dialog.spatial_viz_run = state["spatial_viz_run"]
                
                if "community_analysis_run" in state:
                    if hasattr(dialog, 'community_analysis_run'):
                        dialog.community_analysis_run = state["community_analysis_run"]
                
                # Restore source file filters
                if "selected_source_files" in state and state["selected_source_files"]:
                    if hasattr(dialog, 'selected_source_files'):
                        dialog.selected_source_files = set(state["selected_source_files"])
                
                if "available_source_files" in state and state["available_source_files"]:
                    if hasattr(dialog, 'available_source_files'):
                        dialog.available_source_files = set(state["available_source_files"])
                
                # Restore random seed
                if "rng_seed" in state:
                    if hasattr(dialog, 'rng_seed'):
                        dialog.rng_seed = state["rng_seed"]
                
                # Update tab states after restoring all state
                # This is critical for enabling tabs and buttons based on restored data
                if hasattr(dialog, '_update_tab_states'):
                    try:
                        dialog._update_tab_states()
                        print("Updated tab states after spatial analysis state restoration")
                    except Exception as e:
                        print(f"Warning: Could not update tab states: {e}")
            
            # Advanced Spatial Analysis specific restoration
            elif dialog_type == "advanced":
                # Restore analysis status
                if "analysis_status" in state and state["analysis_status"]:
                    if hasattr(dialog, 'analysis_status'):
                        dialog.analysis_status = state["analysis_status"]
                
                # Restore processed ROIs info
                if "processed_rois" in state and state["processed_rois"]:
                    if hasattr(dialog, 'processed_rois'):
                        dialog.processed_rois = state["processed_rois"]
                
                # Restore aggregated results
                if "aggregated_results" in state and state["aggregated_results"]:
                    if hasattr(dialog, 'aggregated_results'):
                        # Reconstruct TempAnnData objects if needed
                        restored_aggregated = {}
                        for key, value in state["aggregated_results"].items():
                            # Check if it's a serialized TempAnnData object
                            if isinstance(value, dict) and value.get("__type__") == "TempAnnData":
                                # Reconstruct TempAnnData object
                                class TempAnnData:
                                    def __init__(self, matrix, cluster_key, obs, significant_counts=None):
                                        self.uns = {'nhood_enrichment': {'zscore': matrix}}
                                        self.obs = obs
                                        self._cluster_key = cluster_key
                                        self._significant_counts = significant_counts
                                
                                # Extract data from serialized format
                                uns_data = value.get("uns", {})
                                obs_data = value.get("obs")
                                cluster_key = value.get("_cluster_key")
                                significant_counts = value.get("_significant_counts")
                                
                                # Extract matrix from uns structure
                                matrix = None
                                if 'nhood_enrichment' in uns_data:
                                    if isinstance(uns_data['nhood_enrichment'], dict):
                                        matrix = uns_data['nhood_enrichment'].get('zscore')
                                    else:
                                        matrix = uns_data['nhood_enrichment']
                                
                                if matrix is not None and obs_data is not None and cluster_key is not None:
                                    temp_adata = TempAnnData(matrix, cluster_key, obs_data, significant_counts)
                                    restored_aggregated[key] = temp_adata
                                else:
                                    # Fallback: try to use value as-is if it's already an AnnData-like object
                                    restored_aggregated[key] = value
                            else:
                                # Regular object - use as-is
                                restored_aggregated[key] = value
                        
                        dialog.aggregated_results = restored_aggregated
                
                # Restore graph built flag
                if "spatial_graph_built" in state:
                    if hasattr(dialog, 'spatial_graph_built'):
                        dialog.spatial_graph_built = state["spatial_graph_built"]
                
                # Restore graph construction parameters and rebuild graphs if they existed
                if "graph_construction_params" in state and state["has_anndata_cache"]:
                    graph_params = state["graph_construction_params"]
                    
                    # Set UI controls to saved values
                    if "method" in graph_params and hasattr(dialog, 'graph_method_combo'):
                        index = dialog.graph_method_combo.findText(graph_params["method"])
                        if index >= 0:
                            dialog.graph_method_combo.setCurrentIndex(index)
                    
                    if "k" in graph_params and hasattr(dialog, 'graph_k_spin'):
                        dialog.graph_k_spin.setValue(graph_params["k"])
                    
                    if "radius" in graph_params and hasattr(dialog, 'graph_radius_spin'):
                        dialog.graph_radius_spin.setValue(graph_params["radius"])
                    
                    if "seed" in graph_params and hasattr(dialog, 'seed_spinbox'):
                        dialog.seed_spinbox.setValue(graph_params["seed"])
                    
                    # Rebuild spatial graphs with saved parameters
                    # This is critical for enabling analysis tabs
                    if hasattr(dialog, '_create_spatial_graph'):
                        try:
                            print("Rebuilding spatial graphs for advanced spatial analysis...")
                            dialog._create_spatial_graph()
                        except Exception as e:
                            print(f"Warning: Could not rebuild spatial graphs: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Refresh plots for analyses that have aggregated results
                # This ensures visualizations are displayed when loading saved state
                if hasattr(dialog, 'aggregated_results') and dialog.aggregated_results:
                    try:
                        # Refresh neighborhood enrichment plot if available
                        if 'nhood_enrichment' in dialog.aggregated_results:
                            if hasattr(dialog, '_plot_sq_nhood_enrichment'):
                                # Set ROI combo to "All ROIs" to show aggregated results
                                if hasattr(dialog, 'sq_nhood_roi_combo'):
                                    # Find "All ROIs" option (usually index 0)
                                    dialog.sq_nhood_roi_combo.setCurrentIndex(0)
                                dialog._plot_sq_nhood_enrichment(dialog.aggregated_results['nhood_enrichment'])
                                if hasattr(dialog, 'sq_nhood_save_btn'):
                                    dialog.sq_nhood_save_btn.setEnabled(True)
                                print("Refreshed neighborhood enrichment plot from saved state")
                        
                        # Refresh autocorrelation plot if available
                        if 'autocorrelation' in dialog.aggregated_results:
                            if hasattr(dialog, '_plot_sq_autocorrelation'):
                                # Set ROI combo to "All ROIs" to show aggregated results
                                if hasattr(dialog, 'sq_autocorr_roi_combo'):
                                    dialog.sq_autocorr_roi_combo.setCurrentIndex(0)
                                dialog._plot_sq_autocorrelation(dialog.aggregated_results['autocorrelation'])
                                if hasattr(dialog, 'sq_autocorr_save_btn'):
                                    dialog.sq_autocorr_save_btn.setEnabled(True)
                                print("Refreshed autocorrelation plot from saved state")
                    except Exception as e:
                        print(f"Warning: Could not refresh plots from aggregated results: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Update button states after restoring all state
                # This ensures buttons are enabled/disabled based on restored data
                if hasattr(dialog, '_update_button_states'):
                    try:
                        dialog._update_button_states()
                        print("Updated button states after advanced spatial analysis state restoration")
                    except Exception as e:
                        print(f"Warning: Could not update button states: {e}")
                
                # Enable save buttons based on analysis_status
                # This ensures save buttons reflect which analyses have been run
                if hasattr(dialog, 'analysis_status') and dialog.analysis_status:
                    try:
                        # Check if any ROI has neighborhood enrichment
                        has_nhood = any(
                            roi_status.get('nhood_enrichment', False) 
                            for roi_status in dialog.analysis_status.values()
                        )
                        if has_nhood and hasattr(dialog, 'sq_nhood_save_btn'):
                            dialog.sq_nhood_save_btn.setEnabled(True)
                        
                        # Check if any ROI has co-occurrence
                        has_cooccur = any(
                            roi_status.get('co_occurrence', False) 
                            for roi_status in dialog.analysis_status.values()
                        )
                        if has_cooccur and hasattr(dialog, 'sq_cooccur_save_btn'):
                            dialog.sq_cooccur_save_btn.setEnabled(True)
                        
                        # Check if any ROI has autocorrelation
                        has_autocorr = any(
                            roi_status.get('autocorrelation', False) 
                            for roi_status in dialog.analysis_status.values()
                        )
                        if has_autocorr and hasattr(dialog, 'sq_autocorr_save_btn'):
                            dialog.sq_autocorr_save_btn.setEnabled(True)
                        
                        # Check if any ROI has ripley
                        has_ripley = any(
                            roi_status.get('ripley', False) 
                            for roi_status in dialog.analysis_status.values()
                        )
                        if has_ripley and hasattr(dialog, 'sq_ripley_save_btn'):
                            dialog.sq_ripley_save_btn.setEnabled(True)
                    except Exception as e:
                        print(f"Warning: Could not update save button states: {e}")
            
            # Note: For advanced spatial, AnnData objects are rebuilt from the feature dataframe
            # using the saved graph construction parameters.
        except Exception as e:
            print(f"Warning: Could not fully restore spatial state: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_default_analysis_steps_filename(self) -> str:
        """Build a descriptive default filename for analysis-step exports."""
        base_name = "analysis_steps"
        if self.current_path:
            current_path = Path(self.current_path)
            if current_path.is_dir():
                dataset_name = current_path.name
            else:
                dataset_name = current_path.stem
            dataset_name = self._sanitize_filename(dataset_name)
            if dataset_name:
                base_name = f"{dataset_name}_analysis_steps"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}.txt"

    def _save_analysis_steps_to_path(
        self,
        output_path: str,
        *,
        show_success_message: bool = True,
        show_failure_message: bool = True
    ) -> bool:
        """Write the current session's analysis-step summary to disk."""
        exporter = AnalysisStepsExporter()
        success = exporter.export_from_main_window(self, output_path)

        if success:
            acquisitions = [acq.id for acq in self.acquisitions] if self.acquisitions else []
            self._log_export_operation(
                "analysis_steps_txt",
                {
                    "export_scope": "current_session",
                    "session_start_time": self.session_start_time.isoformat(),
                    "source_files": self._get_source_files_for_logging(acquisitions),
                    "output_format": "TXT",
                },
                output_path,
                acquisitions=acquisitions,
                notes="Exported analysis steps summary"
            )
            if show_success_message:
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Analysis steps exported successfully to:\n{output_path}"
                )
            return True

        if show_failure_message:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Failed",
                "Failed to export analysis steps.\n\n"
                "Make sure you have performed some analysis operations that are logged."
            )
        return False

    def _prompt_analysis_steps_export(
        self,
        *,
        dialog_title: str = "Export Analysis Steps",
        show_success_message: bool = True,
        show_failure_message: bool = True
    ) -> Optional[bool]:
        """Prompt for an output path, then export analysis steps."""
        output_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            dialog_title,
            self._get_default_analysis_steps_filename(),
            "Text Files (*.txt);;All Files (*)"
        )
        if not output_path:
            return None
        return self._save_analysis_steps_to_path(
            output_path,
            show_success_message=show_success_message,
            show_failure_message=show_failure_message
        )

    def _export_analysis_steps(self):
        """Export analysis steps to a text file."""
        self._prompt_analysis_steps_export(
            dialog_title="Export Analysis Steps",
            show_success_message=True,
            show_failure_message=True
        )
    
    def _get_openimc_version(self) -> str:
        """Get OpenIMC version information."""
        try:
            import openimc
            if hasattr(openimc, '__version__'):
                return openimc.__version__
        except:
            pass
        
        try:
            from importlib.metadata import version

            return version("openimc")
        except:
            pass
        
        return "Unknown"
