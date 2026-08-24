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

"""
Quality Control Analysis Dialog for OpenIMC

This module provides QC analysis capabilities including:
- Pixel-level QC: SNR calculation using Otsu threshold
- Cell-level QC: SNR calculation using segmentation masks
- Quality metrics: SNR vs intensity, % covered area, cell density, etc.
"""

from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from openimc.utils.logger import get_logger
from openimc.ui.dialogs.figure_save_dialog import save_figure_with_options
from openimc.ui.dialogs.progress_dialog import run_blocking_task_with_progress_then_finalize
from openimc.data.mcd_loader import AcquisitionInfo
from openimc.core import (
    _calculate_qc_snr,
    _compute_cell_signal_metrics,
    aggregate_qc_results,
    qc_analysis,
)
import multiprocessing as mp
import traceback

# Optional scikit-image for Otsu thresholding
_HAVE_SCIKIT_IMAGE = False
try:
    from skimage.filters import threshold_otsu
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False


def _boxplot_with_labels(ax, values, labels):
    """Use the Matplotlib 3.9+ name with a 3.8 compatibility fallback."""
    try:
        return ax.boxplot(values, tick_labels=labels, patch_artist=True)
    except TypeError:
        return ax.boxplot(values, labels=labels, patch_artist=True)


def _calculate_snr(signal_mean: float, background_mean: float, background_std: float, 
                    img_min: Optional[float] = None, img_max: Optional[float] = None) -> float:
    """Compatibility wrapper for the single canonical QC SNR calculation."""
    return _calculate_qc_snr(
        signal_mean,
        background_mean,
        background_std,
        img_min,
        img_max,
    )


CELL_SIGNAL_METHOD_OPTIONS = [
    ("Positive pixels above background", "positive_pixels"),
    ("Upper quantile of cell intensity", "upper_quantile"),
    ("All cell pixels (legacy)", "all_cell_mean"),
]

DEFAULT_QC_SNR_THRESHOLD = 3.0


def _cell_signal_method_label(method: Optional[str]) -> str:
    """Return the UI label for a QC cell signal method identifier."""
    for label, value in CELL_SIGNAL_METHOD_OPTIONS:
        if value == method:
            return label
    return "Cell signal"


# Import worker function from processing module
from openimc.processing.qc_analysis_worker import qc_process_acquisition_worker as _qc_process_acquisition_worker

def _qc_calculate_pixel_metrics_worker(img: np.ndarray, channel: str) -> Optional[Dict[str, Any]]:
    """Calculate pixel-level QC metrics using Otsu threshold (module-level for multiprocessing).
    
    This is a separate worker function for QC analysis to avoid conflicts.
    """
    if not _HAVE_SCIKIT_IMAGE:
        return None
    
    try:
        # Convert to float if needed
        img_float = img.astype(np.float32)
        
        # Calculate Otsu threshold
        threshold = threshold_otsu(img_float)
        
        # Separate signal (foreground) and background
        foreground = img_float[img_float > threshold]
        background = img_float[img_float <= threshold]
        
        if len(foreground) == 0 or len(background) == 0:
            return None
        
        # Calculate metrics
        signal_mean = np.mean(foreground)
        signal_std = np.std(foreground)
        background_mean = np.mean(background)
        background_std = np.std(background)
        
        # Calculate image range for robust SNR calculation
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        
        # SNR: (signal_mean - background_mean) / background_std (with robust handling)
        snr = _calculate_snr(signal_mean, background_mean, background_std, img_min, img_max)
        
        # Intensity metrics (using raw pixel intensities)
        mean_intensity = np.mean(img_float)
        median_intensity = np.median(img_float)
        max_intensity = np.max(img_float)
        min_intensity = np.min(img_float)
        
        # Coverage: percentage of pixels above threshold
        coverage_pct = (len(foreground) / img_float.size) * 100
        
        # Calculate percentiles
        p1 = np.percentile(img_float, 1)
        p25 = np.percentile(img_float, 25)
        p75 = np.percentile(img_float, 75)
        p99 = np.percentile(img_float, 99)
        
        return {
            'snr': snr,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'background_mean': background_mean,
            'background_std': background_std,
            'threshold': threshold,
            'mean_intensity': mean_intensity,  # Raw pixel intensity
            'median_intensity': median_intensity,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'coverage_pct': coverage_pct,
            'p1': p1,
            'p25': p25,
            'p75': p75,
            'p99': p99,
            'total_pixels': img_float.size,
            'foreground_pixels': len(foreground),
            'background_pixels': len(background)
        }
    except Exception as e:
        print(f"Error calculating pixel metrics for {channel}: {e}")
        return None


def _qc_calculate_cell_metrics_worker(
    img: np.ndarray,
    channel: str,
    mask: np.ndarray,
    cell_signal_method: str = "positive_pixels",
    positive_threshold_sd: float = 2.0,
    upper_quantile: float = 0.90,
) -> Optional[Dict[str, Any]]:
    """Calculate cell-level QC metrics using segmentation masks (module-level for multiprocessing).
    
    This is a separate worker function for QC analysis to avoid conflicts.
    """
    try:
        # Convert to float if needed
        img_float = img.astype(np.float32)
        
        # Ensure mask and image have same shape
        if mask.shape != img_float.shape:
            print(f"Warning: Mask shape {mask.shape} doesn't match image shape {img_float.shape}")
            return None
        
        # Separate signal (cells) and background
        cell_mask = mask > 0
        background_mask = mask == 0
        
        if np.sum(cell_mask) == 0 or np.sum(background_mask) == 0:
            return None
        
        background = img_float[background_mask]
        
        background_mean = np.mean(background)
        background_std = np.std(background)
        
        # Calculate image range for robust SNR calculation
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        signal_metrics = _compute_cell_signal_metrics(
            img_float,
            mask,
            float(background_mean),
            float(background_std),
            float(img_min),
            float(img_max),
            cell_signal_method=cell_signal_method,
            positive_threshold_sd=positive_threshold_sd,
            upper_quantile=upper_quantile,
        )
        signal_mean = signal_metrics['signal_mean']
        signal_std = signal_metrics['signal_std']
        snr = signal_metrics['snr']
        
        # Intensity metrics (raw image intensities for consistency with core.qc_analysis)
        mean_intensity = np.mean(img_float)
        median_intensity = np.median(img_float)
        max_intensity = np.max(img_float)
        min_intensity = np.min(img_float)
        
        # Coverage: percentage of pixels covered by cells
        coverage_pct = (np.sum(cell_mask) / img_float.size) * 100
        
        # Cell density: number of cells per unit area
        unique_cells = np.unique(mask[mask > 0])
        num_cells = len(unique_cells)
        area_pixels = img_float.size
        cell_density = num_cells / area_pixels if area_pixels > 0 else 0
        
        # Calculate percentiles for cell intensities
        p1 = np.percentile(img_float[cell_mask], 1)
        p25 = np.percentile(img_float[cell_mask], 25)
        p75 = np.percentile(img_float[cell_mask], 75)
        p99 = np.percentile(img_float[cell_mask], 99)
        
        # Per-cell statistics
        cell_intensities = []
        for cell_id in unique_cells:
            cell_pixels = img_float[mask == cell_id]
            if len(cell_pixels) > 0:
                cell_intensities.append(np.mean(cell_pixels))
        
        mean_cell_intensity = np.mean(cell_intensities) if cell_intensities else 0
        median_cell_intensity = np.median(cell_intensities) if cell_intensities else 0
        
        return {
            'snr': snr,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'background_mean': background_mean,
            'background_std': background_std,
            'cell_signal_method': cell_signal_method,
            'signal_threshold': signal_metrics['signal_threshold'],
            'signal_quantile': signal_metrics['signal_quantile'],
            'n_signal_pixels': signal_metrics['n_signal_pixels'],
            'n_signal_cells': signal_metrics['n_signal_cells'],
            'signal_fraction': signal_metrics['signal_fraction'],
            'signal_coverage_pct': signal_metrics['signal_coverage_pct'],
            'mean_intensity': mean_intensity,  # Raw pixel intensity
            'median_intensity': median_intensity,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'coverage_pct': coverage_pct,
            'cell_density': cell_density,
            'num_cells': num_cells,
            'p1': p1,
            'p25': p25,
            'p75': p75,
            'p99': p99,
            'mean_cell_intensity': mean_cell_intensity,
            'median_cell_intensity': median_cell_intensity,
            'total_pixels': img_float.size,
            'foreground_pixels': np.sum(cell_mask),
            'background_pixels': np.sum(background_mask)
        }
    except Exception as e:
        print(f"Error calculating cell metrics for {channel}: {e}")
        import traceback
        traceback.print_exc()
        return None


class QCAnalysisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quality Control Analysis")
        self.setMinimumSize(1000, 700)
        
        # Get parent window to access images and masks
        self.parent_window = parent
        
        # Store results
        self.qc_results: Optional[pd.DataFrame] = None  # Raw results per ROI per channel
        self.qc_results_aggregated: Optional[pd.DataFrame] = None  # Aggregated results per channel
        self.pixel_level_results: Optional[pd.DataFrame] = None
        self.cell_level_results: Optional[pd.DataFrame] = None
        self.custom_denoise_settings: Dict[str, Dict[str, dict]] = {}
        
        # Analysis mode
        self.analysis_mode = "pixel"  # "pixel" or "cell"
        
        # File set ID for caching results
        self.file_set_id = None
        if self.parent_window and hasattr(self.parent_window, '_get_qc_file_set_id'):
            self.file_set_id = self.parent_window._get_qc_file_set_id()
        
        # Create UI
        self._create_ui()
        
        # Try to restore cached results
        self._restore_cached_results()
        
    def _create_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Options panel (hosted in a separate popup to maximize plot space)
        options_group = QtWidgets.QGroupBox("Analysis Options")
        options_layout = QtWidgets.QVBoxLayout(options_group)
        
        # Analysis mode selection
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("Analysis Level:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Pixel-level", "Cell-level"])
        
        # Check if masks exist and set default mode
        has_masks = self._check_masks_exist()
        if has_masks:
            # Set cell-level as default if masks are available
            self.mode_combo.setCurrentIndex(1)  # Cell-level
            self.analysis_mode = "cell"
        else:
            # Set pixel-level as default if no masks
            self.mode_combo.setCurrentIndex(0)  # Pixel-level
            self.analysis_mode = "pixel"
            self.mode_combo.setItemText(1, "Cell-level (No masks available)")
            self.mode_combo.model().item(1).setEnabled(False)
        
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        options_layout.addLayout(mode_layout)
        
        # Add explanation text
        explanation_text = QtWidgets.QLabel(
            "<b>Pixel-level:</b> Uses Otsu thresholding to automatically separate signal (foreground) "
            "from background pixels. Background-referenced SNR is calculated as "
            "(signal_mean - background_mean) / background_std. "
            "QC intensity plots use the foreground-only mean intensity, not the whole-image mean. "
            "This works on any image without requiring segmentation.<br>"
            "<b>Cell-level:</b> Uses segmentation masks to separate cell pixels from background pixels. "
            "You can define signal as positive in-cell pixels above background, the brightest cells by upper quantile, "
            "or all cell pixels for legacy whole-cell averaging. QC intensity plots use the selected in-cell signal only; "
            "non-cell regions remain the background. "
            "Requires segmentation masks to be available."
        )
        explanation_text.setWordWrap(True)
        explanation_text.setStyleSheet("QLabel { color: #555; font-size: 9pt; padding: 5px; }")
        options_layout.addWidget(explanation_text)
        
        # Acquisition selection
        acq_layout = QtWidgets.QHBoxLayout()
        acq_layout.addWidget(QtWidgets.QLabel("Acquisition:"))
        self.acq_combo = QtWidgets.QComboBox()
        self._populate_acq_combo()
        # Add "All Acquisitions" as the first (default) option
        self.acq_combo.insertItem(0, "All Acquisitions", "all")
        self.acq_combo.setCurrentIndex(0)  # Set as default
        self.acq_combo.currentIndexChanged.connect(self._populate_denoise_channel_list)
        acq_layout.addWidget(self.acq_combo, 1)
        acq_layout.addStretch()
        options_layout.addLayout(acq_layout)
        
        # Number of workers for multiprocessing
        workers_layout = QtWidgets.QHBoxLayout()
        workers_layout.addWidget(QtWidgets.QLabel("Number of workers:"))
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setMinimum(1)
        self.workers_spin.setMaximum(mp.cpu_count())
        self.workers_spin.setValue(max(1, mp.cpu_count() - 2))  # Default to max CPUs - 2
        workers_layout.addWidget(self.workers_spin)
        workers_layout.addStretch()
        options_layout.addLayout(workers_layout)

        snr_threshold_layout = QtWidgets.QHBoxLayout()
        snr_threshold_layout.addWidget(QtWidgets.QLabel("Background-referenced SNR threshold:"))
        self.snr_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.snr_threshold_spin.setRange(0.1, 100.0)
        self.snr_threshold_spin.setSingleStep(0.5)
        self.snr_threshold_spin.setDecimals(1)
        self.snr_threshold_spin.setValue(DEFAULT_QC_SNR_THRESHOLD)
        self.snr_threshold_spin.setToolTip(
            "Reference line used in background-referenced QC SNR plots. "
            "A value above 1.0 is typically more conservative than signal=noise."
        )
        snr_threshold_layout.addWidget(self.snr_threshold_spin)
        snr_threshold_layout.addStretch()
        options_layout.addLayout(snr_threshold_layout)

        self.cell_signal_group = QtWidgets.QGroupBox("Cell Signal Definition")
        cell_signal_layout = QtWidgets.QVBoxLayout(self.cell_signal_group)
        cell_signal_layout.setSpacing(4)
        cell_signal_layout.setContentsMargins(8, 8, 8, 8)

        cell_signal_method_layout = QtWidgets.QHBoxLayout()
        cell_signal_method_layout.addWidget(QtWidgets.QLabel("Signal definition:"))
        self.cell_signal_method_combo = QtWidgets.QComboBox()
        for label, value in CELL_SIGNAL_METHOD_OPTIONS:
            self.cell_signal_method_combo.addItem(label, value)
        self.cell_signal_method_combo.currentIndexChanged.connect(self._on_cell_signal_method_changed)
        cell_signal_method_layout.addWidget(self.cell_signal_method_combo, 1)
        cell_signal_method_layout.addStretch()
        cell_signal_layout.addLayout(cell_signal_method_layout)

        threshold_layout = QtWidgets.QHBoxLayout()
        self.positive_threshold_sd_label = QtWidgets.QLabel("Threshold SD:")
        self.positive_threshold_sd_spin = QtWidgets.QDoubleSpinBox()
        self.positive_threshold_sd_spin.setRange(0.5, 5.0)
        self.positive_threshold_sd_spin.setSingleStep(0.5)
        self.positive_threshold_sd_spin.setDecimals(1)
        self.positive_threshold_sd_spin.setValue(2.0)
        threshold_layout.addWidget(self.positive_threshold_sd_label)
        threshold_layout.addWidget(self.positive_threshold_sd_spin)
        threshold_layout.addStretch()
        cell_signal_layout.addLayout(threshold_layout)

        quantile_layout = QtWidgets.QHBoxLayout()
        self.upper_quantile_label = QtWidgets.QLabel("Upper quantile:")
        self.upper_quantile_spin = QtWidgets.QDoubleSpinBox()
        self.upper_quantile_spin.setRange(50.0, 99.9)
        self.upper_quantile_spin.setSingleStep(1.0)
        self.upper_quantile_spin.setDecimals(1)
        self.upper_quantile_spin.setValue(90.0)
        self.upper_quantile_spin.setSuffix("%")
        quantile_layout.addWidget(self.upper_quantile_label)
        quantile_layout.addWidget(self.upper_quantile_spin)
        quantile_layout.addStretch()
        cell_signal_layout.addLayout(quantile_layout)

        self.cell_signal_help_label = QtWidgets.QLabel("")
        self.cell_signal_help_label.setWordWrap(True)
        self.cell_signal_help_label.setStyleSheet("QLabel { color: #555; font-size: 9pt; }")
        cell_signal_layout.addWidget(self.cell_signal_help_label)
        options_layout.addWidget(self.cell_signal_group)

        denoise_group = QtWidgets.QGroupBox("Pre-QC Denoising")
        denoise_layout = QtWidgets.QVBoxLayout(denoise_group)
        denoise_layout.setSpacing(4)
        denoise_layout.setContentsMargins(8, 8, 8, 8)

        denoise_source_layout = QtWidgets.QHBoxLayout()
        denoise_source_layout.addWidget(QtWidgets.QLabel("Denoising:"))
        self.denoise_source_combo = QtWidgets.QComboBox()
        self.denoise_source_combo.addItems(["None", "Viewer", "Custom"])
        self.denoise_source_combo.currentTextChanged.connect(self._on_denoise_source_changed)
        denoise_source_layout.addWidget(self.denoise_source_combo)
        denoise_source_layout.addStretch()
        denoise_layout.addLayout(denoise_source_layout)

        self.custom_denoise_frame = QtWidgets.QFrame()
        self.custom_denoise_frame.setFrameStyle(QtWidgets.QFrame.Box)
        self.custom_denoise_frame.setVisible(False)
        custom_denoise_layout = QtWidgets.QVBoxLayout(self.custom_denoise_frame)
        custom_denoise_layout.setSpacing(4)
        custom_denoise_layout.setContentsMargins(8, 8, 8, 8)

        denoise_channel_row = QtWidgets.QHBoxLayout()
        denoise_channel_row.addWidget(QtWidgets.QLabel("Channel:"))
        self.denoise_channel_combo = QtWidgets.QComboBox()
        self.denoise_channel_combo.currentTextChanged.connect(self._on_denoise_channel_changed)
        denoise_channel_row.addWidget(self.denoise_channel_combo, 1)
        custom_denoise_layout.addLayout(denoise_channel_row)

        denoise_controls_layout = QtWidgets.QHBoxLayout()

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

        self.apply_all_channels_btn = QtWidgets.QPushButton("Apply to All Channels")
        self.apply_all_channels_btn.clicked.connect(self._apply_denoise_to_all_channels)
        custom_denoise_layout.addWidget(self.apply_all_channels_btn)

        if not _HAVE_SCIKIT_IMAGE:
            self.custom_denoise_frame.setEnabled(False)
            custom_denoise_layout.addWidget(
                QtWidgets.QLabel("scikit-image not available; install to enable custom denoising.")
            )

        denoise_layout.addWidget(self.custom_denoise_frame)
        options_layout.addWidget(denoise_group)

        self._settings_options_group = options_group
        self._build_settings_dialog()

        controls_row = QtWidgets.QHBoxLayout()
        self.qc_settings_btn = QtWidgets.QPushButton("QC Settings...")
        self.qc_settings_btn.setToolTip(
            "Open acquisition, analysis mode, cell signal, denoising, and worker settings"
        )
        self.qc_settings_btn.clicked.connect(self._open_settings_dialog)
        controls_row.addWidget(self.qc_settings_btn)

        self.settings_summary_label = QtWidgets.QLabel("")
        self.settings_summary_label.setStyleSheet("QLabel { color: #666; }")
        controls_row.addWidget(self.settings_summary_label, 1)

        self.run_btn = QtWidgets.QPushButton("Calculate QC Metrics")
        self.run_btn.clicked.connect(self._run_analysis)
        controls_row.addWidget(self.run_btn)
        layout.addLayout(controls_row)
        
        # Results tabs
        self.tabs = QtWidgets.QTabWidget()
        
        # Summary tab
        summary_tab = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_tab)

        self.summary_method_label = QtWidgets.QLabel("")
        self.summary_method_label.setWordWrap(True)
        self.summary_method_label.setStyleSheet("QLabel { color: #555; font-size: 9pt; padding-bottom: 4px; }")
        self.summary_method_label.hide()
        summary_layout.addWidget(self.summary_method_label)
        
        # Summary table
        self.summary_table = QtWidgets.QTableWidget()
        self.summary_table.setColumnCount(0)
        self.summary_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.summary_table)
        
        # Export button
        summary_btn_layout = QtWidgets.QHBoxLayout()
        self.export_summary_btn = QtWidgets.QPushButton("Export Results...")
        self.export_summary_btn.clicked.connect(self._export_results)
        self.export_summary_btn.setEnabled(False)
        summary_btn_layout.addWidget(self.export_summary_btn)
        summary_btn_layout.addStretch()
        summary_layout.addLayout(summary_btn_layout)
        
        self.tabs.addTab(summary_tab, "Summary")
        
        # SNR vs Intensity plot
        snr_intensity_tab = QtWidgets.QWidget()
        snr_intensity_layout = QtWidgets.QVBoxLayout(snr_intensity_tab)
        
        self.snr_intensity_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        snr_intensity_layout.addWidget(self.snr_intensity_canvas)
        
        snr_intensity_btn_layout = QtWidgets.QHBoxLayout()
        self.snr_intensity_save_btn = QtWidgets.QPushButton("Save Plot...")
        self.snr_intensity_save_btn.clicked.connect(self._save_snr_intensity_plot)
        self.snr_intensity_save_btn.setEnabled(False)
        snr_intensity_btn_layout.addWidget(self.snr_intensity_save_btn)
        snr_intensity_btn_layout.addStretch()
        snr_intensity_layout.addLayout(snr_intensity_btn_layout)
        
        self.tabs.addTab(snr_intensity_tab, "SNR vs Intensity")
        
        # Coverage tab
        coverage_tab = QtWidgets.QWidget()
        coverage_layout = QtWidgets.QVBoxLayout(coverage_tab)
        
        self.coverage_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        coverage_layout.addWidget(self.coverage_canvas)
        
        coverage_btn_layout = QtWidgets.QHBoxLayout()
        self.coverage_save_btn = QtWidgets.QPushButton("Save Plot...")
        self.coverage_save_btn.clicked.connect(self._save_coverage_plot)
        self.coverage_save_btn.setEnabled(False)
        coverage_btn_layout.addWidget(self.coverage_save_btn)
        coverage_btn_layout.addStretch()
        coverage_layout.addLayout(coverage_btn_layout)
        
        self.tabs.addTab(coverage_tab, "Coverage & Density")
        
        # Distribution tab (boxplots)
        distribution_tab = QtWidgets.QWidget()
        distribution_layout = QtWidgets.QVBoxLayout(distribution_tab)
        
        self.distribution_canvas = FigureCanvas(Figure(figsize=(12, 8)))
        distribution_layout.addWidget(self.distribution_canvas)
        
        distribution_btn_layout = QtWidgets.QHBoxLayout()
        self.distribution_save_btn = QtWidgets.QPushButton("Save Plot...")
        self.distribution_save_btn.clicked.connect(self._save_distribution_plot)
        self.distribution_save_btn.setEnabled(False)
        distribution_btn_layout.addWidget(self.distribution_save_btn)
        distribution_btn_layout.addStretch()
        distribution_layout.addLayout(distribution_btn_layout)
        
        self.tabs.addTab(distribution_tab, "Distributions")
        
        layout.addWidget(self.tabs, 1)
        
        # Initialize
        self._on_mode_changed()
        self._populate_denoise_channel_list()
        self._on_denoise_source_changed()
        self._update_cell_signal_controls()
        self._connect_settings_summary_signals()
        self._update_settings_summary()

    def _build_settings_dialog(self):
        """Create the popup dialog that hosts QC settings controls."""
        self.qc_settings_dialog = QtWidgets.QDialog(self)
        self.qc_settings_dialog.setWindowTitle("QC Analysis Settings")
        self.qc_settings_dialog.setModal(True)
        self.qc_settings_dialog.setMinimumSize(580, 460)

        parent_size = self.size()
        if parent_size.width() > 0 and parent_size.height() > 0:
            dialog_width = max(580, int(parent_size.width() * 0.6))
            dialog_height = max(460, int(parent_size.height() * 0.75))
            self.qc_settings_dialog.resize(dialog_width, dialog_height)

        dialog_layout = QtWidgets.QVBoxLayout(self.qc_settings_dialog)
        help_label = QtWidgets.QLabel(
            "Configure QC analysis options here. Then click 'Calculate QC Metrics' in the main window "
            "to refresh the tables and plots."
        )
        help_label.setWordWrap(True)
        dialog_layout.addWidget(help_label)

        self.qc_settings_scroll_area = QtWidgets.QScrollArea()
        self.qc_settings_scroll_area.setWidgetResizable(True)
        self.qc_settings_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.qc_settings_scroll_area.setWidget(self._settings_options_group)
        dialog_layout.addWidget(self.qc_settings_scroll_area, 1)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch()
        done_btn = QtWidgets.QPushButton("Done")
        done_btn.clicked.connect(self.qc_settings_dialog.accept)
        buttons_layout.addWidget(done_btn)
        dialog_layout.addLayout(buttons_layout)

    def _open_settings_dialog(self):
        """Open the popup dialog containing QC analysis settings."""
        if not hasattr(self, 'qc_settings_dialog') or self.qc_settings_dialog is None:
            return
        self._on_denoise_source_changed()
        self._update_cell_signal_controls()
        self.qc_settings_dialog.exec_()
        self._update_settings_summary()

    def _connect_settings_summary_signals(self):
        """Keep the compact settings summary synchronized with control changes."""
        signal_bindings = [
            (self.mode_combo.currentIndexChanged, self._update_settings_summary),
            (self.acq_combo.currentIndexChanged, self._update_settings_summary),
            (self.workers_spin.valueChanged, self._update_settings_summary),
            (self.snr_threshold_spin.valueChanged, self._update_settings_summary),
            (self.cell_signal_method_combo.currentIndexChanged, self._update_settings_summary),
            (self.positive_threshold_sd_spin.valueChanged, self._update_settings_summary),
            (self.upper_quantile_spin.valueChanged, self._update_settings_summary),
            (self.denoise_source_combo.currentTextChanged, self._update_settings_summary),
        ]
        for signal, slot in signal_bindings:
            signal.connect(slot)
        self.snr_threshold_spin.valueChanged.connect(self._refresh_existing_qc_plots)

    def _update_settings_summary(self):
        """Update the compact summary shown next to the settings button."""
        if not hasattr(self, 'settings_summary_label'):
            return

        acq_text = self.acq_combo.currentText() if hasattr(self, 'acq_combo') else ""
        mode_text = self.mode_combo.currentText() if hasattr(self, 'mode_combo') else ""
        workers = self.workers_spin.value() if hasattr(self, 'workers_spin') else None
        snr_threshold = self.get_snr_threshold()
        denoise_text = self.denoise_source_combo.currentText() if hasattr(self, 'denoise_source_combo') else "None"

        parts = [part for part in [acq_text, mode_text] if part]
        parts.append(f"SNR threshold: {snr_threshold:.1f}")

        if self.analysis_mode == "cell":
            method = self.get_cell_signal_method()
            if method == "positive_pixels":
                signal_text = f"Signal: Positive pixels (+{self.positive_threshold_sd_spin.value():.1f} SD)"
            elif method == "upper_quantile":
                signal_text = f"Signal: Top {self.upper_quantile_spin.value():.1f}% cells"
            else:
                signal_text = "Signal: All cell pixels"
            parts.append(signal_text)

        parts.append(f"Denoising: {denoise_text}")
        if workers is not None:
            worker_label = "worker" if workers == 1 else "workers"
            parts.append(f"{workers} {worker_label}")

        summary = " | ".join(parts)
        self.settings_summary_label.setText(summary)
        self.settings_summary_label.setToolTip(summary)
    
    def _check_masks_exist(self) -> bool:
        """Check if any segmentation masks exist."""
        if not self.parent_window or not hasattr(self.parent_window, 'segmentation_masks'):
            return False
        return len(self.parent_window.segmentation_masks) > 0
    
    def _populate_acq_combo(self):
        """Populate acquisition combo box."""
        self.acq_combo.clear()
        if not self.parent_window or not hasattr(self.parent_window, 'acquisitions'):
            return
        
        for acq in self.parent_window.acquisitions:
            # Use same format as main window: well [file_name] or name [file_name]
            import os
            file_name = os.path.basename(acq.source_file) if hasattr(acq, 'source_file') and acq.source_file else "Unknown"
            label = acq.well if acq.well else acq.name
            label += f" [{file_name}]"
            self.acq_combo.addItem(label, acq.id)
            item_index = self.acq_combo.count() - 1
            self.acq_combo.setItemData(
                item_index,
                f"Source data file: {file_name}\nFor MCD workflows, this is the source .mcd file.",
                QtCore.Qt.ToolTipRole,
            )

    def _populate_denoise_channel_list(self):
        """Populate the denoise channel combo with channels from the selected acquisition."""
        if not hasattr(self, 'denoise_channel_combo'):
            return

        channels = []
        if self.parent_window and hasattr(self.parent_window, 'acquisitions'):
            selected_acq_id = self.acq_combo.currentData() if hasattr(self, 'acq_combo') else None
            target_acq = None
            if selected_acq_id and selected_acq_id != "all":
                target_acq = self.parent_window._get_acquisition_info(selected_acq_id)
            elif self.parent_window.acquisitions:
                target_acq = self.parent_window.acquisitions[0]

            if target_acq is not None:
                channels = list(target_acq.channels or [])

        self.denoise_channel_combo.blockSignals(True)
        self.denoise_channel_combo.clear()
        for channel in channels:
            self.denoise_channel_combo.addItem(channel)
        self.denoise_channel_combo.blockSignals(False)
        if channels:
            self.denoise_channel_combo.setCurrentIndex(0)
            self._load_denoise_settings()

    def _on_denoise_source_changed(self):
        """Handle changes to the denoise source selection."""
        if hasattr(self, 'custom_denoise_frame'):
            self.custom_denoise_frame.setVisible(self.denoise_source_combo.currentText() == "Custom")
        self._update_settings_summary()

    def _on_denoise_channel_changed(self):
        """Persist current custom settings before switching denoise channels."""
        self._save_current_denoise_settings()
        self._load_denoise_settings()

    def _load_denoise_settings(self):
        """Load saved custom denoise settings for the current channel into the UI."""
        if not hasattr(self, 'denoise_channel_combo'):
            return
        channel = self.denoise_channel_combo.currentText()
        if not channel:
            return

        cfg = self.custom_denoise_settings.get(channel, {})
        hot = cfg.get("hot")
        speckle = cfg.get("speckle")
        bg = cfg.get("background")

        controls = [
            self.hot_pixel_chk,
            self.hot_pixel_method_combo,
            self.hot_pixel_n_spin,
            self.speckle_chk,
            self.speckle_method_combo,
            self.gaussian_sigma_spin,
            self.bg_subtract_chk,
            self.bg_method_combo,
            self.bg_radius_spin,
        ]
        for control in controls:
            control.blockSignals(True)

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
                bg_method = bg.get("method")
                if bg_method == "white_tophat":
                    self.bg_method_combo.setCurrentIndex(0)
                elif bg_method == "black_tophat":
                    self.bg_method_combo.setCurrentIndex(1)
                else:
                    self.bg_method_combo.setCurrentIndex(2)
                self.bg_radius_spin.setValue(int(bg.get("radius", 15)))
            else:
                self.bg_subtract_chk.setChecked(False)
                self.bg_method_combo.setCurrentIndex(0)
                self.bg_radius_spin.setValue(15)
        finally:
            for control in controls:
                control.blockSignals(False)

        self._sync_hot_controls_visibility()

    def _save_current_denoise_settings(self):
        """Save the currently displayed denoise settings for the selected channel."""
        if not hasattr(self, 'denoise_channel_combo'):
            return
        channel = self.denoise_channel_combo.currentText()
        if not channel:
            return

        cfg = {}
        if self.hot_pixel_chk.isChecked():
            cfg["hot"] = {
                "method": "median3" if self.hot_pixel_method_combo.currentIndex() == 0 else "n_sd_local_median",
                "n_sd": float(self.hot_pixel_n_spin.value()),
            }
        if self.speckle_chk.isChecked():
            cfg["speckle"] = {
                "method": "gaussian" if self.speckle_method_combo.currentIndex() == 0 else "nl_means",
                "sigma": float(self.gaussian_sigma_spin.value()),
            }
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

        self.custom_denoise_settings[channel] = cfg

    def _apply_denoise_to_all_channels(self):
        """Apply the current denoise settings to every available channel."""
        channels = [self.denoise_channel_combo.itemText(i) for i in range(self.denoise_channel_combo.count())]
        if not channels:
            return

        self._save_current_denoise_settings()
        current_channel = self.denoise_channel_combo.currentText()
        current_cfg = dict(self.custom_denoise_settings.get(current_channel, {}))
        for channel in channels:
            self.custom_denoise_settings[channel] = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in current_cfg.items()
            }

        self.apply_all_channels_btn.setText("✓ Applied to All Channels")
        self.apply_all_channels_btn.setStyleSheet(
            "QPushButton { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }"
        )
        QtCore.QTimer.singleShot(2000, self._reset_apply_all_button)

    def _reset_apply_all_button(self):
        """Restore the default appearance of the apply-all button."""
        self.apply_all_channels_btn.setText("Apply to All Channels")
        self.apply_all_channels_btn.setStyleSheet("")

    def _sync_hot_controls_visibility(self):
        """Only show the N threshold control for threshold-based hot-pixel removal."""
        is_threshold = self.hot_pixel_method_combo.currentIndex() == 1
        self.hot_pixel_n_spin.setVisible(is_threshold)
        self.hot_pixel_n_label.setVisible(is_threshold)

    def get_denoise_source(self) -> str:
        """Return the selected denoising source for QC."""
        source = self.denoise_source_combo.currentText()
        if source == "Viewer":
            return "viewer"
        if source == "Custom":
            return "custom"
        return "none"

    def get_custom_denoise_settings(self):
        """Return custom per-channel denoise settings."""
        self._save_current_denoise_settings()
        return self.custom_denoise_settings

    def _resolve_qc_denoise_settings(self) -> Optional[Dict[str, Dict[str, dict]]]:
        """Resolve the effective denoise settings used for QC computation."""
        if not self.parent_window:
            return None

        denoise_source = self.get_denoise_source()
        if denoise_source == "viewer":
            if hasattr(self.parent_window, '_get_relevant_denoise_settings'):
                try:
                    return self.parent_window._get_relevant_denoise_settings("viewer")
                except Exception:
                    pass
            return dict(getattr(self.parent_window, 'channel_denoise', {}) or {})

        if denoise_source == "custom":
            return dict(self.get_custom_denoise_settings() or {})

        return None

    def _on_cell_signal_method_changed(self):
        """Update cell signal controls when the selected method changes."""
        self._update_cell_signal_controls()
        self._update_settings_summary()

    def get_cell_signal_method(self) -> str:
        """Return the selected cell signal method identifier."""
        if not hasattr(self, 'cell_signal_method_combo'):
            return "positive_pixels"
        method = self.cell_signal_method_combo.currentData()
        return method if isinstance(method, str) and method else "positive_pixels"

    def get_snr_threshold(self) -> float:
        """Return the user-configured SNR reference threshold for QC plots."""
        if not hasattr(self, 'snr_threshold_spin'):
            return DEFAULT_QC_SNR_THRESHOLD
        return float(self.snr_threshold_spin.value())

    def _get_cell_signal_settings(self) -> Dict[str, Any]:
        """Return the current cell signal settings in core.qc_analysis format."""
        return {
            "cell_signal_method": self.get_cell_signal_method(),
            "positive_threshold_sd": float(self.positive_threshold_sd_spin.value()),
            "upper_quantile": float(self.upper_quantile_spin.value()) / 100.0,
        }

    def _get_current_cache_signature(self) -> str:
        """Return the QC cache signature for the current analysis settings."""
        if self.analysis_mode != "cell":
            return "pixel"
        settings = self._get_cell_signal_settings()
        return (
            f"cell_{settings['cell_signal_method']}"
            f"_sd{settings['positive_threshold_sd']:.1f}"
            f"_q{settings['upper_quantile']:.3f}"
        )

    def _get_qc_cache_key(self) -> Optional[str]:
        """Return the cache key for the current QC configuration."""
        if not self.file_set_id:
            return None
        return f"{self.file_set_id}_{self._get_current_cache_signature()}"

    def _cache_matches_current_settings(self, cache_data: Dict[str, Any]) -> bool:
        """Check whether cached QC results match the current signal definition settings."""
        if not isinstance(cache_data, dict):
            return False
        if cache_data.get('analysis_mode') != self.analysis_mode:
            return False
        if self.analysis_mode != "cell":
            return True
        settings = self._get_cell_signal_settings()
        cached_method = cache_data.get('cell_signal_method')
        cached_threshold = cache_data.get('positive_threshold_sd')
        cached_quantile = cache_data.get('upper_quantile')
        if cached_method is None or cached_threshold is None or cached_quantile is None:
            return False
        return (
            cached_method == settings['cell_signal_method']
            and np.isclose(float(cached_threshold), settings['positive_threshold_sd'])
            and np.isclose(float(cached_quantile), settings['upper_quantile'])
        )

    def _get_results_cell_signal_method(self) -> Optional[str]:
        """Return the method used to generate the currently displayed cell-mode results."""
        if self.analysis_mode != "cell":
            return None
        if self.qc_results is not None and 'cell_signal_method' in self.qc_results.columns:
            methods = [
                method
                for method in self.qc_results['cell_signal_method'].dropna().unique().tolist()
                if isinstance(method, str) and method
            ]
            if len(methods) == 1:
                return methods[0]
        return self.get_cell_signal_method()

    def _get_qc_method_subtitle(self) -> str:
        """Return a human-readable subtitle for the current QC signal definition."""
        method = self._get_results_cell_signal_method()
        if not method:
            return ""
        label = _cell_signal_method_label(method)
        if method == "positive_pixels":
            return f"Cell signal: {label} (> background + {self.positive_threshold_sd_spin.value():.1f} robust SD)"
        if method == "upper_quantile":
            return f"Cell signal: {label} (top {self.upper_quantile_spin.value():.1f}%)"
        return f"Cell signal: {label}"

    def _compose_plot_title(self, base_title: str) -> str:
        """Append the QC cell-signal subtitle to plot titles when relevant."""
        subtitle = self._get_qc_method_subtitle()
        return f"{base_title}\n{subtitle}" if subtitle else base_title

    def _get_plot_intensity_column(self) -> str:
        """Return the best available intensity column used in QC plots."""
        candidate_frames = [
            getattr(self, 'qc_results_aggregated', None),
            getattr(self, 'qc_results', None),
        ]
        for frame in candidate_frames:
            if (
                frame is not None
                and not frame.empty
                and 'signal_mean' in frame.columns
                and frame['signal_mean'].notna().any()
            ):
                return 'signal_mean'
        for frame in candidate_frames:
            if (
                frame is not None
                and not frame.empty
                and 'mean_intensity' in frame.columns
                and frame['mean_intensity'].notna().any()
            ):
                return 'mean_intensity'
        return 'signal_mean'

    def _get_plot_intensity_label(self) -> str:
        """Return a user-facing description of the foreground intensity used in plots."""
        intensity_col = self._get_plot_intensity_column()
        if intensity_col == 'signal_mean' and self.analysis_mode == "cell":
            return 'Foreground Mean Intensity (selected in-cell signal only, log scale)'
        if intensity_col == 'signal_mean':
            return 'Foreground Mean Intensity (Otsu-positive pixels only, log scale)'
        if self.analysis_mode == "cell":
            return 'Mean Intensity (legacy whole-cell average, log scale)'
        return 'Mean Intensity (log scale)'

    def _refresh_existing_qc_plots(self):
        """Refresh plot-only controls without rerunning QC analysis."""
        if self.qc_results_aggregated is not None and not self.qc_results_aggregated.empty:
            self._plot_snr_vs_intensity()
        if self.qc_results is not None and not self.qc_results.empty:
            self._plot_distributions()

    def _get_export_suffix(self) -> str:
        """Return a concise suffix describing the current cell-signal settings."""
        method = self._get_results_cell_signal_method()
        if not method:
            return ""
        if method == "positive_pixels":
            suffix = f"{method}_sd{self.positive_threshold_sd_spin.value():.1f}"
        elif method == "upper_quantile":
            suffix = f"{method}_q{self.upper_quantile_spin.value():.1f}"
        else:
            suffix = method
        return suffix.replace('.', 'p')

    def _update_cell_signal_controls(self):
        """Show the relevant cell signal controls and explanatory text."""
        is_cell_mode = self.analysis_mode == "cell"
        self.cell_signal_group.setVisible(is_cell_mode)
        method = self.get_cell_signal_method()
        show_threshold = is_cell_mode and method == "positive_pixels"
        show_quantile = is_cell_mode and method == "upper_quantile"
        self.positive_threshold_sd_label.setVisible(show_threshold)
        self.positive_threshold_sd_spin.setVisible(show_threshold)
        self.upper_quantile_label.setVisible(show_quantile)
        self.upper_quantile_spin.setVisible(show_quantile)
        if not is_cell_mode:
            self.cell_signal_help_label.clear()
            return
        if method == "positive_pixels":
            self.cell_signal_help_label.setText(
                "Estimate signal from in-cell pixels above background_mean + N times the robust background SD. "
                "This usually works best for sparse markers with many negative cells."
            )
        elif method == "upper_quantile":
            self.cell_signal_help_label.setText(
                "Estimate signal from the brightest cells only, using the upper tail of per-cell mean intensities. "
                "This is robust when signal is confined to a subset of cells."
            )
        else:
            self.cell_signal_help_label.setText(
                "Use all pixels inside cells as signal. This preserves the legacy behavior but can dilute sparse markers."
            )
        
    def _on_mode_changed(self):
        """Handle mode change."""
        mode_idx = self.mode_combo.currentIndex()
        if mode_idx == 0:
            self.analysis_mode = "pixel"
        else:
            self.analysis_mode = "cell"
            # Check if masks exist for selected acquisition
            if not self._check_masks_exist():
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Masks Available",
                    "Cell-level analysis requires segmentation masks. Please segment cells first."
                )
        
        self._update_cell_signal_controls()
        # Try to restore cached results for the new mode
        self._restore_cached_results()
        self._update_settings_summary()
    
    
    def _run_analysis(self):
        """Run QC analysis with responsive progress handling."""
        if not self.parent_window:
            QtWidgets.QMessageBox.warning(self, "Error", "Parent window not available.")
            return
        
        acq_id = self.acq_combo.currentData()
        if not acq_id:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select an acquisition.")
            return
        
        # Determine which acquisitions to process
        if acq_id == "all":
            acquisitions = self.parent_window.acquisitions
        else:
            acq_info = self.parent_window._get_acquisition_info(acq_id)
            if not acq_info:
                QtWidgets.QMessageBox.warning(self, "Error", f"Could not find acquisition {acq_id}.")
                return
            acquisitions = [acq_info]
        
        # For cell-level analysis, filter out acquisitions without masks
        if self.analysis_mode == "cell":
            acquisitions_with_masks = []
            acquisitions_without_masks = []
            for acq in acquisitions:
                if acq.id in self.parent_window.segmentation_masks:
                    acquisitions_with_masks.append(acq)
                else:
                    acquisitions_without_masks.append(acq)
            
            if not acquisitions_with_masks:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Masks Available",
                    "No segmentation masks found for the selected acquisition(s).\n"
                    "Please segment cells first before running cell-level QC analysis."
                )
                return
            
            if acquisitions_without_masks:
                # Show informative message about which acquisitions will be skipped
                skipped_names = [acq.well if acq.well else acq.name for acq in acquisitions_without_masks]
                if len(skipped_names) == 1:
                    skipped_msg = f"Acquisition '{skipped_names[0]}' will be skipped (no mask available)."
                else:
                    skipped_msg = f"{len(skipped_names)} acquisitions will be skipped (no masks available): {', '.join(skipped_names[:5])}"
                    if len(skipped_names) > 5:
                        skipped_msg += f" and {len(skipped_names) - 5} more"
                
                QtWidgets.QMessageBox.information(
                    self,
                    "Cell-Level QC Analysis",
                    f"Cell-level QC analysis will only process acquisitions with segmentation masks.\n\n"
                    f"{skipped_msg}\n\n"
                    f"Processing {len(acquisitions_with_masks)} acquisition(s) with masks."
                )
            
            # Use only acquisitions with masks
            acquisitions = acquisitions_with_masks
        
        # Get number of workers
        num_workers = self.workers_spin.value()
        denoise_settings = self._resolve_qc_denoise_settings()
        cell_signal_settings = self._get_cell_signal_settings() if self.analysis_mode == "cell" else None
        temp_mask_paths = []

        try:
            # Prepare tasks for multiprocessing - one task per acquisition
            tasks = []
            acq_to_file_map = {}  # Map acquisition ID to source file path
            
            for acq_info in acquisitions:
                acq_id = acq_info.id
                
                # Get original acquisition ID for multiple MCD files
                original_acq_id = acq_id
                if (hasattr(self.parent_window, 'unique_acq_to_original') and 
                    acq_id in self.parent_window.unique_acq_to_original):
                    original_acq_id = self.parent_window.unique_acq_to_original[acq_id]
                else:
                    pass
                
                # Get channels
                channels = acq_info.channels
                if not channels:
                    continue
                
                # Get source file path for loader and determine loader type
                source_file = None
                loader_type = None
                if hasattr(acq_info, 'source_file') and acq_info.source_file:
                    source_file = acq_info.source_file
                    # Determine loader type from file extension or path
                    if source_file.lower().endswith(('.mcd', '.mcdx')):
                        loader_type = "mcd"
                    elif os.path.isdir(source_file):
                        loader_type = "ometiff"
                    else:
                        loader_type = "ometiff"  # Assume OME-TIFF for other file types
                elif acq_id in self.parent_window.acq_to_file:
                    source_file = self.parent_window.acq_to_file[acq_id]
                    loader_type = "mcd"  # acq_to_file typically contains MCD files
                else:
                    # Try to get from current_path
                    if hasattr(self.parent_window, 'current_path') and self.parent_window.current_path:
                        if self.parent_window.current_path.endswith(('.mcd', '.mcdx')):
                            source_file = self.parent_window.current_path
                            loader_type = "mcd"
                        elif os.path.isdir(self.parent_window.current_path):
                            source_file = self.parent_window.current_path
                            loader_type = "ometiff"
                
                if not source_file:
                    continue
                
                if not os.path.exists(source_file):
                    continue
                
                acq_to_file_map[acq_id] = source_file
                
                # Get mask path if cell-level analysis
                mask_path = None
                if self.analysis_mode == "cell":
                    mask = self.parent_window.segmentation_masks.get(acq_id)
                    if mask is None:
                        continue
                    # Save mask temporarily to temp location
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    mask_path = os.path.join(temp_dir, f"qc_mask_{acq_id}.tif")
                    try:
                        import tifffile
                        tifffile.imwrite(mask_path, mask.astype(np.uint32))
                        temp_mask_paths.append(mask_path)
                    except Exception as e:
                        continue
                
                # Create one task per acquisition (all channels)
                # Pass both unique ID (for results) and original ID (for loader)
                task = (
                    acq_id,
                    original_acq_id,
                    acq_info.name,
                    channels,
                    self.analysis_mode,
                    mask_path,
                    source_file,
                    loader_type,
                    acq_to_file_map,
                    denoise_settings,
                    cell_signal_settings,
                )
                tasks.append(task)
            
            if not tasks:
                QtWidgets.QMessageBox.warning(self, "Error", "No tasks to process.")
                return

            def _compute_results():
                return self._execute_qc_tasks(tasks, num_workers)

            def _finalize_results(results, _progress):
                if not results:
                    QtWidgets.QMessageBox.warning(self, "Error", "No results generated. Check console for errors.")
                    return

                self.qc_results = pd.DataFrame(results)
                column_mapping = {
                    'intensity_mean': 'mean_intensity',
                    'intensity_std': 'std_intensity',
                    'intensity_median': 'median_intensity',
                    'intensity_min': 'min_intensity',
                    'intensity_max': 'max_intensity',
                    'coverage': 'coverage_pct'
                }
                existing_mappings = {k: v for k, v in column_mapping.items() if k in self.qc_results.columns}
                if existing_mappings:
                    self.qc_results = self.qc_results.rename(columns=existing_mappings)

                if 'coverage_pct' in self.qc_results.columns and not self.qc_results['coverage_pct'].empty:
                    max_coverage = self.qc_results['coverage_pct'].max()
                    if max_coverage <= 1.0:
                        self.qc_results['coverage_pct'] = self.qc_results['coverage_pct'] * 100.0

                self._aggregate_results_by_channel()
                self._save_results_to_cache()
                self._update_summary_table()
                self._update_plots()
                self.export_summary_btn.setEnabled(True)
                self.snr_intensity_save_btn.setEnabled(True)
                self.coverage_save_btn.setEnabled(True)
                self.distribution_save_btn.setEnabled(True)

                logger = get_logger()
                source_files = set()
                for _, source_file in acq_to_file_map.items():
                    if not source_file:
                        continue
                    if source_file.endswith('.mcd') or source_file.endswith('.mcdx'):
                        source_files.add(os.path.basename(source_file))
                    else:
                        folder_path = source_file if os.path.isdir(source_file) else (os.path.dirname(source_file) or source_file)
                        source_files.add(os.path.basename(folder_path))

                source_file_str = None
                if source_files:
                    sorted_files = sorted(source_files)
                    if len(sorted_files) == 1:
                        source_file_str = sorted_files[0]
                    elif len(sorted_files) <= 3:
                        source_file_str = ", ".join(sorted_files)
                    else:
                        source_file_str = ", ".join(sorted_files[:3]) + f" and {len(sorted_files) - 3} more"

                params = {
                    "analysis_mode": self.analysis_mode,
                    "n_acquisitions": len(acquisitions),
                    "n_channels": len(results),
                    "denoise_source": self.get_denoise_source(),
                }
                if denoise_settings:
                    params["denoise_settings"] = denoise_settings
                if cell_signal_settings:
                    params.update(cell_signal_settings)

                logger._write_entry(
                    entry_type="qc_analysis",
                    operation="signal_to_noise_ratio",
                    parameters=params,
                    acquisitions=[acq.id for acq in acquisitions],
                    notes=f"QC analysis (SNR) completed: {len(results)} channels across {len(acquisitions)} acquisitions",
                    source_file=source_file_str
                )

            run_blocking_task_with_progress_then_finalize(
                parent=self,
                window_title="Calculating QC Metrics",
                initial_message="Calculating QC metrics",
                detail_text="Processing acquisitions and channels in the background.",
                task=_compute_results,
                finalize=_finalize_results,
                finishing_message="Rendering QC results",
                finishing_detail_text="Updating tables and plots.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            for mask_path in temp_mask_paths:
                if mask_path and os.path.exists(mask_path):
                    try:
                        os.remove(mask_path)
                    except Exception:
                        pass

    def _execute_qc_tasks(self, tasks, num_workers: int):
        """Execute QC worker tasks without blocking the UI thread."""
        from collections import defaultdict

        file_groups = defaultdict(list)
        ometiff_tasks = []
        for task in tasks:
            source_file = task[6]
            loader_type = task[7]
            if loader_type == "mcd":
                file_groups[source_file].append(task)
            else:
                ometiff_tasks.append(task)

        results = []
        total_tasks = len(tasks)
        worker_timeout = 300
        n_workers = max(1, min(int(num_workers), total_tasks))

        if n_workers > 1 and total_tasks > 1:
            ctx = mp.get_context('spawn')
            try:
                for _, file_tasks in file_groups.items():
                    with ctx.Pool(processes=min(n_workers, len(file_tasks))) as pool:
                        futures = [pool.apply_async(_qc_process_acquisition_worker, (task,)) for task in file_tasks]
                        for future in futures:
                            try:
                                acquisition_results = future.get(timeout=worker_timeout)
                                if acquisition_results:
                                    results.extend(acquisition_results)
                            except mp.TimeoutError:
                                print(f"[QC] [ERROR] QC analysis timed out after {worker_timeout}s")
                            except Exception as exc:
                                print(f"[QC] [ERROR] QC analysis failed: {exc}")

                if ometiff_tasks:
                    with ctx.Pool(processes=min(n_workers, len(ometiff_tasks))) as pool:
                        futures = [pool.apply_async(_qc_process_acquisition_worker, (task,)) for task in ometiff_tasks]
                        for future in futures:
                            try:
                                acquisition_results = future.get(timeout=worker_timeout)
                                if acquisition_results:
                                    results.extend(acquisition_results)
                            except mp.TimeoutError:
                                print(f"[QC] [ERROR] QC analysis timed out after {worker_timeout}s")
                            except Exception as exc:
                                print(f"[QC] [ERROR] QC analysis failed: {exc}")
            except Exception as mp_error:
                print(f"[QC] Multiprocessing failed, falling back to sequential processing: {mp_error}")
                traceback.print_exc()
                results = []
                for task in tasks:
                    try:
                        acquisition_results = _qc_process_acquisition_worker(task)
                        if acquisition_results:
                            results.extend(acquisition_results)
                    except Exception as exc:
                        print(f"[QC] [ERROR] Sequential QC analysis failed: {exc}")
        else:
            for task in tasks:
                try:
                    acquisition_results = _qc_process_acquisition_worker(task)
                    if acquisition_results:
                        results.extend(acquisition_results)
                except Exception as exc:
                    print(f"[QC] [ERROR] Sequential QC analysis failed: {exc}")

        return results
    
    def _calculate_pixel_metrics(self, img: np.ndarray, channel: str) -> Optional[Dict[str, Any]]:
        """Calculate pixel-level QC metrics using Otsu threshold."""
        if not _HAVE_SCIKIT_IMAGE:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing Dependency",
                "scikit-image is required for pixel-level analysis. Please install it: pip install scikit-image"
            )
            return None
        
        try:
            # Convert to float if needed
            img_float = img.astype(np.float32)
            
            # Calculate Otsu threshold
            threshold = threshold_otsu(img_float)
            
            # Separate signal (foreground) and background
            foreground = img_float[img_float > threshold]
            background = img_float[img_float <= threshold]
            
            if len(foreground) == 0 or len(background) == 0:
                return None
            
            # Calculate metrics
            signal_mean = np.mean(foreground)
            signal_std = np.std(foreground)
            background_mean = np.mean(background)
            background_std = np.std(background)
            
            # Calculate image range for robust SNR calculation
            img_min = np.min(img_float)
            img_max = np.max(img_float)
            
            # SNR: (signal_mean - background_mean) / background_std (with robust handling)
            snr = _calculate_snr(signal_mean, background_mean, background_std, img_min, img_max)
            
            # Intensity metrics
            mean_intensity = np.mean(img_float)
            median_intensity = np.median(img_float)
            max_intensity = np.max(img_float)
            min_intensity = np.min(img_float)
            
            # Coverage: percentage of pixels above threshold
            coverage_pct = (len(foreground) / img_float.size) * 100
            
            # Calculate percentiles
            p1 = np.percentile(img_float, 1)
            p25 = np.percentile(img_float, 25)
            p75 = np.percentile(img_float, 75)
            p99 = np.percentile(img_float, 99)
            
            return {
                'snr': snr,
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'background_mean': background_mean,
                'background_std': background_std,
                'threshold': threshold,
                'mean_intensity': mean_intensity,
                'median_intensity': median_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'coverage_pct': coverage_pct,
                'p1': p1,
                'p25': p25,
                'p75': p75,
                'p99': p99,
                'total_pixels': img_float.size,
                'foreground_pixels': len(foreground),
                'background_pixels': len(background)
            }
        except Exception as e:
            print(f"Error calculating pixel metrics for {channel}: {e}")
            return None
    
    def _calculate_cell_metrics(self, img: np.ndarray, channel: str, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Calculate cell-level QC metrics using segmentation masks."""
        try:
            # Convert to float if needed
            img_float = img.astype(np.float32)
            
            # Ensure mask and image have same shape
            if mask.shape != img_float.shape:
                print(f"Warning: Mask shape {mask.shape} doesn't match image shape {img_float.shape}")
                return None
            
            # Separate signal (cells) and background
            cell_mask = mask > 0
            background_mask = mask == 0
            
            if np.sum(cell_mask) == 0 or np.sum(background_mask) == 0:
                return None
            
            background = img_float[background_mask]
            
            background_mean = np.mean(background)
            background_std = np.std(background)
            
            # Calculate image range for robust SNR calculation
            img_min = np.min(img_float)
            img_max = np.max(img_float)
            signal_settings = self._get_cell_signal_settings()
            signal_metrics = _compute_cell_signal_metrics(
                img_float,
                mask,
                float(background_mean),
                float(background_std),
                float(img_min),
                float(img_max),
                **signal_settings,
            )
            signal_mean = signal_metrics['signal_mean']
            signal_std = signal_metrics['signal_std']
            snr = signal_metrics['snr']
            
            # Intensity metrics (raw image intensities for consistency with core.qc_analysis)
            mean_intensity = np.mean(img_float)
            median_intensity = np.median(img_float)
            max_intensity = np.max(img_float)
            min_intensity = np.min(img_float)
            
            # Coverage: percentage of pixels covered by cells
            coverage_pct = (np.sum(cell_mask) / img_float.size) * 100
            
            # Cell density: number of cells per unit area
            unique_cells = np.unique(mask[mask > 0])
            num_cells = len(unique_cells)
            area_pixels = img_float.size
            # Assuming square pixels, convert to cells per mm² if we have pixel size info
            # For now, just report cells per pixel²
            cell_density = num_cells / area_pixels if area_pixels > 0 else 0
            
            # Calculate percentiles for cell intensities
            p1 = np.percentile(img_float[cell_mask], 1)
            p25 = np.percentile(img_float[cell_mask], 25)
            p75 = np.percentile(img_float[cell_mask], 75)
            p99 = np.percentile(img_float[cell_mask], 99)
            
            # Per-cell statistics
            cell_intensities = []
            for cell_id in unique_cells:
                cell_pixels = img_float[mask == cell_id]
                if len(cell_pixels) > 0:
                    cell_intensities.append(np.mean(cell_pixels))
            
            mean_cell_intensity = np.mean(cell_intensities) if cell_intensities else 0
            median_cell_intensity = np.median(cell_intensities) if cell_intensities else 0
            
            return {
                'snr': snr,
                'signal_mean': signal_mean,
                'signal_std': signal_std,
                'background_mean': background_mean,
                'background_std': background_std,
                'cell_signal_method': signal_settings['cell_signal_method'],
                'signal_threshold': signal_metrics['signal_threshold'],
                'signal_quantile': signal_metrics['signal_quantile'],
                'n_signal_pixels': signal_metrics['n_signal_pixels'],
                'n_signal_cells': signal_metrics['n_signal_cells'],
                'signal_fraction': signal_metrics['signal_fraction'],
                'signal_coverage_pct': signal_metrics['signal_coverage_pct'],
                'mean_intensity': mean_intensity,
                'median_intensity': median_intensity,
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'coverage_pct': coverage_pct,
                'cell_density': cell_density,
                'num_cells': num_cells,
                'p1': p1,
                'p25': p25,
                'p75': p75,
                'p99': p99,
                'mean_cell_intensity': mean_cell_intensity,
                'median_cell_intensity': median_cell_intensity,
                'total_pixels': img_float.size,
                'foreground_pixels': np.sum(cell_mask),
                'background_pixels': np.sum(background_mask)
            }
        except Exception as e:
            print(f"Error calculating cell metrics for {channel}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_results_by_channel(self):
        """Pool per-ROI sufficient statistics into reproducible channel summaries."""
        if self.qc_results is None or self.qc_results.empty:
            return
        self.qc_results_aggregated = aggregate_qc_results(self.qc_results)
    
    def _save_results_to_cache(self):
        """Save current QC results to the parent window's cache for persistence."""
        if not self.file_set_id or not self.parent_window:
            return
        
        if not hasattr(self.parent_window, 'qc_results_cache'):
            self.parent_window.qc_results_cache = {}
        
        cache_key = self._get_qc_cache_key()
        if not cache_key:
            return
        signal_settings = self._get_cell_signal_settings()
        self.parent_window.qc_results_cache[cache_key] = {
            'qc_results': self.qc_results.copy() if self.qc_results is not None else None,
            'qc_results_aggregated': self.qc_results_aggregated.copy() if self.qc_results_aggregated is not None else None,
            'analysis_mode': self.analysis_mode,
            'cell_signal_method': signal_settings['cell_signal_method'],
            'positive_threshold_sd': signal_settings['positive_threshold_sd'],
            'upper_quantile': signal_settings['upper_quantile'],
        }
    
    def _restore_cached_results(self):
        """Restore QC results from cache if available for the current file set."""
        if not self.parent_window:
            return
        
        if not hasattr(self.parent_window, 'qc_results_cache'):
            return
        
        # Try to restore results for the current QC configuration
        cached = None
        cache_key = self._get_qc_cache_key()
        if cache_key:
            cached = self.parent_window.qc_results_cache.get(cache_key)
        
        # If no exact match, try to find any cache entry that matches the current settings
        if not cached or cached.get('qc_results') is None:
            for cache_key, cache_data in self.parent_window.qc_results_cache.items():
                if (isinstance(cache_data, dict) and 
                    (not self.file_set_id or str(cache_key).startswith(f"{self.file_set_id}_")) and
                    self._cache_matches_current_settings(cache_data) and
                    cache_data.get('qc_results') is not None):
                    cached = cache_data
                    break
        
        if cached and cached.get('qc_results') is not None:
            # Restore results
            self.qc_results = cached['qc_results'].copy()
            # Recompute summaries so legacy caches cannot restore the old
            # mean-of-ratios aggregation.
            self._aggregate_results_by_channel()
            
            # Update UI
            self._update_summary_table()
            self._update_plots()
            self.export_summary_btn.setEnabled(True)
            self.snr_intensity_save_btn.setEnabled(True)
            self.coverage_save_btn.setEnabled(True)
            self.distribution_save_btn.setEnabled(True)
        
        # Restore UI state if available
        if hasattr(self.parent_window, '_saved_qc_ui_state') and self.parent_window._saved_qc_ui_state:
            ui_state = self.parent_window._saved_qc_ui_state
            if "analysis_mode" in ui_state and hasattr(self, 'mode_combo'):
                index = self.mode_combo.findText(ui_state["analysis_mode"])
                if index >= 0:
                    self.mode_combo.setCurrentIndex(index)
            if "selected_acquisition" in ui_state and hasattr(self, 'acq_combo'):
                index = self.acq_combo.findText(ui_state["selected_acquisition"])
                if index >= 0:
                    self.acq_combo.setCurrentIndex(index)
            if "num_workers" in ui_state and hasattr(self, 'workers_spin'):
                self.workers_spin.setValue(ui_state["num_workers"])
            if "snr_threshold" in ui_state and hasattr(self, 'snr_threshold_spin'):
                self.snr_threshold_spin.setValue(float(ui_state["snr_threshold"]))
            if "denoise_source" in ui_state and hasattr(self, 'denoise_source_combo'):
                index = self.denoise_source_combo.findText(ui_state["denoise_source"])
                if index >= 0:
                    self.denoise_source_combo.setCurrentIndex(index)
            if "custom_denoise_settings" in ui_state:
                self.custom_denoise_settings = dict(ui_state["custom_denoise_settings"] or {})
                self._populate_denoise_channel_list()
                self._load_denoise_settings()
            if "cell_signal_method" in ui_state and hasattr(self, 'cell_signal_method_combo'):
                index = self.cell_signal_method_combo.findData(ui_state["cell_signal_method"])
                if index >= 0:
                    self.cell_signal_method_combo.setCurrentIndex(index)
            if "positive_threshold_sd" in ui_state and hasattr(self, 'positive_threshold_sd_spin'):
                self.positive_threshold_sd_spin.setValue(float(ui_state["positive_threshold_sd"]))
            if "upper_quantile_percent" in ui_state and hasattr(self, 'upper_quantile_spin'):
                self.upper_quantile_spin.setValue(float(ui_state["upper_quantile_percent"]))
            self._update_cell_signal_controls()
    
    def _update_summary_table(self):
        """Update the summary table with aggregated results."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        methods = set()
        if 'cell_signal_method' in self.qc_results_aggregated.columns:
            methods = {
                str(method)
                for method in self.qc_results_aggregated['cell_signal_method'].dropna().tolist()
                if str(method)
            }

        summary_subtitle = self._get_qc_method_subtitle()
        if summary_subtitle:
            self.summary_method_label.setText(summary_subtitle)
            self.summary_method_label.show()
        else:
            self.summary_method_label.hide()

        coverage_col = 'coverage_pct'
        if (
            self.analysis_mode == "cell"
            and methods
            and methods != {'all_cell_mean'}
            and 'signal_coverage_pct' in self.qc_results_aggregated.columns
        ):
            coverage_col = 'signal_coverage_pct'

        # Select relevant columns for display
        display_cols = [
            'channel',
            'n_rois',
            'snr',
            'signal_minus_background',
            'background_std',
            'mean_intensity',
            coverage_col,
        ]
        if 'cell_density' in self.qc_results_aggregated.columns:
            display_cols.append('cell_density')
        
        # Get available columns
        available_cols = [col for col in display_cols if col in self.qc_results_aggregated.columns]
        
        # Set up table
        self.summary_table.setRowCount(len(self.qc_results_aggregated))
        self.summary_table.setColumnCount(len(available_cols))
        self.summary_table.setHorizontalHeaderLabels(available_cols)
        
        # Populate table
        for i, row in self.qc_results_aggregated.iterrows():
            for j, col in enumerate(available_cols):
                value = row[col]
                if isinstance(value, (int, np.integer)):
                    item = QtWidgets.QTableWidgetItem(str(value))
                elif isinstance(value, (float, np.floating)):
                    item = QtWidgets.QTableWidgetItem(f"{value:.3f}")
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                self.summary_table.setItem(i, j, item)
        
        # Resize columns to content
        self.summary_table.setColumnWidth(0, 200)  # Channel name column
        self.summary_table.resizeColumnsToContents()
    
    def _update_plots(self):
        """Update all plots."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        # Update SNR vs Intensity plot (using aggregated values)
        self._plot_snr_vs_intensity()
        
        # Update Coverage plot (using aggregated values)
        self._plot_coverage()
        
        # Update Distribution plots (using raw data to show distributions)
        self._plot_distributions()
    
    def _plot_snr_vs_intensity(self):
        """Plot SNR vs mean intensity using a truthful SNR axis for the observed range."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        fig = self.snr_intensity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        intensity_col = self._get_plot_intensity_column()

        # Get aggregated intensities and SNR
        intensities = self.qc_results_aggregated[intensity_col].values
        snr_values = self.qc_results_aggregated['snr'].values
        channels = self.qc_results_aggregated['channel'].values
        
        # Mean intensities must stay positive for the log-scaled x-axis.
        valid_mask = np.isfinite(intensities) & np.isfinite(snr_values) & (intensities > 0)
        intensities = intensities[valid_mask]
        snr_values = snr_values[valid_mask]
        channels = channels[valid_mask]
        
        if len(intensities) == 0:
            ax.text(0.5, 0.5, 'No valid data points to plot', 
                   ha='center', va='center', transform=ax.transAxes)
            fig.tight_layout()
            self.snr_intensity_canvas.draw()
            return
        
        # Scatter plot
        ax.scatter(
            intensities,
            snr_values,
            alpha=0.6,
            s=50
        )
        
        # Add channel labels
        for i, channel in enumerate(channels):
            ax.annotate(
                channel,
                (intensities[i], snr_values[i]),
                fontsize=7,
                alpha=0.7
            )
        
        # Set log scale on both axes
        ax.set_xscale('log')
        if np.any(snr_values <= 0):
            ax.set_yscale('symlog', linthresh=1.0)
            ax.axhline(y=0.0, color='lightgray', linestyle=':', linewidth=1, alpha=0.6)
            y_axis_label = 'Background-referenced SNR (symlog scale, pooled across ROIs)'
        else:
            ax.set_yscale('log')
            y_axis_label = 'Background-referenced SNR (log scale, pooled across ROIs)'
        
        snr_threshold = self.get_snr_threshold()
        ax.axhline(
            y=snr_threshold,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.6,
            label=f'SNR threshold: {snr_threshold:.1f}',
        )

        ax.set_xlabel(f'{self._get_plot_intensity_label()} (pooled across ROIs)', fontsize=10)
        ax.set_ylabel(y_axis_label, fontsize=10)
        ax.set_title(self._compose_plot_title('Background-referenced SNR vs Foreground Mean Intensity by Channel (Pooled across ROIs)'), fontsize=12, fontweight='bold')
        ax.grid(False)  # Remove gridlines
        ax.legend(fontsize=8, loc='best')
        
        fig.tight_layout()
        self.snr_intensity_canvas.draw()
    
    def _plot_coverage(self):
        """Plot coverage and cell density (using aggregated values)."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        fig = self.coverage_canvas.figure
        fig.clear()
        
        # Create subplots
        n_plots = 2 if 'cell_density' in self.qc_results_aggregated.columns else 1
        if n_plots == 2:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        else:
            ax1 = fig.add_subplot(111)
            ax2 = None
        
        # Get aggregated values
        channels = self.qc_results_aggregated['channel'].values
        coverage = self.qc_results_aggregated['coverage_pct'].values
        cell_density = self.qc_results_aggregated['cell_density'].values if 'cell_density' in self.qc_results_aggregated.columns else None
        
        # Coverage plot
        ax1.bar(range(len(channels)), coverage, alpha=0.7, color='steelblue')
        ax1.set_xticks(range(len(channels)))
        ax1.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('% Coverage (pooled across ROIs)', fontsize=10)
        ax1.set_title(self._compose_plot_title('Coverage by Channel (Pooled across ROIs)'), fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_coverage = np.mean(coverage)
        ax1.axhline(y=mean_coverage, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {mean_coverage:.1f}%')
        ax1.legend(fontsize=8)
        
        # Cell density plot (if available)
        if ax2 is not None and cell_density is not None:
            ax2.bar(range(len(channels)), cell_density, alpha=0.7, color='orange')
            ax2.set_xticks(range(len(channels)))
            ax2.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Cell Density (cells/pixel², pooled across ROIs)', fontsize=10)
            ax2.set_title(self._compose_plot_title('Cell Density by Channel (Pooled across ROIs)'), fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add mean line
            mean_density = np.mean(cell_density)
            ax2.axhline(y=mean_density, color='red', linestyle='--', alpha=0.5, label=f'Overall Mean: {mean_density:.2e}')
            ax2.legend(fontsize=8)
        
        fig.tight_layout()
        self.coverage_canvas.draw()
    
    def _plot_distributions(self):
        """Plot boxplots showing distributions of SNR, intensity, and coverage across ROIs."""
        if self.qc_results is None or self.qc_results.empty:
            return
        
        fig = self.distribution_canvas.figure
        fig.clear()
        
        # Create 3 subplots
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        
        # Get unique channels (sorted)
        channels = sorted(self.qc_results['channel'].unique())
        num_channels = len(channels)
        
        # Scale font size based on number of channels (smaller for more channels)
        # Base font size of 8, scales down to minimum of 5 for many channels
        if num_channels <= 10:
            xlabel_fontsize = 8
        elif num_channels <= 20:
            xlabel_fontsize = 7
        elif num_channels <= 30:
            xlabel_fontsize = 6
        else:
            xlabel_fontsize = 5
        
        # Prepare data for boxplots
        snr_data = []
        intensity_data = []
        coverage_data = []
        intensity_col = self._get_plot_intensity_column()
        
        for channel in channels:
            channel_data = self.qc_results[self.qc_results['channel'] == channel]
            # Filter valid values
            valid_snr = channel_data['snr'][np.isfinite(channel_data['snr'])].values
            valid_intensity = channel_data[intensity_col][channel_data[intensity_col] > 0].values
            valid_coverage = channel_data['coverage_pct'].values
            
            snr_data.append(valid_snr if len(valid_snr) > 0 else [0])
            intensity_data.append(valid_intensity if len(valid_intensity) > 0 else [0])
            coverage_data.append(valid_coverage if len(valid_coverage) > 0 else [0])
        
        # SNR boxplot
        bp1 = _boxplot_with_labels(ax1, snr_data, channels)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        snr_values = np.concatenate([np.asarray(values, dtype=float) for values in snr_data]) if snr_data else np.array([])
        if snr_values.size > 0 and np.any(snr_values <= 0):
            ax1.set_yscale('symlog', linthresh=1.0)
            ax1.axhline(y=0.0, color='lightgray', linestyle=':', linewidth=1, alpha=0.6)
            snr_ylabel = 'SNR (symlog scale)'
        else:
            ax1.set_yscale('log')
            snr_ylabel = 'SNR (log scale)'
        snr_threshold = self.get_snr_threshold()
        ax1.axhline(
            y=snr_threshold,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.6,
            label=f'Threshold: {snr_threshold:.1f}',
        )
        ax1.set_ylabel(snr_ylabel, fontsize=10)
        ax1.set_title(self._compose_plot_title('SNR Distribution across ROIs'), fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=90, labelsize=xlabel_fontsize)  # Fully vertical text
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend(fontsize=8, loc='best')
        
        # Intensity boxplot
        bp2 = _boxplot_with_labels(ax2, intensity_data, channels)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
        ax2.set_yscale('log')
        ax2.set_ylabel(self._get_plot_intensity_label(), fontsize=10)
        ax2.set_title(self._compose_plot_title('Foreground Intensity Distribution across ROIs'), fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=90, labelsize=xlabel_fontsize)  # Fully vertical text
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Coverage boxplot
        bp3 = _boxplot_with_labels(ax3, coverage_data, channels)
        for patch in bp3['boxes']:
            patch.set_facecolor('lightcoral')
        ax3.set_ylabel('% Coverage', fontsize=10)
        ax3.set_title(self._compose_plot_title('Coverage Distribution across ROIs'), fontsize=11, fontweight='bold')
        ax3.tick_params(axis='x', rotation=90, labelsize=xlabel_fontsize)  # Fully vertical text
        ax3.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        self.distribution_canvas.draw()
    
    def _export_results(self):
        """Export results to CSV."""
        if self.qc_results_aggregated is None or self.qc_results_aggregated.empty:
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QC Results",
            f"qc_results_{self._get_export_suffix()}.csv" if self._get_export_suffix() else "qc_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                # Export the pooled channel summary and the per-ROI values used
                # to calculate it so every reported ratio is auditable.
                self.qc_results_aggregated.to_csv(filename, index=False)
                stem, extension = os.path.splitext(filename)
                per_roi_filename = f"{stem}_per_roi{extension or '.csv'}"
                if self.qc_results is not None:
                    self.qc_results.to_csv(per_roi_filename, index=False)
                QtWidgets.QMessageBox.information(
                    self,
                    "Success",
                    f"Channel summary exported to {filename}\n"
                    f"Per-ROI metrics exported to {per_roi_filename}",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
    
    def _save_distribution_plot(self):
        """Save distribution plot."""
        fig = self.distribution_canvas.figure
        suffix = self._get_export_suffix()
        filename = f"QC_Distributions_{suffix}" if suffix else "QC_Distributions"
        save_figure_with_options(fig, filename, self)
    
    def _save_snr_intensity_plot(self):
        """Save SNR vs Intensity plot."""
        fig = self.snr_intensity_canvas.figure
        suffix = self._get_export_suffix()
        filename = f"SNR_vs_Intensity_{suffix}" if suffix else "SNR_vs_Intensity"
        save_figure_with_options(fig, filename, self)
    
    def _save_coverage_plot(self):
        """Save coverage plot."""
        fig = self.coverage_canvas.figure
        suffix = self._get_export_suffix()
        filename = f"Coverage_Density_{suffix}" if suffix else "Coverage_Density"
        save_figure_with_options(fig, filename, self)
