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
Core operations for OpenIMC.

This module provides unified core operations that can be used by both
the GUI and CLI interfaces, ensuring exact parity between them.
"""

import json
import os
import sys
import gc
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import anndata as ad

# Try to import psutil for better memory tracking
try:
    import psutil
    _HAVE_PSUTIL = True
except ImportError:
    _HAVE_PSUTIL = False

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree, Delaunay

from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.feature_worker import (
    _apply_denoise_to_channel,
    extract_features_for_acquisition,
    drop_excluded_channel_feature_columns
)
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.processing.batch_correction import (
    apply_combat_correction,
    apply_harmony_correction,
    detect_batch_variable,
    get_feature_columns_from_dataframe,
    validate_batch_correction_inputs
)
from openimc.processing.spillover_correction import (
    load_spillover,
    compensate_counts,
    compensate_image_counts
)
from openimc.processing.spillover_matrix import (
    compute_spillmat,
    adapt_spillmat,
    build_spillover_from_comp_mcd
)
from openimc.processing.deconvolution_worker import RLD_HRIMC_circle
from openimc.processing.spatial_analysis_worker import roi_enrichment_worker, distance_distribution_worker
from openimc.ui.cluster_utils import canonicalize_cluster_id, sort_cluster_values
from openimc.ui.utils import (
    arcsinh_normalize,
    benjamini_hochberg_adjust_matrix,
    percentile_clip_normalize,
    channelwise_minmax_normalize,
    combine_channels,
    combine_pvalues_fisher,
)

QC_CELL_SIGNAL_METHODS = (
    "positive_pixels",
    "upper_quantile",
    "all_cell_mean",
)


def _validate_qc_cell_signal_method(method: str) -> str:
    """Validate the configured QC cell signal method."""
    if method not in QC_CELL_SIGNAL_METHODS:
        raise ValueError(
            f"Invalid cell_signal_method {method!r}; expected one of {QC_CELL_SIGNAL_METHODS}"
        )
    return method


def _qc_min_background_std(
    background_mean: float,
    background_std: float,
    img_min: Optional[float] = None,
    img_max: Optional[float] = None,
) -> float:
    """Return the robust minimum background standard deviation used in QC."""
    min_std_relative = abs(background_mean) * 0.001
    min_std_absolute = 1e-6
    min_std_range = 0.0
    if img_min is not None and img_max is not None:
        img_range = img_max - img_min
        if img_range > 0:
            min_std_range = img_range * 0.0001
    return max(float(background_std), min_std_relative, min_std_absolute, min_std_range)


def _calculate_qc_snr(
    signal_mean: float,
    background_mean: float,
    background_std: float,
    img_min: Optional[float] = None,
    img_max: Optional[float] = None,
) -> float:
    """Calculate Signal-to-Noise Ratio with robust handling."""
    signal_diff = signal_mean - background_mean
    min_std = _qc_min_background_std(background_mean, background_std, img_min, img_max)
    return signal_diff / min_std


def _pool_population_statistics(
    means: pd.Series,
    standard_deviations: pd.Series,
    counts: pd.Series,
) -> Tuple[float, float, int]:
    """Pool population means and SDs from disjoint pixel groups."""
    mean_values = pd.to_numeric(means, errors="coerce").to_numpy(dtype=float)
    std_values = pd.to_numeric(standard_deviations, errors="coerce").to_numpy(dtype=float)
    count_values = pd.to_numeric(counts, errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(mean_values)
        & np.isfinite(std_values)
        & np.isfinite(count_values)
        & (count_values > 0)
    )
    if not np.any(valid):
        return np.nan, np.nan, 0

    mean_values = mean_values[valid]
    std_values = std_values[valid]
    count_values = count_values[valid]
    total_count = float(np.sum(count_values))
    pooled_mean = float(np.sum(count_values * mean_values) / total_count)
    second_moment = float(
        np.sum(count_values * (np.square(std_values) + np.square(mean_values)))
        / total_count
    )
    pooled_variance = max(second_moment - pooled_mean ** 2, 0.0)
    return pooled_mean, float(np.sqrt(pooled_variance)), int(total_count)


def aggregate_qc_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-ROI QC results with pixel-weighted sufficient statistics.

    Averaging ROI-level ratios independently produces a summary row whose SNR
    cannot be reproduced from its displayed signal and background columns. This
    function pools the underlying means, population SDs, and pixel counts first,
    then calculates the background-referenced SNR once from the pooled values.
    """
    if results is None or results.empty:
        return pd.DataFrame()
    if "channel" not in results.columns:
        raise ValueError("QC results must contain a 'channel' column")

    intensity_mean_col = "mean_intensity" if "mean_intensity" in results.columns else "intensity_mean"
    intensity_std_col = "std_intensity" if "std_intensity" in results.columns else "intensity_std"
    intensity_median_col = "median_intensity" if "median_intensity" in results.columns else "intensity_median"
    intensity_min_col = "min_intensity" if "min_intensity" in results.columns else "intensity_min"
    intensity_max_col = "max_intensity" if "max_intensity" in results.columns else "intensity_max"

    summary_rows = []
    for channel, group in results.groupby("channel", sort=False, dropna=False):
        row: Dict[str, Any] = {
            "channel": channel,
            "n_rois": int(len(group)),
        }
        if "mode" in group.columns:
            modes = group["mode"].dropna().astype(str).unique().tolist()
            row["mode"] = modes[0] if len(modes) == 1 else "mixed"

        for categorical_column in ("cell_signal_method",):
            if categorical_column in group.columns:
                values = group[categorical_column].dropna().unique().tolist()
                row[categorical_column] = values[0] if len(values) == 1 else "mixed"

        count_columns = (
            "n_total_pixels",
            "n_signal_pixels",
            "n_background_pixels",
            "n_cell_pixels",
            "n_signal_cells",
            "n_cells",
            "num_cells",
        )
        for count_column in count_columns:
            if count_column in group.columns:
                row[count_column] = int(
                    pd.to_numeric(group[count_column], errors="coerce").fillna(0).sum()
                )

        signal_mean, signal_std, pooled_signal_count = _pool_population_statistics(
            group["signal_mean"],
            group["signal_std"] if "signal_std" in group.columns else pd.Series(0.0, index=group.index),
            group["n_signal_pixels"] if "n_signal_pixels" in group.columns else pd.Series(1.0, index=group.index),
        )
        background_mean, background_std, pooled_background_count = _pool_population_statistics(
            group["background_mean"],
            group["background_std"],
            group["n_background_pixels"] if "n_background_pixels" in group.columns else pd.Series(1.0, index=group.index),
        )
        intensity_mean, intensity_std, _ = _pool_population_statistics(
            group[intensity_mean_col],
            group[intensity_std_col],
            group["n_total_pixels"] if "n_total_pixels" in group.columns else pd.Series(1.0, index=group.index),
        )

        row.update({
            "signal_mean": signal_mean,
            "signal_std": signal_std,
            "background_mean": background_mean,
            "background_std": background_std,
            intensity_mean_col: intensity_mean,
            intensity_std_col: intensity_std,
        })
        row["signal_minus_background"] = float(signal_mean - background_mean)
        row["signal_to_background_ratio"] = (
            float(signal_mean / background_mean)
            if np.isfinite(background_mean) and background_mean > 0
            else np.nan
        )

        if intensity_min_col in group.columns:
            row[intensity_min_col] = float(pd.to_numeric(group[intensity_min_col], errors="coerce").min())
        if intensity_max_col in group.columns:
            row[intensity_max_col] = float(pd.to_numeric(group[intensity_max_col], errors="coerce").max())
        if intensity_median_col in group.columns:
            row[intensity_median_col] = float(pd.to_numeric(group[intensity_median_col], errors="coerce").mean())

        if pooled_signal_count == 0:
            row["snr"] = 0.0
        elif pooled_background_count == 0:
            row["snr"] = np.nan
        else:
            row["snr"] = _calculate_qc_snr(
                signal_mean,
                background_mean,
                background_std,
                row.get(intensity_min_col),
                row.get(intensity_max_col),
            )

        total_pixels = int(row.get("n_total_pixels", 0))
        mode = row.get("mode")
        if total_pixels > 0:
            coverage_count = row.get("n_cell_pixels", 0) if mode == "cell" else row.get("n_signal_pixels", 0)
            row["coverage_pct"] = float(100.0 * coverage_count / total_pixels)
            row["signal_coverage_pct"] = float(100.0 * row.get("n_signal_pixels", 0) / total_pixels)
            if mode == "cell":
                row["cell_density"] = float(row.get("n_cells", row.get("num_cells", 0)) / total_pixels)
        elif "coverage_pct" in group.columns:
            row["coverage_pct"] = float(pd.to_numeric(group["coverage_pct"], errors="coerce").mean())

        n_cell_pixels = int(row.get("n_cell_pixels", 0))
        if n_cell_pixels > 0:
            row["signal_fraction"] = float(row.get("n_signal_pixels", 0) / n_cell_pixels)

        for averaged_column in ("threshold", "signal_threshold", "signal_quantile", "p1", "p25", "p75", "p99", "mean_cell_intensity", "median_cell_intensity"):
            if averaged_column in group.columns:
                row[averaged_column] = float(pd.to_numeric(group[averaged_column], errors="coerce").mean())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _compute_cell_signal_metrics(
    img: np.ndarray,
    mask: np.ndarray,
    background_mean: float,
    background_std: float,
    img_min: float,
    img_max: float,
    *,
    cell_signal_method: str = "positive_pixels",
    positive_threshold_sd: float = 2.0,
    upper_quantile: float = 0.90,
) -> Dict[str, Any]:
    """Compute cell-level QC signal metrics for the configured signal definition."""
    method = _validate_qc_cell_signal_method(cell_signal_method)
    if positive_threshold_sd < 0:
        raise ValueError("positive_threshold_sd must be non-negative")
    if not (0.0 < upper_quantile <= 1.0):
        raise ValueError("upper_quantile must be in the interval (0, 1]")

    cell_mask = mask > 0
    total_cell_pixels = int(np.count_nonzero(cell_mask))
    total_pixels = int(mask.size)

    signal_threshold = np.nan
    signal_quantile = np.nan

    if total_cell_pixels == 0 or total_pixels == 0:
        return {
            "signal_mean": float(background_mean),
            "signal_std": 0.0,
            "snr": 0.0,
            "signal_threshold": signal_threshold,
            "signal_quantile": signal_quantile,
            "n_signal_pixels": 0,
            "n_signal_cells": 0,
            "signal_fraction": 0.0,
            "signal_coverage_pct": 0.0,
        }

    cell_ids = np.unique(mask[cell_mask])
    n_cells = int(len(cell_ids))
    cell_pixels = img[cell_mask]

    if method == "all_cell_mean":
        signal_pixels = cell_pixels
        n_signal_cells = n_cells
    elif method == "positive_pixels":
        sigma_bg_robust = _qc_min_background_std(background_mean, background_std, img_min, img_max)
        signal_threshold = float(background_mean + (positive_threshold_sd * sigma_bg_robust))
        signal_mask = cell_mask & (img > signal_threshold)
        signal_pixels = img[signal_mask]
        if signal_pixels.size == 0:
            return {
                "signal_mean": float(background_mean),
                "signal_std": 0.0,
                "snr": 0.0,
                "signal_threshold": signal_threshold,
                "signal_quantile": signal_quantile,
                "n_signal_pixels": 0,
                "n_signal_cells": 0,
                "signal_fraction": 0.0,
                "signal_coverage_pct": 0.0,
            }
        n_signal_cells = int(len(np.unique(mask[signal_mask])))
    else:
        signal_quantile = float(upper_quantile)
        try:
            from scipy import ndimage as ndi

            cell_counts = ndi.labeled_comprehension(
                np.ones_like(img, dtype=np.float64),
                mask,
                cell_ids,
                np.sum,
                float,
                0.0,
            )
            cell_sums = ndi.labeled_comprehension(
                img.astype(np.float64),
                mask,
                cell_ids,
                np.sum,
                float,
                0.0,
            )
            valid_cells = cell_counts > 0
            valid_cell_ids = cell_ids[valid_cells]
            if not np.any(valid_cells):
                return {
                    "signal_mean": float(background_mean),
                    "signal_std": 0.0,
                    "snr": 0.0,
                    "signal_threshold": signal_threshold,
                    "signal_quantile": signal_quantile,
                    "n_signal_pixels": 0,
                    "n_signal_cells": 0,
                    "signal_fraction": 0.0,
                    "signal_coverage_pct": 0.0,
                }
            cell_means = np.divide(
                cell_sums[valid_cells],
                cell_counts[valid_cells],
                out=np.zeros_like(cell_sums[valid_cells], dtype=np.float64),
                where=(cell_counts[valid_cells] > 0),
            )
        except ImportError:
            valid_cell_ids = []
            cell_means = []
            cell_mask_values = mask[cell_mask]
            for cell_id in cell_ids:
                cell_values = cell_pixels[cell_mask_values == cell_id]
                if len(cell_values) == 0:
                    continue
                valid_cell_ids.append(cell_id)
                cell_means.append(float(np.mean(cell_values)))
            if not valid_cell_ids:
                return {
                    "signal_mean": float(background_mean),
                    "signal_std": 0.0,
                    "snr": 0.0,
                    "signal_threshold": signal_threshold,
                    "signal_quantile": signal_quantile,
                    "n_signal_pixels": 0,
                    "n_signal_cells": 0,
                    "signal_fraction": 0.0,
                    "signal_coverage_pct": 0.0,
                }
            valid_cell_ids = np.asarray(valid_cell_ids)
            cell_means = np.asarray(cell_means, dtype=np.float64)

        cutoff = float(np.quantile(cell_means, upper_quantile))
        selected_cell_ids = np.asarray(valid_cell_ids)[cell_means >= cutoff]
        if selected_cell_ids.size == 0:
            return {
                "signal_mean": float(background_mean),
                "signal_std": 0.0,
                "snr": 0.0,
                "signal_threshold": signal_threshold,
                "signal_quantile": signal_quantile,
                "n_signal_pixels": 0,
                "n_signal_cells": 0,
                "signal_fraction": 0.0,
                "signal_coverage_pct": 0.0,
            }
        signal_mask = np.isin(mask, selected_cell_ids)
        signal_pixels = img[signal_mask]
        n_signal_cells = int(selected_cell_ids.size)

    n_signal_pixels = int(signal_pixels.size)
    signal_mean = float(np.mean(signal_pixels))
    signal_std = float(np.std(signal_pixels))
    signal_fraction = float(n_signal_pixels / total_cell_pixels) if total_cell_pixels > 0 else 0.0
    signal_coverage_pct = float((n_signal_pixels / total_pixels) * 100.0) if total_pixels > 0 else 0.0

    return {
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "snr": _calculate_qc_snr(signal_mean, background_mean, background_std, img_min, img_max),
        "signal_threshold": signal_threshold,
        "signal_quantile": signal_quantile,
        "n_signal_pixels": n_signal_pixels,
        "n_signal_cells": n_signal_cells,
        "signal_fraction": signal_fraction,
        "signal_coverage_pct": signal_coverage_pct,
    }


def load_mcd(
    input_path: Union[str, Path],
    channel_format: str = 'CHW'
) -> Tuple[Union[MCDLoader, OMETIFFLoader], str]:
    """Load data from MCD file or OME-TIFF directory.
    
    This is the unified data loading function used by both GUI and CLI.
    
    Args:
        input_path: Path to MCD file or OME-TIFF directory
        channel_format: Format for OME-TIFF files ('CHW' or 'HWC'), default is 'CHW'
    
    Returns:
        Tuple of (loader, loader_type) where loader_type is 'mcd' or 'ometiff'
    
    Raises:
        ValueError: If input path is invalid or unsupported format
    """
    input_path = Path(input_path)
    
    if input_path.is_file() and input_path.suffix.lower() in ['.mcd', '.mcdx']:
        # Load MCD file
        loader = MCDLoader()
        loader.open(str(input_path))
        return loader, 'mcd'
    elif input_path.is_dir():
        # Load OME-TIFF directory
        loader = OMETIFFLoader(channel_format=channel_format)
        loader.open(str(input_path))
        return loader, 'ometiff'
    else:
        raise ValueError(
            f"Input path must be an MCD file or directory containing OME-TIFF files: {input_path}"
        )


def parse_denoise_settings(denoise_json: Optional[Union[str, Dict]]) -> Dict:
    """Parse denoise settings from JSON string, file, or dict.
    
    Args:
        denoise_json: JSON string, path to JSON file, or dict with denoise settings
    
    Returns:
        Dictionary with denoise settings per channel
    """
    if not denoise_json:
        return {}
    
    # If already a dict, return as-is
    if isinstance(denoise_json, dict):
        return denoise_json
    
    # Check if it's a file path
    if os.path.isfile(denoise_json):
        with open(denoise_json, 'r') as f:
            return json.load(f)
    
    # Try to parse as JSON string
    try:
        return json.loads(denoise_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for denoise settings: {denoise_json}")


def preprocess(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    output_dir: Union[str, Path],
    denoise_settings: Optional[Dict] = None,
    normalization_method: str = "None",
    arcsinh_cofactor: float = 1.0,
    percentile_params: Tuple[float, float] = (1.0, 99.0),
    viewer_denoise_func: Optional[callable] = None
) -> Path:
    """Preprocess a single acquisition: apply denoising and export to OME-TIFF.
    
    Note: arcsinh normalization is not applied to exported images by default.
    Only denoising is applied. Arcsinh transform should be applied on extracted intensity features.
    
    Args:
        loader: MCDLoader or OMETIFFLoader instance
        acquisition: AcquisitionInfo for the acquisition to process
        output_dir: Directory to save the processed OME-TIFF file
        denoise_settings: Dictionary with denoise settings per channel (optional)
        normalization_method: Normalization method ("None", "arcsinh", "percentile_clip", "channelwise_minmax")
        arcsinh_cofactor: Arcsinh cofactor (only used if normalization_method is "arcsinh")
        percentile_params: Tuple of (low, high) percentiles for percentile_clip normalization
        viewer_denoise_func: Optional function for viewer-based denoising (GUI only)
    
    Returns:
        Path to the saved OME-TIFF file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract original acquisition ID for loader calls (handles multi-file unique IDs)
    original_acq_id = _extract_original_acq_id(acquisition.id)
    
    # Get all channels
    channels = loader.get_channels(original_acq_id)
    img_stack = loader.get_all_channels(original_acq_id)
    
    # Process each channel
    processed_channels = []
    for i, channel_name in enumerate(channels):
        channel_img = img_stack[..., i] if img_stack.ndim == 3 else img_stack
        
        # Apply denoising if configured
        denoise_source = "custom" if (denoise_settings and channel_name in denoise_settings) else "none"
        channel_denoise = denoise_settings.get(channel_name, {}) if denoise_settings else {}
        
        # Process channel - only denoising, no arcsinh normalization for export
        # Note: normalization_method is set to "None" for export to match CLI behavior
        processed = process_channel_for_export(
            channel_img, channel_name, denoise_source,
            {channel_name: channel_denoise} if channel_denoise else {},
            normalization_method,  # Usually "None" for export
            arcsinh_cofactor,
            percentile_params,
            viewer_denoise_func  # Only used in GUI
        )
        
        processed_channels.append(processed)
    
    # Stack channels in CHW format (C, H, W) to match GUI export
    processed_stack = np.stack(processed_channels, axis=0)
    
    # Save as OME-TIFF
    # Use well name if available, otherwise use acquisition name
    if acquisition.well:
        output_filename = f"{acquisition.well}.ome.tif"
    else:
        output_filename = f"{acquisition.name}.ome.tif"
    output_path = output_dir / output_filename
    
    # Create OME metadata
    metadata = {
        'Channel': {'Name': channels}
    }
    
    tifffile.imwrite(
        str(output_path),
        processed_stack,
        metadata=metadata,
        ome=True,
        photometric='minisblack'
    )
    
    return output_path


def _ensure_0_1_range(img: np.ndarray) -> np.ndarray:
    """Ensure image is normalized to 0-1 range using min-max scaling.
    
    Args:
        img: Input image
    
    Returns:
        Image normalized to 0-1 range
    """
    img_float = img.astype(np.float32, copy=True)
    vmin = np.min(img_float)
    vmax = np.max(img_float)
    if vmax > vmin:
        return (img_float - vmin) / (vmax - vmin)
    else:
        return np.zeros_like(img_float)


def _preprocess_channels_for_segmentation(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    nuclear_channels: List[str],
    cyto_channels: List[str],
    denoise_settings: Optional[Dict] = None,
    normalization_method: str = "None",
    arcsinh_cofactor: float = 1.0,
    percentile_params: Tuple[float, float] = (1.0, 99.0),
    nuclear_combo_method: str = "mean",
    cyto_combo_method: str = "mean",
    nuclear_weights: Optional[List[float]] = None,
    cyto_weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Preprocess channels for segmentation: load, denoise, normalize, and combine.
    
    Args:
        loader: MCDLoader or OMETIFFLoader instance
        acquisition: AcquisitionInfo for the acquisition
        nuclear_channels: List of nuclear channel names
        cyto_channels: List of cytoplasm channel names (can be empty)
        denoise_settings: Dictionary with denoise settings per channel (optional)
        normalization_method: Normalization method ("None", "arcsinh", "percentile_clip", "channelwise_minmax")
        arcsinh_cofactor: Arcsinh cofactor (only used if normalization_method is "arcsinh")
        percentile_params: Tuple of (low, high) percentiles for percentile_clip normalization
        nuclear_combo_method: Method to combine nuclear channels ("single", "mean", "weighted", "max", "pca1")
        cyto_combo_method: Method to combine cytoplasm channels
        nuclear_weights: Optional weights for nuclear channels (for weighted combination)
        cyto_weights: Optional weights for cytoplasm channels
    
    Returns:
        Tuple of (nuclear_img, cyto_img) where cyto_img can be None
    """
    # Extract original acquisition ID for loader calls (handles multi-file unique IDs)
    original_acq_id = _extract_original_acq_id(acquisition.id)
    
    # Load and preprocess nuclear channels
    nuclear_imgs = []
    for channel in nuclear_channels:
        img = loader.get_image(original_acq_id, channel)
        # Apply denoising if custom settings provided
        if denoise_settings and channel in denoise_settings:
            img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
        # Apply normalization if configured
        if normalization_method == 'channelwise_minmax':
            img = channelwise_minmax_normalize(img)
        elif normalization_method == 'arcsinh':
            img = arcsinh_normalize(img, cofactor=arcsinh_cofactor)
        elif normalization_method == 'percentile_clip':
            p_low, p_high = percentile_params
            img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
        # Ensure 0-1 range after denoising and normalization
        img = _ensure_0_1_range(img)
        nuclear_imgs.append(img)
    
    # Combine nuclear channels
    nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights)
    nuclear_img = _ensure_0_1_range(nuclear_img)
    # Release intermediate images immediately to free memory
    del nuclear_imgs
    gc.collect()
    
    # Load and preprocess cytoplasm channels
    cyto_img = None
    if cyto_channels:
        cyto_imgs = []
        for channel in cyto_channels:
            img = loader.get_image(original_acq_id, channel)
            # Apply denoising if custom settings provided
            if denoise_settings and channel in denoise_settings:
                img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
            # Apply normalization if configured
            if normalization_method == 'channelwise_minmax':
                img = channelwise_minmax_normalize(img)
            elif normalization_method == 'arcsinh':
                img = arcsinh_normalize(img, cofactor=arcsinh_cofactor)
            elif normalization_method == 'percentile_clip':
                p_low, p_high = percentile_params
                img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
            # Ensure 0-1 range after denoising and normalization
            img = _ensure_0_1_range(img)
            cyto_imgs.append(img)
        
        # Combine cytoplasm channels
        cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights)
        cyto_img = _ensure_0_1_range(cyto_img)
        # Release intermediate images immediately to free memory
        del cyto_imgs
        gc.collect()
    return nuclear_img, cyto_img


# Memory debugging functions removed - no longer used


def _extract_original_acq_id(acq_id: str) -> str:
    """Extract original acquisition ID from unique ID format.
    
    For multi-file support, acquisition IDs may be in format:
    'slide_0_acq_0__file_e149256f' -> 'slide_0_acq_0'
    
    Args:
        acq_id: Acquisition ID (may be unique or original)
    
    Returns:
        Original acquisition ID
    """
    if '__file_' in acq_id:
        # Extract original ID by removing __file_* suffix
        return acq_id.split('__file_')[0]
    return acq_id


def segment(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    method: str,
    nuclear_channels: List[str],
    cyto_channels: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    denoise_settings: Optional[Dict] = None,
    normalization_method: str = "None",
    arcsinh_cofactor: float = 1.0,
    percentile_params: Tuple[float, float] = (1.0, 99.0),
    nuclear_combo_method: str = "mean",
    cyto_combo_method: str = "mean",
    nuclear_weights: Optional[List[float]] = None,
    cyto_weights: Optional[List[float]] = None,
    # Cellpose parameters
    cellpose_model: str = "cyto3",
    diameter: Optional[int] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gpu_id: Optional[Union[int, str]] = None,
    # CellSAM parameters
    deepcell_api_key: Optional[str] = None,
    bbox_threshold: float = 0.4,
    use_wsi: bool = False,
    low_contrast_enhancement: bool = False,
    gauge_cell_size: bool = False,
    # Watershed parameters
    min_cell_area: int = 100,
    max_cell_area: int = 10000,
    compactness: float = 0.01
) -> np.ndarray:
    """Segment cells using CellSAM, Cellpose, or Watershed method.
    
    This is the unified segmentation function used by both GUI and CLI.
    
    Args:
        loader: MCDLoader or OMETIFFLoader instance
        acquisition: AcquisitionInfo for the acquisition to segment
        method: Segmentation method ("cellsam", "cellpose", or "watershed")
        nuclear_channels: List of nuclear channel names (required)
        cyto_channels: List of cytoplasm channel names (optional, required for watershed and cyto3 model)
        output_dir: Optional directory to save mask (if None, mask is not saved)
        denoise_settings: Dictionary with denoise settings per channel (optional)
        normalization_method: Normalization method ("None", "arcsinh", "percentile_clip", "channelwise_minmax")
        arcsinh_cofactor: Arcsinh cofactor (only used if normalization_method is "arcsinh")
        percentile_params: Tuple of (low, high) percentiles for percentile_clip normalization
        nuclear_combo_method: Method to combine nuclear channels
        cyto_combo_method: Method to combine cytoplasm channels
        nuclear_weights: Optional weights for nuclear channels
        cyto_weights: Optional weights for cytoplasm channels
        cellpose_model: Cellpose model type ("cyto3" or "nuclei")
        diameter: Cell diameter in pixels (Cellpose, optional)
        flow_threshold: Flow threshold (Cellpose)
        cellprob_threshold: Cell probability threshold (Cellpose)
        gpu_id: GPU ID to use (Cellpose, optional)
        deepcell_api_key: DeepCell API key (CellSAM, optional, can use DEEPCELL_ACCESS_TOKEN env var)
        bbox_threshold: Bbox threshold for CellSAM
        use_wsi: Use WSI mode for CellSAM
        low_contrast_enhancement: Enable low contrast enhancement for CellSAM
        gauge_cell_size: Enable gauge cell size for CellSAM
        min_cell_area: Minimum cell area in pixels (watershed)
        max_cell_area: Maximum cell area in pixels (watershed)
        compactness: Watershed compactness
    
    Returns:
        Segmentation mask as numpy array (uint32)
    
    Raises:
        ValueError: If method is invalid or required channels are missing
        ImportError: If required dependencies are not installed
    """
    if cyto_channels is None:
        cyto_channels = []
    
    # Extract original acquisition ID for loader calls (handles multi-file unique IDs)
    original_acq_id = _extract_original_acq_id(acquisition.id)
    
    # Validate channels
    channels = loader.get_channels(original_acq_id)
    missing_nuclear = [ch for ch in nuclear_channels if ch not in channels]
    missing_cyto = [ch for ch in cyto_channels if ch not in channels]
    if missing_nuclear:
        raise ValueError(f"Nuclear channels not found: {missing_nuclear}")
    if missing_cyto and method not in ['watershed', 'cellsam']:
        raise ValueError(f"Cytoplasm channels not found: {missing_cyto}")
    if method == 'cellsam' and not nuclear_channels and not cyto_channels:
        raise ValueError("For CellSAM, at least one nuclear or cytoplasm channel must be specified")
    
    # Run segmentation based on method
    if method == 'cellsam':
        # Use our custom CellSAM implementation with proper model caching
        try:
            from openimc.processing.custom_cellsam import cellsam_pipeline_custom
        except (ImportError, OSError) as e:
            raise ImportError(f"CellSAM not installed or failed to load: {e}. Install with: pip install git+https://github.com/vanvalenlab/cellSAM.git")
        
        # Set API key from argument or environment variable
        api_key = deepcell_api_key or os.environ.get("DEEPCELL_ACCESS_TOKEN", "")
        if not api_key:
            raise ValueError("DeepCell API key is required for CellSAM. Set deepcell_api_key or DEEPCELL_ACCESS_TOKEN environment variable.")
        os.environ["DEEPCELL_ACCESS_TOKEN"] = api_key
        
        # Preprocess channels
        nuclear_img, cyto_img = _preprocess_channels_for_segmentation(
            loader, acquisition, nuclear_channels, cyto_channels,
            denoise_settings, normalization_method, arcsinh_cofactor,
            percentile_params, nuclear_combo_method, cyto_combo_method,
            nuclear_weights, cyto_weights
        )
        
        # Prepare input for CellSAM (supports nuclear-only, cyto-only, or combined)
        if nuclear_channels and cyto_channels:
            # Combined mode: H x W x 3 array
            h, w = nuclear_img.shape
            cellsam_input = np.zeros((h, w, 3), dtype=np.float32)
            cellsam_input[:, :, 1] = nuclear_img  # Channel 1 is nuclear
            cellsam_input[:, :, 2] = cyto_img if cyto_img is not None else nuclear_img  # Channel 2 is cyto
        elif nuclear_channels:
            # Nuclear only mode: H x W array
            cellsam_input = nuclear_img
        elif cyto_channels:
            # Cyto only mode: H x W array
            cellsam_input = cyto_img if cyto_img is not None else nuclear_img
        else:
            raise ValueError("At least one channel (nuclear or cyto) must be selected for CellSAM")
        
        # Run CellSAM pipeline using our custom implementation
        mask = cellsam_pipeline_custom(
            cellsam_input,
            bbox_threshold=bbox_threshold,
            use_wsi=use_wsi,
            low_contrast_enhancement=low_contrast_enhancement,
            gauge_cell_size=gauge_cell_size
        )
        
        # Immediately release input images to free memory
        del cellsam_input
        del nuclear_img
        if cyto_img is not None:
            del cyto_img
        gc.collect()
        
        # Use mask directly without modifications
        if isinstance(mask, np.ndarray):
            mask = mask.copy()
    
    elif method == 'cellpose':
        # Try to import Cellpose
        # Catch both ImportError and OSError (Windows DLL loading errors)
        try:
            from cellpose import models
        except (ImportError, OSError):
            raise ImportError("Cellpose not installed or failed to load. Install with: pip install cellpose")
        
        # Preprocess channels
        nuclear_img, cyto_img = _preprocess_channels_for_segmentation(
            loader, acquisition, nuclear_channels, cyto_channels,
            denoise_settings, normalization_method, arcsinh_cofactor,
            percentile_params, nuclear_combo_method, cyto_combo_method,
            nuclear_weights, cyto_weights
        )
        
        # Ensure images are in 0-1 range before passing to Cellpose
        nuclear_img = _ensure_0_1_range(nuclear_img)
        if cyto_img is not None:
            cyto_img = _ensure_0_1_range(cyto_img)
        
        # Prepare input images for Cellpose
        if cellpose_model == 'nuclei':
            # For nuclei model, use only nuclear channel
            images = [nuclear_img]
            channels_cp = [0, 0]  # [cytoplasm, nucleus] - both are nuclear channel
        else:  # cyto3
            # For cyto3 model, use both channels
            if cyto_img is None:
                cyto_img = nuclear_img  # Fallback to nuclear channel
            images = [cyto_img, nuclear_img]
            channels_cp = [0, 1]  # [cytoplasm, nucleus]
        
        # Initialize Cellpose model
        # Note: Cellpose only accepts 'gpu' (boolean), not 'device' parameter
        # Device selection is handled internally by Cellpose when gpu=True
        use_gpu = gpu_id is not None
        model = models.Cellpose(model_type=cellpose_model, gpu=use_gpu)
        
        # Run Cellpose
        masks, flows, styles, diams = model.eval(
            images,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=channels_cp
        )
        mask = masks[0]
    
    elif method == 'watershed':
        # Get image stack and channels for watershed
        # Use original_acq_id extracted above
        channels = loader.get_channels(original_acq_id)
        img_stack = loader.get_all_channels(original_acq_id)
        
        # Run watershed segmentation
        mask = watershed_segmentation(
            img_stack, channels, nuclear_channels, cyto_channels,
            denoise_settings=denoise_settings if denoise_settings else None,
            normalization_method=normalization_method,
            arcsinh_cofactor=arcsinh_cofactor,
            min_cell_area=min_cell_area,
            max_cell_area=max_cell_area,
            compactness=compactness
        )
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Save mask if output directory is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use well name if available, otherwise use acquisition name
        if acquisition.well:
            label = acquisition.well
        else:
            label = acquisition.name
        
        # Sanitize label for filename (replace invalid characters)
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
        
        # Include source file prefix to match GUI format for compatibility
        if acquisition.source_file:
            source_basename = Path(acquisition.source_file).stem
            safe_source = "".join(c if c.isalnum() or c in "._-" else "_" for c in source_basename)
            output_filename = f"{safe_source}_{safe_label}_segmentation_masks.tif"
        else:
            output_filename = f"{safe_label}_segmentation_masks.tif"
        
        output_path = output_dir / output_filename
        
        tifffile.imwrite(str(output_path), mask.astype(np.uint32), compression='lzw')
    
    return mask


def _load_masks_for_acquisitions(
    mask_path: Union[str, Path],
    acquisitions: List[AcquisitionInfo]
) -> Dict[str, np.ndarray]:
    """Load segmentation masks for acquisitions.
    
    Args:
        mask_path: Path to mask directory or single mask file
        acquisitions: List of AcquisitionInfo objects
    
    Returns:
        Dictionary mapping acquisition ID to mask array
    """
    mask_path = Path(mask_path)
    masks_dict = {}
    
    if mask_path.is_dir():
        # Directory of masks - load masks for each acquisition
        for mask_file in sorted(mask_path.glob('*.tif')) + sorted(mask_path.glob('*.tiff')) + sorted(mask_path.glob('*.npy')):
            # Try to match mask filename to acquisition
            # First try well name, then fall back to acquisition name
            mask_name = mask_file.stem
            matched = False
            # Try to find matching acquisition by well name first
            for acq in acquisitions:
                if acq.well and acq.well in mask_name:
                    if mask_file.suffix == '.npy':
                        masks_dict[acq.id] = np.load(str(mask_file))
                    else:
                        masks_dict[acq.id] = tifffile.imread(str(mask_file))
                    matched = True
                    break
            
            # If no match by well name, try acquisition name
            if not matched:
                for acq in acquisitions:
                    if acq.name in mask_name or acq.id in mask_name:
                        if mask_file.suffix == '.npy':
                            masks_dict[acq.id] = np.load(str(mask_file))
                        else:
                            masks_dict[acq.id] = tifffile.imread(str(mask_file))
                        break
    else:
        # Single mask file - use for all acquisitions
        if mask_path.suffix == '.npy':
            mask = np.load(str(mask_path))
        else:
            mask = tifffile.imread(str(mask_path))
        # Use same mask for all acquisitions
        for acq in acquisitions:
            masks_dict[acq.id] = mask
    
    return masks_dict


def _build_feature_selection_dict(
    morphological: bool = True,
    intensity: bool = True
) -> Dict[str, bool]:
    """Build feature selection dictionary.
    
    Args:
        morphological: Whether to include morphological features
        intensity: Whether to include intensity features
    
    Returns:
        Dictionary mapping feature names to True/False
    """
    selected_features = {}
    
    if morphological:
        # Add all morphological features
        selected_features.update({
            'area_um2': True,
            'perimeter_um': True,
            'equivalent_diameter_um': True,
            'eccentricity': True,
            'solidity': True,
            'extent': True,
            'circularity': True,
            'major_axis_len_um': True,
            'minor_axis_len_um': True,
            'aspect_ratio': True,
            'bbox_area_um2': True,
            'touches_border': True,
            'touches_edge': True,
            'holes_count': True,
            'centroid_x': True,
            'centroid_y': True
        })
    
    if intensity:
        # Add all intensity features
        selected_features.update({
            'mean': True,
            'median': True,
            'std': True,
            'mad': True,
            'p10': True,
            'p90': True,
            'integrated': True,
            'frac_pos': True
        })
    
    return selected_features


def extract_features(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisitions: List[AcquisitionInfo],
    mask_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    morphological: bool = True,
    intensity: bool = True,
    denoise_settings: Optional[Dict] = None,
    arcsinh: bool = False,
    arcsinh_cofactor: float = 1.0,
    spillover_config: Optional[Dict] = None,
    excluded_channels: Optional[set] = None,
    selected_features: Optional[Dict[str, bool]] = None
) -> pd.DataFrame:
    """Extract features from segmented cells.
    
    This is the unified feature extraction function used by both GUI and CLI.
    
    Args:
        loader: MCDLoader or OMETIFFLoader instance
        acquisitions: List of AcquisitionInfo objects to process
        mask_path: Path to mask directory or single mask file
        output_path: Optional path to save CSV (if None, features are not saved)
        morphological: Whether to extract morphological features
        intensity: Whether to extract intensity features
        denoise_settings: Dictionary with denoise settings per channel (optional)
        arcsinh: Whether to apply arcsinh transformation to intensity features
        arcsinh_cofactor: Arcsinh cofactor
        spillover_config: Optional spillover correction configuration
        excluded_channels: Optional set of channel names to exclude
        selected_features: Optional custom feature selection dict (overrides morphological/intensity)
    
    Returns:
        DataFrame with extracted features
    """
    # Load masks
    masks_dict = _load_masks_for_acquisitions(mask_path, acquisitions)
    
    # Build feature selection dict
    if selected_features is None:
        selected_features = _build_feature_selection_dict(morphological, intensity)
    
    all_features = []
    
    for acq in acquisitions:
        # Get mask for this acquisition
        if acq.id not in masks_dict:
            continue
        
        # Extract original acquisition ID for loader calls (handles multi-file unique IDs)
        original_acq_id = _extract_original_acq_id(acq.id)
        
        mask = masks_dict[acq.id]
        channels = loader.get_channels(original_acq_id)
        img_stack = loader.get_all_channels(original_acq_id)
        
        # Prepare acquisition info
        acq_info = {
            'channels': channels,
            'channel_metals': acq.channel_metals,
            'channel_labels': acq.channel_labels,
            'well': acq.well  # Include well for source_well column creation
        }
        
        # Extract features
        # Use well name for acquisition label if available, otherwise use acquisition name
        acq_label = acq.well if acq.well else acq.name
        features_df = extract_features_for_acquisition(
            acq.id,
            mask,
            selected_features,
            acq_info,
            acq_label,
            img_stack,
            arcsinh,
            arcsinh_cofactor,
            "custom" if denoise_settings else "None",
            denoise_settings,
            spillover_config,
            acq.source_file,
            excluded_channels
        )
        
        # Add acquisition info
        features_df['acquisition_id'] = acq.id
        features_df['acquisition_name'] = acq.name
        if acq.well:
            features_df['well'] = acq.well
        
        all_features.append(features_df)
    
    # Combine all features
    if len(all_features) > 1:
        combined_features = pd.concat(all_features, ignore_index=True)
    elif len(all_features) == 1:
        combined_features = all_features[0]
    else:
        # No features extracted
        combined_features = pd.DataFrame()

    # Enforce channel exclusion at final table level (after concat) so excluded
    # channels cannot appear as columns with NaN values.
    combined_features = drop_excluded_channel_feature_columns(combined_features, excluded_channels)
    
    # Apply arcsinh transformation to intensity features after combining all acquisitions
    # This is more efficient than applying per-acquisition and ensures consistency
    if arcsinh and not combined_features.empty:
        # Find all intensity feature columns (exclude frac_pos as it's a proportion)
        intensity_cols = [col for col in combined_features.columns 
                         if any(col.endswith(f"_{ft}") for ft in ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated'])
                         and not col.endswith('_frac_pos')]
        
        if intensity_cols:
            # Apply arcsinh to all intensity features at once
            combined_features[intensity_cols] = arcsinh_normalize(
                combined_features[intensity_cols].values, 
                cofactor=arcsinh_cofactor
            )
    
    # Save to CSV if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        combined_features.to_csv(output_path, index=False)
    
    return combined_features


def _apply_pca_to_clustering_matrix(
    data: pd.DataFrame,
    *,
    pca_mode: str = "variance",
    pca_variance: float = 0.95,
    pca_n_components: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Project a cleaned/scaled clustering matrix into PC space."""
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn is required for PCA clustering") from e

    mode = (pca_mode or "variance").strip().lower()
    if mode in {"n_components", "components", "component", "count", "number"}:
        mode = "components"
    elif mode not in {"variance", "components"}:
        raise ValueError("pca_mode must be 'variance' or 'components'")

    max_components = int(min(data.shape[0], data.shape[1]))
    if max_components < 1:
        raise ValueError("PCA requires at least one valid row and one valid feature")

    requested_variance = None
    requested_n_components = None
    if mode == "variance":
        if pca_variance is None:
            raise ValueError("pca_variance is required when pca_mode='variance'")
        requested_variance = float(pca_variance)
        if not (0.0 < requested_variance <= 1.0):
            raise ValueError("pca_variance must be greater than 0 and less than or equal to 1")
        n_components = requested_variance if requested_variance < 1.0 else max_components
    else:
        if pca_n_components is None:
            raise ValueError("pca_n_components is required when pca_mode='components'")
        requested_n_components = int(pca_n_components)
        if requested_n_components < 1:
            raise ValueError("pca_n_components must be at least 1")
        n_components = min(requested_n_components, max_components)

    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    transformed = pca.fit_transform(data.values)
    retained_components = int(transformed.shape[1])
    pc_columns = [f"PC{i}" for i in range(1, retained_components + 1)]
    pca_df = pd.DataFrame(transformed, index=data.index, columns=pc_columns)

    explained_ratios = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0)
    metadata = {
        "feature_representation": "principal_components",
        "use_pca": True,
        "pca_selection_mode": mode,
        "pca_requested_variance": requested_variance,
        "pca_requested_n_components": requested_n_components,
        "pca_n_components_retained": retained_components,
        "pca_variance_retained": float(np.sum(explained_ratios)),
        "pca_input_feature_count": int(data.shape[1]),
        "pca_max_components": max_components,
    }
    return pca_df, metadata


def _prepare_clustering_matrix(
    features_df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaling: str = "zscore",
    *,
    use_pca: bool = False,
    pca_mode: str = "variance",
    pca_variance: float = 0.95,
    pca_n_components: Optional[int] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Build the numeric matrix used by clustering algorithms."""
    import time

    t0 = time.time()
    if columns:
        cluster_columns = columns
    else:
        exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'cell_id',
                       'source_file', 'source_well', 'acquisition_label',
                       'centroid_x', 'centroid_y', 'area', 'area_um2', 'perimeter', 'perimeter_um',
                       'eccentricity', 'solidity', 'circularity', 'major_axis_length',
                       'minor_axis_length', 'orientation', 'extent', 'convex_area', 'euler_number'}
        cluster_columns = [col for col in features_df.columns if col not in exclude_cols]
    print(f"[CORE.CLUSTER] Column selection: {len(cluster_columns)} columns, took {time.time() - t0:.3f}s")

    missing = [col for col in cluster_columns if col not in features_df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    t0 = time.time()
    data = features_df[cluster_columns].copy()
    print(f"[CORE.CLUSTER] Data copy: shape={data.shape}, took {time.time() - t0:.3f}s")

    t0 = time.time()
    data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
    print(f"[CORE.CLUSTER] Handle missing/infinite: took {time.time() - t0:.3f}s")

    t0 = time.time()
    if scaling == 'zscore':
        data_means = data.mean()
        data_stds = data.std(ddof=1)
        zero_var_cols = (data_stds == 0) | data_stds.isna() | data_means.isna()
        if zero_var_cols.any():
            data.loc[:, zero_var_cols] = 0
            non_zero_var_cols = ~zero_var_cols
            if non_zero_var_cols.any():
                normalized_data = (data.loc[:, non_zero_var_cols] - data_means[non_zero_var_cols]) / data_stds[non_zero_var_cols]
                data.loc[:, non_zero_var_cols] = normalized_data
        else:
            data = (data - data_means) / data_stds
    elif scaling == 'mad':
        data_medians = data.median()
        mad_values = {}
        for col in data.columns:
            col_data = data[col].values
            median_val = data_medians[col]
            if pd.isna(median_val):
                mad_values[col] = 0.0
            else:
                mad = np.median(np.abs(col_data - median_val))
                mad_values[col] = 0.0 if pd.isna(mad) else mad

        mad_series = pd.Series(mad_values)
        zero_mad_cols = (mad_series == 0) | mad_series.isna() | data_medians.isna()
        if zero_mad_cols.any():
            data.loc[:, zero_mad_cols] = 0
            non_zero_mad_cols = ~zero_mad_cols
            if non_zero_mad_cols.any():
                for col in data.columns[non_zero_mad_cols]:
                    data[col] = (data[col] - data_medians[col]) / mad_series[col]
        else:
            for col in data.columns:
                data[col] = (data[col] - data_medians[col]) / mad_series[col]

    data = data.replace([np.inf, -np.inf], np.nan)
    print(f"[CORE.CLUSTER] Scaling complete: took {time.time() - t0:.3f}s")

    t0 = time.time()
    data = data.dropna(axis=0, how='any').dropna(axis=1, how='any')
    print(f"[CORE.CLUSTER] Dropna: shape={data.shape}, took {time.time() - t0:.3f}s")

    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Insufficient data for clustering. Need at least 2 rows and 2 columns after cleaning.")

    pca_metadata = {
        "feature_representation": "raw_features",
        "use_pca": False,
        "pca_selection_mode": None,
        "pca_requested_variance": None,
        "pca_requested_n_components": None,
        "pca_n_components_retained": None,
        "pca_variance_retained": None,
        "pca_input_feature_count": int(data.shape[1]),
        "pca_max_components": int(min(data.shape[0], data.shape[1])),
    }
    if use_pca:
        t0 = time.time()
        data, pca_metadata = _apply_pca_to_clustering_matrix(
            data,
            pca_mode=pca_mode,
            pca_variance=pca_variance,
            pca_n_components=pca_n_components,
            seed=seed,
        )
        print(
            "[CORE.CLUSTER] PCA projection: "
            f"{pca_metadata['pca_input_feature_count']} features -> "
            f"{pca_metadata['pca_n_components_retained']} PCs "
            f"({pca_metadata['pca_variance_retained']:.4f} variance), "
            f"took {time.time() - t0:.3f}s"
        )

    return data, cluster_columns, pca_metadata


def cluster(
    features_df: pd.DataFrame,
    method: str = "leiden",
    columns: Optional[List[str]] = None,
    scaling: str = "zscore",
    output_path: Optional[Union[str, Path]] = None,
    # Hierarchical parameters
    n_clusters: Optional[int] = None,
    linkage: str = "ward",
    # Leiden/Louvain parameters
    resolution: float = 1.0,
    seed: int = 42,
    n_neighbors: int = 15,  # Number of neighbors for k-NN graph
    metric: str = "euclidean",  # Distance metric for k-NN graph
    use_jaccard: bool = False,  # Use Jaccard similarity for edge weights (PhenoGraph-like)
    # K-means parameters
    n_init: int = 10,  # Number of initializations for K-means
    # HDBSCAN parameters
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",  # HDBSCAN cluster selection method
    hdbscan_metric: str = "euclidean",  # HDBSCAN distance metric
    # PCA representation parameters
    use_pca: bool = False,
    pca_mode: str = "variance",
    pca_variance: float = 0.95,
    pca_n_components: Optional[int] = None
) -> pd.DataFrame:
    """Perform clustering on feature data.
    
    This is the unified clustering function used by both GUI and CLI.
    
    Args:
        features_df: DataFrame with features to cluster
        method: Clustering method ("hierarchical", "leiden", "louvain", "kmeans", or "hdbscan")
        columns: List of column names to use for clustering (auto-detect if None)
        scaling: Scaling method ("none", "zscore", or "mad")
        output_path: Optional path to save clustered features CSV
        n_clusters: Number of clusters (required for hierarchical)
        linkage: Linkage method for hierarchical clustering ("ward", "complete", "average")
        resolution: Resolution parameter for Leiden clustering
        seed: Random seed for reproducibility
        n_neighbors: Number of neighbors for k-NN graph construction (Leiden/Louvain only, default: 15)
        metric: Distance metric for k-NN graph (Leiden/Louvain only, default: "euclidean")
        use_jaccard: Use Jaccard similarity for edge weights instead of inverse distance (PhenoGraph-like, default: False)
        n_init: Number of initializations for K-means (default: 10)
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 10)
        min_samples: Minimum samples for HDBSCAN (default: 5)
        cluster_selection_method: Cluster selection method for HDBSCAN ("eom" or "leaf", default: "eom")
        hdbscan_metric: Distance metric for HDBSCAN (default: "euclidean")
        use_pca: If True, cluster on principal components instead of raw/scaled features
        pca_mode: PCA selection mode ("variance" or "components")
        pca_variance: Proportion of variance to retain when pca_mode="variance"
        pca_n_components: Number of PCs to retain when pca_mode="components"
    
    Returns:
        DataFrame with cluster labels added in 'cluster' column
    
    Raises:
        ValueError: If method is invalid or required parameters are missing
    """
    import time
    import random
    t_start = time.time()
    print(f"[CORE.CLUSTER] Starting clustering: method={method}, input shape={features_df.shape}")
    # Set explicit seeds once at function entry for reproducible behavior.
    random.seed(seed)
    np.random.seed(seed)
    
    data, cluster_columns, pca_metadata = _prepare_clustering_matrix(
        features_df,
        columns=columns,
        scaling=scaling,
        use_pca=use_pca,
        pca_mode=pca_mode,
        pca_variance=pca_variance,
        pca_n_components=pca_n_components,
        seed=seed,
    )
    
    # Store original indices to map back
    original_indices = data.index
    data_values = data.values
    print(f"[CORE.CLUSTER] Final data shape: {data_values.shape} (n_cells={data_values.shape[0]}, n_features={data_values.shape[1]})")
    
    # Perform clustering
    t_cluster_start = time.time()
    if method == 'hierarchical':
        from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
        from scipy.spatial.distance import pdist
        
        n = data_values.shape[0]
        print(f"[CORE.CLUSTER] Hierarchical: Starting with {n} cells, linkage={linkage}")
        
        t0 = time.time()
        # Calculate distance matrix (efficient condensed form)
        distances = pdist(data_values, metric='euclidean')
        print(f"[CORE.CLUSTER] Hierarchical: pdist took {time.time() - t0:.3f}s (distance array size: {len(distances)})")
        
        t0 = time.time()
        # Perform linkage
        linkage_matrix = scipy_linkage(distances, method=linkage)
        print(f"[CORE.CLUSTER] Hierarchical: linkage took {time.time() - t0:.3f}s")
        
        # Get cluster labels
        if n_clusters is None:
            raise ValueError("n_clusters is required for hierarchical clustering")
        t0 = time.time()
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        print(f"[CORE.CLUSTER] Hierarchical: fcluster took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Hierarchical: Found {len(np.unique(cluster_labels))} clusters")
    
    elif method == 'leiden':
        import igraph as ig
        import leidenalg
        
        # Use k-NN graph (much faster than fully connected graph)
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError("scikit-learn is required for Leiden clustering (k-NN graph construction)")
        
        n = data_values.shape[0]
        print(f"[CORE.CLUSTER] Leiden: Building k-NN graph with {n} nodes, k={n_neighbors}, metric={metric}")
        
        t0 = time.time()
        # Build k-NN graph using sklearn (matching old GUI implementation)
        try:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=1).fit(data_values)
        except TypeError:
            # Backward compatibility with older sklearn versions without n_jobs arg.
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data_values)
        distances_knn, indices_knn = nbrs.kneighbors(data_values)
        print(f"[CORE.CLUSTER] Leiden: k-NN calculation took {time.time() - t0:.3f}s")
        
        t0 = time.time()
        # Create graph from k-NN (matching old GUI implementation)
        edge_weights = {}
        directed_edge_count = 0
        
        if use_jaccard:
            # Compute neighbor sets for Jaccard similarity (PhenoGraph-like)
            # Each node's neighbor set includes itself and its k-nearest neighbors
            neighbor_sets = [set(indices_knn[i]) | {i} for i in range(n)]
            print(f"[CORE.CLUSTER] Leiden: Computed neighbor sets for Jaccard weighting")
            
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices_knn[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        directed_edge_count += 1
                        # Compute Jaccard similarity: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
                        intersection = len(neighbor_sets[i] & neighbor_sets[neighbor_idx])
                        union = len(neighbor_sets[i] | neighbor_sets[neighbor_idx])
                        jaccard = intersection / union if union > 0 else 0.0
                        u = int(i)
                        v = int(neighbor_idx)
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        prev = edge_weights.get(key)
                        if prev is None or jaccard > prev:
                            edge_weights[key] = float(jaccard)
        else:
            # Use inverse distance weighting (default)
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices_knn[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        directed_edge_count += 1
                        # Convert distance to similarity (inverse, normalized) - matching old GUI
                        weight = 1.0 / (1.0 + distances_knn[i][j_idx])
                        u = int(i)
                        v = int(neighbor_idx)
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        prev = edge_weights.get(key)
                        if prev is None or weight > prev:
                            edge_weights[key] = float(weight)
        
        print(f"[CORE.CLUSTER] Leiden: Edge list creation took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Leiden: Created {directed_edge_count} directed edges from k-NN")
        
        t0 = time.time()
        # Convert to a deterministic undirected edge representation.
        symmetric_edges = list(edge_weights.keys())
        symmetric_weights = list(edge_weights.values())
        
        print(f"[CORE.CLUSTER] Leiden: Symmetric graph conversion took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Leiden: Final graph has {len(symmetric_edges)} unique edges")
        
        t0 = time.time()
        # Create igraph
        g = ig.Graph(n)
        g.add_edges(symmetric_edges)
        g.es['weight'] = symmetric_weights
        print(f"[CORE.CLUSTER] Leiden: Graph creation took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Leiden: Graph has {g.vcount()} vertices, {g.ecount()} edges")
        
        t0 = time.time()
        # Run Leiden clustering (matching GUI)
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            seed=seed,
        )
        print(f"[CORE.CLUSTER] Leiden: find_partition took {time.time() - t0:.3f}s")
        cluster_labels = np.array(partition.membership) + 1  # Start from 1 (matching GUI)
        print(f"[CORE.CLUSTER] Leiden: Found {len(np.unique(cluster_labels))} clusters")
    
    elif method == 'louvain':
        import igraph as ig
        import leidenalg
        
        # Use k-NN graph (same as Leiden, but with modularity optimization)
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError("scikit-learn is required for Louvain clustering (k-NN graph construction)")
        
        n = data_values.shape[0]
        print(f"[CORE.CLUSTER] Louvain: Building k-NN graph with {n} nodes, k={n_neighbors}, metric={metric}")
        
        t0 = time.time()
        # Build k-NN graph using sklearn (same as Leiden)
        try:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=1).fit(data_values)
        except TypeError:
            # Backward compatibility with older sklearn versions without n_jobs arg.
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data_values)
        distances_knn, indices_knn = nbrs.kneighbors(data_values)
        print(f"[CORE.CLUSTER] Louvain: k-NN calculation took {time.time() - t0:.3f}s")
        
        t0 = time.time()
        # Create graph from k-NN
        edge_weights = {}
        directed_edge_count = 0
        
        if use_jaccard:
            # Compute neighbor sets for Jaccard similarity (PhenoGraph-like)
            # Each node's neighbor set includes itself and its k-nearest neighbors
            neighbor_sets = [set(indices_knn[i]) | {i} for i in range(n)]
            print(f"[CORE.CLUSTER] Louvain: Computed neighbor sets for Jaccard weighting")
            
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices_knn[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        directed_edge_count += 1
                        # Compute Jaccard similarity: |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
                        intersection = len(neighbor_sets[i] & neighbor_sets[neighbor_idx])
                        union = len(neighbor_sets[i] | neighbor_sets[neighbor_idx])
                        jaccard = intersection / union if union > 0 else 0.0
                        u = int(i)
                        v = int(neighbor_idx)
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        prev = edge_weights.get(key)
                        if prev is None or jaccard > prev:
                            edge_weights[key] = float(jaccard)
        else:
            # Use inverse distance weighting (default)
            for i in range(n):
                for j_idx, neighbor_idx in enumerate(indices_knn[i]):
                    if neighbor_idx != i:  # Don't add self-loops
                        directed_edge_count += 1
                        # Convert distance to similarity (inverse, normalized)
                        weight = 1.0 / (1.0 + distances_knn[i][j_idx])
                        u = int(i)
                        v = int(neighbor_idx)
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        prev = edge_weights.get(key)
                        if prev is None or weight > prev:
                            edge_weights[key] = float(weight)
        
        print(f"[CORE.CLUSTER] Louvain: Edge list creation took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Louvain: Created {directed_edge_count} directed edges from k-NN")
        
        t0 = time.time()
        # Convert to a deterministic undirected edge representation.
        symmetric_edges = list(edge_weights.keys())
        symmetric_weights = list(edge_weights.values())
        
        print(f"[CORE.CLUSTER] Louvain: Symmetric graph conversion took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Louvain: Final graph has {len(symmetric_edges)} unique edges")
        
        t0 = time.time()
        # Create igraph
        g = ig.Graph(n)
        g.add_edges(symmetric_edges)
        g.es['weight'] = symmetric_weights
        print(f"[CORE.CLUSTER] Louvain: Graph creation took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Louvain: Graph has {g.vcount()} vertices, {g.ecount()} edges")
        
        t0 = time.time()
        # Run Louvain clustering (modularity optimization)
        partition = leidenalg.find_partition(
            g,
            leidenalg.ModularityVertexPartition,
            weights='weight',
            seed=seed,
        )
        print(f"[CORE.CLUSTER] Louvain: find_partition took {time.time() - t0:.3f}s")
        cluster_labels = np.array(partition.membership) + 1  # Start from 1 (matching GUI)
        print(f"[CORE.CLUSTER] Louvain: Found {len(np.unique(cluster_labels))} clusters")
    
    elif method == 'kmeans':
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn is required for K-means clustering")
        
        n = data_values.shape[0]
        print(f"[CORE.CLUSTER] K-means: Starting with {n} cells, n_clusters={n_clusters}, n_init={n_init}")
        
        if n_clusters is None:
            raise ValueError("n_clusters is required for K-means clustering")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        t0 = time.time()
        # Use efficient K-means implementation (n_init=10 is good balance)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init, algorithm='lloyd')
        cluster_labels = kmeans.fit_predict(data_values)
        print(f"[CORE.CLUSTER] K-means: fit_predict took {time.time() - t0:.3f}s")
        
        # Convert to 1-based labels (matching GUI)
        cluster_labels = cluster_labels + 1
        print(f"[CORE.CLUSTER] K-means: Found {len(np.unique(cluster_labels))} clusters")
        print(f"[CORE.CLUSTER] K-means: Inertia: {kmeans.inertia_:.2f}")
    
    elif method == 'hdbscan':
        import hdbscan
        
        n = data_values.shape[0]
        print(f"[CORE.CLUSTER] HDBSCAN: Starting with {n} cells")
        print(f"[CORE.CLUSTER] HDBSCAN: Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"[CORE.CLUSTER] HDBSCAN: cluster_selection_method={cluster_selection_method}, metric={hdbscan_metric}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        t0 = time.time()
        # Create HDBSCAN clusterer (efficient implementation)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=hdbscan_metric,
            core_dist_n_jobs=1  # Use single thread for stability
        )
        cluster_labels = clusterer.fit_predict(data_values)
        print(f"[CORE.CLUSTER] HDBSCAN: fit_predict took {time.time() - t0:.3f}s")
        
        # HDBSCAN uses -1 for noise, convert to 1-based (matching GUI)
        n_noise = (cluster_labels == -1).sum()
        cluster_labels = cluster_labels + 1  # -1 becomes 0, others become 1-based
        n_clusters_found = len(np.unique(cluster_labels[cluster_labels > 0]))
        print(f"[CORE.CLUSTER] HDBSCAN: Found {n_clusters_found} clusters, {n_noise} noise points")
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    print(f"[CORE.CLUSTER] Clustering algorithm took {time.time() - t_cluster_start:.3f}s total")
    
    # Map cluster labels back to original dataframe indices
    t0 = time.time()
    # Create a series with cluster labels for the cleaned data
    cluster_series = pd.Series(cluster_labels, index=original_indices)
    
    # Add cluster labels to original dataframe (NaN for rows that were dropped)
    result_df = features_df.copy()
    result_df['cluster'] = cluster_series
    # Fill NaN with 0 (noise/unassigned) if needed
    result_df['cluster'] = result_df['cluster'].fillna(0).astype(int)
    result_df.attrs["pca_metadata"] = pca_metadata
    print(f"[CORE.CLUSTER] Mapping labels back took {time.time() - t0:.3f}s")
    
    # Save output if path is provided
    if output_path is not None:
        t0 = time.time()
        output_path = Path(output_path)
        result_df.to_csv(output_path, index=False)
        print(f"[CORE.CLUSTER] Saving output took {time.time() - t0:.3f}s")
    
    print(f"[CORE.CLUSTER] Total clustering time: {time.time() - t_start:.3f}s")
    return result_df


def build_spatial_graph(
    features_df: pd.DataFrame,
    method: str = "kNN",
    k_neighbors: int = 6,
    radius: Optional[float] = None,
    pixel_size_um: float = 1.0,
    roi_column: Optional[str] = None,
    detect_communities: bool = False,
    community_seed: int = 42,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Build spatial graph from cell centroids.
    
    This function creates a spatial graph connecting cells based on their
    spatial proximity. It supports kNN, radius-based, and Delaunay triangulation
    methods. The graph can be built per-ROI (if roi_column is provided) or globally.
    
    Args:
        features_df: DataFrame with cell features, must contain 'centroid_x' and 'centroid_y'
        method: Graph construction method ('kNN', 'Radius', or 'Delaunay')
        k_neighbors: Number of neighbors for kNN method
        radius: Radius in pixels for radius-based method (required if method='Radius')
        pixel_size_um: Pixel size in micrometers for distance conversion
        roi_column: Column name for ROI grouping (e.g., 'acquisition_id'). If None, builds global graph
        detect_communities: Whether to detect communities using Leiden algorithm
        community_seed: Random seed for community detection
        output_path: Optional path to save edges CSV file
    
    Returns:
        Tuple of (edges_df, features_with_communities_df)
        - edges_df: DataFrame with columns ['roi_id', 'cell_id_A', 'cell_id_B', 'distance_um'] (or ['source', 'target', 'distance', 'distance_um'] for global)
        - features_with_communities_df: DataFrame with 'spatial_community' column if detect_communities=True, else None
    """
    # Validate required columns
    required_cols = ['centroid_x', 'centroid_y']
    missing = [col for col in required_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Required columns for spatial analysis: {missing}")
    
    # Validate method
    if method not in ['kNN', 'Radius', 'Delaunay']:
        raise ValueError(f"Unknown graph method: {method}. Must be 'kNN', 'Radius', or 'Delaunay'")
    
    if method == 'Radius' and radius is None:
        raise ValueError("radius parameter is required for 'Radius' method")
    
    # Determine ROI column
    if roi_column is None:
        # Try to auto-detect ROI column
        for col in ['acquisition_id', 'roi_id', 'roi']:
            if col in features_df.columns:
                roi_column = col
                break
    
    # Build graph per ROI if roi_column exists, otherwise build globally
    if roi_column and roi_column in features_df.columns:
        return _build_spatial_graph_per_roi(
            features_df, method, k_neighbors, radius, pixel_size_um,
            roi_column, detect_communities, community_seed, output_path
        )
    else:
        return _build_spatial_graph_global(
            features_df, method, k_neighbors, radius, pixel_size_um,
            detect_communities, community_seed, output_path
        )


def _build_spatial_graph_per_roi(
    features_df: pd.DataFrame,
    method: str,
    k_neighbors: int,
    radius: Optional[float],
    pixel_size_um: float,
    roi_column: str,
    detect_communities: bool,
    community_seed: int,
    output_path: Optional[Union[str, Path]]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Build spatial graph per ROI."""
    edge_records = []
    
    for roi_id, roi_df in features_df.groupby(roi_column):
        roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])
        if roi_df.empty:
            continue
        
        coords_px = roi_df[["centroid_x", "centroid_y"]].values
        cell_ids = roi_df["cell_id"].values if 'cell_id' in roi_df.columns else roi_df.index.values
        
        # Build spatial tree
        tree = cKDTree(coords_px)
        
        # Use set to deduplicate edges
        roi_edges_set = set()
        
        if method == "kNN":
            # Query k+1 (including self), exclude self idx 0
            query_k = min(k_neighbors + 1, max(2, len(coords_px)))
            dists, idxs = tree.query(coords_px, k=query_k)
            
            # Handle scalar case
            if np.isscalar(dists):
                dists = np.array([[dists]])
                idxs = np.array([[idxs]])
            elif dists.ndim == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            
            for i in range(len(coords_px)):
                src_cell_id = int(cell_ids[i])
                for j in range(1, min(dists.shape[1], k_neighbors + 1)):
                    nbr_idx = int(idxs[i, j])
                    if nbr_idx < 0 or nbr_idx >= len(coords_px):
                        continue
                    dst_cell_id = int(cell_ids[nbr_idx])
                    dist_px = float(dists[i, j])
                    dist_um = dist_px * pixel_size_um
                    
                    # Create canonical edge (smaller cell_id first)
                    edge_key = (min(src_cell_id, dst_cell_id), max(src_cell_id, dst_cell_id))
                    if edge_key not in roi_edges_set:
                        roi_edges_set.add(edge_key)
                        edge_records.append({
                            'roi_id': str(roi_id),
                            'cell_id_A': src_cell_id,
                            'cell_id_B': dst_cell_id,
                            'distance_um': dist_um
                        })
        
        elif method == "Radius":
            # Radius graph: radius is in pixels
            radius_px = radius
            pairs = tree.query_pairs(r=radius_px)
            
            for i, j in pairs:
                a_id = int(cell_ids[int(i)])
                b_id = int(cell_ids[int(j)])
                
                # Create canonical edge (smaller cell_id first)
                edge_key = (min(a_id, b_id), max(a_id, b_id))
                if edge_key not in roi_edges_set:
                    roi_edges_set.add(edge_key)
                    dist_um = float(np.linalg.norm(coords_px[int(i)] - coords_px[int(j)])) * pixel_size_um
                    edge_records.append({
                        'roi_id': str(roi_id),
                        'cell_id_A': a_id,
                        'cell_id_B': b_id,
                        'distance_um': dist_um
                    })
        
        elif method == "Delaunay":
            # Delaunay triangulation
            tri = Delaunay(coords_px)
            edges_set = set()
            
            for simplex in tri.simplices:
                # Each simplex has 3 vertices, create edges between all pairs
                for i in range(3):
                    for j in range(i + 1, 3):
                        v1, v2 = simplex[i], simplex[j]
                        # Create canonical edge (smaller index first)
                        edge_key = (min(v1, v2), max(v1, v2))
                        if edge_key not in edges_set:
                            edges_set.add(edge_key)
                            a_id = int(cell_ids[v1])
                            b_id = int(cell_ids[v2])
                            dist_um = float(np.linalg.norm(coords_px[v1] - coords_px[v2])) * pixel_size_um
                            edge_records.append({
                                'roi_id': str(roi_id),
                                'cell_id_A': a_id,
                                'cell_id_B': b_id,
                                'distance_um': dist_um
                            })
    
    # Create edges dataframe
    edges_df = pd.DataFrame(edge_records)
    
    # Detect communities if requested
    features_with_communities = None
    if detect_communities:
        features_with_communities = _detect_spatial_communities(
            features_df, edges_df, roi_column, pixel_size_um, community_seed
        )
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        edges_df.to_csv(output_path, index=False)
        if features_with_communities is not None:
            community_output = output_path.parent / (output_path.stem + '_with_communities.csv')
            features_with_communities.to_csv(community_output, index=False)
    
    return edges_df, features_with_communities


def _build_spatial_graph_global(
    features_df: pd.DataFrame,
    method: str,
    k_neighbors: int,
    radius: Optional[float],
    pixel_size_um: float,
    detect_communities: bool,
    community_seed: int,
    output_path: Optional[Union[str, Path]]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Build spatial graph globally (single ROI)."""
    coords = features_df[['centroid_x', 'centroid_y']].dropna().values
    
    # Build spatial tree
    tree = cKDTree(coords)
    
    # Build edges
    edge_records = []
    edge_set = set()
    
    if method == "kNN":
        query_k = min(k_neighbors + 1, max(2, len(coords)))
        dists, idxs = tree.query(coords, k=query_k)
        
        # Handle scalar case
        if np.isscalar(dists):
            dists = np.array([[dists]])
            idxs = np.array([[idxs]])
        elif dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        
        for i in range(len(coords)):
            for j in range(1, min(dists.shape[1], k_neighbors + 1)):
                nbr_idx = int(idxs[i, j])
                if nbr_idx < 0 or nbr_idx >= len(coords):
                    continue
                dist_px = float(dists[i, j])
                dist_um = dist_px * pixel_size_um
                
                edge_key = (min(i, nbr_idx), max(i, nbr_idx))
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    edge_records.append({
                        'source': i,
                        'target': nbr_idx,
                        'distance': dist_px,  # Keep in pixels for compatibility
                        'distance_um': dist_um
                    })
    
    elif method == "Radius":
        radius_px = radius
        pairs = tree.query_pairs(r=radius_px)
        
        for i, j in pairs:
            edge_key = (min(i, j), max(i, j))
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                dist_px = float(np.linalg.norm(coords[i] - coords[j]))
                dist_um = dist_px * pixel_size_um
                edge_records.append({
                    'source': i,
                    'target': j,
                    'distance': dist_px,
                    'distance_um': dist_um
                })
    
    elif method == "Delaunay":
        tri = Delaunay(coords)
        edges_set = set()
        
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    v1, v2 = simplex[i], simplex[j]
                    edge_key = (min(v1, v2), max(v1, v2))
                    if edge_key not in edges_set:
                        edges_set.add(edge_key)
                        dist_px = float(np.linalg.norm(coords[v1] - coords[v2]))
                        dist_um = dist_px * pixel_size_um
                        edge_records.append({
                            'source': int(v1),
                            'target': int(v2),
                            'distance': dist_px,
                            'distance_um': dist_um
                        })
    
    # Create edges dataframe
    edges_df = pd.DataFrame(edge_records)
    
    # Detect communities if requested
    features_with_communities = None
    if detect_communities:
        features_with_communities = _detect_spatial_communities_global(
            features_df, edges_df, pixel_size_um, community_seed
        )
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        edges_df.to_csv(output_path, index=False)
        if features_with_communities is not None:
            community_output = output_path.parent / (output_path.stem + '_with_communities.csv')
            features_with_communities.to_csv(community_output, index=False)
    
    return edges_df, features_with_communities


def _detect_spatial_communities(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    roi_column: str,
    pixel_size_um: float,
    seed: int
) -> pd.DataFrame:
    """Detect communities using Leiden algorithm for per-ROI graphs."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError("igraph and leidenalg are required for community detection")
    
    # Build graph from edges
    # Map cell IDs to indices if needed
    if 'cell_id_A' in edges_df.columns and 'cell_id' in features_df.columns:
        # Create mapping from cell_id to index
        cell_id_to_idx = {cell_id: idx for idx, cell_id in enumerate(features_df['cell_id'].values)}
        edge_list = []
        weights = []
        for _, e in edges_df.iterrows():
            cell_a = int(e['cell_id_A'])
            cell_b = int(e['cell_id_B'])
            if cell_a in cell_id_to_idx and cell_b in cell_id_to_idx:
                edge_list.append((cell_id_to_idx[cell_a], cell_id_to_idx[cell_b]))
                dist_um = e.get('distance_um', e.get('distance', 1.0) * pixel_size_um)
                weights.append(1.0 / (dist_um + 1e-6))
        g = ig.Graph(len(features_df))
        g.add_edges(edge_list)
        g.es['weight'] = weights
    else:
        # Use index-based edges
        g = ig.Graph()
        g.add_vertices(len(features_df))
        edge_list = []
        weights = []
        for _, e in edges_df.iterrows():
            source = int(e.get('source', e.get('cell_id_A', 0)))
            target = int(e.get('target', e.get('cell_id_B', 0)))
            edge_list.append((source, target))
            dist_um = e.get('distance_um', e.get('distance', 1.0) * pixel_size_um)
            weights.append(1.0 / (dist_um + 1e-6))
        g.add_edges(edge_list)
        g.es['weight'] = weights
    
    # Run community detection with seed
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight', seed=seed)
    communities = partition.membership
    
    # Map community labels back to dataframe
    result_df = features_df.copy()
    if 'cell_id_A' in edges_df.columns and 'cell_id' in features_df.columns:
        # Map from graph vertex index to cell_id, then to dataframe index
        idx_to_cell_id = {idx: cell_id for idx, cell_id in enumerate(features_df['cell_id'].values)}
        community_series = pd.Series(index=features_df.index, dtype=int)
        for vertex_idx, community in enumerate(communities):
            if vertex_idx < len(features_df):
                community_series.iloc[vertex_idx] = community
        result_df['spatial_community'] = community_series
    else:
        # Direct mapping (vertex index = dataframe index)
        result_df['spatial_community'] = communities[:len(features_df)]
    
    return result_df


def _detect_spatial_communities_global(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    pixel_size_um: float,
    seed: int
) -> pd.DataFrame:
    """Detect communities using Leiden algorithm for global graphs."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError("igraph and leidenalg are required for community detection")
    
    # Build graph from edges
    g = ig.Graph()
    g.add_vertices(len(features_df))
    edge_list = []
    weights = []
    for _, e in edges_df.iterrows():
        source = int(e.get('source', 0))
        target = int(e.get('target', 0))
        edge_list.append((source, target))
        dist_um = e.get('distance_um', e.get('distance', 1.0) * pixel_size_um)
        weights.append(1.0 / (dist_um + 1e-6))
    g.add_edges(edge_list)
    g.es['weight'] = weights
    
    # Run community detection with seed
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight', seed=seed)
    communities = partition.membership
    
    # Map community labels back to dataframe
    result_df = features_df.copy()
    result_df['spatial_community'] = communities[:len(features_df)]
    
    return result_df


def batch_correction(
    features_df: pd.DataFrame,
    method: str = "harmony",
    batch_var: Optional[str] = None,
    features: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    # ComBat parameters
    covariates: Optional[List[str]] = None,
    # Harmony parameters
    n_clusters: int = 30,
    sigma: float = 0.1,
    theta: float = 2.0,
    lambda_reg: float = 1.0,
    max_iter: int = 20,
    pca_variance: float = 0.9
) -> pd.DataFrame:
    """Apply batch correction to feature data.
    
    This function applies batch correction using ComBat or Harmony to remove
    technical variation (batch effects) between different files or batches.
    
    Args:
        features_df: DataFrame with cell features
        method: Batch correction method ('combat' or 'harmony')
        batch_var: Column name containing batch identifiers. If None, auto-detects
        features: List of feature column names to correct. If None, auto-detects
        output_path: Optional path to save corrected features CSV
        covariates: Optional list of covariate column names (ComBat only)
        n_clusters: Number of Harmony clusters (default: 30)
        sigma: Width of soft kmeans clusters for Harmony (default: 0.1)
        theta: Diversity clustering penalty parameter for Harmony (default: 2.0)
        lambda_reg: Regularization parameter for Harmony (default: 1.0)
        max_iter: Maximum number of iterations for Harmony (default: 20)
        pca_variance: Proportion of variance to retain in PCA for Harmony (default: 0.9)
    
    Returns:
        DataFrame with corrected features (all original columns preserved)
    """
    # Auto-detect batch variable if not provided
    if batch_var is None:
        batch_var = detect_batch_variable(features_df)
        if batch_var is None:
            raise ValueError(
                "No batch variable found. Please specify batch_var or ensure "
                "dataframe contains 'source_file' or 'acquisition_id' column."
            )
    
    # Auto-detect features if not provided
    if features is None:
        features = get_feature_columns_from_dataframe(features_df, batch_var=batch_var)
        if not features:
            raise ValueError("No features found to correct. Please specify features.")
    
    # Validate inputs
    validate_batch_correction_inputs(features_df, batch_var, features)
    
    # Apply correction based on method
    if method.lower() == 'combat':
        corrected_df = apply_combat_correction(
            features_df,
            batch_var,
            features,
            covariates=covariates
        )
    elif method.lower() == 'harmony':
        corrected_df = apply_harmony_correction(
            features_df,
            batch_var,
            features,
            n_clusters=n_clusters,
            sigma=sigma,
            theta=theta,
            lambda_reg=lambda_reg,
            max_iter=max_iter,
            pca_variance=pca_variance
        )
    else:
        raise ValueError(f"Unknown batch correction method: {method}. Must be 'combat' or 'harmony'")
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        corrected_df.to_csv(output_path, index=False)
    
    return corrected_df


def pixel_correlation(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    channels: List[str],
    mask: Optional[np.ndarray] = None,
    multiple_testing_correction: Optional[str] = None
) -> pd.DataFrame:
    """Compute pixel-level correlations between marker pairs.
    
    This function computes Spearman correlation coefficients for all pairs of
    markers at the pixel level. Can analyze within cell masks or entire ROI.
    
    Args:
        loader: Data loader (MCDLoader or OMETIFFLoader)
        acquisition: Acquisition information
        channels: List of channel names to analyze
        mask: Optional segmentation mask. If provided, only pixels within cells are analyzed
        multiple_testing_correction: Optional correction method ('bonferroni', 'fdr_bh', etc.)
            If provided, applies correction to p-values
    
    Returns:
        DataFrame with columns: marker1, marker2, correlation, p_value, n_pixels
    """
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    
    # Extract original acquisition ID for loader calls (handles multi-file unique IDs)
    original_acq_id = _extract_original_acq_id(acquisition.id)
    
    # Load image stack for all channels
    img_stack = loader.get_all_channels(original_acq_id)
    
    # Determine shape - loaders return HWC format (H, W, C)
    if img_stack.ndim == 3:
        height, width, n_channels = img_stack.shape
    elif img_stack.ndim == 2:
        # Single channel
        return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])
    else:
        return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])
    
    # Ensure we have the right number of channels
    if len(channels) != n_channels:
        if len(channels) > n_channels:
            channels = channels[:n_channels]
    
    all_channels = loader.get_channels(original_acq_id)
    selected_pairs = []
    for channel in channels:
        if channel not in all_channels:
            continue
        channel_idx = all_channels.index(channel)
        if channel_idx >= n_channels:
            continue
        selected_pairs.append((channel, channel_idx))

    if len(selected_pairs) < 2:
        return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])

    selected_channels = [channel for channel, _ in selected_pairs]
    selected_indices = [channel_idx for _, channel_idx in selected_pairs]
    pixel_matrix = img_stack[:, :, selected_indices].reshape(-1, len(selected_indices)).astype(np.float64, copy=False)

    if mask is not None and mask.shape == img_stack.shape[:2]:
        pixel_matrix = pixel_matrix[mask.reshape(-1) > 0]

    if pixel_matrix.size == 0:
        return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])

    finite_matrix = np.isfinite(pixel_matrix)
    if not np.all(finite_matrix):
        # Preserve all valid observations for each marker pair. Requiring rows
        # to be finite in every selected channel lets an unrelated channel's
        # missing value change another pair's coefficient and sample size.
        correlations = []
        for i, ch1 in enumerate(selected_channels):
            for j in range(i + 1, len(selected_channels)):
                ch2 = selected_channels[j]
                pair_valid = finite_matrix[:, i] & finite_matrix[:, j]
                n_pair_pixels = int(np.count_nonzero(pair_valid))
                if n_pair_pixels < 3:
                    continue
                pair_result = spearmanr(
                    pixel_matrix[pair_valid, i],
                    pixel_matrix[pair_valid, j],
                )
                corr_value = float(getattr(pair_result, 'statistic', pair_result[0]))
                p_value = float(getattr(pair_result, 'pvalue', pair_result[1]))
                if not np.isfinite(corr_value):
                    continue
                correlations.append({
                    'marker1': ch1,
                    'marker2': ch2,
                    'correlation': corr_value,
                    'p_value': p_value,
                    'n_pixels': n_pair_pixels,
                })
    else:
        n_pixels = int(pixel_matrix.shape[0])
        if n_pixels < 3:
            return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])

        # Compute correlations for complete data in one vectorized call.
        corr_result = spearmanr(pixel_matrix, axis=0)
        corr_matrix = getattr(corr_result, 'statistic', corr_result[0] if isinstance(corr_result, tuple) else corr_result)
        p_matrix = getattr(corr_result, 'pvalue', corr_result[1] if isinstance(corr_result, tuple) else None)
        corr_matrix = np.asarray(corr_matrix, dtype=float)
        p_matrix = np.asarray(p_matrix, dtype=float) if p_matrix is not None else np.full_like(corr_matrix, np.nan, dtype=float)

        if corr_matrix.ndim == 0 and len(selected_channels) == 2:
            corr_value = float(corr_matrix)
            p_value = float(p_matrix) if np.ndim(p_matrix) == 0 else np.nan
            correlations = [{
                'marker1': selected_channels[0],
                'marker2': selected_channels[1],
                'correlation': corr_value,
                'p_value': p_value,
                'n_pixels': n_pixels,
            }] if np.isfinite(corr_value) else []
        else:
            correlations = []
            if corr_matrix.ndim != 2:
                return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])
            for i, ch1 in enumerate(selected_channels):
                for j in range(i + 1, len(selected_channels)):
                    corr_coef = corr_matrix[i, j]
                    p_value = p_matrix[i, j] if p_matrix.ndim == 2 else np.nan
                    if not np.isfinite(corr_coef):
                        continue
                    correlations.append({
                        'marker1': ch1,
                        'marker2': selected_channels[j],
                        'correlation': float(corr_coef),
                        'p_value': float(p_value),
                        'n_pixels': n_pixels,
                    })

    # Create results dataframe
    if not correlations:
        return pd.DataFrame(columns=['marker1', 'marker2', 'correlation', 'p_value', 'n_pixels'])
    
    results_df = pd.DataFrame(correlations)
    
    # Apply multiple testing correction if requested
    if multiple_testing_correction and len(results_df) > 0:
        p_values = results_df['p_value'].values
        try:
            _, p_corrected, _, _ = multipletests(p_values, method=multiple_testing_correction)
            results_df['p_value_corrected'] = p_corrected
        except Exception:
            # If correction fails, just continue without it
            pass
    
    return results_df


def qc_analysis(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    channels: List[str],
    mode: str = "pixel",
    mask: Optional[np.ndarray] = None,
    denoise_settings: Optional[Dict[str, Dict[str, dict]]] = None,
    cell_signal_method: Literal["positive_pixels", "upper_quantile", "all_cell_mean"] = "positive_pixels",
    positive_threshold_sd: float = 2.0,
    upper_quantile: float = 0.90,
) -> pd.DataFrame:
    """Perform quality control analysis on IMC data.
    
    This function calculates QC metrics including SNR (Signal-to-Noise Ratio),
    intensity statistics, and coverage metrics. Can analyze at pixel level or
    cell level (if mask is provided).
    
    Args:
        loader: Data loader (MCDLoader or OMETIFFLoader)
        acquisition: Acquisition information
        channels: List of channel names to analyze
        mode: Analysis mode ('pixel' or 'cell')
        mask: Optional segmentation mask (required for 'cell' mode)
        denoise_settings: Optional per-channel denoise settings applied before QC
        cell_signal_method: Cell-mode signal selection method
        positive_threshold_sd: Number of robust background SDs above background_mean
            used for the positive-pixel threshold method
        upper_quantile: Quantile in (0, 1] used for the upper-quantile cell method
    
    Returns:
        DataFrame with QC metrics per channel
    """
    # Extract original acquisition ID for loader calls (handles multi-file unique IDs)
    original_acq_id = _extract_original_acq_id(acquisition.id)
    
    # Optional scikit-image for Otsu thresholding
    try:
        from skimage.filters import threshold_otsu
        _HAVE_SCIKIT_IMAGE = True
    except ImportError:
        _HAVE_SCIKIT_IMAGE = False
    
    if mode not in {"pixel", "cell"}:
        raise ValueError("mode must be either 'pixel' or 'cell'")
    cell_signal_method = _validate_qc_cell_signal_method(cell_signal_method)
    if mode == "cell":
        if mask is None:
            raise ValueError("mask is required for cell-level QC")
        if np.asarray(mask).ndim != 2:
            raise ValueError("mask must be a two-dimensional label image")
        if not np.any(mask > 0):
            raise ValueError("mask must contain at least one labeled cell")
        if not np.any(mask == 0):
            raise ValueError("cell-level QC requires non-cell pixels for background estimation")
    
    results = []
    
    # Optimize: Load all channels at once instead of loading the entire acquisition
    # multiple times (once per channel). This is much faster for larger files.
    # For MCD files, get_image() reads the entire acquisition each time, so loading
    # all channels once and extracting individual channels is much more efficient.
    img_stack = None
    channel_indices = None
    n_channels_total = None
    
    try:
        # Load all channels at once
        img_stack = loader.get_all_channels(original_acq_id)
        
        # Determine shape - loaders return HWC format (H, W, C)
        if img_stack.ndim == 3:
            height, width, n_channels_total = img_stack.shape
        elif img_stack.ndim == 2:
            # Single channel - handle separately
            n_channels_total = 1
        else:
            # Unexpected shape - fall back to per-channel loading
            img_stack = None
            n_channels_total = None
        
        if img_stack is not None:
            # Get channel indices for the requested channels
            all_channels = loader.get_channels(original_acq_id)
            channel_indices = {}
            for channel in channels:
                if channel in all_channels:
                    channel_indices[channel] = all_channels.index(channel)
    except Exception as e:
        # Fallback to per-channel loading if get_all_channels fails
        # (e.g., for some loader implementations)
        img_stack = None
        channel_indices = None
        n_channels_total = None

    if mode == "cell" and img_stack is not None and mask.shape != img_stack.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {img_stack.shape[:2]}"
        )
    
    for channel in channels:
        try:
            # Extract channel from pre-loaded stack if available, otherwise load individually
            if (img_stack is not None and channel_indices is not None and 
                channel in channel_indices and n_channels_total is not None):
                ch_idx = channel_indices[channel]
                if img_stack.ndim == 3:
                    if ch_idx >= n_channels_total:
                        continue
                    img = img_stack[:, :, ch_idx]
                elif img_stack.ndim == 2:
                    img = img_stack
                else:
                    continue
            else:
                # Fallback: load image individually (slower but works for all loaders)
                img = loader.get_image(original_acq_id, channel)
            
            if img is None:
                continue

            if denoise_settings and channel in denoise_settings:
                img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
            
            # Optimize: compute statistics more efficiently without flattening
            # Use np.nanmin/nanmax for better performance on large arrays
            img_min = float(np.min(img))
            img_max = float(np.max(img))
            img_mean = float(np.mean(img))
            img_std = float(np.std(img))
            # Median is expensive, compute only if needed
            img_median = float(np.median(img))
            
            if mode == "pixel":
                # Pixel-level QC using Otsu threshold
                if _HAVE_SCIKIT_IMAGE:
                    try:
                        # For very large images, consider downsampling for Otsu threshold
                        # Otsu thresholding is O(n) but can be slow on huge images
                        img_for_otsu = img
                        downsample_factor = 1
                        if img.size > 50_000_000:  # > ~7000x7000 pixels
                            # Downsample by 2x for Otsu threshold calculation
                            downsample_factor = 2
                            from scipy import ndimage
                            img_for_otsu = img[::downsample_factor, ::downsample_factor]
                        
                        threshold = threshold_otsu(img_for_otsu)
                        # Scale threshold back if downsampled
                        if downsample_factor > 1:
                            # Threshold should be similar, but adjust if needed
                            pass  # Otsu threshold is intensity-based, not position-based
                        
                        signal_mask = img > threshold
                        # Optimize: only compute background_mask if needed, otherwise use ~signal_mask
                        background_mask = ~signal_mask
                        
                        # Vectorized operations - faster than conditional indexing
                        signal_pixels = img[signal_mask]
                        background_pixels = img[background_mask]
                        
                        if len(signal_pixels) > 0:
                            signal_mean = float(np.mean(signal_pixels))
                            signal_std = float(np.std(signal_pixels))
                        else:
                            signal_mean = img_mean
                            signal_std = 0.0
                        
                        if len(background_pixels) > 0:
                            background_mean = float(np.mean(background_pixels))
                            background_std = float(np.std(background_pixels))
                        else:
                            background_mean = img_mean
                            background_std = img_std
                        
                        snr = _calculate_qc_snr(signal_mean, background_mean, background_std, img_min, img_max)
                        coverage = float(len(signal_pixels) / img.size) if img.size > 0 else 0.0
                    except Exception:
                        # Fallback if Otsu fails
                        threshold = np.nan
                        signal_mean = img_mean
                        signal_std = 0.0
                        background_mean = img_mean
                        background_std = img_std
                        signal_pixels = np.empty(0, dtype=img.dtype)
                        background_pixels = img.reshape(-1)
                        snr = 0.0
                        coverage = 0.0
                else:
                    # No scikit-image, use simple statistics
                    threshold = np.nan
                    signal_mean = img_mean
                    signal_std = 0.0
                    background_mean = img_mean
                    background_std = img_std
                    signal_pixels = np.empty(0, dtype=img.dtype)
                    background_pixels = img.reshape(-1)
                    snr = 0.0
                    coverage = 0.0
                
                results.append({
                    'acquisition_id': acquisition.id,
                    'acquisition_name': acquisition.name,
                    'channel': channel,
                    'mode': 'pixel',
                    'snr': snr,
                    'signal_mean': signal_mean,
                    'signal_std': signal_std,
                    'background_mean': background_mean,
                    'background_std': background_std,
                    'threshold': float(threshold),
                    'n_total_pixels': int(img.size),
                    'n_signal_pixels': int(len(signal_pixels)),
                    'n_background_pixels': int(len(background_pixels)),
                    'intensity_mean': img_mean,
                    'intensity_std': img_std,
                    'intensity_median': img_median,
                    'intensity_min': img_min,
                    'intensity_max': img_max,
                    'coverage': coverage
                })
            
            elif mode == "cell":
                # Cell-level QC using segmentation mask
                if mask is None:
                    continue
                
                if mask.shape != img.shape:
                    continue
                
                cell_mask = mask > 0
                background_mask = mask == 0
                cell_ids = np.unique(mask[cell_mask])
                if len(cell_ids) == 0:
                    continue
                
                # Background is pixels outside cells - vectorized
                if np.any(background_mask):
                    background_pixels = img[background_mask]
                    background_mean = float(np.mean(background_pixels))
                    background_std = float(np.std(background_pixels))
                else:
                    background_mean = img_mean
                    background_std = img_std

                signal_metrics = _compute_cell_signal_metrics(
                    img,
                    mask,
                    background_mean,
                    background_std,
                    img_min,
                    img_max,
                    cell_signal_method=cell_signal_method,
                    positive_threshold_sd=positive_threshold_sd,
                    upper_quantile=upper_quantile,
                )
                signal_mean = signal_metrics["signal_mean"]
                signal_std = signal_metrics["signal_std"]
                snr = signal_metrics["snr"]
                
                # Coverage: fraction of pixels covered by cells - optimized
                n_cell_pixels = np.sum(mask > 0)
                coverage = float(n_cell_pixels / mask.size) if mask.size > 0 else 0.0
                
                # Cell density: cells per unit area (assuming pixels)
                n_cells = len(cell_ids)
                cell_density = float(n_cells / mask.size) if mask.size > 0 else 0.0
                
                results.append({
                    'acquisition_id': acquisition.id,
                    'acquisition_name': acquisition.name,
                    'channel': channel,
                    'mode': 'cell',
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
                    'n_total_pixels': int(mask.size),
                    'n_cell_pixels': int(n_cell_pixels),
                    'n_background_pixels': int(np.count_nonzero(background_mask)),
                    'intensity_mean': img_mean,
                    'intensity_std': img_std,
                    'intensity_median': img_median,
                    'intensity_min': img_min,
                    'intensity_max': img_max,
                    'coverage': coverage,
                    'cell_density': cell_density,
                    'n_cells': n_cells
                })
        
        except Exception:
            continue
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def spillover_correction(
    features_df: pd.DataFrame,
    spillover_matrix: Union[str, Path, pd.DataFrame],
    method: str = "pgd",
    arcsinh_cofactor: Optional[float] = None,
    channel_map: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Apply spillover correction to feature data.
    
    This function applies CATALYST-like spillover compensation to remove
    spectral overlap between channels in IMC data.
    
    Args:
        features_df: DataFrame with cell features (cells x channels)
        spillover_matrix: Path to spillover matrix CSV or DataFrame
        method: Compensation method ('nnls' or 'pgd', default: 'pgd')
        arcsinh_cofactor: Optional cofactor for arcsinh transformation
        channel_map: Optional mapping from feature column names to spillover matrix channel names
        output_path: Optional path to save corrected features CSV
    
    Returns:
        DataFrame with corrected features
    """
    # Load spillover matrix if path provided
    if isinstance(spillover_matrix, (str, Path)):
        S = load_spillover(str(spillover_matrix))
    else:
        S = spillover_matrix.copy()
    
    # Extract intensity features (columns that match spillover matrix channels)
    # Auto-detect intensity columns if not all columns are features
    intensity_cols = []
    for col in features_df.columns:
        if col in S.columns or (channel_map and col in channel_map):
            intensity_cols.append(col)
    
    if not intensity_cols:
        # Try to find intensity columns by pattern (e.g., channel names)
        # This is a fallback - ideally user should specify
        raise ValueError(
            "No matching channels found between features and spillover matrix. "
            "Please ensure channel names match or provide channel_map."
        )
    
    # Apply compensation
    comp_counts, comp_asinh = compensate_counts(
        features_df[intensity_cols],
        S,
        method=method,
        arcsinh_cofactor=arcsinh_cofactor,
        channel_map=channel_map,
        strict_align=False,
        return_all_channels=True
    )
    
    # Create result dataframe with all original columns
    result_df = features_df.copy()
    result_df[intensity_cols] = comp_counts[intensity_cols]
    
    # Add arcsinh-transformed version if requested
    if comp_asinh is not None:
        for col in intensity_cols:
            if col in comp_asinh.columns:
                result_df[f"{col}_arcsinh"] = comp_asinh[col]
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        result_df.to_csv(output_path, index=False)
    
    return result_df


def generate_spillover_matrix(
    mcd_path: Union[str, Path],
    donor_label_per_acq: Optional[Dict[str, str]] = None,
    cap: float = 0.3,
    aggregate: str = "median",
    channel_name_field: str = "name",
    p_low: float = 90.0,
    p_high_clip: float = 99.9,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate spillover matrix from single-stain control MCD file.
    
    This function analyzes pixel-level data from single-stain control acquisitions
    to estimate spillover coefficients between channels.
    
    Args:
        mcd_path: Path to MCD file containing single-stain controls
        donor_label_per_acq: Mapping from acquisition ID/index to donor channel name
        cap: Maximum spillover coefficient (default: 0.3)
        aggregate: Aggregation method when multiple acquisitions per donor ('median' or 'mean')
        channel_name_field: Field to use for channel names ('name' or 'fullname', default: 'name')
        p_low: Lower percentile for foreground selection (default: 90.0)
        p_high_clip: Upper percentile for clipping (default: 99.9)
        output_path: Optional path to save spillover matrix CSV
    
    Returns:
        Tuple of (spillover_matrix_df, qc_metrics_df)
    """
    S_df, qc_df = build_spillover_from_comp_mcd(
        str(mcd_path),
        donor_label_per_acq=donor_label_per_acq,
        cap=cap,
        aggregate=aggregate,
        channel_name_field=channel_name_field,
        p_low=p_low,
        p_high_clip=p_high_clip
    )
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        S_df.to_csv(output_path, index=True)
        qc_path = output_path.parent / (output_path.stem + '_qc.csv')
        qc_df.to_csv(qc_path, index=True)
    
    return S_df, qc_df


def deconvolution(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    output_dir: Union[str, Path],
    x0: float = 7.0,
    iterations: int = 7,
    output_format: str = "float",
    loader_path: Optional[Union[str, Path]] = None,
    source_file_path: Optional[Union[str, Path]] = None,
    unique_acq_id: Optional[str] = None,
    passes: Optional[np.ndarray] = None,
    contributions: Optional[np.ndarray] = None,
    kernel: Optional[np.ndarray] = None,
    passes_arr: Optional[np.ndarray] = None,
    contribs_arr: Optional[np.ndarray] = None,
    kernel_dim: Optional[int] = None,
    region_data_full: Optional[list] = None,
    I0: Optional[float] = None,
    resolution: int = 333,
) -> Path:
    """Apply Richardson-Lucy deconvolution to high resolution IMC images.
    
    This function applies deconvolution optimized for high resolution IMC
    images with step sizes of 333 nm and 500 nm.
    
    Args:
        loader: Data loader (MCDLoader or OMETIFFLoader)
        acquisition: Acquisition information
        output_dir: Output directory for deconvolved images
        x0: Parameter for kernel calculation (default: 7.0)
        iterations: Number of Richardson-Lucy iterations (default: 7)
        output_format: Output format ('float' or 'uint16', default: 'float')
        resolution: HR-IMC resolution in nm (333 or 500, default: 333)
        loader_path: Optional explicit path to loader file/directory (if loader doesn't have file_path/directory attribute)
        source_file_path: Optional source file path for filename generation (defaults to loader_path)
        unique_acq_id: Optional unique acquisition ID for filename generation (defaults to acquisition.id)
    
    Returns:
        Path to deconvolved OME-TIFF file
    """
    from openimc.processing.deconvolution_worker import deconvolve_acquisition
    
    # Get loader path
    if loader_path is None:
        if hasattr(loader, 'file_path'):
            loader_path = loader.file_path
        elif hasattr(loader, 'directory'):
            loader_path = loader.directory
        elif hasattr(loader, 'folder_path'):
            loader_path = loader.folder_path
        elif isinstance(loader, MCDLoader) and hasattr(loader, 'mcd') and loader.mcd:
            # Try to get path from McdFile object
            if hasattr(loader.mcd, 'path'):
                loader_path = loader.mcd.path
            elif hasattr(loader.mcd, 'filename'):
                loader_path = loader.mcd.filename
            else:
                raise ValueError("Cannot determine loader path for deconvolution. Please provide loader_path parameter.")
        else:
            raise ValueError("Cannot determine loader path for deconvolution. Please provide loader_path parameter.")
    
    loader_path = str(loader_path)
    
    # Use source_file_path if provided, otherwise use loader_path
    if source_file_path is None:
        source_file_path = loader_path
    else:
        source_file_path = str(source_file_path)
    
    # Use unique_acq_id if provided, otherwise use acquisition.id
    if unique_acq_id is None:
        unique_acq_id = acquisition.id
    
    # Determine loader type
    loader_type = "mcd" if isinstance(loader, MCDLoader) else "ometiff"
    
    # Extract channel names from acquisition
    channel_names = acquisition.channels if acquisition.channels else None
    
    # Extract pixel size from metadata
    pixel_size_x = None
    pixel_size_y = None
    pixel_size_unit = "µm"  # Default unit
    
    if acquisition.metadata:
        # Look for common pixel size keys in metadata
        for key, value in acquisition.metadata.items():
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
            # Also check for OME standard keys
            elif key == 'PhysicalSizeX':
                try:
                    pixel_size_x = float(value)
                except (ValueError, TypeError):
                    pass
            elif key == 'PhysicalSizeY':
                try:
                    pixel_size_y = float(value)
                except (ValueError, TypeError):
                    pass
            elif key == 'PhysicalSizeXUnit' or key == 'PhysicalSizeYUnit':
                pixel_size_unit = str(value)
    
    # If only one dimension found, use it for both
    if pixel_size_x is not None and pixel_size_y is None:
        pixel_size_y = pixel_size_x
    elif pixel_size_y is not None and pixel_size_x is None:
        pixel_size_x = pixel_size_y
    
    # Call deconvolution worker
    output_path = deconvolve_acquisition(
        loader_path,
        acquisition.id,
        str(output_dir),
        x0=x0,
        iterations=iterations,
        output_format=output_format,
        channel_names=channel_names,
        source_file_path=source_file_path,
        unique_acq_id=unique_acq_id,
        loader_type=loader_type,
        well_name=getattr(acquisition, 'well', None),
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        pixel_size_unit=pixel_size_unit,
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
    
    return Path(output_path)


def spatial_enrichment(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cluster_column: str = "cluster",
    n_permutations: int = 100,
    seed: int = 42,
    roi_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """Compute pairwise spatial enrichment between clusters.
    
    This function computes enrichment of spatial interactions between cluster
    pairs using permutation-based null distribution with multiprocessing support.
    
    Args:
        features_df: DataFrame with cell features and cluster labels
        edges_df: DataFrame with spatial graph edges (must have 'cell_id_A', 'cell_id_B', 'roi_id')
        cluster_column: Column name containing cluster labels
        n_permutations: Number of permutations for null distribution (default: 100)
        seed: Random seed for reproducibility (default: 42)
        roi_column: Column name for ROI grouping (auto-detected if None)
        output_path: Optional path to save enrichment results CSV
        n_workers: Number of parallel workers (default: None = use all available CPUs - 2)
    
    Returns:
        DataFrame with enrichment results (cluster_A, cluster_B, observed, expected, p_value, z_score, etc.)
    """
    import random

    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be at least 1")
    
    # Auto-detect ROI column
    if roi_column is None:
        for col in ['acquisition_id', 'roi_id', 'roi']:
            if col in features_df.columns:
                roi_column = col
                break
    
    if roi_column is None:
        roi_column = 'roi_id'  # Default for edges_df
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Determine number of workers
    if n_workers is None:
        try:
            cpu_count = mp.cpu_count()
            n_workers = max(1, cpu_count - 2)
        except (NotImplementedError, RuntimeError):
            n_workers = 1
    n_workers = max(1, n_workers)
    
    
    edge_groups = {}
    if edges_df is not None and not edges_df.empty and 'roi_id' in edges_df.columns:
        for roi_id, roi_edges in edges_df.groupby('roi_id', sort=False):
            edge_groups[str(roi_id)] = roi_edges.loc[:, ['cell_id_A', 'cell_id_B']].copy()

    # Collect all ROIs first
    roi_tasks = []
    for roi_id, roi_df in features_df.groupby(roi_column):
        roi_edges = edge_groups.get(str(roi_id))
        
        if roi_edges is None or roi_edges.empty:
            continue
        
        cluster_values = roi_df[cluster_column].map(canonicalize_cluster_id).dropna()
        unique_clusters = sort_cluster_values(cluster_values.unique(), canonical=True)
        if len(unique_clusters) < 2:
            continue
        
        # Prepare ROI task for worker
        roi_tasks.append((
            roi_id,
            roi_df.loc[:, ['cell_id', cluster_column]].copy(),
            roi_edges,
            cluster_column,
            n_permutations,
            seed + len(roi_tasks)  # Unique seed for each ROI
        ))
    
    n_workers = min(n_workers, len(roi_tasks)) if roi_tasks else 1
    
    # Process ROIs with multiprocessing
    enrichment_results = []
    if len(roi_tasks) > 0:
        if n_workers > 1 and len(roi_tasks) > 1:
            # Use multiprocessing: one ROI per worker
            import time
            start_time = time.time()
            with mp.Pool(processes=n_workers) as pool:
                roi_results = pool.map(roi_enrichment_worker, roi_tasks)
            elapsed = time.time() - start_time
            
            # Flatten results from all ROIs
            for roi_result_list in roi_results:
                enrichment_results.extend(roi_result_list)
        else:
            # Single-threaded fallback: process ROIs sequentially
            import time
            start_time = time.time()
            for roi_task in roi_tasks:
                roi_result_list = roi_enrichment_worker(roi_task)
                enrichment_results.extend(roi_result_list)
            elapsed = time.time() - start_time
    
    if not enrichment_results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(enrichment_results)
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        results_df.to_csv(output_path, index=False)
    
    return results_df


def spatial_distance_distribution(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cluster_column: str = "cluster",
    roi_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    pixel_size_um: float = 1.0,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """Compute distance distributions between clusters.
    
    This function computes nearest neighbor distances from each cell to each cluster type
    using multiprocessing at ROI level for efficiency.
    
    Args:
        features_df: DataFrame with cell features and cluster labels
        edges_df: DataFrame with spatial graph edges (must have 'cell_id_A', 'cell_id_B', 'roi_id')
        cluster_column: Column name containing cluster labels
        roi_column: Column name for ROI grouping (auto-detected if None)
        output_path: Optional path to save distance distribution results CSV
        pixel_size_um: Pixel size in micrometers for coordinate conversion (default: 1.0)
        n_workers: Number of parallel workers (default: None = use all available CPUs - 2)
    
    Returns:
        DataFrame with distance data (cell_A_id, cell_A_cluster, nearest_B_cluster, nearest_B_dist_um, etc.)
    """
    # Auto-detect ROI column
    if roi_column is None:
        for col in ['acquisition_id', 'roi_id', 'roi']:
            if col in features_df.columns:
                roi_column = col
                break
    
    if roi_column is None:
        roi_column = 'roi_id'
    
    # Determine number of workers
    if n_workers is None:
        try:
            cpu_count = mp.cpu_count()
            n_workers = max(1, cpu_count - 2)
        except (NotImplementedError, RuntimeError):
            n_workers = 1
    
    n_workers = max(1, n_workers)
    
    
    # Collect all ROIs first
    roi_tasks = []
    for roi_id, roi_df in features_df.groupby(roi_column):
        cluster_values = roi_df[cluster_column].map(canonicalize_cluster_id).dropna()
        unique_clusters = sort_cluster_values(cluster_values.unique(), canonical=True)
        if len(unique_clusters) < 1:
            continue
        
        # Prepare ROI task for worker
        roi_tasks.append((
            str(roi_id),
            roi_df.loc[:, ['cell_id', 'centroid_x', 'centroid_y', cluster_column]].copy(),
            cluster_column,
            pixel_size_um
        ))
    
    n_workers = min(n_workers, len(roi_tasks)) if roi_tasks else 1
    
    # Process ROIs with multiprocessing
    distance_results = []
    if len(roi_tasks) > 0:
        if n_workers > 1 and len(roi_tasks) > 1:
            # Use multiprocessing: one ROI per worker
            with mp.Pool(processes=n_workers) as pool:
                roi_results = pool.map(distance_distribution_worker, roi_tasks)
                # Flatten results from all ROIs
                for roi_result in roi_results:
                    distance_results.extend(roi_result)
        else:
            # Single-threaded processing
            for roi_task in roi_tasks:
                roi_result = distance_distribution_worker(roi_task)
                distance_results.extend(roi_result)
    
    if not distance_results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(distance_results)
    
    # Save output if path is provided
    if output_path is not None:
        output_path = Path(output_path)
        results_df.to_csv(output_path, index=False)
    
    return results_df


def dataframe_to_anndata(
    df: pd.DataFrame,
    roi_id: Optional[str] = None,
    roi_column: str = 'acquisition_id',
    pixel_size_um: float = 1.0
) -> Optional['ad.AnnData']:
    """
    Convert OpenIMC DataFrame to AnnData format for squidpy analysis.
    
    This is the unified function used by both GUI and CLI.
    
    Args:
        df: Feature dataframe with cells as rows
        roi_id: Optional ROI identifier to filter to a single ROI
        roi_column: Column name for ROI identifier
        pixel_size_um: Pixel size in micrometers for coordinate conversion
        
    Returns:
        AnnData object with spatial coordinates and features, or None if conversion fails
    """
    try:
        import anndata as ad
    except (ImportError, OSError, RuntimeError):
        raise ImportError("anndata is required for AnnData-based spatial analysis. Install with: pip install anndata")
    
    try:
        # Filter to specific ROI if provided
        if roi_id is not None and roi_column in df.columns:
            df = df[df[roi_column] == roi_id].copy()
        
        if df.empty:
            return None
        
        # Ensure required columns exist
        required_cols = ['centroid_x', 'centroid_y', 'cell_id']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None
        
        # Extract centroid coordinates
        centroid_coords = df[['centroid_x', 'centroid_y']].values
        
        # Convert coordinates from pixels to micrometers
        coords_um = centroid_coords * pixel_size_um
        
        # Identify feature columns (exclude metadata)
        metadata_cols = {
            'cell_id', 'acquisition_id', 'acquisition_label', 'source_file', 
            'source_well', 'label', 'centroid_x', 'centroid_y', 'cluster',
            'cluster_phenotype', 'cluster_id', 'well', 'acquisition_name'
        }
        
        # Get intensity and morphology features
        all_feature_cols = [col for col in df.columns if col not in metadata_cols]
        feature_cols = [col for col in all_feature_cols if col.endswith('_mean')]
        
        # Also include morphology features (they don't have _mean suffix)
        morpho_names = {
            'area_um2', 'perimeter_um', 'equivalent_diameter_um', 'eccentricity',
            'solidity', 'extent', 'circularity', 'major_axis_len_um', 'minor_axis_len_um',
            'aspect_ratio', 'bbox_area_um2', 'touches_border', 'touches_edge', 'holes_count'
        }
        morpho_cols = [col for col in all_feature_cols if col in morpho_names]
        feature_cols.extend(morpho_cols)
        
        # Create AnnData object
        # X: feature matrix (intensity and morphology features)
        if feature_cols:
            X = (
                df[feature_cols]
                .apply(pd.to_numeric, errors='coerce')
                .astype(float)
                .to_numpy()
            )
        else:
            X = np.zeros((len(df), 0), dtype=float)
        
        # obs: cell metadata
        obs = df[list(metadata_cols & set(df.columns))].copy()
        obs.index = df['cell_id'].astype(str).values
        
        # obsm: spatial coordinates
        obsm = {'spatial': coords_um}
        
        # var: feature names
        var = pd.DataFrame(index=feature_cols)
        
        # Create AnnData
        adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm)
        
        # Store cluster information in obs if available
        cluster_col = None
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in df.columns:
                cluster_col = col
                break
        
        if cluster_col:
            adata.obs['cluster'] = df[cluster_col].map(canonicalize_cluster_id).values
        
        return adata
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _ordered_cluster_categorical(
    values: pd.Series,
    *,
    canonicalize_ids: bool,
) -> pd.Series:
    """Return a stable ordered categorical series for spatial-analysis cluster columns."""
    normalized = values.map(canonicalize_cluster_id) if canonicalize_ids else values.copy()
    ordered_categories = sort_cluster_values(
        pd.unique(normalized.dropna()),
        canonical=canonicalize_ids,
    )
    return pd.Series(
        pd.Categorical(normalized, categories=ordered_categories, ordered=True),
        index=values.index,
        name=values.name,
    )


def build_spatial_graph_anndata(
    features_df: pd.DataFrame,
    method: str = "kNN",
    k_neighbors: int = 20,
    radius: Optional[float] = None,
    pixel_size_um: float = 1.0,
    roi_column: Optional[str] = None,
    roi_id: Optional[str] = None,
    seed: int = 42
) -> Dict[str, 'ad.AnnData']:
    """Build spatial graph using squidpy and return AnnData objects per ROI.
    
    This function creates AnnData objects with spatial graphs built using squidpy.
    It's the unified function used by both GUI and CLI for AnnData-based spatial analysis.
    
    Args:
        features_df: DataFrame with cell features, must contain 'centroid_x' and 'centroid_y'
        method: Graph construction method ('kNN', 'Radius', or 'Delaunay')
        k_neighbors: Number of neighbors for kNN method (default: 20)
        radius: Radius in micrometers for radius-based method (required if method='Radius')
        pixel_size_um: Pixel size in micrometers for coordinate conversion (default: 1.0)
        roi_column: Column name for ROI grouping (e.g., 'acquisition_id'). Auto-detected if None
        roi_id: Optional specific ROI to process. If None, processes all ROIs
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Dictionary mapping ROI ID to AnnData object with spatial graph built
    
    Raises:
        ImportError: If squidpy or anndata are not installed
        ValueError: If method is invalid or required parameters are missing
    """
    try:
        import squidpy as sq
        import anndata as ad
        from scipy.spatial import Delaunay
        from scipy import sparse as sp
    except (ImportError, OSError, RuntimeError):
        raise ImportError(
            "squidpy, anndata, and scipy are required for AnnData-based spatial analysis. "
            "Install with: pip install squidpy anndata scipy"
        )
    
    # Validate method
    if method not in ['kNN', 'Radius', 'Delaunay']:
        raise ValueError(f"Unknown graph method: {method}. Must be 'kNN', 'Radius', or 'Delaunay'")
    
    if method == 'Radius' and radius is None:
        raise ValueError("radius parameter is required for 'Radius' method")
    
    # Determine ROI column
    if roi_column is None:
        for col in ['acquisition_id', 'source_well', 'roi_id', 'roi']:
            if col in features_df.columns:
                roi_column = col
                break
    
    if roi_column is None:
        roi_column = 'acquisition_id'  # Default
    
    # Get ROIs to process
    if roi_id is not None:
        roi_ids = [roi_id]
    else:
        roi_ids = sorted(features_df[roi_column].unique())
    
    anndata_dict = {}
    skipped_rois: List[Tuple[str, str]] = []
    adjusted_k_rois: List[Tuple[str, int, int]] = []
    
    for current_roi_id in roi_ids:
        # Convert dataframe to AnnData
        adata = dataframe_to_anndata(
            features_df,
            roi_id=current_roi_id,
            roi_column=roi_column,
            pixel_size_um=pixel_size_um
        )
        
        if adata is None:
            skipped_rois.append((str(current_roi_id), "no valid cells after filtering"))
            continue
        n_cells = int(adata.n_obs)
        if n_cells == 0:
            skipped_rois.append((str(current_roi_id), "no valid cells after filtering"))
            continue
        
        # Ensure cluster columns are categorical (required by squidpy)
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in adata.obs.columns:
                adata.obs[col] = _ordered_cluster_categorical(
                    adata.obs[col],
                    canonicalize_ids=(col in {'cluster', 'cluster_id'}),
                )
        
        # Build spatial graph
        coords = adata.obsm['spatial']
        
        try:
            if method == "kNN":
                # squidpy/scikit-learn require n_neighbors < n_samples for self-neighbor queries.
                if n_cells < 2:
                    skipped_rois.append((str(current_roi_id), "fewer than 2 cells for kNN"))
                    continue

                requested_k = int(k_neighbors)
                effective_k = min(max(1, requested_k), n_cells - 1)
                if effective_k != requested_k:
                    adjusted_k_rois.append((str(current_roi_id), requested_k, effective_k))
                sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=effective_k, n_rings=1)
            elif method == "Radius":
                # Radius is in micrometers, coordinates are in micrometers.
                # ROIs with one cell are allowed but will produce no edges.
                sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius, n_rings=1)
            elif method == "Delaunay":
                # Delaunay triangulation requires at least 3 points.
                if n_cells < 3:
                    skipped_rois.append((str(current_roi_id), "fewer than 3 cells for Delaunay"))
                    continue

                tri = Delaunay(coords)
                rows, cols = [], []
                for simplex in tri.simplices:
                    # Each simplex has 3 vertices, create edges between all pairs.
                    for i in range(3):
                        for j in range(i + 1, 3):
                            rows.extend([simplex[i], simplex[j]])
                            cols.extend([simplex[j], simplex[i]])
                
                # Create sparse matrix.
                data = np.ones(len(rows))
                conn = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
                
                # Store in AnnData format.
                adata.obsp['spatial_connectivities'] = conn
                
                # Calculate distances.
                distances = []
                for i, j in zip(rows, cols):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances.append(dist)
                dist_matrix = sp.csr_matrix((distances, (rows, cols)), shape=(n_cells, n_cells))
                adata.obsp['spatial_distances'] = dist_matrix
        except ValueError as e:
            err = str(e)
            if "n_neighbors" in err or "n_samples_fit" in err:
                skipped_rois.append((str(current_roi_id), f"insufficient cells for requested neighborhood graph ({err})"))
                continue
            raise
        except Exception as e:
            # Keep processing other ROIs, especially for Delaunay edge cases (e.g., degenerate geometry).
            skipped_rois.append((str(current_roi_id), f"graph construction failed ({e})"))
            continue
        
        # Verify graph was created
        if 'spatial_connectivities' in adata.obsp:
            anndata_dict[str(current_roi_id)] = adata
    
    if adjusted_k_rois:
        adjusted_preview = ", ".join(
            f"{roi}: {old_k}->{new_k}" for roi, old_k, new_k in adjusted_k_rois[:8]
        )
        if len(adjusted_k_rois) > 8:
            adjusted_preview += f", ... ({len(adjusted_k_rois)} total)"
        print(
            f"[CORE.SPATIAL] Adjusted k-neighbors for small ROIs to avoid failures: {adjusted_preview}"
        )

    if skipped_rois:
        skipped_preview = ", ".join(
            f"{roi} ({reason})" for roi, reason in skipped_rois[:8]
        )
        if len(skipped_rois) > 8:
            skipped_preview += f", ... ({len(skipped_rois)} total)"
        print(
            f"[CORE.SPATIAL] Skipped ROI(s) during spatial graph construction: {skipped_preview}"
        )

    return anndata_dict


def spatial_neighborhood_enrichment(
    anndata_dict: Dict[str, 'ad.AnnData'],
    cluster_key: str = "cluster",
    aggregation: str = "mean",
    significance_threshold: float = 2.0
) -> Dict[str, Any]:
    """Compute neighborhood enrichment using squidpy.
    
    This function computes neighborhood enrichment for each ROI and optionally aggregates results.
    
    Args:
        anndata_dict: Dictionary mapping ROI ID to AnnData object with spatial graph
        cluster_key: Column name containing cluster labels (default: "cluster")
        aggregation: Aggregation method for multiple ROIs ("mean" or "sum", default: "mean")
        significance_threshold: Z-score threshold for significant interactions (default: 2.0)
    
    Returns:
        Dictionary with:
            - 'results': Dict mapping ROI ID to enrichment results
            - 'aggregated': Aggregated enrichment matrix (if multiple ROIs)
            - 'cluster_categories': List of cluster categories
            - 'significant_counts': Matrix counting ROIs with significant interactions per cluster pair
    """
    try:
        import squidpy as sq
        import anndata as ad
        from squidpy._utils import _get_n_cores, parallelize
        from squidpy.gr._nhood import _create_function, _nhood_enrichment_helper
    except (ImportError, OSError, RuntimeError):
        raise ImportError("squidpy and anndata are required for neighborhood enrichment")

    def _run_nhood_permutation_test(adata: 'ad.AnnData', *, seed: int = 0, n_perms: int = 1000) -> Dict[str, np.ndarray]:
        """Mirror Squidpy's permutation test while retaining empirical p-values."""
        adjacency = adata.obsp['spatial_connectivities'].tocsr()
        cluster_series = adata.obs[cluster_key]
        cluster_values = list(cluster_series.cat.categories)
        cluster_to_code = {cluster: idx for idx, cluster in enumerate(cluster_values)}
        int_clust = np.asarray([cluster_to_code[value] for value in cluster_series], dtype=np.uint32)
        indices = adjacency.indices.astype(np.uint32, copy=False)
        indptr = adjacency.indptr.astype(np.uint32, copy=False)
        n_clusters = len(cluster_values)

        count_fn = _create_function(n_clusters, parallel=False)
        observed = count_fn(indices, indptr, int_clust)

        n_jobs = _get_n_cores(None)
        permutations = parallelize(
            _nhood_enrichment_helper,
            collection=np.arange(n_perms).tolist(),
            extractor=np.vstack,
            n_jobs=n_jobs,
            backend='loky',
            show_progress_bar=False,
        )(
            callback=count_fn,
            indices=indices,
            indptr=indptr,
            int_clust=int_clust,
            libraries=None,
            n_cls=n_clusters,
            seed=seed,
        )

        permutation_mean = permutations.mean(axis=0)
        permutation_std = permutations.std(axis=0)
        zscore = np.zeros_like(permutation_mean, dtype=float)
        valid_std = permutation_std > 0
        zscore[valid_std] = (observed[valid_std] - permutation_mean[valid_std]) / permutation_std[valid_std]

        observed_delta = np.abs(observed - permutation_mean)
        permutation_delta = np.abs(permutations - permutation_mean[None, :, :])
        p_values = (np.sum(permutation_delta >= observed_delta[None, :, :], axis=0) + 1.0) / (n_perms + 1.0)
        p_values_adjusted = benjamini_hochberg_adjust_matrix(p_values)

        return {
            'zscore': np.asarray(zscore, dtype=float),
            'count': np.asarray(observed, dtype=float),
            'pvalue': np.asarray(p_values, dtype=float),
            'pvalue_fdr_bh': np.asarray(p_values_adjusted, dtype=float),
            'n_perms': np.asarray(n_perms, dtype=float),
        }
    
    results = {}
    enrichment_matrices = []
    pvalue_matrices = {}
    pvalue_adjusted_matrices = {}
    roi_cluster_map = {}
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        if cluster_key not in adata.obs.columns:
            continue
        
        # Filter out cells with NaN cluster values
        cluster_values = adata.obs[cluster_key]
        nan_mask = pd.isna(cluster_values)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            # Create boolean mask for valid cells (non-NaN)
            valid_mask = ~nan_mask
            # Filter AnnData object (this will also filter spatial_connectivities)
            adata = adata[valid_mask].copy()
        
        # Ensure categorical
        canonicalize_cluster_ids = cluster_key in {'cluster', 'cluster_id'}
        adata.obs[cluster_key] = _ordered_cluster_categorical(
            adata.obs[cluster_key],
            canonicalize_ids=canonicalize_cluster_ids,
        )
        
        
        # Check graph connectivity
        if 'spatial_connectivities' in adata.obsp:
            conn = adata.obsp['spatial_connectivities']
            n_edges = conn.nnz // 2  # Divide by 2 because it's symmetric
            # Check if graph is connected
            from scipy.sparse.csgraph import connected_components
            n_components, labels = connected_components(conn, directed=False, return_labels=True)
        
        # Check clusters
        if hasattr(adata.obs[cluster_key], 'cat'):
            categories = list(adata.obs[cluster_key].cat.categories)
            unique_vals = categories
        else:
            categories = sort_cluster_values(
                adata.obs[cluster_key].unique(),
                canonical=canonicalize_cluster_ids,
            )
            unique_vals = categories
        
        if len(unique_vals) < 2:
            continue
        
        enrichment_data = _run_nhood_permutation_test(adata, seed=int(len(results)))
        adata.uns['nhood_enrichment'] = enrichment_data
        adata.uns[f'{cluster_key}_nhood_enrichment'] = enrichment_data

        matrix = None
        
        if isinstance(enrichment_data, dict):
            if 'zscore' in enrichment_data:
                matrix = enrichment_data['zscore']
            elif 'count' in enrichment_data:
                matrix = enrichment_data['count']
            elif 'stat' in enrichment_data:
                matrix = enrichment_data['stat']
            else:
                for key, value in enrichment_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        matrix = value
                        break
        elif isinstance(enrichment_data, np.ndarray):
            matrix = enrichment_data
        
        if matrix is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 2:
            results[roi_id] = adata
            enrichment_matrices.append((roi_id, matrix))
            if isinstance(enrichment_data, dict):
                p_matrix = enrichment_data.get('pvalue')
                if isinstance(p_matrix, np.ndarray) and p_matrix.shape == matrix.shape:
                    pvalue_matrices[roi_id] = np.asarray(p_matrix, dtype=float)
                p_adjusted_matrix = enrichment_data.get('pvalue_fdr_bh')
                if isinstance(p_adjusted_matrix, np.ndarray) and p_adjusted_matrix.shape == matrix.shape:
                    pvalue_adjusted_matrices[roi_id] = np.asarray(p_adjusted_matrix, dtype=float)
            
            # Get cluster categories
            if hasattr(adata.obs[cluster_key], 'cat'):
                clusters = list(adata.obs[cluster_key].cat.categories)
            else:
                clusters = sort_cluster_values(
                    adata.obs[cluster_key].unique(),
                    canonical=canonicalize_cluster_ids,
                )
            roi_cluster_map[roi_id] = clusters
        else:
            if matrix is not None:
                pass
    
    # Aggregate if multiple ROIs
    aggregated_matrix = None
    aggregated_pvalue_matrix = None
    aggregated_pvalue_adjusted_matrix = None
    significant_counts_matrix = None
    all_clusters_union = []
    
    if len(enrichment_matrices) > 1:
        # Get union of all clusters
        all_cluster_sets = [set(clusters) for clusters in roi_cluster_map.values()]
        all_clusters_union = (
            sort_cluster_values(
                set().union(*all_cluster_sets),
                canonical=(cluster_key in {'cluster', 'cluster_id'}),
            )
            if all_cluster_sets
            else []
        )
        
        if all_clusters_union:
            # Align all matrices to the union of clusters
            aligned_matrices = []
            aligned_pvalue_matrices = []
            significant_matrices = []  # Track significant interactions per ROI
            n_clusters = len(all_clusters_union)
            
            for roi_id, matrix in enrichment_matrices:
                roi_clusters = roi_cluster_map.get(roi_id)
                
                if roi_clusters is not None:
                    # Create aligned matrix
                    aligned_matrix = np.full((n_clusters, n_clusters), np.nan)
                    aligned_pvalue_matrix = np.full((n_clusters, n_clusters), np.nan)
                    significant_matrix = np.zeros((n_clusters, n_clusters), dtype=bool)
                    
                    # Map old indices to new indices
                    cluster_to_new_idx = {clust: idx for idx, clust in enumerate(all_clusters_union)}
                    roi_pvalues = pvalue_matrices.get(roi_id)
                    
                    # Fill in values where clusters overlap
                    for i, old_clust_i in enumerate(roi_clusters):
                        if old_clust_i in cluster_to_new_idx:
                            new_i = cluster_to_new_idx[old_clust_i]
                            for j, old_clust_j in enumerate(roi_clusters):
                                if old_clust_j in cluster_to_new_idx:
                                    new_j = cluster_to_new_idx[old_clust_j]
                                    aligned_matrix[new_i, new_j] = matrix[i, j]
                                    if isinstance(roi_pvalues, np.ndarray) and roi_pvalues.shape == matrix.shape:
                                        aligned_pvalue_matrix[new_i, new_j] = roi_pvalues[i, j]
                                    # Mark as significant if |z-score| > threshold
                                    if not np.isnan(matrix[i, j]) and abs(matrix[i, j]) > significance_threshold:
                                        significant_matrix[new_i, new_j] = True
                    
                    aligned_matrices.append(aligned_matrix)
                    aligned_pvalue_matrices.append(aligned_pvalue_matrix)
                    significant_matrices.append(significant_matrix)
                else:
                    aligned_matrices.append(matrix)
                    aligned_pvalue_matrices.append(np.asarray(pvalue_matrices.get(roi_id), dtype=float) if roi_id in pvalue_matrices else np.full(matrix.shape, np.nan))
                    # Create significant matrix for this ROI
                    significant_matrix = np.abs(matrix) > significance_threshold
                    significant_matrices.append(significant_matrix)
            
            # Aggregate z-scores
            stacked = np.stack(aligned_matrices, axis=0)
            if aggregation == 'mean':
                aggregated_matrix = np.nanmean(stacked, axis=0)
            else:  # sum
                aggregated_matrix = np.nansum(stacked, axis=0)
            
            # Count significant interactions across ROIs
            significant_stacked = np.stack(significant_matrices, axis=0)
            significant_counts_matrix = np.sum(significant_stacked, axis=0).astype(int)
            if aligned_pvalue_matrices:
                aggregated_pvalue_matrix = np.full((n_clusters, n_clusters), np.nan, dtype=float)
                for row_idx in range(n_clusters):
                    for col_idx in range(n_clusters):
                        p_values = [
                            matrix[row_idx, col_idx]
                            for matrix in aligned_pvalue_matrices
                            if isinstance(matrix, np.ndarray) and np.isfinite(matrix[row_idx, col_idx])
                        ]
                        aggregated_pvalue_matrix[row_idx, col_idx] = combine_pvalues_fisher(p_values)
                aggregated_pvalue_adjusted_matrix = benjamini_hochberg_adjust_matrix(aggregated_pvalue_matrix)
        else:
            aggregated_matrix = enrichment_matrices[0][1] if enrichment_matrices else None
            if aggregated_matrix is not None:
                significant_counts_matrix = (np.abs(aggregated_matrix) > significance_threshold).astype(int)
    elif len(enrichment_matrices) == 1:
        aggregated_matrix = enrichment_matrices[0][1]
        all_clusters_union = roi_cluster_map.get(enrichment_matrices[0][0], [])
        if aggregated_matrix is not None:
            significant_counts_matrix = (np.abs(aggregated_matrix) > significance_threshold).astype(int)
        aggregated_pvalue_matrix = pvalue_matrices.get(enrichment_matrices[0][0])
        aggregated_pvalue_adjusted_matrix = pvalue_adjusted_matrices.get(enrichment_matrices[0][0])
    
    
    return {
        'results': results,
        'aggregated': aggregated_matrix,
        'aggregated_p_values': aggregated_pvalue_matrix,
        'aggregated_p_values_adjusted': aggregated_pvalue_adjusted_matrix,
        'cluster_categories': all_clusters_union,
        'significant_counts': significant_counts_matrix
    }


def spatial_cooccurrence(
    anndata_dict: Dict[str, 'ad.AnnData'],
    cluster_key: str = "cluster",
    interval: List[float] = [10, 20, 30, 50, 100],
    reference_cluster: Optional[str] = None
) -> Dict[str, 'ad.AnnData']:
    """Compute co-occurrence analysis using squidpy.
    
    Args:
        anndata_dict: Dictionary mapping ROI ID to AnnData object with spatial graph
        cluster_key: Column name containing cluster labels (default: "cluster")
        interval: List of distances in micrometers for co-occurrence analysis
        reference_cluster: Optional reference cluster for co-occurrence
    
    Returns:
        Dictionary mapping ROI ID to AnnData object with co-occurrence results
    """
    try:
        import squidpy as sq
    except (ImportError, OSError, RuntimeError):
        raise ImportError("squidpy is required for co-occurrence analysis")
    
    if len(interval) < 2:
        raise ValueError("Co-occurrence analysis requires at least 2 distances in interval")
    
    results = {}
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        if cluster_key not in adata.obs.columns:
            continue
        
        # Filter out cells with NaN cluster values
        cluster_values = adata.obs[cluster_key]
        nan_mask = pd.isna(cluster_values)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            valid_mask = ~nan_mask
            adata = adata[valid_mask].copy()
        
        # Ensure categorical
        if not hasattr(adata.obs[cluster_key], 'cat'):
            adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
        
        # Run co-occurrence analysis
        sq.gr.co_occurrence(adata, cluster_key=cluster_key, interval=interval)
        
        results[roi_id] = adata
    
    return results


def spatial_autocorrelation(
    anndata_dict: Dict[str, 'ad.AnnData'],
    markers: Optional[List[str]] = None,
    aggregation: str = "mean",
    n_permutations: Optional[int] = 999,
) -> Dict[str, Any]:
    """Compute spatial autocorrelation (Moran's I) using squidpy.
    
    Args:
        anndata_dict: Dictionary mapping ROI ID to AnnData object with spatial graph
        markers: Optional list of marker names to analyze. If None, analyzes all features
        aggregation: Aggregation method for multiple ROIs ("mean" or "sum", default: "mean")
    
    Returns:
        Dictionary with:
            - 'results': Dict mapping ROI ID to AnnData object with autocorrelation results
            - 'aggregated': Aggregated results (if multiple ROIs)
    """
    try:
        import squidpy as sq
    except (ImportError, OSError, RuntimeError):
        raise ImportError("squidpy is required for spatial autocorrelation")

    def _moran_to_dataframe(moran_data: Any) -> pd.DataFrame:
        if isinstance(moran_data, pd.DataFrame):
            frame = moran_data.copy()
            if 'var_names' not in frame.columns:
                frame.insert(0, 'var_names', frame.index.astype(str))
            return frame.reset_index(drop=True)

        if isinstance(moran_data, dict):
            data: Dict[str, Any] = {}
            gene_names = moran_data.get('var_names')
            if gene_names is not None:
                gene_names = np.asarray(gene_names, dtype=object).reshape(-1)
                data['var_names'] = gene_names.astype(str)
                n_rows = len(gene_names)
            else:
                n_rows = 0

            for key, value in moran_data.items():
                if key == 'var_names':
                    continue
                if np.isscalar(value):
                    if n_rows == 0:
                        continue
                    data[key] = np.repeat(value, n_rows)
                    continue
                array = np.asarray(value).reshape(-1)
                if array.size == 0:
                    continue
                if n_rows == 0:
                    n_rows = array.size
                if array.size == n_rows:
                    data[key] = array

            if 'var_names' not in data and n_rows > 0:
                data['var_names'] = np.asarray([f'feature_{idx}' for idx in range(n_rows)], dtype=object)
            return pd.DataFrame(data)

        return pd.DataFrame()
    
    results = {}
    moran_results = []
    all_genes = set()
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        # Exclude 'touches_edge' from analysis
        var_names_list = list(adata.var_names) if hasattr(adata.var_names, '__iter__') else []
        
        # Run spatial autocorrelation
        if markers is not None:
            # Filter out 'touches_edge' from markers
            markers_filtered = [g for g in markers if g != 'touches_edge']
            available_genes = [g for g in markers_filtered if g in var_names_list]
            if not available_genes:
                continue
            sq.gr.spatial_autocorr(
                adata,
                mode="moran",
                genes=available_genes,
                n_perms=n_permutations,
                corr_method='fdr_bh',
                show_progress_bar=False,
            )
            all_genes.update(available_genes)
        else:
            # Filter out 'touches_edge' before running autocorrelation
            if 'touches_edge' in var_names_list:
                # Create a filtered view excluding touches_edge
                var_mask = adata.var_names != 'touches_edge'
                adata_filtered = adata[:, var_mask].copy()
                sq.gr.spatial_autocorr(
                    adata_filtered,
                    mode="moran",
                    n_perms=n_permutations,
                    corr_method='fdr_bh',
                    show_progress_bar=False,
                )
                # Copy results back to original adata
                if 'moranI' in adata_filtered.uns:
                    adata.uns['moranI'] = adata_filtered.uns['moranI']
                var_names_list_filtered = [v for v in var_names_list if v != 'touches_edge']
                all_genes.update(var_names_list_filtered)
            else:
                sq.gr.spatial_autocorr(
                    adata,
                    mode="moran",
                    n_perms=n_permutations,
                    corr_method='fdr_bh',
                    show_progress_bar=False,
                )
                all_genes.update(var_names_list)
        
        # Extract results
        moran_data = adata.uns.get('moranI', {})
        
        # Check if moran_data is not empty
        has_data = False
        if isinstance(moran_data, pd.DataFrame):
            has_data = not moran_data.empty
        elif isinstance(moran_data, dict):
            has_data = len(moran_data) > 0
        elif hasattr(moran_data, '__len__'):
            try:
                has_data = len(moran_data) > 0
            except (TypeError, ValueError):
                has_data = False
        else:
            has_data = moran_data is not None and moran_data != {}
        
        if has_data:
            results[roi_id] = adata
            moran_results.append({
                'adata': adata,
                'moranI': moran_data,
                'frame': _moran_to_dataframe(moran_data),
            })
    
    # Aggregate results if multiple ROIs
    aggregated_adata = None
    if len(moran_results) > 1:
        common_genes = sorted(all_genes)
        pvalue_columns = sorted(
            {
                column
                for result in moran_results
                for column in result['frame'].columns
                if column.startswith('pval') and not column.endswith('_fdr_bh')
            }
        )
        variance_columns = sorted(
            {
                column
                for result in moran_results
                for column in result['frame'].columns
                if column.startswith('var_')
            }
        )

        aggregated_rows = []
        for gene in common_genes:
            row: Dict[str, Any] = {'var_names': gene}
            I_vals = []
            for result in moran_results:
                frame = result['frame']
                if frame.empty or 'var_names' not in frame.columns:
                    continue
                matched = frame.loc[frame['var_names'].astype(str) == str(gene)]
                if matched.empty:
                    continue
                matched_row = matched.iloc[0]
                if 'I' in matched_row.index and pd.notna(matched_row['I']):
                    I_vals.append(float(matched_row['I']))
                elif 'moranI' in matched_row.index and pd.notna(matched_row['moranI']):
                    I_vals.append(float(matched_row['moranI']))

            if I_vals:
                row['I'] = float(np.nanmean(I_vals) if aggregation == 'mean' else np.nansum(I_vals))

            for variance_column in variance_columns:
                variance_values = []
                for result in moran_results:
                    frame = result['frame']
                    if frame.empty or variance_column not in frame.columns:
                        continue
                    matched = frame.loc[frame['var_names'].astype(str) == str(gene), variance_column]
                    if matched.empty:
                        continue
                    value = pd.to_numeric(matched, errors='coerce').iloc[0]
                    if pd.notna(value):
                        variance_values.append(float(value))
                if variance_values:
                    row[variance_column] = float(np.nanmean(variance_values))

            for pvalue_column in pvalue_columns:
                gene_pvalues = []
                for result in moran_results:
                    frame = result['frame']
                    if frame.empty or pvalue_column not in frame.columns:
                        continue
                    matched = frame.loc[frame['var_names'].astype(str) == str(gene), pvalue_column]
                    if matched.empty:
                        continue
                    value = pd.to_numeric(matched, errors='coerce').iloc[0]
                    if pd.notna(value):
                        gene_pvalues.append(float(value))
                if gene_pvalues:
                    row[pvalue_column] = combine_pvalues_fisher(gene_pvalues)

            aggregated_rows.append(row)

        aggregated_frame = pd.DataFrame(aggregated_rows)
        if not aggregated_frame.empty:
            for pvalue_column in pvalue_columns:
                if pvalue_column in aggregated_frame.columns:
                    aggregated_frame[f'{pvalue_column}_fdr_bh'] = benjamini_hochberg_adjust_matrix(
                        aggregated_frame[pvalue_column].to_numpy(dtype=float, copy=False)
                    )
            if 'I' in aggregated_frame.columns:
                aggregated_frame = aggregated_frame.sort_values('I', ascending=False, na_position='last').reset_index(drop=True)

            class TempAnnData:
                def __init__(self, frame: pd.DataFrame):
                    self.uns = {'moranI': frame}

            aggregated_adata = TempAnnData(aggregated_frame)
    elif len(moran_results) == 1:
        aggregated_adata = moran_results[0]['adata']
    
    return {
        'results': results,
        'aggregated': aggregated_adata
    }


def spatial_ripley(
    anndata_dict: Dict[str, 'ad.AnnData'],
    cluster_key: str = "cluster",
    mode: str = "L",
    max_dist: float = 50.0
) -> Dict[str, 'ad.AnnData']:
    """Compute Ripley functions using squidpy.
    
    Args:
        anndata_dict: Dictionary mapping ROI ID to AnnData object with spatial graph
        cluster_key: Column name containing cluster labels (default: "cluster")
        mode: Ripley function mode ("F", "G", or "L", default: "L")
        max_dist: Maximum distance in micrometers (default: 50.0)
    
    Returns:
        Dictionary mapping ROI ID to AnnData object with Ripley results
    """
    try:
        import squidpy as sq
    except (ImportError, OSError, RuntimeError):
        raise ImportError("squidpy is required for Ripley analysis")
    
    if mode not in ['F', 'G', 'L']:
        raise ValueError(f"Invalid Ripley mode: {mode}. Must be 'F', 'G', or 'L'")
    
    results = {}
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        if cluster_key not in adata.obs.columns:
            continue
        
        # Filter out cells with NaN cluster values
        cluster_values = adata.obs[cluster_key]
        nan_mask = pd.isna(cluster_values)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            valid_mask = ~nan_mask
            adata = adata[valid_mask].copy()
        
        # Ensure categorical
        if not hasattr(adata.obs[cluster_key], 'cat'):
            adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
        
        # Check cluster sizes and filter out clusters with < 2 cells
        cluster_counts = adata.obs[cluster_key].value_counts()
        valid_clusters = cluster_counts[cluster_counts >= 2].index.tolist()
        
        if len(valid_clusters) == 0:
            continue
        
        if len(valid_clusters) < len(cluster_counts):
            # Filter adata to only include valid clusters
            adata_filtered = adata[adata.obs[cluster_key].isin(valid_clusters)].copy()
            if adata_filtered.n_obs == 0:
                continue
            adata = adata_filtered
        
        try:
            # Run Ripley analysis
            sq.gr.ripley(adata, cluster_key=cluster_key, mode=mode, max_dist=max_dist)
            ripley_key = next((key for key in adata.uns.keys() if 'ripley' in key.lower()), None)
            if ripley_key is not None:
                ripley_data = adata.uns.get(ripley_key)
                if isinstance(ripley_data, dict):
                    p_values = ripley_data.get('pvalues')
                    if p_values is not None:
                        ripley_data['pvalues_fdr_bh'] = benjamini_hochberg_adjust_matrix(p_values)
            results[roi_id] = adata
        except (ValueError, Exception) as e:
            # Skip if insufficient samples
            if "n_neighbors" in str(e) or "n_samples_fit" in str(e):
                continue
            raise
    
    return results


def export_anndata(
    anndata_dict: Dict[str, 'ad.AnnData'],
    output_path: Union[str, Path],
    combined: bool = True
) -> Path:
    """Export AnnData objects to file(s).
    
    This is the unified export function used by both GUI and CLI.
    
    Args:
        anndata_dict: Dictionary mapping ROI ID to AnnData object
        output_path: Path to output file (if combined=True) or directory (if combined=False)
        combined: If True, export as single combined file. If False, export separate files per ROI
    
    Returns:
        Path to exported file(s)
    """
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required for AnnData export")
    
    output_path = Path(output_path)
    
    if combined:
        # Export as single combined file
        if not output_path.suffix:
            output_path = output_path.with_suffix('.h5ad')
        
        adata_list = list(anndata_dict.values())
        if len(adata_list) == 1:
            combined_adata = adata_list[0]
        else:
            combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
        
        combined_adata.write(str(output_path))
        return output_path
    else:
        # Export as separate files per ROI
        output_path.mkdir(parents=True, exist_ok=True)
        
        for roi_id, adata in anndata_dict.items():
            file_path = output_path / f"anndata_roi_{roi_id}.h5ad"
            adata.write(str(file_path))
        
        return output_path


def get_panel(
    acq_info: AcquisitionInfo,
    output_path: Union[str, Path]
) -> Path:
    """Generate a panel.csv file from acquisition information.
    
    Creates a CSV file with two columns:
    - channel: Metal tag/channel identifier
    - name: Channel name/label
    
    Args:
        acq_info: AcquisitionInfo object containing channel metadata
        output_path: Path where panel.csv will be saved
    
    Returns:
        Path to the created panel.csv file
    
    Raises:
        ValueError: If channel_metals and channel_labels are empty or mismatched
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get channel metals and labels
    channel_metals = acq_info.channel_metals if acq_info.channel_metals else []
    channel_labels = acq_info.channel_labels if acq_info.channel_labels else []
    
    # If both are empty, try to extract from channel names
    if not channel_metals and not channel_labels and acq_info.channels:
        # Try to parse channel names (format: "Label_Metal" or just "Metal")
        for channel in acq_info.channels:
            if '_' in channel:
                parts = channel.split('_', 1)
                if len(parts) == 2:
                    channel_labels.append(parts[0])
                    channel_metals.append(parts[1])
                else:
                    channel_labels.append("")
                    channel_metals.append(channel)
            else:
                channel_labels.append("")
                channel_metals.append(channel)
    
    # Ensure both lists have the same length
    max_len = max(len(channel_metals), len(channel_labels))
    while len(channel_metals) < max_len:
        channel_metals.append("")
    while len(channel_labels) < max_len:
        channel_labels.append("")
    
    if not channel_metals and not channel_labels:
        raise ValueError("No channel information available in acquisition")
    
    # Create DataFrame
    panel_data = {
        'channel': channel_metals,
        'name': channel_labels
    }
    
    df = pd.DataFrame(panel_data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return output_path
