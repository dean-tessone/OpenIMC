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

from typing import Dict, List, Optional, Tuple
import os
import sys
import time

import numpy as np
import pandas as pd

from skimage.measure import regionprops, regionprops_table

INTENSITY_FEATURE_TYPES = ("mean", "median", "std", "mad", "p10", "p90", "integrated", "frac_pos")


def drop_excluded_channel_feature_columns(
    features_df: pd.DataFrame,
    excluded_channels: Optional[set]
) -> pd.DataFrame:
    """Remove intensity feature columns for excluded channels from a feature table.

    This enforces a strict schema rule: excluded channels should not exist as
    columns in the final feature dataframe (rather than appearing as all-NaN).
    """
    if features_df is None or features_df.empty or not excluded_channels:
        return features_df

    excluded = {str(ch) for ch in excluded_channels if ch is not None}
    if not excluded:
        return features_df

    cols_to_drop: List[str] = []
    for ch_name in excluded:
        for feature_type in INTENSITY_FEATURE_TYPES:
            col = f"{ch_name}_{feature_type}"
            if col in features_df.columns:
                cols_to_drop.append(col)

    if cols_to_drop:
        features_df = features_df.drop(columns=sorted(set(cols_to_drop)))

    return features_df


def _compute_mean_intensity_manual(label_image: np.ndarray, intensity_image: np.ndarray) -> pd.DataFrame:
    """Manually compute mean intensity per label as fallback when regionprops_table hangs.
    
    Uses vectorized operations for efficiency.
    """
    from scipy import ndimage as ndi
    
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
    
    if len(unique_labels) == 0:
        return pd.DataFrame({"label": [], "mean_intensity": []})
    
    # Use scipy.ndimage for efficient per-label statistics
    # This is much faster than looping over labels
    label_counts = ndi.labeled_comprehension(
        np.ones_like(intensity_image, dtype=np.float64),
        label_image, unique_labels, np.sum, float, 0.0
    )
    
    label_sums = ndi.labeled_comprehension(
        intensity_image.astype(np.float64),
        label_image, unique_labels, np.sum, float, 0.0
    )
    
    # Compute mean: sum / count
    mean_intensities = np.divide(
        label_sums, 
        label_counts, 
        out=np.zeros_like(label_sums, dtype=np.float64),
        where=(label_counts > 0)
    )
    
    return pd.DataFrame({
        "label": unique_labels,
        "mean_intensity": mean_intensities.astype(np.float64)
    })

from openimc.data.mcd_loader import MCDLoader
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.denoising import apply_channel_denoise
from openimc.ui.utils import arcsinh_normalize

# Optional spillover correction
try:
    from openimc.processing.spillover_correction import compensate_counts
    _HAVE_SPILLOVER = True
except ImportError:
    _HAVE_SPILLOVER = False

def _apply_denoise_to_channel(channel_img: np.ndarray, channel_name: str, denoise_settings: Dict) -> np.ndarray:
    """Apply denoising to a single channel image based on settings.

    Expects a structure like:
      {
        "hot": {"method": "median3" | "n_sd_local_median", "n_sd": float} | None,
        "speckle": {"method": "gaussian" | "nl_means", "sigma": float} | None,
        "background": {"method": "white_tophat" | "black_tophat" | "rolling_ball", "radius": int} | None
      }
    Any of the three keys may be missing or None.
    """
    return apply_channel_denoise(channel_img, denoise_settings)


def extract_features_for_acquisition(
    acq_id: str,
    mask: np.ndarray,
    selected_features: Dict[str, bool],
    acq_info: Dict,
    acq_label: str,
    img_stack: np.ndarray,
    arcsinh_enabled: bool,
    cofactor: float,
    denoise_source: str = "None",
    custom_denoise_settings: Dict = None,
    spillover_config: Optional[Dict] = None,
    source_file: Optional[str] = None,
    excluded_channels: Optional[set] = None,
) -> pd.DataFrame:
    """Module-level worker that extracts features for a single acquisition.

    Arguments MUST be picklable. Returns an empty DataFrame on error.
    """
    try:
        # Apply denoising per channel only when the source is explicitly "Custom".
        # This ensures we operate on original (raw) images and do not double-denoise
        # images that may already reflect viewer/segmentation preprocessing.
        if denoise_source == "custom" and custom_denoise_settings:
            for idx, ch_name in enumerate(acq_info.get("channels", [])):
                cfg = custom_denoise_settings.get(ch_name)
                if not cfg:
                    continue
                ch_img = img_stack[..., idx]
                denoised_img = _apply_denoise_to_channel(ch_img, ch_name, cfg)
                img_stack[..., idx] = denoised_img
        
        # Note: arcsinh normalization is NOT applied to images before feature extraction.
        # Instead, arcsinh transform is applied to the extracted intensity features after extraction.

        # Ensure mask is int labels
        label_image = mask.astype(np.int32, copy=False)

        # Morphology features
        rows: Dict[str, np.ndarray] = {}
        props_to_compute: List[str] = ["label"]
        if selected_features.get("area_um2", True):
            props_to_compute.append("area")
        if selected_features.get("perimeter_um", True):
            props_to_compute.append("perimeter")
        if selected_features.get("equivalent_diameter_um", False):
            props_to_compute.append("equivalent_diameter")
        if selected_features.get("eccentricity", False):
            props_to_compute.append("eccentricity")
        if selected_features.get("solidity", False):
            props_to_compute.append("solidity")
        if selected_features.get("extent", False):
            props_to_compute.append("extent")
        if selected_features.get("major_axis_len_um", False):
            props_to_compute.append("major_axis_length")
        if selected_features.get("minor_axis_len_um", False):
            props_to_compute.append("minor_axis_length")
        # Add centroid coordinates if requested
        if selected_features.get("centroid_x", False) or selected_features.get("centroid_y", False):
            props_to_compute.append("centroid")

        morph_df = pd.DataFrame(regionprops_table(label_image, properties=tuple(props_to_compute)))

        # Normalize morphometric column names to expected schema used in UI and selectors
        rename_map = {}
        if 'area' in morph_df.columns:
            rename_map['area'] = 'area_um2'
        if 'perimeter' in morph_df.columns:
            rename_map['perimeter'] = 'perimeter_um'
        if 'equivalent_diameter' in morph_df.columns:
            rename_map['equivalent_diameter'] = 'equivalent_diameter_um'
        if 'major_axis_length' in morph_df.columns:
            rename_map['major_axis_length'] = 'major_axis_len_um'
        if 'minor_axis_length' in morph_df.columns:
            rename_map['minor_axis_length'] = 'minor_axis_len_um'
        morph_df.rename(columns=rename_map, inplace=True)

        # Extract centroid coordinates if requested
        if 'centroid-0' in morph_df.columns and 'centroid-1' in morph_df.columns:
            # regionprops_table returns centroid as separate columns
            if selected_features.get("centroid_x", False):
                morph_df['centroid_x'] = morph_df['centroid-1']  # x coordinate (column)
            if selected_features.get("centroid_y", False):
                morph_df['centroid_y'] = morph_df['centroid-0']  # y coordinate (row)
            
            # Remove the original centroid columns
            morph_df.drop(columns=['centroid-0', 'centroid-1'], inplace=True)

        # Derived: aspect_ratio (major/minor) if available
        if {'major_axis_len_um', 'minor_axis_len_um'}.issubset(set(morph_df.columns)):
            with np.errstate(divide='ignore', invalid='ignore'):
                morph_df['aspect_ratio'] = morph_df['major_axis_len_um'] / np.maximum(morph_df['minor_axis_len_um'], 1e-6)

        # Optional derived fields
        if "area_um2" in morph_df.columns and "perimeter_um" in morph_df.columns and selected_features.get("circularity", False):
            with np.errstate(divide="ignore", invalid="ignore"):
                circ = 4.0 * np.pi * morph_df["area_um2"] / np.maximum(morph_df["perimeter_um"], 1e-6) ** 2
            morph_df["circularity"] = circ
        
        # Compute touches_edge: check if cell touches ROI edge (the boundary of the mask/image)
        if selected_features.get("touches_edge", False):
            touches_edge_vals = []
            unique_labels = morph_df["label"].to_numpy()
            
            # Get ROI dimensions
            height, width = label_image.shape
            
            # Create mask for ROI boundary pixels
            # Boundary pixels are at row 0, row (height-1), col 0, or col (width-1)
            roi_edge_mask = np.zeros_like(label_image, dtype=bool)
            roi_edge_mask[0, :] = True  # Top edge
            roi_edge_mask[height - 1, :] = True  # Bottom edge
            roi_edge_mask[:, 0] = True  # Left edge
            roi_edge_mask[:, width - 1] = True  # Right edge
            
            # For each label, check if any of its pixels are on the ROI boundary
            for label in unique_labels:
                label_mask = (label_image == label)
                # Check if any pixels of this cell are on the ROI edge
                touches = np.any(label_mask & roi_edge_mask)
                touches_edge_vals.append(bool(touches))
            
            morph_df["touches_edge"] = touches_edge_vals

        # Intensity features per channel (subset: mean, std, p10, p90, integrated)
        channel_names: List[str] = acq_info.get("channels", [])
        
        # Filter out excluded channels
        if excluded_channels:
            excluded_channels_set = excluded_channels if isinstance(excluded_channels, set) else set(excluded_channels)
            # Create filtered channel list and mapping
            filtered_channels = []
            channel_indices = []
            for idx, ch_name in enumerate(channel_names):
                if ch_name not in excluded_channels_set:
                    filtered_channels.append(ch_name)
                    channel_indices.append(idx)
            channel_names = filtered_channels
        else:
            channel_indices = list(range(len(channel_names)))
        
        selected_intensity_types = {
            feature_type
            for feature_type in INTENSITY_FEATURE_TYPES
            if selected_features.get(feature_type, False)
        }

        if selected_intensity_types:
            for filtered_idx, original_idx in enumerate(channel_indices):
                ch_name = channel_names[filtered_idx]
                ch_img = img_stack[..., original_idx]
                inten_df = _compute_mean_intensity_manual(label_image, ch_img)

                labels = inten_df["label"].to_numpy()
                num_labels = len(labels)
                ch_img_float = ch_img.astype(np.float64)

                # Compute std, median, mad, p10, p90, integrated, frac_pos using
                # vectorized operations wherever possible.
                from scipy import ndimage as ndi

                try:
                    label_sums_sq = ndi.labeled_comprehension(
                        ch_img_float ** 2, label_image, labels, np.sum, float, 0.0
                    )
                    label_counts = ndi.labeled_comprehension(
                        np.ones_like(ch_img_float), label_image, labels, np.sum, float, 0.0
                    )
                    label_sums = ndi.labeled_comprehension(
                        ch_img_float, label_image, labels, np.sum, float, 0.0
                    )

                    mean_vals = np.divide(
                        label_sums,
                        label_counts,
                        out=np.zeros_like(label_sums),
                        where=(label_counts > 0),
                    )
                    mean_sq_vals = np.divide(
                        label_sums_sq,
                        label_counts,
                        out=np.zeros_like(label_sums_sq),
                        where=(label_counts > 0),
                    )
                    std_vals = np.sqrt(np.maximum(mean_sq_vals - mean_vals ** 2, 0.0))

                    median_vals = np.zeros(num_labels, dtype=np.float64)
                    mad_vals = np.zeros(num_labels, dtype=np.float64)
                    p10_vals = np.zeros(num_labels, dtype=np.float64)
                    p90_vals = np.zeros(num_labels, dtype=np.float64)
                    integrated_vals = mean_vals * label_counts

                    positive_mask = (ch_img_float > 0).astype(np.float64)
                    label_positive_counts = ndi.labeled_comprehension(
                        positive_mask, label_image, labels, np.sum, float, 0.0
                    )
                    frac_pos_vals = np.divide(
                        label_positive_counts,
                        label_counts,
                        out=np.zeros_like(label_positive_counts),
                        where=(label_counts > 0),
                    )

                    label_flat = label_image.ravel()
                    img_flat = ch_img_float.ravel()
                    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
                    pixels_by_label = {lbl: [] for lbl in labels}

                    for pixel_val, pixel_label in zip(img_flat, label_flat):
                        if pixel_label in label_to_idx:
                            pixels_by_label[pixel_label].append(pixel_val)

                    for i, lbl in enumerate(labels):
                        if label_counts[i] == 0:
                            median_vals[i] = 0.0
                            mad_vals[i] = 0.0
                            p10_vals[i] = 0.0
                            p90_vals[i] = 0.0
                            continue

                        pix = np.array(pixels_by_label[lbl], dtype=np.float64)
                        if pix.size == 0:
                            continue

                        try:
                            median_vals[i] = float(np.median(pix))
                            p10_vals[i] = float(np.percentile(pix, 10))
                            p90_vals[i] = float(np.percentile(pix, 90))
                            mad_vals[i] = float(
                                np.median(np.abs(pix - median_vals[i]))
                            )
                        except Exception as e:
                            print(f"[feature_worker] [ERROR] Error processing label {lbl} for channel {ch_name}: {e}")
                            median_vals[i] = np.nan
                            mad_vals[i] = np.nan
                            p10_vals[i] = np.nan
                            p90_vals[i] = np.nan

                except Exception as e:
                    print(f"[feature_worker] [ERROR] Error in vectorized computation for channel {ch_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    std_vals = np.zeros(num_labels, dtype=np.float64)
                    median_vals = np.zeros(num_labels, dtype=np.float64)
                    mad_vals = np.zeros(num_labels, dtype=np.float64)
                    p10_vals = np.zeros(num_labels, dtype=np.float64)
                    p90_vals = np.zeros(num_labels, dtype=np.float64)
                    integrated_vals = np.zeros(num_labels, dtype=np.float64)
                    frac_pos_vals = np.zeros(num_labels, dtype=np.float64)

                    for i, lbl in enumerate(labels):
                        mask_lbl = (label_image == lbl)
                        pix = ch_img[mask_lbl]
                        if pix.size == 0:
                            continue
                        try:
                            std_vals[i] = float(np.std(pix))
                            median_vals[i] = float(np.median(pix))
                            mad_vals[i] = float(np.median(np.abs(pix - np.median(pix))))
                            p10_vals[i] = float(np.percentile(pix, 10))
                            p90_vals[i] = float(np.percentile(pix, 90))
                            integrated_vals[i] = float(np.mean(pix) * pix.size)
                            frac_pos_vals[i] = float(np.count_nonzero(pix > 0) / pix.size)
                        except Exception:
                            std_vals[i] = np.nan
                            median_vals[i] = np.nan
                            mad_vals[i] = np.nan
                            p10_vals[i] = np.nan
                            p90_vals[i] = np.nan
                            integrated_vals[i] = np.nan
                            frac_pos_vals[i] = np.nan

                if "mean" in selected_intensity_types:
                    inten_df.rename(columns={"mean_intensity": f"{ch_name}_mean"}, inplace=True)
                else:
                    inten_df = inten_df.drop(columns=["mean_intensity"])
                if "std" in selected_intensity_types:
                    inten_df[f"{ch_name}_std"] = std_vals
                if "median" in selected_intensity_types:
                    inten_df[f"{ch_name}_median"] = median_vals
                if "mad" in selected_intensity_types:
                    inten_df[f"{ch_name}_mad"] = mad_vals
                if "p10" in selected_intensity_types:
                    inten_df[f"{ch_name}_p10"] = p10_vals
                if "p90" in selected_intensity_types:
                    inten_df[f"{ch_name}_p90"] = p90_vals
                if "integrated" in selected_intensity_types:
                    inten_df[f"{ch_name}_integrated"] = integrated_vals
                if "frac_pos" in selected_intensity_types:
                    inten_df[f"{ch_name}_frac_pos"] = frac_pos_vals

                morph_df = morph_df.merge(inten_df, on="label", how="left")

        # Apply spillover correction to extracted intensity features (after feature extraction, before arcsinh)
        # Spillover correction operates on raw intensity values (linear scale)
        if spillover_config and _HAVE_SPILLOVER:
            spillover_matrix = spillover_config.get('matrix')
            spillover_method = spillover_config.get('method', 'pgd')
            channel_names = acq_info.get("channels", [])
            
            if spillover_matrix is not None and len(channel_names) > 0:
                try:
                    # Apply spillover correction to each intensity feature type separately
                    # Intensity features: mean, median, std, mad, p10, p90, integrated
                    # Note: frac_pos is a proportion (0-1), so it should not be corrected
                    intensity_feature_types = ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated']
                    
                    for feature_type in intensity_feature_types:
                        # Extract columns for this feature type across all channels
                        feature_cols = [f"{ch_name}_{feature_type}" for ch_name in channel_names 
                                       if f"{ch_name}_{feature_type}" in morph_df.columns]
                        
                        if not feature_cols:
                            continue
                        
                        # Create a temporary DataFrame with cells x channels for this feature type
                        feature_data = morph_df[feature_cols].copy()
                        # Rename columns to match channel names (remove the feature_type suffix)
                        channel_map = {col: col.replace(f"_{feature_type}", "") for col in feature_cols}
                        feature_data.rename(columns=channel_map, inplace=True)
                        
                        # Apply spillover correction
                        comp_data, _ = compensate_counts(
                            feature_data,
                            spillover_matrix,
                            method=spillover_method,
                            strict_align=False,
                            return_all_channels=True
                        )
                        
                        # Rename columns back and update morph_df
                        comp_data.rename(columns={ch: f"{ch}_{feature_type}" for ch in comp_data.columns}, inplace=True)
                        for col in comp_data.columns:
                            if col in morph_df.columns:
                                morph_df[col] = comp_data[col].values
                    
                except Exception as e:
                    print(f"[feature_worker] WARNING: Spillover correction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue without spillover correction rather than failing
        
        # NOTE: Arcsinh transformation is now applied at the end of feature extraction
        # (after all acquisitions are processed) for efficiency. This code is kept for
        # backward compatibility but arcsinh should be disabled when calling this function
        # if arcsinh_enabled:
        #     print(f"[feature_worker] WARNING: Arcsinh should be applied at end, not per-acquisition")

        # Add acquisition id and cell id
        morph_df.rename(columns={"label": "cell_id"}, inplace=True)
        # Add source file name (just the filename, not full path)
        if source_file:
            import os
            source_filename = os.path.basename(source_file)
        else:
            source_filename = None
        
        # Get well name from acq_info if available
        well_name = acq_info.get("well") if isinstance(acq_info, dict) else None
        
        # Create source_well column: source_file (without extension) + well name
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
        
        # Use pd.concat instead of multiple insert() calls to avoid DataFrame fragmentation
        metadata_df = pd.DataFrame({
            "acquisition_id": [acq_id] * len(morph_df),
            "acquisition_label": [acq_label] * len(morph_df),
            "source_file": [source_filename] * len(morph_df),
            "source_well": [source_well] * len(morph_df)
        })
        morph_df = pd.concat([metadata_df, morph_df], axis=1)
        morph_df = drop_excluded_channel_feature_columns(morph_df, excluded_channels)

        return morph_df

    except Exception as e:
        print(f"[feature_worker] ERROR in extraction acq_id={acq_id}: {e}")
        # Return empty on error to keep pipeline robust
        return pd.DataFrame()


def load_and_extract_features(
    acq_id: str,
    mask: Optional[np.ndarray],
    mask_path: Optional[str],
    selected_features: Dict[str, bool],
    acq_info: Dict,
    acq_label: str,
    file_path: str,
    loader_type: str,  # "mcd" or "ometiff"
    arcsinh_enabled: bool,
    cofactor: float,
    denoise_source: str = "None",
    custom_denoise_settings: Dict = None,
    spillover_config: Optional[Dict] = None,
    source_file: Optional[str] = None,
    excluded_channels: Optional[set] = None,
) -> pd.DataFrame:
    """Load image data and extract features using core.extract_features.
    
    This function combines image loading and feature extraction to enable
    parallelization of both I/O and computation. Now uses core.extract_features
    for unified behavior between CLI and GUI.
    
    Arguments MUST be picklable. Returns an empty DataFrame on error.
    
    Args:
        acq_id: Acquisition ID
        mask: Mask array (None if mask_path is provided)
        mask_path: Path to mask file on disk (None if mask array is provided)
        ... (other args)
    """
    import os
    import tempfile
    import tifffile
    from openimc.data.mcd_loader import AcquisitionInfo
    from openimc.core import extract_features, load_mcd
    
    try:
        # Load and extract features (no file locking - handled by grouping in main process)
        return _load_and_extract_features_unlocked(
            acq_id, mask, mask_path, selected_features, acq_info, acq_label,
            file_path, loader_type, arcsinh_enabled, cofactor, denoise_source,
            custom_denoise_settings, spillover_config, source_file, excluded_channels
        )
    except Exception as e:
        print(f"[feature_worker] [ERROR] Exception in load_and_extract_features for acq_id={acq_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def _load_and_extract_features_unlocked(
    acq_id: str,
    mask: Optional[np.ndarray],
    mask_path: Optional[str],
    selected_features: Dict[str, bool],
    acq_info: Dict,
    acq_label: str,
    file_path: str,
    loader_type: str,
    arcsinh_enabled: bool,
    cofactor: float,
    denoise_source: str,
    custom_denoise_settings: Dict,
    spillover_config: Optional[Dict],
    source_file: Optional[str],
    excluded_channels: Optional[set],
) -> pd.DataFrame:
    """Internal function that performs the actual loading and extraction (without file locking)."""
    import os
    import tempfile
    import tifffile
    from openimc.data.mcd_loader import AcquisitionInfo
    from openimc.core import extract_features, load_mcd
    
    try:
        # Load data using core.load_mcd
        loader, _ = load_mcd(file_path, channel_format='CHW' if loader_type == 'ometiff' else 'CHW')
        
        try:
            # Get metadata from loader or use default
            metadata = acq_info.get('metadata')
            if metadata is None:
                # Try to get metadata from loader (like qc_analysis_worker does)
                metadata = getattr(loader, '_acq_metadata', {}).get(acq_id, {})
            
            # Create AcquisitionInfo
            acq_info_obj = AcquisitionInfo(
                id=acq_id,
                name=acq_info.get('name', acq_id),
                well=acq_info.get('well'),
                size=(None, None),
                channels=acq_info.get('channels', []),
                channel_metals=acq_info.get('channel_metals', []),
                channel_labels=acq_info.get('channel_labels', []),
                metadata=metadata,
                source_file=source_file
            )
            
            # Determine mask path: use provided path if available, otherwise write mask array to temp file
            temp_mask_path = None
            created_temp_file = False
            try:
                if mask_path and os.path.exists(mask_path):
                    # Mask is already on disk - use it directly (no need to copy to temp)
                    temp_mask_path = mask_path
                else:
                    # Mask is in memory - write to temp file (core expects path)
                    if mask is None:
                        raise ValueError(f"No mask provided and mask_path {mask_path} does not exist")
                    temp_mask_path = os.path.join(tempfile.gettempdir(), f"feature_mask_{acq_id}_{os.getpid()}.tif")
                    tifffile.imwrite(temp_mask_path, mask.astype(np.uint32))
                    created_temp_file = True
                
                # Determine feature flags
                morphological = any(k.startswith(('area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'circularity', 'major_axis', 'minor_axis', 'aspect_ratio', 'bbox', 'touches_border', 'holes', 'centroid')) for k, v in selected_features.items() if v)
                intensity = any(k.endswith(('_mean', '_median', '_std', '_mad', '_p10', '_p90', '_integrated', '_frac_pos')) for k, v in selected_features.items() if v)
                
                # Call core function (NOTE: arcsinh is disabled here - will be applied at end)
                result = extract_features(
                    loader=loader,
                    acquisitions=[acq_info_obj],
                    mask_path=temp_mask_path,
                    output_path=None,
                    morphological=morphological,
                    intensity=intensity,
                    denoise_settings=custom_denoise_settings if denoise_source == "custom" else None,
                    arcsinh=False,  # Disable arcsinh here - will apply at end
                    arcsinh_cofactor=cofactor,
                    spillover_config=spillover_config,
                    excluded_channels=excluded_channels,
                    selected_features=selected_features
                )
                
                # NOTE: Arcsinh transformation is now applied at the end after all acquisitions are combined
                # (in main_window.py after pd.concat). This is more efficient than applying per-acquisition.
                
                return result
            finally:
                # Only delete temp file if we created it (not if it was provided as mask_path)
                if created_temp_file and temp_mask_path and os.path.exists(temp_mask_path):
                    os.remove(temp_mask_path)
        finally:
            if hasattr(loader, 'close'):
                loader.close()
    except Exception as e:
        print(f"[feature_worker] [ERROR] Exception in load_and_extract_features for acq_id={acq_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
