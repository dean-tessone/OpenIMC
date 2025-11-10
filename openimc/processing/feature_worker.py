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

import numpy as np
import pandas as pd

from skimage.measure import regionprops, regionprops_table

from openimc.data.mcd_loader import MCDLoader
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.ui.utils import arcsinh_normalize

# Optional spillover correction
try:
    from openimc.processing.spillover_correction import compensate_counts
    _HAVE_SPILLOVER = True
except ImportError:
    _HAVE_SPILLOVER = False

# Optional scikit-image for denoising
try:
    from skimage import morphology, filters
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
    if not _HAVE_SCIKIT_IMAGE or not denoise_settings:
        return channel_img

    result = channel_img.copy()

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
        elif method == "rolling_ball" and _HAVE_ROLLING_BALL:
            # Approximate rolling-ball via top-hat background estimate
            selem = disk(radius)
            background = morphology.white_tophat(result, selem)
            result = result - background

    return result


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
        print(f"[feature_worker] Start extraction acq_id={acq_id}, arcsinh={arcsinh_enabled}, cofactor={cofactor}")
        print(f"[feature_worker] Processing image stack shape: {img_stack.shape}")
        
        # Apply denoising per channel only when the source is explicitly "Custom".
        # This ensures we operate on original (raw) images and do not double-denoise
        # images that may already reflect viewer/segmentation preprocessing.
        if denoise_source == "custom" and custom_denoise_settings:
            print("[feature_worker] Applying custom denoising to raw image stack")
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

        print(f"[feature_worker] Computing morph props: {props_to_compute}")
        morph_df = pd.DataFrame(regionprops_table(label_image, properties=tuple(props_to_compute)))
        print(f"[feature_worker] Morph props rows: {len(morph_df)} cols: {list(morph_df.columns)}")

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
            print(f"[feature_worker] Excluding {len(excluded_channels_set)} channels from feature extraction")
            print(f"[feature_worker] Computing intensity features for {len(channel_names)} channels (after exclusion)")
        else:
            channel_indices = list(range(len(channel_names)))
            print(f"[feature_worker] Computing intensity features for {len(channel_names)} channels")
        
        for filtered_idx, original_idx in enumerate(channel_indices):
            ch_name = channel_names[filtered_idx]
            ch_img = img_stack[..., original_idx]
            if ch_img.ndim != 2:
                print(f"[feature_worker] Warning: channel {ch_name} has invalid shape {ch_img.shape}")
            # Mean intensity via regionprops_table
            inten_df = pd.DataFrame(regionprops_table(label_image, intensity_image=ch_img, properties=("label", "mean_intensity")))
            inten_df.rename(columns={"mean_intensity": f"{ch_name}_mean"}, inplace=True)

            # Compute std, median, mad, p10, p90, integrated, frac_pos manually
            # Build per-label lists
            labels = inten_df["label"].to_numpy()
            std_vals = np.zeros_like(labels, dtype=np.float64)
            median_vals = np.zeros_like(labels, dtype=np.float64)
            mad_vals = np.zeros_like(labels, dtype=np.float64)
            p10_vals = np.zeros_like(labels, dtype=np.float64)
            p90_vals = np.zeros_like(labels, dtype=np.float64)
            integrated_vals = np.zeros_like(labels, dtype=np.float64)
            frac_pos_vals = np.zeros_like(labels, dtype=np.float64)

            for i, lbl in enumerate(labels):
                mask_lbl = (label_image == lbl)
                pix = ch_img[mask_lbl]
                if pix.size == 0:
                    continue
                std_vals[i] = float(np.std(pix))
                median_vals[i] = float(np.median(pix))
                mad_vals[i] = float(np.median(np.abs(pix - np.median(pix))))
                p10_vals[i] = float(np.percentile(pix, 10))
                p90_vals[i] = float(np.percentile(pix, 90))
                integrated_vals[i] = float(np.mean(pix) * pix.size)
                frac_pos_vals[i] = float(np.count_nonzero(pix > 0) / pix.size)

            inten_df[f"{ch_name}_std"] = std_vals
            inten_df[f"{ch_name}_median"] = median_vals
            inten_df[f"{ch_name}_mad"] = mad_vals
            inten_df[f"{ch_name}_p10"] = p10_vals
            inten_df[f"{ch_name}_p90"] = p90_vals
            inten_df[f"{ch_name}_integrated"] = integrated_vals
            inten_df[f"{ch_name}_frac_pos"] = frac_pos_vals

            # Merge with morphology on label
            morph_df = morph_df.merge(inten_df, on="label", how="left")

        # Apply spillover correction to extracted intensity features (after feature extraction, before arcsinh)
        # Spillover correction operates on raw intensity values (linear scale)
        if spillover_config and _HAVE_SPILLOVER:
            spillover_matrix = spillover_config.get('matrix')
            spillover_method = spillover_config.get('method', 'pgd')
            channel_names = acq_info.get("channels", [])
            
            if spillover_matrix is not None and len(channel_names) > 0:
                print(f"[feature_worker] Applying spillover correction to intensity features (method={spillover_method})")
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
                    
                    print(f"[feature_worker] Spillover correction applied successfully to intensity features")
                except Exception as e:
                    print(f"[feature_worker] WARNING: Spillover correction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue without spillover correction rather than failing
        
        # Apply arcsinh transformation to extracted intensity features if enabled
        # Note: frac_pos is a proportion (0-1), so it should not be transformed
        if arcsinh_enabled:
            print(f"[feature_worker] Applying arcsinh transformation to extracted intensity features with cofactor={cofactor}")
            for ch_name in channel_names:
                intensity_feature_cols = [
                    f"{ch_name}_mean",
                    f"{ch_name}_median",
                    f"{ch_name}_std",
                    f"{ch_name}_mad",
                    f"{ch_name}_p10",
                    f"{ch_name}_p90",
                    f"{ch_name}_integrated"
                ]
                for col in intensity_feature_cols:
                    if col in morph_df.columns:
                        # Apply arcsinh transform to the feature values (1D array)
                        morph_df[col] = arcsinh_normalize(morph_df[col].values, cofactor=cofactor)

        # Add acquisition id and cell id
        morph_df.rename(columns={"label": "cell_id"}, inplace=True)
        # Add source file name (just the filename, not full path)
        if source_file:
            import os
            source_filename = os.path.basename(source_file)
        else:
            source_filename = None
        # Use pd.concat instead of multiple insert() calls to avoid DataFrame fragmentation
        metadata_df = pd.DataFrame({
            "acquisition_id": [acq_id] * len(morph_df),
            "acquisition_label": [acq_label] * len(morph_df),
            "source_file": [source_filename] * len(morph_df)
        })
        morph_df = pd.concat([metadata_df, morph_df], axis=1)

        print(f"[feature_worker] Finished extraction acq_id={acq_id}, rows={len(morph_df)}")
        return morph_df

    except Exception as e:
        print(f"[feature_worker] ERROR in extraction acq_id={acq_id}: {e}")
        # Return empty on error to keep pipeline robust
        return pd.DataFrame()


def load_and_extract_features(
    acq_id: str,
    mask: np.ndarray,
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
    """Load image data and extract features in a single worker process.
    
    This function combines image loading and feature extraction to enable
    parallelization of both I/O and computation.
    
    Arguments MUST be picklable. Returns an empty DataFrame on error.
    """
    try:
        print(f"[feature_worker] Loading and extracting features for acq_id={acq_id}, loader_type={loader_type}")
        
        # Create loader and load image data
        if loader_type == "mcd":
            loader = MCDLoader()
            loader.open(file_path)
            img_stack = loader.get_all_channels(acq_id)
            loader.close()
        elif loader_type == "ometiff":
            loader = OMETIFFLoader(channel_format='CHW')  # Default to CHW (matches export format)
            loader.open(file_path)
            img_stack = loader.get_all_channels(acq_id)
            # OMETIFFLoader doesn't need explicit close, but we can clear cache
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        print(f"[feature_worker] Loaded image stack shape: {img_stack.shape} for acq_id={acq_id}")
        
        # Now extract features using the existing extraction function
        return extract_features_for_acquisition(
            acq_id=acq_id,
            mask=mask,
            selected_features=selected_features,
            acq_info=acq_info,
            acq_label=acq_label,
            img_stack=img_stack,
            arcsinh_enabled=arcsinh_enabled,
            cofactor=cofactor,
            denoise_source=denoise_source,
            custom_denoise_settings=custom_denoise_settings,
            spillover_config=spillover_config,
            source_file=source_file,
            excluded_channels=excluded_channels
        )
        
    except Exception as e:
        print(f"[feature_worker] ERROR in load_and_extract_features for acq_id={acq_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

