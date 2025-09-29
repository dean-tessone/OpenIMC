from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from skimage.measure import regionprops, regionprops_table

from openmcd.data.mcd_loader import MCDLoader
from openmcd.ui.utils import arcsinh_normalize

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
        
        if arcsinh_enabled:
            print(f"[feature_worker] Applying arcsinh normalization with cofactor={cofactor}")
            img_stack = arcsinh_normalize(img_stack, cofactor=cofactor)

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
        print(f"[feature_worker] Computing intensity features for {len(channel_names)} channels")
        for idx, ch_name in enumerate(channel_names):
            ch_img = img_stack[..., idx]
            if ch_img.ndim != 2:
                print(f"[feature_worker] Warning: channel {ch_name} has invalid shape {ch_img.shape}")
            # Mean intensity via regionprops_table
            inten_df = pd.DataFrame(regionprops_table(label_image, intensity_image=ch_img, properties=("label", "mean_intensity")))
            inten_df.rename(columns={"mean_intensity": f"{ch_name}_mean"}, inplace=True)

            # Compute std, p10, p90, integrated, frac_pos manually
            # Build per-label lists
            labels = inten_df["label"].to_numpy()
            std_vals = np.zeros_like(labels, dtype=np.float64)
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
                p10_vals[i] = float(np.percentile(pix, 10))
                p90_vals[i] = float(np.percentile(pix, 90))
                integrated_vals[i] = float(np.mean(pix) * pix.size)
                frac_pos_vals[i] = float(np.count_nonzero(pix > 0) / pix.size)

            inten_df[f"{ch_name}_std"] = std_vals
            inten_df[f"{ch_name}_p10"] = p10_vals
            inten_df[f"{ch_name}_p90"] = p90_vals
            inten_df[f"{ch_name}_integrated"] = integrated_vals
            inten_df[f"{ch_name}_frac_pos"] = frac_pos_vals

            # Merge with morphology on label
            morph_df = morph_df.merge(inten_df, on="label", how="left")

        # Add acquisition id and cell id
        morph_df.rename(columns={"label": "cell_id"}, inplace=True)
        morph_df.insert(0, "acquisition_id", acq_id)
        morph_df.insert(1, "acquisition_label", acq_label)

        print(f"[feature_worker] Finished extraction acq_id={acq_id}, rows={len(morph_df)}")
        return morph_df

    except Exception as e:
        print(f"[feature_worker] ERROR in extraction acq_id={acq_id}: {e}")
        # Return empty on error to keep pipeline robust
        return pd.DataFrame()

