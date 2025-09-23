from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from skimage.measure import regionprops, regionprops_table

from openmcd.data.mcd_loader import MCDLoader
from openmcd.ui.utils import arcsinh_normalize


def extract_features_for_acquisition(
    acq_id: str,
    mask: np.ndarray,
    selected_features: Dict[str, bool],
    acq_info: Dict,
    acq_label: str,
    mcd_path: str,
    arcsinh_enabled: bool,
    cofactor: float,
) -> pd.DataFrame:
    """Module-level worker that extracts features for a single acquisition.

    Arguments MUST be picklable. Returns an empty DataFrame on error.
    """
    try:
        loader = MCDLoader()
        loader.open(mcd_path)

        # Load all channels (H, W, C)
        img_stack = loader.get_all_channels(acq_id)
        if arcsinh_enabled:
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
        for idx, ch_name in enumerate(channel_names):
            ch_img = img_stack[..., idx]
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

        return morph_df

    except Exception as e:
        # Return empty on error to keep pipeline robust
        return pd.DataFrame()

