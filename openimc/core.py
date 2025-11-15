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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import anndata as ad

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree, Delaunay

from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.feature_worker import _apply_denoise_to_channel, extract_features_for_acquisition
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.processing.batch_correction import (
    apply_combat_correction,
    apply_harmony_correction,
    detect_batch_variable,
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
from openimc.ui.utils import (
    arcsinh_normalize,
    percentile_clip_normalize,
    channelwise_minmax_normalize,
    combine_channels
)


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
    arcsinh_cofactor: float = 10.0,
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
    
    # Get all channels
    channels = loader.get_channels(acquisition.id)
    img_stack = loader.get_all_channels(acquisition.id)
    
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
    arcsinh_cofactor: float = 10.0,
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
    # Load and preprocess nuclear channels
    nuclear_imgs = []
    for channel in nuclear_channels:
        img = loader.get_image(acquisition.id, channel)
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
    
    # Load and preprocess cytoplasm channels
    cyto_img = None
    if cyto_channels:
        cyto_imgs = []
        for channel in cyto_channels:
            img = loader.get_image(acquisition.id, channel)
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
    
    return nuclear_img, cyto_img


def segment(
    loader: Union[MCDLoader, OMETIFFLoader],
    acquisition: AcquisitionInfo,
    method: str,
    nuclear_channels: List[str],
    cyto_channels: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    denoise_settings: Optional[Dict] = None,
    normalization_method: str = "None",
    arcsinh_cofactor: float = 10.0,
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
    
    # Validate channels
    channels = loader.get_channels(acquisition.id)
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
        # Try to import CellSAM
        # Catch both ImportError and OSError (Windows DLL loading errors)
        try:
            from cellSAM import get_model, cellsam_pipeline
        except (ImportError, OSError):
            raise ImportError("CellSAM not installed or failed to load. Install with: pip install git+https://github.com/vanvalenlab/cellSAM.git")
        
        # Set API key from argument or environment variable
        api_key = deepcell_api_key or os.environ.get("DEEPCELL_ACCESS_TOKEN", "")
        if not api_key:
            raise ValueError("DeepCell API key is required for CellSAM. Set deepcell_api_key or DEEPCELL_ACCESS_TOKEN environment variable.")
        os.environ["DEEPCELL_ACCESS_TOKEN"] = api_key
        
        # Initialize CellSAM model and download weights
        try:
            get_model()  # This downloads weights if not already present
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CellSAM model: {e}. Please check your API key and internet connection.")
        
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
        
        # Run CellSAM pipeline
        mask = cellsam_pipeline(
            cellsam_input,
            bbox_threshold=bbox_threshold,
            use_wsi=use_wsi,
            low_contrast_enhancement=low_contrast_enhancement,
            gauge_cell_size=gauge_cell_size
        )
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
        use_gpu = gpu_id is not None
        model = models.Cellpose(model_type=cellpose_model, gpu=use_gpu, device=gpu_id)
        
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
        channels = loader.get_channels(acquisition.id)
        img_stack = loader.get_all_channels(acquisition.id)
        
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
            output_filename = f"{acquisition.well}_segmentation.tif"
        else:
            output_filename = f"{acquisition.name}_segmentation.tif"
        output_path = output_dir / output_filename
        
        tifffile.imwrite(str(output_path), mask.astype(np.uint32), compression='lzw')
        
        # Also save as numpy array for easier loading
        np.save(str(output_path).replace('.tif', '.npy'), mask)
    
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
    arcsinh_cofactor: float = 10.0,
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
        
        mask = masks_dict[acq.id]
        channels = loader.get_channels(acq.id)
        img_stack = loader.get_all_channels(acq.id)
        
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
    
    # Save to CSV if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        combined_features.to_csv(output_path, index=False)
    
    return combined_features


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
    # K-means parameters
    n_init: int = 10,  # Number of initializations for K-means
    # HDBSCAN parameters
    min_cluster_size: int = 10,
    min_samples: int = 5,
    cluster_selection_method: str = "eom",  # HDBSCAN cluster selection method
    hdbscan_metric: str = "euclidean"  # HDBSCAN distance metric
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
        n_init: Number of initializations for K-means (default: 10)
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 10)
        min_samples: Minimum samples for HDBSCAN (default: 5)
        cluster_selection_method: Cluster selection method for HDBSCAN ("eom" or "leaf", default: "eom")
        hdbscan_metric: Distance metric for HDBSCAN (default: "euclidean")
    
    Returns:
        DataFrame with cluster labels added in 'cluster' column
    
    Raises:
        ValueError: If method is invalid or required parameters are missing
    """
    import time
    t_start = time.time()
    print(f"[CORE.CLUSTER] Starting clustering: method={method}, input shape={features_df.shape}")
    
    # Select columns for clustering
    t0 = time.time()
    if columns:
        cluster_columns = columns
    else:
        # Auto-detect: exclude non-feature columns (matching GUI)
        exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'cell_id',
                       'source_file', 'source_well', 'acquisition_label'}
        cluster_columns = [col for col in features_df.columns if col not in exclude_cols]
    print(f"[CORE.CLUSTER] Column selection: {len(cluster_columns)} columns, took {time.time() - t0:.3f}s")
    
    # Validate columns
    missing = [col for col in cluster_columns if col not in features_df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    
    # Prepare data exactly like GUI _prepare_clustering_data
    t0 = time.time()
    data = features_df[cluster_columns].copy()
    print(f"[CORE.CLUSTER] Data copy: shape={data.shape}, took {time.time() - t0:.3f}s")
    
    # Handle missing/infinite values safely (matching GUI)
    t0 = time.time()
    data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
    print(f"[CORE.CLUSTER] Handle missing/infinite: took {time.time() - t0:.3f}s")
    
    # Apply scaling (matching GUI _apply_scaling)
    t0 = time.time()
    if scaling == 'zscore':
        # Z-score normalization: (x - mean) / std
        data_means = data.mean()
        data_stds = data.std(ddof=0)
        
        # Handle columns with zero variance or NaN std/mean
        zero_var_cols = (data_stds == 0) | data_stds.isna() | data_means.isna()
        if zero_var_cols.any():
            # Set zero variance/NaN columns to 0 (centered but not scaled)
            data.loc[:, zero_var_cols] = 0
            non_zero_var_cols = ~zero_var_cols
            if non_zero_var_cols.any():
                normalized_data = (data.loc[:, non_zero_var_cols] - data_means[non_zero_var_cols]) / data_stds[non_zero_var_cols]
                data.loc[:, non_zero_var_cols] = normalized_data
        else:
            # Normalize all columns
            data = (data - data_means) / data_stds
    elif scaling == 'mad':
        # MAD (Median Absolute Deviation) scaling: (x - median) / MAD
        data_medians = data.median()
        
        # Calculate MAD for each column
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
        
        # Handle columns with zero MAD or NaN
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
    # If scaling == 'none', skip scaling
    
    # Handle any infinities that might have been introduced
    data = data.replace([np.inf, -np.inf], np.nan)
    print(f"[CORE.CLUSTER] Scaling complete: took {time.time() - t0:.3f}s")
    
    # Drop any residual non-finite rows/cols (matching GUI)
    t0 = time.time()
    data = data.dropna(axis=0, how='any').dropna(axis=1, how='any')
    print(f"[CORE.CLUSTER] Dropna: shape={data.shape}, took {time.time() - t0:.3f}s")
    
    # Guard: require at least 2 rows and 2 columns
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Insufficient data for clustering. Need at least 2 rows and 2 columns after cleaning.")
    
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
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data_values)
        distances_knn, indices_knn = nbrs.kneighbors(data_values)
        print(f"[CORE.CLUSTER] Leiden: k-NN calculation took {time.time() - t0:.3f}s")
        
        t0 = time.time()
        # Create graph from k-NN (matching old GUI implementation)
        edges = []
        weights = []
        
        for i in range(n):
            for j_idx, neighbor_idx in enumerate(indices_knn[i]):
                if neighbor_idx != i:  # Don't add self-loops
                    edges.append((i, neighbor_idx))
                    # Convert distance to similarity (inverse, normalized) - matching old GUI
                    weight = 1.0 / (1.0 + distances_knn[i][j_idx])
                    weights.append(weight)
        
        print(f"[CORE.CLUSTER] Leiden: Edge list creation took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Leiden: Created {len(edges)} edges from k-NN")
        
        t0 = time.time()
        # Create symmetric graph (undirected - convert to symmetric)
        edge_set = set()
        symmetric_edges = []
        symmetric_weights = []
        for (i, j), w in zip(edges, weights):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                symmetric_edges.append((i, j))
                symmetric_weights.append(w)
        
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
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(data_values)
        distances_knn, indices_knn = nbrs.kneighbors(data_values)
        print(f"[CORE.CLUSTER] Louvain: k-NN calculation took {time.time() - t0:.3f}s")
        
        t0 = time.time()
        # Create graph from k-NN
        edges = []
        weights = []
        
        for i in range(n):
            for j_idx, neighbor_idx in enumerate(indices_knn[i]):
                if neighbor_idx != i:  # Don't add self-loops
                    edges.append((i, neighbor_idx))
                    # Convert distance to similarity (inverse, normalized)
                    weight = 1.0 / (1.0 + distances_knn[i][j_idx])
                    weights.append(weight)
        
        print(f"[CORE.CLUSTER] Louvain: Edge list creation took {time.time() - t0:.3f}s")
        print(f"[CORE.CLUSTER] Louvain: Created {len(edges)} edges from k-NN")
        
        t0 = time.time()
        # Create symmetric graph (undirected - convert to symmetric)
        edge_set = set()
        symmetric_edges = []
        symmetric_weights = []
        for (i, j), w in zip(edges, weights):
            if (i, j) not in edge_set and (j, i) not in edge_set:
                edge_set.add((i, j))
                symmetric_edges.append((i, j))
                symmetric_weights.append(w)
        
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
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
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
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
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
    max_iter: int = 10,
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
        max_iter: Maximum number of iterations for Harmony (default: 10)
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
        # Exclude non-feature columns
        exclude_cols = {
            'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster',
            'cell_id', 'centroid_x', 'centroid_y', 'source_file', 'source_well',
            batch_var
        }
        features = [col for col in features_df.columns if col not in exclude_cols]
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
    
    # Load image stack for all channels
    img_stack = loader.get_all_channels(acquisition.id)
    
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
    
    # Flatten images and apply mask if provided
    pixel_data = {}
    for i, channel in enumerate(channels):
        if i >= n_channels:
            continue
        # Extract channel from HWC format: (H, W, C) -> (H, W)
        channel_img = img_stack[:, :, i] if img_stack.ndim == 3 else img_stack
        
        if mask is not None:
            # Only use pixels within cells
            if mask.shape == channel_img.shape:
                cell_mask = mask > 0
                pixels = channel_img[cell_mask]
            else:
                # Mask dimensions don't match, skip mask
                pixels = channel_img.flatten()
        else:
            # Use all pixels
            pixels = channel_img.flatten()
        
        # Remove NaN and infinite values
        pixels = pixels[~np.isnan(pixels) & ~np.isinf(pixels)]
        pixel_data[channel] = pixels
    
    # Compute pairwise correlations
    correlations = []
    channel_list = list(pixel_data.keys())
    for i, ch1 in enumerate(channel_list):
        for j, ch2 in enumerate(channel_list):
            if i >= j:  # Only compute upper triangle
                continue
            
            data1 = pixel_data[ch1]
            data2 = pixel_data[ch2]
            
            # Ensure same length (take minimum)
            min_len = min(len(data1), len(data2))
            if min_len < 3:  # Need at least 3 points for correlation
                continue
            
            data1 = data1[:min_len]
            data2 = data2[:min_len]
            
            # Compute Spearman correlation
            try:
                corr_coef, p_value = spearmanr(data1, data2)
                
                if not np.isnan(corr_coef) and not np.isinf(corr_coef):
                    correlations.append({
                        'marker1': ch1,
                        'marker2': ch2,
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n_pixels': min_len
                    })
            except Exception:
                continue
    
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
    mask: Optional[np.ndarray] = None
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
    
    Returns:
        DataFrame with QC metrics per channel
    """
    # Optional scikit-image for Otsu thresholding
    try:
        from skimage.filters import threshold_otsu
        _HAVE_SCIKIT_IMAGE = True
    except ImportError:
        _HAVE_SCIKIT_IMAGE = False
    
    def _calculate_snr(signal_mean: float, background_mean: float, background_std: float,
                      img_min: Optional[float] = None, img_max: Optional[float] = None) -> float:
        """Calculate Signal-to-Noise Ratio with robust handling."""
        signal_diff = signal_mean - background_mean
        min_std_relative = abs(background_mean) * 0.001
        min_std_absolute = 1e-6
        min_std_range = 0.0
        if img_min is not None and img_max is not None:
            img_range = img_max - img_min
            if img_range > 0:
                min_std_range = img_range * 0.0001
        min_std = max(background_std, min_std_relative, min_std_absolute, min_std_range)
        snr = signal_diff / min_std
        return snr
    
    results = []
    
    for channel in channels:
        try:
            # Load image
            img = loader.get_image(acquisition.id, channel)
            if img is None:
                continue
            
            img_flat = img.flatten()
            img_min = float(np.min(img_flat))
            img_max = float(np.max(img_flat))
            img_mean = float(np.mean(img_flat))
            img_std = float(np.std(img_flat))
            img_median = float(np.median(img_flat))
            
            if mode == "pixel":
                # Pixel-level QC using Otsu threshold
                if _HAVE_SCIKIT_IMAGE:
                    try:
                        threshold = threshold_otsu(img)
                        signal_mask = img > threshold
                        background_mask = img <= threshold
                        
                        signal_mean = float(np.mean(img[signal_mask])) if np.any(signal_mask) else img_mean
                        background_mean = float(np.mean(img[background_mask])) if np.any(background_mask) else img_mean
                        background_std = float(np.std(img[background_mask])) if np.any(background_mask) else img_std
                        
                        snr = _calculate_snr(signal_mean, background_mean, background_std, img_min, img_max)
                        coverage = float(np.sum(signal_mask) / signal_mask.size) if signal_mask.size > 0 else 0.0
                    except Exception:
                        # Fallback if Otsu fails
                        signal_mean = img_mean
                        background_mean = img_mean
                        background_std = img_std
                        snr = 0.0
                        coverage = 0.0
                else:
                    # No scikit-image, use simple statistics
                    signal_mean = img_mean
                    background_mean = img_mean
                    background_std = img_std
                    snr = 0.0
                    coverage = 0.0
                
                results.append({
                    'acquisition_id': acquisition.id,
                    'acquisition_name': acquisition.name,
                    'channel': channel,
                    'mode': 'pixel',
                    'snr': snr,
                    'signal_mean': signal_mean,
                    'background_mean': background_mean,
                    'background_std': background_std,
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
                
                # Calculate metrics per cell
                cell_ids = np.unique(mask[mask > 0])
                if len(cell_ids) == 0:
                    continue
                
                cell_intensities = []
                for cell_id in cell_ids:
                    cell_mask = mask == cell_id
                    cell_intensity = np.mean(img[cell_mask])
                    cell_intensities.append(cell_intensity)
                
                if len(cell_intensities) == 0:
                    continue
                
                cell_intensities = np.array(cell_intensities)
                signal_mean = float(np.mean(cell_intensities))
                signal_std = float(np.std(cell_intensities))
                
                # Background is pixels outside cells
                background_mask = mask == 0
                if np.any(background_mask):
                    background_mean = float(np.mean(img[background_mask]))
                    background_std = float(np.std(img[background_mask]))
                else:
                    background_mean = img_mean
                    background_std = img_std
                
                snr = _calculate_snr(signal_mean, background_mean, background_std, img_min, img_max)
                
                # Coverage: fraction of pixels covered by cells
                coverage = float(np.sum(mask > 0) / mask.size) if mask.size > 0 else 0.0
                
                # Cell density: cells per unit area (assuming pixels)
                cell_density = float(len(cell_ids) / mask.size) if mask.size > 0 else 0.0
                
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
                    'intensity_mean': img_mean,
                    'intensity_std': img_std,
                    'intensity_median': img_median,
                    'intensity_min': img_min,
                    'intensity_max': img_max,
                    'coverage': coverage,
                    'cell_density': cell_density,
                    'n_cells': len(cell_ids)
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
        if col in S.columns or (channel_map and col in channel_map.values()):
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
    iterations: int = 4,
    output_format: str = "float",
    loader_path: Optional[Union[str, Path]] = None,
    source_file_path: Optional[Union[str, Path]] = None,
    unique_acq_id: Optional[str] = None
) -> Path:
    """Apply Richardson-Lucy deconvolution to high resolution IMC images.
    
    This function applies deconvolution optimized for high resolution IMC
    images with step sizes of 333 nm and 500 nm.
    
    Args:
        loader: Data loader (MCDLoader or OMETIFFLoader)
        acquisition: Acquisition information
        output_dir: Output directory for deconvolved images
        x0: Parameter for kernel calculation (default: 7.0)
        iterations: Number of Richardson-Lucy iterations (default: 4)
        output_format: Output format ('float' or 'uint16', default: 'float')
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
    
    # Call deconvolution worker
    output_path = deconvolve_acquisition(
        loader_path,
        acquisition.id,
        str(output_dir),
        x0=x0,
        iterations=iterations,
        output_format=output_format,
        channel_names=None,  # Will be auto-detected
        source_file_path=source_file_path,
        unique_acq_id=unique_acq_id,
        well_name=getattr(acquisition, 'well', None)
    )
    
    return Path(output_path)


def spatial_enrichment(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cluster_column: str = "cluster",
    n_permutations: int = 100,
    seed: int = 42,
    roi_column: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Compute pairwise spatial enrichment between clusters.
    
    This function computes enrichment of spatial interactions between cluster
    pairs using permutation-based null distribution.
    
    Args:
        features_df: DataFrame with cell features and cluster labels
        edges_df: DataFrame with spatial graph edges (must have 'cell_id_A', 'cell_id_B', 'roi_id')
        cluster_column: Column name containing cluster labels
        n_permutations: Number of permutations for null distribution (default: 100)
        seed: Random seed for reproducibility (default: 42)
        roi_column: Column name for ROI grouping (auto-detected if None)
        output_path: Optional path to save enrichment results CSV
    
    Returns:
        DataFrame with enrichment results (cluster_A, cluster_B, observed, expected, p_value, z_score, etc.)
    """
    import random
    
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
    
    enrichment_results = []
    
    # Process per ROI
    for roi_id, roi_df in features_df.groupby(roi_column):
        roi_edges = edges_df[edges_df['roi_id'] == str(roi_id)]
        
        if roi_edges.empty:
            continue
        
        # Get unique clusters
        unique_clusters = sorted(roi_df[cluster_column].dropna().unique())
        if len(unique_clusters) < 2:
            continue
        
        # Create cell_id to cluster mapping
        cell_to_cluster = dict(zip(roi_df['cell_id'], roi_df[cluster_column]))
        
        # Count observed edges between cluster pairs
        observed_edges = {}
        for _, edge in roi_edges.iterrows():
            cell_a = int(edge['cell_id_A'])
            cell_b = int(edge['cell_id_B'])
            
            cluster_a = cell_to_cluster.get(cell_a)
            cluster_b = cell_to_cluster.get(cell_b)
            
            if cluster_a is not None and cluster_b is not None:
                pair = tuple(sorted([cluster_a, cluster_b]))
                observed_edges[pair] = observed_edges.get(pair, 0) + 1
        
        # Compute enrichment for each cluster pair
        for i, cluster_a in enumerate(unique_clusters):
            for j, cluster_b in enumerate(unique_clusters):
                if j < i:
                    continue
                
                pair = tuple(sorted([cluster_a, cluster_b]))
                observed = observed_edges.get(pair, 0)
                
                # Get cells in each cluster
                cells_a = set(roi_df[roi_df[cluster_column] == cluster_a]['cell_id'])
                cells_b = set(roi_df[roi_df[cluster_column] == cluster_b]['cell_id'])
                
                # Permutation test
                n_a = len(cells_a)
                n_b = len(cells_b)
                total_edges = len(roi_edges)
                
                if total_edges == 0:
                    continue
                
                # Expected number of edges (proportional to cluster sizes)
                expected = (n_a * n_b / (len(roi_df) * (len(roi_df) - 1) / 2)) * total_edges
                
                # Permutation: randomly shuffle cluster labels
                permuted_counts = []
                cluster_labels = roi_df[cluster_column].values.copy()
                
                for _ in range(n_permutations):
                    np.random.shuffle(cluster_labels)
                    perm_cell_to_cluster = dict(zip(roi_df['cell_id'], cluster_labels))
                    
                    perm_observed = 0
                    for _, edge in roi_edges.iterrows():
                        cell_a = int(edge['cell_id_A'])
                        cell_b = int(edge['cell_id_B'])
                        
                        perm_cluster_a = perm_cell_to_cluster.get(cell_a)
                        perm_cluster_b = perm_cell_to_cluster.get(cell_b)
                        
                        if perm_cluster_a == cluster_a and perm_cluster_b == cluster_b:
                            perm_observed += 1
                        elif perm_cluster_a == cluster_b and perm_cluster_b == cluster_a:
                            perm_observed += 1
                    
                    permuted_counts.append(perm_observed)
                
                # Calculate p-value and z-score
                permuted_counts = np.array(permuted_counts)
                p_value = (np.sum(permuted_counts >= observed) + 1) / (n_permutations + 1)
                z_score = (observed - expected) / (np.std(permuted_counts) + 1e-10)
                
                enrichment_results.append({
                    'roi_id': str(roi_id),
                    'cluster_A': cluster_a,
                    'cluster_B': cluster_b,
                    'observed': observed,
                    'expected': expected,
                    'p_value': p_value,
                    'z_score': z_score,
                    'n_permutations': n_permutations
                })
    
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
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Compute distance distributions between clusters.
    
    This function computes the distribution of spatial distances between
    cells of different clusters.
    
    Args:
        features_df: DataFrame with cell features and cluster labels
        edges_df: DataFrame with spatial graph edges (must have 'cell_id_A', 'cell_id_B', 'distance_um', 'roi_id')
        cluster_column: Column name containing cluster labels
        roi_column: Column name for ROI grouping (auto-detected if None)
        output_path: Optional path to save distance distribution results CSV
    
    Returns:
        DataFrame with distance distribution statistics per cluster pair
    """
    # Auto-detect ROI column
    if roi_column is None:
        for col in ['acquisition_id', 'roi_id', 'roi']:
            if col in features_df.columns:
                roi_column = col
                break
    
    if roi_column is None:
        roi_column = 'roi_id'
    
    distance_results = []
    
    # Process per ROI
    for roi_id, roi_df in features_df.groupby(roi_column):
        roi_edges = edges_df[edges_df['roi_id'] == str(roi_id)]
        
        if roi_edges.empty:
            continue
        
        # Get unique clusters
        unique_clusters = sorted(roi_df[cluster_column].dropna().unique())
        if len(unique_clusters) < 2:
            continue
        
        # Create cell_id to cluster mapping
        cell_to_cluster = dict(zip(roi_df['cell_id'], roi_df[cluster_column]))
        
        # Compute distances for each cluster pair
        for i, cluster_a in enumerate(unique_clusters):
            for j, cluster_b in enumerate(unique_clusters):
                if j < i:
                    continue
                
                # Find edges between these clusters
                pair_distances = []
                for _, edge in roi_edges.iterrows():
                    cell_a = int(edge['cell_id_A'])
                    cell_b = int(edge['cell_id_B'])
                    
                    cluster_a_edge = cell_to_cluster.get(cell_a)
                    cluster_b_edge = cell_to_cluster.get(cell_b)
                    
                    if (cluster_a_edge == cluster_a and cluster_b_edge == cluster_b) or \
                       (cluster_a_edge == cluster_b and cluster_b_edge == cluster_a):
                        dist = edge.get('distance_um', edge.get('distance', 0.0))
                        pair_distances.append(dist)
                
                if not pair_distances:
                    continue
                
                pair_distances = np.array(pair_distances)
                
                distance_results.append({
                    'roi_id': str(roi_id),
                    'cluster_A': cluster_a,
                    'cluster_B': cluster_b,
                    'n_edges': len(pair_distances),
                    'mean_distance': float(np.mean(pair_distances)),
                    'median_distance': float(np.median(pair_distances)),
                    'std_distance': float(np.std(pair_distances)),
                    'min_distance': float(np.min(pair_distances)),
                    'max_distance': float(np.max(pair_distances)),
                    'q25_distance': float(np.percentile(pair_distances, 25)),
                    'q75_distance': float(np.percentile(pair_distances, 75))
                })
    
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
    except ImportError:
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
            'aspect_ratio', 'bbox_area_um2', 'touches_border', 'holes_count'
        }
        morpho_cols = [col for col in all_feature_cols if col in morpho_names]
        feature_cols.extend(morpho_cols)
        
        # Create AnnData object
        # X: feature matrix (intensity and morphology features)
        X = df[feature_cols].values if feature_cols else np.zeros((len(df), 0))
        
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
            adata.obs['cluster'] = df[cluster_col].values
        
        return adata
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


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
    except ImportError:
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
    
    for current_roi_id in roi_ids:
        # Convert dataframe to AnnData
        adata = dataframe_to_anndata(
            features_df,
            roi_id=current_roi_id,
            roi_column=roi_column,
            pixel_size_um=pixel_size_um
        )
        
        if adata is None:
            continue
        
        # Ensure cluster columns are categorical (required by squidpy)
        for col in ['cluster', 'cluster_phenotype', 'cluster_id']:
            if col in adata.obs.columns:
                if not hasattr(adata.obs[col], 'cat'):
                    adata.obs[col] = adata.obs[col].astype('category')
        
        # Build spatial graph
        coords = adata.obsm['spatial']
        
        if method == "kNN":
            # Use squidpy for kNN
            sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=k_neighbors, n_rings=1)
        elif method == "Radius":
            # Radius is in micrometers, coordinates are in micrometers
            sq.gr.spatial_neighbors(adata, coord_type="generic", radius=radius, n_rings=1)
        elif method == "Delaunay":
            # Delaunay triangulation - manual implementation
            tri = Delaunay(coords)
            n_cells = len(coords)
            rows, cols = [], []
            for simplex in tri.simplices:
                # Each simplex has 3 vertices, create edges between all pairs
                for i in range(3):
                    for j in range(i + 1, 3):
                        rows.extend([simplex[i], simplex[j]])
                        cols.extend([simplex[j], simplex[i]])
            
            # Create sparse matrix
            data = np.ones(len(rows))
            conn = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
            
            # Store in AnnData format
            adata.obsp['spatial_connectivities'] = conn
            
            # Calculate distances
            distances = []
            for i, j in zip(rows, cols):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
            dist_matrix = sp.csr_matrix((distances, (rows, cols)), shape=(n_cells, n_cells))
            adata.obsp['spatial_distances'] = dist_matrix
        
        # Verify graph was created
        if 'spatial_connectivities' in adata.obsp:
            anndata_dict[str(current_roi_id)] = adata
    
    return anndata_dict


def spatial_neighborhood_enrichment(
    anndata_dict: Dict[str, 'ad.AnnData'],
    cluster_key: str = "cluster",
    aggregation: str = "mean"
) -> Dict[str, Any]:
    """Compute neighborhood enrichment using squidpy.
    
    This function computes neighborhood enrichment for each ROI and optionally aggregates results.
    
    Args:
        anndata_dict: Dictionary mapping ROI ID to AnnData object with spatial graph
        cluster_key: Column name containing cluster labels (default: "cluster")
        aggregation: Aggregation method for multiple ROIs ("mean" or "sum", default: "mean")
    
    Returns:
        Dictionary with:
            - 'results': Dict mapping ROI ID to enrichment results
            - 'aggregated': Aggregated enrichment matrix (if multiple ROIs)
            - 'cluster_categories': List of cluster categories
    """
    try:
        import squidpy as sq
        import anndata as ad
    except ImportError:
        raise ImportError("squidpy and anndata are required for neighborhood enrichment")
    
    results = {}
    enrichment_matrices = []
    roi_cluster_map = {}
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        if cluster_key not in adata.obs.columns:
            continue
        
        # Ensure categorical
        if not hasattr(adata.obs[cluster_key], 'cat'):
            adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
        
        # Run neighborhood enrichment
        sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)
        
        # Extract matrix
        enrichment_data = adata.uns.get('nhood_enrichment', {})
        matrix = None
        
        if isinstance(enrichment_data, dict):
            if 'zscore' in enrichment_data:
                matrix = enrichment_data['zscore']
            elif 'count' in enrichment_data:
                matrix = enrichment_data['count']
            elif 'stat' in enrichment_data:
                matrix = enrichment_data['stat']
            else:
                for value in enrichment_data.values():
                    if isinstance(value, np.ndarray) and value.ndim == 2:
                        matrix = value
                        break
        elif isinstance(enrichment_data, np.ndarray):
            matrix = enrichment_data
        
        if matrix is not None and isinstance(matrix, np.ndarray) and matrix.ndim == 2:
            results[roi_id] = adata
            enrichment_matrices.append((roi_id, matrix))
            
            # Get cluster categories
            if hasattr(adata.obs[cluster_key], 'cat'):
                clusters = list(adata.obs[cluster_key].cat.categories)
            else:
                clusters = sorted(adata.obs[cluster_key].unique())
            roi_cluster_map[roi_id] = clusters
    
    # Aggregate if multiple ROIs
    aggregated_matrix = None
    all_clusters_union = []
    
    if len(enrichment_matrices) > 1:
        # Get union of all clusters
        all_cluster_sets = [set(clusters) for clusters in roi_cluster_map.values()]
        all_clusters_union = sorted(set().union(*all_cluster_sets)) if all_cluster_sets else []
        
        if all_clusters_union:
            # Align all matrices to the union of clusters
            aligned_matrices = []
            n_clusters = len(all_clusters_union)
            
            for roi_id, matrix in enrichment_matrices:
                roi_clusters = roi_cluster_map.get(roi_id)
                
                if roi_clusters is not None:
                    # Create aligned matrix
                    aligned_matrix = np.full((n_clusters, n_clusters), np.nan)
                    
                    # Map old indices to new indices
                    cluster_to_new_idx = {clust: idx for idx, clust in enumerate(all_clusters_union)}
                    
                    # Fill in values where clusters overlap
                    for i, old_clust_i in enumerate(roi_clusters):
                        if old_clust_i in cluster_to_new_idx:
                            new_i = cluster_to_new_idx[old_clust_i]
                            for j, old_clust_j in enumerate(roi_clusters):
                                if old_clust_j in cluster_to_new_idx:
                                    new_j = cluster_to_new_idx[old_clust_j]
                                    aligned_matrix[new_i, new_j] = matrix[i, j]
                    
                    aligned_matrices.append(aligned_matrix)
                else:
                    aligned_matrices.append(matrix)
            
            # Aggregate
            stacked = np.stack(aligned_matrices, axis=0)
            if aggregation == 'mean':
                aggregated_matrix = np.nanmean(stacked, axis=0)
            else:  # sum
                aggregated_matrix = np.nansum(stacked, axis=0)
        else:
            aggregated_matrix = enrichment_matrices[0][1] if enrichment_matrices else None
    elif len(enrichment_matrices) == 1:
        aggregated_matrix = enrichment_matrices[0][1]
        all_clusters_union = roi_cluster_map.get(enrichment_matrices[0][0], [])
    
    return {
        'results': results,
        'aggregated': aggregated_matrix,
        'cluster_categories': all_clusters_union
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
    except ImportError:
        raise ImportError("squidpy is required for co-occurrence analysis")
    
    if len(interval) < 2:
        raise ValueError("Co-occurrence analysis requires at least 2 distances in interval")
    
    results = {}
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        if cluster_key not in adata.obs.columns:
            continue
        
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
    aggregation: str = "mean"
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
    except ImportError:
        raise ImportError("squidpy is required for spatial autocorrelation")
    
    results = {}
    moran_results = []
    all_genes = set()
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        # Run spatial autocorrelation
        if markers is not None:
            available_genes = [g for g in markers if g in adata.var_names]
            if not available_genes:
                continue
            sq.gr.spatial_autocorr(adata, mode="moran", genes=available_genes)
            all_genes.update(available_genes)
        else:
            sq.gr.spatial_autocorr(adata, mode="moran")
            var_names_list = list(adata.var_names) if hasattr(adata.var_names, '__iter__') else []
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
                'moranI': moran_data
            })
    
    # Aggregate results if multiple ROIs
    aggregated_adata = None
    if len(moran_results) > 1:
        common_genes = sorted(all_genes)
        I_values_agg = []
        p_values_agg = []
        
        for gene in common_genes:
            I_vals = []
            p_vals = []
            for result in moran_results:
                moranI = result['moranI']
                
                # Handle DataFrame format
                if isinstance(moranI, pd.DataFrame):
                    if gene in moranI.index:
                        I_val = moranI.loc[gene, 'I'] if 'I' in moranI.columns else None
                        p_val = moranI.loc[gene, 'pval_norm'] if 'pval_norm' in moranI.columns else None
                        if I_val is not None:
                            I_vals.append(float(I_val))
                        if p_val is not None:
                            p_vals.append(float(p_val))
                # Handle dict format
                elif isinstance(moranI, dict):
                    if 'I' in moranI and 'var_names' in moranI:
                        var_names = moranI.get('var_names', [])
                        if gene in var_names:
                            idx = list(var_names).index(gene)
                            I_vals.append(moranI['I'][idx] if isinstance(moranI['I'], (list, np.ndarray)) else moranI['I'])
                            if 'pval_norm' in moranI:
                                p_vals.append(moranI['pval_norm'][idx] if isinstance(moranI['pval_norm'], (list, np.ndarray)) else moranI['pval_norm'])
            
            if I_vals:
                if aggregation == 'mean':
                    I_values_agg.append(np.nanmean(I_vals))
                else:  # sum
                    I_values_agg.append(np.nansum(I_vals))
                if p_vals:
                    p_values_agg.append(np.nanmean(p_vals))
                else:
                    p_values_agg.append(1.0)
        
        # Create aggregated result
        class TempAnnData:
            def __init__(self, I_vals, p_vals, genes):
                self.uns = {
                    'moranI': {
                        'I': np.array(I_vals),
                        'pval_norm': np.array(p_vals) if p_vals else None,
                        'var_names': genes
                    }
                }
        
        aggregated_adata = TempAnnData(I_values_agg, p_values_agg, common_genes)
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
    except ImportError:
        raise ImportError("squidpy is required for Ripley analysis")
    
    if mode not in ['F', 'G', 'L']:
        raise ValueError(f"Invalid Ripley mode: {mode}. Must be 'F', 'G', or 'L'")
    
    results = {}
    
    for roi_id, adata in anndata_dict.items():
        if 'spatial_connectivities' not in adata.obsp:
            continue
        
        if cluster_key not in adata.obs.columns:
            continue
        
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

