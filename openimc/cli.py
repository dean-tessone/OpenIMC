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
Command-line interface for OpenIMC batch processing.

This module provides CLI commands for HPC/batch processing without the GUI.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
import yaml

# Import processing modules
from openimc.data.mcd_loader import MCDLoader
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.processing.export_worker import process_channel_for_export
from openimc.processing.feature_worker import extract_features_for_acquisition, _apply_denoise_to_channel
from openimc.processing.watershed_worker import watershed_segmentation
from openimc.processing.batch_correction import apply_combat_correction, apply_harmony_correction, detect_batch_variable
from openimc.processing.spillover_correction import load_spillover
from openimc.ui.utils import arcsinh_normalize, percentile_clip_normalize, channelwise_minmax_normalize, combine_channels

# Try to import Cellpose (optional)
try:
    from cellpose import models
    _HAVE_CELLPOSE = True
except ImportError:
    _HAVE_CELLPOSE = False

# Try to import CellSAM (optional)
try:
    from cellSAM import get_model, cellsam_pipeline
    _HAVE_CELLSAM = True
except ImportError:
    _HAVE_CELLSAM = False


def load_data(input_path: str, channel_format: str = 'CHW'):
    """Load data from MCD file or OME-TIFF directory.
    
    Args:
        input_path: Path to MCD file or OME-TIFF directory
        channel_format: Format for OME-TIFF files ('CHW' or 'HWC'), default is 'CHW'
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
        raise ValueError(f"Input path must be an MCD file or directory containing OME-TIFF files: {input_path}")


def parse_denoise_settings(denoise_json: Optional[str]) -> Dict:
    """Parse denoise settings from JSON string or file."""
    if not denoise_json:
        return {}
    
    # Check if it's a file path
    if os.path.isfile(denoise_json):
        with open(denoise_json, 'r') as f:
            return json.load(f)
    
    # Try to parse as JSON string
    try:
        return json.loads(denoise_json)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON for denoise settings: {denoise_json}")


def preprocess_command(args):
    """Preprocess images: denoising and export to OME-TIFF.
    
    Note: arcsinh normalization is not applied to exported images.
    Only denoising is applied. Arcsinh transform should be applied on extracted intensity features.
    """
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_data(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        print(f"Found {len(acquisitions)} acquisition(s)")
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            # Get all channels
            channels = loader.get_channels(acq.id)
            img_stack = loader.get_all_channels(acq.id)
            
            print(f"  Image shape: {img_stack.shape}, Channels: {len(channels)}")
            
            # Process each channel
            processed_channels = []
            for i, channel_name in enumerate(channels):
                channel_img = img_stack[..., i] if img_stack.ndim == 3 else img_stack
                
                # Apply denoising if configured
                denoise_source = "custom" if channel_name in denoise_settings else "none"
                channel_denoise = denoise_settings.get(channel_name, {})
                
                # Process channel - only denoising, no arcsinh normalization
                processed = process_channel_for_export(
                    channel_img, channel_name, denoise_source,
                    {channel_name: channel_denoise} if channel_denoise else {},
                    "None",  # No normalization applied to exported images
                    10.0,  # Unused but kept for function signature
                    (1.0, 99.0),
                    None  # viewer_denoise_func not used in CLI
                )
                
                processed_channels.append(processed)
            
            # Stack channels in CHW format (C, H, W) to match GUI export
            processed_stack = np.stack(processed_channels, axis=0)
            
            # Save as OME-TIFF
            # Use well name if available, otherwise use acquisition name
            if acq.well:
                output_filename = f"{acq.well}.ome.tif"
            else:
                output_filename = f"{acq.name}.ome.tif"
            output_path = output_dir / output_filename
            
            # Create OME metadata
            metadata = {
                'Channel': {'Name': channels}
            }
            
            print(f"  Saving to: {output_path}")
            tifffile.imwrite(
                str(output_path),
                processed_stack,
                metadata=metadata,
                ome=True,
                photometric='minisblack'
            )
        
        print(f"\n✓ Preprocessing complete! Output saved to: {output_dir}")
        
    finally:
        loader.close()


def segment_command(args):
    """Segment cells using DeepCell CellSAM, Cellpose, or watershed method."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_data(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    # Helper function to ensure 0-1 range (used by all segmentation methods)
    def ensure_0_1_range(img):
        """Ensure image is normalized to 0-1 range using min-max scaling."""
        img_float = img.astype(np.float32, copy=True)
        vmin = np.min(img_float)
        vmax = np.max(img_float)
        if vmax > vmin:
            return (img_float - vmin) / (vmax - vmin)
        else:
            return np.zeros_like(img_float)
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition (use first if not specified)
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            channels = loader.get_channels(acq.id)
            img_stack = loader.get_all_channels(acq.id)
            
            # Parse channel lists
            nuclear_channels = args.nuclear_channels.split(',') if args.nuclear_channels else []
            nuclear_channels = [ch.strip() for ch in nuclear_channels]
            cyto_channels = args.cytoplasm_channels.split(',') if args.cytoplasm_channels else []
            cyto_channels = [ch.strip() for ch in cyto_channels]
            
            # Parse weights if provided
            nuclear_weights = None
            if args.nuclear_weights:
                try:
                    nuclear_weights = [float(w.strip()) for w in args.nuclear_weights.split(',')]
                except ValueError:
                    raise ValueError(f"Invalid nuclear weights format: {args.nuclear_weights}")
            
            cyto_weights = None
            if args.cyto_weights:
                try:
                    cyto_weights = [float(w.strip()) for w in args.cyto_weights.split(',')]
                except ValueError:
                    raise ValueError(f"Invalid cyto weights format: {args.cyto_weights}")
            
            # Validate channels
            missing_nuclear = [ch for ch in nuclear_channels if ch not in channels]
            missing_cyto = [ch for ch in cyto_channels if ch not in channels]
            if missing_nuclear:
                raise ValueError(f"Nuclear channels not found: {missing_nuclear}")
            if missing_cyto and args.method not in ['watershed', 'cellsam']:
                raise ValueError(f"Cytoplasm channels not found: {missing_cyto}")
            if args.method == 'cellsam' and not nuclear_channels and not cyto_channels:
                raise ValueError("For CellSAM, at least one nuclear or cytoplasm channel must be specified")
            
            # Run segmentation
            if args.method == 'cellsam':
                if not _HAVE_CELLSAM:
                    raise ImportError("CellSAM not installed. Install with: pip install git+https://github.com/vanvalenlab/cellSAM.git")
                
                # Set API key from argument or environment variable
                api_key = args.deepcell_api_key or os.environ.get("DEEPCELL_ACCESS_TOKEN", "")
                if not api_key:
                    raise ValueError("DeepCell API key is required for CellSAM. Set --deepcell-api-key or DEEPCELL_ACCESS_TOKEN environment variable.")
                os.environ["DEEPCELL_ACCESS_TOKEN"] = api_key
                
                # Initialize CellSAM model and download weights
                print("  Initializing DeepCell CellSAM model (downloading weights if needed)...")
                try:
                    get_model()  # This downloads weights if not already present
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize CellSAM model: {e}. Please check your API key and internet connection.")
                
                # Preprocess channels exactly like GUI: load individually, denoise, normalize, then combine
                # Build preprocessing config
                preprocessing_config = {
                    'nuclear_channels': nuclear_channels,
                    'cyto_channels': cyto_channels,
                    'nuclear_combo_method': args.nuclear_fusion_method,
                    'cyto_combo_method': args.cyto_fusion_method,
                    'nuclear_weights': nuclear_weights,
                    'cyto_weights': cyto_weights,
                    'normalization_method': 'arcsinh' if args.arcsinh else 'None',
                    'arcsinh_cofactor': args.arcsinh_cofactor if args.arcsinh else 10.0,
                    'percentile_params': (1.0, 99.0)
                }
                
                # Load and preprocess nuclear channels
                nuclear_imgs = []
                for channel in nuclear_channels:
                    img = loader.get_image(acq.id, channel)
                    # Apply denoising if custom settings provided
                    if denoise_settings and channel in denoise_settings:
                        img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
                    # Apply normalization if configured
                    if preprocessing_config['normalization_method'] == 'channelwise_minmax':
                        img = channelwise_minmax_normalize(img)
                    elif preprocessing_config['normalization_method'] == 'arcsinh':
                        img = arcsinh_normalize(img, cofactor=preprocessing_config['arcsinh_cofactor'])
                    elif preprocessing_config['normalization_method'] == 'percentile_clip':
                        p_low, p_high = preprocessing_config['percentile_params']
                        img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
                    # Ensure 0-1 range after denoising and normalization
                    img = ensure_0_1_range(img)
                    nuclear_imgs.append(img)
                
                # Combine nuclear channels
                nuclear_combo_method = preprocessing_config['nuclear_combo_method']
                nuclear_weights_list = preprocessing_config['nuclear_weights']
                nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights_list)
                # Ensure combined image is in 0-1 range
                nuclear_img = ensure_0_1_range(nuclear_img)
                
                # Load and preprocess cytoplasm channels
                cyto_img = None
                if cyto_channels:
                    cyto_imgs = []
                    for channel in cyto_channels:
                        img = loader.get_image(acq.id, channel)
                        # Apply denoising if custom settings provided
                        if denoise_settings and channel in denoise_settings:
                            img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
                        # Apply normalization if configured
                        if preprocessing_config['normalization_method'] == 'channelwise_minmax':
                            img = channelwise_minmax_normalize(img)
                        elif preprocessing_config['normalization_method'] == 'arcsinh':
                            img = arcsinh_normalize(img, cofactor=preprocessing_config['arcsinh_cofactor'])
                        elif preprocessing_config['normalization_method'] == 'percentile_clip':
                            p_low, p_high = preprocessing_config['percentile_params']
                            img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
                        # Ensure 0-1 range after denoising and normalization
                        img = ensure_0_1_range(img)
                        cyto_imgs.append(img)
                    
                    # Combine cytoplasm channels
                    cyto_combo_method = preprocessing_config['cyto_combo_method']
                    cyto_weights_list = preprocessing_config['cyto_weights']
                    cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights_list)
                    # Ensure combined image is in 0-1 range
                    cyto_img = ensure_0_1_range(cyto_img)
                
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
                print(f"  Running DeepCell CellSAM segmentation...")
                mask = cellsam_pipeline(
                    cellsam_input,
                    bbox_threshold=args.bbox_threshold,
                    use_wsi=args.use_wsi,
                    low_contrast_enhancement=args.low_contrast_enhancement,
                    gauge_cell_size=args.gauge_cell_size
                )
                # Use mask directly without modifications
                if isinstance(mask, np.ndarray):
                    mask = mask.copy()
                
            elif args.method == 'cellpose':
                if not _HAVE_CELLPOSE:
                    raise ImportError("Cellpose not installed. Install with: pip install cellpose")
                
                # Preprocess channels exactly like GUI: load individually, denoise, normalize, then combine
                # Build preprocessing config
                preprocessing_config = {
                    'nuclear_channels': nuclear_channels,
                    'cyto_channels': cyto_channels,
                    'nuclear_combo_method': args.nuclear_fusion_method,
                    'cyto_combo_method': args.cyto_fusion_method,
                    'nuclear_weights': nuclear_weights,
                    'cyto_weights': cyto_weights,
                    'normalization_method': 'arcsinh' if args.arcsinh else 'None',
                    'arcsinh_cofactor': args.arcsinh_cofactor if args.arcsinh else 10.0,
                    'percentile_params': (1.0, 99.0)
                }
                
                # Load and preprocess nuclear channels (exactly like GUI _preprocess_channels_for_segmentation)
                nuclear_imgs = []
                for channel in nuclear_channels:
                    img = loader.get_image(acq.id, channel)
                    # Apply denoising if custom settings provided
                    if denoise_settings and channel in denoise_settings:
                        img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
                    # Apply normalization if configured
                    if preprocessing_config['normalization_method'] == 'channelwise_minmax':
                        img = channelwise_minmax_normalize(img)
                    elif preprocessing_config['normalization_method'] == 'arcsinh':
                        img = arcsinh_normalize(img, cofactor=preprocessing_config['arcsinh_cofactor'])
                    elif preprocessing_config['normalization_method'] == 'percentile_clip':
                        p_low, p_high = preprocessing_config['percentile_params']
                        img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
                    # Ensure 0-1 range after denoising and normalization
                    img = ensure_0_1_range(img)
                    nuclear_imgs.append(img)
                
                # Combine nuclear channels
                nuclear_combo_method = preprocessing_config['nuclear_combo_method']
                nuclear_weights_list = preprocessing_config['nuclear_weights']
                nuclear_img = combine_channels(nuclear_imgs, nuclear_combo_method, nuclear_weights_list)
                # Ensure combined image is in 0-1 range
                nuclear_img = ensure_0_1_range(nuclear_img)
                
                # Load and preprocess cytoplasm channels
                cyto_img = None
                if cyto_channels:
                    cyto_imgs = []
                    for channel in cyto_channels:
                        img = loader.get_image(acq.id, channel)
                        # Apply denoising if custom settings provided
                        if denoise_settings and channel in denoise_settings:
                            img = _apply_denoise_to_channel(img, channel, denoise_settings[channel])
                        # Apply normalization if configured
                        if preprocessing_config['normalization_method'] == 'channelwise_minmax':
                            img = channelwise_minmax_normalize(img)
                        elif preprocessing_config['normalization_method'] == 'arcsinh':
                            img = arcsinh_normalize(img, cofactor=preprocessing_config['arcsinh_cofactor'])
                        elif preprocessing_config['normalization_method'] == 'percentile_clip':
                            p_low, p_high = preprocessing_config['percentile_params']
                            img = percentile_clip_normalize(img, p_low=p_low, p_high=p_high)
                        # Ensure 0-1 range after denoising and normalization
                        img = ensure_0_1_range(img)
                        cyto_imgs.append(img)
                    
                    # Combine cytoplasm channels
                    cyto_combo_method = preprocessing_config['cyto_combo_method']
                    cyto_weights_list = preprocessing_config['cyto_weights']
                    cyto_img = combine_channels(cyto_imgs, cyto_combo_method, cyto_weights_list)
                    # Ensure combined image is in 0-1 range
                    cyto_img = ensure_0_1_range(cyto_img)
                
                # Ensure images are in 0-1 range before passing to Cellpose
                nuclear_img = ensure_0_1_range(nuclear_img)
                if cyto_img is not None:
                    cyto_img = ensure_0_1_range(cyto_img)
                
                # Prepare input images for Cellpose
                if args.model == 'nuclei':
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
                model = models.Cellpose(model_type=args.model, gpu=args.gpu_id is not None, device=args.gpu_id)
                
                print(f"  Running Cellpose segmentation (model: {args.model})...")
                masks, flows, styles, diams = model.eval(
                    images,
                    diameter=args.diameter,
                    flow_threshold=args.flow_threshold,
                    cellprob_threshold=args.cellprob_threshold,
                    channels=channels_cp
                )
                mask = masks[0]
                
            elif args.method == 'watershed':
                print(f"  Running watershed segmentation...")
                mask = watershed_segmentation(
                    img_stack, channels, nuclear_channels, cyto_channels,
                    denoise_settings=denoise_settings if denoise_settings else None,
                    normalization_method="arcsinh" if args.arcsinh else "None",
                    arcsinh_cofactor=args.arcsinh_cofactor if args.arcsinh else 10.0,
                    min_cell_area=args.min_cell_area,
                    max_cell_area=args.max_cell_area,
                    compactness=args.compactness
                )
            else:
                raise ValueError(f"Unknown segmentation method: {args.method}")
            
            # Save mask
            # Use well name if available, otherwise use acquisition name
            if acq.well:
                output_filename = f"{acq.well}_segmentation.tif"
            else:
                output_filename = f"{acq.name}_segmentation.tif"
            output_path = output_dir / output_filename
            
            print(f"  Saving segmentation mask to: {output_path}")
            tifffile.imwrite(str(output_path), mask.astype(np.uint32), compression='lzw')
            
            # Also save as numpy array for easier loading
            np.save(str(output_path).replace('.tif', '.npy'), mask)
            
            print(f"  ✓ Segmentation complete: {np.max(mask)} cells detected")
        
        print(f"\n✓ Segmentation complete! Output saved to: {output_dir}")
        
    finally:
        loader.close()


def extract_features_command(args):
    """Extract features from segmented cells."""
    print(f"Loading data from: {args.input}")
    loader, loader_type = load_data(args.input, channel_format=getattr(args, 'channel_format', 'CHW'))
    
    try:
        acquisitions = loader.list_acquisitions()
        
        # Get acquisition
        if args.acquisition:
            acq = next((a for a in acquisitions if a.id == args.acquisition or a.name == args.acquisition), None)
            if not acq:
                raise ValueError(f"Acquisition '{args.acquisition}' not found")
            acquisitions = [acq]
        
        # Load segmentation mask(s) - can be a directory or single file
        mask_path = Path(args.mask)
        masks_dict = {}
        
        if mask_path.is_dir():
            # Directory of masks - load masks for each acquisition
            print(f"Loading masks from directory: {mask_path}")
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
                        print(f"  Loaded mask for {acq.well} (well name): {mask_file.name}")
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
                            print(f"  Loaded mask for {acq.name} (acquisition name): {mask_file.name}")
                            break
        else:
            # Single mask file - use for all acquisitions
            print(f"Loading mask from: {mask_path}")
            if mask_path.suffix == '.npy':
                mask = np.load(str(mask_path))
            else:
                mask = tifffile.imread(str(mask_path))
            # Use same mask for all acquisitions
            for acq in acquisitions:
                masks_dict[acq.id] = mask
        
        # Parse denoise settings
        denoise_settings = parse_denoise_settings(args.denoise_settings) if args.denoise_settings else {}
        
        # Build feature selection dict
        selected_features = {}
        # If neither specified, use defaults (both True)
        if not args.morphological and not args.intensity:
            args.morphological = True
            args.intensity = True
        
        if args.morphological:
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
        if args.intensity:
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
        
        all_features = []
        
        for acq in acquisitions:
            print(f"\nProcessing acquisition: {acq.name} (ID: {acq.id})")
            
            # Get mask for this acquisition
            if acq.id not in masks_dict:
                print(f"  Warning: No mask found for acquisition {acq.name}, skipping")
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
                args.arcsinh,
                args.arcsinh_cofactor if args.arcsinh else 10.0,
                "custom" if denoise_settings else "None",
                denoise_settings,
                None,  # spillover_config
                acq.source_file,
                None  # excluded_channels (CLI doesn't support channel exclusion yet)
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
        else:
            combined_features = all_features[0]
        
        # Save to CSV
        output_path = Path(args.output)
        print(f"\nSaving features to: {output_path}")
        combined_features.to_csv(output_path, index=False)
        
        print(f"✓ Feature extraction complete! Extracted {len(combined_features)} cells")
        
    finally:
        loader.close()


def cluster_command(args):
    """Perform clustering on feature data."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Select columns for clustering
    if args.columns:
        cluster_columns = [col.strip() for col in args.columns.split(',')]
    else:
        # Auto-detect: exclude non-feature columns (matching GUI)
        exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'cell_id'}
        cluster_columns = [col for col in features_df.columns if col not in exclude_cols]
    
    # Validate columns
    missing = [col for col in cluster_columns if col not in features_df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    
    # Prepare data exactly like GUI _prepare_clustering_data
    data = features_df[cluster_columns].copy()
    
    # Handle missing/infinite values safely (matching GUI)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median(numeric_only=True))
    
    # Apply scaling (matching GUI _apply_scaling)
    if args.scaling == 'zscore':
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
    elif args.scaling == 'mad':
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
    
    # Handle any infinities that might have been introduced
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Drop any residual non-finite rows/cols (matching GUI)
    data = data.dropna(axis=0, how='any').dropna(axis=1, how='any')
    
    # Guard: require at least 2 rows and 2 columns
    if data.shape[0] < 2 or data.shape[1] < 2:
        raise ValueError("Insufficient data for clustering. Need at least 2 rows and 2 columns after cleaning.")
    
    # Store original indices to map back
    original_indices = data.index
    data_values = data.values
    
    # Perform clustering
    print(f"Running {args.method} clustering...")
    
    if args.method == 'hierarchical':
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        
        # Calculate distance matrix and linkage (matching GUI)
        distances = pdist(data_values, metric='euclidean')
        linkage_matrix = linkage(distances, method=args.linkage)
        
        # Get cluster labels
        if args.n_clusters is None:
            raise ValueError("--n-clusters is required for hierarchical clustering")
        cluster_labels = fcluster(linkage_matrix, args.n_clusters, criterion='maxclust')
        
    elif args.method == 'leiden':
        import igraph as ig
        from scipy.spatial.distance import pdist
        import leidenalg
        
        # Calculate distance matrix (matching GUI exactly)
        distances = pdist(data_values, metric='euclidean')
        
        # Convert to similarity matrix (invert distances) - matching GUI
        max_dist = np.max(distances)
        similarities = max_dist - distances
        
        # Create graph from similarity matrix (matching GUI exactly)
        n = data_values.shape[0]
        edges = []
        weights = []
        
        # Convert condensed distance matrix to edge list
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[idx] > 0:  # Only add positive similarities
                    edges.append((i, j))
                    weights.append(similarities[idx])
                idx += 1
        
        # Create igraph
        g = ig.Graph(n)
        g.add_edges(edges)
        g.es['weight'] = weights
        
        # Run Leiden clustering (matching GUI)
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=args.resolution,
            seed=args.seed,
        )
        cluster_labels = np.array(partition.membership) + 1  # Start from 1 (matching GUI)
        
    elif args.method == 'hdbscan':
        import hdbscan
        # Set random seed for reproducibility
        np.random.seed(args.seed)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples
        )
        cluster_labels = clusterer.fit_predict(data_values)
        # HDBSCAN uses -1 for noise, convert to 1-based (matching GUI)
        cluster_labels = cluster_labels + 1  # -1 becomes 0, others become 1-based
    
    else:
        raise ValueError(f"Unknown clustering method: {args.method}")
    
    # Map cluster labels back to original dataframe indices
    # Create a series with cluster labels for the cleaned data
    cluster_series = pd.Series(cluster_labels, index=original_indices)
    
    # Add cluster labels to original dataframe (NaN for rows that were dropped)
    features_df['cluster'] = cluster_series
    # Fill NaN with 0 (noise/unassigned) if needed
    features_df['cluster'] = features_df['cluster'].fillna(0).astype(int)
    
    # Save output
    output_path = Path(args.output)
    print(f"Saving clustered features to: {output_path}")
    features_df.to_csv(output_path, index=False)
    
    print(f"✓ Clustering complete! Found {len(set(cluster_labels))} clusters")
    if -1 in cluster_labels:
        n_noise = sum(cluster_labels == -1)
        print(f"  ({n_noise} cells marked as noise)")


def spatial_command(args):
    """Perform spatial analysis on feature data (matching GUI workflow)."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    # Check for required columns
    required_cols = ['centroid_x', 'centroid_y']
    missing = [col for col in required_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Required columns for spatial analysis: {missing}")
    
    # Get pixel size (default to 1.0 µm if not available)
    pixel_size_um = getattr(args, 'pixel_size_um', 1.0)
    
    # Build spatial graph per ROI if acquisition_id is present (matching GUI)
    if 'acquisition_id' in features_df.columns:
        print("Building spatial graph per ROI (acquisition)...")
        edge_records = []
        
        for roi_id, roi_df in features_df.groupby('acquisition_id'):
            roi_df = roi_df.dropna(subset=["centroid_x", "centroid_y"])
            if roi_df.empty:
                continue
            
            coords_px = roi_df[["centroid_x", "centroid_y"]].values
            cell_ids = roi_df["cell_id"].values if 'cell_id' in roi_df.columns else roi_df.index.values
            
            # Use cKDTree for efficient spatial queries (matching GUI)
            from scipy.spatial import cKDTree
            tree = cKDTree(coords_px)
            
            # Convert radius from pixels to micrometers if needed
            # CLI uses pixels, but we'll store in micrometers to match GUI
            radius_px = args.radius
            
            # Build edges using kNN within radius (matching GUI)
            roi_edges_set = set()
            query_k = min(args.k_neighbors + 1, max(2, len(coords_px)))
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
                for j in range(1, min(dists.shape[1], args.k_neighbors + 1)):
                    nbr_idx = int(idxs[i, j])
                    if nbr_idx < 0 or nbr_idx >= len(coords_px):
                        continue
                    dst_cell_id = int(cell_ids[nbr_idx])
                    dist_px = float(dists[i, j])
                    dist_um = dist_px * pixel_size_um
                    
                    # Only include edges within radius
                    if dist_px <= radius_px:
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
        
        # Create edges dataframe (matching GUI format)
        edges_df = pd.DataFrame(edge_records)
    else:
        # Single ROI or no ROI grouping - build graph globally
        print("Building spatial graph (single ROI)...")
        coords = features_df[['centroid_x', 'centroid_y']].dropna().values
        
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        
        # Build edges using kNN within radius
        edge_records = []
        edge_set = set()
        query_k = min(args.k_neighbors + 1, max(2, len(coords)))
        dists, idxs = tree.query(coords, k=query_k)
        
        # Handle scalar case
        if np.isscalar(dists):
            dists = np.array([[dists]])
            idxs = np.array([[idxs]])
        elif dists.ndim == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        
        for i in range(len(coords)):
            for j in range(1, min(dists.shape[1], args.k_neighbors + 1)):
                nbr_idx = int(idxs[i, j])
                if nbr_idx < 0 or nbr_idx >= len(coords):
                    continue
                dist_px = float(dists[i, j])
                dist_um = dist_px * pixel_size_um
                
                if dist_px <= args.radius:
                    edge_key = (min(i, nbr_idx), max(i, nbr_idx))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edge_records.append({
                            'source': i,
                            'target': nbr_idx,
                            'distance': dist_px,  # Keep in pixels for compatibility
                            'distance_um': dist_um
                        })
        
        edges_df = pd.DataFrame(edge_records)
    
    # Save edges
    output_path = Path(args.output)
    print(f"Saving spatial graph edges to: {output_path}")
    edges_df.to_csv(output_path, index=False)
    
    print(f"✓ Spatial analysis complete! Found {len(edges_df)} edges")
    
    # Optionally detect communities
    if args.detect_communities:
        print("Detecting spatial communities...")
        import igraph as ig
        
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
        import leidenalg
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=args.seed)
        communities = partition.membership
        
        # Map community labels back to dataframe
        if 'cell_id_A' in edges_df.columns and 'cell_id' in features_df.columns:
            # Map from graph vertex index to cell_id, then to dataframe index
            idx_to_cell_id = {idx: cell_id for idx, cell_id in enumerate(features_df['cell_id'].values)}
            community_series = pd.Series(index=features_df.index, dtype=int)
            for vertex_idx, community in enumerate(communities):
                if vertex_idx < len(features_df):
                    community_series.iloc[vertex_idx] = community
            features_df['spatial_community'] = community_series
        else:
            # Direct mapping (vertex index = dataframe index)
            features_df['spatial_community'] = communities[:len(features_df)]
        
        # Save with communities
        community_output = output_path.parent / (output_path.stem + '_with_communities.csv')
        features_df.to_csv(community_output, index=False)
        print(f"  Saved communities to: {community_output}")


def cluster_figures_command(args):
    """Generate cluster visualization figures."""
    print(f"Loading clustered features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    if 'cluster' not in features_df.columns:
        raise ValueError("Features file must contain 'cluster' column. Run clustering first.")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = args.font_size
    plt.rcParams['figure.dpi'] = args.dpi
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure: UMAP embedding with clusters
    print("Generating UMAP embedding...")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=args.seed)
        
        # Select feature columns
        exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'centroid_x', 'centroid_y'}
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        embedding = reducer.fit_transform(features_df[feature_cols].values)
        
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=features_df['cluster'], cmap='tab20', s=10, alpha=0.6)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Cluster Visualization (UMAP)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        
        output_path = output_dir / 'cluster_umap.png'
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except ImportError:
        print("  Warning: UMAP not available, skipping UMAP plot")
    
    # Create heatmap of cluster means
    print("Generating cluster heatmap...")
    exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'centroid_x', 'centroid_y'}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    cluster_means = features_df.groupby('cluster')[feature_cols].mean()
    
    fig, ax = plt.subplots(figsize=(max(args.width, 12), max(args.height, 8)))
    sns.heatmap(cluster_means.T, annot=False, cmap='viridis', ax=ax, cbar_kws={'label': 'Mean value'})
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Feature')
    ax.set_title('Cluster Mean Feature Values')
    plt.tight_layout()
    
    output_path = output_dir / 'cluster_heatmap.png'
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    print(f"\n✓ Cluster figures saved to: {output_dir}")


def spatial_figures_command(args):
    """Generate spatial analysis visualization figures."""
    print(f"Loading features from: {args.features}")
    features_df = pd.read_csv(args.features)
    
    required_cols = ['centroid_x', 'centroid_y']
    missing = [col for col in required_cols if col not in features_df.columns]
    if missing:
        raise ValueError(f"Required columns for spatial figures: {missing}")
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Set style
    plt.rcParams['font.size'] = args.font_size
    plt.rcParams['figure.dpi'] = args.dpi
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create spatial scatter plot
    print("Generating spatial scatter plot...")
    fig, ax = plt.subplots(figsize=(args.width, args.height))
    
    if 'cluster' in features_df.columns:
        scatter = ax.scatter(
            features_df['centroid_x'], features_df['centroid_y'],
            c=features_df['cluster'], cmap='tab20', s=10, alpha=0.6
        )
        ax.set_title('Spatial Distribution by Cluster')
        plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(features_df['centroid_x'], features_df['centroid_y'], s=10, alpha=0.6)
        ax.set_title('Spatial Distribution of Cells')
    
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('Y coordinate (pixels)')
    ax.set_aspect('equal')
    plt.tight_layout()
    
    output_path = output_dir / 'spatial_distribution.png'
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # Create spatial graph visualization if edges file provided
    if args.edges:
        print("Generating spatial graph visualization...")
        edges_df = pd.read_csv(args.edges)
        
        fig, ax = plt.subplots(figsize=(args.width, args.height))
        
        # Plot cells
        if 'cluster' in features_df.columns:
            scatter = ax.scatter(
                features_df['centroid_x'], features_df['centroid_y'],
                c=features_df['cluster'], cmap='tab20', s=20, alpha=0.8, zorder=2
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(features_df['centroid_x'], features_df['centroid_y'], s=20, alpha=0.8, zorder=2)
        
        # Plot edges
        for _, edge in edges_df.iterrows():
            source_idx = int(edge['source'])
            target_idx = int(edge['target'])
            ax.plot(
                [features_df.loc[source_idx, 'centroid_x'], features_df.loc[target_idx, 'centroid_x']],
                [features_df.loc[source_idx, 'centroid_y'], features_df.loc[target_idx, 'centroid_y']],
                'k-', alpha=0.1, linewidth=0.5, zorder=1
            )
        
        ax.set_xlabel('X coordinate (pixels)')
        ax.set_ylabel('Y coordinate (pixels)')
        ax.set_title('Spatial Graph')
        ax.set_aspect('equal')
        plt.tight_layout()
        
        output_path = output_dir / 'spatial_graph.png'
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    
    print(f"\n✓ Spatial figures saved to: {output_dir}")


def workflow_command(args):
    """Execute a complete workflow from a YAML configuration file.
    
    Supports all OpenIMC functions:
    - preprocessing: Denoising and export to OME-TIFF
    - deconvolution: High resolution deconvolution
    - segmentation: Cell segmentation (CellSAM, Cellpose, Watershed, Ilastik)
    - feature_extraction: Extract features from segmented cells
    - batch_correction: Batch correction (Harmony, ComBat)
    - pixel_correlation: Pixel-level correlation analysis
    - qc_analysis: Quality control analysis
    - clustering: Cell clustering
    - spatial_analysis: Spatial analysis
    
    Each step can specify:
    - enabled: true/false
    - input: path to input file/directory (optional, uses previous step output if not specified)
    - output: path to output file/directory (optional, uses default location if not specified)
    """
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading workflow configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get base paths (optional - steps can specify their own)
    input_path = Path(config.get('input', '.')) if config.get('input') else None
    output_base = Path(config.get('output', 'workflow_output'))
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Track intermediate outputs for chaining steps
    workflow_state = {
        'input': input_path,
        'output_base': output_base,
        'preprocessing_output': None,
        'deconvolution_output': None,
        'segmentation_output': None,
        'features_output': None,
        'batch_corrected_output': None,
    }
    
    # Helper function to get input path for a step
    def get_step_input(step_config, default_input, step_name):
        """Get input path for a step, using config or default."""
        if 'input' in step_config:
            return Path(step_config['input'])
        elif default_input:
            return default_input
        else:
            raise ValueError(f"{step_name} requires 'input' path in config or previous step output")
    
    # Helper function to get output path for a step
    def get_step_output(step_config, default_output, step_name):
        """Get output path for a step, using config or default."""
        if 'output' in step_config:
            return Path(step_config['output'])
        else:
            return default_output
    
    # Step: Preprocessing (should come first)
    if 'preprocessing' in config and config['preprocessing'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: PREPROCESSING")
        print("="*60)
        
        prep_config = config['preprocessing']
        prep_input = get_step_input(prep_config, workflow_state['input'], 'Preprocessing')
        prep_output = get_step_output(prep_config, output_base / 'preprocessed', 'Preprocessing')
        prep_output.mkdir(parents=True, exist_ok=True)
        
        class PreprocessArgs:
            pass
        prep_args = PreprocessArgs()
        prep_args.input = str(prep_input)
        prep_args.output = str(prep_output)
        prep_args.channel_format = prep_config.get('channel_format', config.get('channel_format', 'CHW'))
        prep_args.denoise_settings = prep_config.get('denoise_settings')
        prep_args.arcsinh = prep_config.get('arcsinh', False)
        prep_args.arcsinh_cofactor = prep_config.get('arcsinh_cofactor', 10.0)
        
        preprocess_command(prep_args)
        workflow_state['preprocessing_output'] = prep_output
        workflow_state['input'] = prep_output  # Update input for next steps
    
    # Step: Deconvolution
    if 'deconvolution' in config and config['deconvolution'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: DECONVOLUTION")
        print("="*60)
        
        deconv_config = config['deconvolution']
        deconv_input = get_step_input(deconv_config, workflow_state['input'], 'Deconvolution')
        deconv_output = get_step_output(deconv_config, output_base / 'deconvolved', 'Deconvolution')
        deconv_output.mkdir(parents=True, exist_ok=True)
        
        from openimc.processing.deconvolution_worker import deconvolve_acquisition
        
        # Determine loader type
        loader_type = 'mcd' if str(deconv_input).endswith(('.mcd', '.mcdx')) else 'ometiff'
        
        # Get acquisitions
        loader, _ = load_data(str(deconv_input), channel_format=config.get('channel_format', 'CHW'))
        try:
            acquisitions = loader.list_acquisitions()
            if deconv_config.get('acquisition'):
                acq = next((a for a in acquisitions if a.id == deconv_config['acquisition'] or a.name == deconv_config['acquisition']), None)
                if not acq:
                    raise ValueError(f"Acquisition '{deconv_config['acquisition']}' not found")
                acquisitions = [acq]
            
            for acq in acquisitions:
                print(f"  Deconvolving acquisition: {acq.name} (ID: {acq.id})")
                channels = loader.get_channels(acq.id)
                output_path = deconvolve_acquisition(
                    data_path=str(deconv_input),
                    acq_id=acq.id,
                    output_dir=str(deconv_output),
                    x0=deconv_config.get('x0', 7.0),
                    iterations=deconv_config.get('iterations', 4),
                    output_format=deconv_config.get('output_format', 'float'),
                    channel_names=channels,
                    source_file_path=acq.source_file,
                    unique_acq_id=acq.id,
                    loader_type=loader_type,
                    channel_format=config.get('channel_format', 'CHW'),
                    well_name=acq.well
                )
                print(f"  ✓ Saved: {output_path}")
        finally:
            loader.close()
        
        workflow_state['deconvolution_output'] = deconv_output
        workflow_state['input'] = deconv_output  # Update input for next steps
    
    # Step: Segmentation (if configured)
    if 'segmentation' in config and config['segmentation'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: SEGMENTATION")
        print("="*60)
        
        seg_config = config['segmentation']
        seg_input = get_step_input(seg_config, workflow_state['input'], 'Segmentation')
        seg_output = get_step_output(seg_config, output_base / 'segmentation', 'Segmentation')
        seg_output.mkdir(parents=True, exist_ok=True)
        
        # Create a mock args object for segment_command
        class SegmentArgs:
            pass
        
        seg_args = SegmentArgs()
        seg_args.input = str(seg_input)
        seg_args.output = str(seg_output)
        seg_args.channel_format = seg_config.get('channel_format', config.get('channel_format', 'CHW'))
        seg_args.acquisition = seg_config.get('acquisition')
        seg_args.method = seg_config.get('method', 'cellsam')
        seg_args.nuclear_channels = ','.join(seg_config.get('nuclear_channels', []))
        seg_args.cytoplasm_channels = ','.join(seg_config.get('cytoplasm_channels', [])) if seg_config.get('cytoplasm_channels') else None
        seg_args.nuclear_fusion_method = seg_config.get('nuclear_fusion_method', 'mean')
        seg_args.cyto_fusion_method = seg_config.get('cyto_fusion_method', 'mean')
        seg_args.nuclear_weights = ','.join(map(str, seg_config.get('nuclear_weights', []))) if seg_config.get('nuclear_weights') else None
        seg_args.cyto_weights = ','.join(map(str, seg_config.get('cyto_weights', []))) if seg_config.get('cyto_weights') else None
        seg_args.model = seg_config.get('model', 'cyto3')
        seg_args.diameter = seg_config.get('diameter')
        seg_args.flow_threshold = seg_config.get('flow_threshold', 0.4)
        seg_args.cellprob_threshold = seg_config.get('cellprob_threshold', 0.0)
        seg_args.gpu_id = seg_config.get('gpu_id')
        seg_args.min_cell_area = seg_config.get('min_cell_area', 100)
        seg_args.max_cell_area = seg_config.get('max_cell_area', 10000)
        seg_args.compactness = seg_config.get('compactness', 0.01)
        seg_args.deepcell_api_key = seg_config.get('deepcell_api_key') or os.environ.get("DEEPCELL_ACCESS_TOKEN")
        seg_args.bbox_threshold = seg_config.get('bbox_threshold', 0.4)
        seg_args.use_wsi = seg_config.get('use_wsi', False)
        seg_args.low_contrast_enhancement = seg_config.get('low_contrast_enhancement', False)
        seg_args.gauge_cell_size = seg_config.get('gauge_cell_size', False)
        seg_args.arcsinh = seg_config.get('arcsinh', False)
        seg_args.arcsinh_cofactor = seg_config.get('arcsinh_cofactor', 10.0)
        
        # Handle denoise settings
        denoise_settings = seg_config.get('denoise_settings')
        if denoise_settings:
            if isinstance(denoise_settings, str):
                seg_args.denoise_settings = denoise_settings
            else:
                # Save to temporary JSON file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(denoise_settings, f)
                    seg_args.denoise_settings = f.name
        else:
            seg_args.denoise_settings = None
        
        segment_command(seg_args)
        workflow_state['segmentation_output'] = seg_output
    
    # Step: Feature Extraction (if configured)
    if 'feature_extraction' in config and config['feature_extraction'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: FEATURE EXTRACTION")
        print("="*60)
        
        feat_config = config['feature_extraction']
        feat_input = get_step_input(feat_config, workflow_state['input'], 'Feature Extraction')
        
        # Determine mask path
        if 'mask' in feat_config:
            mask_path = Path(feat_config['mask'])
        elif workflow_state['segmentation_output']:
            mask_path = workflow_state['segmentation_output']
        else:
            raise ValueError("Feature extraction requires either 'mask' path in config or segmentation to be run first")
        
        features_output = get_step_output(feat_config, output_base / 'features.csv', 'Feature Extraction')
        
        # Create a mock args object for extract_features_command
        class ExtractArgs:
            pass
        
        extract_args = ExtractArgs()
        extract_args.input = str(feat_input)
        extract_args.output = str(features_output)
        extract_args.channel_format = feat_config.get('channel_format', config.get('channel_format', 'CHW'))
        extract_args.mask = str(mask_path)
        extract_args.acquisition = feat_config.get('acquisition')
        extract_args.morphological = feat_config.get('morphological', True)
        extract_args.intensity = feat_config.get('intensity', True)
        extract_args.arcsinh = feat_config.get('arcsinh', False)
        extract_args.arcsinh_cofactor = feat_config.get('arcsinh_cofactor', 10.0)
        
        # Handle denoise settings
        denoise_settings = feat_config.get('denoise_settings')
        if denoise_settings:
            if isinstance(denoise_settings, str):
                extract_args.denoise_settings = denoise_settings
            else:
                # Save to temporary JSON file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(denoise_settings, f)
                    extract_args.denoise_settings = f.name
        else:
            extract_args.denoise_settings = None
        
        # Handle spillover correction
        spillover_config = feat_config.get('spillover_correction')
        if spillover_config and spillover_config.get('enabled', False):
            # We need to modify extract_features_for_acquisition to accept spillover config
            # For now, we'll extract features first, then apply spillover correction
            print("  Note: Spillover correction will be applied after feature extraction")
            extract_args._spillover_config = spillover_config
        else:
            extract_args._spillover_config = None
        
        extract_features_command(extract_args)
        
        # Apply spillover correction if configured
        if extract_args._spillover_config:
            print("\n  Applying spillover correction...")
            features_df = pd.read_csv(features_output)
            
            spillover_file = extract_args._spillover_config.get('matrix_file')
            if not spillover_file:
                raise ValueError("spillover_correction.matrix_file must be specified")
            
            spillover_matrix = load_spillover(spillover_file)
            spillover_method = extract_args._spillover_config.get('method', 'nnls')
            
            # Get channel names from feature columns (intensity features end with _mean, _median, etc.)
            # We need to identify which columns are intensity features
            intensity_feature_types = ['mean', 'median', 'std', 'mad', 'p10', 'p90', 'integrated']
            channel_names = set()
            for col in features_df.columns:
                for feat_type in intensity_feature_types:
                    if col.endswith(f'_{feat_type}'):
                        channel_name = col[:-len(f'_{feat_type}')]
                        channel_names.add(channel_name)
                        break
            
            if not channel_names:
                print("  Warning: No intensity features found for spillover correction")
            else:
                # Apply spillover correction to each intensity feature type separately
                # This matches the approach in feature_worker.py
                from openimc.processing.spillover_correction import compensate_counts
                
                for feature_type in intensity_feature_types:
                    # Extract columns for this feature type across all channels
                    feature_cols = [f"{ch_name}_{feature_type}" for ch_name in channel_names 
                                   if f"{ch_name}_{feature_type}" in features_df.columns]
                    
                    if not feature_cols:
                        continue
                    
                    # Create a temporary DataFrame with cells x channels for this feature type
                    feature_data = features_df[feature_cols].copy()
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
                    
                    # Rename columns back and update features_df
                    comp_data.rename(columns={ch: f"{ch}_{feature_type}" for ch in comp_data.columns}, inplace=True)
                    for col in comp_data.columns:
                        if col in features_df.columns:
                            features_df[col] = comp_data[col].values
                
                # Save updated features
                features_df.to_csv(features_output, index=False)
                print(f"  ✓ Spillover correction applied to all intensity features and saved to {features_output}")
        
        workflow_state['features_output'] = features_output
    
    # Step: Batch Correction (if configured)
    if 'batch_correction' in config and config['batch_correction'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: BATCH CORRECTION")
        print("="*60)
        
        batch_config = config['batch_correction']
        
        # Determine input features path
        if 'input_features' in batch_config:
            features_path = Path(batch_config['input_features'])
        elif workflow_state['features_output']:
            features_path = workflow_state['features_output']
        else:
            raise ValueError("Batch correction requires either 'input_features' path in config or feature extraction to be run first")
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        batch_output = get_step_output(batch_config, output_base / 'features_batch_corrected.csv', 'Batch Correction')
        
        # Load features
        print(f"Loading features from: {features_path}")
        features_df = pd.read_csv(features_path)
        
        # Determine batch variable
        batch_var = batch_config.get('batch_variable')
        if not batch_var:
            batch_var = detect_batch_variable(features_df)
            if not batch_var:
                raise ValueError("Could not detect batch variable. Please specify 'batch_variable' in config.")
            print(f"  Auto-detected batch variable: {batch_var}")
        else:
            if batch_var not in features_df.columns:
                raise ValueError(f"Batch variable '{batch_var}' not found in features")
        
        # Determine features to correct
        feature_columns = batch_config.get('features')
        if not feature_columns:
            # Auto-detect: exclude non-feature columns
            exclude_cols = {'label', 'acquisition_id', 'acquisition_name', 'well', 'cluster', 'cell_id', 
                          'centroid_x', 'centroid_y', 'source_file', 'source_well', batch_var}
            feature_columns = [col for col in features_df.columns if col not in exclude_cols]
            print(f"  Auto-detected {len(feature_columns)} features for correction")
        else:
            # Validate specified features
            missing = [f for f in feature_columns if f not in features_df.columns]
            if missing:
                raise ValueError(f"Features not found: {missing}")
        
        # Apply batch correction
        method = batch_config.get('method', 'harmony')
        print(f"  Applying {method} batch correction...")
        
        if method == 'combat':
            covariates = batch_config.get('covariates')
            corrected_df = apply_combat_correction(
                features_df,
                batch_var,
                feature_columns,
                covariates=covariates
            )
        elif method == 'harmony':
            corrected_df = apply_harmony_correction(
                features_df,
                batch_var,
                feature_columns,
                n_clusters=batch_config.get('n_clusters', 30),
                sigma=batch_config.get('sigma', 0.1),
                theta=batch_config.get('theta', 2.0),
                lambda_reg=batch_config.get('lambda_reg', 1.0),
                max_iter=batch_config.get('max_iter', 10),
                pca_variance=batch_config.get('pca_variance', 0.9)
            )
        else:
            raise ValueError(f"Unknown batch correction method: {method}")
        
        # Save corrected features
        print(f"  Saving corrected features to: {batch_output}")
        corrected_df.to_csv(batch_output, index=False)
        print(f"  ✓ Batch correction complete! Output saved to: {batch_output}")
        workflow_state['batch_corrected_output'] = batch_output
    
    # Step: Pixel Correlation
    if 'pixel_correlation' in config and config['pixel_correlation'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: PIXEL CORRELATION")
        print("="*60)
        
        corr_config = config['pixel_correlation']
        corr_input = get_step_input(corr_config, workflow_state['input'], 'Pixel Correlation')
        corr_output = get_step_output(corr_config, output_base / 'pixel_correlation.csv', 'Pixel Correlation')
        
        # This would require implementing a CLI version of pixel correlation
        # For now, we'll note that this needs to be implemented
        print("  Note: Pixel correlation CLI implementation needed")
        print(f"  Would analyze: {corr_input}")
        print(f"  Would save to: {corr_output}")
    
    # Step: QC Analysis
    if 'qc_analysis' in config and config['qc_analysis'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: QC ANALYSIS")
        print("="*60)
        
        qc_config = config['qc_analysis']
        qc_input = get_step_input(qc_config, workflow_state['input'], 'QC Analysis')
        qc_output = get_step_output(qc_config, output_base / 'qc_analysis.csv', 'QC Analysis')
        
        # This would require implementing a CLI version of QC analysis
        # For now, we'll note that this needs to be implemented
        print("  Note: QC analysis CLI implementation needed")
        print(f"  Would analyze: {qc_input}")
        print(f"  Would save to: {qc_output}")
        if qc_config.get('mask'):
            print(f"  Using mask: {qc_config['mask']}")
    
    # Step: Clustering (if configured)
    if 'clustering' in config and config['clustering'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: CLUSTERING")
        print("="*60)
        
        cluster_config = config['clustering']
        
        # Determine input features path
        if 'input_features' in cluster_config:
            features_path = Path(cluster_config['input_features'])
        elif workflow_state['batch_corrected_output']:
            features_path = workflow_state['batch_corrected_output']
        elif workflow_state['features_output']:
            features_path = workflow_state['features_output']
        else:
            raise ValueError("Clustering requires either 'input_features' path in config or feature extraction to be run first")
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        cluster_output = get_step_output(cluster_config, output_base / 'clustered_features.csv', 'Clustering')
        
        class ClusterArgs:
            pass
        cluster_args = ClusterArgs()
        cluster_args.features = str(features_path)
        cluster_args.output = str(cluster_output)
        cluster_args.method = cluster_config.get('method', 'leiden')
        cluster_args.n_clusters = cluster_config.get('n_clusters')
        cluster_args.columns = ','.join(cluster_config.get('columns', [])) if cluster_config.get('columns') else None
        cluster_args.scaling = cluster_config.get('scaling', 'zscore')
        cluster_args.linkage = cluster_config.get('linkage', 'ward')
        cluster_args.resolution = cluster_config.get('resolution', 1.0)
        cluster_args.min_cluster_size = cluster_config.get('min_cluster_size', 10)
        cluster_args.min_samples = cluster_config.get('min_samples', 5)
        cluster_args.seed = cluster_config.get('seed', 42)
        
        cluster_command(cluster_args)
    
    # Step: Spatial Analysis (if configured)
    if 'spatial_analysis' in config and config['spatial_analysis'].get('enabled', False):
        print("\n" + "="*60)
        print("STEP: SPATIAL ANALYSIS")
        print("="*60)
        
        spatial_config = config['spatial_analysis']
        
        # Determine input features path
        if 'input_features' in spatial_config:
            features_path = Path(spatial_config['input_features'])
        elif workflow_state['batch_corrected_output']:
            features_path = workflow_state['batch_corrected_output']
        elif workflow_state['features_output']:
            features_path = workflow_state['features_output']
        else:
            raise ValueError("Spatial analysis requires either 'input_features' path in config or feature extraction to be run first")
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        spatial_output = get_step_output(spatial_config, output_base / 'spatial_edges.csv', 'Spatial Analysis')
        
        class SpatialArgs:
            pass
        spatial_args = SpatialArgs()
        spatial_args.features = str(features_path)
        spatial_args.output = str(spatial_output)
        spatial_args.radius = spatial_config.get('radius')
        if not spatial_args.radius:
            raise ValueError("spatial_analysis.radius is required")
        spatial_args.k_neighbors = spatial_config.get('k_neighbors', 10)
        spatial_args.pixel_size_um = spatial_config.get('pixel_size_um', 1.0)
        spatial_args.detect_communities = spatial_config.get('detect_communities', False)
        spatial_args.seed = spatial_config.get('seed', 42)
        
        spatial_command(spatial_args)
    
    print("\n" + "="*60)
    print("✓ WORKFLOW COMPLETE")
    print("="*60)
    print(f"Output directory: {output_base}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='OpenIMC CLI for batch processing without GUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess images with denoising and arcsinh scaling
  openimc preprocess input.mcd output/ --arcsinh --arcsinh-cofactor 10.0

  # Segment cells using Cellpose (cytoplasm channels optional for cyto3)
  openimc segment input.mcd output/ --method cellpose --nuclear-channels DAPI --model cyto3 --gpu-id 0

  # Segment cells using Watershed (requires both channels)
  openimc segment input.mcd output/ --method watershed --nuclear-channels DNA1 --cytoplasm-channels CK8_CK18

  # Extract features (mask can be directory or single file)
  openimc extract-features input.mcd output/features.csv --mask output/masks/ --morphological --intensity

  # Cluster cells (Leiden uses resolution, not n-clusters)
  openimc cluster features.csv clustered_features.csv --method leiden --resolution 1.0
  openimc cluster features.csv clustered_features.csv --method hierarchical --n-clusters 10
  openimc cluster features.csv clustered_features.csv --method hdbscan --min-cluster-size 10

  # Spatial analysis (--radius is required)
  openimc spatial features.csv edges.csv --radius 50 --k-neighbors 10 --detect-communities

  # Generate figures
  openimc cluster-figures clustered_features.csv output/figures/ --dpi 300 --font-size 12
  openimc spatial-figures features.csv output/figures/ --edges edges.csv --dpi 300

  # Run complete workflow from config file
  openimc workflow config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess images (denoising, export to OME-TIFF). Note: arcsinh normalization is not applied to exported images.')
    preprocess_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    preprocess_parser.add_argument('output', help='Output directory for processed OME-TIFF files')
    preprocess_parser.add_argument('--arcsinh', action='store_true', help='(Deprecated) Arcsinh normalization is not applied to exported images. Use during feature extraction instead.')
    preprocess_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='(Deprecated) Arcsinh cofactor (default: 10.0). Not used for export.')
    preprocess_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    preprocess_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    preprocess_parser.set_defaults(func=preprocess_command)
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment cells (DeepCell CellSAM, Cellpose, or watershed)')
    segment_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    segment_parser.add_argument('output', help='Output directory for segmentation masks')
    segment_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    segment_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    segment_parser.add_argument('--method', choices=['cellsam', 'cellpose', 'watershed'], default='cellsam', help='Segmentation method (default: cellsam)')
    segment_parser.add_argument('--nuclear-channels', type=str, required=True, help='Comma-separated list of nuclear channel names')
    segment_parser.add_argument('--cytoplasm-channels', type=str, help='Comma-separated list of cytoplasm channel names (for cyto3 model)')
    segment_parser.add_argument('--nuclear-fusion-method', choices=['single', 'mean', 'weighted', 'max', 'pca1'], default='mean', help='Method to combine nuclear channels (default: mean)')
    segment_parser.add_argument('--cyto-fusion-method', choices=['single', 'mean', 'weighted', 'max', 'pca1'], default='mean', help='Method to combine cytoplasm channels (default: mean)')
    segment_parser.add_argument('--nuclear-weights', type=str, help='Comma-separated weights for nuclear channels (e.g., "0.5,0.3,0.2")')
    segment_parser.add_argument('--cyto-weights', type=str, help='Comma-separated weights for cytoplasm channels (e.g., "0.5,0.3,0.2")')
    segment_parser.add_argument('--model', choices=['cyto3', 'nuclei'], default='cyto3', help='Cellpose model type')
    segment_parser.add_argument('--diameter', type=int, help='Cell diameter in pixels (Cellpose)')
    segment_parser.add_argument('--flow-threshold', type=float, default=0.4, help='Flow threshold (Cellpose, default: 0.4)')
    segment_parser.add_argument('--cellprob-threshold', type=float, default=0.0, help='Cell probability threshold (Cellpose, default: 0.0)')
    segment_parser.add_argument('--gpu-id', type=int, help='GPU ID to use (Cellpose)')
    segment_parser.add_argument('--min-cell-area', type=int, default=100, help='Minimum cell area in pixels (watershed, default: 100)')
    segment_parser.add_argument('--max-cell-area', type=int, default=10000, help='Maximum cell area in pixels (watershed, default: 10000)')
    segment_parser.add_argument('--compactness', type=float, default=0.01, help='Watershed compactness (default: 0.01)')
    # DeepCell CellSAM parameters
    segment_parser.add_argument('--deepcell-api-key', type=str, help='DeepCell API key (CellSAM). Can also be set via DEEPCELL_ACCESS_TOKEN environment variable')
    segment_parser.add_argument('--bbox-threshold', type=float, default=0.4, help='Bbox threshold for CellSAM (default: 0.4, lower for faint cells: 0.01-0.1)')
    segment_parser.add_argument('--use-wsi', action='store_true', help='Use WSI mode for CellSAM (for ROIs with >500 cells, increases processing time)')
    segment_parser.add_argument('--low-contrast-enhancement', action='store_true', help='Enable low contrast enhancement for CellSAM (for poor contrast images)')
    segment_parser.add_argument('--gauge-cell-size', action='store_true', help='Enable gauge cell size for CellSAM (runs twice: estimates error, then returns mask)')
    segment_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh normalization before segmentation')
    segment_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    segment_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    segment_parser.set_defaults(func=segment_command)
    
    # Extract features command
    extract_parser = subparsers.add_parser('extract-features', help='Extract features from segmented cells')
    extract_parser.add_argument('input', help='Input MCD file or OME-TIFF directory')
    extract_parser.add_argument('output', help='Output CSV file path')
    extract_parser.add_argument('--channel-format', choices=['CHW', 'HWC'], default='CHW', help='Channel format for OME-TIFF files (default: CHW)')
    extract_parser.add_argument('--mask', type=str, required=True, help='Path to segmentation mask directory or single mask file (.tif, .tiff, or .npy). If directory, masks are matched to acquisitions by filename.')
    extract_parser.add_argument('--acquisition', type=str, help='Acquisition ID or name (uses first if not specified)')
    extract_parser.add_argument('--morphological', action='store_true', help='Extract morphological features')
    extract_parser.add_argument('--intensity', action='store_true', help='Extract intensity features')
    extract_parser.add_argument('--arcsinh', action='store_true', help='Apply arcsinh transformation to extracted intensity features (mean, median, std, etc.), not to raw images')
    extract_parser.add_argument('--arcsinh-cofactor', type=float, default=10.0, help='Arcsinh cofactor (default: 10.0)')
    extract_parser.add_argument('--denoise-settings', type=str, help='JSON file or string with denoise settings per channel')
    extract_parser.set_defaults(func=extract_features_command)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Perform clustering on feature data')
    cluster_parser.add_argument('features', help='Input CSV file with features')
    cluster_parser.add_argument('output', help='Output CSV file with cluster labels')
    cluster_parser.add_argument('--method', choices=['hierarchical', 'leiden', 'hdbscan'], default='leiden', help='Clustering method')
    cluster_parser.add_argument('--n-clusters', type=int, help='Number of clusters (required for hierarchical, not used for leiden/hdbscan)')
    cluster_parser.add_argument('--columns', type=str, help='Comma-separated list of columns to use for clustering (auto-detect if not specified)')
    cluster_parser.add_argument('--scaling', choices=['none', 'zscore', 'mad'], default='zscore', help='Feature scaling method (zscore or mad, matching GUI)')
    cluster_parser.add_argument('--linkage', choices=['ward', 'complete', 'average'], default='ward', help='Linkage method for hierarchical clustering')
    cluster_parser.add_argument('--resolution', type=float, default=1.0, help='Resolution parameter for Leiden clustering (default: 1.0)')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=10, help='Minimum cluster size (hdbscan, default: 10)')
    cluster_parser.add_argument('--min-samples', type=int, default=5, help='Minimum samples (hdbscan, default: 5)')
    cluster_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    cluster_parser.set_defaults(func=cluster_command)
    
    # Spatial command
    spatial_parser = subparsers.add_parser('spatial', help='Perform spatial analysis on feature data')
    spatial_parser.add_argument('features', help='Input CSV file with features (must contain centroid_x, centroid_y)')
    spatial_parser.add_argument('output', help='Output CSV file with spatial graph edges')
    spatial_parser.add_argument('--radius', type=float, required=True, help='Maximum distance for edges (pixels)')
    spatial_parser.add_argument('--k-neighbors', type=int, default=10, help='k for k-nearest neighbors (default: 10)')
    spatial_parser.add_argument('--pixel-size-um', type=float, default=1.0, help='Pixel size in micrometers (default: 1.0, used for distance_um conversion)')
    spatial_parser.add_argument('--detect-communities', action='store_true', help='Also detect spatial communities')
    spatial_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    spatial_parser.set_defaults(func=spatial_command)
    
    # Cluster figures command
    cluster_figures_parser = subparsers.add_parser('cluster-figures', help='Generate cluster visualization figures')
    cluster_figures_parser.add_argument('features', help='Input CSV file with clustered features')
    cluster_figures_parser.add_argument('output', help='Output directory for figures')
    cluster_figures_parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    cluster_figures_parser.add_argument('--font-size', type=float, default=10.0, help='Font size in points (default: 10.0)')
    cluster_figures_parser.add_argument('--width', type=float, default=8.0, help='Figure width in inches (default: 8.0)')
    cluster_figures_parser.add_argument('--height', type=float, default=6.0, help='Figure height in inches (default: 6.0)')
    cluster_figures_parser.add_argument('--seed', type=int, default=42, help='Random seed for UMAP reproducibility (default: 42)')
    cluster_figures_parser.set_defaults(func=cluster_figures_command)
    
    # Spatial figures command
    spatial_figures_parser = subparsers.add_parser('spatial-figures', help='Generate spatial analysis visualization figures')
    spatial_figures_parser.add_argument('features', help='Input CSV file with features (must contain centroid_x, centroid_y)')
    spatial_figures_parser.add_argument('output', help='Output directory for figures')
    spatial_figures_parser.add_argument('--edges', type=str, help='Optional CSV file with spatial graph edges')
    spatial_figures_parser.add_argument('--dpi', type=int, default=300, help='Figure DPI (default: 300)')
    spatial_figures_parser.add_argument('--font-size', type=float, default=10.0, help='Font size in points (default: 10.0)')
    spatial_figures_parser.add_argument('--width', type=float, default=8.0, help='Figure width in inches (default: 8.0)')
    spatial_figures_parser.add_argument('--height', type=float, default=6.0, help='Figure height in inches (default: 6.0)')
    spatial_figures_parser.set_defaults(func=spatial_figures_command)
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Execute a complete workflow from a YAML configuration file')
    workflow_parser.add_argument('config', help='Path to YAML configuration file')
    workflow_parser.set_defaults(func=workflow_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run the command
    try:
        args.func(args)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

