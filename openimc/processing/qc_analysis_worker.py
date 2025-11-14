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
Worker functions for QC analysis using multiprocessing.
Can be used by both CLI and GUI.
"""

import os
from typing import List, Dict, Optional, Any
import numpy as np
import tifffile

from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.core import qc_analysis

# Optional scikit-image for pixel-level QC
try:
    from skimage.filters import threshold_otsu
    _HAVE_SCIKIT_IMAGE = True
except ImportError:
    _HAVE_SCIKIT_IMAGE = False
    threshold_otsu = None


def qc_process_acquisition_worker(task_data):
    """Thin wrapper around core.qc_analysis for multiprocessing."""
    print(f"[QC WORKER DEBUG] Starting worker, task_data length: {len(task_data)}")
    
    # Handle both old format (without original_acq_id) and new format (with original_acq_id)
    if len(task_data) == 8:
        acq_id, original_acq_id, acq_name, channels, analysis_mode, mask_path, loader_path, _ = task_data
        print(f"[QC WORKER DEBUG] Using new format - unique ID: {acq_id}, original ID: {original_acq_id}")
    else:
        # Old format: assume acq_id is the original ID (for backward compatibility)
        acq_id, acq_name, channels, analysis_mode, mask_path, loader_path, _ = task_data
        original_acq_id = acq_id
        print(f"[QC WORKER DEBUG] Using old format - acq_id: {acq_id}")
    
    print(f"[QC WORKER DEBUG] Processing: {acq_name}, channels: {len(channels)}, mode: {analysis_mode}, file: {os.path.basename(loader_path) if loader_path else 'None'}")
    
    try:
        # Recreate loader (can't pickle loader objects)
        if loader_path and os.path.exists(loader_path):
            print(f"[QC WORKER DEBUG] Opening loader for: {loader_path}")
            if loader_path.lower().endswith(('.mcd', '.mcdx')):
                loader = MCDLoader()
                loader.open(loader_path)
                print(f"[QC WORKER DEBUG] Opened MCD loader successfully")
            elif os.path.isdir(loader_path):
                loader = OMETIFFLoader(channel_format='CHW')
                loader.open(loader_path)
                print(f"[QC WORKER DEBUG] Opened OME-TIFF loader successfully")
            else:
                print(f"[QC WORKER DEBUG] ERROR: Unknown file type: {loader_path}")
                return []
        else:
            print(f"[QC WORKER DEBUG] ERROR: Loader path invalid or doesn't exist: {loader_path}")
            return []
        
        try:
            # Load mask if needed
            mask = None
            if analysis_mode == "cell" and mask_path and os.path.exists(mask_path):
                try:
                    mask = tifffile.imread(mask_path)
                except Exception:
                    return []
            
            # Get acquisition metadata from loader using original acquisition ID
            print(f"[QC WORKER DEBUG] Getting channels for original_acq_id: {original_acq_id}")
            try:
                all_channels = loader.get_channels(original_acq_id)
                print(f"[QC WORKER DEBUG] Loader returned {len(all_channels) if all_channels else 0} channels")
                if not all_channels:
                    print(f"[QC WORKER DEBUG] WARNING: No channels returned from loader for {original_acq_id}")
                    print(f"[QC WORKER DEBUG] Available acquisition IDs in loader: {list(getattr(loader, '_acq_channels', {}).keys())}")
            except Exception as e:
                print(f"[QC WORKER DEBUG] ERROR getting channels: {e}")
                import traceback
                traceback.print_exc()
                return []
            
            all_channel_metals = getattr(loader, '_acq_channel_metals', {}).get(original_acq_id, [])
            all_channel_labels = getattr(loader, '_acq_channel_labels', {}).get(original_acq_id, [])
            well = getattr(loader, '_acq_well', {}).get(original_acq_id)
            size = getattr(loader, '_acq_size', {}).get(original_acq_id, (None, None))
            metadata = getattr(loader, '_acq_metadata', {}).get(original_acq_id, {})
            print(f"[QC WORKER DEBUG] Metadata retrieved - well: {well}, size: {size}")
            
            # Filter channel_metals and channel_labels to match provided channels
            channel_metals = []
            channel_labels = []
            for ch in channels:
                if ch in all_channels:
                    idx = all_channels.index(ch)
                    if idx < len(all_channel_metals):
                        channel_metals.append(all_channel_metals[idx])
                    else:
                        channel_metals.append("")
                    if idx < len(all_channel_labels):
                        channel_labels.append(all_channel_labels[idx])
                    else:
                        channel_labels.append("")
                else:
                    channel_metals.append("")
                    channel_labels.append("")
            
            # Create AcquisitionInfo with original ID (needed for loader methods)
            # The loader needs the original ID, but we'll preserve the unique ID in results
            acquisition = AcquisitionInfo(
                id=original_acq_id,  # Use original ID for loader compatibility
                name=acq_name,
                well=well,
                size=size,
                channels=channels,
                channel_metals=channel_metals,
                channel_labels=channel_labels,
                metadata=metadata,
                source_file=loader_path
            )
            
            # Call core function
            print(f"[QC WORKER DEBUG] Calling qc_analysis with acquisition.id={acquisition.id}, {len(channels)} channels")
            try:
                results_df = qc_analysis(
                    loader=loader,
                    acquisition=acquisition,
                    channels=channels,
                    mode=analysis_mode,
                    mask=mask
                )
                print(f"[QC WORKER DEBUG] qc_analysis returned {len(results_df)} results")
            except Exception as e:
                print(f"[QC WORKER DEBUG] ERROR in qc_analysis: {e}")
                import traceback
                traceback.print_exc()
                return []
            
            # If we have a unique ID (different from original), update results to use it
            if acq_id != original_acq_id and not results_df.empty:
                print(f"[QC WORKER DEBUG] Updating results acquisition_id from {original_acq_id} to {acq_id}")
                results_df['acquisition_id'] = acq_id
            
            # Convert to list of dicts for GUI compatibility
            result_list = [row.to_dict() for _, row in results_df.iterrows()] if not results_df.empty else []
            print(f"[QC WORKER DEBUG] Returning {len(result_list)} results for {acq_name}")
            return result_list
        finally:
            if hasattr(loader, 'close'):
                loader.close()
    except Exception as e:
        print(f"[QC WORKER DEBUG] EXCEPTION in QC analysis for {acq_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def qc_calculate_pixel_metrics_worker(img: np.ndarray, channel: str) -> Optional[Dict[str, Any]]:
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
        if background_std > 0:
            snr = (signal_mean - background_mean) / background_std
        else:
            # Fallback: use range-based SNR
            if img_max > img_min:
                snr = (signal_mean - background_mean) / (img_max - img_min + 1e-6)
            else:
                snr = 0.0
        
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


def qc_calculate_cell_metrics_worker(img: np.ndarray, channel: str, mask: np.ndarray) -> Optional[Dict[str, Any]]:
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
        
        foreground = img_float[cell_mask]
        background = img_float[background_mask]
        
        # Calculate metrics
        signal_mean = np.mean(foreground)
        signal_std = np.std(foreground)
        background_mean = np.mean(background)
        background_std = np.std(background)
        
        # SNR: (signal_mean - background_mean) / background_std
        if background_std > 0:
            snr = (signal_mean - background_mean) / background_std
        else:
            snr = 0.0
        
        # Intensity metrics
        mean_intensity = np.mean(img_float)
        median_intensity = np.median(img_float)
        max_intensity = np.max(img_float)
        min_intensity = np.min(img_float)
        
        # Coverage: percentage of pixels in cells
        coverage_pct = (np.sum(cell_mask) / img_float.size) * 100
        
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
            'cell_pixels': np.sum(cell_mask),
            'background_pixels': np.sum(background_mask)
        }
    except Exception as e:
        print(f"Error calculating cell metrics for {channel}: {e}")
        return None

