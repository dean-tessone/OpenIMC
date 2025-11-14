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
Worker functions for pixel correlation analysis using multiprocessing.
Can be used by both CLI and GUI.
"""

import os
from typing import Dict, List
import numpy as np
import tifffile

from openimc.data.mcd_loader import MCDLoader, AcquisitionInfo
from openimc.data.ometiff_loader import OMETIFFLoader
from openimc.core import pixel_correlation


def correlation_process_roi_worker(task_data):
    """Thin wrapper around core.pixel_correlation for multiprocessing."""
    (acq_id, acq_name, file_path, loader_type, selected_channels, 
     original_acq_id, analyze_within_masks, mask_path) = task_data
    
    try:
        # Recreate loader (can't pickle loader objects)
        if loader_type == "mcd":
            loader = MCDLoader()
            loader.open(file_path)
        elif loader_type == "ometiff":
            loader = OMETIFFLoader(channel_format='CHW')
            loader.open(file_path)
        else:
            return []
        
        try:
            # Get and filter channels
            all_channels = loader.get_channels(original_acq_id)
            channels = [ch for ch in all_channels if ch in selected_channels] if selected_channels else all_channels
            
            if len(channels) < 2:
                return []
            
            # Load mask if needed
            mask = None
            if analyze_within_masks and mask_path and os.path.exists(mask_path):
                try:
                    mask = np.load(mask_path) if mask_path.endswith('.npy') else tifffile.imread(mask_path)
                except Exception:
                    pass
            
            # Get acquisition metadata from loader
            all_channel_metals = getattr(loader, '_acq_channel_metals', {}).get(original_acq_id, [])
            all_channel_labels = getattr(loader, '_acq_channel_labels', {}).get(original_acq_id, [])
            well = getattr(loader, '_acq_well', {}).get(original_acq_id)
            size = getattr(loader, '_acq_size', {}).get(original_acq_id, (None, None))
            metadata = getattr(loader, '_acq_metadata', {}).get(original_acq_id, {})
            
            # Filter channel_metals and channel_labels to match selected channels
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
            
            # Create AcquisitionInfo with correct parameters
            acquisition = AcquisitionInfo(
                id=original_acq_id,
                name=acq_name,
                well=well,
                size=size,
                channels=channels,
                channel_metals=channel_metals,
                channel_labels=channel_labels,
                metadata=metadata,
                source_file=file_path
            )
            
            # Call core function
            results_df = pixel_correlation(
                loader=loader,
                acquisition=acquisition,
                channels=channels,
                mask=mask,
                multiple_testing_correction=None
            )
            
            # Convert to list of dicts for GUI compatibility
            return [row.to_dict() for _, row in results_df.iterrows()] if not results_df.empty else []
        finally:
            if hasattr(loader, 'close'):
                loader.close()
    except Exception as e:
        print(f"Error processing ROI {acq_name}: {e}")
        import traceback
        traceback.print_exc()
        return []

