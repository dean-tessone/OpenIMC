from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import os
import glob

import numpy as np

from .mcd_loader import AcquisitionInfo

_HAVE_TIFFFILE = True
try:
    import tifffile
except Exception:
    _HAVE_TIFFFILE = False


class OMETIFFLoader:
    """Loader for OME-TIFF files in a directory, treating each file as an acquisition."""

    def __init__(self):
        if not _HAVE_TIFFFILE:
            raise RuntimeError("tifffile is not installed. Run: pip install tifffile")
        self.folder_path: Optional[str] = None
        self._acq_map: Dict[str, str] = {}  # Maps acq_id to file path
        self._acq_channels: Dict[str, List[str]] = {}
        self._acq_channel_metals: Dict[str, List[str]] = {}
        self._acq_channel_labels: Dict[str, List[str]] = {}
        self._acq_size: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        self._acq_name: Dict[str, str] = {}
        self._acq_well: Dict[str, Optional[str]] = {}
        self._acq_metadata: Dict[str, Dict] = {}
        self._image_cache: Dict[str, np.ndarray] = {}  # Cache loaded images

    def open(self, path: str):
        """Open a folder containing OME-TIFF files."""
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
        self.folder_path = path
        self._index()

    def _index(self):
        """Index all OME-TIFF files in the folder."""
        self._acq_map.clear()
        self._acq_channels.clear()
        self._acq_channel_metals.clear()
        self._acq_channel_labels.clear()
        self._acq_size.clear()
        self._acq_name.clear()
        self._acq_well.clear()
        self._acq_metadata.clear()
        self._image_cache.clear()

        if not self.folder_path:
            raise RuntimeError("No folder path set.")

        # Find all .ome.tif and .ome.tiff files (case-insensitive)
        patterns = [
            os.path.join(self.folder_path, "*.ome.tif"),
            os.path.join(self.folder_path, "*.ome.tiff"),
            os.path.join(self.folder_path, "*.tif"),
            os.path.join(self.folder_path, "*.tiff"),
        ]
        
        tiff_files = []
        for pattern in patterns:
            tiff_files.extend(glob.glob(pattern))
            tiff_files.extend(glob.glob(pattern.upper()))

        # Remove duplicates and sort
        tiff_files = sorted(list(set(tiff_files)))

        if not tiff_files:
            raise RuntimeError(f"No OME-TIFF files found in directory: {self.folder_path}")

        for idx, tiff_path in enumerate(tiff_files):
            acq_id = f"file_{idx}"
            filename = os.path.basename(tiff_path)
            name = os.path.splitext(os.path.splitext(filename)[0])[0]  # Remove .ome.tif extensions
            
            # Try to read OME metadata
            try:
                with tifffile.TiffFile(tiff_path) as tif:
                    # Get OME metadata if available
                    ome_metadata = None
                    if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                        import xml.etree.ElementTree as ET
                        ome_metadata = ET.fromstring(tif.ome_metadata)
                    
                    # Get image shape
                    img_shape = tif.series[0].shape if tif.series else None
                    
                    # Extract channel information from OME metadata
                    channels = []
                    channel_metals = []
                    channel_labels = []
                    
                    if ome_metadata is not None:
                        # Try to extract channel names from OME metadata
                        # Look for Channel elements in the Pixels section
                        pixels_elem = ome_metadata.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                        if pixels_elem is not None:
                            for channel in pixels_elem.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel'):
                                channel_id = channel.get('ID', '')
                                
                                # Try to get channel name from Name attribute or element
                                channel_name = channel.get('Name', '')
                                if not channel_name:
                                    # Try to find Name as a child element
                                    name_elem = channel.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Name')
                                    if name_elem is not None:
                                        channel_name = name_elem.text or ''
                                
                                # Fallback to channel ID if no name found
                                if not channel_name:
                                    channel_name = channel_id
                                
                                # Try to extract metal/label info if available
                                metal = ''
                                label = ''
                                if '_' in channel_name:
                                    parts = channel_name.split('_', 1)
                                    if len(parts) == 2:
                                        label, metal = parts
                                else:
                                    metal = channel_name
                                
                                channels.append(channel_name)
                                channel_metals.append(metal)
                                channel_labels.append(label)
                    
                    # If no channels found in metadata, infer from image dimensions
                    if not channels:
                        if img_shape:
                            # Assume last dimension is channels
                            if len(img_shape) == 3:
                                n_channels = img_shape[2]
                            elif len(img_shape) == 4:  # TZCYX format
                                n_channels = img_shape[1] if img_shape[0] == 1 else img_shape[0]
                            elif len(img_shape) == 5:  # TZCYX format
                                n_channels = img_shape[2]
                            else:
                                n_channels = 1
                            
                            for i in range(n_channels):
                                channels.append(f"Channel_{i+1}")
                                channel_metals.append(f"Channel_{i+1}")
                                channel_labels.append("")
                        else:
                            # Default to single channel
                            channels = ["Channel_1"]
                            channel_metals = ["Channel_1"]
                            channel_labels = [""]
                    
                    # Determine image size
                    if img_shape:
                        if len(img_shape) >= 2:
                            # For 2D: (H, W)
                            # For 3D: (H, W, C) or (C, H, W)
                            # For 4D: (T, Z, H, W) or (T, H, W, C)
                            # For 5D: (T, Z, C, H, W)
                            if len(img_shape) == 2:
                                H, W = img_shape
                            elif len(img_shape) == 3:
                                # Try to determine if channels first or last
                                if img_shape[0] < img_shape[2]:
                                    H, W, C = img_shape
                                else:
                                    C, H, W = img_shape
                            elif len(img_shape) == 4:
                                # Assume (T, H, W, C) or (T, C, H, W)
                                if img_shape[1] < img_shape[3]:
                                    H, W = img_shape[1], img_shape[2]
                                else:
                                    H, W = img_shape[2], img_shape[3]
                            elif len(img_shape) == 5:
                                H, W = img_shape[3], img_shape[4]
                            else:
                                H, W = None, None
                        else:
                            H, W = None, None
                    else:
                        H, W = None, None
                    
                    metadata = {}
                    if ome_metadata is not None:
                        # Try to extract well information
                        well_elem = ome_metadata.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Well')
                        if well_elem is not None:
                            well_name = well_elem.get('ID', '')
                            metadata['Well'] = well_name
                    
            except Exception as e:
                # If reading metadata fails, use defaults
                print(f"Warning: Could not read metadata from {filename}: {e}")
                channels = ["Channel_1"]
                channel_metals = ["Channel_1"]
                channel_labels = [""]
                H, W = None, None
                metadata = {}
            
            well = metadata.get('Well') if metadata else None
            
            self._acq_map[acq_id] = tiff_path
            self._acq_channels[acq_id] = channels
            self._acq_channel_metals[acq_id] = channel_metals
            self._acq_channel_labels[acq_id] = channel_labels
            self._acq_size[acq_id] = (H, W) if H and W else (None, None)
            self._acq_name[acq_id] = name
            self._acq_well[acq_id] = well
            self._acq_metadata[acq_id] = metadata

        if not self._acq_map:
            raise RuntimeError("No valid OME-TIFF files found in this directory.")

    def list_acquisitions(self) -> List[AcquisitionInfo]:
        """List all acquisitions (files) in the folder."""
        infos: List[AcquisitionInfo] = []
        for acq_id in self._acq_map:
            infos.append(
                AcquisitionInfo(
                    id=acq_id,
                    name=self._acq_name.get(acq_id, acq_id),
                    well=self._acq_well.get(acq_id),
                    size=self._acq_size.get(acq_id, (None, None)),
                    channels=self._acq_channels.get(acq_id, []),
                    channel_metals=self._acq_channel_metals.get(acq_id, []),
                    channel_labels=self._acq_channel_labels.get(acq_id, []),
                    metadata=self._acq_metadata.get(acq_id, {}),
                )
            )
        return infos

    def get_channels(self, acq_id: str) -> List[str]:
        """Get channel names for a specific acquisition."""
        return self._acq_channels[acq_id]

    def get_image(self, acq_id: str, channel: str) -> np.ndarray:
        """Get image data for a specific acquisition and channel."""
        if acq_id not in self._acq_map:
            raise ValueError(f"Acquisition '{acq_id}' not found.")
        
        channels = self._acq_channels[acq_id]
        if channel not in channels:
            raise ValueError(f"Channel '{channel}' not found in acquisition {acq_id}.")
        ch_idx = channels.index(channel)
        
        tiff_path = self._acq_map[acq_id]
        
        # Try cache first
        cache_key = (acq_id, channel)
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Load the full image stack
        img_stack = self.get_all_channels(acq_id)
        
        # Extract the specific channel
        if img_stack.ndim == 3:
            img = img_stack[..., ch_idx]
        elif img_stack.ndim == 2:
            # Single channel image
            img = img_stack
        else:
            raise ValueError(f"Unexpected image shape: {img_stack.shape}")
        
        # Cache the result
        self._image_cache[cache_key] = img
        return img

    def get_all_channels(self, acq_id: str) -> np.ndarray:
        """Get all channels for a specific acquisition as a 3D array (H, W, C)."""
        if acq_id not in self._acq_map:
            raise ValueError(f"Acquisition '{acq_id}' not found.")
        
        tiff_path = self._acq_map[acq_id]
        
        # Load the image
        try:
            img = tifffile.imread(tiff_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read image from {tiff_path}: {e}")
        
        # Normalize to (H, W, C) format
        if img.ndim == 2:
            # Single channel, add channel dimension
            img = img[..., np.newaxis]
        elif img.ndim == 3:
            # Could be (C, H, W) or (H, W, C)
            # Assume (H, W, C) if last dim is smaller, otherwise transpose
            if img.shape[0] < img.shape[2]:
                # Likely (C, H, W), transpose to (H, W, C)
                img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 4:
            # Could be (T, C, H, W), (T, H, W, C), (Z, C, H, W), etc.
            # For time series, take first time point
            # For z-stack, take first z slice
            if img.shape[0] == 1:
                img = img[0]
                if img.ndim == 3 and img.shape[0] < img.shape[2]:
                    img = np.transpose(img, (1, 2, 0))
            elif img.shape[1] == 1:
                img = img[:, 0]
                if img.ndim == 3 and img.shape[0] < img.shape[2]:
                    img = np.transpose(img, (1, 2, 0))
            else:
                # Take first slice and assume (C, H, W) or (H, W, C)
                img = img[0]
                if img.ndim == 3 and img.shape[0] < img.shape[2]:
                    img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 5:
            # (T, Z, C, H, W) or similar
            img = img[0, 0]  # Take first time point and z slice
            if img.ndim == 3 and img.shape[0] < img.shape[2]:
                img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported image dimensionality: {img.ndim}D")
        
        return img

    def close(self):
        """Close the loader and clear cache."""
        self._image_cache.clear()
        self.folder_path = None

