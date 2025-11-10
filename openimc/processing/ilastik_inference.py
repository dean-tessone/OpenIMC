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
Ilastik model loading and inference pipeline.

This module provides functionality to load Ilastik project files (.ilp) and
run inference on images using trained Ilastik models.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import os
import tempfile
import subprocess
import shutil
from pathlib import Path

# Try to import Ilastik Python API
try:
    import ilastik
    from ilastik.applets.pixelClassification import PixelClassificationApplet
    from lazyflow.operators.ioOperators import OpStackLoader
    _HAVE_ILASTIK_API = True
except ImportError:
    _HAVE_ILASTIK_API = False


class IlastikInferencePipeline:
    """
    Pipeline for loading Ilastik models and running inference.
    
    Supports both Ilastik Python API (if available) and headless command-line mode.
    """
    
    def __init__(self, project_path: str):
        """
        Initialize the pipeline with an Ilastik project file.
        
        Args:
            project_path: Path to the .ilp Ilastik project file
        """
        if not os.path.exists(project_path):
            raise FileNotFoundError(f"Ilastik project file not found: {project_path}")
        
        if not project_path.endswith('.ilp'):
            raise ValueError(f"Expected .ilp file, got: {project_path}")
        
        self.project_path = os.path.abspath(project_path)
        self._check_ilastik_available()
    
    def _check_ilastik_available(self):
        """Check if Ilastik is available either as Python API or command-line."""
        if _HAVE_ILASTIK_API:
            self.use_api = True
            return
        
        # Check if ilastik command is available
        try:
            result = subprocess.run(
                ['ilastik', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.use_api = False
                return
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        raise RuntimeError(
            "Ilastik is not available. Please install Ilastik or ensure it's in your PATH.\n"
            "You can install Ilastik from: https://www.ilastik.org/download"
        )
    
    def run_inference(self, 
                     img_stack: np.ndarray,
                     channel_names: List[str],
                     output_format: str = "Simple Segmentation",
                     tile_size: Optional[int] = None,
                     return_probabilities: bool = False) -> Dict[str, np.ndarray]:
        """
        Run inference on an image stack.
        
        Args:
            img_stack: Image stack of shape (H, W, C) with dtype uint8 or uint16
            channel_names: List of channel names
            output_format: Output format ("Simple Segmentation", "Probabilities", etc.)
            tile_size: Optional tile size for large images (not used with API)
            return_probabilities: Whether to return probability maps
            
        Returns:
            Dictionary with results:
            - 'labels': Instance segmentation labels (if output_format is "Simple Segmentation")
            - 'probabilities': Probability maps (if return_probabilities=True)
        """
        if self.use_api:
            return self._run_inference_api(img_stack, channel_names, return_probabilities)
        else:
            return self._run_inference_headless(img_stack, channel_names, output_format, return_probabilities)
    
    def _run_inference_api(self, 
                          img_stack: np.ndarray,
                          channel_names: List[str],
                          return_probabilities: bool) -> Dict[str, np.ndarray]:
        """
        Run inference using Ilastik Python API.
        
        Note: This is a placeholder - full API integration requires more complex setup.
        For now, we fall back to headless mode.
        """
        # The Ilastik Python API requires more complex initialization
        # For now, we'll use headless mode as it's more reliable
        return self._run_inference_headless(
            img_stack, channel_names, 
            "Probabilities" if return_probabilities else "Simple Segmentation",
            return_probabilities
        )
    
    def _run_inference_headless(self,
                               img_stack: np.ndarray,
                               channel_names: List[str],
                               output_format: str,
                               return_probabilities: bool) -> Dict[str, np.ndarray]:
        """
        Run inference using Ilastik headless command-line mode.
        
        Args:
            img_stack: Image stack of shape (H, W, C)
            channel_names: List of channel names
            output_format: Output format string
            return_probabilities: Whether to return probability maps
            
        Returns:
            Dictionary with results
        """
        height, width, n_channels = img_stack.shape
        
        # Create temporary directory for input/output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save input image as TIFF
            input_path = os.path.join(temp_dir, "input.tif")
            self._save_image_stack(img_stack, input_path)
            
            # Determine output format
            if return_probabilities or output_format == "Probabilities":
                export_source = "Probabilities"
                output_ext = ".h5"
            else:
                export_source = "Simple Segmentation"
                output_ext = ".tif"
            
            # Ilastik appends export source to input filename, so we need to construct the expected output path
            input_basename = os.path.splitext(os.path.basename(input_path))[0]
            if output_ext == '.h5':
                output_path = os.path.join(temp_dir, f"{input_basename}_Probabilities.h5")
            else:
                output_path = os.path.join(temp_dir, f"{input_basename}_Simple Segmentation.tif")
            
            # Run Ilastik headless
            cmd = [
                'ilastik',
                '--headless',
                '--project=' + self.project_path,
                '--export_source=' + export_source,
                '--output_format=tif' if output_ext == '.tif' else '--output_format=hdf5',
                input_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    check=True
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError("Ilastik inference timed out after 5 minutes")
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else e.stdout
                raise RuntimeError(
                    f"Ilastik inference failed:\n{error_msg}\n"
                    f"Command: {' '.join(cmd)}"
                )
            
            # Load results
            results = {}
            
            if os.path.exists(output_path):
                if return_probabilities or export_source == "Probabilities":
                    # Load HDF5 probabilities
                    probabilities = self._load_probabilities(output_path)
                    results['probabilities'] = probabilities
                    
                    # Convert probabilities to labels if needed
                    if not return_probabilities:
                        labels = self._probabilities_to_labels(probabilities)
                        results['labels'] = labels
                else:
                    # Load segmentation labels
                    labels = self._load_segmentation_labels(output_path)
                    results['labels'] = labels
                    
                    # Optionally extract probabilities if available
                    if return_probabilities:
                        # Try to find probability file
                        prob_path = output_path.replace('.tif', '_Probabilities.h5')
                        if os.path.exists(prob_path):
                            probabilities = self._load_probabilities(prob_path)
                            results['probabilities'] = probabilities
            else:
                raise RuntimeError(f"Ilastik did not produce output file: {output_path}")
            
            return results
    
    def _save_image_stack(self, img_stack: np.ndarray, output_path: str):
        """Save image stack as multi-channel TIFF."""
        try:
            import tifffile
            # Ensure uint16 or uint8
            if img_stack.dtype == np.float32 or img_stack.dtype == np.float64:
                # Normalize to uint16
                img_max = np.max(img_stack)
                if img_max > 0:
                    img_stack = (img_stack / img_max * 65535).astype(np.uint16)
                else:
                    img_stack = img_stack.astype(np.uint16)
            elif img_stack.dtype != np.uint16 and img_stack.dtype != np.uint8:
                img_stack = img_stack.astype(np.uint16)
            
            # Save as multi-page TIFF (one page per channel)
            tifffile.imwrite(output_path, img_stack, photometric='minisblack')
        except ImportError:
            raise ImportError("tifffile is required for Ilastik integration")
    
    def _load_segmentation_labels(self, tiff_path: str) -> np.ndarray:
        """Load segmentation labels from TIFF file."""
        try:
            import tifffile
            labels = tifffile.imread(tiff_path)
            # Ensure integer type
            if labels.dtype != np.int32 and labels.dtype != np.int64:
                labels = labels.astype(np.int32)
            return labels
        except ImportError:
            raise ImportError("tifffile is required for Ilastik integration")
    
    def _load_probabilities(self, h5_path: str) -> Dict[str, np.ndarray]:
        """Load probability maps from HDF5 file."""
        try:
            import h5py
            probabilities = {}
            
            with h5py.File(h5_path, 'r') as f:
                # Ilastik typically stores probabilities in 'exported_data' or similar
                if 'exported_data' in f:
                    data = f['exported_data'][:]
                    # Shape is typically (C, H, W) or (H, W, C)
                    if len(data.shape) == 3:
                        if data.shape[0] < data.shape[2]:
                            # (C, H, W) -> (H, W, C)
                            data = np.transpose(data, (1, 2, 0))
                        
                        # Split into class probabilities
                        # Assuming classes: background, class1, class2, etc.
                        for i in range(data.shape[2]):
                            probabilities[f'class_{i}'] = data[:, :, i]
                else:
                    # Try to find any dataset
                    keys = list(f.keys())
                    if keys:
                        data = f[keys[0]][:]
                        if len(data.shape) == 3:
                            if data.shape[0] < data.shape[2]:
                                data = np.transpose(data, (1, 2, 0))
                            for i in range(data.shape[2]):
                                probabilities[f'class_{i}'] = data[:, :, i]
            
            return probabilities
        except ImportError:
            raise ImportError("h5py is required for loading Ilastik probabilities")
    
    def _probabilities_to_labels(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert probability maps to segmentation labels."""
        if not probabilities:
            raise ValueError("No probabilities provided")
        
        # Stack probabilities
        prob_arrays = list(probabilities.values())
        prob_stack = np.stack(prob_arrays, axis=-1)
        
        # Get class with maximum probability
        labels = np.argmax(prob_stack, axis=-1).astype(np.int32)
        
        return labels
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            "project_path": self.project_path,
            "use_api": self.use_api if hasattr(self, 'use_api') else False
        }

