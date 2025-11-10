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
Unit tests for watershed segmentation.
"""
import numpy as np
import pytest


@pytest.mark.unit
class TestWatershedSegmentation:
    """Tests for watershed segmentation functions."""
    
    def test_watershed_segmentation_basic(self, sample_image_stack_chw, sample_channels):
        """Test basic watershed segmentation."""
        pytest.importorskip("skimage")
        
        from openimc.processing.watershed_worker import watershed_segmentation
        
        # Convert CHW to HWC for watershed
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        nuclear_channels = [sample_channels[0]]  # DAPI
        cyto_channels = [sample_channels[1]]  # CD45
        
        result = watershed_segmentation(
            img_stack_hwc,
            sample_channels,
            nuclear_channels,
            cyto_channels,
            denoise_settings=None,
            normalization_method="arcsinh",
            arcsinh_cofactor=10.0,
            min_cell_area=100,
            max_cell_area=10000,
            compactness=0.01
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == img_stack_hwc.shape[:2]  # Should be 2D mask
        assert result.dtype in [np.uint32, np.int32, np.int64]
    
    def test_watershed_segmentation_no_cyto(self, sample_image_stack_chw, sample_channels):
        """Test watershed segmentation with only nuclear channels."""
        pytest.importorskip("skimage")
        
        from openimc.processing.watershed_worker import watershed_segmentation
        
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        nuclear_channels = [sample_channels[0]]
        cyto_channels = []
        
        result = watershed_segmentation(
            img_stack_hwc,
            sample_channels,
            nuclear_channels,
            cyto_channels,
            denoise_settings=None,
            normalization_method="arcsinh",
            arcsinh_cofactor=10.0,
            min_cell_area=100,
            max_cell_area=10000,
            compactness=0.01
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == img_stack_hwc.shape[:2]
    
    def test_watershed_segmentation_with_denoise(self, sample_image_stack_chw, sample_channels, sample_denoise_settings):
        """Test watershed segmentation with denoising."""
        pytest.importorskip("skimage")
        
        from openimc.processing.watershed_worker import watershed_segmentation
        
        img_stack_hwc = np.moveaxis(sample_image_stack_chw, 0, -1)
        
        nuclear_channels = [sample_channels[0]]
        cyto_channels = [sample_channels[1]]
        
        # Create denoise settings dict for channels
        denoise_settings = {
            sample_channels[0]: sample_denoise_settings
        }
        
        result = watershed_segmentation(
            img_stack_hwc,
            sample_channels,
            nuclear_channels,
            cyto_channels,
            denoise_settings=denoise_settings,
            normalization_method="arcsinh",
            arcsinh_cofactor=10.0,
            min_cell_area=100,
            max_cell_area=10000,
            compactness=0.01
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == img_stack_hwc.shape[:2]

