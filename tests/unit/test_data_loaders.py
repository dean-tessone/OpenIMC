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
Unit tests for data loaders (MCDLoader and OMETIFFLoader).
"""
import pytest
import numpy as np
from pathlib import Path

# These tests may require actual data files or sophisticated mocking
# Marking as requiring readimc for MCD tests


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestOMETIFFLoader:
    """Tests for OMETIFFLoader."""
    
    def test_ometiff_loader_init_chw(self):
        """Test OMETIFFLoader initialization with CHW format."""
        from openimc.data.ometiff_loader import OMETIFFLoader
        
        loader = OMETIFFLoader(channel_format='CHW')
        assert loader.channel_format == 'CHW'
    
    def test_ometiff_loader_init_hwc(self):
        """Test OMETIFFLoader initialization with HWC format."""
        from openimc.data.ometiff_loader import OMETIFFLoader
        
        loader = OMETIFFLoader(channel_format='HWC')
        assert loader.channel_format == 'HWC'
    
    def test_ometiff_loader_invalid_format(self):
        """Test OMETIFFLoader with invalid format raises error."""
        from openimc.data.ometiff_loader import OMETIFFLoader
        
        with pytest.raises(ValueError, match="channel_format must be"):
            OMETIFFLoader(channel_format='invalid')
    
    def test_ometiff_loader_open_directory(self, mock_ometiff_directory):
        """Test opening a directory with OME-TIFF files."""
        from openimc.data.ometiff_loader import OMETIFFLoader
        
        loader = OMETIFFLoader(channel_format='CHW')
        loader.open(str(mock_ometiff_directory))
        
        acquisitions = loader.list_acquisitions()
        assert len(acquisitions) > 0
    
    def test_ometiff_loader_open_invalid_path(self):
        """Test opening invalid path raises error."""
        from openimc.data.ometiff_loader import OMETIFFLoader
        
        loader = OMETIFFLoader(channel_format='CHW')
        
        with pytest.raises(ValueError, match="Path is not a directory"):
            loader.open("/nonexistent/path")
    
    def test_ometiff_loader_get_channels(self, mock_ometiff_directory):
        """Test getting channels from OME-TIFF loader."""
        from openimc.data.ometiff_loader import OMETIFFLoader
        
        loader = OMETIFFLoader(channel_format='CHW')
        loader.open(str(mock_ometiff_directory))
        
        acquisitions = loader.list_acquisitions()
        if acquisitions:
            channels = loader.get_channels(acquisitions[0].id)
            assert isinstance(channels, list)
            assert len(channels) > 0


@pytest.mark.unit
@pytest.mark.requires_readimc
class TestMCDLoader:
    """Tests for MCDLoader."""
    
    def test_mcd_loader_init(self):
        """Test MCDLoader initialization."""
        from openimc.data.mcd_loader import MCDLoader
        
        loader = MCDLoader()
        assert loader.mcd is None
    
    def test_mcd_loader_no_readimc(self, monkeypatch):
        """Test MCDLoader raises error when readimc is not available."""
        import sys
        from openimc.data import mcd_loader
        
        # Mock the import to fail
        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'readimc':
                raise ImportError("readimc not available")
            return original_import(name, *args, **kwargs)
        
        # This test is tricky because the import happens at module level
        # We'll skip it for now and test the actual functionality when readimc is available
        pytest.skip("Requires mocking at module import level")


@pytest.mark.unit
class TestAcquisitionInfo:
    """Tests for AcquisitionInfo dataclass."""
    
    def test_acquisition_info_creation(self):
        """Test creating AcquisitionInfo."""
        from openimc.data.mcd_loader import AcquisitionInfo
        
        acq = AcquisitionInfo(
            id='test_id',
            name='Test Acquisition',
            well='A1',
            size=(100, 100),
            channels=['DAPI', 'CD45'],
            channel_metals=['191Ir', '89Y'],
            channel_labels=['DAPI', 'CD45'],
            metadata={},
            source_file='test.mcd'
        )
        
        assert acq.id == 'test_id'
        assert acq.name == 'Test Acquisition'
        assert acq.well == 'A1'
        assert acq.size == (100, 100)
        assert len(acq.channels) == 2

