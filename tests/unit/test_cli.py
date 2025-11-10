"""
Unit tests for CLI functions.
"""
import pytest
import json
import tempfile
from pathlib import Path

from openimc.cli import load_data, parse_denoise_settings


@pytest.mark.unit
class TestLoadData:
    """Tests for load_data CLI function."""
    
    def test_load_data_invalid_path(self):
        """Test load_data with invalid path raises error."""
        with pytest.raises(ValueError, match="Input path must be"):
            load_data("/nonexistent/path")
    
    def test_load_data_directory(self, mock_ometiff_directory):
        """Test load_data with OME-TIFF directory."""
        loader, loader_type = load_data(str(mock_ometiff_directory))
        
        assert loader_type == 'ometiff'
        assert loader is not None
    
    def test_load_data_channel_format(self, mock_ometiff_directory):
        """Test load_data with custom channel format."""
        loader, loader_type = load_data(str(mock_ometiff_directory), channel_format='HWC')
        
        assert loader_type == 'ometiff'
        assert loader.channel_format == 'HWC'


@pytest.mark.unit
class TestParseDenoiseSettings:
    """Tests for parse_denoise_settings CLI function."""
    
    def test_parse_denoise_settings_none(self):
        """Test parse_denoise_settings with None."""
        result = parse_denoise_settings(None)
        assert result == {}
    
    def test_parse_denoise_settings_empty_string(self):
        """Test parse_denoise_settings with empty string."""
        result = parse_denoise_settings("")
        assert result == {}
    
    def test_parse_denoise_settings_json_string(self):
        """Test parse_denoise_settings with JSON string."""
        json_str = '{"DAPI": {"hot": {"method": "median3"}}}'
        result = parse_denoise_settings(json_str)
        
        assert isinstance(result, dict)
        assert "DAPI" in result
    
    def test_parse_denoise_settings_json_file(self, temp_dir):
        """Test parse_denoise_settings with JSON file."""
        settings = {
            "DAPI": {
                "hot": {"method": "median3"},
                "speckle": {"method": "gaussian", "sigma": 0.8}
            }
        }
        
        json_file = temp_dir / "denoise_settings.json"
        with open(json_file, 'w') as f:
            json.dump(settings, f)
        
        result = parse_denoise_settings(str(json_file))
        
        assert isinstance(result, dict)
        assert "DAPI" in result
        assert "hot" in result["DAPI"]
    
    def test_parse_denoise_settings_invalid_json(self):
        """Test parse_denoise_settings with invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_denoise_settings("{invalid json}")

