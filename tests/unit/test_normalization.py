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
Unit tests for normalization utilities.
"""
import numpy as np
import pytest

from openimc.ui.utils import (
    arcsinh_normalize,
    percentile_clip_normalize,
    channelwise_minmax_normalize,
    combine_channels,
    robust_percentile_scale
)


@pytest.mark.unit
class TestArcsinhNormalize:
    """Tests for arcsinh normalization."""
    
    def test_arcsinh_basic(self, sample_image_2d):
        """Test basic arcsinh normalization."""
        result = arcsinh_normalize(sample_image_2d, cofactor=10.0)
        
        assert result.shape == sample_image_2d.shape
        assert result.dtype == np.float32
        # Arcsinh should preserve non-negativity for non-negative inputs
        assert np.all(result >= 0)
    
    def test_arcsinh_cofactor(self, sample_image_2d):
        """Test arcsinh normalization with different cofactors."""
        result1 = arcsinh_normalize(sample_image_2d, cofactor=5.0)
        result2 = arcsinh_normalize(sample_image_2d, cofactor=10.0)
        
        # Smaller cofactor should produce larger values
        assert np.all(result1 >= result2)
    
    def test_arcsinh_zero_input(self):
        """Test arcsinh normalization with zero input."""
        zero_img = np.zeros((10, 10), dtype=np.uint16)
        result = arcsinh_normalize(zero_img, cofactor=10.0)
        
        assert np.all(result == 0)
    
    def test_arcsinh_high_values(self):
        """Test arcsinh normalization with high intensity values."""
        high_img = np.full((10, 10), 10000, dtype=np.uint16)
        result = arcsinh_normalize(high_img, cofactor=10.0)
        
        assert result.shape == high_img.shape
        assert np.all(result > 0)


@pytest.mark.unit
class TestPercentileClipNormalize:
    """Tests for percentile clip normalization."""
    
    def test_percentile_clip_basic(self, sample_image_2d):
        """Test basic percentile clip normalization."""
        result = percentile_clip_normalize(sample_image_2d, p_low=1.0, p_high=99.0)
        
        assert result.shape == sample_image_2d.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)
    
    def test_percentile_clip_extreme_percentiles(self, sample_image_2d):
        """Test percentile clip with extreme percentiles."""
        result = percentile_clip_normalize(sample_image_2d, p_low=0.0, p_high=100.0)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)
    
    def test_percentile_clip_constant_image(self):
        """Test percentile clip with constant image."""
        constant_img = np.full((10, 10), 100, dtype=np.uint16)
        result = percentile_clip_normalize(constant_img, p_low=1.0, p_high=99.0)
        
        # Should return zeros for constant image
        assert np.all(result == 0)
    
    def test_percentile_clip_custom_percentiles(self, sample_image_2d):
        """Test percentile clip with custom percentiles."""
        result = percentile_clip_normalize(sample_image_2d, p_low=5.0, p_high=95.0)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)


@pytest.mark.unit
class TestChannelwiseMinMaxNormalize:
    """Tests for channelwise min-max normalization."""
    
    def test_channelwise_minmax_basic(self, sample_image_2d):
        """Test basic channelwise min-max normalization."""
        result = channelwise_minmax_normalize(sample_image_2d)
        
        assert result.shape == sample_image_2d.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)
        # Check that min is 0 and max is 1 (or very close)
        assert np.isclose(result.min(), 0.0, atol=1e-6) or result.min() == 0
        assert np.isclose(result.max(), 1.0, atol=1e-6) or result.max() == 1
    
    def test_channelwise_minmax_constant_image(self):
        """Test channelwise min-max with constant image."""
        constant_img = np.full((10, 10), 100, dtype=np.uint16)
        result = channelwise_minmax_normalize(constant_img)
        
        # Should return zeros for constant image
        assert np.all(result == 0)
    
    def test_channelwise_minmax_zero_image(self):
        """Test channelwise min-max with zero image."""
        zero_img = np.zeros((10, 10), dtype=np.uint16)
        result = channelwise_minmax_normalize(zero_img)
        
        assert np.all(result == 0)


@pytest.mark.unit
class TestCombineChannels:
    """Tests for channel combination functions."""
    
    def test_combine_single(self, sample_image_2d):
        """Test single channel combination."""
        images = [sample_image_2d]
        result = combine_channels(images, method='single')
        
        assert np.array_equal(result, sample_image_2d)
    
    def test_combine_mean(self, sample_image_2d):
        """Test mean channel combination."""
        images = [sample_image_2d, sample_image_2d * 2, sample_image_2d * 3]
        result = combine_channels(images, method='mean')
        
        expected = np.mean(np.stack(images, axis=0), axis=0)
        assert np.allclose(result, expected)
    
    def test_combine_max(self, sample_image_2d):
        """Test max channel combination."""
        images = [sample_image_2d, sample_image_2d * 2, sample_image_2d * 3]
        result = combine_channels(images, method='max')
        
        expected = np.max(np.stack(images, axis=0), axis=0)
        assert np.allclose(result, expected)
    
    def test_combine_weighted(self, sample_image_2d):
        """Test weighted channel combination."""
        images = [sample_image_2d, sample_image_2d * 2]
        weights = [0.7, 0.3]
        result = combine_channels(images, method='weighted', weights=weights)
        
        # Manual calculation
        expected = 0.7 * images[0] + 0.3 * images[1]
        assert np.allclose(result, expected)
    
    def test_combine_weighted_normalized(self, sample_image_2d):
        """Test that weights are normalized."""
        images = [sample_image_2d, sample_image_2d * 2]
        weights = [2.0, 1.0]  # Should be normalized to [0.67, 0.33]
        result = combine_channels(images, method='weighted', weights=weights)
        
        # Should still work correctly
        assert result.shape == sample_image_2d.shape
    
    def test_combine_pca1(self, sample_image_2d):
        """Test PCA1 channel combination."""
        images = [sample_image_2d, sample_image_2d * 2, sample_image_2d * 3]
        result = combine_channels(images, method='pca1')
        
        assert result.shape == sample_image_2d.shape
    
    def test_combine_empty_list(self):
        """Test combining empty list raises error."""
        with pytest.raises(ValueError, match="No images provided"):
            combine_channels([], method='mean')
    
    def test_combine_invalid_method(self, sample_image_2d):
        """Test invalid combination method raises error."""
        with pytest.raises(ValueError, match="Unknown combination method"):
            combine_channels([sample_image_2d], method='invalid')
    
    def test_combine_weighted_mismatch(self, sample_image_2d):
        """Test weighted combination with mismatched weights raises error."""
        images = [sample_image_2d, sample_image_2d * 2]
        weights = [0.5]  # Wrong length
        
        with pytest.raises(ValueError, match="Weights must be provided"):
            combine_channels(images, method='weighted', weights=weights)


@pytest.mark.unit
class TestRobustPercentileScale:
    """Tests for robust percentile scaling."""
    
    def test_robust_percentile_scale_basic(self, sample_image_2d):
        """Test basic robust percentile scaling."""
        result = robust_percentile_scale(sample_image_2d, low=1.0, high=99.0)
        
        assert result.shape == sample_image_2d.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0)
        assert np.all(result <= 1.0)
    
    def test_robust_percentile_scale_constant_image(self):
        """Test robust percentile scale with constant image."""
        constant_img = np.full((10, 10), 100, dtype=np.uint16)
        result = robust_percentile_scale(constant_img, low=1.0, high=99.0)
        
        # Should handle constant image gracefully
        assert result.shape == constant_img.shape

