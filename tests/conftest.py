"""
Pytest configuration and shared fixtures for OpenIMC tests.
"""
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_2d():
    """Create a sample 2D image array for testing."""
    np.random.seed(42)
    return np.random.randint(0, 1000, size=(100, 100), dtype=np.uint16)


@pytest.fixture
def sample_image_3d():
    """Create a sample 3D image array (H, W, C) for testing."""
    np.random.seed(42)
    return np.random.randint(0, 1000, size=(100, 100, 5), dtype=np.uint16)


@pytest.fixture
def sample_image_stack_chw():
    """Create a sample image stack in CHW format (C, H, W)."""
    np.random.seed(42)
    return np.random.randint(0, 1000, size=(5, 100, 100), dtype=np.uint16)


@pytest.fixture
def sample_segmentation_mask():
    """Create a sample segmentation mask with labeled cells."""
    mask = np.zeros((100, 100), dtype=np.uint32)
    # Create 5 labeled regions
    mask[10:30, 10:30] = 1
    mask[40:60, 40:60] = 2
    mask[70:90, 20:40] = 3
    mask[20:40, 70:90] = 4
    mask[60:80, 60:80] = 5
    return mask


@pytest.fixture
def sample_channels():
    """Sample channel names for testing."""
    return ['DAPI', 'CD45', 'CD3', 'CD4', 'CD8']


@pytest.fixture
def sample_denoise_settings():
    """Sample denoise settings dictionary."""
    return {
        'hot': {
            'method': 'median3',
            'n_sd': 5.0
        },
        'speckle': {
            'method': 'gaussian',
            'sigma': 0.8
        },
        'background': {
            'method': 'white_tophat',
            'radius': 15
        }
    }


@pytest.fixture
def sample_acquisition_info():
    """Sample acquisition info for testing."""
    return {
        'id': 'test_acq_1',
        'name': 'Test Acquisition',
        'well': 'A1',
        'channels': ['DAPI', 'CD45', 'CD3', 'CD4', 'CD8'],
        'channel_metals': ['191Ir', '89Y', '141Pr', '142Nd', '143Nd'],
        'channel_labels': ['DAPI', 'CD45', 'CD3', 'CD4', 'CD8'],
        'metadata': {},
        'source_file': 'test.mcd'
    }


@pytest.fixture
def sample_feature_dataframe():
    """Create a sample feature dataframe for testing."""
    import pandas as pd
    np.random.seed(42)
    
    n_cells = 100
    data = {
        'cell_id': range(1, n_cells + 1),
        'label': range(1, n_cells + 1),
        'centroid_x': np.random.rand(n_cells) * 100,
        'centroid_y': np.random.rand(n_cells) * 100,
        'area_um2': np.random.rand(n_cells) * 1000 + 100,
        'perimeter_um': np.random.rand(n_cells) * 50 + 20,
        'mean_DAPI': np.random.rand(n_cells) * 1000,
        'mean_CD45': np.random.rand(n_cells) * 1000,
        'mean_CD3': np.random.rand(n_cells) * 1000,
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_mcd_file(tmp_path):
    """Create a mock MCD file structure (placeholder for actual MCD file tests)."""
    # This is a placeholder - actual MCD file testing would require real .mcd files
    # or more sophisticated mocking of the readimc library
    mcd_path = tmp_path / "test.mcd"
    mcd_path.touch()
    return mcd_path


@pytest.fixture
def mock_ometiff_file(tmp_path, sample_image_stack_chw):
    """Create a mock OME-TIFF file for testing."""
    import tifffile
    
    ometiff_path = tmp_path / "test.ome.tif"
    
    # Create OME metadata
    metadata = {
        'Channel': {'Name': ['DAPI', 'CD45', 'CD3', 'CD4', 'CD8']}
    }
    
    # Write OME-TIFF file
    tifffile.imwrite(
        str(ometiff_path),
        sample_image_stack_chw,
        metadata=metadata,
        ome=True,
        photometric='minisblack'
    )
    
    return ometiff_path


@pytest.fixture
def mock_ometiff_directory(tmp_path, sample_image_stack_chw):
    """Create a directory with mock OME-TIFF files."""
    import tifffile
    import shutil
    
    # Create directory
    ometiff_dir = tmp_path / "ometiff_dir"
    ometiff_dir.mkdir()
    
    # Create multiple test files
    for i in range(3):
        test_file = ometiff_dir / f"acquisition_{i}.ome.tif"
        metadata = {
            'Channel': {'Name': ['DAPI', 'CD45', 'CD3', 'CD4', 'CD8']}
        }
        tifffile.imwrite(
            str(test_file),
            sample_image_stack_chw,
            metadata=metadata,
            ome=True,
            photometric='minisblack'
        )
    
    return ometiff_dir

