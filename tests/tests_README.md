# OpenIMC Test Suite

This directory contains the test suite for OpenIMC, including unit tests and integration tests.

## Installation

First, install the testing dependencies:

```bash
pip install -r requirements.txt
```

Or install just the testing packages:

```bash
pip install pytest>=7.4.0 pytest-cov>=4.1.0 pytest-mock>=3.11.0 pytest-qt>=4.2.0
```

## Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Unit tests for individual functions
│   ├── test_normalization.py
│   ├── test_denoising.py
│   ├── test_data_loaders.py
│   ├── test_feature_extraction.py
│   ├── test_watershed.py
│   └── test_cli.py
├── integration/             # Integration tests for workflows
│   └── test_cli_workflow.py
└── fixtures/                # Test data and fixtures
```

## Quick Start

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run tests with coverage
```bash
pytest --cov=openimc --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_normalization.py
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Skip slow tests
pytest -m "not slow"
```

### Run tests with markers
```bash
# Run only fast unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run tests that require GPU (if available)
pytest -m requires_gpu
```

## Test Organization

### Unit Tests (`tests/unit/`)
- `test_normalization.py`: Tests for normalization functions (arcsinh, percentile_clip, etc.)
- `test_denoising.py`: Tests for denoising functions
- `test_data_loaders.py`: Tests for MCDLoader and OMETIFFLoader
- `test_feature_extraction.py`: Tests for feature extraction
- `test_watershed.py`: Tests for watershed segmentation
- `test_cli.py`: Tests for CLI utility functions

### Integration Tests (`tests/integration/`)
- `test_cli_workflow.py`: End-to-end CLI workflow tests

## Test Markers

Tests are marked with the following markers:

- `unit`: Unit tests for individual functions
- `integration`: Integration tests for workflows
- `slow`: Tests that take a long time to run
- `requires_gpu`: Tests that require GPU
- `requires_cellpose`: Tests that require Cellpose
- `requires_cellsam`: Tests that require CellSAM
- `requires_readimc`: Tests that require readimc library
- `ui`: Tests that require UI/Qt
- `data`: Tests that require test data files

## Writing Tests

### Unit Tests

Unit tests should:
- Test individual functions in isolation
- Use fixtures from `conftest.py` for common test data
- Be fast and not require external dependencies when possible
- Be marked with `@pytest.mark.unit`

Example:
```python
@pytest.mark.unit
def test_arcsinh_normalize(sample_image_2d):
    result = arcsinh_normalize(sample_image_2d, cofactor=10.0)
    assert result.shape == sample_image_2d.shape
```

### Integration Tests

Integration tests should:
- Test complete workflows
- May require test data files
- Be marked with `@pytest.mark.integration` and `@pytest.mark.slow` if appropriate

### Adding New Tests

1. **Unit Tests**: Add to `tests/unit/test_<module>.py`
   - Test individual functions
   - Use fixtures from `conftest.py`
   - Mark with `@pytest.mark.unit`

2. **Integration Tests**: Add to `tests/integration/test_<workflow>.py`
   - Test complete workflows
   - Mark with `@pytest.mark.integration` and `@pytest.mark.slow` if appropriate

3. **Fixtures**: Add to `tests/conftest.py` if reusable across tests

### Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_image_2d`: 2D test image
- `sample_image_3d`: 3D test image (H, W, C)
- `sample_image_stack_chw`: Image stack in CHW format
- `sample_segmentation_mask`: Segmentation mask with labeled cells
- `sample_channels`: Sample channel names
- `sample_denoise_settings`: Sample denoise settings
- `sample_acquisition_info`: Sample acquisition info
- `sample_feature_dataframe`: Sample feature dataframe
- `temp_dir`: Temporary directory for test outputs
- `mock_ometiff_file`: Mock OME-TIFF file
- `mock_ometiff_directory`: Directory with mock OME-TIFF files

## Test Data

Test data is generated programmatically using fixtures in `conftest.py`. For tests requiring actual data files:

1. Place test data in `tests/fixtures/`
2. Document requirements in test docstrings
3. Mark tests with `@pytest.mark.data`

For tests requiring actual MCD files or large datasets, consider:

1. Using small sample files
2. Mocking data loaders
3. Generating synthetic data
4. Marking tests as `@pytest.mark.data` and documenting requirements

## Common Issues

### Import Errors
If you see import errors, make sure you're running tests from the project root:
```bash
cd /path/to/OpenIMC
pytest
```

### Missing Dependencies
Some tests may be skipped if optional dependencies are not installed:
- `requires_readimc`: Requires `readimc` package
- `requires_cellpose`: Requires `cellpose` package
- `requires_cellsam`: Requires `cellSAM` package
- `requires_gpu`: Requires GPU and CUDA

### Qt/UI Tests
UI tests require PyQt5 and may need a display. Use `pytest-qt` for Qt-specific testing.

## Continuous Integration

The test suite is designed to work in CI/CD environments. Some tests will be automatically skipped if dependencies are unavailable.

## Coverage

Aim for high test coverage of core functionality:
- Normalization utilities
- Data loaders
- Processing functions
- CLI commands

UI components may have lower coverage due to Qt dependencies.

