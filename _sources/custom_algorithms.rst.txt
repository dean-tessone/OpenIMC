Custom Algorithms
=================

OpenIMC provides base classes that allow developers to easily integrate novel
segmentation, clustering, and feature extraction algorithms into the framework.
These base classes define clear interfaces with standardized input/output formats,
making it straightforward to add new methods while maintaining compatibility with
the existing OpenIMC pipeline.

Overview
--------

The base classes are located in ``openimc.processing.base`` and provide:

- **Clear interface definitions**: Standardized input/output formats
- **Input validation**: Automatic validation of inputs before processing
- **Output validation**: Automatic validation of outputs after processing
- **Documentation**: Comprehensive docstrings explaining expected formats
- **Error handling**: Consistent error messages and exception types

Base Classes
------------

BaseSegmenter
~~~~~~~~~~~~~

Abstract base class for segmentation algorithms.

**Location**: ``openimc.processing.base.BaseSegmenter``

**Expected Inputs**:

- ``nuclear_image``: ``np.ndarray``, shape ``(H, W)``, dtype ``float32``
  - Preprocessed nuclear channel image (0-1 normalized)
- ``cyto_image``: ``np.ndarray``, shape ``(H, W)``, dtype ``float32``, optional
  - Preprocessed cytoplasm channel image (0-1 normalized)
- ``**kwargs``: Additional algorithm-specific parameters

**Expected Output**:

- ``mask``: ``np.ndarray``, shape ``(H, W)``, dtype ``uint32``
  - Segmentation mask where each cell has a unique integer label
  - ``0`` = background, ``1+`` = cell labels

**Example Implementation**:

.. code-block:: python

    from openimc.processing.base import BaseSegmenter
    import numpy as np
    
    class MyCustomSegmenter(BaseSegmenter):
        def __init__(self):
            super().__init__(name="my_custom_segmenter")
        
        def segment(self, nuclear_image, cyto_image=None, **kwargs):
            # Validate inputs (optional, but recommended)
            self.validate_inputs(nuclear_image, cyto_image)
            
            # Your segmentation algorithm here
            # ...
            
            # Create mask (example: simple thresholding)
            threshold = kwargs.get('threshold', 0.5)
            mask = (nuclear_image > threshold).astype(np.uint32)
            
            # Validate output (optional, but recommended)
            self.validate_output(mask, nuclear_image.shape)
            
            return mask

BaseClusterer
~~~~~~~~~~~~~

Abstract base class for clustering algorithms.

**Location**: ``openimc.processing.base.BaseClusterer``

**Expected Inputs**:

- ``features_df``: ``pd.DataFrame``
  - Feature matrix with one row per cell and one column per feature
  - Required columns: None (all numeric columns are used)
  - Excluded columns: ``'cell_id'``, ``'acquisition_id'``, ``'acquisition_name'``, ``'well'``, ``'cluster'``, ``'label'``, ``'source_file'``, etc.
- ``columns``: ``List[str]``, optional
  - Specific feature columns to use for clustering
  - If ``None``, auto-detects all numeric columns
- ``**kwargs``: Additional algorithm-specific parameters

**Expected Output**:

- ``features_df``: ``pd.DataFrame``
  - Same DataFrame as input with ``'cluster'`` column added
  - ``'cluster'`` column: ``int``, 1-based cluster labels (``0`` = unassigned/noise)

**Example Implementation**:

.. code-block:: python

    from openimc.processing.base import BaseClusterer
    import pandas as pd
    from sklearn.cluster import KMeans
    
    class MyCustomClusterer(BaseClusterer):
        def __init__(self):
            super().__init__(name="my_custom_clusterer")
        
        def cluster(self, features_df, columns=None, **kwargs):
            # Validate and prepare inputs
            data, column_names = self.validate_inputs(features_df, columns)
            original_shape = features_df.shape
            
            # Your clustering algorithm here
            n_clusters = kwargs.get('n_clusters', 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data.values)
            
            # Convert to 1-based labels
            cluster_labels = (cluster_labels + 1).astype(int)
            
            # Add cluster column
            result_df = features_df.copy()
            result_df['cluster'] = cluster_labels
            
            # Validate output
            self.validate_output(result_df, original_shape)
            
            return result_df

BaseFeatureExtractor
~~~~~~~~~~~~~~~~~~~~

Abstract base class for feature extraction algorithms.

**Location**: ``openimc.processing.base.BaseFeatureExtractor``

**Expected Inputs**:

- ``mask``: ``np.ndarray``, shape ``(H, W)``, dtype ``uint32``
  - Segmentation mask with cell labels (``0`` = background, ``1+`` = cells)
- ``image_stack``: ``np.ndarray``, shape ``(H, W, C)``, dtype ``float32``
  - Image stack with ``C`` channels
- ``channel_names``: ``List[str]``, length ``C``
  - Names of each channel in image_stack
- ``**kwargs``: Additional algorithm-specific parameters

**Expected Output**:

- ``features_df``: ``pd.DataFrame``
  - Feature matrix with one row per cell
  - Required columns:
  
    - ``'cell_id'``: ``int``, unique identifier for each cell (1-based)
    - ``'label'``: ``int``, cell label from mask (1-based)
  
  - Additional feature columns (algorithm-specific)

**Example Implementation**:

.. code-block:: python

    from openimc.processing.base import BaseFeatureExtractor
    import numpy as np
    import pandas as pd
    
    class MyCustomFeatureExtractor(BaseFeatureExtractor):
        def __init__(self):
            super().__init__(name="my_custom_extractor")
        
        def extract(self, mask, image_stack, channel_names, **kwargs):
            # Validate inputs
            self.validate_inputs(mask, image_stack, channel_names)
            
            # Get unique cell labels (exclude background = 0)
            unique_labels = np.unique(mask)
            unique_labels = unique_labels[unique_labels > 0]
            
            features_list = []
            for idx, label in enumerate(unique_labels):
                cell_id = idx + 1  # 1-based
                features = {'cell_id': cell_id, 'label': int(label)}
                
                # Create binary mask for this cell
                cell_mask = (mask == label)
                
                # Extract your custom features here
                # ...
                
                features_list.append(features)
            
            # Create DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Validate output
            expected_n_cells = len(unique_labels)
            self.validate_output(features_df, expected_n_cells)
            
            return features_df

Integration with OpenIMC
------------------------

Once you've implemented a custom algorithm, you can integrate it into OpenIMC
by modifying the core functions to support your new algorithm. Here's how:

Segmentation Integration
~~~~~~~~~~~~~~~~~~~~~~~~

Modify ``openimc.core.segment()`` to add your segmenter:

.. code-block:: python

    def segment(loader, acquisition, method, ...):
        # ... existing code ...
        
        elif method == 'my_custom_segmenter':
            from my_module import MyCustomSegmenter
            
            # Preprocess channels (same as other methods)
            nuclear_img, cyto_img = _preprocess_channels_for_segmentation(...)
            
            # Create segmenter instance
            segmenter = MyCustomSegmenter()
            
            # Run segmentation
            mask = segmenter.segment(
                nuclear_img,
                cyto_image=cyto_img,
                threshold=0.5,  # Your custom parameters
                min_cell_area=50
            )
        
        # ... rest of code ...

Clustering Integration
~~~~~~~~~~~~~~~~~~~~~~

Modify ``openimc.core.cluster()`` to add your clusterer:

.. code-block:: python

    def cluster(features_df, method='leiden', ...):
        # ... existing code ...
        
        elif method == 'my_custom_clusterer':
            from my_module import MyCustomClusterer
            
            # Create clusterer instance
            clusterer = MyCustomClusterer()
            
            # Run clustering
            result_df = clusterer.cluster(
                features_df,
                columns=columns,
                n_clusters=5,  # Your custom parameters
                seed=42
            )
            
            return result_df
        
        # ... rest of code ...

Feature Extraction Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify ``openimc.processing.feature_worker.extract_features_for_acquisition()``
to add your extractor:

.. code-block:: python

    def extract_features_for_acquisition(..., feature_extractor=None):
        # ... existing code ...
        
        if feature_extractor is not None:
            # Use custom extractor
            from my_module import MyCustomFeatureExtractor
            extractor = MyCustomFeatureExtractor()
            features_df = extractor.extract(
                mask,
                img_stack,
                channel_names,
                morphological=True,
                intensity=True
            )
        else:
            # Use default extractor
            # ... existing code ...

Example Implementations
------------------------

Complete example implementations are available in ``openimc.processing.examples``:

- ``ExampleThresholdSegmenter``: Simple thresholding-based segmentation
- ``ExampleKMeansClusterer``: K-means clustering implementation
- ``ExampleBasicFeatureExtractor``: Basic morphological and intensity features

These examples demonstrate:

- Proper input validation
- Correct output format
- Error handling
- Integration patterns

Best Practices
--------------

1. **Always validate inputs**: Use the ``validate_inputs()`` method before processing
2. **Always validate outputs**: Use the ``validate_output()`` method after processing
3. **Handle edge cases**: Empty masks, no cells, missing channels, etc.
4. **Document parameters**: Clearly document all ``**kwargs`` parameters
5. **Preserve data types**: Ensure outputs match expected dtypes (uint32, float32, etc.)
6. **Use 1-based indexing**: Cell IDs and labels should start at 1 (0 = background/unassigned)
7. **Handle memory efficiently**: For large datasets, process in chunks if needed
8. **Provide informative errors**: Raise ``ValueError`` with clear messages for invalid inputs

Common Pitfalls
---------------

1. **Wrong dtype**: Segmentation masks must be ``uint32``, not ``uint8`` or ``int32``
2. **Wrong indexing**: Cell IDs and labels must be 1-based (1, 2, 3, ...), not 0-based
3. **Missing required columns**: Feature DataFrames must have ``'cell_id'`` and ``'label'`` columns
4. **Shape mismatches**: Output shapes must match input shapes
5. **NaN values**: Cluster labels cannot contain NaN (use 0 for unassigned cells)
6. **Background handling**: Background pixels should be labeled as 0 in masks

Testing Your Implementation
----------------------------

Before integrating your algorithm, test it with the base class validation:

.. code-block:: python

    import numpy as np
    from my_module import MyCustomSegmenter
    
    # Create test data
    nuclear_img = np.random.rand(100, 100).astype(np.float32)
    
    # Test segmenter
    segmenter = MyCustomSegmenter()
    mask = segmenter.segment(nuclear_img, threshold=0.5)
    
    # Validation is automatic, but you can also check:
    assert mask.dtype == np.uint32
    assert mask.shape == nuclear_img.shape
    assert mask.min() >= 0

For more complex testing, use the OpenIMC test fixtures and integration tests.

