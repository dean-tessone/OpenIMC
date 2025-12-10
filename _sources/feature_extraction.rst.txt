Feature Extraction
===================

Feature extraction computes quantitative measurements from segmented cells, including morphological features (shape, size) and intensity features (marker expression levels).

Overview
--------

After segmentation, feature extraction quantifies cell properties that will be used for downstream analysis such as clustering, phenotyping, and spatial analysis. Features are extracted per cell and stored in a CSV file with one row per cell.

Options
-------

OpenIMC extracts two main categories of features:

1. **Morphological Features**: Shape and size characteristics of cells
2. **Intensity Features**: Expression levels of each marker channel within each cell

Parameters
----------

Feature Selection
~~~~~~~~~~~~~~~~~

- **morphological** (default: ``true``): Extract morphological features
  - Includes: area, perimeter, eccentricity, solidity, extent, and more
  - See "Morphological Features" section below for complete list

- **intensity** (default: ``true``): Extract intensity features
  - Includes: mean, median, max, min, std for each channel
  - See "Intensity Features" section below for complete list

Preprocessing Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

- **arcsinh** (default: ``false``): Apply arcsinh transformation to intensity features
  - Helps normalize highly skewed intensity distributions
  - Recommended for IMC data with wide dynamic range
  - Applied to extracted features, not raw images

- **arcsinh_cofactor** (default: ``10.0``): Cofactor for arcsinh transformation
  - Lower values (5.0) compress high intensities more
  - Higher values (20.0) preserve more of the original distribution
  - Typical range: 5.0-20.0

- **denoise_settings** (optional): Dictionary with denoise settings per channel
  - Format: ``{"Channel1": {"hot": {"method": "median3"}, "speckle": {"method": "gaussian", "sigma": 0.8}}}``
  - Can also be a path to a JSON file
  - Applied to raw images before feature extraction

- **excluded_channels** (optional): Set of channel names to exclude from feature extraction
  - Useful for excluding channels that are not informative
  - Example: ``{"DAPI", "Background"}``

Spillover Correction
~~~~~~~~~~~~~~~~~~~~

- **spillover_correction** (optional): Configuration for spillover correction
  - **enabled** (default: ``false``): Enable spillover correction
  - **matrix_file** (required if enabled): Path to spillover matrix CSV file
  
    - Matrix format: rows and columns are channel names, values are spillover coefficients
  
  - **method** (default: ``"nnls"``): Correction method
    - ``"nnls"``: Non-negative least squares (recommended)
    - ``"pgd"``: Projected gradient descent

Other Parameters
~~~~~~~~~~~~~~~~

- **acquisition** (optional): Specific acquisition ID or name to process
  - If not specified, processes all acquisitions

- **mask** (required): Path to segmentation mask directory or single mask file
  - Masks are matched to acquisitions by filename
  - Supports ``.tif``, ``.tiff``, or ``.npy`` formats

Morphological Features
----------------------

The following morphological features are extracted for each cell:

- **area**: Cell area in pixels
- **perimeter**: Cell perimeter in pixels
- **eccentricity**: Eccentricity of the ellipse with same second moments (0=circle, 1=line)
- **solidity**: Ratio of cell area to convex hull area
- **extent**: Ratio of cell area to bounding box area
- **major_axis_length**: Length of major axis of fitted ellipse
- **minor_axis_length**: Length of minor axis of fitted ellipse
- **orientation**: Orientation of major axis in radians
- **centroid_x**: X coordinate of cell centroid
- **centroid_y**: Y coordinate of cell centroid
- **equivalent_diameter**: Diameter of circle with same area
- **euler_number**: Euler characteristic (topology measure)

Intensity Features
------------------

For each marker channel, the following intensity features are extracted:

- **{channel}_mean**: Mean intensity within the cell
- **{channel}_median**: Median intensity within the cell
- **{channel}_max**: Maximum intensity within the cell
- **{channel}_min**: Minimum intensity within the cell
- **{channel}_std**: Standard deviation of intensities within the cell
- **{channel}_sum**: Sum of all pixel intensities within the cell

Where ``{channel}`` is replaced by the actual channel name (e.g., ``CD3_1841_mean``).

Using Feature Extraction in the GUI
------------------------------------

1. Ensure segmentation has been completed and masks are available

2. Navigate to **Analysis → Feature Extraction** or click the feature extraction button

3. In the feature extraction dialog:
   - Select which acquisitions to process
   - Choose feature types (morphological, intensity, or both)
   - Configure preprocessing options:
   
     - Enable/disable arcsinh transformation
     - Set arcsinh cofactor
     - Configure denoising if needed
   
   - Optionally configure spillover correction
   - Select channels to exclude if any

4. Choose output location for the features CSV file

5. Click **Extract Features** to start the process

6. Progress is shown in a progress dialog

7. The resulting CSV file contains one row per cell with all extracted features

Using Feature Extraction in the CLI
-----------------------------------

Basic Command
~~~~~~~~~~~~~

.. code-block:: bash

   openimc extract-features input.mcd features.csv \\
       --mask segmentation_masks/ \\
       --morphological \\
       --intensity

With Arcsinh Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc extract-features input.mcd features.csv \\
       --mask segmentation_masks/ \\
       --arcsinh \\
       --arcsinh-cofactor 10.0

With Denoising
~~~~~~~~~~~~~~

.. code-block:: bash

   openimc extract-features input.mcd features.csv \\
       --mask segmentation_masks/ \\
       --denoise-settings denoise_config.json

With Spillover Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc extract-features input.mcd features.csv \\
       --mask segmentation_masks/ \\
       --spillover-matrix spillover_matrix.csv \\
       --spillover-method nnls

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   feature_extraction:
     enabled: true
     morphological: true
     intensity: true
     arcsinh: false
     arcsinh_cofactor: 10.0
     spillover_correction:
       enabled: false
       matrix_file: "spillover_matrix.csv"
       method: "nnls"

Method Details
--------------

Feature extraction uses scikit-image's ``regionprops`` and ``regionprops_table`` functions to compute morphological features from segmentation masks. Intensity features are computed by masking each channel image with the cell segmentation mask and computing statistics.

Morphological Feature Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Morphological features are computed using scikit-image's region properties:

1. Each cell in the segmentation mask is treated as a region
2. Region properties are computed using ``regionprops_table``
3. Properties include geometric measurements (area, perimeter, etc.) and shape descriptors (eccentricity, solidity, etc.)

**Citation:**
- scikit-image: van der Walt, S., et al. (2014). "scikit-image: image processing in Python." PeerJ, 2, e453. `DOI: 10.7717/peerj.453 <https://doi.org/10.7717/peerj.453>`_
- `scikit-image regionprops <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>`_

Intensity Feature Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intensity features are computed per channel:

1. For each channel, the image is masked with the cell segmentation mask
2. Pixel intensities within each cell are extracted
3. Statistical measures (mean, median, max, min, std, sum) are computed
4. Results are stored with channel name prefixes

Arcsinh Transformation
~~~~~~~~~~~~~~~~~~~~~~

The arcsinh (inverse hyperbolic sine) transformation is commonly used in cytometry to stabilize variance and normalize distributions:

.. math::

   \text{arcsinh}(x) = \ln(x + \sqrt{x^2 + 1})

When applied with a cofactor:

.. math::

   \text{arcsinh}(x / \text{cofactor})

This transformation:
- Compresses high-intensity values
- Expands low-intensity values
- Reduces the impact of outliers
- Makes data more suitable for downstream analysis

**Citation:**
- Bendall, S. C., et al. (2011). "Single-cell mass cytometry of differential immune and drug responses across a human hematopoietic continuum." Science, 332(6030), 687-696. `DOI: 10.1126/science.1198704 <https://doi.org/10.1126/science.1198704>`_

Spillover Correction
~~~~~~~~~~~~~~~~~~~~~

Spillover correction compensates for spectral overlap between channels in IMC data. OpenIMC supports two methods:

1. **Non-negative Least Squares (NNLS)**: Solves for corrected intensities that minimize error while ensuring non-negativity
2. **Projected Gradient Descent (PGD)**: Iterative optimization method

The spillover matrix should be provided as a CSV file where:
- Rows and columns are channel names
- Values are spillover coefficients (typically 0-1)
- Diagonal should be 1.0 (self-spillover)

**Citation:**
- Implementation based on standard spillover correction methods used in flow cytometry and mass cytometry
- For IMC-specific spillover correction, see: Chevrier, S., et al. (2018). "Compensation of Signal Spillover in Suspension and Imaging Mass Cytometry." Cell Systems, 6(5), 612-620. `DOI: 10.1016/j.cels.2018.02.010 <https://doi.org/10.1016/j.cels.2018.02.010>`_

Tips and Best Practices
-----------------------

1. **Feature Selection**: Extract both morphological and intensity features for comprehensive analysis. Morphological features are useful for cell type identification, while intensity features capture marker expression.

2. **Arcsinh Transformation**: Apply arcsinh transformation if your intensity distributions are highly skewed or have a wide dynamic range. This is especially important for IMC data.

3. **Denoising**: Apply denoising before feature extraction if your images are noisy. This can improve the accuracy of intensity features.

4. **Spillover Correction**: Use spillover correction if you have a spillover matrix. This is particularly important for channels with significant spectral overlap.

5. **Channel Exclusion**: Exclude channels that are not informative (e.g., background channels, DAPI if not used for analysis) to reduce feature dimensionality.

6. **Validation**: Check the extracted features CSV to ensure:
   - All expected cells are present
   - Feature values are reasonable (no extreme outliers)
   - Missing values are handled appropriately

7. **Memory Considerations**: For large datasets, process acquisitions separately or use the CLI with appropriate resource allocation.

