Quality Control Analysis
=========================

Quality control (QC) analysis assesses image quality by computing metrics such as Signal-to-Noise Ratio (SNR), intensity statistics, and coverage metrics for each channel.

Overview
--------

QC analysis helps identify:
- **Low-quality channels**: Channels with poor signal-to-noise ratios
- **Detection issues**: Channels with low coverage or detection rates
- **Image artifacts**: Unusual intensity distributions or patterns
- **Acquisition problems**: Systematic issues across acquisitions

Options
-------

QC analysis can be performed in two modes:

1. **Pixel-level QC**: Analyzes all pixels using Otsu thresholding to separate signal from background
2. **Cell-level QC**: Uses segmentation masks to separate cell pixels from background pixels

Parameters
----------

- **mode** (default: ``"pixel"``): Analysis mode
  - ``"pixel"``: Pixel-level analysis using Otsu thresholding
  - ``"cell"``: Cell-level analysis using segmentation masks (requires mask)

- **channels** (optional): List of channel names to analyze
  - If not specified, analyzes all channels

- **mask** (required for cell mode): Segmentation mask for cell-level analysis
  - Must match image dimensions
  - Pixels with mask > 0 are considered cells
  - Pixels with mask == 0 are considered background

Using Quality Control Analysis in the GUI
-------------------------------------------

1. Load your IMC data file (``.mcd`` or OME-TIFF directory)

2. Navigate to **Analysis → QC Analysis…** in the menu bar

3. In the QC analysis dialog:
   - Select which acquisitions to analyze
   - Choose analysis mode:
     - **Pixel-level**: Uses Otsu thresholding (no mask required)
     - **Cell-level**: Uses segmentation masks (requires masks to be loaded)
   - Optionally select specific channels (or analyze all channels)
   - If using cell-level mode, ensure segmentation masks are available
   - Click **Run Analysis** to start the process

4. Results are displayed in multiple tabs:
   - **QC Metrics Table**: Detailed metrics for each channel
   - **SNR vs Intensity**: Scatter plot showing SNR vs mean intensity
   - **Distribution Plots**: Boxplots showing distributions across ROIs

5. Export results using the **Export Results** button

Using Quality Control Analysis in the CLI
-------------------------------------------

Basic Command (Pixel-level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc qc-analysis input.mcd output.csv \\
       --mode pixel

Cell-level Command
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc qc-analysis input.mcd output.csv \\
       --mode cell \\
       --mask segmentation_masks/

With Specific Channels
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc qc-analysis input.mcd output.csv \\
       --mode pixel \\
       --channels CD3_1841,CD4_2293

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   qc_analysis:
     enabled: true
     # input: "path/to/input.mcd"  # Optional: uses previous step output if not specified
     # output: "qc_analysis.csv"  # Optional: override default output location
     # mask: "path/to/masks/"  # Optional: for cell-level QC
     mode: "pixel"  # or "cell"

Method Details
--------------

Signal-to-Noise Ratio (SNR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SNR measures the strength of the signal relative to background noise. Higher SNR indicates better image quality.

**SNR Equation:**

.. math::

   \text{SNR} = \frac{\mu_{\text{signal}} - \mu_{\text{background}}}{\sigma_{\text{background}}}

Where:
- :math:`\mu_{\text{signal}}` = Mean intensity of signal (foreground) pixels
- :math:`\mu_{\text{background}}` = Mean intensity of background pixels
- :math:`\sigma_{\text{background}}` = Standard deviation of background pixels

**Robust SNR Calculation:**

To prevent inflated SNR values when background standard deviation is extremely small, OpenIMC uses a minimum background standard deviation:

.. math::

   \sigma_{\text{min}} = \max(\sigma_{\text{background}}, 0.001 \times |\mu_{\text{background}}|, 0.0001 \times \text{range}, 10^{-6})

   \text{SNR} = \frac{\mu_{\text{signal}} - \mu_{\text{background}}}{\sigma_{\text{min}}}

Where:
- :math:`\text{range}` = Image intensity range (max - min)

This ensures SNR values remain reasonable even for very uniform backgrounds or very low-intensity channels.

**Pixel-level Mode:**
- Uses Otsu thresholding to separate foreground (signal) from background
- Otsu method automatically determines optimal threshold
- Foreground: pixels above threshold
- Background: pixels at or below threshold

**Cell-level Mode:**
- Uses segmentation masks to separate cells from background
- Signal: pixels within cells (mask > 0)
- Background: pixels outside cells (mask == 0)

**Citation:**
- Otsu, N. (1979). "A threshold selection method from gray-level histograms." IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66. `DOI: 10.1109/TSMC.1979.4310076 <https://doi.org/10.1109/TSMC.1979.4310076>`_
- Implementation: `scikit-image filters.threshold_otsu <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu>`_

QC Metrics
~~~~~~~~~~

The following metrics are computed for each channel:

**Intensity Metrics:**
- **mean_intensity**: Mean pixel intensity
- **median_intensity**: Median pixel intensity
- **max_intensity**: Maximum pixel intensity
- **min_intensity**: Minimum pixel intensity
- **std_intensity**: Standard deviation of intensities

**Signal Metrics (foreground/cells):**
- **signal_mean**: Mean intensity of signal pixels
- **signal_std**: Standard deviation of signal pixels

**Background Metrics:**
- **background_mean**: Mean intensity of background pixels
- **background_std**: Standard deviation of background pixels

**Quality Metrics:**
- **snr**: Signal-to-Noise Ratio (see equation above)
- **coverage_pct**: Percentage of pixels covered by signal/cells
- **cell_density** (cell mode only): Number of cells per unit area

**Percentile Metrics (pixel mode only):**
- **p1, p25, p75, p99**: 1st, 25th, 75th, and 99th percentiles

Tips and Best Practices
-----------------------

1. **Mode Selection**:
   - Use **pixel-level** mode for quick assessment without segmentation
   - Use **cell-level** mode for more accurate SNR when segmentation is available

2. **SNR Interpretation**:
   - **SNR > 10**: Excellent signal quality
   - **SNR 5-10**: Good signal quality
   - **SNR 2-5**: Acceptable but may need optimization
   - **SNR < 2**: Poor signal quality, consider excluding or optimizing

3. **Coverage Interpretation**:
   - Low coverage may indicate:
     - Sparse marker expression
     - Poor staining
     - Threshold issues (pixel mode)
   - High coverage may indicate:
     - Ubiquitous marker expression
     - Background contamination

4. **Cross-ROI Comparison**:
   - Compare metrics across ROIs to identify systematic issues
   - Look for consistent patterns vs. outliers

5. **Channel Filtering**:
   - Use QC metrics to identify and exclude low-quality channels
   - Set minimum SNR thresholds for downstream analysis

6. **Validation**:
   - Check SNR vs Intensity plot for expected relationships
   - High-intensity channels should generally have higher SNR
   - Investigate outliers in the SNR vs Intensity plot

