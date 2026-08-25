Quality Control Analysis
=========================

Quality control (QC) analysis assesses image quality by computing Signal-to-Noise Ratio (SNR), Contrast-to-Noise Ratio (CNR), intensity statistics, and coverage metrics for each channel.

Overview
--------

QC analysis helps identify:
- **Low-quality channels**: Channels with poor signal-to-noise or contrast-to-noise ratios
- **Detection issues**: Channels with low coverage or detection rates
- **Image artifacts**: Unusual intensity distributions or patterns
- **Acquisition problems**: Systematic issues across acquisitions

Options
-------

QC analysis can be performed in two modes:

1. **Pixel-level QC**: Analyzes all pixels using Otsu thresholding to separate signal from background
2. **Cell-level QC**: Uses segmentation masks to separate cell pixels from background pixels

For cell-level QC, OpenIMC now supports multiple signal definitions so sparse markers are not diluted by marker-negative cells:

- **Positive pixels above background**: Default. Uses in-cell pixels above ``background_mean + N * robust_background_sd``
- **Upper quantile of cell intensity**: Uses the brightest cells only, based on the upper tail of per-cell mean intensities
- **All cell pixels (legacy)**: Uses every in-cell pixel as signal for backward-compatible whole-cell averaging

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

- **cell_signal_method** (cell mode only, default: ``"positive_pixels"``): Cell-mode signal definition
  - ``"positive_pixels"``: Estimate signal from in-cell pixels above a background-derived threshold
  - ``"upper_quantile"``: Estimate signal from the brightest cells only
  - ``"all_cell_mean"``: Legacy behavior using all in-cell pixels as signal

- **positive_threshold_sd** (cell mode only, default: ``2.0``): Number of robust background standard deviations used by ``positive_pixels``

- **upper_quantile** (cell mode only, default: ``0.90``): Quantile in ``(0, 1]`` used by ``upper_quantile``

Using Quality Control Analysis in the GUI
-------------------------------------------

1. Load your IMC data file (``.mcd`` or OME-TIFF directory)

2. Navigate to **Analysis → QC Analysis…** in the menu bar

3. In the QC analysis dialog, click **QC Settings...** to configure:
   - Which acquisitions to analyze
   - The analysis mode:
   
     - **Pixel-level**: Uses Otsu thresholding (no mask required)
     - **Cell-level**: Uses segmentation masks (requires masks to be loaded)

   - In cell-level mode, choose the **Cell Signal Definition**:

     - **Positive pixels above background**: Best default for sparse markers
     - **Upper quantile of cell intensity**: Focuses on the brightest cells
     - **All cell pixels (legacy)**: Preserves the older whole-cell average behavior
   
   - Optional denoising and worker settings
   - If using cell-level mode, ensure segmentation masks are available
   - Click **Done**, then click **Calculate QC Metrics** in the main QC window

4. Results are displayed in multiple tabs:
   - **QC Metrics Table**: Detailed metrics for each channel
   - **SNR / CNR vs Intensity**: Side-by-side plots showing each noise-normalized metric vs mean signal intensity
   - **Distribution Plots**: Boxplots showing distributions across ROIs

5. Export results using the **Export Results** button. OpenIMC writes both:

   - A pooled channel summary (the selected filename)
   - A companion ``*_per_roi.csv`` file containing every ROI-level metric used in the summary

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

Cell-level Command for Sparse Markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc qc-analysis input.mcd output.csv \\
       --mode cell \\
       --mask segmentation_masks/roi1.tif \\
       --cell-signal-method positive_pixels \\
       --positive-threshold-sd 2.0

Cell-level Command with Upper-Quantile Signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc qc-analysis input.mcd output.csv \\
       --mode cell \\
       --mask segmentation_masks/roi1.tif \\
       --cell-signal-method upper_quantile \\
       --upper-quantile 0.95

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

Signal-to-Noise and Contrast-to-Noise Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SNR and CNR answer different questions. SNR measures the magnitude of the
selected signal relative to background noise, while CNR measures how clearly
the selected signal is separated from the background mean. OpenIMC reports both
and states the equations explicitly because microscopy terminology is not
uniform across publications.

**SNR equation:**

.. math::

   \text{SNR} = \frac{\mu_{\text{signal}}}{\sigma_{\text{background}}}

**CNR equation:**

.. math::

   \text{CNR} = \frac{|\mu_{\text{signal}} - \mu_{\text{background}}|}{\sigma_{\text{background}}}

Where:
- :math:`\mu_{\text{signal}}` = Mean intensity of signal (foreground) pixels
- :math:`\mu_{\text{background}}` = Mean intensity of background pixels
- :math:`\sigma_{\text{background}}` = Standard deviation of background pixels

These are the linear forms of the Gaussian region metrics described for
`fluorescence microscopy <https://www.nature.com/articles/srep20640>`_, where
SNR uses the signal-region mean and CNR uses the signal-background difference,
both relative to the background-region SD.

**Robust denominator:**

To prevent inflated SNR or CNR values when background standard deviation is extremely small, OpenIMC uses a minimum background standard deviation:

.. math::

   \sigma_{\text{min}} = \max(\sigma_{\text{background}}, 0.001 \times |\mu_{\text{background}}|, 0.0001 \times \text{range}, 10^{-6})

   \text{SNR} = \frac{\mu_{\text{signal}}}{\sigma_{\text{min}}}

   \text{CNR} = \frac{|\mu_{\text{signal}} - \mu_{\text{background}}|}{\sigma_{\text{min}}}

Where:
- :math:`\text{range}` = Image intensity range (max - min)

This ensures both metrics remain finite and bounded against unrealistically
small denominators in very uniform or very low-intensity backgrounds.

**Multiple-ROI summaries:**

OpenIMC pools signal and background pixel counts, means, and population
variances across ROIs before calculating channel-level SNR and CNR. It does not
average ROI-level ratios. Consequently, both metrics in an exported channel
summary can be reproduced directly from that row's ``signal_mean``,
``background_mean``, and ``background_std`` values. Use the companion
``*_per_roi.csv`` export to inspect ROI heterogeneity.

SNR can remain high when both foreground and background have an elevated
baseline; CNR removes that baseline and measures their separation. Conversely,
a dim channel with a stable background can have a useful CNR despite low raw
intensity. Review the two metrics together with ``signal_minus_background`` and
``background_std``. IMC hot pixels and speckles can increase background SD;
inspect per-ROI values and use the pre-QC denoising controls when artifacts are
present.

**Pixel-level Mode:**
- Uses Otsu thresholding to separate foreground (signal) from background
- Otsu method automatically determines optimal threshold
- Foreground: pixels above threshold
- Background: pixels at or below threshold

**Cell-level Mode:**
- Uses segmentation masks to separate cells from background
- Signal: selected in-cell pixels or cells, depending on ``cell_signal_method``
- Background: pixels outside cells (mask == 0)

**Positive-Pixel Threshold Mode:**

For sparse markers, signal is estimated from in-cell pixels above a background-derived cutoff:

.. math::

   \sigma_{\text{bg,robust}} = \max(\sigma_{\text{background}}, 0.001 \times |\mu_{\text{background}}|, 0.0001 \times \text{range}, 10^{-6})

   T = \mu_{\text{background}} + N \times \sigma_{\text{bg,robust}}

   \text{signal pixels} = \{x \in \text{cells} \mid x > T\}

This follows the IMC practice of background removal plus positive-pixel support for per-cell marker quantification, which is particularly useful for dim or sparse markers.

**Upper-Quantile Mode:**

For markers expected in only a subset of cells, signal can be estimated from the upper tail of the per-cell intensity distribution:

.. math::

   Q = \operatorname{quantile}(\text{cell mean intensities}, q)

   \text{signal cells} = \{c \mid \mu_c \ge Q\}

This keeps the background denominator unchanged while redefining signal from the biologically informative upper tail.

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
- **cnr**: Contrast-to-Noise Ratio (see equation above)
- **signal_minus_background**: Signed foreground-minus-background intensity difference (channel summary)
- **signal_to_background_ratio**: Foreground mean divided by background mean (channel summary)
- **coverage_pct**: Percentage of pixels covered by signal/cells
- **cell_density** (cell mode only): Number of cells per unit area
- **cell_signal_method** (cell mode only): Signal definition used for the reported SNR and CNR
- **signal_threshold** (positive-pixel mode only): Background-derived threshold used to select signal pixels
- **signal_quantile** (upper-quantile mode only): Quantile used to select high-signal cells
- **n_signal_pixels** (cell mode only): Number of signal pixels selected by the chosen cell signal method
- **n_signal_cells** (cell mode only): Number of cells contributing signal under the chosen method
- **signal_fraction** (cell mode only): Fraction of in-cell pixels classified as signal
- **signal_coverage_pct** (cell mode only): Percentage of image pixels contributing signal under the chosen method

**Percentile Metrics (pixel mode only):**
- **p1, p25, p75, p99**: 1st, 25th, 75th, and 99th percentiles

Tips and Best Practices
-----------------------

1. **Mode Selection**:
   - Use **pixel-level** mode for quick assessment without segmentation
   - Use **cell-level** mode for more biologically targeted SNR/CNR when segmentation is available
   - For sparse markers, prefer **positive_pixels** or **upper_quantile** over legacy all-cell averaging

2. **SNR and CNR Interpretation**:
   - Treat both configurable thresholds as study-specific references, not universal pass/fail cutoffs
   - Use SNR for signal magnitude relative to noise and CNR for signal-background separation
   - Compare like-for-like signal definitions, preprocessing settings, and tissue types
   - Review ``signal_minus_background`` and ``background_std`` before concluding that a visually bright channel is low quality
   - Investigate hot pixels, speckles, segmentation leakage, and ROI outliers in the per-ROI export

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
   - Set study-specific SNR and CNR thresholds for downstream analysis

6. **Validation**:
   - Check the side-by-side SNR/CNR vs Intensity plots for expected relationships
   - Investigate channels whose SNR and CNR disagree; an elevated background mean can increase SNR without improving contrast
   - Investigate outliers in either plot using the per-ROI export
