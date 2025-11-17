Pixel Correlation Analysis
============================

Pixel correlation analysis computes pairwise correlations between markers at the pixel level, helping identify co-expression patterns, spatial co-localization, and potential spillover effects.

Overview
--------

Pixel correlation analysis examines how marker intensities co-vary at the pixel level across the image. This can reveal:
- **Co-expression patterns**: Markers that are expressed together in the same cells
- **Spatial co-localization**: Markers that are spatially close
- **Potential spillover**: Unusually high correlations that may indicate spectral overlap
- **Biological relationships**: Functional associations between markers

Options
-------

Pixel correlation analysis can be performed:
- **Across entire ROI**: Analyzes all pixels in the image
- **Within cells only**: Restricts analysis to pixels within segmented cells (requires segmentation mask)

Parameters
----------

- **channels** (optional): List of channel names to analyze
  - If not specified, analyzes all channels
  - Can be used to focus on specific marker pairs

- **mask** (optional): Segmentation mask to restrict analysis to cell pixels
  - If provided, only pixels within cells (mask > 0) are analyzed
  - If not provided, all pixels in the image are analyzed
  - Must match image dimensions

- **multiple_testing_correction** (optional): Method for multiple testing correction
  - ``"bonferroni"``: Bonferroni correction (conservative)
  - ``"fdr_bh"``: Benjamini-Hochberg FDR correction (less conservative)
  - If not specified, no correction is applied
  - Recommended when analyzing many channel pairs

Using Pixel Correlation Analysis in the GUI
--------------------------------------------

1. Load your IMC data file (``.mcd`` or OME-TIFF directory)

2. Navigate to **Analysis → Pixel-Level Correlation…** in the menu bar

3. In the pixel correlation dialog:
   - Select which acquisitions to analyze
   - Optionally select specific channels (or analyze all channels)
   - Optionally load a segmentation mask to restrict analysis to cell pixels
   - Choose multiple testing correction method if desired
   - Click **Run Analysis** to start the process

4. Results are displayed as:
   - **Correlation matrix**: Heatmap showing pairwise correlations
   - **Correlation table**: Detailed table with correlation coefficients and p-values
   - **Condition comparison**: If multiple conditions are present, comparisons across conditions

5. Export results using the **Export Results** button

Using Pixel Correlation Analysis in the CLI
--------------------------------------------

Basic Command
~~~~~~~~~~~~~

.. code-block:: bash

   openimc pixel-correlation input.mcd output.csv

With Specific Channels
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc pixel-correlation input.mcd output.csv \\
       --channels CD3_1841,CD4_2293,CD8_1941

With Segmentation Mask
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc pixel-correlation input.mcd output.csv \\
       --mask segmentation_masks/mask.tif

With Multiple Testing Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc pixel-correlation input.mcd output.csv \\
       --multiple-testing-correction fdr_bh

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   pixel_correlation:
     enabled: true
     # input: "path/to/input.mcd"  # Optional: uses previous step output if not specified
     # output: "pixel_correlation.csv"  # Optional: override default output location
     # mask: "path/to/masks/"  # Optional: restrict to cell pixels

Method Details
--------------

Correlation Computation
~~~~~~~~~~~~~~~~~~~~~~~

Pixel correlation analysis uses **Spearman rank correlation** to measure the relationship between marker intensities at the pixel level.

**Spearman Correlation:**
- Non-parametric measure of monotonic relationship
- Ranks pixel intensities rather than using raw values
- Robust to outliers and non-linear relationships
- Range: -1 to +1
  - +1: Perfect positive correlation
  - 0: No correlation
  - -1: Perfect negative correlation

**How it works:**

1. **Pixel Extraction**: For each channel, extract pixel intensities
   - If mask provided: Only pixels within cells (mask > 0)
   - If no mask: All pixels in the image

2. **Data Cleaning**: Remove NaN and infinite values

3. **Pairwise Correlation**: For each pair of channels:
   - Compute Spearman rank correlation coefficient
   - Compute p-value for significance testing
   - Record number of pixels used

4. **Multiple Testing Correction** (optional): Apply correction to p-values if many channel pairs are tested

**Interpretation:**
- **High positive correlation (>0.7)**: Markers are co-expressed or co-localized
- **High negative correlation (<-0.7)**: Markers are mutually exclusive
- **Low correlation (near 0)**: Markers are independent
- **Significant p-value**: Correlation is statistically significant

**Citation:**
- Spearman, C. (1904). "The proof and measurement of association between two things." American Journal of Psychology, 15(1), 72-101. `DOI: 10.2307/1412159 <https://doi.org/10.2307/1412159>`_
- Implementation: `scipy.stats.spearmanr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html>`_

Tips and Best Practices
-----------------------

1. **Mask Usage**: 
   - Use segmentation masks to focus on cell pixels and reduce background noise
   - Without masks, background pixels may dilute correlations

2. **Channel Selection**: 
   - Analyze all channels for comprehensive overview
   - Focus on specific channels to investigate particular relationships

3. **Multiple Testing Correction**: 
   - Always use correction when analyzing many channel pairs
   - FDR (Benjamini-Hochberg) is less conservative than Bonferroni

4. **Interpretation**: 
   - High correlations may indicate:
     - True co-expression (biological relationship)
     - Spillover (spectral overlap - check spillover matrix)
     - Spatial proximity (markers in same cell types)
   - Validate findings with biological knowledge

5. **Validation**: 
   - Compare correlations across different ROIs or conditions
   - Check for consistency in expected relationships
   - Use spatial visualization to confirm co-localization

