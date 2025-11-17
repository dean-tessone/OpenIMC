Spillover Matrix Generation
===========================

Spillover matrix generation estimates spectral overlap between channels from single-stain control data, enabling compensation for spillover effects in IMC data.

Overview
--------

In IMC, spectral overlap occurs when the signal from one metal isotope is detected in other channels. Spillover matrices quantify these overlaps and are used to correct for them. OpenIMC generates spillover matrices from single-stain control MCD files, where each acquisition contains cells stained with a single metal isotope.

Options
-------

Spillover matrix generation:
- Analyzes pixel-level data from single-stain controls
- Automatically detects donor channels from acquisition/well names
- Estimates spillover coefficients between all channel pairs
- Provides QC metrics for matrix quality assessment

Parameters
----------

Input Parameters
~~~~~~~~~~~~~~~~~

- **mcd_path** (required): Path to MCD file containing single-stain controls
  - Each acquisition should contain a single-stain control for one channel
  - Multiple acquisitions per donor are supported (aggregated)

- **donor_label_per_acq** (optional): Manual mapping from acquisition ID/name to donor channel
  - If not provided, auto-detection is attempted based on well/acquisition names
  - Format: ``{"acquisition_id": "donor_channel_name"}``
  - Example: ``{"acq_1": "Yb176", "acq_2": "Er170"}``

Computation Parameters
~~~~~~~~~~~~~~~~~~~~~~

- **cap** (default: ``0.3``): Maximum spillover coefficient
  - Spillover coefficients above this value are capped
  - Prevents unrealistic spillover estimates
  - Typical range: 0.2-0.5

- **aggregate** (default: ``"median"``): Aggregation method when multiple acquisitions per donor
  - ``"median"``: Use median spillover estimate (more robust to outliers)
  - ``"mean"``: Use mean spillover estimate

- **p_low** (default: ``90.0``): Lower percentile for foreground pixel selection
  - Pixels above this percentile are considered foreground (signal)
  - Higher values (95-99) select only brightest pixels
  - Lower values (80-90) include more pixels

- **p_high_clip** (default: ``99.9``): Upper percentile for intensity clipping
  - Intensities above this percentile are clipped
  - Prevents extreme outliers from affecting estimates
  - Typical range: 99.0-99.9

- **channel_name_field** (default: ``"name"``): Field to use for channel names from MCD
  - ``"name"``: Use channel name field
  - ``"fullname"``: Use full channel name field

Using Spillover Matrix Generation in the GUI
----------------------------------------------

1. Prepare your single-stain control MCD file:
   - Each acquisition should contain cells stained with a single metal isotope
   - Well names or acquisition names should ideally indicate the donor channel (e.g., "Yb176", "Er170")

2. Navigate to **Analysis → Generate Spillover Matrix…** in the menu bar

3. In the spillover matrix dialog:
   - **Configuration Tab**:
     - Click **Browse…** to select your single-stain control MCD file
     - The tool will automatically detect acquisitions and attempt to map them to donor channels
     - Review the **Acquisition to Donor Channel Mapping** table
     - Use **Auto-detect Donors** to re-run auto-detection if needed
     - Manually adjust donor channel assignments in the dropdown menus if auto-detection is incorrect
     - Adjust computation parameters if needed:
       - **Max spillover**: Maximum allowed spillover coefficient (default: 0.3)
       - **Aggregation**: How to combine multiple acquisitions per donor (median or mean)
       - **Percentile (low)**: Lower percentile for foreground selection (default: 90.0)
       - **Percentile (high)**: Upper percentile for clipping (default: 99.9)
     - Click **Generate Matrix** to start computation
   
   - **Matrix Visualization Tab**:
     - Heatmap showing spillover coefficients
     - Diagonal = 1.0 (self-spillover)
     - Off-diagonal = spillover from donor (row) to receiver (column)
   
   - **Matrix Table Tab**:
     - Detailed table with all spillover coefficients
     - Color-coded: diagonal (green), off-diagonal > 0 (yellow/orange)
   
   - **QC Metrics Tab**:
     - Quality control metrics for each donor channel
     - Includes: number of acquisitions, off-diagonal sum, maximum spillover, pixels used

4. Save the matrix using **Save Matrix…** button (saves as CSV)

Using Spillover Matrix Generation in the CLI
---------------------------------------------

Basic Command
~~~~~~~~~~~~~

.. code-block:: bash

   openimc generate-spillover-matrix single_stain_controls.mcd spillover_matrix.csv

With Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc generate-spillover-matrix single_stain_controls.mcd spillover_matrix.csv \\
       --cap 0.3 \\
       --aggregate median \\
       --p-low 90.0 \\
       --p-high 99.9

With Manual Donor Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc generate-spillover-matrix single_stain_controls.mcd spillover_matrix.csv \\
       --donor-mapping '{"acq_1": "Yb176", "acq_2": "Er170"}'

Method Details
--------------

Spillover Coefficient Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spillover coefficients are estimated from pixel-level data using linear regression (slope estimation without intercept).

**Pixel-Level Method:**

For each donor channel :math:`i` and receiver channel :math:`j`, the spillover coefficient :math:`S_{ij}` is computed as:

1. **Foreground Selection**: Select foreground pixels from the donor channel
   - Pixels between :math:`p_{\text{low}}` and :math:`p_{\text{high}}` percentiles are selected
   - Intensities above :math:`p_{\text{high}}` are clipped to prevent outlier effects
   - This focuses on signal pixels where spillover is most apparent

2. **Slope Estimation**: Compute slope without intercept using linear regression
   - :math:`S_{ij} = \frac{\mathbf{x}_i^T \mathbf{y}_j}{\mathbf{x}_i^T \mathbf{x}_i}`
   
   Where:
   - :math:`\mathbf{x}_i` = Donor channel intensities for foreground pixels
   - :math:`\mathbf{y}_j` = Receiver channel intensities for the same foreground pixels

3. **Capping**: Apply maximum spillover cap
   - :math:`S_{ij} = \min(S_{ij}, \text{cap})`
   - Prevents unrealistic spillover estimates

4. **Aggregation**: When multiple acquisitions per donor exist
   - Use median (default) or mean of spillover estimates across acquisitions
   - Median is more robust to outliers

**Alternative Method (for cell-level data):**

For cell-level or event-level data, OpenIMC also supports the CATALYST-style ratio method:

1. **Background Subtraction**: Subtract background from both channels
   - Background estimated using trimmed mean (10% trim) of negative control events
   - :math:`\text{bg}_i = \text{trimmed\_mean}(\text{neg}_i, 0.10)`
   - :math:`\text{bg}_j = \text{trimmed\_mean}(\text{neg}_j, 0.10)`

2. **Signal Calculation**: Compute signal above background
   - :math:`\text{spiller} = \max(\text{pos}_i - \text{bg}_i, 0)`
   - :math:`\text{receiver} = \max(\text{pos}_j - \text{bg}_j, 0)`

3. **Ratio Calculation**: Compute event-wise ratios
   - :math:`\text{ratio} = \frac{\text{receiver}}{\max(\text{spiller}, 10^{-12})}`

4. **Coefficient Estimation**: Use median of ratios
   - :math:`S_{ij} = \text{median}(\text{ratio})`

**Spillover Candidate Selection:**

Only plausible spillover pairs are estimated:
- **M±1 amu**: Channels at mass ±1 atomic mass unit
- **M+16**: Oxide formation (mass +16)
- **Isotopic neighbors**: Other isotopes of the same metal (from isotope list)

This reduces computation and focuses on physically plausible spillover.

**Foreground Selection:**

For pixel-level analysis:
- Pixels above the :math:`p_{\text{low}}` percentile are considered foreground
- Intensities above the :math:`p_{\text{high}}` percentile are clipped
- This focuses on signal pixels and reduces outlier effects

**Citation:**
- Chevrier, S., et al. (2018). "Compensation of Signal Spillover in Suspension and Imaging Mass Cytometry." Cell Systems, 6(5), 612-620. `DOI: 10.1016/j.cels.2018.02.010 <https://doi.org/10.1016/j.cels.2018.02.010>`_
- Implementation based on CATALYST R package: `CATALYST <https://github.com/HelenaLC/CATALYST>`_

Matrix Structure
~~~~~~~~~~~~~~~~

The spillover matrix is a square matrix where:
- **Rows**: Donor channels (emitting signal)
- **Columns**: Receiver channels (receiving spillover)
- **Diagonal**: Self-spillover (always 1.0)
- **Off-diagonal**: Spillover from donor to receiver (0.0 to cap)

Example matrix structure:

.. code-block:: text

            Yb176  Er170  Dy164  ...
   Yb176    1.000  0.023  0.015  ...
   Er170    0.018  1.000  0.031  ...
   Dy164    0.012  0.027  1.000  ...
   ...

QC Metrics
~~~~~~~~~~

Quality control metrics are computed for each donor channel:

- **n_acqs**: Number of acquisitions used per donor
- **offdiag_sum**: Sum of off-diagonal spillover values (total spillover)
- **offdiag_max**: Maximum off-diagonal spillover value
- **pixels_used_median**: Median number of pixels used for estimation

These metrics help assess:
- **Matrix quality**: High off-diagonal sums may indicate issues
- **Estimation reliability**: More acquisitions and pixels improve reliability
- **Spillover magnitude**: Maximum spillover indicates worst-case overlap

Tips and Best Practices
-----------------------

1. **Single-Stain Controls**:
   - Use high-quality single-stain controls
   - Ensure each acquisition contains only one metal isotope
   - Use similar cell types and staining conditions as your experimental data

2. **Donor Mapping**:
   - Well names like "Yb176" or "Er170" enable auto-detection
   - Manually verify auto-detected mappings
   - Correct any mis-mapped acquisitions before generation

3. **Parameter Tuning**:
   - **cap**: Adjust based on expected spillover (typically 0.2-0.5)
   - **p_low**: Higher values (95-99) for cleaner signal, lower (80-90) for more pixels
   - **p_high**: Keep at 99.0-99.9 to clip extreme outliers

4. **Multiple Acquisitions**:
   - Multiple acquisitions per donor improve reliability
   - Use median aggregation (default) for robustness to outliers

5. **Validation**:
   - Check QC metrics for each donor
   - Verify that spillover values are physically plausible
   - Compare with known spillover patterns (e.g., M+1, M+16)

6. **Matrix Usage**:
   - Save the matrix as CSV for use in spillover correction
   - Use the same matrix for all data acquired with the same panel and instrument settings
   - Regenerate if instrument settings or panel change

7. **Troubleshooting**:
   - Low spillover estimates: Check foreground selection (p_low may be too high)
   - High spillover estimates: Check for contamination or mis-mapped donors
   - Missing entries: Verify donor mapping covers all channels

