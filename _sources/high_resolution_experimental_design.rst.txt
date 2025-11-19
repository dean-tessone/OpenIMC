High Resolution Experimental Design
===================================

The High Resolution Experimental Design tool helps determine optimal ablation energy for high-resolution IMC experiments by modeling intensity decay across multiple passes. This tool analyzes how signal intensity changes across consecutive ablation passes of the same region, enabling you to optimize experimental parameters and determine appropriate deconvolution settings.

Overview
--------

In high-resolution IMC experiments, multiple passes are often performed on the same tissue region to maximize signal recovery. The intensity of signal typically decays across passes, and understanding this decay pattern is critical for:

- **Determining optimal ablation energy**: Finding the energy level that maintains signal for the desired number of passes
- **Setting deconvolution parameters**: Determining the x0 parameter for Richardson-Lucy deconvolution
- **Experimental optimization**: Understanding how different energy levels affect signal retention

The tool models intensity decay using an inverse sigmoidal function and provides visualizations to help interpret the results.

Accessing the Tool
------------------

The Experimental Design tool is accessible through the High Resolution Deconvolution dialog:

1. Navigate to **Analysis → High Resolution Deconvolution…** in the menu bar
2. Click on the **Experimental Design** tab

The dialog contains two tabs:
- **ROI Selection**: Configure ROIs and analyze intensity decay
- **Experimental Design Plot**: View and export the analysis results

Parameters
----------

Channel Selection
~~~~~~~~~~~~~~~~~

- **Channel**: Select the channel to analyze for intensity decay
  - The tool sums all pixel intensities within each ROI for the selected channel
  - Each pixel represents 1μm²
  - DNA channels are often good choices for this analysis
  - The combo box automatically selects DNA channels if available

ROI Selection and Pass Ordering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Available ROIs**: List of all available acquisitions/ROIs in your loaded data
  - Use the search box to filter ROIs by name
  - Select one or more ROIs to add to the analysis
  - Each ROI should represent a different pass of the same tissue region

- **Selected ROIs Table**: Table showing ROIs assigned to passes with the following columns:
  
  - **Pass #**: The pass number for this ROI (editable)
    - Defaults to sequential numbers (1, 2, 3, ...)
    - Can be manually edited to use non-sequential or decimal values
    - Allows fine control over pass ordering for better visualization
  
  - **ROI Name**: Display name of the acquisition/ROI
    - Shows well name, acquisition name, and source file if available
  
  - **Energy**: Ablation energy value for this ROI (editable)
    - Numeric value (can be negative)
    - Used to group ROIs by energy level for comparison
    - Multiple ROIs can share the same energy value
  
  - **Actions**: Remove button (×) to delete the ROI from the table

Working with the ROI Table
---------------------------

Adding ROIs
~~~~~~~~~~~

1. Select one or more ROIs from the **Available ROIs** list
2. Click **Add →** to add them to the **Selected ROIs** table
3. ROIs are automatically assigned sequential pass numbers starting from 1
4. You can manually edit pass numbers if needed

Removing ROIs
~~~~~~~~~~~~~

- Select one or more rows in the table and click **← Remove**
- Or click the **×** button in the Actions column for a specific row

Reordering ROIs
~~~~~~~~~~~~~~~

1. Select a row in the table
2. Use **↑ Move Up** or **↓ Move Down** buttons to change the order
3. Pass numbers are preserved when reordering (not auto-updated)
4. You can manually edit pass numbers after reordering if needed

Editing Pass Numbers
~~~~~~~~~~~~~~~~~~~~

- Click on any cell in the **Pass #** column to edit
- Pass numbers can be:
  - Sequential integers: 1, 2, 3, 4, ...
  - Non-sequential: 1, 3, 5, 7, ...
  - Decimal values: 1.0, 1.5, 2.0, 2.5, ...
- Decimal values are useful when passes don't align perfectly or for interpolation

Setting Energy Values
~~~~~~~~~~~~~~~~~~~~~

- Click on any cell in the **Energy** column to edit the energy value
- Multiple ROIs can have the same energy value
- The tool will group ROIs by energy and plot separate curves for each energy level
- This allows comparison of intensity decay patterns across different energy settings

Analysis
--------

Running the Analysis
~~~~~~~~~~~~~~~~~~~~

1. Ensure at least 2 ROIs are added to the table
2. Select the channel to analyze from the **Channel** dropdown
3. Click **Analyze Intensity Decay**

The analysis will:
- Load the selected channel image for each ROI
- Sum all pixel intensities within each ROI (each pixel = 1μm²)
- Group ROIs by energy value
- Fit an inverse sigmoidal decay curve for each energy group
- Generate visualization plots

Mathematical Model
~~~~~~~~~~~~~~~~~~

The tool fits an inverse sigmoidal decay function to the intensity data:

.. math::

   I(x) = \frac{I_{max}}{1 + e^{k(x - x_0)}}

Where:
- **I(x)**: Intensity at pass number x
- **I_max**: Maximum intensity (initial intensity)
- **k**: Decay rate parameter (positive values indicate decay)
- **x_0**: Inflection point (pass number where decay rate is maximum)

The derivative of this function shows the rate of change:

.. math::

   \frac{dI}{dx} = \frac{-I_{max} \cdot k \cdot e^{k(x-x_0)}}{(1 + e^{k(x-x_0)})^2}

Interpreting the Plots
----------------------

The Experimental Design Plot tab displays two side-by-side plots:

Intensity Decay Plot (Left)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **X-axis**: Pass number
- **Y-axis**: Total intensity (sum of all pixels in ROI)
- **Data points**: Scattered points showing measured intensities for each ROI
  - Color-coded by energy level
  - Labeled as "Energy=X.XX (data)"
- **Fitted curves**: Smooth lines showing the fitted inverse sigmoidal decay
  - Color-coded by energy level
  - Labeled as "Energy=X.XX (fit)"
  - Only shown when ≥3 data points are available for that energy group
- **Connecting lines**: Dashed lines connecting data points (for visual reference)

**What to look for:**
- **Steep decay**: Rapid signal loss indicates energy may be too high
- **Gradual decay**: Slow, controlled decay is ideal
- **Plateau**: Flat regions indicate signal is being retained
- **Comparison across energies**: Compare how different energy levels affect decay patterns

Derivative Plot (Right)
~~~~~~~~~~~~~~~~~~~~~~~

- **X-axis**: Pass number
- **Y-axis**: Rate of change (dI/dx)
- **Curves**: Derivative of the fitted curves for each energy level
- **Zero line**: Horizontal dashed line at y=0

**What to look for:**
- **Peak location**: The peak indicates where decay rate is maximum (inflection point)
- **Peak magnitude**: Higher peaks indicate faster decay
- **Negative values**: All values should be negative (indicating decay)
- **Comparison**: Compare peak locations and magnitudes across energy levels

Fit Parameters Display
~~~~~~~~~~~~~~~~~~~~~~

Below the plots, fit parameters are displayed for each energy level:

- **E=X.XX**: Energy value
- **I_max**: Maximum intensity estimate
- **k**: Decay rate parameter
- **x_0**: Inflection point (pass number where decay is fastest)
- **Sum|res|**: Sum of absolute residuals (measure of fit quality)

**Interpreting fit parameters:**
- **I_max**: Should be close to the first (highest) intensity value
- **k**: Higher values indicate faster decay
- **x_0**: The pass number where decay rate peaks; useful for determining optimal x0 for deconvolution
- **Sum|res|**: Lower values indicate better fit quality

Optimal Energy Criteria
-----------------------

Based on the analysis, optimal ablation energy should meet these criteria:

1. **At least 4 consecutive passes**: Signal should be retained for at least 4 consecutive passes without significant decay
2. **At least 7 total passes**: Signal should be retained across at least 7 total passes

These criteria ensure sufficient signal recovery for high-resolution IMC analysis.

Determining x0 for Deconvolution
---------------------------------

The x0 parameter for Richardson-Lucy deconvolution should be set to the **highest-numbered pass** for which signal is still retained (i.e., where intensity has not significantly decayed).

To determine x0:

1. Examine the intensity decay plot
2. Identify the pass number where intensity begins to significantly drop
3. Set x0 to this pass number (or slightly before)
4. Alternatively, use the x_0 value from the fit parameters as a starting point
5. Adjust based on your specific experimental needs

Example: If intensity is maintained through pass 7 and begins dropping at pass 8, set x0 = 7.0.

Exporting Results
-----------------

Export Plot
~~~~~~~~~~~

1. After running the analysis, click **Export Plot…**
2. Choose file format:
   - PNG (recommended for presentations)
   - PDF (recommended for publications)
   - SVG (vector format, scalable)
3. Select save location and filename
4. The plot is exported at 300 DPI for high-quality output

The exported plot includes both the intensity decay plot and the derivative plot, along with all fitted curves and data points.

Best Practices
--------------

1. **Select representative ROIs**: Choose ROIs that represent the same tissue region across different passes
2. **Use appropriate channels**: DNA channels often work well, but any channel of interest can be analyzed
3. **Include multiple energy levels**: Test different energy settings to compare their effects
4. **Ensure sufficient data points**: Include at least 3 ROIs per energy level for curve fitting
5. **Verify ROI alignment**: Ensure ROIs represent the same spatial region across passes
6. **Review fit quality**: Check the Sum|res| values to ensure good fit quality
7. **Consider experimental context**: Combine quantitative analysis with biological knowledge

Troubleshooting
---------------

No Fitted Curves Appearing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Ensure you have at least 3 ROIs for each energy level
- Check that pass numbers are correctly assigned
- Verify that intensity values are reasonable (not all zeros)

Poor Fit Quality
~~~~~~~~~~~~~~~~

- Check that ROIs represent the same tissue region
- Verify that pass numbers are correctly ordered
- Consider removing outliers if they don't represent the expected decay pattern
- Ensure sufficient data points (more points = better fit)

All Intensities Are Zero
~~~~~~~~~~~~~~~~~~~~~~~~

- Verify the selected channel exists in all ROIs
- Check that ROIs contain actual tissue (not background)
- Ensure images are properly loaded

Cannot Add ROI
~~~~~~~~~~~~~~

- Check that the ROI is not already in the table
- Verify that the acquisition is properly loaded
- Try refreshing the available ROIs list

Tips
----

- **Use decimal pass numbers**: If passes don't align perfectly, use decimal values (e.g., 1.0, 1.5, 2.0) for better visualization
- **Compare multiple channels**: Run the analysis for different channels to understand channel-specific decay patterns
- **Document energy settings**: Keep track of energy values used in your experiments for reproducibility
- **Iterative optimization**: Use the tool iteratively to refine energy settings before running full experiments
- **Combine with deconvolution**: Use the determined x0 value directly in the deconvolution tab

