Advanced Spatial Analysis
==========================

Advanced Spatial Analysis provides sophisticated spatial analysis methods using squidpy, including neighborhood enrichment, co-occurrence analysis, spatial autocorrelation, and Ripley functions.

Overview
--------

Advanced Spatial Analysis extends Simple Spatial Analysis with additional methods from the squidpy package, enabling more sophisticated spatial pattern analysis, statistical testing, and spatial statistics.

.. note::
   Advanced Spatial Analysis requires the ``squidpy`` package to be installed. Install with: ``pip install squidpy``

Options
-------

Advanced Spatial Analysis includes:

1. **Neighborhood Enrichment**: Analyze enrichment of cell types in neighborhoods
2. **Co-occurrence Analysis**: Test for spatial co-occurrence patterns
3. **Spatial Autocorrelation**: Measure spatial correlation of features
4. **Ripley Functions**: Analyze spatial point patterns (K and L functions)
5. **Additional Spatial Statistics**: Various spatial metrics and tests

Parameters
----------

Graph Construction
~~~~~~~~~~~~~~~~~~

Same as Simple Spatial Analysis:
- **method**: kNN, Radius, or Delaunay
- **k_neighbors**: Number of neighbors for kNN
- **radius**: Maximum distance for Radius method
- **pixel_size_um**: Pixel size in micrometers

Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~

- **n_permutations** (default: ``100``): Number of permutations for statistical testing
  - More permutations provide more accurate p-values
  - Typical range: 100-1000

- **interaction_threshold** (optional): Threshold for considering interactions significant
  - Used to filter results
  - Default: based on statistical significance

Co-occurrence Analysis
~~~~~~~~~~~~~~~~~~~~~~

- **reference_cluster** (optional): Reference cluster for one-vs-others analysis
  - If specified, compares reference cluster against all others
  - If not specified, performs pairwise comparisons

- **method** (default: ``"pairwise"``): Analysis method
  - ``"pairwise"``: Compare all cluster pairs
  - ``"one_vs_others"``: Compare reference cluster against all others

Spatial Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~

- **feature** (required): Feature column to analyze
  - Can be a marker expression or other numeric feature

- **method** (default: ``"moran"``): Autocorrelation method
  - ``"moran"``: Moran's I statistic
  - ``"geary"``: Geary's C statistic

- **n_permutations** (default: ``100``): Number of permutations for significance testing

Ripley Functions
~~~~~~~~~~~~~~~~

- **cluster_column** (required): Column name containing cluster assignments
  - Typically ``"cluster"``

- **mode** (default: ``"K"``): Ripley function type
  - ``"K"``: Ripley's K function
  - ``"L"``: Ripley's L function (normalized K function)

- **max_dist** (optional): Maximum distance to compute function
  - If not specified, uses a default based on data extent

- **roi_column** (optional): Column name for ROI grouping
  - If specified, computes Ripley functions per ROI

Using Advanced Spatial Analysis in the GUI
--------------------------------------------

1. Ensure clustering has been completed

2. Navigate to **Analysis → Spatial Analysis → Advanced Spatial Analysis**

3. In the advanced spatial analysis dialog:
   - **Build Spatial Graph** (same as Simple Spatial Analysis)
   
   - **Neighborhood Enrichment Tab**:
     - Set number of permutations
     - Click "Run Neighborhood Enrichment"
     - Results show enrichment scores and p-values
   
   - **Co-occurrence Analysis Tab**:
     - Select analysis method (pairwise or one-vs-others)
     - Optionally specify reference cluster
     - Click "Run Co-occurrence Analysis"
   
   - **Spatial Autocorrelation Tab**:
     - Select feature to analyze
     - Choose autocorrelation method
     - Set number of permutations
     - Click "Run Autocorrelation Analysis"
   
   - **Ripley Functions Tab**:
     - Select cluster column
     - Choose function type (K or L)
     - Set maximum distance
     - Click "Run Ripley Analysis"

4. Export results using the export buttons

Using Advanced Spatial Analysis in the CLI
-------------------------------------------

Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial-enrichment features.csv enrichment_results.csv \\
       --method kNN \\
       --k-neighbors 10 \\
       --n-permutations 500

Co-occurrence Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial-cooccurrence features.csv cooccurrence_results.csv \\
       --method pairwise \\
       --reference-cluster "Cluster_1"

Spatial Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial-autocorr features.csv autocorr_results.csv \\
       --feature CD3_1841_mean \\
       --method moran \\
       --n-permutations 500

Ripley Functions
~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial-ripley features.csv ripley_results.h5ad \\
       --cluster-column cluster \\
       --mode K \\
       --max-dist 100.0 \\
       --pixel-size-um 1.0

Method Details
--------------

Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~

Neighborhood enrichment analyzes whether cell types are enriched or depleted in the neighborhoods of other cell types.

**How it works:**

1. **Neighborhood Definition**: For each cell, define its neighborhood (spatially adjacent cells)

2. **Observed Composition**: Compute the composition of cell types in each cell's neighborhood

3. **Expected Composition**: Compute expected composition under random spatial distribution

4. **Enrichment Score**: Compare observed vs. expected composition
   - Positive score: Enrichment
   - Negative score: Depletion

5. **Statistical Testing**: Use permutation tests to assess significance

**Interpretation:**
- Enrichment: Cell type A is more common in neighborhoods of cell type B than expected
- Depletion: Cell type A is less common in neighborhoods of cell type B than expected

**Citation:**
- Based on methods in: Schapiro, D., et al. (2017). "histoCAT: analysis of cell phenotypes and interactions in multiplex image cytometry data." Nature Methods, 14(9), 873-876. `DOI: 10.1038/s41592-017-0001-x <https://doi.org/10.1038/s41592-017-0001-x>`_
- Implementation: `squidpy.gr.nhood_enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html>`_

Co-occurrence Analysis
~~~~~~~~~~~~~~~~~~~~~

Co-occurrence analysis tests whether cell types tend to appear together in spatial proximity.

**How it works:**

1. **Spatial Proximity**: Define spatial proximity based on spatial graph (kNN, Radius, etc.)

2. **Observed Co-occurrence**: Count how often cell type pairs appear in proximity

3. **Expected Co-occurrence**: Compute expected co-occurrence under random distribution

4. **Statistical Testing**: Use permutation tests to assess significance

**Pairwise Mode**: Compares all pairs of cell types

**One-vs-Others Mode**: Compares a reference cell type against all others

**Citation:**
- Implementation: `squidpy.gr.co_occurrence <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.co_occurrence.html>`_

Spatial Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~

Spatial autocorrelation measures how similar feature values are for spatially nearby cells.

**Moran's I**:
- Range: -1 to 1
- Positive values: Similar values cluster together (positive autocorrelation)
- Negative values: Dissimilar values cluster together (negative autocorrelation)
- Near 0: No spatial autocorrelation (random spatial distribution)

**Geary's C**:
- Range: 0 to 2
- Values < 1: Positive autocorrelation
- Values > 1: Negative autocorrelation
- Values near 1: No autocorrelation

**How it works:**

1. **Spatial Weights**: Define spatial weights matrix based on spatial graph

2. **Autocorrelation Statistic**: Compute Moran's I or Geary's C using spatial weights

3. **Statistical Testing**: Use permutation tests to assess significance

**Interpretation:**
- Positive autocorrelation: Feature values are spatially clustered
- Negative autocorrelation: Feature values are spatially dispersed
- Useful for identifying spatial gradients or domains

**Citation:**
- Moran, P. A. P. (1950). "Notes on continuous stochastic phenomena." Biometrika, 37(1/2), 17-23. `DOI: 10.2307/2332142 <https://doi.org/10.2307/2332142>`_
- Geary, R. C. (1954). "The contiguity ratio and statistical mapping." The Incorporated Statistician, 5(3), 115-145. `DOI: 10.2307/2986645 <https://doi.org/10.2307/2986645>`_
- Implementation: `squidpy.gr.spatial_autocorr <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_autocorr.html>`_

Ripley Functions
~~~~~~~~~~~~~~~~

Ripley functions analyze spatial point patterns to test for clustering or dispersion.

**Ripley's K Function**:
- Measures the expected number of points within distance r of a randomly chosen point
- Under complete spatial randomness (CSR): K(r) = πr²
- K(r) > πr²: Clustering
- K(r) < πr²: Dispersion

**Ripley's L Function**:
- Normalized version: L(r) = √(K(r)/π) - r
- Under CSR: L(r) = 0
- L(r) > 0: Clustering
- L(r) < 0: Dispersion

**How it works:**

1. **Distance Calculation**: For each point, count neighbors within distance r

2. **Edge Correction**: Apply edge correction for points near ROI boundaries

3. **Function Computation**: Compute K(r) or L(r) for a range of distances

4. **Comparison to CSR**: Compare observed function to expected under complete spatial randomness

**Interpretation:**
- Clustering: Cell type is more clustered than random
- Dispersion: Cell type is more dispersed than random
- Useful for identifying spatial organization patterns

**Citation:**
- Ripley, B. D. (1976). "The second-order analysis of stationary point processes." Journal of Applied Probability, 13(2), 255-266. `DOI: 10.2307/3212829 <https://doi.org/10.2307/3212829>`_
- Ripley, B. D. (1977). "Modelling spatial patterns." Journal of the Royal Statistical Society: Series B, 39(2), 172-192. `DOI: 10.1111/j.2517-6161.1977.tb01615.x <https://doi.org/10.1111/j.2517-6161.1977.tb01615.x>`_
- Implementation: `squidpy.gr.ripley <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.ripley.html>`_

Squidpy Integration
~~~~~~~~~~~~~~~~~~

Advanced Spatial Analysis uses the squidpy package, which provides a comprehensive toolkit for spatial omics analysis.

**Citation:**
- Palla, G., et al. (2022). "Squidpy: a scalable framework for spatial omics analysis." Nature Methods, 19(2), 171-178. `DOI: 10.1038/s41592-021-01358-2 <https://doi.org/10.1038/s41592-021-01358-2>`_
- `squidpy Documentation <https://squidpy.readthedocs.io/>`_
- `squidpy GitHub <https://github.com/scverse/squidpy>`_

Tips and Best Practices
-----------------------

1. **Installation**: Ensure squidpy is installed: ``pip install squidpy``

2. **Method Selection**:
   - Use **Neighborhood Enrichment** to identify cell type interactions
   - Use **Co-occurrence Analysis** for pairwise spatial relationships
   - Use **Spatial Autocorrelation** to identify spatial gradients
   - Use **Ripley Functions** to test for clustering/dispersion

3. **Parameter Tuning**:
   - **n_permutations**: Use at least 100, preferably 500-1000 for publication
   - **max_dist** (Ripley): Should cover relevant spatial scales (1-5 cell diameters)

4. **Statistical Interpretation**:
   - Always consider both effect size and p-value
   - Multiple testing correction may be needed for many comparisons
   - Visualize results to understand spatial patterns

5. **Validation**:
   - Compare results across different graph construction methods
   - Verify that spatial patterns are biologically meaningful
   - Check edge effects in Ripley functions

6. **Performance**:
   - Advanced methods can be computationally intensive
   - Use parallel processing when available
   - Consider analyzing subsets of data for exploration

7. **Integration with Simple Spatial Analysis**:
   - Use Simple Spatial Analysis for initial exploration
   - Use Advanced Spatial Analysis for detailed statistical testing
   - Combine results from both for comprehensive spatial analysis

