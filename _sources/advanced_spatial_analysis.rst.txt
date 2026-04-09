Advanced Spatial Analysis
==========================

Advanced Spatial Analysis provides sophisticated spatial analysis methods using squidpy, including neighborhood enrichment, co-occurrence analysis, spatial autocorrelation, and Ripley functions.

Overview
--------

Advanced Spatial Analysis extends Simple Spatial Analysis with additional methods from the squidpy package, enabling more sophisticated spatial pattern analysis, statistical testing, and spatial statistics.

**Key Features:**
- **Squidpy Integration**: All analyses are implemented using the squidpy package
- **AnnData Format**: Data is converted to AnnData format for compatibility with the scverse ecosystem
- **Export Capability**: AnnData objects can be exported to H5AD files for downstream analysis in other tools

.. note::
   Advanced Spatial Analysis requires the ``squidpy`` and ``anndata`` packages to be installed. Install with: ``pip install squidpy anndata``

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

The graph-construction controls resemble Simple Spatial Analysis in the GUI, but the implementation is different:

- **method**: ``kNN``, ``Radius``, or ``Delaunay``
- **k_neighbors**: Number of neighbors for kNN graph construction through Squidpy
- **radius**: Maximum distance for ``Radius`` graphs in micrometers after coordinate scaling
- **pixel_size_um**: Pixel size in micrometers used to convert centroid coordinates before building AnnData graphs

Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~

- **ROI** (optional): Run a single ROI or use **All ROIs** to aggregate across the current ROI set
- **Aggregation**: ``Mean`` or ``Sum`` when **All ROIs** is selected
- **cluster_column**: Cluster annotation used for the Squidpy enrichment matrix

.. note::
   The current OpenIMC wrapper does not expose Squidpy's neighborhood-enrichment ``n_perms`` or enrichment-level ``seed`` options. The GUI and CLI surface the resulting z-score/count-style matrices rather than neighborhood-enrichment p-values.

Co-occurrence Analysis
~~~~~~~~~~~~~~~~~~~~~~~

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

2. Navigate to **Analysis → Spatial Analysis → Advanced Spatial Analysis** in the menu bar

3. In the advanced spatial analysis dialog:

   - **Build Spatial Graph**:
   
     - Select graph construction method (kNN, Radius, or Delaunay)
     - Set parameters (k_neighbors, radius, pixel_size_um)
     - Click "Build Graph"
     - This converts your data to AnnData format and builds spatial graphs using squidpy
   
   - **Neighborhood Enrichment Tab**:
   
     - Optionally choose a single ROI or **All ROIs**
     - Choose aggregation mode and cluster column
     - Click "Run Neighborhood Enrichment"
     - Results show a cluster-by-neighbor-cluster z-score matrix
     - Single-ROI results are stored in AnnData objects; multi-ROI results are aligned and aggregated for display
   
   - **Co-occurrence Analysis Tab**:
   
     - Select analysis method (pairwise or one-vs-others)
     - Optionally specify reference cluster
     - Click "Run Co-occurrence Analysis"
     - Results are stored in AnnData objects
   
   - **Spatial Autocorrelation Tab**:
   
     - Select feature to analyze
     - Choose autocorrelation method (Moran's I or Geary's C)
     - Set number of permutations
     - Click "Run Autocorrelation Analysis"
     - Results are stored in AnnData objects
   
   - **Ripley Functions Tab**:
   
     - Select cluster column
     - Choose function type (K or L)
     - Set maximum distance
     - Click "Run Ripley Analysis"
     - Results are stored in AnnData objects

4. **Export AnnData**:
   
   - Click "Export AnnData" button to save AnnData objects
   - Choose to export as:
   
     - **Combined file**: Single H5AD file with all ROIs
     - **Separate files**: One H5AD file per ROI
   
   - Exported H5AD files can be used in other tools (scanpy, squidpy, etc.)

5. Export analysis results using the export buttons

Using Advanced Spatial Analysis in the CLI
-------------------------------------------

Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial-nhood-enrichment features.csv \\
       --output enrichment_results.h5ad \\
       --method kNN \\
       --k-neighbors 10 \\
       --cluster-column cluster \\
       --aggregation mean

Co-occurrence Analysis
~~~~~~~~~~~~~~~~~~~~~~

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

Build Spatial Graph and Export AnnData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial-anndata features.csv --output spatial_graph.h5ad \\
       --method kNN \\
       --k-neighbors 10 \\
       --pixel-size-um 1.0 \\
       --combined

Export AnnData Objects
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc export-anndata input.h5ad output.h5ad \\
       --combined

Method Details
--------------

Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~

OpenIMC advanced neighborhood enrichment wraps ``squidpy.gr.nhood_enrichment`` on ROI-specific AnnData spatial graphs.
The result is best interpreted as a focal-cluster by neighbor-cluster enrichment matrix rather than the undirected edge-pair summary used by Simple Spatial Analysis.

**How it works:**

1. **Graph Construction**: Build or reuse ROI-specific AnnData spatial connectivities with Squidpy

2. **Neighborhood Counting**: For each cluster, count neighboring clusters in the AnnData connectivity graph

3. **Squidpy Statistic**: Run Squidpy's neighborhood-enrichment permutation test to compute z-scores and counts
   - Squidpy stores these results under ``adata.uns['{cluster_key}_nhood_enrichment']``

4. **OpenIMC Aggregation**: If multiple ROIs are selected, OpenIMC aligns ROI matrices to the union of clusters and aggregates the z-score matrices by ``mean`` or ``sum`` for display

**Interpretation:**
- Rows correspond to the focal cell cluster and columns correspond to the neighboring cluster
- Positive z-scores indicate more neighbors than expected; negative z-scores indicate fewer neighbors than expected
- The current OpenIMC wrapper surfaces z-score/count-style outputs from Squidpy, not neighborhood-enrichment p-values
- Depending on graph construction and aggregation, the matrix should not be interpreted as the same statistic as simple pairwise enrichment

Why Results Differ from Simple Pairwise Enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Statistic definition**: Advanced neighborhood enrichment uses Squidpy's cluster-by-neighbor-cluster enrichment statistic. Simple pairwise enrichment counts unordered edges between cluster pairs.
- **Graph representation**: Advanced enrichment uses AnnData spatial connectivities. Simple enrichment uses OpenIMC's deduplicated undirected edge list.
- **Symmetry and directionality**: Advanced heatmaps are interpreted as row cluster versus neighbor cluster. Simple pairwise heatmaps are symmetric by construction.
- **ROI aggregation**: Advanced OpenIMC displays align ROI matrices and aggregate z-scores by ``mean`` or ``sum``. Simple GUI displays average ROI-level ``z_score`` and ``p_value`` values by cluster pair.
- **Reproducibility controls**: Simple pairwise enrichment exposes ``n_permutations`` and ``seed`` directly for the enrichment step. Advanced neighborhood enrichment currently does not expose Squidpy's neighborhood-enrichment ``n_perms`` or enrichment-level ``seed`` controls through the OpenIMC wrapper.

**Citation:**
- Based on methods in: Schapiro, D., et al. (2017). "histoCAT: analysis of cell phenotypes and interactions in multiplex image cytometry data." Nature Methods, 14(9), 873-876. `DOI: 10.1038/s41592-017-0001-x <https://doi.org/10.1038/s41592-017-0001-x>`_
- Implementation: `squidpy.gr.nhood_enrichment <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html>`_
- Additional implementation references: `Squidpy graph builder <https://squidpy.readthedocs.io/en/stable/_modules/squidpy/gr/_build.html>`_ and `Squidpy neighborhood-enrichment source <https://squidpy.readthedocs.io/en/stable/_modules/squidpy/gr/_nhood.html>`_

Co-occurrence Analysis
~~~~~~~~~~~~~~~~~~~~~~

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

Squidpy and AnnData Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced Spatial Analysis uses the squidpy package, which provides a comprehensive toolkit for spatial omics analysis. All analyses are performed using AnnData objects, which provide a standardized format for single-cell and spatial omics data.

**Data Flow:**

1. **Input**: Feature DataFrame (CSV) with cell features and spatial coordinates
2. **Conversion**: DataFrame is converted to AnnData format using ``dataframe_to_anndata()``
   - Features stored in ``adata.X`` or ``adata.obs``
   - Spatial coordinates stored in ``adata.obsm['spatial']``
   - Metadata stored in ``adata.obs``
3. **Graph Construction**: Spatial graphs are built using squidpy
   - Graphs stored in ``adata.obsp['spatial_connectivities']`` and ``adata.obsp['spatial_distances']``
4. **Analysis**: Squidpy functions operate on AnnData objects
   - Results stored in ``adata.uns``, ``adata.obs``, or ``adata.obsm``
5. **Export**: AnnData objects can be exported to H5AD files for use in other tools

**AnnData Export:**

AnnData objects can be exported in two formats:

- **Combined Export**: All ROIs combined into a single H5AD file
  - Useful for downstream analysis in scanpy or other tools
  - Preserves ROI information in ``adata.obs``
  
- **Separate Export**: One H5AD file per ROI
  - Useful when analyzing ROIs independently
  - Files named as ``anndata_roi_{roi_id}.h5ad``

**Using Exported AnnData:**

Exported H5AD files can be used in:

- **scanpy**: For additional single-cell analysis
- **squidpy**: For additional spatial analysis methods
- **Python scripts**: Load with ``anndata.read_h5ad()``
- **R/Bioconductor**: Using the ``zellkonverter`` package

**Citation:**
- Palla, G., et al. (2022). "Squidpy: a scalable framework for spatial omics analysis." Nature Methods, 19(2), 171-178. `DOI: 10.1038/s41592-021-01358-2 <https://doi.org/10.1038/s41592-021-01358-2>`_
- `squidpy Documentation <https://squidpy.readthedocs.io/>`_
- `squidpy GitHub <https://github.com/scverse/squidpy>`_
- AnnData: Virshup, I., et al. (2023). "The scverse project provides a computational ecosystem for single-cell omics data analysis." Nature Biotechnology, 41(5), 604-606. `DOI: 10.1038/s41587-023-01733-8 <https://doi.org/10.1038/s41587-023-01733-8>`_
- `AnnData Documentation <https://anndata.readthedocs.io/>`_

Tips and Best Practices
-----------------------

1. **Installation**: Ensure squidpy is installed: ``pip install squidpy``

2. **Method Selection**:
   - Use **Neighborhood Enrichment** to identify cell type interactions
   - Use **Co-occurrence Analysis** for pairwise spatial relationships
   - Use **Spatial Autocorrelation** to identify spatial gradients
   - Use **Ripley Functions** to test for clustering/dispersion

3. **Parameter Tuning**:
   - Tune graph parameters first (``k_neighbors``, ``radius``, and ROI selection), since they strongly affect neighborhood-based statistics
   - **max_dist** (Ripley): Should cover relevant spatial scales (1-5 cell diameters)

4. **Statistical Interpretation**:
   - Interpret each method using the outputs it actually returns
   - For neighborhood enrichment, focus on the z-score matrix and row/column semantics
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
   - Use Advanced Spatial Analysis for Squidpy-based neighborhood and spatial statistics
   - Treat simple pairwise enrichment and advanced neighborhood enrichment as complementary, not interchangeable

8. **AnnData Export**:
   - Export AnnData objects after building graphs or running analyses
   - Use combined export for multi-ROI analysis in other tools
   - Use separate export for ROI-specific analysis
   - Exported H5AD files are compatible with the entire scverse ecosystem

9. **Workflow Integration**:
   - Build spatial graphs first (creates AnnData objects)
   - Run analyses (results stored in AnnData)
   - Export AnnData for downstream analysis or visualization
   - Can reload exported AnnData files for further analysis

Advanced Spatial Analysis Visualizations
=========================================

Advanced Spatial Analysis provides sophisticated visualization options for exploring spatial patterns using squidpy methods. All visualizations are accessible from the advanced spatial analysis dialog after building the spatial graph.

Available Visualizations
-------------------------

1. **Neighborhood Enrichment**: Heatmap showing enrichment/depletion of cell types in neighborhoods
2. **Co-occurrence Analysis**: Heatmap showing spatial co-occurrence patterns
3. **Spatial Autocorrelation**: Visualization of spatial autocorrelation statistics
4. **Ripley Functions**: Plots of Ripley's K and L functions for spatial point patterns

Neighborhood Enrichment Visualization
--------------------------------------

Shows a heatmap of neighborhood-enrichment z-scores, with focal cell clusters on the rows and neighbor clusters on the columns.

**Parameters:**

- **ROI**: Plot a selected ROI or an aggregated view across **All ROIs**
- **Aggregation**: ``Mean`` or ``Sum`` when plotting multiple ROIs together
- **Cluster column**: Select the cluster annotation used to build the Squidpy enrichment matrix

**How it works:**

1. Build or reuse the AnnData spatial graph for each ROI
2. Run Squidpy neighborhood enrichment on the selected cluster column
3. Read Squidpy's z-score/count results from the AnnData object
4. If multiple ROIs are selected, align cluster sets and aggregate z-score matrices by ``mean`` or ``sum``
5. Display the resulting matrix as a heatmap

**Interpretation:**

- **Positive z-score**: The column cluster appears more often than expected in the row cluster's neighborhood
- **Negative z-score**: The column cluster appears less often than expected in the row cluster's neighborhood
- OpenIMC currently plots z-score matrices for this analysis rather than neighborhood-enrichment p-values
- Color intensity indicates strength of enrichment/depletion

**Export:**

- Click **"Save Plot"** button to export
- Options: PNG, JPG, or PDF format
- Adjustable DPI (default: 300)
- Optional font size and figure size override

Co-occurrence Analysis Visualization
-------------------------------------

Shows a heatmap of co-occurrence scores indicating whether cell types tend to appear together in spatial proximity.

**Parameters:**

- **Method**: Analysis method
  - ``"pairwise"``: Compare all cluster pairs (default)
  - ``"one_vs_others"``: Compare reference cluster against all others
- **Reference cluster** (for one_vs_others mode): Select cluster to compare against all others

**How it works:**

1. Defines spatial proximity based on spatial graph (kNN, Radius, etc.)
2. Counts observed co-occurrence of cell type pairs in proximity
3. Computes expected co-occurrence under random distribution
4. Performs permutation tests to assess significance
5. Displays results as heatmap with co-occurrence scores and p-values

**Pairwise Mode**: Compares all pairs of cell types

**One-vs-Others Mode**: Compares a reference cell type against all others

**Interpretation:**

- **Positive score + significant p-value**: Co-occurrence (cell types appear together more than expected)
- **Negative score + significant p-value**: Avoidance (cell types appear together less than expected)
- **Non-significant**: Random spatial distribution

**Export:**

- Click **"Save Plot"** button to export
- Same export options as other visualizations

Spatial Autocorrelation Visualization
--------------------------------------

Shows spatial autocorrelation statistics (Moran's I or Geary's C) for selected features, indicating spatial clustering or dispersion.

**Parameters:**

- **Feature**: Select feature column to analyze (required)
  - Can be a marker expression or other numeric feature
  - Dropdown with all available features
- **Method**: Autocorrelation method
  - ``"moran"``: Moran's I statistic (default)
  - ``"geary"``: Geary's C statistic
- **n_permutations** (default: ``100``): Number of permutations for significance testing
  - Recommended: 500-1000 for publication

**Moran's I:**
- Range: -1 to 1
- Positive values: Similar values cluster together (positive autocorrelation)
- Negative values: Dissimilar values cluster together (negative autocorrelation)
- Near 0: No spatial autocorrelation (random spatial distribution)

**Geary's C:**
- Range: 0 to 2
- Values < 1: Positive autocorrelation
- Values > 1: Negative autocorrelation
- Values near 1: No autocorrelation

**How it works:**

1. Defines spatial weights matrix based on spatial graph
2. Computes Moran's I or Geary's C using spatial weights
3. Performs permutation tests to assess significance
4. Displays statistic value, p-value, and visualization

**Interpretation:**

- **Positive autocorrelation**: Feature values are spatially clustered
- **Negative autocorrelation**: Feature values are spatially dispersed
- Useful for identifying spatial gradients or domains
- High autocorrelation indicates spatial structure in feature expression

**Export:**

- Click **"Save Plot"** button to export
- Same export options as other visualizations

Ripley Functions Visualization
-------------------------------

Shows Ripley's K or L functions for analyzing spatial point patterns, testing for clustering or dispersion.

**Parameters:**

- **Cluster column**: Column name containing cluster assignments (default: ``"cluster"``)
- **Mode**: Ripley function type
  - ``"K"``: Ripley's K function (default)
  - ``"L"``: Ripley's L function (normalized K function)
- **max_dist** (optional): Maximum distance to compute function
  - If not specified, uses a default based on data extent
  - Should cover relevant spatial scales (1-5 cell diameters)

**Ripley's K Function:**
- Measures the expected number of points within distance r of a randomly chosen point
- Under complete spatial randomness (CSR): K(r) = πr²
- K(r) > πr²: Clustering
- K(r) < πr²: Dispersion

**Ripley's L Function:**
- Normalized version: L(r) = √(K(r)/π) - r
- Under CSR: L(r) = 0
- L(r) > 0: Clustering
- L(r) < 0: Dispersion

**How it works:**

1. For each point, counts neighbors within distance r
2. Applies edge correction for points near ROI boundaries
3. Computes K(r) or L(r) for a range of distances
4. Compares observed function to expected under complete spatial randomness
5. Displays function plot with confidence intervals

**Interpretation:**

- **Clustering**: Cell type is more clustered than random
- **Dispersion**: Cell type is more dispersed than random
- Useful for identifying spatial organization patterns
- Compare functions across different cell types
- Look for peaks (clustering) or valleys (dispersion) at specific distances

**Export:**

- Click **"Save Plot"** button to export
- Same export options as other visualizations

Exporting Plots
---------------

All visualizations can be exported using the **"Save Plot"** button in each tab.

**Export Options:**

1. **Format**: Choose output format
   - ``PNG``: Raster image (default, good for presentations)
   - ``JPG``: Compressed raster image
   - ``PDF``: Vector format (good for publications, scalable)

2. **DPI (Dots Per Inch)**: Resolution for raster formats
   - Default: 300 DPI (publication quality)
   - Range: 72-1200 DPI
   - Higher DPI = larger file size, better quality

3. **Font Size Override**: Optionally override all font sizes
   - Check "Override figure font size"
   - Set font size in points (default: 10.0, range: 6.0-72.0)
   - Useful for adjusting text size for publications

4. **Figure Size Override**: Optionally change figure dimensions
   - Check "Override figure size"
   - Set width and height in inches (default: 8.0 x 6.0)
   - Range: 1.0-100.0 inches

**Export Workflow:**

1. Build spatial graph (creates AnnData objects)
2. Run the desired analysis (neighborhood enrichment, co-occurrence, autocorrelation, or Ripley)
3. Adjust any parameters
4. Click **"Save Plot"** button in the relevant tab
5. In the save dialog:
   - Choose filename and location
   - Select format (PNG/JPG/PDF)
   - Set DPI (for raster formats)
   - Optionally override font size
   - Optionally override figure size
6. Click **"Save"**

**Tips for Export:**

- Use **PDF** format for publications (vector graphics, scalable)
- Use **PNG** at 300 DPI for presentations and web
- Increase font size for small figures in publications
- Adjust figure size to match journal requirements
- Heatmaps may need larger figure sizes to show all labels clearly

Accessing Visualizations in the GUI
------------------------------------

1. **Build Spatial Graph**: First, build the spatial graph using the controls at the top
   - Select graph construction method (kNN, Radius, or Delaunay)
   - Set parameters (k_neighbors, radius, pixel_size_um)
   - Click "Build Graph"
   - This converts data to AnnData format and builds spatial graphs using squidpy
   - Graph must be built before analyses are available

2. **Open Advanced Spatial Analysis Dialog**: Navigate to **Analysis → Spatial Analysis → Advanced Spatial Analysis** in the menu bar

3. **Select Tab**: Use the tabs to access different analyses and visualizations
   - **Neighborhood Enrichment**: Run enrichment analysis and view heatmap
   - **Co-occurrence Analysis**: Run co-occurrence analysis and view heatmap
   - **Spatial Autocorrelation**: Run autocorrelation analysis and view results
   - **Ripley Functions**: Run Ripley analysis and view function plots

4. **Adjust Parameters**: Use controls in each tab to customize analyses

5. **Export**: Click **"Save Plot"** in each tab to export visualizations

6. **Export AnnData**: Click **"Export AnnData"** button to save AnnData objects for downstream analysis

**Tab-Specific Controls:**

- **Neighborhood Enrichment**: ROI selection, Aggregation, Cluster column, Run button, Save Plot button
- **Co-occurrence Analysis**: Method, Reference cluster (for one_vs_others), Run button, Save Plot button
- **Spatial Autocorrelation**: Feature selection, Method, n_permutations, Run button, Save Plot button
- **Ripley Functions**: Cluster column, Mode, max_dist, Run button, Save Plot button

Tips and Best Practices for Visualizations
-------------------------------------------

1. **Neighborhood Enrichment:**
   - Interpret the heatmap by row cluster versus neighbor cluster
   - Compare results across graph settings and ROI aggregation modes
   - Look for consistent patterns across multiple ROIs
   - Consider biological context when interpreting results

2. **Co-occurrence Analysis:**
   - Use pairwise mode to explore all relationships
   - Use one_vs_others mode to focus on specific cell types
   - Compare results across different graph construction methods
   - Consider multiple testing correction for many comparisons

3. **Spatial Autocorrelation:**
   - Select informative features (known spatial markers)
   - Use Moran's I for most cases (more commonly used)
   - Use Geary's C for alternative perspective
   - High autocorrelation indicates spatial structure

4. **Ripley Functions:**
   - Set max_dist to cover relevant spatial scales (1-5 cell diameters)
   - Compare functions across different cell types
   - Look for peaks (clustering) or valleys (dispersion) at specific distances
   - Consider edge effects near ROI boundaries

5. **Export:**
   - Use PDF for publications (vector graphics)
   - Use PNG at 300 DPI for presentations
   - Adjust font sizes for small figures
   - Heatmaps may need larger figure sizes
   - Export AnnData objects for further analysis in other tools
