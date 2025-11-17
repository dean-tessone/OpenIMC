Simple Spatial Analysis
========================

Simple Spatial Analysis provides fundamental spatial analysis tools for exploring cell spatial relationships, including spatial graph construction, pairwise enrichment, distance distributions, and spatial visualization.

Overview
--------

Spatial analysis examines how cells are organized in tissue space, identifying spatial patterns, cell-cell interactions, and tissue architecture. Simple Spatial Analysis includes core spatial analysis methods that work without additional dependencies like squidpy.

Options
-------

Simple Spatial Analysis includes:

1. **Spatial Graph Construction**: Build spatial networks connecting neighboring cells
2. **Pairwise Enrichment**: Test for spatial co-occurrence or avoidance between cell types
3. **Distance Distributions**: Analyze nearest-neighbor distances between cell types
4. **Spatial Visualization**: Visualize cell spatial organization
5. **Spatial Communities**: Identify spatially coherent cell communities

Parameters
----------

Spatial Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **method** (default: ``"kNN"``): Graph construction method
  - ``"kNN"``: k-nearest neighbors graph
  - ``"Radius"``: Connect all cells within a specified radius
  - ``"Delaunay"``: Delaunay triangulation (connects cells in triangular mesh)

- **k_neighbors** (default: ``10``): Number of nearest neighbors for kNN method
  - More neighbors (15-30) create denser graphs
  - Fewer neighbors (5-10) create sparser graphs
  - Typical range: 5-30

- **radius** (required for Radius method): Maximum distance for edges in pixels
  - Only used when method is "Radius"
  - Larger radius (50-100) connects more distant cells
  - Smaller radius (20-50) connects only nearby cells
  - Should be adjusted based on cell density

- **pixel_size_um** (default: ``1.0``): Pixel size in micrometers
  - Used to convert pixel distances to physical distances
  - Important for distance-based analyses
  - Should match your image acquisition settings

- **seed** (default: ``42``): Random seed for reproducibility
  - Used for permutation tests and community detection

Pairwise Enrichment
~~~~~~~~~~~~~~~~~~~

- **n_permutations** (default: ``100``): Number of permutations for statistical testing
  - More permutations (500-1000) provide more accurate p-values
  - Fewer permutations (100-200) are faster but less precise
  - Typical range: 100-1000

- **workers** (default: auto): Number of parallel workers for permutation tests
  - More workers speed up computation
  - Default: number of CPU cores - 2

Distance Distributions
~~~~~~~~~~~~~~~~~~~~~~

- **workers** (default: auto): Number of parallel workers for distance computation
  - More workers speed up computation for large datasets

Spatial Communities
~~~~~~~~~~~~~~~~~

- **min_cells** (default: ``5``): Minimum number of cells in a community
  - Filters out very small communities
  - Increase to focus on larger spatial structures

Using Simple Spatial Analysis in the GUI
-----------------------------------------

1. Ensure clustering has been completed (cells need cluster assignments)

2. Navigate to **Analysis → Spatial Analysis → Simple Spatial Analysis**

3. In the spatial analysis dialog:
   - **Build Spatial Graph**:
     - Select graph construction method (kNN, Radius, or Delaunay)
     - Set k_neighbors (for kNN) or radius (for Radius)
     - Set pixel size if known
     - Click "Build Graph"
   
   - **Pairwise Enrichment Tab**:
     - Set number of permutations
     - Set number of workers
     - Click "Run Enrichment Analysis"
     - Results show z-scores and p-values for each cluster pair
   
   - **Distance Distributions Tab**:
     - Click "Run Distance Analysis"
     - Select clusters to display in the plot
     - Results show nearest-neighbor distance distributions
   
   - **Spatial Visualization Tab**:
     - Select ROI to visualize
     - Choose color encoding (cluster or feature)
     - Optionally show edges
     - Click "Generate Spatial Plot"
   
   - **Spatial Communities Tab**:
     - Select ROI
     - Set minimum cells per community
     - Optionally exclude specific cell types
     - Click "Run Community Analysis"

4. Export results using the "Export Results" or "Export Graph" buttons

Using Simple Spatial Analysis in the CLI
-----------------------------------------

Basic Command
~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial features.csv spatial_edges.csv \\
       --method kNN \\
       --k-neighbors 10 \\
       --pixel-size-um 1.0

With Radius Method
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial features.csv spatial_edges.csv \\
       --method Radius \\
       --radius 50.0 \\
       --pixel-size-um 1.0

With Community Detection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc spatial features.csv spatial_edges.csv \\
       --method kNN \\
       --k-neighbors 10 \\
       --detect-communities \\
       --seed 42

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   spatial_analysis:
     enabled: true
     method: "kNN"
     k_neighbors: 10
     radius: null  # Not used for kNN
     pixel_size_um: 1.0
     detect_communities: false
     seed: 42

Method Details
--------------

Spatial Graph Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Spatial graphs represent cell neighborhoods by connecting cells that are spatially close.

**k-Nearest Neighbors (kNN)**:
- Connects each cell to its k nearest neighbors
- Creates a directed graph (can be made undirected)
- Good for uniform cell densities
- Fast computation using KD-tree

**Radius-based**:
- Connects all cells within a specified radius
- Creates an undirected graph
- Good for variable cell densities
- More edges than kNN for dense regions

**Delaunay Triangulation**:
- Connects cells in a triangular mesh
- Ensures all cells are connected to neighbors
- Good for exploring local neighborhoods
- Creates many edges

**Citation:**
- Implementation based on scipy.spatial: `scipy.spatial.cKDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`_ and `scipy.spatial.Delaunay <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html>`_

Pairwise Enrichment
~~~~~~~~~~~~~~~~~~~

Pairwise enrichment tests whether two cell types co-occur or avoid each other more than expected by chance.

**How it works:**

1. **Observed Co-occurrence**: Count edges between cell type A and cell type B in the spatial graph

2. **Expected Co-occurrence**: Compute expected number of edges under random spatial distribution
   - Based on proportions of each cell type

3. **Permutation Test**: Randomly shuffle cell type labels while preserving graph structure
   - Repeat n_permutations times
   - Compute z-score: (observed - mean(permuted)) / std(permuted)

4. **P-value**: Proportion of permutations with z-score as extreme or more extreme

**Interpretation:**
- Positive z-score + significant p-value: Enrichment (co-occurrence)
- Negative z-score + significant p-value: Depletion (avoidance)
- Non-significant: Random spatial distribution

**Citation:**
- Based on standard spatial co-occurrence analysis methods used in spatial transcriptomics and imaging mass cytometry
- Similar to methods in: Schapiro, D., et al. (2017). "histoCAT: analysis of cell phenotypes and interactions in multiplex image cytometry data." Nature Methods, 14(9), 873-876. `DOI: 10.1038/s41592-017-0001-x <https://doi.org/10.1038/s41592-017-0001-x>`_

Distance Distributions
~~~~~~~~~~~~~~~~~~~~~~~

Distance distribution analysis computes the distribution of nearest-neighbor distances between cell types.

**How it works:**

1. **For each cell**: Find nearest neighbor of each cell type (including same type)

2. **Distance Calculation**: Compute Euclidean distance to nearest neighbor
   - Converted to micrometers using pixel_size_um

3. **Distribution Analysis**: Aggregate distances across all cells
   - Compare distances between different cell type pairs
   - Visualize as violin/box plots

**Interpretation:**
- Shorter distances: Cell types are spatially close
- Longer distances: Cell types are spatially separated
- Compare distributions to identify spatial relationships

Spatial Visualization
~~~~~~~~~~~~~~~~~~~~~

Spatial visualization displays cells in their spatial coordinates, colored by cluster or feature values.

**Features:**
- Color cells by cluster assignment or feature expression
- Optionally display spatial graph edges
- Adjustable point sizes
- Per-ROI visualization

**Use cases:**
- Visual inspection of spatial organization
- Identifying spatial patterns
- Validating clustering results
- Exploring feature spatial distributions

Spatial Communities
~~~~~~~~~~~~~~~~~~~

Spatial community detection identifies spatially coherent groups of cells using graph-based clustering.

**How it works:**

1. **Graph Construction**: Build spatial graph (kNN, Radius, or Delaunay)

2. **Community Detection**: Apply Leiden algorithm to spatial graph
   - Identifies communities based on graph structure
   - Communities are spatially coherent groups

3. **Filtering**: Remove communities smaller than min_cells

**Interpretation:**
- Communities represent spatially organized cell groups
- May correspond to tissue structures or functional units
- Can be used to identify spatial niches

**Citation:**
- Leiden algorithm: Traag, V. A., et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports, 9(1), 5233. `DOI: 10.1038/s41598-019-41695-z <https://doi.org/10.1038/s41598-019-41695-z>`_
- Implementation: `leidenalg Python Package <https://github.com/vtraag/leidenalg>`_

Tips and Best Practices
-----------------------

1. **Graph Construction Method**:
   - Use **kNN** for most cases (fast, good default)
   - Use **Radius** when cell density varies significantly
   - Use **Delaunay** for detailed local neighborhood analysis

2. **Parameter Selection**:
   - **k_neighbors**: Start with 10, adjust based on cell density
   - **radius**: Should be 1-2 cell diameters
   - **pixel_size_um**: Critical for distance-based analyses, verify from metadata

3. **Pairwise Enrichment**:
   - Use at least 100 permutations for reliable results
   - Increase to 500-1000 for publication-quality p-values
   - Interpret z-scores in context of p-values

4. **Distance Distributions**:
   - Compare distances between different cell type pairs
   - Look for systematic differences indicating spatial relationships
   - Consider biological context when interpreting results

5. **Spatial Visualization**:
   - Always visually inspect spatial organization
   - Use different color encodings to explore different aspects
   - Compare across ROIs to identify consistent patterns

6. **Validation**:
   - Verify that spatial patterns are biologically meaningful
   - Check that graph construction parameters are appropriate
   - Ensure pixel size is correct for distance measurements

7. **Performance**:
   - Use parallel workers for large datasets
   - Consider processing ROIs separately if memory is limited
   - Graph construction is fast, but enrichment analysis can be slow for many permutations

Spatial Analysis Visualizations
================================

Simple Spatial Analysis provides several visualization options to explore spatial relationships and patterns. All visualizations are accessible from the spatial analysis dialog after building the spatial graph.

Available Visualizations
-------------------------

1. **Pairwise Enrichment**: Heatmap showing spatial co-occurrence/avoidance between cluster pairs
2. **Distance Distributions**: Violin/box plots of nearest-neighbor distances between cell types
3. **Spatial Visualization**: Scatter plot of cells in spatial coordinates
4. **Spatial Communities**: Visualization of spatially coherent cell communities

Pairwise Enrichment Visualization
----------------------------------

Shows a heatmap of z-scores and p-values for spatial co-occurrence or avoidance between cluster pairs.

**Parameters:**

- **Permutations**: Number of permutations for statistical testing (default: 100, range: 10-10000)
  - More permutations provide more accurate p-values
  - Recommended: 500-1000 for publication
- **Workers**: Number of parallel workers for permutation tests (default: auto)
  - More workers speed up computation
  - Default: number of CPU cores - 2

**How it works:**

1. Computes observed co-occurrence between cluster pairs in the spatial graph
2. Performs permutation tests by randomly shuffling cluster labels
3. Calculates z-scores: (observed - mean(permuted)) / std(permuted)
4. Computes p-values from permutation distribution
5. Displays results as a heatmap with z-scores color-coded

**Interpretation:**

- **Positive z-score + significant p-value**: Enrichment (cell types co-occur more than expected)
- **Negative z-score + significant p-value**: Depletion (cell types avoid each other)
- **Non-significant**: Random spatial distribution
- Color intensity indicates strength of association

**Export:**

- Click **"Save Plot"** button to export
- Options: PNG, JPG, or PDF format
- Adjustable DPI (default: 300)
- Optional font size and figure size override

Distance Distributions Visualization
-------------------------------------

Shows the distribution of nearest-neighbor distances between cell types using violin or box plots.

**Parameters:**

- **Clusters to display**: Select which cluster pairs to visualize (multi-select)
  - Can compare distances between same cluster vs. different clusters
  - Useful for identifying spatial relationships

**How it works:**

1. For each cell, finds the nearest neighbor of each cell type
2. Computes Euclidean distance to nearest neighbor
3. Converts to micrometers using pixel_size_um
4. Aggregates distances across all cells
5. Displays as violin/box plots grouped by cluster pair

**Interpretation:**

- **Shorter distances**: Cell types are spatially close
- **Longer distances**: Cell types are spatially separated
- Compare distributions to identify spatial relationships
- Same-cluster distances show within-cluster spatial organization
- Cross-cluster distances show between-cluster spatial relationships

**Export:**

- Click **"Save Plot"** button to export
- Same export options as other visualizations

Spatial Visualization
---------------------

Displays cells in their spatial coordinates (x, y positions), colored by cluster or feature expression.

**Parameters:**

- **ROI**: Select which ROI to visualize (dropdown)
  - Each ROI is visualized separately
  - Select from available ROIs in the dataset
- **Color by**: Choose how to color cells
  - ``"cluster"``: Color by cluster assignment (default)
  - Feature columns: Color by continuous feature expression (e.g., marker intensities)
  - Searchable dropdown for easy feature selection
- **Point Size**: Multiplier for point sizes (default: 1.0, range: 0.1-10.0)
  - 1.0 = default size
  - Increase for larger points (useful for sparse plots)
  - Decrease for smaller points (useful for dense plots)
- **Show edges**: Checkbox to display spatial graph edges
  - Shows connections between neighboring cells
  - Can be slow for large datasets
  - Useful for visualizing graph structure

**How it works:**

1. Extracts spatial coordinates (centroid_x, centroid_y) for selected ROI
2. Colors cells based on selected attribute (cluster or feature)
3. Optionally draws edges from spatial graph
4. Displays as scatter plot with legend

**Use cases:**

- Visual inspection of spatial organization
- Identifying spatial patterns and domains
- Validating clustering results
- Exploring feature spatial distributions
- Checking for batch effects across ROIs

**Export:**

- Click **"Save Plot"** button to export
- Same export options as other visualizations

Spatial Communities Visualization
----------------------------------

Shows spatially coherent communities of cells identified using graph-based clustering.

**Parameters:**

- **ROI**: Select which ROI to analyze (dropdown)
- **Min cells**: Minimum number of cells in a community (default: 5, range: 1-100)
  - Filters out very small communities
  - Increase to focus on larger spatial structures
- **Exclude cell types**: Optionally exclude specific cell types from community detection
  - Enable exclusion checkbox
  - Multi-select clusters to exclude
  - Useful for focusing on specific cell populations

**How it works:**

1. Builds spatial graph for selected ROI
2. Applies Leiden algorithm to identify communities
3. Filters communities smaller than min_cells
4. Visualizes communities as colored regions in spatial coordinates
5. Shows community assignments and spatial organization

**Interpretation:**

- Communities represent spatially organized cell groups
- May correspond to tissue structures or functional units
- Can be used to identify spatial niches
- Compare community structure across ROIs

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

1. Run the desired analysis (enrichment, distance, spatial viz, or communities)
2. Adjust any parameters (point size, show edges, etc.)
3. Click **"Save Plot"** button in the relevant tab
4. In the save dialog:
   - Choose filename and location
   - Select format (PNG/JPG/PDF)
   - Set DPI (for raster formats)
   - Optionally override font size
   - Optionally override figure size
5. Click **"Save"**

**Tips for Export:**

- Use **PDF** format for publications (vector graphics, scalable)
- Use **PNG** at 300 DPI for presentations and web
- Increase font size for small figures in publications
- Adjust figure size to match journal requirements
- Spatial visualizations benefit from larger figure sizes to show detail

Accessing Visualizations in the GUI
------------------------------------

1. **Build Spatial Graph**: First, build the spatial graph using the controls at the top
   - Select graph construction method (kNN, Radius, or Delaunay)
   - Set parameters (k_neighbors, radius, pixel_size_um)
   - Click "Build Graph"
   - Graph must be built before visualizations are available

2. **Open Spatial Analysis Dialog**: Navigate to **Analysis → Spatial Analysis → Simple Spatial Analysis**

3. **Select Tab**: Use the tabs to access different visualizations
   - **Pairwise Enrichment**: Run enrichment analysis and view heatmap
   - **Distance Distributions**: Run distance analysis and view distributions
   - **Spatial Visualization**: Generate spatial scatter plots
   - **Spatial Communities**: Run community detection and view communities

4. **Adjust Parameters**: Use controls in each tab to customize visualizations

5. **Export**: Click **"Save Plot"** in each tab to export visualizations

**Tab-Specific Controls:**

- **Pairwise Enrichment**: Permutations, Workers, Run button, Save Plot button
- **Distance Distributions**: Cluster selection, Run button, Save Plot button
- **Spatial Visualization**: ROI selection, Color by, Point Size, Show edges, Generate button, Save Plot button
- **Spatial Communities**: ROI selection, Min cells, Exclude cell types, Run button, Save Plot button

Tips and Best Practices for Visualizations
-------------------------------------------

1. **Pairwise Enrichment:**
   - Use at least 100 permutations for reliable results
   - Increase to 500-1000 for publication-quality p-values
   - Interpret z-scores in context of p-values
   - Look for consistent patterns across multiple ROIs

2. **Distance Distributions:**
   - Compare distances between different cell type pairs
   - Look for systematic differences indicating spatial relationships
   - Consider biological context when interpreting results
   - Compare same-cluster vs. cross-cluster distances

3. **Spatial Visualization:**
   - Always visually inspect spatial organization
   - Use different color encodings to explore different aspects
   - Compare across ROIs to identify consistent patterns
   - Adjust point size for optimal visibility
   - Use "Show edges" sparingly (can be slow for large datasets)

4. **Spatial Communities:**
   - Adjust min_cells to focus on relevant spatial scales
   - Exclude cell types that are not of interest
   - Compare community structure across ROIs
   - Use communities to identify spatial niches

5. **Export:**
   - Use PDF for publications (vector graphics)
   - Use PNG at 300 DPI for presentations
   - Adjust font sizes for small figures
   - Spatial visualizations may need larger figure sizes

