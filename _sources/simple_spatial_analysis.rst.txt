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

