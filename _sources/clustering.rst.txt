Clustering
==========

Clustering groups cells into phenotypically similar populations based on their extracted features, enabling cell type identification and population analysis.

Overview
--------

Clustering is a fundamental step in single-cell analysis that identifies distinct cell populations. OpenIMC supports multiple clustering algorithms, each with different characteristics suited to different data types and analysis goals.

Options
-------

OpenIMC supports five clustering methods:

1. **Leiden** (default): Graph-based clustering using the Leiden algorithm
2. **Louvain**: Graph-based clustering using the Louvain algorithm
3. **Hierarchical**: Agglomerative hierarchical clustering
4. **K-means**: Partition-based clustering with k clusters
5. **HDBSCAN**: Density-based clustering that identifies clusters of varying density

Parameters
----------

Common Parameters
~~~~~~~~~~~~~~~~~

- **method** (default: ``"leiden"``): Clustering method
  - Options: ``"leiden"``, ``"louvain"``, ``"hierarchical"``, ``"kmeans"``, ``"hdbscan"``

- **columns** (optional): List of feature column names to use for clustering
  - If not specified, auto-detects all numeric feature columns
  - Excludes metadata columns (cell_id, centroid_x, centroid_y, cluster, etc.)
  - Recommended: Use intensity features, optionally including morphological features

- **scaling** (default: ``"zscore"``): Feature scaling method before clustering
  - ``"none"``: No scaling (use raw features)
  - ``"zscore"``: Z-score normalization (mean=0, std=1)
  - ``"mad"``: Median Absolute Deviation normalization (robust to outliers)

- **use_pca** (default: ``False``): Cluster on principal components instead of the selected marker/morphometric feature matrix
  - PCA is applied after feature selection, missing-value handling, and scaling
  - The original selected features remain available for heatmaps, exports, and interpretation

- **pca_mode** (default: ``"variance"``): PCA retention mode when ``use_pca`` is enabled
  - ``"variance"``: Retain PCs up to ``pca_variance`` cumulative variance
  - ``"components"``: Retain a fixed ``pca_n_components`` PCs

- **pca_variance** (default: ``0.95``): Proportion of variance to retain when ``pca_mode="variance"``

- **pca_n_components** (optional): Number of PCs to retain when ``pca_mode="components"``

- **seed** (default: ``42``): Random seed for reproducibility
  - Ensures consistent results across runs
  - Change to explore different initializations

Leiden/Louvain Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **resolution** (default: ``1.0``): Resolution parameter controlling cluster granularity
  - Higher values (1.5-3.0) create more, smaller clusters
  - Lower values (0.3-0.8) create fewer, larger clusters
  - Typical range: 0.5-2.0
  - Adjust to find optimal cluster granularity

- **n_neighbors** (default: ``15``): Number of neighbors for k-NN graph construction
  - More neighbors (20-30) create denser graphs, smoother clusters
  - Fewer neighbors (5-10) create sparser graphs, more distinct clusters
  - Typical range: 10-30

- **metric** (default: ``"euclidean"``): Distance metric for k-NN graph
  - ``"euclidean"``: Standard Euclidean distance
  - ``"manhattan"``: Manhattan (L1) distance
  - ``"cosine"``: Cosine similarity (good for high-dimensional data)

Hierarchical Parameters
~~~~~~~~~~~~~~~~~~~~~~~

- **n_clusters** (required): Number of clusters to identify
  - Must be specified for hierarchical clustering
  - Use domain knowledge or methods like elbow plot to determine optimal k

- **linkage** (default: ``"ward"``): Linkage criterion for merging clusters
  - ``"ward"``: Minimizes within-cluster variance (recommended for Euclidean distance)
  - ``"complete"``: Maximum distance between clusters
  - ``"average"``: Average distance between clusters

K-means Parameters
~~~~~~~~~~~~~~~~~~

- **n_clusters** (required): Number of clusters to identify
  - Must be specified for K-means
  - Use domain knowledge or methods like elbow plot to determine optimal k

- **n_init** (default: ``10``): Number of initializations
  - K-means is sensitive to initialization
  - More initializations (10-20) improve stability
  - Final result uses the best initialization

HDBSCAN Parameters
~~~~~~~~~~~~~~~~~~

- **min_cluster_size** (default: ``10``): Minimum number of cells in a cluster
  - Smaller values (5-10) identify more, smaller clusters
  - Larger values (20-50) identify fewer, larger clusters
  - Cells not meeting this criterion are marked as noise

- **min_samples** (default: ``5``): Minimum samples in neighborhood for core point
  - Controls cluster density requirement
  - Lower values (3-5) allow sparser clusters
  - Higher values (10-20) require denser clusters

- **cluster_selection_method** (default: ``"eom"``): Method for selecting clusters from tree
  - ``"eom"``: Excess of Mass (recommended, more stable)
  - ``"leaf"``: Leaf selection (more clusters, may be less stable)

- **hdbscan_metric** (default: ``"euclidean"``): Distance metric
  - ``"euclidean"``: Standard Euclidean distance
  - ``"manhattan"``: Manhattan (L1) distance

Using Clustering in the GUI
----------------------------

1. Ensure feature extraction (and optionally batch correction) has been completed

2. Navigate to **Analysis → Clustering** or click the clustering button

3. In the clustering dialog:
   - Select the clustering method
   - Choose feature columns to use (or use auto-detected)
   - Select scaling method
   - Optionally enable **Cluster on principal components** and choose retained variance or number of PCs
   - Adjust method-specific parameters:
   
     - For Leiden/Louvain: Set resolution and n_neighbors
     - For Hierarchical/K-means: Set n_clusters and linkage (hierarchical)
     - For HDBSCAN: Set min_cluster_size and min_samples
   
   - Optionally use "Find Optimal K" tool for hierarchical/K-means

4. Click **Run Clustering** to start the process

5. Cluster assignments are added to the features dataframe in a ``cluster`` column

6. Results can be visualized and exported

Using Clustering in the CLI
---------------------------

Basic Leiden Command
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc cluster features.csv clustered_features.csv \\
       --method leiden \\
       --resolution 1.0

With Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc cluster features.csv clustered_features.csv \\
       --method leiden \\
       --resolution 1.5 \\
       --n-neighbors 20 \\
       --metric euclidean \\
       --scaling zscore

Hierarchical Command
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc cluster features.csv clustered_features.csv \\
       --method hierarchical \\
       --n-clusters 10 \\
       --linkage ward

K-means Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc cluster features.csv clustered_features.csv \\
       --method kmeans \\
       --n-clusters 10 \\
       --n-init 20

HDBSCAN Command
~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc cluster features.csv clustered_features.csv \\
       --method hdbscan \\
       --min-cluster-size 20 \\
       --min-samples 5

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   clustering:
     enabled: true
     method: "leiden"
     columns: null  # Auto-detect
     scaling: "zscore"
     use_pca: false
     pca_mode: "variance"
     pca_variance: 0.95
     pca_n_components: null
     resolution: 1.0
     n_neighbors: 15
     metric: "euclidean"
     seed: 42

Method Details
--------------

Leiden Algorithm
~~~~~~~~~~~~~~~~

The Leiden algorithm is a graph-based clustering method that optimizes modularity, a measure of cluster quality.

**How it works:**

1. **Graph Construction**: Builds a k-nearest neighbor (k-NN) graph from feature space
   - Each cell is a node
   - Edges connect to k nearest neighbors based on distance metric

2. **Modularity Optimization**: Iteratively optimizes modularity by moving cells between clusters
   - Modularity measures how well clusters are separated
   - Resolution parameter controls the trade-off between cluster size and number

3. **Refinement**: Applies local refinement to improve cluster quality

**Advantages:**
- Fast and scalable
- Handles large datasets well
- Resolution parameter provides control over granularity
- Works well with high-dimensional data

**Limitations:**
- Requires tuning of resolution parameter
- Graph construction depends on n_neighbors parameter

**Citation:**
- Traag, V. A., et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports, 9(1), 5233. `DOI: 10.1038/s41598-019-41695-z <https://doi.org/10.1038/s41598-019-41695-z>`_
- `leidenalg Python Package <https://github.com/vtraag/leidenalg>`_

Louvain Algorithm
~~~~~~~~~~~~~~~~~

The Louvain algorithm is similar to Leiden but uses a different optimization strategy.

**How it works:**

1. **Graph Construction**: Same as Leiden (k-NN graph)

2. **Modularity Optimization**: Two-phase iterative optimization
   - Local optimization: Move nodes to maximize modularity
   - Aggregation: Merge nodes in same cluster, repeat

**Advantages:**
- Fast and widely used
- Good default choice for many applications

**Limitations:**
- May produce disconnected communities (Leiden fixes this)
- Requires tuning of resolution parameter

**Citation:**
- Blondel, V. D., et al. (2008). "Fast unfolding of communities in large networks." Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008. `DOI: 10.1088/1742-5468/2008/10/P10008 <https://doi.org/10.1088/1742-5468/2008/10/P10008>`_

Hierarchical Clustering
~~~~~~~~~~~~~~~~~~~~~~~

Hierarchical clustering builds a tree (dendrogram) of clusters by iteratively merging the closest clusters.

**How it works:**

1. **Initialization**: Each cell starts as its own cluster

2. **Iterative Merging**: At each step, merge the two closest clusters
   - Distance between clusters determined by linkage criterion
   - Ward linkage minimizes within-cluster variance

3. **Cut Tree**: Cut the dendrogram at the specified number of clusters (n_clusters)

**Advantages:**
- Provides hierarchical structure (can explore at different resolutions)
- Deterministic results
- Works well with small to medium datasets

**Limitations:**
- Computationally expensive for large datasets (O(n³) complexity)
- Requires specifying number of clusters
- Sensitive to outliers

**Citation:**
- Murtagh, F., & Contreras, P. (2012). "Algorithms for hierarchical clustering: an overview." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2(1), 86-97. `DOI: 10.1002/widm.53 <https://doi.org/10.1002/widm.53>`_
- Implementation: `scipy.cluster.hierarchy <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_

K-means Clustering
~~~~~~~~~~~~~~~~~~

K-means partitions cells into k clusters by minimizing within-cluster variance.

**How it works:**

1. **Initialization**: Randomly assign k cluster centers

2. **Assignment**: Assign each cell to the nearest cluster center

3. **Update**: Recompute cluster centers as means of assigned cells

4. **Iteration**: Repeat steps 2-3 until convergence

5. **Multiple Runs**: Run with different initializations, keep best result

**Advantages:**
- Simple and fast
- Works well with spherical clusters
- Deterministic given initialization

**Limitations:**
- Requires specifying number of clusters
- Assumes clusters are spherical
- Sensitive to initialization (mitigated by n_init)

**Citation:**
- MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1, 281-297.
- Implementation: `scikit-learn KMeans <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_

HDBSCAN
~~~~~~~

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) identifies clusters of varying density and handles noise.

**How it works:**

1. **Mutual Reachability Graph**: Builds a graph based on mutual reachability distance
   - Accounts for local density variations

2. **Minimum Spanning Tree**: Constructs MST from the graph

3. **Hierarchical Clustering**: Performs hierarchical clustering on the MST

4. **Cluster Selection**: Extracts clusters from the hierarchy using selection method
   - EOM (Excess of Mass): More stable, recommended
   - Leaf: More clusters, may be less stable

5. **Noise Assignment**: Cells not meeting min_cluster_size are marked as noise (-1, converted to 0)

**Advantages:**
- Identifies clusters of varying density
- Handles noise/outliers automatically
- Does not require specifying number of clusters
- Robust to outliers

**Limitations:**
- Slower than graph-based methods
- Parameters (min_cluster_size, min_samples) need tuning
- May mark many cells as noise if parameters are too strict

**Citation:**
- McInnes, L., et al. (2017). "Accelerated Hierarchical Density Based Clustering." 2017 IEEE International Conference on Data Mining Workshops (ICDMW), 33-42. `DOI: 10.1109/ICDMW.2017.12 <https://doi.org/10.1109/ICDMW.2017.12>`_
- `HDBSCAN Python Package <https://github.com/scikit-learn-contrib/hdbscan>`_

Tips and Best Practices
-----------------------

1. **Method Selection**:
   - Use **Leiden** for most cases (fast, scalable, good results)
   - Use **Hierarchical** for small datasets or when you need hierarchical structure
   - Use **K-means** when you know the number of clusters and they are spherical
   - Use **HDBSCAN** when you have varying density clusters or want automatic noise detection

2. **Feature Selection**: 
   - Use intensity features (marker expression) as primary features
   - Optionally include morphological features if they are informative
   - Exclude features that are not biologically relevant

3. **Scaling**: 
   - Always use scaling (zscore or mad) unless features are already on the same scale
   - Z-score is standard, MAD is more robust to outliers

4. **Parameter Tuning**:
   - For Leiden/Louvain: Start with resolution=1.0, adjust based on cluster number
   - For Hierarchical/K-means: Use elbow plot or domain knowledge to determine k
   - For HDBSCAN: Adjust min_cluster_size based on expected cluster sizes

5. **Validation**: 
   - Visualize clusters in 2D (e.g., UMAP, t-SNE) to assess quality
   - Check that clusters are biologically meaningful
   - Verify that known cell types are separated

6. **Resolution Parameter (Leiden/Louvain)**: 
   - Lower resolution → fewer, larger clusters
   - Higher resolution → more, smaller clusters
   - Adjust iteratively to find optimal granularity

7. **Downstream Analysis**:
   
   - Use cluster assignments for:
   
     - Cell type annotation
     - Differential expression analysis
     - Spatial analysis
     - Population comparisons

Clustering Visualizations
==========================

After clustering, OpenIMC provides multiple visualization options to explore and interpret cluster results. All visualizations are accessible from the clustering dialog after running clustering.

Available Visualizations
-------------------------

1. **Heatmap**: Shows feature expression patterns across clusters
2. **UMAP**: 2D embedding colored by various attributes
3. **t-SNE**: 2D embedding colored by various attributes (if scikit-learn is installed)
4. **Differential Expression**: Heatmap of top markers per cluster
5. **Stacked Bars**: Cluster composition by grouping variable
6. **Boxplot/Violin Plot**: Distribution of marker expression by cluster

Heatmap Visualization
---------------------

The heatmap is the default visualization after clustering. It displays feature expression patterns across all cells, with cells grouped by cluster.

**Parameters:**

- **Heatmap of**: Choose between "Clusters" (default) or "Manual Gates" (if manual phenotypes are assigned)
- **Scaling**: Method for normalizing features before display
  - ``"Z-score"``: Z-score normalization (default)
  - ``"MAD (Median Absolute Deviation)"``: Robust normalization
  - ``"None (no scaling)"``: Raw feature values
- **Colormap**: Color scheme for the heatmap
  - ``"RdBu_r"``: Red-White-Blue (default, good for z-scored data)
  - ``"viridis"``: Purple-Green-Yellow
  - ``"plasma"``: Purple-Pink-Yellow
  - ``"inferno"``: Purple-Red-Yellow
  - ``"Blues"``, ``"Reds"``, ``"Greens"``, ``"Oranges"``, ``"Purples"``: Sequential colormaps
- **Filter**: Select which clusters/phenotypes to display (click "Filter…" button)
- **Feature label font size**: Font size for feature labels on y-axis (default: 8, range: 4-20)
- **Patient annotation**: Show patient/source file annotation bar above cells
  - Enable/disable with checkbox
  - Select annotation column (source_file, batch_group, source_well)
  - Customize patient labels (click "Customize Patient Labels…")
  - Customize legend label (e.g., "Patient", "Sample", "Source")

**Customization:**

- Click **"Configure Plot"** button to open the plot configuration dialog
- Customize feature labels (click "Customize Feature Labels…" to set friendly names)
- Adjust dendrogram linkage method (automatic, based on data)

**Export:**

- Click **"Save Plot"** button to export the heatmap
- Options: PNG, JPG, or PDF format
- Adjustable DPI (default: 300)
- Optional font size override
- Optional figure size override

UMAP Visualization
------------------

UMAP (Uniform Manifold Approximation and Projection) provides a 2D embedding of the high-dimensional feature space, useful for visualizing cluster structure.

**Parameters:**

- **Color by**: Select one or more attributes to color points (multi-select for faceted plots)
  - ``"Cluster"``: Color by cluster assignment (default)
  - ``"Source File"``: Color by source file/patient (visualize batch effects)
  - ``"Phenotype"``: Color by cluster phenotype (if annotated)
  - ``"Manual Phenotype"``: Color by manual phenotype (if assigned)
  - Feature columns: Color by continuous feature expression (e.g., marker intensities)
- **Point size**: Size of points in scatter plot (default: 18, range: 1-200)
- **Point alpha**: Transparency of points (default: 0.8, range: 0.0-1.0)
  - 0.0 = fully transparent
  - 1.0 = fully opaque
- **Remake UMAP**: Regenerate UMAP with different parameters
  - Select features to use
  - Choose scaling method
  - Set n_neighbors parameter (default: 15)

**Faceted Plotting:**

- Select multiple "Color by" options to create side-by-side plots (up to 3 plots)
- Useful for comparing different coloring schemes

**Export:**

- Click **"Save Plot"** button to export
- Same export options as heatmap (format, DPI, font size, figure size)

t-SNE Visualization
-------------------

t-SNE (t-Distributed Stochastic Neighbor Embedding) provides an alternative 2D embedding method. Requires scikit-learn to be installed.

**Parameters:**

- Same as UMAP:
  - **Color by**: Multi-select for faceted plots
  - **Point size**: Default 18, range 1-200
  - **Point alpha**: Default 0.8, range 0.0-1.0

**Note:** t-SNE is computationally more expensive than UMAP and may take longer for large datasets.

**Export:**

- Same export options as other visualizations

Differential Expression Visualization
--------------------------------------

Shows a heatmap of the top N markers per cluster, highlighting which features are most characteristic of each cluster.

**Parameters:**

- **Top N**: Number of top markers to show per cluster (default: 5, range: 1-20)
- **Colormap**: Color scheme (same options as heatmap)
  - Default: ``"RdBu_r"`` (Red-White-Blue)
- **Feature labels**: Customize feature display names (click "Customize Feature Labels…")

**How it works:**

1. Calculates mean expression per cluster for each feature
2. Computes z-scores: (cluster_mean - overall_mean) / overall_std
3. For each cluster, selects top N features with highest z-scores
4. Displays as heatmap with z-scores color-coded
5. Highlights top N markers for each cluster with black boxes

**Interpretation:**

- Positive z-scores (red): Feature is higher in this cluster than average
- Negative z-scores (blue): Feature is lower in this cluster than average
- Black boxes: Top N markers for each cluster
- Z-score values are displayed as text annotations

**Export:**

- Same export options as other visualizations

Stacked Bars Visualization
---------------------------

Shows the composition of clusters across different groups (e.g., ROIs, conditions, patients).

**Parameters:**

- **Group by**: Select grouping variable
  - ``"acquisition_id"``: Group by acquisition
  - ``"source_file"``: Group by source file/patient
  - ``"batch_group"``: Group by batch
  - ``"source_well"``: Group by well
  - Other available grouping columns
- **Feature labels**: Customize feature display names (optional)

**How it works:**

1. Groups cells by the selected grouping variable
2. Calculates the fraction of each cluster within each group
3. Displays as stacked bar chart
4. Each bar represents one group
5. Colors correspond to clusters (consistent with other visualizations)

**Interpretation:**

- Bar height: Total number of cells in that group
- Segment height: Fraction of cells in that cluster
- Useful for comparing cluster frequencies across conditions/patients

**Export:**

- Same export options as other visualizations

Boxplot/Violin Plot Visualization
-----------------------------------

Shows the distribution of marker expression values across clusters, useful for identifying marker-specific differences.

**Parameters:**

- **Markers**: Select markers to visualize (click "Select Markers…" button)
  - Multi-select dialog with all available markers
  - Can select multiple markers for faceted plots
- **Plot type**: Choose between "Violin Plot" (default) or "Boxplot"
  - **Violin Plot**: Shows full distribution shape (KDE)
  - **Boxplot**: Shows quartiles, median, and outliers
- **Statistical testing**: Enable to perform pairwise comparisons
  
  - **Test mode**:
  
    - ``"Pairwise (all pairs)"``: Compare all cluster pairs
    - ``"One vs Others"``: Compare one cluster against all others
  
  - **Reference cluster**: Select cluster for "One vs Others" mode
  - **Export Statistical Results**: Export p-values to CSV

**Statistical Testing:**

- Uses Mann-Whitney U test (non-parametric, two-sided)
- Applies Benjamini-Hochberg (BH) correction for multiple testing
- Significance levels:
  - ``***``: p < 0.001
  - ``**``: p < 0.01
  - ``*``: p < 0.05
  - ``ns``: not significant (p ≥ 0.05)
- Significance bars are drawn between significantly different clusters
- Only significant results (adjusted p < 0.05) are shown

**Export:**

- **Save Plot**: Export visualization (same options as other plots)
- **Export Statistical Results**: Export test results to CSV
  - Columns: Marker, Cluster_1, Cluster_2, P_value, Adjusted_P_value_BH, Significant, Significance_level

**Customization:**

- Feature labels can be customized (click "Customize Feature Labels…")
- Colors match cluster colors from other visualizations

Exporting Plots
---------------

All visualizations can be exported using the **"Save Plot"** button.

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

1. Generate the desired visualization
2. Adjust any parameters (colormap, point size, etc.)
3. Click **"Save Plot"** button
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
- Use "Override figure size" to create square plots or specific aspect ratios

Customizing Font Sizes
-----------------------

Font sizes can be customized in several ways:

**1. Heatmap Feature Labels:**

- In the clustering dialog, use the **"Configure Plot"** button
- In the plot configuration dialog, adjust "Feature label font size"
- Range: 4-20 points (default: 8)
- Applies to y-axis feature labels in heatmap

**2. Export Font Size Override:**

- When saving a plot, check "Override figure font size"
- Set global font size (applies to all text elements)
- Range: 6.0-72.0 points (default: 10.0)
- Useful for making text larger/smaller for publications

**3. Plot-Specific Font Sizes:**

- Most plots use default font sizes optimized for display
- UMAP/t-SNE: Axis labels (10pt), titles (12pt), legends (8pt)
- Differential Expression: Feature labels (9-10pt), annotations (8-10pt)
- Boxplot/Violin: Axis labels (9-10pt), titles (10-12pt)

**Note:** Font size customization is most important for heatmaps (many feature labels) and publication figures.

Customizing Feature Labels
---------------------------

All visualizations support custom feature labels for better readability.

**How to Customize:**

1. Click **"Customize Feature Labels…"** button (available in all views)
2. In the dialog, set friendly display names for features
   - Example: ``"CD3_1841_mean"`` → ``"CD3 Mean"``
   - Example: ``"Vimentin_mean"`` → ``"Mean Vimentin"``
3. Labels are saved and used in all visualizations
4. Original feature names are preserved in data exports

**Use Cases:**

- Shorten long feature names
- Add units or descriptions
- Use consistent naming conventions
- Improve readability in heatmaps and differential expression plots

Accessing Visualizations in the GUI
------------------------------------

1. **Run Clustering**: Complete clustering analysis first
2. **Open Clustering Dialog**: Navigate to **Analysis → Cell Clustering…**
3. **Select View**: Use the dropdown at the bottom of the dialog
   - Options: "Heatmap", "UMAP", "t-SNE", "Stacked Bars", "Differential Expression", "Boxplot/Violin Plot"
4. **Adjust Parameters**: Use controls visible for the selected view
5. **Configure Plot**: Click **"Configure Plot"** button for advanced settings (heatmap only)
6. **Export**: Click **"Save Plot"** to export the current visualization

**View-Specific Controls:**

- Controls are automatically shown/hidden based on the selected view
- Heatmap: Source, scaling, filter, patient annotation, colormap
- UMAP/t-SNE: Color by, point size, point alpha, remake UMAP (UMAP only)
- Stacked Bars: Group by
- Differential Expression: Top N, colormap
- Boxplot/Violin: Marker selection, plot type, statistical testing

Tips and Best Practices for Visualizations
-------------------------------------------

1. **Heatmap:**
   - Use Z-score scaling for comparing across features
   - Filter clusters to focus on specific populations
   - Enable patient annotation to visualize batch effects
   - Adjust feature label font size for readability

2. **UMAP/t-SNE:**
   - Use "Cluster" coloring to assess cluster separation
   - Use "Source File" coloring to check for batch effects
   - Adjust point size and alpha for dense plots
   - Use faceted plots to compare multiple colorings

3. **Differential Expression:**
   - Adjust "Top N" to show more/fewer markers
   - Use RdBu_r colormap for z-scored data
   - Look for black boxes highlighting top markers

4. **Stacked Bars:**
   - Choose grouping variable based on your experimental design
   - Useful for comparing cluster frequencies across conditions

5. **Boxplot/Violin Plot:**
   - Select informative markers (known cell type markers)
   - Use violin plots to see full distribution shape
   - Enable statistical testing for publication figures
   - Export statistical results for detailed analysis

6. **Export:**
   - Use PDF for publications (vector graphics)
   - Use PNG at 300 DPI for presentations
   - Adjust font sizes for small figures
   - Save multiple formats if needed

