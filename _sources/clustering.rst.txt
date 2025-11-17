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
     resolution: 1.0
     n_neighbors: 15
     metric: "euclidean"
     seed: 42

Method Details
--------------

Leiden Algorithm
~~~~~~~~~~~~~~~

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

