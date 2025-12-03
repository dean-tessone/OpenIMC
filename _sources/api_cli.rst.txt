CLI API Reference
==================

The OpenIMC command-line interface provides batch processing capabilities without the GUI.
All commands can be accessed via the ``openimc`` command-line tool.

Overview
--------

The CLI supports 22 commands covering the complete OpenIMC analysis workflow:

**Data Processing:**
- ``preprocess`` - Denoise and export images to OME-TIFF format
- ``segment`` - Perform cell segmentation using various methods
- ``extract-features`` - Extract morphological and intensity features from segmented cells
- ``deconvolution`` - Apply deconvolution to high resolution IMC images

**Analysis:**
- ``cluster`` - Perform clustering analysis on feature data
- ``spatial`` - Perform spatial analysis including graph construction and community detection
- ``spatial-anndata`` - Build spatial graph using AnnData/squidpy approach
- ``spatial-enrichment`` - Compute pairwise spatial enrichment between clusters
- ``spatial-distance`` - Compute distance distributions between clusters
- ``spatial-nhood-enrichment`` - Run neighborhood enrichment analysis using squidpy
- ``spatial-cooccurrence`` - Run co-occurrence analysis using squidpy
- ``spatial-autocorr`` - Run spatial autocorrelation (Moran's I) analysis using squidpy
- ``spatial-ripley`` - Run Ripley function analysis using squidpy

**Quality Control & Correction:**
- ``batch-correction`` - Apply batch correction methods (Harmony, ComBat)
- ``pixel-correlation`` - Compute pixel-level correlations between markers
- ``qc-analysis`` - Perform quality control analysis
- ``spillover-correction`` - Apply spillover correction to feature data
- ``generate-spillover-matrix`` - Generate spillover matrix from single-stain controls

**Export & Visualization:**
- ``export-anndata`` - Export AnnData objects to H5AD file(s)
- ``cluster-figures`` - Generate cluster visualization figures
- ``spatial-figures`` - Generate spatial analysis visualization figures

**Workflow:**
- ``workflow`` - Execute a complete workflow from a YAML configuration file

Main Entry Point
----------------

.. autofunction:: openimc.cli.main

Command Functions
-----------------

Preprocessing
~~~~~~~~~~~~~

.. autofunction:: openimc.cli.preprocess_command

**Example Usage:**

.. code-block:: bash

    # Basic preprocessing
    openimc preprocess input.mcd output/

    # With denoising on all channels
    openimc preprocess input.mcd output/ --denoise all --denoise-method median3

    # With custom denoise settings per channel
    openimc preprocess input.mcd output/ --denoise-settings denoise_config.json

**Parameters:**
- ``input`` - Input MCD file or OME-TIFF directory (required)
- ``output`` - Output directory for processed OME-TIFF files (required)
- ``--denoise`` - Apply denoising to all channels (use "all" for hot pixel removal)
- ``--denoise-method`` - Hot pixel removal method: ``median3`` (3x3 median filter) or ``n_sd_local_median`` (default: ``median3``)
- ``--denoise-n-sd`` - Number of standard deviations for ``n_sd_local_median`` method (default: 5.0)
- ``--denoise-settings`` - JSON file or string with denoise settings per channel
- ``--channel-format`` - Channel format for OME-TIFF files: ``CHW`` or ``HWC`` (default: ``CHW``)
- ``--workers`` - Number of parallel workers (default: auto)

**Note:** Arcsinh normalization is not applied to exported images. Use during feature extraction instead.

Segmentation
~~~~~~~~~~~~

.. autofunction:: openimc.cli.segment_command

**Example Usage:**

.. code-block:: bash

    # Cellpose segmentation
    openimc segment input.mcd output/ --method cellpose --nuclear-channels DAPI --model cyto3 --gpu-id 0

    # Watershed segmentation (requires both channels)
    openimc segment input.mcd output/ --method watershed --nuclear-channels DNA1 --cytoplasm-channels CK8_CK18

    # CellSAM segmentation (requires API key)
    openimc segment input.mcd output/ --method cellsam --nuclear-channels DAPI --deepcell-api-key YOUR_KEY

**Parameters:**
- ``input`` - Input MCD file or OME-TIFF directory (required)
- ``output`` - Output directory for segmentation masks (required)
- ``--method`` - Segmentation method: ``cellsam``, ``cellpose``, or ``watershed`` (default: ``cellsam``)
- ``--nuclear-channels`` - Comma-separated list of nuclear channel names (required)
- ``--cytoplasm-channels`` - Comma-separated list of cytoplasm channel names (optional, for cyto3 model)
- ``--nuclear-fusion-method`` - Method to combine nuclear channels: ``single``, ``mean``, ``weighted``, ``max``, ``pca1`` (default: ``mean``)
- ``--cyto-fusion-method`` - Method to combine cytoplasm channels (default: ``mean``)
- ``--nuclear-weights`` - Comma-separated weights for nuclear channels (e.g., "0.5,0.3,0.2")
- ``--cyto-weights`` - Comma-separated weights for cytoplasm channels
- ``--model`` - Cellpose model type: ``cyto3`` or ``nuclei`` (default: ``cyto3``)
- ``--diameter`` - Cell diameter in pixels (Cellpose, optional)
- ``--flow-threshold`` - Flow threshold (Cellpose, default: 0.4)
- ``--cellprob-threshold`` - Cell probability threshold (Cellpose, default: 0.0)
- ``--gpu-id`` - GPU ID to use (Cellpose, optional)
- ``--min-cell-area`` - Minimum cell area in pixels (watershed, default: 100)
- ``--max-cell-area`` - Maximum cell area in pixels (watershed, default: 10000)
- ``--compactness`` - Watershed compactness (default: 0.01)
- ``--deepcell-api-key`` - DeepCell API key (CellSAM, can also use DEEPCELL_ACCESS_TOKEN env var)
- ``--bbox-threshold`` - Bbox threshold for CellSAM (default: 0.4, lower for faint cells: 0.01-0.1)
- ``--use-wsi`` - Use WSI mode for CellSAM (for ROIs with >500 cells)
- ``--low-contrast-enhancement`` - Enable low contrast enhancement for CellSAM
- ``--gauge-cell-size`` - Enable gauge cell size for CellSAM (runs twice: estimates error, then returns mask)
- ``--arcsinh`` - Apply arcsinh normalization before segmentation
- ``--arcsinh-cofactor`` - Arcsinh cofactor (default: 1.0)
- ``--denoise`` - Apply denoising to all channels
- ``--denoise-method`` - Denoising method (default: ``median3``)
- ``--denoise-n-sd`` - Number of standard deviations for denoising (default: 5.0)
- ``--denoise-settings`` - JSON file or string with per-channel denoise settings
- ``--workers`` - Number of parallel workers (default: auto)

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.extract_features_command

**Example Usage:**

.. code-block:: bash

    # Extract all features
    openimc extract-features input.mcd features.csv --mask output/masks/ --morphological --intensity

    # Extract with denoising
    openimc extract-features input.mcd features.csv --mask output/masks/ --denoise all --denoise-method median3

    # Extract with arcsinh transformation
    openimc extract-features input.mcd features.csv --mask output/masks/ --arcsinh --arcsinh-cofactor 1.0

**Parameters:**
- ``input`` - Input MCD file or OME-TIFF directory (required)
- ``output`` - Output CSV file path (required)
- ``--mask`` - Path to segmentation mask directory or single mask file (.tif, .tiff, or .npy) (required)
- ``--acquisition`` - Acquisition ID or name (uses first if not specified)
- ``--morphological`` - Extract morphological features (default: True if neither specified)
- ``--intensity`` - Extract intensity features (default: True if neither specified)
- ``--arcsinh`` - Apply arcsinh transformation to extracted intensity features
- ``--arcsinh-cofactor`` - Arcsinh cofactor (default: 1.0)
- ``--denoise`` - Apply denoising to all channels
- ``--denoise-method`` - Denoising method (default: ``median3``)
- ``--denoise-n-sd`` - Number of standard deviations for denoising (default: 5.0)
- ``--denoise-settings`` - JSON file or string with per-channel denoise settings
- ``--channel-format`` - Channel format for OME-TIFF files (default: ``CHW``)
- ``--workers`` - Number of parallel workers (default: auto)

**Note:** The mask can be a directory (masks matched to acquisitions by filename) or a single mask file.

Clustering
~~~~~~~~~~

.. autofunction:: openimc.cli.cluster_command

**Example Usage:**

.. code-block:: bash

    # Leiden clustering
    openimc cluster features.csv clustered.csv --method leiden --resolution 1.0

    # Hierarchical clustering
    openimc cluster features.csv clustered.csv --method hierarchical --n-clusters 10 --linkage ward

    # HDBSCAN clustering
    openimc cluster features.csv clustered.csv --method hdbscan --min-cluster-size 10

    # K-means clustering
    openimc cluster features.csv clustered.csv --method kmeans --n-clusters 8

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``output`` - Output CSV file with cluster labels (required)
- ``--method`` - Clustering method: ``hierarchical``, ``leiden``, ``louvain``, ``kmeans``, or ``hdbscan`` (default: ``leiden``)
- ``--columns`` - Comma-separated list of columns to use for clustering (auto-detect if not specified)
- ``--scaling`` - Feature scaling method: ``none``, ``zscore``, or ``mad`` (default: ``zscore``)
- ``--n-clusters`` - Number of clusters (required for hierarchical and kmeans)
- ``--linkage`` - Linkage method for hierarchical clustering: ``ward``, ``complete``, or ``average`` (default: ``ward``)
- ``--resolution`` - Resolution parameter for Leiden clustering (default: 1.0)
- ``--n-neighbors`` - Number of neighbors for k-NN graph (Leiden/Louvain only, default: 15)
- ``--metric`` - Distance metric for k-NN graph: ``euclidean``, ``manhattan``, or ``cosine`` (default: ``euclidean``)
- ``--use-jaccard`` - Use Jaccard similarity for edge weights (PhenoGraph-like, Leiden/Louvain only)
- ``--n-init`` - Number of initializations for K-means (default: 10)
- ``--min-cluster-size`` - Minimum cluster size (HDBSCAN, default: 10)
- ``--min-samples`` - Minimum samples (HDBSCAN, default: 5)
- ``--cluster-selection-method`` - Cluster selection method for HDBSCAN: ``eom`` or ``leaf`` (default: ``eom``)
- ``--hdbscan-metric`` - Distance metric for HDBSCAN: ``euclidean`` or ``manhattan`` (default: ``euclidean``)
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Analysis
~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_command

**Example Usage:**

.. code-block:: bash

    # Basic spatial graph construction
    openimc spatial features.csv edges.csv --method kNN --k-neighbors 10

    # Radius-based spatial graph
    openimc spatial features.csv edges.csv --method Radius --radius 50 --k-neighbors 10

    # With community detection
    openimc spatial features.csv edges.csv --method kNN --k-neighbors 10 --detect-communities

**Parameters:**
- ``features`` - Input CSV file with features (must contain ``centroid_x``, ``centroid_y``) (required)
- ``output`` - Output CSV file with spatial graph edges (required)
- ``--method`` - Graph construction method: ``kNN``, ``Radius``, or ``Delaunay`` (default: ``kNN``)
- ``--radius`` - Maximum distance for edges in pixels (required for Radius method)
- ``--k-neighbors`` - k for k-nearest neighbors (default: 10, used for kNN method)
- ``--pixel-size-um`` - Pixel size in micrometers (default: 1.0, used for distance_um conversion)
- ``--detect-communities`` - Also detect spatial communities
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--workers`` - Number of parallel workers (default: auto)

Batch Correction
~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.batch_correction_command

**Example Usage:**

.. code-block:: bash

    # Harmony batch correction
    openimc batch-correction features.csv corrected.csv --method harmony --batch-var batch_id

    # ComBat batch correction
    openimc batch-correction features.csv corrected.csv --method combat --batch-var batch_id --covariates condition

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``output`` - Output CSV file with corrected features (required)
- ``--method`` - Batch correction method: ``combat`` or ``harmony`` (default: ``harmony``)
- ``--batch-var`` - Column name containing batch identifiers (auto-detected if not specified)
- ``--columns`` - Comma-separated list of feature columns to correct (auto-detected if not specified)
- ``--covariates`` - Comma-separated list of covariate columns (ComBat only)
- ``--n-clusters`` - Number of Harmony clusters (default: 30)
- ``--sigma`` - Width of soft kmeans clusters for Harmony (default: 0.1)
- ``--theta`` - Diversity clustering penalty parameter for Harmony (default: 2.0)
- ``--lambda-reg`` - Regularization parameter for Harmony (default: 1.0)
- ``--max-iter`` - Maximum iterations for Harmony (default: 20)
- ``--pca-variance`` - Proportion of variance to retain in PCA for Harmony (default: 0.9)
- ``--workers`` - Number of parallel workers (default: auto)

Pixel Correlation
~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.pixel_correlation_command

**Example Usage:**

.. code-block:: bash

    # Analyze all channels
    openimc pixel-correlation input.mcd correlations.csv

    # Analyze specific channels
    openimc pixel-correlation input.mcd correlations.csv --channels CD3,CD4,CD8

    # With multiple testing correction
    openimc pixel-correlation input.mcd correlations.csv --multiple-testing-correction fdr_bh

**Parameters:**
- ``input`` - Input MCD file or OME-TIFF directory (required)
- ``output`` - Output CSV file with correlation results (required)
- ``--channel-format`` - Channel format for OME-TIFF files (default: ``CHW``)
- ``--acquisition`` - Acquisition ID or name (uses first if not specified)
- ``--channels`` - Comma-separated list of channels to analyze (uses all if not specified)
- ``--mask`` - Path to segmentation mask file (optional, for within-cell analysis)
- ``--multiple-testing-correction`` - Multiple testing correction method: ``bonferroni``, ``fdr_bh``, or ``fdr_by``
- ``--workers`` - Number of parallel workers (default: auto)

Quality Control Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.qc_analysis_command

**Example Usage:**

.. code-block:: bash

    # Pixel-level QC analysis
    openimc qc-analysis input.mcd qc_results.csv --mode pixel

    # Cell-level QC analysis (requires mask)
    openimc qc-analysis input.mcd qc_results.csv --mode cell --mask masks/roi1.tif

**Parameters:**
- ``input`` - Input MCD file or OME-TIFF directory (required)
- ``output`` - Output CSV file with QC metrics (required)
- ``--channel-format`` - Channel format for OME-TIFF files (default: ``CHW``)
- ``--acquisition`` - Acquisition ID or name (uses first if not specified)
- ``--channels`` - Comma-separated list of channels to analyze (uses all if not specified)
- ``--mode`` - Analysis mode: ``pixel`` or ``cell`` (default: ``pixel``)
- ``--mask`` - Path to segmentation mask file (required for cell mode)
- ``--workers`` - Number of parallel workers (default: auto)

Spillover Correction
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spillover_correction_command

**Example Usage:**

.. code-block:: bash

    # Apply spillover correction using NNLS
    openimc spillover-correction features.csv spillover_matrix.csv corrected.csv --method nnls

    # Apply spillover correction using PGD
    openimc spillover-correction features.csv spillover_matrix.csv corrected.csv --method pgd --arcsinh-cofactor 1.0

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``spillover_matrix`` - Path to spillover matrix CSV file (required)
- ``output`` - Output CSV file with corrected features (required)
- ``--method`` - Compensation method: ``nnls`` or ``pgd`` (default: ``pgd``)
- ``--arcsinh-cofactor`` - Optional cofactor for arcsinh transformation
- ``--workers`` - Number of parallel workers (default: auto)

Generate Spillover Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.generate_spillover_matrix_command

**Example Usage:**

.. code-block:: bash

    # Generate spillover matrix from single-stain controls
    openimc generate-spillover-matrix controls.mcd spillover_matrix.csv

    # With donor mapping
    openimc generate-spillover-matrix controls.mcd spillover_matrix.csv --donor-map donor_map.json

**Parameters:**
- ``input`` - Input MCD file with single-stain controls (required)
- ``output`` - Output CSV file for spillover matrix (required)
- ``--donor-map`` - JSON file or string mapping acquisition IDs to donor channel names
- ``--cap`` - Maximum spillover coefficient (default: 0.3)
- ``--aggregate`` - Aggregation method: ``median`` or ``mean`` (default: ``median``)
- ``--workers`` - Number of parallel workers (default: auto)

Deconvolution
~~~~~~~~~~~~

.. autofunction:: openimc.cli.deconvolution_command

**Example Usage:**

.. code-block:: bash

    # Basic deconvolution
    openimc deconvolution input.mcd output/ --x0 7.0 --iterations 4

    # With specific acquisition
    openimc deconvolution input.mcd output/ --acquisition ROI1 --x0 7.0 --iterations 4

**Parameters:**
- ``input`` - Input MCD file or OME-TIFF directory (required)
- ``output`` - Output directory for deconvolved images (required)
- ``--channel-format`` - Channel format for OME-TIFF files (default: ``CHW``)
- ``--acquisition`` - Acquisition ID or name (processes all if not specified)
- ``--x0`` - Parameter for kernel calculation (default: 7.0)
- ``--iterations`` - Number of Richardson-Lucy iterations (default: 4)
- ``--output-format`` - Output format: ``float`` or ``uint16`` (default: ``float``)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Enrichment
~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_enrichment_command

**Example Usage:**

.. code-block:: bash

    # Compute spatial enrichment between clusters
    openimc spatial-enrichment features.csv edges.csv enrichment.csv --n-permutations 100

**Parameters:**
- ``features`` - Input CSV file with features and cluster labels (required)
- ``edges`` - Input CSV file with spatial graph edges (required)
- ``output`` - Output CSV file with enrichment results (required)
- ``--cluster-column`` - Column name containing cluster labels (default: ``cluster``)
- ``--n-permutations`` - Number of permutations for null distribution (default: 100)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Distance
~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_distance_command

**Example Usage:**

.. code-block:: bash

    # Compute distance distributions between clusters
    openimc spatial-distance features.csv edges.csv distances.csv

**Parameters:**
- ``features`` - Input CSV file with features and cluster labels (required)
- ``edges`` - Input CSV file with spatial graph edges (required)
- ``output`` - Output CSV file with distance distribution results (required)
- ``--cluster-column`` - Column name containing cluster labels (default: ``cluster``)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial AnnData
~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_anndata_command

**Example Usage:**

.. code-block:: bash

    # Build spatial graph using AnnData/squidpy
    openimc spatial-anndata features.csv --output anndata.h5ad --method kNN --k-neighbors 20

    # Radius-based with specific ROI
    openimc spatial-anndata features.csv --output anndata.h5ad --method Radius --radius 50 --roi-id ROI1

**Parameters:**
- ``features`` - Input CSV file with features (must contain ``centroid_x``, ``centroid_y``) (required)
- ``--output`` - Output H5AD file or directory for AnnData objects
- ``--method`` - Graph construction method: ``kNN``, ``Radius``, or ``Delaunay`` (default: ``kNN``)
- ``--radius`` - Maximum distance for edges in micrometers (required for Radius method)
- ``--k-neighbors`` - k for k-nearest neighbors (default: 20, used for kNN method)
- ``--pixel-size-um`` - Pixel size in micrometers (default: 1.0)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--roi-id`` - Specific ROI ID to process (processes all if not specified)
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--combined`` - Export as single combined file (default: separate files per ROI)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Neighborhood Enrichment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_nhood_enrichment_command

**Example Usage:**

.. code-block:: bash

    # Run neighborhood enrichment analysis
    openimc spatial-nhood-enrichment features.csv --output enrichment.h5ad --cluster-column cluster

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``--output`` - Output H5AD file or directory for results
- ``--method`` - Graph construction method: ``kNN``, ``Radius``, or ``Delaunay`` (default: ``kNN``)
- ``--radius`` - Maximum distance for edges in micrometers (required for Radius method)
- ``--k-neighbors`` - k for k-nearest neighbors (default: 20)
- ``--pixel-size-um`` - Pixel size in micrometers (default: 1.0)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--roi-id`` - Specific ROI ID to process (processes all if not specified)
- ``--cluster-column`` - Column name containing cluster labels (default: ``cluster``)
- ``--aggregation`` - Aggregation method for multiple ROIs: ``mean`` or ``sum`` (default: ``mean``)
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--combined`` - Export as single combined file (default: separate files per ROI)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Co-occurrence
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_cooccurrence_command

**Example Usage:**

.. code-block:: bash

    # Run co-occurrence analysis
    openimc spatial-cooccurrence features.csv --output cooccurrence.h5ad --interval 10,20,30,50,100

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``--output`` - Output H5AD file or directory for results
- ``--method`` - Graph construction method: ``kNN``, ``Radius``, or ``Delaunay`` (default: ``kNN``)
- ``--radius`` - Maximum distance for edges in micrometers (required for Radius method)
- ``--k-neighbors`` - k for k-nearest neighbors (default: 20)
- ``--pixel-size-um`` - Pixel size in micrometers (default: 1.0)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--roi-id`` - Specific ROI ID to process (processes all if not specified)
- ``--cluster-column`` - Column name containing cluster labels (default: ``cluster``)
- ``--interval`` - Comma-separated distances in micrometers (default: ``10,20,30,50,100``)
- ``--reference-cluster`` - Optional reference cluster for co-occurrence
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--combined`` - Export as single combined file (default: separate files per ROI)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Autocorrelation
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_autocorr_command

**Example Usage:**

.. code-block:: bash

    # Run spatial autocorrelation for all markers
    openimc spatial-autocorr features.csv --output autocorr.h5ad

    # Run for specific markers
    openimc spatial-autocorr features.csv --output autocorr.h5ad --markers CD3,CD4,CD8

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``--output`` - Output H5AD file or directory for results
- ``--method`` - Graph construction method: ``kNN``, ``Radius``, or ``Delaunay`` (default: ``kNN``)
- ``--radius`` - Maximum distance for edges in micrometers (required for Radius method)
- ``--k-neighbors`` - k for k-nearest neighbors (default: 20)
- ``--pixel-size-um`` - Pixel size in micrometers (default: 1.0)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--roi-id`` - Specific ROI ID to process (processes all if not specified)
- ``--markers`` - Comma-separated list of markers to analyze, or ``all`` (default: ``all``)
- ``--aggregation`` - Aggregation method for multiple ROIs: ``mean`` or ``sum`` (default: ``mean``)
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--combined`` - Export as single combined file (default: separate files per ROI)
- ``--workers`` - Number of parallel workers (default: auto)

Spatial Ripley
~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_ripley_command

**Example Usage:**

.. code-block:: bash

    # Run Ripley L function analysis
    openimc spatial-ripley features.csv --output ripley.h5ad --mode L --max-dist 50.0

**Parameters:**
- ``features`` - Input CSV file with features (required)
- ``--output`` - Output H5AD file or directory for results
- ``--method`` - Graph construction method: ``kNN``, ``Radius``, or ``Delaunay`` (default: ``kNN``)
- ``--radius`` - Maximum distance for edges in micrometers (required for Radius method)
- ``--k-neighbors`` - k for k-nearest neighbors (default: 20)
- ``--pixel-size-um`` - Pixel size in micrometers (default: 1.0)
- ``--roi-column`` - Column name for ROI grouping (auto-detected if not specified)
- ``--roi-id`` - Specific ROI ID to process (processes all if not specified)
- ``--cluster-column`` - Column name containing cluster labels (default: ``cluster``)
- ``--mode`` - Ripley function mode: ``F``, ``G``, or ``L`` (default: ``L``)
- ``--max-dist`` - Maximum distance in micrometers (default: 50.0)
- ``--seed`` - Random seed for reproducibility (default: 42)
- ``--combined`` - Export as single combined file (default: separate files per ROI)
- ``--workers`` - Number of parallel workers (default: auto)

Export AnnData
~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.export_anndata_command

**Example Usage:**

.. code-block:: bash

    # Export single H5AD file
    openimc export-anndata input.h5ad output.h5ad

    # Export directory of H5AD files
    openimc export-anndata input_dir/ output_dir/

    # Combine multiple files
    openimc export-anndata input_dir/ output.h5ad --combined

**Parameters:**
- ``input`` - Input H5AD file or directory containing H5AD files (required)
- ``output`` - Output H5AD file (if combined) or directory (if separate) (required)
- ``--combined`` - Export as single combined file (default: separate files per ROI)
- ``--workers`` - Number of parallel workers (default: auto)

Cluster Figures
~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.cluster_figures_command

**Example Usage:**

.. code-block:: bash

    # Generate cluster visualization figures
    openimc cluster-figures clustered_features.csv output/figures/ --dpi 300 --font-size 12

**Parameters:**
- ``features`` - Input CSV file with clustered features (must contain ``cluster`` column) (required)
- ``output`` - Output directory for figures (required)
- ``--dpi`` - Figure DPI (default: 300)
- ``--font-size`` - Font size in points (default: 10.0)
- ``--width`` - Figure width in inches (default: 8.0)
- ``--height`` - Figure height in inches (default: 6.0)
- ``--seed`` - Random seed for UMAP reproducibility (default: 42)
- ``--workers`` - Number of parallel workers (default: auto)

**Output Files:**
- ``cluster_umap.png`` - UMAP embedding colored by cluster
- ``cluster_heatmap.png`` - Heatmap of cluster mean feature values

Spatial Figures
~~~~~~~~~~~~~~~

.. autofunction:: openimc.cli.spatial_figures_command

**Example Usage:**

.. code-block:: bash

    # Generate spatial visualization figures
    openimc spatial-figures features.csv output/figures/ --edges edges.csv --dpi 300

**Parameters:**
- ``features`` - Input CSV file with features (must contain ``centroid_x``, ``centroid_y``) (required)
- ``output`` - Output directory for figures (required)
- ``--edges`` - Optional CSV file with spatial graph edges
- ``--dpi`` - Figure DPI (default: 300)
- ``--font-size`` - Font size in points (default: 10.0)
- ``--width`` - Figure width in inches (default: 8.0)
- ``--height`` - Figure height in inches (default: 6.0)
- ``--workers`` - Number of parallel workers (default: auto)

**Output Files:**
- ``spatial_distribution.png`` - Spatial scatter plot of cells
- ``spatial_graph.png`` - Spatial graph visualization (if edges provided)

Workflow
~~~~~~~~

.. autofunction:: openimc.cli.workflow_command

**Example Usage:**

.. code-block:: bash

    # Execute workflow from YAML configuration
    openimc workflow config.yaml

**Parameters:**
- ``config`` - Path to YAML configuration file (required)
- ``--workers`` - Number of parallel workers (default: auto)

**Workflow Configuration:**

The workflow command supports a YAML configuration file that can chain multiple analysis steps:

.. code-block:: yaml

    input: input.mcd
    output: workflow_output/
    channel_format: CHW

    preprocessing:
      enabled: true
      denoise_settings: null

    segmentation:
      enabled: true
      method: cellpose
      nuclear_channels: [DAPI]
      model: cyto3

    feature_extraction:
      enabled: true
      morphological: true
      intensity: true
      arcsinh: false

    clustering:
      enabled: true
      method: leiden
      resolution: 1.0

    spatial_analysis:
      enabled: true
      method: kNN
      k_neighbors: 10
      radius: 50
      detect_communities: true

Each step can specify:
- ``enabled``: true/false
- ``input``: path to input file/directory (optional, uses previous step output if not specified)
- ``output``: path to output file/directory (optional, uses default location if not specified)
- Step-specific parameters (see individual command documentation)

Additional Notes
---------------

**Parallel Processing:**
Most commands support parallel processing via the ``--workers`` parameter. By default, the number of workers is set to ``max(1, cpu_count() - 2)`` to leave resources for the system.

**Channel Format:**
OME-TIFF files can be stored in two formats:
- ``CHW``: Channels × Height × Width (default)
- ``HWC``: Height × Width × Channels

**Denoise Settings:**
Denoise settings can be specified as:
- A JSON file path
- A JSON string
- Using ``--denoise all`` to apply the same method to all channels

Example JSON format:

.. code-block:: json

    {
      "CD3": {
        "hot": {
          "method": "median3"
        }
      },
      "CD4": {
        "hot": {
          "method": "n_sd_local_median",
          "n_sd": 5.0
        }
      }
    }

**Error Handling:**
All commands provide informative error messages and will exit with a non-zero status code on failure. Use ``--help`` on any command to see detailed usage information.
