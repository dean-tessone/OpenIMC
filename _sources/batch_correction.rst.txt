Batch Correction
=================

Batch correction removes technical variation (batch effects) between different experiments, acquisition runs, or files, allowing data from multiple sources to be combined for analysis.

Overview
--------

When combining IMC data from multiple experiments, files, or acquisition runs, technical variation can introduce artifacts that confound biological signals. Batch correction methods identify and remove these technical effects while preserving biological variation.

Options
-------

OpenIMC supports two batch correction methods:

1. **Harmony** (default): Iterative clustering-based correction in PCA space
2. **ComBat**: Empirical Bayes method for batch effect correction

Parameters
----------

Common Parameters
~~~~~~~~~~~~~~~~~

- **method** (default: ``"harmony"``): Batch correction method
  - Options: ``"harmony"``, ``"combat"``

- **batch_variable** (optional): Column name containing batch identifiers
  - If not specified, auto-detects from ``source_file`` or ``acquisition_id`` columns
  - Each unique value represents a batch

- **features** (optional): List of feature column names to correct
  - If not specified, auto-detects all numeric feature columns
  - Excludes metadata columns (cell_id, centroid_x, centroid_y, cluster, etc.)

Harmony Parameters
~~~~~~~~~~~~~~~~~~

- **n_clusters** (default: ``30``): Number of Harmony clusters
  - More clusters capture finer structure but may overfit
  - Typical range: 10-50
  - Increase for datasets with many cell types

- **sigma** (default: ``0.1``): Width of soft k-means clusters
  - Controls how "soft" the cluster assignments are
  - Lower values (0.05-0.1) create sharper clusters
  - Higher values (0.1-0.3) create softer, overlapping clusters

- **theta** (default: ``2.0``): Diversity clustering penalty parameter
  - Higher values (2.0-4.0) encourage more diverse clusters across batches
  - Lower values (0.5-2.0) allow more batch-specific clusters
  - Adjust if batches are not mixing well

- **lambda_reg** (default: ``1.0``): Regularization parameter
  - Controls strength of correction
  - Higher values (1.0-2.0) apply stronger correction
  - Lower values (0.5-1.0) apply weaker correction
  - Increase if batch effects are strong

- **max_iter** (default: ``20``): Maximum number of iterations
  - Harmony typically converges in 10-20 iterations
  - Increase (30-50) if convergence is slow
  - Decrease if processing time is a concern

- **pca_variance** (default: ``0.9``): Proportion of variance to retain in PCA
  - Higher values (0.9-0.95) retain more information but may include noise
  - Lower values (0.8-0.9) reduce dimensionality more aggressively
  - Typical range: 0.8-0.95

ComBat Parameters
~~~~~~~~~~~~~~~~~

- **covariates** (optional): List of covariate column names
  - Covariates are biological variables to preserve (e.g., condition, treatment)
  - ComBat will correct for batch while preserving covariate effects
  - Example: ``["condition", "treatment"]``

Using Batch Correction in the GUI
-----------------------------------

1. Ensure feature extraction has been completed

2. Navigate to **Analysis → Batch Correction** or click the batch correction button

3. In the batch correction dialog:
   - Select the batch correction method (Harmony or ComBat)
   - Choose or verify the batch variable (auto-detected if available)
   - Select features to correct (auto-detected if not specified)
   - Adjust method-specific parameters
   - For ComBat, optionally specify covariates to preserve

4. Choose output location for the corrected features CSV file

5. Click **Apply Batch Correction** to start the process

6. The corrected features are saved and can be used for downstream analysis

Using Batch Correction in the CLI
-----------------------------------

Basic Harmony Command
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc batch-correction features.csv corrected_features.csv \\
       --method harmony \\
       --batch-var source_file

With Custom Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc batch-correction features.csv corrected_features.csv \\
       --method harmony \\
       --batch-var source_file \\
       --n-clusters 40 \\
       --sigma 0.15 \\
       --theta 2.5 \\
       --lambda-reg 1.2

ComBat Command
~~~~~~~~~~~~~~

.. code-block:: bash

   openimc batch-correction features.csv corrected_features.csv \\
       --method combat \\
       --batch-var source_file \\
       --covariates condition,treatment

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   batch_correction:
     enabled: true
     method: "harmony"
     batch_variable: "source_file"
     features: null  # Auto-detect
     n_clusters: 30
     sigma: 0.1
     theta: 2.0
     lambda_reg: 1.0
     max_iter: 20
     pca_variance: 0.9

Method Details
--------------

Harmony
~~~~~~~

Harmony is an iterative clustering-based method that corrects batch effects in low-dimensional space (typically PCA).

**How it works:**

1. **PCA Projection**: Features are projected into PCA space, retaining a specified proportion of variance (pca_variance)

2. **Soft Clustering**: Cells are assigned to clusters using soft k-means with specified width (sigma)

3. **Batch Correction**: Within each cluster, batch-specific effects are estimated and removed

4. **Iteration**: Steps 2-3 are repeated until convergence (up to max_iter iterations)

5. **Diversity Penalty**: The theta parameter encourages clusters to contain cells from multiple batches

6. **Regularization**: The lambda_reg parameter controls the strength of correction

7. **Back-projection**: Corrected PCA coordinates are transformed back to original feature space

**Advantages:**
- Preserves biological variation while removing batch effects
- Works well with many batches
- Handles non-linear batch effects
- Fast and scalable

**Limitations:**
- Requires sufficient cells per batch for stable correction
- May over-correct if batch effects are weak
- Parameters may need tuning for optimal results

**Citation:**
- Korsunsky, I., et al. (2019). "Fast, sensitive and accurate integration of single-cell data with Harmony." Nature Methods, 16(12), 1289-1296. `DOI: 10.1038/s41592-019-0619-0 <https://doi.org/10.1038/s41592-019-0619-0>`_
- `Harmony GitHub <https://github.com/immunogenomics/harmony>`_
- `harmonypy Python Package <https://github.com/slowkow/harmonypy>`_

ComBat
~~~~~~

ComBat is an empirical Bayes method that estimates and removes batch effects while preserving biological variation.

**How it works:**

1. **Model Fitting**: For each feature, ComBat fits a linear model:
   - Feature = Batch Effect + Biological Variation + Error

2. **Empirical Bayes**: Batch effects are estimated using empirical Bayes, which borrows information across features

3. **Covariate Adjustment**: If covariates are specified, their effects are preserved while correcting for batch

4. **Correction**: Estimated batch effects are subtracted from the data

**Advantages:**
- Well-established method with proven track record
- Can preserve specified covariates
- Works well with small batch sizes
- Fast computation

**Limitations:**
- Assumes linear batch effects
- May not handle non-linear batch effects well
- Requires sufficient features for stable estimation

**Citation:**
- Johnson, W. E., et al. (2007). "Adjusting batch effects in microarray expression data using empirical Bayes methods." Biostatistics, 8(1), 118-127. `DOI: 10.1093/biostatistics/kxj037 <https://doi.org/10.1093/biostatistics/kxj037>`_
- Zhang, Y., et al. (2020). "ComBat-seq: batch effect adjustment for RNA-seq count data." NAR Genomics and Bioinformatics, 2(3), lqaa078. `DOI: 10.1093/nargab/lqaa078 <https://doi.org/10.1093/nargab/lqaa078>`_
- `pycombat Python Package <https://github.com/brentp/combat-py>`_

Tips and Best Practices
-----------------------

1. **When to Use Batch Correction**: Apply batch correction when combining data from:
   - Multiple acquisition runs
   - Different experimental days
   - Different instruments or protocols
   - Multiple files with systematic differences

2. **Method Selection**:
   - Use **Harmony** for most cases, especially with many batches or non-linear effects
   - Use **ComBat** when you need to preserve specific covariates or have linear batch effects

3. **Parameter Tuning**:
   - Start with default parameters
   - Visualize results (e.g., PCA plots) to assess correction quality
   - Adjust parameters if batches are not mixing well or if over-correction occurs

4. **Validation**: After batch correction, verify that:
   - Batch effects are reduced (check PCA plots colored by batch)
   - Biological variation is preserved (check that known cell types still separate)
   - Feature distributions are reasonable (no extreme values or artifacts)

5. **Feature Selection**: Only correct features that are affected by batch effects. Exclude:
   - Morphological features (typically less affected by batch)
   - Features that should vary by batch (e.g., acquisition-specific markers)

6. **Covariates**: When using ComBat, carefully select covariates to preserve. Include only true biological variables, not technical variables that correlate with batch.

7. **Downstream Analysis**: Use batch-corrected features for:
   - Clustering across batches
   - Differential expression analysis
   - Spatial analysis combining multiple files
   - Any analysis requiring integrated data

