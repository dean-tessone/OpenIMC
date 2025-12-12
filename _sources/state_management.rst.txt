State Management and Reproducibility
====================================

OpenIMC provides comprehensive state management features that allow you to save your complete analysis session and restore it later. This is essential for reproducibility, collaboration, and sharing your work for publication.

Overview
--------

The state management system saves:

* **Source Data Files**: All .mcd files used in the analysis
* **Segmentation Masks**: All cell segmentation masks
* **Feature Data**: Original and batch-corrected feature dataframes
* **Analysis States**: Complete state from all analysis modules (QC, clustering, spatial analysis, etc.)
* **Analysis Parameters**: All parameters used for each analysis step

Saved states are self-contained and can be:

* Loaded by anyone with OpenIMC to reproduce your analysis
* Uploaded to Zenodo or other data repositories as supplementary material
* Shared with collaborators for review or continuation of work

Saving State
------------

To save your current analysis state:

1. Go to **File → Save State…**
2. Select a directory where you want to save the state
3. The application will create a folder containing all necessary files

**State Folder Structure:**

::

   state_folder/
   ├── README.md              # Documentation for the saved state
   ├── manifest.txt           # Complete file listing
   ├── state.json             # Main state metadata and configuration
   ├── images/                # Source .mcd files (if any)
   ├── masks/                 # Segmentation masks
   ├── features/              # Feature dataframes
   └── analysis/              # Analysis module states

**What Gets Saved:**

* **Images**: All .mcd source files are copied to the ``images/`` folder
* **Masks**: All segmentation masks with proper naming (``{source}_{well}_segmentation_masks.tif``)
* **Features**: Original features (``features.csv``) and batch-corrected features (``features_batch_corrected.csv``)
* **Analysis States**: 
  * QC analysis results and parameters
  * Clustering results, cluster labels, and annotation maps
  * Spatial analysis results and graph states
  * Feature extraction parameters
  * All other analysis module configurations

Loading State
-------------

To load a previously saved state:

1. Go to **File → Load State…**
2. Select the folder containing the saved state
3. The application will:
   * Automatically load all .mcd files from the ``images/`` folder
   * Restore all masks, features, and analysis states
   * Restore the UI to match the saved session

**After Loading:**

* All acquisitions are available and images are displayed
* Masks are properly matched to acquisitions
* Features are loaded and available for analysis
* Analysis dialogs can be reopened with their saved state
* You can continue working exactly where you left off

Exporting Analysis Steps
-------------------------

For documentation and methods sections, you can export a human-readable description of all analysis steps:

1. Go to **File → Export Analysis Steps…**
2. Choose a location to save the text file
3. The file will contain:
   * Dataset information (number of cells, features, etc.)
   * All segmentation steps (method, channels, parameters)
   * Feature extraction details (features extracted, normalization, corrections)
   * Clustering information (algorithm, parameters, features used)
   * Batch correction details
   * Spatial analysis methods
   * All other analysis operations with timestamps and parameters

**Example Output:**

::

   OpenIMC Analysis Steps
   ====================================================================
   
   Generated: 2025-01-15 14:30:22
   
   This document describes all analysis steps performed on the dataset.
   
   Dataset Information
   --------------------------------------------------------------------
   Source file(s): data.mcd
   Number of acquisitions: 12
   Feature data: 45000 cells, 156 features
   Batch correction: Applied
   Segmentation masks: 12 acquisition(s)
   
   Segmentation
   --------------------------------------------------------------------
   1. Watershed
      Date/Time: 2025-01-15 10:15:30
      Parameters:
         - method: watershed
         - nuclear_channels: ['DNA1_Ir191', 'DNA2_Ir193']
         - cyto_channels: ['CD3_1841', 'CD4_2293']
         - n_cells: 45000
      Acquisitions: 12 acquisition(s)
   
   Feature Extraction
   --------------------------------------------------------------------
   1. Extract Features
      Date/Time: 2025-01-15 10:45:12
      Parameters:
         - features_extracted: 156 features
         - morphological: True
         - intensity: True
         - arcsinh: True
         - arcsinh_cofactor: 1.0
      Acquisitions: 12 acquisition(s)
   
   ...

Zenodo Submission
-----------------

Saved states are designed to be uploaded to Zenodo or other data repositories as supplementary material for publications.

**Recommended Submission Structure:**

1. **Source Data**: Original .mcd files (if not already in a separate repository)
2. **State Folder**: Complete saved state folder containing:
   * All masks, features, and analysis states
   * README.md explaining the contents
   * manifest.txt listing all files
3. **Analysis Steps**: Exported analysis steps text file for methods section

**Benefits:**

* **Reproducibility**: Anyone can load the state and reproduce your analysis
* **Transparency**: All parameters and steps are documented
* **Collaboration**: Others can continue or modify your analysis
* **Publication**: Meets requirements for data and code availability

**README.md Contents:**

Each saved state includes a README.md file that:

* Lists all contents of the state folder
* Explains how to load the state
* Documents the OpenIMC version used
* Provides notes for Zenodo submission

Best Practices
-------------

1. **Save State Regularly**: Save your state at key milestones in your analysis
2. **Descriptive Folder Names**: Use descriptive names like ``analysis_20250115_final``
3. **Include Analysis Steps**: Export analysis steps alongside the state for documentation
4. **Version Control**: Consider versioning your states (e.g., ``state_v1/``, ``state_v2/``)
5. **Document Changes**: Note any manual changes or corrections in the README

Technical Details
-----------------

**State Format Version**: The state format is versioned to ensure compatibility. Current version: 1.0

**File Formats**:
* Masks: TIFF files (uint16) with LZW compression
* Features: CSV files (comma-separated)
* Analysis States: JSON files with special handling for DataFrames and numpy arrays
* Images: Original .mcd files (copied, not converted)

**Mask Matching**: When loading state, masks are matched to acquisitions using:
1. Source file basename + well/name (most reliable for multiple MCD files)
2. Acquisition ID matching
3. Well/name matching as fallback

**Multi-File Support**: When multiple .mcd files are saved in state, they are loaded using the same logic as the base application, including:
* Unique acquisition ID generation
* Proper file-to-acquisition mapping
* Channel mismatch detection
* File names in acquisition dropdown

