Quick Start
===========

This quick tutorial walks you through launching OpenIMC, loading an IMC file, visualizing channels, and running a basic segmentation workflow.

GUI Mode
--------

Start the GUI after activating your conda environment:

.. code-block:: bash

   python main.py

This opens the full OpenIMC interface.

Basic Workflow
~~~~~~~~~~~~~~

1. Open an IMC file

   Select **File → Open File/Folder**, then choose the ``.mcd`` file(s) you want to analyze.  
   Multiple files can be selected using Shift or Ctrl.

   .. figure:: _static/images/FileSelection_GUI.png
      :alt: File Selection in GUI
      :width: 600px
      :align: center

      **Figure 1:** Selecting one or more ``.mcd`` files in the OpenIMC GUI.


2. Explore the image

   Newly loaded files appear in the image viewer. By default, channels are shown in a grid layout.

   .. figure:: _static/images/MultiChannelImageView.png
      :alt: View of 3 channels in grid view
      :width: 800px
      :align: center

      **Figure 2:** Grid-based visualization of multiple IMC channels.

   Switch to grayscale by enabling **Grayscale mode**.

   To view channels as a composite RGB image, disable **Grid view for multiple channels**.  
   A dropdown will appear to assign channels to R, G, and B. Composite intensity is computed using the selected merge method.

   .. figure:: _static/images/RGB_view.png
      :alt: View of 3 channels in RGB view
      :width: 1000px
      :align: center

      **Figure 3:** RGB composite view generated from user-selected channels.

   Additional tools:
   - **Custom scaling**: interactively rescale intensities.
   - **Scale bar**: add and configure a micrometer scale bar.

   These options help you quickly assess signal quality, contrast, and cell morphology.


3. Run segmentation

   Click **Cell Segmentation** to open the segmentation dialog.  
   Choose a segmentation method and select the channels appropriate for your experiment.

   Available segmentation engines:
   - DeepCell CellSAM  
   - Cellpose cyto3 (cytoplasm + nucleus)  
   - Cellpose nuclei model  
   - Ilastik (``.ilp`` models)  
   - Watershed  

   For most datasets, **DeepCell CellSAM** provides the best overall performance.
   DeepCell CellSAM is a transformer based model and requires a GPU to run quickly (CPU still works, but will be slow). 
   If you do not have a GPU, Cellpose is a potential alternative, but will still be slow on CPU. 

   If using CellSAM, enter your DeepCell API token under **DeepCell CellSAM Parameters**.  
   Adjustable CellSAM options:
   - **Bbox threshold** (default 0.4)  
   - **Use WSI mode** (tiling for large images)  
   - **Low contrast enhancement**  
   - **Gauge cell size**  

   Before segmenting an entire experiment, it is recommended to test on a single acquisition and refine settings.

   Once segmentation completes, enable **Show Overlay** to visualize masks on top of any channel.

   .. figure:: _static/images/segmentation.png
      :alt: Segmentation overlaid on DNA
      :width: 1000px
      :align: center

      **Figure 4:** Segmentation mask overlaid on the DNA channel after processing with DeepCell CellSAM. Masks and the raw image are shown side by side.

4. Extract features

   Click **Feature Extraction** to open the feature extraction dialog.
   The primary settings in feature extraction are: 
   - **Acquisitions**: the acquisitions to extract features from (default is all acquisitions with segmentation masks)
   - **Output**: the output directory for the features (default is the current directory)
   - **Denoising**: the denoising method to use (for most datasets, hot pixel removal by Median 3x3 is recommended, ensure you click 'Apply to all channels')
   - **Arcsinh scaling**: whether to apply arcsinh scaling to the intensity features (for most datasets, this is recommended)

   .. figure:: _static/images/feature_extraction.png
      :alt: Recommended Settings for Feature Extraction for Most Datasets
      :width: 1000px
      :align: center

    **Figure 5:** Recommended Settings for Feature Extraction for Most Datasets.

   Click 'Extract Features' to start the feature extraction process.
   The features will be saved to the output directory as a CSV file.
   The data will continue to be stored in memory for further analysis, including clustering and spatial analysis and does not need to be reloaded. 

5. Cluster cells

   Click **Cell Clustering** to open the clustering dialog.
   The features extracted from the previous step will be loaded automatically.

   At the top of the clustering dialog, you can select the type of clustering to perform.
   Available clustering methods:
   - Leiden
   - Louvain
   - Hierarchical
   - K-means
   - HDBSCAN

   Leiden and louvain are recommended for most datasets with high cell density and highly varying types of cells.
   Hierarchical clustering is recommended for datasets with a small number of cells or a small number of cell types.
   K-means is recommended for datasets with approximately equal numbers of cells of each type.
   HDBSCAN is recommended for datasets with many outliers or cells that are not well-defined.
   
   After selecting the clustering method, you can set the parameters for the clustering method. See the Clustering section for more details.
   For most datasets, the default settings will be sufficient.

   We recommend using Leiden clustering for most datasets.

   .. figure:: _static/images/clustering_settings.png
      :alt: Clustering Dialog
      :width: 1000px
      :align: center

      **Figure 6:** Clustering Dialog Settings, default to Leiden clustering.

   After you hit 'Run Clustering', a pop up will ask you to select the features to cluster. Features will be subset to the mean intensities and the morphological features.
   At this stage, it is important to select the features that are most informative for the clustering.
   If you know that certain antibodies are not working well, you should exclude them from the clustering.

   .. figure:: _static/images/clustering_feature_selection.png
      :alt: Feature Selection
      :width: 1000px
      :align: center

      **Figure 7:** Feature Selection for Clustering.

   Once the clustering is complete, you can visualize the clusters.
   The default visualization is a heatmap of the clusters.

   .. figure:: _static/images/clustering_heatmap.png
      :alt: Heatmap of the clusters
      :width: 1000px
      :align: center

      **Figure 8:** Heatmap of the clusters, with annotation of both the clusters and the patients from which the cells are derived.

   At the bottom of the clustering dialog, you can change to various other visualizations, including UMAPs, t-SNE, Stacked Bars, Differential Expression, and Boxplot/Violin Plot.
   You can also save the visualization as a PNG file by clicking the 'Save Plot' button.
   You can also save the clustering output as a CSV file by clicking the 'Save Clustering Output' button. This will save the features with cluster labels and/or any manual annotations.

6. Cell phenotyping

   After clustering, you can annotate the clusters by their cell type. 
   This is done by clicking the 'Annotate Phenotypes' button.

   You can either manually annotate the clusters by typing in the cell type name or use the LLM to annotate the clusters.
   The LLM will ask you for an OpenAI API key. If you do not have an OpenAI API key, you can get one by signing up for an account at https://openai.com/ (see the Installation section for more details).
   You can also provide context to the LLM (such as the cancer type, the tissue type, the treatment, the patient metadata, etc.)
   This will help the LLM to generate more accurate cell type annotations.

   Note: the LLM is charged by the token, so ensure your account has enough credits.
   Second Note: the LLM is not perfect, so you should manually check the annotations and correct them if necessary.

   Once the LLM has suggested the cell types, you can choose from its suggestions and update your plot. 

   .. figure:: _static/images/phenotyping.png
      :alt: Phenotyping with ChatGPT suggestions
      :width: 1000px
      :align: center

      **Figure 9:** Phenotyping with ChatGPT suggestions.

7. Spatial analysis

   After clustering, you can perform spatial analysis to investigate the spatial distribution of the cells.
   This is done by clicking the 'Spatial Analysis' button.
   
   Spatial analysis is split into two windows: a simple analysis and a more advanced analysis. 
   For most users, the simple analysis will be sufficient.

   The first step of spatial analysis is to build a spatial graph of the cells.
   There are three methods to build the spatial graph:
   - k-nearest neighbors (kNN)
   - Radius
   - Delaunay

   kNN is the default method and is recommended for most datasets. K is the number of neighbors to consider for the spatial graph for each cell. Set this to a reasonable number based on the density of the cells.
   Radius is recommended for datasets with a small number of cells or a small number of cell types. Radius is the radius in micrometers to consider for the spatial graph for each cell. Set this to a reasonable number based on the density of the cells.
   Delaunay is recommended for datasets with a large number of cells or a large number of cell types. Delaunay is based on the Delaunay triangulation of the cells. See more details in the Spatial Analysis section.

   Some spatial visualizations are available in the simple analysis window, including:
   - Spatial visualization of the cells (represent each cell as a point, color by their cluster, and show the edges between the cells)
   - Distance distribution of the cells (show the distance distribution of the cells by their cluster -> what is the distribution of distances between cells of the same cluster vs. cells of different clusters?)
   - Pairwise enrichment analysis (test for significant spatial co-occurrence or avoidance between cluster pairs using permutation tests)
   - Community detection (detect communities in the spatial graph -> rather than clustering the cells, we cluster based on the spatial relationship between the cells)

   .. figure:: _static/images/spatial_visualization.png
      :alt: Spatial Visualizations in the Simple Analysis Window
      :width: 1000px
      :align: center

      **Figure 10:** Spatial Visualizations in the Simple Analysis Window.


More Analyses and Features in OpenIMC
-------------------------------------

This quickstart covers the essential workflow to get you started with OpenIMC. However, the application includes many additional features and advanced analysis options to support a wide range of IMC experiments. These include:

- **Quality Control (QC)**: Tools for assessing image quality, detecting artifacts, and verifying segmentation accuracy.
- **Pixel Correlation Analysis**: Explore channel relationships and spatial co-localization at the pixel level.
- **Advanced Spatial Analyses**: Beyond the basics, the software supports expanded spatial statistics, neighborhood enrichment, proximity scores, and custom graph-building options.
- **Panel Design and Spillover Tools**: Functions to assist with antibody panel QC and compensation for metal spillover.
- **Batch Processing, Automation, and Reports**: Tools for end-to-end automated analysis, with exportable reports and visualizations.

We recommend exploring each section of the documentation after you have completed your first analysis to take full advantage of OpenIMC’s capabilities. The documentation provides detailed step-by-step usage guides, tips, and explanations for all analysis modules.
