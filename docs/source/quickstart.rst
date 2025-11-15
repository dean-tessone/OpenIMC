Quick Start 
============

A quick start guide for using OpenIMC.

GUI Mode
--------

After installation of the conda environment, you can start the GUI by running:

.. code-block:: bash

   python main.py

This will launch the GUI application.

Basic Workflow
~~~~~~~~~~~~~~

1. Open .mcd file.

   In the GUI, select "Open File/Folder" and navigate to the .mcd file you want to analyze.
   Note: users can select multiple .mcd files to analyze at once (Shift-click or Ctrl-click).

   .. image:: _static/images/FileSelection_gui.png
      :alt: File Selection in GUI
      :width: 600px
      :align: center

2. Use the image viewer to inspect the image.

   The image viewer displays the selected image(s) in a grid view by default.
   Users can select a single channel to view, or multiple channels.

   .. image:: _static/images/MultiChannelImageView.png
      :alt: View of 3 channels in grid view
      :width: 800px
      :align: center

   Users can view the image in grayscale mode by clicking the "Grayscale mode" checkbox.

   To see images in RGB, rather than in the default grid view, disable the "Grid view for multiple channels" checkbox.
   A dropdown will appear to select the RGB channels to use for the composite image. Users can assign as many channels as they want to each of the RGB channels.
   The channels will be combined using the selected method.

   .. image:: _static/images/RGB_view.png
      :alt: View of 3 channels in RGB view
      :width: 1000px
      :align: center

   Users can also scale the image by clicking the "Custom scaling" checkbox.
   A dialog will appear to configure the scaling parameters, which will update in real time.
   Users can also add a scale bar to the image by clicking the "Scale bar" checkbox, which will appear in the bottom right corner of the image with a configurable length in micrometers.

   For more detail on the image viewer, see the Image Viewer section of the documentation.

3. Run segmentation.

   After viewing the image(s), users can run segmentation by clicking the "Cell Segmentation" button.
   The segmentation dialog will appear.
   Users can select the segmentation method and configure the segmentation parameters.
   Users should select the channels to use for segmentation. Importantly, this is unique to each experiment and needs to be determined by the user.

   Five segmentation methods are available:

   - DeepCell CellSAM
   - Cellpose cyto3 model (both cytoplasm and nucleus)
   - Cellpose nuclei model (only nucleus)
   - Ilastik (load and run inference with trained Ilastik models, ``.ilp`` project files)
   - Watershed

   We recommend using the DeepCell CellSAM method for most experiments.
   DeepCell CellSAM is a foundation model-based segmentation method trained on a large dataset of IMC, IF, brightfield, and related images.

   To set up DeepCell CellSAM, users need to provide a DeepCell API token. You can get your token from the DeepCell User Portal (see the Installation section of the documentation for more details).
   Once you have your token, enter it in the "DeepCell CellSAM Parameters" section of the segmentation dialog.

   CellSAM has four parameters that can be configured:

   - **Bbox threshold**: Threshold for bounding boxes. Lower values include more cells (and more false positives), higher values exclude more cells. The default (0.4) is generally appropriate.
   - **Use WSI mode**: For large images (e.g. > 500 cells). Tiles the image and segments tile-by-tile. Slower but more accurate. Default: False.
   - **Low contrast enhancement**: Enhances contrast for low-contrast images. Slower but may improve segmentation. Default: False.
   - **Gauge cell size**: Estimates cell size before running final segmentation. Slower but can improve robustness across datasets. Default: False.

   For more detail on the segmentation dialog, see the Segmentation Dialog section of the documentation.

   At the bottom of the segmentation dialog, users can choose to segment all images in the experiment at once, or only the current acquisition.
   We recommend first segmenting one image, tuning the parameters, and then segmenting the remaining images.

   Segmentation can take several minutes depending on image size and the chosen method.
   Because segmentation is computationally intensive, users can choose to load existing masks into the image viewer to speed up subsequent work.
   To do this, use the "File → Load Segmentation Masks" menu item.

   To visually inspect segmentation results, users can click the "Show Overlay" checkbox once segmentation is complete.

   .. image:: _static/images/segmentation.png
      :alt: View of segmentation overlaid on DAPI, with DAPI-only inset
      :width: 1000px
      :align: center

CLI Mode
--------

After installation of the conda environment, you can start the CLI by running:
