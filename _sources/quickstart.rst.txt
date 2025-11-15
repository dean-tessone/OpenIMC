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
    In the GUI, select 'Open File/Folder' and navigate to the .mcd file you want to analyze.
    Note: users can select multiple .mcd files to analyze at once (shift click or control click).

    .. image:: __static/images/FileSelection_gui.png
        :alt: File Selection in GUI
        :width: 600px
        :class: center

2. Use the image viewer to inspect the image.
    The image viewer displays the selected image(s) in a grid view by default.
    Users can select a single channel to view, or multiple channels.

    .. image:: __static/images/MultiChannelImageView.png
        :alt: View of 3 channels in grid view
        :width: 800px
        :class: center

    Users can view the image in grayscale mode by clicking the 'Grayscale mode' checkbox.

    To see images in RGB, rather than in the default grid view, disable the 'Grid view for multiple channels' checkbox.
    A drop down will appear to select the RGB channels to use for the composite image. Users can assign as many channels as they want to each of the RGB channels.
    The channels will be combined using the selected method.


    .. image:: __static/images/RGB_view.png
        :alt: View of 3 channels in RGB view
        :width: 1000px
        :class: center
    
    Users can also scale the image by clicking the 'Custom scaling' checkbox.
    A dialog will appear to configure the scaling parameters, which will update in real time. 
    Users can also add a scale bar to the image by clicking the 'Scale bar' checkbox, which will appear in the bottom right corner of the image, with a configurable length in micrometers.

    For more detail on the image viewer, see the Image Viewer section of the documentation.

3. Run segmentation.
    After viewing the image(s), users can run segmentation by clicking the 'Cell Segmentation' button.
    The segmentation dialog will appear.
    Users can select the segmentation method, and configure the segmentation parameters.
    Users should select the channels to use for segmentation. Importantly, this is unique to each experiment and needs to be determined by the user.

    5 different segmentation methods are available:
    - DeepCell CellSAM
    - Cellpose cyto3 model (both cytoplasm and nucleus)
    - Cellpose nuclei model (only nucleus)
    - Ilastik (load and run inference with your trained Ilastik models (.ilp project files))
    - Watershed

    We recommend using the DeepCell CellSAM method for most experiments. 
    DeepCell CellSAM is a foundation model-based segmentation method that is trained on a large dataset of IMC images, IF images, brightfield images, and more.
    
    To set up DeepCell CellSAM, users need to provide a DeepCell API token. You can get your token from the DeepCell User Portal (see the Installation section of the documentation for more details).
    Once you have your token, you can enter it in the 'DeepCell CellSAM Parameters' section of the segmentation dialog.

    CellSAM has 4 parameters that can be configured:
    - Bbox threshold: This is the threshold for the bounding box of the cells. Lower values will include more cells, but may include more false positives. Higher values will exclude more cells, but may exclude more true cells. The default value of 0.4 is generally appropriate.
    - Use WSI mode: This is only available for DeepCell CellSAM. It is used to segment large images (e.g. >500 cells). It will take longer to run, but will be more accurate. The default value is False. This involves tiling the image and running the segmentation on each tile separately.
    - Low contrast enhancement: This is only available for DeepCell CellSAM. It is used to enhance the contrast of the image. It will take longer to run, but will be more accurate. The default value is False. This involves applying a contrast enhancement filter to the image.
    - Gauge cell size: This is only available for DeepCell CellSAM. It is used to gauge the cell size of the cells. It will take longer to run, but will be more accurate. The default value is False. This involves running the segmentation twice, once to gauge the cell size, and once to run the segmentation.

    For more detail on the segmentation dialog, see the Segmentation Dialog section of the documentation. 

    At the bottom of the segmentation dialog, users can decide to segment all images in the experiment at once, or segment only the current acquisition. 
    We recommend first segmenting one image, tweaking the parameters as needed, and then segmenting the rest of the images in the experiment.
    Segmentation can take a few minutes to complete, depending on the size of the image and the segmentation method.
    Because segmentation is computationally intensive, users can choose to load masks into the image viewer to speed up the process. To do this, navigate to the 'File -> Load Segmentation Masks' menu item.

    To visually inspect the segmentation, users can click the 'Show Overlay' checkbox once segmentation is complete.

    .. image:: __static/images/segmentation.png
        :alt: View of segmentation overlaid on DAPI, with DAPI only on the side.
        :width: 1000px
        :class: center

CLI Mode
--------
After installation of the conda environment, you can start the CLI by running: