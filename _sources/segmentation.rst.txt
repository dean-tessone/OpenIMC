Segmentation
============

Segmentation is the process of identifying and delineating individual cells in IMC images. OpenIMC provides four segmentation methods: CellSAM (via DeepCell API), Cellpose, Watershed, and Ilastik.

Overview
--------

Cell segmentation is a critical first step in IMC analysis, as it defines the boundaries of individual cells from which features will be extracted. The choice of segmentation method depends on your data characteristics, computational resources, and accuracy requirements.

Options
-------

OpenIMC supports four segmentation methods:

1. **CellSAM** (default): Deep learning-based segmentation using DeepCell's CellSAM model via API
2. **Cellpose**: Local deep learning-based segmentation (supports GPU acceleration)
3. **Watershed**: Traditional marker-controlled watershed segmentation
4. **Ilastik**: Segmentation using pre-trained Ilastik models (``.ilp`` project files)

Parameters
----------

Common Parameters
~~~~~~~~~~~~~~~~~

These parameters apply to all segmentation methods:

- **nuclear_channels** (required): List of channel names to use for nuclear detection
  - Example: ``["DNA1_Ir191", "DNA2_Ir193"]``
  - Used to identify cell nuclei for seeding segmentation

- **cytoplasm_channels** (optional for CellSAM/Cellpose, required for Watershed): List of channel names for cytoplasm detection
  - Example: ``["CD3_1841", "CD4_2293"]``
  - Used to define cell boundaries

- **nuclear_fusion_method** (default: ``"mean"``): Method to combine multiple nuclear channels
  - Options: ``"single"``, ``"mean"``, ``"weighted"``, ``"max"``, ``"pca1"``
  - ``"single"``: Use only the first channel
  - ``"mean"``: Average all channels
  - ``"weighted"``: Weighted average (requires ``nuclear_weights``)
  - ``"max"``: Maximum intensity across channels
  - ``"pca1"``: First principal component

- **cyto_fusion_method** (default: ``"mean"``): Method to combine cytoplasm channels (same options as nuclear_fusion_method)

- **nuclear_weights** (optional): List of weights for weighted fusion of nuclear channels
  - Example: ``[0.5, 0.3, 0.2]``
  - Must match the number of nuclear channels

- **cyto_weights** (optional): List of weights for weighted fusion of cytoplasm channels

- **normalization_method** (default: ``"None"``): Normalization method to apply before segmentation
  - ``"None"``: No normalization
  - ``"arcsinh"``: Arcsinh transformation (helps normalize intensity distributions, recommended for data with high dynamic range)
  - ``"channelwise_minmax"``: Min-max normalization per channel (scales each channel independently to 0-1 range, useful when channels have different intensity ranges)
  - ``"percentile_clip"``: Percentile-based clipping normalization

- **arcsinh_cofactor** (default: ``10.0``): Cofactor for arcsinh transformation (only used if normalization_method is "arcsinh")
  - Lower values increase compression of high intensities
  - Typical range: 5.0-20.0

- **arcsinh** (default: ``false``): Legacy parameter, equivalent to setting normalization_method to "arcsinh"
  - Deprecated: Use normalization_method instead

- **denoise_settings** (optional): Dictionary with denoise settings per channel
  - Format: ``{"Channel1": {"method": "gaussian", "sigma": 1.0}}``
  - Can also be a path to a JSON file

- **acquisition** (optional): Specific acquisition ID or name to segment
  - If not specified, processes all acquisitions

CellSAM Parameters
~~~~~~~~~~~~~~~~~~

- **deepcell_api_key** (optional): DeepCell API access token
  - If not provided, uses ``DEEPCELL_ACCESS_TOKEN`` environment variable
  - Required for CellSAM method
  - Get your API key from `DeepCell <https://deepcell.org/>`_

- **bbox_threshold** (default: ``0.4``): Bounding box detection threshold
  - Lower values (0.01-0.1) detect fainter cells
  - Higher values (0.4-0.8) detect only bright cells
  - Adjust based on cell visibility in your data

- **use_wsi** (default: ``false``): Use whole-slide imaging (WSI) mode
  - Enable for ROIs with >500 cells
  - Improves performance for large images

- **low_contrast_enhancement** (default: ``false``): Apply low contrast enhancement
  - Helps with faint or low-contrast cells

- **gauge_cell_size** (default: ``false``): Automatically gauge cell size
  - Can improve segmentation for variable cell sizes

Cellpose Parameters
~~~~~~~~~~~~~~~~~~

- **model** (default: ``"cyto3"``): Cellpose model type
  - ``"cyto3"``: General cytoplasm segmentation (requires cytoplasm channels)
  - ``"nuclei"``: Nuclear segmentation only

- **diameter** (optional): Expected cell diameter in pixels
  - If not specified, Cellpose estimates automatically
  - Specify if you know the approximate cell size

- **flow_threshold** (default: ``0.4``): Flow field threshold
  - Lower values (0.0-0.2) allow more cell boundaries
  - Higher values (0.4-0.8) require stronger boundaries
  - Adjust if cells are over- or under-segmented

- **cellprob_threshold** (default: ``0.0``): Cell probability threshold
  - Lower values (negative) include more uncertain regions
  - Higher values (0.0-0.5) require higher confidence
  - Typically kept at 0.0 for best results

- **gpu_id** (optional): GPU device ID to use for acceleration
  - Example: ``0`` for first GPU, ``1`` for second GPU
  - If not specified, uses CPU (slower but works on all systems)

Watershed Parameters
~~~~~~~~~~~~~~~~~~~~

- **min_cell_area** (default: ``100``): Minimum cell area in pixels
  - Filters out small objects (likely noise)
  - Increase if you have many small false positives

- **max_cell_area** (default: ``10000``): Maximum cell area in pixels
  - Filters out very large objects (likely merged cells)
  - Decrease if cells are being merged

- **compactness** (default: ``0.01``): Watershed compactness parameter
  - Lower values (0.001-0.01) allow irregular shapes
  - Higher values (0.01-0.1) prefer compact, round shapes
  - Adjust based on expected cell morphology

Ilastik Parameters
~~~~~~~~~~~~~~~~~

- **ilp_file** (required): Path to Ilastik project file (``.ilp``)
  - Must be a trained Ilastik project file
  - Train your model in Ilastik GUI and save as ``.ilp`` file
  - The model should be trained on similar data for best results

- **output_format** (default: ``"Simple Segmentation"``): Output format from Ilastik
  - ``"Simple Segmentation"``: Segmentation masks (default)
  - ``"Probabilities"``: Probability maps for each class

.. note::
   Ilastik must be installed separately and available in your PATH. See the :doc:`installation` guide for details.

Using Segmentation in the GUI
------------------------------

1. Load your IMC data file (``.mcd`` or OME-TIFF directory)

2. Navigate to **Analysis → Segmentation** or click the segmentation button in the toolbar

3. In the segmentation dialog:
   - Select the segmentation method (CellSAM, Cellpose, Watershed, or Ilastik)
   - Choose nuclear channels from the channel list (not required for Ilastik)
   - Optionally select cytoplasm channels (not required for Ilastik)
   - For Ilastik: Browse and select your trained ``.ilp`` project file
   - Adjust method-specific parameters
   - Configure preprocessing options (denoising, normalization)

4. Click **Run Segmentation** to start the process

5. Segmentation masks are automatically saved and can be visualized in the image viewer

6. Masks are stored per acquisition and can be used for subsequent feature extraction

Using Segmentation in the CLI
-------------------------------

Basic Command
~~~~~~~~~~~~~

.. code-block:: bash

   openimc segment input.mcd output/ --method cellpose \\
       --nuclear-channels DNA1_Ir191,DNA2_Ir193 \\
       --cytoplasm-channels CD3_1841,CD4_2293

CellSAM Example
~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc segment input.mcd output/ --method cellsam \\
       --nuclear-channels DNA1_Ir191,DNA2_Ir193 \\
       --cytoplasm-channels CD3_1841 \\
       --bbox-threshold 0.3 \\
       --use-wsi

Cellpose Example
~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc segment input.mcd output/ --method cellpose \\
       --nuclear-channels DNA1_Ir191 \\
       --cytoplasm-channels CD3_1841,CD4_2293 \\
       --model cyto3 \\
       --diameter 30 \\
       --flow-threshold 0.4 \\
       --gpu-id 0

Watershed Example
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   openimc segment input.mcd output/ --method watershed \\
       --nuclear-channels DNA1_Ir191,DNA2_Ir193 \\
       --cytoplasm-channels CD3_1841 \\
       --min-cell-area 100 \\
       --max-cell-area 10000 \\
       --compactness 0.01

Ilastik Example
~~~~~~~~~~~~~~~

.. note::
   Ilastik segmentation is primarily available through the GUI. For CLI usage, ensure Ilastik is installed and the ``ilastik`` command is available in your PATH.

In the GUI, select "Ilastik" as the segmentation method and browse to your trained ``.ilp`` project file.

Workflow YAML Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   segmentation:
     enabled: true
     method: "cellsam"
     nuclear_channels:
       - "DNA1_Ir191"
       - "DNA2_Ir193"
     cytoplasm_channels:
       - "CD3_1841"
     nuclear_fusion_method: "mean"
     cyto_fusion_method: "mean"
     normalization_method: "channelwise_minmax"  # Options: "None", "arcsinh", "channelwise_minmax", "percentile_clip"
     arcsinh_cofactor: 10.0
     bbox_threshold: 0.4
     use_wsi: false

Method Details
--------------

CellSAM
~~~~~~~

CellSAM uses DeepCell's CellSAM model, which is a state-of-the-art deep learning model for cell segmentation. It leverages the Segment Anything Model (SAM) architecture adapted for cell segmentation tasks.

**How it works:**
1. Nuclear channels are combined using the specified fusion method
2. The combined nuclear image is sent to DeepCell's API
3. CellSAM detects cell bounding boxes using the bbox_threshold
4. Within each bounding box, precise cell boundaries are segmented
5. Results are returned as segmentation masks

**Advantages:**
- High accuracy, especially for complex cell morphologies
- Handles variable cell sizes well
- No local GPU required (uses cloud API)

**Limitations:**
- Requires internet connection and API key
- Processing time depends on API availability
- May have usage limits depending on API plan

**Citation:**
- DeepCell CellSAM: `DeepCell Platform <https://deepcell.org/>`_
- Segment Anything Model: Kirillov, A., et al. (2023). "Segment Anything." arXiv:2304.02643

Cellpose
~~~~~~~~

Cellpose is a generalist algorithm for cell segmentation that uses a deep neural network trained on diverse cell types.

**How it works:**
1. Nuclear and/or cytoplasm channels are preprocessed and combined
2. The Cellpose model predicts a flow field and cell probability map
3. The flow field guides cell boundary detection
4. Thresholds (flow_threshold, cellprob_threshold) filter the results
5. Final segmentation masks are generated

**Advantages:**
- Works offline (no API required)
- Supports GPU acceleration for faster processing
- Generalizable across many cell types
- Can be fine-tuned for specific applications

**Limitations:**
- Requires local installation and potentially GPU setup
- May need parameter tuning for optimal results

**Citation:**
- Stringer, C., et al. (2021). "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods, 18(1), 100-106. `DOI: 10.1038/s41592-020-01018-x <https://doi.org/10.1038/s41592-020-01018-x>`_
- `Cellpose GitHub <https://github.com/MouseLand/cellpose>`_

Watershed
~~~~~~~~~

Watershed segmentation is a classical image processing technique that uses marker-controlled watershed transformation.

**How it works:**
1. Nuclear channels are combined to create a nuclear marker image
2. Cytoplasm channels are combined to create a membrane/cytoplasm image
3. Distance transform is applied to the nuclear markers
4. Watershed algorithm floods from markers, using membrane image as boundaries
5. Post-processing filters cells by area (min_cell_area, max_cell_area)

**Advantages:**
- Fast and computationally efficient
- No external dependencies or API keys required
- Deterministic results (reproducible)
- Good for well-separated cells with clear nuclei

**Limitations:**
- Less accurate for touching or overlapping cells
- Requires good nuclear and membrane channel contrast
- May need careful parameter tuning

**Citation:**
- Vincent, L., & Soille, P. (1991). "Watersheds in digital spaces: an efficient algorithm based on immersion simulations." IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(6), 583-598. `DOI: 10.1109/34.87344 <https://doi.org/10.1109/34.87344>`_
- Implementation based on scikit-image: `scikit-image Watershed <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html>`_

Ilastik
~~~~~~~

Ilastik is an interactive learning and segmentation toolkit that allows you to train custom segmentation models on your specific data.

**How it works:**
1. Train a segmentation model in Ilastik's GUI using your IMC data
2. Save the trained model as a ``.ilp`` project file
3. In OpenIMC, select Ilastik as the segmentation method
4. Load your trained ``.ilp`` project file
5. OpenIMC runs inference using Ilastik's headless mode
6. Results are returned as segmentation masks

**Advantages:**
- Train custom models tailored to your specific data
- Interactive training allows fine-tuning on difficult cases
- Can handle complex segmentation tasks
- Works well for specialized cell types or tissue structures

**Limitations:**
- Requires separate Ilastik installation
- Requires training a model before use (time investment)
- Model quality depends on training data quality
- Processing can be slower than other methods

**Citation:**
- Berg, S., et al. (2019). "ilastik: interactive machine learning for (bio)image analysis." Nature Methods, 16(12), 1226-1232. `DOI: 10.1038/s41592-019-0582-9 <https://doi.org/10.1038/s41592-019-0582-9>`_
- `Ilastik Website <https://www.ilastik.org/>`_
- `Ilastik Documentation <https://www.ilastik.org/documentation/>`_

Tips and Best Practices
-----------------------

1. **Channel Selection**: Choose nuclear channels with strong, consistent staining. For cytoplasm channels, select markers that outline cell boundaries well.

2. **Preprocessing**: 
   - Use ``channelwise_minmax`` normalization when channels have different intensity ranges (recommended for most cases)
   - Apply ``arcsinh`` normalization if your data has a wide dynamic range
   - Use denoising for noisy channels

3. **Parameter Tuning**: Start with default parameters and adjust based on results:
   - If cells are over-segmented (too many small pieces), increase thresholds or min_cell_area
   - If cells are under-segmented (merged together), decrease thresholds or adjust fusion methods

4. **Method Selection**:
   - Use CellSAM for best accuracy and when API access is available
   - Use Cellpose for quicker processing with good accuracy
   - Use Watershed for fast processing of well-separated cells
   - Use Ilastik when you need custom segmentation models tailored to your specific data or cell types

5. **Validation**: Always visually inspect segmentation results before proceeding to feature extraction. Adjust parameters as needed.

