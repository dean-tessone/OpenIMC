Image Visualization
===================

OpenIMC provides comprehensive image visualization capabilities for exploring IMC data, including multiple viewing modes, segmentation overlays, scaling options, and comparison tools.

Overview
--------

The image visualization system in OpenIMC allows you to:

- Load and display IMC images with efficient caching
- Select and view multiple channels simultaneously
- Switch between different viewing modes (Grid, RGB, Grayscale)
- Overlay segmentation masks with multiple display options
- Apply custom scaling, denoising, and spillover correction
- Zoom and navigate images
- Compare multiple acquisitions side-by-side

Image Loading and Caching
--------------------------

**Image Cache:**

OpenIMC uses an in-memory cache to store loaded images, improving performance when switching between channels or acquisitions.

- **Cache Key**: Images are cached by acquisition ID and channel name
- **Automatic Prefetching**: When an acquisition is selected, all channels are prefetched in the background
- **Cache Management**: Cache is automatically cleared when switching files or acquisitions
- **Thread-Safe**: Cache operations are thread-safe to support background loading

**Loading Process:**

1. Check cache for existing image
2. If not cached, load from MCD/OME-TIFF file
3. Apply denoising (if enabled)
4. Apply normalization (if configured)
5. Store in cache for future use

Channel Selection
-----------------

**Selecting Channels:**

- **Channel List**: Use the checkbox list to select one or more channels
- **Search**: Use the search box to filter channels by name
- **Select All**: Use "Select All" button to select all visible channels
- **Deselect All**: Use "Deselect All" button to clear selection

**Auto-Selection:**

- When switching acquisitions, previously selected channels are automatically re-selected if they exist in the new acquisition
- DNA channels (DNA1, DNA2) are auto-selected on initial file load if available

Viewing Modes
-------------

OpenIMC supports three viewing modes:

**1. Grid Mode** (default for multiple channels)

- Displays each selected channel in a separate subplot
- Useful for comparing multiple channels side-by-side
- Each channel can have independent scaling
- Enable with "Grid view for multiple channels" checkbox

**2. RGB Composite Mode**

- Combines selected channels into a single RGB image
- Assign channels to Red, Green, and Blue color components
- Supports multiple channels per color (averaged or combined)
- Useful for visualizing co-expression patterns
- Enable by unchecking "Grid view for multiple channels"

**3. Grayscale Mode**

- Converts RGB composite to grayscale (mean across channels)
- Or displays individual channels in grayscale colormap
- Useful for printing or when color is not needed
- Enable with "Grayscale mode" checkbox

**Switching Modes:**

- Toggle "Grid view" checkbox to switch between Grid and RGB modes
- Toggle "Grayscale mode" checkbox to enable/disable grayscale
- View automatically refreshes when mode changes

Segmentation Overlays
---------------------

**Enabling Overlays:**

- Check "Show segmentation overlay" to display segmentation masks
- Overlays are applied to all displayed channels
- Requires loaded segmentation masks (from segmentation or loaded from file)

**Overlay Modes:**

1. **Mask Mode** (default):
   - Displays segmentation masks as semi-transparent colored overlays
   - Each cell is colored based on its mask value
   - Useful for visualizing cell boundaries and spatial distribution

2. **Outline Mode**:
   - Displays only the cell outlines (contours)
   - Red outlines overlaid on the image
   - Useful for precise cell boundary visualization without obscuring image details

3. **Cluster Mode**:
   - Colors cells by their cluster assignment (if clustering has been performed)
   - Shows cluster colors with legend
   - Can display cluster mask only (without channel overlay) when no channels are selected
   - Useful for visualizing spatial distribution of cell types

**Switching Overlay Modes:**

- Use the "Mode" dropdown in the overlay controls
- Mode is applied immediately to all displayed images
- Cluster mode is only available if clustering has been performed

Scale Bar
---------

**Adding Scale Bar:**

- Check "Show scale bar" to display a scale bar on images
- Configure length in micrometers (default: 10 μm)
- Scale bar is automatically positioned in the bottom-right corner

**Parameters:**

- **Length (μm)**: Physical length of the scale bar in micrometers
- **Pixel Size**: Automatically detected from image metadata or can be manually set
- **Position**: Automatically positioned to avoid overlap with image content

**Usage:**

- Essential for publication-quality figures
- Helps readers understand spatial scale
- Works in both Grid and RGB modes

Custom Scaling
--------------

**What is Custom Scaling?**

Custom scaling allows you to adjust the intensity range (min/max values) used for displaying each channel. This controls how pixel intensities are mapped to display colors.

**How it Works:**

1. **Default Range**: Uses the full intensity range of the image (min to max)
2. **Manual Scaling**: Set custom min/max values to focus on specific intensity ranges
3. **Per-Channel Scaling**: Each channel can have independent scaling values
4. **RGB Color Scaling**: In RGB mode, each color component (Red, Green, Blue) can be scaled independently

**Scaling Methods:**

- **Default**: Full image range (min to max)
- **Manual**: Set custom min/max values
- **Arcsinh**: Apply arcsinh transformation with configurable cofactor (useful for high dynamic range data)

**Using Custom Scaling:**

1. Enable "Custom scaling" checkbox
2. Select channel or RGB color from dropdown
3. Adjust min/max spinboxes or use "Default range" button
4. For arcsinh: check "Arcsinh" and set cofactor (default: 10.0)
5. Scaling is applied immediately

**Tips:**

- Lower min values increase brightness
- Higher max values increase contrast
- Arcsinh helps visualize data with very high dynamic range
- In RGB mode, scaling affects the intensity of each color component

Denoising
---------

**What is Denoising?**

Denoising removes noise and artifacts from images to improve visualization quality. Denoising is applied per-channel and operates on raw pixel intensities before normalization.

**Denoising Methods:**

1. **Hot Pixel Removal**:
   - **Median 3x3**: Replace hot pixels with median of 3x3 neighborhood
   - **>N SD above local median**: Remove pixels more than N standard deviations above local median

2. **Speckle Smoothing**:
   - **Gaussian**: Apply Gaussian blur (configurable sigma)
   - **Non-local means**: Advanced denoising (slower but better quality)

3. **Background Subtraction**:
   - **White top-hat**: Remove bright background structures
   - **Black top-hat**: Remove dark background structures
   - **Rolling ball**: Approximate rolling ball background subtraction

**Using Denoising:**

1. Enable "Enable denoising" checkbox
2. Select channel from dropdown
3. Configure each denoising method (hot pixel, speckle, background)
4. Set processing order (Hot → Speckle → Background recommended)
5. Click "Apply to all channels" to use same settings for all channels
6. Denoising is applied automatically when viewing images

**Important Notes:**

- Denoising is **visualization-only** and does not affect feature extraction
- Settings are saved per-channel
- Requires scikit-image package
- Processing order matters: hot pixels first, then speckle, then background

Spillover Correction (Visualization)
-------------------------------------

**What is Spillover Correction?**

Spillover correction compensates for signal leakage between channels due to spectral overlap. In visualization mode, this is applied on-the-fly to show how images would look after correction.

**Important Warning:**

.. warning::
   **Spillover correction in visualization is for preview only.** To actually correct your data, you must apply spillover correction during feature extraction. Visualization correction does not modify the underlying data.

**Using Spillover Correction:**

1. Load a spillover matrix CSV file (via File → Load Spillover Matrix or Analysis → Generate Spillover Matrix)
2. Enable "Spillover correction" checkbox
3. Select correction method:
   - **NNLS** (Non-Negative Least Squares): Recommended, ensures non-negative results
   - **PGD** (Projected Gradient Descent): Alternative method
4. Correction is applied automatically when viewing images

**How it Works:**

1. Loads all selected channels with denoising only
2. Applies spillover correction using the loaded matrix
3. Applies normalization/scaling to corrected images
4. Displays corrected images

**Limitations:**

- Only affects visualization, not actual data
- Requires loaded spillover matrix
- Matrix channels must match image channels
- Correction is computationally intensive for many channels

Zoom Functionality
------------------

**Zooming:**

- **Mouse Wheel**: Scroll to zoom in/out
- **Navigation Toolbar**: Use zoom tools in matplotlib toolbar
- **Zoom to Rectangle**: Click and drag to zoom to selected region
- **Pan**: Click and drag to pan when zoomed in

**Zoom Preservation:**

- Zoom level is automatically preserved when:
  - Switching between channels (if same acquisition)
  - Toggling viewing modes
  - Changing overlay settings
  - Applying scaling or denoising

**Reset Zoom:**

- Click "Reset Zoom" button to return to full image view
- Or use navigation toolbar "Home" button

**Synchronized Zoom (Grid Mode):**

- In grid mode, zooming one channel zooms all channels to the same region
- Useful for comparing channels at the same spatial location

Comparison Mode
---------------

**Opening Comparison Mode:**

- Click "Comparison mode" button in the main window
- Opens a dialog for side-by-side comparison of multiple acquisitions

**Features:**

- **Multi-Acquisition Selection**: Select multiple acquisitions to compare
- **Channel Selection**: Choose channel to display across all acquisitions
- **Synchronized Viewing**: All acquisitions display the same channel
- **Individual Scaling**: Each acquisition can have independent scaling
- **Overlay Support**: Segmentation overlays can be enabled
- **Grayscale Mode**: Optional grayscale display

**Use Cases:**

- Compare the same ROI across different conditions
- Compare different ROIs from the same sample
- Quality control across acquisitions
- Batch visualization

**Controls:**

- Select acquisitions from list (multi-select)
- Choose channel from dropdown
- Adjust scaling per acquisition
- Toggle overlay and grayscale options
- Export individual or all images

Exporting Images
----------------

**Saving Figures:**

- Use File → Save Figure or right-click on image
- Opens save dialog with options:
  - **Format**: PNG, PDF, SVG, EPS, TIFF
  - **DPI**: Resolution (default: 300 for publication quality)
  - **Transparent Background**: Optional
  - **Font Size**: Adjust text size in saved figure

**Suggested Filenames:**

- Automatically generates filename based on:
  - File/slide name
  - Acquisition/ROI name
  - Selected channels
  - Viewing mode (RGB or Grid)

**Export Options:**

- **Single Image**: Current view
- **All Channels**: Export all channels in grid (via "Show all channels" button)
- **Comparison Mode**: Export individual or all comparison images

Tips and Best Practices
-----------------------

1. **Performance**:
   - Use grid mode for comparing many channels
   - Use RGB mode for visualizing co-expression
   - Cache improves performance when switching channels

2. **Visualization Quality**:
   - Adjust scaling to highlight features of interest
   - Use denoising for noisy images
   - Apply arcsinh for high dynamic range data

3. **Segmentation Overlays**:
   - Use Outline mode for precise boundary visualization
   - Use Cluster mode to visualize cell type distribution
   - Use Mask mode for general cell visualization

4. **Scale Bar**:
   - Always include scale bar in publication figures
   - Verify pixel size is correct for your data

5. **Spillover Correction**:
   - Remember: visualization correction is preview only
   - Apply correction during feature extraction for actual data correction

6. **Comparison Mode**:
   - Use for quality control across acquisitions
   - Adjust scaling individually for each acquisition
   - Export side-by-side comparisons for presentations

7. **Export**:
   - Use high DPI (300+) for publication figures
   - PNG for general use, PDF/SVG for vector graphics
   - Adjust font sizes for readability in saved figures

