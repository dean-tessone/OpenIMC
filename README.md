# OpenMCD - IMC .mcd File Viewer

A PyQt5-based viewer for IMC (Imaging Mass Cytometry) .mcd files using the readimc library. This application provides an intuitive graphical interface for visualizing and analyzing multi-channel imaging data from mass cytometry experiments.

## Features

### Core Functionality
- **Multi-acquisition Support**: View multiple acquisitions within a single .mcd file
- **Channel Visualization**: Display individual channels or create RGB composites
- **Flexible Display Modes**: 
  - Single channel view with grayscale or color mapping
  - RGB composite view with custom color channel assignments
  - Grid view for comparing multiple channels simultaneously
- **Dynamic Comparison Mode**: Compare the same channel across different acquisitions with advanced scaling options
- **Metadata Display**: View acquisition metadata including dimensions, channel information, and experimental details
- **Cell Segmentation (optional)**: Run Cellpose (cyto3/nuclei) with CPU/GPU acceleration and overlay masks
- **Feature Extraction (optional)**: Compute per-cell morphology and intensity features with optional multiprocessing and export to CSV
- **Cell Clustering Analysis (optional)**: Perform hierarchical and Leiden clustering on extracted cell features
- **UMAP Dimensionality Reduction (optional)**: Visualize high-dimensional cell data in 2D space using UMAP

### Advanced Features
- **Custom Scaling**: Per-channel intensity scaling with slider controls
- **Auto Contrast**: Automatic contrast optimization using percentile-based scaling
- **Annotation System**: Label channels with quality assessments (High-quality, Low-quality, Artifact/Exclude)
- **Export Capabilities**: Save annotations as CSV files for further analysis
- **Memory Management**: Intelligent caching system for efficient handling of large datasets
- **Multiple Clustering Algorithms**: Choose between hierarchical clustering and Leiden community detection
- **Feature Selection**: Interactive dialog to select morphometric and intensity features for clustering
- **Cluster Visualization**: Heatmaps, dendrograms, and cluster statistics for analysis results
- **UMAP Integration**: 2D embedding visualization with customizable parameters (n_neighbors, min_dist)
- **Cluster Explorer**: Detailed exploration of individual clusters with cell image visualization
- **Enhanced Comparison Mode**: RGB composite support, linked/unlinked scaling, and arcsinh normalization

### User Interface
- **Intuitive Controls**: Easy-to-use interface with clear organization
- **Real-time Updates**: Instant visualization updates when changing parameters
- **Flexible Layout**: Resizable windows and panels for optimal viewing
- **Color Assignment**: Custom RGB channel mapping for composite images

## Requirements

- Python 3.11
- PyQt5 for the graphical interface
- readimc library for .mcd file reading
- Standard scientific Python packages (numpy, pandas, matplotlib, scipy, scikit-image, scikit-learn, etc.)

### Optional Dependencies

For enhanced functionality, the following packages are optional but recommended:

- **Cellpose** (cellpose==3.1.1.2): For cell segmentation functionality
- **PyTorch** (torch>=1.9.0): For GPU-accelerated segmentation (CUDA/MPS support)
- **Leiden Algorithm** (leidenalg>=0.10.0, python-igraph>=0.11.0): For advanced clustering analysis
- **UMAP** (umap-learn>=0.5.0): For dimensionality reduction and visualization
- **Seaborn** (seaborn>=0.13.2): For enhanced statistical visualizations

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd OpenMCD
```

### 2. Create a Virtual Environment
```bash
python3.11 -m venv openmcd_env
source openmcd_env/bin/activate  # On Windows: openmcd_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python main.py
```

## Usage

### Starting the Application
```bash
python main.py
```

### Basic Workflow

1. **Open a .mcd File**
   - Click "Open .mcd" button or use File → Open .mcd
   - Select your IMC .mcd file from the file dialog

2. **Select Acquisition**
   - Choose an acquisition from the dropdown menu
   - View acquisition metadata in the information panel

3. **Select Channels**
   - Check the channels you want to visualize
   - Use "Deselect all" to clear selections
   - Choose display mode (single channel, RGB composite, or grid view)

4. **Customize Display**
   - Toggle grayscale mode for single-channel views
   - Enable custom scaling for fine-tuned intensity control
   - Assign channels to RGB colors for composite views

5. **View Images**
   - Click "View selected" to display your chosen channels
   - Use comparison mode to compare the same channel across acquisitions

6. **Run Segmentation (optional)**
   - Click "Cell Segmentation"
   - Choose model (cyto3 or nuclei) and configure preprocessing if needed
   - Optionally select GPU device (Auto/CPU/GPU) if PyTorch+CUDA or MPS are available
   - Enable "Show segmentation overlay" to visualize masks after completion

7. **Extract Features (optional)**
   - Click "Extract Features"
   - Select acquisitions (or all with masks) and feature sets
   - Choose an output directory/filename or keep results in memory
   - The results DataFrame includes both `acquisition_id` and `acquisition_label` for clarity

8. **Cell Clustering Analysis (optional)**
   - Click "Cell Clustering" (requires feature extraction first)
   - Select features for clustering (morphometric and/or intensity features)
   - Choose clustering method: Hierarchical or Leiden community detection
   - Configure clustering parameters (number of clusters, linkage method, etc.)
   - View results as heatmaps, dendrograms, and cluster statistics

9. **UMAP Visualization (optional)**
   - From the clustering dialog, click "Run UMAP"
   - Select features for dimensionality reduction
   - Customize UMAP parameters (n_neighbors, min_dist)
   - Visualize cell populations in 2D embedding space
   - Overlay cluster information on UMAP plots

### Advanced Features

#### Custom Scaling
- Enable "Custom scaling" checkbox
- Select a channel for scaling adjustment
- Use sliders or buttons to set intensity range:
  - **Auto Contrast**: Optimize contrast using 1st-99th percentiles
  - **Percentile Scaling**: Apply robust percentile scaling
  - **Default Range**: Use full image intensity range
- Click "Apply" to update the display

#### Comparison Mode
- Click "Comparison mode" to open the comparison dialog
- Select multiple acquisitions to compare
- Choose a channel to display across all selected acquisitions
- Use linked scaling for fair comparison or individual scaling for detailed analysis
- Create RGB composite images by selecting multiple channels for red, green, and blue
- Apply arcsinh normalization for improved visualization of high-dimensional data

#### Annotations
- Select channels and choose a label from the dropdown
- Click "Apply label" to annotate channels
- Save annotations as CSV for later analysis
- Load previously saved annotations

#### Clustering Analysis
- **Feature Selection**: Choose from morphometric features (area, perimeter, eccentricity, etc.) and intensity features (mean, median, std, etc.)
- **Clustering Methods**: 
  - Hierarchical clustering with customizable linkage methods (ward, complete, average, single)
  - Leiden community detection for graph-based clustering with two modes:
    - **Resolution parameter**: Control cluster granularity (0.1-5.0, default 1.0)
    - **Modularity optimization**: Automatically optimize for best modularity score
- **Visualization**: View results as heatmaps, dendrograms, and cluster statistics
- **Cluster Explorer**: Examine individual clusters with cell image visualization and channel-specific analysis

#### UMAP Dimensionality Reduction
- **Feature Selection**: Choose features for 2D embedding visualization
- **Parameter Tuning**: Adjust n_neighbors (default: 15) and min_dist (default: 0.1) for optimal visualization
- **Cluster Overlay**: Visualize clustering results on UMAP plots
- **Interactive Exploration**: Zoom, pan, and explore the 2D embedding space

## Troubleshooting

### Common Issues

1. **"readimc is not installed" error**
   - Ensure readimc is installed: `pip install readimc>=0.9.0`

2. **Cellpose/GPU segmentation not available**
   - Cellpose is optional. Install to enable segmentation: `pip install cellpose`
   - For CUDA GPUs, also install PyTorch with CUDA (see PyTorch install guide)
   - On Apple Silicon, MPS is auto-detected when available

3. **PyQt5 installation issues**
   - On some systems, you may need to install PyQt5 separately:
   - Ubuntu/Debian: `sudo apt-get install python3-pyqt5`
   - macOS: `brew install pyqt5`

4. **Display issues**
   - Ensure your system supports the required OpenGL version for PyQt5
   - Try running with different Qt backends if needed 

5. **Memory issues with large files**
   - The application includes caching, but very large datasets may require more RAM
   - Consider closing other applications to free up memory

6. **Multiprocessing during feature extraction**
   - The app uses multiprocessing where possible; if it fails, it falls back to single-threaded processing automatically
   - If you see pickling-related errors, ensure you start via `python main.py` (not from within embedded REPLs)

7. **Clustering and UMAP dependencies**
   - For Leiden clustering: Install `leidenalg` and `python-igraph`: `pip install leidenalg python-igraph`
   - For UMAP visualization: Install `umap-learn`: `pip install umap-learn`
   - For enhanced visualizations: Install `seaborn`: `pip install seaborn`
   - These are optional - the app will work without them but clustering features will be limited

8. **Memory issues with large clustering datasets**
   - UMAP and clustering can be memory-intensive for large cell populations (>10,000 cells)
   - Consider subsampling your data or using fewer features if you encounter memory issues
   - The app includes progress indicators for long-running clustering operations

## Acknowledgments

- **Cellpose**: GPU-accelerated cell segmentation framework used for segmentation functionality.
  - Paper: Stringer et al., Cellpose: a generalist algorithm for cellular segmentation
  - Project: https://www.cellpose.org/
- **readimc**: IMC file reader used to load .mcd acquisitions and channel data.
  - Project: https://github.com/BodenmillerGroup/readimc
- **UMAP**: Uniform Manifold Approximation and Projection for dimensionality reduction.
  - Paper: McInnes et al., UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
  - Project: https://umap-learn.readthedocs.io/
- **Leiden Algorithm**: Community detection algorithm for graph-based clustering.
  - Paper: Traag et al., From Louvain to Leiden: guaranteeing well-connected communities
  - Project: https://github.com/vtraag/leidenalg
- **scikit-learn**: Machine learning library providing clustering algorithms and utilities.
  - Project: https://scikit-learn.org/
- **seaborn**: Statistical data visualization library for enhanced plotting capabilities.
  - Project: https://seaborn.pydata.org/

## License

This project is licensed under the MIT License.
