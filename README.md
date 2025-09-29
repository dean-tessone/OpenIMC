# OpenMCD - Advanced IMC Data Analysis Platform

OpenMCD is a comprehensive PyQt5-based platform for analyzing Imaging Mass Cytometry (IMC) data. It provides an intuitive graphical interface for visualizing, processing, and analyzing multi-channel imaging data from mass cytometry experiments with advanced machine learning capabilities.

## 🚀 Key Features

### 📊 **Image Viewing & Visualization**
- **Multi-acquisition Support**: View and analyze multiple acquisitions within a single .mcd file
- **Flexible Display Modes**: Single channel, RGB composite, and grid view for comprehensive data exploration
- **Custom Scaling**: Advanced intensity scaling with percentile-based optimization and manual controls
- **Dynamic Comparison**: Compare channels across different acquisitions with linked/unlinked scaling

### 🔬 **Marker Quality Control**
- **Interactive Annotation System**: Label channels with quality assessments (High-quality, Low-quality, Artifact/Exclude)
- **Export Capabilities**: Save and load annotations as CSV files for reproducible analysis
- **Visual Quality Assessment**: Real-time visualization with custom scaling for marker evaluation

### 🎯 **Advanced Segmentation**
- **Cellpose Integration**: GPU-accelerated cell segmentation using state-of-the-art Cellpose models
- **Multiple Models**: Support for cyto3 and nuclei segmentation models
- **Overlay Visualization**: Real-time mask overlay on original images
- **GPU Acceleration**: CUDA and MPS support for faster processing

### 📈 **Feature Extraction**
- **Comprehensive Feature Sets**: Extract morphometric (area, perimeter, eccentricity) and intensity features (mean, median, std)
- **Multiprocessing Support**: Parallel processing for efficient feature computation
- **Export Options**: Save results to CSV or keep in memory for further analysis
- **Per-cell Analysis**: Detailed feature extraction for individual cells

### 🧬 **Clustering Analysis**
- **Multiple Algorithms**: Hierarchical clustering and Leiden community detection
- **Feature Selection**: Interactive dialog to select relevant features for clustering
- **Advanced Visualization**: Heatmaps, dendrograms, and cluster statistics
- **Cluster Explorer**: Detailed exploration of individual clusters with cell visualization

### 🤖 **LLM-Based Cell Phenotyping**
- **OpenAI Integration**: Leverage GPT models for intelligent cell phenotype annotation
- **Automated Classification**: AI-powered cell type identification and characterization
- **Custom Phenotypes**: Define and train custom phenotype categories
- **Batch Processing**: Process large datasets with AI-assisted annotation

## 📋 Workflow

### 1. **Image Loading**
- Open .mcd files containing IMC acquisitions
- Browse and select specific acquisitions for analysis
- View acquisition metadata and channel information

### 2. **Marker QC**
- Visualize individual channels with custom scaling
- Annotate channels based on quality assessment
- Export QC annotations for documentation and reproducibility

### 3. **Segmentation**
- Run Cellpose segmentation (cyto3 or nuclei models)
- Configure preprocessing parameters and GPU acceleration
- Visualize segmentation masks with overlay options

### 4. **Feature Extraction**
- Extract comprehensive cell features (morphometric and intensity)
- Select specific feature sets for analysis
- Export results for downstream analysis

### 5. **Clustering on QC Features**
- Perform hierarchical or Leiden clustering on extracted features
- Visualize results with heatmaps and dendrograms
- Explore individual clusters with detailed statistics

### 6. **Phenotype Annotation**
- **LLM-based**: Use OpenAI GPT models for automated cell phenotyping
- **Custom**: Define and apply custom phenotype categories
- Export annotated results for further analysis

## 🛠️ Installation & Setup

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd OpenMCD

# Create conda environment
conda create -n openmcd python=3.11
conda activate openmcd

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py
```

### Option 2: Virtual Environment

```bash
# Clone the repository
git clone <repository-url>
cd OpenMCD

# Create virtual environment
python3.11 -m venv openmcd_env
source openmcd_env/bin/activate  # On Windows: openmcd_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py
```

## 🔑 OpenAI API Key Setup

To use the LLM-based cell phenotyping features, you'll need an OpenAI API key:

### 1. **Generate API Key**
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to the API section
4. Click "Create new secret key"
5. Copy the generated API key (starts with `sk-`)

## 📦 Requirements

### Core Dependencies
- Python 3.11
- PyQt5 (GUI framework)
- readimc (IMC file reading)
- numpy, pandas, matplotlib, scipy (scientific computing)
- scikit-image, scikit-learn (image processing and ML)

### Optional Dependencies
- **Cellpose** (3.1.1.2): Cell segmentation
- **PyTorch** (≥1.9.0): GPU acceleration
- **Leiden Algorithm**: Advanced clustering
- **UMAP** (≥0.5.0): Dimensionality reduction
- **OpenAI** (≥1.42.0): LLM-based phenotyping

## 🚀 Quick Start

```bash
# Start the application
python main.py

# Basic workflow:
# 1. Open .mcd file → 2. Select acquisition → 3. Run segmentation
# 4. Extract features → 5. Perform clustering → 6. Annotate phenotypes
```

## 🔧 Troubleshooting

### Common Issues

1. **"readimc is not installed"**
   ```bash
   pip install readimc>=0.9.0
   ```

2. **GPU segmentation not available**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **OpenAI API errors**
   - Verify your API key is correctly set
   - Check your OpenAI account has sufficient credits
   - Ensure internet connectivity

4. **Memory issues with large datasets**
   - Close other applications to free RAM
   - Consider subsampling for clustering analysis
   - Use multiprocessing for feature extraction

## 📚 Acknowledgments

- **Cellpose**: GPU-accelerated cell segmentation framework
  - Paper: Stringer et al., Cellpose: a generalist algorithm for cellular segmentation
  - Project: https://www.cellpose.org/

- **readimc**: IMC file reader for .mcd acquisitions
  - Project: https://github.com/BodenmillerGroup/readimc

- **OpenAI**: Large language models for cell phenotyping
  - Project: https://openai.com/

- **UMAP**: Uniform Manifold Approximation and Projection
  - Paper: McInnes et al., UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
  - Project: https://umap-learn.readthedocs.io/

- **Leiden Algorithm**: Community detection for graph-based clustering
  - Paper: Traag et al., From Louvain to Leiden: guaranteeing well-connected communities
  - Project: https://github.com/vtraag/leidenalg

- **scikit-learn**: Machine learning algorithms and utilities
  - Project: https://scikit-learn.org/

- **seaborn**: Statistical data visualization
  - Project: https://seaborn.pydata.org/

## 📄 License

This project is licensed under the MIT License.