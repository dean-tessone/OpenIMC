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
- **Dynamic Comparison Mode**: Compare the same channel across different acquisitions
- **Metadata Display**: View acquisition metadata including dimensions, channel information, and experimental details

### Advanced Features
- **Custom Scaling**: Per-channel intensity scaling with slider controls
- **Auto Contrast**: Automatic contrast optimization using percentile-based scaling
- **Annotation System**: Label channels with quality assessments (High-quality, Low-quality, Artifact/Exclude)
- **Export Capabilities**: Save annotations as CSV files for further analysis
- **Memory Management**: Intelligent caching system for efficient handling of large datasets

### User Interface
- **Intuitive Controls**: Easy-to-use interface with clear organization
- **Real-time Updates**: Instant visualization updates when changing parameters
- **Flexible Layout**: Resizable windows and panels for optimal viewing
- **Color Assignment**: Custom RGB channel mapping for composite images

## Requirements

- Python 3.11
- PyQt5 for the graphical interface
- readimc library for .mcd file reading
- Standard scientific Python packages (numpy, pandas, matplotlib, etc.)

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
python open_mcd.py --help
```

## Usage

### Starting the Application
```bash
python open_mcd.py
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

#### Annotations
- Select channels and choose a label from the dropdown
- Click "Apply label" to annotate channels
- Save annotations as CSV for later analysis
- Load previously saved annotations

## Troubleshooting

### Common Issues

1. **"readimc is not installed" error**
   - Ensure readimc is installed: `pip install readimc>=0.9.0`

2. **PyQt5 installation issues**
   - On some systems, you may need to install PyQt5 separately:
   - Ubuntu/Debian: `sudo apt-get install python3-pyqt5`
   - macOS: `brew install pyqt5`

3. **Memory issues with large files**
   - The application includes caching, but very large datasets may require more RAM
   - Consider closing other applications to free up memory

4. **Display issues**
   - Ensure your system supports the required OpenGL version for PyQt5
   - Try running with different Qt backends if needed

### Getting Help

If you encounter issues:
1. Check that all dependencies are correctly installed
2. Verify your .mcd file is not corrupted
3. Ensure you have sufficient system memory
4. Check the console output for detailed error messages

## License

This project is licensed under the MIT License.
