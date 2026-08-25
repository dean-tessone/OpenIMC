# OpenIMC

![OpenIMC Logo](docs/source/_static/images/OpenIMC_Logo.png)

OpenIMC is a comprehensive, open-source PyQt5-based platform for analyzing Imaging Mass Cytometry (IMC) data. It provides an intuitive graphical interface for visualizing, processing, and analyzing multi-channel imaging data from mass cytometry experiments with advanced machine learning capabilities.

## Paper and Citation
See the paper here: **https://link.springer.com/article/10.1186/s12859-026-06547-4**

Cite as: Tessone, D., Kamal, M., Hennes, V. et al. OpenIMC: an open-source platform for analyzing single-cell and spatial proteomics by imaging mass cytometry. BMC Bioinformatics (2026). https://doi.org/10.1186/s12859-026-06547-4

## Documentation

For complete documentation, installation instructions, and usage guides, please visit:

**https://dean-tessone.github.io/OpenIMC/overview.html**

## Download OpenIMC

Most users should install the ready-to-run desktop application. Python, Git,
or a virtual environment are not required. Download OpenIMC 1.1.1 for your
computer:

| Operating system | Download |
| --- | --- |
| Windows 10/11, 64-bit | **[Windows ZIP](https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-windows-x86_64.zip)** |
| Mac with Apple Silicon (M1 or newer) | **[Apple Silicon Mac installer](https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-darwin-arm64.pkg)** |
| Mac with an Intel processor | **[Intel Mac installer](https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-darwin-x86_64.pkg)** |
| Ubuntu 22.04 or newer, 64-bit | **[Ubuntu installer](https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-linux-amd64.deb)** |

Not sure which Mac you have? Choose **Apple menu → About This Mac**. Download
the Apple Silicon installer if the **Chip** line says Apple M1, M2, M3, M4, or
newer. Download the Intel installer if it shows an Intel **Processor**.

### Windows installation

1. Download the Windows ZIP and choose **Extract all** in File Explorer.
2. Open the extracted folder and double-click `OpenIMC.exe`.
3. Windows may show **Windows protected your PC** for a new or unsigned
   download. Confirm that the file came from this official repository, click
   **More info**, then click **Run anyway**. Do not disable SmartScreen.

### macOS installation

1. Download the installer for your Mac and double-click the `.pkg` file.
2. If macOS blocks it, close the warning and open **Apple menu → System
   Settings → Privacy & Security**.
3. Scroll to **Security**, click **Open Anyway**, authenticate, and confirm.
   Re-open the installer if it does not resume automatically.
4. Complete the installer, then open **OpenIMC** from Applications. If macOS
   blocks the app itself on first launch, use **Open Anyway** once more.

The installer may request an administrator password because it places OpenIMC
in the system Applications folder.

### Ubuntu installation

1. Download the Ubuntu `.deb` file.
2. Double-click it, click **Install**, and authenticate when Ubuntu asks.
3. Open **OpenIMC** from the Applications menu.

If the graphical installer does not open, open a terminal in the Downloads
folder and run:

```bash
sudo apt install ./OpenIMC-1.1.1-linux-amd64.deb
```

The Windows and Ubuntu downloads use the CPU by default. On a compatible
NVIDIA computer, OpenIMC offers a **Download CUDA support** button at startup
until the optional GPU packages are installed and verified. Apple Silicon Macs
use Apple's built-in GPU support; Intel Macs use the CPU.

The [release page](https://github.com/dean-tessone/OpenIMC/releases/tag/v1.1.1)
also provides download fingerprints (`SHA256SUMS.txt`) and a zipped software
inventory for security review.

## Quick Start

### Installation from source

The source installation is intended for developers and advanced users.

The preferred installation pattern uses `uv`:

```bash
# Clone the repository
git clone https://github.com/dean-tessone/OpenIMC.git
cd OpenIMC

# Create and activate a uv-managed virtual environment
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Install the package
uv pip install -e .
```

If you prefer Conda, that workflow is still supported in the
[Installation documentation](https://dean-tessone.github.io/OpenIMC/installation.html).

For detailed installation instructions, including alternative methods and troubleshooting, see the [Installation documentation](https://dean-tessone.github.io/OpenIMC/installation.html).

A video tutorial is available here: [Video Tutorial](https://youtu.be/CKSwJE3jdi0?si=J9Eei4c2iC_D_VQc).

### Running OpenIMC

After installation, you can run:

```bash
# Start the GUI application
openimc-gui

# Or run the CLI
openimc --help
```

## License

OpenIMC – Interactive analysis toolkit for IMC data

Copyright (C) 2025 University of Southern California

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program (see LICENSE). If not, see <https://www.gnu.org/licenses/>.
