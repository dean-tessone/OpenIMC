# OpenIMC

![OpenIMC Logo](docs/source/_static/images/OpenIMC_Logo.png)

OpenIMC is a comprehensive, open-source PyQt5-based platform for analyzing Imaging Mass Cytometry (IMC) data. It provides an intuitive graphical interface for visualizing, processing, and analyzing multi-channel imaging data from mass cytometry experiments with advanced machine learning capabilities.

## Preprint and Citation
See the paper here: **https://link.springer.com/article/10.1186/s12859-026-06547-4**

Cite as: Tessone, D., Kamal, M., Hennes, V. et al. OpenIMC: an open-source platform for analyzing single-cell and spatial proteomics by imaging mass cytometry. BMC Bioinformatics (2026). https://doi.org/10.1186/s12859-026-06547-4

## Documentation

For complete documentation, installation instructions, and usage guides, please visit:

**https://dean-tessone.github.io/OpenIMC/overview.html**

## Download OpenIMC

Most users should install the ready-to-run desktop application. Python, Git,
and a virtual environment are not required.

**[Download the latest OpenIMC release](https://github.com/dean-tessone/OpenIMC/releases/latest)**

Choose the file for your computer:

| Operating system | Release file | How to start |
| --- | --- | --- |
| Windows 10/11, 64-bit | `OpenIMC-*-windows-x86_64.zip` | Extract the zip and double-click `OpenIMC.exe`. |
| Mac with Apple Silicon (M1 or newer) | `OpenIMC-*-darwin-arm64.pkg` | Double-click the package and follow the macOS Installer. |
| Mac with an Intel processor | `OpenIMC-*-darwin-x86_64.pkg` | Double-click the package and follow the macOS Installer. |
| Ubuntu 22.04 or newer, 64-bit | `OpenIMC-*-linux-amd64.deb` | Double-click it, click **Install**, then open OpenIMC from the Applications menu. |

Each release also includes SHA-256 checksum files and a software bill of
materials. Windows releases are code-signed. macOS releases are Developer
ID-signed and notarized by Apple. A drag-to-Applications `.dmg` is included as
an alternative for each Mac architecture. The guided macOS Installer may ask
for an administrator password because it installs OpenIMC for every user in
the system Applications folder. Ubuntu releases retain a portable `.tar.gz`
for advanced users who prefer not to install a system package.

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
