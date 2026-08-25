Installation
============

Most users should use the ready-to-run desktop application. It already includes
Python and the packages OpenIMC needs. The source installation sections are
available below for developers, command-line users, and advanced environments.

Desktop application (recommended)
---------------------------------

Download OpenIMC 1.1.1 for your computer:

* `Windows 10/11, 64-bit ZIP <https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-windows-x86_64.zip>`_
* `Mac installer for Apple Silicon <https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-darwin-arm64.pkg>`_
* `Mac installer for Intel processors <https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-darwin-x86_64.pkg>`_
* `Ubuntu 22.04 or newer, 64-bit installer <https://github.com/dean-tessone/OpenIMC/releases/download/v1.1.1/OpenIMC-1.1.1-linux-amd64.deb>`_

You do not need to install Python, Git, Conda, or a virtual environment when
using these downloads.

Windows 10 or 11
~~~~~~~~~~~~~~~~

#. Download the Windows ZIP.
#. In File Explorer, right-click the ZIP and choose **Extract All**. Do not run
   OpenIMC from inside the compressed ZIP.
#. Open the extracted folder and double-click ``OpenIMC.exe``.
#. Microsoft Defender SmartScreen may display **Windows protected your PC** for
   a new or unsigned download. Confirm that you downloaded the file from this
   official OpenIMC repository, click **More info**, and then click **Run
   anyway**.

Do not turn off SmartScreen. Approving only this OpenIMC file preserves the
rest of Windows' protection. A computer managed by an employer or university
may prevent unsigned applications entirely; in that case, contact its IT
administrator.

For more context, see Microsoft's explanation of
`SmartScreen warnings for unsigned applications <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation>`_.

macOS
~~~~~

First choose the correct installer:

* Open **Apple menu → About This Mac**.
* If the **Chip** line says Apple M1, M2, M3, M4, or newer, use the Apple
  Silicon installer.
* If the Mac shows an Intel **Processor**, use the Intel installer.

Then install OpenIMC:

#. Download and double-click the appropriate ``.pkg`` file.
#. If macOS says that it cannot verify or open the installer, close the
   warning.
#. Open **Apple menu → System Settings → Privacy & Security**.
#. Scroll to **Security**, click **Open Anyway**, authenticate with your Mac
   login, and confirm that you want to open it.
#. Re-open the ``.pkg`` installer if it does not resume automatically, then
   complete the guided installation. An administrator password may be required
   because OpenIMC is installed in the system Applications folder.
#. Open **OpenIMC** from Applications. If macOS separately blocks the
   application on its first launch, repeat the **Open Anyway** step for the app.

The **Open Anyway** button is available for about one hour after the blocked
attempt. Approve OpenIMC only when it came from this repository. See Apple's
official `instructions for opening an app from an unknown developer
<https://support.apple.com/guide/mac-help/mh40616/mac>`_ for the current macOS
wording and security guidance.

Ubuntu
~~~~~~

#. Download the Ubuntu ``.deb`` file.
#. Double-click the file, click **Install**, and authenticate when Ubuntu asks.
#. Open **OpenIMC** from the Applications menu.

If Ubuntu's graphical installer does not open, open a terminal in the folder
containing the download and run:

.. code-block:: bash

   sudo apt install ./OpenIMC-1.1.1-linux-amd64.deb

GPU acceleration
~~~~~~~~~~~~~~~~

The Windows and Ubuntu downloads use the CPU by default. If OpenIMC detects a
compatible NVIDIA CUDA system, it offers a **Download CUDA support** button
when it starts. The prompt returns at each startup until the optional packages
finish downloading and pass a real GPU check. Internet access and several
gigabytes of free disk space may be required.

Apple Silicon Macs use Apple's built-in GPU support and do not use CUDA. Intel
Macs use the CPU.

Download verification
~~~~~~~~~~~~~~~~~~~~~

The `OpenIMC 1.1.1 release page
<https://github.com/dean-tessone/OpenIMC/releases/tag/v1.1.1>`_ also contains:

* ``SHA256SUMS.txt``, which lists fingerprints for confirming that downloads
  arrived unchanged.
* ``openimc-sboms.zip``, a software inventory intended mainly for security
  review. It is not needed to run OpenIMC.

Source installation prerequisites
---------------------------------

OpenIMC supports both ``uv`` and Conda-based source workflows. ``uv`` is the
preferred option because it provides a fast, lightweight virtual environment
and package-management workflow.

**Python Version**
   OpenIMC requires Python 3.12 or higher.

**Tested System Configurations**
   OpenIMC has been tested on the following operating systems and hardware configurations:

   **Linux**
      - Ubuntu 24.04.02
      - AMD Ryzen 9 3900x 24 Core CPU
      - 64 GB RAM
      - Dual GPU: NVIDIA RTX 5000 + NVIDIA T1000

   **macOS**
      - M2 MacBook Air
      - 16 GB RAM
      - Tahoe 26.1

   **Windows**
      - Windows 11
      - 16 GB RAM
      - 6 core CPU

**Important: datrie Dependency**
   Some users may encounter issues installing the ``datrie`` package, which is
   a dependency of certain OpenIMC components. If you encounter errors related
   to ``datrie`` during installation, first try installing it into your active
   environment before re-running the main dependency install:

   .. code-block:: bash

      uv pip install datrie

   If ``datrie`` still fails to build, install it from conda-forge instead:

   .. code-block:: bash

      conda install -c conda-forge datrie

   This is especially important on some Linux distributions and macOS systems
   where ``datrie`` may not build correctly from PyPI.

Full Installation
-----------------

The full installation includes all features: GUI interface, CLI tools, and all
optional dependencies for segmentation, clustering, and spatial analysis.

**Option 1: uv Environment (Recommended)**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dean-tessone/OpenIMC.git
   cd OpenIMC

   # Create and activate a uv-managed virtual environment
   uv venv --python 3.12
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # If datrie fails later, see the note in Prerequisites above

   # Install dependencies
   uv pip install -r requirements.txt

   # Install the package in editable mode (enables CLI and GUI commands)
   uv pip install -e .

   # Verify installation - run GUI
   openimc-gui

   # Or verify CLI installation
   openimc --help

**Option 2: Conda Environment**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dean-tessone/OpenIMC.git
   cd OpenIMC

   # Create conda environment
   conda create -n openimc python=3.12
   conda activate openimc

   # If datrie fails later, install it from conda-forge
   # conda install -c conda-forge datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI and GUI commands)
   pip install -e .

   # Verify installation - run GUI
   openimc-gui

   # Or verify CLI installation
   openimc --help

**Option 3: Standard Virtual Environment**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dean-tessone/OpenIMC.git
   cd OpenIMC

   # Create virtual environment
   python3.12 -m venv openimc_env
   source openimc_env/bin/activate  # On Windows: openimc_env\Scripts\activate

   # Install datrie if needed (see Prerequisites above)
   # If using conda, run: conda install -c conda-forge datrie
   # Otherwise, pip may work: pip install datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI and GUI commands)
   pip install -e .

   # Verify installation - run GUI
   openimc-gui

   # Or verify CLI installation
   openimc --help


CLI Installation
---------------------

For headless batch processing on HPC systems or servers without display
capabilities, you can install OpenIMC's CLI tools. Note that some
GUI dependencies (like PyQt5) may still be installed as they are part of the
core requirements, but the CLI can be used without a display.

**Option 1: uv Environment (Recommended)**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dean-tessone/OpenIMC.git
   cd OpenIMC

   # Create and activate a uv-managed virtual environment
   uv venv --python 3.12
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # If datrie fails later, see the note in Prerequisites above

   # Install dependencies
   uv pip install -r requirements.txt

   # Install the package in editable mode (enables CLI)
   uv pip install -e .

   # Verify CLI installation
   openimc --help

**Option 2: Conda Environment**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dean-tessone/OpenIMC.git
   cd OpenIMC

   # Create conda environment
   conda create -n openimc python=3.12
   conda activate openimc

   # If datrie fails later, install it from conda-forge
   # conda install -c conda-forge datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI)
   pip install -e .

   # Verify CLI installation
   openimc --help

**Option 3: Standard Virtual Environment**

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/dean-tessone/OpenIMC.git
   cd OpenIMC

   # Create virtual environment
   python3.12 -m venv openimc_env
   source openimc_env/bin/activate  # On Windows: openimc_env\Scripts\activate

   # Install datrie if needed (see Prerequisites above)
   # If using conda, run: conda install -c conda-forge datrie
   # Otherwise, pip may work: pip install datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI)
   pip install -e .

   # Verify CLI installation
   openimc --help

Optional Software Setup
-----------------------

Ilastik Installation
~~~~~~~~~~~~~~~~~~~~

To use Ilastik segmentation, you need to install Ilastik separately (it's not a
Python package):

1. **Download Ilastik**
   - Visit https://www.ilastik.org/download
   - Download the appropriate version for your operating system
   - Follow the installation instructions for your platform

2. **Verify Installation**
   - Ensure the ``ilastik`` command is available in your PATH
   - Test by running: ``ilastik --version`` in your terminal
   - The integration uses Ilastik's headless mode, so the full installation is required

3. **Using Ilastik Models**
   - Train your segmentation model in Ilastik's GUI
   - Save your trained project as a ``.ilp`` file
   - In OpenIMC, select "Ilastik" as the segmentation method
   - Browse and select your ``.ilp`` project file
   - Run inference on your images

OpenAI API Key Setup
~~~~~~~~~~~~~~~~~~~~

To use the LLM-based cell phenotyping features, you'll need an OpenAI API key:

1. **Generate API Key**
   - Visit `OpenAI Platform <https://platform.openai.com/>`_
   - Sign up or log in to your account
   - Navigate to the API section
   - Click "Create new secret key"
   - Copy the generated API key (starts with ``sk-``)

DeepCell API Token Setup
~~~~~~~~~~~~~~~~~~~~~~~~

To use the DeepCell CellSAM segmentation method, you'll need a DeepCell API token:

1. **Generate API Token**

   - Visit `DeepCell User Portal <https://users.deepcell.org/login/>`_
   - Sign up or log in to your account
   - Your username is your registration email without the domain suffix
     (e.g., if your email is ``user@example.com``, your username is ``user``)
   - Navigate to your account settings or API section
   - Generate or copy your API token
   - The API token is used to download the most up-to-date CellSAM model weights

2. **Set the API Token**

   You can set the API token in one of the following ways:

   **Option A: Environment Variable (Recommended for CLI)**

   .. code-block:: bash

      export DEEPCELL_ACCESS_TOKEN="your-api-token-here"

   **Option B: GUI Settings**

   - When using the GUI, enter your API token in the "DeepCell CellSAM Parameters"
     section of the segmentation dialog
   - The token will be saved in your user preferences for future use


Verification
------------

After installation, verify that OpenIMC is working correctly:

**GUI Mode:**
   Run ``openimc-gui`` to launch the graphical interface.

**CLI Mode:**
   Run ``openimc --help`` to see available commands, or run a specific command
   like ``openimc preprocess --help``.

   Note: ``python -m openimc`` also runs the CLI (equivalent to ``openimc``).

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **"readimc is not installed"**

   .. code-block:: bash

      pip install "readimc>=0.9.0"

2. **GPU segmentation not available**

   .. code-block:: bash

      # Install PyTorch with CUDA support
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

3. **Ilastik not found**

   - Install Ilastik from https://www.ilastik.org/download
   - Ensure the ``ilastik`` command is available in your PATH
   - For headless mode, Ilastik must be properly installed and accessible from
     command line
   - The integration uses Ilastik's headless mode, so full installation is
     required

4. **OpenAI API errors**

   - Verify your API key is correctly set
   - Check your OpenAI account has sufficient credits
   - Ensure internet connectivity

5. **Memory issues with large datasets**

   - Close other applications to free RAM
   - Consider subsampling for clustering analysis
   - Use multiprocessing for feature extraction

6. **datrie installation errors**

   If you encounter build errors with ``datrie``, install it from conda-forge
   after trying ``uv pip install datrie`` or ``pip install datrie`` in your
   active environment:

   .. code-block:: bash

      conda install -c conda-forge datrie
