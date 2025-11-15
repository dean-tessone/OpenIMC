Installation
============

This guide covers installation of OpenIMC for different use cases. Choose the
installation method that best fits your needs: full installation (all features),
GUI-only installation, or CLI-only installation.

Prerequisites
-------------

**Python Version**
   OpenIMC requires Python 3.11 or higher.

**Important: datrie Dependency**
   Some users may encounter issues installing the ``datrie`` package, which is
   a dependency of certain OpenIMC components. If you encounter errors related
   to ``datrie`` during installation, install it from conda-forge **before**
   running ``pip install``:

   .. code-block:: bash

      conda install -c conda-forge datrie

   This is especially important on some Linux distributions and macOS systems
   where ``datrie`` may not build correctly from PyPI.

Full Installation
-----------------

The full installation includes all features: GUI interface, CLI tools, and all
optional dependencies for segmentation, clustering, and spatial analysis.

**Option 1: Conda Environment (Recommended)**

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd OpenIMC

   # Create conda environment
   conda create -n openimc python=3.11
   conda activate openimc

   # Install datrie if needed (see Prerequisites above)
   conda install -c conda-forge datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI)
   pip install -e .

   # Verify installation
   python main.py

**Option 2: Virtual Environment**

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd OpenIMC

   # Create virtual environment
   python3.11 -m venv openimc_env
   source openimc_env/bin/activate  # On Windows: openimc_env\Scripts\activate

   # Install datrie if needed (see Prerequisites above)
   # If using conda, run: conda install -c conda-forge datrie
   # Otherwise, pip may work: pip install datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI)
   pip install -e .

   # Verify installation
   python main.py


CLI Installation
---------------------

For headless batch processing on HPC systems or servers without display
capabilities, you can install OpenIMC's CLI tools. Note that some
GUI dependencies (like PyQt5) may still be installed as they are part of the
core requirements, but the CLI can be used without a display.

**Option 1: Conda Environment**

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd OpenIMC

   # Create conda environment
   conda create -n openimc python=3.11
   conda activate openimc

   # Install datrie if needed (see Prerequisites above)
   conda install -c conda-forge datrie

   # Install dependencies
   pip install -r requirements.txt

   # Install the package in editable mode (enables CLI)
   pip install -e .

   # Verify CLI installation
   openimc --help

**Option 2: Virtual Environment**

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd OpenIMC

   # Create virtual environment
   python3.11 -m venv openimc_env
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
   - The integration uses Ilastik's headless mode, so the full installation is
     required

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
   Run ``python main.py`` to launch the graphical interface.

**CLI Mode:**
   Run ``openimc --help`` to see available commands, or run a specific command
   like ``openimc preprocess --help``.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **"readimc is not installed"**
   .. code-block:: bash
      pip install readimc>=0.9.0

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
   - If you encounter build errors with ``datrie``, install it from conda-forge
     before running ``pip install -r requirements.txt``:
     .. code-block:: bash
        conda install -c conda-forge datrie

