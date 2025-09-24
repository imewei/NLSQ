Installation Guide
==================

This guide provides comprehensive installation instructions for NLSQ across different platforms and use cases.

Quick Start
-----------

For most users, the simplest installation method is:

**Linux/macOS**::

    # For CPU-only
    pip install --upgrade "jax[cpu]>=0.4.20" nlsq

    # For GPU with CUDA 12
    pip install --upgrade "jax[cuda12]>=0.4.20" nlsq

**Windows**::

    # CPU-only (works on all Windows versions)
    pip install "jax[cpu]>=0.4.20" nlsq

System Requirements
-------------------

**Minimum Requirements:**

- Python 3.12 or higher (3.13 also supported)
- 4 GB RAM (8 GB recommended for large datasets)
- 2 GB free disk space

**Recommended Requirements:**

- Python 3.12+
- 8 GB RAM or more
- NVIDIA GPU with CUDA 12+ (for GPU acceleration)
- SSD storage for better I/O performance with large datasets

**Software Dependencies:**

- JAX 0.4.20 - 0.7.2
- NumPy 1.26.0+
- SciPy 1.11.0+
- psutil (for memory monitoring)

Platform-Specific Installation
-------------------------------

Linux (Recommended Platform)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NLSQ works best on Linux systems with full JAX support.

**CPU-only installation:**

.. code-block:: bash

    # Create virtual environment (recommended)
    python -m venv nlsq-env
    source nlsq-env/bin/activate

    # Install NLSQ with CPU support
    pip install --upgrade "jax[cpu]>=0.4.20" nlsq

    # Verify installation
    python -c "import nlsq; print(f'NLSQ {nlsq.__version__} installed successfully')"

**GPU installation (CUDA 12):**

.. code-block:: bash

    # Ensure NVIDIA drivers are installed and up to date
    nvidia-smi

    # Create virtual environment
    python -m venv nlsq-env
    source nlsq-env/bin/activate

    # Install NLSQ with CUDA 12 support
    pip install --upgrade "jax[cuda12]>=0.4.20" nlsq

    # Verify GPU access
    python -c "import jax; print(f'JAX devices: {jax.devices()}')"

macOS
~~~~~

macOS users can install NLSQ with CPU acceleration. GPU support is limited to Apple Silicon Macs.

**Intel Macs:**

.. code-block:: bash

    # Use Homebrew Python (recommended)
    brew install python@3.12

    # Create virtual environment
    python3.12 -m venv nlsq-env
    source nlsq-env/bin/activate

    # Install NLSQ
    pip install --upgrade "jax[cpu]>=0.4.20" nlsq

**Apple Silicon Macs (M1/M2/M3):**

.. code-block:: bash

    # Create virtual environment
    python -m venv nlsq-env
    source nlsq-env/bin/activate

    # Install with Metal support (experimental)
    pip install --upgrade jax-metal>=0.0.5
    pip install --upgrade "jax[cpu]>=0.4.20" nlsq

Windows
~~~~~~~

Windows users have several installation options.

**Option 1: WSL2 (Recommended)**

Windows Subsystem for Linux 2 provides the best compatibility:

.. code-block:: bash

    # Install WSL2 and Ubuntu
    wsl --install -d Ubuntu

    # Inside WSL2, follow Linux installation instructions
    python -m venv nlsq-env
    source nlsq-env/bin/activate
    pip install --upgrade "jax[cpu]>=0.4.20" nlsq

**Option 2: Native Windows (CPU-only)**

.. code-block:: bash

    # Create virtual environment
    python -m venv nlsq-env
    nlsq-env\Scripts\activate

    # Install NLSQ
    pip install "jax[cpu]>=0.4.20" nlsq

**Option 3: Native Windows with GPU (Advanced)**

For CUDA support on Windows:

.. code-block:: bash

    # Prerequisites:
    # 1. Install CUDA Toolkit 12.x from NVIDIA
    # 2. Install Visual Studio Build Tools
    # 3. Install Anaconda/Miniconda (recommended)

    # Create Conda environment
    conda create -n nlsq python=3.12
    conda activate nlsq

    # Install CUDA toolkit
    conda install -c conda-forge cuda-toolkit=12.1

    # Install JAX with CUDA support
    pip install "jax[cuda12_local]>=0.4.20"

    # Install NLSQ
    pip install nlsq

Development Installation
------------------------

For contributors and advanced users who want to modify NLSQ:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/Dipolar-Quantum-Gases/nlsq.git
    cd nlsq

    # Create development environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install in development mode with all extras
    pip install -e ".[dev,test,docs]"

    # Install pre-commit hooks (recommended)
    pre-commit install

    # Run tests to verify installation
    python -m unittest discover tests -p "test*.py"

Docker Installation
-------------------

For containerized environments:

.. code-block:: dockerfile

    FROM python:3.12-slim

    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

    # Install NLSQ
    RUN pip install --upgrade "jax[cpu]>=0.4.20" nlsq

    # Verify installation
    RUN python -c "import nlsq; print(f'NLSQ {nlsq.__version__} ready')"

**GPU Docker (NVIDIA Container Toolkit required):**

.. code-block:: dockerfile

    FROM nvidia/cuda:12.2-devel-ubuntu22.04

    # Install Python
    RUN apt-get update && apt-get install -y \
        python3.12 \
        python3.12-pip \
        python3.12-venv \
        && rm -rf /var/lib/apt/lists/*

    # Install NLSQ with CUDA support
    RUN pip3.12 install --upgrade "jax[cuda12]>=0.4.20" nlsq

Verification and Testing
------------------------

After installation, verify NLSQ is working correctly:

.. code-block:: python

    import numpy as np
    import jax
    from nlsq import CurveFit, curve_fit_large

    # Check NLSQ version
    import nlsq
    print(f"NLSQ version: {nlsq.__version__}")

    # Check JAX devices
    print(f"JAX devices: {jax.devices()}")

    # Test basic functionality
    def linear(x, m, b):
        return m * x + b

    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + 0.1 * np.random.normal(size=len(x))

    cf = CurveFit()
    popt, pcov = cf.curve_fit(linear, x, y)
    print(f"Fitted parameters: m={popt[0]:.2f}, b={popt[1]:.2f}")

    # Test large dataset function
    popt2, pcov2 = curve_fit_large(linear, x, y)
    print("Large dataset fitting: OK")

    print("Installation verification complete!")

Performance Testing
~~~~~~~~~~~~~~~~~~~

Test GPU acceleration (if available):

.. code-block:: python

    import time
    import numpy as np
    import jax.numpy as jnp
    from nlsq import CurveFit

    # Generate large dataset
    n_points = 1_000_000
    x = np.linspace(0, 10, n_points)
    y = 2.5 * np.exp(-0.5 * x) + np.random.normal(0, 0.1, n_points)

    def exponential(x, a, b):
        return a * jnp.exp(-b * x)

    cf = CurveFit()

    # Time the fit
    start = time.time()
    popt, pcov = cf.curve_fit(exponential, x, y, p0=[2.0, 0.4])
    duration = time.time() - start

    print(f"Fitted {n_points:,} points in {duration:.2f} seconds")
    print(f"Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}")

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Import Error: "No module named 'jax'"**

.. code-block:: bash

    # Install JAX explicitly
    pip install --upgrade "jax>=0.4.20"

**CUDA Not Found Error**

.. code-block:: bash

    # Check CUDA installation
    nvcc --version
    nvidia-smi

    # Reinstall JAX with CUDA support
    pip install --upgrade --force-reinstall "jax[cuda12]>=0.4.20"

**Memory Error with Large Datasets**

.. code-block:: python

    # Use curve_fit_large with memory limit
    from nlsq import curve_fit_large

    popt, pcov = curve_fit_large(
        func, x, y,
        memory_limit_gb=4.0,  # Adjust to your system
        show_progress=True
    )

**Windows Installation Issues**

1. Ensure you have Visual Studio Build Tools installed
2. Use Anaconda/Miniconda for better dependency management
3. Consider using WSL2 for full Linux compatibility

**macOS Permission Issues**

.. code-block:: bash

    # Use --user flag if needed
    pip install --user "jax[cpu]>=0.4.20" nlsq

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/Dipolar-Quantum-Gases/nlsq/issues>`_
2. Review the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
3. Ask questions in `GitHub Discussions <https://github.com/Dipolar-Quantum-Gases/nlsq/discussions>`_

Version Compatibility
----------------------

NLSQ is tested with the following version combinations:

**Python Versions:**

- Python 3.12 (recommended)
- Python 3.13 (supported)

**JAX Versions:**

- JAX 0.4.20 - 0.4.35 (stable)
- JAX 0.5.0 - 0.6.0 (stable)
- JAX 0.7.0 - 0.7.2 (latest)

**Operating Systems:**

- Ubuntu 20.04+ (primary testing)
- CentOS/RHEL 8+ (supported)
- macOS 12+ (supported)
- Windows 10/11 (limited testing)

For the most current compatibility information, see the project's CI configuration on GitHub.