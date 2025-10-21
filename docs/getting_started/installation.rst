Installation Guide
==================

This guide provides comprehensive installation instructions for NLSQ across different platforms and use cases.

Quick Start
-----------

**Important**: GPU acceleration is only supported on Linux. Windows and macOS are CPU-only.

For most users, the simplest installation method is:

**Linux**::

    # For CPU-only (basic features)
    pip install nlsq "jax[cpu]>=0.6.0"

    # For GPU with system CUDA 12 (best performance, requires CUDA installed)
    pip install nlsq "jax[cuda12-local]>=0.6.0"

    # For GPU with bundled CUDA 12 (larger download, no system CUDA needed)
    pip install nlsq "jax[cuda12]>=0.6.0"

    # With all advanced features (recommended)
    pip install nlsq[all] "jax[cpu]>=0.6.0"

**macOS/Windows**::

    # CPU-only (GPU not supported)
    pip install nlsq "jax[cpu]>=0.6.0"

    # With all advanced features (recommended)
    pip install nlsq[all] "jax[cpu]>=0.6.0"

Installation Options
--------------------

NLSQ offers different installation options depending on your needs:

**Basic Installation (nlsq):**
- Core curve fitting functionality
- GPU/TPU acceleration via JAX
- SciPy compatibility
- Basic error handling

**Full Installation (nlsq[all]):**
- All basic features
- Advanced memory management with automatic monitoring
- Intelligent algorithm selection
- Real-time diagnostics and convergence monitoring
- Smart caching system for performance optimization
- Robust error recovery and fallback strategies
- Comprehensive input validation
- Large dataset support with progress reporting
- Sparse Jacobian optimization
- Streaming optimizer for unlimited datasets

**Development Installation (nlsq[dev]):**
- All features from nlsq[all]
- Development tools (pytest, mypy, ruff)
- Pre-commit hooks
- Documentation building tools
- Security analysis tools (bandit)

Choose the appropriate installation based on your use case:

.. code-block:: bash

    # Minimal installation for basic curve fitting
    pip install nlsq

    # Recommended for most users
    pip install nlsq[all]

    # For developers and contributors
    pip install nlsq[dev,test,docs]

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

Core dependencies:

- JAX 0.6.0 - 0.7.2 (JIT compilation and automatic differentiation)
- NumPy 1.26.0+ (numerical arrays)
- SciPy 1.11.0+ (optimization algorithms)

Advanced feature dependencies:

- psutil 5.9.0+ (memory monitoring and management)
- tqdm 4.65.0+ (progress bars for large dataset processing)

Optional dependencies for development:

- pytest (testing)
- mypy (type checking)
- ruff (code formatting and linting)
- bandit (security analysis)

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
    pip install nlsq "jax[cpu]>=0.6.0"

    # Verify installation
    python -c "import nlsq; print(f'NLSQ {nlsq.__version__} installed successfully')"

**GPU installation (CUDA 12) - Linux Only:**

.. code-block:: bash

    # Ensure NVIDIA drivers and CUDA 12 are installed
    nvidia-smi
    nvcc --version  # Should show CUDA 12.x

    # Create virtual environment
    python -m venv nlsq-env
    source nlsq-env/bin/activate

    # Option 1: System CUDA 12 (best performance)
    pip install nlsq "jax[cuda12-local]>=0.6.0"

    # Option 2: Bundled CUDA 12 (no system CUDA needed)
    pip install nlsq "jax[cuda12]>=0.6.0"

    # Verify GPU access
    python -c "import jax; print(f'JAX devices: {jax.devices()}')"

macOS
~~~~~

**Important**: macOS does not support GPU acceleration with CUDA. Only CPU mode is available.

**Intel Macs:**

.. code-block:: bash

    # Use Homebrew Python (recommended)
    brew install python@3.12

    # Create virtual environment
    python3.12 -m venv nlsq-env
    source nlsq-env/bin/activate

    # Install NLSQ (CPU-only)
    pip install nlsq "jax[cpu]>=0.6.0"

**Apple Silicon Macs (M1/M2/M3):**

.. code-block:: bash

    # Create virtual environment
    python -m venv nlsq-env
    source nlsq-env/bin/activate

    # Install with Metal support (experimental, CPU-only)
    pip install --upgrade jax-metal>=0.0.5
    pip install nlsq "jax[cpu]>=0.6.0"

Windows
~~~~~~~

**Important**: Windows does not support GPU acceleration with CUDA natively. Use WSL2 for GPU support.

Windows users have two installation options:

**Option 1: WSL2 (Recommended for GPU support)**

Windows Subsystem for Linux 2 provides full Linux compatibility including GPU support:

.. code-block:: bash

    # Install WSL2 and Ubuntu
    wsl --install -d Ubuntu

    # Inside WSL2, follow Linux installation instructions above
    python -m venv nlsq-env
    source nlsq-env/bin/activate

    # For CPU-only
    pip install nlsq "jax[cpu]>=0.6.0"

    # For GPU (requires CUDA 12 installed in WSL2)
    pip install nlsq "jax[cuda12-local]>=0.6.0"

**Option 2: Native Windows (CPU-only)**

.. code-block:: bash

    # Create virtual environment
    python -m venv nlsq-env
    nlsq-env\Scripts\activate

    # Install NLSQ (CPU-only)
    pip install nlsq "jax[cpu]>=0.6.0"

Development Installation
------------------------

For contributors and advanced users who want to modify NLSQ:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/imewei/NLSQ.git
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
    RUN pip install --upgrade "jax[cpu]>=0.6.0" nlsq

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
    RUN pip3.12 install --upgrade "jax[cuda12]>=0.6.0" nlsq

Verification and Testing
------------------------

After installation, verify NLSQ is working correctly:

Basic Verification
~~~~~~~~~~~~~~~~~~

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

    print("Basic installation verification complete!")

Advanced Features Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you installed with ``nlsq[all]``, test the advanced features:

.. code-block:: python

    from nlsq import (
        MemoryConfig,
        memory_context,
        AlgorithmSelector,
        SmartCache,
        DiagnosticMonitor,
        InputValidator,
    )

    print("Testing advanced features...")

    # Test memory management
    config = MemoryConfig(memory_limit_gb=4.0)
    print(f"✓ Memory management available")

    # Test algorithm selection
    selector = AlgorithmSelector()
    print(f"✓ Algorithm selection available")

    # Test caching
    cache = SmartCache()
    print(f"✓ Smart caching available")

    # Test diagnostics
    monitor = DiagnosticMonitor()
    print(f"✓ Diagnostic monitoring available")

    # Test input validation
    validator = InputValidator()
    print(f"✓ Input validation available")

    # Test advanced curve fitting
    with memory_context(config):
        cf_advanced = CurveFit(
            algorithm_selector=selector, cache=cache, diagnostic_monitor=monitor
        )

        result = cf_advanced.curve_fit(linear, x, y)
        print(f"✓ Advanced curve fitting successful")

    print("All advanced features verified successfully!")

Memory and Performance Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test memory management with different dataset sizes:

.. code-block:: python

    from nlsq import curve_fit_large, get_memory_stats, estimate_memory_requirements

    # Test memory estimation
    n_points = 10000
    n_params = 3
    stats = estimate_memory_requirements(n_points, n_params)
    print(
        f"Memory estimate for {n_points:,} points: {stats.total_memory_estimate_gb:.2f} GB"
    )

    # Test automatic dataset size handling
    sizes = [1000, 100000, 1000000]

    for size in sizes:
        x_test = np.linspace(0, 10, size)
        y_test = 2 * x_test + 1 + 0.1 * np.random.normal(size=size)

        popt, pcov = curve_fit_large(linear, x_test, y_test, show_progress=True)
        print(f"✓ Processed {size:,} points successfully")

    print("Memory and performance verification complete!")

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
    pip install --upgrade "jax>=0.6.0"

**CUDA Not Found Error**

.. code-block:: bash

    # Check CUDA installation
    nvcc --version
    nvidia-smi

    # Reinstall JAX with CUDA support
    pip install --upgrade --force-reinstall "jax[cuda12]>=0.6.0"

**Memory Error with Large Datasets**

.. code-block:: python

    # Use curve_fit_large with memory limit
    from nlsq import curve_fit_large

    popt, pcov = curve_fit_large(
        func, x, y, memory_limit_gb=4.0, show_progress=True  # Adjust to your system
    )

**Windows Installation Issues**

1. Ensure you have Visual Studio Build Tools installed
2. Use Anaconda/Miniconda for better dependency management
3. Consider using WSL2 for full Linux compatibility

**macOS Permission Issues**

.. code-block:: bash

    # Use --user flag if needed
    pip install --user "jax[cpu]>=0.6.0" nlsq

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/imewei/NLSQ/issues>`_ for known issues and solutions
2. Review the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_ for JAX-specific setup
3. Search existing issues or create a new one for help

Version Compatibility
----------------------

NLSQ is tested with the following version combinations:

**Python Versions:**

- Python 3.12 (recommended)
- Python 3.13 (supported)

**JAX Versions:**

- JAX 0.6.0 - 0.4.35 (stable)
- JAX 0.5.0 - 0.6.0 (stable)
- JAX 0.7.0 - 0.7.2 (latest)

**Operating Systems:**

- Ubuntu 20.04+ (primary testing)
- CentOS/RHEL 8+ (supported)
- macOS 12+ (supported)
- Windows 10/11 (limited testing)

For the most current compatibility information, see the project's CI configuration on GitHub.
