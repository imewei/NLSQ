GPU Setup
=========

This guide covers installing and configuring JAX for GPU acceleration.

.. important::

   **GPU acceleration is supported on Linux only.** macOS and Windows
   automatically use the CPU backend — NLSQ enforces this at import time
   by setting ``JAX_PLATFORM_NAME=cpu``.

NVIDIA GPU (CUDA) — Linux
--------------------------

**Requirements:**

- **Linux** operating system
- NVIDIA GPU with CUDA support (Maxwell or newer, SM ≥ 5.2)
- CUDA 12.x or 13.x drivers installed
- cuDNN (bundled with JAX)

**Installation:**

.. code-block:: bash

   # Recommended: use the Makefile target (auto-detects CUDA version)
   make install-jax-gpu

   # Or manually for CUDA 12
   pip install --upgrade "jax[cuda12-local]"

   # Or for CUDA 13
   pip install --upgrade "jax[cuda13-local]"

**Verify installation:**

.. code-block:: python

   import jax

   print(f"JAX version: {jax.__version__}")
   print(f"Devices: {jax.devices()}")
   print(f"Default backend: {jax.default_backend()}")

Expected output:

.. code-block:: text

   JAX version: 0.9.0
   Devices: [CudaDevice(id=0)]
   Default backend: gpu

AMD GPU (ROCm) — Linux
-----------------------

**Requirements:**

- **Linux** operating system
- AMD GPU with ROCm support
- ROCm 5.x+ installed

**Installation:**

.. code-block:: bash

   pip install --upgrade "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

macOS and Windows (CPU Only)
-----------------------------

NLSQ enforces CPU-only mode on macOS and Windows at import time. No
additional configuration is needed — the following environment variables
are set automatically:

- ``NLSQ_FORCE_CPU=1``
- ``JAX_PLATFORM_NAME=cpu``
- ``JAX_PLATFORMS=cpu``

On macOS, additional guards prevent SIGBUS crashes from Metal/OpenGL/XLA
conflicts (``XLA_FLAGS``, ``OMP_NUM_THREADS``, ``MPLBACKEND``, etc.).

.. code-block:: bash

   # Just install JAX (CPU backend is automatic)
   pip install jax jaxlib

Docker Setup
------------

For containerized environments:

.. code-block:: dockerfile

   FROM nvidia/cuda:12.1-runtime-ubuntu22.04

   RUN pip install jax[cuda12_pip] nlsq

Run with GPU access:

.. code-block:: bash

   docker run --gpus all my-nlsq-container

Troubleshooting Installation
----------------------------

**CUDA not found:**

.. code-block:: bash

   # Check CUDA installation
   nvidia-smi
   nvcc --version

   # CUDA path may need to be set
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

**JAX falls back to CPU:**

.. code-block:: python

   import jax

   if jax.default_backend() == "cpu":
       print("GPU not detected!")
       # Check CUDA drivers
       # Reinstall JAX with CUDA support

**Out of memory errors:**

.. code-block:: python

   # Limit GPU memory
   import os

   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

**Multiple GPUs not detected:**

.. code-block:: bash

   # Check all GPUs visible
   nvidia-smi

   # Set visible devices
   export CUDA_VISIBLE_DEVICES=0,1

Verifying NLSQ GPU Usage
------------------------

.. code-block:: python

   from nlsq import get_device

   device = get_device()
   print(f"NLSQ device: {device}")

   # Check if GPU is available
   import jax

   if jax.default_backend() == "gpu":
       print("GPU acceleration enabled!")
   else:
       print("Running on CPU")

Next Steps
----------

- :doc:`gpu_usage` - Using GPU in your fits
- :doc:`multi_gpu` - Multiple GPU configuration
