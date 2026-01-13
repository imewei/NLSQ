GPU Setup
=========

This guide covers installing and configuring JAX for GPU acceleration.

NVIDIA GPU (CUDA)
-----------------

**Requirements:**

- NVIDIA GPU with CUDA support
- CUDA 12.x drivers installed
- cuDNN (bundled with JAX)

**Installation:**

.. code-block:: bash

   # Install JAX with CUDA 12 support
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Or for CUDA 11
   pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**Verify installation:**

.. code-block:: python

   import jax

   print(f"JAX version: {jax.__version__}")
   print(f"Devices: {jax.devices()}")
   print(f"Default backend: {jax.default_backend()}")

Expected output:

.. code-block:: text

   JAX version: 0.8.0
   Devices: [CudaDevice(id=0)]
   Default backend: gpu

AMD GPU (ROCm)
--------------

**Requirements:**

- AMD GPU with ROCm support
- ROCm 5.x+ installed

**Installation:**

.. code-block:: bash

   pip install --upgrade "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

Apple Silicon (M1/M2/M3)
------------------------

**Requirements:**

- Apple Silicon Mac (M1, M2, M3)
- macOS 12.0+

**Installation (experimental):**

.. code-block:: bash

   pip install jax-metal

**Verify:**

.. code-block:: python

   import jax

   print(jax.devices())  # Should show Metal device

CPU-Only Setup
--------------

If no GPU is available, JAX works on CPU:

.. code-block:: bash

   pip install jax jaxlib

NLSQ will use CPU automatically.

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
