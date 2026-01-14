Installation
============

This guide covers installing NLSQ and verifying your setup.

Basic Installation
------------------

Install NLSQ using pip:

.. code-block:: bash

   pip install nlsq

Or using uv (recommended for faster installation):

.. code-block:: bash

   uv pip install nlsq

Verify Installation
-------------------

.. code-block:: python

   import nlsq

   print(f"NLSQ version: {nlsq.__version__}")
   print(f"Device: {nlsq.get_device()}")

Expected output:

.. code-block:: text

   NLSQ version: 0.6.4
   Device: cpu  # or 'cuda:0' if GPU is available

GPU Support (Optional)
----------------------

NLSQ uses JAX for GPU acceleration. To enable GPU support:

**NVIDIA CUDA:**

.. code-block:: bash

   # Install JAX with CUDA support
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

**AMD ROCm:**

.. code-block:: bash

   pip install --upgrade "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

**Apple Silicon (M1/M2/M3):**

JAX runs on Apple Silicon via the Metal backend (experimental):

.. code-block:: bash

   pip install jax-metal

Verify GPU Setup
----------------

.. code-block:: python

   import jax

   print(f"JAX devices: {jax.devices()}")
   print(f"Default backend: {jax.default_backend()}")

Expected output with GPU:

.. code-block:: text

   JAX devices: [CudaDevice(id=0)]
   Default backend: gpu

Desktop GUI (Optional)
----------------------

To use the interactive desktop application:

.. code-block:: bash

   pip install "nlsq[gui_qt]"

Launch with:

.. code-block:: bash

   nlsq-gui

Development Installation
------------------------

For development with all optional dependencies:

.. code-block:: bash

   git clone https://github.com/imewei/NLSQ.git
   cd NLSQ
   pip install -e ".[dev,test,docs,gui_qt]"

Common Installation Issues
--------------------------

**JAX installation fails:**
   Try installing JAX separately before NLSQ:

   .. code-block:: bash

      pip install jax jaxlib
      pip install nlsq

**GPU not detected:**
   Ensure CUDA drivers are installed and compatible with your JAX version.
   Check with ``nvidia-smi`` for NVIDIA GPUs.

**Import errors:**
   Ensure Python 3.12+ is being used:

   .. code-block:: bash

      python --version  # Should be 3.12+

Next Steps
----------

Continue to :doc:`first_fit` to fit your first curve.
