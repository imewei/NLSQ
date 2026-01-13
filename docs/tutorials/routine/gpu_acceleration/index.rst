GPU Acceleration
================

NLSQ uses JAX for GPU acceleration, providing significant speedups for
large datasets with no code changes required.

.. toctree::
   :maxdepth: 1

   gpu_setup
   gpu_usage
   multi_gpu

Chapter Overview
----------------

**GPU Setup** (10 min)
   Install JAX with GPU support and verify configuration.

**GPU Usage** (5 min)
   Automatic GPU detection and usage patterns.

**Multi-GPU** (5 min)
   Using multiple GPUs for parallel processing.

Quick Start
-----------

.. code-block:: bash

   # Install JAX with CUDA support
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

.. code-block:: python

   from nlsq import fit

   # GPU is used automatically if available
   popt, pcov = fit(model, x, y, p0=[...])

Performance Benefits
--------------------

GPU acceleration provides significant speedups:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Dataset Size
     - CPU Time
     - GPU Time
     - Speedup
   * - 10K points
     - ~1.0s
     - ~0.5s
     - 2x
   * - 100K points
     - ~5.0s
     - ~0.8s
     - 6x
   * - 1M points
     - ~30s
     - ~2s
     - 15x
   * - 10M points
     - ~5min
     - ~15s
     - 20x

First fit includes JIT compilation overhead. Subsequent fits are faster.
