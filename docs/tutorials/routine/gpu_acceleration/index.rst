GPU Acceleration
================

NLSQ uses JAX for GPU acceleration, which makes large fits faster than on the CPU with no code changes
required. Small fits are slower on the GPU, so the benefit depends on the dataset size.

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

NLSQ's GPU run compared with NLSQ's CPU run (same code, RTX 4090 Laptop GPU and i9-13900H, 3-parameter
exponential decay, warm median, ``curve_fit`` up to 100K points and ``curve_fit_large`` above):

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Points
     - API
     - NLSQ CPU
     - NLSQ GPU
     - CPU time / GPU time
   * - 1,000
     - ``curve_fit``
     - 4.0 ms
     - 24.8 ms
     - 0.16x
   * - 10,000
     - ``curve_fit``
     - 11.8 ms
     - 30.9 ms
     - 0.38x
   * - 100,000
     - ``curve_fit``
     - 44.4 ms
     - 48.0 ms
     - 0.93x
   * - 1,000,000
     - ``curve_fit_large``
     - 90.8 ms
     - 18.5 ms
     - 4.9x
   * - 10,000,000
     - ``curve_fit_large``
     - 977.2 ms
     - 78.1 ms
     - 12.5x
   * - 100,000,000
     - ``curve_fit_large``
     - 6.08 s
     - 618.1 ms
     - 9.8x

CPU time / GPU time above 1 means the GPU was faster. On this model the GPU was slower up to 100K points and faster
from 1M points; at 100K points across four models it ranged from 1.2x slower to 3.2x faster. This compares NLSQ with itself; comparisons with other libraries, made within one device, and the
other three models are in :doc:`/api/performance_benchmarks`.

First fit includes JIT compilation overhead. Subsequent fits are faster.
