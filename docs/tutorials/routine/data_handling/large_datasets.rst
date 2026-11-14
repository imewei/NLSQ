Large Datasets
==============

NLSQ automatically handles datasets from hundreds to 100M+ points using
memory-aware strategies. You don't need to change your code.

Automatic Handling
------------------

NLSQ automatically selects the best strategy:

.. code-block:: python

   from nlsq import fit

   # Same code works for any size
   popt, pcov = fit(model, x, y, p0=[...])

   # 100 points: STANDARD (in-memory)
   # 1M points: CHUNKED (Jacobian chunking)
   # 100M points: STREAMING (batch processing)

Memory Strategy Selection
-------------------------

The system analyzes data size vs available memory:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Strategy
     - When Used
     - Characteristics
   * - **STANDARD**
     - Data + Jacobian fit in memory
     - Fastest, full in-memory processing
   * - **CHUNKED**
     - Data fits, Jacobian doesn't
     - Jacobian computed in chunks
   * - **STREAMING**
     - Data exceeds memory
     - Adaptive batch processing

Monitoring Memory Usage
-----------------------

Check what strategy NLSQ selects:

.. code-block:: python

   from nlsq import fit
   from nlsq.core.workflow import MemoryBudgetSelector

   # Check memory strategy before fitting
   selector = MemoryBudgetSelector()
   strategy, config = selector.select(
       n_points=len(x),
       n_params=3,  # Number of model parameters
   )
   print(f"Strategy: {strategy}")

   # Fit
   popt, pcov = fit(model, x, y, p0=[...])

Memory Override
---------------

Force a specific memory limit:

.. code-block:: python

   # Force chunked/streaming by limiting memory
   popt, pcov = fit(model, x, y, p0=[...], memory_limit_gb=4.0)

   # Useful when:
   # - Running alongside other processes
   # - On shared systems
   # - Testing streaming behavior

Large Dataset Tips
------------------

**1. Use the streaming optimizer for very large data:**

.. code-block:: python

   from nlsq import curve_fit_large

   popt, pcov = curve_fit_large(model, x, y, p0=p0)
   # At or above size_threshold (default 1M points): exact full-data streaming

At or above ``size_threshold`` the default path is an exact full-data fit that
accumulates ``J^T J`` and ``J^T r`` over chunks, so the result does not depend on the
chunking. It needs a pointwise model written with ``jax.numpy`` and 1-D ``xdata``;
other inputs fall back to, or raise an error naming, the chunked fit (see
:ref:`exact-streaming`). The older chunked fit, which fits each chunk and combines
the results and is not the full-batch answer for a nonlinear model, is available with
``strategy=Chunked()``. ``method="exact_streaming"`` selects the exact fit explicitly at
any size and also accepts memmap/HDF5/Zarr sources.

.. code-block:: python

   from nlsq.streaming import Chunked

   popt, pcov = curve_fit_large(model, x, y, p0=p0, strategy=Chunked())

**2. Start with a subset:**

.. code-block:: python

   # Quick test on subset
   n_sample = 10000
   idx = np.random.choice(len(x), n_sample, replace=False)
   popt_test, _ = fit(model, x[idx], y[idx], p0=[...])

   # Full fit with good initial guess
   popt, pcov = fit(model, x, y, p0=popt_test)

**3. Use bounds:**

.. code-block:: python

   # Bounds help convergence with large data
   popt, pcov = fit(model, x, y, p0=[...], bounds=([0, 0], [100, 10]))

**4. Adjust tolerances:**

.. code-block:: python

   # Looser tolerances for faster convergence
   popt, pcov = fit(model, x, y, p0=[...], ftol=1e-6, xtol=1e-6, gtol=1e-6)

Performance Benchmarks
----------------------

``curve_fit_large`` runs exact full-data streaming at or above ``size_threshold`` (1M points). At 1M points it
was 2.2x to 11x faster than in-memory ``curve_fit`` on both devices (warm medians, i9-13900H and RTX 4090 Laptop
GPU, run of 2026-11-13):

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Model
     - ``curve_fit`` CPU
     - ``curve_fit_large`` CPU
     - ``curve_fit`` GPU
     - ``curve_fit_large`` GPU
   * - Exponential decay
     - 362.3 ms
     - 90.8 ms
     - 140.5 ms
     - 18.5 ms
   * - Gaussian
     - 562.0 ms
     - 252.5 ms
     - 165.8 ms
     - 43.5 ms
   * - Two exponentials
     - 930.4 ms
     - 328.2 ms
     - 115.5 ms
     - 48.4 ms
   * - Bounded exponential
     - 1.28 s
     - 293.3 ms
     - 538.6 ms
     - 49.0 ms

``curve_fit`` was not run at 10M points and above; ``curve_fit_large`` fitted 100M points on both devices (for
example 6.08 s on the CPU and 618 ms on the GPU for the exponential decay). The comparison with SciPy, lmfit, JAXopt
and Optimistix, made within one device, the methodology and the limitations are in
:doc:`/api/performance_benchmarks`. Timings on a laptop vary between sessions.

GPU Acceleration
----------------

GPUs run large fits faster than the CPU:

.. code-block:: python

   # Install JAX with CUDA
   # pip install jax[cuda12_pip]

   from nlsq import fit

   # Automatically uses GPU if available
   popt, pcov = fit(model, x, y, p0=[...])

NLSQ on the GPU was faster than NLSQ on the CPU from about 100K points and slower below that
(:doc:`/api/performance_benchmarks`).

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit

   # Generate large dataset
   np.random.seed(42)
   n_points = 1_000_000  # 1 million points

   x = np.linspace(0, 100, n_points)
   y_true = 2.5 * np.exp(-0.05 * x) + 0.3
   y = y_true + 0.2 * np.random.randn(n_points)


   # Model
   def exponential(x, A, k, c):
       return A * jnp.exp(-k * x) + c


   # Fit - NLSQ auto-selects appropriate strategy
   print(f"Fitting {n_points:,} points...")
   popt, pcov = fit(exponential, x, y, p0=[2, 0.05, 0])

   # Results
   A, k, c = popt
   perr = np.sqrt(np.diag(pcov))

   print(f"\nResults:")
   print(f"  A = {A:.4f} +/- {perr[0]:.4f}")
   print(f"  k = {k:.5f} +/- {perr[1]:.5f}")
   print(f"  c = {c:.4f} +/- {perr[2]:.4f}")

Troubleshooting Large Data
--------------------------

**Out of memory:**

- Reduce ``memory_limit_gb``
- Use streaming optimizer
- Close other applications

**Very slow:**

- Use GPU acceleration
- Loosen tolerances
- Check model complexity

**Poor convergence:**

- Improve initial guess (test on subset first)
- Add bounds
- Check data quality

Next Steps
----------

- :doc:`../gpu_acceleration/index` - GPU acceleration
- :doc:`../troubleshooting/common_issues` - Debugging tips
