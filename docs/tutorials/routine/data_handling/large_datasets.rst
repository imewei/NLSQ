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
   # Memory-efficient chunked processing

The default path fits each chunk and combines the results; for a nonlinear model
that is not the full-batch answer. Pass ``method="exact_streaming"`` for an exact
full-data fit that accumulates ``J^T J`` and ``J^T r`` over chunks (any size,
memmap/HDF5/Zarr sources, ``sigma``, ``bounds``). It was the fastest NLSQ path at
100M points in the benchmarks below; see :ref:`exact-streaming`.

.. code-block:: python

   popt, pcov = curve_fit_large(model, x, y, p0=p0, method="exact_streaming")

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

Measured on an i9-13900H (20 logical cores, 62 GB RAM) and an RTX 4090 Laptop GPU (16 GB),
3-parameter ``exp_decay`` model, warm median (2026-11-01, JAX 0.12.0.1). Timings on a laptop
vary by up to about 50% between sessions.

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * - Points
     - ``curve_fit_large`` default, CPU
     - ``exact_streaming``, CPU
     - ``exact_streaming``, GPU
   * - 1K
     - 3.8 ms
     - 60 ms
     - 92 ms
   * - 1M
     - 0.58 s
     - 0.19 s
     - 0.45 s
   * - 10M
     - 3.1 s
     - 1.3 s
     - 0.26 s
   * - 100M
     - 29 s
     - 8.7 s
     - 0.60 s

Plain ``curve_fit`` runs out of memory on the 16 GB GPU at 100M points. On the GPU at 1M
points and below, plain ``curve_fit`` is faster than ``exact_streaming`` for the unbounded
problems (``exp_decay`` at 1M: 53 ms vs 451 ms; not for ``bounded_exp``), and
``exact_streaming`` has a fixed cost of about 60-140 ms at 1K. The first fit also pays JIT
compilation. The full comparison against SciPy, lmfit, JAXopt and Optimistix is in
``benchmarks/baselines/2026-11-01-library-comparison-1k-1M-10M-100M.md``.

GPU Acceleration
----------------

GPUs provide significant speedup for large datasets:

.. code-block:: python

   # Install JAX with CUDA
   # pip install jax[cuda12_pip]

   from nlsq import fit

   # Automatically uses GPU if available
   popt, pcov = fit(model, x, y, p0=[...])

GPU speedup is most significant for 100K+ points.

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
