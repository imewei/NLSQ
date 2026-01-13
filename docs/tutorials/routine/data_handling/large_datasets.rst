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

**1. Use float32 for very large data:**

.. code-block:: python

   x = x.astype(np.float32)
   y = y.astype(np.float32)
   # Halves memory usage

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

Typical performance on modern hardware (8-core CPU, 32GB RAM):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Points
     - Strategy
     - First Fit
     - Subsequent
     - Memory
   * - 10K
     - STANDARD
     - ~1s
     - ~0.1s
     - ~100MB
   * - 100K
     - STANDARD
     - ~2s
     - ~0.3s
     - ~1GB
   * - 1M
     - CHUNKED
     - ~10s
     - ~2s
     - ~2GB
   * - 10M
     - STREAMING
     - ~60s
     - ~20s
     - ~4GB
   * - 100M
     - STREAMING
     - ~10min
     - ~3min
     - ~4GB

First fit includes JIT compilation; subsequent fits are faster.

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
- Use float32 data
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
