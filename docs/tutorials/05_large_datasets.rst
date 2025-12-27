Tutorial 5: Large Datasets
===========================

In this tutorial, you'll learn how to efficiently fit models to large datasets
with hundreds of thousands to millions of data points.

What You'll Learn
-----------------

- When standard fitting isn't enough
- Using ``curve_fit_large`` for automatic handling
- Streaming optimization for unlimited-size data
- Memory management strategies

Prerequisites
-------------

- Completed tutorials 1-4
- Understanding of basic fitting

When You Need Large Dataset Support
-----------------------------------

Consider large dataset strategies when:

- Dataset has **> 100,000 points**
- Fitting causes **memory errors**
- Fitting is **too slow** (> 60 seconds)
- You need to fit datasets **larger than RAM**

Standard Fitting vs Large Dataset Fitting
-----------------------------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit, curve_fit_large
   import time


   def exponential(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Generate a large dataset
   np.random.seed(42)
   n_points = 1_000_000  # 1 million points
   x = np.linspace(0, 10, n_points)
   y = 2.0 * np.exp(-0.5 * x) + 0.3 + 0.05 * np.random.normal(size=n_points)

   # Standard curve_fit - may be slow or fail for very large data
   print("Standard curve_fit:")
   start = time.time()
   popt, pcov = curve_fit(exponential, x, y, p0=[2.0, 0.5, 0.3])
   print(f"  Time: {time.time() - start:.2f}s")
   print(f"  Parameters: {popt}")

   # curve_fit_large - automatic handling
   print("\ncurve_fit_large:")
   start = time.time()
   popt, pcov = curve_fit_large(exponential, x, y, p0=[2.0, 0.5, 0.3], show_progress=True)
   print(f"  Time: {time.time() - start:.2f}s")
   print(f"  Parameters: {popt}")

Using curve_fit_large
---------------------

The ``curve_fit_large`` function automatically detects dataset size and
applies appropriate strategies:

.. code-block:: python

   from nlsq import curve_fit_large

   popt, pcov = curve_fit_large(
       model,
       x_data,
       y_data,
       p0=initial_guess,
       show_progress=True,  # Show progress bar
       memory_limit_gb=4.0,  # Memory limit (default: auto-detect)
   )

Automatic Strategies
~~~~~~~~~~~~~~~~~~~~

Based on dataset size, ``curve_fit_large`` uses:

- **< 100K points**: Standard optimization
- **100K - 1M points**: Optimized batch processing
- **> 1M points**: Streaming optimization with checkpoints

Progress Monitoring
-------------------

For long-running fits, enable progress monitoring:

.. code-block:: python

   from nlsq import curve_fit_large

   popt, pcov = curve_fit_large(
       model,
       x,
       y,
       p0=p0,
       show_progress=True,  # Shows progress bar
   )

You'll see output like:

.. code-block:: text

   Processing: 100%|██████████| 1000000/1000000 [00:15<00:00, 65432.10 pts/s]
   Iterations: 100%|██████████| 25/25 [00:15<00:00, 1.64 it/s]

Streaming Optimization
----------------------

For datasets too large to fit in memory, use streaming optimization:

.. code-block:: python

   from nlsq import StreamingOptimizer

   # Create optimizer
   optimizer = StreamingOptimizer(
       model,
       n_params=3,
       chunk_size=100_000,  # Process 100K points at a time
   )

   # Fit streaming data
   result = optimizer.fit(x_data, y_data, p0=[2.0, 0.5, 0.3], show_progress=True)

   print(f"Parameters: {result.popt}")
   print(f"Covariance: {result.pcov}")

Adaptive Hybrid Streaming
-------------------------

For the best performance on challenging problems, use the adaptive hybrid
streaming optimizer:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer

   optimizer = AdaptiveHybridStreamingOptimizer(
       model,
       n_params=3,
       enable_parameter_normalization=True,  # For multi-scale parameters
   )

   result = optimizer.fit(x_data, y_data, p0=p0)
   print(f"Parameters: {result.popt}")

This optimizer uses a 4-phase approach:

1. **Adam warmup**: Fast initial convergence
2. **Gauss-Newton refinement**: Precise parameter estimation
3. **Multi-start verification**: Avoid local minima
4. **Covariance calculation**: Accurate uncertainties

Memory Management
-----------------

Control memory usage with memory limits:

.. code-block:: python

   from nlsq import curve_fit_large

   # Limit to 2GB of RAM
   popt, pcov = curve_fit_large(model, x, y, p0=p0, memory_limit_gb=2.0)

   # For systems with limited memory
   popt, pcov = curve_fit_large(model, x, y, p0=p0, memory_limit_gb=0.5)  # Only 500MB

Checkpointing for Very Long Fits
--------------------------------

For fits that may take hours, enable checkpointing:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer

   optimizer = AdaptiveHybridStreamingOptimizer(
       model,
       n_params=3,
       checkpoint_dir="./fit_checkpoints",
       checkpoint_interval=60,  # Save every 60 seconds
   )

   # If interrupted, restart from checkpoint
   result = optimizer.fit(x_data, y_data, p0=p0, resume=True)

Using the fit() Function with Presets
-------------------------------------

The unified ``fit()`` function has a ``large`` preset:

.. code-block:: python

   from nlsq import fit

   # Automatic large dataset handling
   popt, pcov = fit(
       model, x, y, p0=p0, preset="large"  # Auto-detect and use best strategy
   )

Complete Example: Fitting 10 Million Points
-------------------------------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit_large
   import time


   def model(x, a, b, c, d):
       """Damped exponential with offset."""
       return a * jnp.exp(-b * x) * jnp.cos(c * x) + d


   # Generate 10 million points
   print("Generating 10M data points...")
   np.random.seed(42)
   n = 10_000_000
   x = np.linspace(0, 100, n)
   y_true = 2.0 * np.exp(-0.05 * x) * np.cos(0.5 * x) + 1.0
   y = y_true + 0.1 * np.random.normal(size=n)

   print(f"Data size: {x.nbytes / 1e9:.2f} GB")

   # Fit with progress monitoring
   print("\nFitting...")
   start = time.time()

   popt, pcov = curve_fit_large(
       model, x, y, p0=[2.0, 0.05, 0.5, 1.0], show_progress=True, memory_limit_gb=4.0
   )

   elapsed = time.time() - start
   print(f"\nCompleted in {elapsed:.1f}s")
   print(f"Rate: {n/elapsed/1e6:.2f} million points/second")

   # Results
   perr = np.sqrt(np.diag(pcov))
   print("\nResults:")
   print(f"  a = {popt[0]:.4f} ± {perr[0]:.4f} (true: 2.0)")
   print(f"  b = {popt[1]:.4f} ± {perr[1]:.4f} (true: 0.05)")
   print(f"  c = {popt[2]:.4f} ± {perr[2]:.4f} (true: 0.5)")
   print(f"  d = {popt[3]:.4f} ± {perr[3]:.4f} (true: 1.0)")

Performance Tips
----------------

1. **Use GPU acceleration** (see :doc:`06_gpu_acceleration`)

   GPU can provide 100-300x speedup for large datasets.

2. **Choose appropriate chunk size**

   .. code-block:: python

      # Balance between memory and speed
      optimizer = StreamingOptimizer(model, n_params=3, chunk_size=500_000)

3. **Pre-allocate arrays**

   .. code-block:: python

      # Avoid creating new arrays in loops
      x = np.empty(n_points, dtype=np.float64)
      y = np.empty(n_points, dtype=np.float64)

4. **Use float32 for very large data**

   .. code-block:: python

      x = x.astype(np.float32)  # Half the memory
      y = y.astype(np.float32)

Key Takeaways
-------------

1. Use ``curve_fit_large`` for datasets > 100K points
2. Enable ``show_progress=True`` for long fits
3. Set ``memory_limit_gb`` for memory-constrained systems
4. Use streaming optimization for data larger than RAM
5. Enable checkpointing for very long fits
6. GPU acceleration provides massive speedups

Next Steps
----------

In :doc:`06_gpu_acceleration`, you'll learn how to:

- Set up GPU/TPU acceleration
- Understand when GPU helps
- Optimize GPU performance

.. seealso::

   - :doc:`/howto/handle_large_data` - Detailed large data guide
   - :doc:`/explanation/streaming` - How streaming works
