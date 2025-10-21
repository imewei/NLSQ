Advanced Features Guide
=======================

This guide covers NLSQ’s advanced features for sophisticated curve
fitting applications. These features enable robust fitting, progress
monitoring, large dataset handling, and performance optimization.

Table of Contents
-----------------

1. `Callbacks and Progress
   Monitoring <#callbacks-and-progress-monitoring>`__
2. `Robust Fitting with Loss
   Functions <#robust-fitting-with-loss-functions>`__
3. `Large Dataset Handling <#large-dataset-handling>`__
4. `Algorithm Selection <#algorithm-selection>`__
5. `Memory Management <#memory-management>`__
6. `Diagnostic Monitoring <#diagnostic-monitoring>`__
7. `Sparse Jacobian Optimization <#sparse-jacobian-optimization>`__
8. `Streaming Optimization <#streaming-optimization>`__

--------------

Callbacks and Progress Monitoring
---------------------------------

Callbacks allow you to monitor optimization progress, log intermediate
results, and implement custom stopping criteria.

Built-in Callbacks
~~~~~~~~~~~~~~~~~~

NLSQ provides three built-in callbacks in the ``nlsq.callbacks`` module:

1. ProgressBar
^^^^^^^^^^^^^^

Display a visual progress bar during optimization:

.. code:: python

   from nlsq import curve_fit
   from nlsq.callbacks import ProgressBar
   import jax.numpy as jnp
   import numpy as np


   def exponential(x, a, b):
       return a * jnp.exp(-b * x)


   x = np.linspace(0, 5, 100)
   y = 2.5 * np.exp(-1.3 * x) + 0.1 * np.random.randn(100)

   # Create progress bar callback
   progress = ProgressBar(max_iterations=100)

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1], callback=progress, max_nfev=100)

**Output:**

::

   Fitting: |████████████████████| 100% [Cost: 0.0234]

2. IterationLogger
^^^^^^^^^^^^^^^^^^

Log detailed information at each iteration:

.. code:: python

   from nlsq.callbacks import IterationLogger

   # Log every iteration
   logger = IterationLogger(log_every=1)

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1], callback=logger)

**Output:**

::

   Iteration 1: cost=1.2345, params=[2.1, 1.05], grad_norm=0.234
   Iteration 2: cost=0.8765, params=[2.3, 1.15], grad_norm=0.156
   ...

3. EarlyStopping
^^^^^^^^^^^^^^^^

Stop optimization when improvement plateaus:

.. code:: python

   from nlsq.callbacks import EarlyStopping

   # Stop if cost doesn't improve by 0.1% for 10 iterations
   early_stop = EarlyStopping(
       patience=10, min_delta=0.001, mode="relative"  # 0.1% relative improvement
   )

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1], callback=early_stop)

   print(f"Stopped early: {early_stop.stopped}")
   print(f"Best cost: {early_stop.best_cost}")

Custom Callbacks
~~~~~~~~~~~~~~~~

Create custom callbacks by defining a function with signature:

.. code:: python

   def custom_callback(iteration, cost, params, info):
       """
       Parameters
       ----------
       iteration : int
           Current iteration number (0-indexed)
       cost : float
           Current cost function value
       params : ndarray
           Current parameter values
       info : dict
           Additional information (gradient norm, step norm, etc.)

       Returns
       -------
       stop : bool
           True to stop optimization early, False to continue
       """
       # Custom logic
       if iteration > 50 and cost < 0.01:
           print("Good enough! Stopping early.")
           return True  # Stop optimization

       if iteration % 10 == 0:
           print(f"Iter {iteration}: cost={cost:.6f}, params={params}")

       return False  # Continue


   # Use custom callback
   popt, pcov = curve_fit(
       exponential, x, y, p0=[2, 1], callback=custom_callback, max_nfev=100
   )

Combining Multiple Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain multiple callbacks together:

.. code:: python

   from nlsq.callbacks import CallbackChain, ProgressBar, IterationLogger, EarlyStopping

   # Create chain of callbacks
   callbacks = CallbackChain(
       [
           ProgressBar(max_iterations=100),
           IterationLogger(log_every=10),
           EarlyStopping(patience=15, min_delta=0.0001),
       ]
   )

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1], callback=callbacks)

--------------

Robust Fitting with Loss Functions
----------------------------------

Robust loss functions reduce the influence of outliers by downweighting
large residuals.

Available Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~

============= ================================== =====================
Loss Function Formula                            Use Case
============= ================================== =====================
``'linear'``  ρ(z) = z                           No outliers (default)
``'soft_l1'`` ρ(z) = 2[(1 + z)^0.5 - 1]          Mild outliers
``'huber'``   ρ(z) = z if z ≤ 1, else 2z^0.5 - 1 Moderate outliers
``'cauchy'``  ρ(z) = ln(1 + z)                   Severe outliers
``'arctan'``  ρ(z) = arctan(z)                   Extreme outliers
============= ================================== =====================

where z = (residual / f_scale)²

Example: Fitting with Outliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import matplotlib.pyplot as plt

   # Generate data with outliers
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 2.5 * np.exp(-0.5 * x)
   y = y_true + 0.1 * np.random.randn(100)

   # Add 10 outliers
   outlier_indices = np.random.choice(100, 10, replace=False)
   y[outlier_indices] += np.random.randn(10) * 2.0

   # Fit with different loss functions
   losses = ["linear", "soft_l1", "huber", "cauchy"]
   results = {}

   for loss in losses:
       popt, pcov = curve_fit(
           exponential,
           x,
           y,
           p0=[2, 0.5],
           loss=loss,
           f_scale=0.5,  # Tuning parameter for robust losses
       )
       results[loss] = popt

   # Compare results
   for loss, popt in results.items():
       y_fit = exponential(x, *popt)
       rmse = np.sqrt(np.mean((y - y_fit) ** 2))
       print(f"{loss:8s}: a={popt[0]:.3f}, b={popt[1]:.3f}, RMSE={rmse:.4f}")

**Output:**

::

   linear  : a=2.234, b=0.447, RMSE=0.5823  (affected by outliers)
   soft_l1 : a=2.487, b=0.496, RMSE=0.4156  (mild robustness)
   huber   : a=2.501, b=0.501, RMSE=0.3982  (better)
   cauchy  : a=2.498, b=0.499, RMSE=0.3845  (best for severe outliers)

Tuning f_scale Parameter
~~~~~~~~~~~~~~~~~~~~~~~~

The ``f_scale`` parameter determines the transition point between
quadratic and linear/constant behavior:

-  **Small f_scale (e.g., 0.1)**: More aggressive outlier rejection
-  **Large f_scale (e.g., 1.0)**: More conservative, closer to least
   squares
-  **Rule of thumb**: Set
   ``f_scale ≈ expected noise standard deviation``

.. code:: python

   # Automatic f_scale from robust MAD estimator
   from nlsq.utils import estimate_f_scale

   # Initial fit to get residuals
   popt_init, _ = curve_fit(exponential, x, y, p0=[2, 0.5])
   residuals = y - exponential(x, *popt_init)
   f_scale = estimate_f_scale(residuals)

   # Refit with estimated f_scale
   popt, pcov = curve_fit(exponential, x, y, p0=popt_init, loss="huber", f_scale=f_scale)

--------------

Large Dataset Handling
----------------------

NLSQ provides specialized handling for datasets with millions of points.

Automatic Large Dataset Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``curve_fit`` function automatically detects large datasets:

.. code:: python

   # For very large datasets (> 20M points)
   x_large = np.linspace(0, 100, 25_000_000)
   y_large = exponential(x_large, 2.5, 0.5) + 0.01 * np.random.randn(25_000_000)

   # Automatically uses chunking and memory management
   popt, pcov = curve_fit(exponential, x_large, y_large, p0=[2, 0.5])

Manual Large Dataset Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For explicit control over chunking and memory:

.. code:: python

   from nlsq.large_dataset import fit_large_dataset

   popt, pcov, info = fit_large_dataset(
       f=exponential,
       xdata=x_large,
       ydata=y_large,
       p0=[2, 0.5],
       chunk_size=1_000_000,  # Process 1M points at a time
       memory_limit_gb=4.0,  # Limit GPU memory usage
       progress=True,  # Show progress bar
       solver="cg",  # Use conjugate gradient for efficiency
   )

   print(f"Chunks processed: {info['n_chunks']}")
   print(f"Peak memory: {info['peak_memory_gb']:.2f} GB")
   print(f"Processing time: {info['time']:.2f} seconds")

Streaming Optimization
~~~~~~~~~~~~~~~~~~~~~~

For datasets too large to fit in memory:

.. code:: python

   from nlsq.streaming_optimizer import StreamingOptimizer


   # Generator that yields data chunks
   def data_generator():
       for i in range(100):  # 100 chunks
           x_chunk = np.linspace(i, i + 1, 100_000)
           y_chunk = exponential(x_chunk, 2.5, 0.5) + 0.01 * np.random.randn(100_000)
           yield x_chunk, y_chunk


   optimizer = StreamingOptimizer(
       model=exponential, p0=[2, 0.5], buffer_size=3  # Keep 3 chunks in memory
   )

   popt, pcov = optimizer.fit(data_generator())

--------------

Algorithm Selection
-------------------

NLSQ automatically selects the best algorithm based on problem
characteristics.

Trust Region Reflective (TRF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default algorithm, suitable for most problems:

.. code:: python

   popt, pcov = curve_fit(
       exponential,
       x,
       y,
       p0=[2, 0.5],
       method="trf",  # Explicit (default)
       bounds=([0, 0], [10, 5]),  # With bounds
   )

**Best for:** - Problems with bounds - Medium to large datasets (100-10M
points) - Most general-purpose applications

Solver Selection
~~~~~~~~~~~~~~~~

Different solvers for different problem structures:

.. code:: python

   # SVD solver (default for small problems)
   popt, pcov = curve_fit(exponential, x, y, solver="svd")

   # Conjugate Gradient (memory efficient for large problems)
   popt, pcov = curve_fit(exponential, x_large, y_large, solver="cg")

   # LSQR (good for sparse Jacobians)
   popt, pcov = curve_fit(exponential, x, y, solver="lsqr")

   # Minibatch (for very large datasets)
   popt, pcov = curve_fit(
       exponential, x_large, y_large, solver="minibatch", batch_size=10_000
   )

   # Auto (recommended - automatically selects best solver)
   popt, pcov = curve_fit(exponential, x, y, solver="auto")

Algorithm Selection Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

============= ========== ====== =================================
Dataset Size  Parameters Bounds Recommended Solver
============= ========== ====== =================================
< 10K points  < 10       No     ``svd``
< 10K points  < 10       Yes    ``trf`` + ``svd``
10K-1M points Any        Any    ``trf`` + ``cg``
> 1M points   Any        Any    ``trf`` + ``cg`` or ``minibatch``
> 20M points  Any        Any    ``fit_large_dataset``
============= ========== ====== =================================

--------------

Memory Management
-----------------

Control memory usage for GPU/TPU acceleration.

Memory Configuration
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq.memory_manager import MemoryConfig, MemoryManager

   # Configure memory limits
   config = MemoryConfig(
       max_memory_gb=8.0,  # Maximum GPU memory
       chunk_size=1_000_000,  # Chunk size for large datasets
       cache_size_mb=512,  # JIT compilation cache
       enable_monitoring=True,  # Monitor memory usage
   )

   # Create memory manager
   manager = MemoryManager(config)

   # Fit with memory monitoring
   with manager.monitor():
       popt, pcov = curve_fit(exponential, x_large, y_large)

   print(f"Peak memory: {manager.peak_memory_gb:.2f} GB")
   print(f"Average memory: {manager.avg_memory_gb:.2f} GB")

Memory Estimation
~~~~~~~~~~~~~~~~~

Estimate memory requirements before fitting:

.. code:: python

   from nlsq.large_dataset import estimate_memory_requirements

   # Estimate memory for a fit
   mem_est = estimate_memory_requirements(
       n_points=10_000_000, n_params=5, dtype=np.float64
   )

   print(f"Estimated memory: {mem_est['total_gb']:.2f} GB")
   print(f"Jacobian memory: {mem_est['jacobian_gb']:.2f} GB")
   print(f"Data memory: {mem_est['data_gb']:.2f} GB")
   print(f"Recommended chunk size: {mem_est['recommended_chunk_size']:,}")

--------------

Diagnostic Monitoring
---------------------

Monitor optimization health and numerical stability.

Diagnostic Tools
~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq.diagnostics import DiagnosticMonitor

   # Create diagnostic monitor
   monitor = DiagnosticMonitor(
       check_condition_number=True,
       check_gradient_norm=True,
       check_step_quality=True,
       log_level="INFO",
   )

   # Fit with diagnostics
   popt, pcov = curve_fit(exponential, x, y, p0=[2, 0.5], diagnostics=monitor)

   # Review diagnostics
   print(monitor.summary())

**Output:**

::

   Diagnostic Summary:
   ├─ Condition number: 12.34 (well-conditioned)
   ├─ Max gradient norm: 0.0123
   ├─ Avg step quality: 0.89 (good)
   ├─ Numerical warnings: 0
   └─ Convergence: SUCCESS

Stability Checks
~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq.stability import check_numerical_stability

   # Check stability of a fit
   stability = check_numerical_stability(
       jacobian=res.jac, residuals=res.fun, parameters=popt
   )

   if not stability["is_stable"]:
       print("Warning: Numerical instability detected!")
       print(f"Condition number: {stability['condition_number']:.2e}")
       print(f"Recommendations: {stability['recommendations']}")

--------------

Sparse Jacobian Optimization
----------------------------

For models with sparse Jacobian structure, provide sparsity pattern for
significant speedups.

Defining Sparsity Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import scipy.sparse as sp


   def complex_model(x, *params):
       # Model where each output depends on only a few parameters
       # (e.g., piecewise models, additive components)
       ...


   # Define sparsity pattern (n_outputs × n_params)
   # 1 = nonzero, 0 = always zero
   sparsity = sp.lil_matrix((len(x), len(p0)))
   sparsity[0:50, 0:2] = 1  # First 50 outputs depend on params 0-1
   sparsity[50:100, 2:4] = 1  # Next 50 outputs depend on params 2-3

   popt, pcov = curve_fit(complex_model, x, y, p0=p0, jac_sparsity=sparsity)

**Performance gain:** 2-10x faster for sparse problems with > 10
parameters.

--------------

.. _streaming-optimization-1:

Streaming Optimization
----------------------

For online learning or real-time fitting scenarios.

Online Fitting
~~~~~~~~~~~~~~

.. code:: python

   from nlsq.streaming_optimizer import OnlineOptimizer

   # Initialize online optimizer
   optimizer = OnlineOptimizer(
       model=exponential, p0=[2, 0.5], learning_rate=0.01, momentum=0.9
   )

   # Process data as it arrives
   for x_batch, y_batch in data_stream:
       popt = optimizer.update(x_batch, y_batch)
       print(f"Current estimate: {popt}")

   # Final parameters
   popt_final = optimizer.get_parameters()
   pcov_final = optimizer.get_covariance()

--------------

Best Practices Summary
----------------------

1.  **Use callbacks** for long-running fits to monitor progress
2.  **Choose robust loss functions** when outliers are present
3.  **Use ``solver='auto'``** for automatic solver selection
4.  **Enable memory monitoring** for large datasets (> 1M points)
5.  **Provide sparsity patterns** for sparse Jacobians (> 10 params)
6.  **Set realistic bounds** to improve convergence
7.  **Use diagnostics** to detect numerical issues early
8.  **Consider ``fit_large_dataset``** for datasets > 20M points
9.  **Tune ``f_scale``** based on expected noise level
10. **Monitor condition numbers** for ill-conditioned problems

--------------

Related Documentation
---------------------

-  :doc:`performance_guide` - GPU acceleration, JIT compilation
-  :doc:`migration_scipy` - Migrating from SciPy
-  :doc:`troubleshooting` - Common issues and solutions
-  :doc:`../api/index` - Complete API documentation
