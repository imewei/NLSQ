Workflow Options Guide
======================

Use this guide when you want to tune workflow behavior without writing
custom pipeline code. It focuses on user-level configuration and common
choices that affect fit quality, robustness, and runtime.

Callbacks and Progress Monitoring
---------------------------------

Callbacks let you monitor optimization progress or stop early.

Built-in callbacks
~~~~~~~~~~~~~~~~~~

NLSQ provides three built-in callbacks in the ``nlsq.callbacks`` module.

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

   early_stop = EarlyStopping(patience=10, min_delta=0.001, mode="relative")

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1], callback=early_stop)

   print(f"Stopped early: {early_stop.stopped}")
   print(f"Best cost: {early_stop.best_cost}")

Combining multiple callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chain multiple callbacks together:

.. code:: python

   from nlsq.callbacks import CallbackChain, ProgressBar, IterationLogger, EarlyStopping

   callbacks = CallbackChain(
       [
           ProgressBar(max_iterations=100),
           IterationLogger(log_every=10),
           EarlyStopping(patience=15, min_delta=0.0001),
       ]
   )

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1], callback=callbacks)

For custom callback logic, see :doc:`advanced_customization`.

Robust Fitting with Loss Functions
----------------------------------

Robust loss functions reduce the influence of outliers by downweighting
large residuals.

Available loss functions
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

where z = (residual / f_scale)^2

Example: fitting with outliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import matplotlib.pyplot as plt

   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = 2.5 * np.exp(-0.5 * x)
   y = y_true + 0.1 * np.random.randn(100)

   outlier_indices = np.random.choice(100, 10, replace=False)
   y[outlier_indices] += np.random.randn(10) * 2.0

   losses = ["linear", "soft_l1", "huber", "cauchy"]
   results = {}

   for loss in losses:
       popt, pcov = curve_fit(
           exponential,
           x,
           y,
           p0=[2, 0.5],
           loss=loss,
           f_scale=0.5,
       )
       results[loss] = popt

   for loss, popt in results.items():
       y_fit = exponential(x, *popt)
       rmse = np.sqrt(np.mean((y - y_fit) ** 2))
       print(f"{loss:8s}: a={popt[0]:.3f}, b={popt[1]:.3f}, RMSE={rmse:.4f}")

Tuning f_scale
~~~~~~~~~~~~~~

The ``f_scale`` parameter controls the transition point between
quadratic and linear behavior:

- Small ``f_scale`` (e.g., 0.1): aggressive outlier rejection
- Large ``f_scale`` (e.g., 1.0): closer to least squares
- Rule of thumb: ``f_scale`` is near the expected noise std

Algorithm Selection
-------------------

NLSQ can select algorithms based on problem characteristics.

Trust Region Reflective (TRF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   popt, pcov = curve_fit(
       exponential,
       x,
       y,
       p0=[2, 0.5],
       method="trf",
       bounds=([0, 0], [10, 5]),
   )

Solver selection
~~~~~~~~~~~~~~~~

.. code:: python

   popt, pcov = curve_fit(exponential, x, y, solver="svd")
   popt, pcov = curve_fit(exponential, x_large, y_large, solver="cg")
   popt, pcov = curve_fit(exponential, x, y, solver="lsqr")
   popt, pcov = curve_fit(
       exponential, x_large, y_large, solver="minibatch", batch_size=10_000
   )
   popt, pcov = curve_fit(exponential, x, y, solver="auto")

Algorithm selection matrix
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

Memory Management
-----------------

Control memory usage for GPU/TPU acceleration and large datasets.

Memory configuration
~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq.memory_manager import MemoryConfig, MemoryManager

   config = MemoryConfig(
       max_memory_gb=8.0,
       chunk_size=1_000_000,
       cache_size_mb=512,
       enable_monitoring=True,
   )

   manager = MemoryManager(config)

   with manager.monitor():
       popt, pcov = curve_fit(exponential, x_large, y_large)

   print(f"Peak memory: {manager.peak_memory_gb:.2f} GB")

Large datasets
--------------

For large datasets or streaming data, see :doc:`large_datasets`.

Defense Layers (v0.3.6+)
------------------------

When using ``hybrid_streaming`` or ``AdaptiveHybridStreamingOptimizer``, the
4-Layer Defense Strategy prevents Adam warmup divergence. This is critical for
**warm-start refinement** where initial parameters are already near optimal.

The 4-layer defense system
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Layer
     - Name
     - Function
   * - 1
     - Warm Start Detection
     - Skips warmup if initial loss < 1% of data variance
   * - 2
     - Adaptive Learning Rate
     - Scales LR based on fit quality (1e-6 to 0.001)
   * - 3
     - Cost-Increase Guard
     - Aborts if loss increases > 5%
   * - 4
     - Step Clipping
     - Limits parameter update magnitude (max norm 0.1)

Using defense presets
~~~~~~~~~~~~~~~~~~~~~

NLSQ provides four defense presets for common scenarios:

.. code-block:: python

   from nlsq import HybridStreamingConfig

   # For warm-start refinement (strictest protection)
   config = HybridStreamingConfig.defense_strict()

   # For exploration (more aggressive learning)
   config = HybridStreamingConfig.defense_relaxed()

   # For production scientific computing (balanced)
   config = HybridStreamingConfig.scientific_default()

   # To disable (pre-0.3.6 behavior)
   config = HybridStreamingConfig.defense_disabled()

Monitoring defense activations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track defense layer activations via the telemetry API:

.. code-block:: python

   from nlsq import curve_fit, get_defense_telemetry, reset_defense_telemetry
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Reset telemetry before fitting
   reset_defense_telemetry()

   # Run fit with defense layers active
   popt, pcov = curve_fit(model, x, y, p0=p0, method="hybrid_streaming")

   # Check what happened
   telemetry = get_defense_telemetry()
   print(telemetry.get_summary())
   # Example output:
   # Defense Layer Activations:
   #   Layer 1 (Warm Start): 1 activation
   #   Layer 2 (Adaptive LR): 3 activations
   #   Layer 3 (Cost Guard): 0 activations
   #   Layer 4 (Step Clip): 5 activations

   # Get trigger rates
   print(telemetry.get_trigger_rates())
   # {'warm_start': 0.1, 'adaptive_lr': 0.3, 'cost_guard': 0.0, 'step_clip': 0.5}

Custom defense configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune individual layer thresholds:

.. code-block:: python

   from nlsq import HybridStreamingConfig

   config = HybridStreamingConfig(
       # Layer 1: Warm start detection
       warm_start_threshold=0.01,  # 1% of data variance
       # Layer 2: Adaptive learning rate
       lr_refinement=1e-6,  # LR when near optimal
       lr_careful=1e-5,  # LR when close to optimal
       lr_exploration=0.001,  # LR when far from optimal
       # Layer 3: Cost-increase guard
       cost_increase_tolerance=0.05,  # 5% increase allowed
       # Layer 4: Step clipping
       max_step_norm=0.1,  # Maximum step magnitude
   )

See :doc:`defense_layers` for the complete guide.

Related documentation
---------------------

- :doc:`defense_layers` - Complete defense layer guide
- :doc:`performance_guide` - GPU acceleration and JIT hints
- :doc:`troubleshooting` - Common issues and solutions
- :doc:`../api/index` - Complete API documentation
