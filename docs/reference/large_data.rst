Large Dataset API Reference
===========================

APIs for handling datasets that exceed GPU memory.

Overview
--------

NLSQ provides three approaches for large datasets:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - API
     - Best For
     - Memory Behavior
   * - ``curve_fit_large``
     - Datasets up to ~100M points
     - Automatic chunking, fits in memory
   * - ``LargeDatasetFitter``
     - Fine-grained control over chunking
     - Automatic memory management
   * - ``AdaptiveHybridStreamingOptimizer``
     - Huge datasets and production pipelines
     - Four-phase optimization

curve_fit_large
---------------

.. autofunction:: nlsq.curve_fit_large
   :no-index:

Automatically chunks data to fit within GPU memory.

**Key Parameters:**

- ``memory_limit_gb``: Maximum GPU memory to use
- ``chunk_size``: Override automatic chunk sizing
- ``progress``: Show progress bar during fitting

**Example:**

.. code-block:: python

   from nlsq import curve_fit_large

   # 50 million data points
   x = jnp.linspace(0, 100, 50_000_000)
   y = 2.5 * jnp.exp(-0.1 * x) + 0.5 + noise

   popt, pcov = curve_fit_large(
       model, x, y, p0=[1.0, 0.1, 0.0], memory_limit_gb=8.0  # Limit to 8 GB
   )

LargeDatasetFitter
------------------

.. autoclass:: nlsq.LargeDatasetFitter
   :members:
   :special-members: __init__
   :no-index:

Class-based interface for large dataset fitting with more control.

**Example:**

.. code-block:: python

   from nlsq import LargeDatasetFitter

   fitter = LargeDatasetFitter(
       memory_limit_gb=8.0, chunk_overlap=0.1  # 10% overlap between chunks
   )

   result = fitter.fit(model, x, y, p0=p0)

AdaptiveHybridStreamingOptimizer
--------------------------------

.. autoclass:: nlsq.AdaptiveHybridStreamingOptimizer
   :members:
   :special-members: __init__
   :no-index:

Production-grade optimizer with four-phase optimization:

1. **Parameter normalization**: Scales parameters for stability
2. **L-BFGS warmup**: Fast initial convergence
3. **Streaming Gauss-Newton**: Precise refinement
4. **Exact covariance**: Accurate uncertainty estimates

**Example:**

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer
   from nlsq import HybridStreamingConfig

   config = HybridStreamingConfig.from_preset("production")

   optimizer = AdaptiveHybridStreamingOptimizer(config)

   result = optimizer.fit((x, y), model, p0=p0)

HybridStreamingConfig
---------------------

.. autoclass:: nlsq.HybridStreamingConfig
   :members:
   :no-index:

Configuration for the hybrid streaming optimizer.

**Presets:**

- ``"fast"``: Quick convergence, lower precision
- ``"balanced"``: Good balance of speed and accuracy
- ``"production"``: Maximum reliability
- ``"research"``: Highest precision

**Example:**

.. code-block:: python

   from nlsq import HybridStreamingConfig

   # From preset
   config = HybridStreamingConfig.from_preset("production")

   # Custom configuration
   config = HybridStreamingConfig(
       warmup_iterations=50,
       gauss_newton_max_iterations=20,
       chunk_size=50_000,
       checkpoint_frequency=10,
   )

Memory Estimation
-----------------

.. autofunction:: nlsq.estimate_memory_requirements
   :no-index:

Estimate memory requirements before fitting:

.. code-block:: python

   from nlsq import estimate_memory_requirements

   mem = estimate_memory_requirements(n_points=10_000_000, n_params=5, dtype="float64")

   print(f"Estimated GPU memory: {mem['gpu_memory_gb']:.2f} GB")
   print(f"Recommended chunk size: {mem['chunk_size']}")

Checkpointing
-------------

AdaptiveHybridStreamingOptimizer supports checkpointing for long-running fits:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

   config = HybridStreamingConfig(
       checkpoint_dir="./checkpoints",
       checkpoint_frequency=100,
   )
   optimizer = AdaptiveHybridStreamingOptimizer(config)

   # If interrupted, resume from checkpoint
   result = optimizer.fit((x, y), model, p0=p0, verbose=1)

Parallel Processing
-------------------

For multi-GPU systems:

.. code-block:: python

   from nlsq import ParallelFitter

   fitter = ParallelFitter(n_gpus=4, memory_per_gpu_gb=16.0)

   result = fitter.fit(model, x, y, p0=p0)

See Also
--------

- :doc:`/tutorials/05_large_datasets` - Tutorial on large data handling
- :doc:`/howto/streaming_checkpoints` - Checkpoint and resume guide
- :doc:`/explanation/streaming` - Streaming concepts
