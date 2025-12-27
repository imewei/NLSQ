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
   * - ``StreamingOptimizer``
     - Unlimited size, disk-based
     - Streams from disk, constant memory
   * - ``AdaptiveHybridStreamingOptimizer``
     - Production pipelines
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

StreamingOptimizer
------------------

.. autoclass:: nlsq.StreamingOptimizer
   :members:
   :special-members: __init__
   :no-index:

Stream data from disk for datasets that cannot fit in memory at all.

**Key Parameters:**

- ``model``: The model function
- ``n_params``: Number of parameters
- ``chunk_size``: Points to process per iteration
- ``checkpoint_interval``: How often to save progress

**Example:**

.. code-block:: python

   from nlsq import StreamingOptimizer

   optimizer = StreamingOptimizer(model=exponential, n_params=3, chunk_size=100_000)

   # From HDF5 file
   result = optimizer.fit_from_hdf5(
       "large_data.h5", x_dataset="x", y_dataset="y", p0=[1.0, 0.1, 0.0]
   )


   # Or from generator
   def data_generator():
       for chunk in load_chunks("data/*.npy"):
           yield chunk["x"], chunk["y"]


   result = optimizer.fit_from_generator(data_generator(), p0=p0)

AdaptiveHybridStreamingOptimizer
--------------------------------

.. autoclass:: nlsq.AdaptiveHybridStreamingOptimizer
   :members:
   :special-members: __init__
   :no-index:

Production-grade optimizer with four-phase optimization:

1. **Parameter normalization**: Scales parameters for stability
2. **Adam warmup**: Fast initial convergence
3. **Streaming Gauss-Newton**: Precise refinement
4. **Exact covariance**: Accurate uncertainty estimates

**Example:**

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer
   from nlsq import HybridStreamingConfig

   config = HybridStreamingConfig.from_preset("production")

   optimizer = AdaptiveHybridStreamingOptimizer(model=model, config=config)

   result = optimizer.fit(x, y, p0=p0)

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
       adam_warmup_epochs=5,
       gauss_newton_iterations=20,
       chunk_size=50_000,
       checkpoint_interval=10,
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

StreamingOptimizer supports checkpointing for long-running fits:

.. code-block:: python

   optimizer = StreamingOptimizer(
       model=model, n_params=3, checkpoint_dir="./checkpoints", checkpoint_interval=100
   )

   # If interrupted, resume from checkpoint
   result = optimizer.fit(x, y, p0=p0, resume=True)

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
