nlsq.large\_dataset module
============================

.. currentmodule:: nlsq.streaming.large_dataset

.. automodule:: nlsq.streaming.large_dataset
   :noindex:

Overview
--------

The ``nlsq.large_dataset`` module provides specialized tools for fitting curves to datasets
that are too large to fit in memory or require chunking for efficient processing. This module
is essential for working with datasets containing millions or billions of points.

Key Features
------------

- **Automatic dataset handling** for 100M+ points
- **Intelligent chunking** with <1% error for well-conditioned problems
- **Memory estimation** and automatic memory management
- **Streaming optimization** for unlimited-size datasets
- **Progress reporting** for long-running fits

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   LargeDatasetFitter

.. autoclass:: LDMemoryConfig
   :members:
   :noindex:

Functions
---------

.. autosummary::
   :toctree: generated/

   fit_large_dataset
   estimate_memory_requirements

Examples
--------

Basic Usage with curve_fit_large
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import curve_fit_large, estimate_memory_requirements
    import jax.numpy as jnp
    import numpy as np

    # Check memory requirements
    n_points = 50_000_000  # 50 million points
    n_params = 3
    stats = estimate_memory_requirements(n_points, n_params)
    print(f"Memory required: {stats.total_memory_estimate_gb:.2f} GB")

    # Generate large dataset
    x = np.linspace(0, 10, n_points)
    y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, n_points)


    # Define fit function
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Fit with automatic chunking
    popt, pcov = curve_fit_large(
        exponential,
        x,
        y,
        p0=[2.5, 0.6, 0.2],
        memory_limit_gb=4.0,
        show_progress=True,
    )

Advanced Usage with LargeDatasetFitter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import LargeDatasetFitter, LDMemoryConfig
    import jax.numpy as jnp

    # Configure memory management
    config = LDMemoryConfig(
        memory_limit_gb=4.0,
        min_chunk_size=10000,
        max_chunk_size=1000000,
        min_success_rate=0.8,  # Require 80% of chunks to succeed
    )

    # Create fitter
    fitter = LargeDatasetFitter(config=config)

    # Fit with progress tracking
    result = fitter.fit_with_progress(exponential, x, y, p0=[2.5, 0.6, 0.2])

    print(f"Fitted parameters: {result.popt}")
    # Note: success_rate and n_chunks only available for multi-chunk fits

Streaming Optimization
~~~~~~~~~~~~~~~~~~~~~~

For datasets that don't fit in memory or are generated on-the-fly:

.. code-block:: python

    from nlsq import StreamingOptimizer, StreamingConfig

    # Configure streaming
    config = StreamingConfig(batch_size=10000, max_epochs=100, convergence_tol=1e-6)

    optimizer = StreamingOptimizer(config)

    # Stream data from generator
    result = optimizer.fit_streaming(func, data_generator, p0=p0)

Sparse Jacobian Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For problems with sparse structure:

.. code-block:: python

    from nlsq import SparseJacobianComputer

    # Detect and exploit sparsity
    sparse_computer = SparseJacobianComputer(sparsity_threshold=0.01)
    pattern, sparsity = sparse_computer.detect_sparsity_pattern(func, p0, x_sample)

    if sparsity > 0.1:  # If more than 10% sparse
        print(f"Jacobian is {sparsity:.1%} sparse")

See Also
--------

- :doc:`../howto/handle_large_data` : Large dataset guide
- :doc:`nlsq.memory_manager` : Memory management utilities
- :doc:`nlsq.streaming_optimizer` : Streaming optimization details
