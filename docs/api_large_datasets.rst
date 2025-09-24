Large Dataset API Reference
===========================

This page documents the API for NLSQ's large dataset handling features, designed for datasets with 20M+ points.

Memory Estimation
-----------------

.. autofunction:: nlsq.estimate_memory_requirements

The ``estimate_memory_requirements`` function returns a ``DatasetStats`` object with the following attributes:

- ``n_points``: Number of data points
- ``n_params``: Number of parameters
- ``total_memory_estimate_gb``: Estimated memory requirement in GB
- ``recommended_chunk_size``: Recommended chunk size for processing
- ``n_chunks``: Number of chunks needed
- ``requires_sampling``: Whether sampling is recommended for this dataset size

Example::

    from nlsq import estimate_memory_requirements

    stats = estimate_memory_requirements(100_000_000, 4)
    if stats.requires_sampling:
        print(f"Dataset too large, sampling recommended")
    else:
        print(f"Process in {stats.n_chunks} chunks")

LargeDatasetFitter
------------------

.. autoclass:: nlsq.LargeDatasetFitter
   :members:
   :undoc-members:
   :show-inheritance:

The main class for handling large datasets with automatic memory management.

**Constructor Parameters:**

- ``memory_limit_gb`` (float): Maximum memory to use (default: 4.0)
- ``config`` (LDMemoryConfig, optional): Advanced configuration object

**Key Methods:**

.. automethod:: nlsq.LargeDatasetFitter.fit

Main fitting method with automatic chunking.

Parameters:
    - ``func``: Model function to fit
    - ``xdata``: Independent variable data
    - ``ydata``: Dependent variable data
    - ``p0``: Initial parameter guess
    - ``bounds``: Optional parameter bounds
    - ``**kwargs``: Additional options passed to underlying solver

Returns:
    ``LDResult`` object with attributes:

    - ``popt``: Optimized parameters
    - ``pcov``: Covariance matrix
    - ``success``: Whether fit succeeded
    - ``message``: Status message
    - ``n_chunks``: Number of chunks used
    - ``chunk_size``: Size of chunks

.. automethod:: nlsq.LargeDatasetFitter.fit_with_progress

Fitting with progress bar for long-running fits.

.. automethod:: nlsq.LargeDatasetFitter.get_memory_recommendations

Get memory usage recommendations before fitting.

Example::

    from nlsq import LargeDatasetFitter

    fitter = LargeDatasetFitter(memory_limit_gb=8.0)

    # Get recommendations
    recs = fitter.get_memory_recommendations(50_000_000, 3)
    print(recs['processing_strategy'])

    # Fit with progress
    result = fitter.fit_with_progress(func, x, y, p0)

Convenience Functions
---------------------

.. autofunction:: nlsq.fit_large_dataset

High-level convenience function for large dataset fitting.

Parameters:
    - ``func``: Model function
    - ``xdata``: Independent variable
    - ``ydata``: Dependent variable
    - ``p0``: Initial parameters
    - ``memory_limit_gb``: Memory limit (default: 4.0)
    - ``show_progress``: Show progress bar (default: False)
    - ``**kwargs``: Additional fitting options

Returns:
    ``LDResult`` object

Example::

    from nlsq import fit_large_dataset

    result = fit_large_dataset(
        exponential, x_data, y_data,
        p0=[1.0, 0.5, 0.1],
        memory_limit_gb=4.0,
        show_progress=True
    )

Sparse Jacobian Support
-----------------------

.. autoclass:: nlsq.SparseJacobianComputer
   :members:
   :undoc-members:

Detects and exploits sparsity in Jacobian matrices.

**Key Methods:**

.. automethod:: nlsq.SparseJacobianComputer.detect_sparsity

Detect sparsity pattern in Jacobian.

.. automethod:: nlsq.SparseJacobianComputer.is_sparse

Check if Jacobian is sufficiently sparse.

.. automethod:: nlsq.SparseJacobianComputer.compute_sparsity_ratio

Compute ratio of zero elements.

.. autoclass:: nlsq.SparseOptimizer
   :members:
   :undoc-members:

Optimizer that exploits sparse Jacobian structure.

.. automethod:: nlsq.SparseOptimizer.optimize_with_sparsity

Optimize using sparse methods.

Example::

    from nlsq import SparseJacobianComputer, SparseOptimizer

    # Detect sparsity
    computer = SparseJacobianComputer(sparsity_threshold=0.01)
    pattern = computer.detect_sparsity(func, x[:1000], p0)

    if computer.is_sparse(pattern):
        # Use sparse optimization
        optimizer = SparseOptimizer()
        result = optimizer.optimize_with_sparsity(
            func, x, y, p0, pattern
        )

Streaming Optimizer
-------------------

.. autoclass:: nlsq.StreamingOptimizer
   :members:
   :undoc-members:

Optimizer for datasets that don't fit in memory.

.. autoclass:: nlsq.StreamingConfig
   :members:
   :undoc-members:

Configuration for streaming optimization.

**Parameters:**

- ``batch_size``: Size of data batches (default: 10000)
- ``max_epochs``: Maximum training epochs (default: 100)
- ``convergence_tol``: Convergence tolerance (default: 1e-6)
- ``use_adam``: Use Adam optimizer (default: True)
- ``learning_rate``: Initial learning rate (default: 0.001)

**Key Methods:**

.. automethod:: nlsq.StreamingOptimizer.fit_unlimited_data

Fit using data generator or iterator.

.. automethod:: nlsq.StreamingOptimizer.fit_from_hdf5

Fit directly from HDF5 file.

HDF5 Dataset Support
--------------------

.. autofunction:: nlsq.create_hdf5_dataset

Create HDF5 dataset for streaming.

Parameters:
    - ``filename``: Output HDF5 file path
    - ``func``: Function to generate data
    - ``params``: True parameters
    - ``n_samples``: Number of samples
    - ``chunk_size``: HDF5 chunk size
    - ``noise_level``: Noise to add (default: 0.1)

.. autofunction:: nlsq.stream_from_hdf5

Stream data from HDF5 file.

Returns:
    Generator yielding (x, y) batches

Example::

    from nlsq import (StreamingOptimizer, StreamingConfig,
                     create_hdf5_dataset, stream_from_hdf5)

    # Create large dataset on disk
    create_hdf5_dataset(
        "data.h5", exponential, [2.0, 0.5, 0.1],
        n_samples=100_000_000,
        chunk_size=10000
    )

    # Configure streaming
    config = StreamingConfig(
        batch_size=10000,
        max_epochs=50,
        use_adam=True
    )

    # Fit from HDF5
    optimizer = StreamingOptimizer(config)
    result = optimizer.fit_from_hdf5("data.h5", exponential, p0)

Memory Configuration
--------------------

.. autoclass:: nlsq.large_dataset.LDMemoryConfig
   :members:
   :undoc-members:

Advanced memory configuration options.

**Parameters:**

- ``memory_limit_gb``: Maximum memory in GB
- ``chunk_size_factor``: Chunk size multiplier (default: 0.8)
- ``min_chunk_size``: Minimum chunk size (default: 1000)
- ``max_chunk_size``: Maximum chunk size (default: 10_000_000)
- ``enable_sampling``: Allow sampling for very large datasets
- ``sampling_threshold_gb``: Memory threshold for sampling (default: 16.0)
- ``sample_rate``: Sampling rate when enabled (default: 0.1)

Example::

    from nlsq import LargeDatasetFitter
    from nlsq.large_dataset import LDMemoryConfig

    # Custom configuration
    config = LDMemoryConfig(
        memory_limit_gb=8.0,
        chunk_size_factor=0.9,
        enable_sampling=True,
        sampling_threshold_gb=10.0,
        sample_rate=0.2
    )

    fitter = LargeDatasetFitter(config=config)

Data Chunking
-------------

.. autoclass:: nlsq.large_dataset.DataChunker
   :members:
   :undoc-members:

Utility class for chunking large arrays.

.. automethod:: nlsq.large_dataset.DataChunker.create_chunks

Create iterator of data chunks.

Returns:
    Iterator yielding (x_chunk, y_chunk, indices) tuples

Example::

    from nlsq.large_dataset import DataChunker

    chunker = DataChunker(chunk_size=100000)

    for x_chunk, y_chunk, idx in chunker.create_chunks(x, y):
        # Process chunk
        result = process_chunk(x_chunk, y_chunk)

Performance Considerations
--------------------------

Memory Usage Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Dataset sizes and recommended approaches:

- **< 1M points**: Use standard ``curve_fit``
- **1M - 10M points**: Use ``LargeDatasetFitter`` with default settings
- **10M - 100M points**: Use ``LargeDatasetFitter`` with chunking
- **100M - 1B points**: Use ``StreamingOptimizer`` with HDF5
- **> 1B points**: Use sampling strategies or distributed computing

Memory Estimation Formula
~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate memory usage::

    memory_gb = n_points * (3 * n_params + 5) * 8 / 1e9

Where:
- 3 factors: x data, y data, residuals
- n_params: Jacobian columns
- 5: Working arrays
- 8: Bytes per float64

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Check sparsity first**: Many large problems have sparse Jacobians
2. **Use iterative solvers**: CG and LSQR use less memory than SVD
3. **Enable sampling**: For exploratory analysis on very large datasets
4. **Stream from disk**: Use HDF5 for datasets larger than RAM
5. **Monitor progress**: Use ``fit_with_progress`` for long fits

Best Practices
--------------

1. **Always estimate memory first**::

    stats = estimate_memory_requirements(n_points, n_params)
    if stats.total_memory_estimate_gb > available_memory:
        use_large_dataset_fitter()

2. **Use appropriate chunk sizes**::

    # Chunk size affects performance
    # Too small: overhead from many iterations
    # Too large: memory issues
    optimal_chunk = int(available_memory_gb * 1e9 / (8 * 3 * n_params))

3. **Leverage sparsity when available**::

    # Many scientific problems have sparse Jacobians
    # (e.g., fitting multiple peaks, piecewise functions)
    if expected_sparsity > 0.9:
        use_sparse_optimizer()

4. **Consider data precision**::

    # If data has limited precision, sampling may not affect accuracy
    if data_precision < 1e-3 and n_points > 100_000_000:
        enable_sampling()

See Also
--------

- :doc:`main` - Main NLSQ documentation
- :doc:`large_dataset_guide` - Detailed guide for large datasets
- :doc:`autodoc/modules` - Complete API reference
- `Examples notebook <https://github.com/Dipolar-Quantum-Gases/nlsq/blob/main/examples/large_dataset_demo.ipynb>`_
