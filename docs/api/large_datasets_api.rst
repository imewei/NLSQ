Large Dataset API Reference
===========================

This page documents the API for NLSQ's large dataset handling features, designed for datasets with 20M+ points.

Memory Estimation
-----------------

.. autofunction:: nlsq.estimate_memory_requirements
   :no-index:

The ``estimate_memory_requirements`` function returns a ``DatasetStats`` object with the following attributes:

- ``n_points``: Number of data points
- ``n_params``: Number of parameters
- ``total_memory_estimate_gb``: Estimated memory requirement in GB
- ``recommended_chunk_size``: Recommended chunk size for processing
- ``n_chunks``: Number of chunks needed

Example::

    from nlsq import estimate_memory_requirements

    stats = estimate_memory_requirements(100_000_000, 4)
    print(f"Total memory: {stats.total_memory_estimate_gb:.2f} GB")
    print(f"Process in {stats.n_chunks} chunks of {stats.recommended_chunk_size} points")

LargeDatasetFitter
------------------

The main class for handling large datasets with automatic memory management.

For complete API documentation, see :class:`nlsq.LargeDatasetFitter` in the :doc:`nlsq.large_dataset` module.

**Key Features:**

- Automatic memory management and chunking
- Progress reporting for long-running fits
- Configurable memory limits and chunk sizes
- Integration with existing NLSQ optimization algorithms

**Constructor Parameters:**

- ``memory_limit_gb`` (float): Maximum memory to use (default: 4.0)
- ``config`` (LDMemoryConfig, optional): Advanced configuration object

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

``curve_fit_large`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Primary large dataset fitting function with automatic dataset size detection.**

For complete API documentation, see :func:`nlsq.curve_fit_large` in the :doc:`nlsq.large_dataset` module.

This function provides a drop-in replacement for ``curve_fit`` with automatic
detection and handling of large datasets. For small datasets (< 1M points),
it behaves identically to ``curve_fit``. For larger datasets, it automatically
switches to memory-efficient processing with chunking and progress reporting.

Parameters:
    - ``func``: Model function f(x, \\*params) -> y
    - ``xdata``: Independent variable data
    - ``ydata``: Dependent variable data
    - ``p0``: Initial parameter guess
    - ``memory_limit_gb``: Memory limit in GB (default: auto-detect)
    - ``auto_size_detection``: Automatically detect dataset size (default: True)
    - ``size_threshold``: Threshold for switching to large dataset processing (default: 1M)
    - ``show_progress``: Show progress bar for large datasets (default: False)
    - ``**kwargs``: Additional fitting options

Returns:
    ``popt, pcov`` tuple (same as scipy.optimize.curve_fit)

Example::

    from nlsq import curve_fit_large
    import jax.numpy as jnp

    # Automatic handling - uses standard curve_fit for small datasets
    popt, pcov = curve_fit_large(func, x_small, y_small, p0=[1, 0.5])

    # Automatic chunking for large datasets
    popt, pcov = curve_fit_large(
        func, x_large, y_large,
        p0=[1, 0.5],
        memory_limit_gb=8.0,
        show_progress=True
    )

``fit_large_dataset`` Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advanced large dataset fitting with OptimizeResult return format.**

For complete API documentation, see :func:`nlsq.fit_large_dataset` in the :doc:`nlsq.large_dataset` module.

Parameters:
    - ``func``: Model function
    - ``xdata``: Independent variable
    - ``ydata``: Dependent variable
    - ``p0``: Initial parameters
    - ``memory_limit_gb``: Memory limit (default: 4.0)
    - ``show_progress``: Show progress bar (default: False)
    - ``**kwargs``: Additional fitting options

Returns:
    ``OptimizeResult`` object with detailed fitting information

Example::

    from nlsq import fit_large_dataset

    result = fit_large_dataset(
        exponential, x_data, y_data,
        p0=[1.0, 0.5, 0.1],
        memory_limit_gb=4.0,
        show_progress=True
    )
    print(f"Success: {result.success}")
    print(f"Parameters: {result.popt}")
    # Note: n_chunks only available for multi-chunk fits
    if hasattr(result, 'n_chunks'):
        print(f"Chunks used: {result.n_chunks}")

Advanced Features
-----------------

**Sparse Jacobian Support**

For problems with sparse Jacobian structures, NLSQ provides:

- Automatic sparsity detection via :class:`nlsq.SparseJacobianComputer`
- Sparse matrix optimizations
- Memory-efficient sparse solvers

See :doc:`nlsq.large_dataset` for usage examples.

Example::

    from nlsq import SparseJacobianComputer

    # Detect sparsity automatically
    sparse_computer = SparseJacobianComputer(sparsity_threshold=0.01)
    pattern, sparsity = sparse_computer.detect_sparsity_pattern(func, p0, x_sample)

    if sparsity > 0.1:  # If more than 10% sparse
        print(f"Jacobian is {sparsity:.1%} sparse")
        # NLSQ will automatically use sparse optimization

**Streaming Optimization**

For unlimited-size datasets, use the :doc:`nlsq.streaming_optimizer` module:

- Batch processing from disk or generators
- SGD and Adam optimizers for online learning
- HDF5 file integration
- Real-time data stream processing

Example::

    from nlsq import fit_unlimited_data, StreamingConfig

    # Fit to HDF5 dataset or generator
    config = StreamingConfig(batch_size=10000, max_epochs=10)
    result = fit_unlimited_data(func, "huge_dataset.h5", p0=p0, config=config)

Memory Configuration
--------------------

Advanced memory configuration options.

For complete API documentation, see :class:`nlsq.large_dataset.LDMemoryConfig` in the :doc:`nlsq.large_dataset` module.

**Parameters:**

- ``memory_limit_gb``: Maximum memory in GB
- ``safety_factor``: Safety factor for memory calculations (default: 0.8)
- ``min_chunk_size``: Minimum chunk size (default: 1000)
- ``max_chunk_size``: Maximum chunk size (default: 1000000)
- ``min_success_rate``: Minimum success rate for chunked fitting (default: 0.5)

Example::

    from nlsq import LargeDatasetFitter
    from nlsq.streaming.large_dataset import LDMemoryConfig

    # Custom configuration
    config = LDMemoryConfig(
        memory_limit_gb=8.0,
        safety_factor=0.9,
        min_chunk_size=10000,
        max_chunk_size=1000000,
        min_success_rate=0.8,  # Require 80% of chunks to succeed
    )

    fitter = LargeDatasetFitter(config=config)

Data Chunking
-------------

Utility class for chunking large arrays.

For complete API documentation, see :class:`nlsq.large_dataset.DataChunker` in the :doc:`nlsq.large_dataset` module.

Returns:
    Iterator yielding (x_chunk, y_chunk, indices) tuples

Example::

    from nlsq.streaming.large_dataset import DataChunker

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

4. **Use streaming for very large datasets**::

    # For datasets >100M points, streaming optimization processes
    # all data in chunks with zero accuracy loss
    if n_points > 100_000_000:
        # Streaming is automatic in curve_fit_large
        popt, pcov = curve_fit_large(func, xdata, ydata, p0=p0,
                                      memory_limit_gb=8.0, show_progress=True)

See Also
--------

- :doc:`../index` - Main NLSQ documentation
- :doc:`../howto/handle_large_data` - Detailed guide for large datasets
- :doc:`modules` - Complete API reference
- `Examples notebook <https://github.com/imewei/NLSQ/blob/main/examples/large_dataset_demo.ipynb>`_
