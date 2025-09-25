NLSQ Large Dataset Implementation
==================================

This document describes the comprehensive large dataset fitting implementation added to NLSQ, including the ``curve_fit_large`` function and ``LargeDatasetFitter`` class for efficiently handling very large datasets (>10M points).

Overview
--------

The large dataset implementation provides intelligent memory management, automatic chunking, and smart sampling strategies for fitting curve parameters to datasets that may not fit in memory or require significant computational resources.

Primary API: curve_fit_large
-----------------------------

The ``curve_fit_large`` function is the recommended way to handle datasets of any size. It provides a drop-in replacement for scipy.optimize.curve_fit with automatic dataset size detection and intelligent processing strategy selection.

Automatic Size Detection
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import curve_fit_large
    import jax.numpy as jnp
    import numpy as np


    # Define your model function
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Small dataset (< 1M points) - uses standard curve_fit
    x_small = np.linspace(0, 5, 1000)
    y_small = exponential(x_small, 2.5, 1.3, 0.1) + np.random.normal(0, 0.01, 1000)

    popt, pcov = curve_fit_large(exponential, x_small, y_small, p0=[2, 1, 0])

    # Large dataset (>= 1M points) - automatically uses chunked processing
    x_large = np.linspace(0, 5, 10_000_000)
    y_large = exponential(x_large, 2.5, 1.3, 0.1) + np.random.normal(0, 0.01, 10_000_000)

    popt, pcov = curve_fit_large(
        exponential,
        x_large,
        y_large,
        p0=[2, 1, 0],
        memory_limit_gb=4.0,
        show_progress=True,  # Progress bar for large datasets
    )

    print(f"Fitted parameters: {popt}")
    print(f"Parameter uncertainties: {np.sqrt(np.diag(pcov))}")

Key Improvements (Recent Updates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Enhanced chunking algorithm**: Improved sequential refinement approach for better stability
- **Better convergence**: Each chunk uses previous result as initial guess
- **JAX tracing compatibility**: Supports functions with 15+ parameters without TracerArrayConversionError
- **Return type consistency**: All code paths return consistent (popt, pcov) format
- **Comprehensive testing**: Full integration test suite ensures reliability

Core Features Implemented
--------------------------

Memory Management
~~~~~~~~~~~~~~~~~

**Automatic Memory Estimation:**

.. code-block:: python

    # Estimates memory requirements based on data size and parameters
    from nlsq import estimate_memory_requirements

    stats = estimate_memory_requirements(n_points=10_000_000, n_params=3)
    print(f"Memory estimate: {stats.total_memory_estimate_gb:.2f} GB")
    print(f"Recommended chunk size: {stats.recommended_chunk_size:,}")

**Configurable Memory Limits:**

.. code-block:: python

    from nlsq.large_dataset import LDMemoryConfig

    config = LDMemoryConfig(
        memory_limit_gb=8.0,
        safety_factor=0.8,
        min_chunk_size=1000,
        max_chunk_size=1_000_000,
    )
    fitter = LargeDatasetFitter(config=config)

Processing Strategies
~~~~~~~~~~~~~~~~~~~~~

**Single Chunk (Fits in Memory):**
- Dataset fits within memory limits
- Uses standard NLSQ curve fitting
- Optimal performance for datasets < 1-2GB

**Chunked Processing:**
- Automatically divides data into manageable chunks
- Progressive parameter refinement across chunks
- Progress reporting for long-running fits
- Handles datasets up to 100M+ points

**Sampling Strategy:**
- For extremely large datasets (>100M points)
- Smart sampling with multiple strategies (random, uniform, stratified)
- Configurable sample sizes and thresholds
- Maintains statistical representativeness

Integration with NLSQ
~~~~~~~~~~~~~~~~~~~~~~

**Seamless API Integration:**

.. code-block:: python

    # Drop-in replacement for large datasets
    from nlsq import fit_large_dataset

    result = fit_large_dataset(
        model_function,
        x_data,
        y_data,
        p0=[1.0, 2.0],
        memory_limit_gb=4.0,
        show_progress=True,
    )

**Compatible with Existing Infrastructure:**
- Uses CurveFit class internally
- Supports all existing optimization methods ('trf', 'lm', etc.)
- Compatible with JAX JIT compilation
- Maintains parameter bounds and constraints support

Progress Reporting and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Real-time Progress Updates:**

.. code-block:: python

    fitter = LargeDatasetFitter(memory_limit_gb=2.0)
    result = fitter.fit_with_progress(model, x_data, y_data)
    # Outputs: Progress: 5/10 chunks (50%) - ETA: 30.2s

**Memory Usage Monitoring:**

.. code-block:: python

    with fitter.memory_monitor():
        result = fitter.fit(model, x_data, y_data)
    # Logs memory usage before/after fitting

Usage Examples
--------------

Basic Large Dataset Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from nlsq import fit_large_dataset

    # Generate large dataset
    x_data = np.linspace(0, 10, 5_000_000)
    y_data = 2.5 * np.exp(-1.3 * x_data) + noise


    # JAX-compatible model function
    def model(x, a, b):
        return a * jnp.exp(-b * x)


    # Fit with automatic memory management
    result = fit_large_dataset(
        model, x_data, y_data, p0=[2.0, 1.0], memory_limit_gb=4.0, show_progress=True
    )

    print(f"Fitted parameters: {result.popt}")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from nlsq import LargeDatasetFitter
    from nlsq.large_dataset import LDMemoryConfig

    # Custom memory configuration
    config = LDMemoryConfig(
        memory_limit_gb=8.0,
        min_chunk_size=5000,
        max_chunk_size=500000,
        enable_sampling=True,
        sampling_threshold=50_000_000,
    )

    fitter = LargeDatasetFitter(config=config)

    # Get processing recommendations
    recommendations = fitter.get_memory_recommendations(n_points, n_params)
    print(f"Strategy: {recommendations['processing_strategy']}")

    # Fit with progress reporting
    result = fitter.fit_with_progress(model, x_data, y_data, p0=[1.0, 1.0])

Performance Characteristics
----------------------------

Memory Efficiency
~~~~~~~~~~~~~~~~~

- **Automatic chunking** prevents memory overflow
- **Progressive processing** maintains constant memory footprint
- **Smart sampling** handles arbitrarily large datasets

Computational Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

- **JAX JIT compilation** provides GPU/TPU acceleration
- **Parallel chunking** potential for future enhancement
- **Optimized memory access** patterns reduce overhead

Scalability Testing Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

==================== =================== ============= ========== =================
Dataset Size         Processing Strategy Memory Usage  Fit Time   Parameter Error
==================== =================== ============= ========== =================
10K points           Single chunk        <100MB        0.1s       <0.1%
1M points            Single chunk        ~200MB        2-3s       <0.01%
10M points           Chunked (10)        <500MB        15-20s     <0.1%
50M points           Chunked (50)        <500MB        60-80s     <0.5%
100M+ points         Sampling            <1GB          5-10s      <1%
==================== =================== ============= ========== =================

Best Practices
--------------

1. **Always estimate memory first**:

   .. code-block:: python

       stats = estimate_memory_requirements(n_points, n_params)
       if stats.total_memory_estimate_gb > available_memory:
           use_large_dataset_fitter()

2. **Use appropriate chunk sizes**:

   .. code-block:: python

       # Chunk size affects performance
       # Too small: overhead from many iterations
       # Too large: memory issues
       optimal_chunk = int(available_memory_gb * 1e9 / (8 * 3 * n_params))

3. **Consider data precision**:

   .. code-block:: python

       # If data has limited precision, sampling may not affect accuracy
       if data_precision < 1e-3 and n_points > 100_000_000:
           enable_sampling()

Architecture Design
-------------------

Key Design Principles
~~~~~~~~~~~~~~~~~~~~~

1. **Memory Safety**: Never exceed configured memory limits
2. **Automatic Management**: Minimal user configuration required
3. **Graceful Degradation**: Falls back to sampling for extreme cases
4. **Progress Transparency**: Clear reporting for long operations
5. **API Compatibility**: Integrates seamlessly with existing NLSQ

Processing Flow
~~~~~~~~~~~~~~~

1. **Analysis Phase**: Estimate memory requirements and choose strategy
2. **Preparation Phase**: Configure chunking or sampling parameters
3. **Processing Phase**: Execute fitting with progress monitoring
4. **Aggregation Phase**: Combine results and compute final parameters
5. **Validation Phase**: Verify results and provide diagnostics

Compatibility and Requirements
------------------------------

Dependencies
~~~~~~~~~~~~

- **JAX**: For JIT compilation and GPU acceleration
- **NumPy**: For numerical computations and data handling
- **psutil**: For system memory monitoring
- **NLSQ core**: Existing curve fitting infrastructure

Platform Support
~~~~~~~~~~~~~~~~~

- **GPU/TPU**: Full JAX acceleration support
- **CPU**: Optimized for multi-core processing
- **Memory**: Automatic detection and management
- **Operating Systems**: Linux, macOS, Windows

Limitations
~~~~~~~~~~~

- **Model functions** must be JAX-compatible (use ``jax.numpy``)
- **Large parameter counts** (>50) may require custom configuration
- **Very small chunks** (<1000 points) may have reduced accuracy
- **Network storage** may impact performance for very large datasets

Conclusion
----------

The NLSQ Large Dataset implementation provides a comprehensive solution for curve fitting on datasets ranging from thousands to billions of points. The automatic memory management, intelligent processing strategies, and seamless integration with existing NLSQ infrastructure make it a powerful tool for scientific computing and data analysis applications requiring high performance and scalability.

The implementation follows NLSQ's design principles of simplicity, performance, and compatibility while extending capabilities to handle the demanding requirements of modern large-scale data analysis.
