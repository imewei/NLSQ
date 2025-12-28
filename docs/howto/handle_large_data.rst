Large Dataset Tutorial
======================

Learn how to efficiently handle datasets with millions to billions of points using NLSQ's advanced large dataset features.

Learning Objectives
-------------------

After completing this tutorial, you will:

- Understand when and how to use ``curve_fit_large``
- Know how to estimate memory requirements before fitting
- Be able to configure chunking and streaming strategies
- Understand sparse Jacobian optimization
- Master streaming optimization for unlimited-size data ()

Introduction to Large Dataset Challenges
-----------------------------------------

Traditional curve fitting algorithms face several challenges with large datasets:

- **Memory limitations**: Cannot load entire dataset into RAM
- **Computational complexity**: O(n²) or O(n³) scaling with data size
- **Numerical stability**: Large matrices can become ill-conditioned
- **Processing time**: Sequential algorithms don't utilize modern hardware efficiently

NLSQ addresses these challenges through:

- Automatic chunking and memory management
- Streaming data processing with zero data loss ()
- GPU/TPU acceleration via JAX
- Sparse matrix optimizations
- Mini-batch gradient descent for unlimited datasets

Automatic Large Dataset Detection
---------------------------------

The simplest approach is using ``curve_fit_large``, which automatically detects dataset size and chooses appropriate strategies:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from nlsq import curve_fit_large, estimate_memory_requirements

    # First, let's estimate memory requirements
    n_points = 10_000_000  # 10 million points
    n_params = 3
    stats = estimate_memory_requirements(n_points, n_params)

    print(f"Dataset: {n_points:,} points, {n_params} parameters")
    print(f"Estimated memory: {stats.total_memory_estimate_gb:.2f} GB")
    print(f"Recommended chunks: {stats.n_chunks}")
    print(f"Chunk size: {stats.recommended_chunk_size:,}")
    print(f"Processing strategy: {stats.processing_strategy}")

**Example Output:**

.. code-block::

    Dataset: 10,000,000 points, 3 parameters
    Estimated memory: 1.34 GB
    Recommended chunks: 4
    Chunk size: 2,500,000
    Processing strategy: chunked

Now let's generate and fit this large dataset:

.. code-block:: python

    # Generate large synthetic dataset
    print("Generating data...")
    x = np.linspace(0, 5, n_points)

    # True parameters
    true_params = [2.5, 0.8, 0.3]

    # Add realistic noise
    y_true = true_params[0] * np.exp(-true_params[1] * x) + true_params[2]
    noise = np.random.normal(0, 0.05, n_points)
    y = y_true + noise

    print(f"Data size: x={x.shape}, y={y.shape}")
    print(f"Memory usage: ~{(x.nbytes + y.nbytes) / 1e6:.1f} MB")


    # Define model function using JAX
    def exponential_model(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Fit using automatic large dataset handling
    print("Starting fit...")
    popt, pcov = curve_fit_large(
        exponential_model,
        x,
        y,
        p0=[2.0, 0.5, 0.2],
        memory_limit_gb=2.0,  # Limit memory usage
        show_progress=True,  # Show progress bar
    )

    # Display results
    param_errors = np.sqrt(np.diag(pcov))
    print("\nFitting Results:")
    print("=" * 40)
    param_names = ["Amplitude (a)", "Decay rate (b)", "Offset (c)"]

    for name, true_val, fit_val, error in zip(param_names, true_params, popt, param_errors):
        percent_error = 100 * abs(fit_val - true_val) / true_val
        print(f"{name}: {fit_val:.6f} ± {error:.6f}")
        print(f"  True value: {true_val}")
        print(f"  Error: {percent_error:.3f}%")
        print()

Manual Configuration with LargeDatasetFitter
---------------------------------------------

For more control over the fitting process, use the ``LargeDatasetFitter`` class:

.. code-block:: python

    from nlsq import LargeDatasetFitter
    from nlsq.streaming.large_dataset import LDMemoryConfig

    # Create custom configuration
    config = LDMemoryConfig(
        memory_limit_gb=4.0,  # Maximum memory usage
        min_chunk_size=50000,  # Minimum points per chunk
        max_chunk_size=2000000,  # Maximum points per chunk
        # : Streaming optimization automatically handles very large datasets
        use_streaming=True,  # Enable streaming for unlimited data
        streaming_batch_size=100000,  # Mini-batch size for streaming
    )

    # Create fitter with custom configuration
    fitter = LargeDatasetFitter(config=config)

    # Get processing recommendations
    recommendations = fitter.get_memory_recommendations(n_points, n_params)

    print("Processing Strategy Recommendations:")
    print(f"  Strategy: {recommendations['processing_strategy']}")
    print(f"  Memory estimate: {recommendations['memory_estimate_gb']:.2f} GB")
    print(f"  Recommended chunks: {recommendations['n_chunks']}")
    print(f"  Chunk size: {recommendations['chunk_size']:,}")

    # Perform fit with detailed progress reporting
    result = fitter.fit_with_progress(
        exponential_model,
        x,
        y,
        p0=[2.0, 0.5, 0.2],
    )

    # Examine detailed results
    print(f"\nDetailed Results:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    # Note: n_chunks only available for multi-chunk fits
    print(f"  Fitted parameters: {result.popt}")
    print(f"  Total function evaluations: {result.nfev}")

Streaming Optimization for Unlimited Datasets
--------------------------------------------------------

For datasets too large to fit in memory, NLSQ uses streaming optimization with mini-batch gradient descent.
**Unlike subsampling (deprecated), streaming processes 100% of data with zero accuracy loss.**

.. code-block:: python

    from nlsq import StreamingOptimizer, StreamingConfig

    # Simulate billion-point dataset (or load from HDF5)
    n_huge = 1_000_000_000  # 1 billion points

    # Check memory requirements
    huge_stats = estimate_memory_requirements(n_huge, 3)
    print(f"Billion-point dataset:")
    print(f"  Memory estimate: {huge_stats.total_memory_estimate_gb:.1f} GB")
    print(f"  Processing strategy: streaming (processes ALL data)")

    # For demonstration, we'll use a smaller dataset
    n_demo = 5_000_000  # 5 million points
    x_demo = np.linspace(0, 10, n_demo)
    y_demo = 3.2 * np.exp(-0.4 * x_demo) + 0.8 + np.random.normal(0, 0.1, n_demo)

    # Configure streaming optimization
    streaming_config = StreamingConfig(
        batch_size=50000,  # Mini-batch size
        max_epochs=10,  # Number of passes through data
        learning_rate=0.01,  # Adaptive learning rate
    )

    # Use LargeDatasetFitter with streaming enabled
    config = LDMemoryConfig(
        memory_limit_gb=2.0,
        use_streaming=True,  # Enable streaming for very large datasets
        streaming_batch_size=50000,
    )

    fitter = LargeDatasetFitter(config=config)

    print(f"\nFitting {n_demo:,} points with streaming (100% of data)...")
    stream_result = fitter.fit_with_progress(
        exponential_model,
        x_demo,
        y_demo,
        p0=[3.0, 0.3, 0.5],
    )

    print(f"Streaming fit parameters: {stream_result.popt}")
    print(f"Points processed: {n_demo:,} (ALL data, no loss)")
    print(f"Convergence: {stream_result.success}")

Sparse Jacobian Optimization
-----------------------------

Many large-scale problems have sparse Jacobian structures. NLSQ can detect and exploit this:

.. code-block:: python

    from nlsq import SparseJacobianComputer


    # Create a problem with sparse structure
    # Example: Multiple independent exponential components
    def multi_exponential(x, *params):
        """Sum of multiple independent exponential decays."""
        n_components = len(params) // 3  # Each component has 3 parameters
        result = jnp.zeros_like(x)

        for i in range(n_components):
            a = params[3 * i]  # amplitude
            b = params[3 * i + 1]  # decay rate
            c = params[3 * i + 2]  # offset
            result += a * jnp.exp(-b * x) + c

        return result


    # Generate data with 5 components (15 parameters total)
    n_components = 5
    n_points_sparse = 100000
    x_sparse = np.linspace(0, 3, n_points_sparse)

    # True parameters for 5 components
    true_sparse_params = []
    for i in range(n_components):
        true_sparse_params.extend(
            [2.0 + 0.5 * i, 0.5 + 0.2 * i, 0.1 * i]  # amplitude  # decay rate  # offset
        )

    y_sparse = multi_exponential(x_sparse, *true_sparse_params)
    y_sparse += 0.02 * np.random.normal(size=len(x_sparse))

    # Detect sparsity
    sparse_computer = SparseJacobianComputer(sparsity_threshold=0.1)

    # Use a sample to detect sparsity pattern
    sample_size = min(1000, len(x_sparse))
    sample_indices = np.random.choice(len(x_sparse), sample_size, replace=False)
    x_sample = x_sparse[sample_indices]

    p0_sparse = [1.8 + 0.4 * i for i in range(n_components * 3)]  # Initial guess

    # Detect sparsity pattern
    pattern, sparsity = sparse_computer.detect_sparsity_pattern(
        multi_exponential, p0_sparse, x_sample
    )

    print(f"Jacobian Analysis:")
    print(f"  Matrix size: {pattern.shape}")
    print(f"  Sparsity ratio: {sparsity:.1%}")
    print(f"  Is sparse: {sparsity > 0.1}")

    if sparsity > 0.1:  # If more than 10% sparse
        print("  -> Using sparse optimization algorithms")
    else:
        print("  -> Using dense optimization algorithms")

Streaming Optimization for Unlimited Data
------------------------------------------

For datasets that cannot fit in memory or are generated on-the-fly, use streaming optimization:

.. code-block:: python

    from nlsq import StreamingOptimizer, StreamingConfig
    from nlsq.streaming.optimizer import create_hdf5_dataset


    # First, create a large HDF5 dataset on disk
    def create_large_dataset():
        print("Creating large HDF5 dataset...")
        create_hdf5_dataset(
            "large_dataset.h5",
            exponential_model,
            [2.8, 0.6, 0.4],  # True parameters
            n_samples=50_000_000,  # 50 million points
            chunk_size=10000,  # HDF5 chunk size
            noise_level=0.08,
        )
        print("Dataset created: large_dataset.h5")


    # Create the dataset (this may take a few minutes)
    create_large_dataset()

    # Configure streaming optimizer
    streaming_config = StreamingConfig(
        batch_size=50000,  # Points per batch
        max_epochs=20,  # Maximum training epochs
        convergence_tol=1e-6,  # Convergence tolerance
        use_adam=True,  # Use Adam optimizer
        learning_rate=0.001,  # Initial learning rate
    )

    # Create streaming optimizer
    stream_optimizer = StreamingOptimizer(streaming_config)

    # Fit directly from HDF5 file (file path passed as data source)
    print("Starting streaming optimization...")
    stream_result = stream_optimizer.fit_streaming(
        exponential_model, "large_dataset.h5", p0=np.array([2.5, 0.4, 0.3])
    )

    print("Streaming Results:")
    print(f"  Converged: {stream_result['success']}")
    print(f"  Final parameters: {stream_result['x']}")
    print(f"  Epochs used: {stream_result['nit']}")
    print(f"  Final loss: {stream_result['fun']:.6f}")


    # Alternative: Stream from custom generator
    def data_generator(batch_size=10000):
        """Generate data batches on-the-fly."""
        while True:
            x_batch = np.random.uniform(0, 5, batch_size)
            x_batch.sort()  # Keep x values sorted
            y_batch = exponential_model(x_batch, 2.8, 0.6, 0.4)
            y_batch += 0.08 * np.random.normal(size=batch_size)
            yield x_batch, y_batch


    # Fit using generator (data stream)
    print("Fitting from data generator...")
    gen_result = stream_optimizer.fit_streaming(
        exponential_model, data_generator(batch_size=20000), p0=np.array([2.5, 0.4, 0.3])
    )

    print("Generator Results:")
    print(f"  Parameters: {gen_result['x']}")
    print(f"  Samples processed: {gen_result['total_samples']:,}")

Performance Comparison
----------------------

Let's compare different strategies for the same large dataset:

.. code-block:: python

    import time

    # Test dataset
    n_test = 2_000_000  # 2 million points
    x_test = np.linspace(0, 4, n_test)
    y_test = 1.8 * np.exp(-0.7 * x_test) + 0.2 + np.random.normal(0, 0.03, n_test)

    strategies = [
        (
            "Standard curve_fit_large",
            lambda: curve_fit_large(exponential_model, x_test, y_test, p0=[1.5, 0.5, 0.1]),
        ),
        (
            "Chunked (4 chunks)",
            lambda: curve_fit_large(
                exponential_model,
                x_test,
                y_test,
                p0=[1.5, 0.5, 0.1],
                memory_limit_gb=0.5,  # Force chunking
            ),
        ),
        (
            "Streaming ()",
            lambda: curve_fit_large(
                exponential_model,
                x_test,
                y_test,
                p0=[1.5, 0.5, 0.1],
                memory_limit_gb=2.0,  # Use streaming for very large data
                # Streaming automatically handles unlimited datasets
            ),
        ),
    ]

    results = {}

    print(f"Performance Comparison ({n_test:,} points)")
    print("=" * 60)

    for name, strategy in strategies:
        print(f"\nTesting: {name}")
        start_time = time.time()

        try:
            popt, pcov = strategy()
            duration = time.time() - start_time
            error = np.sqrt(np.mean((y_test - exponential_model(x_test, *popt)) ** 2))

            results[name] = {
                "time": duration,
                "params": popt,
                "rms_error": error,
                "success": True,
            }

            print(f"  Time: {duration:.2f} seconds")
            print(f"  Parameters: [{popt[0]:.3f}, {popt[1]:.3f}, {popt[2]:.3f}]")
            print(f"  RMS Error: {error:.5f}")

        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {"success": False, "error": str(e)}

    # Summary
    print("\nSummary:")
    print("-" * 40)
    successful_results = {k: v for k, v in results.items() if v.get("success", False)}

    if successful_results:
        fastest = min(successful_results, key=lambda k: successful_results[k]["time"])
        most_accurate = min(
            successful_results, key=lambda k: successful_results[k]["rms_error"]
        )

        print(f"Fastest: {fastest} ({successful_results[fastest]['time']:.2f}s)")
        print(
            f"Most accurate: {most_accurate} (RMS: {successful_results[most_accurate]['rms_error']:.6f})"
        )

Best Practices for Large Datasets
----------------------------------

**1. Estimate Memory First**

Always check memory requirements before fitting:

.. code-block:: python

    # Check before processing
    stats = estimate_memory_requirements(len(x), n_parameters)
    if stats.total_memory_estimate_gb > available_memory_gb:
        print("Consider using chunking or streaming optimization")

**2. Choose Appropriate Strategies**

- **< 1M points**: Use standard ``curve_fit``
- **1M - 10M points**: Use ``curve_fit_large`` with default settings
- **10M - 100M points**: Use chunking with progress monitoring
- **> 100M points**: Use streaming optimization (, processes 100% of data)

**3. Optimize for Your Hardware**

.. code-block:: python

    # Check available devices
    import jax

    print(f"Available devices: {jax.devices()}")

    # GPU memory is typically more limited
    if jax.devices()[0].device_kind == "gpu":
        memory_limit_gb = 2.0  # Conservative for GPU
    else:
        memory_limit_gb = 8.0  # More generous for CPU

**4. Monitor Progress for Long Fits**

.. code-block:: python

    # Always use progress bars for large datasets
    popt, pcov = curve_fit_large(
        func, x, y, show_progress=True, memory_limit_gb=4.0  # Essential for user experience
    )

**5. Validate Results**

.. code-block:: python

    # Check residuals and parameter uncertainties
    residuals = y - func(x, *popt)
    rms_residual = np.sqrt(np.mean(residuals**2))
    param_errors = np.sqrt(np.diag(pcov))

    print(f"RMS residual: {rms_residual:.6f}")
    print(f"Max parameter uncertainty: {np.max(param_errors / np.abs(popt)):.2%}")

Troubleshooting Large Dataset Issues
-------------------------------------

**Memory Errors**

.. code-block:: python

    # Use chunking or streaming for large datasets
    try:
        popt, pcov = curve_fit_large(func, x, y)
    except MemoryError:
        print("Using streaming optimization to handle unlimited data...")
        # : Streaming processes 100% of data with zero accuracy loss
        popt, pcov = curve_fit_large(func, x, y, memory_limit_gb=2.0, chunk_size=100000)

**Convergence Issues**

.. code-block:: python

    # Try different initial guesses or increase tolerances
    popt, pcov = curve_fit_large(
        func, x, y, p0=better_initial_guess, ftol=1e-6, xtol=1e-6  # Looser tolerance
    )

**Performance Issues**

.. code-block:: python

    # Profile your function for JAX compatibility
    import jax

    # Test function compilation
    compiled_func = jax.jit(func)
    test_result = compiled_func(x[:100], *p0)  # Should not raise errors

Interactive Notebooks
---------------------

Hands-on tutorials for large dataset handling:

**Core Tutorials:**

- `Large Dataset Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/02_core_tutorials/large_dataset_demo.ipynb>`_ - Handle millions of data points with automatic chunking

**Streaming and Fault Tolerance:**

- `Basic Fault Tolerance <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/06_streaming/01_basic_fault_tolerance.ipynb>`_ - Error handling for streaming fits
- `Checkpoint and Resume <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/06_streaming/02_checkpoint_resume.ipynb>`_ - Save and restore optimization state
- `Custom Retry Settings <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/06_streaming/03_custom_retry_settings.ipynb>`_ - Configure retry behavior
- `Interpreting Diagnostics <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/06_streaming/04_interpreting_diagnostics.ipynb>`_ - Understanding streaming results
- `Hybrid Streaming API <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/06_streaming/05_hybrid_streaming_api.ipynb>`_ - Parameter normalization and Adam warmup

**HPC and Cluster Computing:**

- `HPC and Checkpointing <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/07_hpc_and_checkpointing.ipynb>`_ - PBS Pro, fault tolerance, and cluster computing

Next Steps
----------

Congratulations! You now have the tools to handle datasets of any size. Continue with:

1. :doc:`../api/large_datasets_api` - Advanced fitting APIs and parameter constraints
2. :doc:`../api/performance_benchmarks` - Performance analysis and optimization
3. Browse the `examples directory <https://github.com/imewei/NLSQ/tree/main/examples>`_ for more complex scenarios

Further Reading
---------------

- :doc:`../api/nlsq.large_dataset` - Comprehensive technical details
- :doc:`../api/large_datasets_api` - Complete function documentation
- `JAX Documentation <https://docs.jax.dev/en/latest/>`_ - Understanding JAX transformations
