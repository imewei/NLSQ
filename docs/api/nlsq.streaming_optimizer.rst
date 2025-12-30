nlsq.streaming\_optimizer module
==================================

.. currentmodule:: nlsq.streaming.optimizer

.. automodule:: nlsq.streaming.optimizer
   :noindex:

Overview
--------

The ``nlsq.streaming_optimizer`` module provides true streaming optimization for datasets
that don't fit in memory. Unlike batch processing, streaming optimization processes data
in an online fashion, enabling curve fitting on datasets of unlimited size including
real-time data streams, HDF5 files, memory-mapped arrays, and data generators.

**New in version 0.1.1**: Complete streaming optimization with SGD and Adam optimizers.

Key Features
------------

- **Unlimited dataset size** - Process datasets larger than available RAM
- **Multiple data sources** - HDF5, memory-mapped files, generators, streaming APIs
- **SGD and Adam optimizers** - Modern optimization algorithms for online learning
- **Adaptive learning rates** - Warmup and cosine annealing schedules
- **Gradient clipping** - Prevent exploding gradients
- **Checkpointing** - Save intermediate results during long optimizations
- **Progress monitoring** - Real-time statistics and convergence tracking
- **Batch processing** - Configurable batch sizes for efficiency

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   StreamingOptimizer
   DataGenerator

.. autoclass:: StreamingConfig
   :members:
   :noindex:

Functions
---------

.. autosummary::
   :toctree: generated/

   fit_unlimited_data
   create_hdf5_dataset

Usage Examples
--------------

Basic Streaming Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit a model to data stored in HDF5 file:

.. code-block:: python

    from nlsq.streaming.optimizer import fit_unlimited_data, StreamingConfig
    import numpy as np
    import jax.numpy as jnp


    # Define model function
    def exponential(x, a, b, c):
        return a * jnp.exp(-b * x) + c


    # Configure streaming
    config = StreamingConfig(
        batch_size=10000, max_epochs=10, use_adam=True, learning_rate=0.01
    )

    # Fit to HDF5 dataset (100M points)
    result = fit_unlimited_data(
        exponential,
        "huge_dataset.h5",  # HDF5 file with 'x' and 'y' datasets
        p0=[2.0, 0.5, 0.3],
        config=config,
        verbose=1,
    )

    print(f"Converged after {result['n_epochs']} epochs")
    print(f"Final parameters: {result['x']}")
    print(f"Final loss: {result['fun']:.6f}")

Streaming from Data Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process data generated on-the-fly:

.. code-block:: python

    from nlsq.streaming.optimizer import StreamingOptimizer, StreamingConfig
    import numpy as np
    import jax.numpy as jnp

    # Create streaming optimizer
    config = StreamingConfig(
        batch_size=5000, max_epochs=20, use_adam=True, convergence_tol=1e-7
    )
    optimizer = StreamingOptimizer(config)


    # Define data generator
    def data_generator():
        """Generate batches of data indefinitely."""
        while True:
            batch_size = 5000
            x = np.random.randn(batch_size) * 10
            # True model: y = 2*x + 1 + noise
            y = 2.0 * x + 1.0 + np.random.randn(batch_size) * 0.5
            yield x, y


    # Define model
    def linear(x, a, b):
        return a * x + b


    # Fit streaming data
    result = optimizer.fit_streaming(linear, data_generator(), p0=[1.0, 0.0], verbose=2)

    print(f"Estimated parameters: a={result['x'][0]:.3f}, b={result['x'][1]:.3f}")
    print(f"True parameters: a=2.000, b=1.000")

Memory-Mapped NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~

Process large NumPy arrays without loading into memory:

.. code-block:: python

    from nlsq.streaming.optimizer import fit_unlimited_data
    import numpy as np

    # Create memory-mapped file
    n_samples = 50_000_000  # 50M samples
    data = np.memmap("large_data.npy", dtype="float64", mode="w+", shape=(n_samples, 2))

    # Generate data in chunks
    chunk_size = 1_000_000
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        x_chunk = np.random.randn(end - i)
        y_chunk = 3.0 * x_chunk**2 - 1.5 * x_chunk + 2.0
        data[i:end, 0] = x_chunk
        data[i:end, 1] = y_chunk
    data.flush()


    # Fit using streaming
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c


    result = fit_unlimited_data(
        quadratic,
        "large_data.npy",
        p0=[1.0, -1.0, 1.0],
        config=StreamingConfig(batch_size=50000, max_epochs=5),
    )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

Fine-tune optimizer settings:

.. code-block:: python

    from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer

    # Custom configuration
    config = StreamingConfig(
        batch_size=20000,  # Larger batches for stability
        learning_rate=0.005,  # Conservative learning rate
        momentum=0.95,  # High momentum for SGD
        max_epochs=50,  # More epochs for convergence
        convergence_tol=1e-8,  # Tight convergence criterion
        checkpoint_interval=50,  # Checkpoint every 50 batches
        use_adam=True,  # Use Adam optimizer
        adam_beta1=0.9,  # Adam momentum
        adam_beta2=0.999,  # Adam RMSprop
        adam_eps=1e-8,  # Numerical stability
        gradient_clip=5.0,  # Clip gradients at 5.0
        warmup_steps=1000,  # 1000 warmup steps
    )

    optimizer = StreamingOptimizer(config)

    result = optimizer.fit_streaming(
        model_func,
        data_source,
        p0=initial_params,
        bounds=(lower_bounds, upper_bounds),  # Optional parameter bounds
        verbose=2,
    )

Creating HDF5 Datasets
~~~~~~~~~~~~~~~~~~~~~~

Generate test datasets in HDF5 format:

.. code-block:: python

    from nlsq.streaming.optimizer import create_hdf5_dataset
    import numpy as np
    import jax.numpy as jnp


    # Define true model
    def true_model(x, a, b):
        return a * jnp.exp(-b * x)


    true_params = [2.5, 0.5]

    # Create HDF5 dataset with 10M samples
    create_hdf5_dataset(
        filename="test_dataset.h5",
        func=true_model,
        params=true_params,
        n_samples=10_000_000,
        chunk_size=10000,
        noise_level=0.1,
    )

    # Now fit to the dataset
    result = fit_unlimited_data(
        lambda x, a, b: a * jnp.exp(-b * x),
        "test_dataset.h5",
        p0=[2.0, 0.4],
        config=StreamingConfig(batch_size=10000, max_epochs=10),
    )

    print(f"Recovered: a={result['x'][0]:.3f}, b={result['x'][1]:.3f}")
    print(f"True: a={true_params[0]:.3f}, b={true_params[1]:.3f}")

Progress Monitoring with Callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor optimization progress:

.. code-block:: python

    from nlsq.streaming.optimizer import StreamingOptimizer, StreamingConfig

    # Track progress
    loss_history = []


    def progress_callback(iteration, params, loss):
        """Called after each batch."""
        loss_history.append(loss)
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: loss={loss:.6f}")


    config = StreamingConfig(batch_size=5000, max_epochs=10)
    optimizer = StreamingOptimizer(config)

    result = optimizer.fit_streaming(
        model_func, data_source, p0=initial_guess, callback=progress_callback, verbose=1
    )

    # Plot loss history
    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Streaming Optimization Progress")
    plt.show()

Bounded Optimization
~~~~~~~~~~~~~~~~~~~~

Apply parameter bounds during streaming:

.. code-block:: python

    from nlsq.streaming.optimizer import fit_unlimited_data, StreamingConfig
    import numpy as np
    import jax.numpy as jnp


    # Constrained exponential decay
    def exponential(x, a, tau):
        return a * jnp.exp(-x / tau)


    # Bounds: a in [0, 10], tau in [0.1, 100]
    lower_bounds = np.array([0.0, 0.1])
    upper_bounds = np.array([10.0, 100.0])

    result = fit_unlimited_data(
        exponential,
        "data.h5",
        p0=[5.0, 10.0],
        bounds=(lower_bounds, upper_bounds),
        config=StreamingConfig(batch_size=10000),
    )

Optimizer State Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reset and reuse optimizer:

.. code-block:: python

    from nlsq.streaming.optimizer import StreamingOptimizer, StreamingConfig

    config = StreamingConfig(batch_size=10000, use_adam=True)
    optimizer = StreamingOptimizer(config)

    # First optimization
    result1 = optimizer.fit_streaming(func1, data1, p0=init1)

    # Reset state for new optimization
    optimizer.reset_state()

    # Second optimization (fresh start)
    result2 = optimizer.fit_streaming(func2, data2, p0=init2)

Comparison: SGD vs Adam
~~~~~~~~~~~~~~~~~~~~~~~~

Compare optimization algorithms:

.. code-block:: python

    from nlsq.streaming.optimizer import StreamingOptimizer, StreamingConfig

    # SGD with momentum
    sgd_config = StreamingConfig(
        use_adam=False, learning_rate=0.01, momentum=0.9, batch_size=5000
    )
    sgd_optimizer = StreamingOptimizer(sgd_config)
    sgd_result = sgd_optimizer.fit_streaming(model, data, p0=p0)

    # Adam optimizer
    adam_config = StreamingConfig(
        use_adam=True, learning_rate=0.01, adam_beta1=0.9, adam_beta2=0.999, batch_size=5000
    )
    adam_optimizer = StreamingOptimizer(adam_config)
    adam_result = adam_optimizer.fit_streaming(model, data, p0=p0)

    print(f"SGD final loss: {sgd_result['fun']:.6f}")
    print(f"Adam final loss: {adam_result['fun']:.6f}")

Performance Characteristics
---------------------------

**Throughput**:

- Typical: 50,000 - 200,000 samples/second (CPU)
- GPU: 500,000 - 2,000,000 samples/second
- Depends on model complexity and batch size

**Memory usage**:

- Fixed memory footprint regardless of dataset size
- Memory ~ batch_size × n_params × 8 bytes
- Example: 10,000 batch × 10 params = ~800 KB

**Convergence**:

- Online methods converge slower than batch methods
- Typical epochs needed: 5-50 depending on problem
- Use larger batch sizes for faster convergence
- Adam generally converges faster than SGD

**Batch size recommendations**:

- Small models (<10 params): 5,000 - 20,000
- Medium models (10-100 params): 10,000 - 50,000
- Large models (>100 params): 20,000 - 100,000

Optimization Tips
-----------------

1. **Batch size**: Larger batches = more stable but slower per epoch
2. **Learning rate**: Start with 0.01, reduce if unstable
3. **Warmup**: Use warmup for first 1000-5000 steps
4. **Gradient clipping**: Set to 5-10 to prevent exploding gradients
5. **Adam vs SGD**: Adam generally better, but SGD with momentum can work
6. **Epochs**: Monitor convergence, stop when loss plateaus
7. **Checkpointing**: Enable for long optimizations (>1 hour)

Data Source Support
-------------------

The ``DataGenerator`` class supports multiple data sources:

**HDF5 Files** (``*.h5``, ``*.hdf5``):

.. code-block:: python

    # File must contain 'x' and 'y' datasets
    optimizer.fit_streaming(model, "data.h5", p0=p0)

**Memory-mapped NumPy** (``*.npy``, ``*.npz``):

.. code-block:: python

    # Single .npy file or .npz with 'x' and 'y' keys
    optimizer.fit_streaming(model, "data.npz", p0=p0)

**Python Generators**:

.. code-block:: python

    def data_gen():
        while True:
            yield x_batch, y_batch


    optimizer.fit_streaming(model, data_gen(), p0=p0)

**In-memory Arrays** (for testing):

.. code-block:: python

    # Tuple of (x, y) arrays
    optimizer.fit_streaming(model, (x_array, y_array), p0=p0)

Limitations
-----------

- **No covariance estimation**: Streaming methods don't compute parameter uncertainties
- **Convergence guarantees**: Not as strong as batch methods
- **Initial guess sensitivity**: Online methods more sensitive to initialization
- **Loss function**: Currently supports L2 loss only
- **Jacobian**: Uses finite differences (autodiff support planned)

When to Use Streaming
----------------------

**Use streaming when**:

- Dataset doesn't fit in memory (>available RAM)
- Data arrives in real-time or from streaming API
- Need to process data from disk without loading
- Memory is constrained (embedded systems)

**Use batch methods when**:

- Dataset fits in memory
- Need parameter uncertainties (covariance)
- Require guaranteed convergence
- Have good initial parameter estimates

See Also
--------

- :doc:`nlsq.large_dataset` : Chunked batch processing for large datasets
- :doc:`nlsq.memory_manager` : Memory management utilities
- :doc:`../howto/handle_large_data` : Large dataset guide
- :doc:`../howto/advanced_api` : Advanced optimization features
