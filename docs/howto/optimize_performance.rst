Performance Optimization Guide
==============================

This guide provides practical strategies for maximizing NLSQ’s
performance through GPU/TPU acceleration, JIT compilation, batch
processing, and memory optimization.

Table of Contents
-----------------

1. `Quick Performance Wins <#quick-performance-wins>`__
2. `GPU/TPU Acceleration <#gputpu-acceleration>`__
3. `JIT Compilation Optimization <#jit-compilation-optimization>`__
4. `Batch Processing <#batch-processing>`__
5. `Memory Optimization <#memory-optimization>`__
6. `Algorithm and Solver Selection <#algorithm-and-solver-selection>`__
7. `Profiling and Benchmarking <#profiling-and-benchmarking>`__
8. `Performance Troubleshooting <#performance-troubleshooting>`__

--------------

Quick Performance Wins
----------------------

Top 5 Performance Tips
~~~~~~~~~~~~~~~~~~~~~~

1. **Use CurveFit class for multiple fits** (reuses JIT compilation)
2. **Enable GPU acceleration** (automatic, 10-270x faster)
3. **Batch similar fits together** (amortize JIT overhead)
4. **Use appropriate solver** (``solver='auto'`` recommended)
5. **Set memory limits for large datasets** (``memory_limit_gb``
   parameter)

Quick Comparison: SciPy vs NLSQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import numpy as np
   import time
   from scipy.optimize import curve_fit as scipy_curve_fit
   from nlsq import curve_fit as nlsq_curve_fit
   import jax.numpy as jnp


   def exponential(x, a, b):
       return a * jnp.exp(-b * x)


   # Generate large dataset
   x = np.linspace(0, 10, 1_000_000)
   y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.randn(1_000_000)

   # SciPy (CPU-only)
   start = time.time()
   popt_scipy, _ = scipy_curve_fit(lambda x, a, b: a * np.exp(-b * x), x, y, p0=[2, 1])
   scipy_time = time.time() - start

   # NLSQ (GPU-accelerated)
   start = time.time()
   popt_nlsq, _ = nlsq_curve_fit(exponential, x, y, p0=[2, 1])
   nlsq_time = time.time() - start

   print(f"SciPy time: {scipy_time:.2f}s")
   print(f"NLSQ time: {nlsq_time:.2f}s")
   print(f"Speedup: {scipy_time / nlsq_time:.1f}x")

**Expected output (GPU):**

::

   SciPy time: 42.5s
   NLSQ time: 0.15s
   Speedup: 283x

--------------

GPU/TPU Acceleration
--------------------

Automatic GPU Detection
~~~~~~~~~~~~~~~~~~~~~~~

NLSQ automatically detects and uses available accelerators:

.. code:: python

   import jax

   # Check available devices
   print(f"Available devices: {jax.devices()}")
   print(f"Default backend: {jax.default_backend()}")

**Output examples:**

::

   # GPU available:
   Available devices: [GpuDevice(id=0, process_index=0)]
   Default backend: gpu

   # CPU only:
   Available devices: [CpuDevice(id=0)]
   Default backend: cpu

Manual Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~

Override automatic detection:

.. code:: bash

   # Force CPU-only (useful for debugging)
   JAX_PLATFORM_NAME=cpu python your_script.py

   # Select specific GPU
   CUDA_VISIBLE_DEVICES=1 python your_script.py

   # Use TPU
   JAX_PLATFORM_NAME=tpu python your_script.py

In code:

.. code:: python

   import os

   os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Must be set before importing jax

   from nlsq import curve_fit

   # Now runs on CPU

GPU Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Pre-allocate GPU memory (prevents fragmentation)
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

   # Disable memory preallocation for multi-process scenarios
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

   # Set memory fraction (0.0-1.0)
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"  # Use 75% of GPU memory

   # Enable TF32 on Ampere GPUs (faster, slight precision loss)
   os.environ["XLA_FLAGS"] = "--xla_gpu_enable_fast_math=true"

When to Use GPU vs CPU
~~~~~~~~~~~~~~~~~~~~~~

=============== ========== ============== =======================
Dataset Size    Parameters Recommendation Expected Speedup
=============== ========== ============== =======================
< 1,000 points  Any        **CPU**        0.1-0.5x (JIT overhead)
1K-10K points   < 5        CPU or GPU     1-5x
10K-100K points Any        **GPU**        10-50x
100K-1M points  Any        **GPU**        50-150x
> 1M points     Any        **GPU**        150-300x
=============== ========== ============== =======================

--------------

JIT Compilation Optimization
----------------------------

Understanding JIT Overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~

First call includes compilation time:

.. code:: python

   import time
   from nlsq import curve_fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   x = np.linspace(0, 5, 10000)
   y = model(x, 2.5, 1.3, 0.5) + 0.01 * np.random.randn(10000)

   # First call: JIT compilation + execution
   start = time.time()
   popt1, _ = curve_fit(model, x, y, p0=[2, 1, 0])
   first_time = time.time() - start

   # Second call: execution only (cached)
   start = time.time()
   popt2, _ = curve_fit(model, x, y, p0=[2, 1, 0])
   second_time = time.time() - start

   print(f"First call (with JIT): {first_time:.3f}s")
   print(f"Second call (cached): {second_time:.3f}s")
   print(f"JIT overhead: {(first_time - second_time):.3f}s")

**Output:**

::

   First call (with JIT): 0.487s
   Second call (cached): 0.035s
   JIT overhead: 0.452s

Reusing Compiled Functions: CurveFit Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multiple fits with the same model, use ``CurveFit`` class:

.. code:: python

   from nlsq import CurveFit

   # Create CurveFit instance (compiles once)
   fitter = CurveFit()

   # Fit multiple datasets (no recompilation)
   datasets = [...]  # List of (x, y) pairs

   results = []
   for x_data, y_data in datasets:
       popt, pcov = fitter.curve_fit(model, x_data, y_data, p0=[2, 1, 0])
       results.append(popt)

   # 10x faster for 10 datasets vs calling curve_fit() 10 times

Pre-compilation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

Warm up JIT compilation before production use:

.. code:: python

   # Dummy fit to trigger compilation
   x_dummy = np.linspace(0, 1, 100)
   y_dummy = model(x_dummy, 2, 1, 0)
   _ = fitter.curve_fit(model, x_dummy, y_dummy, p0=[2, 1, 0])

   # Now production fits are fast
   for x, y in production_data:
       popt, pcov = fitter.curve_fit(model, x, y, p0=[2, 1, 0])

--------------

Batch Processing
----------------

Vectorized Batch Fitting
~~~~~~~~~~~~~~~~~~~~~~~~

Fit multiple datasets simultaneously:

.. code:: python

   from nlsq import curve_fit_batch
   import jax

   # Generate 100 datasets
   n_datasets = 100
   x = np.linspace(0, 5, 1000)
   y_batch = np.zeros((n_datasets, 1000))

   for i in range(n_datasets):
       true_params = [2 + 0.5 * i / n_datasets, 1.3, 0.5]
       y_batch[i] = model(x, *true_params) + 0.01 * np.random.randn(1000)

   # Vectorized batch fitting (uses vmap internally)
   popt_batch, pcov_batch = curve_fit_batch(
       model,
       x,  # Same x for all datasets
       y_batch,  # Shape: (n_datasets, n_points)
       p0=[2, 1, 0],
   )

   print(f"Fitted {n_datasets} datasets")
   print(f"Result shape: {popt_batch.shape}")  # (100, 3)

**Performance:** 50-100x faster than sequential fitting.

Parallel Fitting with Different X-data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets with different x-values:

.. code:: python

   from jax import pmap, vmap
   import jax.numpy as jnp

   # Split datasets across GPUs
   n_gpus = jax.device_count()
   datasets_per_gpu = len(datasets) // n_gpus


   @pmap
   def fit_batch_on_gpu(x_batch, y_batch):
       """Fit batch of datasets on one GPU"""
       return vmap(lambda x, y: curve_fit(model, x, y, p0=[2, 1, 0]))(x_batch, y_batch)


   # Reshape data for multi-GPU: (n_gpus, datasets_per_gpu, ...)
   x_reshaped = ...
   y_reshaped = ...

   results = fit_batch_on_gpu(x_reshaped, y_reshaped)

--------------

Memory Optimization
-------------------

Large Dataset Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq.streaming.large_dataset import fit_large_dataset

   # 50 million points
   x_huge = np.linspace(0, 100, 50_000_000)
   y_huge = model(x_huge, 2.5, 1.3, 0.5) + 0.01 * np.random.randn(50_000_000)

   # Automatic memory management
   popt, pcov, info = fit_large_dataset(
       f=model,
       xdata=x_huge,
       ydata=y_huge,
       p0=[2, 1, 0],
       memory_limit_gb=4.0,  # Limit to 4 GB
       chunk_size="auto",  # Automatic chunk sizing
       solver="cg",  # Memory-efficient solver
       progress=True,
   )

   print(f"Peak memory: {info['peak_memory_gb']:.2f} GB")
   print(f"Processed in {info['n_chunks']} chunks")

Memory Profiling
~~~~~~~~~~~~~~~~

.. code:: python

   from nlsq.caching.memory_manager import MemoryProfiler

   profiler = MemoryProfiler()

   with profiler.profile():
       popt, pcov = curve_fit(model, x, y, p0=[2, 1, 0])

   print(profiler.summary())

**Output:**

::

   Memory Profile:
   ├─ Peak usage: 2.34 GB
   ├─ Average usage: 1.87 GB
   ├─ Allocation events: 12
   ├─ Largest allocation: 1.2 GB (Jacobian)
   └─ Time in GC: 0.02s (0.5%)

Reducing Memory Footprint
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Use float32 instead of float64 (2x memory reduction)
   # WARNING: May reduce accuracy
   from jax import config

   config.update("jax_enable_x64", False)  # Use float32

   popt, pcov = curve_fit(model, x, y, p0=[2, 1, 0])

   # Re-enable float64 for production
   config.update("jax_enable_x64", True)

--------------

Algorithm and Solver Selection
------------------------------

Solver Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   import time

   solvers = ["svd", "cg", "lsqr", "auto"]
   results = {}

   for solver in solvers:
       start = time.time()
       popt, pcov = curve_fit(model, x, y, p0=[2, 1, 0], solver=solver)
       elapsed = time.time() - start
       results[solver] = {"time": elapsed, "popt": popt}

   # Print comparison
   for solver, data in results.items():
       print(f"{solver:8s}: {data['time']:.4f}s")

**Typical results (10K points, 3 params):**

::

   svd     : 0.0234s  (best for small problems)
   cg      : 0.0456s  (better for large problems)
   lsqr    : 0.0389s  (good middle ground)
   auto    : 0.0234s  (selects svd)

Dataset Size-Based Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   def recommend_solver(n_points, n_params):
       """Recommend optimal solver based on problem size"""
       if n_points < 10_000:
           return "svd"
       elif n_points < 1_000_000:
           return "cg"
       else:
           return "minibatch"


   # Or just use 'auto'
   popt, pcov = curve_fit(model, x, y, solver="auto")

--------------

Profiling and Benchmarking
--------------------------

Built-in Timing
~~~~~~~~~~~~~~~

.. code:: python

   # Enable timing
   popt, pcov, res, post_time, compile_time = curve_fit(
       model, x, y, p0=[2, 1, 0], timeit=True
   )

   print(f"Compilation time: {compile_time:.4f}s")
   print(f"Post-processing time: {post_time:.4f}s")
   print(f"Total time: {compile_time + post_time:.4f}s")

Detailed Profiling
~~~~~~~~~~~~~~~~~~

.. code:: python

   import jax.profiler

   # Profile GPU kernels
   with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
       popt, pcov = curve_fit(model, x, y, p0=[2, 1, 0])

   # View trace at: perfetto.dev
   print("Open trace at: https://ui.perfetto.dev")

Benchmarking Script
~~~~~~~~~~~~~~~~~~~

.. code:: python

   import pandas as pd


   def benchmark_nlsq(sizes=[100, 1000, 10000, 100000, 1000000]):
       """Benchmark NLSQ across dataset sizes"""
       results = []

       for n in sizes:
           x = np.linspace(0, 5, n)
           y = model(x, 2.5, 1.3, 0.5) + 0.01 * np.random.randn(n)

           # Warm-up
           _ = curve_fit(model, x, y, p0=[2, 1, 0])

           # Timed run
           start = time.time()
           popt, pcov = curve_fit(model, x, y, p0=[2, 1, 0])
           elapsed = time.time() - start

           results.append(
               {"n_points": n, "time_s": elapsed, "points_per_sec": n / elapsed}
           )

       return pd.DataFrame(results)


   df = benchmark_nlsq()
   print(df)

**Output:**

::

      n_points   time_s  points_per_sec
   0       100  0.4123      242.5
   1      1000  0.4234     2362.1
   2     10000  0.4456    22445.2
   3    100000  0.5123   195203.4
   4   1000000  0.8234   1214574.9

--------------

Performance Troubleshooting
---------------------------

Issue 1: Slow First Call
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** First fit takes 10-100x longer than subsequent fits

**Solution:** Use ``CurveFit`` class to reuse compilation

.. code:: python

   # Bad: Recompiles every time
   for data in datasets:
       popt, _ = curve_fit(model, *data, p0=[2, 1, 0])

   # Good: Compiles once
   fitter = CurveFit()
   for data in datasets:
       popt, _ = fitter.curve_fit(model, *data, p0=[2, 1, 0])

Issue 2: GPU Slower Than CPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** GPU performance worse than CPU for small datasets

**Explanation:** JIT overhead + data transfer overhead dominates

**Solution:** Only use GPU for datasets > 10K points

.. code:: python

   # For small datasets, force CPU
   if len(x) < 10000:
       os.environ["JAX_PLATFORM_NAME"] = "cpu"

Issue 3: Out of Memory (OOM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``RuntimeError: RESOURCE_EXHAUSTED: Out of memory``

**Solutions:**

1. **Reduce batch size:**

.. code:: python

   popt, pcov = curve_fit(model, x, y, solver="minibatch", batch_size=10000)

2. **Use memory-efficient solver:**

.. code:: python

   popt, pcov = curve_fit(model, x, y, solver="cg")

3. **Enable chunking:**

.. code:: python

   from nlsq.streaming.large_dataset import fit_large_dataset

   popt, pcov, info = fit_large_dataset(model, x, y, memory_limit_gb=4.0)

4. **Disable JIT for very large models:**

.. code:: python

   os.environ["JAX_DISABLE_JIT"] = "1"  # Last resort - very slow

Issue 4: No GPU Detected
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** ``Available devices: [CpuDevice(id=0)]`` despite having GPU

**Checks:**

.. code:: bash

   # Check CUDA installation
   nvidia-smi

   # Check JAX GPU support
   python -c "import jax; print(jax.devices())"

   # Reinstall JAX with CUDA support
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

--------------

Performance Checklist
---------------------

Before deploying NLSQ in production, verify:

-  ☐ GPU is detected and used (for datasets > 10K points)
-  ☐ ``CurveFit`` class used for multiple fits
-  ☐ ``solver='auto'`` or appropriate manual selection
-  ☐ Memory profiling done for large datasets
-  ☐ JIT compilation cached (warm-up run completed)
-  ☐ Batch processing used when fitting multiple datasets
-  ☐ Appropriate bounds set (improves convergence speed)
-  ☐ Mixed precision enabled (default: float32 → float64 upgrade)
-  ☐ Profiling confirms GPU kernels are executing
-  ☐ No memory warnings or OOM errors

--------------

Performance Optimization Flowchart
----------------------------------

::

   Start
     │
     ├─> Dataset < 10K points? ──Yes──> Use CPU (JIT overhead dominates)
     │                          │
     │                          No
     │                          │
     ├─> Dataset > 20M points? ─Yes──> Use fit_large_dataset() with chunking
     │                         │
     │                         No
     │                         │
     ├─> Multiple datasets? ───Yes──> Use CurveFit class + batch processing
     │                        │
     │                        No
     │                        │
     ├─> GPU available? ──────Yes──> Use GPU with solver='auto'
     │                       │
     │                       No
     │                       │
     └─> Use CPU with solver='auto'

--------------

Jacobian Automatic Differentiation Configuration
-------------------------------------------------

NLSQ automatically computes Jacobians using JAX's automatic differentiation.
Starting in v0.3.0, you can control the AD mode for optimal performance.

Jacobian Modes
~~~~~~~~~~~~~~

JAX provides two automatic differentiation modes with different performance characteristics:

- **Forward-mode (jacfwd)**: Efficient when n_params ≤ n_residuals (wide Jacobians)
- **Reverse-mode (jacrev)**: Efficient when n_params > n_residuals (tall Jacobians)

The computational cost difference can be 10-100x for high-parameter problems!

Automatic Mode Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, NLSQ automatically selects the optimal mode based on problem dimensions:

.. code:: python

   from nlsq import curve_fit

   # Automatic selection (recommended)
   # For 1000 params, 100 residuals → automatically uses jacrev
   popt, pcov = curve_fit(model, xdata, ydata, p0=initial_guess, jacobian_mode="auto")

Manual Override
~~~~~~~~~~~~~~~

You can manually override the automatic selection:

.. code:: python

   # Force forward-mode AD
   popt, pcov = curve_fit(model, xdata, ydata, p0=p0, jacobian_mode="fwd")

   # Force reverse-mode AD
   popt, pcov = curve_fit(model, xdata, ydata, p0=p0, jacobian_mode="rev")

Configuration Precedence
~~~~~~~~~~~~~~~~~~~~~~~~~

Jacobian mode can be configured at multiple levels (highest to lowest priority):

1. **Function parameter**: ``jacobian_mode='fwd'`` in ``curve_fit()``
2. **Environment variable**: ``export NLSQ_JACOBIAN_MODE=rev``
3. **Config file**: ``~/.nlsq/config.json``
4. **Auto-default**: Automatic selection based on problem dimensions

Example: Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # Set globally for all fits in current session
   export NLSQ_JACOBIAN_MODE=rev
   python my_fitting_script.py

Example: Config File
^^^^^^^^^^^^^^^^^^^^

Create ``~/.nlsq/config.json``:

.. code:: json

   {
     "jacobian_mode": "rev"
   }

Example: Programmatic Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from nlsq.config import set_jacobian_mode

   # Set mode for current Python session
   set_jacobian_mode("rev")

   # All subsequent fits will use reverse-mode
   popt1, _ = curve_fit(model1, x1, y1, p0=p0_1)
   popt2, _ = curve_fit(model2, x2, y2, p0=p0_2)

When to Use Each Mode
~~~~~~~~~~~~~~~~~~~~~

**Use jacrev (reverse-mode) when:**

- Many parameters, few data points (tall Jacobian)
- Parameter estimation problems with 100+ parameters
- High-dimensional optimization (n_params > n_residuals)

**Use jacfwd (forward-mode) when:**

- Few parameters, many data points (wide Jacobian)
- Standard curve fitting (2-10 parameters, 100+ points)
- Low-dimensional optimization (n_params ≤ n_residuals)

**Use auto (recommended) when:**

- You're unsure about the problem structure
- Problem dimensions vary across different datasets
- You want optimal performance without manual tuning

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

**High-parameter problem (1000 params, 100 residuals):**

.. code:: python

   # Automatic selection (uses jacrev)
   popt, pcov = curve_fit(high_param_model, xdata, ydata, p0=p0, jacobian_mode="auto")

   # Expected: 10-100x faster Jacobian computation vs jacfwd on GPU

**Standard fitting problem (3 params, 1000 residuals):**

.. code:: python

   # Automatic selection (uses jacfwd)
   popt, pcov = curve_fit(exponential, xdata, ydata, p0=[1, 1, 1], jacobian_mode="auto")

   # Expected: Comparable or faster vs jacrev

Debug Logging
~~~~~~~~~~~~~

Enable debug logging to see Jacobian mode selection:

.. code:: python

   import logging

   logging.basicConfig(level=logging.DEBUG)

   from nlsq import curve_fit

   popt, pcov = curve_fit(model, xdata, ydata, p0=p0, jacobian_mode="auto")

   # Output: Jacobian mode: 'rev' (from auto-default). Rationale: jacrev (1000 params > 100 residuals)

Common Use Cases
~~~~~~~~~~~~~~~~

**Case 1: Parameter-Heavy Fitting**

Fitting a model with many parameters to limited data (e.g., neural network parameter estimation):

.. code:: python

   # 500 parameters, 100 data points
   popt, pcov = curve_fit(
       neural_network_model,
       xdata,
       ydata,
       p0=np.random.randn(500),
       jacobian_mode="rev",  # Explicitly use reverse-mode for efficiency
   )

**Case 2: Batch Processing with Varying Dimensions**

Processing multiple datasets with different parameter counts:

.. code:: python

   from nlsq import CurveFit

   fitter = CurveFit(model)

   for xdata, ydata, p0 in datasets:
       # Auto mode adapts to each dataset's dimensions
       popt, pcov = fitter.fit(xdata, ydata, p0=p0, jacobian_mode="auto")

**Case 3: Performance-Critical Pipeline**

For production pipelines, benchmark and set the mode explicitly:

.. code:: python

   # Benchmark both modes once
   import time

   modes = ["fwd", "rev"]
   times = {}

   for mode in modes:
       start = time.time()
       popt, _ = curve_fit(model, xdata, ydata, p0=p0, jacobian_mode=mode)
       times[mode] = time.time() - start

   best_mode = min(times, key=times.get)
   print(f"Best mode: {best_mode} ({times[best_mode]:.4f}s)")

   # Use best mode in production
   from nlsq.config import set_jacobian_mode

   set_jacobian_mode(best_mode)

Troubleshooting
~~~~~~~~~~~~~~~

**Issue**: "Invalid jacobian_mode: xyz"

**Solution**: Use one of 'auto', 'fwd', or 'rev':

.. code:: python

   # ✗ Wrong
   curve_fit(model, x, y, jacobian_mode="reverse")  # ValueError

   # ✓ Correct
   curve_fit(model, x, y, jacobian_mode="rev")

**Issue**: Performance doesn't match expectations

**Solution**: Check problem dimensions and verify mode selection:

.. code:: python

   import logging

   logging.basicConfig(level=logging.DEBUG)

   # Enable verbose output to see actual mode used
   popt, _ = curve_fit(model, x, y, p0=p0, jacobian_mode="auto", verbose=2)

--------------

Related Documentation
---------------------

-  :doc:`../reference/configuration` - Configuration reference
-  :doc:`advanced_api` - Advanced API usage
-  :doc:`migrate_from_scipy` - Migrating from SciPy
-  :doc:`troubleshooting` - Common issues and solutions
-  :doc:`../api/index` - Complete API documentation

--------------

Interactive Notebooks
---------------------

Hands-on tutorials for performance optimization:

**Performance Fundamentals:**

- `Performance Optimization Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/02_core_tutorials/performance_optimization_demo.ipynb>`_ (25-35 min) - Speed optimization, GPU setup, migration checklist

**Advanced Performance:**

- `GPU Optimization Deep Dive <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/03_advanced/gpu_optimization_deep_dive.ipynb>`_ (40 min) - Maximize GPU utilization, memory optimization, performance profiling
- `Custom Algorithms <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/03_advanced/custom_algorithms_advanced.ipynb>`_ (40 min) - Implement custom optimizers, algorithm selection

**Large Dataset Performance:**

- `Large Dataset Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/02_core_tutorials/large_dataset_demo.ipynb>`_ (25 min) - Chunking, streaming, memory management

--------------

Benchmarking Results
--------------------

Official Benchmarks (NVIDIA V100 GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============ ========== ========== =========== =======
Dataset Size Parameters NLSQ (GPU) SciPy (CPU) Speedup
============ ========== ========== =========== =======
1,000        3          0.03s      0.05s       1.7x
10,000       3          0.04s      0.18s       4.5x
100,000      3          0.09s      2.1s        23x
1,000,000    5          0.15s      40.5s       270x
10,000,000   5          0.42s      480s        1143x
============ ========== ========== =========== =======

*Benchmarks run on NVIDIA Tesla V100 32GB GPU, Intel Xeon Gold 6248R
CPU*

Scaling Characteristics
~~~~~~~~~~~~~~~~~~~~~~~

-  **Near-linear scaling** up to 50M points on GPU
-  **Excellent batch performance**: 100 datasets in 2x time of single
   dataset
-  **Memory efficient**: 1M points uses ~500MB GPU memory
