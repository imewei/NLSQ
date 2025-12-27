Tutorial 6: GPU Acceleration
=============================

In this tutorial, you'll learn how to leverage GPU/TPU acceleration for
dramatically faster curve fitting.

What You'll Learn
-----------------

- When GPU acceleration helps
- Setting up JAX for GPU
- Verifying GPU usage
- Performance optimization tips

Prerequisites
-------------

- Completed tutorials 1-5
- NVIDIA GPU with CUDA support (for GPU acceleration)

When GPU Acceleration Helps
---------------------------

GPU acceleration provides significant speedups when:

- Dataset has **> 10,000 points** (more points = bigger speedup)
- Model has **many parameters** or **complex computations**
- You're fitting **multiple datasets** (JIT compilation is amortized)

GPU may not help when:

- Dataset has **< 1,000 points** (CPU overhead dominates)
- Model is **very simple** (e.g., linear fit)
- You're doing a **single one-off fit** (JIT compilation overhead)

Expected Speedups
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Dataset Size
     - SciPy (CPU)
     - NLSQ (GPU)
     - Speedup
   * - 10,000
     - 0.18s
     - 0.04s
     - 4.5x
   * - 100,000
     - 2.1s
     - 0.09s
     - 23x
   * - 1,000,000
     - 40.5s
     - 0.15s
     - 270x

*Benchmarks on NVIDIA Tesla V100. Results vary by GPU model.*

Step 1: Check Current Backend
-----------------------------

First, check what JAX backend is currently active:

.. code-block:: python

   import jax

   print(f"JAX version: {jax.__version__}")
   print(f"Available devices: {jax.devices()}")
   print(f"Default backend: {jax.default_backend()}")

Output will show either ``cpu``, ``gpu``, or ``tpu``.

Step 2: Install JAX with GPU Support
------------------------------------

If you see ``cpu`` but have an NVIDIA GPU, install JAX with CUDA:

.. code-block:: bash

   # For CUDA 12.x (most modern systems)
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # For CUDA 11.x (older systems)
   pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Verify installation
   python -c "import jax; print(jax.devices())"

You should see output like ``[cuda(id=0)]``.

Step 3: Verify GPU Usage with NLSQ
----------------------------------

NLSQ automatically uses GPU if available:

.. code-block:: python

   import jax
   from nlsq import curve_fit
   import jax.numpy as jnp
   import numpy as np

   # Check device
   print(f"NLSQ will use: {jax.default_backend()}")


   # Simple test
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = np.linspace(0, 10, 100000)
   y = 2.5 * np.exp(-0.5 * x) + 0.1 * np.random.randn(len(x))

   # Fit - automatically uses GPU
   popt, pcov = curve_fit(model, x, y)
   print(f"Fitted on {jax.default_backend()}: a={popt[0]:.4f}, b={popt[1]:.4f}")

Forcing CPU or GPU
------------------

You can explicitly control the backend:

.. code-block:: python

   import os

   # Force CPU (before importing JAX)
   os.environ["JAX_PLATFORM_NAME"] = "cpu"

   # Or force GPU
   os.environ["JAX_PLATFORM_NAME"] = "cuda"

   # Now import JAX and NLSQ
   import jax
   from nlsq import curve_fit

NLSQ also provides environment variables:

.. code-block:: bash

   # Force CPU for testing
   NLSQ_FORCE_CPU=1 python my_script.py

Benchmarking CPU vs GPU
-----------------------

Compare performance on your system:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import curve_fit
   import time


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   def benchmark(n_points, n_runs=5):
       """Benchmark fitting speed."""
       np.random.seed(42)
       x = np.linspace(0, 10, n_points)
       y = 2.0 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(n_points)

       # Warmup (JIT compilation)
       _ = curve_fit(model, x[:1000], y[:1000], p0=[2.0, 0.5, 0.3])

       # Benchmark
       times = []
       for _ in range(n_runs):
           start = time.time()
           popt, pcov = curve_fit(model, x, y, p0=[2.0, 0.5, 0.3])
           times.append(time.time() - start)

       return np.mean(times), np.std(times)


   # Test different sizes
   import jax

   print(f"Backend: {jax.default_backend()}")
   print("-" * 50)

   for n in [10_000, 100_000, 1_000_000]:
       mean_time, std_time = benchmark(n)
       rate = n / mean_time / 1e6
       print(f"{n:>10,} points: {mean_time:.3f} ± {std_time:.3f}s ({rate:.2f}M pts/s)")

Multi-GPU Usage
---------------

For systems with multiple GPUs:

.. code-block:: python

   import jax

   # See all devices
   devices = jax.devices()
   print(f"Available GPUs: {devices}")

   # NLSQ uses the default device
   # To use a specific GPU:
   import os

   os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0

Memory Management on GPU
------------------------

GPU memory is limited. For very large datasets:

.. code-block:: python

   # Monitor GPU memory
   import jax

   # Get device memory stats (JAX 0.4.20+)
   for device in jax.devices():
       if hasattr(device, "memory_stats"):
           stats = device.memory_stats()
           print(f"{device}: {stats}")

   # For large datasets, use streaming (automatically manages memory)
   from nlsq import curve_fit_large

   popt, pcov = curve_fit_large(model, x, y, memory_limit_gb=8.0)  # Limit GPU memory usage

TPU Acceleration
----------------

NLSQ also supports Google TPUs:

.. code-block:: python

   # On Google Colab with TPU runtime
   import jax

   print(f"TPU devices: {jax.devices()}")

   # Use normally - NLSQ auto-detects TPU
   from nlsq import curve_fit

   popt, pcov = curve_fit(model, x, y)

Common Issues and Solutions
---------------------------

Issue: "No GPU found"
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check CUDA installation
   import subprocess

   result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
   print(result.stdout)

   # Check JAX can see GPU
   import jax

   print(jax.devices())

If no GPU appears:

1. Verify NVIDIA drivers: ``nvidia-smi``
2. Reinstall JAX with CUDA support
3. Check CUDA version matches JAX requirements

Issue: Out of GPU Memory
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reduce batch size
   from nlsq import curve_fit_large

   popt, pcov = curve_fit_large(model, x, y, memory_limit_gb=2.0)

   # Or use streaming
   from nlsq import StreamingOptimizer

   optimizer = StreamingOptimizer(model, n_params=3, chunk_size=50000)

Issue: GPU is Slower Than Expected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This usually happens on first call due to JIT compilation:

.. code-block:: python

   import time

   # First call: includes compilation (slow)
   start = time.time()
   popt1, _ = curve_fit(model, x, y)
   print(f"First call: {time.time() - start:.3f}s")

   # Second call: uses cached compilation (fast)
   start = time.time()
   popt2, _ = curve_fit(model, x, y)
   print(f"Second call: {time.time() - start:.3f}s")

For repeated fits, use the ``CurveFit`` class:

.. code-block:: python

   from nlsq import CurveFit

   fitter = CurveFit()  # Compile once

   for dataset in datasets:
       popt, pcov = fitter.curve_fit(model, dataset.x, dataset.y)
       # All calls after first are fast

Complete Example: GPU Performance Comparison
--------------------------------------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   import jax
   from nlsq import curve_fit
   import time


   def complex_model(x, a, b, c, d, e):
       """Complex model to benefit from GPU."""
       return a * jnp.exp(-b * x) * jnp.sin(c * x + d) + e


   # Generate large dataset
   np.random.seed(42)
   n = 500_000
   x = np.linspace(0, 20, n)
   y_true = 2.0 * np.exp(-0.1 * x) * np.sin(1.5 * x + 0.3) + 0.5
   y = y_true + 0.1 * np.random.randn(n)

   print(f"Dataset: {n:,} points")
   print(f"Backend: {jax.default_backend()}")
   print("-" * 50)

   # Warmup
   _ = curve_fit(complex_model, x[:1000], y[:1000], p0=[2, 0.1, 1.5, 0.3, 0.5])

   # Benchmark
   n_trials = 3
   times = []

   for i in range(n_trials):
       start = time.time()
       popt, pcov = curve_fit(complex_model, x, y, p0=[2, 0.1, 1.5, 0.3, 0.5])
       elapsed = time.time() - start
       times.append(elapsed)
       print(f"Trial {i+1}: {elapsed:.3f}s")

   print("-" * 50)
   print(f"Average: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
   print(f"Throughput: {n / np.mean(times) / 1e6:.2f} million points/second")

   # Results
   perr = np.sqrt(np.diag(pcov))
   print("\nFitted parameters:")
   for i, (p, e) in enumerate(zip(popt, perr)):
       print(f"  p{i} = {p:.4f} ± {e:.4f}")

Key Takeaways
-------------

1. **GPU provides 10-300x speedup** for large datasets
2. Install JAX with CUDA: ``pip install "jax[cuda12_pip]"``
3. Check backend with ``jax.devices()``
4. First call includes JIT compilation overhead
5. Use ``CurveFit`` class for repeated fits
6. Use streaming for datasets larger than GPU memory

Congratulations!
----------------

You've completed the NLSQ tutorial series! You now know how to:

- Fit models to data (:doc:`01_first_fit`)
- Interpret fit results (:doc:`02_understanding_results`)
- Use bounds for physical constraints (:doc:`03_fitting_with_bounds`)
- Handle complex multi-parameter models (:doc:`04_multiple_parameters`)
- Work with large datasets (:doc:`05_large_datasets`)
- Leverage GPU acceleration (this tutorial)

Next Steps
----------

Explore more advanced topics:

- :doc:`/howto/index` - Solve specific problems
- :doc:`/explanation/index` - Understand the theory
- :doc:`/reference/index` - Complete API reference

.. seealso::

   - :doc:`/explanation/gpu_architecture` - How GPU acceleration works
   - :doc:`/howto/optimize_performance` - Performance tuning guide
