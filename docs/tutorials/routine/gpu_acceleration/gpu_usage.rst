GPU Usage
=========

Once JAX is installed with GPU support, NLSQ automatically uses the GPU.
No code changes are required.

Automatic GPU Detection
-----------------------

NLSQ detects available GPUs automatically:

.. code-block:: python

   from nlsq import fit, get_device
   import jax.numpy as jnp

   # Check current device
   print(f"Using device: {get_device()}")


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # GPU used automatically
   popt, pcov = fit(model, x, y, p0=[1, 0.5, 0])

JIT Compilation
---------------

The first fit includes JIT (Just-In-Time) compilation:

.. code-block:: python

   import time

   # First fit: includes JIT compilation (~1-5 seconds)
   start = time.time()
   popt1, pcov1 = fit(model, x1, y1, p0=[1, 0.5, 0])
   print(f"First fit: {time.time() - start:.2f}s")

   # Subsequent fits: cached compilation (~10x faster)
   start = time.time()
   popt2, pcov2 = fit(model, x2, y2, p0=[1, 0.5, 0])
   print(f"Second fit: {time.time() - start:.2f}s")

NLSQ uses persistent JIT caching at ``~/.cache/nlsq/jax_cache``.

Forcing CPU Usage
-----------------

To run on CPU even when GPU is available:

.. code-block:: python

   import os

   os.environ["NLSQ_FORCE_CPU"] = "1"

   # Or set JAX backend
   import jax

   jax.config.update("jax_platform_name", "cpu")

Data Transfer
-------------

Data is transferred to GPU automatically:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp

   # NumPy arrays (transferred to GPU during fit)
   x = np.linspace(0, 10, 100000)
   y = np.random.randn(100000)

   # JAX arrays (already on GPU if JAX uses GPU)
   x_jax = jnp.array(x)
   y_jax = jnp.array(y)

   # Both work - NLSQ handles conversion
   popt, pcov = fit(model, x, y, p0=[...])

GPU Memory Management
---------------------

For large datasets, control GPU memory:

.. code-block:: python

   import os

   # Don't preallocate all GPU memory
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

   # Use only 50% of GPU memory
   os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

   # Must set before importing JAX
   from nlsq import fit

When GPU Helps Most
-------------------

GPU acceleration is most beneficial for:

- **Large datasets**: 100K+ points
- **Complex models**: Many parameters, complex math
- **Repeated fits**: JIT cache amortizes compilation
- **Global optimization**: Many parallel evaluations

GPU may not help or be slower for:

- **Small datasets**: <1000 points (data transfer overhead)
- **Simple models**: Overhead exceeds computation time
- **Single fits**: JIT compilation dominates

Benchmark Your Workload
-----------------------

.. code-block:: python

   import time
   import numpy as np
   import jax
   import jax.numpy as jnp
   from nlsq import fit


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   def benchmark(n_points):
       x = np.linspace(0, 10, n_points)
       y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(n_points)

       # Warm-up (JIT compilation)
       popt, _ = fit(model, x, y, p0=[2, 0.5, 0])

       # Timed fit
       start = time.time()
       for _ in range(5):
           popt, _ = fit(model, x, y, p0=popt)
       elapsed = (time.time() - start) / 5

       return elapsed


   print(f"Backend: {jax.default_backend()}")
   for n in [1000, 10000, 100000, 1000000]:
       t = benchmark(n)
       print(f"{n:>10,} points: {t:.3f}s")

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq import fit, get_device

   # Check device
   print(f"Running on: {get_device()}")

   # Large dataset
   np.random.seed(42)
   n_points = 500_000
   x = np.linspace(0, 100, n_points)
   y_true = 2.5 * np.exp(-0.05 * x) + 0.3
   y = y_true + 0.2 * np.random.randn(n_points)


   # Model
   def exponential(x, A, k, c):
       return A * jnp.exp(-k * x) + c


   # Fit (GPU automatically used)
   import time

   start = time.time()
   popt, pcov = fit(exponential, x, y, p0=[2, 0.05, 0])
   print(f"Fit time: {time.time() - start:.2f}s")
   print(f"Results: A={popt[0]:.3f}, k={popt[1]:.4f}, c={popt[2]:.3f}")

Next Steps
----------

- :doc:`multi_gpu` - Use multiple GPUs
- :doc:`../data_handling/large_datasets` - Large dataset handling
