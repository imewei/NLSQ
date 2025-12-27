GPU Architecture and Acceleration
==================================

This guide explains how NLSQ leverages GPU hardware for massive speedups
and when GPU acceleration is most beneficial.

Why GPUs for Curve Fitting?
---------------------------

GPUs have thousands of cores optimized for parallel computation:

.. list-table::
   :header-rows: 1

   * - Hardware
     - Cores
     - Best For
   * - CPU (typical)
     - 4-16
     - Sequential, complex logic
   * - GPU (NVIDIA V100)
     - 5120
     - Parallel, simple operations
   * - GPU (NVIDIA A100)
     - 6912
     - Even more parallel capacity

Curve fitting benefits because:

1. **Data parallelism**: Same operation on many points
2. **Matrix operations**: Jacobian computation, SVD
3. **Batch processing**: Multiple function evaluations

Where Speedups Come From
------------------------

**Residual computation**:

.. code-block:: text

   CPU: Process each point sequentially
   for i in range(1_000_000):
       r[i] = y[i] - f(x[i])

   GPU: Process all points in parallel
   r = y - f(x)  # All 1M points at once

**Jacobian computation**:

.. code-block:: text

   CPU: Finite differences (2m function evaluations)
   GPU: Single reverse-mode AD pass

**Matrix operations**:

.. code-block:: text

   CPU: Sequential SVD
   GPU: Parallel SVD with cuBLAS/cuSOLVER

Performance Scaling
-------------------

Expected speedups by dataset size:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Dataset Size
     - SciPy (CPU)
     - NLSQ (V100)
     - Speedup
   * - 1,000
     - 0.05s
     - 0.43s (JIT)
     - 0.1x (JIT cost)
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
   * - 10,000,000
     - ~7 min
     - 1.5s
     - ~280x

Key observations:

1. **JIT compilation overhead**: First call is slower
2. **Crossover point**: GPU wins at ~5,000 points
3. **Scaling advantage**: GPU speedup increases with size

Memory Considerations
---------------------

GPU memory is limited (16-80 GB typical). NLSQ handles this with:

**Automatic chunking**:

.. code-block:: python

   from nlsq import curve_fit_large

   # Automatically chunks if data exceeds GPU memory
   popt, pcov = curve_fit_large(model, x, y, memory_limit_gb=8.0)  # Match your GPU

**Streaming optimization**:

.. code-block:: python

   from nlsq import StreamingOptimizer

   # Process data in chunks, accumulate gradients
   optimizer = StreamingOptimizer(model, n_params=3)
   result = optimizer.fit(x, y, p0=p0)

Multi-GPU Usage
---------------

For systems with multiple GPUs:

.. code-block:: python

   import jax

   # See all available devices
   devices = jax.devices()
   print(f"Available GPUs: {devices}")

   # NLSQ uses the default device
   # To select a specific GPU:
   import os

   os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only

Data Transfer Overhead
----------------------

Moving data between CPU and GPU has a cost:

.. code-block:: text

   CPU RAM ←──[PCIe bus]──→ GPU VRAM
                ~12 GB/s

For small datasets, this overhead can dominate. That's why:

- Small data (<1K points): Use CPU
- Large data (>10K points): Use GPU

NLSQ minimizes transfers by:

1. Keeping data on GPU throughout optimization
2. Only transferring results at the end
3. Using JAX's lazy evaluation

Mixed Precision
---------------

Modern GPUs have Tensor Cores for fast float16/bfloat16:

.. code-block:: python

   # Use float32 for larger datasets with acceptable precision
   x = x.astype(jnp.float32)
   y = y.astype(jnp.float32)

   # NLSQ uses float64 by default for numerical accuracy
   # but float32 can be 2x faster with half the memory

When GPU Doesn't Help
---------------------

GPU may not be beneficial when:

1. **Small datasets** (<1K points): JIT overhead dominates
2. **Simple models**: Not enough computation to parallelize
3. **Single fit**: JIT compilation cost not amortized
4. **Memory-bound**: Data larger than GPU memory (use streaming)

CPU may be preferred when:

1. Testing and development
2. Laptops without discrete GPU
3. Need maximum numerical precision

Optimizing GPU Performance
--------------------------

**1. Warm up JIT**

.. code-block:: python

   # Compile on small data first
   _ = curve_fit(model, x[:100], y[:100], p0=p0)

   # Then run on full data (uses cached compilation)
   popt, pcov = curve_fit(model, x, y, p0=p0)

**2. Use CurveFit class for repeated fits**

.. code-block:: python

   from nlsq import CurveFit

   fitter = CurveFit()  # Compile once

   for dataset in datasets:
       popt, pcov = fitter.curve_fit(model, dataset.x, dataset.y)
       # All calls after first are fast

**3. Batch similar models**

.. code-block:: python

   # If fitting many similar datasets, batch them
   import jax

   batched_fit = jax.vmap(single_fit, in_axes=(0, 0))
   results = batched_fit(x_batch, y_batch)

**4. Profile to find bottlenecks**

.. code-block:: python

   # Use JAX profiler
   with jax.profiler.trace("/tmp/jax-trace"):
       popt, pcov = curve_fit(model, x, y)

   # View with TensorBoard

Hardware Requirements
---------------------

Minimum:
- NVIDIA GPU with CUDA Compute Capability 5.0+
- 4 GB VRAM
- CUDA 11.x or 12.x

Recommended:
- NVIDIA GPU with Tensor Cores (V100, A100, RTX series)
- 16+ GB VRAM
- CUDA 12.x
- cuDNN 8.x

Cloud Options:
- Google Colab (free T4 GPU)
- AWS EC2 (p3, p4 instances)
- GCP Compute Engine (A100, T4)

Summary
-------

GPU acceleration in NLSQ provides:

1. **Massive parallelism**: Thousands of cores for data operations
2. **Automatic optimization**: JAX handles GPU placement
3. **Scaling advantage**: Speedup grows with dataset size
4. **Memory management**: Automatic chunking and streaming

Best for:
- Large datasets (>10K points)
- Repeated fits (amortize JIT)
- Production pipelines

See Also
--------

- :doc:`jax_autodiff` - How JAX enables GPU
- :doc:`/tutorials/06_gpu_acceleration` - GPU setup tutorial
- :doc:`/howto/optimize_performance` - Performance guide
