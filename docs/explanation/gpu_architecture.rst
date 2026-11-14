GPU Architecture and Acceleration
==================================

This guide explains how NLSQ uses GPU hardware and when GPU acceleration helps. Measured timings are in
:doc:`/api/performance_benchmarks`; the GPU is faster than the CPU for large problems and slower for small ones.

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

What Differs Between CPU and GPU
--------------------------------

NLSQ runs the same JAX code on both devices. Residuals and the Jacobian (computed by automatic
differentiation, not finite differences) are XLA-compiled and vectorized on the CPU as well as on the GPU, so
the comparison is between two compiled paths, not between a Python loop and a kernel.

**What the GPU adds**: the work that scales with the number of points (evaluating the model and its Jacobian
at every point, and reducing them to the small ``J^T J`` and ``J^T r`` matrices) is spread over thousands of
cores, which pays off once there are enough points to keep them busy.

**What does not shrink**: the fit loop itself runs on the host. Each iteration launches device work and reads
back small results, and each new problem shape is compiled once. These costs do not depend on the dataset size,
which is why the GPU was slower than the CPU for small fits in the measurements below.

Performance Scaling
-------------------

These are NLSQ's own CPU and GPU runs, compared with each other (RTX 4090 Laptop GPU and i9-13900H, four
models, warm medians). They are not a comparison with another library; those, made within one device, are in
:doc:`/api/performance_benchmarks`.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Dataset size
     - GPU time compared with CPU time
   * - 1,000 and 10,000 points
     - GPU 1.4x to 7.1x slower
   * - 100,000 points
     - GPU 1.2x slower to 3.2x faster, depending on the model
   * - 1,000,000 points
     - GPU 4.9x to 6.8x faster (``curve_fit_large``)
   * - 10,000,000 and 100,000,000 points
     - GPU 9.8x to 20.7x faster (``curve_fit_large``)

Key observations:

1. **Fixed per-fit overhead**: JIT dispatch and host/device transfers dominate small problems, and the
   first call for a new shape also compiles the kernel
2. **Crossover point**: on this hardware the GPU was faster on all four models from 1,000,000 points; at
   100,000 points it ranged from 1.2x slower to 3.2x faster depending on the model, and below that the CPU
   was faster
3. **Scaling advantage**: the GPU's advantage over the CPU grows with the dataset size

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

   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

   # Process data in chunks with bounded memory
   config = HybridStreamingConfig(chunk_size=50000)
   optimizer = AdaptiveHybridStreamingOptimizer(config)
   result = optimizer.fit((x, y), model, p0=p0)

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

1. **Data parallelism**: Thousands of cores for per-point operations
2. **Automatic optimization**: JAX handles GPU placement
3. **Scaling advantage**: the GPU's advantage over the CPU grows with dataset size
4. **Memory management**: Automatic chunking and streaming

Best for:
- Large datasets (1M points and above; at 100K points it depends on the model, and the CPU is faster for small fits)
- Repeated fits (amortize JIT)
- Production pipelines

See Also
--------

- :doc:`jax_autodiff` - How JAX enables GPU
- :doc:`/tutorials/routine/gpu_acceleration/gpu_usage` - GPU setup tutorial
- :doc:`/howto/optimize_performance` - Performance guide
