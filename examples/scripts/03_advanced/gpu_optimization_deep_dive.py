"""
Converted from gpu_optimization_deep_dive.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # GPU Optimization and Performance Deep Dive
#
# **Level**: Advanced
# **Time**: 50-70 minutes
# **Prerequisites**: NLSQ Quickstart, JAX basics
#
# ## Overview
#
# This tutorial covers **performance optimization** for NLSQ, focusing on:
# - JAX JIT compilation and profiling
# - GPU acceleration strategies
# - Memory optimization
# - Batch processing for maximum throughput
#
# ### What You'll Learn
#
# 1. **JAX Profiling**: Identifying bottlenecks with JAX tools
# 2. **JIT Compilation**: Understanding and optimizing compilation
# 3. **GPU Acceleration**: When and how to leverage GPUs
# 4. **Memory Management**: Avoiding OOM errors
# 5. **Batch Strategies**: Processing thousands of fits efficiently
# 6. **Benchmarking**: Measuring and comparing performance
#
# ### Performance Targets
#
# Typical NLSQ performance (depends on hardware, problem size):
# - **Cold start (first call)**: 0.5-2 seconds (includes JIT compilation)
# - **Warm calls (cached)**: 1-50 ms per fit
# - **GPU speedup**: 5-50x for large batches vs CPU
# - **Batch throughput**: 100-10,000 fits/second (GPU, batched)
#
# ### Hardware Requirements
#
# This notebook runs on CPU or GPU. GPU examples automatically fall back to CPU if no GPU is available.
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap

from nlsq import CurveFit
from nlsq.utils.error_messages import OptimizationError

# Detect available devices
devices = jax.devices()
has_gpu = any("gpu" in str(d).lower() for d in devices)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


print("Hardware Configuration:")
print(f"  JAX version: {jax.__version__}")
print(f"  Default backend: {jax.default_backend()}")
print(f"  Available devices: {devices}")
print(f"  GPU available: {'✓ Yes' if has_gpu else '✗ No (will use CPU)'}")
print()

if has_gpu:
    print("GPU detected - examples will show GPU acceleration")
else:
    print("Running on CPU - GPU examples will still work but won't show speedup")
    print("To use GPU: Install jax[cuda] or jax[rocm] depending on your hardware")


# ======================================================================
# ## Part 1: JIT Compilation Basics
#
# Understanding JAX's Just-In-Time (JIT) compilation is crucial for performance.
# ======================================================================


# Demonstrating JIT compilation overhead and benefits


# Simple model
def exponential_model(x, a, b):
    return a * jnp.exp(-b * x)


# Test data
x_test = jnp.linspace(0, 5, cap_samples(1000))
y_test = exponential_model(x_test, 3.0, 0.5) + np.random.normal(0, 0.1, len(x_test))

cf = CurveFit()

print("JIT Compilation Analysis:")
print("=" * 60)

# First call: includes compilation time
start = time.time()
popt1, _ = cf.curve_fit(exponential_model, x_test, y_test, p0=[2.0, 0.3])
time_first = (time.time() - start) * 1000  # ms

# Second call: uses cached compilation
start = time.time()
popt2, _ = cf.curve_fit(exponential_model, x_test, y_test, p0=[2.5, 0.4])
time_second = (time.time() - start) * 1000  # ms

# Third call: still cached
start = time.time()
popt3, _ = cf.curve_fit(exponential_model, x_test, y_test, p0=[3.0, 0.5])
time_third = (time.time() - start) * 1000  # ms

print(f"First call (cold):  {time_first:.1f} ms (includes JIT compilation)")
print(f"Second call (warm): {time_second:.1f} ms (cached)")
print(f"Third call (warm):  {time_third:.1f} ms (cached)")
print()
print(f"Speedup after JIT:  {time_first / time_second:.1f}x")
print(
    f"Compilation overhead: {time_first - time_second:.1f} ms ({(time_first - time_second) / time_first * 100:.1f}% of first call)"
)
print()
print("Key insight: First call is slow due to JIT compilation.")
print("            Subsequent calls are much faster (10-100x).")


# Understanding what triggers recompilation

print("Recompilation Triggers:")
print("=" * 60)

# Trigger 1: Different array shapes
print("\n1. Changing array shapes triggers recompilation:")

x_100 = jnp.linspace(0, 5, 100)
y_100 = exponential_model(x_100, 3.0, 0.5) + np.random.normal(0, 0.1, 100)

x_200 = jnp.linspace(0, 5, cap_samples(200))
y_200 = exponential_model(x_200, 3.0, 0.5) + np.random.normal(0, 0.1, cap_samples(200))

cf_new = CurveFit()

start = time.time()
cf_new.curve_fit(exponential_model, x_100, y_100, p0=[2.0, 0.3])
time_100 = (time.time() - start) * 1000

start = time.time()
cf_new.curve_fit(exponential_model, x_200, y_200, p0=[2.0, 0.3])  # Different shape!
time_200 = (time.time() - start) * 1000

start = time.time()
cf_new.curve_fit(exponential_model, x_200, y_200, p0=[2.5, 0.4])  # Same shape
time_200_cached = (time.time() - start) * 1000

print(f"  Fit with shape (100,): {time_100:.1f} ms (first compile)")
print(f"  Fit with shape (200,): {time_200:.1f} ms (recompiled!)")
print(f"  Fit with shape (200,): {time_200_cached:.1f} ms (cached) ✓")
print()
print("  → Keep array shapes consistent to avoid recompilation")

# Trigger 2: Different dtypes
print("\n2. Changing dtypes triggers recompilation:")
print("  float32 vs float64 will trigger separate compilations")
print("  → Use consistent dtype (float32 for GPU, float64 for high precision)")

# Trigger 3: Different parameter counts
print("\n3. Different model signatures trigger recompilation:")
print("  model(x, a, b) vs model(x, a, b, c) are compiled separately")
print("  → Expected - different models need different compilations")


# ======================================================================
# ## Part 2: GPU Acceleration
#
# Leverage GPU for massive speedups on large problems.
# ======================================================================


# CPU vs GPU performance comparison

# Large dataset (GPU shines here)
n_points = cap_samples(10000)
x_large = jnp.linspace(0, 10, n_points)
y_large = (
    3.0 * jnp.exp(-0.5 * x_large)
    + 2.0 * jnp.sin(x_large)
    + np.random.normal(0, 0.1, n_points)
)


def complex_model(x, a, b, c, d):
    return a * jnp.exp(-b * x) + c * jnp.sin(d * x)


print(f"GPU Acceleration Benchmark (n_points={n_points}):")
print("=" * 60)

# Ensure compilation is done (use same settings as benchmark for consistency)
cf_gpu = CurveFit()
try:
    _ = cf_gpu.curve_fit(
        complex_model,
        x_large[:100],
        y_large[:100],
        p0=[3, 0.5, 2, 1],
        maxiter=20 if QUICK else 50,
        max_nfev=200 if QUICK else 1000,
    )
except OptimizationError:
    _ = cf_gpu.curve_fit(
        complex_model,
        x_large[:100],
        y_large[:100],
        p0=[3, 0.5, 2, 1],
        maxiter=20 if QUICK else 50,
        max_nfev=400 if QUICK else 1000,
    )

# Benchmark: 10 fits (reduced in quick mode)
n_runs = 3 if QUICK else 10
times = []

for i in range(n_runs):
    # Slightly vary initial guess to avoid trivial caching
    p0 = [3.0 + i * 0.1, 0.5, 2.0, 1.0]
    start = time.time()
    popt, _ = cf_gpu.curve_fit(
        complex_model, x_large, y_large, p0=p0, maxiter=20 if QUICK else 50
    )
    times.append((time.time() - start) * 1000)

mean_time = np.mean(times)
std_time = np.std(times)

print(f"\nDevice: {jax.devices()[0]}")
print(f"Average fit time: {mean_time:.1f} ± {std_time:.1f} ms")
print(f"Throughput: {1000 / mean_time:.1f} fits/second")
print()

if has_gpu:
    print("✓ Running on GPU - performance is optimized")
    print("  Expected speedup vs CPU: 5-20x for this problem size")
else:
    print("Running on CPU - results are valid but slower than GPU")
    print("  With GPU: Expect 5-50x speedup for large datasets")


# ======================================================================
# ## Part 3: Batch Processing Strategies
#
# Process thousands of fits efficiently with vectorization.
# ======================================================================


# Batch processing with vmap for maximum throughput

print("Batch Processing Benchmark:")
print("=" * 60)

# Generate batch of datasets
n_datasets = min(20, cap_samples(200)) if QUICK else max(10, cap_samples(1000))
n_points_per_dataset = 30 if QUICK else 50

x_batch_data = jnp.linspace(0, 5, n_points_per_dataset)

# Random true parameters for each dataset
np.random.seed(42)
a_true_batch = np.random.uniform(2, 4, n_datasets)
b_true_batch = np.random.uniform(0.3, 0.7, n_datasets)

y_batch_data = jnp.array(
    [
        a * jnp.exp(-b * x_batch_data) + np.random.normal(0, 0.05, n_points_per_dataset)
        for a, b in zip(a_true_batch, b_true_batch, strict=True)
    ]
)

print(f"Batch size: {n_datasets} datasets")
print(f"Points per dataset: {n_points_per_dataset}")
print(f"Total data points: {n_datasets * n_points_per_dataset:,}")
print()

# Method 1: Sequential (slow)
print("Method 1: Sequential fitting (baseline)")
start = time.time()
results_sequential = []
cf_seq = CurveFit()
sequential_runs = min(20 if QUICK else 100, n_datasets)
for i in range(sequential_runs):  # Only fit a subset for speed
    popt, _ = cf_seq.curve_fit(
        exponential_model, x_batch_data, y_batch_data[i], p0=[3.0, 0.5], maxiter=30
    )
    results_sequential.append(popt)
time_sequential = time.time() - start

print(
    f"  Time for {sequential_runs} datasets: {time_sequential * 1000:.0f} ms "
    f"({time_sequential * 1000 / sequential_runs:.1f} ms/fit)"
)
print(
    f"  Estimated time for {n_datasets}: {time_sequential * n_datasets / sequential_runs:.1f} s"
)
print()

# Method 2: Vectorized with vmap (fast)
print("Method 2: Batched fitting with vmap (optimized)")


# Simplified optimizer for vectorization
def fit_one_dataset(y_single):
    """Fit single dataset (simplified gradient descent)."""
    params = jnp.array([3.0, 0.5])

    def loss(p):
        return jnp.sum((y_single - exponential_model(x_batch_data, *p)) ** 2)

    # A few gradient descent steps for demonstration
    for _ in range(5 if QUICK else 20):
        g = jax.grad(loss)(params)
        params = params - 0.05 * g
    return params


# Vectorize over batch dimension
fit_batch = jit(vmap(fit_one_dataset))

# Warm up JIT
warmup_size = min(10, n_datasets)
_ = fit_batch(y_batch_data[:warmup_size])

# Benchmark
start = time.time()
results_batch = fit_batch(y_batch_data)
# Block until computation completes (JAX is async)
results_batch[0].block_until_ready()
time_batch = time.time() - start

print(
    f"  Time for {n_datasets} datasets: {time_batch * 1000:.0f} ms ({time_batch * 1000 / n_datasets:.3f} ms/fit)"
)
print(f"  Throughput: {n_datasets / time_batch:.0f} fits/second")
print()

# Speedup
estimated_sequential_time = time_sequential * n_datasets / sequential_runs
speedup = estimated_sequential_time / time_batch

print(f"Speedup: {speedup:.0f}x faster with vmap + JIT ✓")
print()
print("Key insight: vmap parallelizes across datasets, JIT compiles once")


# ======================================================================
# ## Part 4: Memory Optimization
#
# Avoiding out-of-memory (OOM) errors with large datasets.
# ======================================================================


# Memory optimization strategies

print("Memory Optimization Strategies:")
print("=" * 60)
print()

print("1. Use float32 instead of float64:")
x_f64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
x_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
print(f"   float64 memory: {x_f64.nbytes} bytes per element")
print(f"   float32 memory: {x_f32.nbytes} bytes per element")
print(f"   Savings: {(1 - x_f32.nbytes / x_f64.nbytes) * 100:.0f}%")
print("   → Use float32 unless high precision is critical\n")

print("2. Process data in chunks (streaming):")
print("   # For very large datasets (millions of points)")
print("   chunk_size = 100000")
print("   for i in range(0, len(data), chunk_size):")
print("       chunk = data[i:i+chunk_size]")
print("       result = fit(chunk)")
print("       results.append(result)\n")

print("3. Clear JAX cache if needed:")
print("   from jax import clear_caches")
print("   clear_caches()  # Frees compilation cache\n")

print("4. Monitor memory usage:")


def get_array_memory_mb(arr):
    return arr.nbytes / (1024**2)


large_array = jnp.ones((cap_samples(10000), cap_samples(1000)), dtype=jnp.float32)
print(
    f"   Example: {large_array.shape} array uses {get_array_memory_mb(large_array):.1f} MB"
)
print()

print("5. Typical memory requirements:")
print("   10K points:     ~0.1 MB (negligible)")
print("   1M points:      ~10 MB (easy)")
print("   100M points:    ~1 GB (manageable)")
print("   1B points:      ~10 GB (need chunking or distributed)")
print()
print("→ For datasets >100M points, use chunked processing or streaming")


# ======================================================================
# ## Part 5: Performance Benchmarking
#
# Systematic performance measurement and optimization.
# ======================================================================


# Comprehensive performance benchmark


def benchmark_nlsq(n_points_list, n_params=2, n_runs=5):
    """Benchmark NLSQ across different problem sizes.

    Parameters
    ----------
    n_points_list : list
        List of dataset sizes to test
    n_params : int
        Number of parameters to fit
    n_runs : int
        Number of runs to average

    Returns
    -------
    results : dict
        Benchmark results
    """
    results = {"n_points": [], "mean_time_ms": [], "std_time_ms": []}

    cf_bench = CurveFit()

    for n_points in n_points_list:
        x = jnp.linspace(0, 5, n_points)
        y = 3.0 * jnp.exp(-0.5 * x) + np.random.normal(0, 0.1, n_points)

        # Warm up
        _ = cf_bench.curve_fit(exponential_model, x, y, p0=[2.0, 0.3], maxiter=20)

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.time()
            popt, _ = cf_bench.curve_fit(
                exponential_model, x, y, p0=[2.0, 0.3], maxiter=20
            )
            # Note: popt is numpy array (already synchronous), no need for block_until_ready
            times.append((time.time() - start) * 1000)

        results["n_points"].append(n_points)
        results["mean_time_ms"].append(np.mean(times))
        results["std_time_ms"].append(np.std(times))

    return results


print("Running comprehensive benchmark...")
print("(This may take 30-60 seconds in full mode)")
print()

# Test different problem sizes
size_candidates = [50, 100, 200] if QUICK else [100, 500, 1000, 5000, 10000]
sizes = sorted({cap_samples(s) for s in size_candidates})
bench_results = benchmark_nlsq(sizes, n_runs=2 if QUICK else 5)

# Display results
print("Benchmark Results:")
print("=" * 60)
print(f"{'N Points':<12} {'Mean Time (ms)':<20} {'Throughput (fits/s)'}")
print("-" * 60)

for i, n in enumerate(bench_results["n_points"]):
    mean_t = bench_results["mean_time_ms"][i]
    std_t = bench_results["std_time_ms"][i]
    throughput = 1000 / mean_t
    print(f"{n:<12} {mean_t:>8.2f} ± {std_t:<8.2f} {throughput:>12.1f}")

print()

# Plot scaling
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Time vs problem size
ax1.errorbar(
    bench_results["n_points"],
    bench_results["mean_time_ms"],
    yerr=bench_results["std_time_ms"],
    marker="o",
    capsize=5,
    label="NLSQ",
)
ax1.set_xlabel("Number of Data Points")
ax1.set_ylabel("Time (ms)")
ax1.set_title("Performance Scaling")
ax1.legend()
ax1.grid(alpha=0.3)

# Log-log plot to see scaling behavior
ax2.loglog(bench_results["n_points"], bench_results["mean_time_ms"], "o-", label="NLSQ")
ax2.set_xlabel("Number of Data Points")
ax2.set_ylabel("Time (ms)")
ax2.set_title("Scaling Behavior (log-log)")
ax2.legend()
ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "gpu_optimization_deep_dive"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()

print("Interpretation:")
print("  - Nearly flat scaling: Well-optimized (GPU benefits)")
print("  - Linear scaling: Expected for iterative optimization")
print("  - Superlinear scaling: May indicate memory issues or poor caching")


# ======================================================================
# ## Summary and Best Practices
#
# ### Performance Optimization Checklist
#
# **For Maximum Speed:**
#
# 1. ✅ **Use GPU** if available (5-50x speedup for large problems)
# 2. ✅ **Keep array shapes consistent** to avoid recompilation
# 3. ✅ **Use float32** unless high precision is needed (2x memory savings)
# 4. ✅ **Batch process** with `vmap` for multiple datasets (10-100x faster)
# 5. ✅ **Warm up JIT** with small dataset before benchmarking
# 6. ✅ **Use `block_until_ready()`** when timing (JAX is async)
#
# **For Large Datasets:**
#
# 1. ✅ **Chunk data** if >100M points
# 2. ✅ **Monitor memory** usage
# 3. ✅ **Consider downsampling** for smooth, oversampled data
# 4. ✅ **Use streaming** for datasets that don't fit in memory
#
# ### Performance Expectations
#
# | **Scenario** | **Typical Time** | **Optimization** |
# |--------------|------------------|------------------|
# | First call (cold start) | 0.5-2 seconds | Expected (JIT compilation) |
# | Subsequent calls (warm) | 1-50 ms | Cached compilation |
# | Large dataset (10K points) | 5-100 ms | Use GPU if available |
# | Batch (1000 fits) | 100-5000 ms | Use vmap for parallelization |
# | Huge dataset (1M points) | 50-500 ms | GPU + chunking |
#
# ### Troubleshooting Performance Issues
#
# **Problem**: First call is slow (>5 seconds)
# - **Solution**: Normal for JIT. Subsequent calls will be fast.
#
# **Problem**: All calls are slow (>1 second for small data)
# - **Solution**: Check if recompiling each time (varying shapes/dtypes)
#
# **Problem**: Out of memory errors
# - **Solution**: Use float32, chunk data, or downsample
#
# **Problem**: GPU not being used
# - **Solution**: Check `jax.devices()`, install jax[cuda] or jax[rocm]
#
# **Problem**: Batch processing not faster than sequential
# - **Solution**: Problem may be too small, try larger batches or datasets
#
# ### Advanced Profiling
#
# For detailed profiling:
#
# ```python
# # JAX profiling (requires jax[profiling])
# import jax.profiler
#
# # Profile a code block
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
#     # Your NLSQ code here
#     popt, pcov = cf.curve_fit(model, x, y, p0=...)
#
# # Opens profiling UI in browser
# ```
#
# ### Production Recommendations
#
# ```python
# # Example: Optimized production setup
# import jax
# import jax.numpy as jnp
# from nlsq import CurveFit
#
# # Configure JAX for production
# jax.config.update('jax_enable_x64', False)  # Use float32
#
# # Pre-warm JIT cache at startup
# cf = CurveFit()
# x_dummy = jnp.linspace(0, 1, 100)
# y_dummy = jnp.ones(100)
# _ = cf.curve_fit(model, x_dummy, y_dummy, p0=initial_guess)
#
# # Now ready for fast production fitting
# ```
#
# ### Next Steps
#
# - **Scale up**: Try batch processing 10,000+ datasets with vmap
# - **Optimize models**: Simplify model functions for faster evaluation
# - **Profile**: Use JAX profiler to identify bottlenecks
# - **Distribute**: For massive scale, consider JAX's `pmap` for multi-GPU
#
# ### References
#
# 1. **JAX Performance**: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
# 2. **JAX Profiling**: https://jax.readthedocs.io/en/latest/profiling.html
# 3. **GPU Acceleration**: https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
# 4. **Related examples**:
#    - `custom_algorithms_advanced.ipynb` - vmap for batch fitting
#    - `troubleshooting_guide.ipynb` - Performance debugging
#
# ---
#
# **Remember**: Premature optimization is the root of all evil. Profile first, optimize what matters!
# ======================================================================
