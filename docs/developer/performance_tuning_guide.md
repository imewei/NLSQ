# NLSQ Performance Tuning Guide

**For Users**: How to get the best performance from NLSQ
**Last Updated**: December 2025

---

## Recent Optimizations (v0.4.2+)

NLSQ has received significant performance improvements:

### Lazy Imports (43% Faster Cold Start)

Specialty modules are now lazily imported, reducing initial import time from ~1084ms to ~620ms:

```python
# These modules only load when first accessed:
# - nlsq.global_optimization
# - nlsq.streaming_optimizer
# - nlsq.profiler_visualization
# - nlsq.gui

import nlsq  # Fast (~620ms)

nlsq.curve_fit(...)  # Core functionality loads immediately

# Streaming loads only when needed
nlsq.StreamingOptimizer(...)  # Lazy load happens here
```

### Vectorized Sparse Jacobian (37-50x Speedup)

Sparse Jacobian construction now uses vectorized NumPy operations:

```python
# Old: O(nm) nested loop - slow for large matrices
# New: O(nnz) COO sparse construction - much faster

# 100k x 50 matrix: ~200ms → ~5ms (40x speedup)
```

### LRU Memory Pool

Memory pool now uses LRU eviction with adaptive TTL:

```python
from nlsq.caching.memory_manager import MemoryManager

manager = MemoryManager()
# Arrays are cached and reused
# LRU eviction when pool exceeds max_arrays
manager.optimize_memory_pool(max_arrays=10)
```

---

## Quick Start

NLSQ is already highly optimized and should provide excellent performance out of the box. In most cases, **no tuning is needed**.

**Typical Performance**:
- 100-point fit: ~30ms (after initial JIT compilation)
- 1000-point fit: ~110ms
- 10000-point fit: ~134ms
- 50000-point fit: ~120ms

**Scaling**: 50x more data → only 1.2x slower ✅

---

## Understanding NLSQ Performance

### First Run vs Subsequent Runs

**First run includes JIT compilation**:
```python
from nlsq import curve_fit

# First call: ~430ms (includes ~400ms JIT compilation)
popt1, pcov1 = curve_fit(model, x, y, p0=[1, 1])

# Second call: ~30ms (uses cached compiled function)
popt2, pcov2 = curve_fit(model, x2, y2, p0=[1, 1])
```

**Solution**: JIT compilation is one-time cost, subsequent calls are much faster.

### GPU vs CPU

**Automatic Backend Selection**:
```python
import jax

print(jax.devices())  # Check which devices are available

# NLSQ automatically uses GPU/TPU if available
popt, pcov = curve_fit(model, x, y)  # Runs on GPU automatically
```

**Force CPU** (for debugging or small problems):
```bash
JAX_PLATFORM_NAME=cpu python your_script.py
```

**GPU Benefits**:
- Most noticeable for large problems (>10,000 points)
- Parallel computation of Jacobians
- Faster linear algebra operations

---

## Optimization Techniques

### 1. Reuse Compiled Functions (Highest Impact)

**Problem**: Creating new `curve_fit` calls triggers recompilation

**Solution**: Use `CurveFit` class to reuse compiled functions

```python
from nlsq import CurveFit

# BAD: Recompiles for each fit
for dataset in datasets:
    popt, pcov = curve_fit(model, dataset.x, dataset.y)  # Slow!

# GOOD: Compile once, reuse many times
cf = CurveFit()
for dataset in datasets:
    popt, pcov = cf.curve_fit(model, dataset.x, dataset.y)  # Fast!
```

**Speedup**: 10-100x for batch fitting (avoids repeated JIT compilation)

### 2. Batch Processing

**Problem**: Fitting curves one at a time in a loop

**Solution**: Process multiple fits efficiently

```python
# BAD: Sequential processing
results = []
for i in range(n_curves):
    popt, pcov = cf.curve_fit(model, x_data[i], y_data[i])
    results.append(popt)

# BETTER: Reuse CurveFit instance (as shown above)
cf = CurveFit()
results = []
for i in range(n_curves):
    popt, pcov = cf.curve_fit(model, x_data[i], y_data[i])
    results.append(popt)

# BEST: Use large_dataset module for very large batches
from nlsq.streaming.large_dataset import LargeDatasetFitter

fitter = LargeDatasetFitter()
results = fitter.fit_multiple(model, x_data, y_data, p0_list)
```

### 3. Provide Good Initial Guesses

**Problem**: Poor initial guess → more iterations → slower convergence

**Solution**: Provide reasonable `p0` parameter

```python
# BAD: No initial guess (uses zeros)
popt, pcov = curve_fit(exponential, x, y)  # May take many iterations

# GOOD: Reasonable initial guess
p0 = [max(y), 1.0, min(y)]  # Amplitude, decay rate, offset
popt, pcov = curve_fit(exponential, x, y, p0=p0)  # Faster convergence
```

**Speedup**: 2-5x for well-conditioned problems

### 4. Use Bounds When Appropriate

**Problem**: Unbounded optimization may explore unrealistic parameter space

**Solution**: Provide reasonable bounds

```python
# Example: Exponential decay
# y = a * exp(-b * x) + c
# We know: a > 0, b > 0, c >= 0

bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
popt, pcov = curve_fit(exponential, x, y, p0=p0, bounds=bounds)
```

**Benefits**:
- Faster convergence (avoids unrealistic regions)
- More robust (prevents numerical issues)

### 5. Choose Appropriate Algorithm

**TRF (default)**: Best for bounded problems
```python
popt, pcov = curve_fit(model, x, y, method="trf", bounds=bounds)
```

**LM (Levenberg-Marquardt)**: Best for unbounded problems
```python
popt, pcov = curve_fit(model, x, y, method="lm")  # Slightly faster for unconstrained
```

**Dogbox**: Alternative for bounded problems
```python
popt, pcov = curve_fit(model, x, y, method="dogbox", bounds=bounds)
```

### 6. Reduce Data When Possible

**Problem**: Fitting millions of data points when thousands would suffice

**Solution**: Downsample if appropriate for your problem

```python
# If you have 1M points but only fitting 5 parameters
if len(x) > 10000:
    # Downsample intelligently
    indices = np.linspace(0, len(x) - 1, 10000, dtype=int)
    x_reduced = x[indices]
    y_reduced = y[indices]
    sigma_reduced = sigma[indices] if sigma is not None else None

    popt, pcov = curve_fit(model, x_reduced, y_reduced, sigma=sigma_reduced)
```

**Note**: Only do this if statistically valid for your application!

---

## Profiling Your Workload

### Basic Timing

```python
import time
from nlsq import CurveFit

cf = CurveFit()

# Time first call (includes JIT)
start = time.time()
popt1, pcov1 = cf.curve_fit(model, x, y, p0=p0)
first_call = time.time() - start

# Time second call (cached)
start = time.time()
popt2, pcov2 = cf.curve_fit(model, x2, y2, p0=p0)
second_call = time.time() - start

print(f"First call (with JIT): {first_call*1000:.1f}ms")
print(f"Second call (cached): {second_call*1000:.1f}ms")
print(f"Speedup: {first_call/second_call:.1f}x")
```

### Detailed Profiling

```python
import cProfile
import pstats

# Profile your code
profiler = cProfile.Profile()
profiler.enable()

# Your fitting code here
popt, pcov = curve_fit(model, x, y, p0=p0)

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)  # Top 20 functions
```

### Using pytest-benchmark

```python
# In your test file
def test_fitting_performance(benchmark):
    """Benchmark curve fitting performance"""
    x = np.linspace(0, 10, 1000)
    y = 2.0 * np.exp(-0.5 * x) + 0.3 + 0.05 * np.random.randn(len(x))
    p0 = [2.0, 0.5, 0.3]

    result = benchmark(curve_fit, exponential, x, y, p0=p0)
    popt, pcov = result

    assert len(popt) == 3
```

Run with:
```bash
pytest test_performance.py --benchmark-only
```

---

## Common Performance Issues

### Issue 1: Slow First Call

**Symptom**: First `curve_fit` call takes 200-500ms

**Cause**: JIT compilation overhead

**Solution**: ✅ This is normal and expected
- Subsequent calls will be much faster (~10-50ms)
- Use `CurveFit` class to reuse compiled functions
- Consider warming up the JIT cache on startup

```python
# Warm up JIT cache
cf = CurveFit()
_ = cf.curve_fit(model, x_dummy, y_dummy, p0=p0_dummy)
# Now real fits will be fast
```

### Issue 2: Each Fit Is Slow

**Symptom**: Every call to `curve_fit` takes 200+ ms

**Diagnosis**:
1. Are you recreating the function each time?
2. Are you using different model functions?
3. Is your model function slow?

**Solutions**:
```python
# Make sure you're reusing CurveFit instance
cf = CurveFit()  # Create ONCE
for data in datasets:
    popt, pcov = cf.curve_fit(model, data.x, data.y)  # Reuse

# Profile your model function
import jax.numpy as jnp


@jit  # JIT compile your model
def fast_model(x, a, b, c):
    return a * jnp.exp(-b * x) + c  # Use jnp, not np!
```

### Issue 3: Large Dataset Performance

**Symptom**: Fitting >100,000 points is very slow

**Solution**: Use large dataset optimization features

```python
from nlsq.streaming.large_dataset import LargeDatasetFitter

fitter = LargeDatasetFitter(
    chunk_size=10000, enable_streaming=True  # Process in chunks
)

popt, pcov = fitter.fit(model, x, y, p0=p0)
```

### Issue 4: Fitting Doesn't Converge

**Symptom**: Function takes very long, doesn't converge

**Cause**: Poor initial guess or ill-conditioned problem

**Solutions**:
1. Provide better initial guess
2. Use bounds to constrain search
3. Scale your data
4. Increase max iterations (if needed)

```python
# Scale data to reasonable range
x_scaled = (x - x.mean()) / x.std()
y_scaled = (y - y.mean()) / y.std()

# Fit on scaled data
popt_scaled, pcov_scaled = curve_fit(model, x_scaled, y_scaled, p0=p0)

# Transform parameters back to original scale
popt = transform_params(popt_scaled, x.mean(), x.std(), y.mean(), y.std())
```

---

## Advanced Optimization

### Sparse Jacobian (For Specific Problems)

If your Jacobian has sparse structure, exploit it:

```python
from nlsq.sparse_jacobian import SparseCurveFit

# Define sparsity pattern
# (only if you know your Jacobian is sparse!)
scf = SparseCurveFit(sparsity_pattern=pattern)
popt, pcov = scf.curve_fit(model, x, y, p0=p0)
```

**Speedup**: 2-10x for problems with sparse Jacobians

### Custom Jacobian

If you can provide analytical Jacobian:

```python
def jac_analytical(x, a, b, c):
    """Analytical Jacobian for a*exp(-b*x) + c"""
    J = np.zeros((len(x), 3))
    exp_term = np.exp(-b * x)
    J[:, 0] = exp_term  # d/da
    J[:, 1] = -a * x * exp_term  # d/db
    J[:, 2] = 1.0  # d/dc
    return J


popt, pcov = curve_fit(model, x, y, p0=p0, jac=jac_analytical)
```

**Note**: JAX's autodiff is usually fast enough. Only provide custom Jacobian if:
- You have analytical form
- It's significantly simpler than automatic differentiation
- Profiling shows Jacobian computation is bottleneck

---

## Benchmarking Checklist

Before claiming "NLSQ is slow":

- [ ] Are you using `CurveFit` class for multiple fits?
- [ ] Have you excluded JIT compilation time from measurements?
- [ ] Is your model function JIT-compiled and using JAX operations?
- [ ] Are you providing reasonable initial guesses?
- [ ] Is your problem well-conditioned?
- [ ] Have you profiled to identify the actual bottleneck?
- [ ] Are you comparing fair to fair (NLSQ on CPU vs SciPy on CPU)?

---

## Performance Expectations

### What is Fast?

For reference, here are typical performance numbers on modern CPU:

| Problem Size | Points | Parameters | Expected Time (after JIT) |
|--------------|--------|------------|---------------------------|
| Small | 100 | 2-5 | 10-30ms |
| Medium | 1,000 | 2-5 | 50-150ms |
| Large | 10,000 | 2-5 | 100-200ms |
| XLarge | 50,000 | 2-5 | 100-300ms |
| Huge | 100,000+ | 2-5 | Use large_dataset module |

**GPU acceleration** can provide 2-10x additional speedup for large problems.

### When to Use GPU

**GPU is beneficial when**:
- Problem size > 10,000 points
- Batch fitting many curves
- Complex model functions
- Large Jacobian matrices

**GPU may not help when**:
- Problem size < 1,000 points (overhead dominates)
- Simple model functions
- JIT compilation dominates (first run)

---

## Getting Help

If you're experiencing performance issues:

1. **Profile first**: Identify the actual bottleneck
2. **Check the basics**: CurveFit class, good initial guess, etc.
3. **Review case study**: `docs/optimization_case_study.md`
4. **Open an issue**: With profiling data and minimal reproducible example

**Template for performance issues**:
```python
import numpy as np
from nlsq import CurveFit
import time


# Your model
def model(x, a, b):
    return a * x + b


# Your data
x = np.linspace(0, 10, 1000)
y = 2.0 * x + 1.0 + 0.1 * np.random.randn(len(x))

# Timing
cf = CurveFit()

# First call (with JIT)
start = time.time()
popt1, pcov1 = cf.curve_fit(model, x, y, p0=[1, 0])
first = time.time() - start

# Second call (cached)
start = time.time()
popt2, pcov2 = cf.curve_fit(model, x, y, p0=[1, 0])
second = time.time() - start

print(f"First: {first*1000:.1f}ms, Second: {second*1000:.1f}ms")
print(f"Expected: First ~400ms, Second ~30ms")
```

---

## Summary

**Key Takeaways**:

1. ✅ **NLSQ is already fast** - Well-optimized, excellent scaling
2. ✅ **Use CurveFit class** - Reuse compiled functions (biggest impact)
3. ✅ **Good initial guesses** - Faster convergence
4. ✅ **Profile before optimizing** - Identify actual bottlenecks
5. ✅ **GPU for large problems** - Automatic acceleration when beneficial

**Remember**: Premature optimization is the root of all evil. Profile first, optimize only what matters.

---

**For More Information**:
- Optimization case study: `docs/optimization_case_study.md`
- Benchmark suite: `benchmarks/test_performance_regression.py`
- Examples: `examples/` directory

**Last Updated**: December 2025
