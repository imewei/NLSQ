# NLSQ Examples

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)
[![NLSQ Version](https://img.shields.io/badge/nlsq-0.2.0+-orange.svg)](https://github.com/imewei/NLSQ)

> **⚠️ v0.2.0 Update**: Notebooks updated for NLSQ v0.2.0 with streaming optimization replacing subsampling for zero accuracy loss. See [MIGRATION_V0.2.0.md](../MIGRATION_V0.2.0.md) for details.

Welcome to the NLSQ examples repository! This collection provides comprehensive, interactive tutorials for learning and mastering GPU-accelerated nonlinear least squares curve fitting with JAX.

---

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Available Examples](#available-examples)
- [Learning Paths](#learning-paths)
- [Which Example Should I Use?](#which-example-should-i-use)
- [Setup Instructions](#setup-instructions)
- [Performance Optimization Guide](#performance-optimization-guide)
- [Additional Resources](#additional-resources)

---

## 🚀 Quick Start

**New to NLSQ?** Start here:

```bash
# 1. Install NLSQ
pip install nlsq

# 2. Open your first notebook
jupyter notebook nlsq_quickstart.ipynb

# Or run in Google Colab (click badge in notebook)
```

**Already familiar with NLSQ?** Jump to:
- [Advanced Features](#2-advanced-features-demo) for optimization techniques
- [Performance Optimization](#5-performance-optimization-demo) for maximum performance
- [Large Datasets](#3-large-dataset-demo) for handling big data

---

## 📖 Available Examples

### 1. **NLSQ Quickstart**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/nlsq_quickstart.ipynb)

**File**: `nlsq_quickstart.ipynb` | **Level**: Beginner | **Time**: 15-20 min

**What you'll learn:**
- Basic `curve_fit()` usage compatible with SciPy
- Memory management with `MemoryConfig`
- Performance comparisons with SciPy
- Avoiding JAX retracing with fixed array sizes
- Fitting multiple functions efficiently

**Key topics:**
- ✓ Linear and exponential models
- ✓ Memory estimation and configuration
- ✓ CurveFit class for reusable compiled functions
- ✓ GPU acceleration basics
- ✓ Speed comparisons (150-270x faster than SciPy)

**Perfect for**: First-time users, SciPy migrants, quick prototyping

---

### 2. **Advanced Features Demo**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/advanced_features_demo.ipynb)

**File**: `advanced_features_demo.ipynb` | **Level**: Intermediate | **Time**: 30-40 min

**What you'll learn:**
- Diagnostics and convergence monitoring
- Error recovery mechanisms
- Numerical stability analysis
- Smart caching for repeated computations
- Automatic algorithm selection
- Robust decomposition methods

**Key topics:**
- ✓ `AlgorithmSelector` for optimal method choice
- ✓ `auto_select_algorithm()` automatic tuning
- ✓ Convergence diagnostics
- ✓ Handling ill-conditioned problems
- ✓ Recovery from optimization failures
- ✓ Input validation and sanitization

**Perfect for**: Production applications, robustness requirements, complex models

---

### 3. **Large Dataset Demo**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/large_dataset_demo.ipynb)

**File**: `large_dataset_demo.ipynb` | **Level**: Intermediate | **Time**: 25-35 min

**What you'll learn:**
- Memory estimation for datasets from 100K to 100M+ points
- Automatic chunking for large datasets
- Sampling strategies for extremely large data
- Progress reporting for long-running fits
- Memory-aware optimization

**Key topics:**
- ✓ `fit_large_dataset()` convenience function
- ✓ `curve_fit_large()` with automatic size detection
- ✓ `LargeDatasetFitter` class for chunked processing
- ✓ `large_dataset_context()` for temporary config
- ✓ HDF5 file integration
- ✓ Chunking vs sampling strategies

**Perfect for**: Large-scale data analysis, memory-constrained systems, batch processing

---

### 4. **2D Gaussian Demo**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/nlsq_2d_gaussian_demo.ipynb)

**File**: `nlsq_2d_gaussian_demo.ipynb` | **Level**: Intermediate | **Time**: 20-30 min

**What you'll learn:**
- 2D Gaussian fitting with rotation
- Multi-dimensional parameter optimization
- Memory management for image data
- GPU/CPU fallback handling
- Performance scaling with image size

**Key topics:**
- ✓ Coordinate rotation in 2D
- ✓ Flattening multi-dimensional data
- ✓ Memory estimation for 2D datasets
- ✓ Handling GPU errors gracefully
- ✓ Comparing NLSQ vs SciPy for 2D problems
- ✓ Residual analysis and visualization

**Perfect for**: Image processing, 2D spectroscopy, spatial data analysis

---

### 5. **Performance Optimization Demo** ⭐ NEW

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/performance_optimization_demo.ipynb)

**File**: `performance_optimization_demo.ipynb` | **Level**: Advanced | **Time**: 40-50 min

**What you'll learn:**
- **MemoryPool**: Zero-allocation optimization (2-5x speedup)
- **SparseJacobian**: 10-100x memory reduction for sparse problems
- **StreamingOptimizer**: Process unlimited dataset sizes
- When and how to use each feature
- Combining techniques for maximum performance

**Key topics:**
- ✓ Pre-allocated memory buffers
- ✓ Sparsity pattern detection and exploitation
- ✓ Batch streaming for disk-based data
- ✓ HDF5 integration for massive datasets
- ✓ Performance benchmarking
- ✓ Decision trees for optimization selection

**Perfect for**: Performance-critical applications, very large problems, embedded systems

---

## 🎓 Feature Demonstrations (Python Scripts)

**NEW in v0.1.1**: Phase 1-3 feature demonstrations

Located in `demos/` directory, these Python scripts showcase specific NLSQ v0.1.1 features:

### Phase 1 Features (Quick Wins)

1. **Result Enhancements** (`demos/result_enhancements_demo.py`)
   - Enhanced `.plot()`, `.summary()`, `.confidence_intervals()`
   - Statistical metrics: R², RMSE, AIC, BIC
   - Backward compatible tuple unpacking

2. **Callbacks** (`demos/callbacks_demo.py`)
   - `ProgressBar` - Real-time progress monitoring
   - `EarlyStopping` - Automatic termination
   - `IterationLogger` - Optimization logging

3. **Function Library** (`demos/function_library_demo.py`)
   - Pre-built models: exponential, gaussian, sigmoid
   - Automatic p0 estimation
   - Smart parameter bounds

4. **Enhanced Error Messages** (`demos/enhanced_error_messages_demo.py`)
   - Actionable diagnostics
   - Clear recommendations
   - Better debugging

**Usage**:
```bash
python examples/demos/callbacks_demo.py
```

**Full Documentation**: See [demos/README.md](demos/README.md)

---

## 🎯 Learning Paths

Choose your path based on your goals:

### Path 1: Quick Start (Total: ~45 min)
**Goal**: Get up and running fast

```
1. NLSQ Quickstart (15 min)
   └─> Basic fitting, memory config, performance

2. 2D Gaussian Demo (20 min)
   └─> Multi-dimensional fitting

3. Start using NLSQ! 🚀
```

### Path 2: Production Applications (Total: ~90 min)
**Goal**: Build robust, production-ready systems

```
1. NLSQ Quickstart (15 min)
   └─> Basics

2. Advanced Features Demo (35 min)
   └─> Diagnostics, recovery, validation

3. Large Dataset Demo (30 min)
   └─> Memory management, chunking

4. Deploy with confidence! ✅
```

### Path 3: Performance Optimization (Total: ~120 min)
**Goal**: Maximum performance for challenging problems

```
1. NLSQ Quickstart (15 min)
   └─> Basics

2. Advanced Features Demo (35 min)
   └─> Algorithm selection

3. Large Dataset Demo (30 min)
   └─> Scaling strategies

4. Performance Optimization Demo (45 min)
   └─> MemoryPool, SparseJacobian, Streaming

5. Optimize everything! ⚡
```

### Path 4: Comprehensive Mastery (Total: ~150 min)
**Goal**: Master all NLSQ capabilities

```
1. NLSQ Quickstart (15 min)
2. Advanced Features Demo (35 min)
3. Large Dataset Demo (30 min)
4. 2D Gaussian Demo (25 min)
5. Performance Optimization Demo (45 min)

→ You're now an NLSQ expert! 🎓
```

---

## 🧭 Which Example Should I Use?

### By Problem Size

| Data Points | Recommended Example | Key Features |
|-------------|-------------------|--------------|
| < 1,000 | Quickstart | Basic usage, GPU acceleration |
| 1K - 10K | Quickstart or 2D Gaussian | Standard optimization |
| 10K - 100K | Large Dataset Demo | Memory management |
| 100K - 1M | Large Dataset + Advanced | Chunking, diagnostics |
| 1M - 10M | Large Dataset + Performance | Sparse Jacobian, pooling |
| > 10M or disk | Performance Optimization | Streaming, HDF5 |

### By Use Case

| Use Case | Recommended Example | Why |
|----------|-------------------|-----|
| **First time user** | Quickstart | Gentle introduction |
| **Replacing SciPy** | Quickstart | API compatibility demo |
| **Image/spatial data** | 2D Gaussian | Multi-dimensional fitting |
| **Production deployment** | Advanced Features | Robustness, diagnostics |
| **Memory constraints** | Large Dataset | Chunking, sampling |
| **Real-time systems** | Performance Optimization | MemoryPool, low latency |
| **Sparse problems** | Performance Optimization | SparseJacobian |
| **Huge datasets** | Performance Optimization | StreamingOptimizer |
| **Multi-component models** | Advanced + Performance | Algorithm selection, sparsity |

### By Experience Level

**Beginner** (new to NLSQ or curve fitting):
1. Start with **Quickstart**
2. Try **2D Gaussian** for multi-dimensional problems
3. Explore **Advanced Features** when ready

**Intermediate** (familiar with optimization):
1. Quick review of **Quickstart**
2. Deep dive into **Advanced Features**
3. Study **Large Dataset** for scaling
4. Explore **Performance Optimization** selectively

**Advanced** (optimization expert):
1. Skim **Quickstart** for NLSQ specifics
2. Focus on **Performance Optimization**
3. Reference **Advanced Features** and **Large Dataset** as needed

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.12 or higher** (required)
- **JAX** (automatically installed with NLSQ)
- **NumPy, SciPy** (standard scientific Python)
- **Matplotlib** (for visualizations)
- **Jupyter** (for notebooks) or **Google Colab** (cloud)

### Local Installation

```bash
# Create virtual environment (recommended)
python3.12 -m venv nlsq-env
source nlsq-env/bin/activate  # On Windows: nlsq-env\Scripts\activate

# Install NLSQ with all dependencies
pip install nlsq

# Install Jupyter (if not using Colab)
pip install jupyter

# Clone examples (or download from GitHub)
git clone https://github.com/imewei/NLSQ.git
cd NLSQ/examples

# Launch Jupyter
jupyter notebook
```

### Google Colab (Cloud)

No installation needed! Click the **"Open in Colab"** badge in any notebook.

**Colab setup cell** (run first):
```python
# Install NLSQ in Colab
# !pip install nlsq

# Import and verify
import nlsq

print(f"NLSQ version: {nlsq.__version__}")
```

### GPU Setup (Optional)

NLSQ automatically detects and uses GPUs when available.

**Check GPU availability:**
```python
import jax

print(f"JAX devices: {jax.devices()}")
# Output: [cuda(id=0)] if GPU available
```

**Force CPU (for debugging):**
```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"
```

---

## ⚡ Performance Optimization Guide

This guide explains when and how to use NLSQ's advanced performance optimization features: **MemoryPool**, **SparseJacobian**, and **StreamingOptimizer**.

### Overview

NLSQ provides three advanced features for performance-critical applications:

| Feature | Purpose | Typical Speedup | Memory Reduction |
|---------|---------|-----------------|------------------|
| **MemoryPool** | Reuse pre-allocated buffers | 2-5x (iterations) | 90-99% allocations |
| **SparseJacobian** | Exploit sparsity patterns | 1-3x (compute) | 10-100x (memory) |
| **StreamingOptimizer** | Process unlimited data | N/A | Unlimited |

**When to Optimize:**

✅ **Optimize when you have:**
- Very large datasets (>100K points)
- Memory constraints
- Repeated fitting operations
- Real-time/low-latency requirements
- Sparse problem structure

❌ **Don't optimize prematurely:**
- Profile first to identify bottlenecks
- Standard `CurveFit` handles most cases well
- Optimization adds complexity

---

### MemoryPool - Zero-Allocation Optimization

**What it does:** Pre-allocates and reuses array buffers to eliminate allocation overhead during optimization iterations.

**Use when:**
- ✅ Fitting the same model many times (>10 fits)
- ✅ Optimization has many iterations (>100)
- ✅ Low-latency requirements (real-time systems)
- ✅ Memory allocation shows up in profiling

**Don't use when:**
- ❌ Single fit operations
- ❌ Variable array sizes between fits
- ❌ Memory is abundant and allocation is fast

**Basic Usage:**

```python
from nlsq import CurveFit, MemoryPool

# Method 1: Context manager (recommended)
with MemoryPool(max_pool_size=10, enable_stats=True) as pool:
    cf = CurveFit()
    for dataset in datasets:
        popt, pcov = cf.curve_fit(model, *dataset)

    # Check statistics
    stats = pool.get_stats()
    print(f"Reuse rate: {stats['reuse_rate']:.1%}")

# Method 2: Manual management
pool = MemoryPool(max_pool_size=10)
try:
    arr = pool.allocate((1000, 10))
    # ... work ...
    pool.release(arr)
finally:
    pool.clear()
```

**Performance Example:**

```python
import time
import numpy as np

# Without pool
start = time.time()
for i in range(100):
    arr = np.zeros((10000, 50))
    _ = arr + 1.0
time_without = time.time() - start

# With pool
pool = MemoryPool(max_pool_size=5)
start = time.time()
for i in range(100):
    arr = pool.allocate((10000, 50))
    _ = arr + 1.0
    pool.release(arr)
time_with = time.time() - start

print(f"Speedup: {time_without / time_with:.2f}x")
# Typical output: Speedup: 2-5x
```

**Configuration:**

```python
pool = MemoryPool(
    max_pool_size=10,  # Max buffers per shape/dtype
    enable_stats=True,  # Track allocation statistics
)

# Get statistics
stats = pool.get_stats()
# Returns: {
#   'allocations': int,      # New allocations
#   'reuses': int,           # Reused from pool
#   'releases': int,         # Returned to pool
#   'peak_memory': int,      # Peak memory (bytes)
#   'reuse_rate': float,     # Fraction reused
#   'pool_sizes': dict,      # Buffers per shape
#   'currently_allocated': int,  # Active buffers
# }
```

---

### SparseJacobian - Memory Reduction for Large Problems

**What it does:** Exploits sparsity patterns in the Jacobian matrix to reduce memory usage by 10-100x for large-scale problems.

**Use when:**
- ✅ Jacobian has >90% sparsity (detected automatically)
- ✅ Very large problems (>100K points, >20 parameters)
- ✅ Memory-constrained environments
- ✅ Piecewise or multi-component models
- ✅ Localized parameter effects

**Don't use when:**
- ❌ Dense Jacobians (<50% sparsity)
- ❌ Small problems (<10K points)
- ❌ Simple global models
- ❌ Overhead outweighs benefits

**Sparsity Patterns:**

Sparse Jacobians occur in:

1. **Piecewise Models**: Different parameters for different regions
   ```python
   def piecewise_linear(x, a1, b1, a2, b2):
       # params [a1, b1] only affect x < 0.5
       # params [a2, b2] only affect x >= 0.5
       result = jnp.where(x < 0.5, a1 * x + b1, a2 * x + b2)
       return result
   ```

2. **Multi-Component Models**: Independent sub-models
   ```python
   def multi_gaussian(x, *params):
       # Each Gaussian (3 params) only affects local region
       n_gaussians = len(params) // 3
       result = 0
       for i in range(n_gaussians):
           a, mu, sigma = params[i * 3 : (i + 1) * 3]
           result += a * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))
       return result
   ```

3. **Localized Parameters**: Parameters affecting only nearby points (splines, local polynomials, kernel methods)

**Usage Example:**

```python
from nlsq import SparseJacobianComputer

# Step 1: Detect sparsity pattern
sparse_comp = SparseJacobianComputer(sparsity_threshold=0.01)

pattern, sparsity = sparse_comp.detect_sparsity_pattern(
    func=my_model,
    x0=initial_params,
    xdata_sample=x_data,
    n_samples=100,  # Sample size for detection
)

print(f"Detected sparsity: {sparsity:.1%}")

# Step 2: Estimate memory savings
memory_info = sparse_comp.estimate_memory_usage(
    n_data=1_000_000,
    n_params=50,
    sparsity=0.99,
)

print(f"Memory reduction: {memory_info['reduction_factor']:.1f}x")
print(f"Dense: {memory_info['dense_gb']:.2f} GB")
print(f"Sparse: {memory_info['sparse_gb']:.3f} GB")
```

**Memory Savings Table:**

| Data Points | Parameters | Sparsity | Dense Memory | Sparse Memory | Reduction |
|-------------|------------|----------|--------------|---------------|-----------|
| 10,000 | 10 | 90% | 0.76 MB | 0.15 MB | 5.0x |
| 100,000 | 20 | 95% | 15.26 MB | 1.53 MB | 10.0x |
| 1,000,000 | 50 | 99% | 381.47 MB | 9.54 MB | 40.0x |
| 10,000,000 | 100 | 99.5% | 7.45 GB | 0.09 GB | 82.7x |

**Visualization:**

```python
import matplotlib.pyplot as plt

# Visualize sparsity pattern
plt.imshow(pattern.T, aspect="auto", cmap="binary")
plt.xlabel("Data Point")
plt.ylabel("Parameter")
plt.title(f"Jacobian Sparsity Pattern ({sparsity:.1%} zeros)")
plt.colorbar(label="Non-zero")
plt.show()
```

---

### StreamingOptimizer - Unlimited Dataset Size

**What it does:** Processes data in batches without loading the full dataset into memory, enabling optimization on unlimited dataset sizes.

**Use when:**
- ✅ Dataset >10GB or doesn't fit in memory
- ✅ Data stored in files (HDF5, CSV) or databases
- ✅ Online/incremental learning scenarios
- ✅ Real-time data streams
- ✅ Distributed data sources

**Don't use when:**
- ❌ Data fits comfortably in memory
- ❌ Standard batch methods work well
- ❌ Need exact convergence guarantees
- ❌ Stochastic approximation not acceptable

**Basic Usage:**

```python
from nlsq import StreamingOptimizer, StreamingConfig

# Configure optimizer
config = StreamingConfig(
    batch_size=10000,  # Points per batch
    max_epochs=10,  # Training epochs
    learning_rate=0.01,  # Learning rate
    use_adam=True,  # Use Adam optimizer
    convergence_tol=1e-6,  # Convergence threshold
    warmup_steps=100,  # Learning rate warmup
    checkpoint_interval=1000,  # Save every N iterations
)

optimizer = StreamingOptimizer(config)


# Define model
def model(x, a, b, c):
    return a * np.exp(-b * x) + c


# Fit with streaming data
result = optimizer.fit_streaming(
    func=model,
    data_source="path/to/data.hdf5",  # Or generator
    p0=[1.0, 1.0, 0.0],
    bounds=None,
    verbose=1,
)

print(f"Final params: {result['x']}")
print(f"Final loss: {result['fun']:.6f}")
print(f"Epochs: {result['n_epochs']}")
print(f"Samples processed: {result['total_samples']:,}")
```

**Data Sources:**

1. **HDF5 Files** (recommended for large datasets):
   ```python
   import h5py

   # Create HDF5 dataset
   with h5py.File("large_data.h5", "w") as f:
       f.create_dataset("x", data=x_data, compression="gzip")
       f.create_dataset("y", data=y_data, compression="gzip")

   # Use with StreamingOptimizer
   result = optimizer.fit_streaming(model, "large_data.h5", p0)
   ```

2. **Custom Data Generators**:
   ```python
   class CustomDataGenerator:
       def generate_batches(self, batch_size):
           """Yield (x_batch, y_batch) tuples."""
           while True:
               x_batch = load_next_batch_x()
               y_batch = load_next_batch_y()
               yield x_batch, y_batch

       def close(self):
           """Cleanup resources."""
           pass


   generator = CustomDataGenerator()
   result = optimizer.fit_streaming(model, generator, p0)
   ```

**Configuration Options:**

```python
config = StreamingConfig(
    # Batch processing
    batch_size=10000,  # Points per batch
    max_epochs=10,  # Maximum epochs
    # Optimization
    learning_rate=0.01,  # Initial learning rate
    use_adam=True,  # Adam vs SGD+momentum
    momentum=0.9,  # SGD momentum (if not Adam)
    # Adam parameters
    adam_beta1=0.9,  # First moment decay
    adam_beta2=0.999,  # Second moment decay
    adam_eps=1e-8,  # Numerical stability
    # Regularization
    gradient_clip=10.0,  # Gradient clipping
    warmup_steps=100,  # Learning rate warmup
    # Convergence
    convergence_tol=1e-6,  # Convergence threshold
    checkpoint_interval=1000,  # Checkpoint frequency
)
```

**Memory Comparison:**

| Dataset Size | In-Memory | Streaming | Memory Reduction |
|--------------|-----------|-----------|------------------|
| 100 MB | 100 MB | ~10 MB | 10x |
| 1 GB | 1 GB | ~10 MB | 100x |
| 10 GB | OOM | ~10 MB | 1000x |
| 100 GB+ | OOM | ~10 MB | Unlimited |

---

### Decision Tree - Which Feature to Use?

```
START
│
├─ Dataset size?
│  ├─ <10K points ────────────────────────────────────> Use standard CurveFit
│  │
│  ├─ 10K-1M points
│  │  ├─ Sparse Jacobian (>90% sparsity)? ───────────> SparseJacobian
│  │  ├─ Many repeated fits? ────────────────────────> MemoryPool
│  │  └─ Standard case ───────────────────────────────> CurveFit
│  │
│  └─ >1M points or doesn't fit in memory
│     ├─ Sparse Jacobian (>90% sparsity)? ───────────> SparseJacobian + MemoryPool
│     ├─ Dataset fits in memory? ────────────────────> MemoryPool + CurveFit
│     └─ Dataset >10GB or on disk ───────────────────> StreamingOptimizer
│
└─ Special cases:
   ├─ Low-latency real-time ──────────────────────────> MemoryPool
   ├─ Online learning ────────────────────────────────> StreamingOptimizer
   └─ Memory-constrained embedded ────────────────────> SparseJacobian + MemoryPool
```

**Combination Strategies:**

Small problems (<10K points):
```python
from nlsq import CurveFit

cf = CurveFit()
popt, pcov = cf.curve_fit(model, x, y, p0)
```

Medium problems (10K-1M points, repeated fits):
```python
from nlsq import CurveFit, MemoryPool

with MemoryPool(enable_stats=True) as pool:
    cf = CurveFit()
    for dataset in datasets:
        popt, pcov = cf.curve_fit(model, *dataset)
```

Large sparse problems (>1M points, >90% sparsity):
```python
from nlsq import SparseJacobianComputer, MemoryPool

sparse_comp = SparseJacobianComputer()
pattern, sparsity = sparse_comp.detect_sparsity_pattern(model, p0, x_sample)

if sparsity > 0.9:
    with MemoryPool() as pool:
        # Use sparse-aware fitting
        pass
```

Very large streaming data (>10GB):
```python
from nlsq import StreamingOptimizer, StreamingConfig

config = StreamingConfig(batch_size=10000)
optimizer = StreamingOptimizer(config)
result = optimizer.fit_streaming(model, "data.hdf5", p0)
```

---

### Performance Benchmarks

**MemoryPool Benchmarks** (100 iterations, array shape (10000, 50)):

| Method | Time | Allocations | Memory |
|--------|------|-------------|--------|
| Without Pool | 2.45s | 100 | 381 MB |
| With Pool | 0.52s | 5 | 19 MB |
| **Speedup** | **4.7x** | **20x fewer** | **95% less** |

**SparseJacobian Benchmarks** (1M data points, 50 parameters):

| Sparsity | Dense Memory | Sparse Memory | Reduction | Compute Speedup |
|----------|--------------|---------------|-----------|-----------------|
| 90% | 381 MB | 76 MB | 5.0x | 1.5x |
| 95% | 381 MB | 38 MB | 10.0x | 2.0x |
| 99% | 381 MB | 9.5 MB | 40.0x | 3.5x |
| 99.9% | 381 MB | 1.0 MB | 381.0x | 5.0x |

**StreamingOptimizer Benchmarks** (exponential model, various dataset sizes):

| Dataset Size | In-Memory Time | Streaming Time | Memory Usage |
|--------------|----------------|----------------|--------------|
| 100K points | 0.5s | 2.1s | 7.6 MB → 1.2 MB |
| 1M points | 4.2s | 18.5s | 76 MB → 1.2 MB |
| 10M points | OOM | 185s | OOM → 1.2 MB |
| 100M points | OOM | 1850s | OOM → 1.2 MB |

**Note**: Streaming is slower but enables unlimited dataset size.

---

### Best Practices

**1. Profile Before Optimizing**

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your fitting code
cf = CurveFit()
popt, pcov = cf.curve_fit(model, x, y, p0)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)  # Top 20 functions
```

**2. Measure Improvements**

```python
import time


def benchmark(name, func, *args, n_runs=5):
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func(*args)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"{name}: {avg_time:.3f} ± {std_time:.3f}s")
    return result, avg_time


# Compare approaches
result1, time1 = benchmark("Standard", standard_fit, x, y, p0)
result2, time2 = benchmark("Optimized", optimized_fit, x, y, p0)
print(f"Speedup: {time1/time2:.2f}x")
```

**3. Start Simple, Add Complexity**

```python
# Level 1: Start simple
from nlsq import CurveFit

cf = CurveFit()
popt, pcov = cf.curve_fit(model, x, y, p0)

# Level 2: Add MemoryPool if repeated fits
if n_fits > 10:
    with MemoryPool() as pool:
        for dataset in datasets:
            popt, pcov = cf.curve_fit(model, *dataset)

# Level 3: Add SparseJacobian if sparse
sparse_comp = SparseJacobianComputer()
pattern, sparsity = sparse_comp.detect_sparsity_pattern(model, p0, x_sample)
if sparsity > 0.9:
    pass  # Use sparse methods

# Level 4: Use streaming if data doesn't fit
if data_size > available_memory:
    optimizer = StreamingOptimizer(config)
    result = optimizer.fit_streaming(model, data_source, p0)
```

**4. Monitor Memory Usage**

```python
import psutil
import os


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


print(f"Initial memory: {get_memory_usage():.1f} MB")
popt, pcov = cf.curve_fit(model, x, y, p0)
print(f"Peak memory: {get_memory_usage():.1f} MB")
```

**5. Use Appropriate Batch Sizes**

```python
# Batch size guidelines for StreamingOptimizer
batch_size = min(
    10000,  # Default max
    total_dataset_size // 100,  # At least 100 batches
    available_memory_mb * 1024 // (8 * n_params),  # Fit in memory
)

config = StreamingConfig(batch_size=batch_size)
```

**6. Checkpoint Long-Running Optimizations**

```python
config = StreamingConfig(
    checkpoint_interval=1000,  # Save every 1000 iterations
    max_epochs=100,
)

optimizer = StreamingOptimizer(config)
# Checkpoints saved automatically

# Resume from checkpoint
checkpoint = np.load("checkpoint_iter_5000.npz")
p0 = checkpoint["params"]
result = optimizer.fit_streaming(model, data_source, p0)
```

**7. Validate Sparse Approximations**

```python
sparse_comp = SparseJacobianComputer(sparsity_threshold=0.01)
pattern, sparsity = sparse_comp.detect_sparsity_pattern(model, p0, x_train)

# Compute error from sparsification
J_dense = compute_dense_jacobian(model, p0, x_test[:100])
J_sparse = sparse_comp.compute_sparse_jacobian(...)

error = np.linalg.norm(J_dense - J_sparse.toarray()) / np.linalg.norm(J_dense)
print(f"Sparsification error: {error:.2%}")

if error < 0.01:
    print("✓ Safe to use sparse approximation")
else:
    print("⚠️  Sparsity threshold may be too aggressive")
```

---

### Summary

Choose the right tool for your problem:

| Feature | Best For | Typical Gain |
|---------|----------|--------------|
| **MemoryPool** | Repeated fits, many iterations | 2-5x speedup, 90% less allocation |
| **SparseJacobian** | Large sparse problems | 10-100x memory reduction |
| **StreamingOptimizer** | Unlimited datasets | Enables problems that don't fit in memory |

**Golden Rule**: Profile first, optimize second, measure third!

For interactive examples, see **[performance_optimization_demo.ipynb](performance_optimization_demo.ipynb)**

---

## 📊 Example Comparison

| Notebook | Complexity | Time | Dataset Size | Key Features |
|----------|-----------|------|--------------|--------------|
| **Quickstart** | ⭐ Basic | 15 min | <10K | API basics, memory config |
| **Advanced Features** | ⭐⭐ Medium | 35 min | Any | Diagnostics, recovery, validation |
| **Large Dataset** | ⭐⭐ Medium | 30 min | 10K-100M | Chunking, sampling, HDF5 |
| **2D Gaussian** | ⭐⭐ Medium | 25 min | 2D images | Multi-dimensional, visualization |
| **Performance Opt** | ⭐⭐⭐ Advanced | 45 min | Any | MemoryPool, Sparse, Streaming |

---

## 🔬 Testing and Validation

All examples have been tested and validated:

| Example | Tests | Status | Coverage |
|---------|-------|--------|----------|
| Quickstart | ✅ 5 scenarios | All passing | Core API |
| Advanced Features | ✅ 8 scenarios | All passing | Advanced features |
| Large Dataset | ✅ 6 scenarios | All passing | Scaling |
| 2D Gaussian | ✅ 4 scenarios | All passing | Multi-D |
| Performance Opt | ✅ 12 scenarios | All passing | Optimization |

---

## 🆘 Common Issues and Solutions

### Issue: JAX precision warning
```
UserWarning: JAX is not using 64-bit precision
```
**Solution**: Import NLSQ before importing JAX (NLSQ auto-configures precision)
```python
from nlsq import CurveFit  # Import first
import jax.numpy as jnp  # Import after
```

### Issue: GPU out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU or reduce dataset size
```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # Force CPU
```
Or use chunking/streaming from Large Dataset or Performance examples.

### Issue: Slow first fit
**Expected behavior**: First fit includes JIT compilation (1-2 seconds)
**Solution**: Reuse `CurveFit` objects for subsequent fits
```python
cf = CurveFit()  # Create once
for dataset in datasets:
    popt, pcov = cf.curve_fit(...)  # Fast after first
```

### Issue: Array size mismatch errors
**Cause**: JAX retracing when array sizes change
**Solution**: Use `flength` parameter (see Quickstart example)
```python
cf = CurveFit(flength=max_data_length)
```

---

## 📚 Additional Resources

### Documentation
- **Main Docs**: [https://nlsq.readthedocs.io](https://nlsq.readthedocs.io)
- **GitHub**: [https://github.com/imewei/NLSQ](https://github.com/imewei/NLSQ)
- **API Reference**: [https://nlsq.readthedocs.io/en/latest/api.html](https://nlsq.readthedocs.io/en/latest/api.html)
- **PyPI**: [https://pypi.org/project/nlsq/](https://pypi.org/project/nlsq/)

### Related Resources
- **JAX Documentation**: [https://jax.readthedocs.io](https://jax.readthedocs.io)
- **SciPy curve_fit**: [docs.scipy.org](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
- **JAXFit Paper**: [arXiv:2208.12187](https://doi.org/10.48550/arXiv.2208.12187)

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/imewei/NLSQ/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imewei/NLSQ/discussions)
- **Email**: [maintainer contact](https://github.com/imewei)

---

## 🤝 Contributing

Found an issue or want to improve the examples?

1. **Report bugs**: [GitHub Issues](https://github.com/imewei/NLSQ/issues)
2. **Suggest examples**: [GitHub Discussions](https://github.com/imewei/NLSQ/discussions)
3. **Submit PRs**: Fork, improve, submit!

---

## 📜 License

NLSQ is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

## 🎓 Citation

If you use NLSQ in your research, please cite:

```bibtex
@software{nlsq2024,
  title={NLSQ: GPU-Accelerated Nonlinear Least Squares Curve Fitting},
  author={Chen, Wei},
  year={2024},
  url={https://github.com/imewei/NLSQ},
  note={Argonne National Laboratory}
}
```

Original JAXFit paper:
```bibtex
@article{hofer2022jaxfit,
  title={JAXFit: Trust region reflective algorithms in JAX for high-throughput spectroscopic analysis},
  author={Hofer, Lucas R and Krstajiƒá, Milan and Smith, Robert P},
  journal={arXiv preprint arXiv:2208.12187},
  year={2022}
}
```

---

## 🌟 Highlights

✨ **5 comprehensive notebooks** covering basics to advanced optimization
✨ **Production-ready** with 100% test pass rate
✨ **GPU-accelerated** with 150-270x speedup over SciPy
✨ **Memory-efficient** with chunking and streaming support
✨ **Well-documented** with clear explanations and visualizations
✨ **Beginner-friendly** with progressive learning paths
✨ **Advanced features** for performance-critical applications

---

**Ready to get started?** Open [nlsq_quickstart.ipynb](nlsq_quickstart.ipynb) and begin your journey! 🚀

---

<p align="center">
<i>Last updated: 2025-10-08 | NLSQ v0.1.1</i>
</p>
