<div align="center">
<img src="docs/images/NLSQ_logo.png" alt="NLSQ logo">
</div>

# NLSQ: Nonlinear Least Squares Curve Fitting

[![PyPI version](https://badge.fury.io/py/nlsq.svg)](https://badge.fury.io/py/nlsq)
[![Documentation Status](https://readthedocs.org/projects/nlsq/badge/?version=latest)](https://nlsq.readthedocs.io/en/latest/?badge=latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.8.0-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Examples Validated](https://img.shields.io/badge/examples-validated%202025--11--19-brightgreen?style=flat)](https://github.com/imewei/NLSQ/actions/workflows/readme-examples.yml)

[**Quickstart**](#quickstart-colab-in-the-cloud)
| [**Install guide**](#installation)
| [**ArXiv Paper**](https://doi.org/10.48550/arXiv.2208.12187)
| [**Documentation**](https://nlsq.readthedocs.io/)
| [**Examples**](examples/)

## Acknowledgments

NLSQ is an enhanced fork of [JAXFit](https://github.com/Dipolar-Quantum-Gases/JAXFit), originally developed by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith. We gratefully acknowledge their foundational work on GPU-accelerated curve fitting with JAX. The original JAXFit paper: [arXiv:2208.12187](https://doi.org/10.48550/arXiv.2208.12187).

## What is NLSQ?

NLSQ builds upon JAXFit's foundation, implementing SciPy's nonlinear least squares curve fitting algorithms using [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) for GPU/TPU acceleration. This fork adds significant optimizations, enhanced testing, improved API design, and advanced features for production use. Fit functions are written in Python without CUDA programming.

NLSQ uses JAX's [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) to calculate Jacobians automatically, eliminating the need for manual partial derivatives or numerical approximation.


NLSQ provides a drop-in replacement for SciPy's curve_fit function with advanced features:

## Core Features
- **GPU/TPU acceleration** via JAX JIT compilation
- **Automatic differentiation** for Jacobian calculation
- **Trust Region Reflective** and **Levenberg-Marquardt** algorithms
- **Bounded optimization** with parameter constraints
- **Robust loss functions** for outlier handling
- **Fixed array size optimization** to avoid recompilation
- **Comprehensive test coverage** (>80%) ensuring reliability

## Large Dataset Support
- **Automatic dataset handling** for 100M+ points with `curve_fit_large`
- **Intelligent chunking** with <1% error for well-conditioned problems
- **Memory estimation** and automatic memory management
- **Streaming optimizer** for unlimited-size datasets that don't fit in memory
- **Sparse Jacobian optimization** for problems with sparse structure
- **Progress reporting** for long-running optimizations

## Advanced Memory Management
- **Context-based configuration** with temporary memory settings
- **Automatic memory detection** and chunk sizing
- **Mixed precision fallback** for memory-constrained environments
- **Memory leak prevention** with cleanup
- **Cache management** with eviction policies

## Algorithm Selection
- **Automatic algorithm selection** based on problem characteristics
- **Performance optimization** with problem-specific tuning
- **Convergence analysis** and parameter adjustment
- **Robustness testing** with multiple initialization strategies

## Diagnostics & Monitoring
- **Convergence monitoring** with diagnostics
- **Optimization recovery** from failed fits with fallback strategies
- **Numerical stability analysis** with condition number monitoring
- **Input validation** and error handling
- **Logging** and debugging capabilities

## Caching System
- **JIT compilation caching** to avoid recompilation overhead
- **Function evaluation caching** for repeated calls
- **Jacobian caching** with automatic invalidation
- **Memory-aware cache policies** with size limits

## Basic Usage

```python
import numpy as np
from nlsq import CurveFit


# Define your fit function
def linear(x, m, b):
    return m * x + b


# Prepare data
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Perform the fit
cf = CurveFit()
popt, pcov = cf.curve_fit(linear, x, y)
print(f"Fitted parameters: m={popt[0]:.2f}, b={popt[1]:.2f}")
```

NLSQ leverages JAX's just-in-time (JIT) compilation to [XLA](https://www.tensorflow.org/xla) for GPU/TPU acceleration.
Fit functions must be JIT-compilable. For functions using special operations, use JAX's numpy:

```python
import jax.numpy as jnp
import numpy as np
from nlsq import CurveFit


# Define exponential fit function using JAX numpy
def exponential(x, a, b):
    return jnp.exp(a * x) + b


# Generate synthetic data
x = np.linspace(0, 4, 50)
y_true = np.exp(0.5 * x) + 2.0
y = y_true + 0.1 * np.random.normal(size=len(x))

# Fit with initial guess
cf = CurveFit()
popt, pcov = cf.curve_fit(exponential, x, y, p0=[0.5, 2.0])
print(f"Fitted: a={popt[0]:.3f}, b={popt[1]:.3f}")

# Get parameter uncertainties from covariance
perr = np.sqrt(np.diag(pcov))
print(f"Uncertainties: σ_a={perr[0]:.3f}, σ_b={perr[1]:.3f}")
```


For more complex fit functions there are a few JIT function caveats (see [Current gotchas](#current-gotchas)) such as avoiding control code within the fit function (see [JAX's sharp edges](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
article for a more in-depth look at JAX specific caveats).


### Contents
* [Quickstart: Colab in the Cloud](#quickstart-colab-in-the-cloud)
* [Large Dataset Support](#large-dataset-support)
* [Current gotchas](#current-gotchas)
* [Installation](#installation)
* [Citing NLSQ](#citing-nlsq)
* [Reference documentation](#reference-documentation)

## Quickstart: Colab in the Cloud
The easiest way to test out NLSQ is using a Colab notebook connected to a Google Cloud GPU. JAX comes pre-installed so you'll be able to start fitting right away.

Tutorial notebooks:
- **[Interactive Tutorial: Beginner to Advanced](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ_Interactive_Tutorial.ipynb)** (recommended start! ⭐)
- [The basics: fitting basic functions with NLSQ](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ%20Quickstart.ipynb)
- [Fitting 2D images with NLSQ](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ%202D%20Gaussian%20Demo.ipynb)
- [Large dataset fitting demonstration](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/large_dataset_demo.ipynb)

## Performance Benchmarks

NLSQ delivers massive speedups on GPU hardware compared to SciPy's CPU-based optimization:

| Dataset Size | Parameters | SciPy (CPU) | NLSQ (GPU) | Speedup | Hardware |
|--------------|------------|-------------|------------|---------|----------|
| 1K points    | 3          | 2.5 ms      | 1.7 ms     | **1.5x** | Tesla V100 |
| 10K points   | 5          | 25 ms       | 2.0 ms     | **12x** | Tesla V100 |
| 100K points  | 5          | 450 ms      | 3.2 ms     | **140x** | Tesla V100 |
| 1M points    | 5          | 40.5 s      | 0.15 s     | **270x** | Tesla V100 |
| 50M points   | 3          | >30 min     | 1.8 s      | **>1000x** | Tesla V100 |

**Key Observations:**
- Speedup increases with dataset size due to GPU parallelization
- JIT compilation overhead on first run (~450-650ms), then 1.7-2.0ms on cached runs
- Excellent scaling: 50x more data → only 1.2x slower (1M → 50M points)
- Memory-efficient chunking handles datasets larger than GPU memory

See [Performance Guide](https://nlsq.readthedocs.io/en/latest/guides/performance_guide.html) for detailed benchmarks and optimization strategies.

## Examples Gallery

📂 **[examples/](examples/)** - Complete collection of 34 notebooks & scripts

### 🌟 Getting Started (6 notebooks)
Perfect for first-time users learning NLSQ basics:

- [**Interactive Tutorial**](examples/notebooks/01_getting_started/nlsq_interactive_tutorial.ipynb) - Comprehensive beginner-to-advanced guide ⭐
- [Quick Start](examples/notebooks/01_getting_started/nlsq_quickstart.ipynb) - 5-minute introduction to NLSQ
- [Basic Curve Fitting](examples/notebooks/01_getting_started/basic_curve_fitting.ipynb) - Fundamental fitting concepts
- [Parameter Bounds](examples/notebooks/01_getting_started/parameter_bounds.ipynb) - Constrained optimization
- [Robust Fitting](examples/notebooks/01_getting_started/robust_fitting.ipynb) - Handling outliers with robust loss functions
- [Uncertainty Estimation](examples/notebooks/01_getting_started/uncertainty_estimation.ipynb) - Parameter confidence intervals

### 💡 Core Features (7 notebooks)
Essential NLSQ capabilities for everyday use:

- [GPU vs CPU Performance](examples/notebooks/02_core_tutorials/gpu_vs_cpu.ipynb) - Benchmark GPU acceleration
- [Large Dataset Demo](examples/notebooks/02_core_tutorials/large_dataset_demo.ipynb) - Fitting 50M+ points
- [2D Gaussian Fitting](examples/notebooks/02_core_tutorials/nlsq_2d_gaussian_demo.ipynb) - Image fitting
- [Advanced Features](examples/notebooks/02_core_tutorials/advanced_features_demo.ipynb) - Algorithm selection, caching
- [Performance Optimization](examples/notebooks/02_core_tutorials/performance_optimization_demo.ipynb) - Maximize speed
- [Memory Management](examples/notebooks/02_core_tutorials/memory_management.ipynb) - Configure memory limits
- [Weighted Fitting](examples/notebooks/02_core_tutorials/weighted_fitting.ipynb) - Custom error weights

### 🚀 Advanced Topics (9 notebooks)
Deep dives into specialized features:

- [Custom Algorithms](examples/notebooks/03_advanced/custom_algorithms_advanced.ipynb) - Implement your own optimizers
- [GPU Optimization Deep Dive](examples/notebooks/03_advanced/gpu_optimization_deep_dive.ipynb) - Maximize GPU performance
- [ML Integration](examples/notebooks/03_advanced/ml_integration_tutorial.ipynb) - Combine with JAX ML ecosystem
- [Time Series Analysis](examples/notebooks/03_advanced/time_series_analysis.ipynb) - Temporal data fitting
- [Research Workflow](examples/notebooks/03_advanced/research_workflow_case_study.ipynb) - Real-world Raman spectroscopy
- [Troubleshooting Guide](examples/notebooks/03_advanced/troubleshooting_guide.ipynb) - Debug convergence issues
- [NLSQ Challenges](examples/notebooks/03_advanced/nlsq_challenges.ipynb) - Difficult optimization problems
- [Sparse Jacobian](examples/notebooks/03_advanced/sparse_jacobian.ipynb) - Exploit sparsity patterns
- [Adaptive Algorithms](examples/notebooks/03_advanced/adaptive_algorithms.ipynb) - Auto-tune optimization

### 📚 Application Gallery (12 notebooks)
Domain-specific examples across sciences:

**Biology** (3):
- [Dose-Response Curves](examples/notebooks/04_gallery/biology/dose_response.ipynb)
- [Enzyme Kinetics](examples/notebooks/04_gallery/biology/enzyme_kinetics.ipynb)
- [Growth Curves](examples/notebooks/04_gallery/biology/growth_curves.ipynb)

**Chemistry** (2):
- [Reaction Kinetics](examples/notebooks/04_gallery/chemistry/reaction_kinetics.ipynb)
- [Titration Curves](examples/notebooks/04_gallery/chemistry/titration_curves.ipynb)

**Engineering** (3):
- [Sensor Calibration](examples/notebooks/04_gallery/engineering/sensor_calibration.ipynb)
- [Materials Characterization](examples/notebooks/04_gallery/engineering/materials_characterization.ipynb)
- [System Identification](examples/notebooks/04_gallery/engineering/system_identification.ipynb)

**Physics** (3):
- [Damped Oscillation](examples/notebooks/04_gallery/physics/damped_oscillation.ipynb)
- [Radioactive Decay](examples/notebooks/04_gallery/physics/radioactive_decay.ipynb)
- [Spectroscopy Peaks](examples/notebooks/04_gallery/physics/spectroscopy_peaks.ipynb)

### ⚙️ Feature Demonstrations (4 notebooks)
In-depth feature showcases:

- [Callbacks System](examples/notebooks/05_feature_demos/callbacks_demo.ipynb) - Monitor optimization progress
- [Enhanced Error Messages](examples/notebooks/05_feature_demos/enhanced_error_messages_demo.ipynb) - Helpful diagnostics
- [Function Library](examples/notebooks/05_feature_demos/function_library_demo.ipynb) - Pre-built fitting functions
- [Result Enhancements](examples/notebooks/05_feature_demos/result_enhancements_demo.ipynb) - Rich result objects

### 🔄 Streaming & Fault Tolerance (5 notebooks)
Production-ready reliability features:

- [Basic Fault Tolerance](examples/notebooks/06_streaming/01_basic_fault_tolerance.ipynb) - Handle errors gracefully
- [Checkpoint & Resume](examples/notebooks/06_streaming/02_checkpoint_resume.ipynb) - Save/restore state
- [Custom Retry Settings](examples/notebooks/06_streaming/03_custom_retry_settings.ipynb) - Configure retries
- [Diagnostics Interpretation](examples/notebooks/06_streaming/04_interpreting_diagnostics.ipynb) - Understand results
- [Hybrid Streaming API](examples/notebooks/06_streaming/05_hybrid_streaming_api.ipynb) - 4-phase adaptive optimizer (v0.3.0+)

**All examples available as:**
- 📓 Jupyter notebooks: `examples/notebooks/`
- 🐍 Python scripts: `examples/scripts/`

## Large Dataset Support

> **Note**: The examples below are tested with NLSQ v0.1.1+ (NumPy 2.0+, JAX 0.8.0, Python 3.12+)
> **Last validated**: 2025-11-19 | [Test suite](tests/test_readme_examples.py) | [CI Status](https://github.com/imewei/NLSQ/actions/workflows/readme-examples.yml)

NLSQ includes advanced features for handling very large datasets (20M+ points) that may not fit in memory:

### Automatic Large Dataset Handling with curve_fit_large

```python
from nlsq import curve_fit_large, estimate_memory_requirements
import jax.numpy as jnp
import numpy as np

# Check memory requirements for your dataset
n_points = 50_000_000  # 50 million points
n_params = 3
stats = estimate_memory_requirements(n_points, n_params)
print(f"Memory required: {stats.total_memory_estimate_gb:.2f} GB")
print(f"Recommended chunks: {stats.n_chunks}")

# Generate large dataset
x = np.linspace(0, 10, n_points)
y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, n_points)


# Define fit function using JAX numpy
def exponential(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# Use curve_fit_large for automatic dataset size detection and chunking
popt, pcov = curve_fit_large(
    exponential,
    x,
    y,
    p0=[2.5, 0.6, 0.2],
    memory_limit_gb=4.0,  # Automatic chunking if needed
    show_progress=True,  # Progress bar for large datasets
)

print(f"Fitted parameters: {popt}")
print(f"Parameter uncertainties: {np.sqrt(np.diag(pcov))}")
```

### Advanced Large Dataset Fitting Options

```python
from nlsq import LargeDatasetFitter, fit_large_dataset, LDMemoryConfig
import jax.numpy as jnp

# Option 1: Use the convenience function for simple cases
result = fit_large_dataset(
    exponential,
    x,
    y,
    p0=[2.5, 0.6, 0.2],
    memory_limit_gb=4.0,
    show_progress=True,  # Progress bar for long fits
)

# Option 2: Use LargeDatasetFitter for more control
config = LDMemoryConfig(
    memory_limit_gb=4.0,
    min_chunk_size=10000,
    max_chunk_size=1000000,
    # Streaming optimization is automatic for very large datasets
    # No manual configuration needed - handles unlimited data with zero accuracy loss
    use_streaming=True,  # Enable streaming for datasets > memory limit
    streaming_batch_size=50000,  # Mini-batch size for streaming optimizer
)

fitter = LargeDatasetFitter(config=config)
result = fitter.fit_with_progress(
    exponential,
    x,
    y,
    p0=[2.5, 0.6, 0.2],
)

print(f"Fitted parameters: {result.popt}")
# Note: success_rate and n_chunks are only available for multi-chunk fits
# print(f"Covariance matrix: {result.pcov}")
```

### Sparse Jacobian Optimization

For problems with sparse Jacobian structure (e.g., fitting multiple independent components):

```python
from nlsq import SparseJacobianComputer

# ... (assumes func, p0, x_sample defined from previous example)

# Automatically detect and exploit sparsity
sparse_computer = SparseJacobianComputer(sparsity_threshold=0.01)
pattern, sparsity = sparse_computer.detect_sparsity_pattern(func, p0, x_sample)

if sparsity > 0.1:  # If more than 10% sparse
    print(f"Jacobian is {sparsity:.1%} sparse")
    # Optimization will automatically use sparse methods
```

### Streaming Optimizer for Unlimited Datasets

For datasets that don't fit in memory or are generated on-the-fly:

```python
from nlsq import StreamingOptimizer, StreamingConfig

# Configure streaming optimization
config = StreamingConfig(batch_size=10000, max_epochs=100, convergence_tol=1e-6)

optimizer = StreamingOptimizer(config)

# Stream data from file or generator
result = optimizer.fit_streaming(func, data_generator, p0=p0)
```

### Adaptive Hybrid Streaming Optimizer (v0.3.0+)

Four-phase hybrid optimizer combining parameter normalization, Adam warmup, streaming Gauss-Newton, and exact covariance computation:

```python
from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig
import jax.numpy as jnp

# Configure with presets: aggressive, conservative, or memory_optimized
config = HybridStreamingConfig.aggressive()  # Fast convergence
# config = HybridStreamingConfig.conservative()  # Higher quality
# config = HybridStreamingConfig.memory_optimized()  # Lower memory

optimizer = AdaptiveHybridStreamingOptimizer(config)

# Define model
def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c

# Fit with bounds-based normalization (addresses gradient imbalance)
result = optimizer.fit(
    model=model,
    x_data=x_data,
    y_data=y_data,
    p0=[2.0, 0.5, 0.3],
    bounds=(jnp.array([1.0, 0.1, 0.0]), jnp.array([10.0, 1.0, 2.0]))
)
```

**When to use Adaptive Hybrid Streaming:**
- Parameters span many orders of magnitude (gradient imbalance)
- Large datasets (100K+ points) with memory constraints
- Need production-quality uncertainty estimates
- Standard optimizers converge slowly near optimum

### Key Features for Large Datasets:

- **Automatic Size Detection**: `curve_fit_large` automatically switches between standard and chunked fitting
- **Memory Estimation**: Predict memory requirements before fitting
- **Intelligent Chunking**: Improved algorithm with <1% error for well-conditioned problems
- **Progress Reporting**: Track progress during long-running fits
- **JAX Tracing Support**: Compatible with functions having 15+ parameters
- **Sparse Optimization**: Exploit sparsity in Jacobian matrices
- **Streaming Support**: Process data that doesn't fit in memory
- **Memory-Efficient Solvers**: CG and LSQR solvers for reduced memory usage
- **Adaptive Convergence**: Early stopping when parameters stabilize

For more details, see the [large dataset guide](https://nlsq.readthedocs.io/en/latest/large_datasets.html) and [API documentation](https://nlsq.readthedocs.io/en/latest/api.html).

## Advanced Features

### Memory Management & Configuration

NLSQ provides memory management with context-based configuration:

```python
from nlsq import MemoryConfig, memory_context, get_memory_config
import numpy as np

# Configure memory settings
config = MemoryConfig(
    memory_limit_gb=8.0,
    enable_mixed_precision_fallback=True,
    safety_factor=0.8,
    progress_reporting=True,
)

# Use memory context for temporary settings
with memory_context(config):
    # Memory-optimized fitting
    cf = CurveFit()
    popt, pcov = cf.curve_fit(func, x, y, p0=p0)

# Check current memory configuration
current_config = get_memory_config()
print(f"Memory limit: {current_config.memory_limit_gb} GB")
print(f"Mixed precision fallback: {current_config.enable_mixed_precision_fallback}")
```

### Mixed Precision Fallback

NLSQ includes automatic mixed precision management that provides 50% memory savings while maintaining numerical accuracy:

```python
from nlsq import curve_fit
from nlsq.config import configure_mixed_precision
import jax.numpy as jnp

# Enable mixed precision with custom settings
configure_mixed_precision(
    enable=True,
    max_degradation_iterations=5,  # Grace period before upgrading
    gradient_explosion_threshold=1e10,
    verbose=True,  # Show precision upgrades in logs
)


# Define model function
def exponential(x, a, b):
    return a * jnp.exp(-b * x)


# Fit - starts in float32, automatically upgrades to float64 if needed
popt, pcov = curve_fit(exponential, x, y, p0=[2.0, 0.5])
```

**Key Features:**
- **Automatic float32 → float64 upgrade** when precision issues detected
- **50% memory savings** when using float32
- **Zero-iteration loss** during precision upgrades (state fully preserved)
- **Intelligent fallback** to relaxed float32 if float64 fails
- **Environment variable configuration** for production deployment

**Configuration Options:**
```python
# Programmatic configuration
configure_mixed_precision(
    enable=True,
    max_degradation_iterations=5,
    gradient_explosion_threshold=1e10,
    precision_limit_threshold=1e-7,
    tolerance_relaxation_factor=10.0,
    verbose=False,
)

# Or use environment variables
# export NLSQ_MIXED_PRECISION_VERBOSE=1
# export NLSQ_GRADIENT_EXPLOSION_THRESHOLD=1e8
# export NLSQ_MAX_DEGRADATION_ITERATIONS=3
```

### Algorithm Selection

NLSQ can select the best algorithm based on problem characteristics:

```python
from nlsq.algorithm_selector import AlgorithmSelector, auto_select_algorithm
from nlsq import curve_fit
import jax.numpy as jnp


# Define your model
def model_nonlinear(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# Auto-select best algorithm
recommendations = auto_select_algorithm(
    f=model_nonlinear, xdata=x, ydata=y, p0=[1.0, 0.5, 0.1]
)

# Use recommended algorithm
method = recommendations.get("algorithm", "trf")
popt, pcov = curve_fit(model_nonlinear, x, y, p0=[1.0, 0.5, 0.1], method=method)

print(f"Selected algorithm: {method}")
print(f"Fitted parameters: {popt}")
```

### Diagnostics & Monitoring

Monitor optimization progress:

```python
from nlsq import ConvergenceMonitor, CurveFit
from nlsq.diagnostics import OptimizationDiagnostics
import numpy as np

# Create convergence monitor
monitor = ConvergenceMonitor(window_size=10, sensitivity=1.0)

# Use CurveFit with stability features
cf = CurveFit(enable_stability=True, enable_recovery=True)

# Perform fitting
popt, pcov = cf.curve_fit(func, x, y, p0=p0)
print(f"Fitted parameters: {popt}")

# For detailed diagnostics, create separate diagnostics object
diagnostics = OptimizationDiagnostics()
# (diagnostics would be populated during optimization)
```

### Numerical Stability Mode

NLSQ provides automatic numerical stability monitoring and correction to prevent optimization divergence:

```python
from nlsq import curve_fit
import jax.numpy as jnp
import numpy as np


# Define a model function
def exponential(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# Generate data with challenging characteristics
x = np.linspace(0, 1e6, 1000)  # Large x-range can cause ill-conditioning
y = 2.5 * np.exp(-0.5 * x) + 1.0

# Option 1: stability='check' - warn about issues but don't fix
popt, pcov = curve_fit(exponential, x, y, p0=[2.5, 0.5, 1.0], stability="check")

# Option 2: stability='auto' - automatically detect and fix issues (recommended)
popt, pcov = curve_fit(exponential, x, y, p0=[2.5, 0.5, 1.0], stability="auto")

# Option 3: stability=False - disable stability checks (default)
popt, pcov = curve_fit(exponential, x, y, p0=[2.5, 0.5, 1.0], stability=False)
```

**Stability Modes:**

| Mode | Behavior | Use Case |
|------|----------|----------|
| `stability=False` | No checks (default) | Simple problems, maximum speed |
| `stability='check'` | Warn about issues | Debugging, identify problems |
| `stability='auto'` | Auto-detect and fix | Production use, challenging problems |

**Key Features:**

- **NaN/Inf Detection**: Automatically replaces invalid values in Jacobian
- **Condition Number Monitoring**: Detects ill-conditioned problems
- **Data Rescaling**: Optional rescaling of data to improve conditioning
- **SVD Skip for Large Jacobians**: Avoids expensive SVD computation for >10M elements

**Physics Applications (XPCS, scattering, etc.):**

For physics applications where data must maintain physical units:

```python
# Preserve physical units (don't rescale time delays, scattering vectors, etc.)
popt, pcov = curve_fit(
    g2_model,
    tau,
    y,
    p0=[1.0, 0.3, 100.0],
    stability="auto",
    rescale_data=False,  # Preserve physical units
)
```

**Large Jacobian Optimization:**

For large datasets (>10M Jacobian elements), SVD computation is automatically skipped to prevent performance degradation:

```python
# Custom SVD threshold (default: 10M elements)
popt, pcov = curve_fit(
    model,
    x_large,
    y_large,
    p0=p0,
    stability="auto",
    max_jacobian_elements_for_svd=5_000_000,  # Skip SVD above 5M elements
)
```

**Performance Impact:**
- `stability=False`: No overhead
- `stability='check'`: ~1ms overhead for 1M points
- `stability='auto'`: ~1-5ms overhead, prevents divergence

For more details, see the [Stability Guide](https://nlsq.readthedocs.io/en/latest/guides/stability.html).

### Caching System

Optimize performance with caching:

```python
from nlsq import SmartCache, cached_function, curve_fit
import jax.numpy as jnp

# Configure caching
cache = SmartCache(max_memory_items=1000, disk_cache_enabled=True)


# Define fit function (caching happens at the JIT level)
def exponential(x, a, b):
    return a * jnp.exp(-b * x)


# First fit - compiles function
popt1, pcov1 = curve_fit(exponential, x1, y1, p0=[1.0, 0.1])

# Second fit - reuses JIT compilation from first fit
popt2, pcov2 = curve_fit(exponential, x2, y2, p0=[1.2, 0.15])

# Check cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

### Optimization Recovery & Fallback

Error handling with recovery from failed optimizations:

```python
from nlsq import OptimizationRecovery, CurveFit, curve_fit
import numpy as np

# CurveFit with built-in recovery enabled
cf = CurveFit(enable_recovery=True)

try:
    popt, pcov = cf.curve_fit(func, x, y, p0=p0_initial)
    print(f"Fitted parameters: {popt}")
except Exception as e:
    print(f"Optimization failed: {e}")
    # Manual recovery with OptimizationRecovery
    recovery = OptimizationRecovery(max_retries=3, enable_diagnostics=True)
    # Recovery provides automatic fallback strategies
    popt, pcov = curve_fit(func, x, y, p0=p0_initial)
```

### Input Validation & Error Handling

Input validation for robust operation:

```python
from nlsq import InputValidator, curve_fit
import numpy as np

# Create validator
validator = InputValidator(fast_mode=True)

# Validate inputs before fitting
warnings, errors, clean_x, clean_y = validator.validate_curve_fit_inputs(
    f=func, xdata=x, ydata=y, p0=p0
)

if errors:
    print(f"Validation errors: {errors}")
else:
    # Use validated data
    popt, pcov = curve_fit(func, clean_x, clean_y, p0=p0)
    print(f"Fitted parameters: {popt}")
```

## Performance Optimizations (v0.3.0-beta.2)

NLSQ v0.3.0-beta.2 introduces three major performance optimizations for Phase 1 Priority 2:

### Adaptive Memory Reuse

**12.5% peak memory reduction** through intelligent memory pooling:

```python
from nlsq import curve_fit
from nlsq.memory_manager import MemoryManager

# Automatic memory reuse with size-class bucketing
manager = MemoryManager(enable_pooling=True, enable_stats=True)

# Fit with memory pooling
popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5])

# Check memory statistics
stats = manager.get_stats()
print(f"Memory pool reuse rate: {stats['reuse_rate']:.1%}")  # Typically 90%
print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
```

**Key Features:**
- Size-class bucketing (1KB/10KB/100KB) for 5x better reuse
- Adaptive safety factor (1.2 → 1.05) based on problem characteristics
- 90% memory pool reuse rate achieved
- Zero-copy optimization for reduced malloc/free overhead

### Sparse Jacobian Activation

**Automatic sparse pattern detection** for computational efficiency:

```python
from nlsq.sparse_jacobian import SparseJacobianComputer

# Automatic sparsity detection
computer = SparseJacobianComputer(sparsity_threshold=0.01)

# Detect sparsity pattern
pattern, sparsity = computer.detect_sparsity_pattern(model, p0, x_sample)

if sparsity > 0.1:  # More than 10% sparse
    print(f"Jacobian is {sparsity:.1%} sparse")
    print("Sparse optimizations automatically enabled")
```

**Benefits:**
- Detects sparse patterns (>70% zeros) automatically
- Auto-enables sparse-aware optimizations when beneficial
- Phase 1 infrastructure complete; Phase 2 will deliver 5-50x speedup for sparse problems

### Streaming Batch Padding

**Zero JIT recompiles** after warmup for streaming optimization:

```python
from nlsq import StreamingOptimizer, StreamingConfig

# Enable batch padding for zero recompiles
config = StreamingConfig(
    batch_size=100, use_batch_padding=True, batch_padding_multiple=16  # Default on GPU
)

optimizer = StreamingOptimizer(config)

# First few batches compile, then zero recompiles
result = optimizer.fit_streaming(data_generator, model, p0=[1.0, 0.5])

# Check diagnostics
print(f"Warmup batches: {result['warmup_batches']}")
print(f"Recompiles after warmup: {result['recompiles_after_warmup']}")  # 0
```

**Performance:**
- Eliminates JIT thrashing between streaming batches
- Device-aware auto-selection (GPU default, dynamic on CPU)
- Expected 5-15% GPU throughput improvement

### Host-Device Transfer Profiling (v0.3.0-beta.3)

**Comprehensive profiling and validation infrastructure** for monitoring GPU-CPU transfers:

```python
from nlsq.profiling import profile_optimization, analyze_source_transfers
from nlsq import curve_fit
import jax.numpy as jnp

# Profile optimization performance
with profile_optimization() as metrics:
    popt, pcov = curve_fit(model, x, y, p0=[1.0, 0.5])

print(f"Total time: {metrics.total_time_sec:.3f}s")
print(f"Average iteration: {metrics.avg_iteration_time_ms:.2f}ms")

# Static analysis of transfer patterns
with open("mymodule.py") as f:
    code = f.read()

analysis = analyze_source_transfers(code)
print(f"Potential transfers: {analysis['total_potential_transfers']}")
```

**Key Features:**
- **Async Logging**: JAX-aware logging eliminates GPU-CPU blocking (<5% overhead)
- **JAX Profiler Integration**: Runtime transfer measurement with `jax.profiler.trace()`
- **Static Analysis**: Detect `np.array()`, `np.asarray()`, `.block_until_ready()` patterns
- **Performance Baselines**: Automated baseline generation and CI/CD regression gates
- **Input Validation**: Type checking for all profiling functions

**Performance Regression Protection:**
```python
from nlsq import curve_fit

# Automatic regression detection in CI
# Tests fail if performance degrades >10% vs baseline
# See tests/test_performance_regression.py
```

**Async Logging Benefits:**
- Zero host-device blocking during optimization
- Verbosity control (0=off, 1=every 10th, 2=all iterations)
- Automatic JAX array detection
- Non-blocking callbacks via `jax.debug.callback`

For detailed performance analysis, see the [Performance Guide](https://nlsq.readthedocs.io/en/latest/guides/performance_guide.html).

## Current gotchas

Full disclosure we've copied most of this from the [JAX repo](https://github.com/google/jax#current-gotchas), but NLSQ inherits
JAX's idiosyncrasies and so the "gotchas" are mostly the same.

### Automatic Precision Management (v0.2.0+)
NLSQ **automatically manages numerical precision** for optimal performance and memory usage:

- **Default**: Float32 (single precision) for memory efficiency
- **Automatic upgrade**: Float32 → Float64 when precision issues detected
- **Memory savings**: Up to 50% by starting in float32
- **No manual configuration needed** for most use cases

NLSQ starts with single precision (float32) for memory efficiency. The mixed precision system will automatically upgrade to float64 if convergence stalls or precision issues are detected.

**Advanced users** can manually control precision or disable automatic fallback:

```python
from nlsq import curve_fit
from nlsq.mixed_precision import MixedPrecisionConfig

# Disable automatic fallback (strict float64)
config = MixedPrecisionConfig(enable_fallback=False)
popt, pcov = curve_fit(f, xdata, ydata, mixed_precision_config=config)

# Or manually enable x64 before importing NLSQ
from jax import config

config.update("jax_enable_x64", True)
```

See the [Mixed Precision guide](https://nlsq.readthedocs.io/en/latest/guides/advanced_features.html#mixed-precision-fallback) for advanced configuration options.

### Other caveats
Below are some more things to be careful of, but a full list can be found in [JAX's Gotchas
Notebook](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).
Some standouts:

1. JAX transformations only work on [pure functions](https://en.wikipedia.org/wiki/Pure_function), which don't have side-effects and respect [referential transparency](https://en.wikipedia.org/wiki/Referential_transparency) (i.e. object identity testing with `is` isn't preserved). If you use a JAX transformation on an impure Python function, you might see an error like `Exception: Can't lift Traced...`  or `Exception: Different traces at same level`.
1. [In-place mutating updates of arrays](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates), like `x[i] += y`, aren't supported, but [there are functional alternatives](https://jax.readthedocs.io/en/latest/jax.ops.html). Under a `jit`, those functional alternatives will reuse buffers in-place automatically.
1. Some transformations, like `jit`, [constrain how you can use Python control flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow). You'll always get loud errors if something goes wrong. You might have to use [jit's static_argnums parameter](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit), [structured control flow primitives](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators) like [lax.scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan).
1. Some of NumPy's dtype promotion semantics involving a mix of Python scalars and NumPy types aren't preserved, namely `np.add(1, np.array([2], np.float32)).dtype` is `float64` rather than `float32`.
1. If you're looking for [convolution operators](https://jax.readthedocs.io/en/latest/notebooks/convolutions.html), they're in the `jax.lax` package.


## Installation

### Requirements

- **Python 3.12 or higher** (3.13 also supported)
- **JAX 0.8.0** (locked version)
- **NumPy 2.0+** ⚠️ **Breaking change from NumPy 1.x** (tested with 2.3.4)
- **SciPy 1.14.0+** (tested with 1.16.2)

### Platform Support

| Platform | GPU Support | Performance | Notes |
|----------|-------------|-------------|-------|
| ✅ Linux + CUDA 12.1-12.9 | Full GPU | **150-270x speedup** | Recommended for large datasets |
| ❌ macOS (Intel/Apple Silicon) | CPU only | Baseline | No NVIDIA GPU support |
| ❌ Windows | CPU only | Baseline | Use WSL2 for GPU support |

**GPU Requirements** (Linux only):
- System CUDA 12.1-12.9 installed
- NVIDIA driver >= 525
- Compatible NVIDIA GPU

### Quick Install

#### Linux (CPU Only)

**Using pip:**
```bash
pip install nlsq "jax[cpu]==0.8.0"
```

**Using uv (recommended - faster):**
```bash
uv pip install nlsq "jax[cpu]==0.8.0"
```

#### Linux (GPU Acceleration - Recommended) ⚡

**Option 1: Automated Install (Recommended)**

From the NLSQ repository:

```bash
git clone https://github.com/imewei/NLSQ.git
cd NLSQ
make install-jax-gpu  # Handles uninstall, install, and verification
```

This single command:
- Detects your package manager (uv, conda/mamba, or pip)
- Uninstalls CPU-only JAX
- Installs GPU-enabled JAX with CUDA 12 support
- Verifies GPU detection automatically

**Option 2: Manual Install (pip)**

```bash
# Step 1: Uninstall CPU-only version
pip uninstall -y jax jaxlib

# Step 2: Install JAX with CUDA support (best performance)
pip install "jax[cuda12-local]==0.8.0"

# Step 3: Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
# Expected: [cuda(id=0)] instead of [CpuDevice(id=0)]
```

**Option 3: Manual Install (uv)**

```bash
# Step 1: Uninstall CPU-only version
uv pip uninstall jax jaxlib

# Step 2: Install JAX with CUDA support
uv pip install "jax[cuda12-local]==0.8.0"

# Step 3: Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
```

**Alternative**: For systems without CUDA installed, use bundled CUDA (larger download):

```bash
pip install "jax[cuda12]==0.8.0"
# or with uv:
uv pip install "jax[cuda12]==0.8.0"
```

#### Windows & macOS

```bash
# CPU only (GPU not supported natively)
pip install nlsq "jax[cpu]==0.8.0"
# or with uv:
uv pip install nlsq "jax[cpu]==0.8.0"
```

**Windows GPU Users**: Use WSL2 (Windows Subsystem for Linux) and follow the Linux GPU installation instructions above.

#### Development Installation

**Using pip:**
```bash
git clone https://github.com/imewei/NLSQ.git
cd NLSQ
pip install -e ".[dev,test,docs]"
```

**Using uv (recommended - faster):**
```bash
git clone https://github.com/imewei/NLSQ.git
cd NLSQ
uv pip install -e ".[dev,test,docs]"
```

For GPU support in development:
```bash
make install-jax-gpu
```

### GPU Troubleshooting

#### Diagnostic Tools

Check your environment configuration:

```bash
# From NLSQ repository
make env-info       # Show platform, package manager, GPU hardware, CUDA version
make gpu-check      # Test JAX GPU detection
```

#### Common Issues

**Issue 1: Warning "CUDA-enabled jaxlib is not installed"**

**Symptoms:**
```
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed.
Falling back to cpu.
```

**Solution:**
```bash
# Verify GPU hardware
nvidia-smi  # Should show your GPU

# Verify CUDA version
nvcc --version  # Should show CUDA 12.1-12.9

# Reinstall JAX with GPU support
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]==0.8.0"

# Verify fix
python -c "import jax; print(jax.devices())"
# Expected: [cuda(id=0)]
```

**Issue 2: ImportError or "CUDA library not found"**

**Symptoms:**
```
ImportError: libcudart.so.12: cannot open shared object file
```

**Solution:**
```bash
# Set CUDA library path (add to ~/.bashrc for permanent fix)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
ls /usr/local/cuda/lib64/libcudart.so*
```

**Issue 3: Out of memory errors during computation**

**Symptoms:**
```
jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED
```

**Solution:** Reduce GPU memory usage:
```python
# Option 1: Reduce chunk size for large datasets
from nlsq import curve_fit_large

popt, pcov = curve_fit_large(func, x, y, memory_limit_gb=4.0)  # Reduce from default

# Option 2: Configure JAX memory fraction
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"  # Use 70% of GPU memory
```

**Issue 4: Slow performance despite GPU**

**Symptoms:** First run is slow, subsequent runs are fast

**Explanation:** This is normal! JAX uses JIT (Just-In-Time) compilation:
- First run: 450-650ms (includes compilation)
- Cached runs: 1.7-2.0ms (150-270x faster)

**Solution:** Use `CurveFit` class to reuse compilation:
```python
from nlsq import CurveFit

# ... (assumes model_func, xdata1, ydata1, xdata2, ydata2 defined)

fitter = CurveFit(model_func)
popt1, pcov1 = fitter.fit(xdata1, ydata1)  # First run: JIT compiles
popt2, pcov2 = fitter.fit(xdata2, ydata2)  # Second run: reuses compilation
```

**Issue 5: Suppressing GPU Acceleration Warnings**

**Symptoms:** You see this warning on import even though you intentionally use CPU-only JAX:
```
⚠️  GPU ACCELERATION AVAILABLE
═══════════════════════════════
NVIDIA GPU detected: Tesla V100-SXM2-16GB
JAX is currently using: CPU-only
```

**When you might want to suppress this:**
- Running tests in CI/CD pipelines
- Using CPU-only JAX intentionally (testing, debugging, etc.)
- Parsing stdout programmatically
- Reducing output clutter in Jupyter notebooks

**Solution:** Set the `NLSQ_SKIP_GPU_CHECK` environment variable:
```bash
# Option 1: Set before running Python
export NLSQ_SKIP_GPU_CHECK=1
python your_script.py

# Option 2: Inline with command
NLSQ_SKIP_GPU_CHECK=1 python your_script.py

# Option 3: Add to CI/CD environment variables
# GitHub Actions example:
env:
  NLSQ_SKIP_GPU_CHECK: "1"

# Option 4: Set in Python before importing nlsq
import os
os.environ['NLSQ_SKIP_GPU_CHECK'] = '1'
import nlsq  # No warning printed
```

**Accepted values:** `"1"`, `"true"`, `"yes"` (case-insensitive)

**Note:** This suppresses the warning but does not affect actual GPU usage. If you have GPU-enabled JAX installed, it will still use the GPU for computations.

#### Conda/Mamba Users

NLSQ works seamlessly in conda environments using pip:

```bash
conda create -n nlsq python=3.12
conda activate nlsq
pip install nlsq

# For GPU (Linux only)
git clone https://github.com/imewei/NLSQ.git
cd NLSQ
make install-jax-gpu  # Automatically detects conda/mamba
```

**Note:** Conda extras syntax (`conda install nlsq[gpu-cuda]`) is not supported. Use the Makefile or manual pip installation method above.

<!--For more detail on using these pre-built wheels please see the docs.-->


## Citing NLSQ

If you use NLSQ in your research, please cite both the NLSQ software and the original JAXFit paper:

### NLSQ Software Citation

```bibtex
@software{nlsq2024,
  title={NLSQ: Nonlinear Least Squares Curve Fitting for GPU/TPU},
  author={Chen, Wei and Hofer, Lucas R and Krstaji{\'c}, Milan and Smith, Robert P},
  year={2024},
  url={https://github.com/imewei/NLSQ},
  note={Enhanced fork of JAXFit with advanced features for large datasets, memory management, and algorithm selection}
}
```

### Original JAXFit Paper

```bibtex
@article{jaxfit2022,
  title={JAXFit: Trust Region Method for Nonlinear Least-Squares Curve Fitting on the {GPU}},
  author={Hofer, Lucas R and Krstaji{\'c}, Milan and Smith, Robert P},
  journal={arXiv preprint arXiv:2208.12187},
  year={2022},
  url={https://doi.org/10.48550/arXiv.2208.12187}
}
```


## Reference documentation

For details about the NLSQ API, see the
[reference documentation](https://nlsq.readthedocs.io/).
