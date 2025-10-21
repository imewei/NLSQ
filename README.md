<div align="center">
<img src="docs/images/NLSQ_logo.png" alt="NLSQ logo">
</div>

# NLSQ: Nonlinear Least Squares Curve Fitting

[![PyPI version](https://badge.fury.io/py/nlsq.svg)](https://badge.fury.io/py/nlsq)
[![Documentation Status](https://readthedocs.org/projects/nlsq/badge/?version=latest)](https://nlsq.readthedocs.io/en/latest/?badge=latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.6.0+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Examples Validated](https://img.shields.io/badge/examples-validated%202025--10--09-brightgreen?style=flat)](https://github.com/imewei/NLSQ/actions/workflows/readme-examples.yml)

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

## Large Dataset Support

> **Note**: The examples below are tested with NLSQ v0.1.1+ (NumPy 2.0+, JAX 0.6.0+, Python 3.12+)
> **Last validated**: 2025-10-09 | [Test suite](tests/test_readme_examples.py) | [CI Status](https://github.com/imewei/NLSQ/actions/workflows/readme-examples.yml)

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

## Current gotchas

Full disclosure we've copied most of this from the [JAX repo](https://github.com/google/jax#current-gotchas), but NLSQ inherits
JAX's idiosyncrasies and so the "gotchas" are mostly the same.

### Double precision required
NLSQ requires double precision (64-bit, `float64`) for numerical stability. By default, JAX uses single precision (32-bit, `float32`).

NLSQ **automatically enables double precision** when imported. However, if you import JAX before NLSQ, you must enable it manually:

```python
# If importing JAX first (not recommended)
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from nlsq import CurveFit

# Recommended: Import NLSQ first (auto-enables double precision)
from nlsq import CurveFit
import jax.numpy as jnp
```

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
- **JAX 0.6.0+** (tested with 0.8.0)
- **NumPy 2.0+** ⚠️ **Breaking change from NumPy 1.x** (tested with 2.3.4)
- **SciPy 1.14.0+** (tested with 1.16.2)

**GPU Support**: Only Linux supports GPU acceleration. Windows and macOS are CPU-only.

### Quick Install

#### Linux

```bash
# CPU only
pip install nlsq "jax[cpu]>=0.6.0"

# GPU with system CUDA 12 (best performance, requires CUDA installed)
pip install nlsq "jax[cuda12-local]>=0.6.0"

# GPU with bundled CUDA 12 (larger download, no system CUDA needed)
pip install nlsq "jax[cuda12]>=0.6.0"
```

#### Windows & macOS

```bash
# CPU only (GPU not supported)
pip install nlsq "jax[cpu]>=0.6.0"
```

#### Development Installation

```bash
git clone https://github.com/imewei/NLSQ.git
cd nlsq
pip install -e ".[dev,test,docs]"
```

### Platform-Specific Installation

For detailed installation instructions including Windows support and CUDA configuration, see below.

### Windows Installation

Windows users have several options for installing JAX:

#### Option 1: WSL2 (Recommended for GPU)

Use Windows Subsystem for Linux 2 (WSL2) and follow the Linux installation instructions above for GPU support.

#### Option 2: Native Windows (CPU only)

```bash
# Create a Python 3.12+ environment
conda create -n nlsq python=3.12
conda activate nlsq

# Install JAX CPU version
pip install "jax[cpu]>=0.6.0"
pip install nlsq
```

**Note**: Windows does not support native GPU acceleration for JAX. Use WSL2 for GPU support.

For the latest JAX installation instructions, see the [official JAX documentation](https://jax.readthedocs.io/en/latest/installation.html).

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
