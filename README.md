<div align="center">
<img src="https://raw.githubusercontent.com/imewei/NLSQ/main/docs/_static/NLSQ_logo.png" alt="NLSQ logo" width="400">

# NLSQ: GPU-Accelerated Curve Fitting

**Drop-in replacement for `scipy.optimize.curve_fit` with 150-270x speedup on GPU**

[![PyPI version](https://badge.fury.io/py/nlsq.svg)](https://badge.fury.io/py/nlsq)
[![Documentation](https://readthedocs.org/projects/nlsq/badge/?version=latest)](https://nlsq.readthedocs.io/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX 0.8.0](https://img.shields.io/badge/JAX-0.8.0-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Documentation**](https://nlsq.readthedocs.io/) | [**Examples**](examples/) | [**API Reference**](https://nlsq.readthedocs.io/en/latest/api.html) | [**ArXiv Paper**](https://doi.org/10.48550/arXiv.2208.12187)

</div>

---

## What is NLSQ?

NLSQ is a nonlinear least squares curve fitting library built on [JAX](https://github.com/google/jax). It provides:

- **SciPy-compatible API** - Same function signatures as `scipy.optimize.curve_fit`
- **GPU/TPU acceleration** - JIT-compiled kernels via XLA
- **Automatic differentiation** - No manual Jacobian calculations needed
- **Large dataset support** - Handles 100M+ data points with streaming optimization
- **Interactive GUI** - Native Qt desktop application for no-code curve fitting

## Installation

### Basic Installation (CPU-only)

```bash
pip install nlsq
```

This installs with CPU-only JAX (works on all platforms: Linux, macOS, Windows).

### GPU Installation (Linux + System CUDA)

**Performance Impact:** 20-100x speedup for large datasets (>1M points)

**Prerequisites:**
- NVIDIA GPU with SM >= 5.2 (Maxwell or newer)
- System CUDA 12.x or 13.x installed
- `nvcc` in PATH

<details>
<summary><b>Verify Prerequisites</b></summary>

```bash
# Check CUDA installation
nvcc --version
# Should show: Cuda compilation tools, release 12.x or 13.x

# Check GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# Should show: GPU name and SM version (e.g., "8.9" for RTX 4090)
```

</details>

#### Option 1: Quick Install via Makefile (Recommended)

```bash
git clone https://github.com/imewei/NLSQ.git
cd NLSQ

# Auto-detect system CUDA version and install matching JAX
make install-jax-gpu

# Or explicitly choose CUDA version:
make install-jax-gpu-cuda13  # Requires system CUDA 13.x + SM >= 7.5
make install-jax-gpu-cuda12  # Requires system CUDA 12.x + SM >= 5.2
```

#### Option 2: Manual Installation

```bash
# For System CUDA 13.x (Turing and newer GPUs):
pip uninstall -y jax jaxlib
pip install "jax[cuda13-local]"

# For System CUDA 12.x (Maxwell and newer GPUs):
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]"

# Verify
python -c "import jax; print('Backend:', jax.default_backend())"
# Should show: Backend: gpu
```

<details>
<summary><b>GPU Compatibility Guide</b></summary>

| GPU Generation | Example GPUs | SM Version | CUDA 13 | CUDA 12 |
|----------------|--------------|------------|---------|---------|
| Blackwell | B100, B200 | 10.0 | Yes | Yes |
| Hopper | H100, H200 | 9.0 | Yes | Yes |
| Ada Lovelace | RTX 40xx, L40 | 8.9 | Yes | Yes |
| Ampere | RTX 30xx, A100 | 8.x | Yes | Yes |
| Turing | RTX 20xx, T4 | 7.5 | Yes | Yes |
| Volta | V100, Titan V | 7.0 | No | Yes |
| Pascal | GTX 10xx, P100 | 6.x | No | Yes |
| Maxwell | GTX 9xx, Titan X | 5.x | No | Yes |
| Kepler | GTX 7xx, K80 | 3.x | No | No |

**Recommendation:** SM >= 7.5 (RTX 20xx or newer): Install CUDA 13 for best performance

</details>

<details>
<summary><b>GPU Troubleshooting</b></summary>

**Issue: "nvcc not found"**
```bash
# Option 1: Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian

# Option 2: Add existing CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Issue: "CUDA version mismatch"**
```bash
# Check your system CUDA version
nvcc --version

# Reinstall with correct package
pip uninstall -y jax jaxlib
pip install "jax[cuda12-local]"  # or cuda13-local
```

</details>

### With GUI Support

```bash
pip install nlsq[gui]
```

## Quick Start

```python
import numpy as np
import jax.numpy as jnp
from nlsq import curve_fit


# Define model function (use jax.numpy for GPU acceleration)
def exponential(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# Generate data
x = np.linspace(0, 4, 1000)
y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(len(x))

# Fit - same API as scipy.optimize.curve_fit
popt, pcov = curve_fit(exponential, x, y, p0=[2.0, 0.5, 1.0])

print(f"Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}")
print(f"Uncertainties: {np.sqrt(np.diag(pcov))}")
```

## Graphical User Interface

NLSQ includes an interactive GUI for curve fitting without writing code:

```bash
# Launch the GUI
nlsq-gui
```

The GUI provides:
- **Data Loading**: Import CSV, ASCII, NPZ, or HDF5 files, or paste from clipboard
- **Model Selection**: Choose from 7 built-in models, polynomials, or custom Python functions
- **Fitting Options**: Guided presets (Fast/Robust/Quality) or advanced parameter control
- **Interactive Results**: GPU-accelerated pyqtgraph plots with confidence bands and residuals
- **Export**: Session bundles (ZIP), JSON/CSV results, and reproducible Python code

See the [GUI User Guide](https://nlsq.readthedocs.io/en/latest/gui/index.html) for detailed documentation.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic Jacobians** | JAX's autodiff eliminates manual derivatives |
| **Bounded optimization** | Trust Region Reflective and Levenberg-Marquardt |
| **Large datasets** | Chunked and streaming optimizers for 100M+ points |
| **Multi-start** | Global optimization with LHS/Sobol sampling |
| **Mixed precision** | Automatic float32→float64 upgrade when needed |
| **Workflow system** | Memory-based auto-selection (MemoryBudgetSelector) |
| **CLI interface** | YAML-based workflows with `nlsq fit` and `nlsq batch` |
| **Interactive GUI** | No-code curve fitting with Qt desktop application |
| **Model Diagnostics** | Identifiability analysis, gradient health monitoring, sloppy model detection |

## Architecture

NLSQ is organized into well-separated layers (~72,000 lines):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             USER INTERFACES                                 │
│  Qt GUI (PySide6)       CLI (Click)            Python API                   │
│  ├── 5-page workflow    ├── Model validation   ├── curve_fit(), fit()       │
│  ├── pyqtgraph plots    ├── Security auditing  ├── CurveFit class           │
│  └── Native desktop     └── Export formats     └── LargeDatasetFitter       │
├─────────────────────────────────────────────────────────────────────────────┤
│                        OPTIMIZATION ORCHESTRATION                           │
│  Workflow System         Global Optimization      Streaming Optimizer       │
│  ├── MemoryBudgetSel.    ├── MultiStartOrch.     ├── AdaptiveHybrid         │
│  ├── Strategy: STANDARD/ ├── TournamentSelect    ├── 4-Phase Pipeline:      │
│  │   CHUNKED/STREAMING   ├── LHS/Sobol/Halton    │   0: Normalization       │
│  └── Memory-based auto   └── Sampling            │   1: L-BFGS warmup       │
│                                                  │   2: Gauss-Newton        │
│                                                  └── 3: Denormalization     │
├─────────────────────────────────────────────────────────────────────────────┤
│                          CORE OPTIMIZATION ENGINE                           │
│  curve_fit() ─→ CurveFit ─→ LeastSquares ─→ TrustRegionReflective           │
│       │              │            │                │                        │
│       ▼              ▼            ▼                ▼                        │
│  API Wrapper    Cache+State  Orchestrator+AD   SVD-based TRF                │
│  (SciPy-compat) (UnifiedCache) (AutoDiffJac)   (JIT-compiled)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                          SUPPORT SUBSYSTEMS                                 │
│  stability/           precision/          caching/          diagnostics/    │
│  ├── NumericalGuard   ├── MixedPrecision  ├── UnifiedCache  ├── Identifiab. │
│  ├── SVD fallback     ├── AlgorithmSel.   ├── SmartCache    ├── GradientMon │
│  └── Recovery         └── BoundsInfer.    └── MemoryMgr     └── PluginSys.  │
├─────────────────────────────────────────────────────────────────────────────┤
│                            INFRASTRUCTURE                                   │
│  interfaces/ (Protocols)     config.py (Singleton)    Security              │
│  ├── OptimizerProtocol       ├── JAXConfig            ├── safe_serialize    │
│  ├── CurveFitProtocol        ├── MemoryConfig         ├── model_validation  │
│  └── CacheProtocol           └── LargeDatasetConfig   └── resource_limits   │
├─────────────────────────────────────────────────────────────────────────────┤
│                         JAX RUNTIME (0.8.0)                                 │
│  x64 enabled │ JIT compilation │ Autodiff │ GPU/TPU backend (optional)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

See [Architecture Documentation](https://nlsq.readthedocs.io/en/latest/developer/architecture.html) for detailed design.

## Performance

NLSQ shows increasing speedups as dataset size grows:

| Dataset Size | SciPy (CPU) | NLSQ (GPU) | Speedup |
|--------------|-------------|------------|---------|
| 10K points | 15ms | 2ms | 7x |
| 100K points | 150ms | 3ms | 50x |
| 1M points | 1.5s | 8ms | 190x |
| 10M points | 15s | 55ms | 270x |

> **Note**: First call includes JIT compilation (~500ms). Subsequent calls reuse compiled kernels.

See [Performance Guide](https://nlsq.readthedocs.io/en/latest/guides/performance_guide.html) for benchmarks.

## Large Dataset Example

```python
from nlsq import fit
import numpy as np
import jax.numpy as jnp


def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# 50 million points
x = np.linspace(0, 10, 50_000_000)
y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, len(x))

# Auto-selects optimal strategy (chunked/streaming) based on memory
popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2], show_progress=True)
```

## Advanced Usage

<details>
<summary><b>Multi-start global optimization</b></summary>

```python
from nlsq import curve_fit

popt, pcov = curve_fit(
    model,
    x,
    y,
    p0=[1, 1, 1],
    bounds=([0, 0, 0], [10, 5, 10]),
    multistart=True,
    n_starts=20,
    sampler="lhs",
)
```

</details>

<details>
<summary><b>Workflow presets</b></summary>

```python
from nlsq import fit

# Named presets: 'auto', 'standard', 'quality', 'fast', 'large_robust', 'streaming'
result = fit(model, x, y, p0=[1, 1, 1], workflow="quality")

# Automatic memory-based selection (default)
result = fit(model, x, y, p0=[1, 1, 1], workflow="auto")

# With method='auto' in curve_fit
from nlsq import curve_fit

popt, pcov = curve_fit(model, x, y, p0=[1, 1, 1], method="auto")
```

</details>

<details>
<summary><b>Memory-based strategy selection</b></summary>

```python
from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

# Compute memory budget for your dataset
budget = MemoryBudget.compute(n_points=10_000_000, n_params=10)
print(f"Peak memory: {budget.peak_gb:.2f} GB")
print(f"Fits in memory: {budget.fits_in_memory}")

# Let selector choose strategy
selector = MemoryBudgetSelector(safety_factor=0.75)
strategy, config = selector.select(
    n_points=10_000_000,
    n_params=10,
    memory_limit_gb=16.0,  # Optional override
)
print(f"Selected: {strategy}")  # "streaming", "chunked", or "standard"
```

</details>

<details>
<summary><b>Numerical stability</b></summary>

```python
from nlsq import curve_fit

# Auto-detect and fix numerical issues
popt, pcov = curve_fit(model, x, y, p0=p0, stability="auto")
```

</details>

<details>
<summary><b>Memory management</b></summary>

```python
from nlsq import MemoryConfig, memory_context, CurveFit

config = MemoryConfig(memory_limit_gb=8.0, enable_mixed_precision_fallback=True)

with memory_context(config):
    cf = CurveFit()
    popt, pcov = cf.curve_fit(model, x, y, p0=p0)
```

</details>

<details>
<summary><b>Command-line interface</b></summary>

```bash
# Launch GUI
nlsq gui

# Copy configuration templates
nlsq config

# Single workflow
nlsq fit experiment.yaml

# Output results to stdout (for piping)
nlsq fit experiment.yaml --stdout | jq '.popt'

# Batch processing
nlsq batch configs/*.yaml --summary results.json --workers 4

# System info
nlsq info
```

See [CLI Reference](https://nlsq.readthedocs.io/en/latest/reference/cli.html) for full documentation.

</details>

<details>
<summary><b>Model health diagnostics</b></summary>

```python
from nlsq.diagnostics import (
    DiagnosticsConfig,
    IdentifiabilityAnalyzer,
    GradientMonitor,
    ParameterSensitivityAnalyzer,
    create_health_report,
)

# Analyze parameter identifiability from Jacobian
config = DiagnosticsConfig(condition_threshold=1e8, correlation_threshold=0.95)
analyzer = IdentifiabilityAnalyzer(config)
report = analyzer.analyze(result.jac)

print(f"Condition number: {report.condition_number:.2e}")
print(f"Numerical rank: {report.numerical_rank}/{report.n_params}")

# Monitor gradient health during optimization
monitor = GradientMonitor(config)
callback = monitor.create_callback()
result = curve_fit(model, x, y, p0=p0, callback=callback)
grad_report = monitor.get_report()

# Analyze parameter sensitivity spectrum
sensitivity_analyzer = ParameterSensitivityAnalyzer(config)
sensitivity_report = sensitivity_analyzer.analyze(result.jac)
print(f"Is sloppy: {sensitivity_report.is_sloppy}")
print(f"Effective dimensionality: {sensitivity_report.effective_dimensionality:.1f}")

# Create aggregated health report
health_report = create_health_report(
    identifiability=report,
    gradient_health=grad_report,
    sloppy_model=sensitivity_report,
)
print(health_report.summary())
```

See [Diagnostics API](https://nlsq.readthedocs.io/en/latest/api/nlsq.diagnostics.html) for complete documentation.

</details>

See [Advanced User Guide](https://nlsq.readthedocs.io/en/latest/advanced_guide.html) for complete documentation.

## Examples

Start with the [Interactive Tutorial](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ_Interactive_Tutorial.ipynb) on Google Colab.

**By topic:**
- [Getting Started](examples/notebooks/01_getting_started/) - Basic usage and quickstart
- [Core Tutorials](examples/notebooks/02_core_tutorials/) - Large datasets, bounded optimization
- [Advanced](examples/notebooks/03_advanced/) - GPU optimization, streaming, checkpointing
- [Applications](examples/notebooks/09_gallery_advanced/) - Physics, chemistry, biology, engineering

See [examples/README.md](examples/README.md) for the full index.

## Requirements

- Python 3.12+
- JAX 0.8.0+
- NumPy 2.2+
- SciPy 1.16.0+

**GUI requirements**: PySide6 6.6+, pyqtgraph 0.13+

**GPU support** (Linux only): CUDA 12.x-13.x, NVIDIA driver >= 525

## Citation

If you use NLSQ in your research, please cite:

```bibtex
@software{nlsq2025,
  title={NLSQ: Nonlinear Least Squares Curve Fitting for GPU/TPU},
  author={Chen, Wei and Hofer, Lucas R and Krstaji{\'c}, Milan and Smith, Robert P},
  year={2025},
  url={https://github.com/imewei/NLSQ}
}

@article{jaxfit2022,
  title={JAXFit: Trust Region Method for Nonlinear Least-Squares Curve Fitting on the {GPU}},
  author={Hofer, Lucas R and Krstaji{\'c}, Milan and Smith, Robert P},
  journal={arXiv preprint arXiv:2208.12187},
  year={2022}
}
```

## Acknowledgments

NLSQ is an enhanced fork of [JAXFit](https://github.com/Dipolar-Quantum-Gases/JAXFit) by Lucas R. Hofer, Milan Krstajic, and Robert P. Smith. We gratefully acknowledge their foundational work.

## License

MIT License - see [LICENSE](LICENSE) for details.
