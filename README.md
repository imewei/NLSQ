<div align="center">
<img src="https://raw.githubusercontent.com/imewei/NLSQ/main/docs/images/NLSQ_logo.png" alt="NLSQ logo" width="400">

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

## Installation

```bash
# CPU (all platforms)
pip install nlsq

# GPU (Linux with CUDA 12.1+)
pip install nlsq
pip install "jax[cuda12-local]==0.8.0"
```

<details>
<summary><b>Verify GPU installation</b></summary>

```bash
python -c "import jax; print('Devices:', jax.devices())"
# Expected: [cuda(id=0)] for GPU, [CpuDevice(id=0)] for CPU
```

</details>

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

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic Jacobians** | JAX's autodiff eliminates manual derivatives |
| **Bounded optimization** | Trust Region Reflective and Levenberg-Marquardt |
| **Large datasets** | Chunked and streaming optimizers for 100M+ points |
| **Multi-start** | Global optimization with LHS/Sobol sampling |
| **Mixed precision** | Automatic float32→float64 upgrade when needed |
| **Workflow system** | Auto-selects strategy based on dataset size |
| **CLI interface** | YAML-based workflows with `nlsq fit` and `nlsq batch` |

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

# Presets: 'fast', 'robust', 'global', 'quality', 'memory_efficient'
popt, pcov = fit(model, x, y, preset="robust")
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
# Single workflow
nlsq fit experiment.yaml

# Batch processing
nlsq batch configs/*.yaml --summary results.json

# System info
nlsq info
```

See [CLI Reference](https://nlsq.readthedocs.io/en/latest/user_guide/cli_reference.html) for YAML configuration.

</details>

See [Advanced Features](https://nlsq.readthedocs.io/en/latest/guides/advanced_features.html) for complete documentation.

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
- JAX 0.8.0 (locked version)
- NumPy 2.0+
- SciPy 1.14.0+

**GPU support** (Linux only): CUDA 12.1-12.9, NVIDIA driver >= 525

## Citation

If you use NLSQ in your research, please cite:

```bibtex
@software{nlsq2024,
  title={NLSQ: Nonlinear Least Squares Curve Fitting for GPU/TPU},
  author={Chen, Wei and Hofer, Lucas R and Krstaji{\'c}, Milan and Smith, Robert P},
  year={2024},
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

NLSQ is an enhanced fork of [JAXFit](https://github.com/Dipolar-Quantum-Gases/JAXFit) by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith. We gratefully acknowledge their foundational work.

## License

MIT License - see [LICENSE](LICENSE) for details.
