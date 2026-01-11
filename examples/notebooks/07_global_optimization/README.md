# Section 07: Global Optimization (Notebooks)

**Global optimization with the unified `fit()` API (v0.6.3).**

---

## Overview

Nonlinear least squares problems often have multiple local minima. Standard gradient-based
optimizers can get trapped in a local minimum, returning suboptimal parameter estimates.
Global optimization techniques address this by exploring the parameter space from multiple
starting points.

NLSQ v0.6.3 provides global optimization through the `workflow='auto_global'` parameter:

- **Multi-Start optimization**: Run curve fitting from multiple starting points
- **CMA-ES**: Covariance Matrix Adaptation for multi-scale problems
- **Automatic method selection**: Based on parameter scale ratio

### Method Selection (auto_global)

```
scale_ratio = max(upper-lower) / min(upper-lower)

If scale_ratio > 1000 AND evosax available → CMA-ES
Otherwise → Multi-Start
```

---

## Notebooks

| # | Notebook | Level | Duration | Description |
|---|----------|-------|----------|-------------|
| 01 | [Multi-Start Basics](01_multistart_basics.ipynb) | Intermediate | 20 min | Local minima problem, `fit(workflow='auto_global')`, comparison |
| 02 | [Sampling Strategies](02_sampling_strategies.ipynb) | Intermediate | 25 min | LHS, Sobol, Halton sampling; space-filling properties |
| 03 | [CMA-ES Configuration](03_cmaes_configuration.ipynb) | Intermediate | 20 min | CMAESConfig, MethodSelector, method selection criteria |
| 04 | [Multi-Start Integration](04_multistart_integration.ipynb) | Intermediate | 25 min | Integration patterns, bounds handling, peak fitting |
| 05 | [CMA-ES Basic](05_cmaes_basic.ipynb) | Advanced | 20 min | CMAESOptimizer, presets, diagnostics |
| 06 | [CMA-ES Multi-Scale](06_cmaes_multiscale.ipynb) | Advanced | 25 min | Multi-scale parameters, scale invariance |

**Total time**: ~2.5 hours

---

## Quick Start

```python
from nlsq import fit

# workflow='auto' - Local optimization (default)
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto")

# workflow='auto_global' - Global optimization (bounds required)
# Auto-selects CMA-ES or Multi-Start based on parameter scale ratio
popt, pcov = fit(
    model,
    x,
    y,
    p0=p0,
    bounds=bounds,
    workflow="auto_global",
    n_starts=10,
)

# workflow='hpc' - For HPC cluster jobs with checkpointing
popt, pcov = fit(
    model,
    x,
    y,
    p0=p0,
    bounds=bounds,
    workflow="hpc",
    checkpoint_dir="/scratch/checkpoints",
)
```

---

## API Summary

### Method Selection

```python
from nlsq.global_optimization import MethodSelector

selector = MethodSelector()

# Check scale ratio
scale_ratio = selector.compute_scale_ratio(lower_bounds, upper_bounds)
print(f"Scale ratio: {scale_ratio}")  # If > 1000, CMA-ES recommended

# Get selected method
method = selector.select("auto", lower_bounds, upper_bounds)
# Returns "cmaes" or "multi-start"
```

### Sampling Functions

```python
from nlsq.global_optimization import (
    latin_hypercube_sample,  # Stratified random sampling (default)
    sobol_sample,  # Deterministic quasi-random
    halton_sample,  # Prime-based quasi-random
    scale_samples_to_bounds,  # Scale [0,1] samples to bounds
)
import jax

# Generate samples in [0, 1]^d
key = jax.random.PRNGKey(42)
samples = latin_hypercube_sample(n_samples=10, n_dims=3, rng_key=key)

# Scale to parameter bounds
lb = jnp.array([0.0, 0.0, -1.0])
ub = jnp.array([10.0, 5.0, 1.0])
scaled = scale_samples_to_bounds(samples, lb, ub)
```

### CMA-ES Optimizer (requires evosax)

```python
from nlsq.global_optimization import CMAESConfig, CMAESOptimizer, is_evosax_available

if is_evosax_available():
    # From preset
    optimizer = CMAESOptimizer.from_preset("cmaes")

    # Run optimization
    result = optimizer.fit(model, x, y, bounds=bounds)
    popt = result["popt"]
    diagnostics = result["cmaes_diagnostics"]
```

---

## When to Use Global Optimization

| Scenario | Workflow | Why |
|----------|----------|-----|
| Standard curve fitting | `workflow="auto"` | Local optimization is sufficient |
| Multi-modal problem | `workflow="auto_global"` | Global search for multiple minima |
| Unknown initial guess | `workflow="auto_global"` | Explores parameter space |
| Multi-scale parameters | `workflow="auto_global"` | Auto-selects CMA-ES |
| Long HPC job | `workflow="hpc"` | Checkpointing for fault tolerance |

---

## CMA-ES Presets

| Preset | max_generations | restarts | Use Case |
|--------|-----------------|----------|----------|
| `cmaes-fast` | 50 | 0 | Quick exploration |
| `cmaes` | 100 | 5 (BIPOP) | Default |
| `cmaes-global` | 200 | 10 (BIPOP) | Thorough search |

---

## File Structure

```
07_global_optimization/
├── 01_multistart_basics.ipynb      # Global optimization basics
├── 02_sampling_strategies.ipynb    # LHS, Sobol, Halton sampling
├── 03_cmaes_configuration.ipynb    # CMAESConfig and MethodSelector
├── 04_multistart_integration.ipynb # Integration patterns
├── 05_cmaes_basic.ipynb            # CMAESOptimizer usage
├── 06_cmaes_multiscale.ipynb       # Multi-scale parameter fitting
└── README.md                       # This file
```

---

<p align="center">
<i>NLSQ v0.6.3 | Last updated: 2026-01-11</i>
</p>
