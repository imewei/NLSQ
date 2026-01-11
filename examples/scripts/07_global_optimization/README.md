# Section 07: Global Optimization

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

## Scripts

| # | Script | Level | Description |
|---|--------|-------|-------------|
| 01 | [01_multistart_basics.py](01_multistart_basics.py) | Intermediate | Local minima problem, `fit(workflow='auto_global')`, comparison |
| 02 | [02_sampling_strategies.py](02_sampling_strategies.py) | Intermediate | LHS, Sobol, Halton sampling; space-filling properties |
| 03 | [03_cmaes_configuration.py](03_cmaes_configuration.py) | Intermediate | CMAESConfig, MethodSelector, method selection criteria |
| 04 | [04_multistart_integration.py](04_multistart_integration.py) | Intermediate | Integration patterns, bounds handling, peak fitting |
| 05 | [05_cmaes_basic.py](05_cmaes_basic.py) | Advanced | CMAESOptimizer, presets, diagnostics |
| 06 | [06_cmaes_multiscale.py](06_cmaes_multiscale.py) | Advanced | Multi-scale parameters, scale invariance |

---

## Running the Scripts

```bash
# Run a single script
python examples/scripts/07_global_optimization/01_multistart_basics.py

# Run with quick mode (reduced iterations)
NLSQ_EXAMPLES_QUICK=1 python examples/scripts/07_global_optimization/05_cmaes_basic.py
```

---

## API Summary

### The `fit()` Function with Global Optimization

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
    n_starts=10,  # Number of multi-start runs
    sampler="lhs",  # "lhs", "sobol", or "halton"
)

# workflow='hpc' - For HPC cluster jobs with checkpointing
popt, pcov = fit(
    model,
    x,
    y,
    p0=p0,
    bounds=bounds,
    workflow="hpc",
    checkpoint_dir="/scratch/my_job/checkpoints",
)
```

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
    optimizer = CMAESOptimizer.from_preset("cmaes")  # or "cmaes-fast", "cmaes-global"

    # Custom configuration
    config = CMAESConfig(
        max_generations=100,
        restart_strategy="bipop",
        max_restarts=5,
        seed=42,
        refine_with_nlsq=True,
    )
    optimizer = CMAESOptimizer(config=config)

    # Run optimization
    result = optimizer.fit(model, x, y, bounds=bounds)
    popt = result["popt"]
    pcov = result["pcov"]
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
| Parameters span 3+ orders | `workflow="auto_global"` | CMA-ES handles scale invariance |
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
├── 01_multistart_basics.py      # Global optimization basics
├── 02_sampling_strategies.py    # LHS, Sobol, Halton sampling
├── 03_cmaes_configuration.py    # CMAESConfig and MethodSelector
├── 04_multistart_integration.py # Integration patterns
├── 05_cmaes_basic.py            # CMAESOptimizer usage
├── 06_cmaes_multiscale.py       # Multi-scale parameter fitting
├── figures/                     # Saved visualizations
└── README.md                    # This file
```

---

<p align="center">
<i>NLSQ v0.6.3 | Last updated: 2026-01-11</i>
</p>
