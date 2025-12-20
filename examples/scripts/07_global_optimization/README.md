# Section 07: Global Optimization

**Multi-start optimization for finding global optima in nonlinear least squares fitting.**

---

## Overview

Nonlinear least squares problems often have multiple local minima. Standard gradient-based
optimizers can get trapped in a local minimum, returning suboptimal parameter estimates.
Global optimization techniques address this by exploring the parameter space from multiple
starting points.

This section teaches you how to use NLSQ's global optimization features:

- **Multi-start optimization**: Run curve fitting from multiple starting points and select
  the best result
- **Sampling strategies**: Latin Hypercube (LHS), Sobol, and Halton quasi-random sequences
  for better parameter space coverage
- **Tournament selection**: Memory-efficient candidate selection for streaming large datasets
- **Configuration presets**: Pre-tuned settings for common optimization scenarios

---

## Tutorials

| # | Tutorial | Level | Duration | Description |
|---|----------|-------|----------|-------------|
| 01 | [Multi-Start Basics](01_multistart_basics.ipynb) | Intermediate | 20 min | Local minima problem, `GlobalOptimizationConfig`, single-start vs multi-start comparison |
| 02 | [Sampling Strategies](02_sampling_strategies.ipynb) | Intermediate | 25 min | LHS, Sobol, Halton, and random sampling; space-filling properties; discrepancy metrics |
| 03 | [Presets and Config](03_presets_and_config.ipynb) | Intermediate | 20 min | All `GlobalOptimizationConfig` parameters, built-in presets, custom configurations |
| 04 | [Tournament Selection](04_tournament_selection.ipynb) | Advanced | 30 min | `TournamentSelector` for streaming scenarios, progressive elimination, memory efficiency |
| 05 | [Multi-Start Integration](05_multistart_integration.ipynb) | Intermediate | 25 min | Integration with `curve_fit()` and `curve_fit_large()`, bounds handling, workflow patterns |

**Total time**: ~2 hours

---

## Prerequisites

Before starting this section, you should be familiar with:

- Basic `curve_fit()` usage (see [01_getting_started/nlsq_quickstart](../01_getting_started/nlsq_quickstart.ipynb))
- Parameter bounds specification
- JAX numpy (`jnp`) for model functions

---

## API Summary

### Core Configuration

```python
from nlsq import GlobalOptimizationConfig, curve_fit

# Default configuration
config = GlobalOptimizationConfig()

# From preset
config = GlobalOptimizationConfig.from_preset("robust")

# Custom configuration
config = GlobalOptimizationConfig(
    n_starts=20,
    sampler="lhs",
    center_on_p0=True,
    scale_factor=1.0,
    elimination_rounds=2,
    elimination_fraction=0.5,
    batches_per_round=5,
)

# Use with curve_fit
popt, pcov = curve_fit(
    model,
    x,
    y,
    p0=p0,
    bounds=bounds,
    global_optimization=config,
)
```

### Built-in Presets

| Preset | n_starts | Sampler | center_on_p0 | Use Case |
|--------|----------|---------|--------------|----------|
| `fast` | 3 | lhs | True | Quick exploration |
| `robust` | 5 | lhs | True | Production default |
| `global` | 10 | sobol | False | Thorough search |
| `thorough` | 20 | sobol | False | Publication quality |
| `streaming` | 50 | lhs | False | Large datasets with tournament |

### Sampling Functions

```python
from nlsq.global_optimization import (
    latin_hypercube_sample,  # Stratified random sampling
    sobol_sample,  # Deterministic quasi-random
    halton_sample,  # Prime-based quasi-random
    scale_samples_to_bounds,  # Scale [0,1] samples to bounds
    center_samples_around_p0,  # Center samples on initial guess
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

### Tournament Selection (Streaming)

```python
from nlsq.global_optimization import TournamentSelector, GlobalOptimizationConfig

config = GlobalOptimizationConfig.from_preset("streaming")
selector = TournamentSelector(candidates, config)

# Run tournament on streaming data
best_candidates = selector.run_tournament(
    data_batch_generator,
    model_func,
    top_m=1,
)
```

---

## When to Use Global Optimization

**Enable multi-start when:**

- Your model has periodic functions (sin, cos) that create multiple local minima
- The objective function is multimodal or non-convex
- Initial parameter guesses are poor or unknown
- You need high confidence in finding the global optimum
- Different initial guesses produce different results

**Skip multi-start when:**

- The problem is convex (e.g., linear regression)
- You have a very good initial guess from domain knowledge
- Speed is critical and local solution is acceptable
- The problem is well-conditioned with a single clear optimum

---

## Learning Path

**Recommended sequence:**

1. Start with [01_multistart_basics](01_multistart_basics.ipynb) to understand the local minima problem
2. Learn sampling methods in [02_sampling_strategies](02_sampling_strategies.ipynb)
3. Master configuration in [03_presets_and_config](03_presets_and_config.ipynb)
4. For large datasets, study [04_tournament_selection](04_tournament_selection.ipynb)
5. See practical integration patterns in [05_multistart_integration](05_multistart_integration.ipynb)

**Next steps:**

- Continue to [Section 08: Workflow System](../08_workflow_system/) for the unified `fit()` API
- Explore [Section 04: Gallery](../04_gallery/) for domain-specific examples
- See [Section 06: Streaming](../06_streaming/) for fault-tolerant large dataset fitting

---

## Related Documentation

- [NLSQ Global Optimization Guide](https://nlsq.readthedocs.io/en/latest/global_optimization.html)
- [API Reference: GlobalOptimizationConfig](https://nlsq.readthedocs.io/en/latest/api/global_optimization.html)
- [Multi-Start Optimization Theory](https://nlsq.readthedocs.io/en/latest/theory/multistart.html)

---

## File Structure

```
07_global_optimization/
├── 01_multistart_basics.ipynb      # Notebook version
├── 01_multistart_basics.py         # Script version
├── 02_sampling_strategies.ipynb
├── 02_sampling_strategies.py
├── 03_presets_and_config.ipynb
├── 03_presets_and_config.py
├── 04_tournament_selection.ipynb
├── 04_tournament_selection.py
├── 05_multistart_integration.ipynb
├── 05_multistart_integration.py
├── figures/                        # Saved visualizations
└── README.md                       # This file
```

---

<p align="center">
<i>NLSQ v0.3.3+ | Last updated: 2025-12-19</i>
</p>
