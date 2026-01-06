# Section 08: Workflow System

**The unified `fit()` API with automatic memory-based strategy selection for any dataset size.**

---

## Overview

The workflow system provides a single entry point, `fit()`, that automatically selects
the optimal fitting strategy based on your dataset size, available memory, and optimization
goals. It uses a memory-based decision tree to choose between three strategies:

| Strategy | Dataset Size | Memory Model | Description |
|----------|--------------|--------------|-------------|
| `standard` | Fits in memory | O(N) | Full in-memory computation |
| `chunked` | Jacobian exceeds memory | O(chunk_size) | Memory-managed chunk processing |
| `streaming` | Data exceeds memory | O(batch_size) | Mini-batch gradient descent |

Instead of choosing which function to use, simply call `fit()` and let NLSQ decide:

```python
from nlsq import fit

# Works for any dataset size - automatic strategy selection
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto")

# Or with a named preset for specific behavior
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="quality")
```

---

## Tutorials

| # | Tutorial | Level | Duration | Description |
|---|----------|-------|----------|-------------|
| 01 | [fit() Quickstart](01_fit_quickstart.ipynb) | Beginner | 15 min | Basic `fit()` usage, workflow presets, comparison with `curve_fit()` |
| 02 | [Memory-Based Strategy Selection](02_workflow_tiers.ipynb) | Intermediate | 20 min | MemoryBudget, MemoryBudgetSelector, strategy decision tree |
| 03 | [Optimization Goals](03_optimization_goals.ipynb) | Intermediate | 20 min | FAST, ROBUST, QUALITY goals, adaptive tolerances |
| 04 | [Workflow Presets](04_workflow_presets.ipynb) | Beginner | 15 min | WORKFLOW_PRESETS dictionary, defense layer presets |
| 05 | [YAML Configuration](05_yaml_configuration.ipynb) | Intermediate | 20 min | File-based config, environment variable overrides |
| 06 | [Auto Selection](06_auto_selection.ipynb) | Advanced | 25 min | MemoryBudgetSelector internals, adaptive tolerances |
| 07 | [HPC and Checkpointing](07_hpc_and_checkpointing.ipynb) | Advanced | 30 min | PBS Pro cluster detection, checkpoint/resume, fault tolerance |
| 08 | [Custom Presets](08_custom_presets.ipynb) | Intermediate | 25 min | Building domain-specific configs with kwargs factory pattern |
| 09 | [Kinetics Presets](09_kinetics_presets.ipynb) | Intermediate | 20 min | Chemical/enzyme kinetics: rate constants, Michaelis-Menten |
| 10 | [SAXS Presets](10_saxs_presets.ipynb) | Intermediate | 20 min | Small-angle X-ray scattering form factor fitting |
| 11 | [XPCS Presets](11_xpcs_presets.ipynb) | Intermediate | 20 min | X-ray photon correlation spectroscopy fitting |

**Total time**: ~4 hours

---

## Prerequisites

Before starting this section, you should be familiar with:

- Basic `curve_fit()` usage (see [01_getting_started/nlsq_quickstart](../01_getting_started/nlsq_quickstart.ipynb))
- Global optimization concepts (see [Section 07](../07_global_optimization/))

---

## API Summary

### The `fit()` Function

```python
from nlsq import fit

# Basic usage - automatic workflow selection
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto")

# With a named preset
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="quality")

# With explicit parameters
popt, pcov = fit(
    model,
    x,
    y,
    p0=p0,
    bounds=bounds,
    workflow="standard",
    multistart=True,
    n_starts=10,
    sampler="sobol",
    gtol=1e-10,
)
```

### Memory-Based Strategy Selection

```python
from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

# Compute memory requirements
budget = MemoryBudget.compute(n_points=1_000_000, n_params=5)
print(f"Data: {budget.data_gb:.2f} GB")
print(f"Jacobian: {budget.jacobian_gb:.2f} GB")
print(f"Peak: {budget.peak_gb:.2f} GB")
print(f"Fits in memory: {budget.fits_in_memory}")

# Automatic strategy selection
selector = MemoryBudgetSelector(safety_factor=0.75)
strategy, config = selector.select(n_points=1_000_000, n_params=5)
print(f"Selected strategy: {strategy}")  # 'standard', 'chunked', or 'streaming'
```

### Decision Tree

```
1. Compute MemoryBudget (data_gb, jacobian_gb, peak_gb)
2. If data_gb > threshold -> STREAMING
3. Else if peak_gb > threshold -> CHUNKED
4. Else -> STANDARD
```

### Workflow Presets (WORKFLOW_PRESETS)

```python
from nlsq.core.minpack import WORKFLOW_PRESETS

# Available presets
presets = list(WORKFLOW_PRESETS.keys())
# ['standard', 'quality', 'fast', 'large_robust', 'streaming', 'hpc_distributed', ...]

# Inspect a preset
print(WORKFLOW_PRESETS["quality"])
```

| Preset | Strategy | Multi-start | Best For |
|--------|----------|-------------|----------|
| `fast` | standard | No | Quick exploration |
| `standard` | standard | No | General use |
| `quality` | standard | Yes (10 starts) | Publication results |
| `large_robust` | chunked | Yes (5 starts) | Large datasets |
| `streaming` | streaming | Yes (5 starts) | Very large datasets |
| `hpc_distributed` | streaming | Yes (10 starts) | Cluster computing |

### Optimization Goals

```python
from nlsq import OptimizationGoal
from nlsq.core.workflow import calculate_adaptive_tolerances

# Get adaptive tolerances based on dataset size and goal
tolerances = calculate_adaptive_tolerances(
    n_points=1_000_000, goal=OptimizationGoal.QUALITY
)
print(f"gtol: {tolerances['gtol']}")
```

| Goal | Tolerances | Multi-start | Use Case |
|------|------------|-------------|----------|
| `FAST` | Looser | Disabled | Quick exploration |
| `ROBUST` | Standard | Enabled | Production default |
| `QUALITY` | Tighter | Enabled | Publication results |

### Custom Presets (kwargs Factory Pattern)

```python
from nlsq import fit


def create_my_domain_preset() -> dict:
    """Create a preset for my specific application."""
    return {
        "workflow": "standard",
        "gtol": 1e-9,
        "ftol": 1e-9,
        "xtol": 1e-9,
        "multistart": True,
        "n_starts": 15,
        "sampler": "sobol",
    }


# Use with fit()
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, **create_my_domain_preset())
```

### YAML Configuration

Create `nlsq.yaml` in your project:

```yaml
default_workflow: standard
memory_limit_gb: 16.0

workflows:
  high_precision:
    gtol: 1.0e-10
    ftol: 1.0e-10
    xtol: 1.0e-10
    enable_multistart: true
    n_starts: 20
    sampler: lhs

  quick_explore:
    gtol: 1.0e-5
    ftol: 1.0e-5
    xtol: 1.0e-5
    enable_multistart: false
```

Environment variable overrides:

- `NLSQ_MEMORY_LIMIT_GB`: Override memory limit
- `NLSQ_DEFAULT_WORKFLOW`: Override default workflow
- `NLSQ_CHECKPOINT_DIR`: Override checkpoint directory

### Defense Layer Presets (Streaming)

```python
from nlsq import HybridStreamingConfig

# For checkpoint resume (warm-start protection)
config = HybridStreamingConfig.defense_strict()

# For exploration (rough initial guesses)
config = HybridStreamingConfig.defense_relaxed()

# For production scientific computing
config = HybridStreamingConfig.scientific_default()
```

### HPC / Cluster Support

```python
from nlsq.core.workflow import ClusterDetector, ClusterInfo

# Detect PBS Pro cluster
detector = ClusterDetector()
cluster_info = detector.detect()

if cluster_info:
    print(f"PBS job: {cluster_info.job_id}")
    print(f"Available GPUs: {cluster_info.total_gpus}")
```

---

## Learning Path

### Beginner Path (30 min)

1. [01_fit_quickstart](01_fit_quickstart.ipynb) - Get started with `fit()`
2. [04_workflow_presets](04_workflow_presets.ipynb) - Use named presets

### Intermediate Path (60 min)

3. [02_workflow_tiers](02_workflow_tiers.ipynb) - Memory-based strategy selection
4. [03_optimization_goals](03_optimization_goals.ipynb) - Choose the right goal
5. [05_yaml_configuration](05_yaml_configuration.ipynb) - File-based configuration

### Advanced Path (55 min)

6. [06_auto_selection](06_auto_selection.ipynb) - Deep dive into selection logic
7. [07_hpc_and_checkpointing](07_hpc_and_checkpointing.ipynb) - Cluster computing

### Domain-Specific Path (85 min)

8. [08_custom_presets](08_custom_presets.ipynb) - Building custom presets with kwargs factory
9. [09_kinetics_presets](09_kinetics_presets.ipynb) - Chemical/enzyme kinetics fitting
10. [10_saxs_presets](10_saxs_presets.ipynb) - SAXS form factor fitting
11. [11_xpcs_presets](11_xpcs_presets.ipynb) - XPCS correlation function fitting

---

## When to Use What

| Scenario | Workflow | Why |
|----------|----------|-----|
| Quick prototype | `workflow="fast"` | Speed over precision |
| Production fitting | `workflow="standard"` | Balanced defaults |
| Publication figures | `workflow="quality"` | Highest precision |
| >1M data points | `workflow="auto"` | Automatic chunked selection |
| >10M data points | `workflow="streaming"` | Mini-batch processing |
| HPC cluster job | `workflow="streaming"` + defense presets | Checkpointing for fault tolerance |
| Custom domain | `**create_my_preset()` | kwargs factory pattern |

---

## Related Documentation

- [NLSQ Workflow Guide](https://nlsq.readthedocs.io/en/latest/workflow.html)
- [API Reference: fit()](https://nlsq.readthedocs.io/en/latest/api/fit.html)
- [Section 07: Global Optimization](../07_global_optimization/) - Multi-start details

---

## File Structure

```
08_workflow_system/
├── 01_fit_quickstart.ipynb         # Notebook version
├── 02_workflow_tiers.ipynb         # Memory-based strategy selection
├── 03_optimization_goals.ipynb
├── 04_workflow_presets.ipynb
├── 05_yaml_configuration.ipynb
├── 06_auto_selection.ipynb
├── 07_hpc_and_checkpointing.ipynb
├── 08_custom_presets.ipynb         # kwargs factory pattern guide
├── 09_kinetics_presets.ipynb       # Kinetics fitting examples
├── 10_saxs_presets.ipynb           # SAXS form factor examples
├── 11_xpcs_presets.ipynb           # XPCS correlation examples
├── figures/                        # Saved visualizations
└── README.md                       # This file
```

---

<p align="center">
<i>NLSQ v0.6.0 | Last updated: 2026-01-06</i>
</p>
