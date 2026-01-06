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

## Scripts

| # | Script | Level | Description |
|---|--------|-------|-------------|
| 01 | [01_fit_quickstart.py](01_fit_quickstart.py) | Beginner | Basic `fit()` usage, workflow presets, comparison with `curve_fit()` |
| 02 | [02_workflow_tiers.py](02_workflow_tiers.py) | Intermediate | MemoryBudget, MemoryBudgetSelector, strategy decision tree |
| 03 | [03_optimization_goals.py](03_optimization_goals.py) | Intermediate | FAST, ROBUST, QUALITY goals, adaptive tolerances |
| 04 | [04_workflow_presets.py](04_workflow_presets.py) | Beginner | WORKFLOW_PRESETS dictionary, defense layer presets |
| 05 | [05_yaml_configuration.py](05_yaml_configuration.py) | Intermediate | File-based config, environment variable overrides |
| 06 | [06_auto_selection.py](06_auto_selection.py) | Advanced | MemoryBudgetSelector internals, adaptive tolerances |
| 07 | [07_hpc_and_checkpointing.py](07_hpc_and_checkpointing.py) | Advanced | PBS Pro cluster detection, checkpoint/resume, fault tolerance |
| 08 | [08_custom_presets.py](08_custom_presets.py) | Intermediate | Building domain-specific configs with kwargs factory pattern |
| 09 | [09_kinetics_presets.py](09_kinetics_presets.py) | Intermediate | Chemical/enzyme kinetics: rate constants, Michaelis-Menten |
| 10 | [10_saxs_presets.py](10_saxs_presets.py) | Intermediate | Small-angle X-ray scattering form factor fitting |
| 11 | [11_xpcs_presets.py](11_xpcs_presets.py) | Intermediate | X-ray photon correlation spectroscopy fitting |

---

## Running the Scripts

```bash
# Run a single script
python examples/scripts/08_workflow_system/01_fit_quickstart.py

# Run with quick mode (reduced iterations)
NLSQ_EXAMPLES_QUICK=1 python examples/scripts/08_workflow_system/07_hpc_and_checkpointing.py
```

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

## File Structure

```
08_workflow_system/
├── 01_fit_quickstart.py
├── 02_workflow_tiers.py
├── 03_optimization_goals.py
├── 04_workflow_presets.py
├── 05_yaml_configuration.py
├── 06_auto_selection.py
├── 07_hpc_and_checkpointing.py
├── 08_custom_presets.py
├── 09_kinetics_presets.py
├── 10_saxs_presets.py
├── 11_xpcs_presets.py
├── figures/                    # Saved visualizations
└── README.md                   # This file
```

---

<p align="center">
<i>NLSQ v0.6.0 | Last updated: 2026-01-06</i>
</p>
