# Section 08: Workflow System

**The unified `fit()` API with automatic memory-based strategy selection (v0.6.3).**

---

## Overview

NLSQ v0.6.3 simplifies the workflow system to **three smart workflows** that automatically
select the optimal fitting strategy based on your dataset size, available memory, and
optimization requirements.

### The Three Workflows

| Workflow | Description | Bounds | Use Case |
|----------|-------------|--------|----------|
| `auto` | Memory-aware local optimization | Optional | **Default**. Standard curve fitting. |
| `auto_global` | Memory-aware global optimization | Required | Multi-modal problems, unknown initial guess. |
| `hpc` | `auto_global` + checkpointing | Required | Long-running HPC jobs. |

### Memory Strategies (Auto-Selected)

Each workflow automatically selects the optimal memory strategy:

| Strategy | Dataset Size | Memory Model | Description |
|----------|--------------|--------------|-------------|
| `standard` | Fits in memory | O(N) | Full in-memory computation |
| `chunked` | Jacobian exceeds memory | O(chunk_size) | Memory-managed chunk processing |
| `streaming` | Data exceeds memory | O(batch_size) | Mini-batch gradient descent |

```python
from nlsq import fit

# workflow='auto' - Local optimization (default)
popt, pcov = fit(model, x, y, p0=p0, workflow="auto")

# workflow='auto_global' - Global optimization (bounds required)
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto_global")

# workflow='hpc' - HPC with checkpointing
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

## Scripts

| # | Script | Level | Description |
|---|--------|-------|-------------|
| 01 | [01_fit_quickstart.py](01_fit_quickstart.py) | Beginner | Basic `fit()` usage with the 3 workflows |
| 02 | [02_workflow_tiers.py](02_workflow_tiers.py) | Intermediate | MemoryBudget, MemoryBudgetSelector, strategy decision tree |
| 03 | [03_auto_selection.py](03_auto_selection.py) | Advanced | `auto_global` method selection (CMA-ES vs Multi-Start) |
| 04 | [04_hpc_and_checkpointing.py](04_hpc_and_checkpointing.py) | Advanced | PBS Pro cluster detection, checkpoint/resume |

---

## Running the Scripts

```bash
# Run a single script
python examples/scripts/08_workflow_system/01_fit_quickstart.py

# Run with quick mode (reduced iterations)
NLSQ_EXAMPLES_QUICK=1 python examples/scripts/08_workflow_system/04_hpc_and_checkpointing.py
```

---

## API Summary

### The `fit()` Function

```python
from nlsq import fit

# workflow='auto' - Local optimization (bounds optional)
popt, pcov = fit(model, x, y, p0=p0, workflow="auto")

# workflow='auto' with bounds
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto")

# workflow='auto_global' - Global optimization (bounds required)
# Auto-selects CMA-ES or Multi-Start based on parameter scale ratio
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto_global", n_starts=10)

# workflow='hpc' - For HPC cluster jobs with checkpointing
popt, pcov = fit(
    model,
    x,
    y,
    p0=p0,
    bounds=bounds,
    workflow="hpc",
    checkpoint_dir="/scratch/my_job/checkpoints",
    checkpoint_interval=10,
)

# Tolerance control (set directly, not via presets)
popt, pcov = fit(
    model, x, y, p0=p0, workflow="auto", gtol=1e-10, ftol=1e-10, xtol=1e-10
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

### Global Method Selection (auto_global only)

```python
from nlsq.global_optimization.method_selector import MethodSelector

selector = MethodSelector()
method = selector.select("auto", lower_bounds, upper_bounds)
# Returns "cmaes" or "multi-start"
```

- **CMA-ES**: Selected when parameter scale ratio > 1000 AND evosax is available
- **Multi-Start**: Selected otherwise

---

## When to Use What

| Scenario | Workflow | Why |
|----------|----------|-----|
| Standard curve fitting | `workflow="auto"` | Default, automatic memory handling |
| Well-conditioned problem | `workflow="auto"` | Local optimization is sufficient |
| Multi-modal problem | `workflow="auto_global"` | Global search for multiple minima |
| Unknown initial guess | `workflow="auto_global"` | Explores parameter space |
| Wide parameter bounds | `workflow="auto_global"` | May select CMA-ES for scale-invariance |
| Long HPC job | `workflow="hpc"` | Checkpointing for fault tolerance |
| Fast exploration | `workflow="auto", gtol=1e-6` | Looser tolerances |
| Publication quality | `workflow="auto", gtol=1e-10` | Tighter tolerances |

---

## Migration from Old Presets

The following presets have been removed in v0.6.3. Use these equivalents:

| Old Preset | New Equivalent |
|------------|----------------|
| `standard` | `workflow="auto"` |
| `fast` | `workflow="auto", gtol=1e-6, ftol=1e-6, xtol=1e-6` |
| `quality` | `workflow="auto_global", n_starts=20` |
| `large_robust` | `workflow="auto"` (auto-detects large data) |
| `streaming` | `workflow="auto"` (auto-detects memory pressure) |
| `hpc_distributed` | `workflow="hpc"` |
| `cmaes` | `workflow="auto_global"` (auto-selects CMA-ES) |
| `global_auto` | `workflow="auto_global"` |

---

## File Structure

```
08_workflow_system/
├── 01_fit_quickstart.py       # Basic fit() usage with 3 workflows
├── 02_workflow_tiers.py       # Memory strategy selection
├── 03_auto_selection.py       # Global method selection (CMA-ES vs Multi-Start)
├── 04_hpc_and_checkpointing.py # HPC workflow with checkpointing
├── figures/                   # Saved visualizations
├── nlsq_fit.pbs              # PBS job script template
└── README.md                  # This file
```

---

<p align="center">
<i>NLSQ v0.6.3 | Last updated: 2026-01-11</i>
</p>
