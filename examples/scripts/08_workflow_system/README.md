# Section 08: Workflow System

**The unified `fit()` API with automatic workflow selection for any dataset size.**

---

## Overview

The workflow system provides a single entry point, `fit()`, that automatically selects
the optimal fitting strategy based on your dataset size, available memory, and optimization
goals. It unifies the three core NLSQ functions:

| Dataset Size | Underlying Function | Memory Model |
|--------------|---------------------|--------------|
| Small (<10K points) | `curve_fit()` | O(N) - full data in memory |
| Medium (10K-10M points) | `curve_fit_large()` / `LargeDatasetFitter` | O(chunk_size) - chunked |
| Large (10M+ points) | `AdaptiveHybridStreamingOptimizer` | O(batch_size) - streaming |

Instead of choosing which function to use, simply call `fit()` and let NLSQ decide:

```python
from nlsq import fit

# Works for any dataset size
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds)

# Or with a preset for specific behavior
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, preset="robust")
```

---

## Tutorials

| # | Tutorial | Level | Duration | Description |
|---|----------|-------|----------|-------------|
| 01 | [fit() Quickstart](01_fit_quickstart.ipynb) | Beginner | 15 min | Basic `fit()` usage, presets, comparison with `curve_fit()` |
| 02 | [Workflow Tiers](02_workflow_tiers.ipynb) | Intermediate | 20 min | STANDARD, CHUNKED, STREAMING, STREAMING_CHECKPOINT tiers |
| 03 | [Optimization Goals](03_optimization_goals.ipynb) | Intermediate | 20 min | FAST, ROBUST, GLOBAL, MEMORY_EFFICIENT, QUALITY goals |
| 04 | [Workflow Presets](04_workflow_presets.ipynb) | Beginner | 15 min | All 7 named presets in `WORKFLOW_PRESETS` dictionary |
| 05 | [YAML Configuration](05_yaml_configuration.ipynb) | Intermediate | 20 min | File-based config with `load_yaml_config()`, environment overrides |
| 06 | [Auto Selection](06_auto_selection.ipynb) | Advanced | 25 min | `WorkflowSelector` internals, `auto_select_workflow()`, memory tiers |
| 07 | [HPC and Checkpointing](07_hpc_and_checkpointing.ipynb) | Advanced | 30 min | PBS Pro cluster detection, checkpoint/resume, fault tolerance |

**Total time**: ~2.5 hours

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
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds)

# With a named preset
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, preset="robust")

# With explicit configuration
from nlsq import WorkflowConfig, WorkflowTier, OptimizationGoal

config = WorkflowConfig(
    tier=WorkflowTier.CHUNKED,
    goal=OptimizationGoal.QUALITY,
    enable_multistart=True,
    n_starts=10,
)
popt, pcov = fit(model, x, y, p0=p0, bounds=bounds, config=config)
```

### WorkflowConfig Dataclass

```python
from nlsq import WorkflowConfig, WorkflowTier, OptimizationGoal

# From preset
config = WorkflowConfig.from_preset("quality")

# Custom configuration
config = WorkflowConfig(
    tier=WorkflowTier.STREAMING,
    goal=OptimizationGoal.ROBUST,
    enable_multistart=True,
    n_starts=5,
    gtol=1e-8,
    ftol=1e-8,
    xtol=1e-8,
    max_iterations=1000,
    enable_checkpoints=False,
    checkpoint_dir=None,
    checkpoint_interval=100,
)

# Override specific fields
config = config.with_overrides(enable_multistart=True, n_starts=10)

# Serialize/deserialize
config_dict = config.to_dict()
restored = WorkflowConfig.from_dict(config_dict)
```

### Workflow Tiers

| Tier | Dataset Size | Memory | Description |
|------|-------------|--------|-------------|
| `STANDARD` | <10K points | O(N) | Standard `curve_fit()` |
| `CHUNKED` | 10K-10M points | O(chunk_size) | `LargeDatasetFitter` with chunking |
| `STREAMING` | 10M-100M points | O(batch_size) | Mini-batch gradient descent |
| `STREAMING_CHECKPOINT` | >100M points | O(batch_size) + disk | Streaming with checkpoints |

### Optimization Goals

| Goal | Tolerances | Multi-start | Use Case |
|------|------------|-------------|----------|
| `FAST` | Looser | Disabled | Quick exploration |
| `ROBUST` | Standard | Enabled | Production default |
| `GLOBAL` | Standard | Enabled | Synonym for ROBUST |
| `MEMORY_EFFICIENT` | Standard | Disabled | Memory-constrained |
| `QUALITY` | Tighter | Enabled | Publication results |

### Workflow Presets (WORKFLOW_PRESETS)

```python
from nlsq import WORKFLOW_PRESETS

# Available presets
presets = list(WORKFLOW_PRESETS.keys())
# ['standard', 'quality', 'fast', 'large_robust', 'streaming', 'hpc_distributed', 'memory_efficient']

# Inspect a preset
print(WORKFLOW_PRESETS["quality"])
```

| Preset | Tier | Goal | Multi-start | Best For |
|--------|------|------|-------------|----------|
| `standard` | STANDARD | ROBUST | No | General use |
| `quality` | STANDARD | QUALITY | Yes (10 starts) | Publication results |
| `fast` | STANDARD | FAST | No | Quick exploration |
| `large_robust` | CHUNKED | ROBUST | Yes (5 starts) | Large datasets |
| `streaming` | STREAMING | ROBUST | Yes (5 starts) | Very large datasets |
| `hpc_distributed` | STREAMING_CHECKPOINT | QUALITY | Yes (10 starts) | Cluster computing |
| `memory_efficient` | CHUNKED | MEMORY_EFFICIENT | No | Low memory |

### Auto Selection

```python
from nlsq.core.workflow import auto_select_workflow, WorkflowSelector

# Convenience function
config = auto_select_workflow(
    n_points=5_000_000,
    n_params=5,
    goal=OptimizationGoal.QUALITY,
)

# Full access to selection logic
selector = WorkflowSelector()
config = selector.select(n_points=5_000_000, n_params=5)
```

### YAML Configuration

Create `nlsq.yaml` in your project:

```yaml
default_workflow: "my_workflow"

workflows:
  my_workflow:
    tier: CHUNKED
    goal: QUALITY
    enable_multistart: true
    n_starts: 10
    gtol: 1.0e-10

environment:
  respect_env_vars: true
```

Load and use:

```python
from nlsq.core.workflow import load_yaml_config, get_custom_workflow

config = load_yaml_config("nlsq.yaml")
workflow = get_custom_workflow("my_workflow", "nlsq.yaml")
```

Environment variable overrides:

- `NLSQ_WORKFLOW_GOAL`: Override goal (FAST, ROBUST, QUALITY)
- `NLSQ_MEMORY_LIMIT_GB`: Override memory limit
- `NLSQ_CHECKPOINT_DIR`: Override checkpoint directory

### HPC / Cluster Support

```python
from nlsq.core.workflow import ClusterDetector, create_checkpoint_directory

# Detect PBS Pro cluster
detector = ClusterDetector()
cluster_info = detector.detect()

if cluster_info:
    print(f"PBS job: {cluster_info.job_id}")
    print(f"Available GPUs: {cluster_info.total_gpus}")

# Create timestamped checkpoint directory
checkpoint_dir = create_checkpoint_directory(base_dir="./checkpoints")
```

---

## Learning Path

### Beginner Path (30 min)

1. [01_fit_quickstart](01_fit_quickstart.ipynb) - Get started with `fit()`
2. [04_workflow_presets](04_workflow_presets.ipynb) - Use named presets

### Intermediate Path (60 min)

3. [02_workflow_tiers](02_workflow_tiers.ipynb) - Understand processing tiers
4. [03_optimization_goals](03_optimization_goals.ipynb) - Choose the right goal
5. [05_yaml_configuration](05_yaml_configuration.ipynb) - File-based configuration

### Advanced Path (55 min)

6. [06_auto_selection](06_auto_selection.ipynb) - Deep dive into selection logic
7. [07_hpc_and_checkpointing](07_hpc_and_checkpointing.ipynb) - Cluster computing

---

## When to Use What

| Scenario | Preset | Why |
|----------|--------|-----|
| Quick prototype | `fast` | Speed over precision |
| Production fitting | `standard` or `robust` | Balanced defaults |
| Publication figures | `quality` | Highest precision |
| >1M data points | `large_robust` | Memory-efficient chunking |
| >10M data points | `streaming` | Mini-batch processing |
| HPC cluster job | `hpc_distributed` | Checkpointing for fault tolerance |
| Low memory system | `memory_efficient` | Minimize RAM usage |

---

## Related Documentation

- [NLSQ Workflow Guide](https://nlsq.readthedocs.io/en/latest/workflow.html)
- [API Reference: fit()](https://nlsq.readthedocs.io/en/latest/api/fit.html)
- [API Reference: WorkflowConfig](https://nlsq.readthedocs.io/en/latest/api/workflow.html)
- [Section 07: Global Optimization](../07_global_optimization/) - Multi-start details

---

## File Structure

```
08_workflow_system/
├── 01_fit_quickstart.ipynb         # Notebook version
├── 01_fit_quickstart.py            # Script version
├── 02_workflow_tiers.ipynb
├── 02_workflow_tiers.py
├── 03_optimization_goals.ipynb
├── 03_optimization_goals.py
├── 04_workflow_presets.ipynb
├── 04_workflow_presets.py
├── 05_yaml_configuration.ipynb
├── 05_yaml_configuration.py
├── 06_auto_selection.ipynb
├── 06_auto_selection.py
├── 07_hpc_and_checkpointing.ipynb
├── 07_hpc_and_checkpointing.py
├── figures/                        # Saved visualizations
├── nlsq.yaml                       # Example YAML configuration
├── nlsq_fit.pbs                    # Example PBS Pro job script
└── README.md                       # This file
```

---

<p align="center">
<i>NLSQ v0.3.6+ | Last updated: 2025-12-23</i>
</p>
