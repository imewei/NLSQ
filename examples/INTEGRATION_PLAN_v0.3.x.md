# NLSQ Examples Integration Plan v0.3.x

## Executive Summary

This document outlines the integration plan for adding **12 new tutorials** (24 files: 12 notebooks + 12 scripts) demonstrating the multi-start global optimization (v0.3.3) and unified workflow system (v0.3.4) features to the NLSQ examples directory.

**Total New Files:** 26 (including 2 section README files)

---

## New Directory Structure

```
examples/
├── scripts/
│   ├── 07_global_optimization/      # NEW: 5 tutorials + README
│   │   ├── README.md
│   │   ├── 01_multistart_basics.ipynb
│   │   ├── 01_multistart_basics.py
│   │   ├── 02_sampling_strategies.ipynb
│   │   ├── 02_sampling_strategies.py
│   │   ├── 03_presets_and_config.ipynb
│   │   ├── 03_presets_and_config.py
│   │   ├── 04_tournament_selection.ipynb
│   │   ├── 04_tournament_selection.py
│   │   ├── 05_multistart_integration.ipynb
│   │   └── 05_multistart_integration.py
│   │
│   └── 08_workflow_system/          # NEW: 7 tutorials + README
│       ├── README.md
│       ├── 01_fit_quickstart.ipynb
│       ├── 01_fit_quickstart.py
│       ├── 02_workflow_tiers.ipynb
│       ├── 02_workflow_tiers.py
│       ├── 03_optimization_goals.ipynb
│       ├── 03_optimization_goals.py
│       ├── 04_workflow_presets.ipynb
│       ├── 04_workflow_presets.py
│       ├── 05_yaml_configuration.ipynb
│       ├── 05_yaml_configuration.py
│       ├── 06_auto_selection.ipynb
│       ├── 06_auto_selection.py
│       ├── 07_hpc_and_checkpointing.ipynb
│       └── 07_hpc_and_checkpointing.py
```

---

## Section 07: Global Optimization Tutorials

### 07/01_multistart_basics - Introduction to Multi-Start Optimization

**Level:** ●●○ Intermediate | **Duration:** 20 minutes

**Learning Objectives:**
- Understand why local optimization can fail (local minima trap)
- Configure basic multi-start optimization with `GlobalOptimizationConfig`
- Interpret multi-start results and select best solution

**Key Content:**
```python
from nlsq import curve_fit, GlobalOptimizationConfig
import jax.numpy as jnp

def multimodal_model(x, a, b, c):
    """Model with multiple local minima"""
    return a * jnp.sin(b * x) + c

# Configure multi-start (10 random starting points)
config = GlobalOptimizationConfig(
    n_starts=10,
    sampling_method="lhs",
    seed=42
)

# Fit with global optimization
popt, pcov = curve_fit(
    multimodal_model, x, y,
    p0=[1.0, 1.0, 0.0],
    bounds=([0, 0, -5], [10, 10, 5]),
    global_optimization=config
)
```

**Visualizations:**
- Loss landscape with local minima marked
- Convergence comparison: single-start vs multi-start
- Starting point distribution

---

### 07/02_sampling_strategies - LHS, Sobol, and Halton Samplers

**Level:** ●●○ Intermediate | **Duration:** 25 minutes

**Learning Objectives:**
- Compare Latin Hypercube (LHS), Sobol, and Halton sampling methods
- Understand space-filling properties
- Choose the right sampler for your problem

**Key Content:**
```python
# Compare sampling strategies
configs = {
    "LHS": GlobalOptimizationConfig(n_starts=20, sampling_method="lhs"),
    "Sobol": GlobalOptimizationConfig(n_starts=20, sampling_method="sobol"),
    "Halton": GlobalOptimizationConfig(n_starts=20, sampling_method="halton"),
    "Random": GlobalOptimizationConfig(n_starts=20, sampling_method="random"),
}

for name, config in configs.items():
    popt, _ = curve_fit(model, x, y, global_optimization=config)
    print(f"{name}: residual = {compute_residual(popt):.6f}")
```

**Visualizations:**
- 2D scatter plots showing point distributions for each method
- Discrepancy metric comparison
- Success rate across different starting point counts

---

### 07/03_presets_and_config - GlobalOptimizationConfig Deep Dive

**Level:** ●●○ Intermediate | **Duration:** 20 minutes

**Learning Objectives:**
- Master all `GlobalOptimizationConfig` parameters
- Use built-in presets: 'quick', 'thorough', 'exhaustive'
- Create custom configurations for specific problem types

**Key Content:**
```python
from nlsq import GlobalOptimizationConfig

# Built-in presets
config_quick = GlobalOptimizationConfig.quick()      # 5 starts
config_thorough = GlobalOptimizationConfig.thorough()  # 20 starts
config_exhaustive = GlobalOptimizationConfig.exhaustive()  # 50 starts

# Custom configuration
custom_config = GlobalOptimizationConfig(
    n_starts=15,
    sampling_method="sobol",
    parallel=True,
    seed=42,
    early_stopping=True,
    early_stopping_threshold=1e-10,
)
```

---

### 07/04_tournament_selection - Streaming Tournament Optimization

**Level:** ●●● Advanced | **Duration:** 30 minutes

**Learning Objectives:**
- Understand tournament selection for memory-efficient global optimization
- Use `TournamentSelector` for streaming scenarios
- Configure tournament parameters

**Key Content:**
```python
from nlsq.large_dataset import TournamentSelector

# Tournament selector for streaming data
selector = TournamentSelector(
    tournament_size=8,
    selection_pressure=2.0,
    memory_limit_gb=4.0,
)

# Process candidates in streaming fashion
for chunk_result in streaming_results:
    selector.add_candidate(chunk_result)

# Get best solution
best_params, best_cost = selector.get_best()
```

---

### 07/05_multistart_integration - Combining Multi-Start with curve_fit

**Level:** ●●○ Intermediate | **Duration:** 25 minutes

**Learning Objectives:**
- Integrate multi-start optimization into existing curve_fit workflows
- Handle bounds with multi-start
- Combine with large dataset handling

**Key Content:**
```python
from nlsq import curve_fit, curve_fit_large, GlobalOptimizationConfig

# Standard fitting with global optimization
popt, pcov = curve_fit(
    model, x, y,
    p0=[1.0, 1.0],
    bounds=([0, 0], [10, 10]),
    global_optimization=GlobalOptimizationConfig(n_starts=10)
)

# Large dataset with global optimization
popt, pcov = curve_fit_large(
    model, x_large, y_large,
    p0=[1.0, 1.0],
    memory_limit_gb=4.0,
    global_optimization=GlobalOptimizationConfig(n_starts=5)
)
```

---

## Section 08: Workflow System Tutorials

### 08/01_fit_quickstart - Unified fit() Entry Point

**Level:** ●○○ Beginner | **Duration:** 15 minutes

**Learning Objectives:**
- Use the unified `fit()` function for automatic workflow selection
- Understand return value compatibility with `curve_fit()`
- Choose between `fit()`, `curve_fit()`, and `curve_fit_large()`

**Key Content:**
```python
from nlsq import fit
import jax.numpy as jnp

def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c

# Auto-select workflow (handles any dataset size)
popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2])

# With preset for common scenarios
popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2], preset="robust")

# With custom configuration
from nlsq import WorkflowConfig, OptimizationGoal
config = WorkflowConfig(goal=OptimizationGoal.QUALITY)
popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2], config=config)
```

---

### 08/02_workflow_tiers - Processing Strategies Explained

**Level:** ●●○ Intermediate | **Duration:** 20 minutes

**Learning Objectives:**
- Understand WorkflowTier: STANDARD, CHUNKED, STREAMING, STREAMING_CHECKPOINT
- Know when each tier is automatically selected
- Override automatic tier selection

**Key Content:**
```python
from nlsq import fit, WorkflowConfig, WorkflowTier

# Force specific tier
config = WorkflowConfig(tier=WorkflowTier.STREAMING)
popt, pcov = fit(model, x, y, p0=p0, config=config)

# Tier selection thresholds:
# STANDARD:    < 100K points, fits in memory
# CHUNKED:     100K-1M points, automatic chunking
# STREAMING:   > 1M points, streaming optimization
# STREAMING_CHECKPOINT: > 10M points, with checkpointing
```

**Visualizations:**
- Tier selection decision tree
- Memory usage comparison across tiers
- Performance benchmarks

---

### 08/03_optimization_goals - Goal-Driven Optimization

**Level:** ●●○ Intermediate | **Duration:** 20 minutes

**Learning Objectives:**
- Use OptimizationGoal: FAST, ROBUST, GLOBAL, MEMORY_EFFICIENT, QUALITY
- Understand internal settings each goal applies
- Combine goals with tiers

**Key Content:**
```python
from nlsq import fit, WorkflowConfig, OptimizationGoal

# FAST: Minimum iterations, relaxed tolerances
config_fast = WorkflowConfig(goal=OptimizationGoal.FAST)

# ROBUST: Multi-start with 5 starting points
config_robust = WorkflowConfig(goal=OptimizationGoal.ROBUST)

# GLOBAL: Thorough search with 20 starting points
config_global = WorkflowConfig(goal=OptimizationGoal.GLOBAL)

# MEMORY_EFFICIENT: Aggressive chunking
config_memory = WorkflowConfig(goal=OptimizationGoal.MEMORY_EFFICIENT)

# QUALITY: Tight tolerances with validation
config_quality = WorkflowConfig(goal=OptimizationGoal.QUALITY)
```

---

### 08/04_workflow_presets - Named Configurations

**Level:** ●○○ Beginner | **Duration:** 15 minutes

**Learning Objectives:**
- Use WORKFLOW_PRESETS for common scenarios
- Understand what each preset configures
- Customize presets as starting points

**Key Content:**
```python
from nlsq import fit, WORKFLOW_PRESETS

# Available presets
# 'fast':            Quick results, relaxed tolerances
# 'robust':          Multi-start, 5 starting points
# 'global':          Thorough search, 20 starting points
# 'memory_efficient': Aggressive chunking, streaming fallback
# 'quality':         Tight tolerances, validation passes
# 'hpc':             PBS Pro cluster configuration
# 'streaming':       Tournament selection for streaming

# Use preset directly
popt, pcov = fit(model, x, y, p0=p0, preset="robust")

# Inspect preset configuration
print(WORKFLOW_PRESETS["robust"])
```

---

### 08/05_yaml_configuration - File-Based Configuration

**Level:** ●●○ Intermediate | **Duration:** 20 minutes

**Learning Objectives:**
- Create and use nlsq.yaml configuration files
- Set tolerances, memory limits, and checkpointing via YAML
- Override configuration with environment variables

**Key Content:**
```yaml
# nlsq.yaml
workflow:
  goal: robust
  memory_limit_gb: 16.0
  enable_checkpointing: true
  checkpoint_dir: ./checkpoints

tolerances:
  ftol: 1e-10
  xtol: 1e-10
  gtol: 1e-10

cluster:
  type: pbs
  nodes: 4
  gpus_per_node: 2
```

```python
# Using YAML config
from nlsq.workflow import load_yaml_config, get_custom_workflow

config_dict = load_yaml_config("nlsq.yaml")
workflow_config = get_custom_workflow(config_dict)
popt, pcov = fit(model, x, y, p0=p0, config=workflow_config)
```

**Environment Variables:**
- `NLSQ_WORKFLOW_GOAL`: Override optimization goal
- `NLSQ_MEMORY_LIMIT_GB`: Override memory limit
- `NLSQ_CHECKPOINT_DIR`: Override checkpoint directory

---

### 08/06_auto_selection - Workflow Selection Internals

**Level:** ●●● Advanced | **Duration:** 25 minutes

**Learning Objectives:**
- Understand WorkflowSelector decision logic
- Use auto_select_workflow() for custom scenarios
- Inspect DatasetSizeTier and MemoryTier classification

**Key Content:**
```python
from nlsq.workflow import (
    WorkflowSelector,
    auto_select_workflow,
    DatasetSizeTier,
    MemoryTier,
    get_total_available_memory_gb,
    get_memory_tier,
)

# Check system memory
memory_gb = get_total_available_memory_gb()
memory_tier = get_memory_tier(memory_gb)
print(f"Available memory: {memory_gb:.1f} GB ({memory_tier})")

# Classify dataset size
n_points = len(x)
size_tier = DatasetSizeTier.from_size(n_points)
print(f"Dataset: {n_points:,} points ({size_tier})")

# Get recommended workflow
selector = WorkflowSelector()
workflow = selector.select(n_points=n_points, available_memory_gb=memory_gb)
print(f"Recommended workflow: {workflow}")
```

---

### 08/07_hpc_and_checkpointing - HPC and Long-Running Fits

**Level:** ●●● Advanced | **Duration:** 30 minutes

**Learning Objectives:**
- Detect HPC cluster environments (PBS Pro, SLURM)
- Enable checkpointing for fault-tolerant fitting
- Resume fits from checkpoints

**Key Content:**
```python
from nlsq import fit, WorkflowConfig, WorkflowTier
from nlsq.workflow import ClusterDetector, ClusterInfo

# Detect cluster environment
detector = ClusterDetector()
cluster_info: ClusterInfo = detector.detect()
print(f"Cluster type: {cluster_info.cluster_type}")
print(f"Nodes: {cluster_info.nodes}, GPUs: {cluster_info.gpus_per_node}")

# Configure checkpointing
config = WorkflowConfig(
    tier=WorkflowTier.STREAMING_CHECKPOINT,
    enable_checkpointing=True,
    checkpoint_dir="./nlsq_checkpoints",
    checkpoint_interval=100,  # chunks
)

popt, pcov = fit(model, x, y, p0=p0, config=config)
```

**HPC Job Script Example (PBS Pro):**
```bash
#!/bin/bash
#PBS -N nlsq_fit
#PBS -l select=4:ncpus=40:ngpus=2
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
python fit_script.py
```

---

## Implementation Phases

### Phase 1: Foundation (Estimated: 6 tutorials)
1. `07/01_multistart_basics.{ipynb,py}`
2. `08/01_fit_quickstart.{ipynb,py}`
3. `07/05_multistart_integration.{ipynb,py}`

### Phase 2: Configuration (Estimated: 6 tutorials)
4. `07/03_presets_and_config.{ipynb,py}`
5. `08/03_optimization_goals.{ipynb,py}`
6. `08/04_workflow_presets.{ipynb,py}`

### Phase 3: Advanced Features (Estimated: 6 tutorials)
7. `07/02_sampling_strategies.{ipynb,py}`
8. `08/02_workflow_tiers.{ipynb,py}`
9. `08/06_auto_selection.{ipynb,py}`

### Phase 4: Specialized Topics (Estimated: 6 tutorials)
10. `07/04_tournament_selection.{ipynb,py}`
11. `08/05_yaml_configuration.{ipynb,py}`
12. `08/07_hpc_and_checkpointing.{ipynb,py}`

### Phase 5: Documentation
13. `07_global_optimization/README.md`
14. `08_workflow_system/README.md`

---

## Template Compliance

All tutorials will follow the existing `_templates/00_UNIVERSAL_TEMPLATE.md` structure:

1. **Header:** Title, description, duration, level indicator
2. **Learning Objectives:** 3-4 measurable skills
3. **Learning Path:** Position in tutorial sequence
4. **Prerequisites:** Required knowledge and software
5. **Why This Matters:** Real-world motivation
6. **Quick Start:** 30-second minimal example
7. **Setup:** Import configuration
8. **Tutorial Content:** Progressive sections
9. **Key Takeaways:** Summary points
10. **Common Questions:** FAQ section
11. **Related Resources:** Links to next steps

---

## Dependencies

### Section 07 (Global Optimization)
```python
from nlsq import (
    curve_fit,
    curve_fit_large,
    GlobalOptimizationConfig,
    MultiStartOrchestrator,
    TournamentSelector,
)
```

### Section 08 (Workflow System)
```python
from nlsq import (
    fit,
    WorkflowConfig,
    WorkflowTier,
    OptimizationGoal,
    WORKFLOW_PRESETS,
)
from nlsq.workflow import (
    WorkflowSelector,
    auto_select_workflow,
    DatasetSizeTier,
    MemoryTier,
    ClusterInfo,
    ClusterDetector,
    get_total_available_memory_gb,
    get_memory_tier,
    load_yaml_config,
    get_custom_workflow,
)
```

---

## Validation Checklist

For each tutorial:
- [ ] Notebook executes cleanly (kernel restart + run all)
- [ ] Script runs standalone without errors
- [ ] Output is reproducible (seeds set)
- [ ] Visualizations render correctly
- [ ] Links to documentation are valid
- [ ] Code uses `jnp` (not `np`) in model functions
- [ ] Prerequisites are clearly stated
- [ ] Learning objectives are measurable
- [ ] Follows `_templates/00_UNIVERSAL_TEMPLATE.md` structure

---

## File Count Summary

| Category | Notebooks | Scripts | README | Total |
|----------|-----------|---------|--------|-------|
| 07_global_optimization | 5 | 5 | 1 | 11 |
| 08_workflow_system | 7 | 7 | 1 | 15 |
| **Total** | **12** | **12** | **2** | **26** |

---

## Related Documentation

- [Global Optimization Guide](../docs/guides/global_optimization.md)
- [Workflow System API](../docs/api/nlsq.workflow.rst)
- [Large Dataset Handling](../docs/guides/large_datasets.md)
- [CHANGELOG v0.3.3 & v0.3.4](../CHANGELOG.md)
