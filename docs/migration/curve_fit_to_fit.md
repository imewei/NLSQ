# Migration Guide: curve_fit() to fit()

This guide documents the transition from the lower-level `curve_fit()` API to the unified `fit()` entry point introduced in NLSQ v0.4.0.

## Overview

The new `fit()` function provides a simplified API with automatic workflow selection based on dataset size, memory availability, and optimization goals. It is designed to be the primary entry point for most users while `curve_fit()` remains available for advanced use cases requiring full control.

## Quick Start

### Before (curve_fit)

```python
from nlsq import curve_fit
import jax.numpy as jnp


def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# Basic fitting
popt, pcov = curve_fit(model, xdata, ydata, p0=[1, 1, 0])

# With bounds
popt, pcov = curve_fit(
    model, xdata, ydata, p0=[1, 1, 0], bounds=([0, 0, -10], [10, 5, 10])
)

# With multi-start optimization
popt, pcov = curve_fit(
    model,
    xdata,
    ydata,
    p0=[1, 1, 0],
    bounds=([0, 0, -10], [10, 5, 10]),
    multistart=True,
    n_starts=10,
)
```

### After (fit)

```python
from nlsq import fit
import jax.numpy as jnp


def model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


# Basic fitting (auto-selects appropriate workflow)
popt, pcov = fit(model, xdata, ydata, p0=[1, 1, 0])

# With goal-based configuration
popt, pcov = fit(
    model, xdata, ydata, p0=[1, 1, 0], goal="quality", bounds=([0, 0, -10], [10, 5, 10])
)

# Using named workflow presets
popt, pcov = fit(
    model,
    xdata,
    ydata,
    p0=[1, 1, 0],
    workflow="robust",  # Enables multi-start with 5 starts
    bounds=([0, 0, -10], [10, 5, 10]),
)
```

## Key Concepts

### Workflow Presets

The `fit()` function supports named workflow presets:

| Preset      | Multi-start | n_starts | Use Case                           |
|-------------|-------------|----------|------------------------------------|
| `fast`      | No          | 0        | Quick fits, well-behaved problems  |
| `robust`    | Yes         | 5        | General-purpose, moderate search   |
| `global`    | Yes         | 20       | Thorough search, complex surfaces  |
| `streaming` | Yes         | 10       | Very large datasets (streaming)    |
| `large`     | Yes         | 5        | Auto-detect large dataset strategy |

### Optimization Goals

Goals influence tolerance settings and algorithm behavior:

| Goal               | Description                                     | gtol     |
|--------------------|-------------------------------------------------|----------|
| `fast`             | Loose tolerances, single-start                  | 1e-6     |
| `robust`           | Balanced settings with multi-start              | 1e-8     |
| `quality`          | Tight tolerances, thorough search               | 1e-10    |
| `memory_efficient` | Prioritizes streaming over chunking             | 1e-8     |

### Workflow vs Goal

- **Workflow**: Determines the backend and processing strategy (standard, chunked, streaming)
- **Goal**: Adjusts tolerances and multi-start settings for optimization quality

These can be used independently:

```python
# Use standard workflow with quality goal
fit(model, x, y, workflow="standard", goal="quality")

# Use streaming workflow with fast goal
fit(model, x, y, workflow="streaming", goal="fast")
```

## Migration Examples

### Example 1: Basic Fitting

```python
# Before
popt, pcov = curve_fit(model, x, y, p0=[1, 2, 3])

# After (equivalent)
popt, pcov = fit(model, x, y, p0=[1, 2, 3])
```

### Example 2: Bounded Optimization with Multi-start

```python
# Before
popt, pcov = curve_fit(
    model,
    x,
    y,
    p0=[1, 2, 3],
    bounds=([0, 0, 0], [10, 10, 10]),
    multistart=True,
    n_starts=10,
)

# After (using preset)
popt, pcov = fit(
    model, x, y, p0=[1, 2, 3], bounds=([0, 0, 0], [10, 10, 10]), preset="robust"
)  # Automatically enables multi-start

# Or with goal
popt, pcov = fit(
    model, x, y, p0=[1, 2, 3], bounds=([0, 0, 0], [10, 10, 10]), goal="quality"
)  # Enables multi-start with 20 starts
```

### Example 3: Large Dataset Processing

```python
# Before
from nlsq import curve_fit_large

popt, pcov = curve_fit_large(
    model, large_x, large_y, p0=[1, 2, 3], memory_limit_gb=8.0, show_progress=True
)

# After (automatic detection)
popt, pcov = fit(
    model, large_x, large_y, p0=[1, 2, 3], show_progress=True
)  # Auto-selects large dataset workflow

# Or explicitly
popt, pcov = fit(
    model,
    large_x,
    large_y,
    p0=[1, 2, 3],
    preset="large",
    memory_limit_gb=8.0,
    show_progress=True,
)
```

### Example 4: Streaming Optimization

```python
# Before
from nlsq import AdaptiveHybridStreamingOptimizer
from nlsq.hybrid_streaming_config import HybridStreamingConfig

config = HybridStreamingConfig(
    enable_multistart=True,
    n_starts=10,
)
optimizer = AdaptiveHybridStreamingOptimizer(config=config)
result = optimizer.fit(data_source=(x, y), func=model, p0=p0)

# After
popt, pcov = fit(model, x, y, p0=p0, preset="streaming")
```

### Example 5: Custom Configuration

For advanced users who need full control, custom config objects still work:

```python
from nlsq import fit
from nlsq.large_dataset import LDMemoryConfig

# Create custom config
config = LDMemoryConfig(
    memory_limit_gb=4.0,
    min_chunk_size=100,
    max_chunk_size=10000,
)

# Use with fit()
popt, pcov = fit(model, x, y, p0=[1, 2, 3], workflow=config)
```

## Advanced: Using the Workflow API Directly

For programmatic workflow selection, use the `WorkflowSelector` class:

```python
from nlsq import WorkflowSelector, OptimizationGoal

# Create selector with optional memory limit
selector = WorkflowSelector(memory_limit_gb=32.0)

# Get config for dataset characteristics
config = selector.select(n_points=5_000_000, n_params=5, goal=OptimizationGoal.QUALITY)

# Use the config
fit(model, x, y, p0=p0, workflow=config)
```

Or use the convenience function:

```python
from nlsq import auto_select_workflow, OptimizationGoal

config = auto_select_workflow(
    n_points=len(xdata), n_params=3, goal=OptimizationGoal.ROBUST, memory_limit_gb=16.0
)
```

## YAML Configuration

For reproducible workflows, define configurations in a YAML file:

```yaml
# nlsq.yaml
default_workflow: standard
memory_limit_gb: 32.0

workflows:
  my_production_workflow:
    tier: CHUNKED
    goal: QUALITY
    enable_multistart: true
    n_starts: 15
    gtol: 1e-9
    ftol: 1e-9
    xtol: 1e-9
```

Load and use:

```python
from nlsq.workflow import get_custom_workflow

config = get_custom_workflow("my_production_workflow")
fit(model, x, y, p0=p0, workflow=config)
```

## Environment Variable Overrides

Configure defaults via environment variables:

```bash
export NLSQ_WORKFLOW_GOAL=quality
export NLSQ_MEMORY_LIMIT_GB=16.0
export NLSQ_DEFAULT_WORKFLOW=robust
```

## Backward Compatibility

- `curve_fit()` remains fully supported and unchanged
- Both `curve_fit()` and `fit()` can be imported and used side-by-side
- `curve_fit_large()` remains available for explicit large dataset handling
- All existing code using `curve_fit()` continues to work without modification

## When to Use Which

| Scenario                                     | Recommended API         |
|----------------------------------------------|-------------------------|
| Quick prototyping                            | `fit()`                 |
| Production with reproducible settings        | `fit()` with YAML config|
| Full control over algorithm parameters       | `curve_fit()`           |
| Very large datasets (>1M points)             | `fit(preset="large")`   |
| Explicit streaming requirements              | `fit(preset="streaming")`|
| Custom memory/chunking configuration         | `curve_fit_large()`     |

## API Reference

### fit() Parameters

```python
fit(
    f,  # Model function
    xdata,  # Independent variable data
    ydata,  # Dependent variable data
    p0=None,  # Initial parameters
    sigma=None,  # Measurement uncertainties
    absolute_sigma=False,  # Interpret sigma as absolute
    check_finite=True,  # Check for NaN/Inf
    bounds=(-inf, inf),  # Parameter bounds
    method=None,  # Optimization method
    preset=None,  # Workflow preset name
    workflow="auto",  # Workflow config or "auto"
    goal=None,  # Optimization goal
    multistart=None,  # Override preset multi-start
    n_starts=None,  # Override preset n_starts
    sampler="lhs",  # Multi-start sampling strategy
    center_on_p0=True,  # Center samples on initial guess
    scale_factor=1.0,  # Exploration scale factor
    memory_limit_gb=None,  # Memory limit for large datasets
    size_threshold=1_000_000,  # Threshold for large datasets
    show_progress=False,  # Display progress bar
    chunk_size=None,  # Override chunk size
    **kwargs,  # Additional optimizer parameters
)
```

### Workflow Types

```python
from nlsq import WorkflowTier, OptimizationGoal

# Tiers
WorkflowTier.STANDARD  # Standard in-memory processing
WorkflowTier.CHUNKED  # Chunked processing for large datasets
WorkflowTier.STREAMING  # Streaming optimization
WorkflowTier.STREAMING_CHECKPOINT  # Streaming with checkpoints

# Goals
OptimizationGoal.FAST  # Speed-optimized
OptimizationGoal.ROBUST  # Balanced (default)
OptimizationGoal.QUALITY  # Quality-optimized
OptimizationGoal.MEMORY_EFFICIENT  # Memory-optimized
```

## Troubleshooting

### Issue: "Unknown workflow"

```python
# Error: ValueError: Unknown workflow 'my_workflow'. Valid presets: [...]

# Fix: Use a valid preset name or config object
fit(model, x, y, workflow="standard")  # Use valid preset
```

### Issue: "Unknown goal"

```python
# Error: ValueError: Unknown goal 'best'. Valid goals: [...]

# Fix: Use a valid goal name or enum
from nlsq import OptimizationGoal

fit(model, x, y, goal=OptimizationGoal.QUALITY)
```

### Issue: Slow performance on large datasets

```python
# Try using explicit large dataset settings
fit(
    model,
    x,
    y,
    preset="large",
    memory_limit_gb=16.0,  # Increase if available
    show_progress=True,
)
```
