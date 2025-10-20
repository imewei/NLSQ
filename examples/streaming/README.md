# Streaming Optimizer Examples

This directory contains comprehensive examples demonstrating the fault tolerance features of the NLSQ streaming optimizer. These examples show how to handle errors, recover from interruptions, and analyze optimization diagnostics for datasets that don't fit in memory.

## Overview

The streaming optimizer provides production-ready fault tolerance with:
- **Best parameter tracking**: Never returns initial p0
- **Checkpoint save/resume**: Automatic interruption recovery
- **NaN/Inf detection**: Three validation points
- **Adaptive retry strategies**: Error-specific recovery (max 2 attempts)
- **Success rate validation**: Configurable threshold (default 50%)
- **Detailed diagnostics**: Comprehensive failure tracking
- **Fast mode**: <1% overhead for trusted data

## Examples

### 1. Basic Fault Tolerance (`01_basic_fault_tolerance.py`)

Demonstrates the default fault tolerance behavior with automatic error handling.

**Key features:**
- Automatic best parameter tracking
- NaN/Inf detection at three validation points
- Adaptive retry strategies for failed batches
- Success rate validation
- Comprehensive diagnostics

**Run:**
```bash
python examples/streaming/01_basic_fault_tolerance.py
```

**Output:** Shows how the optimizer automatically handles errors and returns the best parameters found.

---

### 2. Checkpoint Save and Resume (`02_checkpoint_resume.py`)

Demonstrates checkpoint functionality for recovering from interruptions during long-running optimizations.

**Key features:**
- Automatic checkpoint saving at intervals
- Auto-detection of latest checkpoint
- Resume from specific checkpoint path
- Full optimizer state preservation (params, momentum, etc.)

**Run:**
```bash
python examples/streaming/02_checkpoint_resume.py
```

**Output:** Simulates an interruption and demonstrates seamless resume from checkpoint.

---

### 3. Custom Retry Settings (`03_custom_retry_settings.py`)

Demonstrates how to configure retry strategies and success rate thresholds for noisy data.

**Key features:**
- Configurable success rate thresholds
- Adaptive retry strategies
- Error type analysis
- Retry count tracking

**Run:**
```bash
python examples/streaming/03_custom_retry_settings.py
```

**Output:** Shows how to tune settings for datasets with high failure rates.

---

### 4. Interpreting Diagnostics (`04_interpreting_diagnostics.py`)

Demonstrates how to interpret and analyze the comprehensive diagnostic information.

**Key features:**
- Streaming diagnostics structure
- Aggregate statistics interpretation
- Recent batch statistics analysis
- Checkpoint information access
- JSON export for further analysis

**Run:**
```bash
python examples/streaming/04_interpreting_diagnostics.py
```

**Output:** Detailed breakdown of all diagnostic information with interpretation guidance.

---

## Quick Reference

### Basic Usage

```python
from nlsq import StreamingOptimizer, StreamingConfig
import numpy as np

# Configure with fault tolerance (default)
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    enable_fault_tolerance=True,  # Default
    validate_numerics=True,  # Check for NaN/Inf
    min_success_rate=0.5,  # Require 50% success
    max_retries_per_batch=2,  # Max 2 retry attempts
)

# Create optimizer
optimizer = StreamingOptimizer(config)


# Define model
def model(x, a, b):
    return a * np.exp(-b * x)


# Fit with automatic error handling
result = optimizer.fit(
    (x_data, y_data),  # Data as tuple
    model,  # Model function
    p0=[1.0, 0.1],  # Initial parameters
    verbose=1,
)

# Access results
print(f"Best params: {result['x']}")
print(f"Success: {result['success']}")
print(f"Success rate: {result['streaming_diagnostics']['batch_success_rate']:.1%}")
```

### Checkpoint Resume

```python
# Enable checkpoint resume (auto-detect latest)
config = StreamingConfig(
    checkpoint_dir="checkpoints",
    checkpoint_frequency=100,  # Save every 100 iterations
    resume_from_checkpoint=True,  # Auto-detect latest
)

# Or load from specific checkpoint
config = StreamingConfig(resume_from_checkpoint="checkpoints/checkpoint_iter_500.h5")

optimizer = StreamingOptimizer(config)
result = optimizer.fit((x_data, y_data), model, p0=[1.0, 0.1])
```

### Custom Retry Settings

```python
# More permissive for noisy data
config = StreamingConfig(
    min_success_rate=0.3,  # Allow 70% failures
    max_retries_per_batch=2,  # Standard retry limit
    validate_numerics=True,  # Keep validation
)

# Stricter for clean data
config = StreamingConfig(
    min_success_rate=0.8,  # Require 80% success
    max_retries_per_batch=1,  # Fewer retries
    validate_numerics=True,
)

optimizer = StreamingOptimizer(config)
result = optimizer.fit((x_data, y_data), model, p0=[1.0, 0.1])
```

### Fast Mode (Production)

```python
# Disable validation overhead for trusted data
config = StreamingConfig(
    enable_fault_tolerance=False,  # <1% overhead
    enable_checkpoints=True,  # Still save checkpoints
)

optimizer = StreamingOptimizer(config)
result = optimizer.fit((x_data, y_data), model, p0=[1.0, 0.1])
```

### Analyzing Diagnostics

```python
result = optimizer.fit((x_data, y_data), model, p0=[1.0, 0.1])
diag = result["streaming_diagnostics"]

# Overall success
print(f"Success rate: {diag['batch_success_rate']:.1%}")
print(f"Total batches: {diag['total_batches_attempted']}")

# Failure analysis
print(f"Failed batches: {diag['failed_batches']}")
print(f"Error types: {diag['error_types']}")
print(f"Retry counts: {diag['retry_counts']}")

# Recent performance
recent_stats = diag["recent_batch_stats"][-10:]  # Last 10 batches
for stats in recent_stats:
    print(
        f"Batch {stats['batch_idx']}: loss={stats['loss']:.4e}, retries={stats['retry_count']}"
    )

# Aggregate statistics
agg = diag["aggregate_stats"]
print(f"Mean loss: {agg['mean_loss']:.4e}")
print(f"Mean gradient norm: {agg['mean_grad_norm']:.4f}")
```

## Configuration Reference

### Fault Tolerance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_fault_tolerance` | bool | `True` | Master switch for fault tolerance features |
| `validate_numerics` | bool | `True` | Check for NaN/Inf at three validation points |
| `min_success_rate` | float | `0.5` | Minimum batch success rate required (0.0-1.0) |
| `max_retries_per_batch` | int | `2` | Maximum retry attempts per batch |
| `batch_stats_buffer_size` | int | `100` | Size of circular buffer for batch statistics |

### Checkpoint Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | str | `"checkpoints"` | Directory to save checkpoints |
| `checkpoint_frequency` | int | `100` | Save checkpoint every N iterations |
| `enable_checkpoints` | bool | `True` | Whether to enable checkpointing |
| `resume_from_checkpoint` | bool/str/None | `None` | Resume from checkpoint (True=auto, str=path, None=no) |

### Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `32` | Size of batches to process |
| `max_epochs` | int | `10` | Maximum number of epochs |
| `learning_rate` | float | `0.001` | Base learning rate |
| `use_adam` | bool | `True` | Whether to use Adam optimizer |
| `gradient_clip` | float | `1.0` | Maximum gradient norm for clipping |
| `convergence_tol` | float | `1e-6` | Convergence tolerance for loss changes |

## Performance Guide

### Performance Overhead

- **Full fault tolerance**: <5% overhead
- **Fast mode**: <1% overhead
- **Checkpoint saves**: Negligible (async I/O)

### When to Use Each Mode

| Mode | Use Case | Configuration |
|------|----------|---------------|
| **Full fault tolerance** | Long-running optimizations, noisy data, critical results | `enable_fault_tolerance=True` (default) |
| **Fast mode** | Production deployments, trusted data, performance-critical | `enable_fault_tolerance=False` |
| **Permissive** | Very noisy data, exploratory analysis | `min_success_rate=0.3` |
| **Strict** | Clean data, high reliability requirements | `min_success_rate=0.8` |

## Diagnostic Structure

```python
streaming_diagnostics = {
    "failed_batches": [3, 17, 42],  # Indices of failed batches
    "retry_counts": {3: 2, 17: 1},  # Retry attempts per batch
    "error_types": {  # Error categorization
        "NumericalError": 15,
        "SingularMatrix": 2,
        "ValueError": 8,
    },
    "batch_success_rate": 0.92,  # Overall success rate
    "total_batches_attempted": 100,  # Total batches processed
    "total_retries": 25,  # Total retry attempts
    "convergence_achieved": True,  # Whether converged
    "final_epoch": 8,  # Final epoch number
    "elapsed_time": 45.3,  # Total time (seconds)
    "checkpoint_info": {  # Last checkpoint (if any)
        "path": "checkpoints/checkpoint_iter_500.h5",
        "saved_at": "2025-10-20T15:30:00",
        "batch_idx": 500,
    },
    "recent_batch_stats": [...],  # Circular buffer (last 100)
    "aggregate_stats": {  # Aggregate metrics
        "mean_loss": 0.0234,
        "std_loss": 0.0012,
        "mean_grad_norm": 0.456,
        "std_grad_norm": 0.023,
        "mean_batch_time": 0.015,
        "std_batch_time": 0.002,
    },
}
```

## Troubleshooting

### High Failure Rate

**Symptoms:** `batch_success_rate` < 50%, many failed batches

**Solutions:**
1. Check `error_types` to identify common errors
2. Lower `min_success_rate` threshold if data is inherently noisy
3. Inspect failed batches using `failed_batches` indices
4. Examine data quality at failed batch indices

### Slow Performance

**Symptoms:** High `elapsed_time`, slow `mean_batch_time`

**Solutions:**
1. Enable fast mode if data is trusted: `enable_fault_tolerance=False`
2. Increase `batch_size` to reduce overhead
3. Disable numeric validation: `validate_numerics=False` (use with caution)
4. Check if model function is JIT-compiled properly

### Checkpoint Issues

**Symptoms:** Cannot resume from checkpoint, checkpoint not found

**Solutions:**
1. Verify `checkpoint_dir` exists and is writable
2. Check `checkpoint_frequency` is not too large
3. Ensure `enable_checkpoints=True`
4. For auto-detection, ensure checkpoint files exist in `checkpoint_dir`

### NaN/Inf Errors

**Symptoms:** Many `NumericalError` in `error_types`

**Solutions:**
1. Check initial parameters `p0` are reasonable
2. Reduce `learning_rate` for more stable updates
3. Add bounds to parameters if applicable
4. Inspect model function for numerical issues
5. Enable gradient clipping: adjust `gradient_clip` parameter

## Further Reading

- [StreamingOptimizer API Documentation](https://nlsq.readthedocs.io/en/latest/api/nlsq.StreamingOptimizer.html)
- [StreamingConfig Reference](https://nlsq.readthedocs.io/en/latest/api/nlsq.StreamingConfig.html)
- [Fault Tolerance Specification](../../agent-os/specs/2025-10-19-streaming-optimizer-fault-tolerance/spec.md)
- [Migration Guide](../../docs/migration/streaming_fault_tolerance.md)

## Contributing

If you have suggestions for additional examples or find issues with existing examples, please open an issue or pull request on GitHub:
https://github.com/imewei/NLSQ

## License

These examples are part of the NLSQ package and are licensed under the MIT License.
