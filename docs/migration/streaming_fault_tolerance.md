# Migration Guide: Streaming Optimizer Fault Tolerance

**Version:** v0.2.0+
**Date:** 2025-10-20
**Status:** Production Ready

## Overview

This guide helps you migrate existing code to use the new fault tolerance features in the streaming optimizer. All new features are **backward compatible** with sensible defaults, so your existing code will continue to work without modifications.

## Summary of Changes

### New Features (v0.2.0+)

1. **Best Parameter Tracking**: Always returns best parameters found (never initial p0)
2. **Checkpoint Save/Resume**: Automatic detection and full state preservation
3. **NaN/Inf Detection**: Three validation points (gradients, parameters, loss)
4. **Adaptive Retry Strategies**: Error-specific recovery (max 2 attempts per batch)
5. **Success Rate Validation**: Configurable threshold (default 50%)
6. **Detailed Diagnostics**: Comprehensive failure tracking and batch statistics
7. **Batch Statistics**: Circular buffer tracking last 100 batches
8. **Fast Mode**: Disable validation overhead for trusted data (<1% overhead)

### Breaking Changes

**None**. All changes are backward compatible. Existing code continues to work with enhanced reliability.

---

## Backward Compatibility

### Existing Code Works Unchanged

```python
# Your existing code continues to work
from nlsq import StreamingOptimizer, StreamingConfig

config = StreamingConfig(batch_size=100, max_epochs=10)
optimizer = StreamingOptimizer(config)
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])
```

**What's new:**
- Fault tolerance enabled by default
- Best parameters always returned
- Detailed diagnostics available in `result['streaming_diagnostics']`
- NaN/Inf detection at three validation points
- Adaptive retry strategies for failed batches

No code changes required!

---

## Migration Scenarios

### Scenario 1: No Changes Needed

**Use case:** You want the enhanced reliability with zero effort.

**Action:** None required. Your existing code automatically benefits from:
- Best parameter tracking
- NaN/Inf detection
- Adaptive retry strategies
- Success rate validation (50% threshold)

```python
# Existing code - no changes needed
optimizer = StreamingOptimizer(StreamingConfig(batch_size=100))
result = optimizer.fit((x_data, y_data), model, p0=[1.0, 0.1])

# New: Access diagnostics if desired
if not result["success"]:
    diag = result["streaming_diagnostics"]
    print(f"Success rate: {diag['batch_success_rate']:.1%}")
    print(f"Failed batches: {len(diag['failed_batches'])}")
```

---

### Scenario 2: Enable Checkpoint Resume

**Use case:** Your optimization takes hours and you want interruption recovery.

**Changes:**

```python
# Before (v0.1.x)
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    checkpoint_dir="checkpoints",
    checkpoint_frequency=100,
    enable_checkpoints=True,
)

# After (v0.2.0+) - add resume capability
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    checkpoint_dir="checkpoints",
    checkpoint_frequency=100,
    enable_checkpoints=True,
    resume_from_checkpoint=True,  # NEW: Auto-detect latest checkpoint
)

optimizer = StreamingOptimizer(config)
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])

# Check if resumed
if result["streaming_diagnostics"]["checkpoint_info"]:
    print("Resumed from checkpoint")
```

**Benefits:**
- Automatically resumes from latest checkpoint
- No manual checkpoint management
- Full state preservation (params, momentum, iteration, etc.)

---

### Scenario 3: Configure for Noisy Data

**Use case:** Your dataset has high failure rates or noisy sensors.

**Changes:**

```python
# Before (v0.1.x) - might fail with default settings
config = StreamingConfig(batch_size=100, max_epochs=10)

# After (v0.2.0+) - tune for noisy data
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    # NEW: Allow more failures
    min_success_rate=0.3,  # Allow 70% failures (default: 0.5)
    max_retries_per_batch=2,  # Standard retry limit
    validate_numerics=True,  # Keep validation (default: True)
)

optimizer = StreamingOptimizer(config)
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])

# Analyze failures
diag = result["streaming_diagnostics"]
print(f"Error types: {diag['error_types']}")
print(f"Retry counts: {diag['retry_counts']}")
```

**Benefits:**
- Optimization succeeds with noisy data
- Detailed error analysis available
- Adaptive retry strategies handle transient errors

---

### Scenario 4: Production Deployment (Fast Mode)

**Use case:** You trust your data quality and need maximum performance.

**Changes:**

```python
# Before (v0.1.x) - validation overhead included
config = StreamingConfig(batch_size=100, max_epochs=10)

# After (v0.2.0+) - disable validation overhead
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    # NEW: Fast mode for production
    enable_fault_tolerance=False,  # <1% overhead (default: True)
    enable_checkpoints=True,  # Still save checkpoints
)

optimizer = StreamingOptimizer(config)
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])
```

**Benefits:**
- <1% performance overhead (vs <5% with full fault tolerance)
- Still saves checkpoints for recovery
- Basic error handling maintained

---

### Scenario 5: Analyzing Optimization Behavior

**Use case:** You want to understand why optimization fails or behaves unexpectedly.

**Changes:**

```python
# Before (v0.1.x) - limited diagnostics
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])
print(result["message"])

# After (v0.2.0+) - comprehensive diagnostics
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])
diag = result["streaming_diagnostics"]

# Overall success metrics
print(f"Success rate: {diag['batch_success_rate']:.1%}")
print(f"Total batches: {diag['total_batches_attempted']}")
print(f"Total retries: {diag['total_retries']}")
print(f"Elapsed time: {diag['elapsed_time']:.2f}s")

# Failure analysis
if diag["failed_batches"]:
    print(f"Failed batch indices: {diag['failed_batches']}")
    print(f"Error distribution: {diag['error_types']}")
    print(f"Retry patterns: {diag['retry_counts']}")

# Recent performance
recent_stats = diag["recent_batch_stats"][-10:]  # Last 10 batches
for stats in recent_stats:
    status = "SUCCESS" if stats["success"] else "FAILED"
    print(f"Batch {stats['batch_idx']}: {status}, loss={stats['loss']:.4e}")

# Aggregate statistics
agg = diag["aggregate_stats"]
print(f"Mean loss: {agg['mean_loss']:.6e}")
print(f"Mean gradient norm: {agg['mean_grad_norm']:.6f}")
```

**Benefits:**
- Comprehensive failure tracking
- Batch-level statistics (circular buffer, last 100 batches)
- Aggregate metrics for performance analysis
- Error type distribution
- Retry patterns

---

## API Changes

### New Configuration Parameters (StreamingConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_fault_tolerance` | bool | `True` | Master switch for fault tolerance features (NEW) |
| `validate_numerics` | bool | `True` | Check for NaN/Inf at three validation points (NEW) |
| `min_success_rate` | float | `0.5` | Minimum batch success rate required (NEW) |
| `max_retries_per_batch` | int | `2` | Maximum retry attempts per batch (NEW) |
| `batch_stats_buffer_size` | int | `100` | Size of circular buffer for batch statistics (NEW) |
| `resume_from_checkpoint` | bool/str/None | `None` | Resume from checkpoint (True=auto, str=path) (NEW) |

### New Result Fields

```python
result = {
    "x": np.ndarray,  # Best parameters (now guaranteed never p0)
    "success": bool,  # Whether optimization succeeded
    "message": str,  # Human-readable status message
    "fun": float,  # Function value at best parameters
    "best_loss": float,  # Best loss encountered (NEW)
    "final_epoch": int,  # Final epoch number (NEW)
    "streaming_diagnostics": {  # NEW: Comprehensive diagnostics
        "failed_batches": list,
        "retry_counts": dict,
        "error_types": dict,
        "batch_success_rate": float,
        "total_batches_attempted": int,
        "total_retries": int,
        "convergence_achieved": bool,
        "final_epoch": int,
        "elapsed_time": float,
        "checkpoint_info": dict or None,
        "recent_batch_stats": list,
        "aggregate_stats": dict,
    },
}
```

### Behavior Changes

#### 1. Best Parameters Always Returned

**Before (v0.1.x):**
```python
# Might return p0 if optimization diverged
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])
# result['x'] could be [1.0, 0.1] (initial p0)
```

**After (v0.2.0+):**
```python
# Always returns best parameters found (never p0)
result = optimizer.fit(data_source, model, p0=[1.0, 0.1])
# result['x'] is always the best parameters found during optimization
# Even if later batches fail, best params from earlier batches are returned
```

**Migration:** No code changes needed. This is an improvement in reliability.

#### 2. Success Criteria

**Before (v0.1.x):**
```python
# Success if no exceptions raised
result["success"]  # True/False based on completion
```

**After (v0.2.0+):**
```python
# Success if batch success rate >= min_success_rate (default 50%)
result["success"]  # True if >= 50% of batches succeeded
result["message"]  # Explains why optimization failed/succeeded

# Access detailed success metrics
diag = result["streaming_diagnostics"]
actual_rate = diag["batch_success_rate"]
threshold = config.min_success_rate
```

**Migration:** Check `result['success']` and `result['message']` as before. Optionally inspect `batch_success_rate` for more detail.

#### 3. Checkpoint Format

**Before (v0.1.x):**
```
checkpoint_v1.h5
├── parameters/current
├── parameters/best
├── optimizer_state/...
└── progress/...
```

**After (v0.2.0+):**
```
checkpoint_v2.h5  (backward compatible with v1)
├── version: "2.0"
├── parameters/current
├── parameters/best
├── optimizer_state/...
├── progress/...
└── diagnostics/      (NEW)
    ├── failed_batch_indices
    ├── retry_counts
    └── error_types
```

**Migration:** No changes needed. v2 checkpoints auto-detected. v1 checkpoints still loadable.

---

## Configuration Recommendations

### Development/Exploratory Analysis

```python
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    enable_fault_tolerance=True,  # Full diagnostics
    validate_numerics=True,  # Detect numerical issues
    min_success_rate=0.5,  # Standard threshold
    max_retries_per_batch=2,  # Standard retry limit
    enable_checkpoints=True,
    checkpoint_frequency=100,
    batch_stats_buffer_size=100,  # Track last 100 batches
)
```

### Production Deployments

```python
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    enable_fault_tolerance=False,  # Fast mode (<1% overhead)
    enable_checkpoints=True,  # Still save checkpoints
    checkpoint_frequency=1000,  # Less frequent for performance
)
```

### Long-Running Optimizations

```python
config = StreamingConfig(
    batch_size=100,
    max_epochs=50,
    enable_fault_tolerance=True,
    validate_numerics=True,
    min_success_rate=0.5,
    enable_checkpoints=True,
    checkpoint_frequency=100,
    resume_from_checkpoint=True,  # Auto-resume on restart
)
```

### Noisy/Unreliable Data

```python
config = StreamingConfig(
    batch_size=100,
    max_epochs=10,
    enable_fault_tolerance=True,
    validate_numerics=True,
    min_success_rate=0.3,  # Allow 70% failures
    max_retries_per_batch=2,  # Standard retry limit
    enable_checkpoints=True,
)
```

### High-Performance Requirements

```python
config = StreamingConfig(
    batch_size=200,  # Larger batches
    max_epochs=10,
    enable_fault_tolerance=False,  # Fast mode
    validate_numerics=False,  # Skip validation (use with caution)
    enable_checkpoints=False,  # Disable checkpoints for max speed
)
```

---

## Testing Your Migration

### Step 1: Test with Existing Code

Run your existing tests to verify backward compatibility:

```bash
# Your existing tests should pass without modifications
pytest tests/test_streaming.py
```

### Step 2: Test New Features

Add tests for new diagnostic features:

```python
def test_streaming_diagnostics():
    optimizer = StreamingOptimizer(StreamingConfig())
    result = optimizer.fit(data_source, model, p0=[1.0, 0.1])

    # Check new diagnostic fields exist
    diag = result["streaming_diagnostics"]
    assert "batch_success_rate" in diag
    assert "failed_batches" in diag
    assert "error_types" in diag
    assert "retry_counts" in diag
    assert "aggregate_stats" in diag
    assert "recent_batch_stats" in diag
```

### Step 3: Test Checkpoint Resume

```python
def test_checkpoint_resume():
    config = StreamingConfig(
        checkpoint_dir="test_checkpoints",
        checkpoint_frequency=5,
        enable_checkpoints=True,
    )

    # Initial training
    optimizer1 = StreamingOptimizer(config)
    result1 = optimizer1.fit(data_source, model, p0=[1.0, 0.1])

    # Resume training
    config_resume = StreamingConfig(
        checkpoint_dir="test_checkpoints",
        resume_from_checkpoint=True,
    )
    optimizer2 = StreamingOptimizer(config_resume)
    result2 = optimizer2.fit(data_source, model, p0=[1.0, 0.1])

    # Verify resume worked
    assert optimizer2.iteration > 0
    diag = result2["streaming_diagnostics"]
    assert diag["checkpoint_info"] is not None
```

---

## Performance Impact

### Fault Tolerance Enabled (Default)

- **Overhead**: <5%
- **Features**: All fault tolerance, validation, diagnostics
- **Recommended for**: Development, long-running jobs, critical results

### Fast Mode (Production)

- **Overhead**: <1%
- **Features**: Basic error handling, checkpoints (optional)
- **Recommended for**: Production deployments, trusted data, performance-critical

### Comparison

| Feature | Full Fault Tolerance | Fast Mode |
|---------|---------------------|-----------|
| Best parameter tracking | ✅ | ✅ |
| NaN/Inf detection | ✅ | ❌ |
| Adaptive retry strategies | ✅ | ❌ |
| Batch statistics | ✅ | ❌ |
| Checkpoints | ✅ | ✅ (optional) |
| Performance overhead | <5% | <1% |

---

## Troubleshooting

### Migration Issue 1: High Failure Rate

**Symptom:** Optimization fails with `batch_success_rate < min_success_rate`

**Solution:**
```python
# Lower success rate threshold for noisy data
config = StreamingConfig(
    min_success_rate=0.3,  # Allow 70% failures
    validate_numerics=True,  # Keep validation
)
```

### Migration Issue 2: Performance Regression

**Symptom:** Optimization slower than before

**Solution:**
```python
# Enable fast mode for production
config = StreamingConfig(
    enable_fault_tolerance=False,  # <1% overhead
    enable_checkpoints=True,  # Still save checkpoints
)
```

### Migration Issue 3: Checkpoint Incompatibility

**Symptom:** Cannot load old checkpoints

**Solution:**
```python
# Old v1 checkpoints should load automatically
# If issues persist, start fresh:
config = StreamingConfig(
    checkpoint_dir="checkpoints_new", resume_from_checkpoint=None  # Start fresh
)
```

### Migration Issue 4: Too Much Diagnostic Data

**Symptom:** Memory issues from large diagnostic buffers

**Solution:**
```python
# Reduce buffer size
config = StreamingConfig(
    batch_stats_buffer_size=50,  # Reduce from default 100
)

# Or disable diagnostics completely (fast mode)
config = StreamingConfig(enable_fault_tolerance=False)
```

---

## Examples

See the `examples/streaming/` directory for complete working examples:

1. **`01_basic_fault_tolerance.py`**: Basic usage with default settings
2. **`02_checkpoint_resume.py`**: Checkpoint save/resume workflow
3. **`03_custom_retry_settings.py`**: Configuring for noisy data
4. **`04_interpreting_diagnostics.py`**: Analyzing diagnostic information

---

## Support

For questions or issues:

1. **Documentation**: https://nlsq.readthedocs.io
2. **GitHub Issues**: https://github.com/imewei/NLSQ/issues
3. **Examples**: `examples/streaming/`
4. **Specification**: `agent-os/specs/2025-10-19-streaming-optimizer-fault-tolerance/spec.md`

---

## Summary

**What you need to do:**
- Nothing! Existing code works with enhanced reliability.

**What you can do:**
- Enable checkpoint resume for long-running jobs
- Tune `min_success_rate` for noisy data
- Enable fast mode for production
- Analyze detailed diagnostics for debugging

**What you should not do:**
- Assume `result['x']` is p0 (it's now always best parameters)
- Ignore `result['success']` status (may fail with low success rate)
- Use fast mode without understanding trade-offs

**Key takeaway:** All changes are backward compatible. Your code continues to work, but now with production-ready fault tolerance and comprehensive diagnostics.
