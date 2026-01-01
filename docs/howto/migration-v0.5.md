# Migration Guide: v0.4.x to v0.5.0

This guide helps you migrate your NLSQ code from v0.4.x to v0.5.0. The v0.5.0 release includes
architectural improvements for better maintainability, security hardening for CLI model loading,
and improved test reliability.

## Quick Migration Checklist

- [ ] Update deprecated imports (see [Import Path Changes](#import-path-changes))
- [ ] Review custom model files for security compliance
- [ ] Update test code that uses `time.sleep()` for condition waiting

## Import Path Changes

### OptimizeResult and OptimizeWarning

**Before (v0.4.x):**
```python
from nlsq.core._optimize import OptimizeResult, OptimizeWarning
```

**After (v0.5.0):**
```python
# Recommended: Import from nlsq.result
from nlsq.result import OptimizeResult, OptimizeWarning

# Or: Import from package root (also works)
from nlsq import OptimizeResult, OptimizeWarning
```

**Deprecation Timeline:**
- v0.5.0 (2026-01-01): Old imports emit `DeprecationWarning`
- v0.6.0 (2027-01-01): Old imports will be removed

**Suppressing Warnings During Migration:**
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="nlsq")
```

## New Features

### Factory Functions for Optimizer Composition

v0.5.0 introduces factory functions for runtime configuration of optimizers:

```python
from nlsq.core.factories import create_optimizer, configure_curve_fit

# Create a streaming optimizer
optimizer = create_optimizer(streaming=True, chunk_size=10000)
result = optimizer.fit(model, xdata, ydata)

# Create a pre-configured curve_fit with diagnostics
curve_fit = configure_curve_fit(
    enable_diagnostics=True,
    enable_recovery=True,
)
popt, pcov = curve_fit(model, xdata, ydata)
```

### Protocol Adapters for Dependency Injection

For advanced users implementing custom optimizers:

```python
from nlsq.core.adapters import CurveFitAdapter
from nlsq.interfaces import CurveFitProtocol

# Create adapter with custom configuration
adapter = CurveFitAdapter(
    cache=my_cache,
    stability_guard=my_guard,
)

# Use like curve_fit
popt, pcov = adapter.curve_fit(model, xdata, ydata)
```

## Security Hardening

### Custom Model Validation

v0.5.0 validates custom model files loaded via CLI before execution. Models containing
dangerous patterns are blocked:

**Blocked Operations:**
- Code execution: `exec()`, `eval()`, `compile()`, `__import__()`
- System access: `os.system()`, `subprocess.*`, `popen()`
- Network access: `socket`, `urllib`, `http`
- File modification: `open()` with write modes
- Memory manipulation: `ctypes`, `cffi`

**If your model is blocked:**

1. Review your model file for dangerous patterns
2. Remove or refactor the blocked code
3. Use the `--trust` flag to bypass validation (use with caution):

```bash
nlsq fit --model my_model.py --trust
```

**Audit Logging:**
All model loading attempts are logged to `~/.nlsq/audit.log` with:
- Timestamp
- User
- File path
- Validation result
- Any security violations

## Test Migration

### Replacing time.sleep() with wait_for()

v0.5.0 provides a `wait_for()` utility for reliable condition waiting:

**Before (v0.4.x):**
```python
import time

def test_async_operation():
    start_async_task()
    time.sleep(2.0)  # Flaky: might not be enough time
    assert task_completed()
```

**After (v0.5.0):**
```python
from tests.conftest import wait_for

def test_async_operation():
    start_async_task()
    wait_for(
        task_completed,
        timeout=5.0,
        message="Task did not complete",
    )
    assert task_completed()
```

**wait_for() Features:**
- Exponential backoff polling (starts at 10ms, max 1s)
- Configurable timeout
- Clear error messages on timeout
- Returns immediately when condition is met

## Architecture Changes

### Core Module Dependencies

`nlsq/core/minpack.py` now uses lazy imports, reducing its dependency count from 27 to <15.
This improves:
- Import time (now <700ms)
- Code maintainability
- Circular dependency prevention

**Impact:** None for users. This is an internal improvement.

### Protocol Interfaces

The `nlsq/interfaces/` package now exports all protocol types:

```python
from nlsq.interfaces import (
    # Optimizer protocols
    OptimizerProtocol,
    LeastSquaresOptimizerProtocol,
    CurveFitProtocol,

    # Data protocols
    DataSourceProtocol,
    StreamingDataSourceProtocol,

    # Computation protocols
    JacobianProtocol,
    SparseJacobianProtocol,

    # Cache protocols
    CacheProtocol,
    BoundedCacheProtocol,

    # Result protocol
    ResultProtocol,

    # Concrete implementations
    ArrayDataSource,
    AutodiffJacobian,
    DictCache,
)
```

## Breaking Changes

**None.** v0.5.0 is fully backward compatible with v0.4.x.

All changes are additive or use deprecation warnings for removed functionality.

## Getting Help

If you encounter issues during migration:

1. Check the [CHANGELOG](https://github.com/imewei/NLSQ/blob/main/CHANGELOG.md) for details
2. Search existing [GitHub Issues](https://github.com/imewei/NLSQ/issues)
3. Open a new issue with the "migration" label
