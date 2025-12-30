# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLSQ is a GPU/TPU-accelerated nonlinear least squares curve fitting library built on JAX. It provides a drop-in replacement for `scipy.optimize.curve_fit` with automatic differentiation for Jacobian computation and support for datasets up to 100M+ points.

## Development Commands

```bash
# Install for development
make dev                    # Install dev dependencies + pre-commit hooks
pip install -e ".[dev,test,docs]"

# Testing
make test                   # Run all tests (parallel by default, ~2904 tests)
make test-fast              # Skip slow tests (-m "not slow")
pytest tests/core/test_minpack.py::TestCurveFit::test_basic_fit  # Single test
pytest -k "stability"       # Tests matching pattern

# Code quality
make lint                   # Run ruff check
make format                 # Format with ruff
make type-check             # Run mypy

# GPU setup (Linux only)
make install-jax-gpu        # Install JAX with CUDA support
make gpu-check              # Verify GPU detection
make env-info               # Show platform/GPU info
```

## Architecture

### Package Structure (v0.4.2)

The `nlsq` package is organized into logical subpackages:

```
nlsq/
├── core/           # Core optimization algorithms
│   ├── minpack.py         # SciPy-compatible curve_fit() API
│   ├── least_squares.py   # LeastSquares orchestrator
│   ├── trf.py             # Trust Region Reflective optimizer (2544 lines)
│   ├── trf_jit.py         # JIT-compiled TRF functions (474 lines)
│   ├── profiler.py        # TRFProfiler/NullProfiler (181 lines)
│   ├── functions.py       # Built-in model functions
│   ├── sparse_jacobian.py # Sparse Jacobian computation
│   └── workflow.py        # Workflow configuration
├── interfaces/     # Protocol definitions for dependency injection
│   ├── cache_protocol.py     # CacheProtocol, BoundedCacheProtocol
│   ├── optimizer_protocol.py # OptimizerProtocol
│   ├── data_source_protocol.py
│   ├── jacobian_protocol.py
│   └── result_protocol.py
├── streaming/      # Large dataset handling
│   ├── optimizer.py       # Streaming optimization
│   ├── large_dataset.py   # Memory-aware chunking
│   ├── adaptive_hybrid.py # Hybrid streaming strategies (4514 lines)
│   ├── telemetry.py       # DefenseLayerTelemetry (336 lines)
│   ├── validators.py      # Config validation functions (569 lines)
│   └── hybrid_config.py   # HybridStreamingConfig
├── caching/        # Performance optimization
│   ├── memory_manager.py  # Memory pooling with TTL
│   ├── smart_cache.py     # JIT cache (xxhash)
│   └── compilation_cache.py
├── stability/      # Numerical stability
│   ├── guard.py           # Condition monitoring
│   ├── svd_fallback.py    # GPU/CPU fallback SVD
│   └── condition_monitor.py
├── precision/      # Precision controls
│   ├── mixed_precision.py
│   └── parameter_normalizer.py
├── utils/          # Utilities
│   ├── validators.py      # Input validation
│   ├── diagnostics.py     # Convergence monitoring
│   └── logging.py
└── (root modules)  # Core infrastructure
    ├── callbacks.py, config.py, result.py
    ├── common_jax.py, common_scipy.py
    └── types.py, constants.py, device.py
```

### Core Optimization Pipeline

```
curve_fit() → CurveFit → LeastSquares → TrustRegionReflective
     │              │           │              │
     ▼              ▼           ▼              ▼
  core/minpack  core/minpack  core/least_squares  core/trf
```

- **core/minpack.py**: SciPy-compatible `curve_fit()` API wrapper
- **core/least_squares.py**: `LeastSquares` class orchestrating optimization
- **core/trf.py**: Trust Region Reflective algorithm (main optimizer), uses SVD for solving trust-region subproblems

### Key Subsystems

- **core/trf_jit.py**: JIT-compiled TRF helper functions (gradient, SVD, CG solver)
- **core/profiler.py**: TRFProfiler for performance timing, NullProfiler for zero-overhead
- **interfaces/**: Protocol definitions enabling dependency injection and loose coupling
- **stability/guard.py**: Numerical stability monitoring (condition numbers, NaN/Inf detection, data rescaling)
- **utils/validators.py**: Input validation with security constraints (array size limits, bounds checking)
- **caching/memory_manager.py**: Memory pooling with TTL-cached psutil calls
- **caching/smart_cache.py, compilation_cache.py**: JIT compilation caching (xxhash for speed)
- **stability/svd_fallback.py**: GPU/CPU fallback SVD with randomized SVD for large matrices
- **utils/diagnostics.py**: Convergence monitoring with verbosity levels
- **streaming/optimizer.py**: Streaming optimization for unlimited-size datasets
- **streaming/large_dataset.py**: Automatic chunking for datasets exceeding memory
- **streaming/telemetry.py**: DefenseLayerTelemetry for 4-layer defense strategy monitoring
- **streaming/validators.py**: Extracted config validators (reduce HybridStreamingConfig complexity)

### JAX Patterns

All fit functions must be JIT-compilable. Use `jax.numpy` instead of `numpy` in model functions:

```python
import jax.numpy as jnp


def model(x, a, b):
    return a * jnp.exp(-b * x)  # Use jnp, not np
```

Closures that capture different data each call use `@jit` directly (not `cached_jit`) since source-based caching won't help.

## Key Configuration

- **Python**: ≥3.12 required
- **JAX**: Locked to 0.8.0
- **Package manager**: uv preferred (see Makefile for auto-detection)
- **Persistent JAX cache**: `~/.cache/nlsq/jax_cache` (eliminates cold-start overhead)

## Testing Patterns

Tests use pytest with parallel execution (`-n 4`). Key markers:
- `@pytest.mark.slow`: Tests >1s (skip with `-m "not slow"`)
- `@pytest.mark.serial`: Tests that must run on a single xdist worker (prevents resource contention)
- `@pytest.mark.gpu`: Requires GPU
- `@pytest.mark.stability`: Numerical stability tests

### Preventing Test Suite Hangs (Critical)

Tests that spawn subprocesses or consume large memory MUST use `@pytest.mark.serial`:

1. **Subprocess-spawning tests**: Each subprocess initializes JAX (~620ms + 500MB memory). With `-n 4` workers, parallel JAX initializations cause compilation cache deadlocks and system freezes.
   - `tests/regression/test_examples_scripts.py` - 65 example scripts
   - `tests/regression/test_notebooks.py` - 60 notebooks
   - `tests/cli/test_integration.py::TestCLISubprocessInvocation`

2. **Memory-intensive tests**: Tests with 1M+ data points can cause OOM when run in parallel.
   - `tests/streaming/test_streaming_stress.py` - 100K+ points
   - `tests/streaming/test_algorithm_efficiency.py` - memory sweep tests
   - `tests/stability/test_stability_extended.py::test_memory_constraints`

When adding new tests, apply `@pytest.mark.serial` if the test:
- Spawns subprocesses that import JAX/NLSQ
- Creates arrays larger than 100K elements
- Uses multiprocessing.Pool or ThreadPool with JAX operations

## Environment Variables

- `NLSQ_SKIP_GPU_CHECK=1`: Suppress GPU availability warnings
- `NLSQ_DISABLE_PERSISTENT_CACHE=1`: Disable JAX compilation cache
- `NLSQ_DEBUG=1`: Enable debug logging
- `NLSQ_FORCE_CPU=1`: Force CPU backend for testing

## Performance Optimizations (v1.0)

### Lazy Imports
The `nlsq/__init__.py` uses lazy loading for specialty modules. Only core modules (curve_fit, OptimizeResult, etc.) are loaded immediately; others load on first access.

- Import time reduced from ~1084ms to ~620ms (43% reduction)
- JAX initialization (~290ms) is unavoidable
- Specialty modules: streaming, global optimization, sparse jacobian, etc.

### Vectorized Sparse Jacobian
The sparse Jacobian construction in `nlsq/core/sparse_jacobian.py` uses O(nnz) vectorized NumPy operations instead of O(nm) Python loops:

```python
# Fast path: vectorized COO construction
mask = np.abs(J_chunk) > threshold
rows, cols = np.where(mask)
values = J_chunk[rows, cols]
J_sparse = coo_matrix((values, (rows, cols)), shape=shape)
```

- 37-50x speedup for 10k×50 matrices
- Handles 100k×50 matrices in <150ms

### Condition Number Estimation
The stability guard in `nlsq/stability/guard.py` uses `svdvals()` (singular values only) instead of full SVD for condition estimation, avoiding unnecessary U/V computation.

## Active Technologies
- Python ≥3.12 (per pyproject.toml) + JAX 0.8.0, NumPy, SciPy (for reference implementations)
- Python ≥3.12 + pytest, pytest-xdist (parallel execution) (004-reorganize-tests-scripts)
- N/A (file reorganization only) (004-reorganize-tests-scripts)

## Recent Changes
- 004-legacy-modernization: Comprehensive codebase modernization (v0.4.2):
  - Extracted interfaces/ package with Protocol classes for dependency injection
  - Split trf.py: extracted trf_jit.py (474 lines) and profiler.py (181 lines)
  - Split streaming: extracted telemetry.py (336 lines) and validators.py (569 lines)
  - Updated type hints to Python 3.12+ syntax (Union→|, Optional→X|None)
  - Added __slots__ to dataclasses for memory efficiency
- 003-reorganize-imports: Reorganized package into subpackages (core/, streaming/, caching/, stability/, precision/, utils/) while maintaining backwards compatibility
- 002-performance-optimizations: Added Python ≥3.12 (per pyproject.toml) + JAX 0.8.0, NumPy, SciPy (for reference implementations)
