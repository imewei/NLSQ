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
│   ├── minpack.py         # SciPy-compatible curve_fit() API (<15 deps via lazy imports)
│   ├── least_squares.py   # LeastSquares orchestrator
│   ├── trf.py             # Trust Region Reflective optimizer (2544 lines)
│   ├── trf_jit.py         # JIT-compiled TRF functions (474 lines)
│   ├── profiler.py        # TRFProfiler/NullProfiler (181 lines)
│   ├── functions.py       # Built-in model functions
│   ├── sparse_jacobian.py # Sparse Jacobian computation
│   ├── workflow.py        # Workflow configuration
│   ├── factories.py       # NEW: create_optimizer(), configure_curve_fit() builders
│   └── adapters/          # NEW: Protocol adapters for DI
│       └── curve_fit_adapter.py  # CurveFitProtocol implementation
├── interfaces/     # Protocol definitions for dependency injection
│   ├── cache_protocol.py     # CacheProtocol, BoundedCacheProtocol
│   ├── optimizer_protocol.py # OptimizerProtocol, CurveFitProtocol
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
│   ├── safe_serialize.py  # Secure JSON serialization (replaces pickle)
│   └── logging.py
├── cli/            # Command-line interface
│   ├── model_registry.py  # Model loading with security validation
│   └── model_validation.py # SECURITY: AST-based model validation
├── result/         # Result types (consolidated)
│   ├── optimize_result.py # OptimizeResult (moved from core/_optimize.py)
│   └── optimize_warning.py # OptimizeWarning (moved from core/_optimize.py)
├── gui_qt/         # Native Qt desktop application (PySide6)
│   ├── __init__.py        # run_desktop() entry point
│   ├── main_window.py     # MainWindow with sidebar navigation
│   ├── app_state.py       # AppState (Qt signals wrapping SessionState)
│   ├── theme.py           # ThemeConfig, ThemeManager (light/dark)
│   ├── autosave.py        # AutosaveManager for crash recovery
│   ├── pages/             # Workflow pages (data, model, fitting, results, export)
│   ├── widgets/           # Reusable Qt widgets
│   └── plots/             # pyqtgraph-based scientific plots
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
- **core/factories.py**: Factory functions for composing optimizer configurations at runtime
- **core/adapters/**: Protocol adapters enabling dependency injection
- **interfaces/**: Protocol definitions enabling dependency injection and loose coupling
- **stability/guard.py**: Numerical stability monitoring (condition numbers, NaN/Inf detection, data rescaling)
- **utils/validators.py**: Input validation with security constraints (array size limits, bounds checking)
- **utils/safe_serialize.py**: Secure JSON-based serialization for checkpoints (replaces pickle, CWE-502 fix)
- **caching/memory_manager.py**: Memory pooling with TTL-cached psutil calls
- **caching/smart_cache.py, compilation_cache.py**: JIT compilation caching (xxhash for speed)
- **stability/svd_fallback.py**: GPU/CPU fallback SVD with randomized SVD for large matrices
- **utils/diagnostics.py**: Convergence monitoring with verbosity levels
- **streaming/optimizer.py**: Streaming optimization for unlimited-size datasets
- **streaming/large_dataset.py**: Automatic chunking for datasets exceeding memory
- **streaming/telemetry.py**: DefenseLayerTelemetry for 4-layer defense strategy monitoring
- **streaming/validators.py**: Extracted config validators (reduce HybridStreamingConfig complexity)
- **cli/model_validation.py**: Security validation for custom model files (AST inspection, path traversal prevention)

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

## Qt Desktop GUI

NLSQ includes a native Qt desktop application built with PySide6 and pyqtgraph for GPU-accelerated plotting.

### Launching the GUI

```bash
# Via entry point
nlsq-gui

# Or via module
python -m nlsq.gui_qt

# Or programmatically
from nlsq.gui_qt import run_desktop
run_desktop()
```

### GUI Features

- **5-page workflow**: Data Loading → Model Selection → Fitting Options → Results → Export
- **GPU-accelerated plots**: pyqtgraph with OpenGL for 500K+ point datasets
- **Theme support**: Light/dark themes via qdarktheme (Ctrl+T to toggle)
- **Keyboard shortcuts**: Ctrl+1-5 for pages, Ctrl+R run fit, Ctrl+O open file
- **Autosave**: Session recovery on crash (1-minute autosave interval)
- **Window persistence**: Size, position, theme saved between sessions

### GUI Architecture

```
MainWindow (QMainWindow)
├── Sidebar (QListWidget) - Page navigation with guards
├── PageStack (QStackedWidget)
│   ├── DataLoadingPage - File/clipboard import, column selection
│   ├── ModelSelectionPage - Built-in/polynomial/custom models
│   ├── FittingOptionsPage - Guided/Advanced modes, live cost plot
│   ├── ResultsPage - Parameters, statistics, fit/residual plots
│   └── ExportPage - ZIP/JSON/CSV/Python code export
└── StatusBar - Data info, fit status
```

### GUI Dependencies

Install with the `gui_qt` extra:
```bash
pip install "nlsq[gui_qt]"
# Or with uv
uv pip install "nlsq[gui_qt]"
```

Required packages: `PySide6>=6.5`, `pyqtgraph>=0.13`, `pyqtdarktheme>=2.1`

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
- Python ≥3.12 + JAX 0.8.0, NumPy, SciPy (006-legacy-modernization)
- N/A (library, no persistence) (006-legacy-modernization)
- Python 3.12+ + JAX 0.8.0, NumPy, SciPy (reference implementations) (007-performance-optimizations)
- Python ≥3.12 (per pyproject.toml) + JAX 0.8.0 (locked), NumPy, SciPy, pytest, ruff, mypy (008-tech-debt-remediation)
- Python >=3.12 (per pyproject.toml) + JAX 0.8.0, NumPy >=2.2, SciPy >=1.16.0, Optax >=0.2.6 (009-code-quality-refactor)
- Python 3.12+ (per pyproject.toml requires-python) + PySide6 (Qt bindings), pyqtgraph (GPU-accelerated plotting), existing nlsq core (JAX 0.8.0, NumPy 2.x, SciPy 1.16+) (010-streamlit-to-qt)
- N/A (file-based import/export, no database) (010-streamlit-to-qt)

## Recent Changes
- 010-streamlit-to-qt: Native Qt Desktop GUI (v0.5.0):
  - **Native Desktop App**: PySide6-based GUI replacing Streamlit web interface
  - **GPU-Accelerated Plots**: pyqtgraph with OpenGL for 500K+ point datasets
  - **5-Page Workflow**: Data Loading, Model Selection, Fitting Options, Results, Export
  - **Theme Support**: Light/dark themes via qdarktheme with Ctrl+T toggle
  - **Keyboard Shortcuts**: Ctrl+1-5 pages, Ctrl+R run fit, Ctrl+O open file
  - **Crash Recovery**: Autosave manager with 1-minute interval and session recovery
  - **Window Persistence**: Size, position, theme, current page saved via QSettings
  - **Export Formats**: ZIP session bundle, JSON, CSV, Python code generation
  - Spec: `/specs/010-streamlit-to-qt/`
- 006-legacy-modernization: Comprehensive legacy modernization (v0.4.3):
  - **Architecture**: Zero circular dependencies via lazy imports and TYPE_CHECKING
  - **God Module Reduction**: core/minpack.py now has <15 direct dependencies
  - **Security Hardening**: CLI model validation with AST-based pattern detection
    - Blocks dangerous operations: exec, eval, system calls, network access
    - Path traversal prevention for file loading
    - Resource limits (timeout/memory) for model execution
    - Audit logging with rotation (10MB) and retention (90 days)
  - **Factory Pattern**: New `nlsq/core/factories.py` with `create_optimizer()` and `configure_curve_fit()`
  - **Protocol Adapters**: New `nlsq/core/adapters/` with `CurveFitAdapter` implementing `CurveFitProtocol`
  - **Type Consolidation**: OptimizeResult/OptimizeWarning moved to `nlsq/result/`
  - **Test Reliability**: Replaced flaky time.sleep() with wait_for() polling utility
  - **Deprecation**: Old import paths work with deprecation warnings (12-month period)
  - Spec: `/specs/006-legacy-modernization/`
- 005-model-health-diagnostics: Model Health Diagnostics System (planned):
  - New `nlsq/diagnostics/` subpackage for post-fit model health analysis
  - Identifiability analysis (FIM condition number, rank, correlations)
  - Gradient health monitoring (vanishing, imbalance, stagnation detection)
  - Sloppy model analysis (eigenvalue spectrum, stiff/sloppy directions)
  - Plugin system for domain-specific diagnostic extensions
  - Spec: `/specs/005-model-health-diagnostics/`
- 004-legacy-modernization: Comprehensive codebase modernization (v0.4.2):
  - Extracted interfaces/ package with Protocol classes for dependency injection
  - Split trf.py: extracted trf_jit.py (474 lines) and profiler.py (181 lines)
  - Split streaming: extracted telemetry.py (336 lines) and validators.py (569 lines)
  - Updated type hints to Python 3.12+ syntax (Union→|, Optional→X|None)
  - Added __slots__ to dataclasses for memory efficiency
- 003-reorganize-imports: Reorganized package into subpackages (core/, streaming/, caching/, stability/, precision/, utils/) while maintaining backwards compatibility
- 002-performance-optimizations: Added Python ≥3.12 (per pyproject.toml) + JAX 0.8.0, NumPy, SciPy (for reference implementations)
