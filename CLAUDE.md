# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
NLSQ is a GPU/TPU-accelerated nonlinear least squares curve fitting library that ports SciPy's curve fitting algorithms to JAX. It provides a drop-in replacement for `scipy.optimize.curve_fit` with massive speedups on GPU/TPU hardware.

**Repository**: https://github.com/imewei/NLSQ
**Maintainer**: Wei Chen (Argonne National Laboratory)
**Origin**: Enhanced fork of JAXFit by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith

## Development Commands

### Running Tests
```bash
# Run all tests with pytest
make test

# Run only fast tests (excludes slow optimization tests)
make test-fast

# Run only slow optimization tests
make test-slow

# Run tests with coverage report
make test-cov

# Run tests on CPU backend only (avoids GPU compilation)
make test-cpu

# Run tests with debug logging
NLSQ_DEBUG=1 pytest -s

# Run specific test file
pytest tests/test_least_squares.py -v
pytest tests/test_minpack.py -v
```

### Building and Installation
```bash
# Install in development mode with all dependencies
make dev

# Install package only
make install

# Build the package
python -m build

# Run pre-commit hooks
pre-commit run --all-files
```

### Code Quality
```bash
# Run linting checks
make lint

# Format code with black and ruff
make format

# Type checking
make type-check

# Clean build artifacts and cache
make clean
```

## Architecture

### Core Components

The library is organized around several key modules in the `nlsq/` directory:

#### Core Optimization
1. **minpack.py** - Contains the main `CurveFit` class and `curve_fit` function that provides the high-level API compatible with SciPy
2. **least_squares.py** - Implements the `LeastSquares` class with the core least squares solver
3. **trf.py** - Trust Region Reflective algorithm implementation
4. **loss_functions.py** - Various loss functions for robust fitting
5. **_optimize.py** - Optimization result classes and utilities

#### Advanced Features
6. **algorithm_selector.py** - Automatic algorithm selection based on problem characteristics
7. **large_dataset.py** - Large dataset handling with automatic chunking and streaming
8. **memory_manager.py** - Memory management with configurable limits and monitoring
9. **smart_cache.py** - Smart caching system for repeated computations
10. **diagnostics.py** - Diagnostic monitoring and performance analysis
11. **recovery.py** - Error recovery mechanisms and fallback strategies
12. **stability.py** - Numerical stability improvements and condition monitoring
13. **validators.py** - Comprehensive input validation and sanitization
14. **robust_decomposition.py** - Robust linear algebra decompositions

#### Infrastructure
15. **config.py** - Configuration management and context managers
16. **common_jax.py** - JAX-specific utilities and functions
17. **common_scipy.py** - SciPy compatibility layer and shared utilities
18. **logging.py** - Logging and debugging utilities
19. **caching.py** - Core caching infrastructure
20. **optimizer_base.py** - Abstract base classes for optimization algorithms
21. **sparse_jacobian.py** - Sparse Jacobian matrix handling for large-scale problems
22. **streaming_optimizer.py** - Streaming optimization for very large datasets
23. **svd_fallback.py** - Fallback SVD implementations for numerical stability

### Key Design Principles

- **JAX JIT Compilation**: All fit functions must be JIT-compilable. This means avoiding Python control flow in fit functions and using JAX numpy operations.
- **Double Precision**: NLSQ requires 64-bit precision. The library automatically enables this when imported via `config.update("jax_enable_x64", True)`.
- **Automatic Differentiation**: Uses JAX's autodiff for computing Jacobians rather than requiring analytical derivatives or numeric approximation.
- **Backend Selection**: JAX automatically detects available accelerators (GPU/TPU). Use `JAX_PLATFORM_NAME=cpu` environment variable to force CPU-only execution.

### API Compatibility

The library maintains API compatibility with SciPy's `curve_fit`:
- Same function signature for `curve_fit(f, xdata, ydata, p0=None, ...)`
- Returns `popt` (optimized parameters) and `pcov` (covariance matrix)
- The `CurveFit` class allows reusing compiled functions for multiple fits

## Testing Approach

Tests use Python's unittest and pytest frameworks, with comprehensive coverage across all modules:

### Core Test Files
- `test_least_squares.py` - Tests for the least squares solver
- `test_minpack.py` - Tests for the curve_fit interface
- `test_trf_simple.py` - Tests for Trust Region Reflective algorithm
- `test_integration.py` - Integration tests across components
- `test_large_dataset.py` - Tests for large dataset handling
- `test_stability.py` - Numerical stability tests
- `test_streaming_optimizer.py` - Streaming optimization tests
- `test_sparse_jacobian.py` - Sparse Jacobian tests

### Coverage and Additional Tests
- Multiple coverage test files ensure comprehensive testing
- Tests compare results against SciPy implementations for correctness
- Performance benchmarking tests for GPU/TPU acceleration

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating the library's capabilities:
- `NLSQ Quickstart.ipynb` - Getting started guide with basic usage
- `NLSQ_2D_Gaussian_Demo.ipynb` - 2D Gaussian fitting demonstration
- `advanced_features_demo.ipynb` - Showcases advanced optimization features
- `large_dataset_demo.ipynb` - Demonstrates handling of large datasets with chunking and streaming
