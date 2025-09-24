# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
NLSQ is a GPU/TPU-accelerated nonlinear least squares curve fitting library that ports SciPy's curve fitting algorithms to JAX. It provides a drop-in replacement for `scipy.optimize.curve_fit` with massive speedups on GPU/TPU hardware.

## Development Commands

### Running Tests
```bash
# Run all tests using unittest
python -m unittest discover tests -p "test*.py"

# Run specific test file
python -m unittest tests.test_least_squares
python -m unittest tests.test_minpack
```

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Build the package
python -m build
```

## Architecture

### Core Components

The library is organized around several key modules in the `nlsq/` directory:

1. **minpack.py** - Contains the main `CurveFit` class and `curve_fit` function that provides the high-level API compatible with SciPy
2. **least_squares.py** - Implements the `LeastSquares` class with the core least squares solver
3. **trf.py** - Trust Region Reflective algorithm implementation
4. **loss_functions.py** - Various loss functions for robust fitting
5. **common_jax.py** - JAX-specific utilities and functions
6. **common_scipy.py** - SciPy compatibility layer and shared utilities
7. **_optimize.py** - Optimization result classes and utilities

### Key Design Principles

- **JAX JIT Compilation**: All fit functions must be JIT-compilable. This means avoiding Python control flow in fit functions and using JAX numpy operations.
- **Double Precision**: NLSQ requires 64-bit precision. The library automatically enables this when imported via `config.update("jax_enable_x64", True)`.
- **Automatic Differentiation**: Uses JAX's autodiff for computing Jacobians rather than requiring analytical derivatives or numeric approximation.

### API Compatibility

The library maintains API compatibility with SciPy's `curve_fit`:
- Same function signature for `curve_fit(f, xdata, ydata, p0=None, ...)`
- Returns `popt` (optimized parameters) and `pcov` (covariance matrix)
- The `CurveFit` class allows reusing compiled functions for multiple fits

## Testing Approach

Tests use Python's unittest framework and mirror SciPy's optimization test suite:
- `test_least_squares.py` - Tests for the least squares solver
- `test_minpack.py` - Tests for the curve_fit interface
- Tests compare results against SciPy implementations for correctness
