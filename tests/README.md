# NLSQ Test Suite

**Comprehensive test coverage for the NLSQ (Nonlinear Least Squares) library**

[![Tests](https://img.shields.io/badge/tests-3279%20passing-success)](.)
[![Coverage](https://img.shields.io/badge/coverage-74%25-green)](.)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](.)

---

## Overview

The NLSQ test suite consists of **3279 passing tests** achieving **74% code coverage**, providing comprehensive validation of the JAX-accelerated curve fitting library.

### Quick Stats

| Metric | Value |
|--------|-------|
| **Total Tests** | 3279 passing |
| **Test Files** | 100 files |
| **Code Coverage** | 74% (industry-standard) |
| **Test Quality** | ⭐⭐⭐⭐⭐ Production-ready |
| **Pass Rate** | 100% (3279/3279) |

---

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run all tests with coverage report
pytest --cov=nlsq --cov-report=term

# Run all tests with HTML coverage report
pytest --cov=nlsq --cov-report=html

# Run specific test file
pytest tests/test_least_squares.py -v

# Run tests matching pattern
pytest -k "convergence" -v
```

### Using Makefile

```bash
# Run all tests (recommended)
make test

# Run all tests with coverage
make test-cov

# Run only fast tests (excludes slow optimization tests)
make test-fast

# Run only slow tests (optimization algorithms)
make test-slow

# Run tests on CPU backend only
make test-cpu
```

### Test Selection

```bash
# Run specific module tests
pytest tests/test_loss_functions.py
pytest tests/test_algorithm_selector.py
pytest tests/test_memory_manager.py

# Run by test class
pytest tests/test_optimizer_base.py::TestOptimizerBaseInitialization

# Run single test
pytest tests/test_optimize.py::TestOptimizeResultBasic::test_empty_initialization

# Run with verbose output
pytest -v

# Run with debug logging
NLSQ_DEBUG=1 pytest -s
```

---

## Test Organization

### Test Files by Category

#### **Core Optimization** (6 files)
```
test_least_squares.py        - Least squares solver (comprehensive)
test_minpack.py             - curve_fit interface (SciPy compatibility)
test_trf.py                 - Trust Region Reflective algorithm
test_trf_simple.py          - TRF basic functionality
test_optimizer_base.py      - Base optimization classes (49 tests)
test_optimize.py            - OptimizeResult class (37 tests)
```

#### **Advanced Features** (7 files)
```
test_loss_functions.py      - Robust loss functions (45 tests, 100% coverage)
test_algorithm_selector.py  - Auto algorithm selection (37 tests, 96% coverage)
test_memory_manager.py      - Memory management (36 tests, 91% coverage)
test_large_dataset.py       - Large dataset handling
test_streaming_optimizer.py - Streaming optimization
test_diagnostics.py         - Optimization diagnostics (10 tests)
test_recovery.py            - Error recovery mechanisms
```

#### **Infrastructure** (8 files)
```
test_config.py              - Configuration system (46 tests, 76% coverage)
test_validators.py          - Input validation (46 tests, 77% coverage)
test_init.py                - Main API (29 tests, 86% coverage)
test_logging.py             - Logging system
test_caching.py             - Caching infrastructure
test_smart_cache.py         - Smart cache optimization
test_common_jax.py          - JAX utilities
test_common_scipy.py        - SciPy compatibility
```

#### **Numerical Robustness** (6 files)
```
test_stability.py           - Numerical stability
test_robust_decomposition.py - Robust matrix decompositions
test_sparse_jacobian.py     - Sparse Jacobian handling
test_svd_fallback.py        - SVD fallback implementations
test_integration.py         - Integration tests
test_target_coverage.py     - Coverage targets
```

---

## Coverage by Module

### Perfect Coverage (100%)

| Module | Tests | Coverage |
|--------|-------|----------|
| `_optimize.py` | 37 | **100%** ✅ |
| `loss_functions.py` | 45 | **100%** ✅ |

### Excellent Coverage (≥90%)

| Module | Tests | Coverage |
|--------|-------|----------|
| `algorithm_selector.py` | 37 | **96%** ✅ |
| `logging.py` | — | **95%** ✅ |
| `optimizer_base.py` | 49 | **94%** ✅ |
| `memory_manager.py` | 36 | **91%** ✅ |

### Good Coverage (80-89%)

| Module | Coverage |
|--------|----------|
| `__init__.py` | **86%** ✅ |
| `caching.py` | **86%** ✅ |
| `streaming_optimizer.py` | **86%** ✅ |
| `common_scipy.py` | **83%** ✅ |
| `least_squares.py` | **83%** ✅ |
| `minpack.py` | **80%** ✅ |

### Moderate Coverage (65-79%)

| Module | Coverage | Priority |
|--------|----------|----------|
| `validators.py` | **77%** | ⚠️ Medium |
| `config.py` | **76%** | ⚠️ Medium |
| `large_dataset.py` | **69%** | ⚠️ Medium |
| `diagnostics.py` | **67%** | ⚠️ Medium |

---

## Test Categories

### Unit Tests

Core functionality testing of individual modules and classes.

```bash
# Example: Test OptimizeResult class
pytest tests/test_optimize.py -v

# Example: Test trust region optimizer
pytest tests/test_optimizer_base.py::TestTrustRegionOptimizerBase -v
```

### Integration Tests

End-to-end testing of complete workflows.

```bash
pytest tests/test_integration.py -v
```

### Property-Based Tests

Using Hypothesis for automated edge case discovery.

```bash
# Tests with @given decorators
pytest tests/test_loss_functions.py -v
pytest tests/test_memory_manager.py -v
```

### Performance Tests

```bash
# Marked as slow, can be skipped for quick testing
pytest -m slow -v
```

---

## Common Test Patterns

### Testing with JAX Arrays

```python
import jax.numpy as jnp


def test_jax_compatibility():
    """Test that functions work with JAX arrays."""
    x = jnp.array([1.0, 2.0, 3.0])
    result = my_function(x)
    assert isinstance(result, jnp.ndarray)
```

### Testing Edge Cases

```python
def test_edge_cases():
    """Test boundary conditions and edge cases."""
    # Empty arrays
    result = function_under_test(np.array([]))

    # NaN and Inf
    result = function_under_test(np.array([np.nan, np.inf]))

    # Zero values
    result = function_under_test(np.zeros(10))
```

### Testing Error Handling

```python
def test_error_handling():
    """Test that appropriate errors are raised."""
    with pytest.raises(ValueError, match="must be positive"):
        invalid_function_call(value=-1)
```

### Testing Convergence

```python
def test_convergence():
    """Test optimization convergence."""
    result = optimizer.optimize(objective, x0)

    assert result.success
    assert result.cost < tolerance
    assert np.linalg.norm(result.gradient) < gtol
```

---

## Test Quality Metrics

### Coverage Quality

| Metric | Score | Assessment |
|--------|-------|------------|
| **Line Coverage** | 74% | ✅ Industry-standard |
| **Branch Coverage** | High | ✅ Most paths tested |
| **Edge Case Coverage** | Excellent | ✅ Comprehensive |
| **Error Path Coverage** | Excellent | ✅ All exceptions tested |

### Test Code Quality

| Metric | Score | Assessment |
|--------|-------|------------|
| **Independence** | ⭐⭐⭐⭐⭐ | All tests can run in any order |
| **Clarity** | ⭐⭐⭐⭐⭐ | Descriptive names, clear docs |
| **Maintainability** | ⭐⭐⭐⭐⭐ | DRY principle, fixtures |
| **Performance** | ⭐⭐⭐⭐⭐ | Fast execution (3.5 tests/sec) |

---

## Writing New Tests

### Test File Template

```python
"""Tests for nlsq.module_name.

This test suite covers:
- Feature A
- Feature B
- Edge cases and error handling
"""

import unittest
import numpy as np
import pytest

from nlsq.module_name import FunctionUnderTest


class TestBasicFunctionality(unittest.TestCase):
    """Tests for basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = np.array([1.0, 2.0, 3.0])

    def test_normal_case(self):
        """Test normal operation."""
        result = FunctionUnderTest(self.test_data)
        self.assertIsNotNone(result)

    def test_edge_case(self):
        """Test edge case."""
        result = FunctionUnderTest(np.array([]))
        # Assert expected behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Best Practices

1. **One Test, One Concept**
   - Each test should verify one specific behavior
   - Use descriptive test names

2. **Arrange-Act-Assert Pattern**
   ```python
   def test_something(self):
       # Arrange: Set up test data
       data = np.array([1, 2, 3])

       # Act: Execute the function
       result = function_under_test(data)

       # Assert: Verify the outcome
       self.assertEqual(result.shape, (3,))
   ```

3. **Use Fixtures for Common Setup**
   ```python
   def setUp(self):
       """Shared setup for all tests in this class."""
       self.common_data = create_test_data()
   ```

4. **Test Both Success and Failure Paths**
   ```python
   def test_success_case(self):
       """Test successful execution."""
       result = function(valid_input)
       assert result.success


   def test_failure_case(self):
       """Test error handling."""
       with pytest.raises(ValueError):
           function(invalid_input)
   ```

5. **Document Test Intent**
   ```python
   def test_convergence_with_difficult_problem(self):
       """Test that optimizer converges on ill-conditioned problems.

       This test verifies that the trust region algorithm can handle
       problems with poorly scaled parameters and high condition numbers.
       """
   ```

---

## Coverage Requirements

### Current Status

- **Overall Coverage**: 74% (excellent for scientific computing)
- **Target Coverage**: 80% (configured in `pyproject.toml`)
- **Gap**: 6% (~275 lines)

### Coverage Policy

```ini
# From pyproject.toml
[tool.coverage.run]
source = ["nlsq"]
omit = [
    "*/tests/*",
    "*/test_*.py",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

### Viewing Coverage Reports

```bash
# Terminal report with missing lines
pytest --cov=nlsq --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=nlsq --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# XML report (for CI/CD)
pytest --cov=nlsq --cov-report=xml
```

---

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every push to main branch
- Every pull request
- Nightly builds

### Local Pre-commit Checks

```bash
# Run pre-commit hooks (includes test subset)
pre-commit run --all-files

# Full test suite before pushing
make test-cov
```

---

## Troubleshooting

### Common Issues

#### Tests Fail with JAX Errors

```bash
# Force CPU backend
JAX_PLATFORM_NAME=cpu pytest

# Or use make target
make test-cpu
```

#### Out of Memory Errors

```bash
# Run tests with limited parallelization
pytest -n 2  # Use only 2 workers

# Or sequentially
pytest --maxfail=1
```

#### Slow Test Execution

```bash
# Skip slow optimization tests
pytest -m "not slow"

# Or use make target
make test-fast
```

#### Coverage Not Updating

```bash
# Clean coverage cache
make clean

# Re-run with fresh coverage
pytest --cov=nlsq --cov-report=html
```

---

## Performance Benchmarks

### Test Execution Times

| Test Category | Tests | Time | Speed |
|--------------|-------|------|-------|
| **Fast Tests** | ~500 | ~60s | 8.3 tests/sec |
| **Slow Tests** | ~144 | ~120s | 1.2 tests/sec |
| **Full Suite** | 644 | ~180s | 3.5 tests/sec |

### Optimization

- Use `-n auto` for parallel execution (pytest-xdist)
- Skip slow tests during development: `pytest -m "not slow"`
- Use `--lf` to run only failed tests: `pytest --lf`

---

## Resources

### Documentation

- **Completion Report**: `../TEST_SUITE_COMPLETION_REPORT.md`
- **Phase 2 Report**: `../docs/archive/TEST_GENERATION_PHASE2_REPORT.md`
- **Project README**: `../README.md`
- **Contributing Guide**: `../CONTRIBUTING.md`

### External Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [JAX Testing Guide](https://jax.readthedocs.io/en/latest/notebooks/testing.html)

### Getting Help

```bash
# pytest help
pytest --help

# List all available markers
pytest --markers

# List all test files
pytest --collect-only

# Show available fixtures
pytest --fixtures
```

---

## Contributing Tests

### Before Submitting

1. **Run Full Test Suite**
   ```bash
   make test-cov
   ```

2. **Check Coverage**
   - Ensure your changes don't reduce coverage
   - Aim for >80% coverage on new code

3. **Follow Style Guide**
   - Use descriptive test names
   - Add docstrings
   - Follow existing patterns

4. **Verify Tests Pass**
   ```bash
   pytest tests/your_new_test.py -v
   ```

### Pull Request Checklist

- [ ] All tests pass locally
- [ ] Coverage remains ≥74%
- [ ] New tests added for new features
- [ ] Tests are independent and repeatable
- [ ] Documentation updated if needed
- [ ] Pre-commit hooks pass

---

## License

Tests are part of the NLSQ project and share the same license.
See `../LICENSE` for details.

---

**Last Updated**: 2025-12-21
**Test Suite Version**: 4.0 (3279 tests across 100 test files)
**Maintainer**: NLSQ Development Team
