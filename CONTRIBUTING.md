# Contributing to NLSQ

Thank you for your interest in contributing to NLSQ! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code.

## Getting Started

NLSQ is an enhanced fork of JAXFit, providing JAX-based nonlinear least squares curve fitting with GPU/TPU acceleration. We build upon the excellent foundation laid by the original JAXFit authors (Lucas R. Hofer, Milan Krstajić, and Robert P. Smith).

Before contributing:

1. **Read the documentation**: Familiarize yourself with the [user guide](https://nlsq.readthedocs.io/) and [API reference](https://nlsq.readthedocs.io/en/latest/api/).
2. **Check existing issues**: Look at [open issues](https://github.com/imewei/NLSQ/issues) to see if your idea or bug report already exists.
3. **Understand the fork**: Review [AUTHORS.md](AUTHORS.md) to understand the relationship between NLSQ and JAXFit.
4. **Start small**: Consider starting with documentation improvements or small bug fixes before tackling major features.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git
- JAX-compatible hardware (CPU/GPU/TPU)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/imewei/NLSQ.git
   cd nlsq
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,test,docs]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Verify installation**:
   ```bash
   python -c "import nlsq; print(nlsq.__version__)"
   python -m pytest tests/ -x -v
   ```

### GPU Setup (Optional)

For GPU development:

```bash
# CUDA 12.x
pip install --upgrade "jax[cuda12]>=0.6.0"

# Or for CPU-only development
pip install --upgrade "jax[cpu]>=0.6.0"
```

### Dependency Management

For detailed information about dependencies, version requirements, and the project's dependency management strategy, see [REQUIREMENTS.md](REQUIREMENTS.md).

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing functionality issues
- **Features**: Add new curve fitting algorithms, optimizations, or utilities
- **Documentation**: Improve guides, API docs, examples, or tutorials
- **Performance**: Optimize existing code for speed or memory usage
- **Testing**: Add test coverage or improve existing tests

### Before You Start

1. **Create an issue**: For significant changes, create an issue first to discuss the approach.
2. **Check scope**: Ensure your contribution aligns with the project's goals.
3. **Avoid breaking changes**: Maintain backward compatibility when possible.

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=nlsq --cov-report=html

# Run specific test file
python -m pytest tests/test_minpack.py -v

# Run tests in parallel
python -m pytest -n auto

# Run only fast tests (skip slow integration tests)
python -m pytest -m "not slow"
```

### Test Structure

- **Unit tests**: Test individual functions and classes (`tests/test_*.py`)
- **Integration tests**: Test complete workflows (`tests/test_integration.py`)
- **Performance tests**: Benchmark critical paths (`tests/test_performance.py`)

### Writing Tests

- Use `pytest` fixtures for common test data
- Test both CPU and GPU code paths when applicable
- Include edge cases and error conditions
- Add performance tests for critical algorithms
- Use descriptive test names: `test_curve_fit_with_bounds_and_weights`

Example test:

```python
import pytest
import numpy as np
import jax.numpy as jnp
from nlsq import curve_fit


def test_exponential_fit_basic():
    """Test basic exponential curve fitting."""

    def exponential(x, a, b):
        return a * jnp.exp(b * x)

    x = np.linspace(0, 2, 50)
    true_params = [2.0, -0.5]
    y_true = exponential(x, *true_params)
    y_data = y_true + np.random.normal(0, 0.1, len(x))

    popt, pcov = curve_fit(exponential, x, y_data, p0=[1.5, -0.3])

    np.testing.assert_allclose(popt, true_params, rtol=0.1)
    assert pcov.shape == (2, 2)
    assert np.all(np.diag(pcov) > 0)  # Positive variances
```

## Documentation

### Building Documentation

```bash
# Build HTML documentation
cd docs/
make html

# Build and serve locally
make livehtml  # Requires sphinx-autobuild

# Clean build
make clean html
```

### Documentation Standards

- **Docstrings**: Use NumPy-style docstrings for all public APIs
- **Type hints**: Include comprehensive type annotations
- **Examples**: Include working code examples in docstrings
- **Cross-references**: Link to related functions and concepts

Example docstring:

```python
def curve_fit_large(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray | None = None,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Curve fitting with automatic large dataset handling.

    This function provides a drop-in replacement for `curve_fit` with automatic
    detection and handling of large datasets. For small datasets (< 1M points),
    it behaves identically to `curve_fit`.

    Parameters
    ----------
    f : callable
        The model function f(x, *params) -> y
    xdata : np.ndarray
        Independent variable data
    ydata : np.ndarray
        Dependent variable data
    p0 : np.ndarray, optional
        Initial parameter guess

    Returns
    -------
    popt : np.ndarray
        Optimal parameters
    pcov : np.ndarray
        Covariance matrix of the parameters

    Examples
    --------
    >>> import numpy as np
    >>> import jax.numpy as jnp
    >>> from nlsq import curve_fit_large
    >>>
    >>> def exponential(x, a, b):
    ...     return a * jnp.exp(-b * x)
    >>>
    >>> x = np.linspace(0, 5, 1000000)  # Large dataset
    >>> y = exponential(x, 2.5, 1.3) + np.random.normal(0, 0.1, len(x))
    >>> popt, pcov = curve_fit_large(exponential, x, y, p0=[2, 1])
    """
```

## Code Style

We use automated code formatting and linting tools:

### Tools Used

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit**: Automatic checks

### Running Code Style Tools

```bash
# Format code
black nlsq/ tests/ examples/

# Check and fix linting issues
ruff check --fix nlsq/ tests/

# Type checking
mypy nlsq/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Style Guidelines

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, sort with ruff
- **Type hints**: Required for all public APIs
- **Docstrings**: NumPy style for public functions
- **Variable names**: Descriptive names, avoid abbreviations
- **Function length**: Keep functions focused and readable

## Submitting Changes

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Test your changes**:
   ```bash
   python -m pytest
   pre-commit run --all-files
   ```

4. **Update documentation** if needed:
   ```bash
   cd docs/ && make html
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "feat: add support for sparse Jacobians

   - Implement SparseJacobianComputer class
   - Add automatic sparsity detection
   - Include comprehensive tests and documentation

   Closes #123"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes made
   - Any breaking changes noted

### Pull Request Guidelines

- **Atomic commits**: Each commit should represent a single logical change
- **Test coverage**: Maintain or improve test coverage
- **Documentation**: Update docs for API changes
- **Backward compatibility**: Avoid breaking changes when possible
- **Performance**: Consider performance implications

### Review Process

1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer review required
3. **Testing**: Verify changes work as expected
4. **Documentation**: Ensure docs are updated and accurate

## Release Process

Releases are managed by maintainers and follow semantic versioning:

- **Patch** (0.0.X): Bug fixes and minor improvements
- **Minor** (0.X.0): New features, backward compatible
- **Major** (X.0.0): Breaking changes

### Release Workflow

1. Update `CHANGELOG.md` with release notes
2. Bump version using `setuptools-scm`
3. Create release tag and GitHub release
4. Build and publish to PyPI
5. Update documentation on ReadTheDocs

## Getting Help

- **Documentation**: https://nlsq.readthedocs.io/
- **Issues**: https://github.com/imewei/NLSQ/issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors are recognized in:

- `CHANGELOG.md` for their contributions
- GitHub contributor statistics
- Special recognition for significant contributions

Thank you for contributing to NLSQ! 🚀
