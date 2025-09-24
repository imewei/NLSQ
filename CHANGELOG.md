# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Renamed package from JAXFit to NLSQ
- Migrated to modern pyproject.toml configuration
- Updated minimum Python version to 3.12
- Switched to explicit imports throughout the codebase
- Modernized development tooling with ruff, mypy, and pre-commit
- Updated all dependencies to latest stable versions

### Added
- Type hints throughout the codebase (PEP 561 compliant)
- Comprehensive CI/CD with GitHub Actions
- Support for Python 3.13 (development)
- Property-based testing with Hypothesis
- Benchmarking support with pytest-benchmark and ASV
- Modern documentation with MyST parser support

### Removed
- Support for Python < 3.12
- Obsolete setup.cfg and setup.py files
- Debug scripts and test artifacts
- Commented-out code and unused imports

## [0.0.5] - 2024-01-01

### Initial Release as NLSQ
- Core functionality for nonlinear least squares fitting
- GPU/TPU acceleration via JAX
- Drop-in replacement for scipy.optimize.curve_fit
- Trust Region Reflective algorithm implementation
- Multiple loss functions support
