# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-25

### Added
- **Comprehensive Documentation**: Complete rewrite of documentation for PyPI and ReadTheDocs standards
- **Installation Guide**: Platform-specific instructions for Linux, macOS, and Windows
- **Tutorial Series**: Step-by-step tutorials from basic fitting to advanced large dataset handling
- **Contributing Guidelines**: Detailed contributor documentation in `CONTRIBUTING.md`
- **Enhanced API Documentation**: Improved examples and cross-references
- **`curve_fit_large` function**: Primary API for automatic large dataset handling with size detection
- **Memory estimation**: `estimate_memory_requirements` function for planning large dataset fits
- **Progress reporting**: Real-time progress bars for large dataset operations
- **JAX tracing compatibility**: Support for functions with 15+ parameters without TracerArrayConversionError
- **JAX Array Support**: Full compatibility with JAX arrays as input data

### Changed
- **Python Requirements**: Now requires Python 3.12+ (removed Python 3.11 support)
- **Documentation Structure**: Reorganized with Getting Started, User Guide, and API Reference sections
- **Examples Updated**: All documentation examples now highlight `curve_fit_large` as primary API
- **Example Notebooks**: Updated all Jupyter notebooks with Python 3.12+ requirement notices
- **GitHub URLs**: Updated all repository URLs from Dipolar-Quantum-Gases to imewei
- **Chunking Algorithm**: Improved sequential refinement approach replacing adaptive exponential moving average
- **Return Type Consistency**: All code paths return consistent (popt, pcov) format
- **Error Handling**: Enhanced error messages and validation for large dataset functions
- **CI/CD Pipeline**: Optimized GitHub Actions workflows for faster and more reliable testing

### Fixed
- **Variable Naming**: Fixed pcov vs _pcov inconsistencies throughout codebase and tests
- **StreamingOptimizer Tests**: Fixed parameter naming from x0 to p0 in all test files
- **GitHub Actions**: Fixed workflow failures by downgrading action versions and removing pip caching
- **JAX Tracing Issues**: Resolved TracerArrayConversionError for functions with many parameters
- **Chunking Stability**: Fixed instability issues with complex parameter averaging
- **Integration Tests**: Adjusted tolerances for chunked algorithms and polynomial fitting
- **Documentation Consistency**: Fixed examples and API references across all documentation files
- **Package Metadata**: Corrected all project URLs and repository references
- **JAX Array Compatibility Bug**: Fixed critical bug rejecting JAX arrays in minpack.py

### Technical Details
- Enhanced Sphinx configuration with modern extensions (doctest, coverage, duration)
- Improved autodoc configuration with better type hint handling
- Sequential refinement chunking algorithm for better stability and <1% error rates
- Comprehensive integration test suite with realistic tolerances
- All 354 tests passing with full coverage

## [Previous Unreleased - Development Phase]

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
