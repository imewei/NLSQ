# Documentation Changelog

## v0.6.0 (2026-01-06)
- Reorganized documentation into Routine User Guide and Advanced User Guide
- Consolidated 3 migration files into single comprehensive `howto/migration.rst`
- Removed stale deprecation notices from result module
- Removed 8 non-existent workflow presets from CLI documentation
- Fixed broken documentation links in README

## v0.5.4 (2026-01-05)
- Migrated to Furo theme for modern documentation styling
- Removed all deprecated functionality from NLSQ package
- Comprehensive documentation update with AST-based code analysis (94.9% coverage)
- Fixed pre-push hook CI interference
- Fixed Windows resource module import compatibility

## v0.5.3 (2026-01-04)
- Replaced pyqtdarktheme with Qt 6.5+ built-in color scheme (`setColorScheme()`)
- Updated dependency versions: JAX 0.8.2, NumPy 2.4.0, SciPy 1.16.3, ruff 0.14.10, mypy 1.19.1
- Added CHANGELOG entries for v0.5.0, v0.5.1, v0.5.2 release history
- Reorganized benchmark suite into ci/components/microbench directories

## v0.5.2 (2026-01-04)
- 15% performance improvement in TRF optimizer (10K point fit: 1.04s → 0.88s)
- Fixed JIT compatibility in TRF subproblem solvers
- Fixed enum identity issues in parallel test execution

## v0.5.1 (2026-01-03)
- Added `nlsq.gui_qt` API documentation to Sphinx
- Fixed README link to GUI user guide (now points to ReadTheDocs)
- Updated README GUI description to reflect pyqtgraph plots (was incorrectly referencing Plotly)
- Fixed pre-commit issues: refactored complex functions, added contextlib.suppress, created .secrets.baseline
- Added mypy overrides for PySide6, pyqtgraph, and qdarktheme in pyproject.toml

## v0.5.0 (2026-01-02)
- **BREAKING**: Removed Streamlit GUI in favor of native Qt desktop application
  - The `nlsq.gui` package has been removed entirely
  - Use `nlsq.gui_qt` and `nlsq-gui` command for the desktop GUI
  - Removed `gui` optional extra from pyproject.toml; use `gui_qt` instead
  - Install with: `pip install "nlsq[gui_qt]"`
- Removed legacy StreamingOptimizer/StreamingConfig and the Adam warmup path from docs; use AdaptiveHybridStreamingOptimizer with HybridStreamingConfig (L-BFGS warmup) for large datasets.
- Removed DataGenerator, create_hdf5_dataset, and fit_unlimited_data from examples/docs; use LargeDatasetFitter or AdaptiveHybridStreamingOptimizer instead.
- Updated streaming workflow guidance to reflect hybrid streaming only.
