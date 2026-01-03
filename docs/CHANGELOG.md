# Documentation Changelog

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
