# NLSQ Scripts

Utility scripts for maintaining the NLSQ project.

## Available Scripts

### `convert_examples.py`
Bidirectional converter between example notebooks and scripts. Works on single files or whole folders.

**Usage**
- `python scripts/convert_examples.py notebook-to-script examples/notebooks/01_getting_started/nlsq_quickstart.ipynb`
- `python scripts/convert_examples.py script-to-notebook examples/scripts/01_getting_started/nlsq_quickstart.py`
- `python scripts/convert_examples.py notebook-to-script examples/notebooks/01_getting_started/`
- `python scripts/convert_examples.py script-to-notebook examples/scripts/01_getting_started/`

### `configure_notebooks.py`
Click-based utility to apply transforms to notebooks (matplotlib config, import fixes, incremental updates). See `docs/developer/notebook_utilities.rst` for full CLI.

### `visualize_stability_performance.py`
Generates performance plots for stability/overhead benchmarks.

### `quick_sitecustomize/`
Runtime helpers loaded in CI/example harnesses to cap sample sizes, seed RNGs, and force non-interactive matplotlib during automated runs.

## Notes
- Deprecated/unused maintenance scripts were removed (`ast_analysis.py`, `benchmark_notebook_utilities.py`, `configure_matplotlib_notebooks.py`) to keep this folder lean.
