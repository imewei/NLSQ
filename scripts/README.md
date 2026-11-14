# NLSQ Scripts

Utility scripts for maintaining the NLSQ project.

## Available Scripts

### `notebooks/convert_examples.py`
Bidirectional converter between example notebooks and scripts. Works on single files or whole folders.

**Usage**
- `python scripts/notebooks/convert_examples.py notebook-to-script examples/notebooks/01_getting_started/nlsq_quickstart.ipynb`
- `python scripts/notebooks/convert_examples.py script-to-notebook examples/scripts/01_getting_started/nlsq_quickstart.py`
- `python scripts/notebooks/convert_examples.py notebook-to-script examples/notebooks/01_getting_started/`
- `python scripts/notebooks/convert_examples.py script-to-notebook examples/scripts/01_getting_started/`

### `notebooks/configure_notebooks.py`
Click-based utility to apply transforms to notebooks (matplotlib config, import fixes, incremental updates). See `docs/developer/notebook_utilities.rst` for full CLI.

### `notebooks/regenerate_notebooks.py`
Rebuilds the notebooks for a whole scripts folder: `python scripts/notebooks/regenerate_notebooks.py examples/scripts/03_advanced [--banners]`. The notebook folder is the same path with `scripts` swapped for `notebooks`. The default mode splits a script into header, import, function and main cells. `--banners` is for flat tutorial scripts whose prose sits in `# ====` banner blocks: each block becomes a markdown cell, `savefig` becomes `plt.show()`, and a script-level quick-mode `sys.exit(0)` becomes a hook that turns the remaining cells into no-ops (`SystemExit` is a cell error in a notebook). Exits non-zero if any script fails to convert.

### `notebooks/run_scripts.py` and `notebooks/run_notebooks.py`
Run example scripts or notebooks sequentially in quick mode with strict warning handling, and exit non-zero if any of them fail. Pass files or folders as arguments. `script_runner.py` and `io_patch.py` redirect `savefig` output into a per-script artifact folder and leave file-like targets alone.

### `visualization/visualize_stability_performance.py`
Generates four plots of stability/overhead behaviour. The figures are illustrative models built from assumed constants (no NLSQ fit is run and nothing is measured; each title says so). PNGs are written to `--output-dir` (default `<system temp>/nlsq_stability_plots`). Exits non-zero on any plot error.

### `benchmarks/`
Performance benchmark scripts (benchmark_baseline.py, benchmark_us1.py, benchmark_us2.py). `benchmark_us1.py` and `benchmark_us2.py` exit non-zero when a success criterion fails (SC-002 in `benchmark_us2.py` is informational and not gated). They write JSON results under the gitignored `specs/002-performance-optimizations/`.

### `check_core_install.py`
Smoke test that the core install (`pip install nlsq` from a wheel, without the `gui`/dev extras) works: `curve_fit`, the chunked path and plotting, and that no non-core package (PySide6, pyqtgraph, pytest, ruff, sphinx, lmfit, jaxopt, ...) is importable. CI runs it in a wheel-only venv: `.venv-test/bin/python scripts/check_core_install.py`. It fails by design in a development environment (exit 1, "non-core packages importable"), so do not run it in your dev `.venv`.

### `quick_sitecustomize/`
Runtime helpers loaded in CI/example harnesses to cap sample sizes, seed RNGs, and force non-interactive matplotlib during automated runs.

Environment variables: `NLSQ_EXAMPLES_QUICK` (enables quick mode: CPU, no JIT, non-interactive matplotlib, capped samples; empty, `0`, `false`, `no` and `off` leave it off), `NLSQ_EXAMPLES_MAX_SAMPLES` (sample cap, default 5000; the test harnesses use 10), `NLSQ_EXAMPLES_TMPDIR` (matplotlib config directory). Library code called from numpy, scipy, matplotlib or jax is never capped, only example code.

## Notes
- Deprecated/unused maintenance scripts were removed (`ast_analysis.py`, `benchmark_notebook_utilities.py`, `configure_matplotlib_notebooks.py`) to keep this folder lean.
