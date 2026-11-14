# NLSQ Benchmarks

Performance benchmarking suite for NLSQ, comparing against SciPy and tracking performance characteristics.

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── run_benchmarks.py            # CLI for running full benchmark suite
├── benchmark_suite.py           # Core benchmarking infrastructure
├── profile_trf.py               # TRF algorithm profiler
│
├── ci/                          # CI/CD regression tests
│   ├── __init__.py
│   └── test_regression.py       # Performance regression detection
│
├── components/                  # Component-specific benchmarks
│   ├── __init__.py
│   ├── cache_benchmark.py       # JIT compilation cache
│   ├── jacobian_benchmark.py    # Jacobian mode auto-switching
│   ├── sparse_benchmark.py      # Sparse solver detection
│   ├── stability_benchmark.py   # Stability guard overhead
│   └── transfer_benchmark.py    # Host-device transfers
│
├── microbench/                  # Micro-benchmarks (pytest-benchmark)
│   ├── __init__.py
│   ├── test_import.py           # Import time (SC-001)
│   ├── test_condition.py        # Condition estimation (SC-010)
│   ├── test_sparse.py           # Sparse Jacobian (SC-002)
│   └── test_optimizations.py    # Overall optimization
│
└── baselines/                   # Baseline results for comparison
    ├── create_baseline.py
    └── *.json
```

## Quick Start

### Run Full Benchmark Suite

```bash
# Standard benchmarks
python benchmarks/run_benchmarks.py

# Quick benchmarks (smaller sizes, fewer repeats)
python benchmarks/run_benchmarks.py --quick

# Custom configuration
python benchmarks/run_benchmarks.py --problems exponential gaussian --sizes 100 1000
```

### Run Component Benchmarks

```bash
# Cache performance
python benchmarks/components/cache_benchmark.py
python benchmarks/components/cache_benchmark.py --quick

# Jacobian auto-switching
python benchmarks/components/jacobian_benchmark.py
python benchmarks/components/jacobian_benchmark.py --mode=direct

# Sparse solvers
python benchmarks/components/sparse_benchmark.py

# Stability overhead
python benchmarks/components/stability_benchmark.py

# Host-device transfers (GPU)
python benchmarks/components/transfer_benchmark.py --gpu --save-baseline
```

`pyproject.toml` passes `--benchmark-disable` to every pytest run (it avoids pytest-benchmark
warnings under xdist), so the commands below need `--benchmark-enable` and `-n 0`.

### Run Micro-benchmarks

```bash
# All microbenchmarks
pytest benchmarks/microbench/ --benchmark-only --benchmark-enable -n 0

# Save results
pytest benchmarks/microbench/ --benchmark-json=results.json

# Compare with baseline
pytest benchmarks/microbench/ --benchmark-compare=baseline
```

### Run CI Regression Tests

```bash
# Regression detection
pytest benchmarks/ci/ --benchmark-only --benchmark-enable -n 0

# Save new baseline
pytest benchmarks/ci/ --benchmark-save=baseline

# Compare against baseline
pytest benchmarks/ci/ --benchmark-compare=baseline
```

## Benchmark Categories

### CI Regression Tests (`ci/`)

Tests designed for CI/CD pipeline integration:
- Automatic performance regression detection
- Baseline comparison with configurable thresholds
- JSON output for CI reporting

### Component Benchmarks (`components/`)

Standalone scripts for specific NLSQ components:

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `cache_benchmark.py` | JIT compilation cache | >80% hit rate, 2-5x speedup |
| `jacobian_benchmark.py` | jacfwd vs jacrev | 10-100x on high-param problems |
| `sparse_benchmark.py` | Sparse solver detection | 3-10x speedup on sparse |
| `stability_benchmark.py` | Stability guard overhead | <5% overhead |
| `transfer_benchmark.py` | GPU/CPU transfers | 80% transfer reduction |

### Micro-benchmarks (`microbench/`)

Focused pytest-benchmark tests for specific operations:

| Test | Success Criterion |
|------|-------------------|
| `test_import.py` | SC-001: <400ms import time |
| `test_condition.py` | SC-010: 50% faster condition estimation |
| `test_sparse.py` | SC-002: 100x sparse Jacobian speedup |
| `test_optimizations.py` | Overall optimization validation |

## Library comparison

Compares NLSQ (`curve_fit`, `curve_fit_large` default path, and `curve_fit_large(method="exact_streaming")`)
against SciPy, lmfit, JAXopt and Optimistix (`benchmarks/compare/`; adapters `nlsq`, `nlsq_large`, `nlsq_exact`).
Install the comparison libraries with `uv sync --group bench`.

```bash
uv run python -m benchmarks.compare.run_library_comparison \
    --sizes 1000 100000 1000000 --repeats 7 --out benchmarks/results
uv run python -m benchmarks.compare.run_library_comparison --quick --no-batch   # smoke run
# GPU (needs a CUDA jaxlib; run CPU and GPU sequentially, never at the same time):
JAX_PLATFORMS=cuda uv run python -m benchmarks.compare.run_library_comparison --out benchmarks/results
# 100M points: add --no-batch; SciPy and lmfit take minutes per fit, use --libraries/--repeats 1
```

Fairness rules: every library gets the same tolerance (1e-8) for each stopping
criterion it exposes and the same iteration budget; each case runs in a fresh
worker process after a trivial backend warm-up, so "cold" is the first call
(including JIT compilation) and "warm" is the median and IQR of repeated calls;
the cost is recomputed from the returned parameters, and a library is ranked only
if that cost matches the best in the case (a fast early stop is listed as
`did_not_reach_optimum`, not ranked). The raw JSON lands in the git-ignored
`benchmarks/results/`; only the newest rendered report is kept under `baselines/`
(currently `2026-11-13-library-comparison-1k-100M.md`: 1k to 100M points, CPU and GPU, with
the exact invocations; earlier reports are in git history). Timings on the
development laptop drift by up to about 50% between sessions.

## Baselines

Baseline results are stored in `baselines/` as JSON files:

- `cache_unification.json` - Cache performance baselines
- `jacobian_autoswitch.json` - Jacobian mode baselines
- `memory_reuse.json` - Memory optimization baselines
- `sparse_activation.json` - Sparse solver baselines
- `streaming_overhead.json`, `streaming_partial_batch_overhead.json` - Streaming overhead baselines
- `import_baseline.txt` - Import-time baseline read by `microbench/test_import.py`
- `v0.3.0-beta.3-linux.json`, `v0.3.0-beta.3-darwin.json` - Release cold-JIT / hot-path timings
- `create_baseline.py` - Script that writes a release baseline
- `2026-11-13-library-comparison-1k-100M.md` - Newest library comparison report

`ci/test_regression.py` looks for `linux-py312-beta1.json` (the regression gates). It is not
in the repository; without it the module still runs the benchmarks but applies no gates.
Create one with `pytest benchmarks/ci/ --benchmark-enable --benchmark-save=<name>`.

## Creating New Baselines

```bash
# Component baselines (saved automatically)
python benchmarks/components/cache_benchmark.py

# CI baselines
pytest benchmarks/ci/ --benchmark-save=my-baseline

# Custom output path
python benchmarks/components/cache_benchmark.py --output my_baseline.json
```

## Requirements

```bash
# Test tooling comes from `make dev` / `uv sync` (the `dev` dependency group);
# benchmark comparison libraries come from `uv sync --group bench`
pip install pytest pytest-benchmark

# For GPU benchmarks
pip install jax[cuda12]
```
