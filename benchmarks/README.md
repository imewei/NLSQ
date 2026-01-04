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
│   ├── memory_benchmark.py      # Memory pool optimization
│   ├── precision_benchmark.py   # Mixed precision fallback
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

# Memory optimization
python benchmarks/components/memory_benchmark.py

# Sparse solvers
python benchmarks/components/sparse_benchmark.py

# Stability overhead
python benchmarks/components/stability_benchmark.py

# Host-device transfers (GPU)
python benchmarks/components/transfer_benchmark.py --gpu --save-baseline
```

### Run Micro-benchmarks

```bash
# All microbenchmarks
pytest benchmarks/microbench/ --benchmark-only

# Save results
pytest benchmarks/microbench/ --benchmark-json=results.json

# Compare with baseline
pytest benchmarks/microbench/ --benchmark-compare=baseline
```

### Run CI Regression Tests

```bash
# Regression detection
pytest benchmarks/ci/ --benchmark-only

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
| `memory_benchmark.py` | Memory pool reuse | 10-20% memory reduction |
| `precision_benchmark.py` | Float32/64 fallback | Automatic precision upgrade |
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

## Baselines

Baseline results are stored in `baselines/` as JSON files:

- `cache_unification.json` - Cache performance baselines
- `jacobian_autoswitch.json` - Jacobian mode baselines
- `memory_reuse.json` - Memory optimization baselines
- `sparse_activation.json` - Sparse solver baselines
- `host_device_transfers.json` - Transfer profiling baselines
- `linux-py312-beta1.json` - CI regression baseline

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
# Core dependencies (included with nlsq[dev])
pip install pytest pytest-benchmark

# For GPU benchmarks
pip install jax[cuda12]
```
