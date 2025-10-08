# NLSQ Benchmarking Usage Guide

Comprehensive guide for running benchmarks, interpreting results, and selecting optimal configurations.

## Quick Start

### Running Standard Benchmarks

```bash
# Run standard benchmarks (exponential, gaussian, polynomial, sinusoidal)
python benchmark/run_benchmarks.py

# Quick benchmarks (smaller sizes, fewer repeats)
python benchmark/run_benchmarks.py --quick

# Custom output directory
python benchmark/run_benchmarks.py --output ./my_results
```

**Output Files**:
- Text report: `benchmark_results/benchmark_report.txt`
- CSV data: `benchmark_results/benchmark_results.csv`
- HTML dashboard: `benchmark_results/dashboard/dashboard.html`
- Comparison plots: `benchmark_results/dashboard/*.png`
- JSON export: `benchmark_results/dashboard/profiles.json`

---

## Command-Line Options

### Problem Selection

```bash
# Benchmark specific problems
python benchmark/run_benchmarks.py --problems exponential gaussian

# All available problems: exponential, gaussian, polynomial, sinusoidal
python benchmark/run_benchmarks.py --problems exponential gaussian polynomial sinusoidal
```

### Size and Repeat Configuration

```bash
# Custom problem sizes
python benchmark/run_benchmarks.py --sizes 100 1000 10000

# Custom number of repeats
python benchmark/run_benchmarks.py --repeats 10

# Combine options
python benchmark/run_benchmarks.py --sizes 100 1000 --repeats 5
```

### Method Selection

```bash
# Test specific methods
python benchmark/run_benchmarks.py --methods trf lm

# All available methods: trf, lm, dogbox
python benchmark/run_benchmarks.py --methods trf lm dogbox
```

### Comparison Options

```bash
# Skip SciPy comparison (faster)
python benchmark/run_benchmarks.py --no-scipy

# Include SciPy comparison (default)
python benchmark/run_benchmarks.py  # SciPy included by default
```

### Help

```bash
# View all options
python benchmark/run_benchmarks.py --help
```

---

## Advanced Usage

### Custom Benchmark Configuration

Create a Python script for custom benchmarks:

```python
from benchmark.benchmark_suite import (
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkResult
)

# Create custom configuration
config = BenchmarkConfig(
    name="custom_benchmark",
    problem_sizes=[100, 1000, 10000],
    n_repeats=10,
    warmup_runs=2,
    methods=["trf", "lm"],
    backends=["cpu"],
    compare_scipy=True
)

# Run benchmarks
runner = BenchmarkRunner(config)
results = runner.run_all_benchmarks()

# Generate reports
runner.generate_text_report(results, "my_report.txt")
runner.generate_csv_report(results, "my_results.csv")
```

### Profiling Performance

```bash
# Profile TRF algorithm hot paths
python benchmark/profile_trf.py

# View profiling recommendations
# Output includes: timing breakdown, JIT vs runtime, optimization suggestions
```

### CI/CD Performance Regression Tests

```bash
# Run regression tests
pytest benchmark/test_performance_regression.py --benchmark-only

# Save baseline for comparison
pytest benchmark/test_performance_regression.py --benchmark-save=baseline

# Compare against baseline
pytest benchmark/test_performance_regression.py --benchmark-compare=baseline

# Generate JSON report for CI
pytest benchmark/test_performance_regression.py --benchmark-json=ci_report.json
```

---

## Configuration Recommendations

### By Use Case

| Use Case | Configuration | Command Example |
|----------|--------------|-----------------|
| **High Performance Computing** | Default (no stability features) | Standard benchmark commands |
| **Production Systems** | Enable stability | Set `enable_stability=True` in code |
| **Numerical Research** | Stability + overflow check | Full configuration options |
| **Critical Applications** | All features enabled | Maximum reliability settings |

### By Dataset Size

| Dataset Size | Recommendation | Notes |
|-------------|---------------|-------|
| < 1K points | Use default `curve_fit()` | Fast, no overhead |
| 1K - 100K points | `curve_fit()` with optional stability | Configure as needed |
| > 100K points | Use `curve_fit_large()` | Automatic chunking |
| Numerical edge cases | Always enable overflow checking | Safety first |

### Solver Selection

| Problem Type | Best Solver | Rationale |
|-------------|------------|-----------|
| Small (< 100 points) | `auto` or `svd` | Fast and accurate |
| Medium (100-10K) | `cg` | Best performance (15.1ms vs 17.8ms SVD) |
| Large (> 10K) | `cg` or `lsqr` | Memory efficient |
| Ill-conditioned | `svd` | Most stable |
| Sparse Jacobian | `lsqr` | Optimized for sparsity |

### Backend Selection

```python
# Force CPU (useful for debugging, consistent benchmarks)
import jax
jax.config.update("jax_platform_name", "cpu")

# Use GPU if available (automatic detection)
# JAX automatically detects and uses GPU/TPU if available

# Check available backends
print(jax.devices())  # Lists available backends
```

---

## Interpreting Results

### Understanding Timing Output

**Example Output**:
```
Problem: exponential, Size: 1000, Method: trf
  NLSQ:  2.15 ms ± 0.10 ms
  SciPy: 0.24 ms ± 0.01 ms
  Ratio: 0.11x (NLSQ slower on CPU)
```

**Interpretation**:
- **NLSQ Time**: Includes JIT compilation (first run) or cached (subsequent)
- **± value**: Standard deviation across repeats
- **Ratio**: NLSQ/SciPy comparison (expect <1.0x on CPU, >1.0x on GPU)

### JIT Compilation Overhead

**First Run** (includes JIT compilation):
- Small problems: ~500ms total (~30ms runtime after JIT)
- Medium problems: ~600ms total (~110ms runtime after JIT)
- Large problems: ~630ms total (~134ms runtime after JIT)

**Cached Runs** (using CurveFit class):
- **58x faster** than first run
- Example: 8.6ms cached vs 500ms first run

**Tip**: Use `CurveFit` class for multiple fits to reuse JIT compilation:

```python
from nlsq import CurveFit

cf = CurveFit()
# First fit: ~500ms (includes JIT)
result1 = cf.fit(model1, x1, y1, p0=[1, 1])
# Subsequent fits: ~8.6ms (cached)
result2 = cf.fit(model2, x2, y2, p0=[1, 1])
result3 = cf.fit(model3, x3, y3, p0=[1, 1])
```

### Performance Expectations

**CPU Benchmarks** (NLSQ vs SciPy):
- ❌ NLSQ **10-20x slower** on CPU for small problems
- ✅ This is **expected** - NLSQ is optimized for GPU/TPU
- ✅ Recommendation: Use SciPy for <1K points on CPU

**GPU Benchmarks** (NLSQ vs SciPy):
- ✅ NLSQ **150-270x faster** on GPU for large problems
- ✅ Example: 1M points, 5 params: 0.15s (GPU) vs 40.5s (SciPy CPU)

---

## Troubleshooting

### Slow Performance

**Issue**: Benchmarks are extremely slow
**Causes**:
1. First run includes JIT compilation overhead
2. Stability features enabled (25-30% overhead)
3. Large problem sizes without chunking

**Solutions**:
```bash
# Run quick benchmarks
python benchmark/run_benchmarks.py --quick

# Use smaller sizes
python benchmark/run_benchmarks.py --sizes 100 1000

# Skip SciPy comparison
python benchmark/run_benchmarks.py --no-scipy
```

### Memory Errors

**Issue**: Out of memory errors on large datasets
**Solutions**:
1. Use `curve_fit_large()` instead of `curve_fit()`
2. Reduce problem sizes
3. Enable dynamic sizing (automatic)

```python
# For large datasets
from nlsq import curve_fit_large
result = curve_fit_large(model, x, y, p0=[1, 1])
```

### Inconsistent Results

**Issue**: Benchmark results vary significantly between runs
**Causes**:
1. Insufficient warmup runs
2. System resource contention
3. Random initialization variations

**Solutions**:
```bash
# Increase repeats
python benchmark/run_benchmarks.py --repeats 10

# Use more warmup runs (edit benchmark_suite.py)
config = BenchmarkConfig(warmup_runs=3)  # Default is 1
```

### GPU Not Detected

**Issue**: JAX not using GPU
**Check**:
```python
import jax
print(jax.devices())  # Should show GPU devices
```

**Solutions**:
1. Install CUDA-enabled JAX: `pip install -U "jax[cuda12]"`
2. Verify CUDA installation: `nvidia-smi`
3. Check JAX GPU support: https://jax.readthedocs.io/en/latest/installation.html

---

## Best Practices

### For Development

1. **Use quick mode** for fast iteration:
   ```bash
   python benchmark/run_benchmarks.py --quick
   ```

2. **Profile before optimizing**:
   ```bash
   python benchmark/profile_trf.py
   ```

3. **Run regression tests** before committing:
   ```bash
   pytest benchmark/test_performance_regression.py --benchmark-only
   ```

### For CI/CD

1. **Save baseline** after optimization work:
   ```bash
   pytest benchmark/test_performance_regression.py --benchmark-save=v0.1.1
   ```

2. **Compare in CI** to detect regressions:
   ```bash
   pytest benchmark/test_performance_regression.py --benchmark-compare=v0.1.1
   ```

3. **Set thresholds** for acceptable performance changes:
   - Currently: <5% slowdown triggers alert
   - Configurable in test_performance_regression.py

### For Research

1. **Document configuration** in notebooks/scripts
2. **Use sufficient repeats** (≥5) for statistical significance
3. **Report with uncertainty**: "2.15ms ± 0.10ms"
4. **Compare fairly**: CPU vs CPU, GPU vs GPU
5. **Include system specs** in reports

---

## Example Workflows

### Workflow 1: Quick Development Check

```bash
# Fast check during development
python benchmark/run_benchmarks.py --quick --no-scipy

# Output: benchmark_results/benchmark_report.txt
# Time: ~1-2 minutes
```

### Workflow 2: Comprehensive Comparison

```bash
# Full benchmarks with SciPy comparison
python benchmark/run_benchmarks.py \
  --problems exponential gaussian polynomial \
  --sizes 100 1000 10000 \
  --repeats 10 \
  --output ./full_results

# View HTML dashboard
open full_results/dashboard/dashboard.html
```

### Workflow 3: Performance Profiling

```bash
# Profile TRF algorithm
python benchmark/profile_trf.py > profiling_report.txt

# Analyze profiling report
cat profiling_report.txt
# Look for: JIT vs runtime, hot paths, optimization recommendations
```

### Workflow 4: Regression Testing in CI

```yaml
# GitHub Actions example
- name: Run Performance Regression Tests
  run: |
    pytest benchmark/test_performance_regression.py \
      --benchmark-only \
      --benchmark-compare=baseline \
      --benchmark-compare-fail=mean:5%
```

---

## Additional Resources

### Documentation
- [Historical Results](historical_results.md) - Benchmark data 2024-2025
- [Optimization History](completed/) - Completed optimizations
- [Future Work](future/) - Deferred optimizations
- [Main README](../README.md) - Quick start and overview

### External Resources
- [JAX Performance Tips](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [SciPy curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)

### NLSQ Documentation
- [API Documentation](../../docs/api.md)
- [Installation Guide](../../README.md#installation)
- [Examples](../../examples/)
- [CHANGELOG](../../CHANGELOG.md)

---

**Last Updated**: 2025-10-08
**Version**: v0.1.1
