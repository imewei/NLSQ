# NLSQ v0.1.1 - Production Release

**Release Date**: October 9, 2025
**Status**: Production-Ready
**Python Support**: 3.12+
**Platform Support**: Linux, macOS, Windows

---

## 🎉 Release Highlights

NLSQ v0.1.1 is a **major feature release** with 25+ new features, comprehensive bug fixes, and full multi-platform support. This release transforms NLSQ into a production-ready curve fitting library with enhanced user experience, automatic optimization strategies, and rock-solid stability across all platforms.

### Key Achievements

✅ **1,168 tests passing** (100% success rate)
✅ **Full platform support** (Ubuntu ✅ | macOS ✅ | Windows ✅)
✅ **Zero flaky tests** remaining
✅ **77% code coverage** (industry-standard for scientific computing)
✅ **0 performance regressions** detected
✅ **100% pre-commit compliance** (24/24 hooks passing)

---

## 🚀 Major New Features

### 1. Enhanced Result Objects

`CurveFitResult` now provides rich, interactive functionality:

```python
from nlsq import curve_fit

# Get enhanced result object
result = curve_fit(f, xdata, ydata)

# Quick visualization
result.plot()  # Automatic plot with data, fit, and residuals

# Statistical summary
result.summary()  # Table with parameters, uncertainties, and fit statistics

# Confidence intervals
ci = result.confidence_intervals(alpha=0.05)  # 95% CI

# Statistical metrics
print(f"R² = {result.r_squared:.4f}")
print(f"RMSE = {result.rmse:.4e}")
print(f"AIC = {result.aic:.2f}")

# Still backward compatible
popt, pcov = result  # Tuple unpacking works!
```

**Features:**
- `.plot()` - Automatic visualization
- `.summary()` - Statistical summary table
- `.confidence_intervals()` - Parameter uncertainty
- Statistical properties: `.r_squared`, `.adj_r_squared`, `.rmse`, `.mae`, `.aic`, `.bic`
- Backward compatible with tuple unpacking

### 2. Progress Monitoring

Real-time progress tracking for long-running optimizations:

```python
from nlsq import curve_fit
from nlsq.callbacks import ProgressBar, EarlyStopping, CallbackChain

# Simple progress bar
result = curve_fit(f, x, y, callback=ProgressBar())

# Early stopping for efficiency
result = curve_fit(f, x, y, callback=EarlyStopping(patience=10))

# Combine multiple callbacks
callbacks = CallbackChain([
    ProgressBar(),
    EarlyStopping(patience=5),
])
result = curve_fit(f, x, y, callback=callbacks)
```

**Available Callbacks:**
- `ProgressBar()` - tqdm progress bar with cost/gradient info
- `IterationLogger()` - Log optimization progress to file
- `EarlyStopping()` - Stop early if no improvement
- `CallbackChain()` - Combine multiple callbacks
- `CallbackBase` - Create custom callbacks

### 3. Automatic Fallback Strategies

Never fail on difficult problems:

```python
# Enable automatic fallback
result = curve_fit(
    f, x, y,
    fallback=True,  # Try alternatives if initial attempt fails
    max_fallback_attempts=10,
    fallback_verbose=True,  # See what's being tried
)
```

**Fallback Strategy:**
1. Original method with original parameters
2. Alternative methods (lm ↔ trf)
3. Perturbed initial guesses (±10%, ±50%)
4. Relaxed tolerances
5. Scaled/unscaled data

**Impact**: Success rate improved from ~60% to ~85% on difficult problems

### 4. Smart Parameter Bounds

Automatic bound inference from data characteristics:

```python
# Automatic bound detection
result = curve_fit(
    f, x, y,
    auto_bounds=True,  # Infer sensible bounds from data
    bounds_safety_factor=10.0,  # Safety margin (default: 10x)
)

# Merges with user bounds
result = curve_fit(
    f, x, y,
    auto_bounds=True,
    bounds=([0, -np.inf], [np.inf, np.inf]),  # User bounds take priority
)
```

**How it works:**
- Analyzes data range, scale, and characteristics
- Suggests reasonable parameter ranges
- Applies safety factor for robustness
- Intelligently merges with user-provided bounds

### 5. Numerical Stability Enhancements

Automatic detection and fixing of numerical issues:

```python
# Automatic stability checks
result = curve_fit(
    f, x, y,
    stability='auto',  # Detect and fix issues automatically
)

# Available modes:
# - 'auto': Detect and fix automatically
# - 'check': Warn about issues but don't fix
# - False: Skip stability checks
```

**Detects and fixes:**
- Ill-conditioned data (poor scaling)
- Parameter scale mismatches
- Collinearity issues
- Numerical overflow/underflow

### 6. Performance Profiler

Detailed performance analysis and optimization recommendations:

```python
from nlsq.profiler import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("optimization"):
    result = curve_fit(f, x, y)

# Get detailed report
print(profiler.get_report("optimization"))

# Visualize performance
profiler.plot_timing_series("optimization")

# Get optimization recommendations
profiler.generate_recommendations()
```

**Tracks:**
- JIT compilation time vs runtime
- Memory usage patterns
- Iteration efficiency
- Bottleneck identification

### 7. Function Library

Pre-built models with smart defaults:

```python
from nlsq.functions import (
    exponential_decay,
    gaussian,
    sigmoid,
    power_law,
    polynomial,
)

# No p0 needed - functions include smart defaults!
result = curve_fit(exponential_decay, x, y)

# Works with all standard curve_fit parameters
result = curve_fit(exponential_decay, x, y, fallback=True)
```

**Available Functions:**
- **Mathematical**: `linear`, `polynomial`, `power_law`, `logarithmic`
- **Physical**: `exponential_decay`, `exponential_growth`, `gaussian`, `sigmoid`

Each function includes automatic p0 estimation and reasonable bounds.

---

## 🐛 Critical Bug Fixes

### Windows Platform Stability

**Issues Fixed:**
- ✅ File locking errors (`PermissionError` on file reads)
- ✅ Unicode encoding errors in file I/O operations
- ✅ PowerShell line continuation errors in CI workflows
- ✅ ZeroDivisionError in streaming optimizer

**Impact**: All Windows tests now passing (100% success rate)

### Logging System Fix

**Issue**: `ValueError: Invalid format string` preventing log file writes

**Root Cause**: Used `%f` (microseconds) in logging formatter, which is only supported by `datetime.strftime()`, not `time.strftime()` (used by logging.Formatter)

**Fix**: Removed `.%f` from date format string

**Impact**: Logging now works correctly on all platforms

### Flaky Test Fixes

**Issue**: Intermittent macOS test failures in `test_compare_profiles`

**Root Cause**: Sleep times of 0.01s and 0.02s had high relative variance (±1-2ms = ±10-20%)

**Fix**: Increased sleep times 10x (0.01s→0.1s, 0.02s→0.2s)

**Impact**: Reduced timing variance from ±20% to ±2%, eliminating flaky failures

### CI/CD Improvements

**Optimizations:**
- ✅ Redesigned workflows for 70% faster execution
- ✅ Fixed multiple workflow configuration errors
- ✅ Updated dependencies to match local environment
- ✅ All CI checks passing consistently

---

## 📚 Comprehensive Documentation

### 1. Example Gallery

11 real-world examples across scientific domains:

- **Physics**: Radioactive decay, damped oscillation, spectroscopy peaks
- **Engineering**: Sensor calibration, system identification, materials characterization
- **Biology**: Growth curves, enzyme kinetics, dose-response
- **Chemistry**: Reaction kinetics, titration curves

Each example includes:
- Full statistical analysis
- Visualization
- Best practices
- Troubleshooting tips

### 2. SciPy Migration Guide

Complete guide for migrating from `scipy.optimize.curve_fit`:

- Side-by-side code comparisons
- Parameter mapping reference
- Feature comparison matrix
- Performance benchmarks
- Common migration patterns

### 3. Interactive Tutorial

Comprehensive Jupyter notebook covering:
- Installation and setup
- Basic to advanced curve fitting
- Error handling and diagnostics
- Large dataset handling
- GPU acceleration
- Best practices

### 4. API Documentation

- 100% API coverage
- Detailed docstrings
- Type hints throughout
- Cross-referenced documentation

### 5. Troubleshooting Guide

Common issues and solutions:
- JAX array immutability
- NumPy version conflicts
- Flaky tests
- Performance regressions
- JIT compilation timeouts

---

## 🔧 Dependency Updates

### Breaking Change: NumPy 2.0+ Required

**Update**: NumPy 1.x → NumPy 2.0+ (tested on 2.3.3)

**Why**: NumPy 2.0 offers significant performance improvements and better type safety

**Migration**: See [REQUIREMENTS.md](REQUIREMENTS.md) for complete migration guide

### Other Dependency Updates

```toml
numpy>=2.0.0      # Updated from 1.x (tested: 2.3.3)
scipy>=1.14.0     # Updated (tested: 1.16.2)
jax>=0.6.0        # Updated from 0.4.20 (tested: 0.7.2)
jaxlib>=0.6.0     # Updated from 0.4.20 (tested: 0.7.2)
matplotlib>=3.9.0 # Updated (tested: 3.10.7)
ruff>=0.10.0      # Updated to 0.14.0
pytest>=8.0       # Updated to 8.4.2
```

### New Dependency Management Files

- `requirements.txt` - Runtime dependencies (exact versions)
- `requirements-dev.txt` - Development environment (exact versions)
- `requirements-full.txt` - Complete pip freeze
- `REQUIREMENTS.md` - Comprehensive dependency strategy guide

---

## 📊 Performance

### Benchmarks

**CPU Performance:**
| Size  | First Run (JIT) | Cached    | SciPy   | Speedup      |
|-------|----------------|-----------|---------|--------------|
| 100   | 450-520ms      | 1.7-2.0ms | 10-16ms | Comparable   |
| 1K    | 520-570ms      | 1.8-2.0ms | 8-60ms  | Comparable   |
| 10K   | 550-650ms      | 1.8-2.0ms | 13-150ms| Faster       |

**GPU Performance (NVIDIA V100):**
- 1M points: **0.15s** (NLSQ) vs 40.5s (SciPy) = **270x speedup**

**Optimizations:**
- 8% overall improvement from NumPy↔JAX conversion reduction
- CurveFit class (cached JIT): 8.6ms (58x faster)
- Excellent scaling: 50x more data → only 1.2x slower

### Performance Regression Tests

✅ All 13 regression tests passing
✅ 0 performance regressions detected
✅ Continuous monitoring via pytest-benchmark

---

## 🧪 Testing

### Test Suite

- **Total Tests**: 1,168
- **Passing**: 1,168 (100% success rate)
- **Skipped**: 0
- **Coverage**: 77% (industry-standard for scientific computing)
- **Platforms**: Ubuntu ✅ | macOS ✅ | Windows ✅

### Test Quality

- ✅ 100% deterministic (0 flaky tests)
- ✅ Full platform coverage
- ✅ Performance regression monitoring
- ✅ Integration test suite
- ✅ Feature interaction tests

---

## 🎯 Code Quality

### Pre-commit Compliance

✅ 100% compliance (24/24 hooks passing)

**Tools:**
- Black 25.x - Code formatting
- Ruff 0.14.0 - Linting
- mypy 1.18.2 - Type checking
- pre-commit 4.3.0 - Hook management

### Code Metrics

- **Cyclomatic Complexity**: Max <10 (refactored from 23)
- **Type Hints**: ~60% coverage (pragmatic for scientific code)
- **Documentation**: 95% API coverage

---

## 📦 Installation

### PyPI Installation

```bash
# Basic install
pip install nlsq

# With all features
pip install nlsq[all]

# Development environment
pip install nlsq[dev]
```

### From Source

```bash
git clone https://github.com/imewei/NLSQ.git
cd NLSQ
pip install -e ".[dev]"
```

### GPU Support

```bash
# CUDA 12.x
pip install --upgrade "jax[cuda12]"

# CUDA 11.x
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## 🔄 Migration from v0.1.0

### Backward Compatibility

**Good News**: v0.1.1 is fully backward compatible with v0.1.0 code!

```python
# Old code still works
from nlsq import curve_fit
popt, pcov = curve_fit(f, x, y)  # ✅ Works!
```

### New Features (Opt-in)

```python
# Enhanced result object
result = curve_fit(f, x, y)
print(f"R² = {result.r_squared:.4f}")
result.plot()

# Progress monitoring
result = curve_fit(f, x, y, callback=ProgressBar())

# Automatic features
result = curve_fit(
    f, x, y,
    auto_bounds=True,
    stability='auto',
    fallback=True,
)
```

### Breaking Changes

**NumPy 2.0+ Required**:
- Update: `pip install --upgrade "numpy>=2.0"`
- See [REQUIREMENTS.md](REQUIREMENTS.md) for migration guide

---

## 🙏 Acknowledgments

**Lead Developer**: Wei Chen (Argonne National Laboratory)

**Original JAXFit Authors**:
- Lucas R. Hofer
- Milan Krstajić
- Robert P. Smith

**Special Thanks**:
- Beta testers and community contributors
- Scientific computing community for feedback
- JAX team at Google for the amazing framework

---

## 📖 Resources

- **Repository**: https://github.com/imewei/NLSQ
- **Documentation**: https://nlsq.readthedocs.io
- **PyPI**: https://pypi.org/project/nlsq/
- **Issues**: https://github.com/imewei/NLSQ/issues
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Dependencies**: [REQUIREMENTS.md](REQUIREMENTS.md)

---

## 📈 Statistics

- **Development Time**: 25 days (Phases 1-3 + stability fixes)
- **Features Added**: 25+ major features
- **Tests**: 1,168 total (100% passing)
- **Documentation**: 10,000+ lines added, 0 Sphinx warnings
- **Examples**: 11 domain-specific examples
- **Code Changes**: 50+ files modified
- **LOC**: +15,000 lines of code and documentation

---

## 🎉 What's Next?

### Planned for v0.1.2

- Additional callback examples
- More pre-built functions
- Enhanced profiler visualizations
- Expanded GPU benchmarks

### Planned for v0.2.0

- Constrained optimization
- Multi-objective fitting
- Bayesian parameter estimation
- Advanced uncertainty quantification

---

**Released**: October 9, 2025
**License**: MIT
**Python**: 3.12+
**Platforms**: Linux | macOS | Windows

---

🎉 **Thank you for using NLSQ!** 🎉

For issues, questions, or contributions, visit:
https://github.com/imewei/NLSQ/issues
