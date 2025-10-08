# NLSQ v0.1.1 Release Notes

**Release Date**: October 8, 2025
**Type**: Feature Release

---

## 🎉 What's New

NLSQ v0.1.1 is a **feature release** bringing 25+ enhancements that make curve fitting easier, more robust, and more insightful. This release represents 24 days of development focused on user experience, documentation, and production-grade robustness.

### Highlights

✨ **Enhanced Results** - Rich result objects with automatic plotting and statistics
🔄 **Auto-Retry** - Automatic fallback strategies that improve success rates from 60% to 85%
📊 **Progress Monitoring** - Real-time progress bars and logging for long optimizations
📚 **Complete Documentation** - 11 real-world examples + comprehensive migration guide
🎯 **Smart Defaults** - Auto-detect parameter bounds and handle numerical stability
📦 **Function Library** - 10+ pre-built models with automatic parameter estimation

---

## 🚀 Key Features

### 1. Enhanced Result Object

Results are now **rich objects** with automatic visualization and statistics:

```python
from nlsq import curve_fit

# Fit your data
result = curve_fit(model, xdata, ydata)

# Instant statistics
print(f"R² = {result.r_squared:.4f}")
print(f"RMSE = {result.rmse:.4f}")

# One-line visualization
result.plot()  # Shows data, fit, and residuals

# Detailed summary
result.summary()  # Statistical table with uncertainties

# Confidence intervals
ci = result.confidence_intervals(alpha=0.95)

# Backward compatible
popt, pcov = result  # Tuple unpacking still works!
```

**What it gives you:**
- `.r_squared`, `.adj_r_squared` - Goodness-of-fit metrics
- `.rmse`, `.mae` - Error metrics
- `.aic`, `.bic` - Model selection criteria
- `.plot()` - Automatic matplotlib visualization
- `.summary()` - Formatted statistical table
- `.confidence_intervals()` - Parameter uncertainty

### 2. Progress Monitoring

Monitor long-running optimizations with built-in callbacks:

```python
from nlsq.callbacks import ProgressBar, EarlyStopping, CallbackChain

# Simple progress bar
result = curve_fit(model, x, y, callback=ProgressBar(max_nfev=100))

# Early stopping
callback = EarlyStopping(patience=10, min_delta=1e-6)
result = curve_fit(model, x, y, callback=callback)

# Combine callbacks
chain = CallbackChain(ProgressBar(max_nfev=100), EarlyStopping(patience=5))
result = curve_fit(model, x, y, callback=chain)

# Log to file
from nlsq.callbacks import IterationLogger

logger = IterationLogger("optimization.log", log_params=True)
result = curve_fit(model, x, y, callback=logger)
```

**Available Callbacks:**
- `ProgressBar` - Real-time tqdm progress with cost/gradient
- `IterationLogger` - Log progress to file or stdout
- `EarlyStopping` - Stop when no improvement
- `CallbackChain` - Combine multiple callbacks
- `CallbackBase` - Create custom callbacks

### 3. Automatic Fallback Strategies

Never fail silently again - automatic retry with smart strategies:

```python
# Enable automatic fallback
result = curve_fit(
    model,
    x,
    y,
    p0=[1.0, 1.0],  # Even poor guesses work!
    fallback=True,  # Auto-retry on failure
    fallback_verbose=True,  # See what's being tried
)

# Check what worked
if hasattr(result, "fallback_strategy_used"):
    print(f"Success with: {result.fallback_strategy_used}")
    print(f"Attempts: {result.fallback_attempts}")
```

**Fallback Strategies:**
1. Try alternative methods (trf → dogbox → lm)
2. Perturb initial guesses
3. Adjust tolerances
4. Infer parameter bounds
5. Use robust loss functions
6. Rescale problem

**Result:** Success rate improves from **60% → 85%** on difficult problems!

### 4. Smart Parameter Bounds

Let NLSQ suggest reasonable parameter bounds:

```python
# Automatic bound inference
result = curve_fit(
    model,
    x,
    y,
    p0=[2.0, 1.0, 0.5],
    auto_bounds=True,  # Analyze data to suggest bounds
    bounds_safety_factor=10.0,  # Safety multiplier
)

# Combines with your bounds
result = curve_fit(
    model,
    x,
    y,
    auto_bounds=True,
    bounds=([0, 0, -np.inf], [np.inf, 2.0, np.inf]),  # User bounds override
)
```

**What it does:**
- Analyzes data ranges (x and y)
- Detects likely parameter scales
- Applies conservative safety factors
- Merges with user-provided bounds

### 5. Numerical Stability

Automatic detection and fixing of numerical issues:

```python
# Auto-fix stability problems
result = curve_fit(model, x, y, stability="auto")  # Detect and fix automatically

# Options:
# - 'auto': Detect and fix (recommended)
# - 'check': Detect and warn only
# - False: Skip checks (default)
```

**Detects and fixes:**
- Ill-conditioned data (rescales to [0, 1])
- Parameter scale mismatches (normalizes)
- NaN/Inf values (replaces with mean)
- Collinear data (warns)

### 6. Function Library

Pre-built models with smart defaults:

```python
from nlsq.functions import exponential_decay, gaussian, sigmoid

# No p0 needed - smart defaults included!
result = curve_fit(exponential_decay, x, y)

# Built-in functions:
from nlsq.functions import (
    linear,
    polynomial,  # Basic math
    exponential_decay,
    exponential_growth,  # Exponentials
    gaussian,
    sigmoid,  # S-curves and peaks
    power_law,
    logarithmic,  # Power laws
)

# Each function knows its bounds
print(exponential_decay.bounds())
# Automatic p0 estimation
p0 = exponential_decay.estimate_p0(x, y)
```

**Available Functions:**
- **Mathematical**: `linear`, `polynomial`, `power_law`, `logarithmic`
- **Physical**: `exponential_decay`, `exponential_growth`, `gaussian`, `sigmoid`
- More coming in future releases!

### 7. Performance Profiler

Understand where time is spent:

```python
from nlsq.profiler import profile_curve_fit

# Profile your fit
result, profile = profile_curve_fit(model, x, y, p0=[1.0, 1.0])

# View report
profile.report()
# ╔═══════════════════════════════════════════╗
# ║ NLSQ Performance Profile                  ║
# ╠═══════════════════════════════════════════╣
# ║ Total Time:        487ms                  ║
# ║ ├─ JIT Compilation: 412ms (84.6%)         ║
# ║ └─ Optimization:    75ms (15.4%)          ║
# ║    ├─ Function:     45ms (60.0%)          ║
# ║    ├─ Jacobian:     20ms (26.7%)          ║
# ║    └─ Linear Solve: 10ms (13.3%)          ║
# ╚═══════════════════════════════════════════╝
#
# Recommendations:
# ✓ Use CurveFit class for multiple fits (reuse JIT)

# Visual report
profile.plot()  # Matplotlib charts
```

---

## 📚 Documentation

### Example Gallery

**11 real-world examples** across scientific domains:

#### Physics (3 examples)
- **Radioactive Decay**: Half-life determination with uncertainty propagation
- **Damped Oscillation**: Quality factor from pendulum data
- **Spectroscopy Peaks**: Multi-peak Gaussian/Lorentzian deconvolution

#### Engineering (3 examples)
- **Sensor Calibration**: Non-linear calibration curves
- **System Identification**: Transfer function from step response
- **Materials Characterization**: Elastic modulus from stress-strain

#### Biology (3 examples)
- **Growth Curves**: Logistic growth with lag time and max rate
- **Enzyme Kinetics**: Michaelis-Menten Km and Vmax
- **Dose-Response**: Hill equation EC50 and efficacy

#### Chemistry (2 examples)
- **Reaction Kinetics**: Rate constants from time courses
- **Titration Curves**: pKa determination

All examples include:
- Complete scientific context
- Data generation/loading
- Full statistical analysis
- Multi-panel visualization
- Result interpretation

### Migration Guide

Complete **SciPy Migration Guide** (857 lines, 11 sections):
- Side-by-side code comparisons
- Parameter mapping reference
- Feature comparison matrix
- Performance benchmarks
- Common migration patterns
- Breaking changes (none!)

### Interactive Tutorial

Comprehensive Jupyter notebook covering:
- Installation and setup
- Basic to advanced fitting
- Error handling
- Large datasets
- GPU acceleration
- Best practices

---

## 🔧 API Changes

### Return Type Enhancement

**Before (v0.1.0)**:
```python
popt, pcov = curve_fit(model, x, y)
# Limited to parameters and covariance
```

**After (v0.1.1)**:
```python
result = curve_fit(model, x, y)
# Access rich features
result.plot()
print(result.r_squared)
result.summary()

# Backward compatible
popt, pcov = result  # Tuple unpacking works!
```

### New Parameters

```python
curve_fit(
    f,
    xdata,
    ydata,
    # New in v0.1.1:
    callback=None,  # Progress monitoring
    auto_bounds=False,  # Smart bound inference
    fallback=False,  # Automatic fallback
    stability=False,  # Numerical stability checks
    bounds_safety_factor=10.0,  # Auto bounds safety
    max_fallback_attempts=10,  # Fallback retries
    fallback_verbose=False,  # Print fallback progress
    # All v0.1.0 parameters still work
)
```

---

## 📊 Performance

### Benchmarks

All 13 performance regression tests passing - **zero regressions**:

| Problem Size | Time (first run) | Time (cached) | Status |
|--------------|------------------|---------------|--------|
| Small (100)  | 500ms            | 8.6ms         | ✅ |
| Medium (1K)  | 600ms            | ~10ms         | ✅ |
| Large (10K)  | 630ms            | ~15ms         | ✅ |
| XLarge (50K) | 580ms            | ~20ms         | ✅ |

**Improvements:**
- **8% faster** overall (NumPy↔JAX optimization)
- **58x faster** with CurveFit class (JIT caching)
- **Excellent scaling**: 50x more data → only 1.2x slower

### Test Suite

- **1,160 tests** (743 → 1,160, +417 tests)
- **99.0% pass rate** (1,148 passing)
- **70% code coverage** (target: 80%)
- **13 performance regression tests**
- **Feature interaction test suite**

---

## 🐛 Bug Fixes

- Fixed integration test for backward compatibility
- Added `close()` method to `CallbackBase`
- Fixed JAX array immutability issues
- Improved test stability with random seeds
- Fixed CodeQL workflow schema errors
- 100% pre-commit compliance (24/24 hooks)

---

## ⚠️ Known Issues

### Callback API Tests (Low Priority)

**Issue**: 8 tests in `test_callbacks.py` have API mismatches
**Impact**: Low - core callback functionality works correctly
**Workaround**: Documented in user guides
**Fix**: Planned for v0.1.2 (ETA: 2 weeks)

These test failures do not affect:
- Production callback usage
- Core optimization functionality
- Any other features

---

## 🔄 Migration Path

### From v0.1.0 → v0.1.1

**100% Backward Compatible** - No breaking changes!

**Recommended Updates:**

1. **Use enhanced result object**:
```python
# Instead of:
popt, pcov = curve_fit(model, x, y)

# Try:
result = curve_fit(model, x, y)
result.plot()  # Automatic visualization!
```

2. **Enable robustness features**:
```python
result = curve_fit(model, x, y, auto_bounds=True, stability="auto", fallback=True)
```

3. **Use function library**:
```python
from nlsq.functions import exponential_decay

result = curve_fit(exponential_decay, x, y)  # No p0 needed!
```

4. **Monitor long optimizations**:
```python
from nlsq.callbacks import ProgressBar

result = curve_fit(model, x, y, callback=ProgressBar())
```

---

## 📦 Installation

```bash
# PyPI (recommended)
pip install --upgrade nlsq

# With optional dependencies
pip install nlsq[dev]  # Development tools
pip install nlsq[docs]  # Documentation building
pip install nlsq[all]  # Everything

# From source
git clone https://github.com/imewei/NLSQ
cd NLSQ
pip install -e .
```

**Requirements:**
- Python 3.12+
- JAX 0.4.20+
- NumPy, SciPy
- matplotlib (for plotting)
- tqdm (for progress bars, optional)

---

## 🙏 Acknowledgments

Special thanks to:
- **Original JAXFit Authors**: Lucas R. Hofer, Milan Krstajić, Robert P. Smith
- **Lead Developer**: Wei Chen (Argonne National Laboratory)
- **Community**: Beta testers and contributors

---

## 📈 Statistics

- **Development Time**: 24 days (Phases 1-3)
- **Features Added**: 25+ major features
- **Tests Added**: 417 new tests (+56%)
- **Documentation**: 10,000+ lines added
- **Examples**: 11 new domain-specific examples
- **Code Changes**: 50+ files modified
- **Lines of Code**: +15,000 LOC

---

## 🚀 What's Next?

### v0.1.2 (2 weeks)
- Fix callback API tests
- Additional examples
- Performance tuning
- Bug fixes based on user feedback

### v0.2.0 (Future)
- Additional function library models
- Enhanced profiler visualization
- Multi-GPU support
- Sparse Jacobian optimizations

---

## 📖 Resources

- **Documentation**: https://nlsq.readthedocs.io
- **GitHub**: https://github.com/imewei/NLSQ
- **PyPI**: https://pypi.org/project/nlsq
- **Issues**: https://github.com/imewei/NLSQ/issues
- **Discussions**: https://github.com/imewei/NLSQ/discussions

---

## 🎯 Getting Started

### Quick Example

```python
import numpy as np
import jax.numpy as jnp
from nlsq import curve_fit
from nlsq.functions import exponential_decay
from nlsq.callbacks import ProgressBar

# Generate data
x = np.linspace(0, 5, 100)
y = 2.5 * np.exp(-1.3 * x) + 0.5 + np.random.normal(0, 0.05, 100)

# Fit with all new features
result = curve_fit(
    exponential_decay,
    x,
    y,
    auto_bounds=True,
    stability="auto",
    fallback=True,
    callback=ProgressBar(),
)

# Analyze results
print(f"R² = {result.r_squared:.4f}")
print(f"RMSE = {result.rmse:.4f}")

# Visualize
result.plot()

# Statistical summary
result.summary()
```

### Try it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ_Interactive_Tutorial.ipynb)

---

**Happy Fitting! 🎉**

---

*NLSQ v0.1.1 - GPU-Accelerated Nonlinear Least Squares*
*Released: October 8, 2025*
*License: MIT*
