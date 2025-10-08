# NLSQ Feature Demonstrations

Python scripts demonstrating Phase 1-3 features introduced in NLSQ v0.1.1.

## Overview

These demonstrations showcase new features added during the v0.1.1 development sprint (October 2025). Each script is self-contained and can be run independently to explore specific features.

**Why Python Scripts?**

These demos are Python scripts (not notebooks) for:
- ✅ **Quick execution and testing** - Run from command line
- ✅ **Easy integration** - Copy code into your own projects
- ✅ **Minimal dependencies** - No Jupyter required
- ✅ **Command-line usage** - Automate and script

## Phase 1 Features (Days 1-6: Quick Wins)

### 1. Result Enhancements Demo

**File**: `result_enhancements_demo.py`

**Features Demonstrated**:
- Enhanced `CurveFitResult` object
- Automatic visualization with `.plot()`
- Statistical summary with `.summary()`
- Confidence intervals with `.confidence_intervals()`
- Goodness-of-fit metrics: R², adjusted R², RMSE, MAE
- Model selection criteria: AIC, BIC
- Backward compatibility via tuple unpacking

**Usage**:
```bash
python examples/demos/result_enhancements_demo.py
```

**Key Concepts**:
```python
from nlsq import curve_fit

# Fit returns enhanced result object
result = curve_fit(model, x, y)

# New features (v0.1.1)
result.plot()  # Automatic visualization
result.summary()  # Statistical table
print(f"R² = {result.r_squared:.4f}")

# Backward compatible
popt, pcov = result  # Tuple unpacking still works!
```

---

### 2. Callbacks Demo

**File**: `callbacks_demo.py`

**Features Demonstrated**:
- `ProgressBar` - Real-time progress with tqdm
- `EarlyStopping` - Stop when no improvement
- `IterationLogger` - Log optimization progress
- `CallbackChain` - Combine multiple callbacks
- Custom callbacks via `CallbackBase`

**Usage**:
```bash
python examples/demos/callbacks_demo.py
```

**Key Concepts**:
```python
from nlsq import curve_fit
from nlsq.callbacks import ProgressBar, EarlyStopping, CallbackChain

# Progress monitoring
result = curve_fit(model, x, y, callback=ProgressBar(max_nfev=100))

# Early stopping
callback = EarlyStopping(patience=10, min_delta=1e-6)
result = curve_fit(model, x, y, callback=callback)

# Combine callbacks
chain = CallbackChain(ProgressBar(), EarlyStopping(patience=5))
result = curve_fit(model, x, y, callback=chain)
```

---

### 3. Function Library Demo

**File**: `function_library_demo.py`

**Features Demonstrated**:
- Pre-built model functions with smart defaults
- Automatic p0 estimation
- Reasonable parameter bounds
- Common mathematical and physical models

**Available Functions**:
- **Mathematical**: `linear`, `polynomial`, `power_law`, `logarithmic`
- **Physical**: `exponential_decay`, `exponential_growth`, `gaussian`, `sigmoid`

**Usage**:
```bash
python examples/demos/function_library_demo.py
```

**Key Concepts**:
```python
from nlsq import curve_fit
from nlsq.functions import exponential_decay, gaussian, sigmoid

# No p0 needed - smart defaults included!
result = curve_fit(exponential_decay, x, y)

# Each function knows its bounds
print(exponential_decay.bounds())

# Automatic p0 estimation
p0 = exponential_decay.estimate_p0(x, y)
```

---

### 4. Enhanced Error Messages Demo

**File**: `enhanced_error_messages_demo.py`

**Features Demonstrated**:
- Actionable error messages with recommendations
- Clear diagnostics for common issues
- Input validation with helpful suggestions
- Better debugging information

**Usage**:
```bash
python examples/demos/enhanced_error_messages_demo.py
```

**Key Concepts**:
```python
from nlsq import curve_fit

# Example: Poor initial guess
try:
    result = curve_fit(model, x, y, p0=[0, 0])  # May fail
except RuntimeError as e:
    print(f"Error: {e}")
    # Error message includes:
    # - What went wrong
    # - Why it happened
    # - How to fix it
    # - Suggestions for p0
```

---

## Phase 2 Features (Days 7-14: Documentation & Examples)

Phase 2 focused on documentation and example gallery. See:
- **Main Examples**: `../` (Jupyter notebooks)
- **Gallery**: `../gallery/` (11 domain-specific examples)
- **Documentation**: `../../docs/user_guides/`

---

## Phase 3 Features (Days 15-24: Advanced Features)

Phase 3 features are demonstrated in:
- **Automatic Fallback**: `../advanced_features_demo.ipynb`
- **Smart Bounds**: `../advanced_features_demo.ipynb`
- **Stability Enhancements**: `../advanced_features_demo.ipynb`
- **Performance Profiler**: `../performance_optimization_demo.ipynb`

---

## Converting to Jupyter Notebooks

To use these demos interactively in Jupyter:

### Option 1: Using jupytext

```bash
# Install jupytext
pip install jupytext

# Convert single demo
jupytext --to notebook demos/callbacks_demo.py

# Convert all demos
jupytext --to notebook demos/*.py
```

### Option 2: Using nbconvert

```bash
# Convert and execute
jupyter nbconvert --to notebook --execute callbacks_demo.py
```

### Option 3: Manual copy

1. Open the `.py` file in a text editor
2. Copy code sections
3. Paste into Jupyter notebook cells
4. Run interactively

---

## Requirements

All demos require:
- **nlsq** (v0.1.1+) - This package
- **numpy** - Numerical operations
- **matplotlib** - Visualization
- **jax** - Automatically installed with nlsq

Optional dependencies:
- **tqdm** - Progress bars (for callbacks_demo.py)
- **scipy** - Statistical functions (some examples)

Install with:
```bash
pip install nlsq[all]  # Includes all optional dependencies
```

---

## Learning Path

**Recommended Order**:
1. **result_enhancements_demo.py** - Start here, most impactful feature
2. **callbacks_demo.py** - Progress monitoring for long optimizations
3. **function_library_demo.py** - Convenient pre-built models
4. **enhanced_error_messages_demo.py** - Better debugging

**Time Required**: 15-20 minutes total (3-5 min per demo)

---

## Integration with Your Code

These demos are designed to be copied and adapted:

```python
# Example: Adding progress bar to your existing code
from nlsq import curve_fit
from nlsq.callbacks import ProgressBar

# Your existing code
# result = curve_fit(my_model, x, y, p0=[1, 1])

# Enhanced with progress bar (one line change)
result = curve_fit(my_model, x, y, p0=[1, 1], callback=ProgressBar())
```

---

## Running All Demos

To run all demonstrations sequentially:

```bash
for demo in examples/demos/*.py; do
    echo "Running $demo..."
    python "$demo"
    echo "---"
done
```

---

## Comparison with Notebooks

| Feature | Python Scripts (demos/) | Jupyter Notebooks (../) |
|---------|------------------------|-------------------------|
| **Purpose** | Feature demonstrations | Comprehensive tutorials |
| **Length** | Short (~100-200 lines) | Long (~500+ cells) |
| **Execution** | Command-line | Interactive |
| **Focus** | Single feature | End-to-end workflow |
| **Best for** | Quick reference | Learning and exploration |

**Use Demos When**:
- You want to quickly test a specific feature
- You need code to copy into your project
- You prefer command-line execution
- You're integrating into a script

**Use Notebooks When**:
- You're learning NLSQ for the first time
- You want comprehensive tutorials
- You prefer interactive exploration
- You need detailed explanations

---

## Contributing

To add a new demo:

1. Create Python script in `demos/` directory
2. Follow naming convention: `feature_name_demo.py`
3. Include sections:
   - Docstring with description
   - Feature overview
   - Example code with comments
   - Output demonstration
   - Usage instructions
4. Test thoroughly
5. Update this README

---

## References

- **Phase 1 Completion Summary**: `../../docs/development/phase1/sprint1_completion_summary.md`
- **Main Examples README**: `../README.md`
- **NLSQ Documentation**: `../../docs/`
- **CHANGELOG**: `../../CHANGELOG.md` (v0.1.1 section)

---

**Last Updated**: 2025-10-08
**NLSQ Version**: v0.1.1
**Phase**: Phase 1-3 (Complete)
