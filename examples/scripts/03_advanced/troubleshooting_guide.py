"""
Converted from troubleshooting_guide.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # NLSQ Troubleshooting Guide
#
# **Level**: All Levels
# **Time**: Reference guide (browse as needed)
# **Prerequisites**: NLSQ Quickstart
#
# ## Overview
#
# This guide covers **common issues** encountered when using NLSQ and provides **practical solutions**. Each problem includes:
# - Clear symptoms and error messages
# - Root cause explanation
# - Step-by-step fixes
# - Working code examples
#
# ### Quick Navigation
#
# 1. **Convergence Failures**: "Optimal parameters not found", max iterations reached
# 2. **Poor Fit Quality**: High residuals, wrong parameter values
# 3. **Numerical Issues**: NaN, inf, JAX errors
# 4. **Performance Problems**: Slow compilation, memory errors
# 5. **Error Messages**: Decoding common NLSQ/JAX errors
# 6. **Best Practices**: Preventing issues before they occur
# ======================================================================


# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib

import contextlib
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from nlsq import CurveFit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


# Show all warnings (helpful for debugging)
warnings.filterwarnings("default")

print("✓ Setup complete")
print(f"  JAX version: {jax.__version__}")
print(f"  JAX backend: {jax.default_backend()}")


# ======================================================================
# ## Issue 1: Convergence Failures
#
# ### Symptoms
# - Error: `OptimizeWarning: Optimal parameters not found`
# - Error: `RuntimeError: Fitting failed to converge`
# - Warning: `Maximum number of iterations reached`
# - Parameters unchanged or nonsensical values
#
# ### Root Causes
# 1. **Poor initial guess** (`p0` far from true values)
# 2. **Inadequate iteration limit** (complex fits need more iterations)
# 3. **Ill-conditioned problem** (parameters on very different scales)
# 4. **Local minima** (non-convex optimization landscape)
#
# ### Solutions
# ======================================================================


# Demonstrate convergence failure and fix

# Generate test data
x_data = np.linspace(0, 10, 50)
y_data = 5.0 * np.exp(-0.5 * x_data) + np.random.normal(0, 0.1, 50)


def exponential_decay(x, a, b):
    return a * jnp.exp(-b * x)


# Create fresh CurveFit instance (avoid lingering state from previous demos)
cf = CurveFit()

# PROBLEM: Bad initial guess
print("❌ PROBLEM: Poor initial guess")
try:
    p0_bad = [0.1, 10.0]  # Far from true values [5.0, 0.5]
    popt_bad, _ = cf.curve_fit(
        exponential_decay, jnp.array(x_data), jnp.array(y_data), p0=p0_bad, maxiter=100
    )
    print(f"  Fitted params: a={popt_bad[0]:.2f}, b={popt_bad[1]:.2f}")
    print("  True params: a=5.0, b=0.5 (likely poor fit!)\n")
except Exception as e:
    print(f"  Error: {e}\n")

# SOLUTION 1: Better initial guess
print("✓ SOLUTION 1: Improve initial guess")
# Strategy: Estimate from data
a_guess = y_data[0]  # Initial value (t=0)
# Handle edge case where ratio might be negative due to noise
ratio = y_data[-1] / y_data[0]
b_guess = -np.log(max(ratio, 0.01)) / (x_data[-1] - x_data[0])  # Decay rate
p0_good = [a_guess, b_guess]
print(f"  Estimated p0: a={p0_good[0]:.2f}, b={p0_good[1]:.2f}")

popt_good, _ = cf.curve_fit(
    exponential_decay, jnp.array(x_data), jnp.array(y_data), p0=p0_good
)
print(f"  Fitted params: a={popt_good[0]:.2f}, b={popt_good[1]:.2f}")
print("  True params: a=5.0, b=0.5 ✓\n")

# SOLUTION 2: Increase iteration limit
print("✓ SOLUTION 2: Increase maxiter for complex problems")
popt_iter, _ = cf.curve_fit(
    exponential_decay,
    jnp.array(x_data),
    jnp.array(y_data),
    p0=[1.0, 1.0],
    maxiter=1000,  # Default is often 200-400
)
print(f"  Fitted with maxiter=1000: a={popt_iter[0]:.2f}, b={popt_iter[1]:.2f} ✓")


# ======================================================================
# ## Issue 2: Poor Fit Quality
#
# ### Symptoms
# - Fit converges but doesn't match data well
# - High chi-squared or RMSE
# - Parameters don't make physical sense
# - Residuals show clear patterns
#
# ### Root Causes
# 1. **Wrong model** (functional form doesn't match data)
# 2. **Insufficient model complexity** (missing terms)
# 3. **Bounds too restrictive** (excluding true parameters)
# 4. **Outliers** dominating the fit
#
# ### Solutions
# ======================================================================


# Demonstrate poor fit quality and fixes

# Generate data with offset (y = a*exp(-b*x) + c)
x_data = np.linspace(0, 5, 40)
y_true = 3.0 * np.exp(-0.8 * x_data) + 1.5  # Has offset!
y_data = y_true + np.random.normal(0, 0.1, len(x_data))

# PROBLEM: Wrong model (missing offset term)
print("❌ PROBLEM: Model mismatch (missing offset)")


def exp_no_offset(x, a, b):
    return a * jnp.exp(-b * x)


popt_wrong, _ = cf.curve_fit(
    exp_no_offset, jnp.array(x_data), jnp.array(y_data), p0=[3.0, 0.8]
)
y_fit_wrong = exp_no_offset(jnp.array(x_data), *popt_wrong)
rmse_wrong = np.sqrt(np.mean((y_data - y_fit_wrong) ** 2))
print(f"  RMSE with wrong model: {rmse_wrong:.3f} (high!)\n")

# SOLUTION: Add missing offset term
print("✓ SOLUTION: Use correct model with offset")


def exp_with_offset(x, a, b, c):
    return a * jnp.exp(-b * x) + c


popt_correct, _ = cf.curve_fit(
    exp_with_offset, jnp.array(x_data), jnp.array(y_data), p0=[3.0, 0.8, 1.0]
)
y_fit_correct = exp_with_offset(jnp.array(x_data), *popt_correct)
rmse_correct = np.sqrt(np.mean((y_data - y_fit_correct) ** 2))
print(
    f"  Fitted: a={popt_correct[0]:.2f}, b={popt_correct[1]:.2f}, c={popt_correct[2]:.2f}"
)
print(f"  RMSE with correct model: {rmse_correct:.3f} (much better!) ✓")
print(f"  Improvement: {(rmse_wrong - rmse_correct) / rmse_wrong * 100:.1f}%")


# Demonstrate bounds issues

# PROBLEM: Bounds excluding true parameters
print("\n❌ PROBLEM: Overly restrictive bounds")
bounds_wrong = ([0, 0, 0], [2.0, 1.0, 1.0])  # c is actually 1.5!

popt_bounded, _ = cf.curve_fit(
    exp_with_offset,
    jnp.array(x_data),
    jnp.array(y_data),
    p0=[1.9, 0.5, 0.5],  # Adjusted to be strictly within bounds
    bounds=bounds_wrong,
)
print(
    f"  Fitted with tight bounds: a={popt_bounded[0]:.2f}, b={popt_bounded[1]:.2f}, c={popt_bounded[2]:.2f}"
)
print("  True offset c=1.5, but bounds limited to c ≤ 1.0!\n")

# SOLUTION: Relax bounds or remove them
print("✓ SOLUTION: Use wider bounds or let optimizer explore")
bounds_good = ([0, 0, 0], [10.0, 5.0, 5.0])  # More generous

popt_unbounded, _ = cf.curve_fit(
    exp_with_offset,
    jnp.array(x_data),
    jnp.array(y_data),
    p0=[2.0, 0.5, 1.0],
    bounds=bounds_good,
)
print(
    f"  Fitted with relaxed bounds: a={popt_unbounded[0]:.2f}, b={popt_unbounded[1]:.2f}, c={popt_unbounded[2]:.2f} ✓"
)


# ======================================================================
# ## Issue 3: Numerical Instability (NaN, Inf)
#
# ### Symptoms
# - Error: `ValueError: array must not contain infs or NaNs`
# - Warning: `RuntimeWarning: overflow encountered in exp`
# - Parameters become NaN or Inf during optimization
#
# ### Root Causes
# 1. **Overflow** in exponentials (`exp(large_number)` → inf)
# 2. **Underflow** in divisions (divide by zero)
# 3. **Poor parameter scaling** (parameters differ by orders of magnitude)
# 4. **Invalid operations** (sqrt of negative, log of zero)
#
# ### Solutions
# ======================================================================


# Demonstrate numerical instability and fixes

# PROBLEM: Exponential overflow
print("❌ PROBLEM: Exponential overflow")

x_large = np.linspace(0, 100, 50)
y_data_large = 1.0 / (1.0 + np.exp(-0.1 * (x_large - 50))) + np.random.normal(
    0, 0.01, 50
)


def logistic_unstable(x, L, k, x0):
    # UNSTABLE: exp can overflow for large k*(x-x0)
    return L / (1.0 + jnp.exp(-k * (x - x0)))


try:
    popt, _ = cf.curve_fit(
        logistic_unstable,
        jnp.array(x_large),
        jnp.array(y_data_large),
        p0=[1.0, 1.0, 50.0],
    )
    print(f"  Fitted (might have issues): {popt}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}\n")

# SOLUTION 1: Use numerically stable formulation
print("✓ SOLUTION 1: Rewrite function to avoid overflow")


def logistic_stable(x, L, k, x0):
    # STABLE: Uses log-space computations internally
    z = -k * (x - x0)
    # Use jnp.where to handle both overflow regions
    return jnp.where(
        z > 0,
        L / (1.0 + jnp.exp(z)),  # Safe when z > 0
        L * jnp.exp(-z) / (jnp.exp(-z) + 1.0),  # Safe when z ≤ 0
    )


popt_stable, _ = cf.curve_fit(
    logistic_stable, jnp.array(x_large), jnp.array(y_data_large), p0=[1.0, 0.1, 50.0]
)
print(
    f"  Fitted with stable version: L={popt_stable[0]:.2f}, k={popt_stable[1]:.2f}, x0={popt_stable[2]:.1f} ✓\n"
)

# SOLUTION 2: Parameter rescaling
print("✓ SOLUTION 2: Rescale parameters to similar magnitudes")


def model_rescaled(x, L, k_scaled, x0):
    # k_scaled = k * 100 (so we fit k in [0.01, 1] instead of [0.0001, 0.01])
    k = k_scaled / 100.0
    return logistic_stable(x, L, k, x0)


popt_rescaled, _ = cf.curve_fit(
    model_rescaled, jnp.array(x_large), jnp.array(y_data_large), p0=[1.0, 10.0, 50.0]
)
print(
    f"  Fitted with rescaling: L={popt_rescaled[0]:.2f}, k={popt_rescaled[1] / 100:.3f}, x0={popt_rescaled[2]:.1f} ✓"
)


# ======================================================================
# ## Issue 4: Performance Problems
#
# ### Symptoms
# - Very slow first call (compilation time)
# - Subsequent calls still slow
# - Memory errors with large datasets
# - Error: `RESOURCE_EXHAUSTED: Out of memory`
#
# ### Root Causes
# 1. **JIT compilation overhead** (JAX compiles on first call)
# 2. **Large data** (millions of points)
# 3. **Complex models** (many nested operations)
# 4. **Unnecessary recompilation** (changing array shapes)
#
# ### Solutions
# ======================================================================


# Performance optimization tips

import time

# PROBLEM: Slow first call (JIT compilation)
print("ℹ UNDERSTANDING: JIT compilation (first call slow, then fast)")

x_perf = jnp.linspace(0, 10, cap_samples(1000))
y_perf = 2.0 * jnp.sin(x_perf) + np.random.normal(0, 0.1, cap_samples(1000))


def sine_model(x, a, b):
    return a * jnp.sin(b * x)


# First call: includes compilation time
start = time.time()
popt1, _ = cf.curve_fit(sine_model, x_perf, y_perf, p0=[1.0, 1.0])
time1 = time.time() - start

# Second call: already compiled
start = time.time()
popt2, _ = cf.curve_fit(sine_model, x_perf, y_perf, p0=[1.5, 0.8])
time2 = time.time() - start

print(f"  First call: {time1 * 1000:.1f} ms (includes compilation)")
print(f"  Second call: {time2 * 1000:.1f} ms (cached, faster!)")
print(f"  Speedup: {time1 / time2:.1f}x\n")

# SOLUTION 1: Pre-compile with dummy call
print("✓ SOLUTION 1: Warm up JIT cache with dummy call")
cf_new = CurveFit()
# Dummy call with data to compile (use full dataset for stable fit)
with contextlib.suppress(Exception):
    # Compilation happened even if fit didn't fully converge
    _ = cf_new.curve_fit(sine_model, x_perf, y_perf, p0=[1.0, 1.0], max_nfev=50)
print("  JIT cache warmed up (subsequent calls will be fast) ✓\n")

# SOLUTION 2: Use consistent array shapes (avoid recompilation)
print("✓ SOLUTION 2: Keep array shapes consistent")
print("  ❌ Bad: Changing shapes triggers recompilation")
print("     fit(x[:100], ...)  # Compiles for shape (100,)")
print("     fit(x[:200], ...)  # Recompiles for shape (200,) ⚠")
print("  ✓ Good: Same shapes reuse compilation")
print("     fit(x, ...)  # Compiles for shape (1000,)")
print("     fit(x, ...)  # Reuses compilation ✓")


# Handling large datasets

print("\n✓ SOLUTION 3: Strategies for large datasets (>1M points)\n")

print("Option A: Downsample data (if appropriate)")
print("  # For smooth, oversampled data")
print("  stride = 10")
print("  x_sub = x_data[::stride]  # Every 10th point")
print("  y_sub = y_data[::stride]")
print("  popt, pcov = cf.curve_fit(model, x_sub, y_sub, ...)\n")

print("Option B: Binning/averaging")
print("  # Combine neighboring points")
print("  n_bins = 10000")
print("  x_binned = np.array([x_data[i::n_bins].mean() for i in range(n_bins)])")
print("  y_binned = np.array([y_data[i::n_bins].mean() for i in range(n_bins)])\n")

print("Option C: Use float32 instead of float64")
print("  # Halves memory usage, often sufficient precision")
print("  x_data = jnp.array(x_data, dtype=jnp.float32)")
print("  y_data = jnp.array(y_data, dtype=jnp.float32)\n")

print("Option D: Streaming/chunked fitting (advanced)")
print("  # For distributed data, fit chunks separately then combine")
print("  # See: examples/streaming/ directory")


# ======================================================================
# ## Issue 5: Common Error Messages
#
# ### Quick Reference
# ======================================================================


# Common error messages and solutions

print("━" * 80)
print("COMMON NLSQ/JAX ERROR MESSAGES & SOLUTIONS")
print("━" * 80)
print()

errors = [
    {
        "error": "ValueError: operands could not be broadcast together",
        "cause": "Array shape mismatch (x and y different lengths)",
        "fix": "Check len(x) == len(y), reshape arrays if needed",
    },
    {
        "error": "TypeError: Argument 'x' of type <class 'list'> is not a valid JAX type",
        "cause": "Passing Python list instead of JAX/NumPy array",
        "fix": "Convert to array: jnp.array(x) or np.array(x)",
    },
    {
        "error": "LinAlgError: Singular matrix",
        "cause": "Covariance matrix is singular (parameters not identifiable)",
        "fix": "Reduce model complexity, check for redundant parameters, add bounds",
    },
    {
        "error": "ValueError: array must not contain infs or NaNs",
        "cause": "Model produces NaN/Inf during optimization",
        "fix": "Check model for overflow (exp, division), add bounds, rescale params",
    },
    {
        "error": "IndexError: tuple index out of range",
        "cause": "Accessing parameters that don't exist",
        "fix": "Check p0 length matches number of parameters in model",
    },
    {
        "error": "RuntimeError: Fitting failed to converge",
        "cause": "Optimizer couldn't find minimum",
        "fix": "Improve p0, increase maxiter, relax bounds, check model",
    },
    {
        "error": "ValueError: `bounds` must have shape (2, n_params)",
        "cause": "Incorrect bounds format",
        "fix": "Use bounds=([lower1, lower2], [upper1, upper2])",
    },
    {
        "error": "JAX tracer error / ConcretizationTypeError",
        "cause": "Using if/while with traced values inside JIT",
        "fix": "Use jnp.where, jax.lax.cond, or static_argnums",
    },
]

for i, err in enumerate(errors, 1):
    print(f"{i}. ERROR: {err['error']}")
    print(f"   Cause: {err['cause']}")
    print(f"   Fix:   {err['fix']}")
    print()

print("━" * 80)


# ======================================================================
# ## Issue 6: Best Practices Checklist
#
# ### Pre-Flight Checklist (Before Fitting)
#
# Use this checklist to prevent common issues:
# ======================================================================


# Pre-fitting checklist

print("━" * 80)
print("NLSQ PRE-FITTING CHECKLIST")
print("━" * 80)
print()

checklist = [
    (
        "Data Validation",
        [
            "Arrays are JAX/NumPy arrays (not lists)",
            "x and y have same length",
            "No NaN or Inf values in data",
            "Sufficient data points (at least 10x number of parameters)",
            "Data spans appropriate range for model",
        ],
    ),
    (
        "Model Definition",
        [
            "Function uses jnp (not np) for JAX compatibility",
            "No Python if/while statements (use jnp.where, jax.lax.cond)",
            "Model is numerically stable (check for overflow/underflow)",
            "Function signature: model(x, param1, param2, ...)",
        ],
    ),
    (
        "Initial Guess (p0)",
        [
            "p0 length matches number of parameters",
            "Values are reasonable estimates (not random)",
            "Test: plot model(x, *p0) vs. data to verify",
            "Parameters on similar scales (or use rescaling)",
        ],
    ),
    (
        "Bounds (if used)",
        [
            "Format: bounds=([lower1, lower2], [upper1, upper2])",
            "Bounds include true parameter values",
            "Not overly restrictive (allow optimizer to explore)",
            "Physical constraints enforced (e.g., positive rates)",
        ],
    ),
    (
        "Optimization Settings",
        [
            "maxiter sufficient for problem complexity (default: 200-400)",
            "Consider sigma if uncertainties are heteroscedastic",
            "Use absolute_sigma=True if sigma values are reliable",
        ],
    ),
]

for category, items in checklist:
    print(f"📋 {category}:")
    for item in items:
        print(f"   ☐ {item}")
    print()

print("━" * 80)
print("After fitting, always:")
print("  1. Check convergence (no warnings)")
print("  2. Plot residuals (should be random)")
print("  3. Verify parameter uncertainties are reasonable")
print("  4. Test on held-out data if available")
print("━" * 80)


# ======================================================================
# ## Diagnostic Workflow
#
# When troubleshooting a fit that's not working:
# ======================================================================


# Step-by-step diagnostic workflow

print("━" * 80)
print("DIAGNOSTIC WORKFLOW FOR FAILED FITS")
print("━" * 80)
print()

workflow = [
    (
        "Step 1: Visualize the problem",
        [
            "plt.plot(x_data, y_data, 'o', label='Data')",
            "plt.plot(x_data, model(x_data, *p0), '-', label='Initial guess')",
            "plt.legend()",
            "# Does p0 give reasonable shape? If not, fix p0 first!",
        ],
    ),
    (
        "Step 2: Test model function",
        [
            "# Call model directly to check for errors",
            "y_test = model(jnp.array(x_data), *p0)",
            "print(f'Model output: min={y_test.min()}, max={y_test.max()}')",
            "# Check for NaN, Inf, unexpected values",
        ],
    ),
    (
        "Step 3: Simplify the problem",
        [
            "# Try fitting with subset of data",
            "x_sub, y_sub = x_data[:20], y_data[:20]",
            "# Try simpler model (fewer parameters)",
            "# Remove bounds temporarily",
        ],
    ),
    (
        "Step 4: Check convergence details",
        [
            "# Use full_output=True for diagnostics",
            "popt, pcov, infodict = cf.curve_fit(model, x, y, p0=p0, full_output=True)",
            "print(infodict)  # Inspect iteration count, message, etc.",
        ],
    ),
    (
        "Step 5: Try alternative approaches",
        [
            "# Different initial guesses (grid search)",
            "# Different optimization method (if available)",
            "# Add constraints via bounds",
            "# Reformulate model (e.g., log-space)",
        ],
    ),
]

for step, code_lines in workflow:
    print(f"\n{step}")
    print("-" * 60)
    for line in code_lines:
        print(f"  {line}")

print("\n" + "━" * 80)
print("If still stuck: Check NLSQ GitHub issues or ask for help!")
print("  https://github.com/your-nlsq-repo/issues")
print("━" * 80)


# ======================================================================
# ## Summary: Quick Problem Solver
#
# | **Symptom** | **Most Likely Cause** | **Quick Fix** |
# |-------------|----------------------|---------------|
# | "Optimal parameters not found" | Poor initial guess | Improve p0, visualize model(x, *p0) |
# | High residuals, poor fit | Wrong model | Add missing terms, check functional form |
# | NaN or Inf errors | Numerical overflow | Rescale parameters, rewrite model |
# | Very slow (>10s) | JIT compilation | Normal for first call, faster after |
# | Memory error | Too much data | Downsample, use float32, or chunk |
# | Singular matrix | Redundant parameters | Simplify model, add bounds |
# | Parameters hit bounds | Bounds too tight | Relax bounds or remove them |
# | JAX tracer error | if/while in model | Use jnp.where or jax.lax.cond |
#
# ### Pro Tips
#
# 1. **Always visualize** before fitting: `plt.plot(x, model(x, *p0))`
# 2. **Start simple**: Fit with fewer parameters, then add complexity
# 3. **Check units**: Ensure x, y, and parameters are in sensible ranges
# 4. **Use physics**: Prior knowledge helps constrain bounds and p0
# 5. **Read warnings**: They usually tell you exactly what's wrong
#
# ### Additional Resources
#
# - **NLSQ Documentation**: https://nlsq.readthedocs.io/
# - **Advanced Examples**: `examples/advanced_features_demo.ipynb`
# - **JAX Debugging**: https://jax.readthedocs.io/en/latest/debugging/
# - **Gallery Examples**: `examples/gallery/` (domain-specific use cases)
#
# ---
#
# **Remember**: Most fitting issues stem from poor initial guesses, wrong models, or numerical instability. Address these systematically using the diagnostic workflow above.
# ======================================================================
