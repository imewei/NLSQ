Migration Guide: SciPy to NLSQ
==============================

This guide helps you migrate from ``scipy.optimize.curve_fit`` to NLSQ
for GPU/TPU-accelerated curve fitting.

Table of Contents
-----------------

1. `Quick Start Migration <#quick-start-migration>`__
2. `API Compatibility <#api-compatibility>`__
3. `Key Differences <#key-differences>`__
4. `Code Examples <#code-examples>`__
5. `Common Migration Patterns <#common-migration-patterns>`__
6. `Performance Considerations <#performance-considerations>`__
7. `Troubleshooting Migration
   Issues <#troubleshooting-migration-issues>`__

--------------

Quick Start Migration
---------------------

Minimal Changes Required
~~~~~~~~~~~~~~~~~~~~~~~~

**Before (SciPy):**

.. code:: python

   from scipy.optimize import curve_fit
   import numpy as np


   def exponential(x, a, b):
       return a * np.exp(-b * x)


   x = np.linspace(0, 5, 1000)
   y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.randn(1000)

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1])

**After (NLSQ):**

.. code:: python

   from nlsq import curve_fit
   import jax.numpy as jnp  # Changed from numpy
   import numpy as np  # Keep for data generation


   def exponential(x, a, b):
       return a * jnp.exp(-b * x)  # Changed to jnp


   x = np.linspace(0, 5, 1000)
   y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.randn(1000)

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1])

**That’s it!** The API is nearly identical. Just change ``np`` to
``jnp`` in your model function.

--------------

API Compatibility
-----------------

Function Signature Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``curve_fit``
^^^^^^^^^^^^^

**SciPy:**

.. code:: python

   scipy.optimize.curve_fit(
       f,
       xdata,
       ydata,
       p0=None,
       sigma=None,
       absolute_sigma=False,
       check_finite=True,
       bounds=(-np.inf, np.inf),
       method=None,
       jac=None,
       **kwargs
   )

**NLSQ:**

.. code:: python

   nlsq.curve_fit(
       f,
       xdata,
       ydata,
       p0=None,
       sigma=None,
       absolute_sigma=False,
       check_finite=True,
       bounds=(-np.inf, np.inf),
       method="trf",  # Default specified
       jac=None,
       solver="auto",  # NLSQ-specific
       batch_size=None,  # NLSQ-specific
       callback=None,  # NLSQ-specific
       **kwargs
   )

Supported Parameters
~~~~~~~~~~~~~~~~~~~~

+------------------------+---------------+------------+---------------+
| Parameter              | SciPy         | NLSQ       | Notes         |
+========================+===============+============+===============+
| ``f``                  | ✅            | ✅         | Must use      |
|                        |               |            | ``jax.numpy`` |
|                        |               |            | in NLSQ       |
+------------------------+---------------+------------+---------------+
| ``xdata``              | ✅            | ✅         | Identical     |
+------------------------+---------------+------------+---------------+
| ``ydata``              | ✅            | ✅         | Identical     |
+------------------------+---------------+------------+---------------+
| ``p0``                 | ✅            | ✅         | ``'auto'``    |
|                        |               |            | supported in  |
|                        |               |            | NLSQ          |
+------------------------+---------------+------------+---------------+
| ``sigma``              | ✅            | ✅         | Identical     |
+------------------------+---------------+------------+---------------+
| ``absolute_sigma``     | ✅            | ✅         | Identical     |
+------------------------+---------------+------------+---------------+
| ``check_finite``       | ✅            | ✅         | Identical     |
+------------------------+---------------+------------+---------------+
| ``bounds``             | ✅            | ✅         | Identical     |
+------------------------+---------------+------------+---------------+
| ``method``             | ``'lm'``,     | ``'trf'``  | NLSQ uses TRF |
|                        | ``'trf'``,    | only       |               |
|                        | ``'dogbox'``  |            |               |
+------------------------+---------------+------------+---------------+
| ``jac``                | ✅            | ✅         | Autodiff      |
|                        |               |            | recommended   |
|                        |               |            | in NLSQ       |
+------------------------+---------------+------------+---------------+
| ``full_output``        | ✅            | ❌         | Use           |
|                        |               |            | ``return_eval |
|                        |               |            | =True``       |
|                        |               |            | instead       |
+------------------------+---------------+------------+---------------+
| ``solver``             | ❌            | ✅         | NLSQ-specific |
+------------------------+---------------+------------+---------------+
| ``callback``           | ❌            | ✅         | NLSQ-specific |
+------------------------+---------------+------------+---------------+

Return Values
~~~~~~~~~~~~~

Both return ``(popt, pcov)`` by default:

.. code:: python

   popt, pcov = curve_fit(...)

**NLSQ enhancement:** Returns ``CurveFitResult`` object that supports
both tuple unpacking and dictionary access:

.. code:: python

   # Works like SciPy
   popt, pcov = curve_fit(...)

   # NLSQ-specific: access optimization details
   result = curve_fit(...)
   popt = result.x
   pcov = result["pcov"]
   nfev = result.nfev  # Number of function evaluations
   cost = result.cost  # Final cost

--------------

Key Differences
---------------

1. NumPy → JAX NumPy
~~~~~~~~~~~~~~~~~~~~

**Critical Change:** Model functions must use ``jax.numpy`` instead of
``numpy``.

**Why:** JAX enables GPU acceleration and automatic differentiation.

**Migration:**

.. code:: python

   # SciPy version
   import numpy as np


   def model_scipy(x, a, b, c):
       return a * np.exp(-b * x) + c * np.sin(x)


   # NLSQ version
   import jax.numpy as jnp


   def model_nlsq(x, a, b, c):
       return a * jnp.exp(-b * x) + c * jnp.sin(x)

**Gotcha:** Data preparation should still use NumPy:

.. code:: python

   import numpy as np
   import jax.numpy as jnp

   # Data generation: use numpy
   x = np.linspace(0, 10, 1000)
   y = np.sin(x) + 0.1 * np.random.randn(1000)


   # Model function: use jax.numpy
   def model(x, a, omega, phi):
       return a * jnp.sin(omega * x + phi)


   # Fitting: works with both numpy and jax arrays
   popt, pcov = curve_fit(model, x, y, p0=[1, 1, 0])

2. Method Selection
~~~~~~~~~~~~~~~~~~~

**SciPy:** Supports ``'lm'``, ``'trf'``, ``'dogbox'``

**NLSQ:** Only ``'trf'`` (Trust Region Reflective)

.. code:: python

   # SciPy
   popt, pcov = curve_fit(model, x, y, method="lm")  # Levenberg-Marquardt
   popt, pcov = curve_fit(model, x, y, method="trf")  # Trust Region
   popt, pcov = curve_fit(model, x, y, method="dogbox")  # Dogbox

   # NLSQ
   popt, pcov = curve_fit(model, x, y, method="trf")  # Only option (default)

**Migration strategy:** If you were using ``method='lm'``, simply remove
it or change to ``'trf'``. TRF is more robust and handles bounds better.

3. Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:** Uses finite differences for Jacobian by default

**NLSQ:** Uses JAX autodiff (much faster and more accurate)

.. code:: python

   # SciPy - you might have provided analytical Jacobian
   def model(x, a, b):
       return a * np.exp(-b * x)


   def jacobian(x, a, b):
       # Analytical derivatives
       da = np.exp(-b * x)
       db = -a * x * np.exp(-b * x)
       return np.column_stack([da, db])


   popt, pcov = curve_fit(model, x, y, jac=jacobian)


   # NLSQ - no Jacobian needed!
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   popt, pcov = curve_fit(model, x, y)  # Autodiff handles Jacobian

**Recommendation:** Remove manual Jacobian functions when migrating.
Autodiff is faster and less error-prone.

4. Double Precision
~~~~~~~~~~~~~~~~~~~

**NLSQ automatically enables 64-bit precision** (float64) upon import.

.. code:: python

   # This happens automatically when you import nlsq
   from nlsq import curve_fit

   # JAX is now in float64 mode

   # If you need float32 for memory reasons:
   from jax import config

   config.update("jax_enable_x64", False)

5. Enhanced Result Object (NLSQ-Specific)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NLSQ returns an enhanced ``CurveFitResult`` object** that supports
both backward compatibility and new visualization/statistics features.

Backward Compatible
^^^^^^^^^^^^^^^^^^^

.. code:: python

   # Works exactly like SciPy (tuple unpacking)
   popt, pcov = curve_fit(model, x, y)

Enhanced Features
^^^^^^^^^^^^^^^^^

.. code:: python

   # NLSQ: Use result object directly
   result = curve_fit(model, x, y)

   # Access parameters (same as SciPy)
   popt = result.popt  # Or result.x
   pcov = result.pcov

   # NEW: Statistical properties
   print(f"R² = {result.r_squared:.4f}")
   print(f"Adjusted R² = {result.adj_r_squared:.4f}")
   print(f"RMSE = {result.rmse:.4f}")
   print(f"MAE = {result.mae:.4f}")
   print(f"AIC = {result.aic:.2f}")
   print(f"BIC = {result.bic:.2f}")

   # NEW: Confidence intervals
   ci = result.confidence_intervals(alpha=0.95)  # 95% CI
   for i, (lower, upper) in enumerate(ci):
       print(f"Parameter {i}: [{lower:.3f}, {upper:.3f}]")

   # NEW: Automatic visualization
   result.plot(show_residuals=True)  # Data + fit + residuals
   plt.show()

   # NEW: Statistical summary table
   result.summary()  # Prints formatted table

**Example output from ``result.summary()``:**

::

   ======================================================================
   Curve Fit Summary
   ======================================================================

   Fitted Parameters:
   ----------------------------------------------------------------------
   Parameter            Value   Std Error                       95% CI
   ----------------------------------------------------------------------
   p0               2.487654    0.021345   [  2.445532,   2.529776]
   p1               1.302341    0.015234   [  1.272368,   1.332314]
   p2               0.498765    0.018654   [  0.462132,   0.535398]

   Goodness of Fit:
   ----------------------------------------------------------------------
   R²                :     0.987654
   Adjusted R²       :     0.986234
   RMSE              :     0.201234
   MAE               :     0.165432

   Model Selection Criteria:
   ----------------------------------------------------------------------
   AIC               :       -45.23
   BIC               :       -38.76

   Convergence Information:
   ----------------------------------------------------------------------
   Success           : True
   Message           : Gradient norm < gtol
   Iterations        : 12
   Final cost        : 2.01456
   Optimality        : 1.234567e-08
   ======================================================================

--------------

Code Examples
-------------

Example 1: Simple Exponential Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   from scipy.optimize import curve_fit
   import numpy as np
   import matplotlib.pyplot as plt


   def exponential(x, a, b, c):
       return a * np.exp(-b * x) + c


   x = np.linspace(0, 4, 50)
   y = exponential(x, 2.5, 1.3, 0.5)
   y += 0.2 * np.random.normal(size=x.size)

   popt, pcov = curve_fit(exponential, x, y)
   perr = np.sqrt(np.diag(pcov))

   plt.plot(x, y, "o", label="data")
   plt.plot(x, exponential(x, *popt), "-", label="fit")
   plt.legend()
   plt.show()

**NLSQ (minimal changes):**

.. code:: python

   from nlsq import curve_fit
   import jax.numpy as jnp  # Added
   import numpy as np
   import matplotlib.pyplot as plt


   def exponential(x, a, b, c):
       return a * jnp.exp(-b * x) + c  # Changed to jnp


   x = np.linspace(0, 4, 50)
   y = exponential(x, 2.5, 1.3, 0.5)
   y += 0.2 * np.random.normal(size=x.size)

   popt, pcov = curve_fit(exponential, x, y)
   perr = np.sqrt(np.diag(pcov))

   plt.plot(x, y, "o", label="data")
   plt.plot(x, exponential(x, *popt), "-", label="fit")
   plt.legend()
   plt.show()

Example 2: Fitting with Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   popt, pcov = curve_fit(exponential, x, y, bounds=([0, 0, -np.inf], [10, 5, np.inf]))

**NLSQ (identical):**

.. code:: python

   popt, pcov = curve_fit(exponential, x, y, bounds=([0, 0, -np.inf], [10, 5, np.inf]))

Example 3: Weighted Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   sigma = np.ones(len(x)) * 0.1
   sigma[10:20] = 0.5  # Higher uncertainty

   popt, pcov = curve_fit(exponential, x, y, sigma=sigma, absolute_sigma=True)

**NLSQ (identical):**

.. code:: python

   sigma = np.ones(len(x)) * 0.1
   sigma[10:20] = 0.5

   popt, pcov = curve_fit(exponential, x, y, sigma=sigma, absolute_sigma=True)

Example 4: Multi-dimensional X Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   def surface(xdata, a, b, c):
       x, y = xdata
       return a * np.exp(-b * x**2) * np.sin(c * y)


   x1 = np.linspace(-2, 2, 50)
   x2 = np.linspace(-2, 2, 50)
   X1, X2 = np.meshgrid(x1, x2)
   xdata = np.vstack([X1.ravel(), X2.ravel()])
   ydata = surface(xdata, 2.5, 1.0, 3.0) + 0.1 * np.random.randn(xdata.shape[1])

   popt, pcov = curve_fit(surface, xdata, ydata)

**NLSQ:**

.. code:: python

   def surface(xdata, a, b, c):
       x, y = xdata
       return a * jnp.exp(-b * x**2) * jnp.sin(c * y)  # jnp


   x1 = np.linspace(-2, 2, 50)
   x2 = np.linspace(-2, 2, 50)
   X1, X2 = np.meshgrid(x1, x2)
   xdata = np.vstack([X1.ravel(), X2.ravel()])
   ydata = surface(xdata, 2.5, 1.0, 3.0) + 0.1 * np.random.randn(xdata.shape[1])

   popt, pcov = curve_fit(surface, xdata, ydata)

--------------

Common Migration Patterns
-------------------------

Pattern 1: Conditional Logic in Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Python control flow (if/else) not JIT-compatible

**SciPy (works):**

.. code:: python

   def piecewise(x, a, b, c):
       result = np.zeros_like(x)
       mask = x < 5
       result[mask] = a * x[mask] + b
       result[~mask] = c
       return result

**NLSQ (needs JAX control flow):**

.. code:: python

   def piecewise(x, a, b, c):
       return jnp.where(x < 5, a * x + b, c)

Pattern 2: Custom Jacobian → Autodiff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   def model(x, a, b):
       return a * np.exp(-b * x)


   def jac(x, a, b):
       J = np.zeros((len(x), 2))
       J[:, 0] = np.exp(-b * x)
       J[:, 1] = -a * x * np.exp(-b * x)
       return J


   popt, pcov = curve_fit(model, x, y, jac=jac)

**NLSQ (remove Jacobian):**

.. code:: python

   def model(x, a, b):
       return a * jnp.exp(-b * x)


   popt, pcov = curve_fit(model, x, y)  # Autodiff handles it

Pattern 3: Multiple Fits → CurveFit Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   results = []
   for x_data, y_data in datasets:
       popt, pcov = curve_fit(model, x_data, y_data, p0=[2, 1])
       results.append(popt)

**NLSQ (faster with CurveFit class):**

.. code:: python

   from nlsq import CurveFit

   fitter = CurveFit()  # Compile once
   results = []
   for x_data, y_data in datasets:
       popt, pcov = fitter.curve_fit(model, x_data, y_data, p0=[2, 1])
       results.append(popt)

**Speedup:** 5-10x faster for 10+ datasets

Pattern 4: Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   # Might be slow or run out of memory
   x_large = np.linspace(0, 100, 10_000_000)
   y_large = model(x_large, 2.5, 1.3) + np.random.randn(10_000_000) * 0.01

   popt, pcov = curve_fit(model, x_large, y_large)  # Slow!

**NLSQ:**

.. code:: python

   from nlsq.streaming.large_dataset import fit_large_dataset

   x_large = np.linspace(0, 100, 10_000_000)
   y_large = model(x_large, 2.5, 1.3) + np.random.randn(10_000_000) * 0.01

   popt, pcov, info = fit_large_dataset(
       model, x_large, y_large, memory_limit_gb=4.0, progress=True
   )

**Speedup:** 100-300x faster with GPU

--------------

Performance Considerations
--------------------------

When to Migrate
~~~~~~~~~~~~~~~

Migrate to NLSQ when:

-  ✅ Dataset has > 10,000 points (GPU advantage)
-  ✅ Fitting multiple similar datasets (JIT compilation amortized)
-  ✅ Need faster iteration in research/development
-  ✅ Working with very large datasets (> 1M points)

Stay with SciPy when:

-  ✅ Dataset < 1,000 points (JIT overhead not worth it)
-  ✅ One-off fits in simple scripts
-  ✅ No GPU available and dataset is small
-  ✅ Need ``method='lm'`` specifically

Expected Performance Gains
~~~~~~~~~~~~~~~~~~~~~~~~~~

============ ========== ============================= ===========
Dataset Size SciPy Time NLSQ Time (GPU)               Speedup
============ ========== ============================= ===========
1,000        0.05s      0.43s (first), 0.03s (cached) 0.1x → 1.7x
10,000       0.18s      0.04s                         4.5x
100,000      2.1s       0.09s                         23x
1,000,000    40.5s      0.15s                         270x
============ ========== ============================= ===========

*First call includes JIT compilation overhead. Subsequent calls are much
faster.*

--------------

Troubleshooting Migration Issues
--------------------------------

Issue 1: “TypeError: **jit_vectorcall** only supported for built-in functions”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Using NumPy instead of JAX NumPy in model

**Fix:**

.. code:: python

   # Before (error)
   def model(x, a, b):
       return a * np.exp(-b * x)  # np.exp not JIT-compatible


   # After (fixed)
   def model(x, a, b):
       return a * jnp.exp(-b * x)  # jnp.exp works

Issue 2: “TypeError: Shapes must be 1D sequences of concrete values”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Dynamic array shapes (not JIT-compatible)

**Fix:**

.. code:: python

   # Before (error)
   def model(x, a, b):
       if len(x) > 100:
           return a * jnp.exp(-b * x[:100])  # Dynamic slicing
       return a * jnp.exp(-b * x)


   # After (fixed)
   def model(x, a, b):
       return a * jnp.exp(-b * x)  # Avoid conditionals on array size

Issue 3: Results Slightly Different from SciPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Different algorithms (SciPy LM vs NLSQ TRF) or numerical
precision

**Check:**

.. code:: python

   # Compare results
   popt_scipy, _ = scipy_curve_fit(...)
   popt_nlsq, _ = nlsq_curve_fit(...)

   rel_error = np.abs((popt_scipy - popt_nlsq) / popt_scipy)
   print(f"Relative error: {rel_error}")

   # Should be < 1e-6 for well-conditioned problems
   assert np.all(rel_error < 1e-5)

Issue 4: “No GPU/TPU found”
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** JAX not installed with CUDA support

**Fix:**

.. code:: bash

   # Reinstall JAX with CUDA
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # Verify
   python -c "import jax; print(jax.devices())"

Issue 5: Slower Than SciPy
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause:** Small dataset + first call (JIT compilation overhead)

**Fix:**

.. code:: python

   # For small datasets, use CPU
   if len(x) < 10000:
       import os

       os.environ["JAX_PLATFORM_NAME"] = "cpu"

   # Or use CurveFit class for multiple fits
   from nlsq import CurveFit

   fitter = CurveFit()
   # First call: slow (compilation)
   # Subsequent calls: fast

--------------

Migration Checklist
-------------------

Before migrating:

-  ☐ Install NLSQ: ``pip install nlsq``
-  ☐ Verify JAX installation:
   ``python -c "import jax; print(jax.devices())"``
-  ☐ Check dataset size (> 10K recommended for GPU benefit)

During migration:

-  ☐ Replace ``from scipy.optimize import curve_fit`` with
   ``from nlsq import curve_fit``
-  ☐ Add ``import jax.numpy as jnp``
-  ☐ Change ``np`` to ``jnp`` in model functions
-  ☐ Remove custom Jacobian functions (use autodiff)
-  ☐ Change ``method='lm'`` to ``method='trf'`` (or remove, it’s
   default)
-  ☐ Use ``CurveFit`` class if fitting multiple datasets

After migration:

-  ☐ Test that results match SciPy (within tolerance)
-  ☐ Benchmark performance improvement
-  ☐ Verify GPU is being used (check ``jax.devices()``)
-  ☐ Update tests and documentation

--------------

Side-by-Side Comparison
-----------------------

Complete Example: SciPy vs NLSQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**SciPy:**

.. code:: python

   from scipy.optimize import curve_fit
   import numpy as np
   import matplotlib.pyplot as plt


   # Model definition
   def gaussian(x, amp, cen, wid):
       return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = gaussian(x, 2.5, 5.0, 1.0)
   y = y_true + 0.2 * np.random.normal(size=x.size)

   # Fit
   p0 = [2, 5, 1]  # Initial guess
   popt, pcov = curve_fit(gaussian, x, y, p0=p0)

   # Uncertainty
   perr = np.sqrt(np.diag(pcov))

   # Results
   print(f"Amplitude: {popt[0]:.3f} ± {perr[0]:.3f}")
   print(f"Center: {popt[1]:.3f} ± {perr[1]:.3f}")
   print(f"Width: {popt[2]:.3f} ± {perr[2]:.3f}")

   # Plot
   plt.plot(x, y, "o", label="Data")
   plt.plot(x, gaussian(x, *popt), "-", label="Fit")
   plt.legend()
   plt.show()

**NLSQ:**

.. code:: python

   from nlsq import curve_fit
   import jax.numpy as jnp  # Added jnp
   import numpy as np
   import matplotlib.pyplot as plt


   # Model definition
   def gaussian(x, amp, cen, wid):
       return amp * jnp.exp(-((x - cen) ** 2) / (2 * wid**2))  # jnp


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = gaussian(x, 2.5, 5.0, 1.0)
   y = y_true + 0.2 * np.random.normal(size=x.size)

   # Fit
   p0 = [2, 5, 1]  # Initial guess
   popt, pcov = curve_fit(gaussian, x, y, p0=p0)

   # Uncertainty
   perr = np.sqrt(np.diag(pcov))

   # Results
   print(f"Amplitude: {popt[0]:.3f} ± {perr[0]:.3f}")
   print(f"Center: {popt[1]:.3f} ± {perr[1]:.3f}")
   print(f"Width: {popt[2]:.3f} ± {perr[2]:.3f}")

   # Plot
   plt.plot(x, y, "o", label="Data")
   plt.plot(x, gaussian(x, *popt), "-", label="Fit")
   plt.legend()
   plt.show()

**Differences:** Only 3 lines changed (imports + ``jnp`` in model)!

Enhanced NLSQ Example with Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**NLSQ with new features:**

.. code:: python

   from nlsq import curve_fit
   import jax.numpy as jnp
   import numpy as np
   import matplotlib.pyplot as plt


   # Model definition
   def gaussian(x, amp, cen, wid):
       return amp * jnp.exp(-((x - cen) ** 2) / (2 * wid**2))


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y_true = gaussian(x, 2.5, 5.0, 1.0)
   y = y_true + 0.2 * np.random.normal(size=x.size)

   # Fit (use result object, not tuple unpacking)
   result = curve_fit(gaussian, x, y, p0=[2, 5, 1])

   # NEW: Automatic statistical analysis
   print(f"R² = {result.r_squared:.4f}")  # Goodness of fit
   print(f"RMSE = {result.rmse:.4f}")  # Error metric
   print(f"AIC = {result.aic:.2f}")  # Model selection

   # NEW: Confidence intervals
   ci = result.confidence_intervals(alpha=0.95)
   print("\n95% Confidence Intervals:")
   print(f"Amplitude: [{ci[0,0]:.3f}, {ci[0,1]:.3f}]")
   print(f"Center:    [{ci[1,0]:.3f}, {ci[1,1]:.3f}]")
   print(f"Width:     [{ci[2,0]:.3f}, {ci[2,1]:.3f}]")

   # NEW: Automatic visualization
   result.plot(show_residuals=True)
   plt.show()

   # NEW: Complete summary table
   result.summary()

**Output:**

::

   R² = 0.9876
   RMSE = 0.1987
   AIC = -54.32

   95% Confidence Intervals:
   Amplitude: [2.423, 2.587]
   Center:    [4.965, 5.035]
   Width:     [0.962, 1.042]

   ======================================================================
   Curve Fit Summary
   ======================================================================
   [Full statistical summary printed here...]

**Key Advantages:** - **3 lines** instead of ~20 for complete analysis -
**Automatic** confidence intervals (no manual calculation) -
**Built-in** visualization (data + fit + residuals) - **Statistical
metrics** (R², RMSE, AIC, BIC) computed automatically - **Still backward
compatible** with SciPy tuple unpacking

--------------

Additional Resources
--------------------

-  `NLSQ Documentation <https://nlsq.readthedocs.io>`__
-  `JAX Documentation <https://docs.jax.dev/en/latest/>`__
-  :doc:`Performance Optimization Guide <optimize_performance>`
-  :doc:`Configuration Reference <../reference/configuration>`
-  :doc:`Troubleshooting Guide <troubleshooting>`
