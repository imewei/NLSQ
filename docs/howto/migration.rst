Migration Guide
===============

This comprehensive guide covers migration to NLSQ from SciPy and between NLSQ versions.

----

Migrating from SciPy
--------------------

Quick Start
~~~~~~~~~~~

**Minimal changes required to migrate from ``scipy.optimize.curve_fit``:**

**Before (SciPy):**

.. code-block:: python

   from scipy.optimize import curve_fit
   import numpy as np


   def exponential(x, a, b):
       return a * np.exp(-b * x)


   x = np.linspace(0, 5, 1000)
   y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.randn(1000)

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1])

**After (NLSQ):**

.. code-block:: python

   from nlsq import curve_fit
   import jax.numpy as jnp  # Changed from numpy
   import numpy as np  # Keep for data generation


   def exponential(x, a, b):
       return a * jnp.exp(-b * x)  # Changed to jnp


   x = np.linspace(0, 5, 1000)
   y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.randn(1000)

   popt, pcov = curve_fit(exponential, x, y, p0=[2, 1])

**That's it!** The API is nearly identical. Just change ``np`` to ``jnp`` in your model function.

Key Differences from SciPy
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **NumPy → JAX NumPy**: Model functions must use ``jax.numpy`` instead of ``numpy`` for GPU acceleration and automatic differentiation.

2. **Method Selection**: NLSQ uses only ``'trf'`` (Trust Region Reflective). Remove ``method='lm'`` or ``method='dogbox'`` parameters.

3. **Automatic Differentiation**: Remove manual Jacobian functions. NLSQ uses JAX autodiff which is faster and more accurate.

4. **Double Precision**: NLSQ automatically enables float64 precision.

API Compatibility
~~~~~~~~~~~~~~~~~

+------------------------+---------------+------------+
| Parameter              | SciPy         | NLSQ       |
+========================+===============+============+
| ``f``                  | Yes           | Yes*       |
+------------------------+---------------+------------+
| ``xdata``              | Yes           | Yes        |
+------------------------+---------------+------------+
| ``ydata``              | Yes           | Yes        |
+------------------------+---------------+------------+
| ``p0``                 | Yes           | Yes        |
+------------------------+---------------+------------+
| ``sigma``              | Yes           | Yes        |
+------------------------+---------------+------------+
| ``absolute_sigma``     | Yes           | Yes        |
+------------------------+---------------+------------+
| ``check_finite``       | Yes           | Yes        |
+------------------------+---------------+------------+
| ``bounds``             | Yes           | Yes        |
+------------------------+---------------+------------+
| ``method``             | lm/trf/dogbox | trf only   |
+------------------------+---------------+------------+
| ``jac``                | Yes           | Yes**      |
+------------------------+---------------+------------+

\*Must use ``jax.numpy`` in function body

\*\*Autodiff recommended instead

Enhanced Result Object
~~~~~~~~~~~~~~~~~~~~~~

NLSQ returns a ``CurveFitResult`` object with additional features:

.. code-block:: python

   # Works like SciPy (tuple unpacking)
   popt, pcov = curve_fit(model, x, y)

   # NLSQ enhancement: access optimization details
   result = curve_fit(model, x, y)
   print(f"R² = {result.r_squared:.4f}")
   print(f"RMSE = {result.rmse:.4f}")

   # Confidence intervals
   ci = result.confidence_intervals(alpha=0.95)

   # Automatic visualization
   result.plot(show_residuals=True)

   # Statistical summary
   result.summary()

Common Migration Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

**Conditional Logic (JAX control flow):**

.. code-block:: python

   # SciPy (works with Python if/else)
   def piecewise(x, a, b, c):
       result = np.zeros_like(x)
       mask = x < 5
       result[mask] = a * x[mask] + b
       result[~mask] = c
       return result


   # NLSQ (use jnp.where)
   def piecewise(x, a, b, c):
       return jnp.where(x < 5, a * x + b, c)

**Remove Manual Jacobians:**

.. code-block:: python

   # SciPy with manual Jacobian
   popt, pcov = curve_fit(model, x, y, jac=my_jacobian_func)

   # NLSQ (autodiff handles it)
   popt, pcov = curve_fit(model, x, y)

**Large Datasets:**

.. code-block:: python

   # NLSQ streaming for large datasets
   from nlsq import fit

   result = fit(model, x_large, y_large, workflow="streaming", memory_limit_gb=4.0)

When to Migrate
~~~~~~~~~~~~~~~

**Migrate to NLSQ when:**

- Dataset has > 10,000 points (GPU advantage)
- Fitting multiple similar datasets (JIT compilation amortized)
- Working with very large datasets (> 1M points)

**Stay with SciPy when:**

- Dataset < 1,000 points (JIT overhead not worth it)
- Need ``method='lm'`` specifically
- One-off fits in simple scripts

Expected Performance
~~~~~~~~~~~~~~~~~~~~

============ ========== =============== ===========
Dataset Size SciPy Time NLSQ Time (GPU) Speedup
============ ========== =============== ===========
1,000        0.05s      0.43s (first)   0.1x-1.7x
10,000       0.18s      0.04s           4.5x
100,000      2.1s       0.09s           23x
1,000,000    40.5s      0.15s           270x
============ ========== =============== ===========

*First call includes JIT compilation. Subsequent calls are much faster.*

----

Version Migration
-----------------

v0.5.x → v0.6.0
~~~~~~~~~~~~~~~

.. note::

   **v0.6.0 Deprecation Purge Complete**

   All deprecated functionality has been completely removed in v0.6.0.
   There are no deprecation warnings or compatibility shims remaining.
   The ``nlsq.compat`` module has been deleted from the package.

**Removed in v0.6.0:**

1. **Domain-specific workflow presets** - Use core presets instead:

   +------------------+---------------+
   | Removed Preset   | Replacement   |
   +==================+===============+
   | ``xpcs``         | ``standard``  |
   +------------------+---------------+
   | ``saxs``         | ``standard``  |
   +------------------+---------------+
   | ``kinetics``     | ``standard``  |
   +------------------+---------------+
   | ``dose_response``| ``quality``   |
   +------------------+---------------+
   | ``imaging``      | ``streaming`` |
   +------------------+---------------+
   | ``materials``    | ``standard``  |
   +------------------+---------------+
   | ``binding``      | ``standard``  |
   +------------------+---------------+
   | ``synchrotron``  | ``streaming`` |
   +------------------+---------------+

2. **SloppyModelAnalyzer aliases** - Use new names:

   .. code-block:: python

      # Before (v0.5.x)
      from nlsq.diagnostics import SloppyModelAnalyzer, SloppyModelReport

      # After (v0.6.0)
      from nlsq.diagnostics import ParameterSensitivityAnalyzer, ParameterSensitivityReport

3. **IssueCategory.SLOPPY** - Use new enum value:

   .. code-block:: python

      # Before (v0.5.x)
      if issue.category == IssueCategory.SLOPPY:
          pass  # handle sensitivity issue

      # After (v0.6.0)
      if issue.category == IssueCategory.SENSITIVITY:
          pass  # handle sensitivity issue

4. **compute_svd_adaptive()** - Use new function:

   .. code-block:: python

      # Before (v0.5.x)
      from nlsq.stability.svd_fallback import compute_svd_adaptive

      # After (v0.6.0)
      from nlsq.stability.svd_fallback import compute_svd_with_fallback

5. **nlsq.compat module** - Deleted entirely. Import from canonical locations.

v0.4.2 → v0.4.3
~~~~~~~~~~~~~~~

**Import Path Changes:**

.. code-block:: python

   # Before (v0.4.x) - deprecated
   from nlsq.core._optimize import OptimizeResult, OptimizeWarning

   # After (v0.4.3) - recommended
   from nlsq.result import OptimizeResult, OptimizeWarning

   # Or from package root
   from nlsq import OptimizeResult, OptimizeWarning

**New Features in v0.4.3:**

- Factory functions: ``create_optimizer()``, ``configure_curve_fit()``
- Protocol adapters for dependency injection
- Security hardening for CLI model loading
- ``wait_for()`` utility for reliable test condition waiting

----

Deprecation Timeline
--------------------

+---------------------------+-------------+------------------+-------------------+
| Item                      | Deprecated  | Removal Version  | Replacement       |
+===========================+=============+==================+===================+
| ``nlsq.core._optimize``   | v0.4.3      | v0.6.0 (removed) | ``nlsq.result``   |
+---------------------------+-------------+------------------+-------------------+
| Domain presets            | v0.5.0      | v0.6.0 (removed) | Core presets      |
+---------------------------+-------------+------------------+-------------------+
| ``SloppyModelAnalyzer``   | v0.5.0      | v0.6.0 (removed) | ``Parameter...``  |
+---------------------------+-------------+------------------+-------------------+
| ``IssueCategory.SLOPPY``  | v0.5.0      | v0.6.0 (removed) | ``SENSITIVITY``   |
+---------------------------+-------------+------------------+-------------------+
| ``compute_svd_adaptive``  | v0.3.5      | v0.6.0 (removed) | ``compute_svd_..``|
+---------------------------+-------------+------------------+-------------------+
| ``nlsq.compat``           | v0.5.0      | v0.6.0 (removed) | Direct imports    |
+---------------------------+-------------+------------------+-------------------+
| ``result['x']`` syntax    | v0.5.0      | v0.6.0 (removed) | ``result.x``      |
+---------------------------+-------------+------------------+-------------------+

----

Finding Deprecated Usage
------------------------

Run these commands to identify deprecated code in your project:

.. code-block:: bash

   # Deprecated presets (removed in v0.6.0)
   grep -rn "from_preset.*xpcs\|saxs\|kinetics\|dose_response\|imaging\|materials\|binding\|synchrotron" .

   # Deprecated class names (removed in v0.6.0)
   grep -rn "SloppyModelAnalyzer\|SloppyModelReport" .

   # Deprecated enum (removed in v0.6.0)
   grep -rn "IssueCategory.SLOPPY" .

   # Deprecated SVD function (removed in v0.6.0)
   grep -rn "compute_svd_adaptive" .

   # Deprecated compat imports (removed in v0.6.0)
   grep -rn "from nlsq.compat import" .

   # Deprecated dict-style access (removed in v0.6.0)
   grep -rn "result\['" --include="*.py" .

----

Migration Checklist
-------------------

**From SciPy:**

- [ ] Replace ``from scipy.optimize import curve_fit`` with ``from nlsq import curve_fit``
- [ ] Add ``import jax.numpy as jnp``
- [ ] Change ``np`` to ``jnp`` in model functions
- [ ] Remove custom Jacobian functions (use autodiff)
- [ ] Remove ``method='lm'`` or ``method='dogbox'`` parameters
- [ ] Test that results match SciPy (within tolerance)

**To v0.6.0:**

- [ ] Replace domain-specific presets with core presets
- [ ] Replace ``SloppyModelAnalyzer`` with ``ParameterSensitivityAnalyzer``
- [ ] Replace ``IssueCategory.SLOPPY`` with ``IssueCategory.SENSITIVITY``
- [ ] Replace ``compute_svd_adaptive`` with ``compute_svd_with_fallback``
- [ ] Remove imports from ``nlsq.compat``
- [ ] Replace ``result['x']`` with ``result.x`` throughout codebase
- [ ] Use ``result.to_dict()`` if dictionary conversion needed

----

Getting Help
------------

If you encounter issues during migration:

1. Check the `API documentation <https://nlsq.readthedocs.io/en/latest/api/>`_
2. Search `GitHub Issues <https://github.com/imewei/NLSQ/issues>`_
3. Open a new issue with the ``migration`` label
